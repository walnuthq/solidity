/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0
/**
 * libyul-AST importer implementation.
 *
 * Import model (ADR-004, ADR-005):
 *  - every Yul local becomes a yul.var; reads become yul.var_load snapshots
 *    and assignments become yul.assign - fully general for loops and nested
 *    scopes without any phi/SSA construction;
 *  - builtin calls create the identically-named yul.* op generically (the
 *    1:1 design of ADR-003); arguments are evaluated right-to-left per the
 *    Yul specification;
 *  - `switch` is desugared into exclusive eq-guarded ifs (plus a `matched`
 *    variable when a default case exists);
 *  - statements after a terminator (leave/break/continue) are unreachable
 *    in Yul and are dropped;
 *  - the for-loop init block is hoisted before the loop (SSA names are
 *    unique, so Yul's init scoping is preserved).
 */

#include "YulASTImporter.h"

#include "YulDialect.h"
#include "YulOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#pragma GCC diagnostic pop

#include <libyul/AST.h>
#include <libyul/Dialect.h>
#include <libyul/Object.h>
#include <libyul/YulStack.h>

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace solidity;
using namespace solidity::mlirgen;

namespace
{

struct ImportError
{
	std::string message;
};

class Importer
{
public:
	Importer(mlir::MLIRContext& _ctx, yul::Dialect const& _yulDialect): m_builder(&_ctx), m_yulDialect(_yulDialect) {}

	mlir::OwningOpRef<mlir::ModuleOp> run(yul::Block const& _root, std::string& _error)
	{
		mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(m_builder.getUnknownLoc());
		m_builder.setInsertionPointToStart(module->getBody());
		try
		{
			Scope rootScope(*this);
			importBlockInline(_root);
		}
		catch (ImportError const& _e)
		{
			_error = _e.message;
			return nullptr;
		}
		if (failed(mlir::verify(*module)))
		{
			// Pinpoint the classic failure shape (terminator not last).
			std::string info;
			module->walk([&](mlir::Block* block) {
				for (auto it = block->begin(); it != block->end(); ++it)
					if (it->hasTrait<mlir::OpTrait::IsTerminator>() && std::next(it) != block->end())
						info = " [" + it->getName().getStringRef().str() + " followed by "
							   + std::next(it)->getName().getStringRef().str() + " inside "
							   + block->getParentOp()->getName().getStringRef().str() + "]";
			});
			_error = "imported module failed MLIR verification" + info;
			return nullptr;
		}
		return module;
	}

private:
	mlir::OpBuilder m_builder;
	yul::Dialect const& m_yulDialect;

	struct FunctionSig
	{
		size_t numParams = 0;
		size_t numReturns = 0;
	};

	// Lexical scopes for variables (yul.var refs) and function signatures.
	std::vector<std::map<std::string, mlir::Value>> m_varScopes;
	std::vector<std::map<std::string, FunctionSig>> m_funcScopes;
	// Mutable return variables of the function currently being imported.
	std::vector<std::vector<mlir::Value>> m_returnVarStack;

	class Scope
	{
	public:
		explicit Scope(Importer& _importer): m_importer(_importer)
		{
			m_importer.m_varScopes.emplace_back();
			m_importer.m_funcScopes.emplace_back();
		}
		~Scope()
		{
			m_importer.m_varScopes.pop_back();
			m_importer.m_funcScopes.pop_back();
		}

	private:
		Importer& m_importer;
	};

	/// Exception-safe variable isolation for function bodies (Yul functions
	/// cannot see outer locals): swaps the variable scopes out on entry and
	/// back on destruction, and balances the return-variable stack - also
	/// when an ImportError unwinds through a function body.
	class FunctionIsolation
	{
	public:
		explicit FunctionIsolation(Importer& _importer)
			: m_importer(_importer), m_savedVarScopes(std::move(_importer.m_varScopes))
		{
			m_importer.m_varScopes.clear();
			m_importer.m_returnVarStack.emplace_back();
		}
		~FunctionIsolation()
		{
			m_importer.m_returnVarStack.pop_back();
			m_importer.m_varScopes = std::move(m_savedVarScopes);
		}

	private:
		Importer& m_importer;
		std::vector<std::map<std::string, mlir::Value>> m_savedVarScopes;
	};

	[[noreturn]] static void fail(std::string _message) { throw ImportError{std::move(_message)}; }

	mlir::Location loc() { return m_builder.getUnknownLoc(); }

	mlir::IntegerType wordType() { return m_builder.getIntegerType(256); }

	mlir::Value defineVar(std::string const& _name, mlir::Value _init)
	{
		auto var = m_builder.create<mlir::yul::VarOp>(loc(), _init);
		// Kept so a caller splicing this block into something else can find a
		// particular variable again - inline assembly needs to hand Solidity's
		// values in and read what the block assigned back out.
		var->setAttr("yul_name", m_builder.getStringAttr(_name));
		m_varScopes.back()[_name] = var.getResult();
		return var.getResult();
	}

	mlir::Value lookupVar(std::string const& _name)
	{
		for (auto it = m_varScopes.rbegin(); it != m_varScopes.rend(); ++it)
		{
			auto found = it->find(_name);
			if (found != it->end())
				return found->second;
		}
		fail("reference to unknown variable '" + _name + "'");
	}

	FunctionSig lookupFunction(std::string const& _name)
	{
		for (auto it = m_funcScopes.rbegin(); it != m_funcScopes.rend(); ++it)
		{
			auto found = it->find(_name);
			if (found != it->end())
				return found->second;
		}
		fail("reference to unknown function '" + _name + "'");
	}

	mlir::Value constant(llvm::APInt _value)
	{
		return m_builder.create<mlir::yul::ConstOp>(loc(), mlir::IntegerAttr::get(wordType(), std::move(_value)));
	}

	mlir::Value constant(uint64_t _value) { return constant(llvm::APInt(256, _value)); }

	static llvm::APInt toAPInt(solidity::u256 const& _value)
	{
		// u256 -> decimal string -> APInt(256). Slow but exact; fine for import.
		return llvm::APInt(256, _value.str(), 10);
	}

	//===------------------------------------------------------------------===//
	// Expressions
	//===------------------------------------------------------------------===//

	mlir::Value importLiteral(yul::Literal const& _literal)
	{
		if (_literal.value.unlimited())
			fail("unlimited (builtin-argument) string literal outside a special builtin");
		return constant(toAPInt(_literal.value.value()));
	}

	/// Extracts the literal string argument of datasize/dataoffset & co.
	static std::string literalStringArgument(yul::FunctionCall const& _call, char const* _builtinName)
	{
		if (_call.arguments.size() != 1 || !std::holds_alternative<yul::Literal>(_call.arguments[0]))
			fail(std::string("expected a single literal argument to ") + _builtinName);
		yul::Literal const& lit = std::get<yul::Literal>(_call.arguments[0]);
		if (lit.value.unlimited())
			return lit.value.builtinStringLiteralValue();
		fail(std::string("expected a string literal argument to ") + _builtinName);
	}

	/// Imports a call to a builtin function; returns the results (0 or 1).
	llvm::SmallVector<mlir::Value, 1> importBuiltinCall(yul::FunctionCall const& _call, yul::BuiltinName const& _name)
	{
		yul::BuiltinFunction const& builtin = m_yulDialect.builtin(_name.handle);
		std::string const& name = builtin.name;

		// Special forms first.
		if (name == "pop")
		{
			// pop(x): evaluate and discard - no dedicated op in the dialect.
			importExpression(_call.arguments.at(0));
			return {};
		}
		if (name == "dataoffset")
			return {m_builder.create<mlir::yul::DataOffsetOp>(loc(), literalStringArgument(_call, "dataoffset"))};
		if (name == "datasize")
			return {m_builder.create<mlir::yul::DataSizeOp>(loc(), literalStringArgument(_call, "datasize"))};
		if (name == "memoryguard")
		{
			if (_call.arguments.size() != 1 || !std::holds_alternative<yul::Literal>(_call.arguments[0]))
				fail("expected a literal argument to memoryguard");
			yul::Literal const& lit = std::get<yul::Literal>(_call.arguments[0]);
			return {m_builder.create<mlir::yul::MemoryGuardOp>(
				loc(), mlir::IntegerAttr::get(wordType(), toAPInt(lit.value.value())))};
		}
		if (name == "loadimmutable")
			return {m_builder.create<mlir::yul::LoadImmutableOp>(loc(), literalStringArgument(_call, "loadimmutable"))};
		if (name == "setimmutable")
		{
			if (_call.arguments.size() != 3 || !std::holds_alternative<yul::Literal>(_call.arguments[1]))
				fail("expected setimmutable(offset, \"name\", value)");
			yul::Literal const& nameLit = std::get<yul::Literal>(_call.arguments[1]);
			if (!nameLit.value.unlimited())
				fail("expected a string literal immutable name");
			// Right-to-left evaluation of the value operands.
			mlir::Value value = importExpression(_call.arguments[2]);
			mlir::Value offset = importExpression(_call.arguments[0]);
			mlir::OperationState state(loc(), "yul.setimmutable");
			state.addOperands({offset, value});
			state.addAttribute("immutable_name", m_builder.getStringAttr(nameLit.value.builtinStringLiteralValue()));
			m_builder.create(state);
			return {};
		}
		if (name == "linkersymbol")
			return {m_builder.create<mlir::yul::LinkerSymbolOp>(loc(), literalStringArgument(_call, "linkersymbol"))};
		if (name.substr(0, 8) == "verbatim")
			fail("verbatim builtins are not supported yet");

		// Generic 1:1 builtin (ADR-003): op name == builtin name. Arguments
		// are evaluated right-to-left per the Yul specification, keeping
		// positional order in the created op.
		llvm::SmallVector<mlir::Value, 8> args(_call.arguments.size());
		for (size_t i = _call.arguments.size(); i > 0; --i)
			args[i - 1] = importExpression(_call.arguments[i - 1]);

		std::string opName = "yul." + name;
		std::optional<mlir::RegisteredOperationName> registered =
			mlir::RegisteredOperationName::lookup(opName, m_builder.getContext());
		if (!registered)
			fail("builtin '" + name + "' has no yul dialect op");

		mlir::OperationState state(loc(), *registered);
		state.addOperands(args);
		if (builtin.numReturns == 1)
			state.addTypes(wordType());
		else if (builtin.numReturns > 1)
			fail("builtin '" + name + "' with multiple returns is not supported");
		mlir::Operation* op = m_builder.create(state);

		llvm::SmallVector<mlir::Value, 1> results;
		for (mlir::Value result: op->getResults())
			results.push_back(result);
		return results;
	}

	/// Imports a call (builtin or user function); returns all results.
	llvm::SmallVector<mlir::Value, 1> importCall(yul::FunctionCall const& _call)
	{
		if (std::holds_alternative<yul::BuiltinName>(_call.functionName))
			return importBuiltinCall(_call, std::get<yul::BuiltinName>(_call.functionName));

		std::string name = std::get<yul::Identifier>(_call.functionName).name.str();
		FunctionSig sig = lookupFunction(name);
		if (sig.numParams != _call.arguments.size())
			fail("argument count mismatch calling '" + name + "'");

		llvm::SmallVector<mlir::Value, 8> args(_call.arguments.size());
		for (size_t i = _call.arguments.size(); i > 0; --i)
			args[i - 1] = importExpression(_call.arguments[i - 1]);

		llvm::SmallVector<mlir::Type, 2> resultTypes(sig.numReturns, wordType());
		auto call = m_builder.create<mlir::yul::FuncCallOp>(
			loc(), resultTypes, mlir::FlatSymbolRefAttr::get(m_builder.getContext(), name), args);

		llvm::SmallVector<mlir::Value, 1> results;
		for (mlir::Value result: call.getResults())
			results.push_back(result);
		return results;
	}

	/// Imports an expression that must produce exactly one value.
	mlir::Value importExpression(yul::Expression const& _expr)
	{
		if (std::holds_alternative<yul::Literal>(_expr))
			return importLiteral(std::get<yul::Literal>(_expr));
		if (std::holds_alternative<yul::Identifier>(_expr))
			return m_builder.create<mlir::yul::VarLoadOp>(
				loc(), lookupVar(std::get<yul::Identifier>(_expr).name.str()));

		auto results = importCall(std::get<yul::FunctionCall>(_expr));
		if (results.size() != 1)
			fail("expected a single-value expression");
		return results[0];
	}

	/// Imports the RHS of a (multi-)declaration/assignment for _count names.
	llvm::SmallVector<mlir::Value, 1> importRHS(yul::Expression const& _expr, size_t _count)
	{
		if (_count == 1)
			return {importExpression(_expr)};
		if (!std::holds_alternative<yul::FunctionCall>(_expr))
			fail("multi-value RHS must be a function call");
		auto results = importCall(std::get<yul::FunctionCall>(_expr));
		if (results.size() != _count)
			fail("multi-value arity mismatch");
		return results;
	}

	//===------------------------------------------------------------------===//
	// Statements
	//===------------------------------------------------------------------===//

	/// Pre-registers the signatures of all functions defined directly in the
	/// block (Yul functions are visible in the whole enclosing block).
	void hoistFunctionSignatures(yul::Block const& _block)
	{
		for (yul::Statement const& stmt: _block.statements)
			if (std::holds_alternative<yul::FunctionDefinition>(stmt))
			{
				yul::FunctionDefinition const& fun = std::get<yul::FunctionDefinition>(stmt);
				m_funcScopes.back()[fun.name.str()] = FunctionSig{fun.parameters.size(), fun.returnVariables.size()};
			}
	}

	/// Imports the statements of _block into the current insertion point,
	/// opening a fresh lexical scope. @returns true if the block ended in a
	/// terminator (leave/break/continue).
	bool importBlockInline(yul::Block const& _block)
	{
		Scope scope(*this);
		hoistFunctionSignatures(_block);
		// Function definitions are position-independent declarations in Yul
		// (via-ir places them after terminating statements): import their
		// bodies first, then walk the executable statements.
		for (yul::Statement const& stmt: _block.statements)
			if (std::holds_alternative<yul::FunctionDefinition>(stmt))
				importFunction(std::get<yul::FunctionDefinition>(stmt));
		for (yul::Statement const& stmt: _block.statements)
		{
			if (std::holds_alternative<yul::FunctionDefinition>(stmt))
				continue;
			if (importStatement(stmt))
				return true; // terminator: anything further is unreachable (ADR-005)
		}
		return false;
	}

	/// @returns true if the statement terminates the current block.
	bool importStatement(yul::Statement const& _stmt)
	{
		if (std::holds_alternative<yul::ExpressionStatement>(_stmt))
		{
			yul::Expression const& expr = std::get<yul::ExpressionStatement>(_stmt).expression;
			if (!std::holds_alternative<yul::FunctionCall>(expr))
				fail("expression statement must be a call");
			yul::FunctionCall const& call = std::get<yul::FunctionCall>(expr);

			// Terminating builtins (revert/return/stop/invalid/...) end the
			// block; anything following is unreachable and dropped (ADR-005).
			bool terminates = false;
			if (auto const* builtinName = std::get_if<yul::BuiltinName>(&call.functionName))
				terminates = !m_yulDialect.builtin(builtinName->handle).controlFlowSideEffects.canContinue;

			auto results = importCall(call);
			if (!results.empty())
				fail("expression statement with results");
			return terminates;
		}
		if (std::holds_alternative<yul::VariableDeclaration>(_stmt))
		{
			yul::VariableDeclaration const& decl = std::get<yul::VariableDeclaration>(_stmt);
			if (decl.value)
			{
				auto values = importRHS(*decl.value, decl.variables.size());
				for (size_t i = 0; i < decl.variables.size(); ++i)
					defineVar(decl.variables[i].name.str(), values[i]);
			}
			else
				for (auto const& variable: decl.variables)
					defineVar(variable.name.str(), constant(uint64_t(0)));
			return false;
		}
		if (std::holds_alternative<yul::Assignment>(_stmt))
		{
			yul::Assignment const& assignment = std::get<yul::Assignment>(_stmt);
			auto values = importRHS(*assignment.value, assignment.variableNames.size());
			for (size_t i = 0; i < assignment.variableNames.size(); ++i)
				m_builder.create<mlir::yul::AssignOp>(
					loc(), lookupVar(assignment.variableNames[i].name.str()), values[i]);
			return false;
		}
		if (std::holds_alternative<yul::FunctionDefinition>(_stmt))
		{
			importFunction(std::get<yul::FunctionDefinition>(_stmt));
			return false;
		}
		if (std::holds_alternative<yul::If>(_stmt))
		{
			yul::If const& ifStmt = std::get<yul::If>(_stmt);
			mlir::Value cond = importExpression(*ifStmt.condition);
			auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), cond);
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
			importBlockInline(ifStmt.body);
			return false;
		}
		if (std::holds_alternative<yul::Switch>(_stmt))
		{
			importSwitch(std::get<yul::Switch>(_stmt));
			return false;
		}
		if (std::holds_alternative<yul::ForLoop>(_stmt))
		{
			importForLoop(std::get<yul::ForLoop>(_stmt));
			return false;
		}
		if (std::holds_alternative<yul::Break>(_stmt))
		{
			m_builder.create<mlir::yul::BreakOp>(loc());
			return true;
		}
		if (std::holds_alternative<yul::Continue>(_stmt))
		{
			m_builder.create<mlir::yul::ContinueOp>(loc());
			return true;
		}
		if (std::holds_alternative<yul::Leave>(_stmt))
		{
			createLeave();
			return true;
		}
		if (std::holds_alternative<yul::Block>(_stmt))
		{
			importBlockInline(std::get<yul::Block>(_stmt));
			return false;
		}
		fail("unsupported Yul statement");
	}

	void createLeave()
	{
		llvm::SmallVector<mlir::Value, 2> results;
		if (!m_returnVarStack.empty())
			for (mlir::Value returnVar: m_returnVarStack.back())
				results.push_back(m_builder.create<mlir::yul::VarLoadOp>(loc(), returnVar));
		m_builder.create<mlir::yul::LeaveOp>(loc(), results);
	}

	void importFunction(yul::FunctionDefinition const& _fun)
	{
		llvm::SmallVector<mlir::Type, 4> paramTypes(_fun.parameters.size(), wordType());
		llvm::SmallVector<mlir::Type, 2> resultTypes(_fun.returnVariables.size(), wordType());
		auto funcType = mlir::FunctionType::get(m_builder.getContext(), paramTypes, resultTypes);

		auto funcOp = m_builder.create<mlir::yul::FuncOp>(loc(), _fun.name.str(), funcType);
		mlir::Block* body = &funcOp.getBody().emplaceBlock();

		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToStart(body);

		// Yul functions are isolated: only params/returns are in scope.
		// (Function signatures remain visible - symbol calls, not SSA values.)
		FunctionIsolation isolation(*this);
		Scope scope(*this);

		// Parameters and named returns are mutable variables.
		for (auto const& param: _fun.parameters)
			defineVar(param.name.str(), body->addArgument(wordType(), loc()));
		for (auto const& returnVariable: _fun.returnVariables)
			m_returnVarStack.back().push_back(defineVar(returnVariable.name.str(), constant(uint64_t(0))));

		bool terminated = importBlockInline(_fun.body);
		if (!terminated)
			createLeave(); // implicit leave at the end of the body
	}

	void importForLoop(yul::ForLoop const& _for)
	{
		// The init block's scope spans cond/post/body: open a scope, import
		// init statements inline BEFORE the loop op (hoisting; ADR-004).
		Scope scope(*this);
		hoistFunctionSignatures(_for.pre);
		for (yul::Statement const& stmt: _for.pre.statements)
		{
			if (std::holds_alternative<yul::Break>(stmt) || std::holds_alternative<yul::Continue>(stmt)
				|| std::holds_alternative<yul::Leave>(stmt))
				fail("loop init block must not contain terminators");
			importStatement(stmt);
		}

		auto forOp = m_builder.create<mlir::yul::ForOp>(loc());
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getCondRegion().emplaceBlock());
			mlir::Value cond = importExpression(*_for.condition);
			m_builder.create<mlir::yul::ConditionOp>(loc(), cond);
		}
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getBodyRegion().emplaceBlock());
			importBlockInline(_for.body);
		}
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getPostRegion().emplaceBlock());
			importBlockInline(_for.post);
		}
	}

	void importSwitch(yul::Switch const& _switch)
	{
		// Desugar into exclusive eq-guarded ifs; a `matched` variable is only
		// needed when a default case exists. Case values are distinct by
		// analysis, so at most one case body runs.
		mlir::Value scrutinee = importExpression(*_switch.expression);

		bool hasDefault = false;
		for (yul::Case const& switchCase: _switch.cases)
			if (!switchCase.value)
				hasDefault = true;

		mlir::Value matched;
		if (hasDefault)
			matched = m_builder.create<mlir::yul::VarOp>(loc(), constant(uint64_t(0)));

		for (yul::Case const& switchCase: _switch.cases)
		{
			if (!switchCase.value)
				continue; // default handled last
			mlir::Value caseValue = importLiteral(*switchCase.value);
			mlir::Value isMatch = m_builder.create<mlir::yul::EqOp>(loc(), scrutinee, caseValue);
			auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), isMatch);
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
			if (hasDefault)
			{
				mlir::Value one = constant(uint64_t(1));
				m_builder.create<mlir::yul::AssignOp>(loc(), matched, one);
			}
			importBlockInline(switchCase.body);
		}

		if (hasDefault)
			for (yul::Case const& switchCase: _switch.cases)
			{
				if (switchCase.value)
					continue;
				mlir::Value matchedNow = m_builder.create<mlir::yul::VarLoadOp>(loc(), matched);
				mlir::Value noMatch = m_builder.create<mlir::yul::IsZeroOp>(loc(), matchedNow);
				auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), noMatch);
				mlir::OpBuilder::InsertionGuard guard(m_builder);
				m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
				importBlockInline(switchCase.body);
			}
	}
};

} // anonymous namespace

mlir::OwningOpRef<mlir::ModuleOp>
solidity::mlirgen::importYulAST(yul::AST const& _ast, mlir::MLIRContext& _context, std::string& _error)
{
	_context.getOrLoadDialect<mlir::yul::YulDialect>();
	return Importer(_context, _ast.dialect()).run(_ast.root(), _error);
}

mlir::OwningOpRef<mlir::ModuleOp> solidity::mlirgen::importYulSource(
	std::string const& _sourceName, std::string const& _source, mlir::MLIRContext& _context, std::string& _error)
{
	yul::YulStack stack;
	if (!stack.parseAndAnalyze(_sourceName, _source))
	{
		_error = "libyul failed to parse/analyze the source";
		for (auto const& error: stack.errors())
			if (error->comment())
				_error += "\n  " + *error->comment();
		return nullptr;
	}
	std::shared_ptr<yul::Object> object = stack.parserResult();
	if (!object || !object->code())
	{
		_error = "no Yul object was produced";
		return nullptr;
	}
	if (!object->subObjects.empty())
	{
		_error = "objects with sub-objects are not supported yet (use importYulObjects)";
		return nullptr;
	}
	return importYulAST(*object->code(), _context, _error);
}

namespace
{

/// Object names are stored as string literals, quotes included.
std::string unquote(std::string _name)
{
	if (_name.size() >= 2 && _name.front() == '"' && _name.back() == '"')
		return _name.substr(1, _name.size() - 2);
	return _name;
}

/// Appends @a _object to @a _out in pre-order and returns its index, recording
/// each object's children so the tree can be rebuilt by the caller.
size_t collectObjects(
	yul::Object const& _object,
	std::vector<yul::Object const*>& _out,
	std::vector<std::vector<size_t>>& _children,
	std::vector<std::vector<solidity::mlirgen::ImportedDataSegment>>& _data)
{
	size_t const index = _out.size();
	_out.push_back(&_object);
	_children.emplace_back();
	_data.emplace_back();

	for (auto const& sub: _object.subObjects)
	{
		if (auto const* child = dynamic_cast<yul::Object const*>(sub.get()))
		{
			// Recurse first: the call grows _children, so subscripting it in
			// the same expression would hold a reference across a realloc.
			size_t const childIndex = collectObjects(*child, _out, _children, _data);
			_children[index].push_back(childIndex);
		}
		else if (auto const* data = dynamic_cast<yul::Data const*>(sub.get()))
			_data[index].push_back({unquote(data->name), data->data});
	}
	return index;
}

} // anonymous namespace

std::vector<solidity::mlirgen::ImportedObject> solidity::mlirgen::importYulObjects(
	std::string const& _sourceName, std::string const& _source, mlir::MLIRContext& _context, std::string& _parseError)
{
	std::vector<ImportedObject> results;

	yul::YulStack stack;
	if (!stack.parseAndAnalyze(_sourceName, _source))
	{
		_parseError = "libyul failed to parse/analyze the source";
		for (auto const& error: stack.errors())
			if (error->comment())
				_parseError += "\n  " + *error->comment();
		return results;
	}
	std::shared_ptr<yul::Object> root = stack.parserResult();
	if (!root)
	{
		_parseError = "no Yul object was produced";
		return results;
	}

	std::vector<yul::Object const*> objects;
	std::vector<std::vector<size_t>> children;
	std::vector<std::vector<ImportedDataSegment>> data;
	collectObjects(*root, objects, children, data);

	for (size_t index = 0; index < objects.size(); ++index)
	{
		yul::Object const* object = objects[index];
		ImportedObject entry;
		entry.name = unquote(object->name);
		entry.subObjects = children[index];
		entry.dataSegments = data[index];
		if (!object->code())
			entry.error = "object has no code";
		else
			entry.module = importYulAST(*object->code(), _context, entry.error);
		results.push_back(std::move(entry));
	}
	return results;
}
