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
 * Yul-text emitter implementation.
 *
 * Emission model: one `let` per SSA value (each yul.* builtin op prints as
 * the identically-named Yul builtin call - the 1:1 design of ADR-003),
 * mutable yul.var ops print as Yul variables, region control flow prints as
 * Yul `if`/`for`. yul.for's condition region is emitted with the standard
 * `for {} 1 {post} { <cond> if iszero(c) { break } <body> }` normalization,
 * which preserves evaluation order and continue/break semantics.
 */

#include "YulTextEmitter.h"

#include "YulDialect.h"
#include "YulOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#pragma GCC diagnostic pop

#include <cassert>
#include <string>
#include <vector>

using namespace mlir;

namespace
{

/// True for characters allowed in Yul identifiers.
bool isYulIdentChar(char _c)
{
	return (_c >= 'a' && _c <= 'z') || (_c >= 'A' && _c <= 'Z') || (_c >= '0' && _c <= '9') || _c == '_' || _c == '$'
		   || _c == '.';
}

std::string sanitizeIdentifier(llvm::StringRef _name)
{
	std::string result;
	for (char c: _name)
		result += isYulIdentChar(c) ? c : '_';
	if (result.empty() || (result[0] >= '0' && result[0] <= '9'))
		result = "f_" + result;
	return result;
}

class Emitter
{
public:
	std::string run(mlir::ModuleOp _module)
	{
		m_out.clear();
		line("{");
		++m_indent;
		for (mlir::Operation& op: _module.getBody()->getOperations())
			emitStatement(&op);
		--m_indent;
		line("}");
		return m_out;
	}

private:
	std::string m_out;
	unsigned m_indent = 0;
	unsigned m_counter = 0;
	llvm::DenseMap<mlir::Value, std::string> m_names;
	std::vector<std::string> m_resultVars;

	void line(std::string const& _s)
	{
		for (unsigned i = 0; i < m_indent; ++i)
			m_out += "    ";
		m_out += _s;
		m_out += "\n";
	}

	std::string freshName(char const* _prefix) { return std::string(_prefix) + std::to_string(m_counter++); }

	std::string nameOf(mlir::Value _v)
	{
		auto it = m_names.find(_v);
		assert(it != m_names.end() && "use of a value without a Yul name");
		return it == m_names.end() ? std::string("0") : it->second;
	}

	static std::string literal(llvm::APInt const& _value)
	{
		llvm::SmallString<80> str;
		_value.toStringUnsigned(str, 16);
		return "0x" + std::string(str.str());
	}

	/// Builtin call expression: the op mnemonic IS the Yul builtin name.
	std::string callExpr(mlir::Operation* _op)
	{
		std::string expr = _op->getName().stripDialect().str();
		expr += "(";
		bool first = true;
		for (mlir::Value operand: _op->getOperands())
		{
			if (!first)
				expr += ", ";
			first = false;
			expr += nameOf(operand);
		}
		expr += ")";
		return expr;
	}

	void emitBlock(mlir::Block& _block)
	{
		for (mlir::Operation& op: _block.getOperations())
			emitStatement(&op);
	}

	void emitFunction(mlir::yul::FuncOp _func)
	{
		auto savedResultVars = m_resultVars;

		std::string header = "function " + sanitizeIdentifier(_func.getSymName()) + "(";
		mlir::FunctionType type = _func.getFunctionType();

		bool first = true;
		if (!_func.getBody().empty())
			for (mlir::BlockArgument arg: _func.getBody().front().getArguments())
			{
				if (!first)
					header += ", ";
				first = false;
				std::string name = freshName("a");
				m_names[arg] = name;
				header += name;
			}

		header += ")";

		m_resultVars.clear();
		if (type.getNumResults() > 0)
		{
			header += " -> ";
			for (unsigned i = 0; i < type.getNumResults(); ++i)
			{
				if (i > 0)
					header += ", ";
				std::string name = freshName("r");
				m_resultVars.push_back(name);
				header += name;
			}
		}

		header += " {";
		line(header);
		++m_indent;
		if (!_func.getBody().empty())
			emitBlock(_func.getBody().front());
		--m_indent;
		line("}");

		m_resultVars = savedResultVars;
	}

	void emitIf(mlir::yul::IfOp _if)
	{
		line("if " + nameOf(_if.getCondition()) + " {");
		++m_indent;
		if (!_if.getThenRegion().empty())
			emitBlock(_if.getThenRegion().front());
		--m_indent;
		line("}");
	}

	void emitFor(mlir::yul::ForOp _for)
	{
		// for {} 1 { <post> } { <cond stmts>; if iszero(c) { break }; <body> }
		// preserves Yul's evaluation order: cond before every iteration,
		// continue jumps to post, post runs before the next cond check.
		line("for { } 1 {");
		++m_indent;
		if (!_for.getPostRegion().empty())
			emitBlock(_for.getPostRegion().front());
		--m_indent;
		line("} {");
		++m_indent;

		std::string condName;
		if (!_for.getCondRegion().empty())
			for (mlir::Operation& op: _for.getCondRegion().front().getOperations())
			{
				if (auto cond = llvm::dyn_cast<mlir::yul::ConditionOp>(&op))
					condName = nameOf(cond.getCondition());
				else
					emitStatement(&op);
			}
		if (!condName.empty())
			line("if iszero(" + condName + ") { break }");

		if (!_for.getBodyRegion().empty())
			emitBlock(_for.getBodyRegion().front());
		--m_indent;
		line("}");
	}

	void emitStatement(mlir::Operation* _op)
	{
		// Structure and control flow.
		if (auto func = llvm::dyn_cast<mlir::yul::FuncOp>(_op))
			return emitFunction(func);
		if (auto ifOp = llvm::dyn_cast<mlir::yul::IfOp>(_op))
			return emitIf(ifOp);
		if (auto forOp = llvm::dyn_cast<mlir::yul::ForOp>(_op))
			return emitFor(forOp);
		if (llvm::isa<mlir::yul::BreakOp>(_op))
			return line("break");
		if (llvm::isa<mlir::yul::ContinueOp>(_op))
			return line("continue");
		if (auto leave = llvm::dyn_cast<mlir::yul::LeaveOp>(_op))
		{
			unsigned n = std::min<unsigned>(leave.getOperands().size(), m_resultVars.size());
			for (unsigned i = 0; i < n; ++i)
				line(m_resultVars[i] + " := " + nameOf(leave.getOperand(i)));
			return line("leave");
		}
		if (llvm::isa<mlir::yul::ConditionOp>(_op))
			return; // handled by emitFor

		// Literals and variables.
		if (auto constOp = llvm::dyn_cast<mlir::yul::ConstOp>(_op))
		{
			std::string name = freshName("v");
			m_names[constOp.getResult()] = name;
			return line("let " + name + " := " + literal(constOp.getValue()));
		}
		if (auto varOp = llvm::dyn_cast<mlir::yul::VarOp>(_op))
		{
			std::string name = freshName("u");
			m_names[varOp.getRef()] = name;
			return line("let " + name + " := " + nameOf(varOp.getInit()));
		}
		if (auto assign = llvm::dyn_cast<mlir::yul::AssignOp>(_op))
			return line(nameOf(assign.getVar()) + " := " + nameOf(assign.getValue()));
		if (auto load = llvm::dyn_cast<mlir::yul::VarLoadOp>(_op))
		{
			// Snapshot the current value: later reassignments must not be
			// visible through this SSA value.
			std::string name = freshName("v");
			m_names[load.getResult()] = name;
			return line("let " + name + " := " + nameOf(load.getVar()));
		}

		// User function calls (possibly multi-result).
		if (auto call = llvm::dyn_cast<mlir::yul::FuncCallOp>(_op))
		{
			std::string expr = sanitizeIdentifier(call.getCallee()) + "(";
			bool first = true;
			for (mlir::Value operand: call.getOperands())
			{
				if (!first)
					expr += ", ";
				first = false;
				expr += nameOf(operand);
			}
			expr += ")";

			if (call.getNumResults() == 0)
				return line(expr);
			std::string lhs;
			for (mlir::Value result: call.getResults())
			{
				if (!lhs.empty())
					lhs += ", ";
				std::string name = freshName("v");
				m_names[result] = name;
				lhs += name;
			}
			return line("let " + lhs + " := " + expr);
		}

		// Object builtins carrying attributes.
		if (auto dataOffset = llvm::dyn_cast<mlir::yul::DataOffsetOp>(_op))
		{
			std::string name = freshName("v");
			m_names[dataOffset.getResult()] = name;
			return line("let " + name + " := dataoffset(\"" + dataOffset.getSegment().str() + "\")");
		}
		if (auto dataSize = llvm::dyn_cast<mlir::yul::DataSizeOp>(_op))
		{
			std::string name = freshName("v");
			m_names[dataSize.getResult()] = name;
			return line("let " + name + " := datasize(\"" + dataSize.getSegment().str() + "\")");
		}
		if (auto memoryGuard = llvm::dyn_cast<mlir::yul::MemoryGuardOp>(_op))
		{
			std::string name = freshName("v");
			m_names[memoryGuard.getResult()] = name;
			return line("let " + name + " := memoryguard(" + literal(memoryGuard.getSize()) + ")");
		}
		if (auto loadImmutable = llvm::dyn_cast<mlir::yul::LoadImmutableOp>(_op))
		{
			std::string name = freshName("v");
			m_names[loadImmutable.getResult()] = name;
			return line("let " + name + " := loadimmutable(\"" + loadImmutable.getImmutableName().str() + "\")");
		}
		if (auto setImmutable = llvm::dyn_cast<mlir::yul::SetImmutableOp>(_op))
			return line(
				"setimmutable(" + nameOf(setImmutable.getOffset()) + ", \"" + setImmutable.getImmutableName().str()
				+ "\", " + nameOf(setImmutable.getValue()) + ")");

		// Generic builtin: the op mnemonic is the Yul builtin name (ADR-003).
		assert(_op->getDialect() && _op->getDialect()->getNamespace() == "yul" && "non-yul op in Yul text emission");

		if (_op->getNumResults() == 0)
			return line(callExpr(_op));

		assert(_op->getNumResults() == 1 && "yul builtins have at most one result");
		mlir::Value result = _op->getResult(0);
		if (result.use_empty())
			// Unused result: pop() keeps the emission valid regardless of
			// the op's effects (dead pure ops are cleaned up by libyul's
			// optimizer / a later canonicalization pass).
			return line("pop(" + callExpr(_op) + ")");
		std::string name = freshName("v");
		m_names[result] = name;
		line("let " + name + " := " + callExpr(_op));
	}
};

} // anonymous namespace

std::string solidity::mlirgen::emitYulText(mlir::ModuleOp _module)
{
	return Emitter().run(_module);
}
