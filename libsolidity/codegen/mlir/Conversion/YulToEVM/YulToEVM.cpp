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
 * YulToEVM conversion implementation.
 *
 * Mutable variables are promoted to SSA *during* control-flow flattening
 * (structured-CF SSA construction): the converter tracks the current value
 * of every yul.var while walking; at control-flow merge points the variables
 * assigned inside the construct become block arguments -
 *
 *  - yul.if:  the continuation block receives one argument per assigned
 *    variable (then-edge passes the then-end values, the false edge passes
 *    the pre-if values);
 *  - yul.for: the condition block carries the loop-carried variables, the
 *    post and exit blocks receive per-edge values from body-end, continue,
 *    break, and the condition's false edge.
 *
 * Module-level (object) code - real via-ir objects are top-level statements
 * plus functions - is synthesized into a `func.func @__entry()` whose end
 * falls through to evm.stop, matching EVM's implicit halt.
 */

#include "YulToEVM.h"

#include "EVMDialect.h"
#include "EVMOps.h"
#include "YulDialect.h"
#include "YulOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Inliner.h"
#pragma GCC diagnostic pop

#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace mlir;
using namespace solidity::mlirgen;

//===----------------------------------------------------------------------===//
// Block-local variable promotion (mini mem2reg; kept as a cleanup utility -
// the converter below no longer requires it)
//===----------------------------------------------------------------------===//

unsigned solidity::mlirgen::promoteBlockLocalVars(mlir::ModuleOp _module)
{
	llvm::DenseSet<mlir::Operation*> promotable;
	_module->walk([&](mlir::yul::VarOp varOp) {
		for (mlir::Operation* user: varOp.getRef().getUsers())
			if (user->getBlock() != varOp->getBlock())
				return;
		promotable.insert(varOp.getOperation());
	});

	_module->walk([&](mlir::Block* block) {
		llvm::DenseMap<mlir::Value, mlir::Value> currentValue;
		llvm::SmallVector<mlir::Operation*, 8> toErase;
		for (mlir::Operation& op: *block)
		{
			if (auto varOp = llvm::dyn_cast<mlir::yul::VarOp>(&op))
			{
				if (promotable.contains(&op))
					currentValue[varOp.getRef()] = varOp.getInit();
			}
			else if (auto load = llvm::dyn_cast<mlir::yul::VarLoadOp>(&op))
			{
				auto it = currentValue.find(load.getVar());
				if (it != currentValue.end())
				{
					load.getResult().replaceAllUsesWith(it->second);
					toErase.push_back(&op);
				}
			}
			else if (auto assign = llvm::dyn_cast<mlir::yul::AssignOp>(&op))
			{
				auto it = currentValue.find(assign.getVar());
				if (it != currentValue.end())
				{
					it->second = assign.getValue();
					toErase.push_back(&op);
				}
			}
		}
		for (mlir::Operation* op: toErase)
			op->erase();
		for (auto const& entry: currentValue)
			if (mlir::Operation* varOp = entry.first.getDefiningOp(); varOp && entry.first.use_empty())
				varOp->erase();
	});

	unsigned remaining = 0;
	_module->walk([&](mlir::yul::VarOp) { ++remaining; });
	return remaining;
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

namespace
{

struct ConversionError
{
	std::string message;
};

class Converter
{
public:
	explicit Converter(mlir::MLIRContext& _ctx): m_builder(&_ctx) {}

	mlir::OwningOpRef<mlir::ModuleOp> run(mlir::ModuleOp _src, std::string& _error)
	{
		mlir::OwningOpRef<mlir::ModuleOp> dst = mlir::ModuleOp::create(m_builder.getUnknownLoc());
		try
		{
			m_dstModule = *dst;

			// Functions first (they are position-independent declarations).
			for (mlir::Operation& op: _src.getBody()->getOperations())
				if (auto funcOp = llvm::dyn_cast<mlir::yul::FuncOp>(&op))
				{
					m_builder.setInsertionPointToEnd(dst->getBody());
					convertFunction(funcOp);
				}

			// Module-level object code becomes @__entry (implicit stop).
			bool hasTopLevelCode = false;
			for (mlir::Operation& op: _src.getBody()->getOperations())
				if (!llvm::isa<mlir::yul::FuncOp>(&op))
					hasTopLevelCode = true;
			if (hasTopLevelCode)
			{
				m_builder.setInsertionPointToEnd(dst->getBody());
				auto entry = m_builder.create<mlir::func::FuncOp>(
					m_builder.getUnknownLoc(), "__entry", mlir::FunctionType::get(m_builder.getContext(), {}, {}));
				m_currentFunc = entry;
				mlir::Block* block = entry.addEntryBlock();
				mlir::OpBuilder::InsertionGuard guard(m_builder);
				m_builder.setInsertionPointToEnd(block);
				bool terminated = false;
				for (mlir::Operation& op: _src.getBody()->getOperations())
				{
					if (llvm::isa<mlir::yul::FuncOp>(&op))
						continue;
					if (convertOp(op))
					{
						terminated = true;
						break; // anything further at top level is unreachable
					}
				}
				if (!terminated)
					m_builder.create<mlir::evm::StopOp>(loc()); // implicit EVM halt
			}
		}
		catch (ConversionError const& _e)
		{
			_error = _e.message;
			return nullptr;
		}
		if (failed(mlir::verify(*dst)))
		{
			if (std::getenv("SOL_MLIR_DUMP_INVALID"))
				dst->print(llvm::errs());
			_error = "converted module failed MLIR verification";
			return nullptr;
		}
		return dst;
	}

private:
	mlir::OpBuilder m_builder;
	mlir::ModuleOp m_dstModule;
	llvm::DenseMap<mlir::Value, mlir::Value> m_map;
	// Structured-CF SSA construction: current value of every mutable var.
	llvm::DenseMap<mlir::Value, mlir::Value> m_curDef;

	struct LoopCtx
	{
		mlir::Block* post;
		mlir::Block* exit;
		llvm::SmallVector<mlir::Value, 4> vars; // loop-carried variable refs
	};
	std::vector<LoopCtx> m_loops;
	mlir::func::FuncOp m_currentFunc;

	[[noreturn]] static void fail(std::string _message) { throw ConversionError{std::move(_message)}; }

	mlir::Location loc() { return m_builder.getUnknownLoc(); }
	mlir::IntegerType wordType() { return m_builder.getIntegerType(256); }

	mlir::Value mapped(mlir::Value _v)
	{
		auto it = m_map.find(_v);
		if (it == m_map.end())
			fail("use of an unmapped value during conversion");
		return it->second;
	}

	mlir::Value currentDef(mlir::Value _ref)
	{
		auto it = m_curDef.find(_ref);
		if (it == m_curDef.end())
			fail("read of a variable before its definition");
		return it->second;
	}

	mlir::Value wordConstant(llvm::APInt _value)
	{
		return m_builder.create<mlir::arith::ConstantOp>(loc(), mlir::IntegerAttr::get(wordType(), std::move(_value)));
	}

	mlir::Value boolToWord(mlir::Value _i1)
	{
		return m_builder.create<mlir::arith::ExtUIOp>(loc(), wordType(), _i1);
	}

	mlir::Value wordToBool(mlir::Value _word)
	{
		mlir::Value zero = wordConstant(llvm::APInt(256, 0));
		return m_builder.create<mlir::arith::CmpIOp>(loc(), mlir::arith::CmpIPredicate::ne, _word, zero);
	}

	mlir::Block* newBlock(unsigned _numArgs = 0)
	{
		mlir::Block* block = new mlir::Block();
		for (unsigned i = 0; i < _numArgs; ++i)
			block->addArgument(wordType(), loc());
		m_currentFunc.getBody().push_back(block);
		return block;
	}

	/// Variables defined OUTSIDE _scope but assigned inside it - the merge
	/// set for the construct's continuation.
	llvm::SmallVector<mlir::Value, 4> assignedOuterVars(mlir::Operation* _scope)
	{
		llvm::SetVector<mlir::Value> vars;
		_scope->walk([&](mlir::yul::AssignOp assign) {
			mlir::Value ref = assign.getVar();
			mlir::Operation* def = ref.getDefiningOp();
			if (def && !_scope->isAncestor(def))
				vars.insert(ref);
		});
		return llvm::SmallVector<mlir::Value, 4>(vars.begin(), vars.end());
	}

	llvm::SmallVector<mlir::Value, 4> currentValues(llvm::ArrayRef<mlir::Value> _vars)
	{
		llvm::SmallVector<mlir::Value, 4> values;
		for (mlir::Value var: _vars)
			values.push_back(currentDef(var));
		return values;
	}

	void convertFunction(mlir::yul::FuncOp _func)
	{
		auto savedCurDef = std::move(m_curDef);
		m_curDef.clear();
		auto savedFunc = m_currentFunc;
		auto savedLoops = std::move(m_loops);
		m_loops.clear();

		auto dstFunc = m_builder.create<mlir::func::FuncOp>(loc(), _func.getSymName(), _func.getFunctionType());
		dstFunc.setPrivate();
		m_currentFunc = dstFunc;

		mlir::Block* entry = dstFunc.addEntryBlock();
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToEnd(entry);
			bool terminated = false;
			if (!_func.getBody().empty())
			{
				mlir::Block& srcEntry = _func.getBody().front();
				for (unsigned i = 0; i < srcEntry.getNumArguments(); ++i)
					m_map[srcEntry.getArgument(i)] = entry->getArgument(i);
				terminated = convertBlockOps(srcEntry);
			}
			if (!terminated)
			{
				if (_func.getFunctionType().getNumResults() == 0)
					m_builder.create<mlir::func::ReturnOp>(loc());
				else
					fail("function '" + _func.getSymName().str() + "' falls through with results");
			}
		}

		m_loops = std::move(savedLoops);
		m_currentFunc = savedFunc;
		m_curDef = std::move(savedCurDef);
	}

	/// Converts the ops of a structured yul block into the current insertion
	/// block (which changes as control flow is flattened). @returns true if
	/// conversion produced a terminator for the current path.
	bool convertBlockOps(mlir::Block& _src)
	{
		for (mlir::Operation& op: _src.getOperations())
			if (convertOp(op))
				return true;
		return false;
	}

	/// @returns true if the op terminated the current block.
	bool convertOp(mlir::Operation& _op)
	{
		// Mutable variables: pure bookkeeping, no IR emitted (SSA
		// construction happens at the merge points).
		if (auto varOp = llvm::dyn_cast<mlir::yul::VarOp>(&_op))
		{
			m_curDef[varOp.getRef()] = mapped(varOp.getInit());
			return false;
		}
		if (auto assign = llvm::dyn_cast<mlir::yul::AssignOp>(&_op))
		{
			m_curDef[assign.getVar()] = mapped(assign.getValue());
			return false;
		}
		if (auto load = llvm::dyn_cast<mlir::yul::VarLoadOp>(&_op))
		{
			m_map[load.getResult()] = currentDef(load.getVar());
			return false;
		}

		// Literals.
		if (auto constOp = llvm::dyn_cast<mlir::yul::ConstOp>(&_op))
		{
			m_map[constOp.getResult()] = wordConstant(constOp.getValue());
			return false;
		}

		// Upstream-exact arithmetic (ADR-003: arith reuse at this rung).
		if (convertArithExact(_op))
			return false;
		if (convertComparison(_op))
			return false;

		if (auto notOp = llvm::dyn_cast<mlir::yul::NotOp>(&_op))
		{
			mlir::Value allOnes = wordConstant(llvm::APInt::getAllOnes(256));
			m_map[notOp.getResult()] =
				m_builder.create<mlir::arith::XOrIOp>(loc(), mapped(notOp.getOperand()), allOnes);
			return false;
		}

		// memoryguard(size) is semantically its argument.
		if (auto memoryGuard = llvm::dyn_cast<mlir::yul::MemoryGuardOp>(&_op))
		{
			// Kept as an op: the value is the start of the contract's heap, and
			// a backend placing memory of its own below it has to move the
			// boundary up. Folding it to a literal loses that meaning.
			m_dstModule->setAttr("evm.memory_guard", memoryGuard.getSizeAttr());
			m_map[memoryGuard.getResult()]
				= m_builder.create<mlir::evm::MemoryGuardOp>(loc(), memoryGuard.getSizeAttr());
			return false;
		}

		// Object data segments and immutables carry string attributes: map
		// onto the corresponding evm.* ops explicitly.
		if (auto dataOffset = llvm::dyn_cast<mlir::yul::DataOffsetOp>(&_op))
		{
			m_map[dataOffset.getResult()] = createAttrOp("evm.dataoffset", "segment", dataOffset.getSegment());
			return false;
		}
		if (auto dataSize = llvm::dyn_cast<mlir::yul::DataSizeOp>(&_op))
		{
			m_map[dataSize.getResult()] = createAttrOp("evm.datasize", "segment", dataSize.getSegment());
			return false;
		}
		if (auto linkerSymbol = llvm::dyn_cast<mlir::yul::LinkerSymbolOp>(&_op))
		{
			m_map[linkerSymbol.getResult()] = createAttrOp("evm.linkersymbol", "symbol", linkerSymbol.getSymbol());
			return false;
		}
		if (auto loadImmutable = llvm::dyn_cast<mlir::yul::LoadImmutableOp>(&_op))
		{
			m_map[loadImmutable.getResult()] =
				createAttrOp("evm.loadimmutable", "immutable_name", loadImmutable.getImmutableName());
			return false;
		}
		if (auto setImmutable = llvm::dyn_cast<mlir::yul::SetImmutableOp>(&_op))
		{
			mlir::OperationState state(loc(), "evm.setimmutable");
			state.addOperands({mapped(setImmutable.getOffset()), mapped(setImmutable.getValue())});
			state.addAttribute("immutable_name", m_builder.getStringAttr(setImmutable.getImmutableName()));
			m_builder.create(state);
			return false;
		}

		// Control flow.
		if (auto ifOp = llvm::dyn_cast<mlir::yul::IfOp>(&_op))
		{
			convertIf(ifOp);
			return false;
		}
		if (auto forOp = llvm::dyn_cast<mlir::yul::ForOp>(&_op))
		{
			convertFor(forOp);
			return false;
		}
		if (llvm::isa<mlir::yul::BreakOp>(&_op))
		{
			if (m_loops.empty())
				fail("break outside of a loop");
			m_builder.create<mlir::cf::BranchOp>(loc(), m_loops.back().exit, currentValues(m_loops.back().vars));
			return true;
		}
		if (llvm::isa<mlir::yul::ContinueOp>(&_op))
		{
			if (m_loops.empty())
				fail("continue outside of a loop");
			m_builder.create<mlir::cf::BranchOp>(loc(), m_loops.back().post, currentValues(m_loops.back().vars));
			return true;
		}
		if (auto leave = llvm::dyn_cast<mlir::yul::LeaveOp>(&_op))
		{
			llvm::SmallVector<mlir::Value, 2> results;
			for (mlir::Value operand: leave.getOperands())
				results.push_back(mapped(operand));
			m_builder.create<mlir::func::ReturnOp>(loc(), results);
			return true;
		}

		// Calls.
		if (auto call = llvm::dyn_cast<mlir::yul::FuncCallOp>(&_op))
		{
			llvm::SmallVector<mlir::Value, 8> args;
			for (mlir::Value operand: call.getOperands())
				args.push_back(mapped(operand));
			llvm::SmallVector<mlir::Type, 2> resultTypes(call.getNumResults(), wordType());
			auto dstCall = m_builder.create<mlir::func::CallOp>(loc(), call.getCallee(), resultTypes, args);
			for (unsigned i = 0; i < call.getNumResults(); ++i)
				m_map[call.getResult(i)] = dstCall.getResult(i);
			return false;
		}

		// Nested function definitions surface at module level in func-land.
		if (auto funcOp = llvm::dyn_cast<mlir::yul::FuncOp>(&_op))
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToEnd(m_dstModule.getBody());
			convertFunction(funcOp);
			return false;
		}

		if (llvm::isa<mlir::yul::ConditionOp>(&_op))
			fail("yul.condition outside a for-loop condition region");

		// Everything else: the identically-named evm.* op (terminators included).
		return convertGenericEvmOp(_op);
	}

	mlir::Value createAttrOp(llvm::StringRef _opName, llvm::StringRef _attrName, llvm::StringRef _attrValue)
	{
		mlir::OperationState state(loc(), _opName);
		state.addAttribute(_attrName, m_builder.getStringAttr(_attrValue));
		state.addTypes(wordType());
		return m_builder.create(state)->getResult(0);
	}

	bool convertArithExact(mlir::Operation& _op)
	{
		mlir::Value result;
		if (auto addOp = llvm::dyn_cast<mlir::yul::AddOp>(&_op))
			result = m_builder.create<mlir::arith::AddIOp>(loc(), mapped(addOp.getLhs()), mapped(addOp.getRhs()));
		else if (auto subOp = llvm::dyn_cast<mlir::yul::SubOp>(&_op))
			result = m_builder.create<mlir::arith::SubIOp>(loc(), mapped(subOp.getLhs()), mapped(subOp.getRhs()));
		else if (auto mulOp = llvm::dyn_cast<mlir::yul::MulOp>(&_op))
			result = m_builder.create<mlir::arith::MulIOp>(loc(), mapped(mulOp.getLhs()), mapped(mulOp.getRhs()));
		else if (auto andOp = llvm::dyn_cast<mlir::yul::AndOp>(&_op))
			result = m_builder.create<mlir::arith::AndIOp>(loc(), mapped(andOp.getLhs()), mapped(andOp.getRhs()));
		else if (auto orOp = llvm::dyn_cast<mlir::yul::OrOp>(&_op))
			result = m_builder.create<mlir::arith::OrIOp>(loc(), mapped(orOp.getLhs()), mapped(orOp.getRhs()));
		else if (auto xorOp = llvm::dyn_cast<mlir::yul::XorOp>(&_op))
			result = m_builder.create<mlir::arith::XOrIOp>(loc(), mapped(xorOp.getLhs()), mapped(xorOp.getRhs()));
		else
			return false;
		m_map[_op.getResult(0)] = result;
		return true;
	}

	bool convertComparison(mlir::Operation& _op)
	{
		mlir::arith::CmpIPredicate predicate;
		if (llvm::isa<mlir::yul::LtOp>(&_op))
			predicate = mlir::arith::CmpIPredicate::ult;
		else if (llvm::isa<mlir::yul::GtOp>(&_op))
			predicate = mlir::arith::CmpIPredicate::ugt;
		else if (llvm::isa<mlir::yul::SLtOp>(&_op))
			predicate = mlir::arith::CmpIPredicate::slt;
		else if (llvm::isa<mlir::yul::SGtOp>(&_op))
			predicate = mlir::arith::CmpIPredicate::sgt;
		else if (llvm::isa<mlir::yul::EqOp>(&_op))
			predicate = mlir::arith::CmpIPredicate::eq;
		else if (llvm::isa<mlir::yul::IsZeroOp>(&_op))
		{
			mlir::Value zero = wordConstant(llvm::APInt(256, 0));
			mlir::Value cmp = m_builder.create<mlir::arith::CmpIOp>(
				loc(), mlir::arith::CmpIPredicate::eq, mapped(_op.getOperand(0)), zero);
			m_map[_op.getResult(0)] = boolToWord(cmp);
			return true;
		}
		else
			return false;

		mlir::Value cmp = m_builder.create<mlir::arith::CmpIOp>(
			loc(), predicate, mapped(_op.getOperand(0)), mapped(_op.getOperand(1)));
		m_map[_op.getResult(0)] = boolToWord(cmp);
		return true;
	}

	void convertIf(mlir::yul::IfOp _if)
	{
		mlir::Value cond = wordToBool(mapped(_if.getCondition()));

		llvm::SmallVector<mlir::Value, 4> vars = assignedOuterVars(_if.getOperation());
		llvm::SmallVector<mlir::Value, 4> preValues = currentValues(vars);

		mlir::Block* thenBlock = newBlock();
		mlir::Block* contBlock = newBlock(vars.size());
		m_builder.create<mlir::cf::CondBranchOp>(
			loc(), cond, thenBlock, mlir::ValueRange{}, contBlock, preValues);

		m_builder.setInsertionPointToEnd(thenBlock);
		bool terminated = _if.getThenRegion().empty() ? false : convertBlockOps(_if.getThenRegion().front());
		if (!terminated)
			m_builder.create<mlir::cf::BranchOp>(loc(), contBlock, currentValues(vars));

		for (unsigned i = 0; i < vars.size(); ++i)
			m_curDef[vars[i]] = contBlock->getArgument(i);
		m_builder.setInsertionPointToEnd(contBlock);
	}

	void convertFor(mlir::yul::ForOp _for)
	{
		llvm::SmallVector<mlir::Value, 4> vars = assignedOuterVars(_for.getOperation());

		mlir::Block* condBlock = newBlock(vars.size());
		mlir::Block* bodyBlock = newBlock();
		mlir::Block* postBlock = newBlock(vars.size());
		mlir::Block* exitBlock = newBlock(vars.size());

		m_builder.create<mlir::cf::BranchOp>(loc(), condBlock, currentValues(vars));

		// Condition block: loop-carried values arrive as block arguments.
		m_builder.setInsertionPointToEnd(condBlock);
		for (unsigned i = 0; i < vars.size(); ++i)
			m_curDef[vars[i]] = condBlock->getArgument(i);

		bool sawCondition = false;
		if (!_for.getCondRegion().empty())
			for (mlir::Operation& op: _for.getCondRegion().front().getOperations())
			{
				if (auto condition = llvm::dyn_cast<mlir::yul::ConditionOp>(&op))
				{
					mlir::Value cond = wordToBool(mapped(condition.getCondition()));
					m_builder.create<mlir::cf::CondBranchOp>(
						loc(), cond, bodyBlock, mlir::ValueRange{}, exitBlock, currentValues(vars));
					sawCondition = true;
					break;
				}
				if (convertOp(op))
					fail("terminator inside a for-loop condition region");
			}
		if (!sawCondition)
			m_builder.create<mlir::cf::BranchOp>(loc(), bodyBlock); // infinite loop unless the body breaks

		m_loops.push_back(LoopCtx{postBlock, exitBlock, vars});

		m_builder.setInsertionPointToEnd(bodyBlock);
		bool bodyTerminated = _for.getBodyRegion().empty() ? false : convertBlockOps(_for.getBodyRegion().front());
		if (!bodyTerminated)
			m_builder.create<mlir::cf::BranchOp>(loc(), postBlock, currentValues(vars));

		m_builder.setInsertionPointToEnd(postBlock);
		for (unsigned i = 0; i < vars.size(); ++i)
			m_curDef[vars[i]] = postBlock->getArgument(i);
		bool postTerminated = _for.getPostRegion().empty() ? false : convertBlockOps(_for.getPostRegion().front());
		if (!postTerminated)
			m_builder.create<mlir::cf::BranchOp>(loc(), condBlock, currentValues(vars));

		m_loops.pop_back();

		for (unsigned i = 0; i < vars.size(); ++i)
			m_curDef[vars[i]] = exitBlock->getArgument(i);
		m_builder.setInsertionPointToEnd(exitBlock);
	}

	/// Generic 1:1 mapping yul.X -> evm.X. @returns true if X is a terminator.
	bool convertGenericEvmOp(mlir::Operation& _op)
	{
		if (_op.getDialect()->getNamespace() != "yul")
			fail("unexpected non-yul op: " + _op.getName().getStringRef().str());

		std::string evmName = "evm." + _op.getName().stripDialect().str();
		std::optional<mlir::RegisteredOperationName> registered =
			mlir::RegisteredOperationName::lookup(evmName, m_builder.getContext());
		if (!registered)
			fail("no evm dialect equivalent for op '" + _op.getName().getStringRef().str() + "'");

		mlir::OperationState state(loc(), *registered);
		for (mlir::Value operand: _op.getOperands())
			state.addOperands(mapped(operand));
		for (unsigned i = 0; i < _op.getNumResults(); ++i)
			state.addTypes(wordType());
		mlir::Operation* dstOp = m_builder.create(state);
		for (unsigned i = 0; i < _op.getNumResults(); ++i)
			m_map[_op.getResult(i)] = dstOp->getResult(i);

		return dstOp->hasTrait<mlir::OpTrait::IsTerminator>();
	}
};

/// Inline single-use helpers and tiny helpers with at most three call sites.
///
/// This backend's outlined calling convention materializes a static frame,
/// copies every argument/result through memory, and performs two jumps. When a
/// helper has only one caller, keeping both that machinery and a separate copy
/// of the body cannot reduce bytecode. Helpers with no more than two nontrivial
/// operations also remain profitable at up to three sites. The unrestricted
/// upstream inliner is deliberately too aggressive for solc-generated Yul, so
/// these bounds are the backend-specific cost model.
void inlineSingleUseFunctions(mlir::ModuleOp _module)
{
	for (;;)
	{
		std::map<std::string, mlir::func::FuncOp> functions;
		std::map<std::string, std::vector<mlir::func::CallOp>> calls;
		std::map<std::string, std::vector<std::string>> callees;
		std::map<std::string, unsigned> nontrivialOpCounts;
		std::set<std::string> functionsWithMachineTerminators;
		for (mlir::func::FuncOp func: _module.getOps<mlir::func::FuncOp>())
		{
			std::string const functionName = func.getName().str();
			functions.emplace(functionName, func);
			func.walk([&](mlir::Operation* op) {
				if (!llvm::isa<mlir::arith::ConstantOp, mlir::func::ReturnOp>(op))
					++nontrivialOpCounts[functionName];
				if (op->getDialect() && op->getDialect()->getNamespace() == "evm"
					&& op->hasTrait<mlir::OpTrait::IsTerminator>())
					functionsWithMachineTerminators.insert(functionName);
				if (auto call = llvm::dyn_cast<mlir::func::CallOp>(op))
				{
					std::string const callee = call.getCallee().str();
					calls[callee].push_back(call);
					callees[functionName].push_back(callee);
				}
			});
		}

		auto reaches = [&](std::string const& from, std::string const& target) {
			std::set<std::string> seen;
			std::vector<std::string> worklist = callees[from];
			while (!worklist.empty())
			{
				std::string current = std::move(worklist.back());
				worklist.pop_back();
				if (current == target)
					return true;
				if (!seen.insert(current).second)
					continue;
				for (std::string const& callee: callees[current])
					worklist.push_back(callee);
			}
			return false;
		};

		mlir::func::CallOp candidate;
		mlir::func::FuncOp callee;
		bool eraseCallee = false;
		for (auto const& [name, sites]: calls)
		{
			if (sites.empty() || sites.size() > 3)
				continue;
			auto found = functions.find(name);
			if (found == functions.end() || found->second.getName() == "__entry")
				continue;
			if (sites.size() > 1 && nontrivialOpCounts[name] > 2)
				continue;
			if (functionsWithMachineTerminators.count(name))
				continue;
			mlir::func::FuncOp caller = sites.front()->getParentOfType<mlir::func::FuncOp>();
			if (!caller || reaches(name, name) || reaches(name, caller.getName().str()))
				continue;
			candidate = sites.front();
			callee = found->second;
			eraseCallee = sites.size() == 1;
			break;
		}
		if (!candidate)
			return;

		mlir::InlinerInterface interface(_module.getContext());
		auto cloneCallback = [](
			mlir::OpBuilder&,
			mlir::Region* source,
			mlir::Block* inlineBlock,
			mlir::Block* postInsertBlock,
			mlir::IRMapping& mapping,
			bool clone) {
			mlir::Region* destination = inlineBlock->getParent();
			if (clone)
				source->cloneInto(destination, postInsertBlock->getIterator(), mapping);
			else
				destination->getBlocks().splice(
					postInsertBlock->getIterator(), source->getBlocks(), source->begin(), source->end());
		};
		if (mlir::failed(mlir::inlineCall(
			interface,
			cloneCallback,
			mlir::CallOpInterface(candidate.getOperation()),
			mlir::CallableOpInterface(callee.getOperation()),
			&callee.getBody())))
			return;
		candidate.erase();
		if (eraseCallee)
			callee.erase();
	}
}

} // anonymous namespace

mlir::OwningOpRef<mlir::ModuleOp> solidity::mlirgen::convertYulToEVM(mlir::ModuleOp _module, std::string& _error)
{
	mlir::MLIRContext& ctx = *_module.getContext();
	ctx.getOrLoadDialect<mlir::evm::EVMDialect>();
	ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
	ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
	ctx.getOrLoadDialect<mlir::func::FuncDialect>();

	// The func dialect's inliner interface is an opt-in extension; without it
	// the inliner cannot see through a func.call at all and silently does
	// nothing.
	mlir::DialectRegistry registry;
	mlir::func::registerInlinerExtension(registry);
	ctx.appendDialectRegistry(registry);

	mlir::OwningOpRef<mlir::ModuleOp> converted = Converter(ctx).run(_module, _error);
	if (!converted)
		return converted;

	// The evm rung is where upstream canonicalization is meant to pay off
	// (ADR-003): the landmine ops fold with EVM-exact edge behaviour, and the
	// pure ones let CSE and dead-value removal work on what is left. Yul
	// arriving from solc is already optimized, so this is about what the
	// conversion itself introduced rather than a replacement for that.
	mlir::PassManager manager(&ctx);
	inlineSingleUseFunctions(*converted);
	manager.addPass(mlir::createSymbolDCEPass());
	manager.addPass(mlir::createCanonicalizerPass());
	manager.addPass(mlir::createCSEPass());
	if (mlir::failed(manager.run(*converted)))
	{
		_error = "optimization of the converted module failed";
		return nullptr;
	}
	if (std::getenv("SOL_MLIR_DUMP_EVM"))
		converted->print(llvm::errs());
	return converted;
}
