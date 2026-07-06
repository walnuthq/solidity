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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#pragma GCC diagnostic pop

#include <string>
#include <vector>

using namespace mlir;
using namespace solidity::mlirgen;

//===----------------------------------------------------------------------===//
// Block-local variable promotion (mini mem2reg)
//===----------------------------------------------------------------------===//

unsigned solidity::mlirgen::promoteBlockLocalVars(mlir::ModuleOp _module)
{
	// A var is promotable when every user (loads and assigns) lives in the
	// same block as the yul.var itself - then a single linear scan renames
	// loads to the current value.
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
			for (mlir::Operation& op: _src.getBody()->getOperations())
			{
				auto funcOp = llvm::dyn_cast<mlir::yul::FuncOp>(&op);
				if (!funcOp)
					fail("unsupported top-level operation: " + op.getName().getStringRef().str());
				m_builder.setInsertionPointToEnd(dst->getBody());
				convertFunction(funcOp);
			}
		}
		catch (ConversionError const& _e)
		{
			_error = _e.message;
			return nullptr;
		}
		if (failed(mlir::verify(*dst)))
		{
			_error = "converted module failed MLIR verification";
			return nullptr;
		}
		return dst;
	}

private:
	mlir::OpBuilder m_builder;
	llvm::DenseMap<mlir::Value, mlir::Value> m_map;

	struct LoopCtx
	{
		mlir::Block* post;
		mlir::Block* exit;
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

	mlir::Value wordConstant(llvm::APInt _value)
	{
		return m_builder.create<mlir::arith::ConstantOp>(loc(), mlir::IntegerAttr::get(wordType(), std::move(_value)));
	}

	/// i256 (0/1 semantics) from an i1.
	mlir::Value boolToWord(mlir::Value _i1)
	{
		return m_builder.create<mlir::arith::ExtUIOp>(loc(), wordType(), _i1);
	}

	/// i1 truthiness of a word (non-zero).
	mlir::Value wordToBool(mlir::Value _word)
	{
		mlir::Value zero = wordConstant(llvm::APInt(256, 0));
		return m_builder.create<mlir::arith::CmpIOp>(loc(), mlir::arith::CmpIPredicate::ne, _word, zero);
	}

	mlir::Block* newBlock()
	{
		mlir::Block* block = new mlir::Block();
		m_currentFunc.getBody().push_back(block);
		return block;
	}

	void convertFunction(mlir::yul::FuncOp _func)
	{
		auto dstFunc = m_builder.create<mlir::func::FuncOp>(loc(), _func.getSymName(), _func.getFunctionType());
		dstFunc.setPrivate();
		m_currentFunc = dstFunc;

		mlir::Block* entry = dstFunc.addEntryBlock();
		if (!_func.getBody().empty())
		{
			mlir::Block& srcEntry = _func.getBody().front();
			for (unsigned i = 0; i < srcEntry.getNumArguments(); ++i)
				m_map[srcEntry.getArgument(i)] = entry->getArgument(i);

			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToEnd(entry);
			bool terminated = convertBlockOps(srcEntry);
			if (!terminated)
			{
				if (_func.getFunctionType().getNumResults() == 0)
					m_builder.create<mlir::func::ReturnOp>(loc());
				else
					fail("function '" + _func.getSymName().str() + "' falls through with results");
			}
		}
		else
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToEnd(entry);
			m_builder.create<mlir::func::ReturnOp>(loc());
		}
	}

	/// Converts the ops of a structured yul block into the current insertion
	/// block (which may change as control flow is flattened). @returns true
	/// if conversion produced a terminator.
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
		// Literals.
		if (auto constOp = llvm::dyn_cast<mlir::yul::ConstOp>(&_op))
		{
			m_map[constOp.getResult()] = wordConstant(constOp.getValue());
			return false;
		}

		// Upstream-exact arithmetic (ADR-003: arith reuse at this rung).
		if (convertArithExact(_op))
			return false;

		// Comparisons -> arith.cmpi + extui.
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
			m_map[memoryGuard.getResult()] = wordConstant(memoryGuard.getSize());
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
			m_builder.create<mlir::cf::BranchOp>(loc(), m_loops.back().exit);
			return true;
		}
		if (llvm::isa<mlir::yul::ContinueOp>(&_op))
		{
			if (m_loops.empty())
				fail("continue outside of a loop");
			m_builder.create<mlir::cf::BranchOp>(loc(), m_loops.back().post);
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
			auto savedFunc = m_currentFunc;
			auto savedLoops = std::move(m_loops);
			m_loops.clear();
			m_builder.setInsertionPointToEnd(
				&m_currentFunc.getOperation()->getParentOfType<mlir::ModuleOp>().getBodyRegion().front());
			convertFunction(funcOp);
			m_currentFunc = savedFunc;
			m_loops = std::move(savedLoops);
			return false;
		}

		// Unpromoted variables.
		if (llvm::isa<mlir::yul::VarOp, mlir::yul::AssignOp, mlir::yul::VarLoadOp>(&_op))
			fail("mutable variable survives promotion (region-crossing var; "
				 "full promotion is a follow-up milestone)");

		if (llvm::isa<mlir::yul::ConditionOp>(&_op))
			fail("yul.condition outside a for-loop condition region");

		if (llvm::isa<mlir::yul::DataOffsetOp, mlir::yul::DataSizeOp, mlir::yul::DataCopyOp>(&_op))
			fail("object data builtins are not supported yet (evm.program objects "
				 "are a follow-up milestone)");

		// Everything else: the identically-named evm.* op (terminators included).
		return convertGenericEvmOp(_op);
	}

	bool convertArithExact(mlir::Operation& _op)
	{
		mlir::Value result;
		if (auto addOp = llvm::dyn_cast<mlir::yul::AddOp>(&_op))
			result =
				m_builder.create<mlir::arith::AddIOp>(loc(), mapped(addOp.getLhs()), mapped(addOp.getRhs()));
		else if (auto subOp = llvm::dyn_cast<mlir::yul::SubOp>(&_op))
			result =
				m_builder.create<mlir::arith::SubIOp>(loc(), mapped(subOp.getLhs()), mapped(subOp.getRhs()));
		else if (auto mulOp = llvm::dyn_cast<mlir::yul::MulOp>(&_op))
			result =
				m_builder.create<mlir::arith::MulIOp>(loc(), mapped(mulOp.getLhs()), mapped(mulOp.getRhs()));
		else if (auto andOp = llvm::dyn_cast<mlir::yul::AndOp>(&_op))
			result =
				m_builder.create<mlir::arith::AndIOp>(loc(), mapped(andOp.getLhs()), mapped(andOp.getRhs()));
		else if (auto orOp = llvm::dyn_cast<mlir::yul::OrOp>(&_op))
			result = m_builder.create<mlir::arith::OrIOp>(loc(), mapped(orOp.getLhs()), mapped(orOp.getRhs()));
		else if (auto xorOp = llvm::dyn_cast<mlir::yul::XorOp>(&_op))
			result =
				m_builder.create<mlir::arith::XOrIOp>(loc(), mapped(xorOp.getLhs()), mapped(xorOp.getRhs()));
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
		mlir::Block* thenBlock = newBlock();
		mlir::Block* contBlock = newBlock();
		m_builder.create<mlir::cf::CondBranchOp>(loc(), cond, thenBlock, contBlock);

		m_builder.setInsertionPointToEnd(thenBlock);
		bool terminated = _if.getThenRegion().empty() ? false : convertBlockOps(_if.getThenRegion().front());
		if (!terminated)
			m_builder.create<mlir::cf::BranchOp>(loc(), contBlock);

		m_builder.setInsertionPointToEnd(contBlock);
	}

	void convertFor(mlir::yul::ForOp _for)
	{
		mlir::Block* condBlock = newBlock();
		mlir::Block* bodyBlock = newBlock();
		mlir::Block* postBlock = newBlock();
		mlir::Block* exitBlock = newBlock();

		m_builder.create<mlir::cf::BranchOp>(loc(), condBlock);

		// Condition region: ops up to yul.condition, then a conditional branch.
		m_builder.setInsertionPointToEnd(condBlock);
		bool sawCondition = false;
		if (!_for.getCondRegion().empty())
			for (mlir::Operation& op: _for.getCondRegion().front().getOperations())
			{
				if (auto condition = llvm::dyn_cast<mlir::yul::ConditionOp>(&op))
				{
					mlir::Value cond = wordToBool(mapped(condition.getCondition()));
					m_builder.create<mlir::cf::CondBranchOp>(loc(), cond, bodyBlock, exitBlock);
					sawCondition = true;
					break;
				}
				if (convertOp(op))
					fail("terminator inside a for-loop condition region");
			}
		if (!sawCondition)
			// No condition: infinite loop unless the body breaks.
			m_builder.create<mlir::cf::BranchOp>(loc(), bodyBlock);

		m_loops.push_back(LoopCtx{postBlock, exitBlock});

		m_builder.setInsertionPointToEnd(bodyBlock);
		bool bodyTerminated = _for.getBodyRegion().empty() ? false : convertBlockOps(_for.getBodyRegion().front());
		if (!bodyTerminated)
			m_builder.create<mlir::cf::BranchOp>(loc(), postBlock);

		m_builder.setInsertionPointToEnd(postBlock);
		bool postTerminated = _for.getPostRegion().empty() ? false : convertBlockOps(_for.getPostRegion().front());
		if (!postTerminated)
			m_builder.create<mlir::cf::BranchOp>(loc(), condBlock);

		m_loops.pop_back();
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

} // anonymous namespace

mlir::OwningOpRef<mlir::ModuleOp> solidity::mlirgen::convertYulToEVM(mlir::ModuleOp _module, std::string& _error)
{
	mlir::MLIRContext& ctx = *_module.getContext();
	ctx.getOrLoadDialect<mlir::evm::EVMDialect>();
	ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
	ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
	ctx.getOrLoadDialect<mlir::func::FuncDialect>();
	return Converter(ctx).run(_module, _error);
}
