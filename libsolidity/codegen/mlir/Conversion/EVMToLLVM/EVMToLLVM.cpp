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
 * EVMToLLVM conversion implementation.
 */

#include "EVMToLLVM.h"

#include "EVMDialect.h"
#include "EVMOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#pragma GCC diagnostic pop

#include <string>

using namespace mlir;

namespace
{

IntegerType wordType(MLIRContext* _ctx) { return IntegerType::get(_ctx, 256); }

/// Finds or declares `void <name>(ptr, ptr, ...)` at module scope.
LLVM::LLVMFuncOp lookupOrCreateRtFn(ConversionPatternRewriter& _rewriter, Operation* _op, StringRef _name, unsigned _numPtrArgs)
{
	auto module = _op->getParentOfType<ModuleOp>();
	if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(_name))
		return existing;

	auto ptrType = LLVM::LLVMPointerType::get(module.getContext());
	SmallVector<Type, 4> argTypes(_numPtrArgs, ptrType);
	auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(module.getContext()), argTypes);

	OpBuilder::InsertionGuard guard(_rewriter);
	_rewriter.setInsertionPointToStart(module.getBody());
	return _rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), _name, fnType);
}

/// Lowers an N-operand single-result evm op to a by-pointer evm-rt call.
struct RtCallLowering: ConversionPattern
{
	RtCallLowering(MLIRContext* _ctx, StringRef _opName, StringRef _rtName)
		: ConversionPattern(_opName, /*benefit=*/1, _ctx), m_rtName(_rtName.str())
	{
	}

	LogicalResult
	matchAndRewrite(Operation* _op, ArrayRef<Value> _operands, ConversionPatternRewriter& _rewriter) const override
	{
		Location loc = _op->getLoc();
		MLIRContext* ctx = _op->getContext();
		auto ptrType = LLVM::LLVMPointerType::get(ctx);
		auto i256 = wordType(ctx);
		auto i32Type = IntegerType::get(ctx, 32);

		Value one = _rewriter.create<LLVM::ConstantOp>(loc, i32Type, _rewriter.getI32IntegerAttr(1));

		// One slot for the result, one per operand.
		Value resultSlot = _rewriter.create<LLVM::AllocaOp>(loc, ptrType, i256, one);
		SmallVector<Value, 4> callArgs{resultSlot};
		for (Value operand: _operands)
		{
			Value slot = _rewriter.create<LLVM::AllocaOp>(loc, ptrType, i256, one);
			_rewriter.create<LLVM::StoreOp>(loc, operand, slot);
			callArgs.push_back(slot);
		}

		LLVM::LLVMFuncOp fn = lookupOrCreateRtFn(_rewriter, _op, m_rtName, callArgs.size());
		_rewriter.create<LLVM::CallOp>(loc, fn, callArgs);

		Value result = _rewriter.create<LLVM::LoadOp>(loc, i256, resultSlot);
		_rewriter.replaceOp(_op, result);
		return success();
	}

private:
	std::string m_rtName;
};

/// Inline legalization for shl/shr/sar: clamp the shift below the bit width
/// and select the EVM-defined out-of-range result (upstream shifts are
/// poison at >= 256).
enum class ShiftKind
{
	Shl,
	Shr,
	Sar
};

struct ShiftLowering: ConversionPattern
{
	ShiftLowering(MLIRContext* _ctx, StringRef _opName, ShiftKind _kind)
		: ConversionPattern(_opName, /*benefit=*/1, _ctx), m_kind(_kind)
	{
	}

	LogicalResult
	matchAndRewrite(Operation* _op, ArrayRef<Value> _operands, ConversionPatternRewriter& _rewriter) const override
	{
		Location loc = _op->getLoc();
		auto i256 = wordType(_op->getContext());

		// evm shift ops carry (shift, value) like the EVM opcodes.
		Value shift = _operands[0];
		Value value = _operands[1];

		Value c256 = _rewriter.create<LLVM::ConstantOp>(loc, i256, _rewriter.getIntegerAttr(i256, 256));
		Value zero = _rewriter.create<LLVM::ConstantOp>(loc, i256, _rewriter.getIntegerAttr(i256, 0));
		Value inRange = _rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, shift, c256);
		// Clamp so the shift itself is never poison, then select the result.
		Value clamped = _rewriter.create<LLVM::SelectOp>(loc, inRange, shift, zero);

		Value shifted;
		Value outOfRange;
		switch (m_kind)
		{
		case ShiftKind::Shl:
			shifted = _rewriter.create<LLVM::ShlOp>(loc, value, clamped);
			outOfRange = zero;
			break;
		case ShiftKind::Shr:
			shifted = _rewriter.create<LLVM::LShrOp>(loc, value, clamped);
			outOfRange = zero;
			break;
		case ShiftKind::Sar:
		{
			shifted = _rewriter.create<LLVM::AShrOp>(loc, value, clamped);
			Value c255 = _rewriter.create<LLVM::ConstantOp>(loc, i256, _rewriter.getIntegerAttr(i256, 255));
			outOfRange = _rewriter.create<LLVM::AShrOp>(loc, value, c255); // sign fill
			break;
		}
		}

		_rewriter.replaceOp(_op, Value(_rewriter.create<LLVM::SelectOp>(loc, inRange, shifted, outOfRange)));
		return success();
	}

private:
	ShiftKind m_kind;
};

} // anonymous namespace

bool solidity::mlirgen::convertEVMToLLVM(mlir::ModuleOp _module, std::string& _error)
{
	MLIRContext* ctx = _module.getContext();
	ctx->getOrLoadDialect<LLVM::LLVMDialect>();

	LLVMTypeConverter typeConverter(ctx);
	RewritePatternSet patterns(ctx);

	arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
	cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
	populateFuncToLLVMConversionPatterns(typeConverter, patterns);

	patterns.add<RtCallLowering>(ctx, "evm.div", "__evm_rt_div");
	patterns.add<RtCallLowering>(ctx, "evm.sdiv", "__evm_rt_sdiv");
	patterns.add<RtCallLowering>(ctx, "evm.mod", "__evm_rt_mod");
	patterns.add<RtCallLowering>(ctx, "evm.smod", "__evm_rt_smod");
	patterns.add<RtCallLowering>(ctx, "evm.addmod", "__evm_rt_addmod");
	patterns.add<RtCallLowering>(ctx, "evm.mulmod", "__evm_rt_mulmod");
	patterns.add<RtCallLowering>(ctx, "evm.exp", "__evm_rt_exp");
	patterns.add<RtCallLowering>(ctx, "evm.byte", "__evm_rt_byte");
	patterns.add<RtCallLowering>(ctx, "evm.signextend", "__evm_rt_signextend");
	patterns.add<ShiftLowering>(ctx, "evm.shl", ShiftKind::Shl);
	patterns.add<ShiftLowering>(ctx, "evm.shr", ShiftKind::Shr);
	patterns.add<ShiftLowering>(ctx, "evm.sar", ShiftKind::Sar);

	LLVMConversionTarget target(*ctx);
	target.addLegalOp<ModuleOp>();

	if (failed(applyFullConversion(_module, target, std::move(patterns))))
	{
		_error = "EVMToLLVM conversion failed (op outside the supported pure "
				 "subset? state/env/call ops arrive with ERHI in M6)";
		return false;
	}
	return true;
}
