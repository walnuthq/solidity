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
 * Operation implementations of the `evm` MLIR dialect.
 */

#include "EVMOps.h"
#include "EVMDialect.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#pragma GCC diagnostic pop

using namespace mlir;
using namespace mlir::evm;

// Generated operation definitions.
#define GET_OP_CLASSES
#include "EVMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Folders
//
// These are the ops whose result is a pure function of their operands and which
// the EVM defines over the whole input range - the semantic landmines. Folding
// them is where the dialect earns the ADR-003 claim that the `evm` rung is
// where MLIR's canonicalization infrastructure pays off, and it is the same
// edge behaviour the differential corpus already pins down: division by zero
// yields zero rather than trapping, sdiv(MIN, -1) wraps, shifts of 256 or more
// are defined, and addmod/mulmod carry a wide intermediate.
//===----------------------------------------------------------------------===//

namespace
{

/// The 256-bit value of a folded operand, if it is a constant.
std::optional<llvm::APInt> word(Attribute _attribute)
{
	if (auto value = dyn_cast_or_null<IntegerAttr>(_attribute))
		return value.getValue();
	return std::nullopt;
}

OpFoldResult wordResult(Type _type, llvm::APInt const& _value)
{
	return IntegerAttr::get(_type, _value);
}

OpFoldResult zero(Type _type) { return wordResult(_type, llvm::APInt(256, 0)); }

OpFoldResult one(Type _type) { return wordResult(_type, llvm::APInt(256, 1)); }

/// Whether a folded operand is exactly @a _literal.
bool isLiteral(Attribute _attribute, uint64_t _literal)
{
	std::optional<llvm::APInt> const value = word(_attribute);
	return value && *value == _literal;
}

} // anonymous namespace

OpFoldResult DivOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getRhs(), 1))
		return getLhs();
	if (isLiteral(_adaptor.getLhs(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const divisor = word(_adaptor.getRhs());
	if (!divisor)
		return {};
	if (divisor->isZero())
		return zero(getType()); // EVM defines x / 0 as 0
	std::optional<llvm::APInt> const dividend = word(_adaptor.getLhs());
	return dividend ? wordResult(getType(), dividend->udiv(*divisor)) : OpFoldResult{};
}

OpFoldResult SDivOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getRhs(), 1))
		return getLhs();
	if (isLiteral(_adaptor.getLhs(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const divisor = word(_adaptor.getRhs());
	if (!divisor)
		return {};
	if (divisor->isZero())
		return zero(getType());
	std::optional<llvm::APInt> const dividend = word(_adaptor.getLhs());
	if (!dividend)
		return {};
	// The one signed division that overflows: the EVM wraps it to MIN.
	if (dividend->isMinSignedValue() && divisor->isAllOnes())
		return wordResult(getType(), *dividend);
	return wordResult(getType(), dividend->sdiv(*divisor));
}

OpFoldResult ModOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getRhs(), 1) || isLiteral(_adaptor.getLhs(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const modulus = word(_adaptor.getRhs());
	if (!modulus)
		return {};
	if (modulus->isZero())
		return zero(getType());
	std::optional<llvm::APInt> const value = word(_adaptor.getLhs());
	return value ? wordResult(getType(), value->urem(*modulus)) : OpFoldResult{};
}

OpFoldResult SModOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getRhs(), 1) || isLiteral(_adaptor.getLhs(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const modulus = word(_adaptor.getRhs());
	if (!modulus)
		return {};
	if (modulus->isZero())
		return zero(getType());
	std::optional<llvm::APInt> const value = word(_adaptor.getLhs());
	if (!value)
		return {};
	// srem overflows on the same pair sdiv does, and the remainder there is 0.
	if (value->isMinSignedValue() && modulus->isAllOnes())
		return zero(getType());
	return wordResult(getType(), value->srem(*modulus));
}

OpFoldResult AddModOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getC(), 1))
		return zero(getType());
	std::optional<llvm::APInt> const modulus = word(_adaptor.getC());
	if (!modulus)
		return {};
	if (modulus->isZero())
		return zero(getType());
	std::optional<llvm::APInt> const a = word(_adaptor.getA());
	std::optional<llvm::APInt> const b = word(_adaptor.getB());
	if (!a || !b)
		return {};
	// The sum has to be taken before the reduction, so it needs the extra bit.
	unsigned const wide = 264;
	llvm::APInt const sum = a->zext(wide) + b->zext(wide);
	return wordResult(getType(), sum.urem(modulus->zext(wide)).trunc(256));
}

OpFoldResult MulModOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getC(), 1) || isLiteral(_adaptor.getA(), 0) || isLiteral(_adaptor.getB(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const modulus = word(_adaptor.getC());
	if (!modulus)
		return {};
	if (modulus->isZero())
		return zero(getType());
	std::optional<llvm::APInt> const a = word(_adaptor.getA());
	std::optional<llvm::APInt> const b = word(_adaptor.getB());
	if (!a || !b)
		return {};
	unsigned const wide = 512;
	llvm::APInt const product = a->zext(wide) * b->zext(wide);
	return wordResult(getType(), product.urem(modulus->zext(wide)).trunc(256));
}

OpFoldResult ExpOp::fold(FoldAdaptor _adaptor)
{
	// x^1 is x, and both x^0 and 1^x are 1 - including 0^0, which the EVM
	// defines as 1.
	if (isLiteral(_adaptor.getRhs(), 1))
		return getLhs();
	if (isLiteral(_adaptor.getRhs(), 0) || isLiteral(_adaptor.getLhs(), 1))
		return one(getType());
	std::optional<llvm::APInt> const base = word(_adaptor.getLhs());
	std::optional<llvm::APInt> const exponent = word(_adaptor.getRhs());
	if (!base || !exponent)
		return {};
	// Square and multiply, wrapping - the EVM has no overflow here.
	llvm::APInt result(256, 1);
	llvm::APInt factor = *base;
	llvm::APInt remaining = *exponent;
	while (!remaining.isZero())
	{
		if (remaining[0])
			result *= factor;
		remaining.lshrInPlace(1);
		if (!remaining.isZero())
			factor *= factor;
	}
	return wordResult(getType(), result);
}

namespace
{

/// Shifts of 256 or more are defined by the EVM, where upstream shifts are
/// poison, so the amount is checked before it is used.
bool shiftedOut(llvm::APInt const& _amount) { return _amount.uge(256); }

} // anonymous namespace

OpFoldResult ShlOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getLhs(), 0))
		return getRhs(); // shifting by nothing
	if (isLiteral(_adaptor.getRhs(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const amount = word(_adaptor.getLhs());
	if (!amount)
		return {};
	if (shiftedOut(*amount))
		return zero(getType());
	std::optional<llvm::APInt> const value = word(_adaptor.getRhs());
	return value ? wordResult(getType(), value->shl(amount->getZExtValue())) : OpFoldResult{};
}

OpFoldResult ShrOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getLhs(), 0))
		return getRhs(); // shifting by nothing
	if (isLiteral(_adaptor.getRhs(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const amount = word(_adaptor.getLhs());
	if (!amount)
		return {};
	if (shiftedOut(*amount))
		return zero(getType());
	std::optional<llvm::APInt> const value = word(_adaptor.getRhs());
	return value ? wordResult(getType(), value->lshr(amount->getZExtValue())) : OpFoldResult{};
}

OpFoldResult SarOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getLhs(), 0))
		return getRhs();
	if (isLiteral(_adaptor.getRhs(), 0))
		return zero(getType()); // the sign of zero is zero
	std::optional<llvm::APInt> const amount = word(_adaptor.getLhs());
	std::optional<llvm::APInt> const value = word(_adaptor.getRhs());
	if (!amount || !value)
		return {};
	// Past the width the sign fills the word, so the result is 0 or all ones.
	if (shiftedOut(*amount))
		return wordResult(getType(), value->isNegative() ? llvm::APInt::getAllOnes(256) : llvm::APInt(256, 0));
	return wordResult(getType(), value->ashr(amount->getZExtValue()));
}

OpFoldResult ByteOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getRhs(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const index = word(_adaptor.getLhs());
	if (!index)
		return {};
	if (index->uge(32))
		return zero(getType()); // out of range reads as zero
	std::optional<llvm::APInt> const value = word(_adaptor.getRhs());
	if (!value)
		return {};
	// Byte 0 is the most significant one.
	unsigned const shift = (31 - index->getZExtValue()) * 8;
	return wordResult(getType(), value->lshr(shift) & llvm::APInt(256, 0xff));
}

OpFoldResult SignExtendOp::fold(FoldAdaptor _adaptor)
{
	if (isLiteral(_adaptor.getRhs(), 0))
		return zero(getType());
	std::optional<llvm::APInt> const index = word(_adaptor.getLhs());
	std::optional<llvm::APInt> const value = word(_adaptor.getRhs());
	if (!index || !value)
		return {};
	if (index->uge(31))
		return wordResult(getType(), *value); // already the full width
	// Extend from the sign bit of byte `index`, counting from the low end.
	unsigned const bits = (index->getZExtValue() + 1) * 8;
	return wordResult(getType(), value->trunc(bits).sext(256));
}

OpFoldResult ClzOp::fold(FoldAdaptor _adaptor)
{
	std::optional<llvm::APInt> const value = word(_adaptor.getOperand());
	if (!value)
		return {};
	// countLeadingZeros of an all-zero word is the full width, which is what
	// the EVM specifies.
	return wordResult(getType(), llvm::APInt(256, value->countLeadingZeros()));
}
