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

#include "SolidityOps.h"
#include "SolidityDialect.h"
#include <optional>

#ifdef SOLIDITY_HAS_MLIR
// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/ADT/APInt.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#pragma GCC diagnostic pop

// Include the generated operation definitions from TableGen
#define GET_OP_CLASSES
#include "SolidityOps.cpp.inc"

// TypeIDs are now defined by TableGen in the included file above

namespace mlir
{
namespace solidity
{

namespace
{

constexpr unsigned cEVMBitWidth = 256;
constexpr unsigned cWideBitWidth = 512;

mlir::IntegerType getEVMIntegerType(mlir::MLIRContext* context)
{
	return mlir::IntegerType::get(context, cEVMBitWidth);
}

mlir::IntegerAttr getEVMIntegerAttr(mlir::MLIRContext* context, llvm::APInt const& value)
{
	return mlir::IntegerAttr::get(getEVMIntegerType(context), value.zextOrTrunc(cEVMBitWidth));
}

mlir::IntegerAttr getEVMZeroAttr(mlir::MLIRContext* context)
{
	return getEVMIntegerAttr(context, llvm::APInt(cEVMBitWidth, 0));
}

mlir::IntegerAttr getEVMOneAttr(mlir::MLIRContext* context)
{
	return getEVMIntegerAttr(context, llvm::APInt(cEVMBitWidth, 1));
}

bool isShiftOutOfRange(llvm::APInt const& shift)
{
	return shift.uge(llvm::APInt(cEVMBitWidth, cEVMBitWidth));
}

// BoolAttr is also an IntegerAttr; handle bool first to preserve true==1.
mlir::IntegerAttr getConstIntAttr(mlir::Attribute attr)
{
	if (!attr)
		return {};

	if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr))
	{
		return mlir::IntegerAttr::get(
			getEVMIntegerType(attr.getContext()),
			llvm::APInt(cEVMBitWidth, boolAttr.getValue() ? 1 : 0));
	}

	return mlir::dyn_cast<mlir::IntegerAttr>(attr);
}

std::optional<bool> getConstBoolValue(mlir::Attribute attr)
{
	if (!attr)
		return std::nullopt;

	if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr))
		return boolAttr.getValue();

	if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
		return !intAttr.getValue().isZero();

	return std::nullopt;
}

llvm::APInt toEVMUnsigned(mlir::IntegerAttr attr)
{
	return attr.getValue().zextOrTrunc(cEVMBitWidth);
}

llvm::APInt toEVMSigned(mlir::IntegerAttr attr)
{
	return attr.getValue().sextOrTrunc(cEVMBitWidth);
}

std::optional<unsigned> getIntegerWidthFromType(mlir::Type type)
{
	if (auto uintType = mlir::dyn_cast<mlir::solidity::UIntType>(type))
		return uintType.getBitWidth();
	if (auto intType = mlir::dyn_cast<mlir::solidity::IntType>(type))
		return intType.getBitWidth();
	if (mlir::isa<mlir::solidity::BoolType>(type))
		return 1;
	if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type))
		return integerType.getWidth();
	return std::nullopt;
}

} // namespace

//===----------------------------------------------------------------------===//
// Custom builders and verifiers for operations
//===----------------------------------------------------------------------===//

// Custom builder for FunctionOp - needed for MLIRGenerator
void FunctionOp::build(
	mlir::OpBuilder& builder,
	mlir::OperationState& state,
	llvm::StringRef name,
	mlir::FunctionType type,
	llvm::StringRef visibility,
	llvm::StringRef stateMutability)
{
	state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
	state.addAttribute("function_type", mlir::TypeAttr::get(type));
	state.addAttribute("visibility", builder.getStringAttr(visibility));
	state.addAttribute("stateMutability", builder.getStringAttr(stateMutability));
	// Create an empty body region
	state.addRegion();
}

// Custom builder for ConstantOp - needed for MLIRGenerator
void ConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Attribute value, mlir::Type type)
{
	(void)builder;
	state.getOrAddProperties<ConstantOp::Properties>().value = value;
	state.addTypes(type);
}

// Custom printer for ConstantOp - required due to hasCustomAssemblyFormat
void ConstantOp::print(mlir::OpAsmPrinter& p)
{
	p << " ";
	p.printAttribute(getValueAttr());
	p << " : ";
	p.printType(getResult().getType());
}

// Custom parser for ConstantOp - required due to hasCustomAssemblyFormat
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::Attribute value;
	mlir::Type type;

	if (parser.parseAttribute(value) || parser.parseColon() || parser.parseType(type))
		return mlir::failure();

	result.getOrAddProperties<ConstantOp::Properties>().value = value;
	result.addTypes(type);
	return mlir::success();
}

// Custom builder for LoadStateVarOp - commented out as TableGen generates this
// void LoadStateVarOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                            mlir::Type resultType, llvm::StringRef varName) {
// 	state.addAttribute("varName", builder.getStringAttr(varName));
// 	state.addTypes(resultType);
// }

// Custom builder for AddOp - commented out as TableGen generates this
// void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
// 	state.addOperands({lhs, rhs});
// 	state.addTypes(resultType);
// }

// Custom printer for CreateContractOp
void CreateContractOp::print(mlir::OpAsmPrinter& p)
{
	// The parser reads this back as a string attribute, so it has to be
	// printed as one: unquoted, the dialect cannot re-read its own output.
	p << " ";
	p.printAttribute(getContractNameAttr());
	p << "(";
	p.printOperands(getConstructorArgs());
	p << ") value ";
	p.printOperand(getValue());
	p.printOptionalAttrDict((*this)->getAttrs(), {"contract_name"});
	p << " : ";
	p.printFunctionalType((*this)->getOperandTypes(), (*this)->getResultTypes());
}

// Custom parser for CreateContractOp
mlir::ParseResult CreateContractOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::StringAttr contractNameAttr;
	if (parser.parseAttribute(contractNameAttr, "contract_name", result.attributes))
		return mlir::failure();

	llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> args;
	if (parser.parseLParen())
		return mlir::failure();
	if (parser.parseOptionalRParen())
	{
		if (parser.parseOperandList(args) || parser.parseRParen())
			return mlir::failure();
	}

	mlir::OpAsmParser::UnresolvedOperand valueOperand;
	if (parser.parseKeyword("value") || parser.parseOperand(valueOperand))
		return mlir::failure();

	if (parser.parseOptionalAttrDict(result.attributes))
		return mlir::failure();

	mlir::FunctionType funcType;
	if (parser.parseColonType(funcType))
		return mlir::failure();

	// Resolve operands: value is first, then constructor args
	if (parser.resolveOperand(valueOperand, funcType.getInput(0), result.operands))
		return mlir::failure();
	for (unsigned i = 0; i < args.size(); ++i)
	{
		if (parser.resolveOperand(args[i], funcType.getInput(i + 1), result.operands))
			return mlir::failure();
	}

	result.addTypes(funcType.getResults());
	return mlir::success();
}

//===----------------------------------------------------------------------===//
// Folding
//===----------------------------------------------------------------------===//

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor)
{
	return getValue();
}

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor)
{
	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());

	if (rhsAttr && toEVMUnsigned(rhsAttr).isZero())
		return getLhs();
	if (lhsAttr && toEVMUnsigned(lhsAttr).isZero())
		return getRhs();
	if (!lhsAttr || !rhsAttr)
		return {};

	return getEVMIntegerAttr(getContext(), toEVMUnsigned(lhsAttr) + toEVMUnsigned(rhsAttr));
}

mlir::OpFoldResult SubOp::fold(FoldAdaptor adaptor)
{
	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());

	if (rhsAttr && toEVMUnsigned(rhsAttr).isZero())
		return getLhs();
	if (getLhs() == getRhs())
		return getEVMZeroAttr(getContext());
	if (!lhsAttr || !rhsAttr)
		return {};

	return getEVMIntegerAttr(getContext(), toEVMUnsigned(lhsAttr) - toEVMUnsigned(rhsAttr));
}

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor)
{
	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());

	if (lhsAttr)
	{
		auto lhs = toEVMUnsigned(lhsAttr);
		if (lhs.isZero())
			return getEVMZeroAttr(getContext());
		if (lhs.isOne())
			return getRhs();
	}

	if (rhsAttr)
	{
		auto rhs = toEVMUnsigned(rhsAttr);
		if (rhs.isZero())
			return getEVMZeroAttr(getContext());
		if (rhs.isOne())
			return getLhs();
	}

	if (!lhsAttr || !rhsAttr)
		return {};

	return getEVMIntegerAttr(getContext(), toEVMUnsigned(lhsAttr) * toEVMUnsigned(rhsAttr));
}

mlir::OpFoldResult DivOp::fold(FoldAdaptor adaptor)
{
	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());

	if (rhsAttr)
	{
		auto rhs = toEVMUnsigned(rhsAttr);
		if (rhs.isOne())
			return getLhs();
		if (rhs.isZero())
			return getEVMZeroAttr(getContext());
	}

	if (lhsAttr && toEVMUnsigned(lhsAttr).isZero())
		return getEVMZeroAttr(getContext());

	if (!lhsAttr || !rhsAttr)
		return {};

	auto lhs = toEVMUnsigned(lhsAttr);
	auto rhs = toEVMUnsigned(rhsAttr);
	if (rhs.isZero())
		return getEVMZeroAttr(getContext());
	return getEVMIntegerAttr(getContext(), lhs.udiv(rhs));
}

mlir::OpFoldResult ModOp::fold(FoldAdaptor adaptor)
{
	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());

	if (rhsAttr)
	{
		auto rhs = toEVMUnsigned(rhsAttr);
		if (rhs.isOne() || rhs.isZero())
			return getEVMZeroAttr(getContext());
	}

	if (!lhsAttr || !rhsAttr)
		return {};

	auto lhs = toEVMUnsigned(lhsAttr);
	auto rhs = toEVMUnsigned(rhsAttr);
	if (rhs.isZero())
		return getEVMZeroAttr(getContext());
	return getEVMIntegerAttr(getContext(), lhs.urem(rhs));
}

mlir::OpFoldResult ExpOp::fold(FoldAdaptor adaptor)
{
	auto baseAttr = getConstIntAttr(adaptor.getBase());
	auto exponentAttr = getConstIntAttr(adaptor.getExponent());

	if (exponentAttr)
	{
		auto exponent = toEVMUnsigned(exponentAttr);
		if (exponent.isZero())
			return getEVMOneAttr(getContext());
		if (exponent.isOne())
			return getBase();
	}

	if (!baseAttr || !exponentAttr)
		return {};

	llvm::APInt base = toEVMUnsigned(baseAttr);
	llvm::APInt exponent = toEVMUnsigned(exponentAttr);
	llvm::APInt result(cEVMBitWidth, 1);

	for (unsigned i = 0; i < cEVMBitWidth; ++i)
	{
		if (exponent[i])
			result *= base;
		base *= base;
	}

	return getEVMIntegerAttr(getContext(), result);
}

mlir::OpFoldResult CmpOp::fold(FoldAdaptor adaptor)
{
	auto predicate = getPredicate();

	if (getLhs() == getRhs())
	{
		if (
			predicate == "eq" || predicate == "le" || predicate == "ge" || predicate == "sle" || predicate == "sge")
			return mlir::BoolAttr::get(getContext(), true);
		if (
			predicate == "ne" || predicate == "lt" || predicate == "gt" || predicate == "slt" || predicate == "sgt")
			return mlir::BoolAttr::get(getContext(), false);
	}

	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());
	if (!lhsAttr || !rhsAttr)
		return {};

	auto lhsU = toEVMUnsigned(lhsAttr);
	auto rhsU = toEVMUnsigned(rhsAttr);
	auto lhsS = toEVMSigned(lhsAttr);
	auto rhsS = toEVMSigned(rhsAttr);

	bool result;
	if (predicate == "eq")
		result = lhsU == rhsU;
	else if (predicate == "ne")
		result = lhsU != rhsU;
	else if (predicate == "lt")
		result = lhsU.ult(rhsU);
	else if (predicate == "le")
		result = lhsU.ule(rhsU);
	else if (predicate == "gt")
		result = lhsU.ugt(rhsU);
	else if (predicate == "ge")
		result = lhsU.uge(rhsU);
	else if (predicate == "slt")
		result = lhsS.slt(rhsS);
	else if (predicate == "sle")
		result = lhsS.sle(rhsS);
	else if (predicate == "sgt")
		result = lhsS.sgt(rhsS);
	else if (predicate == "sge")
		result = lhsS.sge(rhsS);
	else
		return {};

	return mlir::BoolAttr::get(getContext(), result);
}

mlir::OpFoldResult AndOp::fold(FoldAdaptor adaptor)
{
	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());

	if (lhsAttr)
	{
		auto lhs = toEVMUnsigned(lhsAttr);
		if (lhs.isZero())
			return getEVMZeroAttr(getContext());
		if (lhs.isAllOnes())
			return getRhs();
	}

	if (rhsAttr)
	{
		auto rhs = toEVMUnsigned(rhsAttr);
		if (rhs.isZero())
			return getEVMZeroAttr(getContext());
		if (rhs.isAllOnes())
			return getLhs();
	}

	if (!lhsAttr || !rhsAttr)
		return {};

	return getEVMIntegerAttr(getContext(), toEVMUnsigned(lhsAttr) & toEVMUnsigned(rhsAttr));
}

mlir::OpFoldResult OrOp::fold(FoldAdaptor adaptor)
{
	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());

	if (lhsAttr && toEVMUnsigned(lhsAttr).isZero())
		return getRhs();
	if (rhsAttr && toEVMUnsigned(rhsAttr).isZero())
		return getLhs();
	if (!lhsAttr || !rhsAttr)
		return {};

	return getEVMIntegerAttr(getContext(), toEVMUnsigned(lhsAttr) | toEVMUnsigned(rhsAttr));
}

mlir::OpFoldResult XorOp::fold(FoldAdaptor adaptor)
{
	auto lhsAttr = getConstIntAttr(adaptor.getLhs());
	auto rhsAttr = getConstIntAttr(adaptor.getRhs());

	if (getLhs() == getRhs())
		return getEVMZeroAttr(getContext());
	if (lhsAttr && toEVMUnsigned(lhsAttr).isZero())
		return getRhs();
	if (rhsAttr && toEVMUnsigned(rhsAttr).isZero())
		return getLhs();
	if (!lhsAttr || !rhsAttr)
		return {};

	return getEVMIntegerAttr(getContext(), toEVMUnsigned(lhsAttr) ^ toEVMUnsigned(rhsAttr));
}

mlir::OpFoldResult NotOp::fold(FoldAdaptor adaptor)
{
	auto operandAttr = getConstIntAttr(adaptor.getOperand());
	if (!operandAttr)
		return {};

	return getEVMIntegerAttr(getContext(), ~toEVMUnsigned(operandAttr));
}

mlir::OpFoldResult ShlOp::fold(FoldAdaptor adaptor)
{
	auto valueAttr = getConstIntAttr(adaptor.getValue());
	auto shiftAttr = getConstIntAttr(adaptor.getShift());

	if (valueAttr && toEVMUnsigned(valueAttr).isZero())
		return getEVMZeroAttr(getContext());
	if (!shiftAttr)
		return {};

	auto shift = toEVMUnsigned(shiftAttr);
	if (isShiftOutOfRange(shift))
		return getEVMZeroAttr(getContext());
	if (shift.isZero())
		return getValue();
	if (!valueAttr)
		return {};

	unsigned shiftAmount = static_cast<unsigned>(shift.getZExtValue());
	return getEVMIntegerAttr(getContext(), toEVMUnsigned(valueAttr) << shiftAmount);
}

mlir::OpFoldResult ShrOp::fold(FoldAdaptor adaptor)
{
	auto valueAttr = getConstIntAttr(adaptor.getValue());
	auto shiftAttr = getConstIntAttr(adaptor.getShift());

	if (valueAttr && toEVMUnsigned(valueAttr).isZero())
		return getEVMZeroAttr(getContext());
	if (!shiftAttr)
		return {};

	auto shift = toEVMUnsigned(shiftAttr);
	if (isShiftOutOfRange(shift))
		return getEVMZeroAttr(getContext());
	if (shift.isZero())
		return getValue();
	if (!valueAttr)
		return {};

	unsigned shiftAmount = static_cast<unsigned>(shift.getZExtValue());
	return getEVMIntegerAttr(getContext(), toEVMUnsigned(valueAttr).lshr(shiftAmount));
}

mlir::OpFoldResult SarOp::fold(FoldAdaptor adaptor)
{
	auto valueAttr = getConstIntAttr(adaptor.getValue());
	auto shiftAttr = getConstIntAttr(adaptor.getShift());

	if (valueAttr && toEVMUnsigned(valueAttr).isZero())
		return getEVMZeroAttr(getContext());
	if (!shiftAttr)
		return {};

	auto shift = toEVMUnsigned(shiftAttr);
	if (!valueAttr)
		return {};

	auto value = toEVMUnsigned(valueAttr);
	if (isShiftOutOfRange(shift))
	{
		if (value.isNegative())
			return getEVMIntegerAttr(getContext(), llvm::APInt::getAllOnes(cEVMBitWidth));
		return getEVMZeroAttr(getContext());
	}

	if (shift.isZero())
		return getValue();

	unsigned shiftAmount = static_cast<unsigned>(shift.getZExtValue());
	return getEVMIntegerAttr(getContext(), value.ashr(shiftAmount));
}

mlir::OpFoldResult AddModOp::fold(FoldAdaptor adaptor)
{
	auto aAttr = getConstIntAttr(adaptor.getA());
	auto bAttr = getConstIntAttr(adaptor.getB());
	auto nAttr = getConstIntAttr(adaptor.getN());

	if (nAttr)
	{
		auto modulus = toEVMUnsigned(nAttr);
		if (modulus.isZero() || modulus.isOne())
			return getEVMZeroAttr(getContext());
	}

	if (!aAttr || !bAttr || !nAttr)
		return {};

	llvm::APInt aWide = toEVMUnsigned(aAttr).zext(cWideBitWidth);
	llvm::APInt bWide = toEVMUnsigned(bAttr).zext(cWideBitWidth);
	llvm::APInt nWide = toEVMUnsigned(nAttr).zext(cWideBitWidth);
	if (nWide.isZero())
		return getEVMZeroAttr(getContext());

	llvm::APInt result = (aWide + bWide).urem(nWide).trunc(cEVMBitWidth);
	return getEVMIntegerAttr(getContext(), result);
}

mlir::OpFoldResult MulModOp::fold(FoldAdaptor adaptor)
{
	auto aAttr = getConstIntAttr(adaptor.getA());
	auto bAttr = getConstIntAttr(adaptor.getB());
	auto nAttr = getConstIntAttr(adaptor.getN());

	if (nAttr)
	{
		auto modulus = toEVMUnsigned(nAttr);
		if (modulus.isZero() || modulus.isOne())
			return getEVMZeroAttr(getContext());
	}

	if (!aAttr || !bAttr || !nAttr)
		return {};

	llvm::APInt aWide = toEVMUnsigned(aAttr).zext(cWideBitWidth);
	llvm::APInt bWide = toEVMUnsigned(bAttr).zext(cWideBitWidth);
	llvm::APInt nWide = toEVMUnsigned(nAttr).zext(cWideBitWidth);
	if (nWide.isZero())
		return getEVMZeroAttr(getContext());

	llvm::APInt result = (aWide * bWide).urem(nWide).trunc(cEVMBitWidth);
	return getEVMIntegerAttr(getContext(), result);
}

mlir::OpFoldResult LogicalNotOp::fold(FoldAdaptor adaptor)
{
	auto operand = getConstBoolValue(adaptor.getOperand());
	if (!operand)
		return {};
	return mlir::BoolAttr::get(getContext(), !(*operand));
}

mlir::OpFoldResult LogicalAndOp::fold(FoldAdaptor adaptor)
{
	auto lhs = getConstBoolValue(adaptor.getLhs());
	auto rhs = getConstBoolValue(adaptor.getRhs());

	if (lhs && !(*lhs))
		return mlir::BoolAttr::get(getContext(), false);
	if (rhs && !(*rhs))
		return mlir::BoolAttr::get(getContext(), false);
	if (lhs && *lhs)
		return getRhs();
	if (rhs && *rhs)
		return getLhs();
	if (!lhs || !rhs)
		return {};

	return mlir::BoolAttr::get(getContext(), *lhs && *rhs);
}

mlir::OpFoldResult LogicalOrOp::fold(FoldAdaptor adaptor)
{
	auto lhs = getConstBoolValue(adaptor.getLhs());
	auto rhs = getConstBoolValue(adaptor.getRhs());

	if (lhs && *lhs)
		return mlir::BoolAttr::get(getContext(), true);
	if (rhs && *rhs)
		return mlir::BoolAttr::get(getContext(), true);
	if (lhs && !(*lhs))
		return getRhs();
	if (rhs && !(*rhs))
		return getLhs();
	if (!lhs || !rhs)
		return {};

	return mlir::BoolAttr::get(getContext(), *lhs || *rhs);
}

mlir::OpFoldResult SelectOp::fold(FoldAdaptor adaptor)
{
	if (getTrueValue() == getFalseValue())
		return getTrueValue();

	auto condition = getConstBoolValue(adaptor.getCondition());
	if (!condition)
		return {};

	return *condition ? mlir::OpFoldResult(getTrueValue()) : mlir::OpFoldResult(getFalseValue());
}

mlir::OpFoldResult ConvertOp::fold(FoldAdaptor adaptor)
{
	auto inputAttr = getConstIntAttr(adaptor.getInput());
	if (!inputAttr)
		return {};

	auto outputType = getOutput().getType();
	if (mlir::isa<mlir::solidity::BoolType>(outputType))
		return mlir::BoolAttr::get(getContext(), !inputAttr.getValue().isZero());

	auto width = getIntegerWidthFromType(outputType);
	if (!width)
		return {};

	llvm::APInt converted = inputAttr.getValue().zextOrTrunc(*width);
	return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), *width), converted);
}

mlir::OpFoldResult ToI1Op::fold(FoldAdaptor adaptor)
{
	auto input = getConstBoolValue(adaptor.getInput());
	if (!input)
		return {};

	return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 1), llvm::APInt(1, *input ? 1 : 0));
}

} // namespace solidity
} // namespace mlir

#else // SOLIDITY_HAS_MLIR

// Empty implementation when MLIR is not available

#endif // SOLIDITY_HAS_MLIR
