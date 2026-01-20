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

#pragma once

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#pragma GCC diagnostic pop

namespace mlir {
namespace solidity {

//===----------------------------------------------------------------------===//
// Solidity Dialect
//===----------------------------------------------------------------------===//

class SolidityDialect : public mlir::Dialect {
public:
	explicit SolidityDialect(mlir::MLIRContext *context);

	static llvm::StringRef getDialectNamespace() { return "solidity"; }


	/// Parse a type registered to this dialect.
	mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

	/// Print a type registered to this dialect.
	void printType(mlir::Type type, mlir::DialectAsmPrinter &os) const override;

	/// Parse an attribute registered to this dialect.
	mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
	                                mlir::Type type) const override;

	/// Print an attribute registered to this dialect.
	void printAttribute(mlir::Attribute attr,
	                    mlir::DialectAsmPrinter &os) const override;
};

//===----------------------------------------------------------------------===//
// Solidity Types
//===----------------------------------------------------------------------===//

namespace detail {
struct UIntTypeStorage;
struct IntTypeStorage;
struct BytesTypeStorage;
struct ArrayTypeStorage;
} // namespace detail

/// Unsigned integer type
class UIntType : public mlir::Type::TypeBase<UIntType, mlir::Type, detail::UIntTypeStorage> {
public:
	using Base::Base;
	static constexpr llvm::StringLiteral name = "solidity.uint";
	
	static UIntType get(mlir::MLIRContext *context, unsigned bitwidth);
	
	unsigned getBitWidth() const;
};

/// Signed integer type
class IntType : public mlir::Type::TypeBase<IntType, mlir::Type, detail::IntTypeStorage> {
public:
	using Base::Base;
	static constexpr llvm::StringLiteral name = "solidity.int";
	
	static IntType get(mlir::MLIRContext *context, unsigned bitwidth);
	
	unsigned getBitWidth() const;
};

/// Address type (160-bit)
class AddressType : public mlir::Type::TypeBase<AddressType, mlir::Type, mlir::TypeStorage> {
public:
	using Base::Base;
	static constexpr llvm::StringLiteral name = "solidity.address";
	
	static AddressType get(mlir::MLIRContext *context);
};

/// Boolean type
class BoolType : public mlir::Type::TypeBase<BoolType, mlir::Type, mlir::TypeStorage> {
public:
	using Base::Base;
	static constexpr llvm::StringLiteral name = "solidity.bool";
	
	static BoolType get(mlir::MLIRContext *context);
};

/// Fixed-size bytes type
class BytesType : public mlir::Type::TypeBase<BytesType, mlir::Type, detail::BytesTypeStorage> {
public:
	using Base::Base;
	static constexpr llvm::StringLiteral name = "solidity.bytes";
	
	static BytesType get(mlir::MLIRContext *context, unsigned size);
	
	unsigned getSize() const;
};

/// Dynamic bytes type
class DynamicBytesType : public mlir::Type::TypeBase<DynamicBytesType, mlir::Type, mlir::TypeStorage> {
public:
	using Base::Base;
	static constexpr llvm::StringLiteral name = "solidity.dynamic_bytes";
	
	static DynamicBytesType get(mlir::MLIRContext *context);
};

/// String type
class StringType : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage> {
public:
	using Base::Base;
	static constexpr llvm::StringLiteral name = "solidity.string";
	
	static StringType get(mlir::MLIRContext *context);
};

/// Array type
class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type, detail::ArrayTypeStorage> {
public:
	using Base::Base;
	static constexpr llvm::StringLiteral name = "solidity.array";
	
	static ArrayType get(mlir::Type elementType, int64_t size);
	
	mlir::Type getElementType() const;
	int64_t getSize() const;
	bool isDynamicallySized() const { return getSize() == -1; }
};

//===----------------------------------------------------------------------===//
// Solidity Operations
//===----------------------------------------------------------------------===//

// Forward declarations for operations
class ContractOp;
class StateVarOp;
class FunctionOp;
class LoadStateVarOp;
class StoreStateVarOp;
class AddOp;
class SubOp;
class MulOp;
class DivOp;
class CmpOp;
class IfOp;
class ForOp;
class ReturnOp;
class RevertOp;
class ConstantOp;

} // namespace solidity
} // namespace mlir

// Include the auto-generated header file containing the declaration of the Solidity operations.
// This would be generated from TableGen but we'll define them manually for now
// TODO: Enable this once TableGen is properly set up
// #define GET_OP_CLASSES
// #include "SolidityOps.h.inc"