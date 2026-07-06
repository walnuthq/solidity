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

#include "SolidityDialect.h"
#include "SolidityOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#pragma GCC diagnostic pop

using namespace mlir;
using namespace mlir::solidity;

#ifdef SOLIDITY_HAS_MLIR

//===----------------------------------------------------------------------===//
// Type Storage Definitions
//===----------------------------------------------------------------------===//

namespace mlir
{
namespace solidity
{
namespace detail
{

struct UIntTypeStorage: public mlir::TypeStorage
{
	UIntTypeStorage(unsigned bitwidth): bitwidth(bitwidth) {}

	using KeyTy = unsigned;

	bool operator==(const KeyTy& key) const { return key == bitwidth; }

	static UIntTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
	{
		return new (allocator.allocate<UIntTypeStorage>()) UIntTypeStorage(key);
	}

	unsigned bitwidth;
};

struct IntTypeStorage: public mlir::TypeStorage
{
	IntTypeStorage(unsigned bitwidth): bitwidth(bitwidth) {}

	using KeyTy = unsigned;

	bool operator==(const KeyTy& key) const { return key == bitwidth; }

	static IntTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
	{
		return new (allocator.allocate<IntTypeStorage>()) IntTypeStorage(key);
	}

	unsigned bitwidth;
};

struct BytesTypeStorage: public mlir::TypeStorage
{
	BytesTypeStorage(unsigned size): size(size) {}

	using KeyTy = unsigned;

	bool operator==(const KeyTy& key) const { return key == size; }

	static BytesTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
	{
		return new (allocator.allocate<BytesTypeStorage>()) BytesTypeStorage(key);
	}

	unsigned size;
};

struct ArrayTypeStorage: public mlir::TypeStorage
{
	ArrayTypeStorage(mlir::Type elementType, int64_t size): elementType(elementType), size(size) {}

	using KeyTy = std::pair<mlir::Type, int64_t>;

	bool operator==(const KeyTy& key) const { return key.first == elementType && key.second == size; }

	static ArrayTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
	{
		return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key.first, key.second);
	}

	mlir::Type elementType;
	int64_t size;
};

} // namespace detail
} // namespace solidity
} // namespace mlir

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

UIntType UIntType::get(mlir::MLIRContext* context, unsigned bitwidth) { return Base::get(context, bitwidth); }

unsigned UIntType::getBitWidth() const { return getImpl()->bitwidth; }

IntType IntType::get(mlir::MLIRContext* context, unsigned bitwidth) { return Base::get(context, bitwidth); }

unsigned IntType::getBitWidth() const { return getImpl()->bitwidth; }

AddressType AddressType::get(mlir::MLIRContext* context) { return Base::get(context); }

BoolType BoolType::get(mlir::MLIRContext* context) { return Base::get(context); }

BytesType BytesType::get(mlir::MLIRContext* context, unsigned size) { return Base::get(context, size); }

unsigned BytesType::getSize() const { return getImpl()->size; }

DynamicBytesType DynamicBytesType::get(mlir::MLIRContext* context) { return Base::get(context); }

StringType StringType::get(mlir::MLIRContext* context) { return Base::get(context); }

ArrayType ArrayType::get(mlir::Type elementType, int64_t size)
{
	return Base::get(elementType.getContext(), elementType, size);
}

mlir::Type ArrayType::getElementType() const { return getImpl()->elementType; }

int64_t ArrayType::getSize() const { return getImpl()->size; }

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

SolidityDialect::SolidityDialect(mlir::MLIRContext* context)
	: mlir::Dialect(getDialectNamespace(), context, mlir::TypeID::get<SolidityDialect>())
{
	// Register Solidity types
	addTypes<UIntType, IntType, AddressType, BoolType, BytesType, DynamicBytesType, StringType, ArrayType>();

	// Register operations generated from TableGen
	addOperations<
#define GET_OP_LIST
#include "SolidityOps.cpp.inc"
		>();
}

/// Parse a type registered to this dialect.
mlir::Type SolidityDialect::parseType(mlir::DialectAsmParser& parser) const
{
	llvm::StringRef keyword;
	if (parser.parseKeyword(&keyword))
		return Type();

	// Parse uint<N>
	if (keyword == "uint")
	{
		if (parser.parseLess())
			return Type();

		unsigned bitwidth;
		if (parser.parseInteger(bitwidth))
			return Type();

		if (parser.parseGreater())
			return Type();

		return UIntType::get(getContext(), bitwidth);
	}

	// Parse int<N>
	if (keyword == "int")
	{
		if (parser.parseLess())
			return Type();

		unsigned bitwidth;
		if (parser.parseInteger(bitwidth))
			return Type();

		if (parser.parseGreater())
			return Type();

		return IntType::get(getContext(), bitwidth);
	}

	// Parse address
	if (keyword == "address")
	{
		return AddressType::get(getContext());
	}

	// Parse bool
	if (keyword == "bool")
	{
		return BoolType::get(getContext());
	}

	// Parse bytes<N> (fixed) or bytes (dynamic)
	if (keyword == "bytes")
	{
		// Use parseOptionalLess to avoid emitting "expected '<'" errors for dynamic bytes
		if (parser.parseOptionalLess().succeeded())
		{
			unsigned size;
			if (parser.parseInteger(size))
				return Type();

			if (parser.parseGreater())
				return Type();

			return BytesType::get(getContext(), size);
		}
		else
		{
			// Dynamic bytes (no <N> suffix)
			return DynamicBytesType::get(getContext());
		}
	}

	// Parse string
	if (keyword == "string")
	{
		return StringType::get(getContext());
	}

	// Parse array<ElementType, Size>
	if (keyword == "array")
	{
		if (parser.parseLess())
			return Type();

		mlir::Type elementType;
		if (parser.parseType(elementType))
			return Type();

		if (parser.parseComma())
			return Type();

		int64_t size;
		if (parser.parseInteger(size))
			return Type();

		if (parser.parseGreater())
			return Type();

		return ArrayType::get(elementType, size);
	}

	parser.emitError(parser.getNameLoc(), "unknown Solidity type: ") << keyword;
	return Type();
}

/// Print a type registered to this dialect.
void SolidityDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const
{
	llvm::TypeSwitch<mlir::Type>(type)
		.Case<UIntType>([&](UIntType t) { os << "uint<" << t.getBitWidth() << ">"; })
		.Case<IntType>([&](IntType t) { os << "int<" << t.getBitWidth() << ">"; })
		.Case<AddressType>([&](AddressType) { os << "address"; })
		.Case<BoolType>([&](BoolType) { os << "bool"; })
		.Case<BytesType>([&](BytesType t) { os << "bytes<" << t.getSize() << ">"; })
		.Case<DynamicBytesType>([&](DynamicBytesType) { os << "bytes"; })
		.Case<StringType>([&](StringType) { os << "string"; })
		.Case<ArrayType>(
			[&](ArrayType t)
			{
				os << "array<";
				os.printType(t.getElementType());
				os << ", " << t.getSize() << ">";
			})
		.Default([&](Type) { llvm::errs() << "unknown type\n"; });
}

/// Parse an attribute registered to this dialect.
mlir::Attribute SolidityDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const
{
	// For now, we don't have custom attributes
	return Attribute();
}

/// Print an attribute registered to this dialect.
void SolidityDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& os) const
{
	// For now, we don't have custom attributes
}

mlir::Operation* SolidityDialect::materializeConstant(
	mlir::OpBuilder& builder,
	mlir::Attribute value,
	mlir::Type type,
	mlir::Location loc)
{
	if (isa<mlir::IntegerAttr>(value) || isa<mlir::BoolAttr>(value))
		return builder.create<mlir::solidity::ConstantOp>(loc, value, type);

	return nullptr;
}

//===----------------------------------------------------------------------===//
// Operation Definitions
//===----------------------------------------------------------------------===//

// Auto-generated operation definitions are included in SolidityOps.cpp

//===----------------------------------------------------------------------===//
// Dialect Registration
//===----------------------------------------------------------------------===//

// Not needed - dialects are registered through context->getOrLoadDialect()

#else // !SOLIDITY_HAS_MLIR

// Stub implementations when MLIR is not available

#endif // SOLIDITY_HAS_MLIR
