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
 * Implementation of the `yul` MLIR dialect.
 */

#include "YulDialect.h"
#include "YulOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#pragma GCC diagnostic pop

using namespace mlir;
using namespace mlir::yul;

// Generated dialect definitions (constructor, TypeID, ...).
#include "YulDialect.cpp.inc"

void YulDialect::initialize()
{
	addTypes<VarRefType>();

	addOperations<
#define GET_OP_LIST
#include "YulOps.cpp.inc"
		>();
}

/// Parse a type registered to this dialect (only !yul.varref for now).
mlir::Type YulDialect::parseType(mlir::DialectAsmParser& parser) const
{
	llvm::StringRef keyword;
	if (parser.parseKeyword(&keyword))
		return Type();

	if (keyword == "varref")
		return VarRefType::get(getContext());

	parser.emitError(parser.getNameLoc(), "unknown yul type: ") << keyword;
	return Type();
}

/// Print a type registered to this dialect.
void YulDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const
{
	if (llvm::isa<VarRefType>(type))
	{
		printer << "varref";
		return;
	}
	llvm_unreachable("unknown yul type in printType");
}

/// Materialize a constant for canonicalization/folding support.
mlir::Operation* YulDialect::materializeConstant(
	mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
{
	auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(value);
	if (!intAttr || !type.isSignlessInteger(256))
		return nullptr;
	return builder.create<ConstOp>(loc, type, intAttr);
}
