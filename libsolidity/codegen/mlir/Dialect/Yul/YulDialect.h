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
 * The `yul` MLIR dialect - rung 2 of the solc dialect ladder
 * (sol -> yul -> evm -> RISC-V). See YulDialect.td for the design notes.
 */

#ifndef SOLIDITY_CODEGEN_MLIR_YUL_DIALECT_H
#define SOLIDITY_CODEGEN_MLIR_YUL_DIALECT_H

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#pragma GCC diagnostic pop

// Generated dialect declarations.
#include "YulDialect.h.inc"

namespace mlir
{
namespace yul
{

/// Reference to a mutable Yul variable. Yul locals are mutable; the importer
/// models `let`/assignments with yul.var / yul.assign / yul.var_load over
/// this type, and an SSA-promotion pass eliminates it before lowering to the
/// evm dialect.
class VarRefType: public mlir::Type::TypeBase<VarRefType, mlir::Type, mlir::TypeStorage>
{
public:
	using Base::Base;

	static VarRefType get(mlir::MLIRContext* _ctx) { return Base::get(_ctx); }

	static constexpr mlir::StringLiteral name = "yul.varref";
};

} // namespace yul
} // namespace mlir

#endif // SOLIDITY_CODEGEN_MLIR_YUL_DIALECT_H
