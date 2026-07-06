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
 * YulToEVM conversion (M4 scaffold): lowers `yul` dialect modules to the
 * `evm` dialect composed with upstream arith/cf/func —
 *
 *  - arith-exact builtins map to arith.* (ADR-003), comparisons to
 *    arith.cmpi (+ zero-extension back to i256);
 *  - EVM-semantics builtins map 1:1 to evm.*;
 *  - structured control flow (yul.if / yul.for with break/continue) is
 *    flattened to cf.br/cf.cond_br over explicit blocks;
 *  - yul.func / yul.func_call / yul.leave become func.func / func.call /
 *    func.return.
 *
 * Mutable variables must be promoted to SSA first; promoteBlockLocalVars
 * handles vars whose uses stay within their defining block (a full
 * region-aware mem2reg is a follow-up milestone task).
 */

#ifndef SOLIDITY_CODEGEN_MLIR_CONVERSION_YUL_TO_EVM_H
#define SOLIDITY_CODEGEN_MLIR_CONVERSION_YUL_TO_EVM_H

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#pragma GCC diagnostic pop

#include <string>

namespace solidity::mlirgen
{

/// Promotes yul.var variables whose loads/assigns all live in the variable's
/// own block to pure SSA values. @returns the number of variables that could
/// not be promoted (region-crossing; they require the full promotion pass).
unsigned promoteBlockLocalVars(mlir::ModuleOp _module);

/// Converts a var-free module of `yul` dialect ops into a fresh module of
/// evm/arith/cf/func ops. On failure returns null and sets _error.
mlir::OwningOpRef<mlir::ModuleOp> convertYulToEVM(mlir::ModuleOp _module, std::string& _error);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_CONVERSION_YUL_TO_EVM_H
