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
 * SolToYul conversion (M2 scaffold): lowers `sol` dialect modules (contracts,
 * functions, state variables, checked-free arithmetic, structured control
 * flow) into the `yul` dialect. Descends the ladder one rung:
 *
 *   sol dialect -> yul dialect -> evm dialect -> RISC-V
 *
 * Scaffold semantics: all Solidity value types map to i256 words with
 * EVM-representation conversions (masking / signextend / bool
 * normalization); state variables get sequential storage slots; each
 * sol.func becomes a yul.func (the external-ABI dispatcher belongs to the
 * evm.program objects work and is not generated here yet).
 */

#ifndef SOLIDITY_CODEGEN_MLIR_CONVERSION_SOL_TO_YUL_H
#define SOLIDITY_CODEGEN_MLIR_CONVERSION_SOL_TO_YUL_H

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

/// Converts a module of `sol` dialect ops into a fresh module of `yul`
/// dialect ops. On failure returns null and sets _error.
mlir::OwningOpRef<mlir::ModuleOp> convertSolToYul(mlir::ModuleOp _module, std::string& _error);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_CONVERSION_SOL_TO_YUL_H
