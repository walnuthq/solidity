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
 * EVMToLLVM conversion (M5): lowers the evm/arith/cf/func mix produced by
 * YulToEVM into the LLVM dialect —
 *
 *  - upstream arith/cf/func convert via their standard LLVM patterns
 *    (i256 is a legal LLVM type; the backend legalizes it);
 *  - the landmine ops with runtime-sized cost (div/sdiv/mod/smod/addmod/
 *    mulmod/exp/byte/signextend) become by-pointer calls into evm-rt
 *    (__evm_rt_*), with alloca'd i256 operand slots;
 *  - shl/shr/sar legalize inline: the shift amount is clamped below 256
 *    and the EVM-defined out-of-range result (0, or the sign fill for sar)
 *    is selected - upstream shifts would be poison there.
 *
 * Scope: the pure-computation subset. State/env/call/memory ops lower to
 * ERHI host calls in the next milestone (M6).
 */

#ifndef SOLIDITY_CODEGEN_MLIR_CONVERSION_EVM_TO_LLVM_H
#define SOLIDITY_CODEGEN_MLIR_CONVERSION_EVM_TO_LLVM_H

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/BuiltinOps.h"
#pragma GCC diagnostic pop

#include <string>

namespace solidity::mlirgen
{

/// Converts the module in place to the LLVM dialect. On failure returns
/// false and sets _error.
bool convertEVMToLLVM(mlir::ModuleOp _module, std::string& _error);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_CONVERSION_EVM_TO_LLVM_H
