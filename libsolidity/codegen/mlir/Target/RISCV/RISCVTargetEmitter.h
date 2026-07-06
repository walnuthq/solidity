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
 * RISC-V target emitter (M5): translates LLVM-dialect modules to LLVM IR
 * and emits RV32IM object code in-process (rv32im / ilp32 - the zkVM
 * profile baseline). Also provides by-pointer test wrappers so harnesses
 * (host JIT differential, qemu runners) can call i256 functions without
 * depending on the i256 calling convention.
 */

#ifndef SOLIDITY_CODEGEN_MLIR_TARGET_RISCV_EMITTER_H
#define SOLIDITY_CODEGEN_MLIR_TARGET_RISCV_EMITTER_H

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/BuiltinOps.h"
#pragma GCC diagnostic pop

#include <string>

namespace solidity::mlirgen
{

/// For every defined llvm.func with only-i256 params and a single i256
/// result, adds `void @__test_<name>(ptr result, ptr arg...)` that loads the
/// arguments, calls the function, and stores the result. @returns the number
/// of wrappers created.
unsigned addI256TestWrappers(mlir::ModuleOp _module);

/// Translates the LLVM-dialect module and emits an RV32IM (ilp32) object
/// file to _objectPath. Returns false and sets _error on failure.
bool emitRISCVObject(mlir::ModuleOp _module, std::string const& _objectPath, std::string& _error);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_TARGET_RISCV_EMITTER_H
