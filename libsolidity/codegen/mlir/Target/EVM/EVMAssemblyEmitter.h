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
 * EVM assembly target: lowers a module of `evm` dialect ops composed with
 * upstream arith/cf/func into libevmasm `Assembly`, completing the ladder
 *
 *   sol dialect -> yul dialect -> evm dialect -> EVM assembly -> bytecode
 *
 * without leaving MLIR until the final instruction stream.
 *
 * VALUE PLACEMENT (v0 - memory-resident SSA)
 *
 * Every SSA value that is not a constant is assigned a fixed 32-byte slot in a
 * per-function frame. Operands are materialised at each use (PUSH addr; MLOAD)
 * and results are written back (PUSH addr; MSTORE); constants are
 * rematerialised as a PUSH instead of being stored.
 *
 * The consequence is the invariant this backend is built on: outside the
 * emission of a single instruction, the EVM stack holds nothing but the chain
 * of pending return addresses. Stack-too-deep therefore cannot arise and no
 * stack scheduler is required to produce correct code. The cost is gas: every
 * value round-trips through memory. Replacing this with a real stack scheduler
 * (see solar's backend/evm/stack and LLVM's EVMStackSolver) is the next step,
 * and it can be done behind this same interface because the operand order
 * expected at each opcode is already explicit here.
 *
 * CALLING CONVENTION
 *
 * Arguments and results are passed in the callee's frame; the return address
 * is the only value passed on the stack. Because frames are addressed
 * absolutely, two live activations of one function would alias, so recursion
 * is detected up front and rejected rather than miscompiled.
 *
 * KNOWN DIVERGENCES (must be lifted before this is consensus-usable)
 *
 *  - MSIZE observes the frame region, so a contract that branches on MSIZE
 *    sees a different value than under the legacy backend.
 *  - The frame base is fixed rather than negotiated with `memoryguard`, so a
 *    contract whose heap grows past it would collide.
 */

#ifndef SOLIDITY_CODEGEN_MLIR_TARGET_EVM_ASSEMBLY_EMITTER_H
#define SOLIDITY_CODEGEN_MLIR_TARGET_EVM_ASSEMBLY_EMITTER_H

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/BuiltinOps.h"
#pragma GCC diagnostic pop

#include <libevmasm/Assembly.h>
#include <liblangutil/EVMVersion.h>

#include <cstdint>
#include <memory>
#include <string>

namespace solidity::mlirgen
{

struct EVMAssemblyOptions
{
	/// Byte offset of the first function frame. Everything from here up is
	/// owned by the backend.
	uint64_t frameBase = 0x10000;
	langutil::EVMVersion evmVersion{};
	/// Emitted assembly is a creation object rather than deployed code.
	bool creation = false;
	std::string name = "MLIR";
};

/// Lowers @a _module (evm + arith/cf/func ops, as produced by convertYulToEVM)
/// to EVM assembly. On failure returns nullptr and sets @a _error to a message
/// naming the construct that could not be emitted.
std::shared_ptr<solidity::evmasm::Assembly> emitEVMAssembly(
	mlir::ModuleOp _module,
	EVMAssemblyOptions const& _options,
	std::string& _error);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_TARGET_EVM_ASSEMBLY_EMITTER_H
