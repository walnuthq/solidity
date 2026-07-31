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
 * The `evm` MLIR dialect - rung 3 of the solc dialect ladder. See
 * EVMDialect.td for the design notes. Note: this is the MLIR dialect for
 * machine-level EVM semantics; not to be confused with libyul's
 * solidity::yul::EVMDialect (the Yul builtin table).
 */

#ifndef SOLIDITY_CODEGEN_MLIR_EVM_DIALECT_H
#define SOLIDITY_CODEGEN_MLIR_EVM_DIALECT_H

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#pragma GCC diagnostic pop

// Generated dialect declarations.
namespace mlir::evm
{

// EVM state lives in address spaces that cannot alias one another, so each gets
// its own side-effect resource. Without this every state op would appear to
// touch one undifferentiated resource, and an `sload` could not be moved or
// merged across an `mstore` that has nothing to do with it.
struct StorageResource: public ::mlir::SideEffects::Resource::Base<StorageResource>
{
	::llvm::StringRef getName() final { return "EVMStorage"; }
};

struct TransientStorageResource: public ::mlir::SideEffects::Resource::Base<TransientStorageResource>
{
	::llvm::StringRef getName() final { return "EVMTransientStorage"; }
};

struct MemoryResource: public ::mlir::SideEffects::Resource::Base<MemoryResource>
{
	::llvm::StringRef getName() final { return "EVMMemory"; }
};

} // namespace mlir::evm

#include "EVMDialect.h.inc"

#endif // SOLIDITY_CODEGEN_MLIR_EVM_DIALECT_H
