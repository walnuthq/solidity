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
 * The dialect ladder as one call, with no MLIR types in the interface.
 *
 * solc is built with pedantic warnings that the LLVM and MLIR headers do not
 * satisfy, and the suppressions for that live in this directory. Keeping the
 * entry point free of MLIR lets callers include it without inheriting either.
 */

#ifndef SOLIDITY_CODEGEN_MLIR_TARGET_EVM_PIPELINE_H
#define SOLIDITY_CODEGEN_MLIR_TARGET_EVM_PIPELINE_H

#include <liblangutil/EVMVersion.h>
#include <libsolutil/Common.h>

#include <string>

namespace solidity::mlirgen
{

/// Compiles @a _yulSource - a Yul object tree, as produced by `--ir-optimized` -
/// to creation bytecode by way of the yul and evm dialects. @a _name is used to
/// label the emitted assembly. On failure returns false and describes in
/// @a _error which object stopped the pipeline and why.
bool compileYulToEVMBytecode(
	std::string const& _name,
	std::string const& _yulSource,
	langutil::EVMVersion _evmVersion,
	solidity::bytes& _bytecode,
	std::string& _error);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_TARGET_EVM_PIPELINE_H
