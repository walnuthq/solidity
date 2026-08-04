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
 * The complete compiler-facing MLIR backend as one call.
 *
 * The public interface deliberately contains no MLIR types. This keeps LLVM
 * and MLIR headers, and the warning suppressions they require, out of solc's
 * command-line layer.
 */

#ifndef SOLIDITY_CODEGEN_MLIR_SOLIDITY_PIPELINE_H
#define SOLIDITY_CODEGEN_MLIR_SOLIDITY_PIPELINE_H

#include <liblangutil/EVMVersion.h>
#include <libsolidity/interface/OptimiserSettings.h>
#include <libsolutil/Common.h>

#include <string>

namespace solidity::frontend
{
class CompilerStack;
}

namespace solidity::mlirgen
{

enum class SolidityMLIRFrontend
{
	/// Import solc's production Solidity-to-Yul output into the Yul dialect.
	YulIR,
	/// Lower the analyzed legacy AST through the typed sol dialect first.
	LegacySolidity
};

/// Compiles one analyzed Solidity contract to deployable creation bytecode.
/// The production frontend imports solc's strict Yul into the yul dialect;
/// legacy-only tests can instead enter through the typed sol dialect.
///
/// Runtime and creation modules are lowered independently. Contracts named by
/// `new C(...)` or `type(C).runtimeCode` are recursively compiled and attached
/// as sub-objects. Experimental Solidity, whose AST deliberately has no legacy
/// Type annotations, enters at the generated-Yul rung and still traverses the
/// yul and evm MLIR dialects. On failure returns false and describes the rung
/// that stopped the pipeline in @a _error.
bool compileSolidityToEVMBytecode(
	frontend::CompilerStack& _compilerStack,
	std::string const& _contractName,
	langutil::EVMVersion _evmVersion,
	frontend::OptimiserSettings const& _optimiserSettings,
	SolidityMLIRFrontend _frontend,
	solidity::bytes& _bytecode,
	std::string& _error);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_SOLIDITY_PIPELINE_H
