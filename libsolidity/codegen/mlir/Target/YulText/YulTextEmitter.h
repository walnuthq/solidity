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
 * Yul-text emitter: prints a module of `yul` dialect ops as Yul source that
 * libyul accepts (strict assembly). This keeps the existing EVM backend
 * reachable from the MLIR ladder and provides the differential anchor.
 */

#ifndef SOLIDITY_CODEGEN_MLIR_TARGET_YULTEXT_EMITTER_H
#define SOLIDITY_CODEGEN_MLIR_TARGET_YULTEXT_EMITTER_H

#include <string>

namespace mlir
{
class ModuleOp;
}

namespace solidity::mlirgen
{

/// Emits the given module of `yul` dialect ops as Yul source text
/// (a top-level block containing the functions and top-level statements).
/// The result is valid input for `solc --strict-assembly` / libyul.
std::string emitYulText(mlir::ModuleOp _module);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_TARGET_YULTEXT_EMITTER_H
