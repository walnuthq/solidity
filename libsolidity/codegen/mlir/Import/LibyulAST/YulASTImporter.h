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
 * libyul-AST importer: brings any Yul solc can parse (via-ir output,
 * hand-written strict assembly) into the `yul` MLIR dialect (M2b of the
 * SolcRISCV plan). Mutable Yul locals map to yul.var/yul.assign/
 * yul.var_load, so no SSA construction is needed here (ADR-004).
 */

#ifndef SOLIDITY_CODEGEN_MLIR_IMPORT_LIBYUL_AST_IMPORTER_H
#define SOLIDITY_CODEGEN_MLIR_IMPORT_LIBYUL_AST_IMPORTER_H

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#pragma GCC diagnostic pop

#include <string>
#include <vector>

namespace solidity::yul
{
class AST;
}

namespace solidity::mlirgen
{

/// Imports a parsed and analyzed Yul AST into a module of `yul` dialect ops.
/// On failure returns a null module and stores a message in _error.
mlir::OwningOpRef<mlir::ModuleOp>
importYulAST(solidity::yul::AST const& _ast, mlir::MLIRContext& _context, std::string& _error);

/// Convenience wrapper: parses and analyzes _source with libyul (strict
/// assembly), then imports the resulting AST.
mlir::OwningOpRef<mlir::ModuleOp> importYulSource(
	std::string const& _sourceName, std::string const& _source, mlir::MLIRContext& _context, std::string& _error);

struct ImportedObject
{
	std::string name;
	mlir::OwningOpRef<mlir::ModuleOp> module; ///< null when the import failed
	std::string error;
};

/// Parses and analyzes _source, then imports every Yul object in the object
/// tree (creation objects and deployed sub-objects) as a separate module -
/// the entry point for real-world via-ir output. On parse failure returns an
/// empty vector and sets _parseError.
std::vector<ImportedObject> importYulObjects(
	std::string const& _sourceName, std::string const& _source, mlir::MLIRContext& _context, std::string& _parseError);

} // namespace solidity::mlirgen

#endif // SOLIDITY_CODEGEN_MLIR_IMPORT_LIBYUL_AST_IMPORTER_H
