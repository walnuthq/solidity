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
 * Component that lowers MLIR to Yul IR.
 */

#pragma once

#include <string>
#include <memory>

namespace solidity::yul
{
class Object;
}

namespace solidity::frontend
{

/**
 * Lowering pass from MLIR to Yul IR.
 * Converts MLIR operations from the Solidity dialect to Yul AST.
 */
class MLIRToYulLowering
{
public:
	MLIRToYulLowering();
	~MLIRToYulLowering();
	
	/// Lower MLIR module to Yul AST
	/// @param _mlirModule The MLIR module in textual format
	/// @returns Yul Object containing the AST
	std::shared_ptr<yul::Object> lower(std::string const& _mlirModule);
	
	/// Apply optimization passes before lowering
	/// @param _mlirModule The MLIR module to optimize
	/// @param _printIntermediateMLIR Whether to print MLIR after each pass
	/// @param _mlirFile Path to write the optimized MLIR (empty string to skip)
	/// @returns Optimized MLIR module
	std::string optimize(std::string const& _mlirModule, bool _printIntermediateMLIR = false, std::string const& _mlirFile = "");

private:
	class MLIRToYulLoweringImpl;
	std::unique_ptr<MLIRToYulLoweringImpl> m_impl;
};

} // namespace solidity::frontend
