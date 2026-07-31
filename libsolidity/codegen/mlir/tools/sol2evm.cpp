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
 * sol2evm: drives Solidity down the whole ladder without leaving MLIR and
 * without solc's Yul pipeline -
 *
 *   Solidity AST -> sol dialect -> yul dialect -> evm dialect -> EVM assembly
 *
 * and reports the stage each contract reached. This is the instrument for the
 * rung that does not work yet: `SolToYul` covers a fraction of the `sol`
 * dialect, so what matters is which op stops each contract, not the total.
 *
 * Output: one greppable line per contract,
 *   RESULT contract="<name>" stage=<stage> bytes=N detail="..."
 */

#include "MLIRGenerator.h"
#include "SolToYul.h"
#include "YulToEVM.h"
#include "EVMAssemblyEmitter.h"
#include "Dialect/Sol/SolidityDialect.h"

#include <libsolidity/interface/CompilerStack.h>
#include <libsolidity/interface/OptimiserSettings.h>
#include <liblangutil/EVMVersion.h>
#include <libsolutil/CommonIO.h>

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#pragma GCC diagnostic pop

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace solidity;
using namespace solidity::frontend;

namespace
{

std::string quoted(std::string const& _text)
{
	std::string out;
	for (char c: _text)
		out += (c == '"' || c == '\n') ? '.' : c;
	return out;
}

} // anonymous namespace

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cerr << "usage: sol2evm <file.sol>" << std::endl;
		return 2;
	}

	std::ifstream file(argv[1]);
	if (!file)
	{
		std::cout << "PARSE-ERROR detail=\"cannot open " << quoted(argv[1]) << "\"" << std::endl;
		return 1;
	}
	std::stringstream buffer;
	buffer << file.rdbuf();

	CompilerStack stack;
	stack.setSources({{argv[1], buffer.str()}});
	stack.setOptimiserSettings(OptimiserSettings::minimal());
	if (!stack.parseAndAnalyze())
	{
		std::cout << "PARSE-ERROR detail=\"solc rejected the source\"" << std::endl;
		return 1;
	}

	mlir::MLIRContext context;
	context.disableMultithreading();
	// The generator emits structured control flow and arithmetic alongside the
	// sol dialect, so reading its output back needs all of them loaded.
	context.getOrLoadDialect<mlir::solidity::SolidityDialect>();
	context.getOrLoadDialect<mlir::arith::ArithDialect>();
	context.getOrLoadDialect<mlir::scf::SCFDialect>();
	context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
	context.getOrLoadDialect<mlir::func::FuncDialect>();

	for (std::string const& name: stack.contractNames())
	{
		std::string stage = "none";
		std::string detail;
		size_t byteCount = 0;

		ContractDefinition const& contract = stack.contractDefinition(name);
		MLIRGenerator generator(stack, langutil::EVMVersion{}, OptimiserSettings::minimal());

		std::string const text = generator.generate(contract);
		if (text.empty())
			detail = "the generator produced nothing";
		else
		{
			stage = "sol";
			mlir::OwningOpRef<mlir::ModuleOp> solModule
				= mlir::parseSourceString<mlir::ModuleOp>(text, &context);
			if (!solModule)
				// The parser verifies as it goes, so this is as often a
				// generator emitting ill-typed IR as it is bad syntax.
				detail = "the generated sol dialect does not parse or verify";
			else
			{
				std::string error;
				mlir::OwningOpRef<mlir::ModuleOp> yulModule
					= mlirgen::convertSolToYul(*solModule, error);
				if (!yulModule)
					detail = error;
				else
				{
					stage = "yul";
					mlir::OwningOpRef<mlir::ModuleOp> evmModule
						= mlirgen::convertYulToEVM(*yulModule, error);
					if (!evmModule)
						detail = error;
					else
					{
						stage = "evm";
						mlirgen::EVMAssemblyOptions options;
						options.name = name;
						std::shared_ptr<evmasm::Assembly> assembly
							= mlirgen::emitEVMAssembly(*evmModule, options, error);
						if (!assembly)
							detail = error;
						else
						{
							stage = "asm";
							try
							{
								byteCount = assembly->assemble().bytecode.size();
								stage = "bytecode";
								detail = std::to_string(byteCount) + " bytes";
							}
							catch (std::exception const& _exception)
							{
								detail = std::string("assemble failed: ") + _exception.what();
							}
						}
					}
				}
			}
		}

		std::cout << "RESULT contract=\"" << quoted(name) << "\" stage=" << stage << " bytes=" << byteCount
				  << " detail=\"" << quoted(detail) << "\"" << std::endl;
	}
	return 0;
}
