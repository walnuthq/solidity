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
#include <libsolutil/CommonData.h>
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
#include "llvm/Support/raw_ostream.h"
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
	std::string path;
	std::string emit;   // sol | yul | evm: print that rung's IR and stop
	bool printHex = false;
	for (int i = 1; i < argc; ++i)
	{
		std::string const argument = argv[i];
		if (argument == "--hex")
			printHex = true;
		else if (argument.rfind("--emit=", 0) == 0)
			emit = argument.substr(7);
		else if (path.empty())
			path = argument;
	}
	if (path.empty())
	{
		std::cerr << "usage: sol2evm <file.sol> [--hex]" << std::endl;
		return 2;
	}

	std::ifstream file(path);
	if (!file)
	{
		std::cout << "PARSE-ERROR detail=\"cannot open " << quoted(path) << "\"" << std::endl;
		return 1;
	}
	std::stringstream buffer;
	buffer << file.rdbuf();

	CompilerStack stack;
	stack.setSources({{path, buffer.str()}});
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
		if (emit == "sol")
		{
			std::cout << text << std::endl;
			continue;
		}
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
				else if (emit == "yul")
				{
					yulModule->print(llvm::outs());
					continue;
				}
				else
				{
					stage = "yul";
					mlir::OwningOpRef<mlir::ModuleOp> evmModule
						= mlirgen::convertYulToEVM(*yulModule, error);
					if (!evmModule)
						detail = error;
					else if (emit == "evm")
					{
						evmModule->print(llvm::outs());
						continue;
					}
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

							// The creation half runs the initialisers and the
							// constructor, then returns the runtime half, so it
							// has to carry that half as a nested object.
							std::string creationError;
							mlir::OwningOpRef<mlir::ModuleOp> creationYul
								= mlirgen::convertSolToYul(*solModule, creationError, /*creation=*/true);
							std::shared_ptr<evmasm::Assembly> creation;
							if (creationYul)
							{
								mlir::OwningOpRef<mlir::ModuleOp> creationEvm
									= mlirgen::convertYulToEVM(*creationYul, creationError);
								if (creationEvm)
								{
									mlirgen::EVMAssemblyOptions creationOptions;
									creationOptions.name = name + "_creation";
									creationOptions.creation = true;
									creationOptions.subObjects.push_back({"runtime", assembly, {}});
									creation = mlirgen::emitEVMAssembly(*creationEvm, creationOptions, creationError);
								}
							}

							try
							{
								solidity::evmasm::LinkerObject const& linked
									= (creation ? creation : assembly)->assemble();
								byteCount = linked.bytecode.size();
								stage = "bytecode";
								detail = std::to_string(byteCount) + " bytes";
								if (printHex)
									std::cout << "HEX contract=\"" << quoted(name) << "\" "
											  << solidity::util::toHex(linked.bytecode) << std::endl;
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
