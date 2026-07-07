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
 * yul2rv: drives real-world Yul (e.g. `solc --ir-optimized` output) down the
 * MLIR ladder as far as it currently goes and reports the stage reached per
 * object - the coverage instrument for the benchmark runner.
 *
 *   stages: import -> promote -> evm -> llvm -> riscv
 *
 * Output: one greppable line per object,
 *   RESULT object="<name>" stage=<stage> funcs=N ops=N unpromoted=N detail="..."
 */

#include "EVMToLLVM.h"
#include "RISCVTargetEmitter.h"
#include "YulASTImporter.h"
#include "YulOps.h"
#include "YulToEVM.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/Support/FileSystem.h"
#include "mlir/IR/MLIRContext.h"
#pragma GCC diagnostic pop

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace
{

std::string sanitizeFileName(std::string _name)
{
	for (char& c: _name)
		if (!isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-' && c != '.')
			c = '_';
	return _name.empty() ? "unnamed" : _name;
}

std::string quoted(std::string const& _s)
{
	std::string out;
	for (char c: _s)
		out += (c == '"' || c == '\n') ? '.' : c;
	return out;
}

} // anonymous namespace

int main(int argc, char** argv)
{
	std::string inputPath;
	std::string objDir;
	bool wrappers = false;
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if (arg == "--obj-dir" && i + 1 < argc)
			objDir = argv[++i];
		else if (arg == "--wrappers")
			wrappers = true;
		else if (inputPath.empty())
			inputPath = arg;
		else
		{
			std::cerr << "usage: yul2rv <file.yul> [--obj-dir <dir>] [--wrappers]" << std::endl;
			return 2;
		}
	}
	if (inputPath.empty())
	{
		std::cerr << "usage: yul2rv <file.yul> [--obj-dir <dir>] [--wrappers]" << std::endl;
		return 2;
	}

	std::ifstream file(inputPath);
	if (!file)
	{
		std::cout << "PARSE-ERROR detail=\"cannot open " << quoted(inputPath) << "\"" << std::endl;
		return 1;
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	std::string source = buffer.str();

	mlir::MLIRContext ctx;
	ctx.disableMultithreading(); // deterministic, cheaper for a CLI driver
	std::string parseError;
	auto objects = solidity::mlirgen::importYulObjects(inputPath, source, ctx, parseError);
	if (objects.empty())
	{
		std::cout << "PARSE-ERROR detail=\"" << quoted(parseError) << "\"" << std::endl;
		return 1;
	}

	if (!objDir.empty())
		llvm::sys::fs::create_directories(objDir);

	for (auto& object: objects)
	{
		std::string stage = "none";
		std::string detail = object.error;
		unsigned funcs = 0, ops = 0, unpromoted = 0;

		if (object.module)
		{
			stage = "import";
			object.module->walk([&](mlir::Operation*) { ++ops; });
			object.module->walk([&](mlir::yul::FuncOp) { ++funcs; });

			// Block-local cleanup; region-crossing vars are handled by the
			// converter's structured-CF SSA construction.
			unpromoted = solidity::mlirgen::promoteBlockLocalVars(*object.module);
			{
				stage = "promote";
				std::string error;
				mlir::OwningOpRef<mlir::ModuleOp> evmModule =
					solidity::mlirgen::convertYulToEVM(*object.module, error);
				if (!evmModule)
					detail = error;
				else
				{
					stage = "evm";
					if (!solidity::mlirgen::convertEVMToLLVM(*evmModule, error))
						detail = error;
					else
					{
						stage = "llvm";
						if (wrappers)
							solidity::mlirgen::addI256TestWrappers(*evmModule);
						std::string objPath = (objDir.empty() ? std::string("/tmp") : objDir) + "/"
											  + sanitizeFileName(object.name) + ".o";
						if (!solidity::mlirgen::emitRISCVObject(*evmModule, objPath, error))
							detail = error;
						else
						{
							stage = "riscv";
							detail = objPath;
						}
					}
				}
			}
		}

		std::cout << "RESULT object=\"" << quoted(object.name) << "\" stage=" << stage << " funcs=" << funcs
				  << " ops=" << ops << " unpromoted=" << unpromoted << " detail=\"" << quoted(detail) << "\""
				  << std::endl;
	}
	return 0;
}
