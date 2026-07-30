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
 * yul2evm: drives real-world Yul (e.g. `solc --ir-optimized` output) down the
 * MLIR ladder to EVM bytecode and reports the stage reached per object - the
 * coverage instrument for the EVM backend, mirroring yul2rv.
 *
 *   stages: import -> promote -> evm -> asm -> bytecode
 *
 * Output: one greppable line per object,
 *   RESULT object="<name>" stage=<stage> funcs=N ops=N bytes=N detail="..."
 */

#include "EVMAssemblyEmitter.h"
#include "YulASTImporter.h"
#include "YulOps.h"
#include "YulToEVM.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/MLIRContext.h"
#pragma GCC diagnostic pop

#include <libevmasm/Assembly.h>
#include <libsolutil/CommonData.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace
{

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
	bool printAsm = false;
	bool printHex = false;
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if (arg == "--asm")
			printAsm = true;
		else if (arg == "--hex")
			printHex = true;
		else if (inputPath.empty())
			inputPath = arg;
		else
		{
			std::cerr << "usage: yul2evm <file.yul> [--asm] [--hex]" << std::endl;
			return 2;
		}
	}
	if (inputPath.empty())
	{
		std::cerr << "usage: yul2evm <file.yul> [--asm] [--hex]" << std::endl;
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

	// Objects arrive in pre-order, so emitting back to front guarantees a
	// nested object's assembly exists before its parent registers it as a sub.
	std::vector<std::shared_ptr<solidity::evmasm::Assembly>> emitted(objects.size());
	std::vector<std::string> stages(objects.size(), "none");
	std::vector<std::string> details(objects.size());
	std::vector<size_t> byteCounts(objects.size(), 0);
	std::vector<unsigned> funcCounts(objects.size(), 0), opCounts(objects.size(), 0);
	std::vector<std::string> hexCode(objects.size()), asmText(objects.size());

	for (size_t index = objects.size(); index > 0; --index)
	{
		size_t const current = index - 1;
		auto& object = objects[current];
		std::string& stage = stages[current];
		std::string& detail = details[current];
		unsigned& funcs = funcCounts[current];
		unsigned& ops = opCounts[current];
		size_t& byteCount = byteCounts[current];
		detail = object.error;

		if (object.module)
		{
			stage = "import";
			object.module->walk([&](mlir::Operation*) { ++ops; });
			object.module->walk([&](mlir::yul::FuncOp) { ++funcs; });

			solidity::mlirgen::promoteBlockLocalVars(*object.module);
			stage = "promote";

			std::string error;
			mlir::OwningOpRef<mlir::ModuleOp> evmModule
				= solidity::mlirgen::convertYulToEVM(*object.module, error);
			if (!evmModule)
				detail = error;
			else
			{
				stage = "evm";
				solidity::mlirgen::EVMAssemblyOptions options;
				options.name = object.name;
				// The root object is creation code; everything nested in it is
				// the deployed artifact that code returns.
				options.creation = (current == 0);
				for (size_t child: object.subObjects)
				{
					std::vector<std::string> grandchildren;
					for (size_t inner: objects[child].subObjects)
						grandchildren.push_back(objects[inner].name);
					options.subObjects.push_back({objects[child].name, emitted[child], grandchildren});
				}
				for (auto const& segment: object.dataSegments)
					options.dataSegments.push_back({segment.name, segment.data});

				std::shared_ptr<solidity::evmasm::Assembly> assembly
					= solidity::mlirgen::emitEVMAssembly(*evmModule, options, error);
				if (!assembly)
					detail = error;
				else
				{
					stage = "asm";
					emitted[current] = assembly;
					try
					{
						solidity::evmasm::LinkerObject const& linked = assembly->assemble();
						byteCount = linked.bytecode.size();
						stage = "bytecode";
						detail = std::to_string(byteCount) + " bytes";
						if (printAsm)
							asmText[current] = assembly->assemblyString({});
						if (printHex)
							hexCode[current] = solidity::util::toHex(linked.bytecode);
					}
					catch (std::exception const& _e)
					{
						detail = std::string("assemble failed: ") + _e.what();
					}
				}
			}
		}
	}

	// Report in declaration order even though emission ran the other way, so
	// the root object - the creation code - is always reported first.
	for (size_t index = 0; index < objects.size(); ++index)
	{
		if (!asmText[index].empty())
			std::cout << asmText[index] << std::endl;
		if (!hexCode[index].empty())
			std::cout << "HEX object=\"" << quoted(objects[index].name) << "\" " << hexCode[index] << std::endl;
	}
	for (size_t index = 0; index < objects.size(); ++index)
		std::cout << "RESULT object=\"" << quoted(objects[index].name) << "\" stage=" << stages[index]
				  << " funcs=" << funcCounts[index] << " ops=" << opCounts[index] << " bytes=" << byteCounts[index]
				  << " detail=\"" << quoted(details[index]) << "\"" << std::endl;
	return 0;
}
