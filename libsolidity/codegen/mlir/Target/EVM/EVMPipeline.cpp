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

#include "EVMPipeline.h"

#include "EVMAssemblyEmitter.h"
#include "YulASTImporter.h"
#include "YulToEVM.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/MLIRContext.h"
#pragma GCC diagnostic pop

#include <libevmasm/Assembly.h>

#include <boost/algorithm/string/predicate.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

bool solidity::mlirgen::compileYulToEVMBytecode(
	std::string const& _name,
	std::string const& _yulSource,
	langutil::EVMVersion _evmVersion,
	frontend::OptimiserSettings const& _optimiserSettings,
	solidity::bytes& _bytecode,
	std::string& _error,
	std::map<std::string, solidity::util::h160> const& _libraries)
{
	mlir::MLIRContext context;
	context.disableMultithreading();

	std::vector<ImportedObject> objects = importYulObjects(_name, _yulSource, context, _error, _evmVersion);
	if (objects.empty())
		return false;
	if (std::getenv("SOL_MLIR_DUMP_OBJECTS"))
	{
		std::cerr << "MLIR Yul object tree for " << _name << ":\n";
		for (size_t index = 0; index < objects.size(); ++index)
		{
			std::cerr << "  [" << index << "] " << objects[index].name << " children:";
			for (size_t child: objects[index].subObjects)
				std::cerr << " [" << child << "] " << objects[child].name;
			std::cerr << '\n';
		}
	}

	// A nested object has to exist before the object naming it, so emission
	// runs leaf-first and the root - the creation code - is finished last.
	std::vector<std::shared_ptr<evmasm::Assembly>> emitted(objects.size());
	for (size_t index = objects.size(); index > 0; --index)
	{
		size_t const current = index - 1;
		ImportedObject& object = objects[current];
		if (!object.module)
		{
			_error = "object " + object.name + ": " + object.error;
			return false;
		}

		promoteBlockLocalVars(*object.module);
		mlir::OwningOpRef<mlir::ModuleOp> evmModule = convertYulToEVM(*object.module, _error);
		if (!evmModule)
		{
			_error = "object " + object.name + ": " + _error;
			return false;
		}

		EVMAssemblyOptions options;
		options.name = object.name;
		options.creation = !boost::algorithm::ends_with(object.name, "_deployed");
		options.evmVersion = _evmVersion;
		for (size_t child: object.subObjects)
		{
			std::vector<std::string> grandchildren;
			for (size_t inner: objects[child].subObjects)
				grandchildren.push_back(objects[inner].name);
			options.subObjects.push_back({objects[child].name, emitted[child], grandchildren});
		}
		for (auto const& segment: object.dataSegments)
			options.dataSegments.push_back({segment.name, segment.data});

		emitted[current] = emitEVMAssembly(*evmModule, options, _error);
		if (!emitted[current])
		{
			_error = "object " + object.name + ": " + _error;
			return false;
		}
		if (std::getenv("SOL_MLIR_DUMP_OBJECTS"))
			std::cerr << "  emitted [" << current << "] " << object.name << " subassemblies="
				<< options.subObjects.size() << '\n';
	}

	try
	{
		evmasm::Assembly::OptimiserSettings assemblySettings =
			evmasm::Assembly::OptimiserSettings::translateSettings(_optimiserSettings);
		// The MLIR emitter represents calls with explicit tag/jump blocks. Fold
		// the profitable ones as a mandatory target cleanup; the remaining
		// optimizer switches still follow the user's settings exactly.
		assemblySettings.runInliner = true;
		emitted.front()->optimise(assemblySettings);
		if (std::getenv("SOL_MLIR_DUMP_OBJECTS"))
			for (size_t index = 0; index < objects.size(); ++index)
				std::cerr << "  assembled [" << index << "] " << objects[index].name << " bytes="
					<< emitted[index]->assemble().bytecode.size() << '\n';
		evmasm::LinkerObject linked = emitted.front()->assemble();
		linked.link(_libraries);
		_bytecode = std::move(linked.bytecode);
	}
	catch (std::exception const& _exception)
	{
		_error = std::string("assembling failed: ") + _exception.what();
		return false;
	}
	return true;
}
