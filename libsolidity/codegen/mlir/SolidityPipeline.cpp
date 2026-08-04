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

#include "SolidityPipeline.h"

#include "Dialect/Sol/SolidityDialect.h"
#include "EVMAssemblyEmitter.h"
#include "EVMPipeline.h"
#include "MLIRGenerator.h"
#include "SolToYul.h"
#include "YulToEVM.h"

#include <libevmasm/Assembly.h>
#include <libsolidity/interface/CompilerStack.h>

// Disable warnings for LLVM/MLIR headers.
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

#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace solidity::mlirgen
{
namespace
{

constexpr std::string_view kCreationSuffix = "_new";

struct CompiledObject
{
	std::shared_ptr<evmasm::Assembly> assembly;
	std::vector<std::string> children;
};

class ActiveCompilation
{
public:
	ActiveCompilation(std::set<std::string>& _active, std::string _key): m_active(_active), m_key(std::move(_key)) {}
	~ActiveCompilation() { m_active.erase(m_key); }

private:
	std::set<std::string>& m_active;
	std::string m_key;
};

class SolidityPipeline
{
public:
	SolidityPipeline(
		frontend::CompilerStack const& _compilerStack,
		langutil::EVMVersion _evmVersion,
		frontend::OptimiserSettings const& _optimiserSettings,
		std::string& _error):
		m_compilerStack(_compilerStack),
		m_evmVersion(_evmVersion),
		m_optimiserSettings(_optimiserSettings),
		m_error(_error)
	{
		m_context.disableMultithreading();
		// MLIRGenerator emits these alongside the sol dialect. They must all be
		// loaded before its textual module is parsed back into the pipeline.
		m_context.getOrLoadDialect<mlir::solidity::SolidityDialect>();
		m_context.getOrLoadDialect<mlir::arith::ArithDialect>();
		m_context.getOrLoadDialect<mlir::scf::SCFDialect>();
		m_context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
		m_context.getOrLoadDialect<mlir::func::FuncDialect>();
	}

	std::optional<CompiledObject> compile(std::string const& _contractName)
	{
		std::optional<std::string> canonicalName = resolveContract(_contractName);
		if (!canonicalName)
			return std::nullopt;
		return compileObject(*canonicalName, /*creation=*/true);
	}

private:
	std::optional<std::string> resolveContract(std::string const& _name)
	{
		for (std::string const& candidate: m_compilerStack.contractNames())
			if (candidate == _name)
				return candidate;

		std::optional<std::string> match;
		for (std::string const& candidate: m_compilerStack.contractNames())
		{
			size_t const separator = candidate.rfind(':');
			std::string const shortName = separator == std::string::npos ? candidate : candidate.substr(separator + 1);
			if (shortName != _name)
				continue;
			if (match)
			{
				m_error = "referenced contract name '" + _name + "' is ambiguous between '" + *match +
					"' and '" + candidate + "'";
				return std::nullopt;
			}
			match = candidate;
		}

		if (!match)
			m_error = "referenced contract '" + _name + "' was not found in the analyzed sources";
		return match;
	}

	std::optional<mlir::OwningOpRef<mlir::ModuleOp>> generateSolModule(std::string const& _contractName)
	{
		frontend::MLIRGenerator generator(
			m_compilerStack, m_evmVersion, m_optimiserSettings, true /* legacy codegen semantics */);
		std::string const text = generator.generate(m_compilerStack.contractDefinition(_contractName));
		if (text.empty())
		{
			m_error = "sol generation for " + _contractName + " produced no module";
			return std::nullopt;
		}

		mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(text, &m_context);
		if (!module)
		{
			m_error = "generated sol dialect for " + _contractName + " did not parse or verify";
			return std::nullopt;
		}
		return std::move(module);
	}

	std::optional<CompiledObject> compileReference(std::string const& _objectName)
	{
		bool const creation = _objectName.size() > kCreationSuffix.size() &&
			_objectName.compare(
				_objectName.size() - kCreationSuffix.size(),
				kCreationSuffix.size(),
				kCreationSuffix) == 0;
		std::string const contractName = creation ?
			_objectName.substr(0, _objectName.size() - kCreationSuffix.size()) : _objectName;
		std::optional<std::string> canonicalName = resolveContract(contractName);
		if (!canonicalName)
			return std::nullopt;
		return compileObject(*canonicalName, creation);
	}

	std::optional<CompiledObject> compileObject(std::string const& _contractName, bool _creation)
	{
		std::string const key = _contractName + (_creation ? ":creation" : ":runtime");
		if (auto cached = m_cache.find(key); cached != m_cache.end())
			return cached->second;
		if (!m_active.insert(key).second)
		{
			m_error = "recursive contract object reference while compiling " + key;
			return std::nullopt;
		}
		ActiveCompilation activeCompilation(m_active, key);

		std::optional<CompiledObject> runtime;
		if (_creation)
		{
			runtime = compileObject(_contractName, /*creation=*/false);
			if (!runtime)
				return std::nullopt;
		}

		std::optional<mlir::OwningOpRef<mlir::ModuleOp>> solModule = generateSolModule(_contractName);
		if (!solModule)
			return std::nullopt;

		mlir::OwningOpRef<mlir::ModuleOp> yulModule
			= convertSolToYul(**solModule, m_error, _creation, m_evmVersion);
		if (!yulModule)
		{
			m_error = std::string(_creation ? "creation " : "runtime ") + _contractName + ": " + m_error;
			return std::nullopt;
		}
		// convertSolToYul owns this process-global set; capture it before a
		// recursive conversion replaces it.
		std::set<std::string> const referenced = lastReferencedContracts();

		mlir::OwningOpRef<mlir::ModuleOp> evmModule = convertYulToEVM(*yulModule, m_error);
		if (!evmModule)
		{
			m_error = std::string(_creation ? "creation " : "runtime ") + _contractName + ": " + m_error;
			return std::nullopt;
		}

		EVMAssemblyOptions options;
		options.name = _creation ? _contractName + "_creation" : _contractName;
		options.creation = _creation;
		options.evmVersion = m_evmVersion;
		if (runtime)
			options.subObjects.push_back({"runtime", runtime->assembly, runtime->children});

		for (std::string const& objectName: referenced)
		{
			std::optional<CompiledObject> object = compileReference(objectName);
			if (!object)
				return std::nullopt;
			options.subObjects.push_back({objectName, object->assembly, object->children});
		}

		std::shared_ptr<evmasm::Assembly> assembly = emitEVMAssembly(*evmModule, options, m_error);
		if (!assembly)
		{
			m_error = std::string(_creation ? "creation " : "runtime ") + _contractName + ": " + m_error;
			return std::nullopt;
		}

		CompiledObject result;
		result.assembly = std::move(assembly);
		for (EVMSubObject const& child: options.subObjects)
			result.children.push_back(child.name);
		m_cache.emplace(key, result);
		return result;
	}

	frontend::CompilerStack const& m_compilerStack;
	langutil::EVMVersion m_evmVersion;
	frontend::OptimiserSettings const& m_optimiserSettings;
	std::string& m_error;
	mlir::MLIRContext m_context;
	std::map<std::string, CompiledObject> m_cache;
	std::set<std::string> m_active;
};

} // anonymous namespace

bool compileSolidityToEVMBytecode(
	frontend::CompilerStack& _compilerStack,
	std::string const& _contractName,
	langutil::EVMVersion _evmVersion,
	frontend::OptimiserSettings const& _optimiserSettings,
	SolidityMLIRFrontend _frontend,
	solidity::bytes& _bytecode,
	std::string& _error)
{
	_error.clear();
	// Experimental Solidity uses a distinct type-inference graph and leaves
	// legacy AST Type pointers null by design. Its native front end already
	// produces strict Yul; import that object tree into the yul dialect so the
	// remainder is still the real MLIR -> evm -> assembly pipeline.
	if (_compilerStack.isExperimentalAnalysis() || _frontend == SolidityMLIRFrontend::YulIR)
	{
		std::optional<std::string> yul = _compilerStack.generateYulForMLIR(_contractName);
		if (!yul)
		{
			_error = "Solidity front end produced no Yul object";
			return false;
		}
		return compileYulToEVMBytecode(
			_contractName,
			*yul,
			_evmVersion,
			_optimiserSettings,
			_bytecode,
			_error,
			_compilerStack.libraries()
		);
	}
	SolidityPipeline pipeline(_compilerStack, _evmVersion, _optimiserSettings, _error);
	std::optional<CompiledObject> creation = pipeline.compile(_contractName);
	if (!creation)
		return false;

	try
	{
		evmasm::Assembly::OptimiserSettings assemblySettings =
			evmasm::Assembly::OptimiserSettings::translateSettings(_optimiserSettings);
		assemblySettings.runInliner = true;
		creation->assembly->optimise(assemblySettings);
		evmasm::LinkerObject linked = creation->assembly->assemble();
		linked.link(_compilerStack.libraries());
		_bytecode = std::move(linked.bytecode);
	}
	catch (std::exception const& exception)
	{
		_error = std::string("assembling ") + _contractName + " failed: " + exception.what();
		return false;
	}
	return true;
}

} // namespace solidity::mlirgen
