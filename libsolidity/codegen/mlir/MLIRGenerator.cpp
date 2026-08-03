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

#include <libsolidity/ast/AST.h>
#include <libsolidity/ast/ASTAnnotations.h>
#include <libsolidity/ast/Types.h>
#include <libsolidity/ast/TypeProvider.h>
#include <libsolidity/codegen/mlir/MLIRGenerator.h>
#include <libsolidity/interface/CompilerStack.h>
#include <libsolutil/FunctionSelector.h>
#include <libyul/AST.h>
#include <libyul/AsmPrinter.h>
#include <libyul/backends/evm/EVMDialect.h>

#include <cxxabi.h>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <typeinfo>

#ifdef SOLIDITY_HAS_MLIR
// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#pragma GCC diagnostic pop
#include "Dialect/Sol/SolidityDialect.h"
#include "Dialect/Sol/SolidityOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#endif

namespace solidity::frontend
{

namespace
{
std::string demangle(char const* name)
{
	int status = 0;
	char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
	std::string result = (status == 0 && demangled) ? demangled : name;
	free(demangled);
	return result;
}
} // anonymous namespace

// Private implementation class
class MLIRGenerator::MLIRGeneratorImpl
{
public:
	MLIRGeneratorImpl(
		CompilerStack const& _compilerStack,
		langutil::EVMVersion _evmVersion,
		OptimiserSettings const& _optimiserSettings)
		: m_compilerStack(_compilerStack), m_evmVersion(_evmVersion), m_optimiserSettings(_optimiserSettings)
	{
#ifdef SOLIDITY_HAS_MLIR
		// Initialize MLIR context and builder
		m_context = std::make_unique<mlir::MLIRContext>();
		// Load standard and Solidity dialects
		m_context->getOrLoadDialect<mlir::func::FuncDialect>();
		m_context->getOrLoadDialect<mlir::arith::ArithDialect>();
		m_context->getOrLoadDialect<mlir::solidity::SolidityDialect>();
		m_context->getOrLoadDialect<mlir::scf::SCFDialect>();

		// Create a module
		m_builder = std::make_unique<mlir::OpBuilder>(m_context.get());
		// Module itself doesn't have a source location, use unknown
		m_module = mlir::ModuleOp::create(m_builder->getUnknownLoc());
#endif
	}

#ifdef SOLIDITY_HAS_MLIR
	/// Helper to convert Solidity SourceLocation to MLIR Location
	mlir::Location loc(langutil::SourceLocation const& _location)
	{
		if (!_location.isValid())
			return m_builder->getUnknownLoc();

		// Get the source name - use the one from location if available,
		// otherwise try to get it from the CharStream
		std::string sourceName;
		if (_location.sourceName && !_location.sourceName->empty() && *_location.sourceName != "-")
		{
			sourceName = *_location.sourceName;
		}
		else if (_location.sourceName)
		{
			// Even if sourceName is "-" or empty, we can get the actual filename from CharStream
			auto const& charStream = m_compilerStack.charStream(*_location.sourceName);
			sourceName = charStream.name();
		}
		else
		{
			return m_builder->getUnknownLoc();
		}

		// Get the CharStream for this source to convert position to line/column
		auto const& charStream = m_compilerStack.charStream(_location.sourceName ? *_location.sourceName : sourceName);

		// Convert start position to line:column
		auto lineCol = charStream.translatePositionToLineColumn(_location.start);

		// MLIR uses 1-based line numbers and 0-based column numbers
		// Solidity CharStream uses 0-based for both
		return mlir::FileLineColLoc::
			get(m_builder->getStringAttr(sourceName),
				lineCol.line + 1, // Convert to 1-based
				lineCol.column	  // Keep 0-based for MLIR
			);
	}

	/// Helper to get location from an AST node
	template<typename T>
	mlir::Location loc(T const& _node)
	{
		return loc(_node.location());
	}
#endif

	std::string generateModule(ContractDefinition const& _contract)
	{
		m_mostDerivedContract = &_contract;
#ifdef SOLIDITY_HAS_MLIR
		auto loc = this->loc(_contract);

		// Set insertion point to module body
		auto moduleBody = m_module.getBody();
		m_builder->setInsertionPointToEnd(moduleBody);

		// Create contract operation using typed builder
		auto contractOp = m_builder->create<mlir::solidity::ContractOp>(
			loc, m_builder->getStringAttr(_contract.name()),
			m_builder->getI64IntegerAttr(_contract.id()));

		// Set insertion point to contract body
		m_builder->setInsertionPointToEnd(&contractOp.getBody().emplaceBlock());

		// Compute correct storage slots using the Solidity type system.
		// This uses ContractType::linearizedStateVariables() which already skips
		// constants and immutables, and computes proper storage layout with packing.
		m_stateVarSlots.clear();
		{
			ContractType contractType(_contract);
			for (auto const& [varDecl, slot, offset]: contractType.linearizedStateVariables(DataLocation::Storage))
				m_stateVarSlots[varDecl->name()] = slot;
		}

		// Generate state variables from all base contracts (in reverse order: base to derived)
		for (auto it = _contract.annotation().linearizedBaseContracts.rbegin();
			 it != _contract.annotation().linearizedBaseContracts.rend();
			 ++it)
		{
			ContractDefinition const* baseContract = *it;
			for (auto const& var: baseContract->stateVariables())
			{
				generateStateVariable(*var, _contract);
			}
		}

		// Generate auto-getter functions for public state variables.
		// In Solidity, `uint256 public x` auto-generates `function x() view returns (uint256)`.
		// `mapping(K => V) public m` auto-generates `function m(K) view returns (V)`.
		std::set<std::string> generatedFunctions;
		for (auto it = _contract.annotation().linearizedBaseContracts.rbegin();
			 it != _contract.annotation().linearizedBaseContracts.rend();
			 ++it)
		{
			ContractDefinition const* baseContract = *it;
			for (auto const& var: baseContract->stateVariables())
			{
				if (var->visibility() >= Visibility::Public)
				{
					if (auto* mappingType = dynamic_cast<MappingType const*>(var->type()))
					{
						generateMappingGetter(*var, mappingType);
						// Build signature matching the function parameter format:
						// e.g. mapping(address => mapping(address => uint256)) -> "allowance(address,address)"
						std::string sig = var->name() + "(";
						auto const* mt = mappingType;
						bool first = true;
						while (mt)
						{
							if (!first) sig += ",";
							sig += mt->keyType()->toString();
							first = false;
							mt = dynamic_cast<MappingType const*>(mt->valueType());
						}
						sig += ")";
						generatedFunctions.insert(sig);
					}
					else
					{
						generateSimpleGetter(*var);
						generatedFunctions.insert(var->name() + "()");
					}
				}
			}
		}

		// Generate functions from all contracts in C3 linearization order (derived to base).
		// Derived contract overrides take precedence over base implementations.
		for (auto it = _contract.annotation().linearizedBaseContracts.begin();
			 it != _contract.annotation().linearizedBaseContracts.end();
			 ++it)
		{
			ContractDefinition const* baseContract = *it;
			for (auto const& func: baseContract->definedFunctions())
			{
				if (!func->isConstructor())
				{
					// Create a signature to track what we've generated
					std::string signature = func->name() + "(";
					for (size_t i = 0; i < func->parameters().size(); ++i)
					{
						if (i > 0)
							signature += ",";
						signature += func->parameters()[i]->type()->toString();
					}
					signature += ")";

					// Only generate if we haven't seen this signature yet
					// (derived contract functions take precedence)
					if (generatedFunctions.find(signature) == generatedFunctions.end())
					{
						generatedFunctions.insert(signature);
						generateSolidityFunction(*func);
					}
					else if (func->isImplemented())
						// An overridden base implementation is still reachable
						// by an explicit `Base.f(...)`, so it has to exist -
						// under the name such a call resolves to, qualified by
						// its defining contract so it cannot collide with the
						// override that shadows it.
						generateSolidityFunction(*func, baseContract->name() + "." + func->name());
				}
			}
		}

		// Generate constructor(s) from the linearized base contracts.
		// Walk from most-derived to base; only generate the first constructor found.
		// If the constructor has parameters and the derived contract provides base
		// constructor arguments, generate a merged 0-param constructor.
		{
			bool generatedConstructor = false;
			for (auto it = _contract.annotation().linearizedBaseContracts.begin();
				 it != _contract.annotation().linearizedBaseContracts.end() && !generatedConstructor;
				 ++it)
			{
				ContractDefinition const* baseContract = *it;
				for (auto const& func: baseContract->definedFunctions())
				{
					if (func->isConstructor() && func->isImplemented())
					{
						if (!func->parameters().empty() && baseContract != &_contract)
						{
							// Base constructor with parameters — generate merged 0-param version
							// with base constructor arguments inlined
							generateMergedConstructor(*func, _contract);
						}
						else
						{
							// Own constructor (with or without params) or parameterless — generate as-is
							// The lowering will handle calldata decoding for parameterized own constructors
							generateSolidityFunction(*func);
						}
						generatedConstructor = true;
						break;
					}
				}
			}
		}

		// Generate library functions from all reachable libraries.
		// Libraries can be used via "using for" directives or called directly (Library.func()).
		// Scan all imported source units for library contracts.
		{
			std::set<ContractDefinition const*> visitedLibraries;
			std::set<SourceUnit const*> allUnits = _contract.sourceUnit().referencedSourceUnits(true);
			allUnits.insert(&_contract.sourceUnit());
			for (auto const* unit: allUnits)
			{
				for (auto const& node: unit->nodes())
				{
					auto const* libDecl = dynamic_cast<ContractDefinition const*>(node.get());
					if (libDecl && libDecl->isLibrary() && visitedLibraries.insert(libDecl).second)
					{
						for (auto const& func: libDecl->definedFunctions())
						{
							if (!func->isConstructor() && func->isImplemented())
							{
								std::string signature = func->name() + "(";
								for (size_t i = 0; i < func->parameters().size(); ++i)
								{
									if (i > 0)
										signature += ",";
									signature += func->parameters()[i]->type()->toString();
								}
								signature += ")";

								if (generatedFunctions.find(signature) == generatedFunctions.end())
								{
									generatedFunctions.insert(signature);
									generateSolidityFunction(*func);
								}
							}
						}
					}
				}
			}
		}

		// Generate free functions from all reachable source units (including imports).
		// Free functions are file-level functions not inside any contract.
		{
			std::set<SourceUnit const*> allUnits = _contract.sourceUnit().referencedSourceUnits(true);
			allUnits.insert(&_contract.sourceUnit());
			for (auto const* unit: allUnits)
			{
				for (auto const& node: unit->nodes())
				{
					if (auto const* funcDef = dynamic_cast<FunctionDefinition const*>(node.get()))
					{
						if (funcDef->isFree() && funcDef->isImplemented())
						{
							std::string signature = funcDef->name() + "(";
							for (size_t i = 0; i < funcDef->parameters().size(); ++i)
							{
								if (i > 0)
									signature += ",";
								signature += funcDef->parameters()[i]->type()->toString();
							}
							signature += ")";

							if (generatedFunctions.find(signature) == generatedFunctions.end())
							{
								generatedFunctions.insert(signature);
								generateSolidityFunction(*funcDef);
							}
						}
					}
				}
			}
		}

		if (m_unsupportedCount > 0)
			std::cerr << "Warning: [MLIR] " << _contract.name()
					  << ": " << m_unsupportedCount << " unsupported feature(s) dropped\n";

		// Generate child contracts referenced by `new ContractName()`
		// These become sub-objects in the Yul output
		for (auto const* childContract: m_childContracts)
		{
			auto savedIP = m_builder->saveInsertionPoint();
			m_builder->setInsertionPointToEnd(m_module.getBody());

			auto childLoc = this->loc(*childContract);
			auto childOp = m_builder->create<mlir::solidity::ContractOp>(
				childLoc, m_builder->getStringAttr(childContract->name()),
				m_builder->getI64IntegerAttr(childContract->id()));
			childOp->setAttr("is_subobject", m_builder->getUnitAttr());

			m_builder->setInsertionPointToEnd(&childOp.getBody().emplaceBlock());

			// Generate child contract's state variables
			m_stateVarSlots.clear();
			{
				ContractType contractType(*childContract);
				for (auto const& [varDecl, slot, offset]: contractType.linearizedStateVariables(DataLocation::Storage))
					m_stateVarSlots[varDecl->name()] = slot;
			}
			for (auto cit = childContract->annotation().linearizedBaseContracts.rbegin();
				 cit != childContract->annotation().linearizedBaseContracts.rend(); ++cit)
				for (auto const& var: (*cit)->stateVariables())
					generateStateVariable(*var, *childContract);

			// Generate child contract's functions (including getters)
			std::set<std::string> childGeneratedFunctions;
			for (auto cit = childContract->annotation().linearizedBaseContracts.rbegin();
				 cit != childContract->annotation().linearizedBaseContracts.rend(); ++cit)
				for (auto const& var: (*cit)->stateVariables())
					if (var->visibility() >= Visibility::Public && !var->isConstant())
					{
						if (auto* mt = dynamic_cast<MappingType const*>(var->type()))
						{
							generateMappingGetter(*var, mt);
							std::string sig = var->name() + "(";
							auto const* mmt = mt; bool first = true;
							while (mmt) { if (!first) sig += ","; sig += mmt->keyType()->toString(); first = false;
								mmt = dynamic_cast<MappingType const*>(mmt->valueType()); }
							sig += ")"; childGeneratedFunctions.insert(sig);
						}
						else
						{
							generateSimpleGetter(*var);
							childGeneratedFunctions.insert(var->name() + "()");
						}
					}
			for (auto cit = childContract->annotation().linearizedBaseContracts.begin();
				 cit != childContract->annotation().linearizedBaseContracts.end(); ++cit)
				for (auto const& func: (*cit)->definedFunctions())
					if (!func->isConstructor())
					{
						std::string sig = func->name() + "(";
						for (size_t i = 0; i < func->parameters().size(); ++i)
						{ if (i > 0) sig += ","; sig += func->parameters()[i]->type()->toString(); }
						sig += ")";
						if (childGeneratedFunctions.find(sig) == childGeneratedFunctions.end())
						{ childGeneratedFunctions.insert(sig); generateSolidityFunction(*func); }
					}

			// Generate child contract's constructor
			bool childHasConstructor = false;
			for (auto cit = childContract->annotation().linearizedBaseContracts.begin();
				 cit != childContract->annotation().linearizedBaseContracts.end() && !childHasConstructor; ++cit)
				for (auto const& func: (*cit)->definedFunctions())
					if (func->isConstructor() && func->isImplemented())
					{
						if (!func->parameters().empty() && *cit != childContract)
							generateMergedConstructor(*func, *childContract);
						else
							generateSolidityFunction(*func);
						childHasConstructor = true;
						break;
					}

			m_builder->restoreInsertionPoint(savedIP);
		}

		// Convert module to string with proper MLIR formatting
		std::string output;
		llvm::raw_string_ostream stream(output);
		m_module.print(stream);
		stream.flush();

		return output;
#else
		// MLIR support is not available
		solAssert(false, "MLIR support is not enabled. Cannot generate MLIR module.");
		return "";
#endif
	}

#ifdef SOLIDITY_HAS_MLIR
	void generateStateVariable(VariableDeclaration const& _var, ContractDefinition const& _contract)
	{
		auto loc = this->loc(_var);

		// Determine visibility
		std::string visibility = "private";
		if (_var.visibility() == Visibility::Public)
			visibility = "public";
		else if (_var.visibility() == Visibility::Internal)
			visibility = "internal";

		// Create Solidity type
		auto solidityType = translateSolidityType(*_var.type());

		// The initialiser of an ordinary state variable matters as much as a
		// constant's: the creation code has to store it, or the variable reads
		// zero however it was declared.
		mlir::Attribute initialValue;
		if (_var.value())
		{
			// Try to extract the literal value from the constant expression
			if (auto* literal = dynamic_cast<Literal const*>(_var.value().get()))
			{
				if (literal->token() == langutil::Token::Number)
				{
					// Use the literal's annotation type (RationalNumber) which has literalValue(),
					// not the variable's type (uint256) which doesn't.
					if (literal->annotation().type)
					{
						u256 bigValue = literal->annotation().type->literalValue(literal);
						initialValue = m_builder->getStringAttr(bigValue.str());
					}
					else
					{
						// Fallback: parse the value string directly
						try { initialValue = m_builder->getStringAttr(u256(literal->value()).str()); }
						catch (...) {}
					}
				}
				else if (literal->token() == langutil::Token::TrueLiteral)
					initialValue = m_builder->getStringAttr("1");
				else if (literal->token() == langutil::Token::FalseLiteral)
					initialValue = m_builder->getStringAttr("0");
				else if (literal->token() == langutil::Token::StringLiteral
					  || literal->token() == langutil::Token::HexStringLiteral)
					initialValue = m_builder->getStringAttr(literal->value());
			}
		}

		// Look up the correct storage slot from the precomputed map
		mlir::IntegerAttr slotAttr;
		if (!_var.isConstant() && !_var.immutable())
		{
			auto slotIt = m_stateVarSlots.find(_var.name());
			if (slotIt != m_stateVarSlots.end())
				slotAttr = m_builder->getI64IntegerAttr(static_cast<int64_t>(slotIt->second));
		}

		// Create state variable operation using typed builder
		auto stateVarOp = m_builder->create<mlir::solidity::StateVarOp>(
			loc,
			_var.name(),
			solidityType,
			m_builder->getStringAttr(visibility),
			_var.isConstant(),
			_var.immutable(),
			initialValue,
			slotAttr
		);

		// Store reference for later use - store as Operation*
		m_stateVarOpMap[_var.id()] = stateVarOp;
	}

	/// Generate a getter function for a public mapping variable.
	/// For `mapping(K => V) public m`, generates `function m(K) view returns (V)`.
	/// Handles nested mappings: `mapping(K1 => mapping(K2 => V)) public m` → `m(K1, K2) → V`.
	void generateMappingGetter(VariableDeclaration const& _var, MappingType const* _mappingType)
	{
		auto loc = this->loc(_var);

		// Collect key types and find the innermost value type
		std::vector<mlir::Type> keyTypes;
		Type const* valueType = nullptr;
		MappingType const* current = _mappingType;
		while (current)
		{
			keyTypes.push_back(translateSolidityType(*current->keyType()));
			valueType = current->valueType();
			current = dynamic_cast<MappingType const*>(valueType);
		}

		mlir::Type returnType = translateSolidityType(*valueType);
		auto funcType = m_builder->getFunctionType(keyTypes, {returnType});

		// Create function op (same signature as generateSolidityFunction)
		auto funcOp = m_builder->create<mlir::solidity::FunctionOp>(
			loc, _var.name(), funcType, "public", "view");

		// Create function body
		auto& entryBlock = funcOp.getBody().emplaceBlock();
		std::vector<mlir::Value> keys;
		for (auto keyType: keyTypes)
			keys.push_back(entryBlock.addArgument(keyType, loc));

		auto savedInsertionPoint = m_builder->saveInsertionPoint();
		m_builder->setInsertionPointToEnd(&entryBlock);

		// Create mapping access with all keys
		auto accessResult = m_builder->create<mlir::solidity::MappingAccessOp>(
			loc, returnType, m_builder->getStringAttr(_var.name()), keys);

		// Return the result
		m_builder->create<mlir::solidity::ReturnOp>(loc, mlir::ValueRange{accessResult.getResult()});

		m_builder->restoreInsertionPoint(savedInsertionPoint);
	}

	/// Generate a getter function for a simple public state variable.
	/// For `uint256 public x`, generates `function x() view returns (uint256) { return x; }`.
	void generateSimpleGetter(VariableDeclaration const& _var)
	{
		auto loc = this->loc(_var);
		mlir::Type returnType = translateSolidityType(*_var.type());
		auto funcType = m_builder->getFunctionType({}, {returnType});

		auto funcOp = m_builder->create<mlir::solidity::FunctionOp>(
			loc, _var.name(), funcType, "public", "view");

		auto& entryBlock = funcOp.getBody().emplaceBlock();
		auto savedInsertionPoint = m_builder->saveInsertionPoint();
		m_builder->setInsertionPointToEnd(&entryBlock);

		auto loadResult = m_builder->create<mlir::solidity::LoadStateVarOp>(
			loc, returnType, _var.name());

		m_builder->create<mlir::solidity::ReturnOp>(loc, mlir::ValueRange{loadResult.getResult()});

		m_builder->restoreInsertionPoint(savedInsertionPoint);
	}

	/// Generate a merged 0-parameter constructor for a derived contract.
	/// Base constructor arguments (from InheritanceSpecifier) are evaluated
	/// at the top of the body and mapped to the constructor's parameters.
	void generateMergedConstructor(
		FunctionDefinition const& _baseConstructor,
		ContractDefinition const& _derivedContract)
	{
		auto loc = this->loc(_baseConstructor);

		// Always 0 params, 0 returns for the merged constructor
		auto funcType = m_builder->getFunctionType({}, {});

		auto funcOp = m_builder->create<mlir::solidity::FunctionOp>(
			loc, "", funcType, "public", "nonpayable");
		funcOp->setAttr("kind", m_builder->getStringAttr("constructor"));

		auto& entryBlock = funcOp.getBody().emplaceBlock();

		// Enter new scope for function
		SymbolTableScopeT functionScope(m_symbolTable);

		auto savedIP = m_builder->saveInsertionPoint();
		m_builder->setInsertionPointToEnd(&entryBlock);

		// Evaluate base constructor arguments and map them to the parameter variables.
		// These come from InheritanceSpecifier or ModifierInvocation in the derived contract.
		auto constructorArgsIt = _derivedContract.annotation().baseConstructorArguments.find(&_baseConstructor);
		if (constructorArgsIt != _derivedContract.annotation().baseConstructorArguments.end())
		{
			std::vector<ASTPointer<Expression>> const* args = nullptr;

			if (auto* inhSpec = dynamic_cast<InheritanceSpecifier const*>(constructorArgsIt->second))
				args = inhSpec->arguments();
			else if (auto* modInv = dynamic_cast<ModifierInvocation const*>(constructorArgsIt->second))
				args = modInv->arguments();

			if (args)
			{
				auto params = _baseConstructor.parameters();
				for (size_t i = 0; i < params.size() && i < args->size(); ++i)
				{
					mlir::Value argValue = generateSolidityExpression(*(*args)[i]);
					m_symbolTable.insert(params[i]->name(), argValue);
					m_valueMap[params[i]->id()] = argValue;
				}
			}
		}
		else
		{
			// No base constructor arguments found — zero-initialize params
			auto params = _baseConstructor.parameters();
			for (size_t i = 0; i < params.size(); ++i)
			{
				auto paramType = translateSolidityType(*params[i]->type());
				auto zeroValue = m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getIntegerAttr(
						mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned), 0),
					paramType);
				m_symbolTable.insert(params[i]->name(), zeroValue);
				m_valueMap[params[i]->id()] = zeroValue;
			}
		}

		// Generate the base constructor body
		if (_baseConstructor.isImplemented())
			generateSolidityStatement(_baseConstructor.body());

		// Add return if not already present
		bool hasReturn = false;
		if (!entryBlock.empty())
		{
			auto& lastOp = entryBlock.back();
			if (mlir::isa<mlir::solidity::ReturnOp>(lastOp) || mlir::isa<mlir::solidity::RevertOp>(lastOp))
				hasReturn = true;
		}
		if (!hasReturn)
			emitReturn(loc, {});

		m_builder->restoreInsertionPoint(savedIP);
	}

	void generateSolidityFunction(FunctionDefinition const& _func, std::string const& _nameOverride = {})
	{
		auto loc = this->loc(_func);

		// Build function type using Solidity types
		std::vector<mlir::Type> paramTypes;
		for (auto const& param: _func.parameters())
		{
			paramTypes.push_back(translateSolidityType(*param->type()));
		}

		std::vector<mlir::Type> returnTypes;
		for (auto const& ret: _func.returnParameters())
		{
			returnTypes.push_back(translateSolidityType(*ret->type()));
		}

		auto funcType = m_builder->getFunctionType(paramTypes, returnTypes);

		// Determine visibility and mutability
		// Note: constructors cannot call defaultVisibility() — handle them specially
		std::string visibility = "public";
		if (!_func.isConstructor())
		{
			if (_func.visibility() == Visibility::Private)
				visibility = "private";
			else if (_func.visibility() == Visibility::Internal)
				visibility = "internal";
			else if (_func.visibility() == Visibility::External)
				visibility = "external";
		}

		std::string mutability = "nonpayable";
		if (_func.stateMutability() == StateMutability::Pure)
			mutability = "pure";
		else if (_func.stateMutability() == StateMutability::View)
			mutability = "view";
		else if (_func.stateMutability() == StateMutability::Payable)
			mutability = "payable";

		// Create Solidity function operation using typed builder
		auto funcOp = m_builder->create<mlir::solidity::FunctionOp>(
			loc, _nameOverride.empty() ? _func.name() : _nameOverride, funcType, visibility, mutability);

		// Add kind attribute to distinguish constructors, receive, fallback
		if (_func.isConstructor())
			funcOp->setAttr("kind", m_builder->getStringAttr("constructor"));
		else if (_func.isReceive())
			funcOp->setAttr("kind", m_builder->getStringAttr("receive"));
		else if (_func.isFallback())
			funcOp->setAttr("kind", m_builder->getStringAttr("fallback"));

		// Store original parameter and return variable names for inline assembly bridging
		{
			std::vector<mlir::Attribute> paramNameAttrs;
			for (auto const& param: _func.parameters())
				paramNameAttrs.push_back(m_builder->getStringAttr(param->name()));
			if (!paramNameAttrs.empty())
				funcOp->setAttr("param_names", m_builder->getArrayAttr(paramNameAttrs));

			std::vector<mlir::Attribute> returnNameAttrs;
			for (auto const& ret: _func.returnParameters())
				returnNameAttrs.push_back(m_builder->getStringAttr(ret->name()));
			if (!returnNameAttrs.empty())
				funcOp->setAttr("return_names", m_builder->getArrayAttr(returnNameAttrs));
		}

		// Store the ABI-canonical external signature for correct selector computation.
		// This is needed because contract types become addresses, struct types become
		// tuples, etc. in the ABI encoding — information lost when MLIR erases to uint256.
		if (!_func.isConstructor() && !_func.isReceive() && !_func.isFallback())
		{
			if (auto extFuncType = _func.functionType(false))
			{
				std::string extSig = extFuncType->externalSignature();
				funcOp->setAttr("abi_signature", m_builder->getStringAttr(extSig));
			}
		}

		// Create entry block with arguments
		auto& entryBlock = funcOp.getBody().emplaceBlock();

		// Add block arguments for parameters
		for (size_t i = 0; i < paramTypes.size(); ++i)
		{
			entryBlock.addArgument(paramTypes[i], loc);
		}

		// Enter new scope for function
		SymbolTableScopeT functionScope(m_symbolTable);

		// Map function parameters to block arguments
		auto params = _func.parameters();
		for (size_t i = 0; i < params.size(); ++i)
		{
			// Insert parameter into symbol table with proper scoping
			m_symbolTable.insert(params[i]->name(), entryBlock.getArgument(i));
			m_valueMap[params[i]->id()] = entryBlock.getArgument(i);
		}

		// Save current insertion point and switch to function body
		auto savedIP = m_builder->saveInsertionPoint();
		m_builder->setInsertionPointToEnd(&entryBlock);

		// Generate function body
		if (_func.isImplemented())
		{
			generateSolidityStatement(_func.body());
		}

		// Add return if not already present
		// Check if the last operation is a solidity.return
		bool hasReturn = false;
		if (!entryBlock.empty())
		{
			auto& lastOp = entryBlock.back();
			if (mlir::isa<mlir::solidity::ReturnOp>(lastOp) || mlir::isa<mlir::solidity::RevertOp>(lastOp))
				hasReturn = true;
		}

		if (!hasReturn)
		{
			emitReturn(loc, {});
		}

		// Restore insertion point
		m_builder->restoreInsertionPoint(savedIP);
	}

	// Variables a statement declares. A nested loop's own counter is modified
	// inside it but does not exist outside it, so it must not be carried by the
	// enclosing loop.
	std::set<int64_t> collectDeclaredVariables(Statement const& _stmt)
	{
		std::set<int64_t> declared;
		if (auto* varStmt = dynamic_cast<VariableDeclarationStatement const*>(&_stmt))
		{
			for (auto const& var: varStmt->declarations())
				if (var)
					declared.insert(var->id());
		}
		else if (auto* block = dynamic_cast<Block const*>(&_stmt))
		{
			for (auto const& stmt: block->statements())
				if (stmt)
				{
					auto sub = collectDeclaredVariables(*stmt);
					declared.insert(sub.begin(), sub.end());
				}
		}
		else if (auto* ifStmt = dynamic_cast<IfStatement const*>(&_stmt))
		{
			auto sub = collectDeclaredVariables(ifStmt->trueStatement());
			declared.insert(sub.begin(), sub.end());
			if (ifStmt->falseStatement())
			{
				auto other = collectDeclaredVariables(*ifStmt->falseStatement());
				declared.insert(other.begin(), other.end());
			}
		}
		else if (auto* forStmt = dynamic_cast<ForStatement const*>(&_stmt))
		{
			if (forStmt->initializationExpression())
			{
				auto init = collectDeclaredVariables(*forStmt->initializationExpression());
				declared.insert(init.begin(), init.end());
			}
			auto body = collectDeclaredVariables(forStmt->body());
			declared.insert(body.begin(), body.end());
		}
		else if (auto* whileStmt = dynamic_cast<WhileStatement const*>(&_stmt))
		{
			auto body = collectDeclaredVariables(whileStmt->body());
			declared.insert(body.begin(), body.end());
		}
		return declared;
	}

	// Helper function to collect all variables modified in a statement
	std::set<int64_t> collectModifiedVariables(Statement const& _stmt)
	{
		std::set<int64_t> modifiedVars;

		// Recursively analyze the statement
		if (auto* exprStmt = dynamic_cast<ExpressionStatement const*>(&_stmt))
		{
			if (auto* assignment = dynamic_cast<Assignment const*>(&exprStmt->expression()))
			{
				if (auto* ident = dynamic_cast<Identifier const*>(&assignment->leftHandSide()))
				{
					if (auto* varDecl
						= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (!varDecl->isStateVariable())
							modifiedVars.insert(varDecl->id());
					}
				}
			}
			else if (auto* unary = dynamic_cast<UnaryOperation const*>(&exprStmt->expression()))
			{
				if (unary->getOperator() == Token::Inc || unary->getOperator() == Token::Dec)
				{
					if (auto* ident = dynamic_cast<Identifier const*>(&unary->subExpression()))
					{
						if (auto* varDecl
							= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
						{
							if (!varDecl->isStateVariable())
								modifiedVars.insert(varDecl->id());
						}
					}
				}
			}
		}
		else if (auto* block = dynamic_cast<Block const*>(&_stmt))
		{
			for (auto const& stmt: block->statements())
			{
				auto subVars = collectModifiedVariables(*stmt);
				modifiedVars.insert(subVars.begin(), subVars.end());
			}
		}
		else if (auto* ifStmt = dynamic_cast<IfStatement const*>(&_stmt))
		{
			auto trueVars = collectModifiedVariables(ifStmt->trueStatement());
			modifiedVars.insert(trueVars.begin(), trueVars.end());
			if (ifStmt->falseStatement())
			{
				auto falseVars = collectModifiedVariables(*ifStmt->falseStatement());
				modifiedVars.insert(falseVars.begin(), falseVars.end());
			}
		}
		else if (dynamic_cast<ForStatement const*>(&_stmt) || dynamic_cast<WhileStatement const*>(&_stmt))
		{
			// A variable a nested loop assigns is modified by this loop as
			// well, so both have to carry it. Not recursing here left the outer
			// loop yielding fewer values than it needed, and the code after it
			// naming the inner loop's result - which is out of scope there, so
			// the whole contract failed to verify.
			std::set<int64_t> nested;
			if (auto* forStmt = dynamic_cast<ForStatement const*>(&_stmt))
			{
				if (forStmt->initializationExpression())
				{
					auto init = collectModifiedVariables(*forStmt->initializationExpression());
					nested.insert(init.begin(), init.end());
				}
				if (forStmt->loopExpression())
				{
					auto step = collectModifiedVariables(*forStmt->loopExpression());
					nested.insert(step.begin(), step.end());
				}
				auto body = collectModifiedVariables(forStmt->body());
				nested.insert(body.begin(), body.end());
			}
			else
			{
				auto* whileStmt = dynamic_cast<WhileStatement const*>(&_stmt);
				auto body = collectModifiedVariables(whileStmt->body());
				nested.insert(body.begin(), body.end());
			}

			for (int64_t id: collectDeclaredVariables(_stmt))
				nested.erase(id);
			modifiedVars.insert(nested.begin(), nested.end());
		}

		return modifiedVars;
	}

	// Helper to collect variables referenced in an expression (for init/condition/update)
	std::set<int64_t> collectReferencedVariables(Expression const& _expr)
	{
		std::set<int64_t> referencedVars;

		if (auto* ident = dynamic_cast<Identifier const*>(&_expr))
		{
			if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
			{
				if (!varDecl->isStateVariable())
					referencedVars.insert(varDecl->id());
			}
		}
		else if (auto* binary = dynamic_cast<BinaryOperation const*>(&_expr))
		{
			auto leftVars = collectReferencedVariables(binary->leftExpression());
			auto rightVars = collectReferencedVariables(binary->rightExpression());
			referencedVars.insert(leftVars.begin(), leftVars.end());
			referencedVars.insert(rightVars.begin(), rightVars.end());
		}
		else if (auto* unary = dynamic_cast<UnaryOperation const*>(&_expr))
		{
			auto subVars = collectReferencedVariables(unary->subExpression());
			referencedVars.insert(subVars.begin(), subVars.end());
		}
		// Add more cases as needed

		return referencedVars;
	}

	std::string extractMappingVarName(Expression const& expr)
	{
		if (auto* ident = dynamic_cast<Identifier const*>(&expr))
			if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(
					ident->annotation().referencedDeclaration))
				return varDecl->name();
		if (auto* innerAccess = dynamic_cast<IndexAccess const*>(&expr))
			return extractMappingVarName(innerAccess->baseExpression());
		return "";
	}

	/// Create a MappingStoreOp with variadic keys.
	/// The numKeys attribute records how many of the operands are keys vs the value.
	void createMappingStoreOp(mlir::Location loc, std::string const& varName,
		std::vector<mlir::Value> const& keys, mlir::Value value)
	{
		// Combine keys and value into a single operand list
		std::vector<mlir::Value> operands;
		operands.insert(operands.end(), keys.begin(), keys.end());
		operands.push_back(value);

		m_builder->create<mlir::solidity::MappingStoreOp>(
			loc,
			m_builder->getStringAttr(varName),
			m_builder->getI32IntegerAttr(static_cast<int32_t>(keys.size())),
			operands);
	}

	/// Collect all keys from a nested IndexAccess chain on a mapping.
	/// For allowance[owner][spender], returns {owner_value, spender_value}.
	/// The varName output parameter receives the base mapping variable name.
	std::vector<mlir::Value> collectMappingKeys(IndexAccess const& indexAccess, std::string& varName)
	{
		std::vector<mlir::Value> keys;
		Expression const* current = &indexAccess;

		while (auto* idx = dynamic_cast<IndexAccess const*>(current))
		{
			if (dynamic_cast<MappingType const*>(idx->baseExpression().annotation().type))
			{
				keys.push_back(generateSolidityExpression(*idx->indexExpression()));
				current = &idx->baseExpression();
			}
			else
				break;
		}

		varName = extractMappingVarName(*current);

		// Keys are collected outermost-first. Reverse to get innermost-first
		// (base mapping key first, then nested keys).
		std::reverse(keys.begin(), keys.end());

		return keys;
	}

	/// The callee of an internal call, qualified by the contract that defines
	/// it. Solidity has already resolved which declaration a call reaches -
	/// `Base.value(1)` and a virtual `value(1)` inside the override name
	/// different ones - and dropping that resolution here is what made both
	/// lower to the same symbol and bind to whichever won.
	/// The FunctionDefinition a call reaches, after virtual resolution.
	/// Strips parentheses. `((super).f)()` is the same call as `super.f()`, but
	/// each level arrives as a one-component tuple, so anything matching on the
	/// node type sees a TupleExpression and gives up.
	static Expression const& withoutParentheses(Expression const& _expr)
	{
		Expression const* expr = &_expr;
		while (auto const* tuple = dynamic_cast<TupleExpression const*>(expr))
		{
			if (tuple->isInlineArray() || tuple->components().size() != 1 || !tuple->components().front())
				break;
			expr = tuple->components().front().get();
		}
		return *expr;
	}

	/// The contract a `super` member access searches *after*, or null when the
	/// base is not `super`. `super` carries it as the contract of its own type,
	/// which is the one the expression is written in - so a body inherited into
	/// a derived contract searches from where it is defined, not from where it
	/// ended up.
	static ContractDefinition const* superSearchStart(MemberAccess const& _member)
	{
		auto const* typeType
			= dynamic_cast<TypeType const*>(withoutParentheses(_member.expression()).annotation().type);
		if (!typeType)
			return nullptr;
		auto const* contractType = dynamic_cast<ContractType const*>(typeType->actualType());
		return contractType && contractType->isSuper() ? &contractType->contractDefinition() : nullptr;
	}

	/// The next implementation of `_function` after `_from` in the most-derived
	/// contract's linearisation - what `super.f()` means.
	///
	/// `FunctionDefinition::resolveVirtual` takes a search start, but searches
	/// from it inclusively, so handing it the contract the call is written in
	/// returns that same contract's override and the function calls itself.
	/// The step `super` asks for is strictly the next one along.
	FunctionDefinition const* resolveSuper(FunctionDefinition const& _function, ContractDefinition const& _from)
	{
		if (!m_mostDerivedContract || _function.name().empty())
			return nullptr;

		auto const& linearised = m_mostDerivedContract->annotation().linearizedBaseContracts;
		auto position = std::find(linearised.begin(), linearised.end(), &_from);
		if (position == linearised.end())
			return nullptr;

		FunctionType const* wanted = TypeProvider::function(_function)->asExternallyCallableFunction(false);
		for (++position; position != linearised.end(); ++position)
			for (FunctionDefinition const* candidate: (*position)->definedFunctions(_function.name()))
				if (candidate->isImplemented()
					&& FunctionType(*candidate).asExternallyCallableFunction(false)->hasEqualParameterTypes(*wanted))
					return candidate;
		return nullptr;
	}

	FunctionDefinition const* resolvedCallee(FunctionCall const& _call)
	{
		Expression const& callee = withoutParentheses(_call.expression());
		auto const* member = dynamic_cast<MemberAccess const*>(&callee);

		Declaration const* declaration = nullptr;
		if (auto const* ident = dynamic_cast<Identifier const*>(&callee))
			declaration = ident->annotation().referencedDeclaration;
		else if (member)
			declaration = member->annotation().referencedDeclaration;

		auto const* function = dynamic_cast<FunctionDefinition const*>(declaration);
		if (!function || !dynamic_cast<ContractDefinition const*>(function->scope()))
			return nullptr;
		if (!m_mostDerivedContract || function->isConstructor())
			return function;

		// An unqualified call is virtual: `hook()` inside a body Concrete
		// inherits must reach Concrete's override, not the declaration the name
		// resolved to in the base. An explicit `Base.f(...)` arrives as a
		// MemberAccess and stays direct.
		if (!member)
			return &function->resolveVirtual(*m_mostDerivedContract);

		// `super.f()` is neither: it resolves against the most-derived
		// contract's linearisation, starting after the contract the call is
		// written in. Treating it as a direct call reaches the override that is
		// asking, which is a loop; treating it as virtual does the same.
		if (ContractDefinition const* from = superSearchStart(*member))
		{
			if (FunctionDefinition const* next = resolveSuper(*function, *from))
				return next;
		}
		return function;
	}

	/// Makes an argument list agree with the callee's arity. An argument whose
	/// expression was dropped leaves the list short, and the call the verifier
	/// then rejects costs the whole contract rather than the one argument.
	/// A dropped sub-expression comes back as a uint256 placeholder, so an op
	/// needing a boolean or an array is handed the wrong type and the verifier
	/// rejects it - costing the whole contract for the one expression. Re-place
	/// the operand as a placeholder of the type the op declares.
	mlir::Value typedOperand(mlir::Location _loc, mlir::Value _value, mlir::Type _expected, std::string const& _what)
	{
		if (!_value || _value.getType() == _expected)
			return _value;
		return emitUnsupported(_loc, _expected, _what);
	}

	/// A value defined inside a region does not exist after it. When a loop or
	/// an `if` assigned a variable and the region did not yield it, the map
	/// still named the inner value, and every later use of that variable was an
	/// undeclared SSA name - so the contract was lost for one assignment. Park
	/// those as placeholders instead.
	void dropValuesEscaping(mlir::Operation* _op, mlir::Location _loc)
	{
		if (!_op)
			return;
		for (auto& [id, value]: m_valueMap)
		{
			if (!value)
				continue;
			mlir::Operation* owner = value.getDefiningOp();
			if (!owner)
				if (mlir::Block* block = value.getParentBlock())
					owner = block->getParentOp();
			if (owner && owner != _op && _op->isProperAncestor(owner))
				value = emitUnsupported(_loc, value.getType(), "value assigned only inside a region that does not yield it");
		}
	}

	/// One yielded value per loop-carried variable, in order.
	///
	/// A variable the body never assigned has no entry in the map, and dropping
	/// it left the yield shorter than the arity the region declares; yielding
	/// argument 0 in its place named a different variable. The value that
	/// entered the iteration is this position's own block argument, which is
	/// what "unchanged" means here.
	llvm::SmallVector<mlir::Value, 4> loopYieldValues(mlir::Block* _block, std::vector<int64_t> const& _varIds)
	{
		llvm::SmallVector<mlir::Value, 4> values;
		if (!_block)
			return values;
		// The region declares the arity; the variable list only says what each
		// position means. An empty body has nothing in the map and still has to
		// yield one value per argument.
		for (unsigned i = 0; i < _block->getNumArguments(); ++i)
		{
			mlir::Value value;
			if (i < _varIds.size())
			{
				auto it = m_valueMap.find(_varIds[i]);
				if (it != m_valueMap.end() && it->second)
					value = it->second;
			}
			values.push_back(value ? value : _block->getArgument(i));
		}
		return values;
	}

	/// The block a break or continue would yield from, but only when it is the
	/// loop's own body block. Inside a nested `scf.if` a yield belongs to the
	/// `if`, not the loop, so emitting the loop's values there builds a yield
	/// under the wrong parent - and the contract is lost for one `break`.
	mlir::Block* immediateLoopBody()
	{
		mlir::Block* block = m_builder->getInsertionBlock();
		return block && mlir::isa_and_nonnull<mlir::scf::WhileOp>(block->getParentOp()) ? block : nullptr;
	}

	/// The result types a call site must declare. A function returning a tuple
	/// declares one result per component, and collapsing them to the single
	/// type the annotation translates to left the call one result short of the
	/// callee - which the verifier rejects, costing the contract.
	llvm::SmallVector<mlir::Type, 2> callResultTypes(Type const& _annotated)
	{
		llvm::SmallVector<mlir::Type, 2> types;
		if (auto const* tuple = dynamic_cast<TupleType const*>(&_annotated))
		{
			for (auto const& component: tuple->components())
				if (component)
					types.push_back(translateSolidityType(*component));
			return types;
		}
		types.push_back(translateSolidityType(_annotated));
		return types;
	}

	void padCallArguments(FunctionCall const& _call, std::vector<mlir::Value>& _args, mlir::Location _loc)
	{
		FunctionDefinition const* callee = resolvedCallee(_call);
		if (!callee)
			return;
		auto const& parameters = callee->parameters();
		for (size_t i = _args.size(); i < parameters.size(); ++i)
			_args.push_back(emitUnsupported(_loc, translateSolidityType(*parameters[i]->type()), "call argument not generated"));
		if (_args.size() > parameters.size())
			_args.resize(parameters.size());
	}

	std::string resolvedCalleeName(FunctionCall const& _call, std::string const& _fallback)
	{
		FunctionDefinition const* function = resolvedCallee(_call);
		if (!function)
			return _fallback;
		auto const* contract = dynamic_cast<ContractDefinition const*>(function->scope());
		if (!contract)
			return _fallback;
		// A constructor has no name of its own; the conversion knows it as
		// `Contract.constructor`, and both sides have to agree or the call
		// references nothing.
		return contract->name() + "." + (function->isConstructor() ? "constructor" : function->name());
	}

	/// Emits a return, made to agree with the enclosing function's arity.
	///
	/// The verifier checks the two against each other, and several shapes come
	/// up short: `return;` in a function with named results, a tuple the
	/// expression generator produced one value for, and a body that falls off
	/// its end. Emitting the op anyway loses the whole contract instead of the
	/// one construct, so the difference is made up with the placeholder used
	/// everywhere else - which warns and counts as a drop rather than quietly
	/// returning something wrong.
	void emitReturn(mlir::Location _loc, llvm::SmallVector<mlir::Value, 4> _values)
	{
		mlir::Operation* parent = m_builder->getInsertionBlock()->getParentOp();
		while (parent && !mlir::isa<mlir::solidity::FunctionOp>(parent))
			parent = parent->getParentOp();
		if (auto enclosing = mlir::dyn_cast_or_null<mlir::solidity::FunctionOp>(parent))
		{
			mlir::ArrayRef<mlir::Type> const results = enclosing.getResultTypes();
			for (size_t i = _values.size(); i < results.size(); ++i)
				_values.push_back(emitUnsupported(_loc, results[i], "return value not generated"));
			_values.resize(std::min(_values.size(), results.size()));
		}
		m_builder->create<mlir::solidity::ReturnOp>(_loc, mlir::ValueRange{_values});
	}

	ContractDefinition const* m_mostDerivedContract = nullptr;

	mlir::Value generateSolidityExpression(Expression const& _expr)
	{
		auto loc = this->loc(_expr);

		if (auto* literal = dynamic_cast<Literal const*>(&_expr))
		{
			auto type = translateSolidityType(*_expr.annotation().type);

			if (literal->token() == langutil::Token::Number)
			{
				// Use the type's literalValue() to properly parse all numeric formats
				// (decimal, hex, underscores, etc.) and handle values up to 256 bits
				u256 bigValue = _expr.annotation().type->literalValue(literal);
				auto attr = m_builder->getIntegerAttr(
					mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
					llvm::APInt(256, bigValue.str(), 10));
				return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, type);
			}
			else if (literal->token() == langutil::Token::TrueLiteral)
			{
				auto attr = m_builder->getBoolAttr(true);
				return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, type);
			}
			else if (literal->token() == langutil::Token::FalseLiteral)
			{
				auto attr = m_builder->getBoolAttr(false);
				return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, type);
			}
			else if (literal->token() == langutil::Token::StringLiteral
				  || literal->token() == langutil::Token::UnicodeStringLiteral
				  || literal->token() == langutil::Token::HexStringLiteral)
			{
				return m_builder->create<mlir::solidity::StringLiteralOp>(
					loc, type, m_builder->getStringAttr(literal->value()));
			}
		}
		else if (auto* ident = dynamic_cast<Identifier const*>(&_expr))
		{
			if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
			{
				if (varDecl->isStateVariable())
				{
					auto type = translateSolidityType(*_expr.annotation().type);
					return m_builder->create<mlir::solidity::LoadStateVarOp>(loc, type, varDecl->name());
				}
				else if (m_valueMap.count(varDecl->id()))
				{
					return m_valueMap[varDecl->id()];
				}
				else
				{
					// Variable not found - this can happen with loop-local variables
					auto type = translateSolidityType(*_expr.annotation().type);
					auto zeroValue = emitUnsupported(loc, type, "undefined variable '" + varDecl->name() + "'");
					m_valueMap[varDecl->id()] = zeroValue;
					return zeroValue;
				}
			}
			else if (auto* enumValue = dynamic_cast<EnumValue const*>(ident->annotation().referencedDeclaration))
			{
				// Enum values are integer constants - find the index in the parent enum
				auto type = translateSolidityType(*_expr.annotation().type);
				int64_t val = 0;
				if (auto* enumDef = dynamic_cast<EnumDefinition const*>(enumValue->scope()))
				{
					auto const& members = enumDef->members();
					for (size_t i = 0; i < members.size(); ++i)
					{
						if (members[i].get() == enumValue)
						{
							val = static_cast<int64_t>(i);
							break;
						}
					}
				}
				return m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getIntegerAttr(m_builder->getI64Type(), val), type);
			}
			else if (ident->name() == "this")
			{
				return m_builder->create<mlir::solidity::SelfAddressOp>(
					loc, mlir::solidity::AddressType::get(m_context.get()));
			}
			else if (dynamic_cast<ContractDefinition const*>(ident->annotation().referencedDeclaration))
			{
				// Contract/interface/library reference — return zero placeholder.
				// Actual semantics handled at FunctionCall/MemberAccess level.
				auto type = translateSolidityType(*_expr.annotation().type);
				return m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getIntegerAttr(m_builder->getI64Type(), 0), type);
			}
			else
			{
				// Function references, contract references, etc.
				auto type = translateSolidityType(*_expr.annotation().type);
				return emitUnsupported(loc, type, "unresolved identifier '" + ident->name() + "'");
			}
		}
		else if (auto* binOp = dynamic_cast<BinaryOperation const*>(&_expr))
		{
			auto lhs = generateSolidityExpression(binOp->leftExpression());
			auto rhs = generateSolidityExpression(binOp->rightExpression());

			// Check if either operand is null
			if (!lhs || !rhs)
			{
				auto type = translateSolidityType(*_expr.annotation().type);
				return emitUnsupported(loc, type, "binary op with unsupported operand");
			}

			auto resultType = lhs.getType();

			// Signedness decides the opcode, not just the type: EVM has one set
			// of comparisons for unsigned words and another for two's
			// complement. Comparing int256(-1) with lt() asks whether
			// 2^256-1 < 1, which is false where Solidity says true - a wrong
			// answer with nothing to signal it. Div and Mod already pick their
			// signed form in the lowering, from the same operand type.
			bool const isSigned = llvm::isa<mlir::solidity::IntType>(lhs.getType())
				|| llvm::isa<mlir::solidity::IntType>(rhs.getType());
			auto predicate = [&](char const* _unsigned, char const* _signed) {
				return m_builder->getStringAttr(isSigned ? _signed : _unsigned);
			};

			switch (binOp->getOperator())
			{
			case langutil::Token::Add:
			{
				return m_builder->create<mlir::solidity::AddOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::Sub:
			{
				return m_builder->create<mlir::solidity::SubOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::Mul:
			{
				return m_builder->create<mlir::solidity::MulOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::Div:
			{
				return m_builder->create<mlir::solidity::DivOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::Mod:
			{
				return m_builder->create<mlir::solidity::ModOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::LessThan:
			{
				return m_builder->create<mlir::solidity::CmpOp>(
					loc, mlir::solidity::BoolType::get(m_context.get()),
					lhs, rhs, predicate("lt", "slt"));
			}
			case langutil::Token::GreaterThan:
			{
				return m_builder->create<mlir::solidity::CmpOp>(
					loc, mlir::solidity::BoolType::get(m_context.get()),
					lhs, rhs, predicate("gt", "sgt"));
			}
			case langutil::Token::Equal:
			{
				return m_builder->create<mlir::solidity::CmpOp>(
					loc, mlir::solidity::BoolType::get(m_context.get()),
					lhs, rhs, m_builder->getStringAttr("eq"));
			}
			case langutil::Token::LessThanOrEqual:
			{
				return m_builder->create<mlir::solidity::CmpOp>(
					loc, mlir::solidity::BoolType::get(m_context.get()),
					lhs, rhs, predicate("le", "sle"));
			}
			case langutil::Token::GreaterThanOrEqual:
			{
				return m_builder->create<mlir::solidity::CmpOp>(
					loc, mlir::solidity::BoolType::get(m_context.get()),
					lhs, rhs, predicate("ge", "sge"));
			}
			case langutil::Token::NotEqual:
			{
				return m_builder->create<mlir::solidity::CmpOp>(
					loc, mlir::solidity::BoolType::get(m_context.get()),
					lhs, rhs, m_builder->getStringAttr("ne"));
			}
			case langutil::Token::Exp:
			{
				return m_builder->create<mlir::solidity::ExpOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::BitAnd:
			{
				return m_builder->create<mlir::solidity::AndOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::BitOr:
			{
				return m_builder->create<mlir::solidity::OrOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::BitXor:
			{
				return m_builder->create<mlir::solidity::XorOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::SHL:
			{
				return m_builder->create<mlir::solidity::ShlOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::SAR:
			{
				// `>>` on a signed operand keeps the sign bit; on an unsigned
				// one it does not. These two were mapped to each other's ops.
				if (isSigned)
					return m_builder->create<mlir::solidity::SarOp>(loc, resultType, lhs, rhs);
				return m_builder->create<mlir::solidity::ShrOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::SHR:
			{
				return m_builder->create<mlir::solidity::ShrOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::And:
			{
				auto boolType = mlir::solidity::BoolType::get(m_context.get());
				return m_builder->create<mlir::solidity::LogicalAndOp>(
					loc,
					boolType,
					typedOperand(loc, lhs, boolType, "left operand of '&&' not generated as a boolean"),
					typedOperand(loc, rhs, boolType, "right operand of '&&' not generated as a boolean"));
			}
			case langutil::Token::Or:
			{
				auto boolType = mlir::solidity::BoolType::get(m_context.get());
				return m_builder->create<mlir::solidity::LogicalOrOp>(
					loc,
					boolType,
					typedOperand(loc, lhs, boolType, "left operand of '||' not generated as a boolean"),
					typedOperand(loc, rhs, boolType, "right operand of '||' not generated as a boolean"));
			}
			default:
				break;
			}
		}
		else if (auto* assignment = dynamic_cast<Assignment const*>(&_expr))
		{
			mlir::Value value;

			// Handle compound assignments (e.g., +=, *=, etc.)
			if (assignment->assignmentOperator() != Token::Assign)
			{
				// First, get the current value of the left-hand side
				mlir::Value currentValue;
				if (auto* ident = dynamic_cast<Identifier const*>(&assignment->leftHandSide()))
				{
					if (auto* varDecl
						= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (varDecl->isStateVariable())
						{
							currentValue = m_builder->create<mlir::solidity::LoadStateVarOp>(
								loc, translateSolidityType(*varDecl->type()), varDecl->name());
						}
						else
						{
							currentValue = m_valueMap[varDecl->id()];
						}
					}
				}
				else if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&assignment->leftHandSide()))
				{
					// Handle compound assignment to mapping elements: m[k] += v
					if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
					{
						std::string varName;
						auto keys = collectMappingKeys(*indexAccess, varName);
						auto valueType = translateSolidityType(*_expr.annotation().type);

						currentValue = m_builder->create<mlir::solidity::MappingAccessOp>(
							loc, valueType, m_builder->getStringAttr(varName), keys);
					}
				}
				else
				{
					// Generic fallback: use generateSolidityExpression for any LHS
					// Handles MemberAccess on IndexAccess (e.g., getStream[id].balance)
					currentValue = generateSolidityExpression(assignment->leftHandSide());
				}

				// Get the right-hand side value
				auto rightValue = generateSolidityExpression(assignment->rightHandSide());

				// Check if either operand is null
				if (!currentValue || !rightValue)
				{
					auto type = translateSolidityType(*_expr.annotation().type);
					return emitUnsupported(loc, type, "compound assignment with unsupported operand");
				}

				// Apply the operation based on the compound operator
				auto resultType = translateSolidityType(*_expr.annotation().type);
				switch (assignment->assignmentOperator())
				{
				case Token::AssignAdd:
					value = m_builder->create<mlir::solidity::AddOp>(loc, resultType, currentValue, rightValue);
					break;
				case Token::AssignSub:
					value = m_builder->create<mlir::solidity::SubOp>(loc, resultType, currentValue, rightValue);
					break;
				case Token::AssignMul:
					value = m_builder->create<mlir::solidity::MulOp>(loc, resultType, currentValue, rightValue);
					break;
				case Token::AssignDiv:
					value = m_builder->create<mlir::solidity::DivOp>(loc, resultType, currentValue, rightValue);
					break;
				case Token::AssignMod:
					value = m_builder->create<mlir::solidity::ModOp>(loc, resultType, currentValue, rightValue);
					break;
				default:
					// For unsupported compound operators, fall back to simple assignment
					value = rightValue;
					break;
				}
			}
			else
			{
				// Simple assignment
				value = generateSolidityExpression(assignment->rightHandSide());
			}

			// Store the value
			if (auto* ident = dynamic_cast<Identifier const*>(&assignment->leftHandSide()))
			{
				if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
				{
					if (varDecl->isStateVariable())
					{
						m_builder->create<mlir::solidity::StoreStateVarOp>(loc, varDecl->name(), value);
					}
					else
					{
						m_valueMap[varDecl->id()] = value;
					}
				}
			}
			else if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&assignment->leftHandSide()))
			{
				// Handle store to mapping elements: m[k] = v or m[k] += v
				if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
				{
					std::string varName;
					auto keys = collectMappingKeys(*indexAccess, varName);
					createMappingStoreOp(loc, varName, keys, value);
				}
				else if (dynamic_cast<ArrayType const*>(indexAccess->baseExpression().annotation().type))
				{
					auto base = generateSolidityExpression(indexAccess->baseExpression());
					auto index = generateSolidityExpression(*indexAccess->indexExpression());
					std::string varName = extractMappingVarName(indexAccess->baseExpression());
					if (!mlir::isa<mlir::solidity::ArrayType>(base.getType()))
						return emitUnsupported(loc, value.getType(), "store into an unsupported array expression");
					auto storeOp = m_builder->create<mlir::solidity::ArrayStoreOp>(loc, base, index, value);
					if (!varName.empty())
						storeOp->setAttr("varName", m_builder->getStringAttr(varName));
				}
			}
			else if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&assignment->leftHandSide()))
			{
				// Handle store to struct field in mapping: m[k].field = v or m[k].field += v
				if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&memberAccess->expression()))
				{
					if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
					{
						std::string varName;
						auto keys = collectMappingKeys(*indexAccess, varName);
						createMappingStoreOp(loc, varName, keys, value);
					}
				}
			}

			return value;
		}
		else if (auto* unaryOp = dynamic_cast<UnaryOperation const*>(&_expr))
		{
			auto operand = generateSolidityExpression(unaryOp->subExpression());
			auto resultType = translateSolidityType(*_expr.annotation().type);

			switch (unaryOp->getOperator())
			{
			case langutil::Token::BitNot:
			{
				return m_builder->create<mlir::solidity::NotOp>(loc, resultType, operand);
			}
			case langutil::Token::Inc:
			{
				// Pre/post increment: x++ or ++x
				auto one = m_builder->getIntegerAttr(m_builder->getI64Type(), 1);
				auto oneValue = m_builder->create<mlir::solidity::ConstantOp>(loc, one, resultType);

				mlir::Value result = m_builder->create<mlir::solidity::AddOp>(loc, resultType, operand, oneValue);

				// Store back to variable if it's an lvalue
				if (auto* ident = dynamic_cast<Identifier const*>(&unaryOp->subExpression()))
				{
					if (auto* varDecl
						= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (varDecl->isStateVariable())
						{
							m_builder->create<mlir::solidity::StoreStateVarOp>(loc, varDecl->name(), result);
						}
						else
						{
							m_valueMap[varDecl->id()] = result;
						}
					}
				}
				else if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&unaryOp->subExpression()))
				{
					if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
					{
						std::string varName;
						auto keys = collectMappingKeys(*indexAccess, varName);
						createMappingStoreOp(loc, varName, keys, result);
					}
				}

				// Return old value for post-increment, new value for pre-increment
				return unaryOp->isPrefixOperation() ? result : operand;
			}
			case langutil::Token::Dec:
			{
				// Pre/post decrement: x-- or --x
				auto one = m_builder->getIntegerAttr(m_builder->getI64Type(), 1);
				auto oneValue = m_builder->create<mlir::solidity::ConstantOp>(loc, one, resultType);

				mlir::Value result = m_builder->create<mlir::solidity::SubOp>(loc, resultType, operand, oneValue);

				// Store back to variable if it's an lvalue
				if (auto* ident = dynamic_cast<Identifier const*>(&unaryOp->subExpression()))
				{
					if (auto* varDecl
						= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (varDecl->isStateVariable())
						{
							m_builder->create<mlir::solidity::StoreStateVarOp>(loc, varDecl->name(), result);
						}
						else
						{
							m_valueMap[varDecl->id()] = result;
						}
					}
				}
				else if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&unaryOp->subExpression()))
				{
					if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
					{
						std::string varName;
						auto keys = collectMappingKeys(*indexAccess, varName);
						createMappingStoreOp(loc, varName, keys, result);
					}
				}

				// Return old value for post-decrement, new value for pre-decrement
				return unaryOp->isPrefixOperation() ? result : operand;
			}
			case langutil::Token::Delete:
			{
				auto zero = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
				return m_builder->create<mlir::solidity::ConstantOp>(loc, zero, resultType);
			}
			case langutil::Token::Sub:
			{
				// Unary minus
				auto zero = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
				auto zeroValue = m_builder->create<mlir::solidity::ConstantOp>(loc, zero, resultType);

				return m_builder->create<mlir::solidity::SubOp>(loc, resultType, zeroValue, operand);
			}
			case langutil::Token::Not:
			{
				// Logical NOT
				auto boolTy = mlir::solidity::BoolType::get(m_context.get());
				return m_builder->create<mlir::solidity::LogicalNotOp>(
					loc, boolTy, typedOperand(loc, operand, boolTy, "operand of '!' not generated as a boolean"));
			}
			default:
				break;
			}
		}
		else if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
		{
			// Handle member access (e.g., array.length, struct.field, msg.sender)
			std::string memberName = memberAccess->memberName();
			auto baseType = memberAccess->expression().annotation().type;

			// Handle msg.sender, msg.value, etc. (MagicType members)
			if (auto* magicType = dynamic_cast<MagicType const*>(baseType))
			{
				if (magicType->kind() == MagicType::Kind::Message)
				{
					if (memberName == "sender")
					{
						return m_builder->create<mlir::solidity::MsgSenderOp>(
							loc, mlir::solidity::AddressType::get(m_context.get()));
					}
					else if (memberName == "value")
					{
						return m_builder->create<mlir::solidity::MsgValueOp>(
							loc, mlir::solidity::UIntType::get(m_context.get(), 256));
					}
					else if (memberName == "data")
					{
						return m_builder->create<mlir::solidity::MsgDataOp>(
							loc, mlir::solidity::ArrayType::get(mlir::solidity::UIntType::get(m_context.get(), 8), -1));
					}
					else if (memberName == "sig")
					{
						return m_builder->create<mlir::solidity::MsgSigOp>(
							loc, mlir::solidity::BytesType::get(m_context.get(), 4));
					}
				}
				else if (magicType->kind() == MagicType::Kind::Block)
				{
					if (memberName == "timestamp")
					{
						return m_builder->create<mlir::solidity::BlockTimestampOp>(
							loc, mlir::solidity::UIntType::get(m_context.get(), 256));
					}
					else if (memberName == "number")
					{
						return m_builder->create<mlir::solidity::BlockNumberOp>(
							loc, mlir::solidity::UIntType::get(m_context.get(), 256));
					}
					else if (memberName == "chainid")
					{
						return m_builder->create<mlir::solidity::BlockChainIdOp>(
							loc, mlir::solidity::UIntType::get(m_context.get(), 256));
					}
				}
				else if (magicType->kind() == MagicType::Kind::Transaction)
				{
					if (memberName == "origin")
					{
						return m_builder->create<mlir::solidity::TxOriginOp>(
							loc, mlir::solidity::AddressType::get(m_context.get()));
					}
					else if (memberName == "gasprice")
					{
						return m_builder->create<mlir::solidity::TxGasPriceOp>(
							loc, mlir::solidity::UIntType::get(m_context.get(), 256));
					}
				}
			}

			// Handle type(X).max, type(X).min — base is a FunctionCall to type()
			if (auto* funcCallExpr = dynamic_cast<FunctionCall const*>(&memberAccess->expression()))
			{
				if (auto* typeIdent = dynamic_cast<Identifier const*>(&funcCallExpr->expression()))
				{
					if (typeIdent->name() == "type" && !funcCallExpr->arguments().empty())
					{
						auto const* typeArg = funcCallExpr->arguments()[0].get();
						if (auto* typeArgExpr = dynamic_cast<ElementaryTypeNameExpression const*>(typeArg))
						{
							auto const& typeName = typeArgExpr->type();
							auto resultType = translateSolidityType(*_expr.annotation().type);

							if (memberName == "max")
							{
								mlir::Attribute valueAttr;
								if (auto* intType = dynamic_cast<IntegerType const*>(&typeName))
								{
									if (intType->isSigned())
									{
										unsigned bits = intType->numBits();
										u256 maxVal = (u256(1) << (bits - 1)) - 1;
										valueAttr = m_builder->getIntegerAttr(
											mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
											llvm::APInt(256, maxVal.str(), 10));
									}
									else
									{
										unsigned bits = intType->numBits();
										u256 maxVal = (u256(1) << bits) - 1;
										valueAttr = m_builder->getIntegerAttr(
											mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
											llvm::APInt(256, maxVal.str(), 10));
									}
								}
								else
								{
									valueAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
								}
								return m_builder->create<mlir::solidity::ConstantOp>(loc, valueAttr, resultType);
							}
							else if (memberName == "min")
							{
								mlir::Attribute valueAttr;
								if (auto* intType = dynamic_cast<IntegerType const*>(&typeName))
								{
									if (intType->isSigned())
									{
										unsigned bits = intType->numBits();
										u256 minVal = u256(1) << (bits - 1);
										valueAttr = m_builder->getIntegerAttr(
											mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
											llvm::APInt(256, minVal.str(), 10));
									}
									else
									{
										valueAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
									}
								}
								else
								{
									valueAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
								}
								return m_builder->create<mlir::solidity::ConstantOp>(loc, valueAttr, resultType);
							}
						}
					}
				}
			}

			// `type(T).min`, `.max` and `.interfaceId` are compile-time
			// constants the analyser has already worked out; there is nothing
			// to evaluate at run time. Without these the whole expression was
			// dropped and the function answered zero.
			if (auto* magic = dynamic_cast<MagicType const*>(memberAccess->expression().annotation().type))
				if (magic->kind() == MagicType::Kind::MetaType)
				{
					auto resultType = translateSolidityType(*_expr.annotation().type);
					auto wordConstant = [&](bigint const& _value) {
						// Two's complement for the signed minima.
						bigint wrapped = _value < 0 ? (bigint(1) << 256) + _value : _value;
						// A fixed-bytes value sits at the top of the word, so
						// `interfaceId` as bytes4 is the selector shifted up -
						// right-aligned it reads as zero from the ABI.
						if (auto* fixedBytes = dynamic_cast<FixedBytesType const*>(_expr.annotation().type))
							wrapped <<= (32 - fixedBytes->numBytes()) * 8;
						auto attr = m_builder->getIntegerAttr(
							mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
							llvm::APInt(256, u256(wrapped).str(), 10));
						return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, resultType).getResult();
					};

					Type const* argument = magic->typeArgument();
					if (memberName == "interfaceId")
					{
						if (auto* contractType = dynamic_cast<ContractType const*>(argument))
							return wordConstant(bigint(contractType->contractDefinition().interfaceId()));
					}
					else if (memberName == "min" || memberName == "max")
					{
						bool const wantMin = memberName == "min";
						if (auto* enumType = dynamic_cast<EnumType const*>(argument))
							return wordConstant(wantMin ? bigint(0) : bigint(enumType->numberOfMembers() - 1));
						if (auto* integerType = dynamic_cast<IntegerType const*>(argument))
							return wordConstant(wantMin ? integerType->minValue() : integerType->maxValue());
					}
				}

			// Handle EnumDefinition.member (e.g., State.OPEN)
			if (auto* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
			{
				if (auto* enumDef = dynamic_cast<EnumDefinition const*>(baseIdent->annotation().referencedDeclaration))
				{
					auto type = translateSolidityType(*_expr.annotation().type);
					int64_t val = 0;
					for (size_t i = 0; i < enumDef->members().size(); ++i)
						if (enumDef->members()[i]->name() == memberAccess->memberName())
						{ val = static_cast<int64_t>(i); break; }
					return m_builder->create<mlir::solidity::ConstantOp>(
						loc, m_builder->getIntegerAttr(m_builder->getI64Type(), val), type);
				}
			}

			// Handle .selector on FunctionType (e.g., ERC721TokenReceiver.onERC721Received.selector)
			if (memberName == "selector")
			{
				auto* exprType = memberAccess->expression().annotation().type;
				if (auto* funcType = dynamic_cast<FunctionType const*>(exprType))
				{
					if (funcType->hasDeclaration())
					{
						u256 selectorVal = util::selectorFromSignatureU256(funcType->externalSignature());
						auto resultType = translateSolidityType(*_expr.annotation().type);
						auto attr = m_builder->getIntegerAttr(
							mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
							llvm::APInt(256, selectorVal.str(), 10));
						return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, resultType);
					}
				}
			}

			auto base = generateSolidityExpression(memberAccess->expression());
			if (!base)
			{
				auto type = translateSolidityType(*_expr.annotation().type);
				return emitUnsupported(loc, type, "member access '." + memberAccess->memberName() + "' on unsupported base");
			}

			// Handle array.length
			if (auto* arrayType = dynamic_cast<ArrayType const*>(baseType))
			{
				if (memberName == "length")
				{
					// The Solidity type says array, but a dropped sub-expression
					// leaves a plain word in its place. Building the op anyway
					// produces IR the verifier rejects, which loses the whole
					// contract instead of just the feature that was dropped.
					if (!mlir::isa<mlir::solidity::ArrayType>(base.getType()))
						return emitUnsupported(
							loc,
							translateSolidityType(*_expr.annotation().type),
							"length of an unsupported array expression");
					return m_builder->create<mlir::solidity::ArrayLengthOp>(
						loc, mlir::solidity::UIntType::get(m_context.get(), 256), base);
				}
			}
			// Handle struct member access
			else if (auto* structType = dynamic_cast<StructType const*>(baseType))
			{
				auto op = m_builder->create<mlir::solidity::MemberAccessOp>(
					loc, translateSolidityType(*_expr.annotation().type), base, m_builder->getStringAttr(memberName));
				// Compute field index for lowering
				int64_t fieldIdx = 0;
				auto const& members = structType->structDefinition().members();
				for (size_t i = 0; i < members.size(); ++i)
					if (members[i]->name() == memberName) { fieldIdx = static_cast<int64_t>(i); break; }
				op->setAttr("fieldIndex", m_builder->getI64IntegerAttr(fieldIdx));
				return op;
			}
			// Handle address member access (balance, code, codehash)
			else if (dynamic_cast<AddressType const*>(baseType))
			{
				if (memberName == "balance")
				{
					return m_builder->create<mlir::solidity::AddressBalanceOp>(
						loc, mlir::solidity::UIntType::get(m_context.get(), 256), base);
				}
				else if (memberName == "code")
				{
					return m_builder->create<mlir::solidity::AddressCodeOp>(
						loc,
						mlir::solidity::ArrayType::get(
							mlir::solidity::UIntType::get(m_context.get(), 8),
							/*size=*/-1),
						base);
				}
				else if (memberName == "codehash")
				{
					return m_builder->create<mlir::solidity::AddressCodehashOp>(
						loc, mlir::solidity::BytesType::get(m_context.get(), 32), base);
				}
			}

			// Return base for unhandled member access
			return base;
		}
		else if (auto* funcCall = dynamic_cast<FunctionCall const*>(&_expr))
		{
			// `((super).f)()` is the same call as `super.f()`. Each level of
			// parentheses arrives as a one-component tuple, so dispatching on
			// the node type without stripping them misses every parenthesised
			// call and it falls through to a path that cannot resolve it.
			Expression const& callee = withoutParentheses(funcCall->expression());
			// Handle .call{value: amount}("") pattern — FunctionCallOptions wrapping
			if (auto* funcCallOpts = dynamic_cast<FunctionCallOptions const*>(&callee))
			{
				if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&funcCallOpts->expression()))
				{
					if (memberAccess->memberName() == "call")
					{
						auto address = generateSolidityExpression(memberAccess->expression());

						// Extract {value: ...} option
						mlir::Value callValue;
						for (size_t i = 0; i < funcCallOpts->names().size(); ++i)
						{
							if (*funcCallOpts->names()[i] == "value")
							{
								callValue = generateSolidityExpression(*funcCallOpts->options()[i]);
								break;
							}
						}

						if (!callValue)
						{
							auto zero = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
							callValue = m_builder->create<mlir::solidity::ConstantOp>(
								loc, zero, mlir::solidity::UIntType::get(m_context.get(), 256));
						}

						auto boolType = mlir::solidity::BoolType::get(m_context.get());
						return m_builder->create<mlir::solidity::LowLevelCallOp>(
							loc, boolType, address, callValue);
					}
				}
			}

			// Handle member function calls first (e.g., array.push(), .transfer())
			if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&callee))
			{
				// Don't generate base for abi.* calls or contract/library calls
				bool skipBaseGen = false;
				if (auto* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
				{
					skipBaseGen = (baseIdent->name() == "abi");
					if (!skipBaseGen && dynamic_cast<ContractDefinition const*>(baseIdent->annotation().referencedDeclaration))
						skipBaseGen = true;
				}

				if (!skipBaseGen)
				{
					auto base = generateSolidityExpression(memberAccess->expression());
					std::string memberName = memberAccess->memberName();
					auto baseType = memberAccess->expression().annotation().type;

					// Handle array.push()
					if (auto* arrayType = dynamic_cast<ArrayType const*>(baseType))
					{
						if (memberName == "push" && !funcCall->arguments().empty())
						{
							auto value = generateSolidityExpression(*funcCall->arguments()[0]);
							if (!mlir::isa<mlir::solidity::ArrayType>(base.getType()))
								return emitUnsupported(loc, value.getType(), "push onto an unsupported array expression");
							m_builder->create<mlir::solidity::ArrayPushOp>(loc, base, value);
							return base;
						}
					}

					// Handle .transfer(amount) — reverts on failure
					if (memberName == "transfer" && funcCall->arguments().size() == 1 && base)
					{
						auto amount = generateSolidityExpression(*funcCall->arguments()[0]);
						if (amount)
						{
							m_builder->create<mlir::solidity::TransferOp>(loc, base, amount);
							return nullptr;
						}
					}
				}
			}
			// Handle struct constructor calls
			else if (auto* typeConversion = dynamic_cast<Identifier const*>(&callee))
			{
				if (auto* structDecl
					= dynamic_cast<StructDefinition const*>(typeConversion->annotation().referencedDeclaration))
				{
					// Collect field values
					std::vector<mlir::Value> operands;
					for (auto const& arg: funcCall->arguments())
					{
						auto value = generateSolidityExpression(*arg);
						if (!value)
						{
							auto type = translateSolidityType(*arg->annotation().type);
							value = emitUnsupported(loc, type, "struct constructor argument");
						}
						operands.push_back(value);
					}

					auto resultType = translateSolidityType(*_expr.annotation().type);
					return m_builder->create<mlir::solidity::StructCreateOp>(
						loc, resultType, m_builder->getStringAttr(structDecl->name()), operands);
				}
			}
			// Handle new ContractName(args) — contract creation via CREATE opcode
			else if (auto* newExpr = dynamic_cast<NewExpression const*>(&callee))
			{
				std::string typeName;
				if (auto* userDefined = dynamic_cast<UserDefinedTypeName const*>(&newExpr->typeName()))
					typeName = userDefined->pathNode().path().back();
				else
					typeName = "UnknownContract";

				// Track the child contract for sub-object generation
				// Also build the full object name (Name_ID) for Yul dataoffset/datasize
				std::string fullObjectName = typeName;
				if (auto* funcType = dynamic_cast<FunctionType const*>(funcCall->expression().annotation().type))
				{
					if (!funcType->returnParameterTypes().empty())
					{
						if (auto* contractType = dynamic_cast<ContractType const*>(funcType->returnParameterTypes().front()))
						{
							m_childContracts.insert(&contractType->contractDefinition());
							fullObjectName = typeName + "_" + std::to_string(contractType->contractDefinition().id());
						}
					}
				}

				std::vector<mlir::Value> args;
				for (auto const& arg: funcCall->arguments())
				{
					auto argValue = generateSolidityExpression(*arg);
					if (argValue)
						args.push_back(argValue);
				}

				// msg.value = 0 (no value sent with plain `new`)
				auto zeroValue = m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getIntegerAttr(
						mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned), 0),
					mlir::solidity::UIntType::get(m_context.get(), 256));

				auto resultType = translateSolidityType(*_expr.annotation().type);
				return m_builder->create<mlir::solidity::CreateContractOp>(
					loc, resultType,
					m_builder->getStringAttr(fullObjectName),
					zeroValue, args).getResult();
			}

			if (auto* ident = dynamic_cast<Identifier const*>(&callee))
			{
				// Handle special functions like require, assert, revert
				if (ident->name() == "require")
				{
					if (!funcCall->arguments().empty())
					{
						auto cond = generateSolidityExpression(*funcCall->arguments()[0]);
						mlir::StringAttr msgAttr;
						if (funcCall->arguments().size() > 1)
						{
							if (auto* literal = dynamic_cast<Literal const*>(funcCall->arguments()[1].get()))
								msgAttr = m_builder->getStringAttr(literal->value());
						}
						m_builder->create<mlir::solidity::RequireOp>(
							loc,
							typedOperand(loc, cond, mlir::solidity::BoolType::get(m_context.get()),
								"require condition not generated as a boolean"),
							msgAttr);
					}
					return nullptr;
				}
				else if (ident->name() == "assert")
				{
					if (!funcCall->arguments().empty())
					{
						auto cond = generateSolidityExpression(*funcCall->arguments()[0]);
						m_builder->create<mlir::solidity::AssertOp>(
							loc,
							typedOperand(loc, cond, mlir::solidity::BoolType::get(m_context.get()),
								"assert condition not generated as a boolean"));
					}
					return nullptr;
				}
				else if (ident->name() == "revert")
				{
					mlir::StringAttr reasonAttr;
					if (!funcCall->arguments().empty())
					{
						if (auto* literal = dynamic_cast<Literal const*>(funcCall->arguments()[0].get()))
							reasonAttr = m_builder->getStringAttr(literal->value());
					}
					m_builder->create<mlir::solidity::RevertOp>(loc, reasonAttr);
					// Revert is a terminating operation, return nullptr to indicate no value
					// The dummy value creation below will be skipped
					return nullptr;
				}
			}

			// Handle type conversions (e.g., address(this), uint256(x))
			if (funcCall->annotation().kind.set() && *funcCall->annotation().kind == FunctionCallKind::TypeConversion)
			{
				// Get the argument to convert
				if (!funcCall->arguments().empty())
				{
					auto argValue = generateSolidityExpression(*funcCall->arguments()[0]);
					auto targetType = translateSolidityType(*funcCall->annotation().type);

					// Create a type conversion operation
					return m_builder->create<mlir::solidity::ConvertOp>(loc, targetType, argValue);
				}
			}

			// Handle regular function calls (internal functions, etc.)
			if (funcCall->annotation().kind.set() && *funcCall->annotation().kind == FunctionCallKind::FunctionCall)
			{
				std::string funcName;

				// Get function name from identifier
				if (auto* ident = dynamic_cast<Identifier const*>(&callee))
				{
					funcName = ident->name();

					// Handle built-in functions with proper MLIR operations

					// addmod(a, b, n) - modular addition
					if (funcName == "addmod" && funcCall->arguments().size() == 3)
					{
						auto a = generateSolidityExpression(*funcCall->arguments()[0]);
						auto b = generateSolidityExpression(*funcCall->arguments()[1]);
						auto n = generateSolidityExpression(*funcCall->arguments()[2]);
						auto resultType = translateSolidityType(*_expr.annotation().type);

						return m_builder->create<mlir::solidity::AddModOp>(loc, resultType, a, b, n);
					}

					// mulmod(a, b, n) - modular multiplication
					if (funcName == "mulmod" && funcCall->arguments().size() == 3)
					{
						auto a = generateSolidityExpression(*funcCall->arguments()[0]);
						auto b = generateSolidityExpression(*funcCall->arguments()[1]);
						auto n = generateSolidityExpression(*funcCall->arguments()[2]);
						auto resultType = translateSolidityType(*_expr.annotation().type);

						return m_builder->create<mlir::solidity::MulModOp>(loc, resultType, a, b, n);
					}

					// gasleft() - get remaining gas
					if (funcName == "gasleft" && funcCall->arguments().empty())
					{
						return m_builder->create<mlir::solidity::GasLeftOp>(
							loc, mlir::solidity::UIntType::get(m_context.get(), 256));
					}

					// blockhash(blockNumber) - get block hash
					if (funcName == "blockhash" && funcCall->arguments().size() == 1)
					{
						auto blockNumber = generateSolidityExpression(*funcCall->arguments()[0]);

						return m_builder->create<mlir::solidity::BlockhashOp>(
							loc, mlir::solidity::BytesType::get(m_context.get(), 32), blockNumber);
					}

					// keccak256(data) - hash function
					if (funcName == "keccak256" && funcCall->arguments().size() == 1)
					{
						auto data = generateSolidityExpression(*funcCall->arguments()[0]);

						return m_builder->create<mlir::solidity::Keccak256Op>(
							loc, mlir::solidity::BytesType::get(m_context.get(), 32), data);
					}

					// sha256(data) - hash function
					if (funcName == "sha256" && funcCall->arguments().size() == 1)
					{
						auto data = generateSolidityExpression(*funcCall->arguments()[0]);

						return m_builder->create<mlir::solidity::Sha256Op>(
							loc, mlir::solidity::BytesType::get(m_context.get(), 32), data);
					}

					// ripemd160(data) - hash function
					if (funcName == "ripemd160" && funcCall->arguments().size() == 1)
					{
						auto data = generateSolidityExpression(*funcCall->arguments()[0]);

						// ripemd160 returns bytes20, but padded to 32 bytes in EVM
						return m_builder->create<mlir::solidity::Ripemd160Op>(
							loc, mlir::solidity::BytesType::get(m_context.get(), 20), data);
					}

					// ecrecover(hash, v, r, s) - signature recovery
					if (funcName == "ecrecover" && funcCall->arguments().size() == 4)
					{
						auto hash = generateSolidityExpression(*funcCall->arguments()[0]);
						auto v = generateSolidityExpression(*funcCall->arguments()[1]);
						auto r = generateSolidityExpression(*funcCall->arguments()[2]);
						auto s = generateSolidityExpression(*funcCall->arguments()[3]);

						return m_builder->create<mlir::solidity::EcrecoverOp>(
							loc, mlir::solidity::AddressType::get(m_context.get()), hash, v, r, s);
					}

					// selfdestruct(recipient) - destroy contract
					if (funcName == "selfdestruct" && funcCall->arguments().size() == 1)
					{
						auto recipient = generateSolidityExpression(*funcCall->arguments()[0]);

						m_builder->create<mlir::solidity::SelfdestructOp>(loc, recipient);
						return nullptr;
					}

					// type(X) - handled separately as member access (type(uint256).max etc.)
					if (funcName == "type")
					{
						auto dummyType = translateSolidityType(*_expr.annotation().type);
						return emitUnsupported(loc, dummyType, "type() expression");
					}
				}
				// Handle member function calls (e.g., library.func() or obj.method())
				else if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&callee))
				{
					// Check for abi.encode, abi.encodePacked, abi.decode, type().max, etc.
					// These need special handling and cannot be generated as function calls
					if (auto* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
					{
						std::string baseName = baseIdent->name();
						if (baseName == "abi")
						{
							std::string abiFunc = memberAccess->memberName();

							// Collect all arguments
							std::vector<mlir::Value> argValues;
							for (auto const& arg: funcCall->arguments())
							{
								auto val = generateSolidityExpression(*arg);
								if (val)
									argValues.push_back(val);
							}

							auto resultType = translateSolidityType(*_expr.annotation().type);

							if (abiFunc == "encode")
							{
								return m_builder->create<mlir::solidity::AbiEncodeOp>(loc, resultType, argValues);
							}
							else if (abiFunc == "encodePacked")
							{
								return m_builder->create<mlir::solidity::AbiEncodePackedOp>(loc, resultType, argValues);
							}
							else if (abiFunc == "encodeWithSelector")
							{
								return m_builder->create<mlir::solidity::AbiEncodeWithSelectorOp>(loc, resultType, argValues);
							}
							else if (abiFunc == "encodeWithSignature")
							{
								return m_builder->create<mlir::solidity::AbiEncodeWithSignatureOp>(loc, resultType, argValues);
							}
							else if (abiFunc == "decode")
							{
								return m_builder->create<mlir::solidity::AbiDecodeOp>(loc, resultType, argValues);
							}
							else
							{
								return emitUnsupported(loc, resultType, "abi." + abiFunc + "()");
							}
						}
					}
					funcName = memberAccess->memberName();

					// Check for remaining low-level calls (send, delegatecall, staticcall)
					// Note: .transfer() is handled earlier, .call{value:} via FunctionCallOptions
					if (funcName == "send" || funcName == "delegatecall" || funcName == "staticcall")
					{
						auto dummyType = translateSolidityType(*_expr.annotation().type);
						return emitUnsupported(loc, dummyType, "low-level call '." + funcName + "()'");
					}

					// Check for external contract/interface calls
					auto* funcType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type);
					if (funcType && funcType->kind() == FunctionType::Kind::External)
					{
						// Evaluate base expression to get target address
						auto targetAddr = generateSolidityExpression(memberAccess->expression());
						if (!targetAddr)
						{
							auto dummyType = translateSolidityType(*_expr.annotation().type);
							return emitUnsupported(loc, dummyType, "external call to '" + funcName + "': failed to evaluate target");
						}

						// Generate arguments
						std::vector<mlir::Value> args;
						for (auto const& arg: funcCall->arguments())
						{
							auto argValue = generateSolidityExpression(*arg);
							if (argValue)
								args.push_back(argValue);
						}

						// Compute selector and signature
						uint32_t selector = util::selectorFromSignatureU32(funcType->externalSignature());
						std::string signature = funcType->externalSignature();

						// Create op with appropriate result types
						auto resultType = translateSolidityType(*_expr.annotation().type);
						return m_builder->create<mlir::solidity::ExternalFunctionCallOp>(
							loc, mlir::TypeRange{resultType}, targetAddr,
							m_builder->getI32IntegerAttr(selector),
							m_builder->getStringAttr(signature),
							args).getResult(0);
					}

					// Check for library "using for" bound function calls
					// e.g., to.safeTransferETH(amount) => SafeTransferLib.safeTransferETH(to, amount)
					if (funcType && funcType->hasBoundFirstArgument())
					{
						// Evaluate base expression as first argument
						auto baseValue = generateSolidityExpression(memberAccess->expression());

						std::vector<mlir::Value> args;
						if (baseValue)
							args.push_back(baseValue);
						for (auto const& arg: funcCall->arguments())
						{
							auto argValue = generateSolidityExpression(*arg);
							if (argValue)
								args.push_back(argValue);
						}

						padCallArguments(*funcCall, args, loc);

						if (!dynamic_cast<TupleType const*>(_expr.annotation().type)
							|| !dynamic_cast<TupleType const*>(_expr.annotation().type)->components().empty())
						{
							auto resultTypes = callResultTypes(*_expr.annotation().type);
							return m_builder->create<mlir::solidity::FunctionCallOp>(
								loc, mlir::TypeRange{resultTypes}, m_builder->getStringAttr(resolvedCalleeName(*funcCall, funcName)), args).getResult(0);
						}
						else
						{
							m_builder->create<mlir::solidity::FunctionCallOp>(
								loc, mlir::TypeRange{}, m_builder->getStringAttr(resolvedCalleeName(*funcCall, funcName)), args);
							return nullptr;
						}
					}
				}

				if (!funcName.empty())
				{
					// Generate arguments
					std::vector<mlir::Value> args;
					for (auto const& arg: funcCall->arguments())
					{
						auto argValue = generateSolidityExpression(*arg);
						if (argValue)
							args.push_back(argValue);
					}

					// Create function call operation
					// Add result type if the function has a return value
					padCallArguments(*funcCall, args, loc);

					if (!dynamic_cast<TupleType const*>(_expr.annotation().type)
						|| !dynamic_cast<TupleType const*>(_expr.annotation().type)->components().empty())
					{
						auto resultTypes = callResultTypes(*_expr.annotation().type);
						return m_builder->create<mlir::solidity::FunctionCallOp>(
							loc, mlir::TypeRange{resultTypes}, m_builder->getStringAttr(resolvedCalleeName(*funcCall, funcName)), args).getResult(0);
					}
					else
					{
						// Void function
						m_builder->create<mlir::solidity::FunctionCallOp>(
							loc, mlir::TypeRange{}, m_builder->getStringAttr(resolvedCalleeName(*funcCall, funcName)), args);
						return nullptr;
					}
				}
			}
		}
		else if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&_expr))
		{
			if (dynamic_cast<MappingType const*>(
					indexAccess->baseExpression().annotation().type))
			{
				// Collect all keys from nested IndexAccess chain
				std::string varName;
				auto keys = collectMappingKeys(*indexAccess, varName);
				auto valueType = translateSolidityType(*_expr.annotation().type);

				return m_builder->create<mlir::solidity::MappingAccessOp>(
					loc, valueType, m_builder->getStringAttr(varName), keys);
			}
			else if (dynamic_cast<ArrayType const*>(indexAccess->baseExpression().annotation().type))
			{
				auto base = generateSolidityExpression(indexAccess->baseExpression());
				mlir::Value index;
				if (indexAccess->indexExpression())
					index = generateSolidityExpression(*indexAccess->indexExpression());
				auto elementType = translateSolidityType(*_expr.annotation().type);
				if (base && index)
				{
					if (!mlir::isa<mlir::solidity::ArrayType>(base.getType()))
						return emitUnsupported(loc, elementType, "index into an unsupported array expression");
					auto op = m_builder->create<mlir::solidity::ArrayAccessOp>(loc, elementType, base, index);
					std::string varName = extractMappingVarName(indexAccess->baseExpression());
					if (!varName.empty())
						op->setAttr("varName", m_builder->getStringAttr(varName));
					return op;
				}
				return emitUnsupported(loc, elementType, "array element access with null operand");
			}
			else
			{
				generateSolidityExpression(indexAccess->baseExpression());
				if (indexAccess->indexExpression())
					generateSolidityExpression(*indexAccess->indexExpression());
				auto type = translateSolidityType(*_expr.annotation().type);
				return emitUnsupported(loc, type, "index access");
			}
		}
		else if (auto* conditional = dynamic_cast<Conditional const*>(&_expr))
		{
			auto condition = generateSolidityExpression(conditional->condition());
			auto trueVal = generateSolidityExpression(conditional->trueExpression());
			auto falseVal = generateSolidityExpression(conditional->falseExpression());
			auto resultType = translateSolidityType(*_expr.annotation().type);

			return m_builder->create<mlir::solidity::SelectOp>(
				loc, resultType, condition, trueVal, falseVal);
		}
		else if (auto* tupleExpr = dynamic_cast<TupleExpression const*>(&_expr))
		{
			auto const& components = tupleExpr->components();
			if (!tupleExpr->isInlineArray() && components.size() == 1 && components[0])
				return generateSolidityExpression(*components[0]);
			for (auto const& comp : components)
				if (comp)
					return generateSolidityExpression(*comp);
			auto type = translateSolidityType(*_expr.annotation().type);
			return emitUnsupported(loc, type, "empty tuple expression");
		}

		// Unhandled expression type
		auto dummyType = translateSolidityType(*_expr.annotation().type);
		return emitUnsupported(loc, dummyType, "unsupported expression: " + demangle(typeid(_expr).name()));
	}

	mlir::Value generateSolidityStatement(Statement const& _stmt)
	{
		auto loc = this->loc(_stmt);

		if (auto* block = dynamic_cast<Block const*>(&_stmt))
		{
			if (block->unchecked())
			{
				// Generate unchecked block statements inline to avoid SSA scope issues.
				// Creating a nested UncheckedOp region causes values defined inside
				// (like loop counter increments) to be inaccessible in the parent scope.
				// Unchecked semantics don't affect Yul lowering.
				mlir::Value lastValue;
				for (auto const& stmt: block->statements())
				{
					// A break, continue or return ends the block. Statements
					// after it are unreachable, and emitting them puts ops past
					// a terminator - which costs the whole contract, not the
					// dead code.
					mlir::Block* current = m_builder->getInsertionBlock();
					if (current && !current->empty() && current->back().hasTrait<mlir::OpTrait::IsTerminator>())
						break;
					lastValue = generateSolidityStatement(*stmt);
				}
				return lastValue;
			}
			else
			{
				mlir::Value lastValue;
				for (auto const& stmt: block->statements())
				{
					// A break, continue or return ends the block. Statements
					// after it are unreachable, and emitting them puts ops past
					// a terminator - which costs the whole contract, not the
					// dead code.
					mlir::Block* current = m_builder->getInsertionBlock();
					if (current && !current->empty() && current->back().hasTrait<mlir::OpTrait::IsTerminator>())
						break;
					lastValue = generateSolidityStatement(*stmt);
				}
				return lastValue;
			}
		}
		else if (auto* ifStmt = dynamic_cast<IfStatement const*>(&_stmt))
		{
			auto condition = generateSolidityExpression(ifStmt->condition());
			bool hasElse = ifStmt->falseStatement() != nullptr;

			auto ifOp = m_builder->create<mlir::solidity::IfOp>(loc, condition);

			// Generate then region
			m_builder->setInsertionPointToEnd(&ifOp.getThenRegion().emplaceBlock());
			generateSolidityStatement(ifStmt->trueStatement());

			// Generate else region if present
			if (hasElse)
			{
				m_builder->setInsertionPointToEnd(&ifOp.getElseRegion().emplaceBlock());
				generateSolidityStatement(*ifStmt->falseStatement());
			}
			else
			{
				// Create an empty else region - it will be empty which is valid for solidity.if
				ifOp.getElseRegion().emplaceBlock();
			}

			// Reset insertion point after the if statement
			// This ensures subsequent statements are generated after the if, not inside it
			m_builder->setInsertionPointAfter(ifOp.getOperation());

			// Same rule as for loops: a variable assigned only in a branch has
			// no value after the `if`, so it must not stay in the map naming
			// one from inside the region.
			dropValuesEscaping(ifOp.getOperation(), loc);
		}
		else if (auto* forStmt = dynamic_cast<ForStatement const*>(&_stmt))
		{
			// Properly use scf.while with loop-carried values
			// Analyze the loop to find all modified variables

			// Collect all variables that need to be loop-carried:
			// 1. Variables modified in the loop body
			// 2. Variables modified in the loop update expression
			// 3. Loop counter variable from initialization
			std::set<int64_t> loopCarriedVarIds;

			// Analyze loop body for modified variables
			auto bodyModifiedVars = collectModifiedVariables(forStmt->body());
			loopCarriedVarIds.insert(bodyModifiedVars.begin(), bodyModifiedVars.end());

			// Analyze loop update expression
			if (forStmt->loopExpression())
			{
				if (auto* exprStmt = dynamic_cast<ExpressionStatement const*>(forStmt->loopExpression()))
				{
					auto updateModifiedVars = collectModifiedVariables(*exprStmt);
					loopCarriedVarIds.insert(updateModifiedVars.begin(), updateModifiedVars.end());
				}
			}

			// Handle initialization and identify loop counter variable
			int64_t loopCounterId = -1;
			if (forStmt->initializationExpression())
			{
				generateSolidityStatement(*forStmt->initializationExpression());

				// Find the loop counter variable from initialization
				if (auto* varDeclStmt
					= dynamic_cast<VariableDeclarationStatement const*>(forStmt->initializationExpression()))
				{
					for (auto const& decl: varDeclStmt->declarations())
					{
						if (decl)
						{
							loopCounterId = decl->id();
							loopCarriedVarIds.insert(loopCounterId);
							break;
						}
					}
				}
			}

			// Also check condition for referenced variables that might need to be carried
			if (forStmt->condition())
			{
				auto condReferencedVars = collectReferencedVariables(*forStmt->condition());
				// Only add variables that are also modified (already in the set)
				for (auto varId: condReferencedVars)
				{
					if (bodyModifiedVars.count(varId) > 0 || varId == loopCounterId)
						loopCarriedVarIds.insert(varId);
				}
			}

			// Store loop-carried variables for break/continue handling
			m_loopCarriedVarIds = loopCarriedVarIds;
			m_immediateLoopContext = LoopContext::SCF; // We're in an SCF loop

			// Prepare initial values and types for loop-carried variables
			std::vector<mlir::Value> initialValues;
			std::vector<mlir::Type> loopCarriedTypes;
			std::vector<int64_t> loopCarriedVarIdsList; // Keep ordered list

			for (auto varId: loopCarriedVarIds)
			{
				if (m_valueMap.count(varId) > 0)
				{
					initialValues.push_back(m_valueMap[varId]);
					loopCarriedTypes.push_back(m_valueMap[varId].getType());
					loopCarriedVarIdsList.push_back(varId);
				}
			}

			// If no loop-carried values, create a dummy one to satisfy SCF requirements
			if (initialValues.empty())
			{
				auto zeroValue = m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getI64IntegerAttr(0), mlir::solidity::UIntType::get(m_context.get(), 256));
				initialValues.push_back(zeroValue);
				loopCarriedTypes.push_back(mlir::solidity::UIntType::get(m_context.get(), 256));
			}

			// Create the scf.while with proper loop-carried values
			auto whileOp = m_builder->create<
				mlir::scf::WhileOp>(loc, mlir::TypeRange{loopCarriedTypes}, mlir::ValueRange{initialValues});

			// Build the "before" region (condition check)
			{
				auto& beforeRegion = whileOp.getBefore();
				auto* beforeBlock = m_builder->createBlock(&beforeRegion);

				// Add block arguments for loop-carried values
				for (auto type: loopCarriedTypes)
					beforeBlock->addArgument(type, loc);

				m_builder->setInsertionPointToEnd(beforeBlock);

				// Map block arguments to variables
				for (size_t i = 0; i < loopCarriedVarIdsList.size(); ++i)
				{
					m_valueMap[loopCarriedVarIdsList[i]] = beforeBlock->getArgument(i);
				}

				// Generate the condition
				mlir::Value condValue;
				if (forStmt->condition())
				{
					auto solidityBool = generateSolidityExpression(*forStmt->condition());
					// Convert solidity.bool to i1 for SCF
					condValue = m_builder->create<mlir::solidity::ToI1Op>(
						loc, m_builder->getI1Type(), solidityBool);
				}
				else
				{
					// No condition means infinite loop (true)
					condValue = m_builder->create<mlir::arith::ConstantIntOp>(loc, 1, 1);
				}

				// Pass all loop-carried values to the after region
				std::vector<mlir::Value> blockArgs;
				for (unsigned i = 0; i < beforeBlock->getNumArguments(); ++i)
					blockArgs.push_back(beforeBlock->getArgument(i));

				m_builder->create<mlir::scf::ConditionOp>(loc, condValue, mlir::ValueRange{blockArgs});
			}

			// Build the "after" region (body and update)
			{
				auto& afterRegion = whileOp.getAfter();
				auto* afterBlock = m_builder->createBlock(&afterRegion);

				// Add block arguments for loop-carried values
				for (auto type: loopCarriedTypes)
					afterBlock->addArgument(type, loc);

				m_builder->setInsertionPointToEnd(afterBlock);

				// Map block arguments to variables
				for (size_t i = 0; i < loopCarriedVarIdsList.size(); ++i)
				{
					m_valueMap[loopCarriedVarIdsList[i]] = afterBlock->getArgument(i);
				}

				// Generate the loop body
				generateSolidityStatement(forStmt->body());

				// Generate the update expression only if we haven't already terminated
				if (afterBlock->empty() || !afterBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
				{
					if (forStmt->loopExpression())
					{
						generateSolidityStatement(*forStmt->loopExpression());
					}
				}

				// Only add scf::YieldOp if the block doesn't already have a terminator
				// (e.g., from a break statement in the loop body)
				if (afterBlock->empty() || !afterBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
				{
					m_builder->create<mlir::scf::YieldOp>(
						loc, mlir::ValueRange{loopYieldValues(afterBlock, loopCarriedVarIdsList)});
				}
			}

			// Update variable map with final values
			for (size_t i = 0; i < loopCarriedVarIdsList.size(); ++i)
			{
				if (i < whileOp->getNumResults())
					m_valueMap[loopCarriedVarIdsList[i]] = whileOp->getResult(i);
			}

			// Reset insertion point after the loop
			m_builder->setInsertionPointAfter(whileOp);

			// Anything the loop assigned but did not yield is gone once the
			// loop ends, so it must not stay in the map as a live value.
			dropValuesEscaping(whileOp, loc);

			// Clear loop-carried variables tracking
			m_loopCarriedVarIds.clear();
			m_immediateLoopContext = LoopContext::None; // Exiting SCF loop

			// Return the first result if available
			if (whileOp->getNumResults() > 0)
				return whileOp->getResult(0);
		}
		else if (auto* whileStmt = dynamic_cast<WhileStatement const*>(&_stmt))
		{
			// Check if this is a do-while loop
			if (whileStmt->isDoWhile())
			{
				// Implement do-while using SCF while with an initial true condition
				// do { body } while (condition) becomes:
				// scf.while (true) { body; if (!condition) break; }

				// Analyze the loop body for modified variables
				auto bodyModifiedVars = collectModifiedVariables(whileStmt->body());
				std::set<int64_t> loopCarriedVarIds;
				loopCarriedVarIds.insert(bodyModifiedVars.begin(), bodyModifiedVars.end());

				// Store loop-carried variables for break/continue handling
				m_loopCarriedVarIds = loopCarriedVarIds;
				m_immediateLoopContext = LoopContext::SCF; // We're in an SCF loop

				// Prepare initial values and types
				std::vector<mlir::Value> initialValues;
				std::vector<mlir::Type> loopCarriedTypes;
				std::vector<int64_t> loopCarriedVarIdsList;

				for (auto varId: loopCarriedVarIds)
				{
					if (m_valueMap.count(varId) > 0)
					{
						initialValues.push_back(m_valueMap[varId]);
						loopCarriedTypes.push_back(m_valueMap[varId].getType());
						loopCarriedVarIdsList.push_back(varId);
					}
				}

				// If no loop-carried values, add a dummy value
				if (initialValues.empty())
				{
					auto dummyValue = m_builder->create<mlir::solidity::ConstantOp>(
						loc, m_builder->getI64IntegerAttr(0), mlir::solidity::UIntType::get(m_context.get(), 256));
					initialValues.push_back(dummyValue);
					loopCarriedTypes.push_back(mlir::solidity::UIntType::get(m_context.get(), 256));
				}

				// Create the SCF while operation
				auto whileOp = m_builder->create<mlir::scf::WhileOp>(loc, loopCarriedTypes, initialValues);

				// Build the "before" region (condition check - always true for first iteration)
				auto& beforeRegion = whileOp.getBefore();
				auto* beforeBlock = m_builder->createBlock(&beforeRegion);
				// Add block arguments for loop-carried values
				for (auto type: loopCarriedTypes)
				{
					beforeBlock->addArgument(type, loc);
				}
				m_builder->setInsertionPointToEnd(beforeBlock);

				// Always continue for do-while (condition checked at end)
				auto trueVal = m_builder->create<mlir::arith::ConstantIntOp>(loc, 1, 1);
				m_builder->create<mlir::scf::ConditionOp>(loc, trueVal, beforeBlock->getArguments());

				// Build the "after" region (loop body)
				auto& afterRegion = whileOp.getAfter();
				auto* afterBlock = m_builder->createBlock(&afterRegion);
				// Add block arguments for loop-carried values
				for (auto type: loopCarriedTypes)
				{
					afterBlock->addArgument(type, loc);
				}
				m_builder->setInsertionPointToEnd(afterBlock);

				// Update value map with block arguments
				for (size_t i = 0; i < loopCarriedVarIdsList.size(); ++i)
				{
					m_valueMap[loopCarriedVarIdsList[i]] = afterBlock->getArgument(i);
				}

				// Generate the loop body
				generateSolidityStatement(whileStmt->body());

				// Generate condition check at the end (for do-while)
				auto condValue = generateSolidityExpression(whileStmt->condition());

				// After generating the body and condition, ensure proper termination
				// Check if the current block needs a terminator
				auto* currentBlock = m_builder->getBlock();
				bool needsTerminator
					= currentBlock
					  && (currentBlock->empty() || !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>());

				if (needsTerminator)
				{
					m_builder->create<mlir::scf::YieldOp>(
						loc, mlir::ValueRange{loopYieldValues(afterBlock, loopCarriedVarIdsList)});
				}

				// Ensure all blocks in the after region have terminators
				// This is critical for SCF while loops
				for (auto& block: whileOp.getAfter())
				{
					if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>())
					{
						// Add a default yield to this block
						m_builder->setInsertionPointToEnd(&block);
						std::vector<mlir::Value> defaultValues;
						// Use the block's own arguments as the yielded values
						for (unsigned i = 0; i < block.getNumArguments(); ++i)
						{
							defaultValues.push_back(block.getArgument(i));
						}
						// If no arguments, use the original loop-carried values
						if (defaultValues.empty() && !initialValues.empty())
						{
							defaultValues = {initialValues[0]};
						}
						m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{defaultValues});
					}
				}

				// Clear loop-carried variables tracking
				m_loopCarriedVarIds.clear();
				m_immediateLoopContext = LoopContext::None; // Exiting SCF loop

				// IMPORTANT: Set insertion point after the while operation
				// This ensures subsequent operations aren't added to the loop body
				m_builder->setInsertionPointAfter(whileOp.getOperation());

				// Map the loop result to the loop-carried variables
				// This allows subsequent code to reference the final values
				if (whileOp->getNumResults() > 0 && !loopCarriedVarIdsList.empty())
				{
					m_valueMap[loopCarriedVarIdsList[0]] = whileOp->getResult(0);
					return whileOp->getResult(0);
				}
				else if (whileOp->getNumResults() > 0)
				{
					return whileOp->getResult(0);
				}

				return nullptr;
			}

			// Properly use scf.while with loop-carried values
			// Analyze the loop to find all modified variables

			// Collect all variables modified in the loop body
			auto bodyModifiedVars = collectModifiedVariables(whileStmt->body());

			// Also check condition for referenced variables
			std::set<int64_t> loopCarriedVarIds;
			{
				auto condReferencedVars = collectReferencedVariables(whileStmt->condition());
				// Add variables that are both referenced and modified
				for (auto varId: condReferencedVars)
				{
					if (bodyModifiedVars.count(varId) > 0)
						loopCarriedVarIds.insert(varId);
				}
			}

			// Add all body-modified variables
			loopCarriedVarIds.insert(bodyModifiedVars.begin(), bodyModifiedVars.end());

			// Store loop-carried variables for break/continue handling
			m_loopCarriedVarIds = loopCarriedVarIds;
			m_immediateLoopContext = LoopContext::SCF; // We're in an SCF loop

			// Prepare initial values and types for loop-carried variables
			std::vector<mlir::Value> initialValues;
			std::vector<mlir::Type> loopCarriedTypes;
			std::vector<int64_t> loopCarriedVarIdsList; // Keep ordered list

			for (auto varId: loopCarriedVarIds)
			{
				if (m_valueMap.count(varId) > 0)
				{
					initialValues.push_back(m_valueMap[varId]);
					loopCarriedTypes.push_back(m_valueMap[varId].getType());
					loopCarriedVarIdsList.push_back(varId);
				}
			}

			// If no loop-carried values, create a dummy one to satisfy SCF requirements
			if (initialValues.empty())
			{
				auto zeroValue = m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getI64IntegerAttr(0), mlir::solidity::UIntType::get(m_context.get(), 256));
				initialValues.push_back(zeroValue);
				loopCarriedTypes.push_back(mlir::solidity::UIntType::get(m_context.get(), 256));
			}

			// Create the scf.while with proper loop-carried values
			auto whileOp = m_builder->create<
				mlir::scf::WhileOp>(loc, mlir::TypeRange{loopCarriedTypes}, mlir::ValueRange{initialValues});

			// Build the "before" region (condition check)
			{
				auto& beforeRegion = whileOp.getBefore();
				auto* beforeBlock = m_builder->createBlock(&beforeRegion);

				// Add block arguments for loop-carried values
				for (auto type: loopCarriedTypes)
					beforeBlock->addArgument(type, loc);

				m_builder->setInsertionPointToEnd(beforeBlock);

				// Map block arguments to variables
				for (size_t i = 0; i < loopCarriedVarIdsList.size(); ++i)
				{
					m_valueMap[loopCarriedVarIdsList[i]] = beforeBlock->getArgument(i);
				}

				// Generate the condition (while always has a condition)
				auto solidityBool = generateSolidityExpression(whileStmt->condition());
				// Convert solidity.bool to i1 for SCF
				auto condValue = m_builder->create<mlir::solidity::ToI1Op>(
					loc, m_builder->getI1Type(), solidityBool);

				// Pass all loop-carried values to the after region
				std::vector<mlir::Value> blockArgs;
				for (unsigned i = 0; i < beforeBlock->getNumArguments(); ++i)
					blockArgs.push_back(beforeBlock->getArgument(i));

				m_builder->create<mlir::scf::ConditionOp>(loc, condValue, mlir::ValueRange{blockArgs});
			}

			// Build the "after" region (loop body)
			{
				auto& afterRegion = whileOp.getAfter();
				auto* afterBlock = m_builder->createBlock(&afterRegion);

				// Add block arguments for loop-carried values
				for (auto type: loopCarriedTypes)
					afterBlock->addArgument(type, loc);

				m_builder->setInsertionPointToEnd(afterBlock);

				// Map block arguments to variables
				for (size_t i = 0; i < loopCarriedVarIdsList.size(); ++i)
				{
					m_valueMap[loopCarriedVarIdsList[i]] = afterBlock->getArgument(i);
				}

				// Generate the loop body
				generateSolidityStatement(whileStmt->body());

				// Only add scf::YieldOp if the block doesn't already have a terminator
				// (e.g., from a break statement in the loop body)
				if (afterBlock->empty() || !afterBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
				{
					m_builder->create<mlir::scf::YieldOp>(
						loc, mlir::ValueRange{loopYieldValues(afterBlock, loopCarriedVarIdsList)});
				}
			}

			// Update variable map with final values
			for (size_t i = 0; i < loopCarriedVarIdsList.size(); ++i)
			{
				if (i < whileOp->getNumResults())
					m_valueMap[loopCarriedVarIdsList[i]] = whileOp->getResult(i);
			}

			// Reset insertion point after the loop
			m_builder->setInsertionPointAfter(whileOp);

			// Anything the loop assigned but did not yield is gone once the
			// loop ends, so it must not stay in the map as a live value.
			dropValuesEscaping(whileOp, loc);

			// Clear loop-carried variables tracking
			m_loopCarriedVarIds.clear();
			m_immediateLoopContext = LoopContext::None; // Exiting SCF loop

			// Return the first result if available
			if (whileOp->getNumResults() > 0)
				return whileOp->getResult(0);
		}
		else if (auto* ret = dynamic_cast<Return const*>(&_stmt))
		{
			llvm::SmallVector<mlir::Value, 4> values;
			if (ret->expression())
				if (mlir::Value value = generateSolidityExpression(*ret->expression()))
					values.push_back(value);

			emitReturn(loc, values);
		}
		else if (auto* breakStmt = dynamic_cast<Break const*>(&_stmt))
		{
			// Only generate scf::YieldOp if we're in an immediate SCF loop
			if (m_immediateLoopContext == LoopContext::SCF && immediateLoopBody())
			{
				// For break statements in SCF loops, we need to yield with current values
				// This is a simplified handling - proper implementation would track loop contexts
				// and branch to the appropriate exit block

				// The yield has to match the region's arity exactly. A dummy
				// zero when there was nothing to carry, or a short list when a
				// carried variable was untouched, is what the verifier rejected.
				std::vector<int64_t> const carried(m_loopCarriedVarIds.begin(), m_loopCarriedVarIds.end());
				m_builder->create<mlir::scf::YieldOp>(
					loc, mlir::ValueRange{loopYieldValues(immediateLoopBody(), carried)});
			}
			else
			{
				// Not in SCF context, generate a solidity.break operation
				m_builder->create<mlir::solidity::BreakOp>(loc);
			}
			return nullptr; // Return null to indicate we've handled the terminator
		}
		else if (auto* continueStmt = dynamic_cast<Continue const*>(&_stmt))
		{
			// Only generate scf::YieldOp if we're in an immediate SCF loop
			if (m_immediateLoopContext == LoopContext::SCF && immediateLoopBody())
			{
				// For continue statements, we need to yield with updated values
				// to continue to the next iteration
				std::vector<int64_t> const carried(m_loopCarriedVarIds.begin(), m_loopCarriedVarIds.end());
				m_builder->create<mlir::scf::YieldOp>(
					loc, mlir::ValueRange{loopYieldValues(immediateLoopBody(), carried)});
			}
			else
			{
				// Not in SCF context, generate a solidity.continue operation
				m_builder->create<mlir::solidity::ContinueOp>(loc);
			}
			return nullptr; // Return null to indicate we've handled the terminator
		}
		else if (auto* revertStmt = dynamic_cast<RevertStatement const*>(&_stmt))
		{
			// Handle revert statement with custom error
			mlir::StringAttr reasonAttr;
			// Get the error name from the function call if possible
			if (auto* ident = dynamic_cast<Identifier const*>(&revertStmt->errorCall().expression()))
			{
				reasonAttr = m_builder->getStringAttr(ident->name());
			}
			m_builder->create<mlir::solidity::RevertOp>(loc, reasonAttr);
			return nullptr; // Revert is a terminator
		}
		else if (auto* emitStmt = dynamic_cast<EmitStatement const*>(&_stmt))
		{
			// Handle emit statement
			auto const& eventCall = emitStmt->eventCall();
			std::string eventName;

			// Get event name from function call expression
			if (auto* ident = dynamic_cast<Identifier const*>(&eventCall.expression()))
			{
				eventName = ident->name();
			}
			else if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&eventCall.expression()))
			{
				eventName = memberAccess->memberName();
			}

			// Build the full event signature string for computing topic0
			// e.g., "Transfer(address,address,uint256)"
			std::string eventSignature = eventName + "(";
			auto const* exprType = eventCall.expression().annotation().type;
			if (auto const* funcType = dynamic_cast<FunctionType const*>(exprType))
			{
				auto const& paramTypes = funcType->parameterTypes();
				for (size_t i = 0; i < paramTypes.size(); ++i)
				{
					if (i > 0)
						eventSignature += ",";
					eventSignature += paramTypes[i]->canonicalName();
				}
			}
			eventSignature += ")";

			// Determine which parameters are indexed
			std::vector<bool> indexed;
			if (auto const* funcType = dynamic_cast<FunctionType const*>(exprType))
			{
				// Try to get the event definition to find indexed parameters
				if (auto const* eventDef = dynamic_cast<EventDefinition const*>(&funcType->declaration()))
				{
					for (auto const& param: eventDef->parameters())
						indexed.push_back(param->isIndexed());
				}
			}

			// Generate arguments for the event
			std::vector<mlir::Value> args;
			for (auto const& arg: eventCall.arguments())
			{
				auto argValue = generateSolidityExpression(*arg);
				if (argValue)
					args.push_back(argValue);
			}

			// Create emit operation with full event signature
			mlir::ArrayAttr indexedArrayAttr;
			if (!indexed.empty())
			{
				std::vector<mlir::Attribute> indexedAttrs;
				for (bool isIndexed: indexed)
					indexedAttrs.push_back(m_builder->getBoolAttr(isIndexed));
				indexedArrayAttr = m_builder->getArrayAttr(indexedAttrs);
			}

			m_builder->create<mlir::solidity::EmitOp>(
				loc, m_builder->getStringAttr(eventName),
				m_builder->getStringAttr(eventSignature),
				indexedArrayAttr,
				args);
			return nullptr;
		}
		else if (auto* exprStmt = dynamic_cast<ExpressionStatement const*>(&_stmt))
		{
			return generateSolidityExpression(exprStmt->expression());
		}
		else if (auto* varDeclStmt = dynamic_cast<VariableDeclarationStatement const*>(&_stmt))
		{
			if (varDeclStmt->initialValue())
			{
				auto const& declarations = varDeclStmt->declarations();

				// Check if this is tuple destructuring (multiple declarations)
				// For cases like: (bool ok, ) = msg.sender.call{value: bal}("")
				if (declarations.size() > 1)
				{
					// Generate the init expression (e.g., the low-level call)
					auto initValue = generateSolidityExpression(*varDeclStmt->initialValue());

					// One result per component. Binding the first and zeroing the
					// rest read `(a, b) = pair()` as `a = pair(); b = 0`, which
					// compiles and answers wrongly - the worst of the two. Where
					// the initialiser genuinely produces no such result, say so
					// with a placeholder rather than a zero that looks like data.
					mlir::Operation* producer = initValue ? initValue.getDefiningOp() : nullptr;
					for (size_t i = 0; i < declarations.size(); ++i)
					{
						auto const& decl = declarations[i];
						if (!decl)
							// An omitted component still occupies its position.
							continue;

						auto declType = translateSolidityType(*decl->type());
						auto declLoc = this->loc(*decl);

						if (producer && i < producer->getNumResults() && producer->getResult(i).getType() == declType)
							m_valueMap[decl->id()] = producer->getResult(i);
						else if (i == 0 && initValue)
							m_valueMap[decl->id()] = initValue;
						else
							m_valueMap[decl->id()] = emitUnsupported(
								declLoc, declType, "component of a destructuring its initialiser does not produce");
					}
					return initValue ? initValue : mlir::Value();
				}

				// Single declaration.
				auto value = generateSolidityExpression(*varDeclStmt->initialValue());
				if (!value)
				{
					auto loc = this->loc(*varDeclStmt);
					auto type = translateSolidityType(*varDeclStmt->initialValue()->annotation().type);
					value = emitUnsupported(loc, type, "variable declaration with unsupported initializer");
				}
				for (auto const& decl: declarations)
				{
					if (decl)
						m_valueMap[decl->id()] = value;
				}
				return value;
			}
			else
			{
				// No initializer — zero-initialize all declared variables
				for (auto const& decl: varDeclStmt->declarations())
				{
					if (decl)
					{
						auto declType = translateSolidityType(*decl->type());
						auto declLoc = this->loc(*decl);
						m_valueMap[decl->id()] = m_builder->create<mlir::solidity::ConstantOp>(
							declLoc, m_builder->getIntegerAttr(m_builder->getI64Type(), 0), declType);
					}
				}
			}
		}
		else if (auto* inlineAsm = dynamic_cast<InlineAssembly const*>(&_stmt))
		{
			auto loc = this->loc(_stmt);
			// Serialize the Yul block to text using AsmPrinter
			yul::AsmPrinter printer(inlineAsm->dialect());
			std::string yulSource = printer(inlineAsm->operations().root());

			// Create the InlineAssemblyOp with serialized Yul source
			m_builder->create<mlir::solidity::InlineAssemblyOp>(
				loc, m_builder->getStringAttr(yulSource));

			// For external references pointing to Solidity variables,
			// create AssemblyBindOp so subsequent code can reference them.
			// This handles patterns like: address proxy; assembly { proxy := create2(...) }
			for (auto const& [yulIdent, info]: inlineAsm->annotation().externalReferences)
			{
				if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(info.declaration))
				{
					auto varType = translateSolidityType(*varDecl->type());
					auto bindOp = m_builder->create<mlir::solidity::AssemblyBindOp>(
						loc, varType, m_builder->getStringAttr(yulIdent->name.str()));
					m_valueMap[varDecl->id()] = bindOp.getResult();
				}
			}
		}
		else
		{
			auto loc = this->loc(_stmt);
			auto dummyType = mlir::solidity::UIntType::get(m_context.get(), 256);
			emitUnsupported(loc, dummyType, "unsupported statement: " + demangle(typeid(_stmt).name()));
		}
		return mlir::Value();
	}

	mlir::Type translateSolidityType(Type const& _type)
	{
		if (auto* intType = dynamic_cast<IntegerType const*>(&_type))
		{
			if (intType->isSigned())
				return mlir::solidity::IntType::get(m_context.get(), intType->numBits());
			else
				return mlir::solidity::UIntType::get(m_context.get(), intType->numBits());
		}
		else if (dynamic_cast<BoolType const*>(&_type))
		{
			return mlir::solidity::BoolType::get(m_context.get());
		}
		else if (dynamic_cast<AddressType const*>(&_type))
		{
			return mlir::solidity::AddressType::get(m_context.get());
		}
		else if (auto* bytesType = dynamic_cast<FixedBytesType const*>(&_type))
		{
			return mlir::solidity::BytesType::get(m_context.get(), bytesType->numBytes());
		}
		else if (auto* arrayType = dynamic_cast<ArrayType const*>(&_type))
		{
			// string and bytes are special ArrayType sub-kinds
			if (arrayType->isString())
				return mlir::solidity::StringType::get(m_context.get());
			if (arrayType->isByteArray())
				return mlir::solidity::DynamicBytesType::get(m_context.get());

			auto elemType = translateSolidityType(*arrayType->baseType());
			if (arrayType->isDynamicallySized())
				return mlir::solidity::ArrayType::get(elemType, -1);
			else
				return mlir::solidity::ArrayType::get(elemType, static_cast<int64_t>(arrayType->length()));
		}
		else if (auto* mappingType = dynamic_cast<MappingType const*>(&_type))
		{
			// For now, represent mappings as uint256 since we don't have a mapping type in MLIR dialect yet
			// TODO: Add proper MappingType to MLIR dialect
			return mlir::solidity::UIntType::get(m_context.get(), 256);
		}
		else if (auto* structType = dynamic_cast<StructType const*>(&_type))
		{
			// For now, represent structs as uint256 since we don't have a struct type in MLIR dialect yet
			// TODO: Add proper StructType to MLIR dialect
			return mlir::solidity::UIntType::get(m_context.get(), 256);
		}
		else if (dynamic_cast<StringLiteralType const*>(&_type))
		{
			return mlir::solidity::StringType::get(m_context.get());
		}
		else if (dynamic_cast<ContractType const*>(&_type))
		{
			// Contract/interface types are addresses on the ABI level
			return mlir::solidity::AddressType::get(m_context.get());
		}

		// Default to uint256
		return mlir::solidity::UIntType::get(m_context.get(), 256);
	}

	void generateStandardFunction(FunctionDefinition const& _func)
	{
		auto loc = this->loc(_func);

		// Build function type using standard MLIR types
		std::vector<mlir::Type> inputTypes;
		std::vector<mlir::Type> resultTypes;

		// Use simple types for demonstration
		for (auto const& param: _func.parameters())
		{
			inputTypes.push_back(m_builder->getI64Type()); // Simplified to i64
		}

		for (auto const& ret: _func.returnParameters())
		{
			resultTypes.push_back(m_builder->getI64Type()); // Simplified to i64
		}

		auto funcType = m_builder->getFunctionType(inputTypes, resultTypes);

		// Create function using standard func dialect
		auto funcOp = m_builder->create<mlir::func::FuncOp>(loc, _func.name(), funcType);

		// Create entry block
		auto& entryBlock = funcOp.getBody().emplaceBlock();

		// Add block arguments
		for (auto inputType: inputTypes)
		{
			entryBlock.addArgument(inputType, loc);
		}

		// Set insertion point to function body
		m_builder->setInsertionPointToEnd(&entryBlock);

		// Add a simple return
		if (resultTypes.empty())
		{
			m_builder->create<mlir::func::ReturnOp>(loc);
		}
		else
		{
			// Create dummy constants for return values
			std::vector<mlir::Value> returnVals;
			for (auto resultType: resultTypes)
			{
				auto constant
					= m_builder
						  ->create<mlir::arith::ConstantOp>(loc, resultType, m_builder->getIntegerAttr(resultType, 42));
				returnVals.push_back(constant);
			}
			m_builder->create<mlir::func::ReturnOp>(loc, returnVals);
		}
	}

	void generateFunction(FunctionDefinition const& _func)
	{
		auto loc = this->loc(_func);

		// Build function type
		std::vector<mlir::Type> paramTypes;
		for (auto const& param: _func.parameters())
		{
			paramTypes.push_back(translateType(*param->type()));
		}

		std::vector<mlir::Type> returnTypes;
		for (auto const& ret: _func.returnParameters())
		{
			returnTypes.push_back(translateType(*ret->type()));
		}

		auto funcType = mlir::FunctionType::get(m_context.get(), paramTypes, returnTypes);

		// Determine visibility and mutability
		// Note: constructors cannot call defaultVisibility() — handle them specially
		std::string visibility = "public";
		if (!_func.isConstructor())
		{
			if (_func.visibility() == Visibility::Private)
				visibility = "private";
			else if (_func.visibility() == Visibility::Internal)
				visibility = "internal";
			else if (_func.visibility() == Visibility::External)
				visibility = "external";
		}

		std::string mutability = "nonpayable";
		if (_func.stateMutability() == StateMutability::Pure)
			mutability = "pure";
		else if (_func.stateMutability() == StateMutability::View)
			mutability = "view";
		else if (_func.stateMutability() == StateMutability::Payable)
			mutability = "payable";

		// Create function operation
		// Use standard func dialect instead of custom solidity operations
		auto funcOp = m_builder->create<mlir::func::FuncOp>(loc, _func.name(), funcType);

		// Create entry block with arguments
		auto& entryBlock = funcOp.getBody().emplaceBlock();

		// Add block arguments for parameters
		for (size_t i = 0; i < paramTypes.size(); ++i)
		{
			entryBlock.addArgument(paramTypes[i], loc);
		}

		// Enter new scope for function
		SymbolTableScopeT functionScope(m_symbolTable);

		// Map function parameters to block arguments
		auto params = _func.parameters();
		for (size_t i = 0; i < params.size(); ++i)
		{
			// Insert parameter into symbol table with proper scoping
			m_symbolTable.insert(params[i]->name(), entryBlock.getArgument(i));
			m_valueMap[params[i]->id()] = entryBlock.getArgument(i);
		}

		// Save current insertion point and switch to function body
		auto savedIP = m_builder->saveInsertionPoint();
		m_builder->setInsertionPointToEnd(&entryBlock);

		// Generate function body
		if (_func.isImplemented())
		{
			generateStatement(_func.body());
		}

		// Add return if not already present
		if (entryBlock.empty() || !entryBlock.back().hasTrait<mlir::OpTrait::IsTerminator>())
		{
			m_builder->create<mlir::func::ReturnOp>(loc);
		}

		// Restore insertion point
		m_builder->restoreInsertionPoint(savedIP);
	}

	mlir::Value generateExpression(Expression const& _expr)
	{
		auto loc = this->loc(_expr);

		if (auto* literal = dynamic_cast<Literal const*>(&_expr))
		{
			auto type = translateType(*_expr.annotation().type);

			if (literal->token() == langutil::Token::Number)
			{
				// Use the type's literalValue() to properly parse all numeric formats
				// (decimal, hex, underscores, etc.) and handle values up to 256 bits
				u256 bigValue = _expr.annotation().type->literalValue(literal);
				auto attr = m_builder->getIntegerAttr(
					mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
					llvm::APInt(256, bigValue.str(), 10));
				return m_builder->create<mlir::arith::ConstantOp>(loc, attr).getResult();
			}
			else if (literal->token() == langutil::Token::TrueLiteral)
			{
				auto attr = m_builder->getBoolAttr(true);
				return m_builder->create<mlir::arith::ConstantOp>(loc, attr).getResult();
			}
			else if (literal->token() == langutil::Token::FalseLiteral)
			{
				auto attr = m_builder->getBoolAttr(false);
				return m_builder->create<mlir::arith::ConstantOp>(loc, attr).getResult();
			}
		}
		else if (auto* ident = dynamic_cast<Identifier const*>(&_expr))
		{
			if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
			{
				if (varDecl->isStateVariable())
				{
					auto type = translateType(*_expr.annotation().type);
					// TODO: Implement state variable loading with standard MLIR
					// For now, return a constant zero as placeholder
					auto zeroAttr = m_builder->getIntegerAttr(type, 0);
					return m_builder->create<mlir::arith::ConstantOp>(loc, zeroAttr).getResult();
				}
				else if (m_valueMap.count(varDecl->id()))
				{
					return m_valueMap[varDecl->id()];
				}
				else
				{
					// Variable not found - create a zero value as fallback
					auto type = translateType(*_expr.annotation().type);
					auto zeroAttr = m_builder->getIntegerAttr(type, 0);
					auto zeroValue = m_builder->create<mlir::arith::ConstantOp>(loc, zeroAttr).getResult();
					// Store it for future reference
					m_valueMap[varDecl->id()] = zeroValue;
					return zeroValue;
				}
			}
		}
		else if (auto* binOp = dynamic_cast<BinaryOperation const*>(&_expr))
		{
			auto lhs = generateExpression(binOp->leftExpression());
			auto rhs = generateExpression(binOp->rightExpression());

			switch (binOp->getOperator())
			{
			case langutil::Token::Add:
				return m_builder->create<mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
			case langutil::Token::Sub:
				return m_builder->create<mlir::arith::SubIOp>(loc, lhs, rhs).getResult();
			case langutil::Token::Mul:
				return m_builder->create<mlir::arith::MulIOp>(loc, lhs, rhs).getResult();
			case langutil::Token::Div:
				return m_builder->create<mlir::arith::DivUIOp>(loc, lhs, rhs)
					.getResult(); // Using unsigned division as default
			case langutil::Token::Mod:
				return m_builder->create<mlir::arith::RemUIOp>(loc, lhs, rhs)
					.getResult(); // Using unsigned remainder as default
			case langutil::Token::Exp:
				// TODO: Implement exponentiation using standard MLIR operations
				// For now, return lhs as placeholder
				return lhs;
			case langutil::Token::BitAnd:
				return m_builder->create<mlir::arith::AndIOp>(loc, lhs, rhs).getResult();
			case langutil::Token::BitOr:
				return m_builder->create<mlir::arith::OrIOp>(loc, lhs, rhs).getResult();
			case langutil::Token::BitXor:
				return m_builder->create<mlir::arith::XOrIOp>(loc, lhs, rhs).getResult();
			case langutil::Token::SHL:
				return m_builder->create<mlir::arith::ShLIOp>(loc, lhs, rhs).getResult();
			case langutil::Token::SHR:
				return m_builder->create<mlir::arith::ShRUIOp>(loc, lhs, rhs).getResult(); // Logical right shift
			case langutil::Token::SAR:
				return m_builder->create<mlir::arith::ShRSIOp>(loc, lhs, rhs).getResult(); // Arithmetic right shift
			case langutil::Token::LessThan:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, lhs, rhs)
					.getResult();
			case langutil::Token::GreaterThan:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs)
					.getResult();
			case langutil::Token::LessThanOrEqual:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, lhs, rhs)
					.getResult();
			case langutil::Token::GreaterThanOrEqual:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, lhs, rhs)
					.getResult();
			case langutil::Token::Equal:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhs, rhs)
					.getResult();
			case langutil::Token::NotEqual:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhs, rhs)
					.getResult();
			default:
				break;
			}
		}
		else if (auto* assignment = dynamic_cast<Assignment const*>(&_expr))
		{
			auto value = generateExpression(assignment->rightHandSide());

			if (auto* ident = dynamic_cast<Identifier const*>(&assignment->leftHandSide()))
			{
				if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
				{
					if (varDecl->isStateVariable())
					{
						// TODO: Implement state variable storing with standard MLIR
						// For now, skip the operation
					}
					else
					{
						m_valueMap[varDecl->id()] = value;
					}
				}
				else if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&assignment->leftHandSide()))
				{
					auto base = generateExpression(indexAccess->baseExpression());
					auto index = generateExpression(*indexAccess->indexExpression());

					// Check if it's an array or mapping based on base type
					if (dynamic_cast<ArrayType const*>(indexAccess->baseExpression().annotation().type))
					{
						// TODO: Implement array store with standard MLIR operations
					}
					else if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
					{
						// TODO: Implement mapping store with standard MLIR operations
					}
				}
			}

			return value;
		}
		else if (auto* unaryOp = dynamic_cast<UnaryOperation const*>(&_expr))
		{
			auto operand = generateExpression(unaryOp->subExpression());

			switch (unaryOp->getOperator())
			{
			case langutil::Token::BitNot:
				// TODO: Implement logical NOT with standard MLIR
				return operand; // Placeholder
			case langutil::Token::Inc:
			{
				auto one
					= m_builder->create<mlir::arith::ConstantOp>(loc, m_builder->getIntegerAttr(operand.getType(), 1))
						  .getResult();
				return m_builder->create<mlir::arith::AddIOp>(loc, operand, one).getResult();
			}
			case langutil::Token::Dec:
			{
				auto one
					= m_builder->create<mlir::arith::ConstantOp>(loc, m_builder->getIntegerAttr(operand.getType(), 1))
						  .getResult();
				return m_builder->create<mlir::arith::SubIOp>(loc, operand, one).getResult();
			}
			default:
				break;
			}
		}
		else if (auto* funcCall = dynamic_cast<FunctionCall const*>(&_expr))
		{
			if (auto* ident = dynamic_cast<Identifier const*>(&funcCall->expression()))
			{
				if (auto* funcDef = dynamic_cast<FunctionDefinition const*>(ident->annotation().referencedDeclaration))
				{
					// Collect arguments
					std::vector<mlir::Value> args;
					for (auto const& arg: funcCall->arguments())
					{
						if (arg)
							args.push_back(generateExpression(*arg));
					}

					// Determine result types
					std::vector<mlir::Type> resultTypes;
					for (auto const& ret: funcDef->returnParameters())
					{
						resultTypes.push_back(translateType(*ret->type()));
					}

					auto dummyType = translateType(*_expr.annotation().type);
					return emitUnsupported(loc, dummyType, "function call '" + ident->name() + "()'");
				}
				// Handle special functions like require, assert, revert
				else if (ident->name() == "require")
				{
					if (!funcCall->arguments().empty())
					{
						auto cond = generateExpression(*funcCall->arguments()[0]);
						std::string msg;
						if (funcCall->arguments().size() > 1)
						{
							if (auto* literal = dynamic_cast<Literal const*>(funcCall->arguments()[1].get()))
								msg = literal->value();
						}
						// TODO: Implement require with standard MLIR operations
					}
				}
				else if (ident->name() == "assert")
				{
					if (!funcCall->arguments().empty())
					{
						auto cond = generateExpression(*funcCall->arguments()[0]);
						// TODO: Implement assert with standard MLIR operations
					}
				}
				else if (ident->name() == "revert")
				{
					std::string reason;
					if (!funcCall->arguments().empty())
					{
						if (auto* literal = dynamic_cast<Literal const*>(funcCall->arguments()[0].get()))
							reason = literal->value();
					}
					// TODO: Implement revert with standard MLIR operations
				}
			}
		}
		else if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&_expr))
		{
			auto object = generateExpression(memberAccess->expression());
			std::string memberName = memberAccess->memberName();

			// Special case for array.length
			if (memberName == "length")
			{
				auto dummyType = translateType(*_expr.annotation().type);
				return emitUnsupported(loc, dummyType, "array.length access");
			}
			else
			{
				auto dummyType = translateType(*_expr.annotation().type);
				return emitUnsupported(loc, dummyType, "member access '." + memberName + "'");
			}
		}
		else if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&_expr))
		{
			auto base = generateExpression(indexAccess->baseExpression());
			auto index = generateExpression(*indexAccess->indexExpression());
			auto elementType = translateType(*_expr.annotation().type);

			// Check if it's an array or mapping based on base type
			if (dynamic_cast<ArrayType const*>(indexAccess->baseExpression().annotation().type))
			{
				return emitUnsupported(loc, elementType, "array element access");
			}
			else if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
			{
				return emitUnsupported(loc, elementType, "mapping access");
			}
		}

		// Unhandled expression type
		auto dummyType = translateType(*_expr.annotation().type);
		return emitUnsupported(loc, dummyType, "unsupported expression: " + demangle(typeid(_expr).name()));
	}

	void generateStatement(Statement const& _stmt)
	{
		auto loc = this->loc(_stmt);

		if (auto* block = dynamic_cast<Block const*>(&_stmt))
		{
			for (auto const& stmt: block->statements())
				generateStatement(*stmt);
		}
		else if (auto* ifStmt = dynamic_cast<IfStatement const*>(&_stmt))
		{
			auto condition = generateExpression(ifStmt->condition());

			// TODO: Implement if statements using standard MLIR control flow
			// For now, just generate the condition and then statement
			generateStatement(ifStmt->trueStatement());
			if (ifStmt->falseStatement())
				generateStatement(*ifStmt->falseStatement());

			// The simplified implementation already handled in the TODO above
		}
		else if (auto* forStmt = dynamic_cast<ForStatement const*>(&_stmt))
		{
			// TODO: Implement for loops using standard MLIR control flow
			// For now, just generate initialization and body
			if (forStmt->initializationExpression())
				generateStatement(*forStmt->initializationExpression());
			if (forStmt->condition())
				generateExpression(*forStmt->condition());
			generateStatement(forStmt->body());
			if (forStmt->loopExpression())
				generateStatement(*forStmt->loopExpression());
		}
		else if (auto* whileStmt = dynamic_cast<WhileStatement const*>(&_stmt))
		{
			// TODO: Implement while loops using standard MLIR control flow
			// For now, just generate condition and body
			generateExpression(whileStmt->condition());
			generateStatement(whileStmt->body());
		}
		// TODO: DoWhileStatement support would need to be added
		// else if (auto* doWhileStmt = dynamic_cast<DoWhileStatement const*>(&_stmt))
		// {
		// TODO: DoWhileStatement implementation commented out due to compilation issues
		// }
		else if (auto* ret = dynamic_cast<Return const*>(&_stmt))
		{
			if (ret->expression())
			{
				auto value = generateExpression(*ret->expression());
				m_builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{value});
			}
			else
			{
				m_builder->create<mlir::func::ReturnOp>(loc);
			}
		}
		else if (auto* exprStmt = dynamic_cast<ExpressionStatement const*>(&_stmt))
		{
			generateExpression(exprStmt->expression());
		}
		else if (auto* varDeclStmt = dynamic_cast<VariableDeclarationStatement const*>(&_stmt))
		{
			if (varDeclStmt->initialValue())
			{
				auto value = generateExpression(*varDeclStmt->initialValue());
				for (auto const& decl: varDeclStmt->declarations())
				{
					if (decl)
						m_valueMap[decl->id()] = value;
				}
			}
		}
		else
		{
			std::cerr << "Warning: unsupported statement type in MLIRGen: "
					  << demangle(typeid(_stmt).name()) << "\n";
		}
	}

	mlir::Type translateType(Type const& _type)
	{
		if (auto* intType = dynamic_cast<IntegerType const*>(&_type))
		{
			// Use standard MLIR integer types
			return m_builder->getIntegerType(intType->numBits());
		}
		else if (dynamic_cast<BoolType const*>(&_type))
		{
			return mlir::solidity::BoolType::get(m_context.get());
		}
		else if (dynamic_cast<AddressType const*>(&_type))
		{
			// Address is typically 160 bits in Ethereum
			return m_builder->getIntegerType(160);
		}
		else if (auto* bytesType = dynamic_cast<FixedBytesType const*>(&_type))
		{
			// Represent bytes as integer with appropriate bit width
			return m_builder->getIntegerType(bytesType->numBytes() * 8);
		}
		else if (dynamic_cast<StringLiteralType const*>(&_type))
		{
			// For now, represent strings as generic pointers
			return m_builder->getIndexType();
		}
		else if (auto* arrayType = dynamic_cast<ArrayType const*>(&_type))
		{
			// For now, represent arrays as generic pointers
			return m_builder->getIndexType();
		}

		// Default to 256-bit integer (common in Solidity)
		return m_builder->getIntegerType(256);
	}

#endif // SOLIDITY_HAS_MLIR

	std::string translateTypeString(Type const& _type)
	{
		if (auto* intType = dynamic_cast<IntegerType const*>(&_type))
		{
			if (intType->isSigned())
				return "!solidity.int<" + std::to_string(intType->numBits()) + ">";
			else
				return "!solidity.uint<" + std::to_string(intType->numBits()) + ">";
		}
		else if (dynamic_cast<BoolType const*>(&_type))
		{
			return "!solidity.bool";
		}
		else if (dynamic_cast<AddressType const*>(&_type))
		{
			return "!solidity.address";
		}
		else if (auto* bytesType = dynamic_cast<FixedBytesType const*>(&_type))
		{
			return "!solidity.bytes<" + std::to_string(bytesType->numBytes()) + ">";
		}
		else if (dynamic_cast<StringLiteralType const*>(&_type))
		{
			return "!solidity.string";
		}
		else if (auto* arrayType = dynamic_cast<ArrayType const*>(&_type))
		{
			std::string elemType = translateTypeString(*arrayType->baseType());
			if (arrayType->isDynamicallySized())
				return "!solidity.array<" + elemType + ", -1>";
			else
			{
				uint64_t length = static_cast<uint64_t>(arrayType->length());
				return "!solidity.array<" + elemType + ", " + std::to_string(length) + ">";
			}
		}

		// Default to uint256
		return "!solidity.uint<256>";
	}

private:
	[[maybe_unused]] CompilerStack const& m_compilerStack;
	[[maybe_unused]] langutil::EVMVersion m_evmVersion;
	[[maybe_unused]] OptimiserSettings const& m_optimiserSettings;

#ifdef SOLIDITY_HAS_MLIR
	std::unique_ptr<mlir::MLIRContext> m_context;
	std::unique_ptr<mlir::OpBuilder> m_builder;
	mlir::ModuleOp m_module;

	// Symbol table for proper scoping of variables (following low-lang patterns)
	llvm::ScopedHashTable<llvm::StringRef, mlir::Value> m_symbolTable;
	using SymbolTableScopeT = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

	// Legacy value map for compatibility
	std::map<int64_t, mlir::Value> m_valueMap;
	std::map<int64_t, mlir::Operation*> m_stateVarOpMap;

	// Map state variable names to their computed storage slots
	std::map<std::string, u256> m_stateVarSlots;

	// Track loop-carried variables for break/continue statements
	std::set<int64_t> m_loopCarriedVarIds;

	// Track the immediate loop context for break/continue handling
	enum class LoopContext
	{
		None,
		SCF,
		Solidity
	};
	LoopContext m_immediateLoopContext = LoopContext::None;

	// Track child contracts referenced by `new ContractName()`
	std::set<ContractDefinition const*> m_childContracts;

	// Counter for unsupported expression/statement drops
	size_t m_unsupportedCount = 0;

	/// Emit a warning about an unsupported pattern and return a zero-constant placeholder.
	/// This replaces silent drops so that every dropped expression is visible.
	mlir::Value emitUnsupported(mlir::Location _loc, mlir::Type _type, std::string const& _what)
	{
		m_unsupportedCount++;
		std::cerr << "Warning: [MLIR unsupported] " << _what << " (drop #" << m_unsupportedCount << ")\n";
		return m_builder->create<mlir::solidity::ConstantOp>(
			_loc, m_builder->getIntegerAttr(m_builder->getI64Type(), 0), _type);
	}

	/// Return the total number of unsupported drops encountered
	size_t unsupportedCount() const { return m_unsupportedCount; }
#endif
};

MLIRGenerator::MLIRGenerator(
	CompilerStack const& _compilerStack, langutil::EVMVersion _evmVersion, OptimiserSettings const& _optimiserSettings)
	: m_impl(std::make_unique<MLIRGeneratorImpl>(_compilerStack, _evmVersion, _optimiserSettings)),
	  m_compilerStack(_compilerStack), m_evmVersion(_evmVersion), m_optimiserSettings(_optimiserSettings)
{
}

MLIRGenerator::~MLIRGenerator() = default;

std::string MLIRGenerator::generate(ContractDefinition const& _contract)
{
	// Visit the contract to build the MLIR representation
	_contract.accept(*this);

	// Generate the MLIR module
	return m_impl->generateModule(_contract);
}

// ASTVisitor implementations
bool MLIRGenerator::visit(ContractDefinition const& /*_contract*/) { return true; }

void MLIRGenerator::endVisit(ContractDefinition const& /*_contract*/) {}

bool MLIRGenerator::visit(FunctionDefinition const& /*_function*/) { return true; }

void MLIRGenerator::endVisit(FunctionDefinition const& /*_function*/) {}

bool MLIRGenerator::visit(VariableDeclaration const& /*_variable*/) { return true; }

void MLIRGenerator::endVisit(VariableDeclaration const& /*_variable*/) {}

bool MLIRGenerator::visit(Block const& /*_block*/) { return true; }

void MLIRGenerator::endVisit(Block const& /*_block*/) {}

bool MLIRGenerator::visit(IfStatement const& /*_ifStatement*/) { return true; }

void MLIRGenerator::endVisit(IfStatement const& /*_ifStatement*/) {}

bool MLIRGenerator::visit(WhileStatement const& /*_whileStatement*/) { return true; }

void MLIRGenerator::endVisit(WhileStatement const& /*_whileStatement*/) {}

bool MLIRGenerator::visit(ForStatement const& /*_forStatement*/) { return true; }

void MLIRGenerator::endVisit(ForStatement const& /*_forStatement*/) {}

bool MLIRGenerator::visit(Return const& /*_return*/) { return true; }

void MLIRGenerator::endVisit(Return const& /*_return*/) {}

bool MLIRGenerator::visit(Assignment const& /*_assignment*/) { return true; }

void MLIRGenerator::endVisit(Assignment const& /*_assignment*/) {}

bool MLIRGenerator::visit(BinaryOperation const& /*_operation*/) { return true; }

void MLIRGenerator::endVisit(BinaryOperation const& /*_operation*/) {}

bool MLIRGenerator::visit(UnaryOperation const& /*_operation*/) { return true; }

void MLIRGenerator::endVisit(UnaryOperation const& /*_operation*/) {}

bool MLIRGenerator::visit(FunctionCall const& /*_functionCall*/) { return true; }

void MLIRGenerator::endVisit(FunctionCall const& /*_functionCall*/) {}

bool MLIRGenerator::visit(Identifier const& /*_identifier*/) { return true; }

void MLIRGenerator::endVisit(Identifier const& /*_identifier*/) {}

bool MLIRGenerator::visit(Literal const& /*_literal*/) { return true; }

void MLIRGenerator::endVisit(Literal const& /*_literal*/) {}

} // namespace solidity::frontend
