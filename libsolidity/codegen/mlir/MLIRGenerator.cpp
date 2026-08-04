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
#include <libsolidity/ast/ASTVisitor.h>
#include <libsolidity/ast/Types.h>
#include <libsolidity/ast/TypeProvider.h>
#include <libsolutil/Keccak256.h>
#include <libsolidity/codegen/mlir/MLIRGenerator.h>
#include <libsolidity/interface/CompilerStack.h>
#include <libsolutil/FunctionSelector.h>
#include <libyul/AST.h>
#include <libyul/AsmPrinter.h>
#include <libyul/backends/evm/EVMDialect.h>

#include <cxxabi.h>
#include <functional>
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
		OptimiserSettings const& _optimiserSettings,
		bool _legacyCodegen)
		: m_compilerStack(_compilerStack), m_evmVersion(_evmVersion),
		  m_optimiserSettings(_optimiserSettings), m_legacyCodegen(_legacyCodegen)
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
		if (m_legacyCodegen)
			contractOp->setAttr("legacy_codegen", m_builder->getUnitAttr());
		if (m_compilerStack.revertStringBehaviour() >= RevertStrings::Debug)
			contractOp->setAttr("revert_strings_debug", m_builder->getUnitAttr());
		if (m_evmVersion.hasBitwiseShifting())
			contractOp->setAttr("has_bitwise_shifting", m_builder->getUnitAttr());
		if (m_evmVersion.supportsReturndata())
			contractOp->setAttr("supports_returndata", m_builder->getUnitAttr());
		if (m_evmVersion.hasStaticCall())
			contractOp->setAttr("has_staticcall", m_builder->getUnitAttr());
		if (m_evmVersion.hasMcopy())
			contractOp->setAttr("has_mcopy", m_builder->getUnitAttr());
		if (m_evmVersion.canOverchargeGasForCall())
			contractOp->setAttr("can_overcharge_call_gas", m_builder->getUnitAttr());

		// Set insertion point to contract body
		m_builder->setInsertionPointToEnd(&contractOp.getBody().emplaceBlock());

		// Compute correct storage slots using the Solidity type system.
		// This uses ContractType::linearizedStateVariables() which already skips
		// constants and immutables, and computes proper storage layout with packing.
		m_stateVarSlots.clear();
		m_stateVarOffsets.clear();
		m_stateVarNames.clear();
		m_usedStateVarNames.clear();
		{
			ContractType contractType(_contract);
			for (auto const& [varDecl, slot, offset]: contractType.linearizedStateVariables(DataLocation::Storage))
			{
				m_stateVarSlots[stateVarName(*varDecl)] = slot;
				m_stateVarOffsets[stateVarName(*varDecl)] = offset;
			}
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

		// Initialisers run before the constructor body, so they are generated
		// as their own function the creation half calls first.
		generateInitializer(_contract);

		// Generate auto-getter functions for public state variables.
		// In Solidity, `uint256 public x` auto-generates `function x() view returns (uint256)`.
		// `mapping(K => V) public m` auto-generates `function m(K) view returns (V)`.
		std::set<std::string> generatedFunctions;
		std::set<std::string> generatedSymbolNames;

		// File-level functions live in the object that calls them, but importing
		// every free function into every contract is not merely wasteful: an
		// otherwise unrelated `function make() { new C(); }` makes C's own
		// runtime recursively reference C's creation object. Discover the
		// transitive call/reference closure rooted in this contract and its
		// linearised bases instead. Looking at every expression also covers
		// first-class function values, not just immediate calls.
		std::set<FunctionDefinition const*> reachableFreeFunctions;
		std::vector<ASTNode const*> reachabilityRoots;
		for (ContractDefinition const* baseContract: _contract.annotation().linearizedBaseContracts)
			reachabilityRoots.push_back(baseContract);
		for (size_t root = 0; root < reachabilityRoots.size(); ++root)
		{
			SimpleASTVisitor visitor(
				[&](ASTNode const& node)
				{
					auto const* expression = dynamic_cast<Expression const*>(&node);
					if (!expression)
						return true;
					auto const* function = dynamic_cast<FunctionDefinition const*>(
						ASTNode::referencedDeclaration(*expression));
					if (function && function->isFree() && function->isImplemented()
						&& reachableFreeFunctions.insert(function).second)
						reachabilityRoots.push_back(function);
					return true;
				},
				[](ASTNode const&) {}
			);
			reachabilityRoots[root]->accept(visitor);
		}

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
						generatedSymbolNames.insert(var->name());
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
					else if (auto const* arrayType = dynamic_cast<ArrayType const*>(var->type());
						arrayType && !arrayType->isByteArrayOrString())
					{
						generateArrayGetter(*var, *arrayType);
						generatedSymbolNames.insert(var->name());
						generatedFunctions.insert(TypeProvider::function(*var)->externalSignature());
					}
					else
					{
						generateSimpleGetter(*var);
						generatedSymbolNames.insert(var->name());
						generatedFunctions.insert(var->name() + "()");
					}
				}
			}
		}

		// Plan names for all functions before generating any body. A body can
		// call an overload declared later in source/C3 order; assigning names as
		// we emitted made that forward call fall back to the unsuffixed overload.
		// The resulting symbol existed, but with the wrong arity.
		std::vector<std::pair<FunctionDefinition const*, std::string>> plannedFunctions;
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
					std::string signature;
					if (func->isFallback())
						signature = "$fallback";
					else if (func->isReceive())
						signature = "$receive";
					else
					{
						signature = func->name() + "(";
						for (size_t i = 0; i < func->parameters().size(); ++i)
						{
							if (i > 0)
								signature += ",";
							signature += func->parameters()[i]->type()->toString();
						}
						signature += ")";
					}

					// Only generate if we haven't seen this signature yet
					// (derived contract functions take precedence)
					if (generatedFunctions.find(signature) == generatedFunctions.end())
					{
						generatedFunctions.insert(signature);
						std::string name = uniqueFunctionName(*func, generatedSymbolNames);
						m_emittedFunctionNames[func] = name;
						plannedFunctions.emplace_back(func, std::move(name));
					}
					else if (func->isImplemented() && !func->isFallback() && !func->isReceive())
						// An overridden base implementation is still reachable
						// by an explicit `Base.f(...)`, so it has to exist -
						// under the name such a call resolves to, qualified by
						// its defining contract so it cannot collide with the
						// override that shadows it.
					{
						std::string name = uniqueFunctionName(
							*func, generatedSymbolNames, baseContract->name() + ".");
						m_emittedFunctionNames[func] = name;
						plannedFunctions.emplace_back(func, std::move(name));
					}
				}
			}
		}

		// Contract bodies can call free/library overloads emitted later. Reserve
		// those symbols now as well, so forward references use the same unique
		// name as the eventual definition.
		{
			std::set<SourceUnit const*> allUnits = _contract.sourceUnit().referencedSourceUnits(true);
			allUnits.insert(&_contract.sourceUnit());
			for (SourceUnit const* unit: allUnits)
				for (auto const& node: unit->nodes())
				{
					if (auto const* freeFunction = dynamic_cast<FunctionDefinition const*>(node.get()))
					{
						if (reachableFreeFunctions.count(freeFunction)
							&& !m_emittedFunctionNames.count(freeFunction))
							m_emittedFunctionNames[freeFunction]
								= uniqueFunctionName(*freeFunction, generatedSymbolNames);
						continue;
					}
					if (auto const* library = dynamic_cast<ContractDefinition const*>(node.get());
						library && library->isLibrary() && library != &_contract)
						for (auto const* function: library->definedFunctions())
							if (!function->isConstructor() && function->isImplemented()
								&& !m_emittedFunctionNames.count(function))
								m_emittedFunctionNames[function] = uniqueFunctionName(
									*function, generatedSymbolNames, library->name() + ".");
				}
		}
		for (auto const& [function, name]: plannedFunctions)
			generateSolidityFunction(*function, name);

		// Generate constructor(s) from the linearized base contracts.
		// Walk from most-derived to base; only generate the first constructor found.
		// If the constructor has parameters and the derived contract provides base
		// constructor arguments, generate a merged 0-param constructor.
		{
			bool anyConstructor = false;
			for (ContractDefinition const* base: _contract.annotation().linearizedBaseContracts)
				for (auto const& func: base->definedFunctions())
					anyConstructor = anyConstructor || func->isConstructor();

			if (anyConstructor)
				generateConstructorChain(_contract);
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
					if (libDecl && libDecl->isLibrary() && libDecl != &_contract
						&& visitedLibraries.insert(libDecl).second)
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

								std::string const librarySignature = libDecl->name() + "." + signature;
								if (generatedFunctions.find(librarySignature) == generatedFunctions.end())
								{
									generatedFunctions.insert(librarySignature);
									// Internal library functions are pulled into the caller's
									// object, but retain a library-qualified symbol. This both
									// matches resolved call sites and prevents collisions with
									// same-named contract functions.
									std::string name = emittedNameOf(*func);
									m_emittedFunctionNames[func] = name;
									generateSolidityFunction(*func, name);
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
						if (reachableFreeFunctions.count(funcDef))
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
								std::string name = emittedNameOf(*funcDef);
								m_emittedFunctionNames[funcDef] = name;
								generateSolidityFunction(*funcDef, name);
							}
						}
					}
				}
			}
		}

		// Generated after contract, library, and free-function bodies, because
		// any of them can introduce a first-class internal-function call shape.
		generateIndirectDispatchers();

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
			m_stateVarOffsets.clear();
			m_stateVarNames.clear();
			m_usedStateVarNames.clear();
			{
				ContractType contractType(*childContract);
				for (auto const& [varDecl, slot, offset]: contractType.linearizedStateVariables(DataLocation::Storage))
				{
					m_stateVarSlots[stateVarName(*varDecl)] = slot;
					m_stateVarOffsets[stateVarName(*varDecl)] = offset;
				}
			}
			for (auto cit = childContract->annotation().linearizedBaseContracts.rbegin();
				 cit != childContract->annotation().linearizedBaseContracts.rend(); ++cit)
				for (auto const& var: (*cit)->stateVariables())
					generateStateVariable(*var, *childContract);

			// Generate child contract's functions (including getters)
			std::set<std::string> childGeneratedFunctions;
			std::set<std::string> childGeneratedSymbolNames;
			for (auto cit = childContract->annotation().linearizedBaseContracts.rbegin();
				 cit != childContract->annotation().linearizedBaseContracts.rend(); ++cit)
				for (auto const& var: (*cit)->stateVariables())
					if (var->visibility() >= Visibility::Public && !var->isConstant())
					{
						if (auto* mt = dynamic_cast<MappingType const*>(var->type()))
						{
							generateMappingGetter(*var, mt);
							childGeneratedSymbolNames.insert(var->name());
							std::string sig = var->name() + "(";
							auto const* mmt = mt; bool first = true;
							while (mmt) { if (!first) sig += ","; sig += mmt->keyType()->toString(); first = false;
								mmt = dynamic_cast<MappingType const*>(mmt->valueType()); }
							sig += ")"; childGeneratedFunctions.insert(sig);
						}
						else if (auto const* arrayType = dynamic_cast<ArrayType const*>(var->type());
							arrayType && !arrayType->isByteArrayOrString())
						{
							generateArrayGetter(*var, *arrayType);
							generatedSymbolNames.insert(var->name());
							generatedFunctions.insert(TypeProvider::function(*var)->externalSignature());
						}
						else
						{
							generateSimpleGetter(*var);
							childGeneratedSymbolNames.insert(var->name());
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
						{
							childGeneratedFunctions.insert(sig);
							std::string name = uniqueFunctionName(*func, childGeneratedSymbolNames);
							m_emittedFunctionNames[func] = name;
							generateSolidityFunction(*func, name);
						}
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
			auto slotIt = m_stateVarSlots.find(stateVarName(_var));
			if (slotIt != m_stateVarSlots.end())
				slotAttr = m_builder->getI64IntegerAttr(static_cast<int64_t>(slotIt->second));
		}

		// Create state variable operation using typed builder
		auto stateVarOp = m_builder->create<mlir::solidity::StateVarOp>(
			loc,
			stateVarName(_var),
			solidityType,
			m_builder->getStringAttr(visibility),
			_var.isConstant(),
			_var.immutable(),
			initialValue,
			slotAttr
		);
		if (!_var.isConstant() && !_var.immutable())
		{
			auto offset = m_stateVarOffsets.find(stateVarName(_var));
			if (offset != m_stateVarOffsets.end())
				stateVarOp->setAttr(
					"storageOffset", m_builder->getI32IntegerAttr(static_cast<int32_t>(offset->second)));
			stateVarOp->setAttr(
				"storageBytes",
				m_builder->getI32IntegerAttr(static_cast<int32_t>(_var.type()->storageBytes())));
			if (dynamic_cast<FunctionType const*>(_var.type())
				&& dynamic_cast<FunctionType const*>(_var.type())->kind() == FunctionType::Kind::External)
				stateVarOp->setAttr("externalFunction", m_builder->getUnitAttr());
			if (dynamic_cast<FunctionType const*>(_var.type())
				&& dynamic_cast<FunctionType const*>(_var.type())->kind() == FunctionType::Kind::Internal)
				stateVarOp->setAttr("internalFunction", m_builder->getUnitAttr());
		}

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

	/// The name a function was emitted under. A base implementation shadowed by
	/// an override exists under a qualified name, and the dispatcher calls by
	/// name - so `super.f` as a pointer has to name the same one the equivalent
	/// call would reach.
	std::string emittedNameOf(FunctionDefinition const& _function)
	{
		auto known = m_emittedFunctionNames.find(&_function);
		if (known != m_emittedFunctionNames.end())
			return known->second;
		return _function.name();
	}

	/// Experimental Solidity functions express their result through
	/// experimentalReturnExpression and legitimately have no ParameterList for
	/// returns. The classic AST accessor assumes that list exists, so every
	/// generic MLIR traversal must treat a missing list as an empty one.
	static std::vector<ASTPointer<VariableDeclaration>> const& safeReturnParameters(
		FunctionDefinition const& _function)
	{
		static std::vector<ASTPointer<VariableDeclaration>> const empty;
		return _function.returnParameterList() ? _function.returnParameters() : empty;
	}

	/// MLIR/Yul symbols do not overload by type. Preserve the source spelling
	/// for the first declaration (which keeps readable IR), then suffix later
	/// overloads with the stable AST id. The ABI signature remains a separate
	/// attribute and therefore still determines the external selector.
	std::string uniqueFunctionName(
		FunctionDefinition const& _function,
		std::set<std::string>& _used,
		std::string const& _prefix = {})
	{
		std::string const bare = _prefix + _function.name();
		if (_used.insert(bare).second)
			return bare;
		std::string const unique = bare + "$" + std::to_string(_function.id());
		_used.insert(unique);
		return unique;
	}

	/// One dispatcher per shape, turning a pointer back into a call by name.
	///
	/// Generated after every body, because only then is it known which
	/// functions had their address taken. An unknown id reverts - it cannot be
	/// a pointer this contract ever made.
	void generateIndirectDispatchers()
	{
		for (auto const& [arity, resultCount]: m_indirectShapes)
		{
			auto loc = m_builder->getUnknownLoc();
			auto word = mlir::solidity::UIntType::get(m_context.get(), 256);
			llvm::SmallVector<mlir::Type, 4> inputs(arity + 1, word);
			llvm::SmallVector<mlir::Type, 2> results(resultCount, word);

			auto funcOp = m_builder->create<mlir::solidity::FunctionOp>(
				loc,
				indirectDispatcherName(arity, resultCount),
				m_builder->getFunctionType(inputs, results),
				"internal",
				"nonpayable");

			auto& entryBlock = funcOp.getBody().emplaceBlock();
			llvm::SmallVector<mlir::Value, 4> params;
			for (mlir::Type type: inputs)
				params.push_back(entryBlock.addArgument(type, loc));

			auto savedIP = m_builder->saveInsertionPoint();
			m_builder->setInsertionPointToEnd(&entryBlock);

			for (FunctionDefinition const* target: m_functionPointerOrder)
			{
				if (target->parameters().size() != arity || safeReturnParameters(*target).size() != resultCount)
					continue;

				mlir::Value id = m_builder->create<mlir::solidity::ConstantOp>(
					loc,
					m_builder->getIntegerAttr(
						m_builder->getI64Type(), static_cast<int64_t>(m_functionPointerIds[target])),
					word);
				mlir::Value matches = m_builder->create<mlir::solidity::CmpOp>(
					loc, mlir::solidity::BoolType::get(m_context.get()), params[0], id,
					m_builder->getStringAttr("eq"));

				auto ifOp = m_builder->create<mlir::solidity::IfOp>(loc, matches);
				mlir::OpBuilder::InsertionGuard guard(*m_builder);
				m_builder->setInsertionPointToEnd(&ifOp.getThenRegion().emplaceBlock());

				auto called = m_builder->create<mlir::solidity::FunctionCallOp>(
					loc,
					mlir::TypeRange{results},
					m_builder->getStringAttr(emittedNameOf(*target)),
					mlir::ValueRange{llvm::ArrayRef<mlir::Value>(params).drop_front()});
				llvm::SmallVector<mlir::Value, 2> returned(called->getResults());
				m_builder->create<mlir::solidity::ReturnOp>(loc, mlir::ValueRange{returned});
				ifOp.getElseRegion().emplaceBlock();
			}

			// An id this contract never handed out cannot be a pointer it made.
			auto failure = m_builder->create<mlir::solidity::RevertOp>(
				loc,
				m_legacyCodegen
					? mlir::StringAttr{}
					: m_builder->getStringAttr("call through an unknown function pointer"));
			// Solidity specifies Panic(0x51) for calling a zero-initialised or
			// otherwise invalid internal function pointer.  Keep this typed on
			// the terminating op so lowering does not confuse it with Error(string).
			// The legacy backend instead represents the zero value with a special
			// nonzero entry tag whose stub performs a bare revert. Preserve that
			// observable compatibility behavior in the legacy direct-MLIR route.
			if (!m_legacyCodegen)
				failure->setAttr("panicCode", m_builder->getI64IntegerAttr(0x51));
			m_builder->restoreInsertionPoint(savedIP);
		}
	}

	/// State variable initialisers, as a function the creation half calls.
	///
	/// They were recorded as a decimal string on the declaration, which only a
	/// plain integer literal fits - so `bytes32 immutable v = keccak256("x")`,
	/// and every other expression, was dropped and read zero. Generating them
	/// as ordinary IR costs nothing: it is the same expression generator the
	/// rest of the contract already uses.
	void generateInitializer(ContractDefinition const& _contract)
	{
		std::vector<VariableDeclaration const*> initialised;
		for (auto it = _contract.annotation().linearizedBaseContracts.rbegin();
			 it != _contract.annotation().linearizedBaseContracts.rend();
			 ++it)
			for (auto const& var: (*it)->stateVariables())
				// A constant has no storage to initialise; it is substituted
				// wherever it is named.
				if (var->value() && !var->isConstant())
					initialised.push_back(var);

		if (initialised.empty())
			return;

		auto loc = this->loc(_contract);
		auto funcOp = m_builder->create<mlir::solidity::FunctionOp>(
			loc, kInitializerName, m_builder->getFunctionType({}, {}), "internal", "nonpayable");
		// The dispatcher skips anything carrying a `kind`: this is reached from
		// the creation code, never by a selector.
		funcOp->setAttr("kind", m_builder->getStringAttr("initializer"));

		auto& entryBlock = funcOp.getBody().emplaceBlock();
		auto savedInsertionPoint = m_builder->saveInsertionPoint();
		m_builder->setInsertionPointToEnd(&entryBlock);

		for (VariableDeclaration const* var: initialised)
		{
			if (mlir::Value value = generateSolidityExpression(*var->value()))
			{
				// Array literals infer their narrowest element and fixed length,
				// while the declared state variable may be wider and dynamic.
				// Preserve that assignment conversion so storage receives the
				// declared shape (including its length slot).
				value = coerceValue(loc, value, assignmentValueType(var->type()));
				m_builder->create<mlir::solidity::StoreStateVarOp>(loc, stateVarName(*var), value);
			}
		}
		m_builder->create<mlir::solidity::ReturnOp>(loc, mlir::ValueRange{});

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

		// A constant has no slot: it is substituted wherever it is named, so
		// loading one read whatever happened to be in storage - zero.
		mlir::Value result;
		if (_var.isConstant() && _var.value())
			result = generateSolidityExpression(*_var.value());
		if (!result)
			result = m_builder->create<mlir::solidity::LoadStateVarOp>(loc, returnType, stateVarName(_var))
						 .getResult();

		m_builder->create<mlir::solidity::ReturnOp>(loc, mlir::ValueRange{result});

		m_builder->restoreInsertionPoint(savedInsertionPoint);
	}

	/// Generate the indexed ABI getter for a public state array. Solidity adds
	/// one uint256 parameter for every non-byte-array dimension and returns the
	/// selected element; treating the declaration as a simple value instead
	/// emitted `a()` and made every real `a(i, ...)` selector miss.
	void generateArrayGetter(VariableDeclaration const& _var, ArrayType const& _arrayType)
	{
		auto loc = this->loc(_var);
		auto const* getterType = TypeProvider::function(_var);
		llvm::SmallVector<mlir::Type, 4> parameterTypes;
		for (Type const* type: getterType->parameterTypes())
			parameterTypes.push_back(translateSolidityType(*type));
		llvm::SmallVector<mlir::Type, 2> resultTypes;
		for (Type const* type: getterType->returnParameterTypes())
			resultTypes.push_back(translateSolidityType(*type));

		auto funcOp = m_builder->create<mlir::solidity::FunctionOp>(
			loc, _var.name(), m_builder->getFunctionType(parameterTypes, resultTypes), "public", "view");
		auto& entryBlock = funcOp.getBody().emplaceBlock();
		llvm::SmallVector<mlir::Value, 4> parameters;
		for (mlir::Type type: parameterTypes)
			parameters.push_back(entryBlock.addArgument(type, loc));

		auto savedInsertionPoint = m_builder->saveInsertionPoint();
		m_builder->setInsertionPointToEnd(&entryBlock);
		mlir::Type rootType = translateSolidityType(_arrayType);
		auto load = m_builder->create<mlir::solidity::LoadStateVarOp>(
			loc, rootType, stateVarName(_var));
		load->setAttr("asReference", m_builder->getUnitAttr());
		mlir::Value value = load.getResult();
		Type const* sourceType = &_arrayType;

		for (mlir::Value index: parameters)
		{
			auto const* array = dynamic_cast<ArrayType const*>(sourceType);
			if (!array || array->isByteArrayOrString())
				break;
			mlir::Type elementType = translateSolidityType(*array->baseType());
			auto access = m_builder->create<mlir::solidity::ArrayAccessOp>(
				loc, elementType, value, index);
			markStorageArray(access.getOperation(), array, stateVarName(_var));
			// The legacy public-getter ABI rejects an invalid index with empty
			// revert data. Ordinary source-level indexing remains Panic(0x32).
			access->setAttr("getterBounds", m_builder->getUnitAttr());
			if (isStorageReferenceType(array->baseType()))
				access->setAttr("asReference", m_builder->getUnitAttr());
			value = access.getResult();
			sourceType = array->baseType();
		}

		// Struct-valued public getters have multiple projected results and are
		// handled separately; all array getters reaching this path have one ABI
		// result after consuming their dimensions.
		if (resultTypes.size() == 1)
		{
			auto ret = m_builder->create<mlir::solidity::ReturnOp>(loc, mlir::ValueRange{value});
			if (isStorageReferenceType(sourceType))
				ret->setAttr("storage_results", m_builder->getArrayAttr({m_builder->getI32IntegerAttr(0)}));
		}
		else if (auto const* structure = dynamic_cast<StructType const*>(sourceType))
		{
			llvm::SmallVector<mlir::Value, 4> results;
			std::vector<mlir::Attribute> storageResults;
			auto const& names = getterType->returnParameterNames();
			for (auto [index, name]: llvm::enumerate(names))
			{
				if (index >= resultTypes.size())
					break;
				Type const* memberType = structure->memberType(name);
				if (!memberType)
					continue;
				if (memberType->category() == Type::Category::Mapping)
					continue;
				if (auto const* memberArray = dynamic_cast<ArrayType const*>(memberType);
					memberArray && !memberArray->isByteArrayOrString())
					continue;
				auto const [slotOffset, byteOffset] = structure->storageOffsetsOfMember(name);
				(void)byteOffset;
				auto member = m_builder->create<mlir::solidity::StorageMemberLoadOp>(
					loc, resultTypes[index], value,
					m_builder->getI64IntegerAttr(static_cast<int64_t>(slotOffset)));
				markStorageMember(member.getOperation(), *structure, name);
				if (isStorageReferenceType(memberType))
				{
					member->setAttr("asReference", m_builder->getUnitAttr());
					storageResults.push_back(
						m_builder->getI32IntegerAttr(static_cast<int32_t>(results.size())));
				}
				results.push_back(member.getResult());
			}
			auto ret = m_builder->create<mlir::solidity::ReturnOp>(
				loc, mlir::ValueRange{results});
			if (!storageResults.empty())
				ret->setAttr("storage_results", m_builder->getArrayAttr(storageResults));
		}
		else
			m_builder->create<mlir::solidity::ReturnOp>(loc, mlir::ValueRange{});

		m_builder->restoreInsertionPoint(savedInsertionPoint);
	}

	/// Generate a merged 0-parameter constructor for a derived contract.
	/// Base constructor arguments (from InheritanceSpecifier) are evaluated
	/// at the top of the body and mapped to the constructor's parameters.
	/// Evaluate and bind one base constructor's arguments. Solidity performs
	/// this phase in derived-to-base linearisation order, independently of the
	/// base-to-derived order in which constructor bodies subsequently execute.
	void bindConstructorParameters(
		FunctionDefinition const& _constructor,
		ContractDefinition const& _derivedContract,
		mlir::Location _loc)
	{
		auto const params = _constructor.parameters();
		auto supplied = _derivedContract.annotation().baseConstructorArguments.find(&_constructor);
		std::vector<ASTPointer<Expression>> const* args = nullptr;
		if (supplied != _derivedContract.annotation().baseConstructorArguments.end())
		{
			if (auto* inheritance = dynamic_cast<InheritanceSpecifier const*>(supplied->second))
				args = inheritance->arguments();
			else if (auto* modifier = dynamic_cast<ModifierInvocation const*>(supplied->second))
				args = modifier->arguments();
		}

		for (size_t i = 0; i < params.size(); ++i)
		{
			mlir::Value value;
			if (args && i < args->size())
				value = generateSolidityExpression(*(*args)[i]);
			if (!value)
				value = emitUnsupported(
					_loc, translateSolidityType(*params[i]->type()), "base constructor argument not supplied");
			m_symbolTable.insert(params[i]->name(), value);
			m_valueMap[params[i]->id()] = value;
		}

	}

	void inlineConstructorBody(FunctionDefinition const& _constructor)
	{
		if (_constructor.isImplemented())
			emitModifierChain(_constructor, 0, {}, [&]() {
				generateSolidityStatement(_constructor.body());
			});
	}

	/// The whole constructor chain as one function. Its ABI-visible parameters
	/// are those of the most-derived constructor; base arguments are expressions
	/// evaluated from those parameters. This is essential for `D(bytes s) is
	/// B(f(s))`: emitting only D's body silently skips every base constructor.
	///
	/// Solidity runs every base constructor, most-base first, then the derived
	/// body. Only the first constructor found walking towards the bases used to
	/// be generated, so `contract D is B { constructor() B("x") {} }` ran D's
	/// empty body and never B's - and everything B was given to store stayed
	/// zero.
	void generateConstructorChain(ContractDefinition const& _contract)
	{
		auto loc = this->loc(_contract);
		m_valueMap.clear();
		FunctionDefinition const* own = nullptr;
		for (auto const& function: _contract.definedFunctions())
			if (function->isConstructor())
			{
				own = function;
				break;
			}

		std::vector<mlir::Type> parameterTypes;
		if (own)
			for (auto const& parameter: own->parameters())
				parameterTypes.push_back(translateSolidityType(*parameter->type()));
		auto funcOp = m_builder->create<mlir::solidity::FunctionOp>(
			loc, "", m_builder->getFunctionType(parameterTypes, {}), "public", "nonpayable");
		funcOp->setAttr("kind", m_builder->getStringAttr("constructor"));
		if (own && !own->parameters().empty())
		{
			std::vector<mlir::Attribute> names;
			std::vector<mlir::Attribute> externalFunctions;
			for (auto [index, parameter]: llvm::enumerate(own->parameters()))
			{
				names.push_back(m_builder->getStringAttr(parameter->name()));
				if (auto const* functionType = dynamic_cast<FunctionType const*>(parameter->type());
					functionType && functionType->kind() == FunctionType::Kind::External)
					externalFunctions.push_back(
						m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
			}
			funcOp->setAttr("param_names", m_builder->getArrayAttr(names));
			if (!externalFunctions.empty())
				funcOp->setAttr(
					"external_function_params", m_builder->getArrayAttr(externalFunctions));
		}

		auto& entryBlock = funcOp.getBody().emplaceBlock();
		for (mlir::Type parameterType: parameterTypes)
			entryBlock.addArgument(parameterType, loc);
		SymbolTableScopeT functionScope(m_symbolTable);
		if (own)
			for (auto [index, parameter]: llvm::enumerate(own->parameters()))
			{
				m_symbolTable.insert(parameter->name(), entryBlock.getArgument(index));
				m_valueMap[parameter->id()] = entryBlock.getArgument(index);
			}
		auto savedIP = m_builder->saveInsertionPoint();
		m_builder->setInsertionPointToEnd(&entryBlock);

		// Argument expressions have observable side effects. Bind them in the
		// compiler's linearised derived-to-base order before executing any base
		// body (D,C,B,A here, excluding D's already-bound own parameters).
		for (ContractDefinition const* base: _contract.annotation().linearizedBaseContracts)
		{
			if (base == &_contract)
				continue;
			for (auto const& func: base->definedFunctions())
				if (func->isConstructor())
				{
					bindConstructorParameters(*func, _contract, loc);
					break;
				}
		}

		for (auto it = _contract.annotation().linearizedBaseContracts.rbegin();
			 it != _contract.annotation().linearizedBaseContracts.rend();
			 ++it)
			for (auto const& func: (*it)->definedFunctions())
				if (func->isConstructor())
				{
					if (func == own)
					{
						if (func->isImplemented())
							emitModifierChain(*func, 0, {}, [&]() {
								generateSolidityStatement(func->body());
							});
					}
					else
						inlineConstructorBody(*func);
					break;
				}

		bool terminated = false;
		if (!entryBlock.empty())
			terminated = mlir::isa<
				mlir::solidity::ReturnOp,
				mlir::solidity::RevertOp,
				mlir::solidity::RevertValueOp>(entryBlock.back());
		if (!terminated)
			emitReturn(loc, {});

		m_builder->restoreInsertionPoint(savedIP);
	}

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
			if (mlir::isa<
				mlir::solidity::ReturnOp,
				mlir::solidity::RevertOp,
				mlir::solidity::RevertValueOp>(lastOp))
				hasReturn = true;
		}
		if (!hasReturn)
			emitReturn(loc, {});

		m_builder->restoreInsertionPoint(savedIP);
	}

	/// Emit the textual modifier expansion around a private helper containing
	/// the original function body. Each `_` invokes the next modifier (or the
	/// helper), so modifiers with zero or multiple placeholders retain their
	/// Solidity control flow and postludes run after a body-level return.
	void emitModifierChain(
		FunctionDefinition const& _function,
		size_t _index,
		std::string const& _bodyCallee,
		std::function<void()> _terminal = {})
	{
		if (_index >= _function.modifiers().size())
		{
			if (_terminal)
			{
				_terminal();
				return;
			}
			llvm::SmallVector<mlir::Value, 4> arguments;
			for (auto const& parameter: _function.parameters())
				arguments.push_back(m_valueMap[parameter->id()]);
			// The legacy code generator gives named return variables shared
			// storage across repeated `_` invocations. Pass their current SSA
			// values into the body helper so a second invocation resumes from
			// the result of the first instead of reinitialising them.
			for (auto const& result: safeReturnParameters(_function))
				arguments.push_back(m_valueMap[result->id()]);
			llvm::SmallVector<mlir::Type, 4> resultTypes;
			for (auto const& result: safeReturnParameters(_function))
				resultTypes.push_back(translateSolidityType(*result->type()));
			auto call = m_builder->create<mlir::solidity::FunctionCallOp>(
				loc(_function), mlir::TypeRange{resultTypes},
				m_builder->getStringAttr(_bodyCallee), arguments);
			m_modifierResults.assign(call->getResults().begin(), call->getResults().end());
			for (auto [index, result]: llvm::enumerate(safeReturnParameters(_function)))
				m_valueMap[result->id()] = call->getResult(index);
			return;
		}

		ModifierInvocation const& invocation = *_function.modifiers()[_index];
		auto const* definition = dynamic_cast<ModifierDefinition const*>(
			invocation.name().annotation().referencedDeclaration);
		// Constructor invocations share the AST list but are handled by the
		// constructor chain, not as function modifiers.
		if (!definition)
		{
			emitModifierChain(_function, _index + 1, _bodyCallee, std::move(_terminal));
			return;
		}
		if (m_mostDerivedContract
			&& *invocation.name().annotation().requiredLookup == VirtualLookup::Virtual)
			definition = &definition->resolveVirtual(*m_mostDerivedContract);

		std::vector<std::pair<int64_t, std::optional<mlir::Value>>> savedParameters;
		std::vector<std::pair<int64_t, std::optional<mlir::Value>>> savedLocals;
		for (int64_t id: collectDeclaredVariables(definition->body()))
		{
			auto known = m_valueMap.find(id);
			savedLocals.emplace_back(
				id, known == m_valueMap.end() ? std::optional<mlir::Value>{} : known->second);
		}
		auto const parameters = definition->parameters();
		auto const* supplied = invocation.arguments();
		for (size_t i = 0; i < parameters.size(); ++i)
		{
			auto known = m_valueMap.find(parameters[i]->id());
			savedParameters.emplace_back(
				parameters[i]->id(),
				known == m_valueMap.end() ? std::optional<mlir::Value>{} : known->second);
			mlir::Value value;
			if (supplied && i < supplied->size())
				value = generateSolidityExpression(*(*supplied)[i]);
			if (!value)
				value = emitUnsupported(
					loc(invocation), translateSolidityType(*parameters[i]->type()),
					"modifier argument not generated");
			m_valueMap[parameters[i]->id()] = coerceValue(
				loc(invocation), value, translateSolidityType(*parameters[i]->type()));
		}

		auto savedContinuation = std::move(m_modifierContinuation);
		m_modifierContinuation = [this, &_function, _index, &_bodyCallee, _terminal]() mutable {
			emitModifierChain(_function, _index + 1, _bodyCallee, _terminal);
		};
		generateSolidityStatement(definition->body());
		m_modifierContinuation = std::move(savedContinuation);
		for (auto const& [id, value]: savedLocals)
			if (value)
				m_valueMap[id] = *value;
			else
				m_valueMap.erase(id);
		for (auto const& [id, value]: savedParameters)
			if (value)
				m_valueMap[id] = *value;
			else
				m_valueMap.erase(id);
	}

	mlir::solidity::FunctionOp generateSolidityFunction(
		FunctionDefinition const& _func,
		std::string const& _nameOverride = {},
		bool _applyModifiers = true)
	{
		auto loc = this->loc(_func);
		std::optional<std::string> modifierBodyCallee;
		if (_applyModifiers && !_func.isConstructor() && !_func.modifiers().empty())
		{
			std::string const publicName = _nameOverride.empty() ? _func.name() : _nameOverride;
			std::string const helperName
				= publicName + "$modifier_body$" + std::to_string(_func.id());
			auto helper = generateSolidityFunction(_func, helperName, false);
			helper->removeAttr("abi_signature");
			helper->setAttr("visibility", m_builder->getStringAttr("internal"));
			modifierBodyCallee = helperName.find('.') == std::string::npos && m_mostDerivedContract
				? m_mostDerivedContract->name() + "." + helperName
				: helperName;
		}
		// Function-local AST declarations can be emitted more than once (for
		// example when the same inherited body is nested as a child object).
		// No SSA value from the earlier body is valid in this one.
		m_valueMap.clear();

		// Build function type using Solidity types
		std::vector<mlir::Type> paramTypes;
		for (auto const& param: _func.parameters())
		{
			paramTypes.push_back(translateSolidityType(*param->type()));
		}
		if (!_applyModifiers)
			for (auto const& result: safeReturnParameters(_func))
				paramTypes.push_back(translateSolidityType(*result->type()));

		std::vector<mlir::Type> returnTypes;
		for (auto const& ret: safeReturnParameters(_func))
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
			std::vector<mlir::Attribute> externalFunctionParams;
			std::vector<mlir::Attribute> externalFunctionArrayParams;
			std::vector<mlir::Attribute> enumArrayLimits;
			bool hasEnumArray = false;
			for (auto [index, param]: llvm::enumerate(_func.parameters()))
			{
				paramNameAttrs.push_back(m_builder->getStringAttr(param->name()));
				uint64_t enumArrayLimit = 0;
				if (auto const* arrayType = dynamic_cast<ArrayType const*>(param->type()))
					if (auto const* enumType = dynamic_cast<EnumType const*>(arrayType->baseType()))
						enumArrayLimit = static_cast<uint64_t>(enumType->numberOfMembers());
				enumArrayLimits.push_back(
					m_builder->getI64IntegerAttr(static_cast<int64_t>(enumArrayLimit)));
				hasEnumArray = hasEnumArray || enumArrayLimit != 0;
				if (auto const* functionType = dynamic_cast<FunctionType const*>(param->type());
					functionType && functionType->kind() == FunctionType::Kind::External)
					externalFunctionParams.push_back(
						m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
				else if (dynamic_cast<ArrayType const*>(param->type())
					&& containsExternalFunction(param->type()))
					externalFunctionArrayParams.push_back(
						m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
			}
			if (!paramNameAttrs.empty())
				funcOp->setAttr("param_names", m_builder->getArrayAttr(paramNameAttrs));
			if (!externalFunctionParams.empty())
				funcOp->setAttr(
					"external_function_params", m_builder->getArrayAttr(externalFunctionParams));
			if (!externalFunctionArrayParams.empty())
				funcOp->setAttr(
					"external_function_array_params", m_builder->getArrayAttr(externalFunctionArrayParams));
			if (hasEnumArray)
				funcOp->setAttr("enum_array_limits", m_builder->getArrayAttr(enumArrayLimits));

			std::vector<mlir::Attribute> returnNameAttrs;
			std::vector<mlir::Attribute> externalFunctionResults;
			// Remembered so a body that falls off its end returns what the
			// named results hold rather than a placeholder.
			m_returnParameters.clear();
			for (auto [index, ret]: llvm::enumerate(safeReturnParameters(_func)))
			{
				returnNameAttrs.push_back(m_builder->getStringAttr(ret->name()));
				m_returnParameters.push_back(ret.get());
				if (auto const* functionType = dynamic_cast<FunctionType const*>(ret->type());
					functionType && functionType->kind() == FunctionType::Kind::External)
					externalFunctionResults.push_back(
						m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
			}
			if (!returnNameAttrs.empty())
				funcOp->setAttr("return_names", m_builder->getArrayAttr(returnNameAttrs));
			if (!externalFunctionResults.empty())
				funcOp->setAttr(
					"external_function_results", m_builder->getArrayAttr(externalFunctionResults));
		}

		// Store the ABI-canonical external signature for correct selector computation.
		// This is needed because contract types become addresses, struct types become
		// tuples, etc. in the ABI encoding — information lost when MLIR erases to uint256.
		if (!_func.isConstructor() && !_func.isReceive() && !_func.isFallback()
			&& (_func.visibility() == Visibility::Public || _func.visibility() == Visibility::External))
			// FunctionDefinition supplies the declaration context required to
			// canonicalise structs containing external-function members. Calling
			// externalSignature() on the detached type throws an ICE for those
			// perfectly valid corpus contracts.
			funcOp->setAttr(
				"abi_signature", m_builder->getStringAttr(_func.externalSignature()));

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

		// Named return variables are ordinary zero-initialised locals. Besides
		// implementing that Solidity rule, defining them before structured
		// control flow lets loops carry an otherwise not-yet-assigned result.
		for (auto [returnIndex, ret]: llvm::enumerate(safeReturnParameters(_func)))
		{
			mlir::Type type = translateSolidityType(*ret->type());
			mlir::Value initial;
			if (!_applyModifiers)
				initial = entryBlock.getArgument(params.size() + returnIndex);
			else if (llvm::isa<mlir::solidity::StringType, mlir::solidity::DynamicBytesType>(type)
				|| llvm::isa<mlir::solidity::ArrayType>(type))
			{
				uint64_t initialLength = 0;
				if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(type);
					array && !array.isDynamicallySized())
					initialLength = static_cast<uint64_t>(array.getSize());
				mlir::Value zero = m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getIntegerAttr(m_builder->getI64Type(), initialLength),
					mlir::solidity::UIntType::get(m_context.get(), 256));
					initial = m_builder->create<mlir::solidity::MemoryArrayCreateOp>(loc, type, zero);
				}
				else
					initial = m_builder->create<mlir::solidity::ConstantOp>(
						loc, m_builder->getIntegerAttr(m_builder->getI64Type(), 0), type);
				m_valueMap[ret->id()] = initial;
		}

		// Generate function body
		if (_func.isImplemented())
		{
			if (modifierBodyCallee)
			{
				auto savedResults = std::move(m_modifierResults);
				m_modifierResults.clear();
				for (auto const& result: safeReturnParameters(_func))
					m_modifierResults.push_back(m_valueMap[result->id()]);
				emitModifierChain(_func, 0, *modifierBodyCallee);
				if (!endsSourceControlFlow(&entryBlock))
				{
					llvm::SmallVector<mlir::Value, 4> results;
					for (auto [index, result]: llvm::enumerate(safeReturnParameters(_func)))
					{
						auto known = m_valueMap.find(result->id());
						results.push_back(
							known != m_valueMap.end() && known->second
								? known->second : m_modifierResults[index]);
					}
					emitReturn(loc, results);
				}
				m_modifierResults = std::move(savedResults);
			}
			else
				generateSolidityStatement(_func.body());
		}

		// Add return if not already present
		// Check if the last operation is a solidity.return
		bool hasReturn = false;
		if (!entryBlock.empty())
		{
			auto& lastOp = entryBlock.back();
			if (mlir::isa<
				mlir::solidity::ReturnOp,
				mlir::solidity::RevertOp,
				mlir::solidity::RevertValueOp>(lastOp))
				hasReturn = true;
		}

		if (!hasReturn)
		{
			emitReturn(loc, {});
		}

		// Restore insertion point
		m_builder->restoreInsertionPoint(savedIP);
		return funcOp;
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
		else if (dynamic_cast<PlaceholderStatement const*>(&_stmt))
			for (VariableDeclaration const* result: m_returnParameters)
				modifiedVars.insert(result->id());

		return modifiedVars;
	}

	/// Local SSA values assigned while evaluating an expression. Short-circuit
	/// operators need these as additional `scf.if` results: a right-hand-side
	/// assignment happens only in the selected arm, but its value is live after
	/// the expression. State writes are operations and therefore need no SSA
	/// result here.
	void collectExpressionModifiedVariables(Expression const& _expr, std::set<int64_t>& _out)
	{
		auto collectTarget = [&](auto&& self, Expression const& _target) -> void {
			Expression const& target = withoutParentheses(_target);
			if (auto const* identifier = dynamic_cast<Identifier const*>(&target))
			{
				if (auto const* variable
					= dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration))
					if (!variable->isStateVariable())
						_out.insert(variable->id());
			}
			else if (auto const* tuple = dynamic_cast<TupleExpression const*>(&target))
				for (auto const& component: tuple->components())
					if (component)
						self(self, *component);
		};

		if (auto const* assignment = dynamic_cast<Assignment const*>(&_expr))
		{
			collectTarget(collectTarget, assignment->leftHandSide());
			collectExpressionModifiedVariables(assignment->leftHandSide(), _out);
			collectExpressionModifiedVariables(assignment->rightHandSide(), _out);
		}
		else if (auto const* unary = dynamic_cast<UnaryOperation const*>(&_expr))
		{
			if (unary->getOperator() == Token::Inc || unary->getOperator() == Token::Dec)
				collectTarget(collectTarget, unary->subExpression());
			collectExpressionModifiedVariables(unary->subExpression(), _out);
		}
		else if (auto const* binary = dynamic_cast<BinaryOperation const*>(&_expr))
		{
			collectExpressionModifiedVariables(binary->leftExpression(), _out);
			collectExpressionModifiedVariables(binary->rightExpression(), _out);
		}
		else if (auto const* conditional = dynamic_cast<Conditional const*>(&_expr))
		{
			collectExpressionModifiedVariables(conditional->condition(), _out);
			collectExpressionModifiedVariables(conditional->trueExpression(), _out);
			collectExpressionModifiedVariables(conditional->falseExpression(), _out);
		}
		else if (auto const* tuple = dynamic_cast<TupleExpression const*>(&_expr))
		{
			for (auto const& component: tuple->components())
				if (component)
					collectExpressionModifiedVariables(*component, _out);
		}
		else if (auto const* call = dynamic_cast<FunctionCall const*>(&_expr))
		{
			collectExpressionModifiedVariables(call->expression(), _out);
			for (auto const& argument: call->arguments())
				collectExpressionModifiedVariables(*argument, _out);
		}
		else if (auto const* options = dynamic_cast<FunctionCallOptions const*>(&_expr))
		{
			collectExpressionModifiedVariables(options->expression(), _out);
			for (auto const& option: options->options())
				collectExpressionModifiedVariables(*option, _out);
		}
		else if (auto const* member = dynamic_cast<MemberAccess const*>(&_expr))
			collectExpressionModifiedVariables(member->expression(), _out);
		else if (auto const* index = dynamic_cast<IndexAccess const*>(&_expr))
		{
			collectExpressionModifiedVariables(index->baseExpression(), _out);
			if (index->indexExpression())
				collectExpressionModifiedVariables(*index->indexExpression(), _out);
		}
		else if (auto const* range = dynamic_cast<IndexRangeAccess const*>(&_expr))
		{
			collectExpressionModifiedVariables(range->baseExpression(), _out);
			if (range->startExpression())
				collectExpressionModifiedVariables(*range->startExpression(), _out);
			if (range->endExpression())
				collectExpressionModifiedVariables(*range->endExpression(), _out);
		}
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
	mlir::solidity::MappingStoreOp createMappingStoreOp(mlir::Location loc, std::string const& varName,
		std::vector<mlir::Value> const& keys, mlir::Value value, mlir::Value baseSlot = {})
	{
		// Combine keys and value into a single operand list
		std::vector<mlir::Value> operands;
		if (baseSlot)
			operands.push_back(baseSlot);
		operands.insert(operands.end(), keys.begin(), keys.end());
		operands.push_back(value);

		auto store = m_builder->create<mlir::solidity::MappingStoreOp>(
			loc,
			m_builder->getStringAttr(varName),
			m_builder->getI32IntegerAttr(static_cast<int32_t>(keys.size())),
			operands);
		if (baseSlot)
			store->setAttr("baseSlotOperand", m_builder->getUnitAttr());
		return store;
	}

	mlir::Value mappingBaseSlot(IndexAccess const& _access)
	{
		Expression const* current = &_access;
		while (auto const* index = dynamic_cast<IndexAccess const*>(current))
		{
			if (!dynamic_cast<MappingType const*>(index->baseExpression().annotation().type))
				break;
			current = &withoutParentheses(index->baseExpression());
		}
		auto const* identifier = dynamic_cast<Identifier const*>(current);
		if (identifier)
		{
			auto const* variable
				= dynamic_cast<VariableDeclaration const*>(identifier->annotation().referencedDeclaration);
			if (variable)
			{
				// A direct state mapping is still described most compactly by its
				// name; the lowering obtains its statically assigned slot.
				if (variable->isStateVariable())
					return {};
				auto known = m_valueMap.find(variable->id());
				if (known != m_valueMap.end())
					return known->second;
			}
		}

		// A mapping returned by a call, selected from an array, or stored in a
		// struct member is itself an expression whose value is its base slot.
		// Evaluate that expression once and carry the resulting slot explicitly.
		return generateSolidityExpression(*current);
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

	/// Preserve the source enum domains on ordinary mapping operations. ABI-v1
	/// values are uint8 here, so their MLIR type no longer identifies enum keys.
	/// Synthesized public getters intentionally omit this marker because legacy
	/// getters accept out-of-range keys while source-level `m[e]` must panic.
	void markMappingEnumKeyLimits(mlir::Operation* _operation, IndexAccess const& _access)
	{
		std::vector<mlir::Attribute> limits;
		Expression const* current = &_access;
		while (auto const* index = dynamic_cast<IndexAccess const*>(current))
		{
			auto const* mapping = dynamic_cast<MappingType const*>(
				index->baseExpression().annotation().type);
			if (!mapping)
				break;
			uint64_t limit = 0;
			if (auto const* enumType = dynamic_cast<EnumType const*>(mapping->keyType()))
				limit = static_cast<uint64_t>(enumType->numberOfMembers());
			limits.push_back(m_builder->getI64IntegerAttr(static_cast<int64_t>(limit)));
			current = &index->baseExpression();
		}
		std::reverse(limits.begin(), limits.end());
		if (llvm::any_of(limits, [](mlir::Attribute _limit) {
				return llvm::cast<mlir::IntegerAttr>(_limit).getInt() != 0;
			}))
			_operation->setAttr("enum_key_limits", m_builder->getArrayAttr(limits));
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

	/// The name a state variable is known by in the IR.
	///
	/// A derived contract may declare one with a name a base already used.
	/// Keyed by name alone the two share a storage slot, so the second silently
	/// reads and writes the first - and both answer whichever was stored last.
	/// The first declaration seen (bases first) keeps the plain name; a
	/// shadowing one is qualified by the contract that declares it.
	std::string stateVarName(VariableDeclaration const& _var)
	{
		auto known = m_stateVarNames.find(_var.id());
		if (known != m_stateVarNames.end())
			return known->second;

		std::string name = _var.name();
		if (m_usedStateVarNames.count(name))
			if (auto const* contract = dynamic_cast<ContractDefinition const*>(_var.scope()))
				name = contract->name() + "." + _var.name();
		m_usedStateVarNames.insert(name);
		m_stateVarNames[_var.id()] = name;
		return name;
	}

	/// Stamps an arithmetic operation with whether it is inside `unchecked`.
	/// Solidity >= 0.8 panics on overflow everywhere else, and the block is
	/// inlined, so the operation is the only place left to say so.
	template<typename Op>
	mlir::Value marked(Op _op)
	{
		if (m_uncheckedDepth > 0)
			_op->setAttr("unchecked", m_builder->getUnitAttr());
		return _op.getResult();
	}

	/// `a op= b` as `a op b`, for the assignment forms.
	mlir::Value compoundResult(
		mlir::Location _loc, Token _operator, mlir::Value _current, mlir::Value _operand, mlir::Type _type)
	{
		switch (_operator)
		{
		case Token::AssignAdd:
			return marked(m_builder->create<mlir::solidity::AddOp>(_loc, _type, _current, _operand));
		case Token::AssignSub:
			return marked(m_builder->create<mlir::solidity::SubOp>(_loc, _type, _current, _operand));
		case Token::AssignMul:
			return marked(m_builder->create<mlir::solidity::MulOp>(_loc, _type, _current, _operand));
		case Token::AssignDiv:
			return marked(m_builder->create<mlir::solidity::DivOp>(_loc, _type, _current, _operand));
		case Token::AssignMod:
			return marked(m_builder->create<mlir::solidity::ModOp>(_loc, _type, _current, _operand));
		case Token::AssignBitAnd:
			return m_builder->create<mlir::solidity::AndOp>(_loc, _type, _current, _operand);
		case Token::AssignBitOr:
			return m_builder->create<mlir::solidity::OrOp>(_loc, _type, _current, _operand);
		case Token::AssignBitXor:
			return m_builder->create<mlir::solidity::XorOp>(_loc, _type, _current, _operand);
		case Token::AssignShl:
			return m_builder->create<mlir::solidity::ShlOp>(_loc, _type, _current, _operand);
		case Token::AssignSar:
			if (llvm::isa<mlir::solidity::IntType>(_type))
				return m_builder->create<mlir::solidity::SarOp>(_loc, _type, _current, _operand);
			return m_builder->create<mlir::solidity::ShrOp>(_loc, _type, _current, _operand);
		case Token::AssignShr:
			return m_builder->create<mlir::solidity::ShrOp>(_loc, _type, _current, _operand);
		default:
			return _operand;
		}
	}

	/// Marks a storage array operation with the layout the lowering needs: a
	/// dynamic array keeps its length in the slot and its data at
	/// keccak256(slot), a fixed one starts at the slot itself.
	///
	/// Only word-sized elements. An element occupying several slots - a struct,
	/// or a nested array - needs the element size threaded through as well, and
	/// guessing one slot would read the wrong place rather than fail.
	bool markStorageArray(mlir::Operation* _op, Type const* _arrayType, std::string const& _varName)
	{
		auto const* array = dynamic_cast<ArrayType const*>(_arrayType);
		if (!array)
			return false;

		// Dynamic-ness matters wherever the array lives: in storage it decides
		// whether the elements are at the slot or at keccak256(slot), and in
		// memory whether a length word precedes them. A fixed-size memory array
		// has none, so skipping one reads the wrong element.
		if (array->isDynamicallySized())
			_op->setAttr("dynamic", m_builder->getUnitAttr());

		if (array->location() != DataLocation::Storage)
			return false;
		uint64_t const elementSlots = static_cast<uint64_t>(array->baseType()->storageSize());
		if (array->baseType()->isValueType() && array->baseType()->storageBytes() <= 32)
			_op->setAttr(
				"elementBytes",
				m_builder->getI64IntegerAttr(static_cast<int64_t>(array->baseType()->storageBytes())));
		// State arrays can be recovered by name. Storage-reference locals and
		// parameters instead carry the slot in the array operand, so an empty
		// name is still a valid storage array.
		if (!_varName.empty())
			_op->setAttr("varName", m_builder->getStringAttr(_varName));
		_op->setAttr("storageArray", m_builder->getUnitAttr());
		if (elementSlots != 1)
			_op->setAttr(
				"elementSlots", m_builder->getI64IntegerAttr(static_cast<int64_t>(elementSlots)));
		if (array->isDynamicallySized())
			_op->setAttr("dynamic", m_builder->getUnitAttr());
		return true;
	}

	/// A struct in storage is named by a slot, not held as a value: a word
	/// cannot carry one. `m[k]` and a struct state variable therefore produce
	/// the place, and members are read and written through it.
	static bool isStorageStruct(Expression const& _expr)
	{
		auto const* structType = dynamic_cast<StructType const*>(_expr.annotation().type);
		return structType && structType->location() == DataLocation::Storage;
	}

	static bool isStorageReferenceType(Type const* _type)
	{
		if (dynamic_cast<MappingType const*>(_type))
			return true;
		if (auto const* array = dynamic_cast<ArrayType const*>(_type))
			return array->location() == DataLocation::Storage;
		if (auto const* structure = dynamic_cast<StructType const*>(_type))
			return structure->location() == DataLocation::Storage;
		return false;
	}

	static bool containsExternalFunction(Type const* _type)
	{
		if (auto const* function = dynamic_cast<FunctionType const*>(_type))
			return function->kind() == FunctionType::Kind::External;
		if (auto const* array = dynamic_cast<ArrayType const*>(_type))
			return containsExternalFunction(array->baseType());
		return false;
	}

	mlir::Type assignmentValueType(Type const* _type)
	{
		if (!_type)
			return mlir::solidity::UIntType::get(m_context.get(), 256);
		if (auto const* reference = dynamic_cast<ReferenceType const*>(_type);
			reference && reference->location() == DataLocation::Storage
			&& !dynamic_cast<MappingType const*>(_type))
		{
			if (auto const* structure = dynamic_cast<StructType const*>(_type);
				structure && (structure->recursive() || structure->containsNestedMapping()))
				return translateSolidityType(*_type);
			return translateSolidityType(
				*TypeProvider::withLocationIfReference(DataLocation::Memory, _type));
		}
		return translateSolidityType(*_type);
	}

	/// Which slot, counting from the struct's own, a member lives in.
	///
	std::optional<int64_t> memberSlotOffset(StructType const& _struct, std::string const& _member)
	{
		for (auto const& [slot, offset]: {_struct.storageOffsetsOfMember(_member)})
			return static_cast<int64_t>(slot);
		return std::nullopt;
	}

	void markStorageMember(
		mlir::Operation* _op, StructType const& _struct, std::string const& _member)
	{
		auto const [slot, byteOffset] = _struct.storageOffsetsOfMember(_member);
		(void)slot;
		_op->setAttr(
			"storageByteOffset",
			m_builder->getI32IntegerAttr(static_cast<int32_t>(byteOffset)));
		for (auto const& member: _struct.structDefinition().members())
			if (member->name() == _member)
			{
				_op->setAttr(
					"storageBytes",
					m_builder->getI32IntegerAttr(static_cast<int32_t>(member->type()->storageBytes())));
				break;
			}
	}

	/// An internal function pointer is a small integer, not an address.
	///
	/// The EVM has no callable value: solc uses a jump destination, which only
	/// works because it lays the code out itself. Here every function is a Yul
	/// function reached by name, so a pointer is an id and calling one goes
	/// through a generated dispatcher that turns the id back into a name.
	uint64_t functionPointerId(FunctionDefinition const& _function)
	{
		auto known = m_functionPointerIds.find(&_function);
		if (known != m_functionPointerIds.end())
			return known->second;
		uint64_t const id = m_functionPointerIds.size() + 1;
		m_functionPointerIds[&_function] = id;
		m_functionPointerOrder.push_back(&_function);
		return id;
	}

	/// The dispatcher's name for a given shape. One per distinct signature,
	/// because the call has to agree with the callee's arity on both sides.
	std::string indirectDispatcherName(size_t _arguments, size_t _results)
	{
		return "$call." + std::to_string(_arguments) + "." + std::to_string(_results);
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
		if (!function)
			return nullptr;
		// File-level functions participate in overload resolution too. They are
		// emitted into the current object, so keep the resolved declaration and
		// let resolvedCalleeName apply that object's namespace below.
		if (!dynamic_cast<ContractDefinition const*>(function->scope()))
			return function;
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

	/// Solidity's type checker has already approved implicit conversions at
	/// assignment and declaration boundaries. Make those boundaries explicit
	/// in the dialect so an integer literal cannot silently change a uint8 local
	/// (and therefore an SCF loop-carried value) into uint256.
	mlir::Value coerceValue(mlir::Location _loc, mlir::Value _value, mlir::Type _target)
	{
		if (!_value || _value.getType() == _target)
			return _value;

		// The front end keeps a one-character string literal as a literal-string
		// type even when the assignment target is bytes1.  In the MLIR dialect a
		// string literal is represented by a memory pointer, whereas bytesN is a
		// left-aligned word.  A generic convert would therefore convert the
		// pointer rather than the character.  Materialise the already-approved
		// literal conversion at the assignment boundary instead.
		if (auto targetBytes = llvm::dyn_cast<mlir::solidity::BytesType>(_target))
			if (auto literal = _value.getDefiningOp<mlir::solidity::StringLiteralOp>())
			{
				llvm::APInt word(256, 0);
				llvm::StringRef const text = literal.getValue();
				for (unsigned byte = 0; byte < 32; ++byte)
				{
					word <<= 8;
					if (byte < text.size() && byte < targetBytes.getSize())
						word |= llvm::APInt(256, static_cast<uint8_t>(text[byte]));
				}
				return m_builder->create<mlir::solidity::ConstantOp>(
					_loc,
					m_builder->getIntegerAttr(
						mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned), word),
					_target).getResult();
			}
		return m_builder->create<mlir::solidity::ConvertOp>(_loc, _target, _value).getResult();
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

	bool insideSCFRegion() const
	{
		mlir::Operation* parent = m_builder->getInsertionBlock()
			? m_builder->getInsertionBlock()->getParentOp()
			: nullptr;
		while (parent && !mlir::isa<mlir::solidity::FunctionOp>(parent))
		{
			if (mlir::isa<mlir::scf::IfOp, mlir::scf::WhileOp>(parent))
				return true;
			parent = parent->getParentOp();
		}
		return false;
	}

	mlir::solidity::ControlTransferOp emitNestedControlTransfer(
		mlir::Location _loc, llvm::StringRef _kind, mlir::ValueRange _operands = {})
	{
		return m_builder->create<mlir::solidity::ControlTransferOp>(
			_loc, m_builder->getStringAttr(_kind), _operands);
	}

	llvm::SmallVector<mlir::Value, 4> currentLoopValues() const
	{
		llvm::SmallVector<mlir::Value, 4> values;
		for (int64_t id: m_loopCarriedVarIds)
		{
			auto known = m_valueMap.find(id);
			if (known != m_valueMap.end() && known->second)
				values.push_back(known->second);
		}
		return values;
	}

	static bool endsSourceControlFlow(mlir::Block* _block)
	{
		if (!_block || _block->empty())
			return false;
		mlir::Operation& op = _block->back();
		return op.hasTrait<mlir::OpTrait::IsTerminator>()
			|| mlir::isa<mlir::solidity::ControlTransferOp>(op);
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

	mlir::Value generateLowLevelCall(
		FunctionCall const& _call,
		MemberAccess const& _member,
		mlir::Value _target,
		mlir::Value _value,
		llvm::StringRef _kind,
		mlir::Location _loc)
	{
		mlir::Value target = _target ? _target : generateSolidityExpression(_member.expression());
		if (!_value)
			_value = m_builder->create<mlir::solidity::ConstantOp>(
				_loc,
				m_builder->getIntegerAttr(m_builder->getI64Type(), 0),
				mlir::solidity::UIntType::get(m_context.get(), 256));

		mlir::Value data;
		if (!_call.arguments().empty())
			data = generateSolidityExpression(*_call.arguments().front());
		else
			data = m_builder->create<mlir::solidity::StringLiteralOp>(
				_loc,
				mlir::solidity::DynamicBytesType::get(m_context.get()),
				m_builder->getStringAttr(""));

		if (!target || !data)
			return emitUnsupported(
				_loc, mlir::solidity::BoolType::get(m_context.get()), "low-level call operand");
		auto call = m_builder->create<mlir::solidity::LowLevelCallOp>(
			_loc,
			mlir::TypeRange{
				mlir::solidity::BoolType::get(m_context.get()),
				mlir::solidity::DynamicBytesType::get(m_context.get())},
			target,
			_value,
			data,
			m_builder->getStringAttr(_kind));
		return call.getSuccess();
	}

	std::string resolvedCalleeName(FunctionCall const& _call, std::string const& _fallback)
	{
		FunctionDefinition const* function = resolvedCallee(_call);
		if (!function)
			return _fallback;
		if (function->isConstructor())
		{
			auto const* contract = dynamic_cast<ContractDefinition const*>(function->scope());
			return contract ? contract->name() + ".constructor" : _fallback;
		}
		std::string const emitted = emittedNameOf(*function);
		// Overridden base implementations are already emitted with an explicit
		// contract prefix. Ordinary and overloaded functions are local symbols in
		// the sol contract op, but call references are module-qualified so the Yul
		// conversion can resolve them unambiguously.
		if (emitted.find('.') != std::string::npos)
			return emitted;
		// An inherited implementation emitted under a bare name belongs to the
		// most-derived object's namespace (for example Child.f), not the source
		// contract where it was first declared (Base.f). A shadowed base body was
		// handled above because its emitted name is already explicitly qualified.
		return m_mostDerivedContract ? m_mostDerivedContract->name() + "." + emitted : _fallback;
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
	void emitReturn(
		mlir::Location _loc,
		llvm::SmallVector<mlir::Value, 4> _values,
		std::set<unsigned> const& _storageResults = {})
	{
		mlir::Operation* parent = m_builder->getInsertionBlock()->getParentOp();
		while (parent && !mlir::isa<mlir::solidity::FunctionOp>(parent))
			parent = parent->getParentOp();
		if (auto enclosing = mlir::dyn_cast_or_null<mlir::solidity::FunctionOp>(parent))
		{
			mlir::ArrayRef<mlir::Type> const results = enclosing.getResultTypes();
			for (size_t i = _values.size(); i < results.size(); ++i)
			{
				// `returns (uint256 x)` with no explicit return - and `return;`
				// in the same function - returns whatever x holds. Padding with
				// a placeholder answered zero however x was assigned.
				mlir::Value named;
				if (i < m_returnParameters.size() && m_returnParameters[i])
				{
					auto known = m_valueMap.find(m_returnParameters[i]->id());
					if (known != m_valueMap.end() && known->second && known->second.getType() == results[i])
						named = known->second;
				}
				_values.push_back(named ? named : emitUnsupported(_loc, results[i], "return value not generated"));
			}
			_values.resize(std::min(_values.size(), results.size()));

			// A string literal returned where fixed bytes are declared is a
			// word, not a memory value - `return hex"01"` from a bytes1
			// function is 0x01 at the top of the word. The conversion is
			// implicit in the source, so the literal carries the string type
			// and only the declared result says otherwise.
			for (size_t i = 0; i < _values.size(); ++i)
			{
				auto fixedBytes = llvm::dyn_cast<mlir::solidity::BytesType>(results[i]);
				if (!fixedBytes || _values[i].getType() == results[i])
					continue;
				auto literal = _values[i].getDefiningOp<mlir::solidity::StringLiteralOp>();
				if (!literal)
					continue;

				llvm::StringRef const text = literal.getValue();
				llvm::APInt word(256, 0);
				for (unsigned byte = 0; byte < 32; ++byte)
				{
					word <<= 8;
					if (byte < text.size() && byte < fixedBytes.getSize())
						word |= llvm::APInt(256, static_cast<uint8_t>(text[byte]));
				}
				mlir::OpBuilder::InsertionGuard guard(*m_builder);
				m_builder->setInsertionPointAfter(literal);
				_values[i] = m_builder->create<mlir::solidity::ConstantOp>(
					_loc,
					m_builder->getIntegerAttr(
						mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned), word),
					results[i]).getResult();
			}
		}
		if (insideSCFRegion())
		{
			auto transfer = emitNestedControlTransfer(_loc, "return", _values);
			if (!_storageResults.empty())
			{
				std::vector<mlir::Attribute> indices;
				for (unsigned index: _storageResults)
					indices.push_back(m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
				transfer->setAttr("storage_results", m_builder->getArrayAttr(indices));
			}
		}
		else
		{
			auto ret = m_builder->create<mlir::solidity::ReturnOp>(_loc, mlir::ValueRange{_values});
			if (!_storageResults.empty())
			{
				std::vector<mlir::Attribute> indices;
				for (unsigned index: _storageResults)
					indices.push_back(m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
				ret->setAttr("storage_results", m_builder->getArrayAttr(indices));
			}
		}
	}

	static constexpr char const* kInitializerName = "init";

	std::map<FunctionDefinition const*, uint64_t> m_functionPointerIds;
	std::vector<FunctionDefinition const*> m_functionPointerOrder;
	std::set<std::pair<size_t, size_t>> m_indirectShapes;
	std::map<FunctionDefinition const*, std::string> m_emittedFunctionNames;

	std::vector<VariableDeclaration const*> m_returnParameters;
	std::function<void()> m_modifierContinuation;
	llvm::SmallVector<mlir::Value, 4> m_modifierResults;
	unsigned m_uncheckedDepth = 0;

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
				// A literal whose type is a fixed-bytes one is a word, not a
				// memory value: `hex"01"` as bytes1 is 0x01 at the top of the
				// word. Building it as a string put a pointer there instead.
				if (auto* fixedBytes = dynamic_cast<FixedBytesType const*>(_expr.annotation().type))
				{
					llvm::APInt word(256, 0);
					for (unsigned i = 0; i < 32; ++i)
					{
						word <<= 8;
						if (i < literal->value().size() && i < fixedBytes->numBytes())
							word |= llvm::APInt(256, static_cast<uint8_t>(literal->value()[i]));
					}
					auto attr = m_builder->getIntegerAttr(
						mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned), word);
					return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, type).getResult();
				}

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
					if (varDecl->isConstant() && varDecl->value())
						if (mlir::Value substituted = generateSolidityExpression(*varDecl->value()))
							return substituted;
					auto load = m_builder->create<mlir::solidity::LoadStateVarOp>(
						loc, type, stateVarName(*varDecl));
					if (dynamic_cast<MappingType const*>(varDecl->type())
						|| (dynamic_cast<ArrayType const*>(varDecl->type())
							&& dynamic_cast<ArrayType const*>(varDecl->type())->location() == DataLocation::Storage)
						|| (dynamic_cast<StructType const*>(varDecl->type())
							&& dynamic_cast<StructType const*>(varDecl->type())->location() == DataLocation::Storage))
						load->setAttr("asReference", m_builder->getUnitAttr());
					return load;
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
			else if (auto* referenced
					 = dynamic_cast<FunctionDefinition const*>(ident->annotation().referencedDeclaration))
			{
				// A function named where a value is wanted is a pointer to it.
				// This used to fall through to the zero placeholder, so every
				// pointer was null and calling one did nothing.
				return m_builder->create<mlir::solidity::ConstantOp>(
					loc,
					m_builder->getIntegerAttr(
						m_builder->getI64Type(), static_cast<int64_t>(functionPointerId(*referenced))),
					mlir::solidity::UIntType::get(m_context.get(), 256));
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
			// Constant rational expressions are evaluated by Solidity's type
			// system with arbitrary precision.  Lowering their syntax directly to
			// EVM words changes expressions such as `2**256 / 32`: EXP wraps the
			// intermediate to zero even though the final constant, 2**251, fits a
			// uint256.  Preserve the front end's folded value just like the legacy
			// and IR generators do.
			if (auto const* rational
				= dynamic_cast<RationalNumberType const*>(_expr.annotation().type))
			{
				auto type = translateSolidityType(*_expr.annotation().type);
				auto attr = m_builder->getIntegerAttr(
					mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
					llvm::APInt(256, rational->literalValue(nullptr).str(), 10));
				return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, type).getResult();
			}

			auto lhs = generateSolidityExpression(binOp->leftExpression());

			// `&&` and `||` are control flow, not eager boolean arithmetic. Put the
			// RHS in only the arm that evaluates it, and carry any local assignments
			// it performs out as additional SSA results.
			if (binOp->getOperator() == langutil::Token::And
				|| binOp->getOperator() == langutil::Token::Or)
			{
				auto boolType = mlir::solidity::BoolType::get(m_context.get());
				if (!lhs)
					return emitUnsupported(loc, boolType, "short-circuit expression with unsupported left operand");
				lhs = typedOperand(loc, lhs, boolType, "left short-circuit operand not generated as a boolean");

				std::set<int64_t> modifiedSet;
				collectExpressionModifiedVariables(binOp->rightExpression(), modifiedSet);
				std::vector<int64_t> modified;
				llvm::SmallVector<mlir::Value, 4> entry;
				llvm::SmallVector<mlir::Type, 4> resultTypes{boolType};
				for (int64_t id: modifiedSet)
					if (m_valueMap.count(id) && m_valueMap[id])
					{
						modified.push_back(id);
						entry.push_back(m_valueMap[id]);
						resultTypes.push_back(m_valueMap[id].getType());
					}

				auto savedValues = m_valueMap;
				mlir::Value condition
					= m_builder->create<mlir::solidity::ToI1Op>(loc, m_builder->getI1Type(), lhs);
				auto branch = m_builder->create<mlir::scf::IfOp>(
					loc, mlir::TypeRange{resultTypes}, condition, true);

				auto emitArm = [&](mlir::Block& _block, bool _evaluateRight, bool _constant) {
					mlir::OpBuilder::InsertionGuard guard(*m_builder);
					m_builder->setInsertionPointToEnd(&_block);
					m_valueMap = savedValues;
					mlir::Value value;
					if (_evaluateRight)
						value = generateSolidityExpression(binOp->rightExpression());
					else
						value = m_builder->create<mlir::solidity::ConstantOp>(
							loc, m_builder->getBoolAttr(_constant), boolType);
					value = typedOperand(loc, value, boolType, "right short-circuit operand not generated as a boolean");

					llvm::SmallVector<mlir::Value, 4> yielded{value};
					for (size_t i = 0; i < modified.size(); ++i)
					{
						mlir::Value current = m_valueMap.count(modified[i]) ? m_valueMap[modified[i]] : mlir::Value();
						yielded.push_back(
							current && current.getType() == entry[i].getType() ? current : entry[i]);
					}
					m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{yielded});
				};

				bool const isAnd = binOp->getOperator() == langutil::Token::And;
				emitArm(branch.getThenRegion().front(), isAnd, true);
				emitArm(branch.getElseRegion().front(), !isAnd, false);
				m_builder->setInsertionPointAfter(branch.getOperation());
				m_valueMap = savedValues;
				for (size_t i = 0; i < modified.size(); ++i)
					m_valueMap[modified[i]] = branch->getResult(i + 1);
				return branch->getResult(0);
			}

			auto rhs = generateSolidityExpression(binOp->rightExpression());

			// Check if either operand is null
			if (!lhs || !rhs)
			{
				auto type = translateSolidityType(*_expr.annotation().type);
				return emitUnsupported(loc, type, "binary op with unsupported operand");
			}

			// A one-character string literal is implicitly comparable with
			// bytes1. Keep the literal's contents, not its temporary memory
			// pointer, at that comparison boundary. Do not apply this to shifts:
			// their right operand is an integer count, not another bytes value.
			Token const binaryToken = binOp->getOperator();
			bool const comparison = binaryToken == Token::LessThan
				|| binaryToken == Token::GreaterThan
				|| binaryToken == Token::Equal
				|| binaryToken == Token::NotEqual
				|| binaryToken == Token::LessThanOrEqual
				|| binaryToken == Token::GreaterThanOrEqual;
			if (comparison && llvm::isa<mlir::solidity::BytesType>(lhs.getType())
				&& lhs.getType() != rhs.getType())
				rhs = coerceValue(loc, rhs, lhs.getType());
			else if (comparison && llvm::isa<mlir::solidity::BytesType>(rhs.getType())
				&& lhs.getType() != rhs.getType())
				lhs = coerceValue(loc, lhs, rhs.getType());

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
				return marked(m_builder->create<mlir::solidity::AddOp>(loc, resultType, lhs, rhs));
			}
			case langutil::Token::Sub:
			{
				return marked(m_builder->create<mlir::solidity::SubOp>(loc, resultType, lhs, rhs));
			}
			case langutil::Token::Mul:
			{
				return marked(m_builder->create<mlir::solidity::MulOp>(loc, resultType, lhs, rhs));
			}
			case langutil::Token::Div:
			{
				return marked(m_builder->create<mlir::solidity::DivOp>(loc, resultType, lhs, rhs));
			}
			case langutil::Token::Mod:
			{
				return marked(m_builder->create<mlir::solidity::ModOp>(loc, resultType, lhs, rhs));
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

			// `array.push() = value` is an lvalue whose write and length update
			// are one storage operation. A generic FunctionCall lvalue has no
			// identifier/index destination and previously disappeared entirely.
			if (assignment->assignmentOperator() == Token::Assign)
				if (auto const* pushCall = dynamic_cast<FunctionCall const*>(
					&assignment->leftHandSide()))
					if (auto const* member = dynamic_cast<MemberAccess const*>(&pushCall->expression());
						member && member->memberName() == "push" && pushCall->arguments().empty())
						if (auto const* arrayType = dynamic_cast<ArrayType const*>(
							member->expression().annotation().type);
							arrayType && arrayType->location() == DataLocation::Storage
							&& arrayType->isDynamicallySized())
						{
							mlir::Value base = generateSolidityExpression(member->expression());
							mlir::Value assigned = generateSolidityExpression(assignment->rightHandSide());
							assigned = coerceValue(loc, assigned, assignmentValueType(arrayType->baseType()));
							if (arrayType->isByteArray())
								m_builder->create<mlir::solidity::StorageBytesPushOp>(loc, base, assigned);
							else
							{
								auto push = m_builder->create<mlir::solidity::ArrayPushOp>(loc, base, assigned);
								if (isStorageReferenceType(assignment->rightHandSide().annotation().type))
									push->setAttr("storageValue", m_builder->getUnitAttr());
								if (!markStorageArray(
										push.getOperation(), arrayType,
										extractMappingVarName(member->expression())))
									return emitUnsupported(loc, assigned.getType(), "push assignment onto an unnamed storage array");
							}
							return assigned;
						}

			// Evaluate all tuple RHS components before performing any write.
			// This implements simultaneous multi-assignment for the common
			// identifier form and, importantly, does not collapse `(a, b)` to
			// only its first expression.
			if (assignment->assignmentOperator() == Token::Assign)
				if (auto const* targets = dynamic_cast<TupleExpression const*>(
					&assignment->leftHandSide()))
					if (auto const* sources = dynamic_cast<TupleExpression const*>(
						&assignment->rightHandSide());
						sources && !targets->isInlineArray() && !sources->isInlineArray()
						&& targets->components().size() == sources->components().size())
					{
						bool identifiersOnly = true;
						for (auto const& target: targets->components())
							if (target && !dynamic_cast<Identifier const*>(target.get()))
								identifiersOnly = false;
						if (identifiersOnly)
						{
							llvm::SmallVector<mlir::Value, 4> values;
							for (auto const& source: sources->components())
								values.push_back(source
									? generateSolidityExpression(*source) : mlir::Value());
							for (auto [index, target]: llvm::enumerate(targets->components()))
							{
								if (!target)
									continue;
								auto const* identifier = dynamic_cast<Identifier const*>(target.get());
								auto const* variable = dynamic_cast<VariableDeclaration const*>(
									identifier->annotation().referencedDeclaration);
								if (!variable || !values[index])
									continue;
								mlir::Type targetType = variable->isStateVariable()
									? assignmentValueType(variable->type())
									: translateSolidityType(*variable->type());
								mlir::Value assigned = coerceValue(loc, values[index], targetType);
								if (variable->isStateVariable())
								{
									auto store = m_builder->create<mlir::solidity::StoreStateVarOp>(
										loc, stateVarName(*variable), assigned);
									if (sources->components()[index]
										&& isStorageReferenceType(sources->components()[index]->annotation().type))
										store->setAttr("storageValue", m_builder->getUnitAttr());
								}
								else
									m_valueMap[variable->id()] = assigned;
							}
							return values.empty() ? mlir::Value() : values.front();
						}
					}

			// `s.member = v` where s is in storage writes through the slot. It
			// used to fall to a generic path that produced the member's value
			// and then had nowhere to put anything, so the write was lost.
			if (auto* target = dynamic_cast<MemberAccess const*>(&assignment->leftHandSide()))
				if (auto* structType
					= dynamic_cast<StructType const*>(target->expression().annotation().type))
					if (structType->location() == DataLocation::Storage)
					{
						std::optional<int64_t> const offset
							= memberSlotOffset(*structType, target->memberName());
						mlir::Value slot = generateSolidityExpression(target->expression());
						mlir::Value assigned = generateSolidityExpression(assignment->rightHandSide());
						if (!offset || !slot || !assigned)
							return emitUnsupported(
								loc,
								translateSolidityType(*_expr.annotation().type),
								"assignment to struct member '" + target->memberName() + "'");

						assigned = coerceValue(
							loc, assigned, assignmentValueType(assignment->leftHandSide().annotation().type));
						if (assignment->assignmentOperator() != Token::Assign)
						{
							auto load = m_builder->create<mlir::solidity::StorageMemberLoadOp>(
								loc, assigned.getType(), slot, m_builder->getI64IntegerAttr(*offset));
							markStorageMember(load.getOperation(), *structType, target->memberName());
							mlir::Value current = load.getResult();
							assigned = compoundResult(
								loc, assignment->assignmentOperator(), current, assigned, assigned.getType());
						}

						auto store = m_builder->create<mlir::solidity::StorageMemberStoreOp>(
							loc, slot, assigned, m_builder->getI64IntegerAttr(*offset));
						markStorageMember(store.getOperation(), *structType, target->memberName());
						if (isStorageReferenceType(assignment->rightHandSide().annotation().type))
							store->setAttr("storageValue", m_builder->getUnitAttr());
						return assigned;
					}

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
								loc, translateSolidityType(*varDecl->type()), stateVarName(*varDecl));
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
						mlir::Value baseSlot = mappingBaseSlot(*indexAccess);
						if (baseSlot)
							keys.insert(keys.begin(), baseSlot);
						auto access = m_builder->create<mlir::solidity::MappingAccessOp>(
							loc, valueType, m_builder->getStringAttr(varName), keys);
						if (baseSlot)
							access->setAttr("baseSlotOperand", m_builder->getUnitAttr());
						markMappingEnumKeyLimits(access.getOperation(), *indexAccess);
						currentValue = access.getResult();
					}
					else
						currentValue = generateSolidityExpression(assignment->leftHandSide());
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
				// Compound assignment performs the implicit conversion at the
				// assignment boundary before applying the operator. This matters for
				// bytesN, whose value is left-aligned while an integer literal is
				// initially represented in the low bits of the word.
				rightValue = coerceValue(loc, rightValue, currentValue.getType());

				auto resultType = translateSolidityType(*_expr.annotation().type);
				value = compoundResult(
					loc, assignment->assignmentOperator(), currentValue, rightValue, resultType);
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
					mlir::Type targetType = varDecl->isStateVariable()
						? assignmentValueType(varDecl->type())
						: translateSolidityType(*varDecl->type());
					auto const* sourceReference = dynamic_cast<ReferenceType const*>(
						assignment->rightHandSide().annotation().type);
					auto const* targetReference = dynamic_cast<ReferenceType const*>(varDecl->type());
					if (!varDecl->isStateVariable() && sourceReference && targetReference
						&& sourceReference->location() == DataLocation::Storage
						&& targetReference->location() == DataLocation::Memory)
						value = m_builder->create<mlir::solidity::StorageToMemoryOp>(
							loc, targetType, value).getResult();
					else
						value = coerceValue(loc, value, targetType);
					if (varDecl->isStateVariable())
					{
						auto store = m_builder->create<mlir::solidity::StoreStateVarOp>(
							loc, stateVarName(*varDecl), value);
						if (isStorageReferenceType(assignment->rightHandSide().annotation().type))
							store->setAttr("storageValue", m_builder->getUnitAttr());
					}
					else
					{
						m_valueMap[varDecl->id()] = value;
					}
				}
			}
			else if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&assignment->leftHandSide()))
			{
				value = coerceValue(
					loc, value, assignmentValueType(assignment->leftHandSide().annotation().type));
				// Handle store to mapping elements: m[k] = v or m[k] += v
				if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
				{
					std::string varName;
					auto keys = collectMappingKeys(*indexAccess, varName);
					auto store = createMappingStoreOp(
						loc, varName, keys, value, mappingBaseSlot(*indexAccess));
					markMappingEnumKeyLimits(store.getOperation(), *indexAccess);
					if (isStorageReferenceType(assignment->rightHandSide().annotation().type))
						store->setAttr("storageValue", m_builder->getUnitAttr());
				}
				else if (dynamic_cast<ArrayType const*>(indexAccess->baseExpression().annotation().type))
				{
					auto base = generateSolidityExpression(indexAccess->baseExpression());
					auto index = generateSolidityExpression(*indexAccess->indexExpression());
					std::string varName = extractMappingVarName(indexAccess->baseExpression());
					if (!mlir::isa<
						mlir::solidity::ArrayType,
						mlir::solidity::DynamicBytesType,
						mlir::solidity::StringType>(base.getType()))
						return emitUnsupported(loc, value.getType(), "store into an unsupported array expression");
					auto storeOp = m_builder->create<mlir::solidity::ArrayStoreOp>(loc, base, index, value);
					if (isStorageReferenceType(assignment->rightHandSide().annotation().type))
						storeOp->setAttr("storageValue", m_builder->getUnitAttr());
					markStorageArray(
						storeOp.getOperation(),
						indexAccess->baseExpression().annotation().type,
						extractMappingVarName(indexAccess->baseExpression()));
					if (!varName.empty())
						storeOp->setAttr("varName", m_builder->getStringAttr(varName));
				}
			}
			else if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&assignment->leftHandSide()))
			{
				value = coerceValue(
					loc, value, translateSolidityType(*assignment->leftHandSide().annotation().type));
				if (auto const* structure
					= dynamic_cast<StructType const*>(memberAccess->expression().annotation().type);
					structure && structure->location() == DataLocation::Memory)
				{
					mlir::Value object = generateSolidityExpression(memberAccess->expression());
					int64_t fieldIndex = 0;
					for (auto [index, member]: llvm::enumerate(structure->structDefinition().members()))
						if (member->name() == memberAccess->memberName())
						{
							fieldIndex = static_cast<int64_t>(index);
							break;
						}
					m_builder->create<mlir::solidity::MemoryMemberStoreOp>(
						loc, object, value, m_builder->getI64IntegerAttr(fieldIndex));
					return value;
				}
				// Handle store to struct field in mapping: m[k].field = v or m[k].field += v
				if (auto* indexAccess = dynamic_cast<IndexAccess const*>(&memberAccess->expression()))
				{
					if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
					{
						std::string varName;
						auto keys = collectMappingKeys(*indexAccess, varName);
						auto store = createMappingStoreOp(
							loc, varName, keys, value, mappingBaseSlot(*indexAccess));
						markMappingEnumKeyLimits(store.getOperation(), *indexAccess);
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
				if (auto member = operand.getDefiningOp<mlir::solidity::StorageMemberLoadOp>())
					m_builder->create<mlir::solidity::StorageMemberStoreOp>(
						loc, member.getSlot(), result, member.getOffsetAttr());
				else if (auto* ident = dynamic_cast<Identifier const*>(&unaryOp->subExpression()))
				{
					if (auto* varDecl
						= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (varDecl->isStateVariable())
						{
							m_builder->create<mlir::solidity::StoreStateVarOp>(loc, stateVarName(*varDecl), result);
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
						auto store = createMappingStoreOp(
							loc, varName, keys, result, mappingBaseSlot(*indexAccess));
						markMappingEnumKeyLimits(store.getOperation(), *indexAccess);
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
				if (auto member = operand.getDefiningOp<mlir::solidity::StorageMemberLoadOp>())
					m_builder->create<mlir::solidity::StorageMemberStoreOp>(
						loc, member.getSlot(), result, member.getOffsetAttr());
				else if (auto* ident = dynamic_cast<Identifier const*>(&unaryOp->subExpression()))
				{
					if (auto* varDecl
						= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (varDecl->isStateVariable())
						{
							m_builder->create<mlir::solidity::StoreStateVarOp>(loc, stateVarName(*varDecl), result);
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
						auto store = createMappingStoreOp(
							loc, varName, keys, result, mappingBaseSlot(*indexAccess));
						markMappingEnumKeyLimits(store.getOperation(), *indexAccess);
					}
				}

				// Return old value for post-decrement, new value for pre-decrement
				return unaryOp->isPrefixOperation() ? result : operand;
			}
			case langutil::Token::Delete:
			{
				// `delete x` writes the zero value back to x. Yielding a zero
				// and storing it nowhere made every delete a no-op that looked
				// like it worked - a state variable kept its value, and so did
				// a memory array.
				auto zeroAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
				Expression const& target = withoutParentheses(unaryOp->subExpression());

				if (auto const* ident = dynamic_cast<Identifier const*>(&target))
					if (auto const* varDecl
						= dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						auto varType = translateSolidityType(*varDecl->type());
						// A fixed memory array is zeroed in place: the variable
						// still names the same words.
						if (auto const* array = dynamic_cast<ArrayType const*>(varDecl->type()))
							if (array->location() == DataLocation::Memory && !array->isDynamicallySized()
								&& m_valueMap.count(varDecl->id()))
							{
								mlir::Value pointer = m_valueMap[varDecl->id()];
								auto word = mlir::solidity::UIntType::get(m_context.get(), 256);
								for (uint64_t i = 0; i < static_cast<uint64_t>(array->length()); ++i)
								{
									mlir::Value index = m_builder->create<mlir::solidity::ConstantOp>(
										loc, m_builder->getIntegerAttr(m_builder->getI64Type(), int64_t(i)), word);
									mlir::Value zero = m_builder->create<mlir::solidity::ConstantOp>(
										loc, zeroAttr, word);
									auto store = m_builder->create<mlir::solidity::ArrayStoreOp>(
										loc, pointer, index, zero);
									markStorageArray(store.getOperation(), varDecl->type(), "");
								}
								return pointer;
							}

						mlir::Value zero
							= m_builder->create<mlir::solidity::ConstantOp>(loc, zeroAttr, varType);
						if (varDecl->isStateVariable())
						{
							auto store = m_builder->create<mlir::solidity::StoreStateVarOp>(
								loc, stateVarName(*varDecl), zero);
							if (auto const* array = dynamic_cast<ArrayType const*>(varDecl->type());
								array && !array->isByteArrayOrString())
								store->setAttr("deleteValue", m_builder->getUnitAttr());
							else if (dynamic_cast<StructType const*>(varDecl->type()))
								store->setAttr("deleteValue", m_builder->getUnitAttr());
						}
						else
							m_valueMap[varDecl->id()] = zero;
						return zero;
					}

				if (auto const* indexAccess = dynamic_cast<IndexAccess const*>(&target))
				{
					Type const* targetType = target.annotation().type;
					if (auto const* reference = dynamic_cast<ReferenceType const*>(targetType);
						reference && reference->location() == DataLocation::Memory)
					{
						mlir::Value aggregate = generateSolidityExpression(target);
						m_builder->create<mlir::solidity::MemoryClearOp>(loc, aggregate);
						return aggregate;
					}
					mlir::Value base = generateSolidityExpression(indexAccess->baseExpression());
					mlir::Value index = indexAccess->indexExpression()
						? generateSolidityExpression(*indexAccess->indexExpression()) : mlir::Value();
					mlir::Type valueType = assignmentValueType(targetType);
					mlir::Value zero = m_builder->create<mlir::solidity::ConstantOp>(
						loc, zeroAttr, valueType);
					if (base && index)
					{
						auto store = m_builder->create<mlir::solidity::ArrayStoreOp>(
							loc, base, index, zero);
						markStorageArray(
							store.getOperation(),
							indexAccess->baseExpression().annotation().type,
							extractMappingVarName(indexAccess->baseExpression()));
						return zero;
					}
				}

				return emitUnsupported(loc, resultType, "delete of an unsupported target");
			}
			case langutil::Token::Sub:
			{
				// A negated constant is a constant. Emitting `0 - 7` instead
				// made it a subtraction on a type that reads as unsigned, so
				// the overflow check saw it underflow - which it does, as
				// unsigned. The value is known here; there is nothing to
				// compute at run time.
				if (auto const* rational
					= dynamic_cast<RationalNumberType const*>(_expr.annotation().type))
				{
					auto attr = m_builder->getIntegerAttr(
						mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
						llvm::APInt(256, rational->literalValue(nullptr).str(), 10));
					return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, resultType).getResult();
				}

				auto zero = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
				auto zeroValue = m_builder->create<mlir::solidity::ConstantOp>(loc, zero, resultType);

				return marked(m_builder->create<mlir::solidity::SubOp>(loc, resultType, zeroValue, operand));
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
							loc, mlir::solidity::DynamicBytesType::get(m_context.get()));
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
					if (memberName == "runtimeCode" || memberName == "creationCode")
					{
						if (auto* contractType = dynamic_cast<ContractType const*>(argument))
							return m_builder->create<mlir::solidity::ContractCodeOp>(
								loc,
								mlir::solidity::DynamicBytesType::get(m_context.get()),
								m_builder->getStringAttr(contractType->contractDefinition().name()),
								m_builder->getBoolAttr(memberName == "creationCode")).getResult();
					}
					else if (memberName == "interfaceId")
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

			// `super.f` or `Base.f` named where a value is wanted is a pointer,
			// and it has to resolve to the same implementation the equivalent
			// call would reach - `super` steps along the linearisation rather
			// than binding to the override that is asking.
			if (auto* referenced
				= dynamic_cast<FunctionDefinition const*>(memberAccess->annotation().referencedDeclaration))
				if (auto* memberType = dynamic_cast<FunctionType const*>(memberAccess->annotation().type))
				{
					if (memberType->kind() == FunctionType::Kind::Internal)
					{
						FunctionDefinition const* target = referenced;
						if (ContractDefinition const* from = superSearchStart(*memberAccess))
							if (FunctionDefinition const* next = resolveSuper(*referenced, *from))
								target = next;
						return m_builder->create<mlir::solidity::ConstantOp>(
							loc,
							m_builder->getIntegerAttr(
								m_builder->getI64Type(), static_cast<int64_t>(functionPointerId(*target))),
							mlir::solidity::UIntType::get(m_context.get(), 256)).getResult();
					}
					if (memberType->kind() == FunctionType::Kind::External && memberType->hasDeclaration())
					{
						mlir::Value target = generateSolidityExpression(memberAccess->expression());
						if (!target)
							return emitUnsupported(
								loc,
								mlir::solidity::UIntType::get(m_context.get(), 256),
								"external function reference target");
						uint32_t const selector
							= util::selectorFromSignatureU32(memberType->externalSignature());
						return m_builder->create<mlir::solidity::ExternalFunctionRefOp>(
							loc,
							mlir::solidity::UIntType::get(m_context.get(), 256),
							target,
							m_builder->getI32IntegerAttr(selector)).getResult();
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
			if (memberName == "length")
				if (auto const* fixedBytes = dynamic_cast<FixedBytesType const*>(baseType))
					return m_builder->create<mlir::solidity::ConstantOp>(
						loc,
						m_builder->getIntegerAttr(
							m_builder->getI64Type(), static_cast<int64_t>(fixedBytes->numBytes())),
						mlir::solidity::UIntType::get(m_context.get(), 256)).getResult();

			// Handle array.length
			if (auto* arrayType = dynamic_cast<ArrayType const*>(baseType))
			{
				if (memberName == "length")
				{
					// The Solidity type says array, but a dropped sub-expression
					// leaves a plain word in its place. Building the op anyway
					// produces IR the verifier rejects, which loses the whole
					// contract instead of just the feature that was dropped.
					// `string` and `bytes` are a pointer to [length][data...],
					// so their length is the word at the pointer rather than
					// anything the array ops describe.
					if (mlir::isa<mlir::solidity::StringType, mlir::solidity::DynamicBytesType>(base.getType()))
					{
						auto length = m_builder->create<mlir::solidity::MemoryLengthOp>(
							loc, mlir::solidity::UIntType::get(m_context.get(), 256), base);
						if (arrayType->location() == DataLocation::Storage)
						{
							length->setAttr("storageArray", m_builder->getUnitAttr());
							std::string const name = extractMappingVarName(memberAccess->expression());
							if (!name.empty())
								length->setAttr("varName", m_builder->getStringAttr(name));
						}
						return length.getResult();
					}
					if (!mlir::isa<mlir::solidity::ArrayType>(base.getType()))
						return emitUnsupported(
							loc,
							translateSolidityType(*_expr.annotation().type),
							"length of an unsupported array expression");
					{
						auto lengthOp = m_builder->create<mlir::solidity::ArrayLengthOp>(
							loc, mlir::solidity::UIntType::get(m_context.get(), 256), base);
						markStorageArray(
							lengthOp.getOperation(),
							memberAccess->expression().annotation().type,
							extractMappingVarName(memberAccess->expression()));
						return lengthOp.getResult();
					}
				}
			}
			// Handle struct member access
			else if (auto* structType = dynamic_cast<StructType const*>(baseType))
			{
				if (structType->location() == DataLocation::Storage)
				{
					std::optional<int64_t> const offset = memberSlotOffset(*structType, memberName);
					if (!offset)
						return emitUnsupported(
							loc,
							translateSolidityType(*_expr.annotation().type),
							"struct member '" + memberName + "' shares a slot with another");
					auto member = m_builder->create<mlir::solidity::StorageMemberLoadOp>(
						loc, translateSolidityType(*_expr.annotation().type), base,
						m_builder->getI64IntegerAttr(*offset));
					markStorageMember(member.getOperation(), *structType, memberName);
					if (isStorageReferenceType(_expr.annotation().type))
						member->setAttr("asReference", m_builder->getUnitAttr());
					return member.getResult();
				}

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
						mlir::solidity::DynamicBytesType::get(m_context.get()),
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
			if (auto const* builtinType = dynamic_cast<FunctionType const*>(callee.annotation().type))
				if ((builtinType->kind() == FunctionType::Kind::Wrap
						|| builtinType->kind() == FunctionType::Kind::Unwrap)
					&& funcCall->arguments().size() == 1)
				{
					mlir::Value value = generateSolidityExpression(*funcCall->arguments().front());
					auto resultType = translateSolidityType(*_expr.annotation().type);
					if (!value)
						return emitUnsupported(loc, resultType, "user-defined value type conversion");
					return value.getType() == resultType
						? value
						: m_builder->create<mlir::solidity::ConvertOp>(loc, resultType, value).getResult();
				}
			// Handle .call{value: amount}(data) — FunctionCallOptions wrapping.
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

						return generateLowLevelCall(
							*funcCall, *memberAccess, address, callValue, "call", loc);
					}
				}
			}

			// Handle member function calls first (e.g., array.push(), .transfer())
			if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&callee))
			{
				if (memberAccess->memberName() == "call"
					|| memberAccess->memberName() == "staticcall"
					|| memberAccess->memberName() == "delegatecall")
					return generateLowLevelCall(
						*funcCall,
						*memberAccess,
						mlir::Value(),
						mlir::Value(),
						memberAccess->memberName(),
						loc);

				// A user-defined value type is a zero-cost wrapper. Both static
				// builtins preserve the stack representation; only the source-level
				// type changes. This also covers qualified forms such as C.T.wrap(x),
				// where the declaration is attached to a nested MemberAccess.
				Declaration const* typeDeclaration = nullptr;
				if (auto const* identifier = dynamic_cast<Identifier const*>(&memberAccess->expression()))
					typeDeclaration = identifier->annotation().referencedDeclaration;
				else if (auto const* qualifier = dynamic_cast<MemberAccess const*>(&memberAccess->expression()))
					typeDeclaration = qualifier->annotation().referencedDeclaration;
				if (dynamic_cast<UserDefinedValueTypeDefinition const*>(typeDeclaration)
					&& (memberAccess->memberName() == "wrap" || memberAccess->memberName() == "unwrap")
					&& funcCall->arguments().size() == 1)
				{
					mlir::Value value = generateSolidityExpression(*funcCall->arguments().front());
					auto resultType = translateSolidityType(*_expr.annotation().type);
					if (!value)
						return emitUnsupported(loc, resultType, "user-defined value type conversion");
					if (value.getType() == resultType)
						return value;
					return m_builder->create<mlir::solidity::ConvertOp>(loc, resultType, value).getResult();
				}

				// Don't generate base for abi.* calls or contract/library calls
				bool skipBaseGen = false;
				if (auto* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
				{
					skipBaseGen = (baseIdent->name() == "abi");
					if (!skipBaseGen && dynamic_cast<ContractDefinition const*>(baseIdent->annotation().referencedDeclaration))
						skipBaseGen = true;
				}
				else if (dynamic_cast<ElementaryTypeNameExpression const*>(&memberAccess->expression()))
					// Static builtins such as string.concat and bytes.concat have no
					// runtime receiver to evaluate.
					skipBaseGen = true;

				if (!skipBaseGen)
				{
					auto base = generateSolidityExpression(memberAccess->expression());
					std::string memberName = memberAccess->memberName();
					auto baseType = memberAccess->expression().annotation().type;

					// Handle dynamic storage array push/pop, including recursive values.
					if (auto* arrayType = dynamic_cast<ArrayType const*>(baseType))
					{
						if (arrayType->location() == DataLocation::Storage
							&& arrayType->isByteArray()
							&& memberName == "push" && funcCall->arguments().empty())
						{
							auto resultType = translateSolidityType(*_expr.annotation().type);
							return m_builder->create<mlir::solidity::StorageBytesPushEmptyOp>(
								loc, resultType, base).getResult();
						}
						if (arrayType->location() == DataLocation::Storage
							&& arrayType->isByteArray()
							&& memberName == "push" && funcCall->arguments().size() == 1)
						{
							mlir::Value value = generateSolidityExpression(*funcCall->arguments().front());
							value = coerceValue(loc, value, assignmentValueType(arrayType->baseType()));
							m_builder->create<mlir::solidity::StorageBytesPushOp>(loc, base, value);
							return base;
						}
						if (arrayType->location() == DataLocation::Storage
							&& arrayType->isByteArray()
							&& memberName == "pop" && funcCall->arguments().empty())
						{
							m_builder->create<mlir::solidity::StorageBytesPopOp>(loc, base);
							return nullptr;
						}
						auto dynamicStorageArray = [&]() {
							return arrayType->location() == DataLocation::Storage
								&& arrayType->isDynamicallySized();
						};
						if (memberName == "push" && !funcCall->arguments().empty())
						{
							auto value = generateSolidityExpression(*funcCall->arguments()[0]);
							if (!mlir::isa<mlir::solidity::ArrayType>(base.getType()) || !dynamicStorageArray())
								return emitUnsupported(loc, value.getType(), "push onto an unsupported array expression");
							value = coerceValue(loc, value, assignmentValueType(arrayType->baseType()));
							auto pushOp = m_builder->create<mlir::solidity::ArrayPushOp>(loc, base, value);
							if (isStorageReferenceType(funcCall->arguments()[0]->annotation().type))
								pushOp->setAttr("storageValue", m_builder->getUnitAttr());
							if (!markStorageArray(
									pushOp.getOperation(),
									memberAccess->expression().annotation().type,
									extractMappingVarName(memberAccess->expression())))
								return emitUnsupported(loc, value.getType(), "push onto an unnamed storage array");
							return base;
						}
						if (memberName == "push" && funcCall->arguments().empty())
						{
							auto resultType = translateSolidityType(*_expr.annotation().type);
							if (!mlir::isa<mlir::solidity::ArrayType>(base.getType()) || !dynamicStorageArray())
								return emitUnsupported(loc, resultType, "empty push onto an unsupported array expression");
							auto pushOp = m_builder->create<mlir::solidity::ArrayPushEmptyOp>(loc, resultType, base);
							if (!markStorageArray(
									pushOp.getOperation(),
									memberAccess->expression().annotation().type,
									extractMappingVarName(memberAccess->expression())))
								return emitUnsupported(loc, resultType, "empty push onto an unnamed storage array");
							return pushOp.getResult();
						}
						if (memberName == "pop" && funcCall->arguments().empty())
						{
							auto resultType = mlir::solidity::UIntType::get(m_context.get(), 256);
							if (!mlir::isa<mlir::solidity::ArrayType>(base.getType()) || !dynamicStorageArray())
								return emitUnsupported(loc, resultType, "pop from an unsupported array expression");
							auto popOp = m_builder->create<mlir::solidity::ArrayPopOp>(loc, base);
							if (!markStorageArray(
									popOp.getOperation(),
									memberAccess->expression().annotation().type,
									extractMappingVarName(memberAccess->expression())))
								return emitUnsupported(loc, resultType, "pop from an unnamed storage array");
							return nullptr;
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
					for (auto [index, arg]: llvm::enumerate(funcCall->arguments()))
					{
						auto value = generateSolidityExpression(*arg);
						if (!value)
						{
							auto type = translateSolidityType(*arg->annotation().type);
							value = emitUnsupported(loc, type, "struct constructor argument");
						}
						if (value && isStorageReferenceType(arg->annotation().type)
							&& index < structDecl->members().size())
						{
							Type const* memberType = TypeProvider::withLocationIfReference(
								DataLocation::Memory, structDecl->members()[index]->type());
							value = m_builder->create<mlir::solidity::StorageToMemoryOp>(
								loc, translateSolidityType(*memberType), value).getResult();
						}
						operands.push_back(value);
					}

					auto resultType = translateSolidityType(*_expr.annotation().type);
					return m_builder->create<mlir::solidity::StructCreateOp>(
						loc, resultType, m_builder->getStringAttr(structDecl->name()), operands);
				}
			}
			// Handle both forms of `new`: a dynamic memory array owns a freshly
			// allocated [length][data...] region, while a user-defined contract
			// becomes a nested object and a CREATE below. Treating every NewExpression
			// as the latter used to ask the assembler for an "UnknownContract" object
			// for perfectly ordinary expressions such as `new uint256[](3)`.
			else if (auto* newExpr = dynamic_cast<NewExpression const*>(&callee))
			{
				if (auto const* arrayType = dynamic_cast<ArrayType const*>(_expr.annotation().type))
				{
					if (funcCall->arguments().size() != 1)
						return emitUnsupported(
							loc,
							translateSolidityType(*arrayType),
							"memory array construction without exactly one length");
					auto length = generateSolidityExpression(*funcCall->arguments().front());
					if (!length)
						return emitUnsupported(
							loc,
							translateSolidityType(*arrayType),
							"memory array construction with an unsupported length");
					return m_builder->create<mlir::solidity::MemoryArrayCreateOp>(
						loc, translateSolidityType(*arrayType), length).getResult();
				}

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
							// The plain name: this is what the object is
							// compiled and nested under, and an id suffix
							// matches no contract the driver can look up.
							fullObjectName = typeName;
						}
					}
				}

				std::vector<mlir::Value> args;
				std::vector<mlir::Attribute> externalArgs;
				auto const* constructorType
					= dynamic_cast<FunctionType const*>(funcCall->expression().annotation().type);
				for (auto [index, arg]: llvm::enumerate(funcCall->arguments()))
				{
					mlir::Value argValue;
					Type const* parameterType = constructorType && index < constructorType->parameterTypes().size()
						? constructorType->parameterTypes()[index]
						: nullptr;
					if (auto const* fixed = dynamic_cast<FixedBytesType const*>(parameterType))
						if (auto const* literal = dynamic_cast<Literal const*>(&withoutParentheses(*arg)))
							if (literal->token() == langutil::Token::StringLiteral
								|| literal->token() == langutil::Token::HexStringLiteral)
							{
								llvm::APInt word(256, 0);
								for (unsigned i = 0; i < 32; ++i)
								{
									word <<= 8;
									if (i < literal->value().size() && i < fixed->numBytes())
										word |= llvm::APInt(256, static_cast<uint8_t>(literal->value()[i]));
								}
								argValue = m_builder->create<mlir::solidity::ConstantOp>(
									loc,
									m_builder->getIntegerAttr(
										mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned), word),
									translateSolidityType(*parameterType));
							}
					if (!argValue)
						argValue = generateSolidityExpression(*arg);
					if (argValue && parameterType)
						argValue = coerceValue(loc, argValue, translateSolidityType(*parameterType));
					if (argValue)
						args.push_back(argValue);
					if (auto const* argumentType
						= dynamic_cast<FunctionType const*>(arg->annotation().type);
						argumentType && argumentType->kind() == FunctionType::Kind::External)
						externalArgs.push_back(
							m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
				}

				// msg.value = 0 (no value sent with plain `new`)
				auto zeroValue = m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getIntegerAttr(
						mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned), 0),
					mlir::solidity::UIntType::get(m_context.get(), 256));

				auto resultType = translateSolidityType(*_expr.annotation().type);
				auto create = m_builder->create<mlir::solidity::CreateContractOp>(
					loc, resultType,
					m_builder->getStringAttr(fullObjectName),
					zeroValue, args);
				if (!externalArgs.empty())
					create->setAttr("external_function_args", m_builder->getArrayAttr(externalArgs));
				return create.getResult();
			}

			// A call whose callee is a value rather than a name - `fn()` where
			// `fn` holds a pointer - goes through the dispatcher for its shape.
			if (auto* calleeType = dynamic_cast<FunctionType const*>(callee.annotation().type))
				// A local holding a pointer is an Identifier too, so what
				// separates the two is what the name refers to: a variable
				// means the callee is a value.
				if (calleeType->kind() == FunctionType::Kind::Internal && !resolvedCallee(*funcCall)
					&& !dynamic_cast<FunctionDefinition const*>(
						dynamic_cast<Identifier const*>(&callee)
							? dynamic_cast<Identifier const*>(&callee)->annotation().referencedDeclaration
							: nullptr))
				{
					mlir::Value pointer = generateSolidityExpression(callee);
					if (pointer)
					{
						std::vector<mlir::Value> args{pointer};
						for (auto const& argument: funcCall->arguments())
							if (mlir::Value value = generateSolidityExpression(*argument))
								args.push_back(value);

						auto resultTypes = callResultTypes(*_expr.annotation().type);
						if (auto const* tuple = dynamic_cast<TupleType const*>(_expr.annotation().type))
							if (tuple->components().empty())
								resultTypes.clear();

						size_t const arity = args.size() - 1;
						m_indirectShapes.insert({arity, resultTypes.size()});
						auto call = m_builder->create<mlir::solidity::FunctionCallOp>(
							loc,
							mlir::TypeRange{resultTypes},
							m_builder->getStringAttr(indirectDispatcherName(arity, resultTypes.size())),
							args);
						return call->getNumResults() > 0 ? call->getResult(0) : mlir::Value();
					}
				}

			if (auto* calleeType = dynamic_cast<FunctionType const*>(callee.annotation().type))
				if (calleeType->kind() == FunctionType::Kind::External && !resolvedCallee(*funcCall)
					&& !(dynamic_cast<MemberAccess const*>(&callee)
						&& dynamic_cast<ContractType const*>(
							dynamic_cast<MemberAccess const*>(&callee)->expression().annotation().type)))
				{
					mlir::Value pointer = generateSolidityExpression(callee);
					std::vector<mlir::Value> args;
					std::vector<mlir::Attribute> externalArgs;
					std::vector<mlir::Attribute> storageArgs;
					std::vector<mlir::Attribute> enumArgLimits;
					bool hasEnumArg = false;
					for (auto [index, argument]: llvm::enumerate(funcCall->arguments()))
					{
						uint64_t enumLimit = 0;
						if (index < calleeType->parameterTypes().size())
							if (auto const* enumType = dynamic_cast<EnumType const*>(
								calleeType->parameterTypes()[index]))
								enumLimit = static_cast<uint64_t>(enumType->numberOfMembers());
						enumArgLimits.push_back(
							m_builder->getI64IntegerAttr(static_cast<int64_t>(enumLimit)));
						hasEnumArg = hasEnumArg || enumLimit != 0;
						if (mlir::Value value = generateSolidityExpression(*argument))
						{
							args.push_back(value);
							if (auto const* referenceType
								= dynamic_cast<ReferenceType const*>(argument->annotation().type);
								referenceType && referenceType->location() == DataLocation::Storage)
								storageArgs.push_back(m_builder->getI32IntegerAttr(
									static_cast<int32_t>(args.size() - 1)));
						}
						if (auto const* argumentType
							= dynamic_cast<FunctionType const*>(argument->annotation().type);
							argumentType && argumentType->kind() == FunctionType::Kind::External)
							externalArgs.push_back(
								m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
					}
					auto resultTypes = callResultTypes(*_expr.annotation().type);
					if (auto const* tuple = dynamic_cast<TupleType const*>(_expr.annotation().type))
						if (tuple->components().empty())
							resultTypes.clear();
					auto call = m_builder->create<mlir::solidity::ExternalFunctionCallOp>(
						loc,
						mlir::TypeRange{resultTypes},
						pointer,
						m_builder->getI32IntegerAttr(0),
						m_builder->getStringAttr(""),
						m_builder->getBoolAttr(
							calleeType->stateMutability() == StateMutability::Pure
							|| calleeType->stateMutability() == StateMutability::View),
						args);
					call->setAttr("functionPointer", m_builder->getUnitAttr());
					if (!externalArgs.empty())
						call->setAttr("external_function_args", m_builder->getArrayAttr(externalArgs));
					if (!storageArgs.empty())
						call->setAttr("storage_args", m_builder->getArrayAttr(storageArgs));
					if (hasEnumArg)
						call->setAttr("enum_arg_limits", m_builder->getArrayAttr(enumArgLimits));
					return call->getNumResults() > 0 ? call->getResult(0) : mlir::Value();
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
						mlir::Value msgValue;
						if (funcCall->arguments().size() > 1)
						{
							if (auto* literal = dynamic_cast<Literal const*>(funcCall->arguments()[1].get()))
								msgAttr = m_builder->getStringAttr(literal->value());
							else if (dynamic_cast<ReferenceType const*>(
								funcCall->arguments()[1]->annotation().type))
								msgValue = generateSolidityExpression(*funcCall->arguments()[1]);
						}
						m_builder->create<mlir::solidity::RequireOp>(
							loc,
							typedOperand(loc, cond, mlir::solidity::BoolType::get(m_context.get()),
								"require condition not generated as a boolean"),
							msgValue,
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
					mlir::Value reasonValue;
					if (!funcCall->arguments().empty())
					{
						if (auto* literal = dynamic_cast<Literal const*>(funcCall->arguments()[0].get()))
							reasonAttr = m_builder->getStringAttr(literal->value());
						else
							reasonValue = generateSolidityExpression(*funcCall->arguments()[0]);
					}
					if (insideSCFRegion())
						emitNestedControlTransfer(loc, "revert");
					else if (reasonValue)
						m_builder->create<mlir::solidity::RevertValueOp>(loc, reasonValue);
					else
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

					// Create a type conversion operation. Storage reference values are
					// slots at this rung, so conversions that inspect their contents
					// must retain the source location for lowering.
					auto conversion = m_builder->create<mlir::solidity::ConvertOp>(loc, targetType, argValue);
					if (auto const* source = dynamic_cast<ReferenceType const*>(
						funcCall->arguments()[0]->annotation().type);
						source && source->location() == DataLocation::Storage)
						conversion->setAttr("storageValue", m_builder->getUnitAttr());
					return conversion;
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

					if (funcName == "blobhash" && funcCall->arguments().size() == 1)
					{
						auto index = generateSolidityExpression(*funcCall->arguments()[0]);
						return m_builder->create<mlir::solidity::BlobHashOp>(
							loc, mlir::solidity::BytesType::get(m_context.get(), 32), index);
					}

					// EIP-7201 namespace slot:
					// keccak256(abi.encode(uint256(keccak256(bytes(id))) - 1)) & ~uint256(0xff)
					if (funcName == "erc7201" && funcCall->arguments().size() == 1)
					{
						auto uint256Type = mlir::solidity::UIntType::get(m_context.get(), 256);
						auto bytes32Type = mlir::solidity::BytesType::get(m_context.get(), 32);
						auto data = generateSolidityExpression(*funcCall->arguments()[0]);
						auto inner = m_builder->create<mlir::solidity::Keccak256Op>(loc, bytes32Type, data);
						auto asInteger = m_builder->create<mlir::solidity::ConvertOp>(loc, uint256Type, inner);
						auto one = m_builder->create<mlir::solidity::ConstantOp>(
							loc, m_builder->getIntegerAttr(m_builder->getI64Type(), 1), uint256Type);
						auto previous = m_builder->create<mlir::solidity::SubOp>(loc, uint256Type, asInteger, one);
						previous->setAttr("unchecked", m_builder->getUnitAttr());
						auto encoded = m_builder->create<mlir::solidity::AbiEncodeOp>(
							loc,
							mlir::solidity::DynamicBytesType::get(m_context.get()),
							mlir::ValueRange{previous});
						auto outer = m_builder->create<mlir::solidity::Keccak256Op>(loc, bytes32Type, encoded);
						auto mask = m_builder->create<mlir::solidity::ConstantOp>(
							loc,
							m_builder->getIntegerAttr(
								mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
								~llvm::APInt(256, 0xff)),
							uint256Type);
						return m_builder->create<mlir::solidity::AndOp>(loc, uint256Type, outer, mask);
					}

					// keccak256(data) - hash function
					if (funcName == "keccak256" && funcCall->arguments().size() == 1)
					{
						// Of a literal it is a compile-time constant, and worth
						// folding here: the operand would otherwise be a
						// `solidity.string_literal`, which needs the memory
						// allocator that does not exist yet, and the whole
						// expression was dropped for want of it.
						if (auto* literal = dynamic_cast<Literal const*>(
								&withoutParentheses(*funcCall->arguments()[0])))
							if (literal->token() == langutil::Token::StringLiteral
								|| literal->token() == langutil::Token::HexStringLiteral)
							{
								auto const hash = solidity::util::keccak256(literal->value());
								auto attr = m_builder->getIntegerAttr(
									mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
									llvm::APInt(256, u256(hash).str(), 10));
								return m_builder->create<mlir::solidity::ConstantOp>(
									loc, attr, mlir::solidity::BytesType::get(m_context.get(), 32)).getResult();
							}

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
					// string.concat(...) and bytes.concat(...) have an elementary type
					// expression as their base, rather than a runtime receiver. Preserve
					// each argument's type so bytesN values keep their exact width.
					if (memberAccess->memberName() == "concat"
						&& dynamic_cast<ElementaryTypeNameExpression const*>(&memberAccess->expression()))
					{
						auto resultType = translateSolidityType(*_expr.annotation().type);
						std::vector<mlir::Value> values;
						std::vector<mlir::Attribute> storageArgs;
						for (auto [index, argument]: llvm::enumerate(funcCall->arguments()))
						{
							mlir::Value value = generateSolidityExpression(*argument);
							if (!value)
								return emitUnsupported(loc, resultType, "concat argument not generated");
							values.push_back(value);
							if (isStorageReferenceType(argument->annotation().type))
								storageArgs.push_back(
									m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
						}
						auto concat = m_builder->create<mlir::solidity::ConcatOp>(loc, resultType, values);
						if (!storageArgs.empty())
							concat->setAttr("storage_args", m_builder->getArrayAttr(storageArgs));
						return concat.getResult();
					}

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
							std::vector<mlir::Attribute> storageArgs;
							for (auto [index, arg]: llvm::enumerate(funcCall->arguments()))
							{
								// The second operand of abi.decode is syntax describing
								// result types, not a runtime value. Its components are
								// ElementaryTypeNameExpression nodes and must not become
								// placeholder operands in the dialect.
								if (abiFunc == "decode" && index > 0)
									continue;
								auto val = generateSolidityExpression(*arg);
								if (val)
								{
									argValues.push_back(val);
									if (auto const* referenceType
										= dynamic_cast<ReferenceType const*>(arg->annotation().type);
										referenceType && referenceType->location() == DataLocation::Storage)
										storageArgs.push_back(m_builder->getI32IntegerAttr(
											static_cast<int32_t>(argValues.size() - 1)));
								}
							}

							auto resultType = translateSolidityType(*_expr.annotation().type);

							if (abiFunc == "encode")
							{
								auto op = m_builder->create<mlir::solidity::AbiEncodeOp>(loc, resultType, argValues);
								if (!storageArgs.empty())
									op->setAttr("storage_args", m_builder->getArrayAttr(storageArgs));
								return op.getResult();
							}
							else if (abiFunc == "encodePacked")
							{
								auto op = m_builder->create<mlir::solidity::AbiEncodePackedOp>(loc, resultType, argValues);
								if (!storageArgs.empty())
									op->setAttr("storage_args", m_builder->getArrayAttr(storageArgs));
								return op.getResult();
							}
			else if (abiFunc == "encodeWithSelector")
			{
				// A selector is bytes4, including when its source spelling is an
				// integer literal. Fixed bytes are left-aligned in an EVM word;
				// leaving the literal right-aligned encoded four leading zero bytes.
				if (!argValues.empty())
					argValues.front() = coerceValue(
						loc, argValues.front(), mlir::solidity::BytesType::get(m_context.get(), 4));
				auto op = m_builder->create<mlir::solidity::AbiEncodeWithSelectorOp>(loc, resultType, argValues);
								if (!storageArgs.empty())
									op->setAttr("storage_args", m_builder->getArrayAttr(storageArgs));
								return op.getResult();
							}
							else if (abiFunc == "encodeWithSignature")
							{
								auto op = m_builder->create<mlir::solidity::AbiEncodeWithSignatureOp>(loc, resultType, argValues);
								if (!storageArgs.empty())
									op->setAttr("storage_args", m_builder->getArrayAttr(storageArgs));
								return op.getResult();
							}
							else if (abiFunc == "decode")
							{
								// One result per decoded component: `abi.decode`
								// of a tuple yields all of them, and declaring
								// one left the call short of its own type.
								auto decoded = callResultTypes(*_expr.annotation().type);
								auto op = m_builder->create<mlir::solidity::AbiDecodeOp>(
									loc, mlir::TypeRange{decoded}, argValues);
								return op->getNumResults() > 0 ? op->getResult(0) : mlir::Value();
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
						// A first-class external function stored in a variable or
						// struct member has no declaration from which a canonical
						// signature can be recovered. FunctionType deliberately
						// rejects externalSignature() in that case; do not turn a
						// supported Solidity input into a compiler-process crash.
						if (!funcType->hasDeclaration())
						{
							for (auto const& arg: funcCall->arguments())
								generateSolidityExpression(*arg);
							return emitUnsupported(
								loc, translateSolidityType(*_expr.annotation().type),
								"call through a first-class external function value");
						}
						// Evaluate base expression to get target address
						auto targetAddr = generateSolidityExpression(memberAccess->expression());
						if (!targetAddr)
						{
							auto dummyType = translateSolidityType(*_expr.annotation().type);
							return emitUnsupported(loc, dummyType, "external call to '" + funcName + "': failed to evaluate target");
						}

						// Generate arguments
						std::vector<mlir::Value> args;
						std::vector<mlir::Attribute> externalArgs;
						std::vector<mlir::Attribute> externalArrayArgs;
						std::vector<mlir::Attribute> storageArgs;
						std::vector<mlir::Attribute> enumArgLimits;
						bool hasEnumArg = false;
						for (auto [index, arg]: llvm::enumerate(funcCall->arguments()))
						{
							uint64_t enumLimit = 0;
							if (index < funcType->parameterTypes().size())
								if (auto const* enumType = dynamic_cast<EnumType const*>(
									funcType->parameterTypes()[index]))
									enumLimit = static_cast<uint64_t>(enumType->numberOfMembers());
							enumArgLimits.push_back(
								m_builder->getI64IntegerAttr(static_cast<int64_t>(enumLimit)));
							hasEnumArg = hasEnumArg || enumLimit != 0;
							auto argValue = generateSolidityExpression(*arg);
							if (argValue)
							{
								args.push_back(argValue);
								if (auto const* referenceType
									= dynamic_cast<ReferenceType const*>(arg->annotation().type);
									referenceType && referenceType->location() == DataLocation::Storage)
									storageArgs.push_back(m_builder->getI32IntegerAttr(
										static_cast<int32_t>(args.size() - 1)));
							}
							if (auto const* argumentType
								= dynamic_cast<FunctionType const*>(arg->annotation().type);
								argumentType && argumentType->kind() == FunctionType::Kind::External)
								externalArgs.push_back(
									m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
							else if (dynamic_cast<ArrayType const*>(arg->annotation().type)
								&& containsExternalFunction(arg->annotation().type))
								externalArrayArgs.push_back(
									m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
						}

						// Compute selector and signature
						uint32_t selector = util::selectorFromSignatureU32(funcType->externalSignature());
						std::string signature = funcType->externalSignature();

						// A view/pure external function must use STATICCALL: CALL would
						// let a malicious implementation mutate state despite the type.
						auto resultTypes = callResultTypes(*_expr.annotation().type);
						if (auto const* tuple = dynamic_cast<TupleType const*>(_expr.annotation().type))
							if (tuple->components().empty())
								resultTypes.clear();
						auto call = m_builder->create<mlir::solidity::ExternalFunctionCallOp>(
							loc, mlir::TypeRange{resultTypes}, targetAddr,
							m_builder->getI32IntegerAttr(selector),
							m_builder->getStringAttr(signature),
							m_builder->getBoolAttr(
								funcType->stateMutability() == StateMutability::Pure ||
								funcType->stateMutability() == StateMutability::View),
							args);
						if (!externalArgs.empty())
							call->setAttr("external_function_args", m_builder->getArrayAttr(externalArgs));
						if (!externalArrayArgs.empty())
							call->setAttr(
								"external_function_array_args", m_builder->getArrayAttr(externalArrayArgs));
						if (!storageArgs.empty())
							call->setAttr("storage_args", m_builder->getArrayAttr(storageArgs));
						if (hasEnumArg)
							call->setAttr("enum_arg_limits", m_builder->getArrayAttr(enumArgLimits));
						std::vector<mlir::Attribute> externalResults;
						std::vector<mlir::Attribute> externalArrayResults;
						for (auto [index, result]: llvm::enumerate(funcType->returnParameterTypes()))
							if (auto const* resultType = dynamic_cast<FunctionType const*>(result);
								resultType && resultType->kind() == FunctionType::Kind::External)
								externalResults.push_back(
									m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
							else if (dynamic_cast<ArrayType const*>(result) && containsExternalFunction(result))
								externalArrayResults.push_back(
									m_builder->getI32IntegerAttr(static_cast<int32_t>(index)));
						if (!externalResults.empty())
							call->setAttr("external_function_results", m_builder->getArrayAttr(externalResults));
						if (!externalArrayResults.empty())
							call->setAttr(
								"external_function_array_results", m_builder->getArrayAttr(externalArrayResults));
						return call->getNumResults() > 0 ? call->getResult(0) : mlir::Value();
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
					auto const* calledType = dynamic_cast<FunctionType const*>(callee.annotation().type);
					for (auto [index, arg]: llvm::enumerate(funcCall->arguments()))
					{
						auto argValue = generateSolidityExpression(*arg);
						if (argValue && calledType && index < calledType->parameterTypes().size())
						{
							auto const* source = dynamic_cast<ReferenceType const*>(arg->annotation().type);
							auto const* target = dynamic_cast<ReferenceType const*>(
								calledType->parameterTypes()[index]);
							if (source && target
								&& source->location() == DataLocation::Storage
								&& target->location() == DataLocation::Memory)
								argValue = m_builder->create<mlir::solidity::StorageToMemoryOp>(
									loc, translateSolidityType(*calledType->parameterTypes()[index]),
									argValue).getResult();
						}
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
		else if (auto* range = dynamic_cast<IndexRangeAccess const*>(&_expr))
		{
			mlir::Value base = generateSolidityExpression(range->baseExpression());
			mlir::Value start = range->startExpression()
				? generateSolidityExpression(*range->startExpression())
				: m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getI64IntegerAttr(0),
					mlir::solidity::UIntType::get(m_context.get(), 256)).getResult();
			mlir::Value end = range->endExpression()
				? generateSolidityExpression(*range->endExpression())
				: m_builder->create<mlir::solidity::ConstantOp>(
					loc, m_builder->getI64IntegerAttr(0),
					mlir::solidity::UIntType::get(m_context.get(), 256)).getResult();
			if (!base || !start || !end)
				return emitUnsupported(
					loc, translateSolidityType(*_expr.annotation().type), "array slice operand");
			return m_builder->create<mlir::solidity::ArraySliceOp>(
				loc,
				translateSolidityType(*_expr.annotation().type),
				base, start, end,
				range->startExpression() ? m_builder->getUnitAttr() : mlir::UnitAttr(),
				range->endExpression() ? m_builder->getUnitAttr() : mlir::UnitAttr()).getResult();
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
				mlir::Value baseSlot = mappingBaseSlot(*indexAccess);
				if (baseSlot)
					keys.insert(keys.begin(), baseSlot);
				auto access = m_builder->create<mlir::solidity::MappingAccessOp>(
					loc, valueType, m_builder->getStringAttr(varName), keys);
				if (baseSlot)
					access->setAttr("baseSlotOperand", m_builder->getUnitAttr());
				markMappingEnumKeyLimits(access.getOperation(), *indexAccess);
				if (isStorageReferenceType(_expr.annotation().type))
					// A reference value is its storage slot, not the word in it.
					access.setAsReference(true);
				return access.getResult();
			}
			else if (dynamic_cast<FixedBytesType const*>(
				indexAccess->baseExpression().annotation().type))
			{
				mlir::Value base = generateSolidityExpression(indexAccess->baseExpression());
				mlir::Value index = indexAccess->indexExpression()
					? generateSolidityExpression(*indexAccess->indexExpression()) : mlir::Value();
				if (base && index)
					return m_builder->create<mlir::solidity::FixedBytesAccessOp>(
						loc, translateSolidityType(*_expr.annotation().type), base, index).getResult();
				return emitUnsupported(
					loc, translateSolidityType(*_expr.annotation().type), "fixed-bytes index operand");
			}
			else if (
				dynamic_cast<ArrayType const*>(indexAccess->baseExpression().annotation().type)
				|| dynamic_cast<ArraySliceType const*>(indexAccess->baseExpression().annotation().type))
			{
				auto base = generateSolidityExpression(indexAccess->baseExpression());
				mlir::Value index;
				if (indexAccess->indexExpression())
					index = generateSolidityExpression(*indexAccess->indexExpression());
				auto elementType = translateSolidityType(*_expr.annotation().type);
				if (base && index)
				{
					if (!mlir::isa<
						mlir::solidity::ArrayType,
						mlir::solidity::DynamicBytesType,
						mlir::solidity::StringType>(base.getType()))
						return emitUnsupported(loc, elementType, "index into an unsupported array expression");
					auto op = m_builder->create<mlir::solidity::ArrayAccessOp>(loc, elementType, base, index);
					std::string varName = extractMappingVarName(indexAccess->baseExpression());
					if (!varName.empty())
						op->setAttr("varName", m_builder->getStringAttr(varName));
					// Memory arrays go through unmarked: the lowering reads the
					// element out of the pointer rather than out of a slot.
					markStorageArray(
						op.getOperation(), indexAccess->baseExpression().annotation().type, varName);
					if (isStorageReferenceType(_expr.annotation().type))
						op->setAttr("asReference", m_builder->getUnitAttr());
					return op.getResult();
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
			auto resultType = translateSolidityType(*_expr.annotation().type);
			mlir::Value condition = generateSolidityExpression(conditional->condition());
			condition = typedOperand(
				loc, condition, mlir::solidity::BoolType::get(m_context.get()),
				"conditional condition not generated as a boolean");

			std::set<int64_t> modifiedSet;
			collectExpressionModifiedVariables(conditional->trueExpression(), modifiedSet);
			collectExpressionModifiedVariables(conditional->falseExpression(), modifiedSet);
			std::vector<int64_t> modified;
			llvm::SmallVector<mlir::Value, 4> entry;
			llvm::SmallVector<mlir::Type, 4> resultTypes{resultType};
			for (int64_t id: modifiedSet)
				if (m_valueMap.count(id) && m_valueMap[id])
				{
					modified.push_back(id);
					entry.push_back(m_valueMap[id]);
					resultTypes.push_back(m_valueMap[id].getType());
				}

			auto savedValues = m_valueMap;
			auto branch = m_builder->create<mlir::scf::IfOp>(
				loc, mlir::TypeRange{resultTypes},
				m_builder->create<mlir::solidity::ToI1Op>(loc, m_builder->getI1Type(), condition), true);
			auto emitArm = [&](mlir::Block& _block, Expression const& _expression) {
				mlir::OpBuilder::InsertionGuard guard(*m_builder);
				m_builder->setInsertionPointToEnd(&_block);
				m_valueMap = savedValues;
				mlir::Value value = generateSolidityExpression(_expression);
				value = coerceValue(loc, value, resultType);
				llvm::SmallVector<mlir::Value, 4> yielded{value};
				for (size_t i = 0; i < modified.size(); ++i)
				{
					mlir::Value current = m_valueMap.count(modified[i]) ? m_valueMap[modified[i]] : mlir::Value();
					yielded.push_back(
						current && current.getType() == entry[i].getType() ? current : entry[i]);
				}
				m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{yielded});
			};
			emitArm(branch.getThenRegion().front(), conditional->trueExpression());
			emitArm(branch.getElseRegion().front(), conditional->falseExpression());
			m_builder->setInsertionPointAfter(branch.getOperation());
			m_valueMap = savedValues;
			for (size_t i = 0; i < modified.size(); ++i)
				m_valueMap[modified[i]] = branch->getResult(i + 1);
			return branch->getResult(0);
		}
		else if (auto* tupleExpr = dynamic_cast<TupleExpression const*>(&_expr))
		{
			auto const& components = tupleExpr->components();
			if (tupleExpr->isInlineArray())
			{
				auto arrayType = llvm::dyn_cast<mlir::solidity::ArrayType>(
					translateSolidityType(*_expr.annotation().type));
				if (!arrayType)
					return emitUnsupported(
						loc, translateSolidityType(*_expr.annotation().type), "inline array type");
				mlir::Value length = m_builder->create<mlir::solidity::ConstantOp>(
					loc,
					m_builder->getIntegerAttr(
						m_builder->getI64Type(), static_cast<int64_t>(components.size())),
					mlir::solidity::UIntType::get(m_context.get(), 256));
				mlir::Value array = m_builder->create<mlir::solidity::MemoryArrayCreateOp>(
					loc, arrayType, length);
				for (auto [index, component]: llvm::enumerate(components))
				{
					if (!component)
						continue;
					mlir::Value value = generateSolidityExpression(*component);
					mlir::Value position = m_builder->create<mlir::solidity::ConstantOp>(
						loc,
						m_builder->getIntegerAttr(m_builder->getI64Type(), static_cast<int64_t>(index)),
						mlir::solidity::UIntType::get(m_context.get(), 256));
					m_builder->create<mlir::solidity::ArrayStoreOp>(loc, array, position, value);
				}
				return array;
			}
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
				// `unchecked { }` has to reach the lowering: the block is
				// inlined here, so without a mark on the operations themselves
				// nothing downstream can tell the difference.
				++m_uncheckedDepth;
				struct Restore
				{
					unsigned& depth;
					~Restore() { --depth; }
				} restore{m_uncheckedDepth};
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
					if (endsSourceControlFlow(current))
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
					if (endsSourceControlFlow(current))
						break;
					lastValue = generateSolidityStatement(*stmt);
				}
				return lastValue;
			}
		}
		else if (dynamic_cast<PlaceholderStatement const*>(&_stmt))
		{
			if (m_modifierContinuation)
				m_modifierContinuation();
			else
				emitUnsupported(
					loc, mlir::solidity::UIntType::get(m_context.get(), 256),
					"modifier placeholder outside modifier expansion");
			return mlir::Value();
		}
		else if (auto* ifStmt = dynamic_cast<IfStatement const*>(&_stmt))
		{
			auto condition = generateSolidityExpression(ifStmt->condition());
			bool hasElse = ifStmt->falseStatement() != nullptr;

			// A variable assigned in a branch is live after the `if`, so the
			// branches have to yield it - the same problem the loop solves by
			// carrying values. Parking it as a placeholder instead, which is
			// what happened before, meant `if (c) x = 1; else x = 2;` read zero
			// afterwards.
			std::set<int64_t> assigned = collectModifiedVariables(ifStmt->trueStatement());
			if (hasElse)
			{
				auto other = collectModifiedVariables(*ifStmt->falseStatement());
				assigned.insert(other.begin(), other.end());
			}
			for (int64_t declared: collectDeclaredVariables(ifStmt->trueStatement()))
				assigned.erase(declared);
			if (hasElse)
				for (int64_t declared: collectDeclaredVariables(*ifStmt->falseStatement()))
					assigned.erase(declared);

			std::vector<int64_t> const carried(assigned.begin(), assigned.end());
			if (carried.empty())
			{
				auto const savedValues = m_valueMap;
				auto ifOp = m_builder->create<mlir::solidity::IfOp>(loc, condition);
				m_builder->setInsertionPointToEnd(&ifOp.getThenRegion().emplaceBlock());
				m_valueMap = savedValues;
				generateSolidityStatement(ifStmt->trueStatement());
				if (hasElse)
				{
					m_builder->setInsertionPointToEnd(&ifOp.getElseRegion().emplaceBlock());
					m_valueMap = savedValues;
					generateSolidityStatement(*ifStmt->falseStatement());
				}
				else
					ifOp.getElseRegion().emplaceBlock();

				m_builder->setInsertionPointAfter(ifOp.getOperation());
				m_valueMap = savedValues;
				dropValuesEscaping(ifOp.getOperation(), loc);
				return mlir::Value();
			}

			// The value each carried variable has on entry, which an arm that
			// does not touch it yields unchanged.
			llvm::SmallVector<mlir::Value, 4> entry;
			llvm::SmallVector<mlir::Type, 4> carriedTypes;
			for (int64_t id: carried)
			{
				mlir::Value value = m_valueMap.count(id) ? m_valueMap[id] : mlir::Value();
				if (!value)
					value = emitUnsupported(
						loc, mlir::solidity::UIntType::get(m_context.get(), 256), "variable used before assignment");
				entry.push_back(value);
				carriedTypes.push_back(value.getType());
			}

			mlir::Value i1Condition = m_builder->create<mlir::solidity::ToI1Op>(
				loc, m_builder->getI1Type(), condition);
			auto scfIf = m_builder->create<mlir::scf::IfOp>(loc, mlir::TypeRange{carriedTypes}, i1Condition, true);

			auto const savedValues = m_valueMap;
			auto emitArm = [&](mlir::Block& _block, Statement const* _statement) {
				mlir::OpBuilder::InsertionGuard guard(*m_builder);
				m_builder->setInsertionPointToEnd(&_block);
				m_valueMap = savedValues;
				for (size_t i = 0; i < carried.size(); ++i)
					m_valueMap[carried[i]] = entry[i];
				if (_statement)
					generateSolidityStatement(*_statement);
				llvm::SmallVector<mlir::Value, 4> yielded;
				for (size_t i = 0; i < carried.size(); ++i)
				{
					mlir::Value value = m_valueMap.count(carried[i]) ? m_valueMap[carried[i]] : mlir::Value();
					yielded.push_back(value && value.getType() == carriedTypes[i] ? value : entry[i]);
				}
				if (_block.empty() || !_block.back().hasTrait<mlir::OpTrait::IsTerminator>())
					m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{yielded});
			};
			emitArm(scfIf.getThenRegion().front(), &ifStmt->trueStatement());
			emitArm(scfIf.getElseRegion().front(), ifStmt->falseStatement());

			m_builder->setInsertionPointAfter(scfIf.getOperation());
			m_valueMap = savedValues;
			for (size_t i = 0; i < carried.size(); ++i)
				m_valueMap[carried[i]] = scfIf->getResult(i);
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
			if (forStmt->condition())
			{
				std::set<int64_t> conditionModified;
				collectExpressionModifiedVariables(*forStmt->condition(), conditionModified);
				loopCarriedVarIds.insert(conditionModified.begin(), conditionModified.end());
			}

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

			// Store loop-carried variables for break/continue handling. Preserve
			// an enclosing loop: a nested loop must not erase its transfer state.
			auto const savedLoopCarried = m_loopCarriedVarIds;
			auto const savedLoopContext = m_immediateLoopContext;
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
				for (size_t i = 0; i < loopCarriedTypes.size(); ++i)
					blockArgs.push_back(i < loopCarriedVarIdsList.size()
						? m_valueMap.at(loopCarriedVarIdsList[i])
						: beforeBlock->getArgument(i));

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

				// Everything from here belongs to Yul's post region. A continue
				// executes it; a break does not.
				m_builder->create<mlir::solidity::LoopPostOp>(
					loc, mlir::ValueRange{currentLoopValues()});

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

			// Restore an enclosing loop's transfer state.
			m_loopCarriedVarIds = savedLoopCarried;
			m_immediateLoopContext = savedLoopContext;

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
				std::set<int64_t> conditionModified;
				collectExpressionModifiedVariables(whileStmt->condition(), conditionModified);
				loopCarriedVarIds.insert(conditionModified.begin(), conditionModified.end());

				// Store loop-carried variables for break/continue handling
				auto const savedLoopCarried = m_loopCarriedVarIds;
				auto const savedLoopContext = m_immediateLoopContext;
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
				whileOp->setAttr("do_while", m_builder->getUnitAttr());

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

				// A do-while condition is Yul post-region work: continue must
				// evaluate it, while break must skip it.
				m_builder->create<mlir::solidity::LoopPostOp>(
					loc, mlir::ValueRange{currentLoopValues()});
				auto condValue = generateSolidityExpression(whileStmt->condition());
				m_builder->create<mlir::solidity::LoopConditionOp>(loc, condValue);

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

				// IMPORTANT: Set insertion point after the while operation
				// This ensures subsequent operations aren't added to the loop body
				m_builder->setInsertionPointAfter(whileOp.getOperation());

				// Map every loop result back to its source variable.
				for (size_t i = 0; i < loopCarriedVarIdsList.size(); ++i)
					if (i < whileOp->getNumResults())
						m_valueMap[loopCarriedVarIdsList[i]] = whileOp->getResult(i);

				m_loopCarriedVarIds = savedLoopCarried;
				m_immediateLoopContext = savedLoopContext;

				// This allows subsequent code to reference the final values
				if (whileOp->getNumResults() > 0)
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
			std::set<int64_t> conditionModified;
			collectExpressionModifiedVariables(whileStmt->condition(), conditionModified);
			loopCarriedVarIds.insert(conditionModified.begin(), conditionModified.end());

			// Store loop-carried variables for break/continue handling
			auto const savedLoopCarried = m_loopCarriedVarIds;
			auto const savedLoopContext = m_immediateLoopContext;
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
				for (size_t i = 0; i < loopCarriedTypes.size(); ++i)
					blockArgs.push_back(i < loopCarriedVarIdsList.size()
						? m_valueMap.at(loopCarriedVarIdsList[i])
						: beforeBlock->getArgument(i));

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

			// Restore an enclosing loop's transfer state.
			m_loopCarriedVarIds = savedLoopCarried;
			m_immediateLoopContext = savedLoopContext;

			// Return the first result if available
			if (whileOp->getNumResults() > 0)
				return whileOp->getResult(0);
		}
		else if (auto* ret = dynamic_cast<Return const*>(&_stmt))
		{
			llvm::SmallVector<mlir::Value, 4> values;
			std::set<unsigned> storageResults;
			if (ret->expression())
			{
				Expression const& returned = withoutParentheses(*ret->expression());
				auto const* tuple = dynamic_cast<TupleExpression const*>(&returned);
				if (tuple && !tuple->isInlineArray() && tuple->components().size() > 1)
					// `return (a, b)` is one value per component. Generating the
					// tuple as a single expression produced only the first, and
					// the arity padding then made up the difference with
					// placeholders - so every result after the first came back
					// zero, with nothing to say so.
					for (auto const& component: tuple->components())
					{
						if (!component)
							continue;
						if (isStorageReferenceType(component->annotation().type))
							storageResults.insert(static_cast<unsigned>(values.size()));
						if (mlir::Value value = generateSolidityExpression(*component))
							values.push_back(value);
					}
				else if (mlir::Value value = generateSolidityExpression(returned))
				{
					if (isStorageReferenceType(returned.annotation().type))
						storageResults.insert(0);
					// `return f()` where f returns a tuple is the same story:
					// the expression is one value, the call op has several.
					mlir::Operation* producer = value.getDefiningOp();
					if (producer && producer->getNumResults() > 1 && producer->getResult(0) == value)
						for (mlir::Value result: producer->getResults())
							values.push_back(result);
					else
						values.push_back(value);
				}
			}

			emitReturn(loc, values, storageResults);
		}
		else if (auto* breakStmt = dynamic_cast<Break const*>(&_stmt))
		{
			(void) breakStmt;
			if (insideSCFRegion())
				emitNestedControlTransfer(loc, "break", currentLoopValues());
			else
				m_builder->create<mlir::solidity::BreakOp>(loc);
			return nullptr; // Return null to indicate we've handled the terminator
		}
		else if (auto* continueStmt = dynamic_cast<Continue const*>(&_stmt))
		{
			(void) continueStmt;
			if (insideSCFRegion())
				emitNestedControlTransfer(loc, "continue", currentLoopValues());
			else
				m_builder->create<mlir::solidity::ContinueOp>(loc);
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
			if (insideSCFRegion())
				emitNestedControlTransfer(loc, "revert");
			else
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
			bool anonymousEvent = false;
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
					anonymousEvent = eventDef->isAnonymous();
				}
				if (auto const* eventDef = dynamic_cast<EventDefinition const*>(&funcType->declaration()))
				{
					for (auto const& param: eventDef->parameters())
						indexed.push_back(param->isIndexed());
				}
			}

			// Generate arguments for the event
			std::vector<mlir::Value> args;
			std::vector<mlir::Attribute> storageArgs;
			for (auto const& arg: eventCall.arguments())
			{
				auto argValue = generateSolidityExpression(*arg);
				if (argValue)
				{
					args.push_back(argValue);
					if (auto const* referenceType
						= dynamic_cast<ReferenceType const*>(arg->annotation().type);
						referenceType && referenceType->location() == DataLocation::Storage)
						storageArgs.push_back(m_builder->getI32IntegerAttr(
							static_cast<int32_t>(args.size() - 1)));
				}
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

			auto emit = m_builder->create<mlir::solidity::EmitOp>(
				loc, m_builder->getStringAttr(eventName),
				m_builder->getStringAttr(eventSignature),
				indexedArrayAttr,
				anonymousEvent ? m_builder->getUnitAttr() : mlir::UnitAttr(),
				args);
			if (!storageArgs.empty())
				emit->setAttr("storage_args", m_builder->getArrayAttr(storageArgs));
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

						if (producer && i < producer->getNumResults())
							m_valueMap[decl->id()] = coerceValue(declLoc, producer->getResult(i), declType);
						else if (i == 0 && initValue)
							m_valueMap[decl->id()] = coerceValue(declLoc, initValue, declType);
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
				mlir::Value declaredValue = value;
				for (auto const& decl: declarations)
				{
					if (decl)
					{
						mlir::Type declaredType = translateSolidityType(*decl->type());
						auto const* sourceReference = dynamic_cast<ReferenceType const*>(
							varDeclStmt->initialValue()->annotation().type);
						auto const* targetReference = dynamic_cast<ReferenceType const*>(decl->type());
						if (sourceReference && targetReference
							&& sourceReference->location() == DataLocation::Storage
							&& targetReference->location() == DataLocation::Memory)
							declaredValue = m_builder->create<mlir::solidity::StorageToMemoryOp>(
								this->loc(*decl), declaredType, value).getResult();
						else
							declaredValue = coerceValue(this->loc(*decl), value, declaredType);
						m_valueMap[decl->id()] = declaredValue;
					}
				}
				return declaredValue;
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

						// A memory array is a pointer, and zero is not one:
						// leaving it there put the elements in the scratch area
						// at address 0, over the free memory pointer that lives
						// at 0x40. It has to own the words it addresses.
						if (auto const* array = dynamic_cast<ArrayType const*>(decl->type()))
							if (array->location() == DataLocation::Memory)
							{
								int64_t const initialLength = array->isDynamicallySized()
									? 0 : static_cast<int64_t>(array->length());
								mlir::Value length = m_builder->create<mlir::solidity::ConstantOp>(
									declLoc,
									m_builder->getIntegerAttr(
										m_builder->getI64Type(), initialLength),
									mlir::solidity::UIntType::get(m_context.get(), 256));
								m_valueMap[decl->id()] = m_builder->create<mlir::solidity::MemoryArrayCreateOp>(
									declLoc, declType, length).getResult();
								continue;
							}
						if (auto const* structure = dynamic_cast<StructType const*>(decl->type());
							structure && structure->location() == DataLocation::Memory)
						{
							m_valueMap[decl->id()] = m_builder->create<mlir::solidity::StructCreateOp>(
								declLoc, declType,
								m_builder->getStringAttr(structure->structDefinition().name()),
								mlir::ValueRange{}).getResult();
							continue;
						}
						if (mlir::isa<mlir::solidity::StringType, mlir::solidity::DynamicBytesType>(declType))
						{
							mlir::Value zero = m_builder->create<mlir::solidity::ConstantOp>(
								declLoc, m_builder->getIntegerAttr(m_builder->getI64Type(), 0),
								mlir::solidity::UIntType::get(m_context.get(), 256));
							m_valueMap[decl->id()] = m_builder->create<mlir::solidity::MemoryArrayCreateOp>(
								declLoc, declType, zero).getResult();
							continue;
						}

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

			// The block reads Solidity variables as well as writing them, so
			// the values it starts from are operands. Without them every
			// reference began at zero.
			std::vector<mlir::Value> inputs;
			std::vector<mlir::Attribute> inputNames;
			std::set<std::string> seenInputs;
			for (auto const& [yulIdent, info]: inlineAsm->annotation().externalReferences)
				if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(info.declaration))
				{
					std::string const inputName = yulIdent->name.str();
					if (!seenInputs.insert(inputName).second)
						continue;
					mlir::Value value;
					if (varDecl->isStateVariable() && (info.suffix == "slot" || info.suffix == "offset"))
					{
						std::string const name = stateVarName(*varDecl);
						uint64_t raw = 0;
						if (info.suffix == "slot")
						{
							auto slot = m_stateVarSlots.find(name);
							if (slot == m_stateVarSlots.end())
								continue;
							raw = static_cast<uint64_t>(slot->second);
						}
						else
						{
							auto offset = m_stateVarOffsets.find(name);
							if (offset == m_stateVarOffsets.end())
								continue;
							raw = offset->second;
						}
						value = m_builder->create<mlir::solidity::ConstantOp>(
							loc,
							m_builder->getIntegerAttr(m_builder->getI64Type(), static_cast<int64_t>(raw)),
							mlir::solidity::UIntType::get(m_context.get(), 256));
					}
					else if (info.suffix == "offset" && varDecl->type()->dataStoredIn(DataLocation::Storage))
						// Local storage references carry a whole-word slot in the sol
						// dialect. They cannot point into a packed word, so their offset
						// is exactly zero.
						value = m_builder->create<mlir::solidity::ConstantOp>(
							loc,
							m_builder->getIntegerAttr(m_builder->getI64Type(), 0),
							mlir::solidity::UIntType::get(m_context.get(), 256));
					else if (auto known = m_valueMap.find(varDecl->id()); known != m_valueMap.end())
						value = known->second;

					if (!value)
						continue;
					inputs.push_back(value);
					inputNames.push_back(m_builder->getStringAttr(inputName));
				}

			m_builder->create<mlir::solidity::InlineAssemblyOp>(
				loc, m_builder->getStringAttr(yulSource), inputs, m_builder->getArrayAttr(inputNames));

			// For external references pointing to Solidity variables,
			// create AssemblyBindOp so subsequent code can reference them.
			// This handles patterns like: address proxy; assembly { proxy := create2(...) }
			std::set<std::string> seenBindings;
			for (auto const& [yulIdent, info]: inlineAsm->annotation().externalReferences)
			{
				if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(info.declaration))
				{
					std::string const bindingName = yulIdent->name.str();
					if (!seenBindings.insert(bindingName).second)
						continue;
					auto varType = translateSolidityType(*varDecl->type());
					auto bindOp = m_builder->create<mlir::solidity::AssemblyBindOp>(
						loc, varType, m_builder->getStringAttr(bindingName));
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
		else if (auto* enumType = dynamic_cast<EnumType const*>(&_type))
		{
			// ABI coder v1 intentionally accepts dirty enum inputs, but the
			// value representation is still the enum's uint8 encoding type.
			return translateSolidityType(*enumType->encodingType());
		}
		else if (auto* userType = dynamic_cast<UserDefinedValueType const*>(&_type))
		{
			// Keep the dialect representation identical to the wrapped elementary
			// type. In particular, signed widths, addresses and bytesN must not
			// silently become uint256.
			return translateSolidityType(userType->underlyingType());
		}
		else if (auto* sliceType = dynamic_cast<ArraySliceType const*>(&_type))
		{
			// ArraySliceType is distinct in the Solidity AST, but after the
			// slice operation materialises it its value representation is the
			// same as the byte array being sliced.
			return translateSolidityType(sliceType->arrayType());
		}
		else if (auto* arrayType = dynamic_cast<ArrayType const*>(&_type))
		{
			// string and bytes are special ArrayType sub-kinds
			if (arrayType->isString())
				return mlir::solidity::StringType::get(m_context.get());
			if (arrayType->isByteArray())
				return mlir::solidity::DynamicBytesType::get(m_context.get());

			Type const* element = arrayType->baseType();
			// A storage array value is still represented by its base slot, but
			// retaining its recursive element shape is essential for indexing and
			// deep copies involving arrays of structs.
			if (arrayType->location() == DataLocation::Storage)
				element = TypeProvider::withLocationIfReference(DataLocation::Memory, element);
			auto elemType = translateSolidityType(*element);
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
			// Storage structs are represented by their base slot throughout the
			// storage-reference lowering. Recursive structs cannot be represented
			// by a finite MLIR type, and mapping-bearing structs are not ABI values.
			// Keep those pointer/reference-only shapes as a word; memory/calldata
			// ABI structs get the full recursive field description below.
			if (structType->location() == DataLocation::Storage
				|| structType->recursive() || structType->containsNestedMapping())
				return mlir::solidity::UIntType::get(m_context.get(), 256);
			llvm::SmallVector<mlir::Type, 4> fieldTypes;
			llvm::SmallVector<int64_t, 4> fieldSlots;
			llvm::SmallVector<int64_t, 4> fieldByteOffsets;
			for (Type const* memberType: structType->memoryMemberTypes())
				fieldTypes.push_back(translateSolidityType(*memberType));
			for (auto const& member: structType->structDefinition().members())
			{
				auto const [slot, byteOffset] = structType->storageOffsetsOfMember(member->name());
				fieldSlots.push_back(static_cast<int64_t>(slot));
				fieldByteOffsets.push_back(static_cast<int64_t>(byteOffset));
			}
			return mlir::solidity::StructType::get(
				m_context.get(), fieldTypes, fieldSlots, fieldByteOffsets);
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

		for (auto const& ret: safeReturnParameters(_func))
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
		for (auto const& ret: safeReturnParameters(_func))
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
					for (auto const& ret: safeReturnParameters(*funcDef))
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
	bool m_legacyCodegen = false;

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
	std::map<std::string, unsigned> m_stateVarOffsets;
	std::map<int64_t, std::string> m_stateVarNames;
	std::set<std::string> m_usedStateVarNames;

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
	CompilerStack const& _compilerStack,
	langutil::EVMVersion _evmVersion,
	OptimiserSettings const& _optimiserSettings,
	bool _legacyCodegen)
	: m_impl(std::make_unique<MLIRGeneratorImpl>(
		_compilerStack, _evmVersion, _optimiserSettings, _legacyCodegen)),
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
