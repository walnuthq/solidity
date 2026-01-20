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

#include <libsolidity/codegen/mlir/MLIRGenerator.h>
#include <libsolidity/codegen/mlir/MLIRToYulLowering.h>
#include <libsolidity/ast/AST.h>
#include <libsolidity/ast/ASTAnnotations.h>
#include <libsolidity/ast/Types.h>
#include <libsolidity/interface/CompilerStack.h>

#include <sstream>
#include <stack>
#include <map>
#include <set>

#ifdef SOLIDITY_HAS_MLIR
// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#pragma GCC diagnostic pop
#include "Dialect/SolidityDialect.h" 
#include "Dialect/SolidityOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#endif

namespace solidity::frontend
{

// Private implementation class
class MLIRGenerator::MLIRGeneratorImpl
{
public:
	MLIRGeneratorImpl(
		CompilerStack const& _compilerStack,
		langutil::EVMVersion _evmVersion,
		OptimiserSettings const& _optimiserSettings
	):
		m_compilerStack(_compilerStack),
		m_evmVersion(_evmVersion),
		m_optimiserSettings(_optimiserSettings)
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
		auto const& charStream = m_compilerStack.charStream(
			_location.sourceName ? *_location.sourceName : sourceName
		);
		
		// Convert start position to line:column
		auto lineCol = charStream.translatePositionToLineColumn(_location.start);
		
		// MLIR uses 1-based line numbers and 0-based column numbers
		// Solidity CharStream uses 0-based for both
		return mlir::FileLineColLoc::get(
			m_builder->getStringAttr(sourceName),
			lineCol.line + 1,  // Convert to 1-based
			lineCol.column     // Keep 0-based for MLIR
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
#ifdef SOLIDITY_HAS_MLIR
		auto loc = this->loc(_contract);
		
		// Set insertion point to module body
		auto moduleBody = m_module.getBody();
		m_builder->setInsertionPointToEnd(moduleBody);
		
		// Create contract operation using manual operation creation
		auto contractType = mlir::FunctionType::get(m_context.get(), {}, {});
		mlir::OperationState contractState(loc, "solidity.contract");
		contractState.addAttribute("name", m_builder->getStringAttr(_contract.name()));
		contractState.addAttribute("id", m_builder->getI64IntegerAttr(_contract.id()));
		contractState.addRegion();
		auto contractOp = m_builder->create(contractState);
		
		// Set insertion point to contract body
		m_builder->setInsertionPointToEnd(&contractOp->getRegion(0).emplaceBlock());
		
		// Generate state variables from all base contracts (in reverse order: base to derived)
		for (auto it = _contract.annotation().linearizedBaseContracts.rbegin();
		     it != _contract.annotation().linearizedBaseContracts.rend(); ++it)
		{
			ContractDefinition const* baseContract = *it;
			for (auto const& var : baseContract->stateVariables())
			{
				generateStateVariable(*var);
			}
		}

		// Generate functions from all base contracts (in reverse order: base to derived)
		// Track generated functions by signature to avoid duplicates from overrides
		std::set<std::string> generatedFunctions;
		for (auto it = _contract.annotation().linearizedBaseContracts.rbegin();
		     it != _contract.annotation().linearizedBaseContracts.rend(); ++it)
		{
			ContractDefinition const* baseContract = *it;
			for (auto const& func : baseContract->definedFunctions())
			{
				if (!func->isConstructor())
				{
					// Create a signature to track what we've generated
					std::string signature = func->name() + "(";
					for (size_t i = 0; i < func->parameters().size(); ++i)
					{
						if (i > 0) signature += ",";
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
				}
			}
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
	void generateStateVariable(VariableDeclaration const& _var)
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
		
		// Create state variable operation manually
		mlir::OperationState stateVarState(loc, "solidity.state_var");
		stateVarState.addAttribute("name", m_builder->getStringAttr(_var.name()));
		stateVarState.addAttribute("type", mlir::TypeAttr::get(solidityType));
		stateVarState.addAttribute("visibility", m_builder->getStringAttr(visibility));
		if (_var.isConstant())
			stateVarState.addAttribute("isConstant", m_builder->getUnitAttr());
		auto stateVarOp = m_builder->create(stateVarState);
		
		// Store reference for later use - store as Operation*
		m_stateVarOpMap[_var.id()] = stateVarOp;
	}
	
	void generateSolidityFunction(FunctionDefinition const& _func)
	{
		auto loc = this->loc(_func);
		
		// Build function type using Solidity types
		std::vector<mlir::Type> paramTypes;
		for (auto const& param : _func.parameters())
		{
			paramTypes.push_back(translateSolidityType(*param->type()));
		}
		
		std::vector<mlir::Type> returnTypes;
		for (auto const& ret : _func.returnParameters())
		{
			returnTypes.push_back(translateSolidityType(*ret->type()));
		}
		
		auto funcType = m_builder->getFunctionType(paramTypes, returnTypes);
		
		// Determine visibility and mutability
		std::string visibility = "public";
		if (_func.visibility() == Visibility::Private)
			visibility = "private";
		else if (_func.visibility() == Visibility::Internal)
			visibility = "internal";
		else if (_func.visibility() == Visibility::External)
			visibility = "external";
		
		std::string mutability = "nonpayable";
		if (_func.stateMutability() == StateMutability::Pure)
			mutability = "pure";
		else if (_func.stateMutability() == StateMutability::View)
			mutability = "view";
		else if (_func.stateMutability() == StateMutability::Payable)
			mutability = "payable";
		
		// Create Solidity function operation manually
		mlir::OperationState funcState(loc, "solidity.func");
		funcState.addAttribute("sym_name", m_builder->getStringAttr(_func.name()));
		funcState.addAttribute("function_type", mlir::TypeAttr::get(funcType));
		funcState.addAttribute("visibility", m_builder->getStringAttr(visibility));
		funcState.addAttribute("stateMutability", m_builder->getStringAttr(mutability));
		funcState.addRegion();
		auto funcOp = m_builder->create(funcState);
		
		// Create entry block with arguments
		auto& entryBlock = funcOp->getRegion(0).emplaceBlock();
		
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
			if (lastOp.getName().getStringRef() == "solidity.return")
				hasReturn = true;
		}
		
		if (!hasReturn)
		{
			mlir::OperationState returnState(loc, "solidity.return");
			m_builder->create(returnState);
		}
		
		// Restore insertion point
		m_builder->restoreInsertionPoint(savedIP);
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
					if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
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
						if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
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
			for (auto const& stmt : block->statements())
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
		// Note: Don't recurse into nested loops
		
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
	
	mlir::Value generateSolidityExpression(Expression const& _expr)
	{
		auto loc = this->loc(_expr);
		
		if (auto* literal = dynamic_cast<Literal const*>(&_expr))
		{
			auto type = translateSolidityType(*_expr.annotation().type);
			
			if (literal->token() == langutil::Token::Number)
			{
				auto value = std::stoull(literal->value());
				auto attr = m_builder->getIntegerAttr(m_builder->getI64Type(), value);
				return m_builder->create<mlir::solidity::ConstantOp>(loc, attr, type);
			}
			else if (literal->token() == langutil::Token::TrueLiteral)
			{
				auto attr = m_builder->getBoolAttr(true);
				mlir::OperationState constantState(loc, "solidity.constant");
				constantState.addAttribute("value", attr);
				constantState.addTypes(type);
				return m_builder->create(constantState)->getResult(0);
			}
			else if (literal->token() == langutil::Token::FalseLiteral)
			{
				auto attr = m_builder->getBoolAttr(false);
				mlir::OperationState constantState(loc, "solidity.constant");
				constantState.addAttribute("value", attr);
				constantState.addTypes(type);
				return m_builder->create(constantState)->getResult(0);
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
					// Create a zero value as a fallback to prevent crashes
					auto type = translateSolidityType(*_expr.annotation().type);
					auto zeroAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
					mlir::OperationState constState(loc, "solidity.constant");
					constState.addAttribute("value", zeroAttr);
					constState.addTypes(type);
					auto zeroValue = m_builder->create(constState)->getResult(0);
					// Store it for future reference
					m_valueMap[varDecl->id()] = zeroValue;
					return zeroValue;
				}
			}
		}
		else if (auto* binOp = dynamic_cast<BinaryOperation const*>(&_expr))
		{
			auto lhs = generateSolidityExpression(binOp->leftExpression());
			auto rhs = generateSolidityExpression(binOp->rightExpression());
			
			// Check if either operand is null
			if (!lhs || !rhs)
			{
				// Return a zero constant as fallback
				auto type = translateSolidityType(*_expr.annotation().type);
				auto zeroAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
				mlir::OperationState constState(loc, "solidity.constant");
				constState.addAttribute("value", zeroAttr);
				constState.addTypes(type);
				return m_builder->create(constState)->getResult(0);
			}
			
			auto resultType = lhs.getType();
			
			switch (binOp->getOperator())
			{
			case langutil::Token::Add:
			{
				return m_builder->create<mlir::solidity::AddOp>(loc, resultType, lhs, rhs);
			}
			case langutil::Token::Sub:
			{
				mlir::OperationState subState(loc, "solidity.sub");
				subState.addOperands({lhs, rhs});
				subState.addTypes(resultType);
				return m_builder->create(subState)->getResult(0);
			}
			case langutil::Token::Mul:
			{
				mlir::OperationState mulState(loc, "solidity.mul");
				mulState.addOperands({lhs, rhs});
				mulState.addTypes(resultType);
				return m_builder->create(mulState)->getResult(0);
			}
			case langutil::Token::Div:
			{
				mlir::OperationState divState(loc, "solidity.div");
				divState.addOperands({lhs, rhs});
				divState.addTypes(resultType);
				return m_builder->create(divState)->getResult(0);
			}
			case langutil::Token::Mod:
			{
				mlir::OperationState modState(loc, "solidity.mod");
				modState.addOperands({lhs, rhs});
				modState.addTypes(resultType);
				return m_builder->create(modState)->getResult(0);
			}
			case langutil::Token::LessThan:
			{
				mlir::OperationState cmpState(loc, "solidity.cmp");
				cmpState.addAttribute("predicate", m_builder->getStringAttr("lt"));
				cmpState.addOperands({lhs, rhs});
				cmpState.addTypes(mlir::solidity::BoolType::get(m_context.get()));
				return m_builder->create(cmpState)->getResult(0);
			}
			case langutil::Token::GreaterThan:
			{
				mlir::OperationState cmpState(loc, "solidity.cmp");
				cmpState.addAttribute("predicate", m_builder->getStringAttr("gt"));
				cmpState.addOperands({lhs, rhs});
				cmpState.addTypes(mlir::solidity::BoolType::get(m_context.get()));
				return m_builder->create(cmpState)->getResult(0);
			}
			case langutil::Token::Equal:
			{
				mlir::OperationState cmpState(loc, "solidity.cmp");
				cmpState.addAttribute("predicate", m_builder->getStringAttr("eq"));
				cmpState.addOperands({lhs, rhs});
				cmpState.addTypes(mlir::solidity::BoolType::get(m_context.get()));
				return m_builder->create(cmpState)->getResult(0);
			}
			case langutil::Token::LessThanOrEqual:
			{
				mlir::OperationState cmpState(loc, "solidity.cmp");
				cmpState.addAttribute("predicate", m_builder->getStringAttr("le"));
				cmpState.addOperands({lhs, rhs});
				cmpState.addTypes(mlir::solidity::BoolType::get(m_context.get()));
				return m_builder->create(cmpState)->getResult(0);
			}
			case langutil::Token::GreaterThanOrEqual:
			{
				mlir::OperationState cmpState(loc, "solidity.cmp");
				cmpState.addAttribute("predicate", m_builder->getStringAttr("ge"));
				cmpState.addOperands({lhs, rhs});
				cmpState.addTypes(mlir::solidity::BoolType::get(m_context.get()));
				return m_builder->create(cmpState)->getResult(0);
			}
			case langutil::Token::NotEqual:
			{
				mlir::OperationState cmpState(loc, "solidity.cmp");
				cmpState.addAttribute("predicate", m_builder->getStringAttr("ne"));
				cmpState.addOperands({lhs, rhs});
				cmpState.addTypes(mlir::solidity::BoolType::get(m_context.get()));
				return m_builder->create(cmpState)->getResult(0);
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
					if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (varDecl->isStateVariable())
						{
							mlir::OperationState loadState(loc, "solidity.load_state");
							loadState.addAttribute("varName", m_builder->getStringAttr(varDecl->name()));
							loadState.addTypes(translateSolidityType(*varDecl->type()));
							auto* loadOp = m_builder->create(loadState);
							currentValue = loadOp->getResult(0);
						}
						else
						{
							currentValue = m_valueMap[varDecl->id()];
						}
					}
				}
				
				// Get the right-hand side value
				auto rightValue = generateSolidityExpression(assignment->rightHandSide());
				
				// Check if either operand is null
				if (!currentValue || !rightValue)
				{
					// Return a zero constant as fallback
					auto type = translateSolidityType(*_expr.annotation().type);
					auto zeroAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
					mlir::OperationState constState(loc, "solidity.constant");
					constState.addAttribute("value", zeroAttr);
					constState.addTypes(type);
					return m_builder->create(constState)->getResult(0);
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
						mlir::OperationState storeState(loc, "solidity.store_state");
						storeState.addAttribute("varName", m_builder->getStringAttr(varDecl->name()));
						storeState.addOperands(value);
						m_builder->create(storeState);
					}
					else
					{
						m_valueMap[varDecl->id()] = value;
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
				mlir::OperationState notState(loc, "solidity.not");
				notState.addOperands(operand);
				notState.addTypes(resultType);
				return m_builder->create(notState)->getResult(0);
			}
			case langutil::Token::Inc:
			{
				// Pre/post increment: x++ or ++x
				auto one = m_builder->getIntegerAttr(m_builder->getI64Type(), 1);
				mlir::OperationState constState(loc, "solidity.constant");
				constState.addAttribute("value", one);
				constState.addTypes(resultType);
				auto oneValue = m_builder->create(constState)->getResult(0);
				
				mlir::OperationState addState(loc, "solidity.add");
				addState.addOperands({operand, oneValue});
				addState.addTypes(resultType);
				auto result = m_builder->create(addState)->getResult(0);
				
				// Store back to variable if it's an lvalue
				if (auto* ident = dynamic_cast<Identifier const*>(&unaryOp->subExpression()))
				{
					if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (varDecl->isStateVariable())
						{
							mlir::OperationState storeState(loc, "solidity.store_state");
							storeState.addAttribute("varName", m_builder->getStringAttr(varDecl->name()));
							storeState.addOperands(result);
							m_builder->create(storeState);
						}
						else
						{
							m_valueMap[varDecl->id()] = result;
						}
					}
				}
				
				// Return old value for post-increment, new value for pre-increment
				return unaryOp->isPrefixOperation() ? result : operand;
			}
			case langutil::Token::Dec:
			{
				// Pre/post decrement: x-- or --x
				auto one = m_builder->getIntegerAttr(m_builder->getI64Type(), 1);
				mlir::OperationState constState(loc, "solidity.constant");
				constState.addAttribute("value", one);
				constState.addTypes(resultType);
				auto oneValue = m_builder->create(constState)->getResult(0);
				
				mlir::OperationState subState(loc, "solidity.sub");
				subState.addOperands({operand, oneValue});
				subState.addTypes(resultType);
				auto result = m_builder->create(subState)->getResult(0);
				
				// Store back to variable if it's an lvalue
				if (auto* ident = dynamic_cast<Identifier const*>(&unaryOp->subExpression()))
				{
					if (auto* varDecl = dynamic_cast<VariableDeclaration const*>(ident->annotation().referencedDeclaration))
					{
						if (varDecl->isStateVariable())
						{
							mlir::OperationState storeState(loc, "solidity.store_state");
							storeState.addAttribute("varName", m_builder->getStringAttr(varDecl->name()));
							storeState.addOperands(result);
							m_builder->create(storeState);
						}
						else
						{
							m_valueMap[varDecl->id()] = result;
						}
					}
				}
				
				// Return old value for post-decrement, new value for pre-decrement
				return unaryOp->isPrefixOperation() ? result : operand;
			}
			case langutil::Token::Sub:
			{
				// Unary minus
				auto zero = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
				mlir::OperationState constState(loc, "solidity.constant");
				constState.addAttribute("value", zero);
				constState.addTypes(resultType);
				auto zeroValue = m_builder->create(constState)->getResult(0);
				
				mlir::OperationState subState(loc, "solidity.sub");
				subState.addOperands({zeroValue, operand});
				subState.addTypes(resultType);
				return m_builder->create(subState)->getResult(0);
			}
			case langutil::Token::Not:
			{
				// Logical NOT
				mlir::OperationState notState(loc, "solidity.logical_not");
				notState.addOperands(operand);
				notState.addTypes(mlir::solidity::BoolType::get(m_context.get()));
				return m_builder->create(notState)->getResult(0);
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
						mlir::OperationState senderState(loc, "solidity.msg_sender");
						senderState.addTypes(mlir::solidity::AddressType::get(m_context.get()));
						return m_builder->create(senderState)->getResult(0);
					}
					else if (memberName == "value")
					{
						mlir::OperationState valueState(loc, "solidity.msg_value");
						valueState.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
						return m_builder->create(valueState)->getResult(0);
					}
					else if (memberName == "data")
					{
						mlir::OperationState dataState(loc, "solidity.msg_data");
						dataState.addTypes(mlir::solidity::ArrayType::get(
							mlir::solidity::UIntType::get(m_context.get(), 8), -1));
						return m_builder->create(dataState)->getResult(0);
					}
					else if (memberName == "sig")
					{
						mlir::OperationState sigState(loc, "solidity.msg_sig");
						sigState.addTypes(mlir::solidity::BytesType::get(m_context.get(), 4));
						return m_builder->create(sigState)->getResult(0);
					}
				}
				else if (magicType->kind() == MagicType::Kind::Block)
				{
					if (memberName == "timestamp")
					{
						mlir::OperationState timestampState(loc, "solidity.block_timestamp");
						timestampState.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
						return m_builder->create(timestampState)->getResult(0);
					}
					else if (memberName == "number")
					{
						mlir::OperationState numberState(loc, "solidity.block_number");
						numberState.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
						return m_builder->create(numberState)->getResult(0);
					}
					else if (memberName == "chainid")
					{
						mlir::OperationState chainidState(loc, "solidity.block_chainid");
						chainidState.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
						return m_builder->create(chainidState)->getResult(0);
					}
				}
				else if (magicType->kind() == MagicType::Kind::Transaction)
				{
					if (memberName == "origin")
					{
						mlir::OperationState originState(loc, "solidity.tx_origin");
						originState.addTypes(mlir::solidity::AddressType::get(m_context.get()));
						return m_builder->create(originState)->getResult(0);
					}
					else if (memberName == "gasprice")
					{
						mlir::OperationState gaspriceState(loc, "solidity.tx_gasprice");
						gaspriceState.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
						return m_builder->create(gaspriceState)->getResult(0);
					}
				}
			}

			auto base = generateSolidityExpression(memberAccess->expression());
			if (!base)
			{
				// Create a zero constant if base is null
				auto type = translateSolidityType(*_expr.annotation().type);
				auto zeroAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
				mlir::OperationState constState(loc, "solidity.constant");
				constState.addAttribute("value", zeroAttr);
				constState.addTypes(type);
				return m_builder->create(constState)->getResult(0);
			}

			// Handle array.length
			if (auto* arrayType = dynamic_cast<ArrayType const*>(baseType))
			{
				if (memberName == "length")
				{
					mlir::OperationState lengthState(loc, "solidity.array_length");
					lengthState.addOperands(base);
					lengthState.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
					return m_builder->create(lengthState)->getResult(0);
				}
			}
			// Handle struct member access
			else if (auto* structType = dynamic_cast<StructType const*>(baseType))
			{
				mlir::OperationState memberState(loc, "solidity.member_access");
				memberState.addOperands(base);
				memberState.addAttribute("member", m_builder->getStringAttr(memberName));
				memberState.addTypes(translateSolidityType(*_expr.annotation().type));
				return m_builder->create(memberState)->getResult(0);
			}
			// Handle address member access (balance, code, codehash)
			else if (dynamic_cast<AddressType const*>(baseType))
			{
				if (memberName == "balance")
				{
					mlir::OperationState balanceState(loc, "solidity.address_balance");
					balanceState.addOperands(base);
					balanceState.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
					return m_builder->create(balanceState)->getResult(0);
				}
				else if (memberName == "code")
				{
					mlir::OperationState codeState(loc, "solidity.address_code");
					codeState.addOperands(base);
					// bytes is a dynamic array of uint8, size -1 indicates dynamic
					codeState.addTypes(mlir::solidity::ArrayType::get(
						mlir::solidity::UIntType::get(m_context.get(), 8),
						/*size=*/-1
					));
					return m_builder->create(codeState)->getResult(0);
				}
				else if (memberName == "codehash")
				{
					mlir::OperationState codehashState(loc, "solidity.address_codehash");
					codehashState.addOperands(base);
					codehashState.addTypes(mlir::solidity::BytesType::get(m_context.get(), 32));
					return m_builder->create(codehashState)->getResult(0);
				}
			}

			// Return base for unhandled member access
			return base;
		}
		else if (auto* funcCall = dynamic_cast<FunctionCall const*>(&_expr))
		{
			// Handle member function calls first (e.g., array.push())
			if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&funcCall->expression()))
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
						mlir::OperationState pushState(loc, "solidity.array_push");
						pushState.addOperands({base, value});
						m_builder->create(pushState);
						return base;
					}
				}
			}
			// Handle struct constructor calls
			else if (auto* typeConversion = dynamic_cast<Identifier const*>(&funcCall->expression()))
			{
				if (auto* structDecl = dynamic_cast<StructDefinition const*>(typeConversion->annotation().referencedDeclaration))
				{
					// Create struct with named arguments
					mlir::OperationState structState(loc, "solidity.struct_create");
					structState.addAttribute("name", m_builder->getStringAttr(structDecl->name()));
					
					// Add field values
					for (auto const& arg : funcCall->arguments())
					{
						auto value = generateSolidityExpression(*arg);
						if (!value)
						{
							// Create a zero constant if value is null
							auto type = translateSolidityType(*arg->annotation().type);
							auto zeroAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
							mlir::OperationState constState(loc, "solidity.constant");
							constState.addAttribute("value", zeroAttr);
							constState.addTypes(type);
							value = m_builder->create(constState)->getResult(0);
						}
						structState.addOperands(value);
					}
					
					structState.addTypes(translateSolidityType(*_expr.annotation().type));
					return m_builder->create(structState)->getResult(0);
				}
			}
			
			if (auto* ident = dynamic_cast<Identifier const*>(&funcCall->expression()))
			{
				// Handle special functions like require, assert, revert
				if (ident->name() == "require")
				{
					if (!funcCall->arguments().empty())
					{
						auto cond = generateSolidityExpression(*funcCall->arguments()[0]);
						mlir::OperationState requireState(loc, "solidity.require");
						requireState.addOperands(cond);
						if (funcCall->arguments().size() > 1)
						{
							if (auto* literal = dynamic_cast<Literal const*>(funcCall->arguments()[1].get()))
								requireState.addAttribute("msg", m_builder->getStringAttr(literal->value()));
						}
						m_builder->create(requireState);
					}
				}
				else if (ident->name() == "assert")
				{
					if (!funcCall->arguments().empty())
					{
						auto cond = generateSolidityExpression(*funcCall->arguments()[0]);
						mlir::OperationState assertState(loc, "solidity.assert");
						assertState.addOperands(cond);
						m_builder->create(assertState);
					}
				}
				else if (ident->name() == "revert")
				{
					mlir::OperationState revertState(loc, "solidity.revert");
					if (!funcCall->arguments().empty())
					{
						if (auto* literal = dynamic_cast<Literal const*>(funcCall->arguments()[0].get()))
							revertState.addAttribute("reason", m_builder->getStringAttr(literal->value()));
					}
					m_builder->create(revertState);
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
					mlir::OperationState convertState(loc, "solidity.convert");
					convertState.addOperands(argValue);
					convertState.addTypes(targetType);
					return m_builder->create(convertState)->getResult(0);
				}
			}

			// Handle regular function calls (internal functions, etc.)
			if (funcCall->annotation().kind.set() && *funcCall->annotation().kind == FunctionCallKind::FunctionCall)
			{
				std::string funcName;

				// Get function name from identifier
				if (auto* ident = dynamic_cast<Identifier const*>(&funcCall->expression()))
				{
					funcName = ident->name();

					// Skip if already handled (require/assert/revert)
					if (funcName == "require" || funcName == "assert" || funcName == "revert")
					{
						// Already handled above, return dummy for expression value
						auto dummyType = translateSolidityType(*_expr.annotation().type);
						mlir::OperationState constState(loc, "solidity.constant");
						constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
						constState.addTypes(dummyType);
						return m_builder->create(constState)->getResult(0);
					}

					// Handle built-in functions with proper MLIR operations

					// addmod(a, b, n) - modular addition
					if (funcName == "addmod" && funcCall->arguments().size() == 3)
					{
						auto a = generateSolidityExpression(*funcCall->arguments()[0]);
						auto b = generateSolidityExpression(*funcCall->arguments()[1]);
						auto n = generateSolidityExpression(*funcCall->arguments()[2]);
						auto resultType = translateSolidityType(*_expr.annotation().type);

						mlir::OperationState addmodState(loc, "solidity.addmod");
						addmodState.addOperands({a, b, n});
						addmodState.addTypes(resultType);
						return m_builder->create(addmodState)->getResult(0);
					}

					// mulmod(a, b, n) - modular multiplication
					if (funcName == "mulmod" && funcCall->arguments().size() == 3)
					{
						auto a = generateSolidityExpression(*funcCall->arguments()[0]);
						auto b = generateSolidityExpression(*funcCall->arguments()[1]);
						auto n = generateSolidityExpression(*funcCall->arguments()[2]);
						auto resultType = translateSolidityType(*_expr.annotation().type);

						mlir::OperationState mulmodState(loc, "solidity.mulmod");
						mulmodState.addOperands({a, b, n});
						mulmodState.addTypes(resultType);
						return m_builder->create(mulmodState)->getResult(0);
					}

					// gasleft() - get remaining gas
					if (funcName == "gasleft" && funcCall->arguments().empty())
					{
						mlir::OperationState gasleftState(loc, "solidity.gasleft");
						gasleftState.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
						return m_builder->create(gasleftState)->getResult(0);
					}

					// blockhash(blockNumber) - get block hash
					if (funcName == "blockhash" && funcCall->arguments().size() == 1)
					{
						auto blockNumber = generateSolidityExpression(*funcCall->arguments()[0]);

						mlir::OperationState blockhashState(loc, "solidity.blockhash");
						blockhashState.addOperands(blockNumber);
						blockhashState.addTypes(mlir::solidity::BytesType::get(m_context.get(), 32));
						return m_builder->create(blockhashState)->getResult(0);
					}

					// keccak256(data) - hash function
					if (funcName == "keccak256" && funcCall->arguments().size() == 1)
					{
						auto data = generateSolidityExpression(*funcCall->arguments()[0]);

						mlir::OperationState keccakState(loc, "solidity.keccak256");
						keccakState.addOperands(data);
						keccakState.addTypes(mlir::solidity::BytesType::get(m_context.get(), 32));
						return m_builder->create(keccakState)->getResult(0);
					}

					// sha256(data) - hash function
					if (funcName == "sha256" && funcCall->arguments().size() == 1)
					{
						auto data = generateSolidityExpression(*funcCall->arguments()[0]);

						mlir::OperationState sha256State(loc, "solidity.sha256");
						sha256State.addOperands(data);
						sha256State.addTypes(mlir::solidity::BytesType::get(m_context.get(), 32));
						return m_builder->create(sha256State)->getResult(0);
					}

					// ripemd160(data) - hash function
					if (funcName == "ripemd160" && funcCall->arguments().size() == 1)
					{
						auto data = generateSolidityExpression(*funcCall->arguments()[0]);

						mlir::OperationState ripemdState(loc, "solidity.ripemd160");
						ripemdState.addOperands(data);
						// ripemd160 returns bytes20, but padded to 32 bytes in EVM
						ripemdState.addTypes(mlir::solidity::BytesType::get(m_context.get(), 20));
						return m_builder->create(ripemdState)->getResult(0);
					}

					// ecrecover(hash, v, r, s) - signature recovery
					if (funcName == "ecrecover" && funcCall->arguments().size() == 4)
					{
						auto hash = generateSolidityExpression(*funcCall->arguments()[0]);
						auto v = generateSolidityExpression(*funcCall->arguments()[1]);
						auto r = generateSolidityExpression(*funcCall->arguments()[2]);
						auto s = generateSolidityExpression(*funcCall->arguments()[3]);

						mlir::OperationState ecrecoverState(loc, "solidity.ecrecover");
						ecrecoverState.addOperands({hash, v, r, s});
						ecrecoverState.addTypes(mlir::solidity::AddressType::get(m_context.get()));
						return m_builder->create(ecrecoverState)->getResult(0);
					}

					// selfdestruct(recipient) - destroy contract
					if (funcName == "selfdestruct" && funcCall->arguments().size() == 1)
					{
						auto recipient = generateSolidityExpression(*funcCall->arguments()[0]);

						mlir::OperationState selfdestructState(loc, "solidity.selfdestruct");
						selfdestructState.addOperands(recipient);
						m_builder->create(selfdestructState);

						// selfdestruct doesn't return a value, return a dummy for expression context
						auto dummyType = translateSolidityType(*_expr.annotation().type);
						mlir::OperationState constState(loc, "solidity.constant");
						constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
						constState.addTypes(dummyType);
						return m_builder->create(constState)->getResult(0);
					}

					// type(X) - handled separately as member access (type(uint256).max etc.)
					if (funcName == "type")
					{
						// The actual value is extracted from member access (type(X).max, type(X).min)
						// Just return a dummy here - real handling is in member access below
						auto dummyType = translateSolidityType(*_expr.annotation().type);
						mlir::OperationState constState(loc, "solidity.constant");
						constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
						constState.addTypes(dummyType);
						return m_builder->create(constState)->getResult(0);
					}
				}
				// Handle member function calls (e.g., library.func() or obj.method())
				else if (auto* memberAccess = dynamic_cast<MemberAccess const*>(&funcCall->expression()))
				{
					// Check for abi.encode, abi.encodePacked, abi.decode, type().max, etc.
					// These need special handling and cannot be generated as function calls
					if (auto* baseIdent = dynamic_cast<Identifier const*>(&memberAccess->expression()))
					{
						std::string baseName = baseIdent->name();
						if (baseName == "abi")
						{
							// abi.encode, abi.encodePacked, abi.encodeWithSelector, abi.decode, etc.
							// Skip to dummy value - these require special memory handling
							auto dummyType = translateSolidityType(*_expr.annotation().type);
							mlir::OperationState constState(loc, "solidity.constant");
							constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
							constState.addTypes(dummyType);
							return m_builder->create(constState)->getResult(0);
						}
					}
					// Check for type(X).max, type(X).min
					else if (auto* funcCallExpr = dynamic_cast<FunctionCall const*>(&memberAccess->expression()))
					{
						if (auto* typeIdent = dynamic_cast<Identifier const*>(&funcCallExpr->expression()))
						{
							if (typeIdent->name() == "type")
							{
								// Extract the type argument from type(X)
								std::string memberName = memberAccess->memberName();
								auto resultType = translateSolidityType(*_expr.annotation().type);

								// Get the type being queried from the type() call
								if (!funcCallExpr->arguments().empty())
								{
									auto const* typeArg = funcCallExpr->arguments()[0].get();
									if (auto* typeArgIdent = dynamic_cast<ElementaryTypeNameExpression const*>(typeArg))
									{
										auto const& typeName = typeArgIdent->type();

										if (memberName == "max")
										{
											// Generate max value constant
											mlir::OperationState constState(loc, "solidity.constant");

											if (auto* intType = dynamic_cast<IntegerType const*>(&typeName))
											{
												if (intType->isSigned())
												{
													// type(intN).max = 2^(N-1) - 1
													// For int256: 2^255 - 1
													unsigned bits = intType->numBits();
													u256 maxVal = (u256(1) << (bits - 1)) - 1;
													constState.addAttribute("value", m_builder->getIntegerAttr(
														mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
														llvm::APInt(256, maxVal.str(), 10)));
												}
												else
												{
													// type(uintN).max = 2^N - 1
													// For uint256: 2^256 - 1 = 0xffffffff...
													unsigned bits = intType->numBits();
													u256 maxVal = (u256(1) << bits) - 1;
													constState.addAttribute("value", m_builder->getIntegerAttr(
														mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
														llvm::APInt(256, maxVal.str(), 10)));
												}
											}
											else
											{
												// Default fallback
												constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
											}

											constState.addTypes(resultType);
											return m_builder->create(constState)->getResult(0);
										}
										else if (memberName == "min")
										{
											// Generate min value constant
											mlir::OperationState constState(loc, "solidity.constant");

											if (auto* intType = dynamic_cast<IntegerType const*>(&typeName))
											{
												if (intType->isSigned())
												{
													// type(intN).min = -2^(N-1)
													// For int256: -2^255
													// We need to represent this as a two's complement value
													unsigned bits = intType->numBits();
													// -2^(N-1) in two's complement is the value with only the sign bit set
													u256 minVal = u256(1) << (bits - 1);  // This is 2^(N-1), treated as signed it's the min
													constState.addAttribute("value", m_builder->getIntegerAttr(
														mlir::IntegerType::get(m_context.get(), 256, mlir::IntegerType::Unsigned),
														llvm::APInt(256, minVal.str(), 10)));
												}
												else
												{
													// type(uintN).min = 0
													constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
												}
											}
											else
											{
												// Default fallback
												constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
											}

											constState.addTypes(resultType);
											return m_builder->create(constState)->getResult(0);
										}
									}
								}

								// Fallback for other type() members
								auto dummyType = translateSolidityType(*_expr.annotation().type);
								mlir::OperationState constState(loc, "solidity.constant");
								constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
								constState.addTypes(dummyType);
								return m_builder->create(constState)->getResult(0);
							}
						}
					}

					funcName = memberAccess->memberName();

					// Check for address member functions (transfer, send, call, delegatecall, staticcall)
					// These are built-in functions on address types that need special Yul handling
					if (funcName == "transfer" || funcName == "send" || funcName == "call" ||
					    funcName == "delegatecall" || funcName == "staticcall")
					{
						// Skip to dummy value - these require special external call handling
						auto dummyType = translateSolidityType(*_expr.annotation().type);
						mlir::OperationState constState(loc, "solidity.constant");
						constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
						constState.addTypes(dummyType);
						return m_builder->create(constState)->getResult(0);
					}

					// For all other member access function calls (external contract/interface calls),
					// skip to dummy value as these are external calls that need special handling
					// Examples: receiver.onERC721Received(...), token.transfer(...), etc.
					auto dummyType = translateSolidityType(*_expr.annotation().type);
					mlir::OperationState constState(loc, "solidity.constant");
					constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
					constState.addTypes(dummyType);
					return m_builder->create(constState)->getResult(0);
				}

				if (!funcName.empty())
				{
					// Generate arguments
					std::vector<mlir::Value> args;
					for (auto const& arg : funcCall->arguments())
					{
						auto argValue = generateSolidityExpression(*arg);
						if (argValue)
							args.push_back(argValue);
					}

					// Create function call operation
					mlir::OperationState callState(loc, "solidity.function_call");
					callState.addAttribute("callee", m_builder->getStringAttr(funcName));
					callState.addOperands(args);

					// Add result type if the function has a return value
					if (!dynamic_cast<TupleType const*>(_expr.annotation().type) ||
					    !dynamic_cast<TupleType const*>(_expr.annotation().type)->components().empty())
					{
						auto resultType = translateSolidityType(*_expr.annotation().type);
						callState.addTypes(resultType);
						return m_builder->create(callState)->getResult(0);
					}
					else
					{
						// Void function
						m_builder->create(callState);
						return nullptr;
					}
				}
			}
		}

		// Return dummy value for unhandled cases
		// Create a default value with the correct type to avoid type mismatches
		auto dummyType = translateSolidityType(*_expr.annotation().type);
		mlir::OperationState constState(loc, "solidity.constant");
		constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
		constState.addTypes(dummyType);
		return m_builder->create(constState)->getResult(0);
	}

	mlir::Value generateSolidityStatement(Statement const& _stmt)
	{
		auto loc = this->loc(_stmt);
		
		if (auto* block = dynamic_cast<Block const*>(&_stmt))
		{
			if (block->unchecked())
			{
				// Unchecked blocks disable overflow/underflow checks
				mlir::OperationState uncheckedState(loc, "solidity.unchecked");
				uncheckedState.addRegion();
				auto uncheckedOp = m_builder->create(uncheckedState);

				// Generate body region with overflow checking disabled
				m_builder->setInsertionPointToEnd(&uncheckedOp->getRegion(0).emplaceBlock());
				for (auto const& stmt : block->statements())
					generateSolidityStatement(*stmt);

				// Reset insertion point after the unchecked block
				m_builder->setInsertionPointAfter(uncheckedOp);
			}
			else
			{
				mlir::Value lastValue;
				for (auto const& stmt : block->statements())
					lastValue = generateSolidityStatement(*stmt);
				return lastValue;
			}
		}
		else if (auto* ifStmt = dynamic_cast<IfStatement const*>(&_stmt))
		{
			auto condition = generateSolidityExpression(ifStmt->condition());
			bool hasElse = ifStmt->falseStatement() != nullptr;
			
			mlir::OperationState ifState(loc, "solidity.if");
			ifState.addOperands(condition);
			ifState.addRegion(); // then region
			ifState.addRegion(); // else region (always add both regions)
			auto ifOp = m_builder->create(ifState);
			
			// Generate then region
			m_builder->setInsertionPointToEnd(&ifOp->getRegion(0).emplaceBlock());
			generateSolidityStatement(ifStmt->trueStatement());
			
			// Generate else region if present
			if (hasElse)
			{
				m_builder->setInsertionPointToEnd(&ifOp->getRegion(1).emplaceBlock());
				generateSolidityStatement(*ifStmt->falseStatement());
			}
			else
			{
				// Create an empty else region - it will be empty which is valid for solidity.if
				ifOp->getRegion(1).emplaceBlock();
			}
			
			// Reset insertion point after the if statement
			// This ensures subsequent statements are generated after the if, not inside it
			m_builder->setInsertionPointAfter(ifOp);
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
				if (auto* varDeclStmt = dynamic_cast<VariableDeclarationStatement const*>(forStmt->initializationExpression()))
				{
					for (auto const& decl : varDeclStmt->declarations())
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
				for (auto varId : condReferencedVars)
				{
					if (bodyModifiedVars.count(varId) > 0 || varId == loopCounterId)
						loopCarriedVarIds.insert(varId);
				}
			}
			
			// Store loop-carried variables for break/continue handling
			m_loopCarriedVarIds = loopCarriedVarIds;
			m_immediateLoopContext = LoopContext::SCF;  // We're in an SCF loop
			
			// Prepare initial values and types for loop-carried variables
			std::vector<mlir::Value> initialValues;
			std::vector<mlir::Type> loopCarriedTypes;
			std::vector<int64_t> loopCarriedVarIdsList; // Keep ordered list
			
			for (auto varId : loopCarriedVarIds)
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
				mlir::OperationState zeroOp(loc, "solidity.constant");
				zeroOp.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
				zeroOp.addAttribute("value", m_builder->getI64IntegerAttr(0));
				auto zeroValue = m_builder->create(zeroOp);
				initialValues.push_back(zeroValue->getResult(0));
				loopCarriedTypes.push_back(mlir::solidity::UIntType::get(m_context.get(), 256));
			}
			
			// Create the scf.while with proper loop-carried values
			auto whileOp = m_builder->create<mlir::scf::WhileOp>(
				loc,
				mlir::TypeRange{loopCarriedTypes},
				mlir::ValueRange{initialValues}
			);
			
			// Build the "before" region (condition check)
			{
				auto& beforeRegion = whileOp.getBefore();
				auto* beforeBlock = m_builder->createBlock(&beforeRegion);
				
				// Add block arguments for loop-carried values
				for (auto type : loopCarriedTypes)
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
					mlir::OperationState toBoolOp(loc, "solidity.to_i1");
					toBoolOp.addOperands(solidityBool);
					toBoolOp.addTypes(m_builder->getI1Type());
					condValue = m_builder->create(toBoolOp)->getResult(0);
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
				for (auto type : loopCarriedTypes)
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
					// Collect updated values for all loop-carried variables
					std::vector<mlir::Value> updatedValues;
					for (auto varId : loopCarriedVarIdsList)
					{
						if (m_valueMap.count(varId) > 0)
							updatedValues.push_back(m_valueMap[varId]);
						else
							updatedValues.push_back(afterBlock->getArgument(0)); // Fallback
					}
					
					// Yield the updated loop-carried values
					m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{updatedValues});
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
			
			// Clear loop-carried variables tracking
			m_loopCarriedVarIds.clear();
			m_immediateLoopContext = LoopContext::None;  // Exiting SCF loop
			
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
				m_immediateLoopContext = LoopContext::SCF;  // We're in an SCF loop
				
				// Prepare initial values and types
				std::vector<mlir::Value> initialValues;
				std::vector<mlir::Type> loopCarriedTypes;
				std::vector<int64_t> loopCarriedVarIdsList;
				
				for (auto varId : loopCarriedVarIds)
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
					mlir::OperationState dummyOp(loc, "solidity.constant");
					dummyOp.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
					dummyOp.addAttribute("value", m_builder->getI64IntegerAttr(0));
					auto dummyValue = m_builder->create(dummyOp);
					initialValues.push_back(dummyValue->getResult(0));
					loopCarriedTypes.push_back(dummyValue->getResult(0).getType());
				}
				
				// Create the SCF while operation
				auto whileOp = m_builder->create<mlir::scf::WhileOp>(
					loc,
					loopCarriedTypes,
					initialValues
				);
				
				// Build the "before" region (condition check - always true for first iteration)
				auto& beforeRegion = whileOp.getBefore();
				auto* beforeBlock = m_builder->createBlock(&beforeRegion);
				// Add block arguments for loop-carried values
				for (auto type : loopCarriedTypes)
				{
					beforeBlock->addArgument(type, loc);
				}
				m_builder->setInsertionPointToEnd(beforeBlock);
				
				// Always continue for do-while (condition checked at end)
				auto trueVal = m_builder->create<mlir::arith::ConstantIntOp>(loc, 1, 1);
				m_builder->create<mlir::scf::ConditionOp>(
					loc,
					trueVal,
					beforeBlock->getArguments()
				);
				
				// Build the "after" region (loop body)
				auto& afterRegion = whileOp.getAfter();
				auto* afterBlock = m_builder->createBlock(&afterRegion);
				// Add block arguments for loop-carried values
				for (auto type : loopCarriedTypes)
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
				bool needsTerminator = currentBlock && 
					(currentBlock->empty() || !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>());
				
				if (needsTerminator)
				{
					
					// Prepare the values to yield
					std::vector<mlir::Value> updatedValues;
					for (auto varId : loopCarriedVarIdsList)
					{
						if (m_valueMap.count(varId) > 0)
							updatedValues.push_back(m_valueMap[varId]);
					}
					
					if (updatedValues.empty() && !initialValues.empty())
						updatedValues = {afterBlock->getArgument(0)};
					
					// Always add the yield for proper termination
					m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{updatedValues});
				}
				
				// Ensure all blocks in the after region have terminators
				// This is critical for SCF while loops
				for (auto& block : whileOp.getAfter())
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
				m_immediateLoopContext = LoopContext::None;  // Exiting SCF loop
				
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
				for (auto varId : condReferencedVars)
				{
					if (bodyModifiedVars.count(varId) > 0)
						loopCarriedVarIds.insert(varId);
				}
			}
			
			// Add all body-modified variables
			loopCarriedVarIds.insert(bodyModifiedVars.begin(), bodyModifiedVars.end());
			
			// Store loop-carried variables for break/continue handling
			m_loopCarriedVarIds = loopCarriedVarIds;
			m_immediateLoopContext = LoopContext::SCF;  // We're in an SCF loop
			
			// Prepare initial values and types for loop-carried variables
			std::vector<mlir::Value> initialValues;
			std::vector<mlir::Type> loopCarriedTypes;
			std::vector<int64_t> loopCarriedVarIdsList; // Keep ordered list
			
			for (auto varId : loopCarriedVarIds)
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
				mlir::OperationState zeroOp(loc, "solidity.constant");
				zeroOp.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
				zeroOp.addAttribute("value", m_builder->getI64IntegerAttr(0));
				auto zeroValue = m_builder->create(zeroOp);
				initialValues.push_back(zeroValue->getResult(0));
				loopCarriedTypes.push_back(mlir::solidity::UIntType::get(m_context.get(), 256));
			}
			
			// Create the scf.while with proper loop-carried values
			auto whileOp = m_builder->create<mlir::scf::WhileOp>(
				loc,
				mlir::TypeRange{loopCarriedTypes},
				mlir::ValueRange{initialValues}
			);
			
			// Build the "before" region (condition check)
			{
				auto& beforeRegion = whileOp.getBefore();
				auto* beforeBlock = m_builder->createBlock(&beforeRegion);
				
				// Add block arguments for loop-carried values
				for (auto type : loopCarriedTypes)
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
				mlir::OperationState toBoolOp(loc, "solidity.to_i1");
				toBoolOp.addOperands(solidityBool);
				toBoolOp.addTypes(m_builder->getI1Type());
				auto condValue = m_builder->create(toBoolOp)->getResult(0);
				
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
				for (auto type : loopCarriedTypes)
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
					// Collect updated values for all loop-carried variables
					std::vector<mlir::Value> updatedValues;
					for (auto varId : loopCarriedVarIdsList)
					{
						if (m_valueMap.count(varId) > 0)
							updatedValues.push_back(m_valueMap[varId]);
						else
							updatedValues.push_back(afterBlock->getArgument(0)); // Fallback
					}
					
					// Yield the updated loop-carried values
					m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{updatedValues});
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
			
			// Clear loop-carried variables tracking
			m_loopCarriedVarIds.clear();
			m_immediateLoopContext = LoopContext::None;  // Exiting SCF loop
			
			// Return the first result if available
			if (whileOp->getNumResults() > 0)
				return whileOp->getResult(0);
		}
		else if (auto* ret = dynamic_cast<Return const*>(&_stmt))
		{
			mlir::OperationState returnState(loc, "solidity.return");
			if (ret->expression())
			{
				auto value = generateSolidityExpression(*ret->expression());
				returnState.addOperands(value);
			}
			m_builder->create(returnState);
		}
		else if (auto* breakStmt = dynamic_cast<Break const*>(&_stmt))
		{
			// Only generate scf::YieldOp if we're in an immediate SCF loop
			if (m_immediateLoopContext == LoopContext::SCF)
			{
				// For break statements in SCF loops, we need to yield with current values
				// This is a simplified handling - proper implementation would track loop contexts
				// and branch to the appropriate exit block
				
				// If we're in a loop context (which we should be), create a yield
				// with the current loop-carried values
				// For now, just create a dummy yield to satisfy SCF requirements
				std::vector<mlir::Value> currentValues;
				
				// This is a simplified approach - we should track the loop context
				// and use the appropriate loop-carried values
				if (!m_loopCarriedVarIds.empty())
				{
					for (auto varId : m_loopCarriedVarIds)
					{
						if (m_valueMap.count(varId) > 0)
							currentValues.push_back(m_valueMap[varId]);
					}
				}
				
				// If no loop-carried values, create a dummy value
				if (currentValues.empty())
				{
					mlir::OperationState zeroOp(loc, "solidity.constant");
					zeroOp.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
					zeroOp.addAttribute("value", m_builder->getI64IntegerAttr(0));
					auto zeroValue = m_builder->create(zeroOp);
					currentValues.push_back(zeroValue->getResult(0));
				}
				
				// Create the yield to exit the loop
				m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{currentValues});
			}
			else
			{
				// Not in SCF context, generate a solidity.break operation
				mlir::OperationState breakState(loc, "solidity.break");
				m_builder->create(breakState);
			}
			return nullptr; // Return null to indicate we've handled the terminator
		}
		else if (auto* continueStmt = dynamic_cast<Continue const*>(&_stmt))
		{
			// Only generate scf::YieldOp if we're in an immediate SCF loop
			if (m_immediateLoopContext == LoopContext::SCF)
			{
				// For continue statements, we need to yield with updated values
				// to continue to the next iteration
				std::vector<mlir::Value> currentValues;
				
				if (!m_loopCarriedVarIds.empty())
				{
					for (auto varId : m_loopCarriedVarIds)
					{
						if (m_valueMap.count(varId) > 0)
							currentValues.push_back(m_valueMap[varId]);
					}
				}
				
				// If no loop-carried values, create a dummy value
				if (currentValues.empty())
				{
					mlir::OperationState zeroOp(loc, "solidity.constant");
					zeroOp.addTypes(mlir::solidity::UIntType::get(m_context.get(), 256));
					zeroOp.addAttribute("value", m_builder->getI64IntegerAttr(0));
					auto zeroValue = m_builder->create(zeroOp);
					currentValues.push_back(zeroValue->getResult(0));
				}
				
				// Create the yield to continue the loop
				m_builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{currentValues});
			}
			else
			{
				// Not in SCF context, generate a solidity.continue operation
				mlir::OperationState continueState(loc, "solidity.continue");
				m_builder->create(continueState);
			}
			return nullptr; // Return null to indicate we've handled the terminator
		}
		else if (auto* revertStmt = dynamic_cast<RevertStatement const*>(&_stmt))
		{
			// Handle revert statement with custom error
			mlir::OperationState revertState(loc, "solidity.revert");
			// Get the error name from the function call if possible
			if (auto* ident = dynamic_cast<Identifier const*>(&revertStmt->errorCall().expression()))
			{
				revertState.addAttribute("reason", m_builder->getStringAttr(ident->name()));
			}
			m_builder->create(revertState);
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

			// Generate arguments for the event
			std::vector<mlir::Value> args;
			for (auto const& arg : eventCall.arguments())
			{
				auto argValue = generateSolidityExpression(*arg);
				if (argValue)
					args.push_back(argValue);
			}

			// Create emit operation
			mlir::OperationState emitState(loc, "solidity.emit");
			emitState.addAttribute("event", m_builder->getStringAttr(eventName));
			emitState.addOperands(args);
			m_builder->create(emitState);
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
					// For tuple destructuring, create values with correct types for each declaration
					// The initialValue might return a tuple, but we handle each component separately
					for (auto const& decl : declarations)
					{
						if (decl)
						{
							// Create a value with the declaration's actual type
							auto declType = translateSolidityType(*decl->type());
							auto declLoc = this->loc(*decl);

							// Generate a placeholder value with the correct type
							// This handles cases like low-level calls returning (bool, bytes)
							if (dynamic_cast<BoolType const*>(decl->type()))
							{
								// For bool declarations, create a bool constant (will be set by the call)
								mlir::OperationState boolState(declLoc, "solidity.constant");
								boolState.addAttribute("value", m_builder->getBoolAttr(false));
								boolState.addTypes(mlir::solidity::BoolType::get(m_context.get()));
								auto boolValue = m_builder->create(boolState)->getResult(0);
								m_valueMap[decl->id()] = boolValue;
							}
							else
							{
								// For other types, create appropriate placeholder
								mlir::OperationState constState(declLoc, "solidity.constant");
								constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
								constState.addTypes(declType);
								auto value = m_builder->create(constState)->getResult(0);
								m_valueMap[decl->id()] = value;
							}
						}
					}
					// Generate the actual expression (e.g., the low-level call)
					// This is done for side effects
					generateSolidityExpression(*varDeclStmt->initialValue());
					return mlir::Value();
				}

				// Single declaration.
				auto value = generateSolidityExpression(*varDeclStmt->initialValue());
				if (!value)
				{
					// Create a zero constant if value is null
					auto loc = this->loc(*varDeclStmt);
					auto type = translateSolidityType(*varDeclStmt->initialValue()->annotation().type);
					auto zeroAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
					mlir::OperationState constState(loc, "solidity.constant");
					constState.addAttribute("value", zeroAttr);
					constState.addTypes(type);
					value = m_builder->create(constState)->getResult(0);
				}
				for (auto const& decl : declarations)
				{
					if (decl)
						m_valueMap[decl->id()] = value;
				}
				return value;
			}
		}
		
		// Most statements don't return a value
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
		for (auto const& param : _func.parameters())
		{
			inputTypes.push_back(m_builder->getI64Type()); // Simplified to i64
		}
		
		for (auto const& ret : _func.returnParameters())
		{
			resultTypes.push_back(m_builder->getI64Type()); // Simplified to i64  
		}
		
		auto funcType = m_builder->getFunctionType(inputTypes, resultTypes);
		
		// Create function using standard func dialect
		auto funcOp = m_builder->create<mlir::func::FuncOp>(
			loc, _func.name(), funcType
		);
		
		// Create entry block
		auto& entryBlock = funcOp.getBody().emplaceBlock();
		
		// Add block arguments
		for (auto inputType : inputTypes)
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
			for (auto resultType : resultTypes)
			{
				auto constant = m_builder->create<mlir::arith::ConstantOp>(
					loc, resultType, m_builder->getIntegerAttr(resultType, 42)
				);
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
		for (auto const& param : _func.parameters())
		{
			paramTypes.push_back(translateType(*param->type()));
		}
		
		std::vector<mlir::Type> returnTypes;
		for (auto const& ret : _func.returnParameters())
		{
			returnTypes.push_back(translateType(*ret->type()));
		}
		
		auto funcType = mlir::FunctionType::get(m_context.get(), paramTypes, returnTypes);
		
		// Determine visibility and mutability
		std::string visibility = "public";
		if (_func.visibility() == Visibility::Private)
			visibility = "private";
		else if (_func.visibility() == Visibility::Internal)
			visibility = "internal";
		else if (_func.visibility() == Visibility::External)
			visibility = "external";
		
		std::string mutability = "nonpayable";
		if (_func.stateMutability() == StateMutability::Pure)
			mutability = "pure";
		else if (_func.stateMutability() == StateMutability::View)
			mutability = "view";
		else if (_func.stateMutability() == StateMutability::Payable)
			mutability = "payable";
		
		// Create function operation
		// Use standard func dialect instead of custom solidity operations
		auto funcOp = m_builder->create<mlir::func::FuncOp>(
			loc, _func.name(), funcType
		);
		
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
				auto value = std::stoull(literal->value());
				auto attr = m_builder->getIntegerAttr(type, value);
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
				return m_builder->create<mlir::arith::DivUIOp>(loc, lhs, rhs).getResult(); // Using unsigned division as default
			case langutil::Token::Mod:
				return m_builder->create<mlir::arith::RemUIOp>(loc, lhs, rhs).getResult(); // Using unsigned remainder as default
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
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, lhs, rhs).getResult();
			case langutil::Token::GreaterThan:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs).getResult();
			case langutil::Token::LessThanOrEqual:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, lhs, rhs).getResult();
			case langutil::Token::GreaterThanOrEqual:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, lhs, rhs).getResult();
			case langutil::Token::Equal:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhs, rhs).getResult();
			case langutil::Token::NotEqual:
				return m_builder->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhs, rhs).getResult();
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
				auto one = m_builder->create<mlir::arith::ConstantOp>(
					loc, m_builder->getIntegerAttr(operand.getType(), 1)
				).getResult();
				return m_builder->create<mlir::arith::AddIOp>(loc, operand, one).getResult();
			}
			case langutil::Token::Dec:
			{
				auto one = m_builder->create<mlir::arith::ConstantOp>(
					loc, m_builder->getIntegerAttr(operand.getType(), 1)
				).getResult();
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
					for (auto const& arg : funcCall->arguments())
					{
						if (arg)
							args.push_back(generateExpression(*arg));
					}
					
					// Determine result types
					std::vector<mlir::Type> resultTypes;
					for (auto const& ret : funcDef->returnParameters())
					{
						resultTypes.push_back(translateType(*ret->type()));
					}
					
					// TODO: Implement function calls using standard MLIR operations
					// For now, return first argument as placeholder
					if (!args.empty())
						return args[0];
					else
					{
						auto zeroAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 0);
						return m_builder->create<mlir::arith::ConstantOp>(loc, zeroAttr).getResult();
					}
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
				// TODO: Implement array length with standard MLIR
				// For now, return constant placeholder
				auto lengthAttr = m_builder->getIntegerAttr(m_builder->getI64Type(), 10);
				return m_builder->create<mlir::arith::ConstantOp>(loc, lengthAttr).getResult();
			}
			else
			{
				// TODO: Implement member access with standard MLIR
				// For now, return the object itself as placeholder
				return object;
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
				// TODO: Implement array access with standard MLIR
				// For now, return the base as placeholder
				return base;
			}
			else if (dynamic_cast<MappingType const*>(indexAccess->baseExpression().annotation().type))
			{
				// TODO: Implement mapping access with standard MLIR
				// For now, return the base as placeholder
				return base;
			}
		}

		// Return dummy value for unhandled cases
		// Create a default value with the correct type to avoid type mismatches
		auto dummyType = translateType(*_expr.annotation().type);
		mlir::OperationState constState(loc, "solidity.constant");
		constState.addAttribute("value", m_builder->getIntegerAttr(m_builder->getI64Type(), 0));
		constState.addTypes(dummyType);
		return m_builder->create(constState)->getResult(0);
	}

	void generateStatement(Statement const& _stmt)
	{
		auto loc = this->loc(_stmt);
		
		if (auto* block = dynamic_cast<Block const*>(&_stmt))
		{
			for (auto const& stmt : block->statements())
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
				for (auto const& decl : varDeclStmt->declarations())
				{
					if (decl)
						m_valueMap[decl->id()] = value;
				}
			}
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
	
	std::shared_ptr<yul::Object> lowerToYul(std::string const& _mlirModule, bool _printIntermediateMLIR = false, std::string const& _mlirFile = "", bool _runAnalysis = false)
	{
		// Use the MLIRToYulLowering class to perform the conversion
		MLIRToYulLowering lowering;

		// First optimize the MLIR module
		std::string optimizedModule = lowering.optimize(_mlirModule, _printIntermediateMLIR, _mlirFile, _runAnalysis);
		
		// Then lower to Yul
		auto yulObject = lowering.lower(optimizedModule);
		
		// Extract contract name from MLIR if needed
		size_t contractPos = _mlirModule.find("\"solidity.contract\"");
		if (contractPos != std::string::npos && yulObject)
		{
			// Find the name attribute after the contract operation
			size_t namePos = _mlirModule.find("{name = \"", contractPos);
			if (namePos != std::string::npos)
			{
				namePos += 9; // Length of "{name = \""
				size_t endPos = _mlirModule.find("\"", namePos);
				if (endPos != std::string::npos)
				{
					std::string contractName = _mlirModule.substr(namePos, endPos - namePos);
					// Update object name if it's still "Contract"
					if (yulObject->name == "Contract")
						yulObject->name = contractName;
				}
			}
		}
		
		return yulObject;
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
	
	// Track loop-carried variables for break/continue statements
	std::set<int64_t> m_loopCarriedVarIds;
	
	// Track the immediate loop context for break/continue handling
	enum class LoopContext { None, SCF, Solidity };
	LoopContext m_immediateLoopContext = LoopContext::None;
#endif
};

MLIRGenerator::MLIRGenerator(
	CompilerStack const& _compilerStack,
	langutil::EVMVersion _evmVersion,
	OptimiserSettings const& _optimiserSettings
):
	m_impl(std::make_unique<MLIRGeneratorImpl>(_compilerStack, _evmVersion, _optimiserSettings)),
	m_compilerStack(_compilerStack),
	m_evmVersion(_evmVersion),
	m_optimiserSettings(_optimiserSettings)
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

std::shared_ptr<yul::Object> MLIRGenerator::lowerToYul(std::string const& _mlirModule, bool _printIntermediateMLIR, std::string const& _mlirFile)
{
	return m_impl->lowerToYul(_mlirModule, _printIntermediateMLIR, _mlirFile, m_optimiserSettings.mlirAnalyze);
}

// ASTVisitor implementations
bool MLIRGenerator::visit(ContractDefinition const& /*_contract*/)
{
	return true;
}

void MLIRGenerator::endVisit(ContractDefinition const& /*_contract*/)
{
}

bool MLIRGenerator::visit(FunctionDefinition const& /*_function*/)
{
	return true;
}

void MLIRGenerator::endVisit(FunctionDefinition const& /*_function*/)
{
}

bool MLIRGenerator::visit(VariableDeclaration const& /*_variable*/)
{
	return true;
}

void MLIRGenerator::endVisit(VariableDeclaration const& /*_variable*/)
{
}

bool MLIRGenerator::visit(Block const& /*_block*/)
{
	return true;
}

void MLIRGenerator::endVisit(Block const& /*_block*/)
{
}

bool MLIRGenerator::visit(IfStatement const& /*_ifStatement*/)
{
	return true;
}

void MLIRGenerator::endVisit(IfStatement const& /*_ifStatement*/)
{
}

bool MLIRGenerator::visit(WhileStatement const& /*_whileStatement*/)
{
	return true;
}

void MLIRGenerator::endVisit(WhileStatement const& /*_whileStatement*/)
{
}

bool MLIRGenerator::visit(ForStatement const& /*_forStatement*/)
{
	return true;
}

void MLIRGenerator::endVisit(ForStatement const& /*_forStatement*/)
{
}

bool MLIRGenerator::visit(Return const& /*_return*/)
{
	return true;
}

void MLIRGenerator::endVisit(Return const& /*_return*/)
{
}

bool MLIRGenerator::visit(Assignment const& /*_assignment*/)
{
	return true;
}

void MLIRGenerator::endVisit(Assignment const& /*_assignment*/)
{
}

bool MLIRGenerator::visit(BinaryOperation const& /*_operation*/)
{
	return true;
}

void MLIRGenerator::endVisit(BinaryOperation const& /*_operation*/)
{
}

bool MLIRGenerator::visit(UnaryOperation const& /*_operation*/)
{
	return true;
}

void MLIRGenerator::endVisit(UnaryOperation const& /*_operation*/)
{
}

bool MLIRGenerator::visit(FunctionCall const& /*_functionCall*/)
{
	return true;
}

void MLIRGenerator::endVisit(FunctionCall const& /*_functionCall*/)
{
}

bool MLIRGenerator::visit(Identifier const& /*_identifier*/)
{
	return true;
}

void MLIRGenerator::endVisit(Identifier const& /*_identifier*/)
{
}

bool MLIRGenerator::visit(Literal const& /*_literal*/)
{
	return true;
}

void MLIRGenerator::endVisit(Literal const& /*_literal*/)
{
}

} // namespace solidity::frontend
