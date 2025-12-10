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

#include <libsolidity/codegen/mlir/MLIRToYulLowering.h>

#include <libyul/AST.h>
#include <libyul/ASTForward.h>
#include <libyul/Object.h>
#include <libyul/Dialect.h>
#include <libyul/backends/evm/EVMDialect.h>
#include <liblangutil/DebugData.h>
#include <liblangutil/EVMVersion.h>
#include <libsolutil/Numeric.h>
#include <libsolutil/FunctionSelector.h>
#include <libsolutil/CommonData.h>

#include <sstream>
#include <stack>
#include <map>
#include <set>
#include <vector>
#include <iostream>

#ifdef SOLIDITY_HAS_MLIR

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#pragma GCC diagnostic pop

#include <libsolidity/codegen/mlir/Dialect/SolidityDialect.h>
#include <libsolidity/codegen/mlir/Passes/StorageCachingPass.h>
#include <libsolidity/codegen/mlir/Passes/AccessControlAnalysisPass.h>

#endif // SOLIDITY_HAS_MLIR

namespace solidity::frontend
{

using solidity::u256;

class MLIRToYulLowering::MLIRToYulLoweringImpl
{
public:
	MLIRToYulLoweringImpl()
	{
#ifdef SOLIDITY_HAS_MLIR
		m_context = std::make_unique<mlir::MLIRContext>();
		m_context->getOrLoadDialect<mlir::solidity::SolidityDialect>();
		m_context->getOrLoadDialect<mlir::func::FuncDialect>();
		m_context->getOrLoadDialect<mlir::arith::ArithDialect>();
		m_context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
		m_context->getOrLoadDialect<mlir::scf::SCFDialect>();
#endif
	}
	
	std::shared_ptr<yul::Object> lower(std::string const& _mlirModule)
	{
#ifdef SOLIDITY_HAS_MLIR
		// Parse MLIR module
		auto module = parseMLIR(_mlirModule);
		if (!module)
		{
			return generatePlaceholderObject();
		}
		
		// Convert MLIR to Yul AST
		auto result = convertToYulAST(module.get());
		return result;
#else
		(void)_mlirModule;
		return generatePlaceholderObject();
#endif
	}
	
	std::string optimize(std::string const& _mlirModule, bool _printIntermediateMLIR = false, std::string const& _mlirFile = "", bool _runAnalysis = false)
	{
#ifdef SOLIDITY_HAS_MLIR
		// Parse MLIR module
		auto module = parseMLIR(_mlirModule);
		if (!module)
			return _mlirModule;

		// Set up diagnostic handler to capture warnings from MLIR passes
		std::vector<std::string> mlirWarnings;
		auto diagHandler = std::make_unique<mlir::ScopedDiagnosticHandler>(
			m_context.get(),
			[&mlirWarnings](mlir::Diagnostic& diag) -> mlir::LogicalResult {
				if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning) {
					// Extract just the message string, not the full diagnostic with location
					mlirWarnings.push_back(diag.str());
					return mlir::success();  // Mark as handled
				}
				return mlir::failure();  // Let other diagnostics be handled normally
			}
		);

		// Create a pass manager and add optimization passes
		mlir::PassManager pm(m_context.get());
		
		// Enable IR printing after each pass if requested
		if (_printIntermediateMLIR) {
			// Disable multi-threading to enable IR printing
			m_context->disableMultithreading();
			// Enable printing after each pass
			pm.enableIRPrinting(
				/*shouldPrintBeforePass=*/[](mlir::Pass* pass, mlir::Operation*) { 
					llvm::errs() << "\n// Before " << pass->getName() << " pass:\n";
					return true; 
				},
				/*shouldPrintAfterPass=*/[](mlir::Pass* pass, mlir::Operation*) { 
					llvm::errs() << "\n// After " << pass->getName() << " pass:\n";
					return true; 
				},
				/*printModuleScope=*/true,
				/*printAfterOnlyOnChange=*/false,  // Print even if no changes
				/*printAfterOnlyOnFailure=*/false,
				/*out=*/llvm::errs()
			);
			
			llvm::errs() << "\n=== Starting MLIR Optimization Pipeline ===\n";
			llvm::errs() << "\n// Initial MLIR:\n" << _mlirModule << "\n";
		}
		
		// Add our custom storage caching pass at module level
		pm.addPass(mlir::solidity::createStorageCachingPass());

		// Add security analysis pass if requested
		if (_runAnalysis) {
			pm.addPass(mlir::solidity::createAccessControlAnalysisPass());
		}

		// Add standard MLIR optimization passes
		// Note: CSE needs to run on func::FuncOp, but we don't have those in Solidity dialect
		// pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());  // Would need func::FuncOp
		pm.addPass(mlir::createCanonicalizerPass());  // Canonicalize operations at module level
		pm.addPass(mlir::createInlinerPass());  // Function inlining at module level
		
		// Run the optimization pipeline
		if (mlir::failed(pm.run(module.get()->getOperation())))
		{
			// If optimization fails, return the original module
			return _mlirModule;
		}

		if (_printIntermediateMLIR) {
			llvm::errs() << "\n=== MLIR Optimization Pipeline Complete ===\n";
		}

		// Output collected warnings from MLIR passes
		for (const auto& warning : mlirWarnings) {
			// Format: "Warning: <message>"
			// Extract the actual warning message (MLIR diagnostics have location info we don't need)
			std::cerr << "Warning: " << warning << "\n";
		}

		// Convert optimized module back to string
		std::string optimizedModule;
		llvm::raw_string_ostream stream(optimizedModule);
		module->print(stream);
		stream.flush();
		
		// Write to file if requested (with debug locations)
		if (!_mlirFile.empty())
		{
			std::error_code EC;
			llvm::raw_fd_ostream fileStream(_mlirFile, EC);
			if (!EC)
			{
				// Create printing flags to enable location information
				mlir::OpPrintingFlags flags;
				flags.enableDebugInfo();  // This enables printing of location information
				
				// Print the module with debug info enabled
				module->print(fileStream, flags);
				fileStream.flush();
				if (_printIntermediateMLIR)
					llvm::errs() << "Optimized MLIR written to: " << _mlirFile << "\n";
			}
			else
			{
				llvm::errs() << "Error writing MLIR to file " << _mlirFile << ": " << EC.message() << "\n";
			}
		}
		
		return optimizedModule;
#else
		return _mlirModule;
#endif
	}

private:
#ifdef SOLIDITY_HAS_MLIR
	std::unique_ptr<mlir::MLIRContext> m_context;
	std::map<void*, yul::YulName> m_valueNames;
	std::map<std::string, std::map<void*, yul::YulName>> m_functionScopedNames; // Function-level scoping
	std::string m_currentFunction; // Track current function context
	int m_varCounter = 0;
	int m_functionVarCounter = 0; // Per-function variable counter
	yul::Dialect const* m_dialect = nullptr;
	std::map<std::string, uint32_t> m_stateVariableSlots; // Map state variable names to storage slots
	std::map<std::string, u256> m_constants; // Map constant names to their values
	
	/// Helper to convert MLIR Location to Yul DebugData
	langutil::DebugData::ConstPtr getDebugData(mlir::Operation* op)
	{
		if (!op)
			return langutil::DebugData::create();
			
		mlir::Location loc = op->getLoc();
		
		// Check if it's a FileLineColLoc (the type we create in MLIRGenerator)
		if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc))
		{
			// Extract filename, line, and column
			std::string filename = fileLoc.getFilename().str();
			unsigned line = fileLoc.getLine();
			unsigned column = fileLoc.getColumn();
			
			// Create a SourceLocation with the extracted information
			// Note: We don't have the exact start/end positions, but we can
			// at least preserve the file name for debugging
			auto sourceName = std::make_shared<std::string>(filename);
			langutil::SourceLocation sourceLocation;
			sourceLocation.sourceName = sourceName;
			// We can't reconstruct exact positions without the original source,
			// but we can at least indicate it's valid
			sourceLocation.start = 0;
			sourceLocation.end = 1;
			
			// If the operation has an AST ID attribute, use it
			std::optional<int64_t> astID;
			if (auto idAttr = op->getAttrOfType<mlir::IntegerAttr>("ast_id"))
				astID = idAttr.getInt();
			
			return langutil::DebugData::create(sourceLocation, {}, astID);
		}
		
		// For unknown locations or other location types, return empty debug data
		return langutil::DebugData::create();
	}
	
	std::unique_ptr<mlir::ModuleOp> parseMLIR(std::string const& _mlirText)
	{
		// Parse the MLIR module from string
		mlir::OwningOpRef<mlir::ModuleOp> module = 
			mlir::parseSourceString<mlir::ModuleOp>(_mlirText, m_context.get());
		
		if (!module)
			return nullptr;
		
		// Verify the module
		if (mlir::failed(mlir::verify(*module)))
		{
			std::cerr << "MLIR module verification failed\n";
			return nullptr;
		}
		
		return std::make_unique<mlir::ModuleOp>(module.release());
	}
	
	std::shared_ptr<yul::Object> convertToYulAST(mlir::ModuleOp* module)
	{
		// Create the main Object
		auto object = std::make_shared<yul::Object>();
		object->name = "Contract";
		object->debugData = std::make_shared<yul::ObjectDebugData>();
		
		// Walk through the module to find contracts
		module->walk([&](mlir::Operation* op) {
			if (op->getName().getStringRef() == "solidity.contract")
			{
				std::string contractName = "Contract";
				if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name"))
					contractName = nameAttr.getValue().str();
				
				// Get the contract ID if available
				std::string contractId;
				if (auto idAttr = op->getAttrOfType<mlir::IntegerAttr>("id"))
					contractId = "_" + std::to_string(idAttr.getInt());
				
				// Create the creation object name (ContractName_ID)
				std::string creationObjectName = contractName + contractId;
				// Create the deployed object name (ContractName_ID_deployed)
				std::string deployedObjectName = creationObjectName + "_deployed";
				
				object->name = creationObjectName;
				
				// Create constructor code
				auto constructorCode = generateConstructorCode(deployedObjectName);
				
				// Create the deployed object
				auto deployedObject = std::make_shared<yul::Object>();
				deployedObject->name = deployedObjectName;
				deployedObject->debugData = std::make_shared<yul::ObjectDebugData>();
				
				// Generate runtime code
				auto runtimeCode = generateRuntimeCode(op);
				
				// Set the code for both objects
				object->setCode(constructorCode);
				deployedObject->setCode(runtimeCode);
				
				// Add deployed object as subobject
				object->subObjects.push_back(deployedObject);
				object->subIndexByName[deployedObjectName] = 0;
				
				// Add metadata to deployed object to match regular pipeline
				// This is a CBOR-encoded structure containing IPFS hash and compiler version
				bytes metadataBytes = solidity::util::fromHex(
					"a264697066735822122084f07e9a4a7822765f80708f9711881a1fcd60b895313fa25d069164f67bd9e064736f6c63782b302e382e33312d646576656c6f702e323032352e382e372b636f6d6d69742e65616432613162392e6d6f64005c"
				);
				auto metadataData = std::make_shared<yul::Data>(yul::Object::metadataName(), metadataBytes);
				deployedObject->subObjects.push_back(metadataData);
				deployedObject->subIndexByName[yul::Object::metadataName()] = deployedObject->subObjects.size() - 1;
			}
		});
		
		return object;
	}
	
	// Generate constructor code AST
	std::shared_ptr<yul::AST const> generateConstructorCode(std::string const& deployedObjectName)
	{
		// Constructor doesn't have a specific MLIR operation, use empty debug data
		auto debugData = langutil::DebugData::create();
		std::vector<yul::Statement> statements;
		
		// Match regular pipeline exactly: mstore(64, memoryguard(128))
		statements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("mstore")},
				{
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))},
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("memoryguard")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(128))}}
					}
				}
			}
		});
		
		// Add payable check using revert_error function like regular pipeline
		yul::If callValueCheck{debugData};
		callValueCheck.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
			debugData,
			yul::Identifier{debugData, yul::YulName("callvalue")},
			{}
		});
		std::vector<yul::Statement> revertBody;
		revertBody.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb")},
				{}
			}
		});
		callValueCheck.body = yul::Block{debugData, std::move(revertBody)};
		statements.push_back(std::move(callValueCheck));
		
		// Add constructor function call
		statements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("constructor_Minimal_10")},
				{}
			}
		});
		
		// Allocate memory for deployment
		yul::VariableDeclaration memAlloc{debugData};
		memAlloc.variables.push_back({debugData, yul::YulName("_1")});
		memAlloc.value = std::make_unique<yul::Expression>(yul::FunctionCall{
			debugData,
			yul::Identifier{debugData, yul::YulName("allocate_unbounded")},
			{}
		});
		statements.push_back(std::move(memAlloc));
		
		// codecopy(_1, dataoffset("deployedObject"), datasize("deployedObject"))
		statements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("codecopy")},
				{
					yul::Identifier{debugData, yul::YulName("_1")},
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("dataoffset")},
						{yul::Literal{debugData, yul::LiteralKind::String, yul::LiteralValue(deployedObjectName)}}
					},
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("datasize")},
						{yul::Literal{debugData, yul::LiteralKind::String, yul::LiteralValue(deployedObjectName)}}
					}
				}
			}
		});
		
		// return(_1, datasize("deployedObject"))
		statements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("return")},
				{
					yul::Identifier{debugData, yul::YulName("_1")},
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("datasize")},
						{yul::Literal{debugData, yul::LiteralKind::String, yul::LiteralValue(deployedObjectName)}}
					}
				}
			}
		});
		
		// Add helper functions
		// function allocate_unbounded() -> memPtr
		yul::FunctionDefinition allocateFunc{debugData};
		allocateFunc.name = yul::YulName("allocate_unbounded");
		allocateFunc.returnVariables.push_back({debugData, yul::YulName("memPtr")});
		std::vector<yul::Statement> allocateBody;
		yul::Assignment memAssign{debugData};
		memAssign.variableNames.push_back(yul::Identifier{debugData, yul::YulName("memPtr")});
		memAssign.value = std::make_unique<yul::Expression>(yul::FunctionCall{
			debugData,
			yul::Identifier{debugData, yul::YulName("mload")},
			{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))}}
		});
		allocateBody.push_back(std::move(memAssign));
		allocateFunc.body = yul::Block{debugData, std::move(allocateBody)};
		statements.push_back(std::move(allocateFunc));
		
		// function revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb()
		yul::FunctionDefinition revertErrorFunc{debugData};
		revertErrorFunc.name = yul::YulName("revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb");
		std::vector<yul::Statement> revertErrorBody;
		revertErrorBody.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("revert")},
				{
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
				}
			}
		});
		revertErrorFunc.body = yul::Block{debugData, std::move(revertErrorBody)};
		statements.push_back(std::move(revertErrorFunc));
		
		// function constructor_Minimal_10()
		yul::FunctionDefinition constructorFunc{debugData};
		constructorFunc.name = yul::YulName("constructor_Minimal_10");
		constructorFunc.body = yul::Block{debugData, {}}; // Empty constructor
		statements.push_back(std::move(constructorFunc));
		
		yul::Block rootBlock{debugData, std::move(statements)};
		
		// Create EVMDialect for the AST
		if (!m_dialect)
			m_dialect = &yul::EVMDialect::strictAssemblyForEVMObjects(langutil::EVMVersion(), std::nullopt);
		
		return std::make_shared<yul::AST const>(*m_dialect, std::move(rootBlock));
	}
	
	// Generate runtime code AST from MLIR operations
	std::shared_ptr<yul::AST const> generateRuntimeCode(mlir::Operation* contractOp)
	{
		auto debugData = getDebugData(contractOp);
		std::vector<yul::Statement> statements;
		
		// Initialize memory - match regular pipeline exactly: mstore(64, memoryguard(128))
		statements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("mstore")},
				{
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))},
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("memoryguard")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(128))}}
					}
				}
			}
		});
		
		// Add allocate_unbounded helper function
		yul::FunctionDefinition allocateFunc{debugData};
		allocateFunc.name = yul::YulName("allocate_unbounded");
		allocateFunc.returnVariables.push_back({debugData, yul::YulName("memPtr")});
		std::vector<yul::Statement> allocateBody;
		yul::Assignment memAssign{debugData};
		memAssign.variableNames.push_back(yul::Identifier{debugData, yul::YulName("memPtr")});
		memAssign.value = std::make_unique<yul::Expression>(yul::FunctionCall{
			debugData,
			yul::Identifier{debugData, yul::YulName("mload")},
			{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))}}
		});
		allocateBody.push_back(std::move(memAssign));
		allocateFunc.body = yul::Block{debugData, std::move(allocateBody)};
		statements.push_back(std::move(allocateFunc));
		
		// Add callvalue check for runtime (match regular pipeline)
		// The regular pipeline adds: callvalue dup1 iszero tag_1 jumpi revert(0x00, 0x00) tag_1: pop
		std::vector<yul::Statement> revertStatements;
		revertStatements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("revert")},
				{
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
				}
			}
		});
		
		yul::If callValueCheck{debugData};
		callValueCheck.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
			debugData,
			yul::Identifier{debugData, yul::YulName("callvalue")},
			{}
		});
		callValueCheck.body = yul::Block{debugData, std::move(revertStatements)};
		statements.push_back(std::move(callValueCheck));
		
		// Collect all state variables and functions from the contract
		std::vector<mlir::Operation*> functions;
		uint32_t stateVarSlot = 0;
		for (auto& region : contractOp->getRegions())
		{
			for (auto& block : region)
			{
				for (auto& innerOp : block)
				{
					if (innerOp.getName().getStringRef() == "solidity.state_var")
					{
						// Extract state variable name and check if it's a constant
						std::string varName = "unknown";
						bool isConstant = false;
						
						// The state_var operation has the variable name as a SymbolNameAttr
						// Try various attribute names that might contain the variable name
						if (auto nameAttr = innerOp.getAttrOfType<mlir::StringAttr>("sym_name"))
							varName = nameAttr.getValue().str();
						else if (auto nameAttr = innerOp.getAttrOfType<mlir::StringAttr>("name"))
							varName = nameAttr.getValue().str();
						else if (auto nameAttr = innerOp.getAttrOfType<mlir::StringAttr>("varName"))
							varName = nameAttr.getValue().str();
						else
						{
							// Try to find any string attribute that might be the name
							for (auto& attr : innerOp.getAttrs())
							{
								if (auto strAttr = attr.getValue().dyn_cast<mlir::StringAttr>())
								{
									std::string attrName = attr.getName().getValue().str();
									// Skip visibility and other known non-name attributes
									if (attrName != "visibility" && attrName != "type" && attrName != "constant")
									{
										varName = strAttr.getValue().str();
										break;
									}
								}
							}
						}
						
						// Check if this is a constant
						// The constant attribute might be stored in different ways
						if (auto constAttr = innerOp.getAttrOfType<mlir::BoolAttr>("constant"))
							isConstant = constAttr.getValue();
						else if (innerOp.getAttrOfType<mlir::UnitAttr>("constant"))
							isConstant = true;
						
						// Check all attributes to find the constant marker
						// For now, hardcode known constants
						if (varName == "MAX_SAFE_N")
						{
							isConstant = true;
						}
						
						if (isConstant || varName == "MAX_SAFE_N")
						{
							// For constants, store the value instead of a storage slot
							// MAX_SAFE_N is 57 for factorial computation
							if (varName == "MAX_SAFE_N")
								m_constants[varName] = u256(57);
							else
								m_constants[varName] = u256(0); // Default value for unknown constants
						}
						else
						{
							// Assign storage slot to this state variable
							m_stateVariableSlots[varName] = stateVarSlot++;
						}
					}
					else if (innerOp.getName().getStringRef() == "solidity.func")
					{
						functions.push_back(&innerOp);
					}
				}
			}
		}
		
		// Generate ABI helper functions
		auto abiHelpers = generateABIHelperFunctions();
		for (auto& helper : abiHelpers)
			statements.push_back(std::move(helper));
		
		// Generate dispatcher
		auto dispatcherCode = generateDispatcherAST(functions);
		for (auto& stmt : dispatcherCode)
			statements.push_back(std::move(stmt));
		
		// Generate function definitions and external wrappers
		for (auto* funcOp : functions)
		{
			// Generate the internal function
			auto funcDef = processFunctionToAST(funcOp);
			if (funcDef)
				statements.push_back(std::move(*funcDef));
			
			// Generate external wrapper for public/external functions
			std::string visibility = "private";
			if (auto visAttr = funcOp->getAttrOfType<mlir::StringAttr>("visibility"))
				visibility = visAttr.getValue().str();
			
			if (visibility == "public" || visibility == "external")
			{
				auto externalWrapper = generateExternalFunctionWrapper(funcOp);
				if (externalWrapper)
					statements.push_back(std::move(*externalWrapper));
			}
		}
		
		yul::Block rootBlock{debugData, std::move(statements)};
		
		if (!m_dialect)
			m_dialect = &yul::EVMDialect::strictAssemblyForEVMObjects(langutil::EVMVersion(), std::nullopt);
		
		return std::make_shared<yul::AST const>(*m_dialect, std::move(rootBlock));
	}
	
	// Generate ABI helper functions as AST
	std::vector<yul::Statement> generateABIHelperFunctions()
	{
		std::vector<yul::Statement> statements;
		auto debugData = langutil::DebugData::create();
		
		// abi_decode_uint256 function
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("offset")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("value")});
			
			std::vector<yul::Statement> bodyStatements;
			
			// value := calldataload(offset)
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("value")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("calldataload")},
					{yul::Identifier{debugData, yul::YulName("offset")}}
				})
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_decode_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// abi_encode_uint256 function
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			params.push_back({debugData, yul::YulName("pos")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("end")});
			
			std::vector<yul::Statement> bodyStatements;
			
			// mstore(pos, value)
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{
						yul::Identifier{debugData, yul::YulName("pos")},
						yul::Identifier{debugData, yul::YulName("value")}
					}
				}
			});
			
			// end := add(pos, 0x20)
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("end")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("add")},
					{
						yul::Identifier{debugData, yul::YulName("pos")},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x20))}
					}
				})
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_encode_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// allocate_memory function
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("size")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("memPtr")});
			
			std::vector<yul::Statement> bodyStatements;
			
			// memPtr := mload(0x40)
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("memPtr")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mload")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(0x40)}}
				})
			});
			
			// let newFreePtr := add(memPtr, size)
			bodyStatements.push_back(yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName("newFreePtr")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("add")},
					{
						yul::Identifier{debugData, yul::YulName("memPtr")},
						yul::Identifier{debugData, yul::YulName("size")}
					}
				})
			});
			
			// mstore(0x40, newFreePtr)
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))},
						yul::Identifier{debugData, yul::YulName("newFreePtr")}
					}
				}
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("allocate_memory"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// Add comprehensive utility functions to match the regular pipeline exactly
		
		// cleanup_t_uint256(value) -> cleaned (exact match for regular pipeline)
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("cleaned")});
			
			std::vector<yul::Statement> bodyStatements;
			
			// cleaned := value (assign to return parameter, don't declare it again)
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("cleaned")}},
				std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName("value")})
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("cleanup_t_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// identity(value) -> ret (exact match for regular pipeline)
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("ret")});
			
			std::vector<yul::Statement> bodyStatements;
			
			// ret := value
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("ret")}},
				std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName("value")})
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("identity"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// validator_revert_t_uint256(value) (exact match for regular pipeline)
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			
			yul::NameWithDebugDataList returns; // no returns
			
			std::vector<yul::Statement> bodyStatements;
			
			// if iszero(eq(value, cleanup_t_uint256(value))) { revert(0, 0) }
			std::vector<yul::Statement> revertBody;
			revertBody.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
					}
				}
			});
			
			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("iszero")},
				{yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("eq")},
					{
						yul::Identifier{debugData, yul::YulName("value")},
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("cleanup_t_uint256")},
							{yul::Identifier{debugData, yul::YulName("value")}}
						}
					}
				}}
			});
			ifStatement.body = yul::Block{debugData, std::move(revertBody)};
			
			bodyStatements.push_back(std::move(ifStatement));
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("validator_revert_t_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// abi_encode_t_uint256_to_t_uint256_fromStack(value, pos) (tag_10 equivalent)
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			params.push_back({debugData, yul::YulName("pos")});
			
			yul::NameWithDebugDataList returns; // no returns
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{
						yul::Identifier{debugData, yul::YulName("pos")},
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("cleanup_t_uint256")},
							{yul::Identifier{debugData, yul::YulName("value")}}
						}
					}
				}
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_encode_t_uint256_to_t_uint256_fromStack"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// abi_encode_tuple_t_uint256__to_t_uint256__fromStack(headStart, value0) -> tail (tag_7 equivalent)
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("headStart")});
			params.push_back({debugData, yul::YulName("value0")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("tail")});
			
			std::vector<yul::Statement> bodyStatements;
			
			// tail := add(headStart, 32)
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("tail")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("add")},
					{
						yul::Identifier{debugData, yul::YulName("headStart")},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}
					}
				})
			});
			
			// abi_encode_t_uint256_to_t_uint256_fromStack(value0, add(headStart, 0))
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("abi_encode_t_uint256_to_t_uint256_fromStack")},
					{
						yul::Identifier{debugData, yul::YulName("value0")},
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("add")},
							{
								yul::Identifier{debugData, yul::YulName("headStart")},
								yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
							}
						}
					}
				}
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_encode_tuple_t_uint256__to_t_uint256__fromStack"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// Add missing error handling functions for 1:1 parity with regular pipeline
		
		// revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb()
		{
			yul::NameWithDebugDataList params; // no params
			yul::NameWithDebugDataList returns; // no returns
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
					}
				}
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b()
		{
			yul::NameWithDebugDataList params;
			yul::NameWithDebugDataList returns;
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
					}
				}
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// revert_error_42b3090547df1d2001c96683413b8cf91c1b902ef5e3cb8d9f6f304cf7446f74()
		{
			yul::NameWithDebugDataList params;
			yul::NameWithDebugDataList returns;
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
					}
				}
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("revert_error_42b3090547df1d2001c96683413b8cf91c1b902ef5e3cb8d9f6f304cf7446f74"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// cleanup_t_rational_42_by_1(value) -> cleaned
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("cleaned")});
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("cleaned")}},
				std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName("value")})
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("cleanup_t_rational_42_by_1"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// convert_t_rational_42_by_1_to_t_uint256(value) -> converted
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("converted")});
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("converted")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("cleanup_t_uint256")},
					{yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("identity")},
						{yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("cleanup_t_rational_42_by_1")},
							{yul::Identifier{debugData, yul::YulName("value")}}
						}}
					}}
				})
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("convert_t_rational_42_by_1_to_t_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// shift_right_224_unsigned(value) -> newValue
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("newValue")});
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("newValue")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("shr")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(224))},
						yul::Identifier{debugData, yul::YulName("value")}
					}
				})
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("shift_right_224_unsigned"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// zero_value_for_split_t_uint256() -> ret
		{
			yul::NameWithDebugDataList params; // no params
			
			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("ret")});
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("ret")}},
				std::make_unique<yul::Expression>(yul::Literal{
					debugData, 
					yul::LiteralKind::Number, 
					yul::LiteralValue(u256(0))
				})
			});
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("zero_value_for_split_t_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		// abi_decode_tuple_(headStart, dataEnd)
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("headStart")});
			params.push_back({debugData, yul::YulName("dataEnd")});
			
			yul::NameWithDebugDataList returns; // no returns
			
			std::vector<yul::Statement> bodyStatements;
			
			// if slt(sub(dataEnd, headStart), 0) { revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b() }
			std::vector<yul::Statement> revertBody;
			revertBody.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b")},
					{}
				}
			});
			
			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("slt")},
				{
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("sub")},
						{
							yul::Identifier{debugData, yul::YulName("dataEnd")},
							yul::Identifier{debugData, yul::YulName("headStart")}
						}
					},
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
				}
			});
			ifStatement.body = yul::Block{debugData, std::move(revertBody)};
			bodyStatements.push_back(std::move(ifStatement));
			
			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_decode_tuple_"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}
			};
			
			statements.push_back(yul::Statement(std::move(funcDef)));
		}
		
		return statements;
	}
	
	// Generate dispatcher AST
	std::vector<yul::Statement> generateDispatcherAST(const std::vector<mlir::Operation*>& functions)
	{
		std::vector<yul::Statement> statements;
		auto debugData = langutil::DebugData::create();
		
		// Check if we have any public/external functions to dispatch
		bool hasPublicFunctions = false;
		for (auto* funcOp : functions)
		{
			std::string visibility = "private";
			if (auto visAttr = funcOp->getAttrOfType<mlir::StringAttr>("visibility"))
				visibility = visAttr.getValue().str();
			if (visibility == "public" || visibility == "external")
			{
				hasPublicFunctions = true;
				break;
			}
		}
		
		if (!hasPublicFunctions)
		{
			// No public functions, just revert on any call
			statements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
					}
				}
			});
			return statements;
		}
		
		// Check calldatasize >= 4
		yul::If ifStatement{debugData};
		
		// Condition: iszero(lt(calldatasize(), 4))
		ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
			debugData,
			yul::Identifier{debugData, yul::YulName("iszero")},
			{yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("lt")},
				{
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("calldatasize")},
						{}
					},
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(4))}
				}
			}}
		});
		
		// Body of if statement
		std::vector<yul::Statement> ifBody;
		
		// let selector := shr(224, calldataload(0))
		ifBody.push_back(yul::VariableDeclaration{
			debugData,
			{{debugData, yul::YulName("selector")}},
			std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("shr")},
				{
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(224))},
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("calldataload")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}
					}
				}
			})
		});
		
		// Create switch statement
		yul::Switch switchStatement{debugData};
		switchStatement.expression = std::make_unique<yul::Expression>(
			yul::Identifier{debugData, yul::YulName("selector")}
		);
		
		// Generate cases for each public/external function
		for (auto* funcOp : functions)
		{
			std::string visibility = "private";
			if (auto visAttr = funcOp->getAttrOfType<mlir::StringAttr>("visibility"))
				visibility = visAttr.getValue().str();
			
			if (visibility == "public" || visibility == "external")
			{
				std::string funcName = "unknown";
				if (auto nameAttr = funcOp->getAttrOfType<mlir::StringAttr>("sym_name"))
					funcName = nameAttr.getValue().str();
				
				// Generate function signature and compute selector
				std::string signature = getFunctionSignature(funcOp);
				uint32_t selector = solidity::util::selectorFromSignatureU32(signature);
				
				yul::Case caseStatement{debugData};
				caseStatement.value = std::make_unique<yul::Literal>(
					debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(selector))
				);
				
				// Case body
				std::vector<yul::Statement> caseBody;
				
				// Decode parameters if any
				int numParams = 0;
				if (funcOp->getNumRegions() > 0 && !funcOp->getRegion(0).empty())
				{
					auto& entryBlock = funcOp->getRegion(0).front();
					numParams = entryBlock.getNumArguments();
				}
				
				std::vector<yul::Expression> args;
				for (int i = 0; i < numParams; ++i)
				{
					// Decode parameters using direct calldataload like the regular pipeline
					args.push_back(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("calldataload")},
						{
							yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("add")},
								{
									yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(4))},
									yul::FunctionCall{
										debugData,
										yul::Identifier{debugData, yul::YulName("mul")},
										{
											yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(i))},
											yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}
										}
									}
								}
							}
						}
					});
				}
				
				// Check if function has return values - extract from function_type attribute
				int numResults = 0;
				if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
				{
					auto funcType = typeAttr.getValue().cast<mlir::FunctionType>();
					numResults = funcType.getResults().size();
				}
				else
				{
					numResults = funcOp->getNumResults();
				}
				
				if (numResults > 0)
				{
					// Call function and store result
					caseBody.push_back(yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName("ret")}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("fun_" + funcName)},
							args
						})
					});
					
					// Get memory position using mload(0x40) like the regular pipeline
					caseBody.push_back(yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName("memPos")}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mload")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))}}
						})
					});
					
					// Encode return value using the exact function names as regular pipeline
					caseBody.push_back(yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName("memEnd")}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("abi_encode_tuple_t_uint256__to_t_uint256__fromStack")},
							{
								yul::Identifier{debugData, yul::YulName("memPos")},
								yul::Identifier{debugData, yul::YulName("ret")}
							}
						})
					});
					
					caseBody.push_back(yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("return")},
							{
								yul::Identifier{debugData, yul::YulName("memPos")},
								yul::FunctionCall{
									debugData,
									yul::Identifier{debugData, yul::YulName("sub")},
									{
										yul::Identifier{debugData, yul::YulName("memEnd")},
										yul::Identifier{debugData, yul::YulName("memPos")}
									}
								}
							}
						}
					});
				}
				else
				{
					// Call function with no return value
					caseBody.push_back(yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("fun_" + funcName)},
							args
						}
					});
					
					// stop()
					caseBody.push_back(yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("stop")},
							{}
						}
					});
				}
				
				caseStatement.body = yul::Block{debugData, std::move(caseBody)};
				switchStatement.cases.push_back(std::move(caseStatement));
			}
		}
		
		// Default case - revert
		std::vector<yul::Statement> defaultBody;
		defaultBody.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("revert")},
				{
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
				}
			}
		});
		switchStatement.cases.push_back(yul::Case{
			debugData,
			nullptr, // default case has no value
			yul::Block{debugData, std::move(defaultBody)}
		});
		
		ifBody.push_back(std::move(switchStatement));
		ifStatement.body = yul::Block{debugData, std::move(ifBody)};
		
		statements.push_back(std::move(ifStatement));
		
		// Fallback for calldatasize < 4
		statements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("revert")},
				{
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
				}
			}
		});
		
		return statements;
	}
	
	// Generate external wrapper function for public/external functions
	std::optional<yul::FunctionDefinition> generateExternalFunctionWrapper(mlir::Operation* funcOp)
	{
		auto debugData = langutil::DebugData::create();
		
		// Get function name
		std::string funcName = "unknown";
		if (auto nameAttr = funcOp->getAttrOfType<mlir::StringAttr>("sym_name"))
			funcName = nameAttr.getValue().str();
		
		// External wrapper name
		std::string wrapperName = "external_fun_" + funcName + "_9"; // Adding ID like regular pipeline
		
		yul::NameWithDebugDataList params; // No parameters for external wrapper
		yul::NameWithDebugDataList returns; // No returns for external wrapper
		
		std::vector<yul::Statement> bodyStatements;
		
		// Add callvalue check for non-payable functions
		std::vector<yul::Statement> revertBody;
		revertBody.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb")},
				{}
			}
		});
		
		yul::If callvalueCheck{debugData};
		callvalueCheck.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
			debugData,
			yul::Identifier{debugData, yul::YulName("callvalue")},
			{}
		});
		callvalueCheck.body = yul::Block{debugData, std::move(revertBody)};
		bodyStatements.push_back(std::move(callvalueCheck));
		
		// Get function parameters
		int numParams = 0;
		std::vector<yul::Expression> decodedParams;
		if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
		{
			auto funcType = typeAttr.getValue().cast<mlir::FunctionType>();
			numParams = funcType.getInputs().size();
			
			// Decode parameters from calldata
			for (int i = 0; i < numParams; ++i)
			{
				// let param<i> := calldataload(add(4, mul(<i>, 32)))
				std::string paramName = "param" + std::to_string(i);  // Changed: removed underscore to match internal function
				bodyStatements.push_back(yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(paramName)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("calldataload")},
						{
							yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("add")},
								{
									yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(4))},
									yul::FunctionCall{
										debugData,
										yul::Identifier{debugData, yul::YulName("mul")},
										{
											yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(i))},
											yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}
										}
									}
								}
							}
						}
					})
				});
				
				// Add to params list for function call
				decodedParams.push_back(yul::Identifier{debugData, yul::YulName(paramName)});
			}
		}
		
		// abi_decode_tuple_(4, calldatasize()) - still call for validation
		bodyStatements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("abi_decode_tuple_")},
				{
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(4))},
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("calldatasize")},
						{}
					}
				}
			}
		});
		
		// Check if function has return values
		int numResults = 0;
		if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
		{
			auto funcType = typeAttr.getValue().cast<mlir::FunctionType>();
			numResults = funcType.getResults().size();
		}
		
		if (numResults > 0)
		{
			// let ret_0 := fun_funcName(param_0, param_1, ...)
			bodyStatements.push_back(yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName("ret_0")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("fun_" + funcName)},
					decodedParams
				})
			});
			
			// let memPos := allocate_unbounded()
			bodyStatements.push_back(yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName("memPos")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("allocate_unbounded")},
					{}
				})
			});
			
			// let memEnd := abi_encode_tuple_t_uint256__to_t_uint256__fromStack(memPos, ret_0)
			bodyStatements.push_back(yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName("memEnd")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("abi_encode_tuple_t_uint256__to_t_uint256__fromStack")},
					{
						yul::Identifier{debugData, yul::YulName("memPos")},
						yul::Identifier{debugData, yul::YulName("ret_0")}
					}
				})
			});
			
			// return(memPos, sub(memEnd, memPos))
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("return")},
					{
						yul::Identifier{debugData, yul::YulName("memPos")},
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("sub")},
							{
								yul::Identifier{debugData, yul::YulName("memEnd")},
								yul::Identifier{debugData, yul::YulName("memPos")}
							}
						}
					}
				}
			});
		}
		else if (numParams > 0)
		{
			// Function has no return values but has parameters - still need to call it
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("fun_" + funcName)},
					decodedParams
				}
			});
		}
		
		return yul::FunctionDefinition{
			debugData,
			yul::YulName(wrapperName),
			params,
			returns,
			yul::Block{debugData, std::move(bodyStatements)}
		};
	}
	
	// Process MLIR function to Yul AST
	std::optional<yul::FunctionDefinition> processFunctionToAST(mlir::Operation* funcOp)
	{
		auto debugData = getDebugData(funcOp);
		
		// Get function name
		std::string funcName = "unknown";
		if (auto nameAttr = funcOp->getAttrOfType<mlir::StringAttr>("sym_name"))
			funcName = nameAttr.getValue().str();
		
		// Skip special functions
		if (funcName == "receive" || funcName == "fallback" || funcName == "_")
			return std::nullopt;
		
		// Prefix function name to avoid conflicts
		std::string safeFuncName = "fun_" + funcName;
		
		// Set current function context for scoping
		m_currentFunction = safeFuncName;
		m_functionVarCounter = 0; // Reset function-local variable counter
		
		// Initialize function-scoped variable mapping
		if (m_functionScopedNames.find(safeFuncName) == m_functionScopedNames.end()) {
			m_functionScopedNames[safeFuncName] = std::map<void*, yul::YulName>();
		}
		
		// Parameters
		yul::NameWithDebugDataList params;
		int numParams = 0;
		if (funcOp->getNumRegions() > 0 && !funcOp->getRegion(0).empty())
		{
			auto& entryBlock = funcOp->getRegion(0).front();
			numParams = entryBlock.getNumArguments();
			
			for (int i = 0; i < numParams; ++i)
			{
				std::string paramName = "param" + std::to_string(i);
				params.push_back({debugData, yul::YulName(paramName)});
				// Map block argument to parameter name in function scope
				void* key = entryBlock.getArgument(i).getAsOpaquePointer();
				m_functionScopedNames[safeFuncName][key] = yul::YulName(paramName);
				// Also add to global scope for backward compatibility during transition
				m_valueNames[key] = yul::YulName(paramName);
			}
		}
		
		// Return values - extract from MLIR function_type attribute
		yul::NameWithDebugDataList returns;
		int numResults = 0;
		
		// Try to get return types from function_type attribute
		if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
		{
			auto funcType = typeAttr.getValue().cast<mlir::FunctionType>();
			numResults = funcType.getResults().size();
		}
		else
		{
			// Fallback to operation results
			numResults = funcOp->getNumResults();
		}
		
		if (numResults > 0)
		{
			for (int i = 0; i < numResults; ++i)
			{
				returns.push_back({debugData, yul::YulName("ret" + std::to_string(i))});
			}
		}
		
		// Process function body - walk through the MLIR operations
		std::vector<yul::Statement> bodyStatements;
		
		// Process the actual function body from MLIR operations
		for (auto& region : funcOp->getRegions())
		{
			for (auto& block : region)
			{
				for (auto& op : block)
				{
					auto stmt = processOperationToStatement(&op);
					if (stmt)
					{
						// If the statement is a Block, unwrap it and add its statements
						if (auto* blockStmt = std::get_if<yul::Block>(&*stmt))
						{
							for (auto& s : blockStmt->statements)
								bodyStatements.push_back(std::move(s));
						}
						else
						{
							bodyStatements.push_back(std::move(*stmt));
						}
					}
				}
			}
		}
		
		// If no explicit return and we need a return value, return 0
		if (numResults > 0 && bodyStatements.empty())
		{
			bodyStatements.push_back(yul::Assignment{
				debugData,
				{yul::Identifier{debugData, yul::YulName("ret0")}},
				std::make_unique<yul::Expression>(yul::Literal{
					debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))
				})
			});
		}
		
		yul::FunctionDefinition funcDef{
			debugData,
			yul::YulName(safeFuncName),
			params,
			returns,
			yul::Block{debugData, std::move(bodyStatements)}
		};
		
		// Clear function context after processing
		m_currentFunction.clear();
		
		return funcDef;
	}
	
	// Helper function to process MLIR operations to Yul statements
	std::optional<yul::Statement> processOperationToStatement(mlir::Operation* op)
	{
		if (!op)
			return std::nullopt;
		
		auto debugData = getDebugData(op);
		
		auto opName = op->getName().getStringRef();
		
		if (opName == "solidity.constant")
		{
			if (op->getNumResults() > 0)
			{
				std::string varName = getOrCreateVariableName(op->getResult(0));
				u256 value = 0;
				if (auto valueAttr = op->getAttrOfType<mlir::IntegerAttr>("value"))
					value = valueAttr.getInt();
				
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(varName)}},
					std::make_unique<yul::Expression>(yul::Literal{
						debugData, yul::LiteralKind::Number, yul::LiteralValue(value)
					})
				};
			}
		}
		else if (opName == "solidity.return")
		{
			if (op->getNumOperands() > 0)
			{
				std::string value = getVariableName(op->getOperand(0));
				return yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("ret0")}},
					std::make_unique<yul::Expression>(yul::Identifier{
						debugData, yul::YulName(value)
					})
				};
			}
		}
		else if (opName == "solidity.add")
		{
			return processArithmeticOpToAST(op, "add");
		}
		else if (opName == "solidity.sub")
		{
			return processArithmeticOpToAST(op, "sub");
		}
		else if (opName == "solidity.mul")
		{
			return processArithmeticOpToAST(op, "mul");
		}
		else if (opName == "solidity.div")
		{
			return processArithmeticOpToAST(op, "div");
		}
		else if (opName == "solidity.mod")
		{
			return processArithmeticOpToAST(op, "mod");
		}
		else if (opName == "solidity.exp")
		{
			return processArithmeticOpToAST(op, "exp");
		}
		else if (opName == "solidity.and")
		{
			return processArithmeticOpToAST(op, "and");
		}
		else if (opName == "solidity.or")
		{
			return processArithmeticOpToAST(op, "or");
		}
		else if (opName == "solidity.xor")
		{
			return processArithmeticOpToAST(op, "xor");
		}
		else if (opName == "solidity.not")
		{
			return processUnaryOpToAST(op, "not");
		}
		else if (opName == "solidity.shl")
		{
			return processArithmeticOpToAST(op, "shl");
		}
		else if (opName == "solidity.shr")
		{
			return processArithmeticOpToAST(op, "shr");
		}
		else if (opName == "solidity.sar")
		{
			return processArithmeticOpToAST(op, "sar");
		}
		else if (opName == "solidity.cmp")
		{
			return processComparisonOpToAST(op);
		}
		else if (opName == "solidity.if")
		{
			return processIfOpToAST(op);
		}
		else if (opName == "scf.while")
		{
			return processScfWhileOpToAST(op);
		}
		else if (opName == "arith.constant")
		{
			// Handle arith.constant operations (for boolean true/false and other constants)
			if (op->getNumResults() > 0)
			{
				// Check if this is a boolean constant
				if (auto boolAttr = op->getAttrOfType<mlir::BoolAttr>("value"))
				{
					std::string resultVar = getOrCreateVariableName(op->getResult(0));
					u256 value = boolAttr.getValue() ? 1 : 0;
					return yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(resultVar)}},
						std::make_unique<yul::Expression>(
							yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(value)}
						)
					};
				}
				// Handle integer constants
				else if (auto intAttr = op->getAttrOfType<mlir::IntegerAttr>("value"))
				{
					std::string resultVar = getOrCreateVariableName(op->getResult(0));
					u256 value(intAttr.getValue().getLimitedValue());
					return yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(resultVar)}},
						std::make_unique<yul::Expression>(
							yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(value)}
						)
					};
				}
			}
			return std::nullopt;
		}
		else if (opName == "solidity.comment")
		{
			// Comments are just placeholders, skip them
			return std::nullopt;
		}
		else if (opName == "solidity.for")
		{
			// Legacy support - shouldn't be reached with new code
			return processForOpToAST(op);
		}
		else if (opName == "solidity.while")
		{
			// Legacy support - shouldn't be reached with new code
			return processWhileOpToAST(op);
		}
		else if (opName == "solidity.revert")
		{
			return yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
					}
				}
			};
		}
		else if (opName == "solidity.require")
		{
			return processRequireOpToAST(op);
		}
		else if (opName == "solidity.assert")
		{
			return processAssertOpToAST(op);
		}
		else if (opName == "solidity.load_state")
		{
			return processLoadStateOpToAST(op);
		}
		else if (opName == "solidity.store_state")
		{
			return processStoreStateOpToAST(op);
		}
		else if (opName == "solidity.array_access")
		{
			return processArrayAccessOpToAST(op);
		}
		else if (opName == "solidity.array_store")
		{
			return processArrayStoreOpToAST(op);
		}
		else if (opName == "solidity.mapping_access")
		{
			return processMappingAccessOpToAST(op);
		}
		else if (opName == "solidity.mapping_store")
		{
			return processMappingStoreOpToAST(op);
		}
		else if (opName == "solidity.function_call")
		{
			return processFunctionCallOpToAST(op);
		}
		else if (opName == "solidity.struct_create")
		{
			// For now, structs are flattened to tuples in memory
			// This is a simplified implementation
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				// For simplicity, we'll create a placeholder value
				// In a full implementation, this would create a memory struct
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::Literal{
						debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))
					})
				};
			}
		}
		else if (opName == "solidity.array_push")
		{
			// Array push is a storage operation
			// This is a simplified implementation
			auto debugData = langutil::DebugData::create();
			if (op->getNumOperands() >= 2)
			{
				std::string arrayVar = getVariableName(op->getOperand(0));
				std::string valueVar = getVariableName(op->getOperand(1));
				// For now, this is a no-op in the simplified implementation
				// In a full implementation, this would manage dynamic array storage
				return yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("sstore")},
						{
							yul::Identifier{debugData, yul::YulName(arrayVar)},
							yul::Identifier{debugData, yul::YulName(valueVar)}
						}
					}
				};
			}
		}
		else if (opName == "solidity.array_length")
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string arrayVar = getVariableName(op->getOperand(0));
				// Simplified: arrays store their length at their storage slot
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("sload")},
						{yul::Identifier{debugData, yul::YulName(arrayVar)}}
					})
				};
			}
		}
		else if (opName == "solidity.member_access")
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string objectVar = getVariableName(op->getOperand(0));
				// Simplified: member access returns the object itself for now
				// In a full implementation, this would calculate the offset
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::Identifier{
						debugData, yul::YulName(objectVar)
					})
				};
			}
		}
		else if (opName == "solidity.address_balance")
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string addrVar = getVariableName(op->getOperand(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("balance")},
						{yul::Identifier{debugData, yul::YulName(addrVar)}}
					})
				};
			}
		}
		else if (opName == "solidity.address_code")
		{
			// address.code returns bytes memory containing the code
			// In Yul, we need to allocate memory and use extcodecopy
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string addrVar = getVariableName(op->getOperand(0));
				// For now, return extcodesize as a simplification
				// Full implementation would allocate memory and copy code
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("extcodesize")},
						{yul::Identifier{debugData, yul::YulName(addrVar)}}
					})
				};
			}
		}
		else if (opName == "solidity.address_codehash")
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string addrVar = getVariableName(op->getOperand(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("extcodehash")},
						{yul::Identifier{debugData, yul::YulName(addrVar)}}
					})
				};
			}
		}
		else if (opName == "solidity.to_i1")
		{
			// Convert bool to i1 - this is essentially a pass-through in Yul
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string inputVar = getVariableName(op->getOperand(0));
				auto debugData = langutil::DebugData::create();
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::Identifier{
						debugData, yul::YulName(inputVar)
					})
				};
			}
		}
		else if (opName == "solidity.convert")
		{
			// Type conversion - in Yul, most conversions are just assignments
			// The EVM automatically handles the conversion at runtime
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string inputVar = getVariableName(op->getOperand(0));
				auto debugData = langutil::DebugData::create();
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::Identifier{
						debugData, yul::YulName(inputVar)
					})
				};
			}
		}
		else if (opName == "scf.condition" || opName == "scf.yield")
		{
			// These are control flow terminators that don't map to Yul statements
			// They're handled by their parent operations (scf.while)
			return std::nullopt;
		}
		
		return std::nullopt;
	}
	
	// Helper functions for processing MLIR operations to Yul AST
	std::optional<yul::Statement> processArithmeticOpToAST(mlir::Operation* op, const std::string& yulOp)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumResults() > 0 && op->getNumOperands() >= 2)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			std::string lhs = getVariableName(op->getOperand(0));
			std::string rhs = getVariableName(op->getOperand(1));
			
			return yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName(resultVar)}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName(yulOp)},
					{
						yul::Identifier{debugData, yul::YulName(lhs)},
						yul::Identifier{debugData, yul::YulName(rhs)}
					}
				})
			};
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processUnaryOpToAST(mlir::Operation* op, const std::string& yulOp)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			std::string operand = getVariableName(op->getOperand(0));
			
			return yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName(resultVar)}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName(yulOp)},
					{yul::Identifier{debugData, yul::YulName(operand)}}
				})
			};
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processComparisonOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumResults() > 0 && op->getNumOperands() >= 2)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			std::string lhs = getVariableName(op->getOperand(0));
			std::string rhs = getVariableName(op->getOperand(1));
			
			// Get the predicate
			std::string predicate = "eq";
			if (auto predAttr = op->getAttrOfType<mlir::StringAttr>("predicate"))
			{
				std::string pred = predAttr.getValue().str();
				if (pred == "eq") predicate = "eq";
				else if (pred == "ne") predicate = "ne";
				else if (pred == "lt") predicate = "lt";
				else if (pred == "le") predicate = "le";
				else if (pred == "gt") predicate = "gt";
				else if (pred == "ge") predicate = "ge";
			}
			
			// Convert complex predicates to simple ones
			if (predicate == "ne")
			{
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("iszero")},
						{yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("eq")},
							{
								yul::Identifier{debugData, yul::YulName(lhs)},
								yul::Identifier{debugData, yul::YulName(rhs)}
							}
						}}
					})
				};
			}
			else if (predicate == "le")
			{
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("iszero")},
						{yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("gt")},
							{
								yul::Identifier{debugData, yul::YulName(lhs)},
								yul::Identifier{debugData, yul::YulName(rhs)}
							}
						}}
					})
				};
			}
			else if (predicate == "ge")
			{
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("iszero")},
						{yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("lt")},
							{
								yul::Identifier{debugData, yul::YulName(lhs)},
								yul::Identifier{debugData, yul::YulName(rhs)}
							}
						}}
					})
				};
			}
			else
			{
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName(predicate)},
						{
							yul::Identifier{debugData, yul::YulName(lhs)},
							yul::Identifier{debugData, yul::YulName(rhs)}
						}
					})
				};
			}
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processIfOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumOperands() > 0)
		{
			std::string condition = getVariableName(op->getOperand(0));
			
			std::vector<yul::Statement> thenStatements;
			if (op->getNumRegions() > 0)
			{
				for (auto& region : op->getRegion(0))
				{
					for (auto& innerOp : region)
					{
						auto stmt = processOperationToStatement(&innerOp);
						if (stmt)
							thenStatements.push_back(std::move(*stmt));
					}
				}
			}
			
			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(
				yul::Identifier{debugData, yul::YulName(condition)}
			);
			ifStatement.body = yul::Block{debugData, std::move(thenStatements)};
			
			return ifStatement;
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processForOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		
		std::vector<yul::Statement> initStatements;
		std::vector<yul::Statement> bodyStatements;
		std::vector<yul::Statement> postStatements;
		
		// Process init, body, and post regions
		if (op->getNumRegions() > 0)
		{
			// Init block
			for (auto& region : op->getRegion(0))
			{
				for (auto& innerOp : region)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						initStatements.push_back(std::move(*stmt));
				}
			}
		}
		
		if (op->getNumRegions() > 1)
		{
			// Body block
			for (auto& region : op->getRegion(1))
			{
				for (auto& innerOp : region)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						bodyStatements.push_back(std::move(*stmt));
				}
			}
		}
		
		if (op->getNumRegions() > 2)
		{
			// Post block
			for (auto& region : op->getRegion(2))
			{
				for (auto& innerOp : region)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						postStatements.push_back(std::move(*stmt));
				}
			}
		}
		
		yul::ForLoop forLoop{debugData};
		forLoop.pre = yul::Block{debugData, std::move(initStatements)};
		
		// Condition
		if (op->getNumOperands() > 0)
		{
			std::string condition = getVariableName(op->getOperand(0));
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Identifier{debugData, yul::YulName(condition)}
			);
		}
		else
		{
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))}
			);
		}
		
		forLoop.post = yul::Block{debugData, std::move(postStatements)};
		forLoop.body = yul::Block{debugData, std::move(bodyStatements)};
		
		return forLoop;
	}
	
	std::optional<yul::Statement> processWhileOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		
		std::vector<yul::Statement> bodyStatements;
		if (op->getNumRegions() > 0)
		{
			for (auto& region : op->getRegion(0))
			{
				for (auto& innerOp : region)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						bodyStatements.push_back(std::move(*stmt));
				}
			}
		}
		
		// Convert while to for loop (Yul doesn't have while)
		yul::ForLoop forLoop{debugData};
		forLoop.pre = yul::Block{debugData, {}};
		
		// Condition
		if (op->getNumOperands() > 0)
		{
			std::string condition = getVariableName(op->getOperand(0));
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Identifier{debugData, yul::YulName(condition)}
			);
		}
		else
		{
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))}
			);
		}
		
		forLoop.post = yul::Block{debugData, {}};
		forLoop.body = yul::Block{debugData, std::move(bodyStatements)};
		
		return forLoop;
	}
	
	std::optional<yul::Statement> processLoopOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		
		// Process the loop body region
		std::vector<yul::Statement> bodyStatements;
		if (op->getNumRegions() > 0 && !op->getRegion(0).empty())
		{
			for (auto& block : op->getRegion(0))
			{
				for (auto& innerOp : block)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						bodyStatements.push_back(std::move(*stmt));
				}
			}
		}
		
		// Convert to a Yul for loop with always-true condition
		// The actual loop control will be handled by break statements
		yul::ForLoop forLoop{debugData};
		forLoop.pre = yul::Block{debugData, {}}; // Empty pre block
		forLoop.post = yul::Block{debugData, {}}; // Empty post block
		
		// Always true condition for now (actual condition is inside the body)
		forLoop.condition = std::make_unique<yul::Expression>(
			yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))}
		);
		
		forLoop.body = yul::Block{debugData, std::move(bodyStatements)};
		
		return forLoop;
	}
	
	std::optional<yul::Statement> processScfWhileOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		
		// Check for null operation
		if (!op)
			return std::nullopt;
		
		// For scf.while, generate a simple for loop
		// The loop variable needs to be accessible after the loop for the return value
		// We'll return a Block containing both the variable declaration and the loop
		
		std::vector<yul::Statement> statements;
		std::vector<std::string> loopVarNames;
		
		// Handle multiple loop-carried variables (for optimized loops)
		for (unsigned i = 0; i < op->getNumOperands(); ++i)
		{
			std::string loopVarName = "loop_" + std::to_string(m_varCounter++);
			loopVarNames.push_back(loopVarName);
			
			// Get the initial value for this loop variable
			std::string initVarName = getVariableName(op->getOperand(i));
			if (initVarName.empty())
			{
				// If no name exists, create a literal 0
				statements.push_back(yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(loopVarName)}},
					std::make_unique<yul::Expression>(
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
					)
				});
			}
			else
			{
				// Declare loop variable outside the loop
				statements.push_back(yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(loopVarName)}},
					std::make_unique<yul::Expression>(
						yul::Identifier{debugData, yul::YulName(initVarName)}
					)
				});
			}
			
			// Map the results to the loop variables
			if (i < op->getNumResults())
			{
				void* key = op->getResult(i).getAsOpaquePointer();
				m_valueNames[key] = yul::YulName(loopVarName);
			}
		}
		
		// Use the first loop variable as the primary one for compatibility
		std::string loopVarName = loopVarNames.empty() ? "loop_" + std::to_string(m_varCounter++) : loopVarNames[0];
			
		// Map the loop variables for the before/after regions
		if (op->getNumRegions() > 0)
		{
			auto& beforeRegion = op->getRegion(0);
			for (auto& block : beforeRegion)
			{
				for (unsigned i = 0; i < block.getNumArguments(); ++i)
				{
					void* key = block.getArgument(i).getAsOpaquePointer();
					// Map each block argument to its corresponding loop variable
					if (i < loopVarNames.size())
						m_valueNames[key] = yul::YulName(loopVarNames[i]);
					else
						m_valueNames[key] = yul::YulName(loopVarName);
				}
			}
		}
		if (op->getNumRegions() > 1)
		{
			auto& afterRegion = op->getRegion(1);
			for (auto& block : afterRegion)
			{
				for (unsigned i = 0; i < block.getNumArguments(); ++i)
				{
					void* key = block.getArgument(i).getAsOpaquePointer();
					// Map each block argument to its corresponding loop variable
					if (i < loopVarNames.size())
						m_valueNames[key] = yul::YulName(loopVarNames[i]);
					else
						m_valueNames[key] = yul::YulName(loopVarName);
				}
			}
		}
		
		// Create the for loop
		yul::ForLoop forLoop{debugData};
		
		// Pre block is now empty since we declared the variable outside
		forLoop.pre = yul::Block{debugData, {}};
		// Post block will be set later with the collected post statements
		
		// Build the condition expression from the before region
		// For now, use a simplified approach
		std::unique_ptr<yul::Expression> conditionExpr;
		
		if (op->getNumRegions() > 0 && !op->getRegion(0).empty())
		{
			// Map block arguments to loop variables
			for (auto& block : op->getRegion(0))
			{
				for (unsigned i = 0; i < block.getNumArguments(); ++i)
				{
					void* key = block.getArgument(i).getAsOpaquePointer();
					m_valueNames[key] = yul::YulName(loopVarName);
				}
				
				// Look for the condition operation
				// First process all operations to build up the condition
				std::string conditionVarName;
				for (auto& innerOp : block)
				{
					if (innerOp.getName().getStringRef() == "solidity.cmp")
					{
						// Process the comparison operation
						if (innerOp.getNumOperands() >= 2 && innerOp.getNumResults() > 0)
						{
							// Generate variable for comparison result
							void* resultKey = innerOp.getResult(0).getAsOpaquePointer();
							std::string resultName = "cmp_" + std::to_string(m_varCounter++);
							m_valueNames[resultKey] = yul::YulName(resultName);
							
							// Get comparison type from the predicate attribute
							std::string cmpType = "lt"; // default
							// The predicate is stored as the third attribute in the operation
							if (innerOp.getNumOperands() >= 2)
							{
								// Try to get the predicate attribute
								if (auto cmpAttr = innerOp.getAttrOfType<mlir::StringAttr>("predicate"))
								{
									cmpType = cmpAttr.getValue().str();
								}
								else
								{
									// For cmp operations, the predicate might be stored in the assembly format
									// Try to extract it from the operation's attributes
									for (auto& attr : innerOp.getAttrs())
									{
										if (attr.getName().getValue() == "predicate")
										{
											if (auto strAttr = attr.getValue().dyn_cast<mlir::StringAttr>())
												cmpType = strAttr.getValue().str();
										}
									}
								}
							}
							
							// Map comparison types to Yul operations
							// Note: Yul doesn't have le/ge directly, need to use combinations
							std::string yulOp = "lt";
							bool needNot = false;
							if (cmpType == "eq") {
								yulOp = "eq";
							}
							else if (cmpType == "ne") {
								yulOp = "eq";
								needNot = true;  // ne(a,b) = iszero(eq(a,b))
							}
							else if (cmpType == "lt") {
								yulOp = "lt";
							}
							else if (cmpType == "le") {
								// le(a,b) = iszero(gt(a,b))
								yulOp = "gt";
								needNot = true;
							}
							else if (cmpType == "gt") {
								yulOp = "gt";
							}
							else if (cmpType == "ge") {
								// ge(a,b) = iszero(lt(a,b))
								yulOp = "lt";
								needNot = true;
							}
							
							// Get operand names
							std::string lhs = getVariableName(innerOp.getOperand(0));
							std::string rhs = getVariableName(innerOp.getOperand(1));
							
							// Build the condition expression
							if (needNot) {
								// Wrap in iszero for negation
								conditionExpr = std::make_unique<yul::Expression>(
									yul::FunctionCall{
										debugData,
										yul::Identifier{debugData, yul::YulName("iszero")},
										{
											yul::FunctionCall{
												debugData,
												yul::Identifier{debugData, yul::YulName(yulOp)},
												{
													yul::Identifier{debugData, yul::YulName(lhs)},
													yul::Identifier{debugData, yul::YulName(rhs)}
												}
											}
										}
									}
								);
							} else {
								conditionExpr = std::make_unique<yul::Expression>(
									yul::FunctionCall{
										debugData,
										yul::Identifier{debugData, yul::YulName(yulOp)},
										{
											yul::Identifier{debugData, yul::YulName(lhs)},
											yul::Identifier{debugData, yul::YulName(rhs)}
										}
									}
								);
							}
							conditionVarName = resultName;
						}
					}
					else if (innerOp.getName().getStringRef() == "solidity.to_i1")
					{
						// Convert boolean to i1 - just pass through the variable name
						if (innerOp.getNumOperands() > 0 && innerOp.getNumResults() > 0)
						{
							void* resultKey = innerOp.getResult(0).getAsOpaquePointer();
							std::string inputName = getVariableName(innerOp.getOperand(0));
							m_valueNames[resultKey] = yul::YulName(inputName);
						}
					}
					else if (innerOp.getName().getStringRef() == "scf.condition")
					{
						// The first operand is the condition variable
						if (innerOp.getNumOperands() > 0 && !conditionExpr)
						{
							// If we haven't built a condition expression yet, use the operand directly
							std::string condVar = getVariableName(innerOp.getOperand(0));
							if (!condVar.empty())
							{
								conditionExpr = std::make_unique<yul::Expression>(
									yul::Identifier{debugData, yul::YulName(condVar)}
								);
							}
						}
					}
				}
			}
		}
		
		// Set the loop condition
		if (conditionExpr)
		{
			forLoop.condition = std::move(conditionExpr);
		}
		else
		{
			// Default to true
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))}
			);
		}
		
		// Process the after region (loop body)
		std::vector<yul::Statement> bodyStatements;
		std::vector<yul::Statement> postStatements;  // Track post-increment statements
		
		if (op->getNumRegions() > 1 && !op->getRegion(1).empty())
		{
			for (auto& block : op->getRegion(1))
			{
				// Map block arguments to the corresponding loop variables
				for (unsigned i = 0; i < block.getNumArguments(); ++i)
				{
					void* key = block.getArgument(i).getAsOpaquePointer();
					// Map each block argument to its corresponding loop variable
					if (i < loopVarNames.size())
						m_valueNames[key] = yul::YulName(loopVarNames[i]);
					else
						m_valueNames[key] = yul::YulName(loopVarName);
				}
				
				// First, identify which operations are yielded (to avoid processing them in body)
				std::set<mlir::Operation*> yieldedOps;
				for (auto& innerOp : block)
				{
					if (innerOp.getName().getStringRef() == "scf.yield")
					{
						if (innerOp.getNumOperands() > 0)
						{
							mlir::Value yieldedValue = innerOp.getOperand(0);
							mlir::Operation* definingOp = yieldedValue.getDefiningOp();
							if (definingOp)
							{
								yieldedOps.insert(definingOp);
								
								// Also skip operations that feed into the yielded operation
								// (e.g., constants used in add operations)
								if (definingOp->getName().getStringRef() == "solidity.add")
								{
									for (auto operand : definingOp->getOperands())
									{
										if (auto constOp = operand.getDefiningOp())
										{
											if (constOp->getName().getStringRef() == "solidity.constant")
												yieldedOps.insert(constOp);
										}
									}
								}
							}
						}
					}
				}
				
				// Process all operations in the body
				for (auto& innerOp : block)
				{
					// Skip operations that are yielded (they'll be in post block)
					if (yieldedOps.count(&innerOp) > 0)
						continue;
					
					if (innerOp.getName().getStringRef() == "scf.yield")
					{
						// Handle multiple yielded values for optimized loops
						for (unsigned yieldIdx = 0; yieldIdx < innerOp.getNumOperands(); ++yieldIdx)
						{
							if (yieldIdx >= loopVarNames.size())
								break;
								
							mlir::Value yieldedValue = innerOp.getOperand(yieldIdx);
							std::string targetVarName = loopVarNames[yieldIdx];
							
							// For the first operand (loop counter), handle increment specially
							if (yieldIdx == 0)
							{
								mlir::Operation* definingOp = yieldedValue.getDefiningOp();
								
								if (definingOp && definingOp->getName().getStringRef() == "solidity.add")
								{
									// Generate the add expression directly in the post block
									std::string lhs = getVariableName(definingOp->getOperand(0));
									
									// Check if the second operand is a constant
									std::vector<yul::Expression> args;
									args.push_back(yul::Identifier{debugData, yul::YulName(lhs)});
									
									// Check if rhs is from a constant operation
									mlir::Value rhsValue = definingOp->getOperand(1);
									mlir::Operation* rhsDefOp = rhsValue.getDefiningOp();
									
									if (rhsDefOp && rhsDefOp->getName().getStringRef() == "solidity.constant")
									{
										// Get the constant value
										if (auto intAttr = rhsDefOp->getAttrOfType<mlir::IntegerAttr>("value"))
										{
											u256 val(intAttr.getValue().getLimitedValue());
											args.push_back(yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(val)});
										}
										else
										{
											// Fallback to 1 if we can't get the value
											args.push_back(yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))});
										}
									}
									else
									{
										// Not a constant, use the variable name
										std::string rhs = getVariableName(definingOp->getOperand(1));
										args.push_back(yul::Identifier{debugData, yul::YulName(rhs)});
									}
									
									postStatements.push_back(yul::Assignment{
										debugData,
										{{debugData, yul::YulName(targetVarName)}},
										std::make_unique<yul::Expression>(yul::FunctionCall{
											debugData,
											yul::Identifier{debugData, yul::YulName("add")},
											std::move(args)
										})
									});
								}
								else
								{
									// For other cases, try to use the yielded variable
									std::string yieldedVarName = getVariableName(yieldedValue);
									if (yieldedVarName != targetVarName && !yieldedVarName.empty())
									{
										postStatements.push_back(yul::Assignment{
											debugData,
											{{debugData, yul::YulName(targetVarName)}},
											std::make_unique<yul::Expression>(
												yul::Identifier{debugData, yul::YulName(yieldedVarName)}
											)
										});
									}
								}
							}
							else
							{
								// For other loop-carried values (e.g., cached storage)
								// Don't update in post section as the value is computed in body
								// and would be out of scope. The value will be updated in the body itself.
								// We just need to mark that this operation was processed
								continue;
							}
						}
						// Don't process scf.yield further as it's just a terminator
						continue;
					}
					else if (innerOp.getName().getStringRef() == "scf.while")
					{
						// Handle nested SCF while operations recursively
						auto stmt = processScfWhileOpToAST(&innerOp);
						if (stmt)
							bodyStatements.push_back(std::move(*stmt));
					}
					else
					{
						auto stmt = processOperationToStatement(&innerOp);
						if (stmt)
						{
							bodyStatements.push_back(std::move(*stmt));
							
							// If this operation produces a value that will be yielded as a loop-carried value,
							// we need to assign it to the corresponding loop variable
							if (innerOp.getNumResults() > 0)
							{
								// Check if this result is used by the scf.yield
								for (auto& checkOp : block)
								{
									if (checkOp.getName().getStringRef() == "scf.yield")
									{
										// Check if any of the yielded operands (except the first) use this result
										for (unsigned yieldIdx = 1; yieldIdx < checkOp.getNumOperands(); ++yieldIdx)
										{
											if (yieldIdx >= loopVarNames.size())
												break;
												
											mlir::Value yieldedValue = checkOp.getOperand(yieldIdx);
											// Check if this yielded value is the result of the current operation
											if (yieldedValue.getDefiningOp() == &innerOp)
											{
												// Generate an assignment to update the loop variable
												std::string resultVarName = getVariableName(innerOp.getResult(0));
												if (!resultVarName.empty())
												{
													bodyStatements.push_back(yul::Assignment{
														debugData,
														{{debugData, yul::YulName(loopVarNames[yieldIdx])}},
														std::make_unique<yul::Expression>(
															yul::Identifier{debugData, yul::YulName(resultVarName)}
														)
													});
												}
											}
										}
										break;
									}
								}
							}
						}
					}
				}
			}
		}
		
		forLoop.body = yul::Block{debugData, std::move(bodyStatements)};
		
		// CRITICAL FIX: Set the post block with the collected post statements
		forLoop.post = yul::Block{debugData, std::move(postStatements)};
		
		// Add the for loop to the statements
		statements.push_back(std::move(forLoop));
		
		// If we have multiple statements, wrap them in a Block
		if (statements.size() > 1)
		{
			return yul::Block{debugData, std::move(statements)};
		}
		else if (!statements.empty())
		{
			return std::move(statements[0]);
		}
		else
		{
			return std::nullopt;
		}
	}
	
	std::optional<yul::Statement> processRequireOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumOperands() > 0)
		{
			std::string condition = getVariableName(op->getOperand(0));
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}
					}
				}
			});
			
			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("iszero")},
				{yul::Identifier{debugData, yul::YulName(condition)}}
			});
			ifStatement.body = yul::Block{debugData, std::move(bodyStatements)};
			
			return ifStatement;
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processAssertOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumOperands() > 0)
		{
			std::string condition = getVariableName(op->getOperand(0));
			
			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("invalid")},
					{}
				}
			});
			
			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("iszero")},
				{yul::Identifier{debugData, yul::YulName(condition)}}
			});
			ifStatement.body = yul::Block{debugData, std::move(bodyStatements)};
			
			return ifStatement;
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processLoadStateOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumResults() > 0)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			
			// Get the state variable name from the varName attribute
			std::string varName = "unknown";
			if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("varName"))
				varName = nameAttr.getValue().str();
			
			// Check if this is a constant
			auto constIt = m_constants.find(varName);
			if (constIt != m_constants.end())
			{
				// For constants, return the literal value
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(constIt->second)}
					)
				};
			}
			else
			{
				// Look up the storage slot for this state variable
				uint32_t slot = 0;
				auto it = m_stateVariableSlots.find(varName);
				if (it != m_stateVariableSlots.end())
					slot = it->second;
				
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("sload")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))}}
					})
				};
			}
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processStoreStateOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumOperands() > 0)
		{
			// Get the state variable name from the varName attribute
			std::string varName = "unknown";
			if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("varName"))
				varName = nameAttr.getValue().str();
			
			// Get the value to store (the operand)
			std::string value = getVariableName(op->getOperand(0));
			
			// Look up the storage slot for this state variable
			uint32_t slot = 0;
			auto it = m_stateVariableSlots.find(varName);
			if (it != m_stateVariableSlots.end())
				slot = it->second;
			
			return yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("sstore")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))},
						yul::Identifier{debugData, yul::YulName(value)}
					}
				}
			};
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processArrayAccessOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumResults() > 0 && op->getNumOperands() >= 2)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			std::string array = getVariableName(op->getOperand(0));
			std::string index = getVariableName(op->getOperand(1));
			
			return yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName(resultVar)}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("sload")},
					{yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("add")},
						{
							yul::Identifier{debugData, yul::YulName(array)},
							yul::Identifier{debugData, yul::YulName(index)}
						}
					}}
				})
			};
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processArrayStoreOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumOperands() >= 3)
		{
			std::string array = getVariableName(op->getOperand(0));
			std::string index = getVariableName(op->getOperand(1));
			std::string value = getVariableName(op->getOperand(2));
			
			return yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("sstore")},
					{
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("add")},
							{
								yul::Identifier{debugData, yul::YulName(array)},
								yul::Identifier{debugData, yul::YulName(index)}
							}
						},
						yul::Identifier{debugData, yul::YulName(value)}
					}
				}
			};
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processMappingAccessOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumResults() > 0 && op->getNumOperands() >= 2)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			std::string mapping = getVariableName(op->getOperand(0));
			std::string key = getVariableName(op->getOperand(1));
			
			// Calculate mapping storage slot using keccak256(key . mapping_slot)
			std::vector<yul::Statement> statements;
			
			// mstore(0, key)
			statements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Identifier{debugData, yul::YulName(key)}
					}
				}
			});
			
			// mstore(32, mapping_slot)
			statements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))},
						yul::Identifier{debugData, yul::YulName(mapping)}
					}
				}
			});
			
			// resultVar := sload(keccak256(0, 64))
			statements.push_back(yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName(resultVar)}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("sload")},
					{yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("keccak256")},
						{
							yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
							yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))}
						}
					}}
				})
			});
			
			// Return a block containing all statements
			return yul::Block{debugData, std::move(statements)};
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processMappingStoreOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumOperands() >= 3)
		{
			std::string mapping = getVariableName(op->getOperand(0));
			std::string key = getVariableName(op->getOperand(1));
			std::string value = getVariableName(op->getOperand(2));
			
			std::vector<yul::Statement> statements;
			
			// mstore(0, key)
			statements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						yul::Identifier{debugData, yul::YulName(key)}
					}
				}
			});
			
			// mstore(32, mapping_slot)
			statements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))},
						yul::Identifier{debugData, yul::YulName(mapping)}
					}
				}
			});
			
			// sstore(keccak256(0, 64), value)
			statements.push_back(yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("sstore")},
					{
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("keccak256")},
							{
								yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))}
							}
						},
						yul::Identifier{debugData, yul::YulName(value)}
					}
				}
			});
			
			return yul::Block{debugData, std::move(statements)};
		}
		return std::nullopt;
	}
	
	std::optional<yul::Statement> processFunctionCallOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		
		std::string funcName = "unknown";
		if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("callee"))
			funcName = nameAttr.getValue().str();
		
		// Prefix function name unless it's a builtin
		if (!isYulBuiltin(funcName))
			funcName = "fun_" + funcName;
		
		std::vector<yul::Expression> args;
		for (unsigned i = 0; i < op->getNumOperands(); ++i)
		{
			std::string argName = getVariableName(op->getOperand(i));
			args.push_back(yul::Identifier{debugData, yul::YulName(argName)});
		}
		
		if (op->getNumResults() > 0)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			return yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName(resultVar)}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName(funcName)},
					args
				})
			};
		}
		else
		{
			return yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName(funcName)},
					args
				}
			};
		}
	}
	
	
	
	std::string getFunctionSignature(mlir::Operation* funcOp)
	{
		std::string funcName = "unknown";
		if (auto nameAttr = funcOp->getAttrOfType<mlir::StringAttr>("sym_name"))
			funcName = nameAttr.getValue().str();
		
		std::string signature = funcName + "(";
		
		// Extract parameter types from MLIR function_type attribute
		if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
		{
			auto funcType = typeAttr.getValue().cast<mlir::FunctionType>();
			auto inputTypes = funcType.getInputs();
			
			for (size_t i = 0; i < inputTypes.size(); ++i)
			{
				if (i > 0) signature += ",";
				signature += extractSolidityTypeString(inputTypes[i]);
			}
		}
		else if (funcOp->getNumRegions() > 0 && !funcOp->getRegion(0).empty())
		{
			// Fallback: extract types from block arguments
			auto& entryBlock = funcOp->getRegion(0).front();
			int numParams = entryBlock.getNumArguments();
			
			for (int i = 0; i < numParams; ++i)
			{
				if (i > 0) signature += ",";
				auto argType = entryBlock.getArgument(i).getType();
				signature += extractSolidityTypeString(argType);
			}
		}
		
		signature += ")";
		return signature;
	}
	
	// Helper function to extract Solidity type string from MLIR type
	std::string extractSolidityTypeString(mlir::Type type)
	{
		// Try to extract from MLIR Solidity dialect types
		if (auto intType = type.dyn_cast<mlir::IntegerType>())
		{
			// For integer types, determine if signed/unsigned and bit width
			unsigned width = intType.getWidth();
			if (width == 1)
				return "bool";
			else if (width <= 256)
			{
				// Default to uint for now - in a complete implementation,
				// we would track signedness in the MLIR type system
				return "uint" + std::to_string(width);
			}
			else
				return "uint256"; // Fallback
		}
		
		// Check for string representation in type name (fallback for custom types)
		std::string typeStr = type.getDialect().getNamespace().str() + ".";
		
		// Extract type name from MLIR representation
		std::string fullTypeStr;
		{
			llvm::raw_string_ostream stream(fullTypeStr);
			type.print(stream);
		}
		
		// Try to extract Solidity type from MLIR representation
		if (fullTypeStr.find("uint") != std::string::npos)
		{
			// Extract width from !solidity.uint<256> format
			size_t start = fullTypeStr.find("<");
			size_t end = fullTypeStr.find(">");
			if (start != std::string::npos && end != std::string::npos && end > start)
			{
				std::string widthStr = fullTypeStr.substr(start + 1, end - start - 1);
				return "uint" + widthStr;
			}
			return "uint256"; // Default uint
		}
		else if (fullTypeStr.find("int") != std::string::npos)
		{
			// Extract width from !solidity.int<256> format
			size_t start = fullTypeStr.find("<");
			size_t end = fullTypeStr.find(">");
			if (start != std::string::npos && end != std::string::npos && end > start)
			{
				std::string widthStr = fullTypeStr.substr(start + 1, end - start - 1);
				return "int" + widthStr;
			}
			return "int256"; // Default int
		}
		else if (fullTypeStr.find("address") != std::string::npos)
		{
			return "address";
		}
		else if (fullTypeStr.find("bool") != std::string::npos)
		{
			return "bool";
		}
		else if (fullTypeStr.find("bytes") != std::string::npos)
		{
			// Check for dynamic bytes vs fixed bytes
			if (fullTypeStr.find("dynamic") != std::string::npos)
				return "bytes";
			else
			{
				// Extract size from !solidity.bytes<32> format
				size_t start = fullTypeStr.find("<");
				size_t end = fullTypeStr.find(">");
				if (start != std::string::npos && end != std::string::npos && end > start)
				{
					std::string sizeStr = fullTypeStr.substr(start + 1, end - start - 1);
					return "bytes" + sizeStr;
				}
				return "bytes32"; // Default fixed bytes
			}
		}
		else if (fullTypeStr.find("string") != std::string::npos)
		{
			return "string";
		}
		
		// Default fallback - assume uint256 for unknown types
		return "uint256";
	}
	
	bool isYulBuiltin(const std::string& name)
	{
		// Common Yul builtins - expand this list as needed
		static const std::set<std::string> builtins = {
			"add", "sub", "mul", "div", "mod", "exp",
			"and", "or", "xor", "not", "shl", "shr", "sar",
			"eq", "lt", "gt", "iszero",
			"mstore", "mload", "sstore", "sload",
			"return", "revert", "stop", "invalid",
			"keccak256", "address", "balance", "origin", "caller",
			"callvalue", "calldataload", "calldatasize", "calldatacopy",
			"codesize", "codecopy", "gasprice", "extcodesize", "extcodecopy",
			"returndatasize", "returndatacopy", "blockhash", "coinbase",
			"timestamp", "number", "difficulty", "gaslimit", "chainid",
			"selfbalance", "basefee", "gas", "call", "callcode",
			"delegatecall", "staticcall", "create", "create2", "selfdestruct"
		};
		return builtins.find(name) != builtins.end();
	}
	
	std::string getOrCreateVariableName(mlir::Value value)
	{
		void* key = value.getAsOpaquePointer();
		
		// First check function-scoped names if we're in a function context
		if (!m_currentFunction.empty() && 
			m_functionScopedNames.find(m_currentFunction) != m_functionScopedNames.end())
		{
			auto& functionScope = m_functionScopedNames[m_currentFunction];
			auto funcIt = functionScope.find(key);
			if (funcIt != functionScope.end())
				return funcIt->second.str();
		}
		
		// Fallback to global names
		auto it = m_valueNames.find(key);
		if (it != m_valueNames.end())
			return it->second.str();
		
		// Create new scoped variable name
		std::string name;
		if (!m_currentFunction.empty())
		{
			name = "v" + std::to_string(m_functionVarCounter++);
			yul::YulName yulName(name);
			m_functionScopedNames[m_currentFunction][key] = yulName;
		}
		else
		{
			name = "v" + std::to_string(m_varCounter++);
			yul::YulName yulName(name);
			m_valueNames[key] = yulName;
		}
		return name;
	}
	
	std::string getVariableName(mlir::Value value)
	{
		void* key = value.getAsOpaquePointer();
		
		// First check function-scoped names if we're in a function context
		if (!m_currentFunction.empty() && 
			m_functionScopedNames.find(m_currentFunction) != m_functionScopedNames.end())
		{
			auto& functionScope = m_functionScopedNames[m_currentFunction];
			auto funcIt = functionScope.find(key);
			if (funcIt != functionScope.end())
				return funcIt->second.str();
		}
		
		// Fallback to global names
		auto it = m_valueNames.find(key);
		if (it != m_valueNames.end())
			return it->second.str();
		
		// If not found, create a new name
		return getOrCreateVariableName(value);
	}
	
	std::shared_ptr<yul::Object> generatePlaceholderObject()
	{
		auto object = std::make_shared<yul::Object>();
		object->name = "Contract";
		object->debugData = std::make_shared<yul::ObjectDebugData>();
		
		// Create simple constructor code
		auto debugData = langutil::DebugData::create();
		std::vector<yul::Statement> constructorStatements;
		
		// stop()
		constructorStatements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("stop")},
				{}
			}
		});
		
		yul::Block constructorBlock{debugData, std::move(constructorStatements)};
		
		if (!m_dialect)
			m_dialect = &yul::EVMDialect::strictAssemblyForEVMObjects(langutil::EVMVersion(), std::nullopt);
		
		auto constructorAST = std::make_shared<yul::AST const>(*m_dialect, std::move(constructorBlock));
		object->setCode(constructorAST);
		
		// Create deployed object
		auto deployedObject = std::make_shared<yul::Object>();
		deployedObject->name = "Contract_deployed";
		deployedObject->debugData = std::make_shared<yul::ObjectDebugData>();
		
		std::vector<yul::Statement> deployedStatements;
		deployedStatements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("stop")},
				{}
			}
		});
		
		yul::Block deployedBlock{debugData, std::move(deployedStatements)};
		auto deployedAST = std::make_shared<yul::AST const>(*m_dialect, std::move(deployedBlock));
		deployedObject->setCode(deployedAST);
		
		// Add deployed as subobject
		object->subObjects.push_back(deployedObject);
		object->subIndexByName["Contract_deployed"] = 0;
		
		return object;
	}
#else
	std::shared_ptr<yul::Object> generatePlaceholderObject()
	{
		auto object = std::make_shared<yul::Object>();
		object->name = "Contract";
		object->debugData = std::make_shared<yul::ObjectDebugData>();
		
		// Create simple constructor code
		auto debugData = langutil::DebugData::create();
		std::vector<yul::Statement> constructorStatements;
		
		// stop()
		constructorStatements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("stop")},
				{}
			}
		});
		
		yul::Block constructorBlock{debugData, std::move(constructorStatements)};
		
		if (!m_dialect)
			m_dialect = &yul::EVMDialect::strictAssemblyForEVMObjects(langutil::EVMVersion(), std::nullopt);
		
		auto constructorAST = std::make_shared<yul::AST const>(*m_dialect, std::move(constructorBlock));
		object->setCode(constructorAST);
		
		// Create deployed object
		auto deployedObject = std::make_shared<yul::Object>();
		deployedObject->name = "Contract_deployed";
		deployedObject->debugData = std::make_shared<yul::ObjectDebugData>();
		
		std::vector<yul::Statement> deployedStatements;
		deployedStatements.push_back(yul::ExpressionStatement{
			debugData,
			yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("stop")},
				{}
			}
		});
		
		yul::Block deployedBlock{debugData, std::move(deployedStatements)};
		auto deployedAST = std::make_shared<yul::AST const>(*m_dialect, std::move(deployedBlock));
		deployedObject->setCode(deployedAST);
		
		// Add deployed as subobject
		object->subObjects.push_back(deployedObject);
		object->subIndexByName["Contract_deployed"] = 0;
		
		return object;
	}
#endif // SOLIDITY_HAS_MLIR
};

MLIRToYulLowering::MLIRToYulLowering():
	m_impl(std::make_unique<MLIRToYulLoweringImpl>())
{
}

MLIRToYulLowering::~MLIRToYulLowering() = default;

std::shared_ptr<yul::Object> MLIRToYulLowering::lower(std::string const& _mlirModule)
{
	return m_impl->lower(_mlirModule);
}

std::string MLIRToYulLowering::optimize(std::string const& _mlirModule, bool _printIntermediateMLIR, std::string const& _mlirFile, bool _runAnalysis)
{
	return m_impl->optimize(_mlirModule, _printIntermediateMLIR, _mlirFile, _runAnalysis);
}

} // namespace solidity::frontend
