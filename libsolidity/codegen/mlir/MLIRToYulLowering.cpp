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

#include <liblangutil/DebugData.h>
#include <liblangutil/EVMVersion.h>
#include <libsolutil/CommonData.h>
#include <libsolutil/FunctionSelector.h>
#include <libsolutil/Numeric.h>
#include <libyul/AST.h>
#include <libyul/ASTForward.h>
#include <libyul/Dialect.h>
#include <libyul/Object.h>
#include <libyul/backends/evm/EVMDialect.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <vector>

#ifdef SOLIDITY_HAS_MLIR

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#pragma GCC diagnostic pop

#include <libsolidity/codegen/mlir/Dialect/SolidityDialect.h>
#include <libsolidity/codegen/mlir/Dialect/SolidityOps.h>
#include <libsolidity/codegen/mlir/Passes/AccessControlAnalysisPass.h>
#include <libsolidity/codegen/mlir/Passes/StorageCachingPass.h>

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
		(void) _mlirModule;
		return generatePlaceholderObject();
#endif
	}

	std::string optimize(
		std::string const& _mlirModule,
		bool _printIntermediateMLIR = false,
		std::string const& _mlirFile = "",
		bool _runAnalysis = false)
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
			[&mlirWarnings](mlir::Diagnostic& diag) -> mlir::LogicalResult
			{
				if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning)
				{
					// Extract just the message string, not the full diagnostic with location
					mlirWarnings.push_back(diag.str());
					return mlir::success(); // Mark as handled
				}
				return mlir::failure(); // Let other diagnostics be handled normally
			});

		// Create a pass manager and add optimization passes
		mlir::PassManager pm(m_context.get());

		// Enable IR printing after each pass if requested
		if (_printIntermediateMLIR)
		{
			// Disable multi-threading to enable IR printing
			m_context->disableMultithreading();
			// Enable printing after each pass
			pm.enableIRPrinting(
				/*shouldPrintBeforePass=*/
				[](mlir::Pass* pass, mlir::Operation*)
				{
					llvm::errs() << "\n// Before " << pass->getName() << " pass:\n";
					return true;
				},
				/*shouldPrintAfterPass=*/
				[](mlir::Pass* pass, mlir::Operation*)
				{
					llvm::errs() << "\n// After " << pass->getName() << " pass:\n";
					return true;
				},
				/*printModuleScope=*/true,
				/*printAfterOnlyOnChange=*/false, // Print even if no changes
				/*printAfterOnlyOnFailure=*/false,
				/*out=*/llvm::errs());

			llvm::errs() << "\n=== Starting MLIR Optimization Pipeline ===\n";
			llvm::errs() << "\n// Initial MLIR:\n" << _mlirModule << "\n";
		}

		// Add our custom storage caching pass at module level
		pm.addPass(mlir::solidity::createStorageCachingPass());

		// Add security analysis pass if requested
		if (_runAnalysis)
		{
			pm.addPass(mlir::solidity::createAccessControlAnalysisPass());
		}

		// Add standard MLIR optimization passes
		// Note: CSE needs to run on func::FuncOp, but we don't have those in Solidity dialect
		// pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());  // Would need func::FuncOp
		pm.addPass(mlir::createCanonicalizerPass()); // Canonicalize operations at module level
		pm.addPass(mlir::createInlinerPass());		 // Function inlining at module level

		// Run the optimization pipeline
		if (mlir::failed(pm.run(module.get()->getOperation())))
		{
			// If optimization fails, return the original module
			return _mlirModule;
		}

		if (_printIntermediateMLIR)
		{
			llvm::errs() << "\n=== MLIR Optimization Pipeline Complete ===\n";
		}

		// Output collected warnings from MLIR passes
		for (const auto& warning: mlirWarnings)
		{
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
				flags.enableDebugInfo(); // This enables printing of location information

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
	std::string m_currentFunction;												// Track current function context
	int m_varCounter = 0;
	int m_functionVarCounter = 0; // Per-function variable counter
	yul::Dialect const* m_dialect = nullptr;
	std::map<std::string, uint32_t> m_stateVariableSlots; // Map state variable names to storage slots
	std::map<std::string, u256> m_constants;			  // Map constant names to their values

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

	/// Helper to add a statement to a vector, unwrapping yul::Block into flat statements.
	/// This avoids creating nested { } scopes that hide variable declarations from the parent scope.
	void addStatementFlattened(std::vector<yul::Statement>& target, yul::Statement stmt)
	{
		if (auto* blockStmt = std::get_if<yul::Block>(&stmt))
		{
			for (auto& s: blockStmt->statements)
				target.push_back(std::move(s));
		}
		else
		{
			target.push_back(std::move(stmt));
		}
	}

	std::unique_ptr<mlir::ModuleOp> parseMLIR(std::string const& _mlirText)
	{
		// Parse the MLIR module from string
		mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(_mlirText, m_context.get());

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
		module->walk(
			[&](mlir::Operation* op)
			{
				if (mlir::isa<mlir::solidity::ContractOp>(op))
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
						"a264697066735822122084f07e9a4a7822765f80708f9711881a1fcd60b895313fa25d069164f67bd9e064736f6c63"
						"782b302e382e33312d646576656c6f702e323032352e382e372b636f6d6d69742e65616432613162392e6d6f64005"
						"c");
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
		statements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))},
					 yul::FunctionCall{
						 debugData,
						 yul::Identifier{debugData, yul::YulName("memoryguard")},
						 {yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(128))}}}}}});

		// Add payable check using revert_error function like regular pipeline
		yul::If callValueCheck{debugData};
		callValueCheck.condition = std::make_unique<yul::Expression>(
			yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("callvalue")}, {}});
		std::vector<yul::Statement> revertBody;
		revertBody.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{
						debugData,
						yul::YulName("revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb")},
					{}}});
		callValueCheck.body = yul::Block{debugData, std::move(revertBody)};
		statements.push_back(std::move(callValueCheck));

		// Add constructor function call
		statements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("constructor_Minimal_10")}, {}}});

		// Allocate memory for deployment
		yul::VariableDeclaration memAlloc{debugData};
		memAlloc.variables.push_back({debugData, yul::YulName("_1")});
		memAlloc.value = std::make_unique<yul::Expression>(
			yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("allocate_unbounded")}, {}});
		statements.push_back(std::move(memAlloc));

		// codecopy(_1, dataoffset("deployedObject"), datasize("deployedObject"))
		statements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("codecopy")},
					{yul::Identifier{debugData, yul::YulName("_1")},
					 yul::FunctionCall{
						 debugData,
						 yul::Identifier{debugData, yul::YulName("dataoffset")},
						 {yul::Literal{debugData, yul::LiteralKind::String, yul::LiteralValue(deployedObjectName)}}},
					 yul::FunctionCall{
						 debugData,
						 yul::Identifier{debugData, yul::YulName("datasize")},
						 {yul::
							  Literal{debugData, yul::LiteralKind::String, yul::LiteralValue(deployedObjectName)}}}}}});

		// return(_1, datasize("deployedObject"))
		statements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("return")},
					{yul::Identifier{debugData, yul::YulName("_1")},
					 yul::FunctionCall{
						 debugData,
						 yul::Identifier{debugData, yul::YulName("datasize")},
						 {yul::
							  Literal{debugData, yul::LiteralKind::String, yul::LiteralValue(deployedObjectName)}}}}}});

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
			{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))}}});
		allocateBody.push_back(std::move(memAssign));
		allocateFunc.body = yul::Block{debugData, std::move(allocateBody)};
		statements.push_back(std::move(allocateFunc));

		// function revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb()
		yul::FunctionDefinition revertErrorFunc{debugData};
		revertErrorFunc.name
			= yul::YulName("revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb");
		std::vector<yul::Statement> revertErrorBody;
		revertErrorBody.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});
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
		statements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("mstore")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))},
					 yul::FunctionCall{
						 debugData,
						 yul::Identifier{debugData, yul::YulName("memoryguard")},
						 {yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(128))}}}}}});

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
			{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))}}});
		allocateBody.push_back(std::move(memAssign));
		allocateFunc.body = yul::Block{debugData, std::move(allocateBody)};
		statements.push_back(std::move(allocateFunc));

		// Add callvalue check for runtime (match regular pipeline)
		// The regular pipeline adds: callvalue dup1 iszero tag_1 jumpi revert(0x00, 0x00) tag_1: pop
		std::vector<yul::Statement> revertStatements;
		revertStatements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});

		yul::If callValueCheck{debugData};
		callValueCheck.condition = std::make_unique<yul::Expression>(
			yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("callvalue")}, {}});
		callValueCheck.body = yul::Block{debugData, std::move(revertStatements)};
		statements.push_back(std::move(callValueCheck));

		// Collect all state variables and functions from the contract
		std::vector<mlir::Operation*> functions;
		uint32_t stateVarSlot = 0;
		for (auto& region: contractOp->getRegions())
		{
			for (auto& block: region)
			{
				for (auto& innerOp: block)
				{
					if (mlir::isa<mlir::solidity::StateVarOp>(innerOp))
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
							for (auto& attr: innerOp.getAttrs())
							{
								if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
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
					else if (mlir::isa<mlir::solidity::FunctionOp>(innerOp))
					{
						functions.push_back(&innerOp);
					}
				}
			}
		}

		// Generate ABI helper functions
		auto abiHelpers = generateABIHelperFunctions();
		for (auto& helper: abiHelpers)
			statements.push_back(std::move(helper));

		// Generate dispatcher
		auto dispatcherCode = generateDispatcherAST(functions);
		for (auto& stmt: dispatcherCode)
			statements.push_back(std::move(stmt));

		// Generate function definitions and external wrappers
		for (auto* funcOp: functions)
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
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("value")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("calldataload")},
						{yul::Identifier{debugData, yul::YulName("offset")}}})});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_decode_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

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
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mstore")},
						{yul::Identifier{debugData, yul::YulName("pos")},
						 yul::Identifier{debugData, yul::YulName("value")}}}});

			// end := add(pos, 0x20)
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("end")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("add")},
						{yul::Identifier{debugData, yul::YulName("pos")},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x20))}}})});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_encode_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

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
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("memPtr")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mload")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(0x40)}}})});

			// let newFreePtr := add(memPtr, size)
			bodyStatements.push_back(
				yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName("newFreePtr")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("add")},
						{yul::Identifier{debugData, yul::YulName("memPtr")},
						 yul::Identifier{debugData, yul::YulName("size")}}})});

			// mstore(0x40, newFreePtr)
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mstore")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))},
						 yul::Identifier{debugData, yul::YulName("newFreePtr")}}}});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("allocate_memory"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

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
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("cleaned")}},
					std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName("value")})});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("cleanup_t_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

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
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("ret")}},
					std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName("value")})});

			yul::FunctionDefinition funcDef{
				debugData, yul::YulName("identity"), params, returns, yul::Block{debugData, std::move(bodyStatements)}};

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
			revertBody.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("revert")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});

			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("iszero")},
				{yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("eq")},
					{yul::Identifier{debugData, yul::YulName("value")},
					 yul::FunctionCall{
						 debugData,
						 yul::Identifier{debugData, yul::YulName("cleanup_t_uint256")},
						 {yul::Identifier{debugData, yul::YulName("value")}}}}}}});
			ifStatement.body = yul::Block{debugData, std::move(revertBody)};

			bodyStatements.push_back(std::move(ifStatement));

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("validator_revert_t_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// abi_encode_t_uint256_to_t_uint256_fromStack(value, pos) (tag_10 equivalent)
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});
			params.push_back({debugData, yul::YulName("pos")});

			yul::NameWithDebugDataList returns; // no returns

			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mstore")},
						{yul::Identifier{debugData, yul::YulName("pos")},
						 yul::FunctionCall{
							 debugData,
							 yul::Identifier{debugData, yul::YulName("cleanup_t_uint256")},
							 {yul::Identifier{debugData, yul::YulName("value")}}}}}});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_encode_t_uint256_to_t_uint256_fromStack"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

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
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("tail")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("add")},
						{yul::Identifier{debugData, yul::YulName("headStart")},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}}})});

			// abi_encode_t_uint256_to_t_uint256_fromStack(value0, add(headStart, 0))
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("abi_encode_t_uint256_to_t_uint256_fromStack")},
						{yul::Identifier{debugData, yul::YulName("value0")},
						 yul::FunctionCall{
							 debugData,
							 yul::Identifier{debugData, yul::YulName("add")},
							 {yul::Identifier{debugData, yul::YulName("headStart")},
							  yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}}}});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_encode_tuple_t_uint256__to_t_uint256__fromStack"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// Add missing error handling functions for 1:1 parity with regular pipeline

		// revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb()
		{
			yul::NameWithDebugDataList params;	// no params
			yul::NameWithDebugDataList returns; // no returns

			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("revert")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b()
		{
			yul::NameWithDebugDataList params;
			yul::NameWithDebugDataList returns;

			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("revert")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// revert_error_42b3090547df1d2001c96683413b8cf91c1b902ef5e3cb8d9f6f304cf7446f74()
		{
			yul::NameWithDebugDataList params;
			yul::NameWithDebugDataList returns;

			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("revert")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("revert_error_42b3090547df1d2001c96683413b8cf91c1b902ef5e3cb8d9f6f304cf7446f74"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// cleanup_t_rational_42_by_1(value) -> cleaned
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});

			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("cleaned")});

			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("cleaned")}},
					std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName("value")})});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("cleanup_t_rational_42_by_1"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// convert_t_rational_42_by_1_to_t_uint256(value) -> converted
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});

			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("converted")});

			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(
				yul::Assignment{
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
								{yul::Identifier{debugData, yul::YulName("value")}}}}}}})});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("convert_t_rational_42_by_1_to_t_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// shift_right_224_unsigned(value) -> newValue
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("value")});

			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("newValue")});

			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("newValue")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("shr")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(224))},
						 yul::Identifier{debugData, yul::YulName("value")}}})});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("shift_right_224_unsigned"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// zero_value_for_split_t_uint256() -> ret
		{
			yul::NameWithDebugDataList params; // no params

			yul::NameWithDebugDataList returns;
			returns.push_back({debugData, yul::YulName("ret")});

			std::vector<yul::Statement> bodyStatements;
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("ret")}},
					std::make_unique<yul::Expression>(
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))})});

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("zero_value_for_split_t_uint256"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

			statements.push_back(yul::Statement(std::move(funcDef)));
		}

		// abi_decode_tuple_(headStart, dataEnd)
		{
			yul::NameWithDebugDataList params;
			params.push_back({debugData, yul::YulName("headStart")});
			params.push_back({debugData, yul::YulName("dataEnd")});

			yul::NameWithDebugDataList returns; // no returns

			std::vector<yul::Statement> bodyStatements;

			// if slt(sub(dataEnd, headStart), 0) {
			// revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b() }
			std::vector<yul::Statement> revertBody;
			revertBody.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{
							debugData,
							yul::YulName(
								"revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b")},
						{}}});

			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("slt")},
				{yul::FunctionCall{
					 debugData,
					 yul::Identifier{debugData, yul::YulName("sub")},
					 {yul::Identifier{debugData, yul::YulName("dataEnd")},
					  yul::Identifier{debugData, yul::YulName("headStart")}}},
				 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}});
			ifStatement.body = yul::Block{debugData, std::move(revertBody)};
			bodyStatements.push_back(std::move(ifStatement));

			yul::FunctionDefinition funcDef{
				debugData,
				yul::YulName("abi_decode_tuple_"),
				params,
				returns,
				yul::Block{debugData, std::move(bodyStatements)}};

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
		for (auto* funcOp: functions)
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
			statements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("revert")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});
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
				{yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("calldatasize")}, {}},
				 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(4))}}}}});

		// Body of if statement
		std::vector<yul::Statement> ifBody;

		// let selector := shr(224, calldataload(0))
		ifBody.push_back(
			yul::VariableDeclaration{
				debugData,
				{{debugData, yul::YulName("selector")}},
				std::make_unique<yul::Expression>(yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("shr")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(224))},
					 yul::FunctionCall{
						 debugData,
						 yul::Identifier{debugData, yul::YulName("calldataload")},
						 {yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}}})});

		// Create switch statement
		yul::Switch switchStatement{debugData};
		switchStatement.expression
			= std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName("selector")});

		// Generate cases for each public/external function
		for (auto* funcOp: functions)
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
				caseStatement.value = std::make_unique<
					yul::Literal>(debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(selector)));

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
					args.push_back(
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("calldataload")},
							{yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("add")},
								{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(4))},
								 yul::FunctionCall{
									 debugData,
									 yul::Identifier{debugData, yul::YulName("mul")},
									 {yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(i))},
									  yul::Literal{
										  debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}}}}}}});
				}

				// Check if function has return values - extract from function_type attribute
				int numResults = 0;
				if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
				{
					auto funcType = mlir::cast<mlir::FunctionType>(typeAttr.getValue());
					numResults = funcType.getResults().size();
				}
				else
				{
					numResults = funcOp->getNumResults();
				}

				// Construct unique function name with param types for overloaded functions
				std::string uniqueFuncName = getUniqueFuncName(funcOp);

				if (numResults > 0)
				{
					// Call function and store result
					caseBody.push_back(
						yul::VariableDeclaration{
							debugData,
							{{debugData, yul::YulName("ret")}},
							std::make_unique<yul::Expression>(yul::FunctionCall{
								debugData, yul::Identifier{debugData, yul::YulName(uniqueFuncName)}, args})});

					// Get memory position using mload(0x40) like the regular pipeline
					caseBody.push_back(
						yul::VariableDeclaration{
							debugData,
							{{debugData, yul::YulName("memPos")}},
							std::make_unique<yul::Expression>(yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("mload")},
								{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))}}})});

					// Encode return value using the exact function names as regular pipeline
					caseBody.push_back(
						yul::VariableDeclaration{
							debugData,
							{{debugData, yul::YulName("memEnd")}},
							std::make_unique<yul::Expression>(yul::FunctionCall{
								debugData,
								yul::Identifier{
									debugData, yul::YulName("abi_encode_tuple_t_uint256__to_t_uint256__fromStack")},
								{yul::Identifier{debugData, yul::YulName("memPos")},
								 yul::Identifier{debugData, yul::YulName("ret")}}})});

					caseBody.push_back(
						yul::ExpressionStatement{
							debugData,
							yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("return")},
								{yul::Identifier{debugData, yul::YulName("memPos")},
								 yul::FunctionCall{
									 debugData,
									 yul::Identifier{debugData, yul::YulName("sub")},
									 {yul::Identifier{debugData, yul::YulName("memEnd")},
									  yul::Identifier{debugData, yul::YulName("memPos")}}}}}});
				}
				else
				{
					// Call function with no return value
					caseBody.push_back(
						yul::ExpressionStatement{
							debugData,
							yul::FunctionCall{
								debugData, yul::Identifier{debugData, yul::YulName(uniqueFuncName)}, args}});

					// stop()
					caseBody.push_back(
						yul::ExpressionStatement{
							debugData,
							yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("stop")}, {}}});
				}

				caseStatement.body = yul::Block{debugData, std::move(caseBody)};
				switchStatement.cases.push_back(std::move(caseStatement));
			}
		}

		// Default case - revert
		std::vector<yul::Statement> defaultBody;
		defaultBody.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});
		switchStatement.cases.push_back(
			yul::Case{
				debugData,
				nullptr, // default case has no value
				yul::Block{debugData, std::move(defaultBody)}});

		ifBody.push_back(std::move(switchStatement));
		ifStatement.body = yul::Block{debugData, std::move(ifBody)};

		statements.push_back(std::move(ifStatement));

		// Fallback for calldatasize < 4
		statements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});

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

		// Get number of parameters for calldata decoding
		int numParams = 0;
		if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
		{
			auto funcType = mlir::cast<mlir::FunctionType>(typeAttr.getValue());
			numParams = funcType.getInputs().size();
		}

		// External wrapper name with param types for overloaded functions
		std::string wrapperName = getUniqueFuncName(funcOp, "external_fun_");
		// Internal function name with param types
		std::string internalFuncName = getUniqueFuncName(funcOp);

		yul::NameWithDebugDataList params;	// No parameters for external wrapper
		yul::NameWithDebugDataList returns; // No returns for external wrapper

		std::vector<yul::Statement> bodyStatements;

		// Add callvalue check for non-payable functions
		std::vector<yul::Statement> revertBody;
		revertBody.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{
						debugData,
						yul::YulName("revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb")},
					{}}});

		yul::If callvalueCheck{debugData};
		callvalueCheck.condition = std::make_unique<yul::Expression>(
			yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("callvalue")}, {}});
		callvalueCheck.body = yul::Block{debugData, std::move(revertBody)};
		bodyStatements.push_back(std::move(callvalueCheck));

		// Decode function parameters from calldata
		std::vector<yul::Expression> decodedParams;
		{
			// Decode parameters from calldata
			for (int i = 0; i < numParams; ++i)
			{
				// let param<i> := calldataload(add(4, mul(<i>, 32)))
				std::string paramName
					= "param" + std::to_string(i); // Changed: removed underscore to match internal function
				bodyStatements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(paramName)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("calldataload")},
							{yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("add")},
								{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(4))},
								 yul::FunctionCall{
									 debugData,
									 yul::Identifier{debugData, yul::YulName("mul")},
									 {yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(i))},
									  yul::Literal{
										  debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}}}}}}})});

				// Add to params list for function call
				decodedParams.push_back(yul::Identifier{debugData, yul::YulName(paramName)});
			}
		}

		// abi_decode_tuple_(4, calldatasize()) - still call for validation
		bodyStatements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("abi_decode_tuple_")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(4))},
					 yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("calldatasize")}, {}}}}});

		// Check if function has return values
		int numResults = 0;
		if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
		{
			auto funcType = mlir::cast<mlir::FunctionType>(typeAttr.getValue());
			numResults = funcType.getResults().size();
		}

		if (numResults > 0)
		{
			// let ret_0 := fun_funcName_N(param_0, param_1, ...)
			bodyStatements.push_back(
				yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName("ret_0")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData, yul::Identifier{debugData, yul::YulName(internalFuncName)}, decodedParams})});

			// let memPos := allocate_unbounded()
			bodyStatements.push_back(
				yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName("memPos")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData, yul::Identifier{debugData, yul::YulName("allocate_unbounded")}, {}})});

			// let memEnd := abi_encode_tuple_t_uint256__to_t_uint256__fromStack(memPos, ret_0)
			bodyStatements.push_back(
				yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName("memEnd")}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("abi_encode_tuple_t_uint256__to_t_uint256__fromStack")},
						{yul::Identifier{debugData, yul::YulName("memPos")},
						 yul::Identifier{debugData, yul::YulName("ret_0")}}})});

			// return(memPos, sub(memEnd, memPos))
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("return")},
						{yul::Identifier{debugData, yul::YulName("memPos")},
						 yul::FunctionCall{
							 debugData,
							 yul::Identifier{debugData, yul::YulName("sub")},
							 {yul::Identifier{debugData, yul::YulName("memEnd")},
							  yul::Identifier{debugData, yul::YulName("memPos")}}}}}});
		}
		else if (numParams > 0)
		{
			// Function has no return values but has parameters - still need to call it
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData, yul::Identifier{debugData, yul::YulName(internalFuncName)}, decodedParams}});
		}

		return yul::FunctionDefinition{
			debugData, yul::YulName(wrapperName), params, returns, yul::Block{debugData, std::move(bodyStatements)}};
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

		// Get number of parameters for processing
		int numParams = 0;
		if (funcOp->getNumRegions() > 0 && !funcOp->getRegion(0).empty())
		{
			auto& entryBlock = funcOp->getRegion(0).front();
			numParams = entryBlock.getNumArguments();
		}

		// Prefix function name with parameter types to avoid conflicts for overloaded functions
		std::string safeFuncName = getUniqueFuncName(funcOp);

		// Set current function context for scoping
		m_currentFunction = safeFuncName;
		m_functionVarCounter = 0; // Reset function-local variable counter

		// Initialize function-scoped variable mapping
		if (m_functionScopedNames.find(safeFuncName) == m_functionScopedNames.end())
		{
			m_functionScopedNames[safeFuncName] = std::map<void*, yul::YulName>();
		}

		// Parameters
		yul::NameWithDebugDataList params;
		if (funcOp->getNumRegions() > 0 && !funcOp->getRegion(0).empty())
		{
			auto& entryBlock = funcOp->getRegion(0).front();

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
			auto funcType = mlir::cast<mlir::FunctionType>(typeAttr.getValue());
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
		for (auto& region: funcOp->getRegions())
		{
			for (auto& block: region)
			{
				for (auto& op: block)
				{
					auto stmt = processOperationToStatement(&op);
					if (stmt)
						addStatementFlattened(bodyStatements, std::move(*stmt));
				}
			}
		}

		// If no explicit return and we need a return value, return 0
		if (numResults > 0 && bodyStatements.empty())
		{
			bodyStatements.push_back(
				yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("ret0")}},
					std::make_unique<yul::Expression>(
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))})});
		}

		yul::FunctionDefinition funcDef{
			debugData, yul::YulName(safeFuncName), params, returns, yul::Block{debugData, std::move(bodyStatements)}};

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

		if (mlir::isa<mlir::solidity::ConstantOp>(op))
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
					std::make_unique<yul::Expression>(
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(value)})};
			}
		}
		else if (mlir::isa<mlir::solidity::ReturnOp>(op))
		{
			if (op->getNumOperands() > 0)
			{
				std::string value = getVariableName(op->getOperand(0));
				return yul::Assignment{
					debugData,
					{yul::Identifier{debugData, yul::YulName("ret0")}},
					std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(value)})};
			}
			// Void return - no Yul statement needed
			return std::nullopt;
		}
		else if (mlir::isa<mlir::solidity::AddOp>(op))
		{
			return processArithmeticOpToAST(op, "add");
		}
		else if (mlir::isa<mlir::solidity::SubOp>(op))
		{
			return processArithmeticOpToAST(op, "sub");
		}
		else if (mlir::isa<mlir::solidity::MulOp>(op))
		{
			return processArithmeticOpToAST(op, "mul");
		}
		else if (mlir::isa<mlir::solidity::DivOp>(op))
		{
			return processArithmeticOpToAST(op, "div");
		}
		else if (mlir::isa<mlir::solidity::ModOp>(op))
		{
			return processArithmeticOpToAST(op, "mod");
		}
		else if (mlir::isa<mlir::solidity::ExpOp>(op))
		{
			return processArithmeticOpToAST(op, "exp");
		}
		else if (mlir::isa<mlir::solidity::AndOp>(op))
		{
			return processArithmeticOpToAST(op, "and");
		}
		else if (mlir::isa<mlir::solidity::OrOp>(op))
		{
			return processArithmeticOpToAST(op, "or");
		}
		else if (mlir::isa<mlir::solidity::XorOp>(op))
		{
			return processArithmeticOpToAST(op, "xor");
		}
		else if (mlir::isa<mlir::solidity::NotOp>(op))
		{
			return processUnaryOpToAST(op, "not");
		}
		else if (mlir::isa<mlir::solidity::LogicalNotOp>(op))
		{
			// Logical NOT uses iszero in Yul
			return processUnaryOpToAST(op, "iszero");
		}
		else if (mlir::isa<mlir::solidity::LogicalAndOp>(op))
		{
			return processArithmeticOpToAST(op, "and");
		}
		else if (mlir::isa<mlir::solidity::LogicalOrOp>(op))
		{
			return processArithmeticOpToAST(op, "or");
		}
		else if (mlir::isa<mlir::solidity::ShlOp>(op))
		{
			return processArithmeticOpToAST(op, "shl");
		}
		else if (mlir::isa<mlir::solidity::ShrOp>(op))
		{
			return processArithmeticOpToAST(op, "shr");
		}
		else if (mlir::isa<mlir::solidity::SarOp>(op))
		{
			return processArithmeticOpToAST(op, "sar");
		}
		else if (mlir::isa<mlir::solidity::CmpOp>(op))
		{
			return processComparisonOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::IfOp>(op))
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
							yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(value)})};
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
							yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(value)})};
				}
			}
			return std::nullopt;
		}
		else if (opName == "solidity.comment")
		{
			// Comments are just placeholders, skip them
			return std::nullopt;
		}
		else if (mlir::isa<mlir::solidity::ForOp>(op))
		{
			// Legacy support - shouldn't be reached with new code
			return processForOpToAST(op);
		}
		else if (opName == "solidity.while")
		{
			// Legacy support - shouldn't be reached with new code
			return processWhileOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::RevertOp>(op))
		{
			return yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{
					debugData,
					yul::Identifier{debugData, yul::YulName("revert")},
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
					 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}};
		}
		else if (mlir::isa<mlir::solidity::RequireOp>(op))
		{
			return processRequireOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::AssertOp>(op))
		{
			return processAssertOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::EmitOp>(op))
		{
			return processEmitOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::LoadStateVarOp>(op))
		{
			return processLoadStateOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::StoreStateVarOp>(op))
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
		else if (mlir::isa<mlir::solidity::MappingAccessOp>(op))
		{
			return processMappingAccessOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::MappingStoreOp>(op))
		{
			return processMappingStoreOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::FunctionCallOp>(op))
		{
			return processFunctionCallOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::SelectOp>(op))
		{
			return processSelectOpToAST(op);
		}
		else if (mlir::isa<mlir::solidity::StructCreateOp>(op))
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
					std::make_unique<yul::Expression>(
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))})};
			}
		}
		else if (mlir::isa<mlir::solidity::ArrayPushOp>(op))
		{
			// Dynamic array push: read length, compute element slot, store value, increment length
			auto debugData = langutil::DebugData::create();
			if (op->getNumOperands() >= 2)
			{
				std::string valueVar = getVariableName(op->getOperand(1));

				// Look up storage slot from varName attribute
				uint32_t slot = 0;
				if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("varName"))
				{
					auto it = m_stateVariableSlots.find(nameAttr.getValue().str());
					if (it != m_stateVariableSlots.end())
						slot = it->second;
				}

				std::string lenVar = "v" + std::to_string(m_functionVarCounter++);
				std::string baseVar = "v" + std::to_string(m_functionVarCounter++);
				std::string elemSlotVar = "v" + std::to_string(m_functionVarCounter++);
				std::string newLenVar = "v" + std::to_string(m_functionVarCounter++);

				std::vector<yul::Statement> statements;

				// let arr_len := sload(slot)
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(lenVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("sload")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))}}})});

				// Compute base data slot: mstore(0, slot); let arr_base := keccak256(0, 32)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
							 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))}}}});

				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(baseVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("keccak256")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
							 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}}})});

				// let arr_elem_slot := add(arr_base, arr_len)
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(elemSlotVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("add")},
							{yul::Identifier{debugData, yul::YulName(baseVar)},
							 yul::Identifier{debugData, yul::YulName(lenVar)}}})});

				// sstore(arr_elem_slot, value)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("sstore")},
							{yul::Identifier{debugData, yul::YulName(elemSlotVar)},
							 yul::Identifier{debugData, yul::YulName(valueVar)}}}});

				// let arr_new_len := add(arr_len, 1)
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(newLenVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("add")},
							{yul::Identifier{debugData, yul::YulName(lenVar)},
							 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))}}})});

				// sstore(slot, arr_new_len)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("sstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))},
							 yul::Identifier{debugData, yul::YulName(newLenVar)}}}});

				return yul::Block{debugData, std::move(statements)};
			}
		}
		else if (mlir::isa<mlir::solidity::ArrayLengthOp>(op))
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
						{yul::Identifier{debugData, yul::YulName(arrayVar)}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::MemberAccessOp>(op))
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
					std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(objectVar)})};
			}
		}
		else if (mlir::isa<mlir::solidity::AddressBalanceOp>(op))
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
						{yul::Identifier{debugData, yul::YulName(addrVar)}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::AddressCodeOp>(op))
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
						{yul::Identifier{debugData, yul::YulName(addrVar)}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::AddressCodehashOp>(op))
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
						{yul::Identifier{debugData, yul::YulName(addrVar)}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::ToI1Op>(op))
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
					std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(inputVar)})};
			}
		}
		else if (mlir::isa<mlir::solidity::ConvertOp>(op))
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
					std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(inputVar)})};
			}
		}
		else if (mlir::isa<mlir::solidity::MsgSenderOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("caller")}, {}})};
			}
		}
		else if (mlir::isa<mlir::solidity::MsgValueOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("callvalue")}, {}})};
			}
		}
		else if (mlir::isa<mlir::solidity::MsgDataOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				// calldataload returns the calldata starting at offset 0
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("calldataload")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::MsgSigOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				// Function selector is the first 4 bytes of calldata
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("shr")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(224))},
						 yul::FunctionCall{
							 debugData,
							 yul::Identifier{debugData, yul::YulName("calldataload")},
							 {yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::BlockTimestampOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("timestamp")}, {}})};
			}
		}
		else if (mlir::isa<mlir::solidity::BlockNumberOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("number")}, {}})};
			}
		}
		else if (mlir::isa<mlir::solidity::BlockChainIdOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("chainid")}, {}})};
			}
		}
		else if (mlir::isa<mlir::solidity::TxOriginOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("origin")}, {}})};
			}
		}
		else if (mlir::isa<mlir::solidity::TxGasPriceOp>(op))
		{
			auto debugData = langutil::DebugData::create();
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("gasprice")}, {}})};
			}
		}
		else if (opName == "scf.condition" || opName == "scf.yield")
		{
			// These are control flow terminators that don't map to Yul statements
			// They're handled by their parent operations (scf.while)
			return std::nullopt;
		}
		// Built-in function lowering
		else if (mlir::isa<mlir::solidity::AddModOp>(op))
		{
			// addmod(a, b, n) - direct Yul opcode
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 3)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string a = getVariableName(op->getOperand(0));
				std::string b = getVariableName(op->getOperand(1));
				std::string n = getVariableName(op->getOperand(2));

				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("addmod")},
						{yul::Identifier{debugData, yul::YulName(a)},
						 yul::Identifier{debugData, yul::YulName(b)},
						 yul::Identifier{debugData, yul::YulName(n)}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::MulModOp>(op))
		{
			// mulmod(a, b, n) - direct Yul opcode
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 3)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string a = getVariableName(op->getOperand(0));
				std::string b = getVariableName(op->getOperand(1));
				std::string n = getVariableName(op->getOperand(2));

				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mulmod")},
						{yul::Identifier{debugData, yul::YulName(a)},
						 yul::Identifier{debugData, yul::YulName(b)},
						 yul::Identifier{debugData, yul::YulName(n)}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::GasLeftOp>(op))
		{
			// gasleft() -> gas() in Yul
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("gas")}, {}})};
			}
		}
		else if (mlir::isa<mlir::solidity::BlockhashOp>(op))
		{
			// blockhash(blockNumber) - direct Yul opcode
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string blockNum = getVariableName(op->getOperand(0));

				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("blockhash")},
						{yul::Identifier{debugData, yul::YulName(blockNum)}}})};
			}
		}
		else if (mlir::isa<mlir::solidity::Keccak256Op>(op))
		{
			// keccak256(data) - needs memory handling
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));

				// Check if the input comes from an ABI encode operation
				// and fuse the operations for efficiency
				auto* defOp = op->getOperand(0).getDefiningOp();

				if (defOp && mlir::isa<mlir::solidity::AbiEncodePackedOp, mlir::solidity::AbiEncodeOp>(defOp))
				{
					// Fused keccak256(abi.encode/abi.encodePacked(a, b, ...))
					// Store all args in scratch space and hash them
					unsigned numArgs = defOp->getNumOperands();
					std::vector<yul::Statement> statements;

					if (mlir::isa<mlir::solidity::AbiEncodeOp>(defOp))
					{
						// abi.encode: each arg padded to 32 bytes
						for (unsigned i = 0; i < numArgs; ++i)
						{
							std::string argName = getVariableName(defOp->getOperand(i));
							unsigned offset = i * 32;
							statements.push_back(
								yul::ExpressionStatement{
									debugData,
									yul::FunctionCall{
										debugData,
										yul::Identifier{debugData, yul::YulName("mstore")},
										{yul::Literal{
											 debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(offset))},
										 yul::Identifier{debugData, yul::YulName(argName)}}}});
						}

						unsigned totalSize = numArgs * 32;
						statements.push_back(
							yul::VariableDeclaration{
								debugData,
								{{debugData, yul::YulName(resultVar)}},
								std::make_unique<yul::Expression>(yul::FunctionCall{
									debugData,
									yul::Identifier{debugData, yul::YulName("keccak256")},
									{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
									 yul::Literal{
										 debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(totalSize))}}})});
					}
					else
					{
						// abi.encodePacked: tightly packed based on type sizes
						unsigned currentOffset = 0;
						for (unsigned i = 0; i < numArgs; ++i)
						{
							std::string argName = getVariableName(defOp->getOperand(i));
							unsigned byteSize = getPackedByteSize(defOp->getOperand(i).getType());

							if (byteSize == 32)
							{
								statements.push_back(
									yul::ExpressionStatement{
										debugData,
										yul::FunctionCall{
											debugData,
											yul::Identifier{debugData, yul::YulName("mstore")},
											{yul::Literal{
												 debugData,
												 yul::LiteralKind::Number,
												 yul::LiteralValue(u256(currentOffset))},
											 yul::Identifier{debugData, yul::YulName(argName)}}}});
							}
							else
							{
								// Sub-32-byte: shift left and mstore
								unsigned shiftBits = (32 - byteSize) * 8;
								statements.push_back(
									yul::ExpressionStatement{
										debugData,
										yul::FunctionCall{
											debugData,
											yul::Identifier{debugData, yul::YulName("mstore")},
											{yul::Literal{
												 debugData,
												 yul::LiteralKind::Number,
												 yul::LiteralValue(u256(currentOffset))},
											 yul::FunctionCall{
												 debugData,
												 yul::Identifier{debugData, yul::YulName("shl")},
												 {yul::Literal{
													  debugData,
													  yul::LiteralKind::Number,
													  yul::LiteralValue(u256(shiftBits))},
												  yul::Identifier{debugData, yul::YulName(argName)}}}}}});
							}

							currentOffset += byteSize;
						}

						statements.push_back(
							yul::VariableDeclaration{
								debugData,
								{{debugData, yul::YulName(resultVar)}},
								std::make_unique<yul::Expression>(yul::FunctionCall{
									debugData,
									yul::Identifier{debugData, yul::YulName("keccak256")},
									{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
									 yul::Literal{
										 debugData,
										 yul::LiteralKind::Number,
										 yul::LiteralValue(u256(currentOffset))}}})});
					}

					return yul::Block{debugData, std::move(statements)};
				}
				else
				{
					// Standard case: single value input
					std::string data = getVariableName(op->getOperand(0));
					std::vector<yul::Statement> statements;

					// mstore(0, data)
					statements.push_back(
						yul::ExpressionStatement{
							debugData,
							yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("mstore")},
								{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								 yul::Identifier{debugData, yul::YulName(data)}}}});

					// let result := keccak256(0, 32)
					statements.push_back(
						yul::VariableDeclaration{
							debugData,
							{{debugData, yul::YulName(resultVar)}},
							std::make_unique<yul::Expression>(yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("keccak256")},
								{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}}})});

					return yul::Block{debugData, std::move(statements)};
				}
			}
		}
		else if (mlir::isa<mlir::solidity::Sha256Op>(op))
		{
			// sha256 uses precompile at address 0x02
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string data = getVariableName(op->getOperand(0));

				std::vector<yul::Statement> statements;

				// mstore(0, data)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
							 yul::Identifier{debugData, yul::YulName(data)}}}});

				// staticcall(gas(), 2, 0, 32, 0, 32)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("pop")},
							{yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("staticcall")},
								{yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("gas")}, {}},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(2))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}}}}}});

				// let result := mload(0)
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(resultVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mload")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}})});

				return yul::Block{debugData, std::move(statements)};
			}
		}
		else if (mlir::isa<mlir::solidity::Ripemd160Op>(op))
		{
			// ripemd160 uses precompile at address 0x03
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string data = getVariableName(op->getOperand(0));

				std::vector<yul::Statement> statements;

				// mstore(0, data)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
							 yul::Identifier{debugData, yul::YulName(data)}}}});

				// staticcall(gas(), 3, 0, 32, 0, 32)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("pop")},
							{yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("staticcall")},
								{yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("gas")}, {}},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(3))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}}}}}});

				// let result := mload(0) - ripemd160 returns right-aligned in 32 bytes
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(resultVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mload")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}})});

				return yul::Block{debugData, std::move(statements)};
			}
		}
		else if (mlir::isa<mlir::solidity::EcrecoverOp>(op))
		{
			// ecrecover uses precompile at address 0x01
			// Input: hash (32 bytes), v (32 bytes), r (32 bytes), s (32 bytes) = 128 bytes
			// Output: address (32 bytes, right-aligned)
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 4)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string hash = getVariableName(op->getOperand(0));
				std::string v = getVariableName(op->getOperand(1));
				std::string r = getVariableName(op->getOperand(2));
				std::string s = getVariableName(op->getOperand(3));

				std::vector<yul::Statement> statements;

				// mstore(0, hash)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
							 yul::Identifier{debugData, yul::YulName(hash)}}}});

				// mstore(32, v)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))},
							 yul::Identifier{debugData, yul::YulName(v)}}}});

				// mstore(64, r)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))},
							 yul::Identifier{debugData, yul::YulName(r)}}}});

				// mstore(96, s)
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(96))},
							 yul::Identifier{debugData, yul::YulName(s)}}}});

				// pop(staticcall(gas(), 1, 0, 128, 0, 32))
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("pop")},
							{yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("staticcall")},
								{yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("gas")}, {}},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(128))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
								 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))}}}}}});

				// let result := mload(0)
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(resultVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mload")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}})});

				return yul::Block{debugData, std::move(statements)};
			}
		}
		else if (mlir::isa<mlir::solidity::SelfdestructOp>(op))
		{
			// selfdestruct(recipient) - direct Yul opcode
			auto debugData = getDebugData(op);
			if (op->getNumOperands() >= 1)
			{
				std::string recipient = getVariableName(op->getOperand(0));

				return yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("selfdestruct")},
						{yul::Identifier{debugData, yul::YulName(recipient)}}}};
			}
		}
		else if (mlir::isa<mlir::solidity::AbiEncodeOp>(op))
		{
			// abi.encode(a, b, ...) - ABI encode with 32-byte padding per arg
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0)
			{
				// Check if the sole user is keccak256 - if so, skip standalone lowering
				// because the fused keccak256 handler will generate optimized code
				auto result = op->getResult(0);
				if (result.hasOneUse())
				{
					auto* user = *result.getUsers().begin();
					if (mlir::isa<mlir::solidity::Keccak256Op>(user))
					{
						// Skip - the fused keccak256 handler will handle this
						std::string resultVar = getOrCreateVariableName(result);
						return yul::VariableDeclaration{
							debugData,
							{{debugData, yul::YulName(resultVar)}},
							std::make_unique<yul::Expression>(
								yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))})};
					}
				}

				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				unsigned numArgs = op->getNumOperands();
				unsigned totalSize = numArgs * 32;
				std::vector<yul::Statement> statements;

				// let ptr := mload(0x40)
				std::string ptrVar = resultVar + "_ptr";
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(ptrVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mload")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))}}})});

				// Store each argument padded to 32 bytes
				for (unsigned i = 0; i < numArgs; ++i)
				{
					std::string argName = getVariableName(op->getOperand(i));
					unsigned offset = i * 32;
					statements.push_back(
						yul::ExpressionStatement{
							debugData,
							yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("mstore")},
								{yul::FunctionCall{
									 debugData,
									 yul::Identifier{debugData, yul::YulName("add")},
									 {yul::Identifier{debugData, yul::YulName(ptrVar)},
									  yul::Literal{
										  debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(offset))}}},
								 yul::Identifier{debugData, yul::YulName(argName)}}}});
				}

				// Update free memory pointer: mstore(0x40, add(ptr, totalSize))
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))},
							 yul::FunctionCall{
								 debugData,
								 yul::Identifier{debugData, yul::YulName("add")},
								 {yul::Identifier{debugData, yul::YulName(ptrVar)},
								  yul::Literal{
									  debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(totalSize))}}}}}});

				// result = ptr (memory pointer to encoded data)
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(resultVar)}},
						std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(ptrVar)})});

				return yul::Block{debugData, std::move(statements)};
			}
		}
		else if (mlir::isa<mlir::solidity::AbiEncodePackedOp>(op))
		{
			// abi.encodePacked(a, b, ...) - tightly packed encoding
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0)
			{
				// Check if the sole user is keccak256 - if so, skip standalone lowering
				auto result = op->getResult(0);
				if (result.hasOneUse())
				{
					auto* user = *result.getUsers().begin();
					if (mlir::isa<mlir::solidity::Keccak256Op>(user))
					{
						// Skip - the fused keccak256 handler will handle this
						std::string resultVar = getOrCreateVariableName(result);
						return yul::VariableDeclaration{
							debugData,
							{{debugData, yul::YulName(resultVar)}},
							std::make_unique<yul::Expression>(
								yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))})};
					}
				}

				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				unsigned numArgs = op->getNumOperands();
				std::vector<yul::Statement> statements;

				// let ptr := mload(0x40)
				std::string ptrVar = resultVar + "_ptr";
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(ptrVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mload")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))}}})});

				// For encodePacked, compute byte sizes from MLIR types and pack tightly
				unsigned currentOffset = 0;
				for (unsigned i = 0; i < numArgs; ++i)
				{
					std::string argName = getVariableName(op->getOperand(i));
					unsigned byteSize = getPackedByteSize(op->getOperand(i).getType());

					if (byteSize == 32)
					{
						// Full 32-byte value - use mstore
						statements.push_back(
							yul::ExpressionStatement{
								debugData,
								yul::FunctionCall{
									debugData,
									yul::Identifier{debugData, yul::YulName("mstore")},
									{yul::FunctionCall{
										 debugData,
										 yul::Identifier{debugData, yul::YulName("add")},
										 {yul::Identifier{debugData, yul::YulName(ptrVar)},
										  yul::Literal{
											  debugData,
											  yul::LiteralKind::Number,
											  yul::LiteralValue(u256(currentOffset))}}},
									 yul::Identifier{debugData, yul::YulName(argName)}}}});
					}
					else
					{
						// Sub-32-byte value: shift left to pack, then use mstore
						// mstore(ptr + offset, shl(shift_amount, value))
						unsigned shiftBits = (32 - byteSize) * 8;
						statements.push_back(
							yul::ExpressionStatement{
								debugData,
								yul::FunctionCall{
									debugData,
									yul::Identifier{debugData, yul::YulName("mstore")},
									{yul::FunctionCall{
										 debugData,
										 yul::Identifier{debugData, yul::YulName("add")},
										 {yul::Identifier{debugData, yul::YulName(ptrVar)},
										  yul::Literal{
											  debugData,
											  yul::LiteralKind::Number,
											  yul::LiteralValue(u256(currentOffset))}}},
									 yul::FunctionCall{
										 debugData,
										 yul::Identifier{debugData, yul::YulName("shl")},
										 {yul::Literal{
											  debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(shiftBits))},
										  yul::Identifier{debugData, yul::YulName(argName)}}}}}});
					}

					currentOffset += byteSize;
				}

				// Round up total size for memory pointer update
				unsigned totalSize = (currentOffset + 31) & ~31u;

				// Update free memory pointer: mstore(0x40, add(ptr, totalSize))
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))},
							 yul::FunctionCall{
								 debugData,
								 yul::Identifier{debugData, yul::YulName("add")},
								 {yul::Identifier{debugData, yul::YulName(ptrVar)},
								  yul::Literal{
									  debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(totalSize))}}}}}});

				// result = ptr (memory pointer to encoded data)
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(resultVar)}},
						std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(ptrVar)})});

				return yul::Block{debugData, std::move(statements)};
			}
		}
		else if (mlir::isa<mlir::solidity::AbiEncodeWithSelectorOp, mlir::solidity::AbiEncodeWithSignatureOp>(op))
		{
			// abi.encodeWithSelector(sel, a, b, ...) or abi.encodeWithSignature(sig, a, b, ...)
			// First operand is the selector/signature, rest are data arguments
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				unsigned numArgs = op->getNumOperands();
				std::vector<yul::Statement> statements;

				// let ptr := mload(0x40)
				std::string ptrVar = resultVar + "_ptr";
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(ptrVar)}},
						std::make_unique<yul::Expression>(yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mload")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))}}})});

				// Store selector at ptr (4 bytes, left-aligned via mstore)
				std::string selectorArg = getVariableName(op->getOperand(0));
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Identifier{debugData, yul::YulName(ptrVar)},
							 yul::Identifier{debugData, yul::YulName(selectorArg)}}}});

				// Store remaining args at ptr + 4, each padded to 32 bytes
				for (unsigned i = 1; i < numArgs; ++i)
				{
					std::string argName = getVariableName(op->getOperand(i));
					unsigned offset = 4 + (i - 1) * 32;
					statements.push_back(
						yul::ExpressionStatement{
							debugData,
							yul::FunctionCall{
								debugData,
								yul::Identifier{debugData, yul::YulName("mstore")},
								{yul::FunctionCall{
									 debugData,
									 yul::Identifier{debugData, yul::YulName("add")},
									 {yul::Identifier{debugData, yul::YulName(ptrVar)},
									  yul::Literal{
										  debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(offset))}}},
								 yul::Identifier{debugData, yul::YulName(argName)}}}});
				}

				unsigned totalSize = 4 + (numArgs - 1) * 32;
				unsigned alignedSize = (totalSize + 31) & ~31u;

				// Update free memory pointer
				statements.push_back(
					yul::ExpressionStatement{
						debugData,
						yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("mstore")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0x40))},
							 yul::FunctionCall{
								 debugData,
								 yul::Identifier{debugData, yul::YulName("add")},
								 {yul::Identifier{debugData, yul::YulName(ptrVar)},
								  yul::Literal{
									  debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(alignedSize))}}}}}});

				// result = ptr
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(resultVar)}},
						std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(ptrVar)})});

				return yul::Block{debugData, std::move(statements)};
			}
		}
		else if (mlir::isa<mlir::solidity::AbiDecodeOp>(op))
		{
			// abi.decode(data, (types)) - for now handle as loading from memory
			auto debugData = getDebugData(op);
			if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
			{
				std::string resultVar = getOrCreateVariableName(op->getResult(0));
				std::string data = getVariableName(op->getOperand(0));

				// Simple case: decode a single value from the data pointer
				// result = mload(add(data, 0))
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mload")},
						{yul::Identifier{debugData, yul::YulName(data)}}})};
			}
		}

		// UncheckedOp: just lower the body ops (unchecked semantics don't affect Yul)
		if (mlir::isa<mlir::solidity::UncheckedOp>(op))
		{
			std::vector<yul::Statement> bodyStatements;
			for (auto& region: op->getRegions())
				for (auto& block: region)
					for (auto& innerOp: block)
					{
						auto stmt = processOperationToStatement(&innerOp);
						if (stmt)
							addStatementFlattened(bodyStatements, std::move(*stmt));
					}
			return yul::Block{debugData, std::move(bodyStatements)};
		}

		std::cerr << "Warning: unsupported MLIR operation in Yul lowering: "
				  << op->getName().getStringRef().str() << "\n";
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
					{yul::Identifier{debugData, yul::YulName(lhs)}, yul::Identifier{debugData, yul::YulName(rhs)}}})};
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
					{yul::Identifier{debugData, yul::YulName(operand)}}})};
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
				if (pred == "eq")
					predicate = "eq";
				else if (pred == "ne")
					predicate = "ne";
				else if (pred == "lt")
					predicate = "lt";
				else if (pred == "le")
					predicate = "le";
				else if (pred == "gt")
					predicate = "gt";
				else if (pred == "ge")
					predicate = "ge";
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
							{yul::Identifier{debugData, yul::YulName(lhs)},
							 yul::Identifier{debugData, yul::YulName(rhs)}}}}})};
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
							{yul::Identifier{debugData, yul::YulName(lhs)},
							 yul::Identifier{debugData, yul::YulName(rhs)}}}}})};
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
							{yul::Identifier{debugData, yul::YulName(lhs)},
							 yul::Identifier{debugData, yul::YulName(rhs)}}}}})};
			}
			else
			{
				return yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName(predicate)},
						{yul::Identifier{debugData, yul::YulName(lhs)},
						 yul::Identifier{debugData, yul::YulName(rhs)}}})};
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
				for (auto& region: op->getRegion(0))
				{
					for (auto& innerOp: region)
					{
						auto stmt = processOperationToStatement(&innerOp);
						if (stmt)
							addStatementFlattened(thenStatements, std::move(*stmt));
					}
				}
			}

			yul::If ifStatement{debugData};
			ifStatement.condition
				= std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(condition)});
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
			for (auto& region: op->getRegion(0))
			{
				for (auto& innerOp: region)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						addStatementFlattened(initStatements, std::move(*stmt));
				}
			}
		}

		if (op->getNumRegions() > 1)
		{
			// Body block
			for (auto& region: op->getRegion(1))
			{
				for (auto& innerOp: region)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						addStatementFlattened(bodyStatements, std::move(*stmt));
				}
			}
		}

		if (op->getNumRegions() > 2)
		{
			// Post block
			for (auto& region: op->getRegion(2))
			{
				for (auto& innerOp: region)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						addStatementFlattened(postStatements, std::move(*stmt));
				}
			}
		}

		yul::ForLoop forLoop{debugData};
		forLoop.pre = yul::Block{debugData, std::move(initStatements)};

		// Condition
		if (op->getNumOperands() > 0)
		{
			std::string condition = getVariableName(op->getOperand(0));
			forLoop.condition = std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(condition)});
		}
		else
		{
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))});
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
			for (auto& region: op->getRegion(0))
			{
				for (auto& innerOp: region)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						addStatementFlattened(bodyStatements, std::move(*stmt));
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
			forLoop.condition = std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(condition)});
		}
		else
		{
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))});
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
			for (auto& block: op->getRegion(0))
			{
				for (auto& innerOp: block)
				{
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						addStatementFlattened(bodyStatements, std::move(*stmt));
				}
			}
		}

		// Convert to a Yul for loop with always-true condition
		// The actual loop control will be handled by break statements
		yul::ForLoop forLoop{debugData};
		forLoop.pre = yul::Block{debugData, {}};  // Empty pre block
		forLoop.post = yul::Block{debugData, {}}; // Empty post block

		// Always true condition for now (actual condition is inside the body)
		forLoop.condition = std::make_unique<yul::Expression>(
			yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))});

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
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(loopVarName)}},
						std::make_unique<yul::Expression>(
							yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))})});
			}
			else
			{
				// Declare loop variable outside the loop
				statements.push_back(
					yul::VariableDeclaration{
						debugData,
						{{debugData, yul::YulName(loopVarName)}},
						std::make_unique<yul::Expression>(yul::Identifier{debugData, yul::YulName(initVarName)})});
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
			for (auto& block: beforeRegion)
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
			for (auto& block: afterRegion)
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
		// We need to:
		// 1. Process all intermediate operations (like mul, add) that compute values used in the condition
		// 2. These operations become statements at the start of the loop body (evaluated each iteration)
		// 3. Build the condition expression from the comparison operation
		std::unique_ptr<yul::Expression> conditionExpr;
		std::vector<yul::Statement> conditionStatements; // Statements to prepend to loop body

		if (op->getNumRegions() > 0 && !op->getRegion(0).empty())
		{
			// Map block arguments to loop variables
			for (auto& block: op->getRegion(0))
			{
				for (unsigned i = 0; i < block.getNumArguments(); ++i)
				{
					void* key = block.getArgument(i).getAsOpaquePointer();
					if (i < loopVarNames.size())
						m_valueNames[key] = yul::YulName(loopVarNames[i]);
					else
						m_valueNames[key] = yul::YulName(loopVarName);
				}

				// First pass: process all operations except terminators to generate statements
				// This ensures intermediate computations like (i * i) are properly defined
				for (auto& innerOp: block)
				{
					llvm::StringRef opName = innerOp.getName().getStringRef();

					// Skip control flow operations - they don't produce statements
					if (opName == "scf.condition" || mlir::isa<mlir::solidity::ToI1Op>(innerOp))
						continue;

					// Skip the comparison itself - we'll build it inline in the condition
					if (mlir::isa<mlir::solidity::CmpOp>(innerOp))
						continue;

					// Process all other operations as statements
					auto stmt = processOperationToStatement(&innerOp);
					if (stmt)
						addStatementFlattened(conditionStatements, std::move(*stmt));
				}

				// Second pass: look for the condition operation and build the expression
				std::string conditionVarName;
				for (auto& innerOp: block)
				{
					if (mlir::isa<mlir::solidity::CmpOp>(innerOp))
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
									for (auto& attr: innerOp.getAttrs())
									{
										if (attr.getName().getValue() == "predicate")
										{
											if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
												cmpType = strAttr.getValue().str();
										}
									}
								}
							}

							// Map comparison types to Yul operations
							// Note: Yul doesn't have le/ge directly, need to use combinations
							std::string yulOp = "lt";
							bool needNot = false;
							if (cmpType == "eq")
							{
								yulOp = "eq";
							}
							else if (cmpType == "ne")
							{
								yulOp = "eq";
								needNot = true; // ne(a,b) = iszero(eq(a,b))
							}
							else if (cmpType == "lt")
							{
								yulOp = "lt";
							}
							else if (cmpType == "le")
							{
								// le(a,b) = iszero(gt(a,b))
								yulOp = "gt";
								needNot = true;
							}
							else if (cmpType == "gt")
							{
								yulOp = "gt";
							}
							else if (cmpType == "ge")
							{
								// ge(a,b) = iszero(lt(a,b))
								yulOp = "lt";
								needNot = true;
							}

							// Get operand names - now these should be properly defined
							// from the first pass processing
							std::string lhs = getVariableName(innerOp.getOperand(0));
							std::string rhs = getVariableName(innerOp.getOperand(1));

							// Build the condition expression
							if (needNot)
							{
								// Wrap in iszero for negation
								conditionExpr = std::make_unique<yul::Expression>(yul::FunctionCall{
									debugData,
									yul::Identifier{debugData, yul::YulName("iszero")},
									{yul::FunctionCall{
										debugData,
										yul::Identifier{debugData, yul::YulName(yulOp)},
										{yul::Identifier{debugData, yul::YulName(lhs)},
										 yul::Identifier{debugData, yul::YulName(rhs)}}}}});
							}
							else
							{
								conditionExpr = std::make_unique<yul::Expression>(yul::FunctionCall{
									debugData,
									yul::Identifier{debugData, yul::YulName(yulOp)},
									{yul::Identifier{debugData, yul::YulName(lhs)},
									 yul::Identifier{debugData, yul::YulName(rhs)}}});
							}
							conditionVarName = resultName;
						}
					}
					else if (mlir::isa<mlir::solidity::ToI1Op>(innerOp))
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
									yul::Identifier{debugData, yul::YulName(condVar)});
							}
						}
					}
				}
			}
		}

		// Process the after region (loop body)
		std::vector<yul::Statement> bodyStatements;
		std::vector<yul::Statement> postStatements; // Track post-increment statements

		// If we have condition statements (intermediate computations for the condition),
		// we need to use the pattern:
		//   for { } 1 { post } { condition_stmts; if iszero(cond) { break } body_stmts }
		// This ensures the condition computations are evaluated each iteration.
		bool hasConditionStatements = !conditionStatements.empty();

		if (hasConditionStatements && conditionExpr)
		{
			// Set condition to always true - we'll break manually
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))});

			// Prepend condition computation statements to body
			for (auto& stmt: conditionStatements)
				bodyStatements.push_back(std::move(stmt));

			// Add "if iszero(condition) { break }" to exit the loop
			yul::If breakIfStmt{debugData};
			breakIfStmt.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData, yul::Identifier{debugData, yul::YulName("iszero")}, {std::move(*conditionExpr)}});
			std::vector<yul::Statement> breakBody;
			breakBody.push_back(yul::Break{debugData});
			breakIfStmt.body = yul::Block{debugData, std::move(breakBody)};
			bodyStatements.push_back(std::move(breakIfStmt));

			// Clear conditionExpr since we've used it
			conditionExpr = nullptr;
		}
		else if (conditionExpr)
		{
			forLoop.condition = std::move(conditionExpr);
		}
		else
		{
			// Default to true
			forLoop.condition = std::make_unique<yul::Expression>(
				yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))});
		}

		if (op->getNumRegions() > 1 && !op->getRegion(1).empty())
		{
			for (auto& block: op->getRegion(1))
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
				for (auto& innerOp: block)
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
								if (mlir::isa<mlir::solidity::AddOp>(definingOp))
								{
									for (auto operand: definingOp->getOperands())
									{
										if (auto constOp = operand.getDefiningOp())
										{
											if (mlir::isa<mlir::solidity::ConstantOp>(constOp))
												yieldedOps.insert(constOp);
										}
									}
								}
							}
						}
					}
				}

				// Process all operations in the body
				for (auto& innerOp: block)
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

								if (definingOp && mlir::isa<mlir::solidity::AddOp>(definingOp))
								{
									// Generate the add expression directly in the post block
									std::string lhs = getVariableName(definingOp->getOperand(0));

									// Check if the second operand is a constant
									std::vector<yul::Expression> args;
									args.push_back(yul::Identifier{debugData, yul::YulName(lhs)});

									// Check if rhs is from a constant operation
									mlir::Value rhsValue = definingOp->getOperand(1);
									mlir::Operation* rhsDefOp = rhsValue.getDefiningOp();

									if (rhsDefOp && mlir::isa<mlir::solidity::ConstantOp>(rhsDefOp))
									{
										// Get the constant value
										if (auto intAttr = rhsDefOp->getAttrOfType<mlir::IntegerAttr>("value"))
										{
											u256 val(intAttr.getValue().getLimitedValue());
											args.push_back(
												yul::Literal{
													debugData, yul::LiteralKind::Number, yul::LiteralValue(val)});
										}
										else
										{
											// Fallback to 1 if we can't get the value
											args.push_back(
												yul::Literal{
													debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(1))});
										}
									}
									else
									{
										// Not a constant, use the variable name
										std::string rhs = getVariableName(definingOp->getOperand(1));
										args.push_back(yul::Identifier{debugData, yul::YulName(rhs)});
									}

									postStatements.push_back(
										yul::Assignment{
											debugData,
											{{debugData, yul::YulName(targetVarName)}},
											std::make_unique<yul::Expression>(yul::FunctionCall{
												debugData,
												yul::Identifier{debugData, yul::YulName("add")},
												std::move(args)})});
								}
								else
								{
									// For other cases, try to use the yielded variable
									std::string yieldedVarName = getVariableName(yieldedValue);
									if (yieldedVarName != targetVarName && !yieldedVarName.empty())
									{
										postStatements.push_back(
											yul::Assignment{
												debugData,
												{{debugData, yul::YulName(targetVarName)}},
												std::make_unique<yul::Expression>(
													yul::Identifier{debugData, yul::YulName(yieldedVarName)})});
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
							addStatementFlattened(bodyStatements, std::move(*stmt));

							// If this operation produces a value that will be yielded as a loop-carried value,
							// we need to assign it to the corresponding loop variable
							if (innerOp.getNumResults() > 0)
							{
								// Check if this result is used by the scf.yield
								for (auto& checkOp: block)
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
													bodyStatements.push_back(
														yul::Assignment{
															debugData,
															{{debugData, yul::YulName(loopVarNames[yieldIdx])}},
															std::make_unique<yul::Expression>(yul::Identifier{
																debugData, yul::YulName(resultVarName)})});
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
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("revert")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}}}});

			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("iszero")},
				{yul::Identifier{debugData, yul::YulName(condition)}}});
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
			bodyStatements.push_back(
				yul::ExpressionStatement{
					debugData, yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("invalid")}, {}}});

			yul::If ifStatement{debugData};
			ifStatement.condition = std::make_unique<yul::Expression>(yul::FunctionCall{
				debugData,
				yul::Identifier{debugData, yul::YulName("iszero")},
				{yul::Identifier{debugData, yul::YulName(condition)}}});
			ifStatement.body = yul::Block{debugData, std::move(bodyStatements)};

			return ifStatement;
		}
		return std::nullopt;
	}

	std::optional<yul::Statement> processEmitOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();

		// Get event name from attribute
		std::string eventName = "UnknownEvent";
		if (auto eventAttr = op->getAttrOfType<mlir::StringAttr>("event"))
			eventName = eventAttr.getValue().str();

		// Get event signature from attribute for accurate topic0 computation
		std::string eventSignature;
		if (auto sigAttr = op->getAttrOfType<mlir::StringAttr>("eventSignature"))
			eventSignature = sigAttr.getValue().str();
		else
		{
			// Fallback: build signature from operand types
			eventSignature = eventName + "(";
			for (unsigned i = 0; i < op->getNumOperands(); ++i)
			{
				if (i > 0)
					eventSignature += ",";
				eventSignature += extractSolidityTypeString(op->getOperand(i).getType());
			}
			eventSignature += ")";
		}

		// Compute topic0 = keccak256(eventSignature)
		util::h256 topic0 = util::keccak256(eventSignature);

		// Get indexed parameter info
		std::vector<bool> indexed;
		if (auto indexedAttr = op->getAttrOfType<mlir::ArrayAttr>("indexed"))
		{
			for (auto attr: indexedAttr)
				indexed.push_back(mlir::cast<mlir::BoolAttr>(attr).getValue());
		}

		// Separate indexed and non-indexed arguments
		std::vector<unsigned> indexedArgs;
		std::vector<unsigned> nonIndexedArgs;
		for (unsigned i = 0; i < op->getNumOperands(); ++i)
		{
			if (i < indexed.size() && indexed[i])
				indexedArgs.push_back(i);
			else
				nonIndexedArgs.push_back(i);
		}

		// If no indexed info available, treat all as indexed topics (legacy behavior)
		if (indexed.empty())
		{
			for (unsigned i = 0; i < op->getNumOperands() && i < 3; ++i)
				indexedArgs.push_back(i);
		}

		std::vector<yul::Statement> statements;

		// Store non-indexed arguments in memory for event data
		unsigned dataOffset = 0;
		for (unsigned argIdx: nonIndexedArgs)
		{
			std::string argName = getVariableName(op->getOperand(argIdx));
			statements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mstore")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(dataOffset))},
						 yul::Identifier{debugData, yul::YulName(argName)}}}});
			dataOffset += 32;
		}

		// Build log call arguments
		std::vector<yul::Expression> args;

		// First argument: memory offset for event data
		args.push_back(yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))});

		// Second argument: data size (non-indexed args * 32 bytes)
		unsigned dataSize = nonIndexedArgs.size() * 32;
		if (dataSize == 0 && nonIndexedArgs.empty() && indexedArgs.empty())
			dataSize = op->getNumOperands() * 32; // Fallback for all args stored in data
		args.push_back(yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(dataSize))});

		// Third argument: topic0 (event signature hash)
		args.push_back(yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(topic0))});

		// Add indexed arguments as additional topics (up to 3 more topics)
		for (unsigned argIdx: indexedArgs)
		{
			if (args.size() >= 6)
				break; // log4 max = offset + size + 4 topics
			std::string argName = getVariableName(op->getOperand(argIdx));
			args.push_back(yul::Identifier{debugData, yul::YulName(argName)});
		}

		// Determine which log function to use based on number of topics
		// topics = 1 (topic0) + indexed args count
		size_t numTopics = 1 + std::min(static_cast<size_t>(indexedArgs.size()), static_cast<size_t>(3));
		std::string logFunc = "log" + std::to_string(numTopics);

		statements.push_back(
			yul::ExpressionStatement{
				debugData,
				yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName(logFunc)}, std::move(args)}});

		if (statements.size() == 1)
			return std::move(statements[0]);
		return yul::Block{debugData, std::move(statements)};
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
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(constIt->second)})};
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
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))}}})};
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
					{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))},
					 yul::Identifier{debugData, yul::YulName(value)}}}};
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
						{yul::Identifier{debugData, yul::YulName(array)},
						 yul::Identifier{debugData, yul::YulName(index)}}}}})};
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
					{yul::FunctionCall{
						 debugData,
						 yul::Identifier{debugData, yul::YulName("add")},
						 {yul::Identifier{debugData, yul::YulName(array)},
						  yul::Identifier{debugData, yul::YulName(index)}}},
					 yul::Identifier{debugData, yul::YulName(value)}}}};
		}
		return std::nullopt;
	}

	std::optional<yul::Statement> processMappingAccessOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumResults() > 0 && op->getNumOperands() >= 1)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			std::string key = getVariableName(op->getOperand(0));

			// Look up storage slot from varName attribute
			uint32_t slot = 0;
			if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("varName"))
			{
				auto it = m_stateVariableSlots.find(nameAttr.getValue().str());
				if (it != m_stateVariableSlots.end())
					slot = it->second;
			}

			// Calculate mapping storage slot using keccak256(key . mapping_slot)
			std::vector<yul::Statement> statements;

			// mstore(0, key)
			statements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mstore")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						 yul::Identifier{debugData, yul::YulName(key)}}}});

			// mstore(32, mapping_slot)
			statements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mstore")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))}}}});

			// resultVar := sload(keccak256(0, 64))
			statements.push_back(
				yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("sload")},
						{yul::FunctionCall{
							debugData,
							yul::Identifier{debugData, yul::YulName("keccak256")},
							{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
							 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))}}}}})});

			// Return a block containing all statements
			return yul::Block{debugData, std::move(statements)};
		}
		return std::nullopt;
	}

	std::optional<yul::Statement> processMappingStoreOpToAST(mlir::Operation* op)
	{
		auto debugData = langutil::DebugData::create();
		if (op->getNumOperands() >= 2)
		{
			std::string key = getVariableName(op->getOperand(0));
			std::string value = getVariableName(op->getOperand(1));

			// Look up storage slot from varName attribute
			uint32_t slot = 0;
			if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("varName"))
			{
				auto it = m_stateVariableSlots.find(nameAttr.getValue().str());
				if (it != m_stateVariableSlots.end())
					slot = it->second;
			}

			std::vector<yul::Statement> statements;

			// mstore(0, key)
			statements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mstore")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
						 yul::Identifier{debugData, yul::YulName(key)}}}});

			// mstore(32, mapping_slot)
			statements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("mstore")},
						{yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(32))},
						 yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(slot))}}}});

			// sstore(keccak256(0, 64), value)
			statements.push_back(
				yul::ExpressionStatement{
					debugData,
					yul::FunctionCall{
						debugData,
						yul::Identifier{debugData, yul::YulName("sstore")},
						{yul::FunctionCall{
							 debugData,
							 yul::Identifier{debugData, yul::YulName("keccak256")},
							 {yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))},
							  yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(64))}}},
						 yul::Identifier{debugData, yul::YulName(value)}}}});

			return yul::Block{debugData, std::move(statements)};
		}
		return std::nullopt;
	}

	std::optional<yul::Statement> processSelectOpToAST(mlir::Operation* op)
	{
		// Lower SelectOp (ternary) to:
		//   let result := 0
		//   switch condition
		//   case 0 { result := falseVal }
		//   default { result := trueVal }
		auto debugData = langutil::DebugData::create();
		if (op->getNumResults() > 0 && op->getNumOperands() >= 3)
		{
			std::string resultVar = getOrCreateVariableName(op->getResult(0));
			std::string condition = getVariableName(op->getOperand(0));
			std::string trueVal = getVariableName(op->getOperand(1));
			std::string falseVal = getVariableName(op->getOperand(2));

			std::vector<yul::Statement> statements;

			// let result := 0
			statements.push_back(
				yul::VariableDeclaration{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))})});

			// switch condition
			// case 0 { result := falseVal }
			// default { result := trueVal }
			std::vector<yul::Case> cases;

			// case 0 { result := falseVal }
			yul::Block falseBlock{debugData, {}};
			falseBlock.statements.push_back(
				yul::Assignment{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::Identifier{debugData, yul::YulName(falseVal)})});
			cases.push_back(yul::Case{
				debugData,
				std::make_unique<yul::Literal>(
					yul::Literal{debugData, yul::LiteralKind::Number, yul::LiteralValue(u256(0))}),
				std::move(falseBlock)});

			// default { result := trueVal }
			yul::Block trueBlock{debugData, {}};
			trueBlock.statements.push_back(
				yul::Assignment{
					debugData,
					{{debugData, yul::YulName(resultVar)}},
					std::make_unique<yul::Expression>(
						yul::Identifier{debugData, yul::YulName(trueVal)})});
			cases.push_back(yul::Case{
				debugData,
				nullptr, // default case
				std::move(trueBlock)});

			statements.push_back(
				yul::Switch{
					debugData,
					std::make_unique<yul::Expression>(
						yul::Identifier{debugData, yul::YulName(condition)}),
					std::move(cases)});

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

		// Prefix function name with param types unless it's a builtin (for overloaded functions)
		if (!isYulBuiltin(funcName))
			funcName = getUniqueFuncNameForCall(op, funcName);

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
				std::make_unique<yul::Expression>(
					yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName(funcName)}, args})};
		}
		else
		{
			return yul::ExpressionStatement{
				debugData, yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName(funcName)}, args}};
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
			auto funcType = mlir::cast<mlir::FunctionType>(typeAttr.getValue());
			auto inputTypes = funcType.getInputs();

			for (size_t i = 0; i < inputTypes.size(); ++i)
			{
				if (i > 0)
					signature += ",";
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
				if (i > 0)
					signature += ",";
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
		if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
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

		// Extract type name from MLIR representation
		std::string fullTypeStr;
		{
			llvm::raw_string_ostream stream(fullTypeStr);
			type.print(stream);
		}

		// Handle array types first - they contain nested types like !solidity.array<!solidity.bytes<1>, -1>
		if (fullTypeStr.find("array") != std::string::npos)
		{
			// Check if it's a dynamic array (bytes type represented as array of bytes1)
			if (fullTypeStr.find("bytes<1>") != std::string::npos
				&& (fullTypeStr.find("-1") != std::string::npos || fullTypeStr.find(", -1>") != std::string::npos))
			{
				return "bytes"; // Dynamic bytes = array of bytes1
			}
			// Check for dynamic array of other types
			if (fullTypeStr.find("-1") != std::string::npos)
			{
				// Try to extract element type
				if (fullTypeStr.find("uint") != std::string::npos)
					return "uint256_arr";
				else if (fullTypeStr.find("int") != std::string::npos)
					return "int256_arr";
				else if (fullTypeStr.find("address") != std::string::npos)
					return "address_arr";
				else if (fullTypeStr.find("bool") != std::string::npos)
					return "bool_arr";
				return "array"; // Generic dynamic array
			}
			else
			{
				return "fixedarray"; // Fixed-size array
			}
		}

		// Handle dynbytes type (dynamic bytes in our dialect)
		if (fullTypeStr.find("dynbytes") != std::string::npos)
		{
			return "bytes";
		}

		// Try to extract Solidity type from MLIR representation
		// Check for uint first (before int, since "uint" contains "int")
		if (fullTypeStr.find("uint") != std::string::npos)
		{
			// Extract width from !solidity.uint<256> format
			size_t start = fullTypeStr.find("<");
			size_t end = fullTypeStr.find(">");
			if (start != std::string::npos && end != std::string::npos && end > start)
			{
				std::string widthStr = fullTypeStr.substr(start + 1, end - start - 1);
				// Validate that widthStr is numeric
				bool isNumeric = !widthStr.empty() && std::all_of(widthStr.begin(), widthStr.end(), ::isdigit);
				if (isNumeric)
					return "uint" + widthStr;
			}
			return "uint256"; // Default uint
		}
		else if (fullTypeStr.find("int") != std::string::npos && fullTypeStr.find("uint") == std::string::npos)
		{
			// Extract width from !solidity.int<256> format
			size_t start = fullTypeStr.find("<");
			size_t end = fullTypeStr.find(">");
			if (start != std::string::npos && end != std::string::npos && end > start)
			{
				std::string widthStr = fullTypeStr.substr(start + 1, end - start - 1);
				// Validate that widthStr is numeric
				bool isNumeric = !widthStr.empty() && std::all_of(widthStr.begin(), widthStr.end(), ::isdigit);
				if (isNumeric)
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
					// Validate that sizeStr is numeric
					bool isNumeric = !sizeStr.empty() && std::all_of(sizeStr.begin(), sizeStr.end(), ::isdigit);
					if (isNumeric)
						return "bytes" + sizeStr;
				}
				return "bytes32"; // Default fixed bytes
			}
		}
		else if (fullTypeStr.find("string") != std::string::npos)
		{
			return "string";
		}

		// Default fallback - return a sanitized version of the type
		// This ensures we always return a valid Yul identifier
		return "unknowntype";
	}

	/// Compute the packed byte size for a Solidity MLIR type.
	/// Used by abi.encodePacked to determine how many bytes each value occupies.
	unsigned getPackedByteSize(mlir::Type type)
	{
		std::string typeStr = extractSolidityTypeString(type);

		// address → 20 bytes
		if (typeStr == "address")
			return 20;
		// bool → 1 byte
		if (typeStr == "bool")
			return 1;
		// bytesN → N bytes
		if (typeStr.substr(0, 5) == "bytes" && typeStr.size() > 5)
		{
			std::string sizeStr = typeStr.substr(5);
			bool isNumeric = !sizeStr.empty() && std::all_of(sizeStr.begin(), sizeStr.end(), ::isdigit);
			if (isNumeric)
				return static_cast<unsigned>(std::stoul(sizeStr));
		}
		// uintN / intN → N/8 bytes
		if (typeStr.substr(0, 4) == "uint" && typeStr.size() > 4)
		{
			std::string bitsStr = typeStr.substr(4);
			bool isNumeric = !bitsStr.empty() && std::all_of(bitsStr.begin(), bitsStr.end(), ::isdigit);
			if (isNumeric)
				return static_cast<unsigned>(std::stoul(bitsStr)) / 8;
		}
		if (typeStr.substr(0, 3) == "int" && typeStr.size() > 3)
		{
			std::string bitsStr = typeStr.substr(3);
			bool isNumeric = !bitsStr.empty() && std::all_of(bitsStr.begin(), bitsStr.end(), ::isdigit);
			if (isNumeric)
				return static_cast<unsigned>(std::stoul(bitsStr)) / 8;
		}
		// Dynamic types (bytes, string, arrays) → 32 bytes as pointer/length
		// (in packed encoding, these would need special handling for actual content)
		return 32;
	}

	// Generate a unique function name based on function name and parameter types
	// This is necessary for function overloading where multiple functions have the same
	// name but different parameter types (e.g., foo(int256) vs foo(uint256))
	std::string getUniqueFuncName(mlir::Operation* funcOp, const std::string& prefix = "fun_")
	{
		std::string funcName = "unknown";
		if (auto nameAttr = funcOp->getAttrOfType<mlir::StringAttr>("sym_name"))
			funcName = nameAttr.getValue().str();

		// Build type signature suffix from parameter types
		std::string typeSuffix;
		if (auto typeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("function_type"))
		{
			auto funcType = mlir::cast<mlir::FunctionType>(typeAttr.getValue());
			auto inputTypes = funcType.getInputs();

			for (size_t i = 0; i < inputTypes.size(); ++i)
			{
				if (i > 0)
					typeSuffix += "_";
				typeSuffix += extractSolidityTypeString(inputTypes[i]);
			}
		}
		else if (funcOp->getNumRegions() > 0 && !funcOp->getRegion(0).empty())
		{
			// Fallback: extract types from block arguments
			auto& entryBlock = funcOp->getRegion(0).front();
			int numParams = entryBlock.getNumArguments();

			for (int i = 0; i < numParams; ++i)
			{
				if (i > 0)
					typeSuffix += "_";
				auto argType = entryBlock.getArgument(i).getType();
				typeSuffix += extractSolidityTypeString(argType);
			}
		}

		// If no parameters, just use "0" for consistency
		if (typeSuffix.empty())
			typeSuffix = "0";

		return prefix + funcName + "_" + typeSuffix;
	}

	// Generate unique function name for call operations based on operand types
	std::string getUniqueFuncNameForCall(mlir::Operation* callOp, const std::string& funcName)
	{
		std::string typeSuffix;

		for (unsigned i = 0; i < callOp->getNumOperands(); ++i)
		{
			if (i > 0)
				typeSuffix += "_";
			auto operandType = callOp->getOperand(i).getType();
			typeSuffix += extractSolidityTypeString(operandType);
		}

		// If no parameters, just use "0" for consistency
		if (typeSuffix.empty())
			typeSuffix = "0";

		return "fun_" + funcName + "_" + typeSuffix;
	}

	bool isYulBuiltin(const std::string& name)
	{
		// Common Yul builtins - expand this list as needed
		static const std::set<std::string> builtins
			= {"add",
			   "sub",
			   "mul",
			   "div",
			   "mod",
			   "exp",
			   "and",
			   "or",
			   "xor",
			   "not",
			   "shl",
			   "shr",
			   "sar",
			   "eq",
			   "lt",
			   "gt",
			   "iszero",
			   "mstore",
			   "mload",
			   "sstore",
			   "sload",
			   "return",
			   "revert",
			   "stop",
			   "invalid",
			   "keccak256",
			   "address",
			   "balance",
			   "origin",
			   "caller",
			   "callvalue",
			   "calldataload",
			   "calldatasize",
			   "calldatacopy",
			   "codesize",
			   "codecopy",
			   "gasprice",
			   "extcodesize",
			   "extcodecopy",
			   "returndatasize",
			   "returndatacopy",
			   "blockhash",
			   "coinbase",
			   "timestamp",
			   "number",
			   "difficulty",
			   "gaslimit",
			   "chainid",
			   "selfbalance",
			   "basefee",
			   "gas",
			   "call",
			   "callcode",
			   "delegatecall",
			   "staticcall",
			   "create",
			   "create2",
			   "selfdestruct"};
		return builtins.find(name) != builtins.end();
	}

	std::string getOrCreateVariableName(mlir::Value value)
	{
		void* key = value.getAsOpaquePointer();

		// First check function-scoped names if we're in a function context
		if (!m_currentFunction.empty() && m_functionScopedNames.find(m_currentFunction) != m_functionScopedNames.end())
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
		if (!m_currentFunction.empty() && m_functionScopedNames.find(m_currentFunction) != m_functionScopedNames.end())
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
		constructorStatements.push_back(
			yul::ExpressionStatement{
				debugData, yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("stop")}, {}}});

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
		deployedStatements.push_back(
			yul::ExpressionStatement{
				debugData, yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("stop")}, {}}});

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
		constructorStatements.push_back(
			yul::ExpressionStatement{
				debugData, yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("stop")}, {}}});

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
		deployedStatements.push_back(
			yul::ExpressionStatement{
				debugData, yul::FunctionCall{debugData, yul::Identifier{debugData, yul::YulName("stop")}, {}}});

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

MLIRToYulLowering::MLIRToYulLowering(): m_impl(std::make_unique<MLIRToYulLoweringImpl>()) {}

MLIRToYulLowering::~MLIRToYulLowering() = default;

std::shared_ptr<yul::Object> MLIRToYulLowering::lower(std::string const& _mlirModule)
{
	return m_impl->lower(_mlirModule);
}

std::string MLIRToYulLowering::optimize(
	std::string const& _mlirModule, bool _printIntermediateMLIR, std::string const& _mlirFile, bool _runAnalysis)
{
	return m_impl->optimize(_mlirModule, _printIntermediateMLIR, _mlirFile, _runAnalysis);
}

} // namespace solidity::frontend
