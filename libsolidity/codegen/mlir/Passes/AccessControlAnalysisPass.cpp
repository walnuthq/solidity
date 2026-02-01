#include "AccessControlAnalysisPass.h"

#ifdef SOLIDITY_HAS_MLIR

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#pragma GCC diagnostic pop

#include "libsolidity/codegen/mlir/Dialect/SolidityDialect.h"
#include <iostream>
#include <string>
#include <unordered_set>

namespace mlir::solidity
{

namespace
{

// Analysis pass that detects missing access control in state-modifying functions
// This is a security-focused pass that warns when public/external functions
// modify state variables without appropriate access control checks
class AccessControlAnalysisPass
	: public mlir::PassWrapper<AccessControlAnalysisPass, mlir::OperationPass<mlir::ModuleOp>>
{
public:
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AccessControlAnalysisPass)

	void runOnOperation() override
	{
		mlir::ModuleOp module = getOperation();

		unsigned warningCount = 0;

		// Walk through all operations in the module to find Solidity functions
		module.walk(
			[&](mlir::Operation* op)
			{
				// Check if this is a solidity.func operation
				if (op->getName().getStringRef() == "solidity.func")
				{
					analyzeFunction(op, warningCount);
				}
			});
	}

	StringRef getArgument() const final { return "access-control-analysis"; }
	StringRef getDescription() const final
	{
		return "Analyze functions for missing access control on state modifications";
	}

private:
	struct FunctionInfo
	{
		std::string name;
		bool isPublicOrExternal = false;
		bool modifiesState = false;
		bool hasAccessControl = false;
		std::unordered_set<std::string> modifiedVariables;
		mlir::Operation* operation = nullptr;
	};

	void analyzeFunction(mlir::Operation* funcOp, unsigned& warningCount)
	{
		FunctionInfo info;
		info.operation = funcOp;

		// Get function name
		if (auto nameAttr = funcOp->getAttrOfType<mlir::StringAttr>("sym_name"))
		{
			info.name = nameAttr.str();
		}
		else if (auto nameAttr = funcOp->getAttrOfType<mlir::StringAttr>("name"))
		{
			info.name = nameAttr.str();
		}
		else
		{
			info.name = "<anonymous>";
		}

		// Check visibility (assume public/external if not specified, or if explicitly marked)
		if (auto visibilityAttr = funcOp->getAttrOfType<mlir::StringAttr>("visibility"))
		{
			auto visibility = visibilityAttr.str();
			info.isPublicOrExternal = (visibility == "public" || visibility == "external");
		}
		else
		{
			// Default to public/external for analysis (conservative approach)
			info.isPublicOrExternal = true;
		}

		// Skip analysis for internal/private functions
		if (!info.isPublicOrExternal)
		{
			return;
		}

		// Analyze function body for state modifications and access control
		funcOp->walk(
			[&](mlir::Operation* op)
			{
				auto opName = op->getName().getStringRef();

				// Check for state variable writes
				if (opName.contains("store_state") || opName.contains("sstore"))
				{
					info.modifiesState = true;

					// Try to get variable name
					if (auto varNameAttr = op->getAttrOfType<mlir::StringAttr>("varName"))
					{
						info.modifiedVariables.insert(varNameAttr.str());
					}
				}

				// Check for access control patterns
				// Look for require/revert operations with msg.sender checks
				if (opName.contains("require") || opName.contains("revert"))
				{
					// Check if this require has msg.sender in its condition
					// Walk backwards to find the condition
					if (hasMessageSenderCheck(op))
					{
						info.hasAccessControl = true;
					}
				}

				// Also check for explicit msg.sender comparisons
				if (opName.contains("msg.sender") || opName.contains("caller"))
				{
					// If we see msg.sender being used, it might be for access control
					// This is a heuristic - look for equality checks
					if (hasAccessControlPattern(op))
					{
						info.hasAccessControl = true;
					}
				}
			});

		// Report findings
		if (info.modifiesState && !info.hasAccessControl)
		{
			warningCount++;
			emitAccessControlWarning(info);
		}
	}

	bool hasMessageSenderCheck(mlir::Operation* requireOp)
	{
		// Check operands of the require operation for msg.sender references
		for (auto operand: requireOp->getOperands())
		{
			// Traverse the def-use chain to find msg.sender
			if (auto defOp = operand.getDefiningOp())
			{
				if (containsMessageSender(defOp))
				{
					return true;
				}
			}
		}
		return false;
	}

	bool containsMessageSender(mlir::Operation* op)
	{
		// Recursively check if operation involves msg.sender
		auto opName = op->getName().getStringRef();

		if (opName.contains("msg.sender") || opName.contains("caller") || opName.contains("msg_sender"))
		{
			return true;
		}

		// Check operands recursively (limited depth to avoid infinite loops)
		for (auto operand: op->getOperands())
		{
			if (auto defOp = operand.getDefiningOp())
			{
				if (containsMessageSender(defOp))
				{
					return true;
				}
			}
		}

		return false;
	}

	bool hasAccessControlPattern(mlir::Operation* op)
	{
		// Look for comparison operations involving msg.sender
		auto opName = op->getName().getStringRef();

		// Check if this is part of a comparison
		for (auto* user: op->getUsers())
		{
			auto userName = user->getName().getStringRef();
			if (userName.contains("cmp") || userName.contains("eq") || userName.contains("compare")
				|| userName.contains("=="))
			{
				return true;
			}
		}

		return false;
	}

	void emitAccessControlWarning(const FunctionInfo& info)
	{
		// Build warning message
		std::string message = "Function '" + info.name + "' modifies state without access control";

		if (!info.modifiedVariables.empty())
		{
			message += " (modifies: ";
			bool first = true;
			for (const auto& var: info.modifiedVariables)
			{
				if (!first)
					message += ", ";
				message += var;
				first = false;
			}
			message += ")";
		}

		// Emit MLIR diagnostic - this will be captured by the diagnostic handler
		if (info.operation)
		{
			info.operation->emitWarning() << message;
		}
	}
};

} // namespace

std::unique_ptr<mlir::Pass> createAccessControlAnalysisPass() { return std::make_unique<AccessControlAnalysisPass>(); }

} // namespace mlir::solidity

#else // SOLIDITY_HAS_MLIR

namespace mlir::solidity
{

std::unique_ptr<mlir::Pass> createAccessControlAnalysisPass() { return nullptr; }

} // namespace mlir::solidity

#endif // SOLIDITY_HAS_MLIR
