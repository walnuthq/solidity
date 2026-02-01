#include "StorageCachingPass.h"

#ifdef SOLIDITY_HAS_MLIR

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#pragma GCC diagnostic pop

#include "libsolidity/codegen/mlir/Dialect/SolidityDialect.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlir::solidity
{

namespace
{

// Pass that performs storage caching optimization for loops
// This optimization hoists storage loads out of loops and stores only once after
class StorageCachingPass: public mlir::PassWrapper<StorageCachingPass, mlir::OperationPass<mlir::ModuleOp>>
{
public:
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StorageCachingPass)

	void runOnOperation() override
	{
		mlir::ModuleOp module = getOperation();

		// Walk through all operations in the module to find Solidity functions
		module.walk(
			[&](mlir::Operation* op)
			{
				// Check if this is a solidity.func operation
				if (op->getName().getStringRef() == "solidity.func")
				{
					// Process loops within this function
					op->walk([&](mlir::scf::ForOp forOp) { optimizeForLoop(forOp); });

					op->walk([&](mlir::scf::WhileOp whileOp) { optimizeWhileLoop(whileOp); });
				}
			});
	}

	StringRef getArgument() const final { return "storage-caching"; }
	StringRef getDescription() const final { return "Cache storage variables in loops to reduce gas costs"; }

private:
	struct StorageInfo
	{
		std::string varName;
		mlir::Operation* firstLoad = nullptr;
		mlir::Operation* lastStore = nullptr;
		mlir::Type storageType;
		int loadCount = 0;
		int storeCount = 0;
	};

	void analyzeStorageOp(mlir::Operation* op, std::unordered_map<std::string, StorageInfo>& storageOps)
	{
		auto opName = op->getName().getStringRef();

		// Check for load_state operations
		if (opName.contains("load_state"))
		{
			// Get the variable name attribute
			if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("varName"))
			{
				auto varName = nameAttr.str();
				storageOps[varName].varName = varName;
				storageOps[varName].loadCount++;
				if (!storageOps[varName].firstLoad)
					storageOps[varName].firstLoad = op;
				if (op->getNumResults() > 0)
					storageOps[varName].storageType = op->getResult(0).getType();
			}
		}
		// Check for store_state operations
		else if (opName.contains("store_state"))
		{
			if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("varName"))
			{
				auto varName = nameAttr.str();
				storageOps[varName].varName = varName;
				storageOps[varName].storeCount++;
				storageOps[varName].lastStore = op;
			}
		}
	}

	bool optimizeForLoop(mlir::scf::ForOp forOp)
	{
		// Analyze storage access patterns in the loop
		std::unordered_map<std::string, StorageInfo> storageOps;

		// Walk through the loop body to find storage operations
		forOp.getBody()->walk([this, &storageOps](mlir::Operation* op) { analyzeStorageOp(op, storageOps); });

		// Find variables that are both loaded and stored (candidates for caching)
		StorageInfo* bestCandidate = nullptr;
		int maxAccesses = 0;

		for (auto& [name, info]: storageOps)
		{
			int totalAccesses = info.loadCount + info.storeCount;
			if (info.loadCount > 0 && info.storeCount > 0 && totalAccesses > maxAccesses)
			{
				maxAccesses = totalAccesses;
				bestCandidate = &info;
			}
		}

		if (!bestCandidate || maxAccesses < 2)
			return false; // No optimization opportunity

		// Perform the transformation
		return transformForLoopWithCaching(forOp, *bestCandidate);
	}

	bool transformForLoopWithCaching(mlir::scf::ForOp forOp, const StorageInfo& info)
	{
		mlir::OpBuilder builder(forOp);
		mlir::Location loc = forOp.getLoc();

		// Step 1: Create a load before the loop
		builder.setInsertionPoint(forOp);

		// Create a load_state operation using the generic operation builder
		mlir::OperationState loadState(loc, "solidity.load_state");
		loadState.addAttribute("varName", builder.getStringAttr(info.varName));
		loadState.addTypes(info.storageType);
		auto* initialLoad = builder.create(loadState);

		if (!initialLoad || initialLoad->getNumResults() == 0)
			return false;

		mlir::Value initialValue = initialLoad->getResult(0);

		// Step 2: Add as loop-carried variable
		auto oldIterArgs = llvm::to_vector(forOp.getInitArgs());
		oldIterArgs.push_back(initialValue);

		// Step 3: Create new for loop
		auto newForOp = builder.create<
			mlir::scf::ForOp>(loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), oldIterArgs);

		// Step 4: Map old to new values
		mlir::IRMapping mapping;
		mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

		auto oldRegionArgs = forOp.getRegionIterArgs();
		auto newRegionArgs = newForOp.getRegionIterArgs();
		for (size_t i = 0; i < oldRegionArgs.size(); ++i)
		{
			mapping.map(oldRegionArgs[i], newRegionArgs[i]);
		}

		// The cached value is the last argument
		mlir::Value cachedValue = newRegionArgs.back();
		mlir::Value currentCachedValue = cachedValue;

		// Step 5: Clone loop body with replacements
		builder.setInsertionPointToStart(newForOp.getBody());

		for (auto& op: forOp.getBody()->getOperations())
		{
			if (isa<mlir::scf::YieldOp>(op))
				continue;

			bool replaced = false;
			auto opName = op.getName().getStringRef();

			// Replace loads with cached value
			if (opName.contains("load_state"))
			{
				if (auto nameAttr = op.getAttrOfType<mlir::StringAttr>("varName"))
				{
					if (nameAttr.str() == info.varName)
					{
						mapping.map(op.getResult(0), currentCachedValue);
						replaced = true;
					}
				}
			}
			// Track stores to update cached value
			else if (opName.contains("store_state"))
			{
				if (auto nameAttr = op.getAttrOfType<mlir::StringAttr>("varName"))
				{
					if (nameAttr.str() == info.varName && op.getNumOperands() > 0)
					{
						// The value being stored becomes the new cached value
						currentCachedValue = mapping.lookupOrDefault(op.getOperand(0));
						replaced = true;
					}
				}
			}

			if (!replaced)
			{
				auto* clonedOp = builder.clone(op, mapping);
				for (size_t i = 0; i < op.getNumResults(); ++i)
				{
					mapping.map(op.getResult(i), clonedOp->getResult(i));
				}
			}
		}

		// Step 6: Update yield with cached value
		auto oldYield = cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
		auto yieldOperands = llvm::to_vector(oldYield.getOperands());
		for (auto& operand: yieldOperands)
		{
			operand = mapping.lookupOrDefault(operand);
		}
		yieldOperands.push_back(currentCachedValue);
		builder.create<mlir::scf::YieldOp>(loc, yieldOperands);

		// Step 7: Store final value after loop
		builder.setInsertionPointAfter(newForOp);

		mlir::Value finalValue = newForOp.getResults().back();
		mlir::OperationState storeState(loc, "solidity.store_state");
		storeState.addAttribute("varName", builder.getStringAttr(info.varName));
		storeState.addOperands(finalValue);
		builder.create(storeState);

		// Step 8: Replace old loop results
		for (size_t i = 0; i < forOp.getNumResults(); ++i)
		{
			forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
		}

		// Step 9: Erase old loop
		forOp.erase();

		return true;
	}

	bool optimizeWhileLoop(mlir::scf::WhileOp whileOp)
	{
		// Analyze storage access patterns in the while loop's after region
		std::unordered_map<std::string, StorageInfo> storageOps;

		// Analyze the after region (the loop body)
		whileOp.getAfter().walk([this, &storageOps](mlir::Operation* op) { analyzeStorageOp(op, storageOps); });

		// Find best candidate for optimization
		StorageInfo* bestCandidate = nullptr;
		int maxAccesses = 0;

		for (auto& [name, info]: storageOps)
		{
			int totalAccesses = info.loadCount + info.storeCount;
			if (info.loadCount > 0 && info.storeCount > 0 && totalAccesses > maxAccesses)
			{
				maxAccesses = totalAccesses;
				bestCandidate = &info;
			}
		}

		if (!bestCandidate || maxAccesses < 2)
			return false; // No optimization opportunity

		// Perform the transformation
		return transformWhileLoopWithCaching(whileOp, *bestCandidate);
	}

	bool transformWhileLoopWithCaching(mlir::scf::WhileOp whileOp, const StorageInfo& info)
	{
		mlir::OpBuilder builder(whileOp);
		mlir::Location loc = whileOp.getLoc();

		// Step 1: Create a load before the loop
		builder.setInsertionPoint(whileOp);

		// Create a load_state operation
		mlir::OperationState loadState(loc, "solidity.load_state");
		loadState.addAttribute("varName", builder.getStringAttr(info.varName));
		loadState.addTypes(info.storageType);
		auto* initialLoad = builder.create(loadState);

		if (!initialLoad || initialLoad->getNumResults() == 0)
			return false;

		mlir::Value initialValue = initialLoad->getResult(0);

		// Step 2: Add cached value to loop-carried variables
		auto oldInits = llvm::to_vector(whileOp.getInits());
		oldInits.push_back(initialValue);

		auto oldResultTypes = llvm::to_vector(whileOp.getResultTypes());
		oldResultTypes.push_back(info.storageType);

		// Step 3: Create new while loop with additional loop-carried variable
		auto newWhileOp = builder.create<mlir::scf::WhileOp>(loc, oldResultTypes, oldInits);

		// Step 4: Clone the before region with new block arguments
		auto& oldBefore = whileOp.getBefore();
		auto& newBefore = newWhileOp.getBefore();

		// Create block in before region with updated arguments
		auto* newBeforeBlock = builder.createBlock(&newBefore);
		for (auto type: llvm::to_vector(oldBefore.front().getArgumentTypes()))
		{
			newBeforeBlock->addArgument(type, loc);
		}
		// Add argument for cached value
		newBeforeBlock->addArgument(info.storageType, loc);

		// Map old to new block arguments
		mlir::IRMapping beforeMapping;
		auto oldBeforeArgs = oldBefore.front().getArguments();
		auto newBeforeArgs = newBeforeBlock->getArguments();
		for (size_t i = 0; i < oldBeforeArgs.size(); ++i)
		{
			beforeMapping.map(oldBeforeArgs[i], newBeforeArgs[i]);
		}

		// Clone operations in before block, fixing scf.condition
		builder.setInsertionPointToStart(newBeforeBlock);
		for (auto& op: oldBefore.front().getOperations())
		{
			if (auto condOp = dyn_cast<mlir::scf::ConditionOp>(op))
			{
				// The condition op needs to pass the cached value too
				auto condOperands = llvm::to_vector(condOp.getArgs());
				for (auto& operand: condOperands)
				{
					operand = beforeMapping.lookupOrDefault(operand);
				}
				// Add the cached value to the condition arguments
				condOperands.push_back(newBeforeArgs.back());
				builder.create<mlir::scf::ConditionOp>(
					condOp.getLoc(), beforeMapping.lookupOrDefault(condOp.getCondition()), condOperands);
			}
			else
			{
				builder.clone(op, beforeMapping);
			}
		}

		// Step 5: Clone the after region with storage replacements
		auto& oldAfter = whileOp.getAfter();
		auto& newAfter = newWhileOp.getAfter();

		// Create block in after region
		auto* newAfterBlock = builder.createBlock(&newAfter);
		for (auto type: llvm::to_vector(oldAfter.front().getArgumentTypes()))
		{
			newAfterBlock->addArgument(type, loc);
		}
		// Add argument for cached value
		newAfterBlock->addArgument(info.storageType, loc);

		// Map old to new block arguments
		mlir::IRMapping afterMapping;
		auto oldAfterArgs = oldAfter.front().getArguments();
		auto newAfterArgs = newAfterBlock->getArguments();
		for (size_t i = 0; i < oldAfterArgs.size(); ++i)
		{
			afterMapping.map(oldAfterArgs[i], newAfterArgs[i]);
		}

		// The cached value is the last argument
		mlir::Value cachedValue = newAfterArgs.back();
		mlir::Value currentCachedValue = cachedValue;

		// Clone operations with replacements
		builder.setInsertionPointToStart(newAfterBlock);
		for (auto& op: oldAfter.front().getOperations())
		{
			if (isa<mlir::scf::YieldOp>(op))
				continue;

			bool replaced = false;
			auto opName = op.getName().getStringRef();

			// Replace loads with cached value
			if (opName.contains("load_state"))
			{
				if (auto nameAttr = op.getAttrOfType<mlir::StringAttr>("varName"))
				{
					if (nameAttr.str() == info.varName)
					{
						afterMapping.map(op.getResult(0), currentCachedValue);
						replaced = true;
					}
				}
			}
			// Track stores to update cached value
			else if (opName.contains("store_state"))
			{
				if (auto nameAttr = op.getAttrOfType<mlir::StringAttr>("varName"))
				{
					if (nameAttr.str() == info.varName && op.getNumOperands() > 0)
					{
						// The value being stored becomes the new cached value
						currentCachedValue = afterMapping.lookupOrDefault(op.getOperand(0));
						replaced = true;
					}
				}
			}

			if (!replaced)
			{
				auto* clonedOp = builder.clone(op, afterMapping);
				for (size_t i = 0; i < op.getNumResults(); ++i)
				{
					afterMapping.map(op.getResult(i), clonedOp->getResult(i));
				}
			}
		}

		// Update yield with cached value
		auto oldYield = cast<mlir::scf::YieldOp>(oldAfter.front().getTerminator());
		auto yieldOperands = llvm::to_vector(oldYield.getOperands());
		for (auto& operand: yieldOperands)
		{
			operand = afterMapping.lookupOrDefault(operand);
		}
		yieldOperands.push_back(currentCachedValue);
		builder.create<mlir::scf::YieldOp>(loc, yieldOperands);

		// Step 6: Store final value after loop
		builder.setInsertionPointAfter(newWhileOp);

		mlir::Value finalValue = newWhileOp.getResults().back();
		mlir::OperationState storeState(loc, "solidity.store_state");
		storeState.addAttribute("varName", builder.getStringAttr(info.varName));
		storeState.addOperands(finalValue);
		builder.create(storeState);

		// Step 7: Replace old loop results
		for (size_t i = 0; i < whileOp.getNumResults(); ++i)
		{
			whileOp.getResult(i).replaceAllUsesWith(newWhileOp.getResult(i));
		}

		// Step 8: Erase old loop
		whileOp.erase();

		return true;
	}
};

} // namespace

std::unique_ptr<mlir::Pass> createStorageCachingPass() { return std::make_unique<StorageCachingPass>(); }

} // namespace mlir::solidity

#else // SOLIDITY_HAS_MLIR

namespace mlir::solidity
{

std::unique_ptr<mlir::Pass> createStorageCachingPass() { return nullptr; }

} // namespace mlir::solidity

#endif // SOLIDITY_HAS_MLIR