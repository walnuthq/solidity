#pragma once

#include <memory>

#ifdef SOLIDITY_HAS_MLIR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/Pass/Pass.h"
#pragma GCC diagnostic pop
#endif

namespace mlir {
class Pass;

namespace solidity {

std::unique_ptr<mlir::Pass> createStorageCachingPass();

} // namespace solidity
} // namespace mlir