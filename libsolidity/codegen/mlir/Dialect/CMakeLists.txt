# The dialect ladder, one subdirectory per rung.
add_subdirectory(Sol)
add_subdirectory(Yul)
add_subdirectory(EVM)

# Propagate MLIR availability upwards (Sol/CMakeLists sets it).
set(SOLIDITY_HAS_MLIR ${SOLIDITY_HAS_MLIR} PARENT_SCOPE)
