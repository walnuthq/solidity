# MLIR pipeline for solc — the dialect ladder:
#
#   sol dialect -> yul dialect -> evm dialect -> RISC-V (via LLVM)
#
# Everything below is only entered when MLIR was found (see root CMakeLists).

add_subdirectory(Dialect)
add_subdirectory(Target/YulText)
add_subdirectory(tools)
