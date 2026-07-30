# MLIR pipeline for solc — the dialect ladder:
#
#   sol dialect -> yul dialect -> evm dialect -> RISC-V (via LLVM)
#
# Everything below is only entered when MLIR was found (see root CMakeLists).

add_subdirectory(Dialect)
add_subdirectory(Target/YulText)
add_subdirectory(Target/EVM)
add_subdirectory(Target/RISCV)
add_subdirectory(Import/LibyulAST)
add_subdirectory(Conversion/SolToYul)
add_subdirectory(Conversion/YulToEVM)
add_subdirectory(Conversion/EVMToLLVM)
add_subdirectory(tools)
