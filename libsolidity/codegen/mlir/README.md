# MLIR-based Solidity Codegen

This directory contains an experimental MLIR-based code generation pipeline for Solidity.

## Build

Requires LLVM/MLIR 21.x installed (e.g. via Homebrew: `brew install llvm@21`).

```bash
mkdir build && cd build
cmake .. \
  -DSOLIDITY_HAS_MLIR=1 \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/mlir
make -j$(nproc)
```

## Test

```bash
make check-solidity-mlir
```

## Directory Structure

- `Dialect/` -- Solidity MLIR dialect (types, operations defined in TableGen)
- `Passes/` -- Custom MLIR optimization and analysis passes
- `MLIRGenerator.cpp` -- AST to MLIR lowering
- `MLIRToYulLowering.cpp` -- MLIR to Yul AST lowering
