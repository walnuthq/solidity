# Solidity lit tests

This directory contains examples for testing Solidity command-line behavior with LLVM's `lit` runner. The initial suite focuses on the experimental ETHDebug output because it benefits from compact command-line tests with precise output checks.

## Running the suite

Build `solc`, then run:

```bash
lit -sv test/lit --param solc=build/solc/solc
```

If the CMake target is available, the same tests can be run with:

```bash
cmake --build build --target check-solidity-lit
```

The `solc` parameter is optional when `build/solc/solc` exists. You can also set `SOLC=/path/to/solc`.

## Writing tests

Tests are ordinary Solidity files with `// RUN:` lines. Use `%solc` for the compiler and `%FileCheck` for textual checks.
