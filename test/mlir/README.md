# MLIR Test Suite for Solidity Compiler

This directory contains comprehensive tests for the MLIR integration in the Solidity compiler using LLVM's `lit` testing framework and `FileCheck` for output verification.

## Test Structure

### Test Files
- **factorial.sol**: Tests state variables, arithmetic operations, control flow, and revert statements
- **control_flow.sol**: Tests nested loops, while loops, do-while loops, and break statements  
- **structs_memory.sol**: Tests struct operations, memory patterns, arrays, and mappings
- **arithmetic_operations.sol**: Comprehensive test of all arithmetic and bitwise operations
- **type_system.sol**: Tests MLIR type generation for all Solidity types

### Configuration
- **lit.cfg.py**: LLVM lit configuration for test execution
- Tests use the `--mlir-optimize` flag to trigger MLIR code generation
- FileCheck patterns verify expected MLIR operations are generated

## Prerequisites

1. **LLVM/MLIR**: Must be installed with `lit` and `FileCheck` tools available
2. **Solidity Build**: The `solc` binary must be built with MLIR support enabled
3. **Build Directory**: Tests expect the build directory at `../../build/`

## Running Tests

### Run All MLIR Tests
```bash
# From the test/mlir directory
lit .

# Or from the project root
lit test/mlir/
```

### Run Individual Tests
```bash
# Test specific functionality
lit factorial.sol
lit control_flow.sol
lit arithmetic_operations.sol
```

### Run with Verbose Output
```bash
lit -v .
```

## Test Verification

Each test uses FileCheck patterns to verify:

### Core MLIR Infrastructure
- Contract generation: `solidity.contract @ContractName`
- Function generation: `solidity.func @functionName`  
- Type system: `!solidity.uint<256>`, `!solidity.address`, etc.

### Operations Coverage
- **Arithmetic**: `solidity.add`, `solidity.mul`, `solidity.div`, etc.
- **Bitwise**: `solidity.and`, `solidity.or`, `solidity.shl`, etc.
- **Comparison**: `solidity.cmp "gt"`, `solidity.cmp "eq"`, etc.
- **Control Flow**: `solidity.if`, `solidity.for`, `solidity.while`
- **Memory/Storage**: `solidity.load_state`, `solidity.store_state`
- **Arrays/Mappings**: `solidity.array_access`, `solidity.mapping_store`
- **Functions**: `solidity.function_call`, `solidity.return`

### Error Handling
- **Assertions**: `solidity.require`, `solidity.assert`  
- **Reverts**: `solidity.revert`

## Adding New Tests

1. Create a `.sol` file in this directory
2. Add the standard RUN line: `// RUN: %solc --mlir-optimize %s 2>&1 | %FileCheck %s`
3. Add `// REQUIRES: mlir` for conditional execution
4. Include FileCheck patterns to verify expected MLIR operations
5. Use `// CHECK:` patterns to match generated MLIR output

### Example Test Structure
```solidity
// RUN: %solc --mlir-optimize %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

pragma solidity ^0.8.0;

contract TestContract {
    function testFunction() public pure returns (uint256) {
        return 42;
    }
}

// CHECK: solidity.contract @TestContract
// CHECK: solidity.func @testFunction
// CHECK: solidity.constant
// CHECK: solidity.return
```

## Expected Output

When tests pass, you should see:
```
Testing Time: X.XX seconds
Passed: 5
```

Failed tests will show the specific FileCheck patterns that didn't match, helping identify issues in MLIR generation.