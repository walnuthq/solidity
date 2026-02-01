// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test Dynamic Array Push MLIR Codegen
/// @notice Tests that array.push() generates correct MLIR operations
contract TestArrayPush {

    uint256[] public values;

    // ========== Array Push ==========

    /// @notice Test array push with a single value
    function pushValue(uint256 val) external {
        values.push(val);
    }
}

// Test Contract Generation
// CHECK: solidity.contract "TestArrayPush"

// Test state variable declaration
// CHECK: solidity.state_var "values" : !solidity.array<!solidity.uint<256>, -1>

// Test load_state for array
// CHECK: solidity.load_state "values" : !solidity.array<!solidity.uint<256>, -1>

// Test array_push with varName attribute
// CHECK: solidity.array_push %{{.*}}, %{{.*}} {varName = "values"} : !solidity.array<!solidity.uint<256>, -1>, !solidity.uint<256>

// Test Function Declarations
// CHECK: sym_name = "pushValue"

// Test Return Statements
// CHECK: solidity.return
