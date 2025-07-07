// RUN: %solc --mlir-optimize %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FactorialStorage {
    uint256 private result;
    uint256 constant MAX_SAFE_N = 57;

    function computeFactorial(uint256 n) external {
        if (n > MAX_SAFE_N) {
            revert("Overflow: n too large");
        }
        
        result = 1;
        for (uint256 i = 2; i <= n; ++i) {
            result *= i;
        }
    }

    function getResult() external view returns (uint256) {
        return result;
    }
}

// Test MLIR Contract Generation
// CHECK: solidity.contract @FactorialStorage

// Test State Variable Declaration
// CHECK: solidity.state_var "result" : !solidity.uint<256>
// CHECK: solidity.state_var "MAX_SAFE_N" : !solidity.uint<256>

// Test Function Declaration with Visibility and Mutability
// CHECK: solidity.func @computeFactorial
// CHECK-SAME: visibility = "external"
// CHECK-SAME: state_mutability = "nonpayable"

// Test Function Declaration for View Function  
// CHECK: solidity.func @getResult
// CHECK-SAME: visibility = "external"
// CHECK-SAME: state_mutability = "view"

// Test Control Flow Operations
// CHECK: solidity.if
// CHECK: solidity.for

// Test Arithmetic Operations
// CHECK: solidity.cmp "gt"
// CHECK: solidity.mul
// CHECK: solidity.add

// Test State Variable Operations
// CHECK: solidity.store_state
// CHECK: solidity.load_state

// Test Control Flow Termination
// CHECK: solidity.revert
// CHECK: solidity.return