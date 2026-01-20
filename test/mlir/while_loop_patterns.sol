// RUN: %solc --mlir-optimize %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Test case for while loop patterns including modulo in conditions
contract WhileLoopPatterns {
    uint256 public result;

    // GCD using while loop with modulo in condition
    function gcd(uint256 a, uint256 b) external {
        while (b != 0) {
            uint256 temp = b;
            b = a % b;
            a = temp;
        }
        result = a;
    }

    // Fibonacci with for loop
    function fibonacci(uint256 n) external {
        if (n <= 1) {
            result = n;
            return;
        }
        uint256 a = 0;
        uint256 b = 1;
        for (uint256 i = 2; i <= n; i++) {
            uint256 temp = a + b;
            a = b;
            b = temp;
        }
        result = b;
    }

    // Power function with for loop
    function power(uint256 base, uint256 exp) external {
        result = 1;
        for (uint256 i = 0; i < exp; i++) {
            result *= base;
        }
    }
}

// Test MLIR Contract Generation
// CHECK: solidity.contract @WhileLoopPatterns

// Test State Variable
// CHECK: solidity.state_var "result" : !solidity.uint<256>

// Test Function Declarations
// CHECK: solidity.func @gcd
// CHECK: solidity.func @fibonacci
// CHECK: solidity.func @power

// Test While Loop for GCD - should generate scf.while
// CHECK: scf.while
// CHECK: solidity.mod
// CHECK: solidity.cmp "ne"
// CHECK: scf.condition

// Test For Loop patterns
// CHECK: solidity.cmp "le"
// CHECK: solidity.add
// CHECK: solidity.mul

// Test Store/Load State
// CHECK: solidity.store_state
