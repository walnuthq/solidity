// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
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

// CHECK: solidity.contract "WhileLoopPatterns"
// CHECK: solidity.state_var "result"
// CHECK: sym_name = "gcd"
// CHECK: scf.while
// CHECK: solidity.cmp "ne"
// CHECK: solidity.mod
// CHECK: scf.yield
// CHECK: solidity.store_state "result"
// CHECK: sym_name = "fibonacci"
// CHECK: solidity.cmp "le"
// CHECK: solidity.if
// CHECK: scf.while
// CHECK: solidity.add
// CHECK: scf.yield
// CHECK: sym_name = "power"