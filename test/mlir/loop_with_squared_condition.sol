// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Test case for loop conditions with intermediate computations
// Bug fix: scf.while conditions that require computing intermediate values (like i*i)
// must have those computations processed as statements before the condition is evaluated.
contract LoopWithSquaredCondition {
    uint256 public result;

    // Loop where condition involves a squared variable: i * i <= n
    // This requires computing i*i before comparing with n
    function squaredLoop(uint256 n) external {
        for (uint256 i = 3; i * i <= n; i += 2) {
            result = i;
        }
    }

    // Prime check uses similar pattern
    function isPrime(uint256 n) external returns (bool) {
        if (n < 2) {
            result = 0;
            return false;
        }
        if (n == 2) {
            result = 1;
            return true;
        }
        if (n % 2 == 0) {
            result = 0;
            return false;
        }
        for (uint256 i = 3; i * i <= n; i += 2) {
            if (n % i == 0) {
                result = 0;
                return false;
            }
        }
        result = 1;
        return true;
    }
}

// CHECK: solidity.contract "LoopWithSquaredCondition"
// CHECK: solidity.state_var "result"
// CHECK: sym_name = "squaredLoop"
// CHECK: solidity.constant 3
// CHECK: scf.while
// CHECK: solidity.mul %arg1, %arg1
// CHECK: solidity.cmp "le"
// CHECK: scf.condition
// CHECK: solidity.store_state "result"
// CHECK: scf.yield
// CHECK: sym_name = "isPrime"
// CHECK: solidity.cmp "lt"
// CHECK: solidity.if