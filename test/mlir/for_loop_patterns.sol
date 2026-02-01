// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ForLoopPatterns {
    uint256 public result;
    function sumToN(uint256 n) external {
        uint256 s = 0;
        for (uint256 i = 1; i <= n; i++) {
            s += i;
        }
        result = s;
    }
    function countDown(uint256 n) external {
        uint256 s = 0;
        for (uint256 i = n; i > 0; i--) {
            s += i;
        }
        result = s;
    }
}

// CHECK: solidity.contract "ForLoopPatterns"
// CHECK: solidity.state_var "result"
// CHECK: sym_name = "sumToN"
// CHECK: scf.while
// CHECK: solidity.cmp "le"
// CHECK: solidity.to_i1
// CHECK: scf.condition
// CHECK: solidity.add
// CHECK: scf.yield
// CHECK: solidity.store_state "result"
// CHECK: sym_name = "countDown"
// CHECK: scf.while
// CHECK: solidity.cmp "gt"
// CHECK: solidity.to_i1
// CHECK: scf.condition
// CHECK: solidity.add
// CHECK: solidity.sub
// CHECK: scf.yield
// CHECK: solidity.store_state "result"
