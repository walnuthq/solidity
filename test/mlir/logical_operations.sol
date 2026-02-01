// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LogicalOpsTest {
    uint256 public result;
    function testLogicalOps(uint256 a, uint256 b) external {
        if (a > 0 && b > 0) {
            result = a + b;
        }
        if (a == 0 || b == 0) {
            result = 0;
        }
        bool flag = !(a > b);
        if (flag) {
            result = b;
        }
    }
}

// CHECK: solidity.contract "LogicalOpsTest"
// CHECK: solidity.state_var "result"
// CHECK: sym_name = "testLogicalOps"
// CHECK: solidity.cmp "gt"
// CHECK: solidity.cmp "gt"
// CHECK: solidity.if
// CHECK: solidity.add
// CHECK: solidity.store_state "result"
// CHECK: solidity.cmp "eq"
// CHECK: solidity.cmp "eq"
// CHECK: solidity.if
// CHECK: solidity.store_state "result"
// CHECK: solidity.logical_not
// CHECK: solidity.if
// CHECK: solidity.store_state "result"
