// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CfgTest {
    function classify(uint256 x) public pure returns (uint256) {
        if (x > 100) {
            return 3;
        } else if (x > 50) {
            return 2;
        } else if (x > 10) {
            return 1;
        }
        return 0;
    }
    function earlyReturn(uint256 a, uint256 b) public pure returns (uint256) {
        if (a == 0) return b;
        if (b == 0) return a;
        return a + b;
    }
}

// CHECK: solidity.contract "CfgTest"
// CHECK: sym_name = "classify"
// CHECK: solidity.constant 100
// CHECK: solidity.cmp "gt"
// CHECK: solidity.if
// CHECK: solidity.constant 3
// CHECK: solidity.return
// CHECK: solidity.constant 50
// CHECK: solidity.cmp "gt"
// CHECK: solidity.if
// CHECK: solidity.constant 2
// CHECK: solidity.return
// CHECK: solidity.constant 10
// CHECK: solidity.cmp "gt"
// CHECK: solidity.if
// CHECK: solidity.constant 1
// CHECK: solidity.constant 0
// CHECK: sym_name = "earlyReturn"
// CHECK: solidity.cmp "eq"
// CHECK: solidity.if
// CHECK: solidity.cmp "eq"
// CHECK: solidity.add
