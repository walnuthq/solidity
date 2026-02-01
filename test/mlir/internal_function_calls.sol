// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract InternalCallTest {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
    function double(uint256 x) public pure returns (uint256) {
        return add(x, x);
    }
}

// CHECK: solidity.contract "InternalCallTest"
// CHECK: sym_name = "add", visibility = "internal"
// CHECK: solidity.add
// CHECK: solidity.return
// CHECK: sym_name = "double"
// CHECK: "solidity.function_call"
// CHECK-SAME: callee = "add"
// CHECK: solidity.return
