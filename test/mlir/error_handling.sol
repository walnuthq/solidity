// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ErrorTest {
    function testRequire(uint256 x) external pure returns (uint256) {
        require(x > 0, "Must be positive");
        return x;
    }
    function testRevert(uint256 x) external pure returns (uint256) {
        if (x == 0) revert("Zero not allowed");
        return x;
    }
    function testAssert(uint256 x) external pure returns (uint256) {
        assert(x != 0);
        return x;
    }
}

// CHECK: solidity.contract "ErrorTest"
// CHECK: sym_name = "testRequire"
// CHECK: solidity.cmp "gt"
// CHECK: solidity.require
// CHECK-SAME: "Must be positive"
// CHECK: sym_name = "testRevert"
// CHECK: solidity.cmp "eq"
// CHECK: solidity.if
// CHECK: solidity.revert "Zero not allowed"
// CHECK: sym_name = "testAssert"
// CHECK: solidity.cmp "ne"
// CHECK: solidity.assert
