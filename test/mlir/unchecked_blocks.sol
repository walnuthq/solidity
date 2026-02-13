// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// Test that unchecked blocks are properly represented in MLIR
/// as solidity.unchecked regions wrapping their body operations.
contract UncheckedTest {
    function uncheckedAdd(uint256 a, uint256 b) public pure returns (uint256) {
        unchecked {
            return a + b;
        }
    }
}

// CHECK: solidity.contract "UncheckedTest"
// CHECK: sym_name = "uncheckedAdd", visibility = "public"
// CHECK: solidity.unchecked
// CHECK: solidity.add
// CHECK: solidity.return
