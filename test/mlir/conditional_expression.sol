// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test Conditional (ternary) MLIR Codegen
/// @notice Tests that ternary expressions generate SelectOp
contract ConditionalTest {

    /// @notice Simple ternary: return the max of two values
    function max(uint256 a, uint256 b) external pure returns (uint256) {
        return a > b ? a : b;
    }

    /// @notice Ternary with zero fallback
    function nonZero(uint256 x) external pure returns (uint256) {
        return x > 0 ? x : 1;
    }
}

// CHECK: solidity.contract "ConditionalTest"

// Test ternary in max function
// CHECK: sym_name = "max"
// CHECK: solidity.cmp
// CHECK: solidity.select
// CHECK: solidity.return

// Test ternary in nonZero function
// CHECK: sym_name = "nonZero"
// CHECK: solidity.cmp
// CHECK: solidity.select
// CHECK: solidity.return
