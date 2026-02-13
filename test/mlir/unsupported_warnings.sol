// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// Test that unsupported constructs are handled properly.
/// Inline assembly is silently skipped (compiled via normal Yul path).
/// Ternary (Conditional) is supported and produces a SelectOp.
contract WarningTest {
    function getBalance(address addr) public view returns (uint256 bal) {
        assembly {
            bal := balance(addr)
        }
    }

    function ternary(uint256 x) public pure returns (uint256) {
        return x > 0 ? x : 1;
    }
}

// CHECK-NOT: Warning: unsupported statement type in MLIRGen: solidity::frontend::InlineAssembly
// CHECK: solidity.contract "WarningTest"
// CHECK: sym_name = "getBalance", visibility = "public"
// CHECK: sym_name = "ternary", visibility = "public"
// CHECK: solidity.select
