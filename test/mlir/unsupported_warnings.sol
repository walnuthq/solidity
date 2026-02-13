// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// Test that unsupported constructs emit warnings rather than silently
/// returning null. Inline assembly and ternary (Conditional) expressions
/// are not yet supported in the MLIR pipeline and should produce warnings.
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

// CHECK: Warning: unsupported statement type in MLIRGen: solidity::frontend::InlineAssembly
// CHECK: Warning: unsupported expression type in MLIRGen: solidity::frontend::Conditional
// CHECK: solidity.contract "WarningTest"
// CHECK: sym_name = "getBalance", visibility = "public"
// CHECK: sym_name = "ternary", visibility = "public"
