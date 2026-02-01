// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

pragma solidity ^0.8.0;

contract Minimal {
    function test() public pure returns (uint256) {
        return 42;
    }
}

// CHECK: solidity.contract "Minimal"
// CHECK: sym_name = "test"
// CHECK: solidity.constant 42
// CHECK: solidity.return
