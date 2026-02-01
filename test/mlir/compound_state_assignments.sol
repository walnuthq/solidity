// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CompoundTest {
    uint256 public total;
    function increment(uint256 amount) external {
        total += amount;
    }
    function decrement(uint256 amount) external {
        total -= amount;
    }
    function scale(uint256 factor) external {
        total *= factor;
    }
}

// CHECK: solidity.contract "CompoundTest"
// CHECK: solidity.state_var "total" : !solidity.uint<256>
// CHECK: sym_name = "increment"
// CHECK: solidity.load_state "total"
// CHECK: solidity.add
// CHECK: solidity.store_state "total"
// CHECK: sym_name = "decrement"
// CHECK: solidity.load_state "total"
// CHECK: solidity.sub
// CHECK: solidity.store_state "total"
// CHECK: sym_name = "scale"
// CHECK: solidity.load_state "total"
// CHECK: solidity.mul
// CHECK: solidity.store_state "total"
