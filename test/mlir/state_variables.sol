// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StateVarTest {
    uint256 public counter;
    bool public paused;
    address public owner;
    int256 public balance;
    function setCounter(uint256 v) external { counter = v; }
    function getCounter() external view returns (uint256) { return counter; }
    function togglePause() external { paused = !paused; }
    function setOwner(address a) external { owner = a; }
}

// CHECK: solidity.contract "StateVarTest"
// CHECK: solidity.state_var "counter" : !solidity.uint<256>
// CHECK: solidity.state_var "paused" : !solidity.bool
// CHECK: solidity.state_var "owner" : !solidity.address
// CHECK: solidity.state_var "balance" : !solidity.int<256>
// CHECK: sym_name = "setCounter"
// CHECK: solidity.store_state "counter"
// CHECK: sym_name = "getCounter"
// CHECK: solidity.load_state "counter" : !solidity.uint<256>
// CHECK: sym_name = "togglePause"
// CHECK: solidity.load_state "paused" : !solidity.bool
// CHECK: solidity.logical_not
// CHECK: solidity.store_state "paused"
// CHECK: sym_name = "setOwner"
// CHECK: solidity.store_state "owner"
