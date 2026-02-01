// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PayableTest {
    mapping(address => uint256) public deposits;
    function deposit() external payable {
        require(msg.value > 0, "No ether sent");
        deposits[msg.sender] += msg.value;
    }
    function getBalance() external view returns (uint256) {
        return deposits[msg.sender];
    }
}

// CHECK: solidity.contract "PayableTest"
// CHECK: solidity.state_var "deposits"
// CHECK: stateMutability = "payable", sym_name = "deposit"
// CHECK: "solidity.msg_value"
// CHECK: solidity.cmp "gt"
// CHECK: solidity.require
// CHECK-SAME: msg = "No ether sent"
// CHECK: "solidity.msg_sender"
// CHECK: "solidity.mapping_access"
// CHECK-SAME: varName = "deposits"
// CHECK: "solidity.msg_value"
// CHECK: solidity.add
// CHECK: "solidity.mapping_store"
// CHECK-SAME: varName = "deposits"
