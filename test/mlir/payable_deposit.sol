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
// CHECK: solidity.msg_value : !solidity.uint<256>
// CHECK: solidity.cmp "gt"
// CHECK: solidity.require
// CHECK-SAME: "No ether sent"
// CHECK: solidity.msg_sender : !solidity.address
// CHECK: solidity.mapping_access "deposits"
// CHECK: solidity.msg_value : !solidity.uint<256>
// CHECK: solidity.add
// CHECK: solidity.mapping_store "deposits"
