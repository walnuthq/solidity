// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MsgTest {
    function getSender() external view returns (address) {
        return msg.sender;
    }
    function getValue() external payable returns (uint256) {
        return msg.value;
    }
}

// CHECK: solidity.contract "MsgTest"
// CHECK: stateMutability = "view", sym_name = "getSender"
// CHECK: "solidity.msg_sender"() : () -> !solidity.address
// CHECK: solidity.return
// CHECK: stateMutability = "payable", sym_name = "getValue"
// CHECK: "solidity.msg_value"() : () -> !solidity.uint<256>
// CHECK: solidity.return
