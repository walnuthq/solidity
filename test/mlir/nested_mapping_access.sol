// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract NestedMapTest {
    mapping(address => mapping(address => uint256)) public allowances;
    function approve(address spender, uint256 amount) external {
        allowances[msg.sender][spender] = amount;
    }
    function getAllowance(address owner, address spender) external view returns (uint256) {
        return allowances[owner][spender];
    }
}

// CHECK: solidity.contract "NestedMapTest"
// CHECK: solidity.state_var "allowances"
// CHECK: sym_name = "approve"
// CHECK: solidity.mapping_store
// CHECK: solidity.return
// CHECK: sym_name = "getAllowance"
// CHECK: solidity.return
