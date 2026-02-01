// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test Selfdestruct MLIR Codegen
/// @notice Tests that selfdestruct generates correct MLIR operation
contract TestSelfdestruct {
    address payable public owner;

    constructor() {
        owner = payable(msg.sender);
    }

    /// @notice Test selfdestruct - destroy contract and send balance
    function destroy() public {
        selfdestruct(owner);
    }
}

// CHECK: solidity.contract "TestSelfdestruct"
// CHECK: solidity.state_var "owner" : !solidity.address
// CHECK: sym_name = "destroy"
// CHECK: solidity.load_state "owner"
// CHECK: solidity.selfdestruct
