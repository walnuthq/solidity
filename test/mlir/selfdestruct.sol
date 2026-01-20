// RUN: %solc --mlir-optimize %s 2>&1 | %FileCheck %s
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

// Test Contract Generation
// CHECK: solidity.contract "TestSelfdestruct"

// Test Selfdestruct Operation
// CHECK: solidity.selfdestruct %{{.*}} : !solidity.address

// Test Function Declaration
// CHECK: sym_name = "destroy"
