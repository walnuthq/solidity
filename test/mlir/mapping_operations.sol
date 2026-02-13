// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test Mapping Operations MLIR Codegen
/// @notice Tests that mapping compound assignments generate correct MLIR operations
contract TestMappingOps {

    mapping(address => uint256) public balances;

    // ========== Compound Assignment ==========

    /// @notice Test mapping compound subtraction and addition (transfer pattern)
    function transfer(address to, uint256 amount) public {
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }

    /// @notice Test mapping compound addition with msg.value
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}

// Test Contract Generation
// CHECK: solidity.contract "TestMappingOps"

// Test state variable
// CHECK: solidity.state_var "balances"

// Test mapping_access for balances[msg.sender] read
// CHECK: solidity.mapping_access "balances"[%{{.*}}] : !solidity.address -> !solidity.uint<256>

// Test subtraction for -= operator
// CHECK: solidity.sub %{{.*}}, %{{.*}} : !solidity.uint<256>, !solidity.uint<256> -> !solidity.uint<256>

// Test mapping_store for writing back
// CHECK: solidity.mapping_store "balances"[%{{.*}}], %{{.*}} : !solidity.address, !solidity.uint<256>

// Test mapping_access for balances[to] read
// CHECK: solidity.mapping_access "balances"[%{{.*}}] : !solidity.address -> !solidity.uint<256>

// Test addition for += operator
// CHECK: solidity.add %{{.*}}, %{{.*}} : !solidity.uint<256>, !solidity.uint<256> -> !solidity.uint<256>

// Test mapping_store for writing balances[to]
// CHECK: solidity.mapping_store "balances"[%{{.*}}], %{{.*}} : !solidity.address, !solidity.uint<256>

// Test Function Declarations
// CHECK: sym_name = "deposit"

// Test Return Statements
// CHECK: solidity.return
