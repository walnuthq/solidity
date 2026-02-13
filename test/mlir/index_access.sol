// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test IndexAccess (mapping read) MLIR Codegen
/// @notice Tests that mapping reads generate MappingAccessOp
contract IndexAccessTest {

    mapping(address => uint256) public balances;

    /// @notice Read a mapping value
    function getBalance(address account) external view returns (uint256) {
        return balances[account];
    }

    /// @notice Read and use a mapping value in arithmetic
    function doubleBalance(address account) external view returns (uint256) {
        uint256 bal = balances[account];
        return bal * 2;
    }
}

// CHECK: solidity.contract "IndexAccessTest"
// CHECK: solidity.state_var "balances"

// Test mapping read in getBalance
// CHECK: sym_name = "getBalance"
// CHECK: solidity.mapping_access "balances"[%{{.*}}] : !solidity.address -> !solidity.uint<256>
// CHECK: solidity.return

// Test mapping read in doubleBalance
// CHECK: sym_name = "doubleBalance"
// CHECK: solidity.mapping_access "balances"[%{{.*}}] : !solidity.address -> !solidity.uint<256>
// CHECK: solidity.mul
// CHECK: solidity.return
