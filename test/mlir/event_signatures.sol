// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test Event Emission MLIR Codegen
/// @notice Tests that Solidity event emissions generate correct MLIR operations with proper signatures and indexing
contract TestEventSignatures {

    // ========== Event Declarations ==========

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Deposit(address indexed account, uint256 amount);
    event AllIndexed(address indexed a, uint256 indexed b, bool indexed c);
    event NoIndexed(address from, address to, uint256 value);

    // ========== Event Emissions ==========

    /// @notice Test Transfer event with two indexed params
    function emitTransfer(address from, address to, uint256 value) public {
        emit Transfer(from, to, value);
    }

    /// @notice Test Deposit event with one indexed param
    function emitDeposit(address account, uint256 amount) public {
        emit Deposit(account, amount);
    }

    /// @notice Test event with all indexed params
    function emitAllIndexed(address a, uint256 b, bool c) public {
        emit AllIndexed(a, b, c);
    }

    /// @notice Test event with no indexed params
    function emitNoIndexed(address from, address to, uint256 value) public {
        emit NoIndexed(from, to, value);
    }
}

// Test Contract Generation
// CHECK: solidity.contract "TestEventSignatures"

// Test Transfer event - two indexed, one non-indexed
// CHECK: solidity.emit "Transfer"(%{{.*}}, %{{.*}}, %{{.*}}) {eventSignature = "Transfer(address,address,uint256)", indexed = [true, true, false]} : !solidity.address, !solidity.address, !solidity.uint<256>

// Test Deposit event - one indexed, one non-indexed
// CHECK: solidity.emit "Deposit"(%{{.*}}, %{{.*}}) {eventSignature = "Deposit(address,uint256)", indexed = [true, false]} : !solidity.address, !solidity.uint<256>

// Test AllIndexed event - all params indexed
// CHECK: solidity.emit "AllIndexed"(%{{.*}}, %{{.*}}, %{{.*}}) {eventSignature = "AllIndexed(address,uint256,bool)", indexed = [true, true, true]} : !solidity.address, !solidity.uint<256>, !solidity.bool

// Test NoIndexed event - no params indexed
// CHECK: solidity.emit "NoIndexed"(%{{.*}}, %{{.*}}, %{{.*}}) {eventSignature = "NoIndexed(address,address,uint256)", indexed = [false, false, false]} : !solidity.address, !solidity.address, !solidity.uint<256>

// Test Return Statements
// CHECK: solidity.return
