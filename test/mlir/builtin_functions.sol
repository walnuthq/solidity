// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test Built-in Functions MLIR Codegen
/// @notice Tests that Solidity built-in functions generate correct MLIR operations
contract TestBuiltinFunctions {

    // ========== Modular Arithmetic ==========

    /// @notice Test addmod - modular addition without overflow
    function testAddMod(uint256 a, uint256 b, uint256 n) public pure returns (uint256) {
        return addmod(a, b, n);
    }

    /// @notice Test mulmod - modular multiplication without overflow
    function testMulMod(uint256 a, uint256 b, uint256 n) public pure returns (uint256) {
        return mulmod(a, b, n);
    }

    // ========== Block/Transaction Functions ==========

    /// @notice Test gasleft - get remaining gas
    function testGasLeft() public view returns (uint256) {
        return gasleft();
    }

    /// @notice Test blockhash - get hash of a block
    function testBlockhash(uint256 blockNumber) public view returns (bytes32) {
        return blockhash(blockNumber);
    }

    // ========== Type Introspection ==========

    /// @notice Test type(uint256).max - maximum value of uint256
    function testTypeMaxUint() public pure returns (uint256) {
        return type(uint256).max;
    }

    /// @notice Test type(int256).min - minimum value of int256
    function testTypeMinInt() public pure returns (int256) {
        return type(int256).min;
    }

    /// @notice Test type(uint8).max - maximum value of uint8
    function testTypeMaxUint8() public pure returns (uint8) {
        return type(uint8).max;
    }
}

// CHECK: solidity.contract "TestBuiltinFunctions"
// CHECK: sym_name = "testAddMod"
// CHECK: solidity.addmod
// CHECK: sym_name = "testMulMod"
// CHECK: solidity.mulmod
// CHECK: sym_name = "testGasLeft"
// CHECK: solidity.gasleft
// CHECK: sym_name = "testBlockhash"
// CHECK: solidity.blockhash
