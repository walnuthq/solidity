// RUN: %solc --mlir-optimize %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test Hash Functions MLIR Codegen
/// @notice Tests that Solidity hash functions generate correct MLIR operations
contract TestHashFunctions {

    // ========== Hash Functions ==========

    /// @notice Test keccak256 - Keccak-256 hash
    function testKeccak256(uint256 value) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(value));
    }

    /// @notice Test sha256 - SHA-256 hash (precompile 0x02)
    function testSha256(uint256 value) public pure returns (bytes32) {
        return sha256(abi.encodePacked(value));
    }

    /// @notice Test ripemd160 - RIPEMD-160 hash (precompile 0x03)
    function testRipemd160(uint256 value) public pure returns (bytes20) {
        return ripemd160(abi.encodePacked(value));
    }

    // ========== Cryptographic Recovery ==========

    /// @notice Test ecrecover - recover address from signature (precompile 0x01)
    function testEcrecover(
        bytes32 hash,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) public pure returns (address) {
        return ecrecover(hash, v, r, s);
    }
}

// Test Contract Generation
// CHECK: solidity.contract "TestHashFunctions"

// Test Hash Function Operations - keccak256
// CHECK: solidity.keccak256 %{{.*}} : !solidity.array<!solidity.bytes<1>, -1> -> !solidity.bytes<32>

// Test Hash Function Operations - sha256 (precompile 0x02)
// CHECK: solidity.sha256 %{{.*}} : !solidity.array<!solidity.bytes<1>, -1> -> !solidity.bytes<32>

// Test Hash Function Operations - ripemd160 (precompile 0x03)
// CHECK: solidity.ripemd160 %{{.*}} : !solidity.array<!solidity.bytes<1>, -1> -> !solidity.bytes<20>

// Test Cryptographic Recovery - ecrecover (precompile 0x01)
// CHECK: solidity.ecrecover %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !solidity.bytes<32>, !solidity.uint<8>, !solidity.bytes<32>, !solidity.bytes<32> -> !solidity.address

// Test Function Declarations
// CHECK: sym_name = "testKeccak256"
// CHECK: sym_name = "testSha256"
// CHECK: sym_name = "testRipemd160"
// CHECK: sym_name = "testEcrecover"

// Test Return Statements
// CHECK: solidity.return
