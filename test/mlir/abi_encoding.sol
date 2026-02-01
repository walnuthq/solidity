// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Test ABI Encoding and Keccak256 MLIR Codegen
/// @notice Tests that abi.encode, abi.encodePacked, and keccak256 generate correct MLIR operations
contract TestAbiEncoding {

    // ========== ABI Encoding ==========

    /// @notice Test abi.encode with two uint256 values
    function testAbiEncode(uint256 a, uint256 b) public pure returns (bytes memory) {
        return abi.encode(a, b);
    }

    /// @notice Test abi.encodePacked with address and uint256
    function testAbiEncodePacked(address addr, uint256 val) public pure returns (bytes memory) {
        return abi.encodePacked(addr, val);
    }

    // ========== Keccak256 with ABI Encoding ==========

    /// @notice Test keccak256(abi.encode(a, b)) - fused pattern
    function testKeccak256AbiEncode(uint256 a, uint256 b) public pure returns (bytes32) {
        return keccak256(abi.encode(a, b));
    }

    /// @notice Test keccak256(abi.encodePacked(addr, val)) - fused pattern with packed encoding
    function testKeccak256AbiEncodePacked(address addr, uint256 val) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(addr, val));
    }

    /// @notice Test keccak256(abi.encodePacked(a, b, c)) - three-value packed encoding
    function testKeccak256ThreePacked(uint256 a, uint256 b, uint256 c) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(a, b, c));
    }

    // ========== Standalone Encoding ==========

    /// @notice Test standalone abi.encode returning bytes memory
    function testStandaloneEncode(uint256 x) public pure returns (bytes memory) {
        return abi.encode(x);
    }

    /// @notice Test standalone abi.encodePacked returning bytes memory
    function testStandaloneEncodePacked(uint256 x) public pure returns (bytes memory) {
        return abi.encodePacked(x);
    }
}

// Test Contract Generation
// CHECK: solidity.contract "TestAbiEncoding"

// Test abi.encode with two uint256 values
// CHECK: "solidity.abi_encode"(%{{.*}}, %{{.*}}) : (!solidity.uint<256>, !solidity.uint<256>) -> !solidity.array<!solidity.bytes<1>, -1>

// Test abi.encodePacked with address and uint256
// CHECK: "solidity.abi_encode_packed"(%{{.*}}, %{{.*}}) : (!solidity.address, !solidity.uint<256>) -> !solidity.array<!solidity.bytes<1>, -1>

// Test keccak256(abi.encode(a, b)) - fused pattern
// CHECK: "solidity.abi_encode"(%{{.*}}, %{{.*}}) : (!solidity.uint<256>, !solidity.uint<256>) -> !solidity.array<!solidity.bytes<1>, -1>
// CHECK-NEXT: %{{.*}} = solidity.keccak256 %{{.*}} : !solidity.array<!solidity.bytes<1>, -1> -> !solidity.bytes<32>

// Test keccak256(abi.encodePacked(addr, val)) - fused pattern with packed encoding
// CHECK: "solidity.abi_encode_packed"(%{{.*}}, %{{.*}}) : (!solidity.address, !solidity.uint<256>) -> !solidity.array<!solidity.bytes<1>, -1>
// CHECK-NEXT: %{{.*}} = solidity.keccak256 %{{.*}} : !solidity.array<!solidity.bytes<1>, -1> -> !solidity.bytes<32>

// Test keccak256(abi.encodePacked(a, b, c)) - three-value packed encoding
// CHECK: "solidity.abi_encode_packed"(%{{.*}}, %{{.*}}, %{{.*}}) : (!solidity.uint<256>, !solidity.uint<256>, !solidity.uint<256>) -> !solidity.array<!solidity.bytes<1>, -1>
// CHECK-NEXT: %{{.*}} = solidity.keccak256 %{{.*}} : !solidity.array<!solidity.bytes<1>, -1> -> !solidity.bytes<32>

// Test standalone abi.encode
// CHECK: "solidity.abi_encode"(%{{.*}}) : (!solidity.uint<256>) -> !solidity.array<!solidity.bytes<1>, -1>

// Test standalone abi.encodePacked
// CHECK: "solidity.abi_encode_packed"(%{{.*}}) : (!solidity.uint<256>) -> !solidity.array<!solidity.bytes<1>, -1>

// Test Function Declarations
// CHECK: sym_name = "testAbiEncode"
// CHECK: sym_name = "testAbiEncodePacked"
// CHECK: sym_name = "testKeccak256AbiEncode"
// CHECK: sym_name = "testKeccak256AbiEncodePacked"
// CHECK: sym_name = "testKeccak256ThreePacked"
// CHECK: sym_name = "testStandaloneEncode"
// CHECK: sym_name = "testStandaloneEncodePacked"

// Test Return Statements
// CHECK: solidity.return
