// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ArithmeticTest {
    function testArithmetic(uint256 a, uint256 b) public pure returns (uint256) {
        uint256 result = 0;
        
        // Basic arithmetic
        result += a + b;        // addition
        result += a - b;        // subtraction  
        result += a * b;        // multiplication
        result += a / b;        // division
        result += a % b;        // modulo
        result += a ** 2;       // exponentiation
        
        return result;
    }
    
    function testBitwise(uint256 x, uint256 y) public pure returns (uint256) {
        uint256 result = 0;
        
        // Bitwise operations
        result |= x & y;        // bitwise AND
        result |= x | y;        // bitwise OR
        result |= x ^ y;        // bitwise XOR
        result |= ~x;           // bitwise NOT
        result |= x << 2;       // left shift
        result |= x >> 2;       // right shift
        
        return result;
    }
    
    function testComparisons(uint256 a, uint256 b) public pure returns (bool) {
        bool result = true;
        
        result = result && (a > b);   // greater than
        result = result && (a < b);   // less than
        result = result && (a >= b);  // greater or equal
        result = result && (a <= b);  // less or equal  
        result = result && (a == b);  // equal
        result = result && (a != b);  // not equal
        
        return result;
    }
    
    function testUnaryOps(uint256 x) public pure returns (uint256) {
        uint256 y = x;
        ++y;  // pre-increment
        y++;  // post-increment
        --y;  // pre-decrement
        y--;  // post-decrement
        return y;
    }
}

// CHECK: solidity.contract "ArithmeticTest"
// CHECK: sym_name = "testArithmetic", visibility = "public"
// CHECK: solidity.add %arg0, %arg1
// CHECK: solidity.sub %arg0, %arg1
// CHECK: solidity.mul %arg0, %arg1
// CHECK: solidity.div %arg0, %arg1
// CHECK: solidity.mod %arg0, %arg1
// CHECK: sym_name = "testBitwise"
// CHECK: solidity.not %arg0
// CHECK: sym_name = "testComparisons"
// CHECK: sym_name = "testUnaryOps"
// CHECK: solidity.add
// CHECK: solidity.sub