// RUN: %solc --mlir-optimize %s 2>&1 | %FileCheck %s
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

// Test Contract Generation
// CHECK: solidity.contract @ArithmeticTest

// Test All Arithmetic Operations
// CHECK: solidity.add
// CHECK: solidity.sub
// CHECK: solidity.mul
// CHECK: solidity.div
// CHECK: solidity.mod
// CHECK: solidity.exp

// Test All Bitwise Operations
// CHECK: solidity.and
// CHECK: solidity.or
// CHECK: solidity.xor
// CHECK: solidity.not
// CHECK: solidity.shl
// CHECK: solidity.shr

// Test All Comparison Operations
// CHECK: solidity.cmp "gt"
// CHECK: solidity.cmp "lt"
// CHECK: solidity.cmp "gte"
// CHECK: solidity.cmp "lte"
// CHECK: solidity.cmp "eq"
// CHECK: solidity.cmp "ne"

// Test Function Declarations
// CHECK: solidity.func @testArithmetic
// CHECK: solidity.func @testBitwise
// CHECK: solidity.func @testComparisons
// CHECK: solidity.func @testUnaryOps

// Test Return Statements
// CHECK: solidity.return