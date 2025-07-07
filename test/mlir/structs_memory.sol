// RUN: %solc --mlir-optimize %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract StructMemoryTest {
    struct Item {
        uint256 x;
        uint256 y;
    }
    
    mapping(uint256 => Item) public items;
    uint256[] public values;

    function computeWithStruct() public pure returns (uint256 sum) {
        for (uint256 i = 0; i < 10; ++i) {
            Item memory item = Item({x: i, y: i * 2});
            sum += item.x + item.y;
        }
    }
    
    function testArrayOperations() public {
        values.push(42);
        values[0] = 100;
        uint256 len = values.length;
        require(len > 0, "Array should not be empty");
    }
    
    function testMappingOperations(uint256 key) public {
        items[key] = Item({x: key, y: key * 3});
        Item storage item = items[key];
        item.x = item.x + 1;
    }
}

// Test Contract Generation
// CHECK: solidity.contract @StructMemoryTest

// Test Struct Type Usage
// CHECK: !solidity.struct<"Item">

// Test State Variable Declarations
// CHECK: solidity.state_var "items" : !solidity.mapping
// CHECK: solidity.state_var "values" : !solidity.array

// Test Function Declarations
// CHECK: solidity.func @computeWithStruct
// CHECK: solidity.func @testArrayOperations  
// CHECK: solidity.func @testMappingOperations

// Test Struct Creation and Member Access
// CHECK: solidity.struct_create
// CHECK: solidity.member_access

// Test Array Operations
// CHECK: solidity.array_push
// CHECK: solidity.array_store
// CHECK: solidity.array_length

// Test Mapping Operations  
// CHECK: solidity.mapping_store
// CHECK: solidity.mapping_access

// Test Arithmetic in Loop Context
// CHECK: solidity.mul
// CHECK: solidity.add

// Test For Loop with Struct Operations
// CHECK: solidity.for
// CHECK: solidity.cmp "lt"

// Test Require Statement
// CHECK: solidity.require