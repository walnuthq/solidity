// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
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

// CHECK: solidity.contract "StructMemoryTest"
// CHECK: solidity.state_var "items"
// CHECK: solidity.state_var "values"
// CHECK: sym_name = "computeWithStruct"
// CHECK: scf.while
// CHECK: solidity.struct_create "Item"
// CHECK: solidity.member_access {{.*}} "x"
// CHECK: solidity.member_access {{.*}} "y"
// CHECK: solidity.add
// CHECK: scf.yield
// CHECK: sym_name = "testArrayOperations"
// CHECK: solidity.array_push
// CHECK: solidity.array_length
// CHECK: solidity.require