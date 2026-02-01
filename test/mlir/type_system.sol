// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TypeSystemTest {
    // Test various Solidity types
    uint8 public smallInt;
    uint256 public largeInt;
    int128 public signedInt; 
    bool public flag;
    address public owner;
    bytes32 public hash;
    bytes public dynamicBytes;
    string public name;
    
    // Test array types
    uint256[] public dynamicArray;
    uint256[10] public fixedArray;
    
    // Test mapping types
    mapping(address => uint256) public balances;
    mapping(uint256 => string) public names;
    
    function testTypes() public {
        smallInt = 255;
        largeInt = 2**256 - 1;
        signedInt = -1000;
        flag = true;
        owner = msg.sender;
        hash = keccak256("test");
        dynamicBytes = "hello";
        name = "contract";
        
        dynamicArray.push(42);
        fixedArray[0] = 100;
        
        balances[msg.sender] = 1000;
        names[1] = "first";
    }
}

// CHECK: solidity.contract "TypeSystemTest"
// CHECK: solidity.state_var "smallInt" : !solidity.uint<8>
// CHECK: solidity.state_var "largeInt" : !solidity.uint<256>
// CHECK: solidity.state_var "signedInt" : !solidity.int<128>
// CHECK: solidity.state_var "flag" : !solidity.bool
// CHECK: solidity.state_var "owner" : !solidity.address
// CHECK: solidity.state_var "hash" : !solidity.bytes<32>
// CHECK: solidity.state_var "dynamicArray" : !solidity.array<!solidity.uint<256>, -1>
// CHECK: solidity.state_var "fixedArray" : !solidity.array<!solidity.uint<256>, 10>
// CHECK: sym_name = "testTypes"
// CHECK: solidity.store_state "smallInt"
// CHECK: solidity.store_state "largeInt"
// CHECK: solidity.store_state "signedInt"
// CHECK: solidity.store_state "flag"
// CHECK: solidity.msg_sender
// CHECK: solidity.store_state "owner"
// CHECK: solidity.keccak256
// CHECK: solidity.array_push
// CHECK: solidity.mapping_store