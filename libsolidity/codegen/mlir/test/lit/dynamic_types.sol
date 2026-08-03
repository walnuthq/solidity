// `string` and `bytes` do not fit a word, so they live in memory and are named
// by a pointer to [length][data...]. Nothing allocated before this, so nothing
// dynamic worked: the getters were not even describable in the ABI, so the
// dispatcher left them out and every call fell through to the revert.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Dynamic {
    string public name;

    constructor() {
        name = "ERC20Mock";
    }

    function tag() external pure returns (string memory) {
        return "literal";
    }
}

// The free memory pointer holds the start of the heap before the first
// allocation. 0x40 is where Solidity keeps it and 0x80 is where the heap
// starts, leaving 0x00-0x3f as the scratch the mapping slot derivation uses.
// CHECK: yul.const 64
// CHECK: yul.const 128
// CHECK: yul.mstore

// Both directions of storage access go through one helper each, because a
// dynamic value is stored two ways: up to 31 bytes in the slot itself with
// 2*length in the low byte, and from 32 bytes as 2*length+1 with the data at
// keccak256(slot).
// CHECK-DAG: sym_name = "$loadStorageBytes"
// CHECK-DAG: sym_name = "$storeStorageBytes"

// The long form needs a loop, which the short one does not.
// CHECK-DAG: yul.for
