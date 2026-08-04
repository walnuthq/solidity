// In memory an array is a pointer and so is a struct - the same shape `string`
// and `bytes` already had. Nothing needed designing for this; the ops only had
// to say which side of the divide they were on.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Memory {
    struct P { uint256 a; uint256 b; }

    function fixedArray(uint256 i) external pure returns (uint256) {
        uint256[4] memory values;
        values[2] = 5;
        delete values;
        return values[i];
    }

    function makeStruct() external pure returns (uint256) {
        P memory p = P(3, 4);
        return p.b;
    }

    function freshArray() external pure returns (uint256) {
        uint256[] memory values = new uint256[](3);
        return values.length + values[1];
    }

    function freshBytes() external pure returns (bytes32) {
        bytes memory value = new bytes(3);
        return keccak256(value);
    }
}

// A fixed memory array owns the words it addresses - zero is not a pointer, and
// leaving one there put the elements over the free memory pointer at 0x40.
// Allocation advances the free-memory pointer into fresh zeroed memory,
// `values[2] = 5` writes the element, and `delete` writes all four elements -
// rather than yielding a zero and dropping it, which is what made every delete
// a no-op that looked like it worked. The bounds-check panic contributes two
// more stores before the following load.
// CHECK-LABEL: sym_name = "Memory.fixedArray"
// CHECK-COUNT-8: yul.mstore

// No length word in front of a fixed array, so the index is scaled straight off
// the pointer.
// CHECK: yul.mload

// A struct in memory is its fields, one word each, in order.
// CHECK-LABEL: sym_name = "Memory.makeStruct"
// CHECK: yul.mstore
// CHECK: yul.mstore
// CHECK: yul.mload

// A dynamic word array owns a length word and one word per element. Fresh EVM
// memory supplies the zero initialization.
// CHECK-LABEL: sym_name = "Memory.freshArray"
// CHECK: yul.mul
// CHECK: yul.mstore
// CHECK: yul.mload

// Dynamic bytes use the same length-prefixed shape but round their byte-sized
// payload up to a whole memory word.
// CHECK-LABEL: sym_name = "Memory.freshBytes"
// CHECK: yul.and
// CHECK: yul.mstore
// CHECK: yul.keccak256
