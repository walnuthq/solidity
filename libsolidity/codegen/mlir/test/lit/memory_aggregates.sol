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
}

// A fixed memory array owns the words it addresses - zero is not a pointer, and
// leaving one there put the elements over the free memory pointer at 0x40.
// The four words are written on declaration, `values[2] = 5` writes a fifth
// time, and `delete` writes four more - rather than yielding a zero and
// dropping it, which is what made every delete a no-op that looked like it
// worked.
// CHECK-LABEL: sym_name = "Memory.fixedArray"
// CHECK-COUNT-9: yul.mstore

// No length word in front of a fixed array, so the index is scaled straight off
// the pointer.
// CHECK: yul.mload

// A struct in memory is its fields, one word each, in order.
// CHECK-LABEL: sym_name = "Memory.makeStruct"
// CHECK: yul.mstore
// CHECK: yul.mstore
// CHECK: yul.mload
