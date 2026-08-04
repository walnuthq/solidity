// string.concat and bytes.concat allocate fresh dynamic memory and copy each
// argument without ABI length words or padding.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s --check-prefix=SOL
// RUN: %sol2evm %s --emit=yul | %FileCheck %s --check-prefix=YUL

contract Concat {
    function stringHash() external pure returns (bytes32) {
        return keccak256(bytes(string.concat("ab", "c")));
    }

    function bytesHash() external pure returns (bytes32) {
        return keccak256(bytes.concat(hex"01", bytes2(0x0203)));
    }

    function nestedHash() external pure returns (bytes32) {
        return keccak256(bytes(string.concat(string.concat("a", "b"), "c")));
    }
}

// SOL-COUNT-4: solidity.concat
// YUL-LABEL: sym_name = "Concat.stringHash"
// YUL: yul.for
// YUL: yul.keccak256
// YUL-LABEL: sym_name = "Concat.bytesHash"
// YUL: yul.for
// YUL: yul.mstore8
// YUL: yul.keccak256
// YUL-LABEL: sym_name = "Concat.nestedHash"
// YUL: yul.for
// YUL: yul.keccak256
