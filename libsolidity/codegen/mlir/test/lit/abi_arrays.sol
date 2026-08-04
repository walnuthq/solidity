// Value arrays use their real ABI shape through encode/decode, dispatcher
// calldata materialisation, high-level external calls and dispatcher returns.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract ABIArray {
    function encodeDynamic() external pure returns (bytes32) {
        uint32[] memory values = new uint32[](3);
        values[0] = 7;
        values[1] = 11;
        values[2] = 13;
        return keccak256(abi.encode(uint16(5), values));
    }

    function encodeFixed() external pure returns (bytes32) {
        uint16[3] memory values;
        values[0] = 17;
        values[1] = 19;
        values[2] = 23;
        return keccak256(abi.encode(values, uint8(29)));
    }

    function callDynamic() external view returns (bytes32) {
        uint32[] memory values = new uint32[](2);
        values[0] = 31;
        values[1] = 37;
        return this.acceptDynamic(41, values, "array-call");
    }

    function acceptDynamic(uint16 prefix, uint32[] calldata values, string calldata text)
        external
        pure
        returns (bytes32)
    {
        return keccak256(abi.encode(prefix, values, text));
    }

    function callFixed() external view returns (bytes32) {
        uint16[3] memory values;
        values[0] = 43;
        values[1] = 47;
        values[2] = 53;
        return this.acceptFixed(values);
    }

    function acceptFixed(uint16[3] calldata values) external pure returns (bytes32) {
        return keccak256(abi.encode(values));
    }

    function decodeDynamic() external pure returns (bytes32) {
        string memory text = "decode-me";
        bytes memory encoded = abi.encode(uint256(59), text);
        (uint256 value, string memory decoded) = abi.decode(encoded, (uint256, string));
        return keccak256(abi.encode(value, decoded));
    }
}

// CHECK-LABEL: sym_name = "ABIArray.encodeDynamic"
// Value arrays are copied element-by-element so each narrow element is cleaned
// at its ABI boundary; byte arrays below use the bulk-copy operation.
// CHECK: yul.for
// CHECK-LABEL: sym_name = "ABIArray.callDynamic"
// CHECK: yul.staticcall
// CHECK-LABEL: sym_name = "ABIArray.decodeDynamic"
// CHECK: yul.mload
// CHECK: yul.mcopy
// CHECK: yul.keccak256
