// High-level external calls ABI-encode calldata, choose STATICCALL for a
// view/pure target, decode all fixed-word returns, and bubble failure data.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract ExternalCallee {
    function double(uint256 value) external pure returns (uint256) { return value * 2; }
    function words() external pure returns (uint256, uint256) { return (7, 9); }
    function bounce(bytes calldata value) external pure returns (bytes memory) { return value; }
}

contract ExternalCaller {
    ExternalCallee private callee;

    constructor() { callee = new ExternalCallee(); }

    function answer() external view returns (uint256) { return callee.double(21); }
    function pair() external view returns (uint256, uint256) { return callee.words(); }
    function bounceHash() external view returns (bytes32) {
        return keccak256(callee.bounce(hex"01020304"));
    }
}

// CHECK-LABEL: sym_name = "ExternalCaller.answer"
// CHECK: yul.mstore
// CHECK: yul.mcopy
// CHECK: yul.staticcall
// CHECK: yul.returndatasize
// CHECK: yul.returndatacopy

// CHECK-LABEL: sym_name = "ExternalCaller.pair"
// CHECK: yul.staticcall
// CHECK: yul.mload
// CHECK: yul.mload

// Dynamic arguments are ABI head/tail encoded and dynamic returndata is copied,
// bounds-checked and materialized as a Solidity memory byte array.
// CHECK-LABEL: sym_name = "ExternalCaller.bounceHash"
// CHECK: yul.mcopy
// CHECK: yul.staticcall
// CHECK: yul.returndatasize
// CHECK: yul.returndatacopy
// CHECK: yul.mload
// CHECK: yul.mcopy
// CHECK: yul.keccak256
