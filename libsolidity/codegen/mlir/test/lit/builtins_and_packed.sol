// Builtins that already have exact Yul/EVM operations must not stop at the sol
// rung, and packed encoding must preserve byte widths and alignment.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract BuiltinsAndPacked {
    function modular(uint256 a, uint256 b, uint256 n) external pure returns (uint256, uint256) {
        return (addmod(a, b, n), mulmod(a, b, n));
    }

    function context(address account, uint256 oldBlock) external payable returns (
        address, address, address, uint256, uint256, uint256, uint256, bytes32, uint256, uint256, bytes32
    ) {
        return (
            address(this), msg.sender, tx.origin, msg.value,
            block.timestamp, block.number, block.chainid, blockhash(oldBlock),
            tx.gasprice, account.balance, account.codehash
        );
    }

    function selector() external pure returns (bytes4) { return msg.sig; }

    function gasIsAvailable() external view returns (bool) { return gasleft() > 0; }

    function destroy(address payable recipient) external { selfdestruct(recipient); }

    function packedHash() external pure returns (bytes32) {
        return keccak256(abi.encodePacked(
            uint32(0x11223344),
            address(0x1234567890123456789012345678901234567890),
            "abc",
            bytes4(0xaabbccdd)
        ));
    }

    function standardHash() external pure returns (bytes32) {
        return keccak256(abi.encode(uint32(0x11223344), "abc", bytes4(0xaabbccdd)));
    }

    function messageHash() external pure returns (bytes32) { return keccak256(msg.data); }

    function emptyCodeHash() external view returns (bytes32) { return keccak256(address(0).code); }

    function negate() external pure returns (bool) { return !false; }

    function selectorEncodingHash() external pure returns (bytes32) {
        return keccak256(abi.encodeWithSelector(bytes4(0x12345678), uint32(7), "abc"));
    }

    function signatureEncodingHash() external pure returns (bytes32) {
        return keccak256(abi.encodeWithSignature("f(uint32,string)", uint32(7), "abc"));
    }
}

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.modular"
// CHECK: yul.addmod
// CHECK: yul.mulmod

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.context"
// CHECK: yul.address
// CHECK: yul.caller
// CHECK: yul.origin
// CHECK: yul.callvalue
// CHECK: yul.timestamp
// CHECK: yul.number
// CHECK: yul.chainid
// CHECK: yul.blockhash
// CHECK: yul.gasprice
// CHECK: yul.balance
// CHECK: yul.extcodehash

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.selector"
// CHECK: yul.calldataload
// CHECK: yul.shr
// CHECK: yul.shl

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.gasIsAvailable"
// CHECK: yul.gas

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.destroy"
// CHECK: yul.selfdestruct

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.packedHash"
// CHECK: yul.mstore8
// CHECK: yul.keccak256

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.standardHash"
// CHECK: yul.mcopy
// CHECK: yul.keccak256

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.messageHash"
// CHECK: yul.calldatasize
// CHECK: yul.calldatacopy

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.emptyCodeHash"
// CHECK: yul.extcodesize
// CHECK: yul.extcodecopy

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.negate"
// CHECK: yul.iszero

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.selectorEncodingHash"
// CHECK: yul.mstore
// CHECK: yul.mcopy
// CHECK: yul.keccak256

// CHECK-LABEL: sym_name = "BuiltinsAndPacked.signatureEncodingHash"
// CHECK: yul.keccak256
// CHECK: yul.mcopy
// CHECK: yul.keccak256
