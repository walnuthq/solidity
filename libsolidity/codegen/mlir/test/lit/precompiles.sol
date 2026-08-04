// Solidity crypto builtins lower to their canonical EVM precompiles, including
// output alignment and malformed ecrecover input returning address(0).
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s --check-prefix=SOL
// RUN: %sol2evm %s --emit=yul | %FileCheck %s --check-prefix=YUL

contract Precompiles {
    function shaEmpty() external view returns (bytes32) {
        return sha256("");
    }

    function ripemdEmpty() external view returns (bytes20) {
        return ripemd160("");
    }

    function invalidRecovery() external pure returns (address) {
        return ecrecover(bytes32(type(uint256).max), 1, bytes32(uint256(2)), bytes32(uint256(3)));
    }
}

// SOL: solidity.sha256
// SOL: solidity.ripemd160
// SOL: solidity.ecrecover

// YUL-LABEL: sym_name = "Precompiles.shaEmpty"
// YUL: yul.staticcall
// YUL-LABEL: sym_name = "Precompiles.ripemdEmpty"
// YUL: yul.staticcall
// YUL: yul.shl
// YUL-LABEL: sym_name = "Precompiles.invalidRecovery"
// YUL: yul.staticcall
