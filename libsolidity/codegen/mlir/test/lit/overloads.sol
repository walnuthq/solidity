// Overloaded Solidity functions must have distinct internal MLIR/Yul symbols
// while retaining their source ABI signatures in the dispatcher.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s --check-prefix=SOL
// RUN: %sol2evm %s --emit=yul | %FileCheck %s --check-prefix=YUL

contract Overloads {
    function choose(uint256 value) public pure returns (uint256) {
        return value + 1;
    }

    function choose(bool value) public pure returns (uint256) {
        return value ? 7 : 9;
    }

    function uintChoice() external pure returns (uint256) {
        return choose(41);
    }

    function boolChoice() external pure returns (uint256) {
        return choose(true);
    }
}

// The first overload retains the readable source name; subsequent overloads
// receive a stable AST-id suffix.
// SOL: sym_name = "choose"
// SOL: sym_name = "choose$
// SOL-LABEL: sym_name = "uintChoice"
// SOL: solidity.function_call "Overloads.choose"
// SOL-LABEL: sym_name = "boolChoice"
// SOL: solidity.function_call "Overloads.choose$

// ABI dispatch still uses choose(uint256) and choose(bool), not the mangled
// internal names, and the wrappers invoke different Yul functions.
// YUL-COUNT-2: sym_name = "Overloads.choose
// YUL-LABEL: sym_name = "Overloads.uintChoice"
// YUL: yul.func_call @Overloads.choose(
// YUL-LABEL: sym_name = "Overloads.boolChoice"
// YUL: yul.func_call @Overloads.choose$
