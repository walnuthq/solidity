// Logical operators evaluate their right-hand side only when Solidity says
// they should, and assignments performed there remain live afterward.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s --check-prefix=SOL
// RUN: %sol2evm %s --emit=yul | %FileCheck %s --check-prefix=YUL

contract ShortCircuit {
    function orSkips() external pure returns (uint256) {
        uint256 x = 0;
        bool result = x == 0 || ((x = 8) > 0);
        return result ? x : 99;
    }

    function orTakes() external pure returns (uint256) {
        uint256 x = 1;
        bool result = x == 0 || ((x = 8) > 0);
        return result ? x : 99;
    }

    function andSkips() external pure returns (uint256) {
        uint256 x = 1;
        bool result = x == 0 && ((x = 8) > 0);
        return result ? 99 : x;
    }

    function andTakes() external pure returns (uint256) {
        uint256 x = 0;
        bool result = x == 0 && ((x = 8) > 0);
        return result ? x : 99;
    }
}

// Both logical forms become result-carrying SCF branches, not eager logical
// ops whose RHS has already run.
// SOL-COUNT-4: scf.if
// SOL-NOT: solidity.logical_and
// SOL-NOT: solidity.logical_or

// The SCF branches become guarded Yul regions and mutable result slots.
// YUL-LABEL: sym_name = "ShortCircuit.orSkips"
// YUL: yul.if
// YUL: yul.if
// YUL-LABEL: sym_name = "ShortCircuit.andTakes"
// YUL: yul.if
// YUL: yul.if
