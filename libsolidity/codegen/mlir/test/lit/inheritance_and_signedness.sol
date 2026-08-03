// Three things that compiled and answered wrongly, which is worse than not
// compiling: there was nothing to signal any of them.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

interface Marker {
    function ping() external pure;
}

contract A {
    function f() public pure virtual returns (uint256) { return 1; }
}

contract B is A {
    function f() public pure virtual override returns (uint256) { return super.f() | 2; }
}

contract C is A {
    function f() public pure virtual override returns (uint256) { return ((super).f)() | 4; }
}

// `super` walks the most-derived contract's linearisation, so in D the chain is
// D -> C -> B -> A. Resolving it as a direct call or as a virtual one reaches
// the override that is asking, and the function calls itself.
contract D is B, C {
    function f() public pure override(B, C) returns (uint256) { return super.f() | 8; }

    // EVM compares unsigned words with lt and two's complement with slt.
    // int256(-1) < int256(1) asked whether 2^256-1 < 1, which is false where
    // Solidity says true.
    function signedLess() public pure returns (bool) { return int256(-1) < int256(1); }

    // `type(T).interfaceId` is a constant the analyser has already worked out.
    // It is a bytes4, so it sits at the top of the word - right-aligned it
    // reads as zero through the ABI.
    function markerId() public pure returns (bytes4) { return type(Marker).interfaceId; }

    // A contract's own functions are emitted before the base implementations it
    // shadows, so the chain reads D, then C, then B.
    // CHECK-LABEL: sym_name = "f"
    // CHECK: solidity.function_call "C.f"

    // CHECK-LABEL: sym_name = "signedLess"
    // CHECK: solidity.cmp "slt"

    // CHECK-LABEL: sym_name = "markerId"
    // CHECK: solidity.constant {{[0-9]+}} : ui256 : !solidity.bytes<4>

    // The parenthesised form is the same call, and C's own super still steps to
    // B rather than back to itself.
    // CHECK-LABEL: sym_name = "C.f"
    // CHECK: solidity.function_call "B.f"

    // CHECK-LABEL: sym_name = "B.f"
    // CHECK: solidity.function_call "A.f"
}
