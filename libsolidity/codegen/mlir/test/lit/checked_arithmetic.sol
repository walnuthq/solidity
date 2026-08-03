// Arithmetic lowered unchecked, so every contract wrapped silently where
// Solidity >= 0.8 must panic. Nothing in the corpus overflowed, which is why
// the differential never caught it - the reason to do this deliberately rather
// than wait for evidence.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Arithmetic {
    function add(uint256 a, uint256 b) external pure returns (uint256) {
        return a + b;
    }

    function wrapping(uint256 a, uint256 b) external pure returns (uint256) {
        unchecked { return a + b; }
    }

    function divide(uint256 a, uint256 b) external pure returns (uint256) {
        return a / b;
    }
}

// A sum below either operand wrapped. The failure is reported as
// `Panic(uint256)` - selector 0x4e487b71, then 0x11 - because reverting bare
// would be the right control flow with the wrong answer to "why".
// CHECK-LABEL: sym_name = "Arithmetic.add"
// CHECK: yul.add
// CHECK: yul.lt
// CHECK: yul.if
// CHECK: yul.const 35408467139433450592217433187231851964531694900788300625387963629091585785856
// CHECK: yul.const 17
// CHECK: yul.revert

// `unchecked` has to reach the lowering. The block is inlined by the generator,
// so the operations themselves carry the mark - and none of the above appears.
// CHECK-LABEL: sym_name = "Arithmetic.wrapping"
// CHECK: yul.add
// CHECK-NOT: yul.revert

// EVM defines x/0 as 0; Solidity does not allow it at all. 0x12, not 0x11.
// CHECK-LABEL: sym_name = "Arithmetic.divide"
// CHECK: yul.iszero
// CHECK: yul.if
// CHECK: yul.const 18
// CHECK: yul.revert
