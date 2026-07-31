// The evm rung folds the semantic landmines with EVM-exact edges: division by
// zero is zero, not a trap, and a shift past the width is defined.
// RUN: %yul2evm %s --asm | %FileCheck %s

{
    sstore(0, div(100, 0))
    sstore(1, shl(256, 1))
    sstore(2, exp(7, 1))
}

// Nothing survives as a div or a shift - all three fold away.
// CHECK-NOT: DIV
// CHECK-NOT: SHL
// CHECK-NOT: EXP
