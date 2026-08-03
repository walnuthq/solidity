// No contract with a loop got past this rung: the generator emits `scf.while`
// for every `for` and `while`, and nothing lowered it.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Loops {
    function sum(uint256 n) external pure returns (uint256) {
        uint256 total = 0;
        for (uint256 i = 0; i < n; i++)
            total = total + i;
        return total;
    }

    // Named result, assigned only inside the branches and never returned
    // explicitly: the implicit return has to pick it up.
    function branch(bool c) external pure returns (uint256 x) {
        if (c) x = 1; else x = 2;
    }
}

// The two dialects disagree about where a loop's state lives. `scf.while`
// carries it as region arguments and yielded results, which is SSA and has no
// place in Yul; `yul.for` has mutable variables instead. So each carried value
// becomes a var, the region arguments read it, and the terminators assign it.
//
// CHECK-LABEL: sym_name = "Loops.sum"
// CHECK: yul.var
// CHECK: yul.var
// CHECK: yul.for

// The condition forwards what the body sees and what the loop yields on exit,
// so it is assigned in the condition region either way.
// CHECK: cond
// CHECK: yul.var_load
// CHECK: yul.assign
// CHECK: yul.condition

// The body assigns what it yields.
// CHECK: body
// CHECK: yul.assign

// After the loop the carried values are read back out of the same slots.
// CHECK: yul.var_load

// A variable assigned in a branch is live after it, so the branches have to
// yield it - the same problem the loop solves by carrying values. Parking it as
// a placeholder meant `if (c) x = 1; else x = 2;` read zero afterwards.
//
// CHECK-LABEL: sym_name = "Loops.branch"
// CHECK: yul.var
// CHECK: yul.if
// CHECK: yul.assign
// CHECK: yul.if
// CHECK: yul.assign
// CHECK: yul.var_load
