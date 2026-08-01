// The creation half stores each state variable's initialiser, runs the
// constructor body, then copies the runtime half out of the object's data and
// returns it - which is what makes that half the deployed code.
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Init {
    uint256 public x = 42;
    uint256 public y;
    constructor() { y = 7; }
}

// The initialiser is recorded on the declaration, not thrown away.
// RUN: %sol2evm %s --emit=sol | %FileCheck %s --check-prefix=SOL
// SOL: solidity.state_var "x"
// SOL-SAME: = "42"

// The runtime half is a dispatcher, so it reads a selector rather than
// returning code.
// CHECK: yul.calldataload
// CHECK: yul.shr
