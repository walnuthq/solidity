// The external entry point: selector read from calldata, matched per function,
// calldata length checked before the arguments are decoded.
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Counter {
    uint256 public number;
    function setNumber(uint256 newNumber) public { number = newNumber; }
}

// CHECK: yul.calldataload
// CHECK: yul.shr
// A short call must revert rather than decode zeros.
// CHECK: yul.calldatasize
// CHECK: yul.lt
// CHECK: yul.revert
// CHECK: yul.func_call
