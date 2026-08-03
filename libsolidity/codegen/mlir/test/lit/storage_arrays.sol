// A dynamic array keeps its length in its own slot and its elements at
// keccak256(slot), which is what keeps them clear of whatever the layout put
// nearby. A fixed one starts at the slot itself.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Arrays {
    uint256[] dyn;
    uint256[3] fixedArr;

    function push(uint256 v) external { dyn.push(v); }
    function get(uint256 i) external view returns (uint256) { return dyn[i]; }
    function len() external view returns (uint256) { return dyn.length; }
    function setFixed(uint256 i, uint256 v) external { fixedArr[i] = v; }
}

// Pushing moves the length on and writes where it used to point.
// CHECK-LABEL: sym_name = "Arrays.push"
// CHECK: yul.sload
// CHECK: yul.sstore
// CHECK: yul.keccak256
// CHECK: yul.sstore

// Reading an element hashes the slot to find the data.
// CHECK-LABEL: sym_name = "Arrays.get"
// CHECK: yul.keccak256
// CHECK: yul.add
// CHECK: yul.sload

// The length is the slot's own contents - no hashing.
// CHECK-LABEL: sym_name = "Arrays.len"
// CHECK-NOT: yul.keccak256
// CHECK: yul.sload

// A fixed array starts at its slot, so the index is added straight to it.
// CHECK-LABEL: sym_name = "Arrays.setFixed"
// CHECK-NOT: yul.keccak256
// CHECK: yul.add
// CHECK: yul.sstore
