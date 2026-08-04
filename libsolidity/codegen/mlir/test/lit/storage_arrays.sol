// A dynamic array keeps its length in its own slot and its elements at
// keccak256(slot), which is what keeps them clear of whatever the layout put
// nearby. A fixed one starts at the slot itself.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Arrays {
    uint256[] dyn;
    uint256[] emptyDyn;
    uint256[3] fixedArr;

    function push(uint256 v) external { dyn.push(v); }
    function pushEmpty() external returns (uint256) { return dyn.push(); }
    function pushThenPop() external returns (uint256) {
        dyn.push(9);
        dyn.pop();
        return dyn.length;
    }
    function popEmpty() external { emptyDyn.pop(); }
    function get(uint256 i) external view returns (uint256) { return dyn[i]; }
    function len() external view returns (uint256) { return dyn.length; }
    function setFixed(uint256 i, uint256 v) external { fixedArr[i] = v; }
}

// Pushing moves the length on and writes where it used to point.
// CHECK-LABEL: sym_name = "Arrays.push"
// CHECK: yul.sload
// CHECK: yul.sstore

// An empty push explicitly clears the returned element and advances the
// length. Its value result is the default zero.
// CHECK-LABEL: sym_name = "Arrays.pushEmpty"
// CHECK: yul.sload
// CHECK: yul.keccak256
// CHECK: yul.sstore
// CHECK: yul.sstore

// Pop checks for an empty array, decrements the length and clears the removed
// word so a later empty push cannot expose stale storage.
// CHECK-LABEL: sym_name = "Arrays.pushThenPop"
// CHECK: yul.iszero
// CHECK: yul.revert
// CHECK: yul.sstore
// CHECK: yul.keccak256
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
