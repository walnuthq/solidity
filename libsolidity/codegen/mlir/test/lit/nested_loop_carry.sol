// A variable assigned inside a nested loop is modified by the enclosing loop
// too, so both have to carry it. When only the inner loop did, the outer loop
// yielded fewer values than it needed and the code after it named the inner
// loop's result - a value out of scope there, which cost the whole contract at
// the first rung.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

contract NestedLoopCarry {
    // `total` is assigned only in the inner loop, `i` only by the outer one.
    // Both cross the outer loop's back edge, so it carries two values and the
    // sum read afterwards is the outer loop's own result.
    function sum(uint256 n, uint256 m) external pure returns (uint256) {
        uint256 total = 0;
        for (uint256 i = 0; i < n; i++)
            for (uint256 j = 0; j < m; j++)
                total = total + j;
        return total;
    }

    // CHECK-LABEL: sym_name = "sum"
    // The outer loop carries the accumulator as well as its counter.
    // CHECK: scf.while
    // CHECK-SAME: -> (!solidity.uint<256>, !solidity.uint<256>)

    // The inner loop's counter is declared inside it and does not outlive it,
    // so it must not be added to what the outer loop carries.
    // CHECK: scf.while
    // CHECK-SAME: -> (!solidity.uint<256>, !solidity.uint<256>)
}
