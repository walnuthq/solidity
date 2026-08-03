// Four ways control flow used to build IR the verifier rejects, each costing
// the whole contract rather than the construct.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

contract ControlFlowShape {
    // An empty body still has to yield one value per region argument. It had
    // nothing in the variable map and yielded nothing, so the back edge did
    // not match the region's arity.
    function emptyBody(uint256 n) external pure returns (uint256) {
        uint256 i = 0;
        while (i < n) { }
        return i;
    }
    // CHECK-LABEL: sym_name = "emptyBody"
    // CHECK: scf.yield
    // CHECK-SAME: !solidity.uint<256>

    // A `break` inside an `if` belongs to the `if`, not the loop. Emitting the
    // loop's carried values there put a yield under the wrong parent.
    function breakInBranch(uint256 n) external pure returns (uint256) {
        uint256 i = 0;
        while (i < n) {
            if (i == 3) break;
            i = i + 1;
        }
        return i;
    }
    // CHECK-LABEL: sym_name = "breakInBranch"

    // Statements after a terminator are unreachable; generating them put ops
    // past the terminator.
    function afterReturn(uint256 x) external pure returns (uint256) {
        return x;
    }
    // CHECK-LABEL: sym_name = "afterReturn"

    // A function returning a tuple declares one result per component, so the
    // call site has to as well - collapsing them left the call short.
    function pair() internal pure returns (uint256, uint256) { return (1, 2); }

    function usePair() external pure returns (uint256) {
        (uint256 a, uint256 b) = pair();
        return a + b;
    }
    // CHECK-LABEL: sym_name = "usePair"
    // CHECK: solidity.function_call "ControlFlowShape.pair"
    // CHECK-SAME: -> (!solidity.uint<256>, !solidity.uint<256>)
    // Both components bind to a result of the call - the second used to be a
    // zero constant, so `a + b` quietly read `a + 0`.
    // CHECK: solidity.add %[[C:.*]]#0, %[[C]]#1
}
