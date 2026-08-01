// A dropped feature must cost the feature, not the contract. Each of these
// would previously emit IR the dialect could not read back or the verifier
// rejected, losing everything.
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

contract Drops {
    // Falls off the end with a named result: the return has to agree with the
    // declared arity or the verifier rejects it.
    function named() internal pure returns (uint256 r) {}

    function use() external pure returns (uint256) { return named(); }
}

// The function still exists and its return matches the declared arity.
// CHECK: sym_name = "named"
// CHECK: solidity.return
