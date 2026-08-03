// Initialisers and multi-value returns, both of which used to answer zero.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

contract Base {
    // Shadowed by Derived's own `value` - legal because it is private, so it
    // is not in the derived scope. Slots were keyed by name, so the two
    // shared one word and each read whatever the other stored last.
    uint256 private value = 1;
}

contract Derived is Base {
    uint256 private value = 2;

    // Recorded as a decimal string on the declaration before, which only a
    // plain integer literal fits - so this read zero. keccak256 of a literal is
    // a compile-time constant, folded rather than left to the memory allocator
    // that does not exist yet.
    bytes32 private hashed = keccak256("solar");

    // Each contract is its own module and Base has an `init` of its own, so
    // these are sequential checks anchored on Derived's module - a CHECK-LABEL
    // would partition the input at Base's `init` instead.
    // CHECK: solidity.contract "Derived"
    // CHECK: solidity.state_var "value"
    // CHECK: solidity.state_var "Derived.value"

    // Initialisers are their own function, called by the creation half before
    // the constructor body.
    // CHECK: sym_name = "init"
    // CHECK: solidity.store_state "value"
    // CHECK: solidity.store_state "Derived.value"
    // CHECK: solidity.constant {{[0-9]+}} : ui256 : !solidity.bytes<32>
    // CHECK: solidity.store_state "hashed"

    // `return (a, b)` is one value per component. Generated as a single
    // expression it produced only the first, and the arity padding made up the
    // rest with placeholders - so the second result came back zero.
    function pair() external pure returns (uint256, uint256) {
        return (4, 5);
    }
    // CHECK: sym_name = "pair"
    // CHECK: solidity.constant 4
    // CHECK: solidity.constant 5
    // CHECK: solidity.return %{{[0-9]+}}, %{{[0-9]+}}
}
