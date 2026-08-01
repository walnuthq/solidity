// The three internal-call semantics inheritance creates. An explicit base call
// reaches the shadowed implementation; a virtual call reaches the override;
// and a virtual call inside an INHERITED body re-resolves against the contract
// it was inherited into, not the one that declared it.
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

contract Base {
    function value(uint256 x) internal pure virtual returns (uint256) { return x + 1; }
}

contract Derived is Base {
    function value(uint256 x) internal pure override returns (uint256) { return x + 100; }
    function exact() external pure returns (uint256) { return Base.value(1); }
    function dynamic() external pure returns (uint256) { return value(1); }
}

abstract contract A2 {
    function hook() public pure virtual returns (uint256);
    function through() public pure returns (uint256) { return hook(); }
}

contract C2 is A2 {
    function hook() public pure override returns (uint256) { return 7; }
}

// The shadowed base implementation exists under its qualified name...
// CHECK-DAG: sym_name = "Base.value"
// ...and the two calls in Derived reach different functions.
// CHECK-DAG: solidity.function_call "Base.value"
// CHECK-DAG: solidity.function_call "Derived.value"
// The inherited body's virtual call re-resolved to the override.
// CHECK-DAG: solidity.function_call "C2.hook"
