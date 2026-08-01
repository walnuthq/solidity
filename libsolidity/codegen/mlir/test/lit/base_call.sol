// An explicit base call and a virtual call of the same name must not lower to
// the same callee: Base.value(1) is 2, value(1) inside the override is 101.
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

contract Base {
    function value(uint256 x) internal pure virtual returns (uint256) { return x + 1; }
}

contract Derived is Base {
    function value(uint256 x) internal pure override returns (uint256) { return x + 100; }
    function exact() external pure returns (uint256) { return Base.value(1); }
    function dynamic() external pure returns (uint256) { return value(1); }
}

// CHECK-DAG: solidity.function_call "Base.value"
// CHECK-DAG: solidity.function_call "Derived.value"
