// Two things the EVM has no direct representation for.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

contract Target {
    function v() external pure returns (uint256) { return 7; }
}

contract Uses {
    function one() internal pure returns (uint256) { return 1; }
    function two() internal pure returns (uint256) { return 2; }

    // There is no callable value on the EVM. solc uses a jump destination,
    // which works because it lays the code out itself; here every function is
    // a Yul function reached by name, so a pointer is an id and calling one
    // goes through a dispatcher that turns the id back into a call.
    function pick(bool first) external pure returns (uint256) {
        function() internal pure returns (uint256) fn = first ? one : two;
        return fn();
    }
    // CHECK-LABEL: sym_name = "pick"
    // CHECK: solidity.function_call "$call.0.1"

    // `type(C).runtimeCode` names another contract's code as data, so that
    // contract is nested as a sub-object of this one and its size is known
    // when the object is assembled - no reading code from an address.
    function size() external pure returns (uint256) {
        return type(Target).runtimeCode.length;
    }
    // CHECK-LABEL: sym_name = "size"
    // CHECK: solidity.contract_code "Target"
    // The length of a dynamic value is the word its pointer addresses.
    // CHECK: solidity.memory_length

    // One dispatcher per shape, generated after every body because only then is
    // it known which functions had their address taken. An id this contract
    // never handed out cannot be a pointer it made.
    // CHECK-LABEL: sym_name = "$call.0.1"
    // CHECK: solidity.function_call "one"
    // CHECK: solidity.function_call "two"
    // CHECK: solidity.revert
}
