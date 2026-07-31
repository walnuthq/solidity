// A mapping element lives at keccak256(key . slot), computed over the scratch
// region the language reserves at 0x00 and 0x20.
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract M {
    mapping(address => uint256) balances;
    function get(address who) public view returns (uint256) { return balances[who]; }
}

// CHECK: yul.mstore
// CHECK: yul.mstore
// CHECK: yul.keccak256
// CHECK: yul.sload
