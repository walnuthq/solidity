// An event is a log: the signature hash is topic0 unless the event is
// anonymous, each indexed argument is a further topic, and the rest are encoded
// into the data.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Events {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Flat(uint256 a, uint256 b);
    event Anon(uint256 a) anonymous;

    function three() external { emit Transfer(address(1), address(2), 3); }
    function data() external { emit Flat(1, 2); }
    function anon() external { emit Anon(1); }
}

// Two indexed arguments plus the signature hash is three topics, and the value
// is the only word of data.
// CHECK-LABEL: sym_name = "Events.three"
// CHECK: yul.const {{[0-9]+}}
// CHECK: yul.log3

// No indexed arguments: one topic, two words of data.
// CHECK-LABEL: sym_name = "Events.data"
// CHECK: yul.mstore
// CHECK: yul.mstore
// CHECK: yul.log1

// Anonymous means no signature topic at all, so a lone unindexed argument
// leaves nothing to put in one.
// CHECK-LABEL: sym_name = "Events.anon"
// CHECK: yul.log0
