// User-defined value types are zero-cost wrappers in the Solidity dialect:
// wrap and unwrap retain the underlying value representation.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s --check-prefix=SOL
// RUN: %sol2evm %s --emit=yul | %FileCheck %s --check-prefix=YUL

type SmallSigned is int8;

contract Types {
    type Counter is uint256;
}

contract ValueTypes {
    function roundTrip() external pure returns (uint256) {
        return Types.Counter.unwrap(Types.Counter.wrap(42));
    }

    function signedRoundTrip() external pure returns (int8) {
        return SmallSigned.unwrap(SmallSigned.wrap(-7));
    }
}

// No nonexistent @wrap/@unwrap calls survive generation.
// SOL-LABEL: sym_name = "roundTrip"
// SOL-NOT: callee = "wrap"
// SOL-NOT: callee = "unwrap"
// SOL: solidity.return
// SOL-LABEL: sym_name = "signedRoundTrip"
// SOL-NOT: callee = "wrap"
// SOL-NOT: callee = "unwrap"
// SOL: solidity.return

// The wrapped uint and signed int reach Yul as their underlying words.
// YUL-LABEL: sym_name = "ValueTypes.roundTrip"
// YUL: yul.const 42
// YUL-LABEL: sym_name = "ValueTypes.signedRoundTrip"
// YUL: yul.const -7
