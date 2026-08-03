// The generator serialises the Yul it was given, and the libyul importer turns
// that back into `yul` dialect ops - so this only has to splice them in and
// connect the ends, rather than re-parse Yul by hand.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Assembly {
    function roundTrip(uint256 input) external pure returns (uint256 output) {
        assembly {
            output := add(input, 1)
        }
    }
}

// Values go in through a prologue of declarations, so the block is valid strict
// assembly on its own; the zero each one starts at is then replaced by the
// value the caller actually has. Without that every reference began at zero -
// which compiled, ran, and answered wrongly.
// CHECK-LABEL: sym_name = "Assembly.roundTrip"
// CHECK: yul.var
// CHECK-SAME: yul_name = "input"
// CHECK: yul.var
// CHECK-SAME: yul_name = "output"

// The block's own work, imported rather than re-invented.
// CHECK: yul.add

// And what it assigned is read back out.
// CHECK: yul.var_load
