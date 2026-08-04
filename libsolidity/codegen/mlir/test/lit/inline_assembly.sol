// The generator serialises the Yul it was given, and the libyul importer turns
// that back into `yul` dialect ops - so this only has to splice them in and
// connect the ends, rather than re-parse Yul by hand.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Assembly {
    uint8 private packedA;
    uint32 private packedB;

    function roundTrip(uint256 input) external pure returns (uint256 output) {
        assembly {
            output := add(input, 1)
        }
    }

    function packedLocation() external view returns (uint256 slot, uint256 offset) {
        assembly {
            slot := packedB.slot
            offset := packedB.offset
        }
    }
}

// Values go in through a prologue of declarations, so the block is valid strict
// assembly on its own; the zero each one starts at is then replaced by the
// value the caller actually has. Without that every reference began at zero -
// which compiled, ran, and answered wrongly.
// CHECK-LABEL: sym_name = "Assembly.roundTrip"
// Their declaration order is not semantically significant.
// CHECK-DAG: yul.var {{.*}} {yul_name = "input"}
// CHECK-DAG: yul.var {{.*}} {yul_name = "output"}

// The block's own work, imported rather than re-invented.
// CHECK: yul.add

// And what it assigned is read back out.
// CHECK: yul.var_load

// State-variable suffixes are compile-time layout facts, not variables named
// `packedB.slot` that happen to start at zero. packedB shares slot zero with
// packedA and begins at byte one.
// CHECK-LABEL: sym_name = "Assembly.packedLocation"
// CHECK: yul.const 0
// CHECK: yul.const 1
