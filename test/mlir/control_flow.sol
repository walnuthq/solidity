// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

pragma solidity >=0.8.0;
contract Test {
    function test() public pure returns (uint256) {
        uint256 i = 0;
        do {
            i = i + 1;
        } while (i < 5);
        return i;
    }
}

// CHECK: solidity.contract "Test"
// CHECK: sym_name = "test"
// CHECK: solidity.constant 0
// CHECK: scf.while
// CHECK: solidity.constant 1
// CHECK: solidity.add
// CHECK: scf.yield
// CHECK: solidity.return
