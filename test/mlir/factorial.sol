// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FactorialStorage {
    uint256 private result;
    uint256 constant MAX_SAFE_N = 57;

    function computeFactorial(uint256 n) external {
        if (n > MAX_SAFE_N) {
            revert("Overflow: n too large");
        }
        
        result = 1;
        for (uint256 i = 2; i <= n; ++i) {
            result *= i;
        }
    }

    function getResult() external view returns (uint256) {
        return result;
    }
}

// CHECK: solidity.contract "FactorialStorage"
// CHECK: solidity.state_var "result" : !solidity.uint<256>
// CHECK: solidity.state_var "MAX_SAFE_N" : !solidity.uint<256>
// CHECK: sym_name = "computeFactorial"
// CHECK: solidity.load_state "MAX_SAFE_N"
// CHECK: solidity.cmp "gt"
// CHECK: solidity.if
// CHECK: solidity.revert "Overflow: n too large"
// CHECK: solidity.store_state "result"
// CHECK: scf.while
// CHECK: solidity.mul
// CHECK: scf.yield
// CHECK: sym_name = "getResult"
// CHECK: solidity.load_state "result"