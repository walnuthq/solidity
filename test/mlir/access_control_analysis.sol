// RUN: %solc --mlir-optimize --mlir-analyze --bin %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract NoAccess {
    address public owner;
    uint256 public criticalParam;

    constructor() {
        owner = msg.sender;
    }

    // ❌ Anyone can change the critical parameter
    function setCriticalParam(uint256 x) external {
        criticalParam = x;
    }

    // ❌ Owner can be overwritten by anyone
    function setOwner(address newOwner) external {
        owner = newOwner;
    }
}

// CHECK: Warning: Function 'setCriticalParam' modifies state without access control (modifies: criticalParam)
// CHECK: Warning: Function 'setOwner' modifies state without access control (modifies: owner)
