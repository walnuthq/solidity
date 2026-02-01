// RUN: %solc --mlir-optimize --print-mlir %s 2>&1 | %FileCheck %s
// REQUIRES: mlir

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Comprehensive test exercising all gas benchmark patterns
// This tests: storage loops, ERC20-style ops, math operations, nested mappings
contract GasBenchmarkPatterns {
    // State variables
    uint256 public result;
    uint256 public counter;
    uint256[] public values;
    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowances;

    // Counter operations with storage caching opportunity
    function incrementCounter(uint256 times) external {
        for (uint256 i = 0; i < times; i++) {
            counter++;
        }
    }

    // Factorial with storage caching
    function computeFactorial(uint256 n) external {
        result = 1;
        for (uint256 i = 2; i <= n; ++i) {
            result *= i;
        }
    }

    // Sum range with storage writes
    function sumRange(uint256 start, uint256 end) external {
        result = 0;
        for (uint256 i = start; i <= end; ++i) {
            result += i;
        }
    }

    // Mixed arithmetic operations
    function compute(uint256 a, uint256 b, uint256 iterations) external {
        result = a;
        for (uint256 i = 0; i < iterations; ++i) {
            result = (result * b + a) / 2;
            result = result % 1000000 + 1;
        }
    }

    // ERC20-style operations
    function approve(address spender, uint256 amount) public returns (bool) {
        allowances[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        balances[msg.sender] -= amount;
        unchecked { balances[to] += amount; }
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        uint256 allowed = allowances[from][msg.sender];
        if (allowed != type(uint256).max) {
            allowances[from][msg.sender] = allowed - amount;
        }
        balances[from] -= amount;
        unchecked { balances[to] += amount; }
        return true;
    }

    function mint(address to, uint256 amount) public {
        unchecked { balances[to] += amount; }
    }

    // Math-intensive operations
    function fibonacci(uint256 n) external {
        if (n <= 1) {
            result = n;
            return;
        }
        uint256 a = 0;
        uint256 b = 1;
        for (uint256 i = 2; i <= n; i++) {
            uint256 temp = a + b;
            a = b;
            b = temp;
        }
        result = b;
    }

    function isPrime(uint256 n) external returns (bool) {
        if (n < 2) {
            result = 0;
            return false;
        }
        if (n == 2) {
            result = 1;
            return true;
        }
        if (n % 2 == 0) {
            result = 0;
            return false;
        }
        for (uint256 i = 3; i * i <= n; i += 2) {
            if (n % i == 0) {
                result = 0;
                return false;
            }
        }
        result = 1;
        return true;
    }

    function power(uint256 base, uint256 exp) external {
        result = 1;
        for (uint256 i = 0; i < exp; i++) {
            result *= base;
        }
    }

    function gcd(uint256 a, uint256 b) external {
        while (b != 0) {
            uint256 temp = b;
            b = a % b;
            a = temp;
        }
        result = a;
    }
}

// CHECK: solidity.contract "GasBenchmarkPatterns"
// CHECK: solidity.state_var "result"
// CHECK: solidity.state_var "counter"
// CHECK: solidity.state_var "values"
// CHECK: solidity.state_var "balances"
// CHECK: sym_name = "incrementCounter"
// CHECK: scf.while
// CHECK: solidity.load_state "counter"
// CHECK: solidity.store_state "counter"
// CHECK: sym_name = "computeFactorial"
// CHECK: solidity.store_state "result"
// CHECK: scf.while
// CHECK: solidity.mul
// CHECK: sym_name = "sumRange"
// CHECK: sym_name = "compute"