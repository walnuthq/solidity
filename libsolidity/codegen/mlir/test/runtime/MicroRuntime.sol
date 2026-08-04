// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RuntimeFactorial {
    uint256 public result;

    function computeFactorial(uint256 n) external {
        result = 1;
        for (uint256 index = 2; index <= n; ++index)
            result *= index;
    }

    function getResult() external view returns (uint256) {
        return result;
    }
}

contract RuntimeCounter {
    uint256 public count;

    function increment(uint256 times) external {
        for (uint256 index = 0; index < times; ++index)
            count += 1;
    }

    function reset() external {
        count = 0;
    }
}

contract RuntimeSumRange {
    uint256 public total;

    function sumRange(uint256 start, uint256 end) external {
        total = 0;
        for (uint256 index = start; index <= end; ++index)
            total += index;
    }
}

contract RuntimeArithmetic {
    uint256 public value;

    function compute(uint256 left, uint256 right, uint256 iterations) external {
        value = left;
        for (uint256 index = 0; index < iterations; ++index) {
            value = (value * right + left) / 2;
            value = value % 1_000_000 + 1;
        }
    }
}
