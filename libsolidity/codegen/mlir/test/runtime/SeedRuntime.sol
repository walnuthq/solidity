// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

abstract contract RuntimeSeedBase {
    event Seen(uint256 indexed tag, uint256 value);

    uint256 public value;
    mapping(uint256 => uint256) public values;

    function setup(uint256 seed) external {
        unchecked {
            value = seed & 1023;
            values[seed & 7] = seed + 1;
            emit Seen(0, value);
        }
    }

    function observe(uint256 key) external view returns (uint256, uint256) {
        return (value, values[key & 7]);
    }

    function helper(uint256 input) internal pure returns (uint256) {
        unchecked {
            return (input * 7) ^ 3;
        }
    }

    function mix(uint256 left, uint256 right) internal pure returns (uint256) {
        unchecked {
            return (left ^ right) + (left & 15);
        }
    }

    function run(uint256 left, uint256 right, bytes calldata data)
        external
        virtual
        returns (uint256 result);
}

contract RuntimeSeedBranch is RuntimeSeedBase {
    function run(uint256 left, uint256 right, bytes calldata)
        external
        override
        returns (uint256 result)
    {
        unchecked {
            if ((left & 3) == 0)
                result = mix(left, value);
            else if ((right & 3) == 0)
                result = mix(right, value);
            else
                result = mix(left, right);
            value = result;
            values[left & 7] = result;
            emit Seen(1, result);
        }
    }
}

contract RuntimeSeedBytes is RuntimeSeedBase {
    function run(uint256 left, uint256, bytes calldata data)
        external
        override
        returns (uint256 result)
    {
        unchecked {
            uint256 limit = data.length < 4 ? data.length : 4;
            result = value;
            for (uint256 index = 0; index < limit; ++index)
                result = (result << 1) + uint8(data[index]);
            value = result;
            values[left & 7] = result;
            emit Seen(1, result);
        }
    }
}

contract RuntimeSeedBytesLoop is RuntimeSeedBase {
    function run(uint256 left, uint256, bytes calldata data)
        external
        override
        returns (uint256 result)
    {
        unchecked {
            uint256 limit = data.length < 5 ? data.length : 5;
            result = value;
            for (uint256 index = 0; index < limit; ++index)
                result += uint8(data[index]);
            value = result;
            values[left & 7] = result;
            emit Seen(1, result);
        }
    }
}

contract RuntimeSeedCombined is RuntimeSeedBase {
    function run(uint256 left, uint256 right, bytes calldata)
        external
        override
        returns (uint256 result)
    {
        unchecked {
            uint256 key = left & 7;
            uint256[2] memory items = [values[key], helper(right)];
            result = items[0] + items[1] + value;
            value = result;
            values[key] = result;
            emit Seen(1, result);
        }
    }
}

contract RuntimeSeedLoop is RuntimeSeedBase {
    function run(uint256 left, uint256 right, bytes calldata)
        external
        override
        returns (uint256 result)
    {
        unchecked {
            uint256 limit = left & 7;
            result = value;
            for (uint256 index = 0; index < limit; ++index)
                result += index + right;
            value = result;
            values[left & 7] = result;
            emit Seen(1, result);
        }
    }
}

contract RuntimeSeedMapping is RuntimeSeedBase {
    function run(uint256 left, uint256 right, bytes calldata)
        external
        override
        returns (uint256 result)
    {
        unchecked {
            uint256 key = left & 7;
            values[key] = values[key] + right + 1;
            result = values[key] + value;
            value = result;
            values[key] = result;
            emit Seen(1, result);
        }
    }
}

contract RuntimeSeedMemoryArray is RuntimeSeedBase {
    function run(uint256 left, uint256 right, bytes calldata)
        external
        override
        returns (uint256 result)
    {
        unchecked {
            uint256[2] memory items;
            items[0] = left + 1;
            items[1] = right + 2;
            result = items[(left ^ right) & 1] + value;
            value = result;
            values[left & 7] = result;
            emit Seen(1, result);
        }
    }
}

contract RuntimeSeedStorageLoop is RuntimeSeedBase {
    function run(uint256 left, uint256 right, bytes calldata)
        external
        override
        returns (uint256 result)
    {
        unchecked {
            uint256 limit = (left & 3) + 1;
            result = values[right & 7];
            for (uint256 index = 0; index < limit; ++index)
                result += mix(index, value);
            value = result;
            values[left & 7] = result;
            emit Seen(1, result);
        }
    }
}
