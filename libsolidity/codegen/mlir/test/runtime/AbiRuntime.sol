// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.0;

contract AbiRuntime {
    error MissingRecord(bytes32 key);

    mapping(bytes32 => bytes) private records;
    uint256 public sum;
    string public label;

    event Stored(bytes32 indexed key, bytes value, bytes32 digest);
    event Accumulated(uint256 count, uint256 sum, string label);

    function store(bytes32 key, bytes calldata value) external returns (bytes32 digest) {
        records[key] = value;
        digest = keccak256(value);
        emit Stored(key, value, digest);
    }

    function accumulate(uint256[] calldata numbers, string calldata newLabel)
        external
        returns (uint256 current)
    {
        current = sum;
        for (uint256 index = 0; index < numbers.length; ++index)
            current += numbers[index];
        sum = current;
        label = newLabel;
        emit Accumulated(numbers.length, current, newLabel);
    }

    function read(bytes32 key)
        external
        view
        returns (bytes memory value, uint256 current, string memory text)
    {
        value = records[key];
        current = sum;
        text = label;
    }

    function digest(bytes32 key) external view returns (bytes32) {
        return keccak256(records[key]);
    }

    function failIfMissing(bytes32 key) external view {
        if (records[key].length == 0)
            revert MissingRecord(key);
    }
}
