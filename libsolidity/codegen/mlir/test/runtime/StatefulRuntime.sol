// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.0;

contract RuntimeChild {
    uint256 public value;

    event Bumped(uint256 previous, uint256 current);

    constructor(uint256 initial) payable {
        value = initial;
    }

    function bump(uint256 delta) external returns (uint256 current) {
        uint256 previous = value;
        current = previous + delta;
        value = current;
        emit Bumped(previous, current);
    }
}

contract StatefulRuntime {
    error RuntimeFailure(uint256 mode, uint256 total);

    mapping(uint256 => uint256) public values;
    uint256[] public history;
    bytes public blob;
    uint256 public total;
    RuntimeChild public child;

    event Updated(uint256 indexed key, uint256 previous, uint256 current, bytes32 digest);
    event ChildCreated(address indexed child, uint256 initial, uint256 balance);
    event Received(address indexed sender, uint256 value);

    constructor() payable {
        values[0] = 7;
        history.push(7);
        blob = hex"010203";
        total = 7;
    }

    function setup(uint256 seed) external payable returns (bytes32 digest) {
        uint256 key = seed & 7;
        uint256 previous = values[key];
        values[key] = seed;
        total = total - previous + seed;
        history.push(seed);
        blob = bytes.concat(blob, bytes1(uint8(seed)));
        digest = keccak256(abi.encode(key, seed, blob));
        emit Updated(key, previous, seed, digest);
    }

    function mutate(uint256 key, uint256 delta, bytes calldata data)
        external
        payable
        returns (uint256 current, bytes32 digest)
    {
        uint256 previous = values[key];
        current = previous + delta;
        values[key] = current;
        total += delta;
        history.push(current);
        blob = bytes.concat(blob, data);
        digest = keccak256(abi.encodePacked(key, current, data, blob));
        emit Updated(key, previous, current, digest);
    }

    function branch(uint256 count) external returns (uint256 result) {
        for (uint256 index = 0; index < count; ++index) {
            if (index % 3 == 0) {
                result += index * 2;
                continue;
            }
            result += index;
            if (result > 80)
                break;
        }
        values[9] = result;
        total += result;
        history.push(result);
    }

    function createChild(uint256 initial) external payable returns (address created) {
        require(address(child) == address(0), "child-already-created");
        child = new RuntimeChild{value: msg.value}(initial);
        created = address(child);
        emit ChildCreated(created, initial, created.balance);
    }

    function bumpChild(uint256 delta) external returns (uint256) {
        return child.bump(delta);
    }

    function observe(uint256 key)
        external
        view
        returns (
            uint256 value,
            uint256 entries,
            bytes32 blobHash,
            uint256 aggregate,
            uint256 balance,
            address childAddress,
            uint256 childValue,
            bytes memory blobValue
        )
    {
        value = values[key];
        entries = history.length;
        blobHash = keccak256(blob);
        aggregate = total;
        balance = address(this).balance;
        childAddress = address(child);
        childValue = childAddress == address(0) ? 0 : child.value();
        blobValue = blob;
    }

    function intentionalRevert(uint256 mode) external view {
        if (mode == 1)
            revert RuntimeFailure(mode, total);
        require(mode == 0, "runtime-revert");
    }

    receive() external payable {
        emit Received(msg.sender, msg.value);
    }
}
