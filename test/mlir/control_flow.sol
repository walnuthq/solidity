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
