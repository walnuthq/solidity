// Low-level calls preserve calldata and return the full (success, returndata)
// pair without turning a callee revert into a caller revert.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s --check-prefix=SOL
// RUN: %sol2evm %s --emit=yul | %FileCheck %s --check-prefix=YUL

contract LowLevelCall {
    function answer() external pure returns (uint256) {
        return 7;
    }

    function callAnswer() external returns (bool, uint256) {
        (bool success, bytes memory data) = address(this).call(
            abi.encodeWithSignature("answer()")
        );
        return (success, success ? abi.decode(data, (uint256)) : 0);
    }

    function missingFails() external returns (bool) {
        (bool success, bytes memory data) = address(this).call(hex"deadbeef");
        return !success && data.length == 0;
    }

    function transferZero() external returns (bool) {
        payable(address(1)).transfer(0);
        return true;
    }
}

// SOL-COUNT-2: solidity.low_level_call
// YUL-LABEL: sym_name = "LowLevelCall.callAnswer"
// YUL: yul.call
// YUL: yul.returndatasize
// YUL: yul.returndatacopy
// YUL-LABEL: sym_name = "LowLevelCall.missingFails"
// YUL: yul.call
// YUL: yul.returndatasize
// YUL-LABEL: sym_name = "LowLevelCall.transferZero"
// YUL: yul.call
