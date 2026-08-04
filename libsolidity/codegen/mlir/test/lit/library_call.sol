// Internal library functions are pulled into the caller object under
// library-qualified symbols and share the caller's memory.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s --check-prefix=SOL
// RUN: %sol2evm %s --emit=yul | %FileCheck %s --check-prefix=YUL

library LibraryCallHelper {
    function setFourth(uint256[] memory values) internal {
        values[3] = 2;
    }
}

contract LibraryCall {
    function run() external returns (uint256) {
        uint256[] memory values = new uint256[](7);
        values[3] = 8;
        LibraryCallHelper.setFourth(values);
        return values[3];
    }
}

// SOL: solidity.function_call "LibraryCallHelper.setFourth"
// SOL: sym_name = "LibraryCallHelper.setFourth"
// YUL: yul.func_call @LibraryCallHelper.setFourth
// YUL: sym_name = "LibraryCallHelper.setFourth"
