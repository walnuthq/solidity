// RUN: %solc --experimental --ethdebug-program --ethdebug-program-runtime --via-ir --pretty-json --json-indent 2 %s 2>&1 | %FileCheck %s --check-prefix=TEXT

// TEXT-DAG: Debug Data (ethdebug/format/program):
// TEXT-DAG: Debug Data of the runtime part (ethdebug/format/program):
// TEXT-DAG: "environment": "create"
// TEXT-DAG: "environment": "call"
// TEXT-DAG: "name": "LitEthdebugProgram"
// TEXT-DAG: "instructions": [
// TEXT-DAG: "offset":
// TEXT-DAG: "operation": {
// TEXT-DAG: "mnemonic":

pragma solidity >=0.0;

contract LitEthdebugProgram {
    function value() public pure returns (uint256) {
        return 42;
    }
}
