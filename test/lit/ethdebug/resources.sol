// RUN: %solc --experimental --ethdebug-resources --pretty-json --json-indent 2 %s 2>&1 | %FileCheck %s --check-prefix=TEXT

// TEXT: ======= Debug Data (ethdebug/format/info/resources) =======
// TEXT: "compilation": {
// TEXT: "compiler": {
// TEXT: "name": "solc"
// TEXT: "sources": [
// TEXT: "path": "{{.*}}resources.sol"

pragma solidity >=0.0;

contract LitEthdebugResources {
    function f() public {}
}
