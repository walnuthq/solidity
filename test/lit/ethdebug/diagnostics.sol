// RUN: not %solc --debug-info ethdebug %s 2>&1 | %FileCheck %s --check-prefix=NO-EXPERIMENTAL
// RUN: not %solc --experimental --debug-info ethdebug --via-ssa-cfg %s 2>&1 | %FileCheck %s --check-prefix=SSA-CFG

// NO-EXPERIMENTAL: Ethdebug annotations are experimental
// SSA-CFG: ethdebug is not yet supported with --via-ssa-cfg

pragma solidity >=0.0;

contract LitEthdebugDiagnostics {
    function f() public {}
}
