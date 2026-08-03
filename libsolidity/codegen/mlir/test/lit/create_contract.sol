// `new C(...)` deploys C's creation code, which is a different object from the
// runtime code `type(C).runtimeCode` names - so it is nested under its own
// name, and that name has to be one the driver can look a contract up by.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Child {
    function v() external pure returns (uint256) { return 42; }
}

contract Parent {
    function spawn() external returns (address) {
        return address(new Child());
    }
}

// The creation code is copied into memory and handed to create, which is what
// makes the child a real deployment rather than a reference to code that is
// already somewhere.
// CHECK-LABEL: sym_name = "Parent.spawn"
// CHECK: yul.datasize "Child_new"
// CHECK: yul.dataoffset "Child_new"
// CHECK: yul.datacopy
// CHECK: yul.create

// The child is carried along in the same sol module so it can be named, but it
// is compiled as its own object - converting it here would append its functions
// after this object's dispatcher, which ends in a revert.
// CHECK-NOT: sym_name = "Child.v"
