// A constructor's arguments follow the creation code, which is the only place
// they can be: there is no calldata during construction. Both ends of that had
// to agree - `new C(...)` writes them after the code it copies, and the code
// finds its own end to read them back.
//
// RUN: %sol2evm %s --emit=yul | %FileCheck %s

contract Child {
    uint256 public v;
    constructor(uint256 a, uint256 b) { v = a + b; }
}

contract Parent {
    function spawn() external returns (address) { return address(new Child(20, 22)); }
}

// The caller allocates code-plus-arguments, copies the code in, and writes the
// arguments after it - so `create` is handed both as one region.
// CHECK-LABEL: sym_name = "Parent.spawn"
// CHECK: yul.datasize "Child_new"
// CHECK: yul.datacopy
// CHECK: yul.mstore
// CHECK: yul.mstore
// CHECK: yul.create
