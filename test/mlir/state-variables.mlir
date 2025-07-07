// RUN: %solc --emit-mlir %s | %FileCheck %s

// CHECK: solidity.contract @Storage
// CHECK: solidity.state_var @value : \!solidity.uint<256> visibility = "private"
// CHECK: solidity.func @setValue
// CHECK: solidity.store_state @value, %arg0
// CHECK: solidity.func @getValue
// CHECK: %0 = solidity.load_state @value : \!solidity.uint<256>
// CHECK: solidity.return %0

contract Storage {
    uint256 private value;
    
    function setValue(uint256 v) public {
        value = v;
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
}
EOF < /dev/null