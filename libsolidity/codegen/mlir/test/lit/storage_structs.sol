// A struct does not fit a word, so a struct in storage is named by a slot
// rather than held as a value. Without that distinction `m[k]` lowered straight
// to an SLOAD, which is already the wrong thing before a member is reached -
// and there was no way to write one back at all.
//
// RUN: %sol2evm %s --emit=sol | %FileCheck %s

contract Structs {
    struct Item {
        bool set;
        uint256 value;
    }

    uint256 public before;
    mapping(bytes32 => Item) private items;
    uint256 public after_;

    function put(bytes32 key, uint256 value) external returns (uint256) {
        Item storage item = items[key];
        item.set = true;
        item.value = value;
        return items[key].value;
    }
}

// The struct occupies the slots its members need, so what follows it does not
// land on top of them: numbering state variables one apiece put `after_` on the
// struct's second member. These come before the functions in the module.
// CHECK: solidity.state_var "before"
// CHECK-SAME: storageSlot = 0
// CHECK: solidity.state_var "items"
// CHECK-SAME: storageSlot = 1
// CHECK: solidity.state_var "after_"
// CHECK-SAME: storageSlot = 2

// The mapping names the place, not what is in it.
// CHECK-LABEL: sym_name = "put"
// CHECK: solidity.mapping_access "items"
// CHECK-SAME: asReference

// Members are read and written at a fixed offset from that place.
// CHECK: solidity.storage_member_store
// CHECK: solidity.storage_member_store
// CHECK: solidity.storage_member_load
