{
    let a := 0x0123456789abcdef
    let b := 0xfedcba9876543210
    sstore(0, add(a, b))
    sstore(1, sub(a, b))
    sstore(2, mul(a, b))
    sstore(3, and(a, b))
    sstore(4, or(a, b))
    sstore(5, xor(a, b))
    sstore(6, not(a))
    mstore(0, add(a, b))
    return(0, 32)
}
