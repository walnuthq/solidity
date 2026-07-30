{
    // Shifts of 256 or more are defined: zero, or sign for SAR.
    let negone := not(0)
    sstore(0, shl(256, 1))
    sstore(1, shr(256, negone))
    sstore(2, sar(256, negone))
    sstore(3, sar(300, 5))
    sstore(4, shl(8, 1))
    sstore(5, shr(4, 0xff))
    sstore(6, sar(4, negone))
    mstore(0, sar(256, negone))
    return(0, 32)
}
