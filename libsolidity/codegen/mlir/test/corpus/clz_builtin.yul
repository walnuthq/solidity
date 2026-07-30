{
    sstore(0, clz(1))
    sstore(1, clz(0xff))
    sstore(2, shr(255, 1))
    mstore(0, clz(1))
    return(0, 32)
}
