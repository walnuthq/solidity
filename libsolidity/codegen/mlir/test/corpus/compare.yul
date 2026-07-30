{
    let negone := not(0)
    sstore(0, lt(1, 2))
    sstore(1, gt(1, 2))
    sstore(2, slt(negone, 1))
    sstore(3, sgt(negone, 1))
    sstore(4, eq(7, 7))
    sstore(5, iszero(0))
    sstore(6, iszero(5))
    sstore(7, lt(negone, 1))
    mstore(0, slt(negone, 1))
    return(0, 32)
}
