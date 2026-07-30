{
    // div/mod by zero yield 0; sdiv(MIN, -1) wraps to MIN.
    let min := 0x8000000000000000000000000000000000000000000000000000000000000000
    let negone := not(0)
    sstore(0, div(100, 0))
    sstore(1, mod(100, 0))
    sstore(2, sdiv(100, 0))
    sstore(3, smod(100, 0))
    sstore(4, sdiv(min, negone))
    sstore(5, smod(min, negone))
    sstore(6, div(100, 7))
    sstore(7, sdiv(negone, 2))
    mstore(0, sdiv(min, negone))
    return(0, 32)
}
