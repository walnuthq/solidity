{
    let x := 5
    if lt(x, 10) {
        mstore(0, 0xbadc0de)
        revert(0, 32)
    }
    mstore(0, 1)
    return(0, 32)
}
