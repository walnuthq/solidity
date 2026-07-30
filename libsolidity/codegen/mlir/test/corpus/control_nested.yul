{
    let acc := 0
    for { let i := 0 } lt(i, 5) { i := add(i, 1) } {
        for { let j := 0 } lt(j, 5) { j := add(j, 1) } {
            if eq(j, 3) { break }
            acc := add(acc, mul(i, j))
        }
    }
    sstore(0, acc)
    mstore(0, acc)
    return(0, 32)
}
