{
    let total := 0
    let i := 0
    for { } lt(i, 20) { i := add(i, 1) } {
        if eq(i, 5) { i := add(i, 1) continue }
        if eq(i, 15) { break }
        total := add(total, i)
    }
    sstore(0, total)
    sstore(1, i)
    mstore(0, total)
    return(0, 32)
}
