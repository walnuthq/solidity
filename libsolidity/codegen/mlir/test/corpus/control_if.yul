{
    let x := 0
    if lt(3, 5) { x := add(x, 10) }
    if gt(3, 5) { x := add(x, 100) }
    if iszero(x) { x := 999 }
    sstore(0, x)
    mstore(0, x)
    return(0, 32)
}
