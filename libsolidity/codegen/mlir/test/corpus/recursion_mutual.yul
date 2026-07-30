{
    function isEven(n) -> r {
        switch n
        case 0 { r := 1 }
        default { r := isOdd(sub(n, 1)) }
    }
    function isOdd(n) -> r {
        let scratch := add(n, 100)
        switch n
        case 0 { r := 0 }
        default { r := isEven(sub(n, 1)) }
        // scratch must still be intact after the mutual call chain
        sstore(9, scratch)
    }
    sstore(0, isEven(10))
    sstore(1, isOdd(10))
    sstore(2, isEven(7))
    sstore(3, isOdd(7))
    mstore(0, isEven(10))
    return(0, 32)
}
