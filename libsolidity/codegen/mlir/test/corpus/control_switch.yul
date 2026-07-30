{
    function classify(v) -> r {
        switch v
        case 0 { r := 100 }
        case 1 { r := 200 }
        default { r := 300 }
    }
    sstore(0, classify(0))
    sstore(1, classify(1))
    sstore(2, classify(9))
    mstore(0, classify(9))
    return(0, 32)
}
