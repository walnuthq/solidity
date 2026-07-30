{
    function pair(a, b) -> x, y {
        x := add(a, b)
        y := mul(a, b)
    }
    function apply(v) -> r {
        let p, q := pair(v, 3)
        r := add(p, q)
    }
    function early(v) -> r {
        r := 1
        if lt(v, 10) { leave }
        r := 2
    }
    let p, q := pair(6, 7)
    sstore(0, p)
    sstore(1, q)
    sstore(2, apply(4))
    sstore(3, early(5))
    sstore(4, early(50))
    mstore(0, apply(4))
    return(0, 32)
}
