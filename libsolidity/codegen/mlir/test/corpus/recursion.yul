{
    function fact(n) -> r {
        switch lt(n, 2)
        case 1 { r := 1 }
        default { r := mul(n, fact(sub(n, 1))) }
    }
    function fib(n) -> r {
        switch lt(n, 2)
        case 1 { r := n }
        default { r := add(fib(sub(n, 1)), fib(sub(n, 2))) }
    }
    // Locals must survive the recursive call that clobbers the shared frame.
    function sumdown(n) -> total {
        let keep := mul(n, 3)
        switch n
        case 0 { total := 0 }
        default { total := add(keep, sumdown(sub(n, 1))) }
    }
    sstore(0, fact(6))
    sstore(1, fib(12))
    sstore(2, sumdown(10))
    mstore(0, fib(12))
    return(0, 32)
}
