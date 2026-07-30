{
    // addmod/mulmod use a wide intermediate; exp wraps mod 2^256.
    let big := not(0)
    sstore(0, addmod(big, big, 7))
    sstore(1, mulmod(big, big, 11))
    sstore(2, addmod(5, 3, 0))
    sstore(3, mulmod(5, 3, 0))
    sstore(4, exp(2, 255))
    sstore(5, exp(3, 200))
    sstore(6, byte(31, 0xff))
    sstore(7, signextend(0, 0xff))
    mstore(0, mulmod(big, big, 11))
    return(0, 32)
}
