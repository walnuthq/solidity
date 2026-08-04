// Abstract declarations are ordinary members of real multi-file project
// closures. The production --mlir-bin driver must skip their empty creation
// output and continue compiling the concrete contracts in the same input.

abstract contract AbstractRuntime {
    function value() external view virtual returns (uint256);
}

interface RuntimeInterface {
    function touch(uint256 input) external returns (uint256);
}

contract ConcreteRuntime is AbstractRuntime {
    uint256 private stored = 7;

    function value() external view override returns (uint256) {
        return stored;
    }
}
