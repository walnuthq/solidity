# solc MLIR pipeline — the dialect ladder

MLIR-based compilation pipeline for solc, targeting Vitalik's "EVM as a
compiler-level feature" endgame (see `SolcRISCV.md` proposal):

```
sol dialect  ->  yul dialect  ->  evm dialect  ->  RISC-V (via LLVM)
   (rung 1)        (rung 2)         (rung 3)         (rung 4)
```

Every rung stays executable or emittable so each conversion is
differentially testable: the `yul` rung emits compilable Yul text (keeping
today's EVM bytecode backend reachable as the differential anchor), the
`evm` rung lowers to a runnable RISC-V ELF.

## Layout

| Path | Content | Status |
|---|---|---|
| `Dialect/Sol/` | `sol` dialect (ported from `feature/mlir-pipeline`) | ported (M0) |
| `Dialect/Yul/` | `yul` dialect: builtins 1:1, if/for CF, mutable vars, immutables | working (M1) |
| `Dialect/EVM/` | `evm` dialect: machine-level EVM, landmine-exact ops | working (M3) |
| `Target/YulText/` | `yul` dialect -> Yul source emitter | working (M1) |
| `Target/RISCV/` | LLVM-dialect -> RV32IM objects (in-process llc) + test wrappers | working (M5) |
| `Import/LibyulAST/` | libyul AST -> `yul` dialect (incl. object trees, real via-ir) | working (M2b) |
| `Conversion/SolToYul/` | `sol` -> `yul` (storage slots, cmp predicates, if/else, require) | scaffolded (M2) |
| `Conversion/YulToEVM/` | block-local var promotion + CF flattening + builtin mapping | scaffolded (M4) |
| `Conversion/EVMToLLVM/` | landmine legalization + evm-rt calls + clamped shifts | working (M5, pure subset) |
| `runtime/evm-rt/` | i256/i512 runtime in LLVM IR (div family, exp, byte, signextend) | working (M5) |
| `hosts/risc0/` | RISC Zero guest+host: zkVM execution, cycle counts, receipts | scaffolded (M7 groundwork) |
| `tools/` | 5 test drivers + `yul2rv` (ladder-stage reporter for real contracts) | working |

Real-contract coverage (see `riscv_bench.py` in solidity-compiler-benchmarks):
**82/82 objects** of the 30-contract suite import + verify; all reach
`promote` — next unlock is the region-aware variable promotion pass
(loop-carried vars -> block args), then module-level statements in
YulToEVM (`evm.program` objects), then ERHI host ops in EVMToLLVM (M6).

The landmine semantics are execution-proven: `evm-riscv-test` JITs the
compiled functions against an independent APInt oracle (24/24 vectors) and
links a real RV32IM ELF (rv32im/ilp32, `ld.lld`).

## Design decisions

See `ADRs.md`. Key ones:

- **ADR-003**: `yul` dialect ops mirror Yul builtins 1:1 over signless
  `i256`; upstream `arith` reuse happens at the `evm` rung.
- **ADR-004**: structured CF as region ops (`yul.if`, `yul.for` with
  `break`/`continue`/`leave` terminators); Yul's mutable locals modeled with
  `yul.var`/`yul.assign`/`yul.var_load` (clang-style; SSA promotion happens
  before the `evm` rung), which makes the libyul-AST importer a trivial walk
  with no phi construction.

## Build & test

MLIR support is optional and auto-detected (see `cmake/FindMLIR.cmake`;
Homebrew `llvm` with MLIR works out of the box):

```sh
cmake -S . -B build            # "MLIR support enabled" in output
cmake --build build --target yul-dialect-test && ./build/libsolidity/codegen/mlir/tools/yul-dialect-test
```

The test builds sample functions as `yul` dialect IR, verifies, round-trips
through the MLIR printer/parser, emits Yul text, and validates the result
with libyul's parser + analyzer.
