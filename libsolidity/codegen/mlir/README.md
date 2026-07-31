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
**82/82 objects of the 30-contract suite compile to RV32IM object files**
(`riscv` stage). Loop-carried variables are promoted via structured-CF SSA
construction inside YulToEVM (merge points become block arguments),
module-level object code synthesizes `@__entry`, and the full evm op set
lowers through the ERHI v0 by-pointer C ABI (`__evm_<op>`; landmines keep
optimized `__evm_rt_*` routes).

Execution-proven, twice:
- `evm-riscv-test`: landmine functions JIT-executed against an independent
  APInt oracle - 24/24 vectors exact; static RV32 ELF links (`ld.lld`).
- `erhi-host-test`: a realistic deployed object (selector dispatcher,
  keccak mapping slots, storage, loops, reverts) executes against a native
  ERHI host - state transitions, revert-rollback, and returndata all exact
  (18/18).

Next: ERHI host ops in the RISC Zero guest (hosts/risc0), evmone/revm
differential harness, gas modes, call family via host re-entry.

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

## The EVM assembly rung (`Target/EVM`)

`Target/EVM/EVMAssemblyEmitter` closes the ladder on the EVM side:

```
Yul text -> yul dialect -> evm dialect -> libevmasm Assembly -> bytecode
```

No stage leaves MLIR until the final instruction stream, so the EVM backend is
now reachable without the `Target/YulText` detour through libyul.

**Value placement.** Every non-constant SSA value gets a fixed 32-byte slot in
a per-function frame; operands are materialised at each use and results written
back, while constants are rematerialised as pushes. Memory is therefore always
authoritative, which makes stack-too-deep unreachable and means no scheduling is
needed for correctness — scheduling only removes traffic.

**The stack model.** `m_stack` mirrors the physical slots a block has pushed
above its own baseline, each holding the value it carries or null for a machine
word with no SSA identity. A value is materialised as a literal, as a `DUP` of a
slot the stack already holds, or failing both by reloading it — and because
every value is also in its frame slot, a lookup that misses or lands past
`DUP16` costs correctness nothing. A result with a later use in the same block
is kept with one `DUP1` before its store; results nothing reads are popped
rather than stored.

Two cases sit on top of it. A result whose single use is the very next
instruction skips its slot entirely: operands are pushed back to front, so it
can serve as the deepest operand for free or as the lower of a pair for one
`SWAP1`. And the cache is dropped — `flushStack` — before every call and every
non-halting branch, because a callee runs with our stack beneath it and `JUMPI`
consumes only its condition, so anything cached underneath would survive into a
successor that expects a bare stack.

That last point is also why the win is bounded. Yul from solc branches and calls
often, so straight-line runs are short and the cache rarely lives long enough to
repay itself; emitted code is still 3-4x the reference. Going further means not
storing a value at all when every use can be proven reachable on the stack, and
carrying a layout across block boundaries — solar's stack-phi planner in
`backend/evm/stack/mod.rs` is the design document for that.

**Calling convention.** Arguments and results live in the callee's frame; only
the return address travels on the stack. Frames are addressed absolutely, so a
function that can re-enter itself would share slots between two live
activations: those functions are found up front, and a call into one banks the
callee's frame before the call and puts it back after. Results are parked on the
stack across the restore, because in a self-call their destination slot is
inside the frame being restored.

**Object model.** The Yul object tree survives the import (`ImportedObject`
carries its sub-objects and data segments), and objects are emitted leaf-first
so a parent can register each child as a sub-assembly before referring to it.
`dataoffset`/`datasize` become relocations - `PushSub`/`PushSubSize` for nested
objects, a data item for `data` segments, and `appendProgramSize` when an object
names itself - `datacopy` is a `CODECOPY`, and immutables map to
`appendImmutable`/`appendImmutableAssignment`. Creation objects therefore
compile and deploy.

**Memory layout.** `memoryguard(x)` is the start of the contract's heap:
memory below it is reserved, memory above it is allocated from. The frames go
at `x` and the guard is rewritten to `x + frame size`, moving the heap above
them — which is why `evm.memoryguard` stays an op rather than being folded to
its literal. Placing the frames *at* the guard does not work: that is where the
heap begins and they collide. An object with no guard that provably never
touches memory gets the frames at `0x80`; anything else keeps a fixed high base,
as do objects with recursion, whose saved frames grow without a static bound so
nothing can sit above them.

This matters beyond code size: a fallback reached by a plain `send` has 2300 gas,
which is not enough to expand memory out to a fixed high address before doing
anything else.

**Known divergences.** `MSIZE` observes the frame region. `semanticTests/various/
code_access_content.sol` hashes its own runtime code and so cannot match a
different backend by construction.

### Validation

Two instruments, both under `test/`:

```sh
anvil --silent --port 8546 &

# Differential: same Yul through solc's backend and through the ladder,
# both executed, comparing returndata, halt status and storage.
python3 libsolidity/codegen/mlir/test/evm_differential.py \
    --corpus libsolidity/codegen/mlir/test/corpus \
    --solc build/solc/solc \
    --yul2evm build/libsolidity/codegen/mlir/tools/yul2evm

# Coverage: how far every Yul object of a real contract gets down the ladder.
python3 libsolidity/codegen/mlir/test/corpus_coverage.py \
    --solc build/solc/solc \
    --yul2evm build/libsolidity/codegen/mlir/tools/yul2evm \
    --corpus <dir-of-sol-files>
```

Current status:

- **15/15 differential match** on the hand-written corpus, which covers the
  §5 landmines (div/mod by zero, `SDIV(MIN,-1)`, shifts >= 256, addmod/mulmod
  wide intermediates, `EXP`, `BYTE`, `SIGNEXTEND`), comparisons, if/switch/
  nested loops with break and continue, multi-return functions with `leave`,
  memory and storage, revert paths, and direct and mutual recursion with locals
  live across the recursive call.
- **solc's own semantic test suite compiles**: 1661 sources, **4128 Yul objects,
  all of them reach bytecode**.
- **622 contracts from that suite deploy and agree** per selector with the
  `solc --via-ir` build, with one divergence that is by construction (above).

### Running it

```sh
anvil --silent --port 8546 --code-size-limit 200000000 --disable-block-gas-limit &

python3 libsolidity/codegen/mlir/test/evm_differential.py \
    --corpus libsolidity/codegen/mlir/test/corpus \
    --solc build/solc/solc --yul2evm build/libsolidity/codegen/mlir/tools/yul2evm

python3 libsolidity/codegen/mlir/test/deploy_differential.py \
    --solc build/solc/solc --yul2evm build/libsolidity/codegen/mlir/tools/yul2evm \
    --corpus test/libsolidity/semanticTests/functionCall

python3 libsolidity/codegen/mlir/test/corpus_coverage.py \
    --solc build/solc/solc --yul2evm build/libsolidity/codegen/mlir/tools/yul2evm \
    --corpus test/libsolidity/semanticTests
```

Or from solc itself, on an MLIR-enabled build:

```sh
solc --via-ir --optimize --mlir-bin Contract.sol
```

## The `sol` rung: a producer, and a measured gap

`MLIRGenerator` (Solidity AST -> `sol` dialect) is ported onto this branch, so
the first rung has a producer for the first time. It lives in `libsolidity`
rather than under `codegen/mlir` because it needs the AST. Its Yul-text
lowering was left behind deliberately: `Conversion/SolToYul` supersedes it, and
porting a rival lowering would defeat the point.

`tools/sol2evm` drives the whole ladder with no solc Yul pipeline anywhere:

```
Solidity AST -> sol dialect -> yul dialect -> evm dialect -> EVM assembly
```

Over 195 contracts from solar's `tests/ui/codegen`:

| Stage reached | Contracts |
|---|---|
| `bytecode` | 52 |
| `evm` | 2 |
| `yul` | 12 |
| `sol` | 129 |

**Reaching `bytecode` here does not yet mean a working contract.** `SolToYul`
generates no external dispatcher, no ABI encoding and no constructor, so
`Counter.sol` compiles to 33 bytes - the function bodies with nothing to call
them. That is a larger gap than the op list below and is not measured by it.

What stops the other 144, most frequent first:

| Blocker | Contracts |
|---|---|
| `solidity.function_call` | 34 |
| generated `sol` dialect does not parse or verify | 20 |
| `solidity.inline_assembly` | 11 |
| `solidity.member_access` | 8 |
| `solidity.mapping_access` | 8 |
| `solidity.string_literal` | 7 |
| `solidity.abi_decode` | 5 |
| `solidity.array_access` | 4 |
| `solidity.emit` | 3 |
| `struct_create`, `mapping_store`, `external_call`, `array_store`, `addmod`, `abi_encode`, `abi_encode_packed` | 2 each |

`function_call` is internal calls, and `yul.func_call` already exists, so it is
likely the cheapest large win.

The round-trip failures started at 31 and are down to 20. They are a defect in
what exists rather than an absence, and they had two causes. Ten were the
driver's own fault - it registered only the `sol` dialect, while the generator
also emits `scf` and `arith`. The rest are the generator building ill-typed IR
after it drops a feature: the Solidity type still says array, but the dropped
sub-expression left a plain word behind, so `solidity.array_length` fails to
verify and the whole contract is lost rather than the one feature. Guarding that
site fixed the largest group; the same shape remains in the boolean ops, the
multi-value returns, and a few printed attributes.
