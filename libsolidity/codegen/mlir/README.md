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
| `Conversion/SolToYul/` | `sol` -> `yul` (creation/runtime, dispatch, storage, memory, ABI subset) | working (M2) |
| `Conversion/YulToEVM/` | block-local var promotion + CF flattening + builtin mapping | mature (M4) |
| `Conversion/EVMToLLVM/` | landmine legalization + evm-rt calls + clamped shifts | working (M5, pure subset) |
| `runtime/evm-rt/` | i256/i512 runtime in LLVM IR (div family, exp, byte, signextend) | working (M5) |
| `hosts/risc0/` | RISC Zero guest+host: zkVM execution, cycle counts, receipts | scaffolded (M7 groundwork) |
| `tools/` | rung emitters, bytecode drivers, and `yul2rv` | working |

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
- **697 contracts from that suite deploy and agree** per selector with the
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
solc --mlir-bin Contract.sol
```

`--mlir-bin` uses solc's production Solidity-to-strict-Yul frontend and imports
the resulting object tree into MLIR:

```
Solidity AST -> strict Yul -> yul dialect -> evm dialect -> EVM assembly
```

Strict-Yul generation is the frontend boundary; after that boundary the route
does not fall back to either legacy EVM code generation or the Yul bytecode
backend. Adding `--via-ir` does not change the selected MLIR backend.

## Production and legacy-compatibility frontends

`MLIRGenerator` also provides the typed first rung:

```
Solidity AST -> sol dialect -> yul dialect -> evm dialect -> EVM assembly
```

This route is used by `tools/sol2evm` and by semantic fixtures explicitly marked
`compileViaYul: false`. It preserves legacy-only observable behavior without
forcing the unfinished direct AST lowering to replace solc's production
frontend for ordinary contracts. Frontend choice is an enum in the compiler
API, not an environment-variable switch. Experimental analysis always uses its
native strict-Yul producer because its AST intentionally lacks legacy type
annotations.

Both routes share the same MLIR Yul-to-EVM conversion, object linker, memory
frame policy, bounded inliner, and EVM assembly emitter. Runtime and creation
objects, nested `new C` objects, immutables, `dataoffset`/`datasize`, library
linking, and optimizer settings therefore go through the same backend.

### Full-corpus validation

The complete in-repo semantic corpus is green on 2026-08-03. The primary
current-EVM run is:

```sh
yes s | build_develop/test/tools/isoltest \
  --mlir --no-smt --no-color -t 'semanticTests/*'
```

```text
Semantic Test Summary: 1590/1661 tests successful (71 tests skipped).
```

There were zero failed tests, MLIR compilation failures, or unhandled
exceptions. The 71 skipped cases are configuration exclusions made by the
semantic harness; every test selected by this EVM/options run passed.

The skipped set is covered by the matching configurations, rather than counted
as success without execution:

```text
default:       1590/1661 successful, 71 skipped
ABI encoder v1: 1614/1661 successful, 47 skipped
Homestead:     1401/1661 successful, 260 skipped
Shanghai-only selfdestruct fixtures: 2/2 successful
@future-only fixture:                 1/1 successful
```

The union executes all **1634 standalone semantic fixture files**, with zero
failures. The other 27 Solidity files below `semanticTests/` are imported
helper sources and have no `// ----` test section. All 38 fixtures marked
`compileViaYul: false` execute successfully in their compatible configuration
through the direct typed-`sol` route.

The MLIR-specific integration suite is green as well:

```sh
ctest --test-dir build_develop -L mlir --output-on-failure
```

It contains 21 registered tests, including **31/31 lit/FileCheck tests**, the
real `solc --mlir-bin` CLI, reach gates, deployed Yul/Solidity differentials,
and the stateful runtime differential below. Focused tests that require an
existing JSON-RPC endpoint report a CTest skip when no node is available; the
stateful test starts a disposable Anvil instance itself. The Osaka `CLZ` case
similarly probes node opcode support before attempting a cross-backend
execution comparison.

### Stateful runtime differential

`runtime_differential.py` is the independent execution harness for the
production compiler path. It compiles every workload twice—reference
`solc --via-ir` and `solc --mlir-bin`—then starts each version from the same EVM
snapshot and executes the same ordered call/transaction sequence. It compares
deployment success, return and revert data, receipt status, event logs, storage
roots, balances, nonces, value transfer, child creation, and external child
calls. Compiler timing, bytecode size, and gas are reported separately and do
not participate in the correctness comparison.

```sh
python3 libsolidity/codegen/mlir/test/runtime_differential.py \
  --solc build_develop/solc/solc \
  --workloads libsolidity/codegen/mlir/test/runtime \
  --spawn-anvil --evm-version cancun \
  --report-json build_develop/libsolidity/codegen/mlir/test/runtime-report.json
```

The deterministic gate is **26/26 workloads passing** over **371 ordered
runtime steps**: 225 transactions (206 successful and 19 deliberate reverts),
143 calls, two block advances, and one time advance. Each workload also deploys
both backends from the same snapshot.

| Group | Workloads | Runtime steps | What it exercises |
|---|---:|---:|---|
| core stateful | 2 | 20 | dynamic ABI, storage, events, value, reverts, `CREATE`, and child calls |
| hot micro | 4 | 22 | arithmetic, factorial, counter, and range loops |
| fixed edge seeds | 8 | 192 | branches, bytes, mappings, memory arrays, and storage loops across eight fixed inputs each |
| real projects | 9 | 87 | Gnosis, Corion, and multisig contracts already present in `test/compilationTests` |
| large state machines | 3 | 50 | multisig quorum/value transfer, daily-limit rollover, and milestone roles/cancellation |

The default is Cancun. Two historical Gnosis difficulty-oracle workloads are
compiled and executed on London: after Paris the `DIFFICULTY` opcode exposes
`prevrandao`, whose random value is not snapshot-replayable in Anvil. The
runner groups workloads by revision and starts an isolated matching node for
each group.

The same run records a performance probe separately from correctness. These
are min/median/max ratios across this single run, not a statistically stable
benchmark:

| Metric | MLIR/reference ratio |
|---|---:|
| runtime bytecode size | 1.50x / 1.75x / 2.04x |
| deployment gas | 1.24x / 1.43x / 1.91x |
| transaction gas | 1.003x / 1.014x / 1.358x |

A semantic differential failure writes a JSON artifact containing the source,
workload, EVM revision, and both observations. Single-source fixture failures
can be reproduced with:

```sh
python3 libsolidity/codegen/mlir/test/runtime_differential.py \
  --solc build_develop/solc/solc \
  --replay path/to/runtime-failure.json --spawn-anvil
```

Generated contracts, ABI fuzz vectors, multi-source failure packaging, and
automatic reduction are intentionally deferred to the fuzzing phase.

The older 1788/2027 `sol2evm` number measured standalone reach of the direct
AST-to-`sol` experiment, not the routed production compiler and not behavior of
the semantic corpus. It remains useful when widening that optional frontend,
but is no longer the project-level pass result. See
`solc-riscv-mlir-tickets.md` beside the source tree for the current evidence and
the historical ticket trail.
