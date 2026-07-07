# Demo: an EVM contract executing in the RISC Zero zkVM

**What this demonstrates:** a token contract — selector dispatcher, keccak
mapping slots, storage, a loop, reverts — compiled by solc's MLIR dialect
ladder to RISC-V machine code and executed **inside the RISC Zero zkVM**,
with call results checked and cycle counts reported. There is **no EVM
interpreter anywhere in the proved trace**: every cycle is the contract
itself plus a thin host interface. This is the working seed of the
SolcRISCV "money demo" (proposal `../SolcRISCV.md`, milestone M7).

```
 token.yul ──> yul dialect ──> evm dialect ──> LLVM ──> rv32im object ─┐
 (via-ir shape)   (import)      (SSA + CFG)    (ERHI v0 calls)         │ llvm-ar
                                                                       ▼
             RISC Zero guest (Rust: ERHI v0 impl + keccak accelerator patch)
                                                                       ▼
             host: ExecutorEnv -> execute -> journal checks + cycle counts
                                        └-> PROVE=1: receipt + verify
```

The demo lives in the RISC Zero monorepo checkout:
`../risc0/examples/solc-evm-token/` (standalone cargo workspace using the
in-repo `risc0-zkvm`/`risc0-build` crates via path dependencies).

## Prerequisites (one-time)

1. **This repo's pipeline tools** (branch `feat/mlir-pipeline`):

   ```sh
   cmake -S . -B build          # needs Homebrew LLVM+MLIR and lld
   cmake --build build --target yul2rv
   ```

2. **The RISC Zero toolchain** (official installer — run it yourself):

   ```sh
   curl -L https://risczero.com/install | bash
   rzup install
   ```

## Run

```sh
cd ../risc0/examples/solc-evm-token

# 1. Compile the contract down the MLIR ladder into the guest archive
#    (lib/libevmcontract.a - rv32im objects: contract + evm-rt):
./build-contract.sh

# 2. Execute inside the zkVM (executor mode - fast, no proving) and check
#    every call; prints user/total cycle counts:
cargo run --release

# 3. Optionally produce and verify a real receipt:
PROVE=1 cargo run --release --features prove
```

## Expected output

```
== solc MLIR ladder -> RISC-V -> RISC Zero zkVM ==
zkVM execution: <N> user cycles, <M> total cycles (<K> segments)

[PASS] balanceOf(alice)                       -> 100
[PASS] transfer(bob, 60)                      -> 1
[PASS] balanceOf(alice)                       -> 40
[PASS] balanceOf(bob)                         -> 60
[PASS] transfer(bob, 1000) [insufficient]     -> REVERT
[PASS] sumTo(1000)                            -> 499500

[PASS] all 6 calls exact inside the zkVM - no EVM interpreter in the trace
```

The same contract and the same expectations run natively in this repo's
`build/libsolidity/codegen/mlir/tools/erhi-host-test` — the two executions
are directly comparable (that is the differential story).

## How it works

- `build-contract.sh` drives `yul2rv`: the token's deployed object descends
  yul dialect -> evm dialect (structured-CF SSA construction turns the
  loop's mutable variables into block arguments; the dispatcher becomes a
  CFG) -> LLVM dialect (state/env ops become ERHI v0 by-pointer calls:
  `__evm_sload(ptr result, ptr slot)`, ...) -> in-process RISC-V codegen.
  `llvm-ar` bundles it with `evm-rt` (i256/i512 runtime in LLVM IR).
- The **guest** (`methods/guest/src/main.rs`) links that archive and
  implements ERHI v0 natively: big-endian memory arena vs little-endian
  word ABI conversions at the boundary, a storage map, calldata/env, and
  halting via a hand-rolled rv32 setjmp/longjmp (the compiled contract's
  `return`/`revert`/`stop` never return - the host unwinds). Keccak uses
  RISC Zero's accelerator-patched `tiny-keccak`, so mapping-slot hashes ride
  the circuit.
- The **host** (`host/src/main.rs`) pre-mints a balance by writing the same
  mapping slot the contract computes, sends six calls, checks the journal
  (status + returndata per call), and prints cycle counts. `PROVE=1` runs
  the prover and verifies the receipt against the guest image ID.

## Troubleshooting

- `yul2rv not found` — build it in this repo first (prerequisite 1); or set
  `SOLC_BUILD_DIR` / `SOLIDITY_DIR` env vars for non-default layouts.
- Toolchain errors from `cargo` — the guest needs the RISC Zero Rust
  toolchain from `rzup install` (prerequisite 2).
- LLVM tools not at `/opt/homebrew/opt/llvm@21/bin` — set `LLVM_BIN_DIR`.
- The example pins `risc0-*` crates by **path** into the monorepo checkout,
  so there is no crates.io version skew; keep the checkout on a release-ish
  commit if guest-toolchain and crate versions drift apart.

## What this demo is not (yet)

Honest scope notes, per the proposal's milestones: single contract, no
cross-contract calls (`CALL` family lands with host re-entry, M7), no gas
accounting (zkVM cycles are the meter), storage is in-guest state rather
than committed pre/post state roots, and there is no interpreted-EVM
baseline measurement yet — that comparison (same calls through revm-in-zkVM)
is the headline number the full M7 demo adds on top of this.
