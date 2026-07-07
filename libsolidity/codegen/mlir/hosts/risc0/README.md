# RISC Zero adapter (M7 groundwork)

Runs MLIR-ladder-compiled EVM code **inside the RISC Zero zkVM** — the
production target of the SolcRISCV pipeline. The host JIT / qemu harnesses
stay as the cheap per-commit correctness rigs; this adapter is where
target-fidelity execution, **cycle counts** (the metric the pipeline
optimizes), and receipts come from.

The guest links the RV32IM objects emitted by `Target/RISCV` (contract code
with by-pointer `__test_*` wrappers + `evm-rt`) and is a thin ERHI-precursor
shim: read vectors, call compiled code, commit results to the journal.

## One-time setup

The RISC Zero toolchain installs via the official installer (run yourself):

```sh
curl -L https://risczero.com/install | bash
rzup install
```

## Run

```sh
# 1. Build the pipeline tools once:      cmake --build build --target yul2rv
# 2. Produce the guest-side archive from the MLIR ladder:
./build-guest-lib.sh
# 3. Execute in the zkVM (executor mode: no proving, prints cycle counts):
cargo run --release -p evm-risc0-host
# 4. Optionally produce and verify a real receipt:
PROVE=1 cargo run --release -p evm-risc0-host
```

Expected output: `[PASS] all 24 landmine vectors exact inside the RISC Zero
zkVM` plus the user/total cycle counts.

Notes:
- `risc0-*` crates are pinned to major version 2; bump if your `rzup`
  toolchain is newer.
- The archive is plain `riscv32-unknown-elf` rv32im/ilp32 code; the R0 guest
  linker consumes it directly.
- Next steps here (per the proposal): route keccak/bigint through the R0
  accelerator syscalls, replace the vector shim with the ERHI env-block +
  syscall adapter (M6/M7), and benchmark cycles vs an interpreted-EVM
  baseline.
