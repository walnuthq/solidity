# Architecture Decision Records — solc MLIR pipeline

## ADR-001: The pipeline is an extension of solc itself

The dialect ladder lives in this repository (`libsolidity/codegen/mlir/`),
not in a standalone tool. Rationale: type information survives the whole
descent (declared widths enable 1–2-limb RISC-V ops instead of 8-limb i256),
the solc semantic-test corpus becomes a backend-vs-backend differential
suite, and distribution is `solc --riscv`. GPLv3 like the rest of solc.

## ADR-002: LLVM/MLIR dependency policy

MLIR is optional at build time (`find_package(MLIR QUIET)`); without it,
vanilla solc builds are completely unaffected. Currently tracking Homebrew
LLVM (22.x); a pinned-version policy comes with CI (M0 exit criteria in the
proposal). TableGen'd dialects follow the Sol dialect's manual `tablegen()`
CMake pattern.

## ADR-003: 1:1 builtins at the `yul` rung; upstream `arith` at the `evm` rung

`yul` dialect ops mirror Yul builtins one-to-one over signless `i256` with
Yul/EVM-exact semantics (`yul.div` by zero = 0, shift >= 256 defined, ...).
This makes the Yul-text emitter a printer, the libyul-AST importer a walk,
and keeps rung 2 a faithful model of the language. Upstream `arith`/`cf`/
`func` reuse — where semantics coincide exactly — happens one rung down in
the `evm` dialect, where MLIR's canonicalization infrastructure pays off.
Cost: folders must be ported rather than inherited at rung 2. Accepted.

## ADR-004: Region-based CF + mutable-variable ops at the `yul` rung

Control flow is modeled with region ops (`yul.if` with a single then-region
per Yul's grammar; `yul.for` with cond/body/post regions and
`yul.condition`), with `yul.break`/`yul.continue`/`yul.leave` as
terminators. Regions carry `NoTerminator` so fall-through blocks stay valid.
`scf` was rejected: it cannot express break/continue/leave, and text
round-trip fidelity is a requirement.

Yul locals are mutable, so rung 2 models them explicitly:
`yul.var` (declare) / `yul.assign` / `yul.var_load` over a hand-written
`!yul.varref` type — the clang/flang "alloca first, promote later" approach.
Consequences: the importer needs no SSA/phi construction (fully general for
loops and nested scopes), and a mem2reg-style promotion pass must run before
lowering to the `evm` rung. The for-loop init block is hoisted before the
loop (SSA names are unique, so Yul's init scoping is preserved trivially).

Emission normalizations (semantics-preserving):
- `for {} cond {post} {body}` prints as
  `for {} 1 {post} { <cond stmts> if iszero(c) { break } <body> }`, which
  preserves evaluation order and continue/break semantics.
- `yul.var_load` prints as a fresh `let` snapshot so later reassignments are
  not visible through the loaded SSA value.
- Unused single results print as `pop(...)`.

## ADR-005: Terminator statements end import of a block

Yul tolerates unreachable statements after `leave`/`break`/`continue`; MLIR
requires terminators to be last in their block. The importer drops dead
statements after a terminator (semantics-preserving by construction).
