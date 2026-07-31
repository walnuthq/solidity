# Porting the Yul optimizer to the `yul`/`evm` dialects

The MLIR ladder currently takes its input from `solc --ir-optimized`, so it
inherits solc's Yul optimizer for free. Generating the `yul` dialect directly
from the `sol` dialect removes that, and unoptimized input would cost far more
than the backend has so far gained. This is the inventory of what would have to
replace it.

## The headline

`libyul/optimiser` is 9260 lines, of which the 32 registered steps are 4115 and
the rest is the machinery they sit on. But **most of it does not need porting**,
for two different reasons — and knowing which bucket a pass is in matters more
than its size.

| | Passes | Lines | Why |
|---|---|---|---|
| Unnecessary | 13 | ~2400 | MLIR's IR does not have the problem |
| Free from upstream | 11 | ~1800 | An MLIR pass already does it, once our ops declare enough |
| Must port | 8 | ~1300 (+950 infra) | Encodes EVM semantics no upstream pass knows |

So the real question is not "can we port 32 passes" but "can we write the 8 that
carry EVM knowledge, and declare enough about our ops that upstream does the
rest". That is a much smaller job than the line count suggests.

## Unnecessary — the IR does not have the problem

Nothing to write. Listed so nobody ports them by reflex.

| Pass | Lines | Why not |
|---|---|---|
| `ExpressionSplitter` | 116 | MLIR is already flat SSA; there are no nested expressions to split |
| `ExpressionJoiner` | 148 | ...and none to rejoin |
| `SSATransform` | 377 | `YulToEVM` constructs SSA during CF flattening; merges are block arguments |
| `SSAReverser` | 120 | Only exists to undo the above for a stack backend |
| `VarDeclInitializer` | 60 | Yul AST normalization |
| `ForLoopInitRewriter` | 52 | Yul AST normalization |
| `ForLoopConditionIntoBody` | 68 | Yul AST normalization |
| `ForLoopConditionOutOfBody` | 78 | Yul AST normalization |
| `BlockFlattener` | 56 | MLIR blocks are already flat |
| `FunctionGrouper` | 59 | Module structure is explicit |
| `FunctionHoister` | 52 | Module structure is explicit |
| `Disambiguator` | 76 | MLIR values have no names to collide |
| `NameSimplifier` / `VarNameCleaner` | 260 | Same |
| `StackCompressor` | 296 | Stack-too-deep is unreachable in this backend |
| `StackLimitEvader` | 239 | Same |
| `StackToMemoryMover` | 335 | Values are already memory-resident |

The last three are worth noticing: three of the hairiest passes in the Yul
optimizer exist to fight stack-too-deep, and the value model this backend chose
makes them meaningless.

## Free from upstream — declare, do not implement

These need op traits and interfaces rather than passes. The work is in the
TableGen, and it is shared with everything else that wants to reason about the
dialects.

| Pass | Lines | Upstream equivalent | What we must declare |
|---|---|---|---|
| `CommonSubexpressionEliminator` | 134 | `-cse` | `Pure` on pure ops |
| `DeadCodeEliminator` | 70 | `-remove-dead-values` | Memory effects |
| `UnusedPruner` | 170 | `-remove-dead-values`, `-symbol-dce` | Memory effects |
| `UnusedAssignEliminator` | 164 | dead-value removal over SSA | — |
| `CircularReferencesPruner` | 63 | `-symbol-dce` | — |
| `StructuralSimplifier` | 128 | `-canonicalize` | Folders |
| `ConditionalSimplifier` | 94 | `-sccp`, `-canonicalize` | Folders |
| `ConditionalUnsimplifier` | 107 | `-canonicalize` | — |
| `ControlFlowSimplifier` | 228 | `-canonicalize` on `cf` | — |
| `LoopInvariantCodeMotion` | 121 | `-loop-invariant-code-motion` | Memory effects |
| `Rematerialiser` | 105 | canonicalization + folders | Folders |
| `FullInliner` / `ExpressionInliner` | 456 | `-inline` | `DialectInlinerInterface` — declared, not enabled |

**Started here. `-canonicalize` and `-cse` now run** on the converted module,
and the 13 landmine ops have folders with EVM-exact edge behaviour (division by
zero, `SDIV(MIN,-1)`, shifts >= 256, wide addmod/mulmod, `BYTE`, `SIGNEXTEND`,
`CLZ`). Emitted bytes over the semantic suite fell from 5325389 to 4227297, a
fifth, with all 4128 objects still reaching bytecode and 636 contracts still
agreeing per selector.

Two things had to be taught to the emitter first, because canonicalization
produces IR the conversion never did: `arith.select`, which the EVM has no
opcode for but an `i1` condition makes branch-free as
`b xor ((a xor b) * c)`; and a conditional branch whose arms have been merged
into one block with differing arguments, where there is no branch left to take
and the arguments themselves are the choice. `-remove-dead-values` is left out —
it fails on 41 objects of the suite, and canonicalization already removes dead
pure ops.

**The inliner interface is in, and deliberately not switched on.** The `evm`
dialect declares `DialectInlinerInterface` and the `func` extension is
registered, so `-inline` works — `functions.yul` in the differential corpus goes
from 369 bytes to 34, a third of what the reference backend emits. Over the
whole semantic suite it is a loss: 4177690 bytes becomes 5202861, a quarter
more, because the input has already been through solc's inliner and upstream's
default threshold only duplicates from there. Turning it on needs a cost model
of this backend's own — a call here costs a stack flush, an argument write and
a result read — which is a separate piece of work from having the interface.

Two things had to be right before it would run at all. The `func` dialect's
inliner interface is an opt-in extension; without registering it the pass
silently does nothing and reports success. And every terminator this dialect
defines halts execution rather than returning, so both `handleTerminator` hooks
are empty — the base class declares them unreachable, which is what any function
containing a `revert` tripped over.

**Memory effects are in.** EVM state spaces cannot alias, so storage, transient
storage and memory are three distinct side-effect resources (`EVMDialect.h`),
and the fourteen state ops declare which one they read or write. Eighteen
environment ops that are fixed for the whole call frame are `Pure`. This is the
interface `LoadResolver` and the store eliminators will ask, so it is done once
for both columns.

It also taught us something about this backend. Marking the environment ops
`Pure` let CSE merge them and made the output *larger*: a value held in a frame
slot costs a store and a reload, eight bytes, where re-emitting `CALLER` costs
one. They are now rematerialised at each use like constants — restricted to ops
that really are a single opcode, since `dataoffset`, immutables and linker
symbols are relocations and can be far larger than a reload. Net over the suite:
4227297 bytes down to 4177690.

## Must port — EVM knowledge lives here

| Pass | Lines | Note |
|---|---|---|
| `SimplificationRules` | 279 | The algebraic identity table. Mostly declarative, so it maps well onto MLIR's rewrite-rule generators rather than hand-written C++ |
| `ExpressionSimplifier` | 93 | The driver for the above |
| `LoadResolver` | 160 | Storage/memory load forwarding |
| `UnusedStoreEliminator` | 438 | Dead store elimination under EVM's side-effect rules |
| `EqualStoreEliminator` | 69 | |
| `UnusedFunctionParameterPruner` | 126 | |
| `EquivalentFunctionCombiner` | 45 | |
| `FunctionSpecializer` | 158 | |
| `ConstantOptimiser` | (in `backends/evm`) | Cheapest way to materialize a constant — pure EVM cost modelling |

Three of these depend on machinery that has to come first:

| Support | Lines | Needed by |
|---|---|---|
| `Semantics` (side effects) | 249 | everything that reorders or deletes |
| `DataFlowAnalyzer` | 488 | `LoadResolver`, both store eliminators |
| `KnowledgeBase` | 216 | `LoadResolver`, `ExpressionSimplifier` |

The side-effect model is the real prerequisite. It is also what the upstream
passes need in the form of `MemoryEffectOpInterface`, so writing it once serves
both columns.

## Suggested order

1. **Traits and effects on the `evm` and `yul` ops** — `Pure` where it holds,
   `MemoryEffectOpInterface` for the state ops. Unlocks `-cse`,
   `-canonicalize`, dead-value removal and LICM in one move. Best ratio by far.
2. **Folders** on the `evm` ops, with the landmine semantics baked in
   (`div` by zero, `SDIV(MIN,-1)`, shifts >= 256). These are already specified
   and already covered by the differential corpus.
3. **`DialectInlinerInterface`**, which replaces 456 lines of inliner.
4. **`SimplificationRules` as rewrite patterns** — the largest single piece of
   EVM-specific value, and declarative rather than procedural.
5. **Side-effect model, then `LoadResolver` and the store eliminators.**
6. **`ConstantOptimiser`**, which is a backend concern and belongs next to the
   assembly emitter rather than in the optimizer.

Steps 1-3 are what make the direct `sol` -> `yul` path viable at all; 4-6 are
what make it competitive.

## How to know it works

Every step above is measurable with what already exists: `corpus_coverage.py`
reports emitted bytes over solc's semantic suite, and `deploy_differential.py`
answers whether the result still behaves. An optimization that costs a single
divergence is not an optimization.
