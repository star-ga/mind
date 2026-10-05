# IR stability contract

> **Status:** stable from mindc 0.2.5 onward.
> **Surface:** `libmind::ir::{IRModule, Instr, BinOp, ValueId, ShapeDim, DType, load, save}`
> plus the textual format `mic@1`.

## Why this exists

mindc 0.2.4 and earlier had a soft contract: backends and runtimes could
reach into `libmind::parser::parse(...)` directly and consume the AST, then
walk it themselves. `mind-runtime/src/eval_entry.rs:88` did exactly that —
re-parsing embedded MIND source on every `mind_main()` invocation. This
made parser performance a runtime hot-path concern across all 12+ backends
on the roadmap (CPU, CUDA, Metal, ROCm, WebGPU, WebNN, ARM, TPU, NPU,
LPU, DPU, FPGA, Quantum) and locked the runtime to whatever extensions
the parser happened to support.

mindc 0.2.5 makes the **public IR layer** (`mic@1`) the canonical contract.
The runtime calls [`ir::load(bytes)`](../src/ir/mod.rs) once at module load
and never invokes the parser again.

## What is stable

The following surface follows the **mind-spec Core v1 stability contract**
and will not change in incompatible ways without a major-version bump:

| Item | Status |
|------|--------|
| `IRModule` struct shape | stable |
| `Instr` variants (existing) | stable |
| `BinOp` enum | stable |
| `ValueId(usize)` | stable |
| `ShapeDim::Known` / `ShapeDim::Sym` | stable |
| `DType` enum (existing variants) | stable |
| `mic@1` textual form | **stable** (RFC-0001 determinism) |
| `ir::load(bytes) -> Result<IRModule, LoadError>` | stable |
| `ir::save(module) -> String` | stable |
| `pipeline::compile_to_mic_text(src, opts) -> String` | stable |

## What is conditionally stable

- `mic@2` text and `MIC-B` binary formats: stable within 0.x but produce
  the lighter-weight `Graph` type, not `IRModule`. Use them via
  `compact::v2::parse_mic2` / `compact::v2::parse_micb`. These are a **library
  surface only** — no `mindc` invocation EMITS mic@2 or MIC-B. A `Mind.toml`
  declaring `ir-format = "mic@2"` is therefore refused by the toolchain pin
  (`src/project/toolchain_pin.rs`): the compiler cannot produce the format that
  manifest claims compatibility with. Parsing stays supported.
- `mic@2.1` adds a trailing canonical **MAP** epilogue (key/value metadata,
  e.g. `evidence_chain.*` per RFC 0016, and Ed25519 `signature.*`) on top of
  `mic@2`/`MIC-B`. Back-compatible by omission: a `Graph` with an empty MAP
  serialises byte-identically to `mic@2`, so `MICB_VERSION` is unchanged. The
  MAP is a separable provenance section, not an IR-grammar change.
- **Canonical-IR direction (RFC 0021, steps 1–3 shipped).** The `Graph`/`mic@2.x`
  lineage carries the lighter dataflow IR + the provenance MAP, but it is **not**
  the compiled-artifact IR — `IRModule`/`mic@1` is (it is what the pipeline and
  the self-host compiler produce, and the RFC 0016 evidence anchor
  `ir::ir_trace_hash` hashes its canonical `mic@3` bytes — `trace_hash =
  SHA-256(canonical mic@3 bytes)`, re-anchored from mic@1 text on 2026-05-31
  after a collision audit found mic@1 text can drop function-body semantics;
  mic@3 binary commits its supported IR fields, so it supersedes the original RFC
  0016 GAP-1 mic@1-text rule). RFC 0021 unifies on the
  `IRModule` data shape with two canonical serialisations: `mic@1` for text and
  **`mic@3` for binary** (magic `MIC3`). Binary and text surfaces have different
  supported metadata; checked APIs refuse metadata loss. The
  provenance MAP attaches to `mic@3` as a `0x4D`-sentinel epilogue using the
  same key/value form as the `mic@2.1` MAP — exposed via
  `mindc --emit-mic3` / `--emit-evidence` (steps 1–3 shipped, `src/ir/compact/v3/`).
  The earlier `mic@1e` proposal was superseded by `mic@3` once `mic@2/2.1`
  shipped (a third minor on the text form would have collided with the
  binary-attestation use case). Steps 4–6 (`mindc verify` CLI, demotion of the
  `Graph` lineage to a `mind-model@2` model-exchange artifact, oracle + CI gate)
  remain in flight; until they land the two coexist and `mic@1` remains the
  canonical contract above.
- New `Instr` variants may be added in minor releases; consumers should
  match exhaustively and treat unknown variants as a hard error.
- New `BinOp` variants may be added in minor releases (e.g. `BinOp::Mod`
  added in 0.2.10). Consumers that exhaustively match on `BinOp` will
  refuse to compile until they handle the new variant; this is
  intentional per the conditionally-stable contract.

## What is experimental

- The unreleased canonical `0x04` binary revision preserves supplied semantic
  metadata for a core instruction subset. Its wire format is draft; source
  lowering, standard-surface transport, pure-MIND parity, and native consumption
  are separate dependencies. See [the current scope](mic3-v04-draft.md).
- The structured `IrVerifyError` body (only the existence/absence of an
  error is stable; specific messages may change).
- New optimisation passes added to `opt::ir_canonical`.
- Backend-specific lowering paths (gated by Cargo features).

## Determinism (RFC-0001)

`save(module)` is deterministic: given the same `IRModule`, output bytes
are byte-identical across runs and platforms. The `save → load → save`
round-trip is a fixed point (verified by `tests/ir_load_save.rs`).

## Migration path for runtimes

Before 0.2.5 (do not do this):
```rust
let module = libmind::parser::parse(&source)?;
let typed = libmind::type_checker::check(&module);
let ir = libmind::eval::lower_to_ir(&module);
// re-runs every invocation
```

After 0.2.5 (the supported pattern):
```rust
// Build time:
let mic = libmind::compile_to_mic_text(&source, &opts)?;
std::fs::write("kernel.mic", &mic)?;

// Run time:
let bytes = std::fs::read("kernel.mic")?;
let ir = libmind::ir::load(&bytes)?;
backend.execute(&ir);
```

## Bench-gate enforcement

The reference MIND implementation's CI bench gate measures the T1 frontend
(`cargo bench --bench compiler --no-default-features`) for
`small_matmul`, `medium_mlp`, and `large_network` against the frozen
correctness floor `.bench-baseline-2026-06-01-correctness.txt`. It applies a
one-sided **+10% regression threshold**: any speedup passes, while a
trustworthy result above the threshold is a decision-triggering regression.
Criterion spread above **12%** is inconclusive, and the canonical pipeline
must provide all three present, trustworthy fixture comparisons. This is a
no-regression guard; it does not establish or require a speedup objective.
The comparator supports an optional separately published champion reference,
but the current workflow supplies only the correctness floor.

## Cross-repo coordination

- `mind-runtime` consumes `ir::load`; do not call `parser::parse` from
  runtime hot paths.
- `mind-law`, `rfn-mind`, `mind-inference`, `MindLLM`, `ctp-mind`,
  `bitnet-mind-governance` may emit `mic@1` from `mindc` and ship it
  alongside the source for ahead-of-time deployment.
- `mindlang.dev/docs/ir/` mirrors this document.
- `mind-spec/spec/v1.x` lists `mic@1` in its stability surface.
