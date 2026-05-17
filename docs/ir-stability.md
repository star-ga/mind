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
  `compact::v2::parse_mic2` / `compact::v2::parse_micb`.
- New `Instr` variants may be added in minor releases; consumers should
  match exhaustively and treat unknown variants as a hard error.
- New `BinOp` variants may be added in minor releases (e.g. `BinOp::Mod`
  added in 0.2.10). Consumers that exhaustively match on `BinOp` will
  refuse to compile until they handle the new variant; this is
  intentional per the conditionally-stable contract.

## What is experimental

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

Because the parser sits on the AOT compile path (and historically on the
runtime hot path), regressions are policed by
`.github/workflows/bench-gate.yml` against the frozen baseline at
`.bench-baseline-2026-04-28-pratt.txt`. The threshold is **+2% mean** on
any of `small_matmul / medium_mlp / large_network`. See
[docs/versioning.md](versioning.md) for the rationale.

## Cross-repo coordination

- `mind-runtime` consumes `ir::load`; do not call `parser::parse` from
  runtime hot paths.
- `512-mind`, `rfn-mind`, `mind-inference`, `MindLLM`, `ctp-mind`,
  `bitnet-mind-governance` may emit `mic@1` from `mindc` and ship it
  alongside the source for ahead-of-time deployment.
- `mindlang.dev/docs/ir/` mirrors this document.
- `mind-spec/spec/v1.x` lists `mic@1` in its stability surface.
