# RFC 0002: `pub fn` → C ABI Symbol Export

- **Start Date**: 2026-05-16
- **RFC PR**: TBD
- **Status**: Draft
- **Target Release**: v0.2.6
- **Phase**: 1 of 2 (Phase 2 = cdylib emit + AOT lowering, lands with v0.3.0)
- **Normative reference**: `mind-spec` RFC-0007 (`spec/v1.0/ffi.md`) — the
  C ABI contract this RFC implements. ABI stability guarantees,
  type mapping, and header-generation rules are owned by the spec; this
  RFC owns the compiler-side mechanics that satisfy them.

## Summary

Turn `mindc` from an executable-only compiler into a library-emitting one.
After this RFC, each `pub fn` declared in a MIND module — together with any
`[exports]` entry in `Mind.toml` — produces a deterministic C ABI wrapper
symbol of the form

```c
int mind_fn_<name>_invoke(const MindIO *inputs, size_t in_count,
                          MindIO *outputs,      size_t out_count);
```

Host runtimes (mind-runtime, mind-nerve, mind-mem, MindLLM, rfn-mind) call
the wrapper through `dlsym()` and pass the existing `MindIO` calling
convention. No more "embed source, re-parse on `dlopen`" path.

## Motivation

### Today
- `Mind.toml [exports]` is parsed into `ProjectManifest.exports` but the
  build pipeline never reads it back.
- `Node::Export { names: Vec<String>, span: Span }` is parsed and
  discarded by the AST → IR lowering pass.
- The .so produced by `mind build` has no AOT symbols for user `pub fn`.
- Downstream runtimes — `mind-runtime`'s embedded parser, `mind-nerve`'s
  daemon — re-parse MIND source at runtime. This is the largest residual
  drift surface between mindc and the runtimes.

### After this RFC
- `Node::Export` → `IRModule.exports: HashSet<FnId>`.
- Each `IRModule.exports` entry produces a C wrapper symbol at MLIR
  lowering time.
- `Mind.toml [exports] c_abi = ["foo"]` and `pub fn foo(...) {...}` produce
  the same symbol; both paths converge on the same `IRModule.exports`
  set.
- mind-nerve's Phase 2 path compiles `pub fn preselect_pre_tokenized`
  with `Mind.toml [exports] c_abi = ["preselect_pre_tokenized"]` and the
  daemon calls `mind_fn_preselect_pre_tokenized_invoke()` via `dlsym`.

## Guide-level explanation

### A worked example

```mind
// nerve_export.mind
pub fn preselect_pre_tokenized(token_ids: Tensor[i32,(256,)], top_k: i32)
    -> Tensor[i32,(top_k,)]
{
    // ... implementation
}
```

```toml
# Mind.toml
[package]
name = "nerve"
version = "0.1.0"

[exports]
c_abi = ["preselect_pre_tokenized"]
```

```bash
$ mindc build --features ffi-c-user
   Compiling nerve v0.1.0
$ nm -D target/release/libnerve.so | grep mind_fn_
T mind_fn_preselect_pre_tokenized_invoke
```

```c
// host.c
#include "mind_runtime.h"

void call_preselect(const int32_t *token_ids, int32_t top_k) {
    MindIO inputs[2] = {
        MindIO_tensor_i32(token_ids, /*shape*/ (size_t[]){256}, 1),
        MindIO_scalar_i32(top_k),
    };
    MindIO outputs[1];
    int rc = mind_fn_preselect_pre_tokenized_invoke(inputs, 2, outputs, 1);
    /* ... */
}
```

The wrapper symbol is the only stable contract. Internal lowering
details (MLIR dialect, LLVM passes, evidence-chain hooks) stay private.

## Reference-level explanation

### AST
`Node::Export { names: Vec<String>, span: Span }` stays as-is. Lowering
moves to AST → IR.

### IR additions

```rust
// crates/mind-ir/src/module.rs
pub struct IRModule {
    /* existing fields ... */

    /// User-exported function IDs. Populated by AST → IR lowering and by
    /// the build pipeline (which prepends `Mind.toml [exports]` entries).
    /// Empty in the default code path → no extra codegen, no benchmark
    /// movement.
    pub exports: HashSet<FnId>,
}
```

### Codegen pass

A new `c_export_wrapper` pass runs **after** MLIR lowering. For each
`fn_id` in `IRModule.exports` it emits a wrapper symbol
`mind_fn_<name>_invoke` that

1. validates `in_count` / `out_count` against the function signature,
2. unpacks each `MindIO` slot into the function's parameter ABI,
3. tail-calls the internal function,
4. packs the return value back into `outputs[0]`,
5. returns `0` on success, `<0` on shape / type mismatch.

The pass is gated behind `feature = "ffi-c-user"`. When the feature is
off the codegen path is unchanged and the export set is unused.

### Build pipeline

```rust
let mut ir_module = lower_to_ir(&ast)?;
// Manifest-declared exports merge into the AST-declared ones.
ir_module.exports.extend(
    manifest.exports.c_abi.iter().map(FnId::from_name)
);
if !ir_module.exports.is_empty() {
    verify_all_exports_exist(&ir_module)?;
}
lower_to_mlir(&ir_module)?;
```

The "merge" call is a no-op when `manifest.exports.c_abi` is empty
(the typical case). The verification scan runs once per `pub fn`
declaration — never per statement — so it cannot move the parser /
typecheck / IR-lowering bench numbers.

## Compile-speed invariant

Frontend headline numbers (1.8–15.5 µs) MUST NOT regress. Discipline:

- `Node::Export` already has a dedicated branch in the parser. No new
  parse cost.
- The IR `exports` field is empty when there is no `export` keyword
  AND no `[exports]` table. Branchless cost: one `HashSet::is_empty()`
  check in the build pipeline.
- The C-wrapper codegen pass runs at MLIR lowering time, after the
  frontend. It only runs if `exports.is_empty() == false`. The default
  benchmark inputs (`small_matmul` / `medium_mlp` / `large_network`)
  have zero exports → pass exits in O(1).
- A new sub-benchmark `bench_c_export_wrapper(1_export | 10_exports |
  100_exports)` lives separately so the headline numbers in
  `.bench-baseline-2026-04-28-pratt.txt` stay frozen.

Acceptance gate: `cargo bench --bench compiler -- --quick` shows ≤ 2 %
mean regression on the three headline benches (the existing
`bench-gate.yml` threshold).

## Drawbacks

1. Wrapper symbols add an indirection vs. direct internal-symbol export.
   The MindIO calling convention is the stable surface; that's the
   intentional trade-off.
2. `dlsym` lookup is one-time at runtime load. Caller caches.
3. Compile time grows linearly in number of exports — for a 100-export
   library, the wrapper pass is the cost. Most projects have ≤ 10
   exports; the dedicated sub-bench keeps this visible.

## Alternatives

- **Embed source, parse at runtime** (today's path). Eliminated: drift
  between mindc and embedded parsers is the largest residual risk in the
  ecosystem.
- **MLIR-native export attribute.** Considered. The two-stage
  AST→IR→MLIR pipeline keeps the export concept at IR level so non-MLIR
  backends (e.g., the cranelift fallback) can still consume it.
- **Cargo-style symbol versioning.** Deferred to v0.3.0 (`mind_fn_X_v1`)
  alongside the cdylib emit work in RFC 0003.

## Future direction (out of scope for this RFC)

- `--profile=<default|systems|embedded>` CLI flag (deliverable 5 of
  Phase 10.6) so the same `Mind.toml` produces three distinct artifacts.
- `mind build --lib` cdylib emit and AOT lowering — covered by the
  upcoming RFC 0003 alongside `mindc` v0.3.0.
- `Mind.toml [protection]` action transforms — replaces the hand-rolled
  `build.sh`-driven post-processing in the protected build variants.
  Covered by RFC 0004 with v0.3.0.

## Open questions

1. Should `pub fn` *imply* `c_abi` export, or always require explicit
   listing in `Mind.toml`? Lean: explicit (manifest is the single source
   of truth for ABI commitment).
2. Should the wrapper validate tensor shapes at the boundary, or trust
   the caller? Lean: validate. The cost is negligible vs. the runtime
   inference time it precedes.

## Drop-in compatibility

- Default build (`mindc build` without `--features ffi-c-user`) is bit-
  identical to v0.2.5.
- mind-nerve, mind-mem, MindLLM, rfn-mind continue to compile as before
  until they opt into the export path.

## Adoption plan

1. Land `IRModule.exports` field + the AST → IR lowering hook —
   feature-gated, no codegen change. Sub-bench added.
2. Land the `c_export_wrapper` codegen pass — feature-gated.
3. Wire `Mind.toml [exports]` merge in the build pipeline.
4. mind-nerve switches its Phase 2 daemon path to call
   `mind_fn_preselect_pre_tokenized_invoke()` instead of re-parsing
   source.
5. Document `mic@1` IR consumption pattern: canonical parse happens at
   compile time; runtimes consume `.mic` via `ir::load()`.
