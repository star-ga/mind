# RFC 0005 Phase D₂b — Cross-arg Named-struct identity matching

> **Status:** Design note, multi-session pickup artifact. Not yet implemented.
>
> **Predecessor:** Phase D₂a (mindc v0.4.4) — Named struct parameter
> names preserved in error diagnostics.
>
> **Position in landing table:** [`docs/rfcs/0005-pure-mind-std-surface.md`](./0005-pure-mind-std-surface.md) §"D phases" row D₂b.

## Problem

Under Option-C (RFC 0005 P0e heap-record struct ABI), every `Named`
struct lowers to an `i64` base-address from `__mind_alloc`. Today's
Phase B per-arg compatibility check folds Named struct annotations to
`ValueType::ScalarI64` and only checks i32↔i64 widening:

```rust
// src/type_checker/mod.rs ~ line 2040
fn cm_arg_compatible(expected: &ValueType, actual: &ValueType) -> bool {
    if expected == actual { return true; }
    matches!(
        (expected, actual),
        (ValueType::ScalarI64, ValueType::ScalarI32) |
        (ValueType::ScalarI32, ValueType::ScalarI64)
    )
}
```

This means a call like:

```mind
use std.vec
use std.string

let s: String = string_new()
let r: i64 = vec_set(s, 0, 99)   // currently accepted at type-check
                                  // — both Vec and String lower to i64,
                                  // so cm_arg_compatible returns true.
```

…passes type-checking but will crash at runtime because the heap-record
layouts for `Vec { addr, len, cap }` and `String { addr, len }` differ.
Phase D₂a improved the *error* message when widening fails on a
different arg, but the actual compatibility check is still permissive.

## Goal

Reject `vec_set(s: String, …)` at the call site while preserving the
escape hatch where a raw `i64` value (literal or expression) is
intentionally used as a heap-record addr in low-level code. This
matches the design intent of the landing table row:

> "Cross-arg Named-struct *identity* matching (Vec ≠ String at call site)"

## Non-goal

Removing the i64↔Named permissive coercion entirely. That would break
the Option-C escape hatch — runtime code that legitimately constructs
or marshals heap-record addresses via raw i64 (e.g. cross-FFI
boundaries, test fixtures, `__mind_alloc` callers in pure-MIND
runtimes).

## Approach — option (c), surgical

Plumb the *syntactic* type of the actual argument through to the
per-arg compat check, **only** when the actual argument's declared
type is a Named struct. Compare struct-name to struct-name; if they
differ, error. If the actual is *not* annotated as a Named struct
(literal, arithmetic expression, function return), fall through to
the existing permissive i64-widening rule.

Specifically:

1. Extend `infer_expr` (or add a sibling helper) to return an
   `Option<&'static str>` "struct-name witness" alongside the
   `ValueType` when the expression's declared/inferred type is a
   Named struct.
2. In `check_imported_fn_call`, after calling `infer_expr`, if both
   the declared param's `TypeAnn::Named(p)` and the actual's
   struct-name witness `a` are present, compare them:
   - `p == a` → pass
   - `p != a` → error with message "expects {p} (heap-record i64
     addr), got {a} (heap-record i64 addr)"
3. If the actual has no struct-name witness, fall through to
   `cm_arg_compatible` (current behaviour preserved).

## Witness sources

The struct-name witness should be derivable from:

- `Ident` whose declared type in `TypeEnv` is `TypeAnn::Named(n)` — most
  common case (local `let v: Vec = vec_new()` then `vec_set(v, …)`).
- Fn-call expression whose imported `pub fn` declares
  `ret_type = Some(TypeAnn::Named(n))` — chains like
  `vec_push(vec_new(), 1)`.
- Struct-literal `Vec { … }` — exact match.

For everything else (arithmetic, indexing, casts), no witness is
emitted and the permissive i64 path stays. This keeps the change
surgical.

## Test contract

Drop tests into `tests/std_surface_use_import_phase_d2b.rs`
(`#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]`):

1. **Wrong-named-struct rejected.** `let s: String = string_new();
   vec_set(s, 0, 99)` → error mentioning both `Vec` and `String`.
2. **Right-named-struct accepted.** `let v: Vec = vec_new();
   vec_set(v, 0, 99)` → clean.
3. **Raw i64 still accepted (Option-C escape hatch).**
   `vec_set(0, 0, 99)` → clean.
4. **i64 expression accepted.** `let addr: i64 = __mind_alloc(24);
   vec_set(addr, 0, 99)` → clean.
5. **Chained fn-return accepted.** `vec_set(vec_new(), 0, 99)` → clean.
6. **Cross-module donor without `pub` decl falls back to Phase A
   loose typing** — already covered by
   `phase_a_fallback_when_no_signature_available`; add a regression
   test that this stays loose under D₂b.

## Compile-speed budget

Phase D₂b is cold-path-only (per-arg error path of imported `pub fn`
calls under `cross-module-imports`). The hot path (default build, no
`use foo` resolution) is byte-identical. Bench gate stays at +7%
against `.bench-baseline-2026-05-18-rfc0005.txt`.

## ABI risk

None at the wire level — Option-C heap-record ABI unchanged. Risk is
purely user-visible: existing user code that *intentionally* passes a
`String`-typed value to a `Vec`-typed parameter (cross-marshalling
without going through raw i64) breaks. Triage:

- The bundled `std/*.mind` modules don't do this internally.
- mind-nerve doesn't do this (Phase 1 PyTorch path; native MIND path
  Phase 2 will adopt D₂b semantics from the start).
- 512-mind / MindLLM consumers — check for cross-Named coercion
  before tagging.

## Tag

Target `mindc v0.4.5`. Independent of any other work; can ship as a
standalone patch tag.

## Open question

Should D₂b *also* emit a warning (not error) when the actual is a
raw i64 expression and the declared param is Named? This would flag
intentional Option-C usage as "you know what you're doing"
documentation. Punt — implement the error path first, see what user
patterns surface, then decide.

---

Authored 2026-05-18 alongside the mindc v0.4.4 / Phase D₂a release.
Pickup artifact for the next compiler session.
