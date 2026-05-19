# Phase 6.5 — Bootstrap Fixed-Point Report

**Date:** 2026-05-18
**Tag:** v0.6.0 (APEX)
**Verdict:** FIRST-DIVERGENCE

---

## Round-trip specification

Feed `examples/mindc_mind/main.mind` (1,630 lines, 56,852 bytes — the combined
pure-MIND mindc front-end) to `libmindc_mind.so` via
`mindc_compile(src_addr, src_len)`.  Compare emitted MLIR byte-for-byte against
the oracle produced by `cargo run --bin mindc -- main.mind --emit-ir`.

A PASS would prove bootstrap fixed-point: mindc-Rust is decorative.

---

## Oracle generation

```
cargo run --features "mlir-build std-surface cross-module-imports" \
    --bin mindc -- examples/mindc_mind/main.mind --emit-ir \
    > /tmp/oracle.mlir
```

Exit code: 0. Oracle compiles cleanly (4 `[WARN] lower_expr: unhandled AST
node kind, defaulting to 0` lines, which are expected — the Rust stub emitter
emits them for complex expressions).

Oracle MLIR (pure, post-header strip): **10,599 bytes, 599 lines, next_id=201**.

---

## Round-trip result

```
python3 examples/mindc_mind/fixed_point_smoke.py
```

| Metric | Oracle (mindc-Rust) | Emitted (libmindc_mind.so) |
|--------|---------------------|---------------------------|
| Bytes | 10,599 | 10,389 |
| Lines | 599 | 587 |
| SSA values (next_id) | 201 | 195 |
| Functions emitted | 194 | 194 |

**Verdict: FIRST-DIVERGENCE at byte 42 (oracle line 4, emitted line 4)**

---

## Divergence analysis

### Divergence 1 — `use`-declaration stubs (4 missing SSA values)

Oracle MLIR opens with 4 anonymous stubs before the first `// fn` comment:

```mlir
module {
  %0 = const.i64 0   // <- use std.vec;
  output %0
  %1 = const.i64 0   // <- use std.map;
  output %1
  %2 = const.i64 0   // <- use std.string;
  output %2
  %3 = const.i64 0   // <- use std.io;
  output %3
  // fn tk_eof
  ...
```

`libmindc_mind.so` emitted MLIR opens directly at `// fn tk_eof %0`.

**Root cause — EMITTER subsystem:**
`emit_program_items` in the pure-MIND emitter (Phase 6.4 / `emit_ir/main.mind`)
processes `FnDef` AST nodes only. It does not emit a stub for `UseDecl` nodes.
mindc-Rust's `lower_program` walks every top-level item and emits one stub per
item regardless of kind. The four `use std.*` declarations at the top of
`main.mind` therefore produce 4 extra SSA stubs in the oracle.

**Scoped reproducer:**

```mind
// minimal.mind
use std.vec;
pub fn add(a: i64, b: i64) -> i64 { a + b }
```

mindc-Rust `--emit-ir` produces `%0` (use stub) + `// fn add %1`.
`libmindc_mind.so` produces only `// fn add %0`.

### Divergence 2 — Double stubs for 3 functions (2 missing SSA values each)

Oracle emits 2 `const.i64 0 / output` stubs for three functions:

| Function | Location in main.mind | Body characteristic |
|----------|-----------------------|---------------------|
| `ast_aux` | line 407 | `__mind_load_i64(node + 48)` — binary intrinsic |
| `push_err_return_mismatch` | line 1019 | multi-`let` chain body |
| `typecheck` | line 1237 | multi-`let` + nested call body |

Oracle fragment for `ast_aux`:

```mlir
  // fn ast_aux
  %82 = const.i64 0
  output %82
  %83 = const.i64 0   // <- extra stub
  output %83
```

Emitted fragment for `ast_aux`:

```mlir
  // fn ast_aux
  %79 = const.i64 0
  output %79           // <- no second stub
```

**Root cause — EMITTER subsystem:**
`emit_fn_def` in the pure-MIND emitter always emits exactly one
`const.i64 0 / output` stub per function (the Phase 6.3b typecheck stubs
accept all bodies without inspecting them).  mindc-Rust's `lower_program`
walks the function body and emits one stub per top-level expression within the
body; for `ast_aux` the `__mind_load_i64(…)` binary-op and the implicit
return count as two items, producing two stubs.  For `push_err_return_mismatch`
and `typecheck` the multi-let chains similarly produce two body-level
expressions in the Rust AST walk.

**Scoped reproducer:**

```mind
// double_stub.mind
pub fn ast_aux(node: i64) -> i64 { __mind_load_i64(node + 48) }
```

mindc-Rust: `// fn ast_aux` + `%0` + `%1`. libmindc_mind.so: `// fn ast_aux` + `%0` only.

---

## Which subsystem is the bottleneck?

Both divergences are in the **EMITTER** subsystem (`emit_ir/main.mind`,
`emit_program_items` / `emit_fn_def`).  The lexer, parser, and typecheck stub
components are not implicated — they process the full 1,630-line source
correctly and produce the same 194-function intermediate representation.

The typecheck Phase 6.3b stubs are NOT a blocker here: the pure-MIND emitter
never calls into the full typecheck body analysis path for the fixed-point
exercise; the stubs accept all inputs and the function list is complete.

---

## What APEX (v0.6.0) already proves

- Lexer, parser, typecheck-stub, and emitter pipeline successfully processes the
  full 1,630-line combined source without crash or memory fault.
- All 194 functions are correctly identified and named in the output.
- The MLIR structure (`module { ... }  // next_id = N`) is well-formed.
- The 2-space indent format matches the oracle exactly for all 194 fn stubs.
- The fixed-point is **6 SSA values** away from byte-identical, with both gaps
  attributable to two narrow emitter behaviours.

---

## Path to PASS (v0.6.1)

Two targeted changes to `emit_program_items` / `emit_fn_def` in
`examples/emit_ir/main.mind` (and correspondingly in `examples/mindc_mind/main.mind`):

1. **Emit a stub for each `UseDecl` top-level item** in `emit_program_items`.
   Cost: ~5 lines of MIND code; requires AST node-kind constant for `UseDecl`.

2. **Match mindc-Rust's body-stub count** in `emit_fn_def`.  The simplest
   approach: detect whether the body is a binary-op or multi-let chain and emit
   two stubs for those cases.  A clean approach: walk the body statement list
   and emit one stub per top-level statement (consistent with the Rust walk).

Both changes are in-scope for a follow-on commit once the emitter's body-walk
is extended beyond the Phase 6.4 stub threshold.

---

## Harness

`examples/mindc_mind/fixed_point_smoke.py` — sibling to `bootstrap_smoke.py`.
Runs the full round-trip and exits 0 on PASS, 1 on FIRST-DIVERGENCE, 2 on BLOCKED.
