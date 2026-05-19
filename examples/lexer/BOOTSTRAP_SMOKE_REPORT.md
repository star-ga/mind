# Phase 6.5 Stage 1 — Bootstrap Smoke Report

**Date:** 2026-05-18
**Branch:** main (`mind@HEAD`)
**Compiler version:** mindc v0.5.0
**Status:** BLOCKED

---

## Build command

```bash
cargo run --features "mlir-build std-surface cross-module-imports" \
    --bin mindc -- \
    examples/lexer/main.mind \
    --emit-shared examples/lexer/libmindc_lexer.so
```

**Observed exit code:** 1
**MLIR tools found:** `mlir-opt` (LLVM 18), `mlir-translate` (LLVM 18), `clang-18`

---

## Failure mode

`mlir-opt` rejects the generated MLIR with:

```
<stdin>:100:5: error: 'func.return' op must be the last operation in the parent block
    return %3 : i64
    ^
<stdin>:100:5: note: see current operation: "func.return"(%2) : (i64) -> ()
```

The MIND IR lowering pipeline produces MLIR where `return` instructions appear
in the middle of a basic block, not at the end. MLIR requires each basic block
to end with exactly one terminator; intermediate `return` ops are invalid.

---

## Root cause — three compounding gaps

### Gap 1 (blocking): `Instr::If` does not exist in the IR

`examples/lexer/main.mind` uses `if`/`return` as its only control-flow
primitive (no `while`, since Phase 6.1 predates while-statement support).
Every function body contains patterns like:

```mind
if b == 32 { return 1; }
if b == 9  { return 1; }
0
```

The AST-to-IR lowerer (`src/eval/lower.rs`, line 601) handles `ast::Node::If`
by lowering the condition and both branches **sequentially into the same flat
instruction stream**. It does not emit any conditional branching IR node —
the IR has no `Instr::If` variant. Each `ast::Node::Return` inside a then-branch
becomes an `Instr::Return` emitted mid-block.

The MLIR lowerer (`src/mlir/lowering.rs`, line 350) faithfully emits each
`Instr::Return` as `return %v : i64`, placing it in the middle of the function
body's single basic block. `mlir-opt` then rejects the output.

**Reproducer:**

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/lexer/main.mind --emit-mlir 2>/dev/null | head -40
```

The output for `@is_space` shows:

```mlir
func.func @is_space(%0: i64) -> i64 {
  %1 = arith.constant 32 : i64
  %2 = arith.cmpi "eq", %0, %1 : i64
  %3 = arith.constant 1 : i64
  return %3 : i64        ← mid-block terminator; MLIR invalid
  %4 = arith.constant 0 : i64
  ...
}
```

**Fix required:** Add `Instr::If { cond, then_body, else_body, dst }` to
`src/ir/mod.rs`, lower `ast::Node::If` into it in `src/eval/lower.rs`,
and emit `scf.if` / `cf.cond_br`+`^then`/`^else`/`^merge` basic-block
structure in `src/mlir/lowering.rs`. This is Stage 2 of the Phase 6.5 work.

### Gap 2 (secondary): `BitOp::And` not mapped to IR

`load_byte(buf, i)` uses `__mind_load_i64(buf + i) & 255`. The `&` operator
parses to `ast::BitOp::And`, but `lower_expr` has no arm for
`ast::Node::BitBinary` (the node wrapping `BitOp`). It falls through to the
`_` catch-all and emits `const.i64 0`, silently discarding the mask.

The `BinOp` enum in `src/ir/mod.rs` has no `And`/`Or`/`Xor`/`Shl`/`Shr`
variants. Both `src/ir/mod.rs` and `src/eval/lower.rs` need updating.

**Reproducer:**

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/lexer/main.mind --emit-mlir 2>/dev/null | grep -A5 "load_byte"
```

Shows:

```mlir
func.func @load_byte(%0: i64, %1: i64) -> i64 {
  %2 = arith.constant 0 : i64
  return %2 : i64
}
```

`arith.addi %0, %1` (the `buf + i`) and `arith.andi` (the `& 255`) are both
missing. The `add` is suppressed because the whole body of `load_byte` is
treated as an unhandled `ast::Node::Return { value: Some(Binary(... &...)) }`
which contains the `&` node.

### Gap 3 (latent): `let` bindings with multi-statement bodies not threading SSA across `if`

Even if Gaps 1 and 2 were fixed, functions like `classify_ident` bind
intermediate `let` values (`c0`, `c1`, `c2`) inside a nested `if` body.
The current `FnDef` lowering treats each `ast::Node::Let` in the function
body correctly, but `ast::Node::If` lowering runs both branches with a
*read-only* view of `fn_env` — mutations inside the then/else branches are
not propagated back to the outer env. This causes `c0`/`c1`/`c2` to appear
as "undefined identifier" warnings. Proper `scf.if` lowering with block
arguments resolves this automatically.

---

## MLIR tool availability

| Tool | Path | Version |
|------|------|---------|
| `mlir-opt` | `/usr/bin/mlir-opt` | LLVM 18 |
| `mlir-translate` | `/usr/bin/mlir-translate` | LLVM 18 |
| `clang` | `/usr/bin/clang` | 18.1.3 |

MLIR toolchain is present and functional. The blocker is entirely within the
mindc IR representation and lowering, not the external toolchain.

---

## What does compile and pass

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/lexer/main.mind --emit-ir
```

Produces `next_id = 36`, confirming all 36 top-level declarations are parsed,
type-checked, and reach the IR lowering stage. The `FnDef` stubs are
registered. Simple constant-returning functions (`tk_eof` through `tk_kw_pub`,
22 total) lower correctly to MLIR `arith.constant` + `return`. The failure
is confined to functions that contain `if`/`return` or `&` expressions.

---

## Python harness status

`examples/lexer/bootstrap_smoke.py` is complete and correct.
It will execute without modification once `libmindc_lexer.so` is built:

1. Loads the .so via `ctypes.CDLL`.
2. Allocates the fixture bytes via `ctypes.create_string_buffer`.
3. Calls `lex(buf_addr: i64, buf_len: i64) -> i64` (Vec handle).
4. Decodes the Vec heap record: `data_ptr` at offset 0, `length` at offset 8.
5. Reads `length / 3` stride-3 `(kind, lo, hi)` triples.
6. Compares against the 32-token expected stream derived from EXPECTED.md
   (byte offsets corrected to the actual 263-byte fixture; EXPECTED.md
   documents an older 254-byte version — delta +4 on first-block offsets).

---

## Fixture byte-offset correction

EXPECTED.md documents offsets assuming a 254-byte fixture file. The actual
`examples/lexer/fixture.mind` is **263 bytes** (LF-terminated, confirmed via
`wc -c`). The first token `use` appears at offset 181, not 177. All
token positions are shifted by +4 for the `use std.vec;` line and by the
same accumulated delta through the rest of the file. The Python harness
uses the corrected offsets verified directly against fixture content.

---

## Total tokens expected

32 tokens × 3 i64 = 96 flat Vec entries (per EXPECTED.md).

---

## Comparison result

BLOCKED — token stream comparison could not be performed; `.so` was not
produced.

---

## Stage 1 verdict

**BLOCKED-BY: Instr::If missing from mindc IR (Gap 1)**

One-line reproducer:

```bash
cargo run --features "std-surface cross-module-imports" --bin mindc -- \
    examples/lexer/main.mind --emit-mlir 2>/dev/null | grep -c "return"
```

Returns `> 1` for any function with `if`/`return` — each extra `return`
is a mid-block terminator that `mlir-opt` will reject.

---

## Stage 2 prerequisites (to unblock Stage 1)

1. Add `Instr::If { cond: ValueId, then_body: Vec<Instr>, else_body: Vec<Instr>, dst: ValueId }`
   to `src/ir/mod.rs`.
2. Lower `ast::Node::If` → `Instr::If` in `src/eval/lower.rs` (replacing the
   current sequential-flatten stub).
3. Emit `scf.if(%cond_bool) -> i64 { then_region } else { else_region }` in
   `src/mlir/lowering.rs`.
4. Add `BinOp::And` / `BinOp::Or` etc. to `src/ir/mod.rs` and lower
   `ast::Node::BitBinary` in `src/eval/lower.rs`, emitting `arith.andi` /
   `arith.ori` in the MLIR lowerer.

Steps 1–3 are required for Stage 1. Step 4 is required for `load_byte` to
produce correct output after the conditional-branch gap is closed.
