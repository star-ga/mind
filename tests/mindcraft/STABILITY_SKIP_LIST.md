# Formatter Stability Skip List

Files in this list drift from `format_source(src) == src` for known,
documented reasons.  They are EXCLUDED from the byte-equality assertion in
`tests/fmt_stdlib_stability.rs` but remain subject to the idempotence gate
in `tests/fmt_idempotence.rs` (idempotence has zero skips).

When a root cause is resolved, remove the entry and re-run the stability
tests.  A stale entry (file is now stable) triggers a warning from
`stability_summary`.

---

## MINDCRAFT-001 — AST drops `pub` keyword on `FnDef`

**Affects:** `std/vec.mind`, `std/string.mind`, `std/io.mind`, `std/map.mind`, `std/blas.mind`

**Root cause:** The parser recognises and consumes the `pub` visibility
modifier on function definitions (`pub fn foo() { ... }`) but the AST
`Node::FnDef` struct has no `is_pub: bool` field.  The information is
silently dropped.  The formatter therefore always emits `fn foo() { ... }`
regardless of whether `pub` was present in the source.

All five stdlib files use `pub fn` for their public API surface.  Every
function declaration in those files drifts from the canonical source.

**Drift example:**
```
source:    pub fn vec_new() -> Vec {
formatted: fn vec_new() -> Vec {
```

**Resolution path:**
1. Add `is_pub: bool` to `Node::FnDef` in `src/ast/mod.rs`.
2. Update `parse_fn_def` in `src/parser/mod.rs` to store the flag.
3. Update `emit_fn_def` in `src/fmt/printer.rs` to emit `pub ` when `is_pub`.
4. Update IR lowering if the flag affects symbol visibility in the IR.
5. Remove this entry and the five skip-list entries in `fmt_stdlib_stability.rs`.

**Note:** The `pub` keyword also appears on functions in `examples/`
(`examples/parser/main.mind`, `examples/typecheck/main.mind`,
`examples/emit_ir/main.mind`, `examples/mindc_mind/main.mind`) but those
files are not in the stdlib stability gate scope.  They are subject to
idempotence only.

**Note:** `blas.mind` also has one non-`pub` function (`matmul_rmajor_q16_v_row`)
which is correctly emitted as `fn` — only the `pub fn` → `fn` drift is
covered by this entry.
