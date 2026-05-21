# Formatter Stability Skip List

Files in this list drift from `format_source(src) == src` for known,
documented reasons.  They are EXCLUDED from the byte-equality assertion in
`tests/fmt_stdlib_stability.rs` but remain subject to the idempotence gate
in `tests/fmt_idempotence.rs` (idempotence has zero skips).

When a root cause is resolved, remove the entry and re-run the stability
tests.  A stale entry (file is now stable) triggers a warning from
`stability_summary`.

---

## MINDCRAFT-001 — RESOLVED

**Status:** RESOLVED.  All five stdlib files are now stable.

**Root cause (closed):** The parser recognised and consumed the `pub` visibility
modifier on function definitions (`pub fn foo() { ... }`) but the AST
`Node::FnDef` struct had no `is_pub: bool` field.  The information was
silently dropped.  The formatter therefore always emitted `fn foo() { ... }`
regardless of whether `pub` was present in the source.

**Resolution:** Added `is_pub: bool` to `Node::FnDef`, `Node::StructDef`,
`Node::EnumDef`, and `Field` in `src/ast/mod.rs`.  Updated the parser to
capture the `pub` keyword and store it.  Updated `src/fmt/printer.rs` to
emit `pub ` when the flag is set.  Reformatted all five stdlib files to
canonical form (the inline `if/else` and multi-line fn-signature style used
in `vec.mind`, `string.mind`, `map.mind`, and `blas.mind` was also
canonicalised at the same time — those were pre-existing minor drifts that
the MINDCRAFT-001 skip had masked).

All five skip-list entries have been removed from `tests/fmt_stdlib_stability.rs`.
The `stability_summary` gate now passes with 5 files stable, 0 skipped.
