# Self-host nfn driver — gap inventory (fuzz-discovered)

The whole-module mic@3 flip (`selftest_mic3_module_nfn(main.mind)` == `mindc --emit-mic3
main.mind`) is **byte-identical** — every construct `main.mind` actually uses is correct.
A 497-program fuzz across construct families surfaced gaps that fire only on shapes **outside
main.mind's subset** (so they do not affect self-host). This catalogs them for follow-up
toward a fully general mic@3 driver.

## Fixed (byte-exact, byte-identity preserved)
- **Same-name phi across value-if-expr branches** — `if c {let v=A; v} else {let v=B; v}`.
  `blk_layout` now unifies a name bound on both branches into one F2 phi (union-of-names).
  Fixed 11 mismatch repros. (commit: blk_layout union)
- **Per-ctor struct-lit alloc handle** — two `S{..}` ctors in one body. `letenv_lookup`
  last-match binds each ctor's stores to its own `__mind_alloc` handle. (commit: letenv_lookup)

## Open — MISMATCH (silent wrong bytes; out-of-subset; priority for general-compiler use)
1. **value-if-expr branch-value escape bubble** (`value-ifexpr_1/3/4`, some `deep-combos`).
   When a value if-expr branch VALUE is itself a nested value if-expr (else-if chain) whose
   branches carry lets, the nested merge bindings must bubble into the enclosing merge. The
   single-expr path `emit_if_expr_lv` emits `nb=0` and `emit_if_expr_block_lv` passes `0,0` to
   `emit_branch_value_lv`, so the nested escapes are dropped. Proper fix: probe + capture the
   branch-value escapes (vbind/vnb, already built for `emit_mixed_value_branch`) and emit them
   as merge rows (placeholder on the non-defining side), in the value-if-expr emitters.
2. **fall-through shadow trailing-read** (`fallthrough-shadow_1..6`). `let y=p; if c {let y=..};
   y` — the inner `let y` shadows the outer; the merge is byte-exact but the trailing `y`
   resolves to the OUTER vid, not the merged vid. Root cause: the trailing value is flattened
   in Pass A against `env` (letenv, slots), which has no representation for a Pass-C fall-through
   merge vid; `flatten_expr_env` drops the name span on ident leaves so post-emit re-resolution
   against the live env is blocked. Proper fix: thread the segment merge bindings into the
   trailing-value resolution (cross-pass) — or fail-closed on shadow-read until supported.
3. **mixed-prefix / call-arg / deep-combos residuals** — combinations of (1) above with prefixes.

## Open — FAIL_CLOSED (safe refusal; in-subset coverage gaps)
- struct-lit in non-let-RHS positions: call argument, nested ctor field, field-read of a
  let-bound ctor (`struct-lit_2/3/4`). Desugar only handles let-RHS / direct-return.
- field-read `recv.field` as a value-if-expr branch / general non-let positions (`field-read_*`).
- escaped-outer-binding reuse inside value-if-expr branches (`value-ifexpr_5/6`).
- unary `-x` at top level.

These FAIL_CLOSED safely (empty output, never wrong bytes). Repros preserved alongside this file.
