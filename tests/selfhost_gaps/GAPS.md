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

## Fixed (cont.)
- **value-if-expr branch-value escape bubble — single-expr path** (`value-ifexpr_1/3`).
  `emit_if_expr_lv` now probes each branch value's escape count and, when nonzero, routes to
  `emit_if_expr_lv_bubble`, re-merging the captured escapes via the two-sided F2 block tail
  (gated so the empty-merge happy path stays byte-identical). (commit: emit_if_expr_lv bubble)

## Fixed (cont.)
- **value-if-expr branch-value escape bubble — LEADING-LET-BLOCK path** (`value-ifexpr_4`,
  `mixed-prefix_2`). `emit_if_expr_block_lv` now probes (`branch_escape_probe`) + appends the
  branch-value escapes after the leading lets so blk_layout/blk_fill_own union them. (commit:
  block_lv bubble)

## Fixed (cont.)
- **let-shadows-param inside a value-if-expr branch** (`deep-combos_3/4`, `call-arg-nesting_4`).
  `let p = if c { let p = p + 1; p } else { p }` — an inner `let p` shadows the same-named
  param `p`. Two fixes, one logical change: (1) `blk_layout` resolves a branch binding's source
  via `lvenv_lookup` first and falls back to `resolve_param`, so the merge binds the let's vid
  (not the param's); (2) `flatten_ast_lv`'s ident arm resolves the lv-env (lets in scope) BEFORE
  params, so the trailing read of the shadowed name picks the let. Additive — main.mind has no
  let-shadows-param, so the flip stays byte-identical. (commit: let-shadows-param)

## Progress: fuzz mismatch set 26 -> 8 (70% cleared byte-exact, flip preserved throughout)

## Open — MISMATCH (deep combinations / cross-pass; out-of-subset; for a focused follow-up)
1. **fall-through-shadow trailing-read** (`fallthrough-shadow_1..6`, 6 of 8). The merge is
   byte-exact but the trailing read of a name SHADOWED inside a fall-through if resolves to the
   OUTER vid, not the merge vid. Cross-pass: the trailing value flattens in Pass A against `env`
   (which stores let SLOTS); a fall-through merge has no flatten slot, and its vid is only known
   in Pass C — and `flatten_expr_env`'s let-ident reuses the let's slot, so there is nowhere to
   inject the merge vid. Proper fix: synthesise a slot per fall-through-merge binding whose
   vidbuf is set to the merge vid in Pass B, and bind it in `env` before the trailing flatten
   (or fail-closed on shadow-read until supported). The single deepest remaining class.
2. **mixed-prefix residuals** (`mixed-prefix_3/4`, 2 of 8): a `let X = <value if-expr>` whose
   if-expr nests same-name phis / bubbles behind a multi-stmt prefix — the fixed pieces
   interacting through the let-init (type-7) path. Each needs the same probe-decode-match loop,
   additive + gated.

## Open — FAIL_CLOSED (safe refusal; in-subset coverage gaps)
- struct-lit in non-let-RHS positions: call argument, nested ctor field, field-read of a
  let-bound ctor (`struct-lit_2/3/4`). Desugar only handles let-RHS / direct-return.
- field-read `recv.field` as a value-if-expr branch / general non-let positions (`field-read_*`).
- escaped-outer-binding reuse inside value-if-expr branches (`value-ifexpr_5/6`).
- unary `-x` at top level.

These FAIL_CLOSED safely (empty output, never wrong bytes). Repros preserved alongside this file.
