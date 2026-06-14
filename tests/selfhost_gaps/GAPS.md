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

## Fixed (cont.)
- **leading-let INIT escape bubble** (`mixed-prefix_3/4`). A leading let whose INIT is an
  escaping value if-expr — `let x = if b>0 { let m=b; m } else { let m=c; m }` — surfaces the
  inner same-name phi `m` into the enclosing block scope (the Rust lowering keeps it live after
  the let), so it must bubble into the enclosing if's merge as an extra phi. `block_init_escape_probe`
  probes each leading let's if-expr init at the init's vid base and appends its escapes BEFORE the
  value escapes (union order [lets, init-escapes, value-escapes]); the value real-append shifts
  past them. blk_layout/blk_fill_own then re-merge them uniformly. (commit: block_init_escape)

## Fixed (cont.)
- **struct-lit desugar preserves the let TYPE annotation** (`slit_emit_ctor`). The desugared
  `let __sl = __mind_alloc(..)` now carries the original let's `ast_child1` type instead of `0`,
  so a later field-read prefold (`s.a` where `let s: S = T{..}`) can resolve the receiver's
  struct type. Net +3 gap fixtures byte-exact (whole-fixture survey 29 -> 32 BYTE_EXACT of 66),
  zero regressions; the whole-module FLIP stays byte-identical (227264 B). (commit: struct-lit type-preserve)

## Progress: whole-fixture survey 29 -> 32 / 66 BYTE_EXACT (operator-edges fully closed; flip byte-identical throughout)

### Integrity breakdown (66-fixture whole-module survey, HEAD fc8e582)
- **32 BYTE_EXACT** (nfn == `--emit-mic3`).
- **28 FAIL_CLOSED (safe)** — nfn emits *nothing* for an unsupported out-of-subset construct;
  it NEVER emits wrong bytes. These are coverage gaps (call-arg/nested struct-lit hoisting,
  field-read in non-let positions, value-ifexpr escaped-outer reuse, …), not correctness bugs.
- **6 TRUE wrong-bytes miscompiles** — ALL of them `fallthrough-shadow_1..6` (one class). This is
  the only integrity target. It is out-of-subset, so the whole-module FLIP stays byte-identical.

So the driver is correct (byte-exact or safe-refuse) for 60/66; the single wrong-bytes class is
fall-through-shadow, whose proper fix is the Pass-B merge-slot synthesis described below.

## Open — MISMATCH (deep cross-pass; out-of-subset; one focused class)
1. **fall-through-shadow trailing-read** (`fallthrough-shadow_1..6`, all 6 remaining). The merge is
   byte-exact but the trailing read of a name SHADOWED inside a fall-through if resolves to the
   OUTER vid, not the merge vid. Cross-pass: the trailing value flattens in Pass A against `env`
   (which stores let SLOTS); a fall-through merge has no flatten slot, and its vid is only known
   in Pass C — and `flatten_expr_env`'s let-ident reuses the let's slot, so there is nowhere to
   inject the merge vid. Proper fix: synthesise a slot per fall-through-merge binding whose
   vidbuf is set to the merge vid in Pass B, and bind it in `env` before the trailing flatten
   (or fail-closed on shadow-read until supported). The single deepest remaining class.

## Fixed (cont.)
- **unary `-x`** (`operator-edges_4/5/6`). `-x` desugars to `0 - x` in the general
  tree flattener: `flatten_ast`/`count_nonparam_nodes` gained an `ast_neg()` arm that
  flattens the operand, then a synthetic const-0 leaf, then a `sub` binop LAST
  (post-order CONST then BINOP, byte-exact vs --emit-mic3). Covers top-level, call-arg,
  and value-if-expr-branch positions (the lv emitter recurses through the same arm).
  Additive — main.mind has no unary neg, flip byte-identical. (commit: unary neg desugar)

## Open — FAIL_CLOSED (safe refusal; in-subset coverage gaps)
- struct-lit in non-let-RHS positions: call argument, nested ctor field, field-read of a
  let-bound ctor (`struct-lit_2/3/4`). Desugar only handles let-RHS / direct-return.
- field-read `recv.field` as a value-if-expr branch / general non-let positions (`field-read_*`).
- escaped-outer-binding reuse inside value-if-expr branches (`value-ifexpr_5/6`).

These FAIL_CLOSED safely (empty output, never wrong bytes). Repros preserved alongside this file.
