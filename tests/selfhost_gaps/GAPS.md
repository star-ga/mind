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

## Progress: whole-fixture survey 64 / 66 BYTE_EXACT, 2 FAIL_CLOSED, 0 WRONG_BYTES — the nfn driver is byte-exact on nearly the whole corpus, NEVER wrong; CI-enforced by `examples/mindc_mind/gap_corpus_smoke.py` (flip byte-identical throughout)

### Integrity breakdown (66-fixture whole-module survey — measured fresh-load, fork-isolated)
- **64 BYTE_EXACT** (nfn == `--emit-mic3`) — lowers byte-for-byte.
- **2 FAIL_CLOSED (safe, deterministic)** — `value-ifexpr_5`, `mixed-prefix_12`: a value
  if-expr branch whose `let` SHADOWS a sequence let (`let v = x + 1` shadowing seq `v`).
  Its branch emit reads an under-initialised scratch slot, so the output is arena-state-
  dependent (a warm/reused .so produced byte-exact bytes — a *fake* pass the fresh-load gate
  exposes). Declined DETERMINISTICALLY (`ifexpr_shadows_seq` guard) rather than emit
  non-determinism; the proper byte-exact lowering is the remaining work.
- **0 WRONG_BYTES** — no silent miscompiles. The cardinal invariant.

> **Measurement caveat (load-bearing).** Survey each fixture with the .so loaded FRESH
> (fork + dlopen in the child, as `gap_corpus_smoke.py` does), NOT a warm/reused handle:
> the driver's per-process bump arena can leak prior-call bytes that masquerade as a
> byte-exact result. A warm survey over-reports byte-exact by these 2 fixtures.

### The big lesson — the dominant gap was PASSES GATED ON SOURCE LITERALS, not missing lowering
The struct-lit desugar/annotation and the field-read prefold were each gated on the source
literally containing `__mind_alloc`/`__mind_store_i64`/`__mind_load_i64` (searched to get the
name spans the synthetic alloc/store/load callees intern from). A module that constructs a
struct or reads a field but never spells those intrinsics had no span → the pass was skipped →
the construct fell closed. main.mind self-hosts only because its OWN body contains all three.
Fix (`build_src_intrinsics` in `selftest_mic3_module_nfn`): when the source lacks a literal,
append it to a src copy (invisible to the lexer, which scans `[0,src_len)`) and use
appended-offset spans; when present (main.mind) keep the real spans BYTE-FOR-BYTE so the flip
is unchanged. This single gate fix closed the struct-lit-as-expression class (deep-combos,
value-ifexpr_7, fallthrough-shadow_7, mixed-prefix_8) and the literal-less field-read class
(mixed-prefix_9). Remaining singletons: an unresolvable field on a non-struct receiver →
synthetic `CONST 0` (value-ifexpr_8); a both-branches-`return` if-else body → one OP_IF with a
`Return` per branch and a phantom merge dst (`emit_mic3_if_both_return_instr`, let-ifexpr-seq_5).

## Fixed (cont.) — fall-through-shadow class (the last wrong-bytes; `fallthrough-shadow_1..6`)
A name bound outside a fall-through if and SHADOWED inside it (`let y=p; if c { let y=p+5 } y`)
merges as an SSA F2 phi; a later read of that name must resolve to the MERGE vid, not the pre-if
outer slot. Two facets, one root:
1. **Over-emission** (`fallthrough-shadow_3`): a then-block binding a name twice (leading let +
   same-name bubbled if-segment binding) emitted two escaping merges; deduped via
   `bind_append_dedup` (first position, last value) before `caseB_layout`.
2. **Trailing-read resolution** (`fallthrough-shadow_1/2/4/5/6`): the cross-pass slot problem —
   the trailing value flattens in Pass A against `env` (let SLOTS); a fall-through merge has no
   flatten slot, vid known only in Pass C. Fix: `seq_fix_deltas` synthesises a placeholder slot
   per escaping binding (`synth_rebind_slots`) and records (name, slot, base0-merge_id) into the
   plan (+48/+56); Pass B (`seq_assign_vids`/`seq_set_rebind_vids`) sets vidbuf[slot] =
   entry_base + merge_id; the driver weaves the slots into a POSITION-ORDERED trailing env
   (`build_trail_env`) so the FN_DEF result resolves a shadowed name to its latest binding via
   letenv_lookup last-match (a later seq let shadows an earlier merge — required for `scan`,
   which re-binds `kind`/`next_toks` at the seq level after an early-return if). Additive: a
   program with no shadow-then-read keeps the whole-module flip byte-identical (231447 B).

## Fixed (cont.)
- **unary `-x`** (`operator-edges_4/5/6`). `-x` desugars to `0 - x` in the general
  tree flattener: `flatten_ast`/`count_nonparam_nodes` gained an `ast_neg()` arm that
  flattens the operand, then a synthetic const-0 leaf, then a `sub` binop LAST
  (post-order CONST then BINOP, byte-exact vs --emit-mic3). Covers top-level, call-arg,
  and value-if-expr-branch positions (the lv emitter recurses through the same arm).
  Additive — main.mind has no unary neg, flip byte-identical. (commit: unary neg desugar)

## Open — value-if-expr branch that shadows a sequence let (`value-ifexpr_5`, `mixed-prefix_12`)
The only remaining gap: a value if-expr used as the FN_DEF trailing value, where a branch
declares a `let` shadowing a sequence let — `let v = x + 10; if c == 0 { let v = x + 1; v }
else { v }`. The branch emit (`emit_if_expr_any_lv` → `blk_layout`) reads an under-initialised
scratch slot for the shadow, making the result depend on the bump-arena state. Declined
DETERMINISTICALLY by the `ifexpr_shadows_seq` guard (fail-closed, never wrong). The proper
fix is to zero-or-fully-initialise that branch-layout slot so the shadow lowers byte-exactly
(then raise the `gap_corpus_smoke.py` FLOOR to 66). Every other previously-fail-closed class —
struct-lit in call-arg / if-expr-branch positions, field-read in non-let / literal-less
modules — now lowers byte-exactly (the lowering was correct; the passes were just gated on
source-literal presence; see "The big lesson" above). Repros preserved as a regression corpus.
