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

## Progress: whole-fixture survey 66 / 66 BYTE_EXACT, 0 FAIL_CLOSED, 0 WRONG_BYTES — the nfn driver is byte-exact on the WHOLE corpus, NEVER wrong; CI-enforced by `examples/mindc_mind/gap_corpus_smoke.py` (FLOOR 66; flip byte-identical throughout)

### Integrity breakdown (66-fixture whole-module survey — measured fresh-load, fork-isolated)
- **66 BYTE_EXACT** (nfn == `--emit-mic3`) — the WHOLE corpus lowers byte-for-byte, verified
  three ways that agree: a brand-new process per fixture, fork + dlopen-in-child, and a warm
  reused handle all return identical bytes (so the lowering is NOT arena-state-dependent).
- **0 FAIL_CLOSED** — every fuzz-discovered shape now lowers; nothing is declined.
- **0 WRONG_BYTES** — no silent miscompiles. The cardinal invariant.

> **Measurement discipline (a lesson learned the hard way).** `value-ifexpr_5` and
> `mixed-prefix_12` were briefly mis-reported as fail-closed (a phantom "64/66"). Root cause:
> a **single test-harness bug** — reading the wrong field of the `EmitState` result, offset
> `+8` (`next_id`) instead of the buffer handle at offset `+0`. That one mistake produced both
> symptoms: a false "empty" (fail-closed) when `next_id` was 0, and a SIGSEGV (read as a crash)
> when it held non-zero garbage dereferenced as `(addr, len)`. The correct read is
> `EmitState.buf` (offset 0) → `String`(addr@0, len@8). Measured correctly it is 66/66, and a
> warm reused handle agrees byte-for-byte with a fresh per-process load — there is no
> arena-state divergence. The gate still loads the `.so` fresh in the child as defensive
> hygiene (it matches a real `mindc` invocation), but the lowering never depended on it.

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

## Fixed (cont.) — value-if-expr branch that shadows a sequence let (`value-ifexpr_5`, `mixed-prefix_12`)
A value if-expr used as the FN_DEF trailing value where a branch declares a `let` shadowing a
sequence let — `let v = x + 10; if c == 0 { let v = x + 1; v } else { v }`. The lowering
(`emit_if_expr_any_lv` → `blk_layout`, which resolves the else-side bare reference to the outer
`v`'s vid with no placeholder) was **already correct and deterministic**: the let-bound form
`let r = if … ; r` is byte-exact across fresh loads, and the trailing form is too. The brief
"fail-closed" was a measurement artifact (see the discipline note above), so the temporary
`ifexpr_shadows_seq` decline guard was removed — declining a shape that actually lowers
byte-exactly was needless. The whole corpus is now 66/66, FLOOR-pinned. Repros preserved as a
regression corpus. Every previously-fail-closed class — struct-lit in call-arg / if-expr-branch
positions, field-read in non-let / literal-less modules — also lowers byte-exactly (the lowering
was correct; the passes were just gated on source-literal presence; see "The big lesson" above).

## Fixed (cont.) — statement-position numeric `print(x)` (`print_lone`, `print_int_discarded`)
The Rust oracle (lower.rs `Node::Print`) lowers a statement-level numeric `print(x)` to
`<arg instrs> ; CALL printI64(arg) ; CALL printNewline() ; CONST 0` (the unit placeholder,
also the fn's value in tail position). The self-host now models this with a PARSE-TIME
statement desugar (`print_desugar_module`, before the strtab pass): every DIRECT fn-body
`print(x)` with one flatten-supported arg expands to `[printI64(x); printNewline(); 0]`,
the synthetic callee names interning first-seen at exactly the oracle's positions from
spans appended to the src copy (the slit intrinsics idiom). The seq emitter then lowers
the expansion with zero new emit code. String-arg, multi-arg, and nested-expression
`print` stay fail-closed (never_wrong locks intact). gap-corpus floor 121 -> 123; loop
anchor re-frozen; flip byte-identical.

## Fixed (cont.) — let-init call-scrutinee match (`matchcall_letinit_1`)
`let x [: T] = match g(a) { .. };` — the oracle evaluates the scrutinee call ONCE
(Call instr first), then lowers the match as a value-if comparing each pattern
against that single result vid, the outer let binding the if's dst. Parse-time
desugar in parse_block_stmts (is_fn_body==1 gated): `let TMP = call ; let X = if
TMP == P0 { .. } ..` — the parse_match_stmt hoisted-temp idiom composed with the
type-7 let-bound-to-value-if seq path. Covers typed/untyped lets and 3+-arm chains;
non-call scrutinees take the unchanged per-arm path; nested-in-if stays fail-closed.
gap-corpus floor 123 -> 124; loop anchor re-frozen; flip byte-identical.

## Fixed (cont.) — depth-2 nested array-type annotation (`nested_array_typed`)
`let m: [[i64; 2]; 2] = [[a, b], [b, a]]; m[1][0]` fail-closed at parse_let's
element-type gate, which refused any `[` element token even though the untyped
nested literal already lowers byte-exactly via the vec_new/vec_push heap path.
The gate now accepts a depth-2 nested annotation whose INNER element type is
i64/f64 (assertion-only, ty = 0, same contract as the flat case); narrow-int
inner types and depth-3+ stay refused. gap-corpus floor 124 -> 125; loop anchor
re-frozen; flip byte-identical.

## Fixed (cont.) — multi-construction bodies: unique per-construction handle spans
Every synthetic struct-construction handle (alloc-let name, store addr refs,
trailing/using refs) used to carry the SAME `__mind_alloc` source span, so two
constructions in one body made the field prefold's first-match type scan (and
letenv) mis-resolve — the reason the field-receiver hoist was gated to
first-construction-only (fr_ok). Each construction's handle now carries the
struct-lit's OWN source span (unique per site, compile-time-only, zero emitted
bytes); the alloc/store CALL callees keep the interned spans. fr_ok is retired.
Closes `prior_let_then_field_recv`, `qfield_nested`, `two_scalar_prior`; the main
corpus (incl. every single-construction struct-lit shape) stays byte-exact.
Still fail-closed (separate features): chained ident field reads (`p.q.z`),
field-receivers inside call args, nested struct-lits as field values,
struct-in-array. gap-corpus floor 125 -> 128; loop anchor re-frozen; flip
byte-identical.

## Fixed (cont.) — if-CONDITION && / || (`andor_and_ifcond_1`, `andor_or_ifcond_1`)
`if a > 0 && b > 0 { 1 } else { 0 }` — the oracle lowers the condition's Logical
as ONE nested OP_IF in the cond region (cond_count=1, cond_id = the inner if's
dst; branches = synth + value), not as nested statement-ifs. emit_mic3_if_instr
now takes a nested-OP_IF arm in all three positions (cond / then / else): an
ast_if expression probe-emits via the recursive emit_if_value_node to learn its
dst, then frames and re-emits deterministically (the established probe/re-emit
pattern); plain trees take the unchanged flatten path. 3+-operand chains
(`a && b && c`, where the desugared inner if becomes a `!= 0` binop operand)
stay fail-closed. gap-corpus floor 128 -> 130; loop anchor re-frozen; flip
byte-identical.

## Fixed (cont.) — chained struct field reads (`chained_field_pqz`)
`p.q.z` — the field prefold recursed into the receiver, folded the inner read,
then required an IDENT receiver and gave up on the outer one (the struct
registry stored field NAMES only, so the intermediate type was unknowable).
The prefold now resolves a field's declared TYPE from the source declaration
(srt_field_ty_span, on-demand scan of the `field: Type` token — no registry
reshape) and recurses through field receivers (field_access_ty), folding the
outer read on the folded receiver: `__mind_load_i64(__mind_load_i64(p) + idx*8)`,
byte-exact vs the oracle for param/let receivers, nonzero indices, and chains
inside larger expressions. Non-struct-typed intermediate fields (Vec, tuples)
stay fail-closed. gap-corpus floor 130 -> 131; loop anchor re-frozen; flip
byte-identical.

## Fixed (cont.) — field-read on a struct-lit call arg (`callarg_then_field_recv`)
`mk(P{..}.x)` — the call-arg hoist matched only BARE struct-lit args, so a
field-read wrapped around one fell through to flatten (fail-closed). The arg
scan now also matches `ast_field(struct_lit)`, hoists the construction with the
struct-TYPE annotation (the field-receiver hoist's fty idiom, so the prefold
resolves `<handle>.x`), and replaces the field's receiver with the handle ident;
the prefold then folds it to a load of the hoisted alloc — exactly the oracle's
`construction; __mind_load_i64(alloc); mk(load)`. Bare struct-lit args keep the
untyped handle (byte-identical to before). gap-corpus floor 131 -> 132; loop
anchor re-frozen; flip byte-identical.

## Fixed (cont.) — nested struct-lits as field values (`nested_slit_field`)
`P { q: Q { z: a }, w: b }` — the store fill wrapped every field value in a
value-let, so a struct-lit VALUE produced a `let … = Q{..}` with no construction
(fail-closed). The fill now runs a running offset and, for a struct-lit value,
inlines the inner construction chain (alloc + recursive stores — the oracle
allocs the OUTER record first, then evaluates nested values in order) before the
store-let that writes its handle. Depth-3 nesting verified. One real bug caught
by the never-wrong gate pre-merge: the nested chain's alloc CALL callee must
keep the interned `__mind_alloc` span (talo/tahi threading) — passing the outer
handle span emitted a call to a nonexistent name (wrong-bytes class).
gap-corpus floor 132 -> 133; loop anchor re-frozen; flip byte-identical.
