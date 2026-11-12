# Compiler part files

`main.mind` is the crate root of the self-host compiler. The compiler also has
part files (`nb_*.mind`, `nbx_*.mind`, `tc_std_surface_table.mind`). One list,
`COMPILER_COMPONENTS` in `_selfhost_sources.py`, names them in their fixed
order, and every composer reads it (native stream, `.so` build, mic3 flip
oracle, fixed-point and native-ELF smokes).

## The part rule

A part calls `std.*` and other parts only. It never calls a `main.mind` fn: in
the `Mind.toml` build each part is a separate module, and `main.mind` (the
crate root) is not importable, so such a call fails with E2003.

## Shared emitter helpers

The emitter helpers that `main.mind` and the parts share live in
`nb_emit_core*.mind` (today: `nb_emit_core.mind`). A lane part imports them:

```
import nb_emit_core;
```

They are: `load_byte`, `nb_push`, `nb_le`, `nb_slot_disp`, `nb_store_rax`,
`nb_load_rax`, `nb_i32le_signed`, `nb_emit_store_i64`, `nb_emit_load_i64`,
`nb_emit_free_xor`, the value-home group `nb_plan_reg`, `nb_val_load_rax`,
`nb_val_store_rax`, `nb_vex2`, `nb_vex3`, `tc_shoc_word_byte`,
`tc_shoc_span_word_len`, `tc_shoc_span_word_eq`, `tc_shoc_span_eq`, `nbx_put`,
and the system-call group `nb_emit_syscall`, `nb_emit_sysargs`,
`nb_load_sysarg`, `nb_load_argreg`, `nb_argreg_field`, `nb_argreg_ext`.

Never move or copy a shared helper inside a lane, and never write a private
twin of one (`nbm_le`, `nbm_store_rax` and the like). Two definitions of one
name break the composed unit; a private twin is a second encoder that can
drift from the first.

## Adding a shared helper

1. Find the helper and every `main.mind` fn it calls (its call closure), with
   `mg <symbol>`, `mg <symbol> --callers`, `mg <symbol> --impact --expand` and
   grep.
2. Check that no Rust test reads the fn out of `main.mind` by name (grep
   `tests/` and `src/` for the name). If one does, the helper stays in
   `main.mind` until that test reads the composed compiler library.
3. Move (never copy) the helper and its closure, each with its comment, into
   `nb_emit_core.mind`. If the file would pass 400 lines, start
   `nb_emit_core2.mind` and add it to `COMPILER_COMPONENTS` right after
   `nb_emit_core.mind`, and add `import nb_emit_core2;` to `main.mind`.
4. Python checks that cut encoders out of `main.mind` by name read
   `main.mind` plus `nb_emit_core.mind`; update any new one the same way.
5. Lower `MAIN_MIND_LINE_PIN` in `stdlib_manifest_lint.py` to the new
   `main.mind` line count in the same commit. Pins only go down.
6. Gate: stage2 == stage3 (`native_lane_value_shapes_smoke.py --mint-stage2
   OUT --assert-fixed-point`), program ELFs byte-identical over the native
   corpus against the previous stage2, keystone, `mic3_flip_smoke.py`,
   `stdlib_manifest_lint.py`.

## The system-call group

`nb_emit_syscall` and its closure moved into `nb_emit_core.mind` in one
commit, with the test change it needed: `tests/native_lane_extern_fixed.rs`
reads `nb_argreg_field` and `nb_argreg_ext` out of the composed compiler
(`main.mind` plus the `COMPILER_COMPONENTS` parts), and
`tests/native_lane_extern_support` mints stage1 from the same composition.
That commit lowered `MAIN_MIND_LINE_PIN` once. A part that emits a system call
(`nb_shm.mind`) imports `nb_emit_core` and calls the shared group.

## The extraction parts (0.10.3)

Three families moved out of `main.mind` verbatim, each into two parts that keep
the one-way call order (the second part imports the first):

- `nb_intern_names.mind`, `nb_intern_names2.mind`: the `intern_*` probes and the
  `load_i64_name_*`, `vec_new_name_*`, `vec_push_name_*`, `conv_name_*` helpers.
- `nb_synth_sha.mind`, `nb_synth_sha2.mind`: `nb_synth_entry` and the SHA-256
  word primitives, then the compression and block loops of the trace-hash section.
- `nb_float_lit.mind`, `nb_float_lit2.mind`: the integer-literal span decode
  (`is_digit`, `char_escape_value`, `parse_char_span`, `parse_int_span`), the
  radix-2^16 big-number helpers, then the decimal to binary64 conversion. The two
  `selftest_float_lit_*` exports stay in `main.mind`.

A Python check that cuts `char_escape_value` out of the source reads
`nb_float_lit.mind` as well as `main.mind` and `nb_emit_core.mind`.
`nbx_row` and its table fns stay in `main.mind`: `tests/native_lane_extern_fixed.rs`
cuts them out of `main.mind` alone.

## The last-definition table (0.10.3)

`nb_lastdef.mind` holds `span_bytes_equal` (moved from `main.mind`) and the
name -> last-definition table that `nb_last_def_idx` reads. Every fn array that
reaches `nb_last_def_idx` carries its table in the spare word after its last
entry (`fns[n_fns]`), written by `nb_last_def_attach` where the array is
filled: the five `all_fns` prune sites and the two `ordered` arrays. A new fn
array that reaches the prune, the dedup or a call check must be attached the
same way. The table header holds a magic word, the array address and its
length. Each fn array producer (`nb_collect_fns`, `nb_order_rest`,
`nb_build_kept`, `nb_build_kept_all`) ends its array with `nbld_end`, which
writes 0 to the spare word. A lookup on an array with no matching header
writes `mindc: internal error: a fn array reached nb_last_def_idx without
nb_last_def_attach` to fd 2 and faults on a load from address 8.

The same part holds the slot layout of the fn return-type registry
(`frt_cap`, `frt_slot_words`, `frt_count`, `frt_slot_base`, `frt_name_lo`,
`frt_name_hi`, moved from `main.mind`) and the name index that `frt_find`
reads (`nbfx_find`). `frt_alloc` reserves the index cell (word
`frt_u64_word() + 4`, 0 until the first lookup). The index is built for one
source buffer at the first lookup. A lookup with another buffer, or after a
refused allocation, scans the registry as before (`nbfx_scan`).

## The string-table index (0.10.3)

`nb_strtab_index.mind` holds `span_bytes_equal2` (moved from `main.mind`) and
the name index that `intern_find` reads (`nbst_find`). The five `offs` arrays
that `main.mind` allocates for a mic@3 string table come from `nbst_offs`,
which puts a 3-word header before `offs[0]` (an address-tied magic word and
the index). Any other `offs` array (a caller's buffer, a per-function name
table) and a table past 262144 entries take the scan, so every
answer is the first equal entry, as before. A new string-table allocation
site in `main.mind` must use `nbst_offs` to get the index.

## The lowering loop and its arena reclaim (0.10.3)

`nb_lower_loop.mind` holds `nb_set_owners` and `nb_poison_merge` (moved from
`main.mind`) and the ownership rule of a checkpoint reclaim. `nb_lower_all_ctx`
marks the arena before each function, copies the function's code into the
spare capacity of the image, and reclaims everything the function allocated
(its SSA, slot, env and code tables). `nblr_ckpt` vetoes the reclaim when the
image grew into a new buffer or a long-lived index cell points at or above the
mark, so nothing live is ever freed. A table that must outlive one function is
built before the first mark (the frt name index is built eagerly for this).
The mic@3 twin's per-item reclaim (`nb_emit_item_reclaimed`) takes the same
veto for its string-table index.

## Lexer scans and the stack budget

`nb_lex_scan.mind` holds the lexer byte classes (`is_space`, `is_alpha`,
`is_alnum`) and the sub-scans over one token's bytes (`lx_leaf` to
`scan_nl_span`), moved verbatim from `main.mind`. They read bytes only. The
token kinds and the scan loop stay in `main.mind`.

`nb_stack_budget.mind` holds the constants of the whole-program stack budget
(`nb_stack_tok_bytes`, `nb_stack_len_bytes`, `nb_stack_base`) and the
measurement they come from. `nb_stack_short` and `nb_stack_tok_cap` in
`main.mind` read them. A walk over the token stream, over a statement list or
over the items of a module is a loop: a recursion there costs one native
frame per element (the native backend has no tail calls), and the budget must
then grow with it.

## The import binding part

`nbx_bind.mind` holds the import ids of the dynamic image writer (`nbx_imports`,
`nbx_import_step`, moved out of `main.mind` unchanged) and the pure steps of
the seed-region extern binding (`nbs_*`). Its header has the binding rule. The
steps that call a `main.mind` fn (`nbx_seed_item`, `nbx_seed_walk`,
`nbx_seed_clash`, `nbx_build_image`) stay in `main.mind`. The glibc version
buckets (`nbx_version_*`, `nbx_same_version`, `nbx_row_unversioned`,
`nbx_chain_next`, `nbx_need_next`, `nbx_vers_tags`) and the writer byte helpers
`nbx_round` and `nbx_to` moved out of `main.mind` unchanged into `nb_exit.mind`.

## The print part (0.10.3)

`nb_print.mind` lowers the builtin `print` statement on the native backend: one
raw `write(1, ..)` per string literal, and per integer or bool argument one write
of the signed decimal and one of `"\n"`, the bytes and writes of the MLIR
route's runtime. `main.mind` keeps one hook in `nb_expr`'s call arm. Its mark
`nb_cap_native_print_v1` ("nbprint1") is what `src/build/native_bridge/seed.rs`
reads before the lane admits `print`. The value-home helpers it needs
(`nb_plan_reg`, `nb_val_load_rax`, `nb_val_store_rax`) moved to
`nb_emit_core.mind`, and `ast_str_lit` moved to `nb_ast.mind`, unchanged.
The stdout table is `testdata/native_lane_values/print_*.mind`, run by
`native_lane_value_shapes_smoke.py` on both routes.

## Constant array elements (0.10.4)

`nb_arr_fold.mind` holds `arr_elem_is_int`, `arrlit_all_int` and
`arr_elem_int`, moved out of `main.mind`. An array element is a constant when
it is an integer constant expression that folds by spec 3.2 (`+ - * /`,
negation, parentheses). A division by zero or an overflow does not fold. The
part mirrors the Rust fold `opt::fold::const_int` arm by arm, so the mic@3 twin
and the Rust oracle bake the same `OP_CONST_ARRAY` values. The part reads AST
kinds and binop tags as numbers, because a part cannot call a `main.mind` fn.

## The branch-write memo part (0.10.3)

`nb_branch_memo.mind` holds the memo of the native if-merge analysis: the merged
branch writes of each `if` node, made once per node and fn instead of once per
enclosing branch walk. A native let cell grows to 136 bytes (`nbwm_cell`); its word
+128 holds the memo. The walks that fill a record (`nbwm_merged`,
`nbwm_ensure_list`, `nbwm_append`) stay in `main.mind`, next to `nb_branch_writes`,
because they call it. The leaf-closed if-merge table helpers (`mt2names`,
`nb_merge_zero_diverged`, `nb_merge_ph_count`, `nb_merge_assign_slots`,
`nb_name_record`) moved here verbatim; `MAIN_MIND_LINE_PIN` went down to 68746.
The same slot table holds one count record per else list (`nbwm_cget`,
`nbwm_cput`): `nb_count_else` in `main.mind` reads the id count an enclosing
count walk made instead of a new walk per enclosing arm.
`nbwm_jcap` sizes the SSA jump table: 512 records, or 400 + 2 per `>` byte when
that is more (a `match` arm desugars to an `if` with two jump records, the router's
jump estimate does not count arms, and each arm has a `>` token in any spelling).

## The exit-carry and value-if parts (0.10.3)

`nb_exit_carry.mind` holds the branch-exit-carry pre-gate of the native router
(`nb_stmts_branch_exit_carry`, called by `hm_fn_pregate`). One bottom-up walk makes
three facts of each statement list (loop exit, assign, exit-carry branch); it
replaces `nb_block_loop_exit` and `nb_block_any_assign`, which re-walked every
later arm at each arm and had no other caller. `MAIN_MIND_LINE_PIN` went down to
67921.
`nb_value_if_memo.mind` holds `nbvi_value_if`, the `is_value_if_expr` test of
`nb_stmt` with each answer kept in the fn's branch memo under the odd key node + 1;
its import and the two-step test take four lines of `main.mind` (67925, pin 67925).
`nb_stack_budget.mind` holds `nb_stack_arm_units`: each `=>` adds 8 nesting units
to the base of its brace level (a match arm is one level of the desugared if
chain, 7968 B of stack measured per arm), so a match too deep for the compiler's
own stack is refused (rc 3). The nesting-unit notes moved there with it (pin 67923).

## The jump table part (0.10.3)

`nbt_jump_table.mind` holds the forward-jump table of a native function: the count
cell with its capacity (`nbjt_cell`), the checked push (`nb_jmps_push`) and
`nb_jmp_set_target`, moved from `main.mind`. A push past the table writes nothing,
marks the cell and poisons the function bound with `nbjt_bind`; the SSA path then
gives that function to the HOME path. Every table site makes its cell with
`nbjt_cell(records)`. `MAIN_MIND_LINE_PIN` went down to 67871.

## Twin-shaped loops and the extern scan part (0.10.3)

The mic@3 twin lays out a `while` body made of `let`, assign, a bare call, and an
`if` whose branches only assign. It refuses a nested `if` inside a loop `if`, a
value `if` in a loop `let`, an `if` that holds a `let` or a call, and a `return`
inside a loop. A loop that falls outside those shapes makes `nb_trace_hash` zero
and `selftest_native_elf_u` return an empty image (`selftest_twin_refusals` names
each refused fn). Write such a scan as one flat loop and put the per-position test
in its own fn (`ent_found_at`, `const_let_at`, `policy_attr_at`, `nb_patch_call`).
A loop that must stop early sets its result, sets the index to the bound, then
steps (`r = j; j = len; j = j + 1;`). Never bring back one native frame per element.

`nbt_extern_scan.mind` holds the second half of the extern ABI scan (`nxa_header_cc`
to `nxa_user_refusal`) in that shape; `nb_extern_abi.mind` keeps the byte scans it
calls. `nbt_fvc_names.mind` holds `tc_fvc_eq2` and `tc_fvc_is_bare_builtin`, moved
verbatim; `MAIN_MIND_LINE_PIN` went down to 68062.
