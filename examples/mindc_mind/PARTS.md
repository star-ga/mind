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
   `main.mind` line count in the same commit. Pins only go down. The lint
   accepts only the exact count: a longer `main.mind` and a shorter one with
   the old pin both fail.
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
enclosing branch walk. A native let cell grows to 144 bytes (`nbwm_cell`); its word
+128 holds the memo, and `nbwm_cell` itself zeroes word +136 (the allocator class, below). The walks that fill a record (`nbwm_merged`,
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

## The AST kind tags (0.10.4)

`nb_ast.mind` holds the 34 `pub fn ast_*() -> i64` kind tags that `main.mind`
held (`ast_int_lit` to `ast_field_assign`, and `ast_enum_ctor`), moved out
unchanged with their comments. `ast_str_lit` (34) was already there, so the
part now holds all 35 tags. A part that needs a kind by name imports
`nb_ast`, as 13 parts already do; its place in `COMPILER_COMPONENTS` does not
decide that, because `nb_x86_ymm_bank.mind` (fourth in the list) imports it. `ast_alloc` and the
`ast_new_*` node constructors stay in `main.mind`: they allocate and fill the
node record, and they are not tags. `MAIN_MIND_LINE_PIN` went down to 67666.
The pin lint now fails when `main.mind` is shorter than the pin too, and
`stdlib_manifest_lint.py --self-test` has one case for each direction.
`lockstep_lint.py` reads the tags from `nb_ast.mind` beside `main.mind`.

## The lane frame (0.10.4)

Three parts hold the refusal frame of the native lane. The string walk is the wired family.
Lane mode (the default) runs it. Self mode and the no-cell (.so) form skip it, so a program
lane mode refuses can still compile in self mode. A refusal keeps exit status 3.

- `nb_lane_keys.mind` is the ONE table of every key the compiler can name
  (`nb_lane_key_name`: ids 16-26 source, type and binop keys, 32-42 string keys (40, 41 and 42 are
  the three renamed refusals), 64-67 extern keys) and the packing of a refusal reason: `nb_lane_pack(key_id, site) =
  key_id * 4294967296 + site`, with `site` the byte offset of the refused node from the
  start of the composed image. A lane word is at least `nb_lane_floor()`; 1 (capacity) and
  2 (arena) keep their meaning in word 5 of the layout cell. The input magic "MINDKEY"
  (byte 1) makes the compiler print the table, one `<id> <name>` line per key; the
  committed copy is `testdata/lane_keys.txt`, and `self_host_native_diag_smoke.py`
  compares the two.
- `nb_lane_report.mind` holds the text and position helpers of the one-line report that
  `nb_report_lane` (selfhost_driver.mind) writes for a lane word: key, fn, line and column
  counted from the start of the user region (a std-region site says "in the std region"),
  exit 3, no artifact, no other backend named. The token reads stay in the driver.
- `nb_lane_rules.mind` holds `nb_lane_rules_run`, called once by `nb_lower_parts_hb_route`
  (the one hook in `main.mind`, after the item table and the fn lists exist; the ctx
  length and function-capacity guards run before the hook). Parameters are in the part
  header. It returns 0 when the mode is not lane (self mode, and the no-cell form). In
  lane mode it runs the string walk.

Mode (layout cell word 12, written by the stage driver): 1 lane (the default), 2 self (the
stdin image started with the 8-byte magic "MINDSLF" and byte 1, before the v1 or v2
header), and no cell (the shared-library form) is off. The one sender is
`_selfhost_sources.self_mode_image`: the self-compile, mint, advance and reseed paths feed the
compiler source through it, and it adds the magic only for a compiler that carries it.
`stdlib_manifest_lint.py` allows the input magics in a named list of files only. The host's
first word is `user_lo` or the layout magic, so it never sends them. A third magic,
"MINDHOOK", is a test input: its next 8 bytes are the word `nb_lane_rules_run` returns in
lane mode (cell word 13).

Three refusals the compiler already made are now named: a raw line feed in a string literal
(`lane.string.raw_newline`) and an escape the lexer does not know (`lane.string.escape`), both
named in the driver's failure path from the lexer's refusal, and a literal the rodata window
refuses (`lane.string.literal_capacity`), recorded where the window check fails
(`nb_lane_literal_full`, one call in `nb_expr`). A chain of fns that each pass a literal of 65,000 bytes to a String parameter
shows it: 129 fns compile (8,385,000 bytes), the 130th (8,450,000 bytes) crosses the 8,388,608-byte
window, and both committed seeds name it in fn `g129` (the seed committed before this release gave the
generic text). On the direct feed with literals of 1,000,000 bytes, the committed loop seed builds 8 fns
and names the 9th `lane.string.literal_capacity`, and the committed bounded seed gives the generic text
for both chains; one literal of 8,388,609 bytes gets the generic text from the loop seed and the
stack-budget text from the bounded seed. The cause of these generic refusals is not identified.
`MAIN_MIND_LINE_PIN` is 67850.
## The native lane string rule (0.10.4)

`nb_lane_scan.mind` (the byte scans), `nb_lane_str.mind` (the walk record, the
binding rule, the call table) and `nb_lane_str2.mind` (the node handlers and the
entry) hold the string rule of the native lane as ONE rule family, `nb_lane_str_walk(src, items_addr, items_len, user_lo,
fns, n_fns)`. It returns 0 or `nb_lane_pack(key, site)` (`nb_lane_keys.mind`):
`key * 4294967296 + site`, site = byte offset of the refused node in the
composed image. The parts read no `main.mind` fn. The caller passes the source
buffer, the top-level items, `user_lo` and the kept fn array.

A String is a PARAMETER type: a param whose type text is exactly `String`. A
String value is a name that resolves to such a param (a `let` of the same name
in an enclosing block shadows it, and the shadow ends with that block) or a
string literal; parentheses are transparent. Keys are the fixed ids of the
interface (16 source.string, 17 type.mismatch, 18 binop.non_scalar, 19
source.neg_type, 20 source.cast, 21 source.print_argument, 22 source.struct,
23 source.index, 24 source.method_call, 32 string.argument, 33 string.result,
34 string.binding, 35 string.const, 36 string.assign, 37 string.equality,
38 string.cap, 39 string.use).

- A call argument in a String slot of a kept fn is a literal or a resolved
  String name, else `lane.string.argument`. A String name in another slot is
  `lane.type.mismatch` (an extern callee: `lane.string.use`; `print`:
  `lane.source.print_argument`). A literal in another slot is
  `lane.source.string`, except a `print` argument.
- `.addr` and `.len` of a resolved String name are plain i64 values. `.cap` is
  `lane.string.cap`. Any other field of a String name is `lane.source.struct`.
  A field read on a receiver that is KNOWN to be a scalar (a number, an
  operator result, a cast, an if-expression, a name bound to a scalar param or
  let, a call with a scalar result or of an extern) is `lane.source.struct`; a receiver of a
  record type is not judged by the walk (the seed reads its fields). A method
  call on a String is `lane.source.method_call`.
- Refused by name: a String result, a `let` of String (annotated: at the let;
  unannotated: at the init), a String `const`, an assignment to a String name
  or to its field, `==` and `!=`, any other operator (two Strings:
  `lane.binop.non_scalar`; one String and an i64: `lane.type.mismatch`),
  unary minus, `!`, a condition (also the operand of `&&` and `||`), a cast, an
  index base, any other use (`return`, an assignment value, a block tail).
- The refusal with the smallest site wins. At the same site the inner node
  wins (`"abc".len` and `"x" == a` are `lane.source.string`).
- The walk covers every user fn (called or not), every kept std fn and every
  const item. Only String things are judged; every child of every kind is
  visited. One explicit work stack (4 words per item: node, context, shadow
  list, depth) replaces recursion, so the walk costs no native frame per node or
  per list element and every loop has the twin shape.
- A call finds its callee in a table of the kept fns that have a String param.
  Another callee is looked up in the kept fns only when an argument is a String
  name, to tell a known fn from an extern.
- Not walked: the arguments of a method call (the node keeps only the count)
  and the names the Rust check owns (`lane.type.named`, `lane.source.binding_type`).
- A literal the lexer refuses (a raw line feed, an escape outside n t r 0 \\ ' ")
  is named by the walk (`nb_lane_scan.mind`, `nbls_lex_scan`: one byte scan of
  the user region that skips `//` comments and char literals as the lexer does)
  with key 40 or 41 at its opening quote. `nb_lane_str_walk` compares that word
  with the walk's first refusal: the smaller site wins, the literal at the same
  site (as inside the family: the refused construct with the smallest source position wins,
  and at one position the inner node wins). An unclosed literal gets no key.
- Order across families: the string family first (the whole image, with the position
  rule above inside it), then the extern family in declaration order. The hook runs before the
  extern check, so a string refusal at a later position wins over an extern
  refusal at an earlier one (one row pins it).
## The extern key parts (0.10.4)

`nb_lane_keys.mind` holds `nb_lane_pack(key_id, site)`, the one packing of a lane
refusal word: `key_id * 4294967296 + site`, the site being the byte offset of the
refused node from the start of the composed image. `nbt_extern_keys.mind` holds
`nb_lane_extern_check`, the one rule family of the extern declarations: it takes
the facts of one declaration (convention, variadic tail, table index of the name,
parameter count, packed signature words, the row record) and returns 0 or the word
of the first failing step, in this order: 67 `lane.extern.callconv`, 64
`lane.extern.unknown_symbol`, 65 `lane.extern.arity`, 66
`lane.extern.signature_mismatch`. `main.mind` `nbx_check_row` passes the facts and
`nbx_refused` keeps the first word in the stage's reason cell (`nb_xcell(5)`).
A declaration of a name is judged against the row of its name whatever other
declarations of the name exist, so a repeated declaration with equal types is
bound once (the table binds the first row of a name) and a repeat with other
types is refused. The registry of the native lane is built tolerant (word 6 of
the registry is 1, `nb_cabi_build_mode`): a repeated name is a row of its own, a
convention with no row is code 4 or more, a `__mind_` name is a declaration (no row
has such a name: 64) and a parameter or result type that is no scalar of the registry
is kept as tag 127 (`nb_lane_type_tag`; it has no code in `nbx_code`, so the
signature word is -1 and the declaration is 66, or 65 when its count differs), so a
defect that stands earlier in the source keeps its word. The strict registry of the
gates (word 6 is 0) still refuses a repeated name, a `__mind_` name, a type outside
the set and an unknown convention.
## The allocator class part (0.10.4)

`nb_alloc_select.mind` chooses the allocator of a default-layout native image from
the composed source: BUMP (the inline bump core) or RECLAIM (the free-list runtime,
capped at the arena). `nbas_alloc_flag` returns 16 or 0, which `main.mind` adds to
the per-compile flags word (bit 4) and stores in `parts + 200`; each production
lowering cell holds it at `lcell + 136` (`nbwm_cell` zeroes the word, so a hand-made
driver cell reads BUMP). `nb_alloc_freelist(am)` is true for an explicit layout or
class RECLAIM, at the alloc, realloc and free decisions and in `nb_rt_prefix`.
`nb_emit_alloc_ctx` stays as a 3-argument wrapper of `nb_emit_alloc_am`;
`nb_emit_intrinsic_ctx` and `nb_materialize_field_array_ctx` take `am` as their last
argument. The walk is one flat worklist: each emitted fn is walked once at depth 0
and, when a call at a loop position names it, once more with its whole body at a
loop position; every loop has a helper fn per position (twin-shaped loops above).
The directive scan reads the leading `//` block of the user region. The part calls
`nb_ast.mind` and `nb_lastdef.mind` only. `MAIN_MIND_LINE_PIN` went up to 67668. The dev
drivers (`native_emission_context.mind`,
`native_scoped_source_image.mind`) allocate a 208-byte `parts` record, the size the
route writes: an i64 at `finished + 192` on every successful return, and the class
word at `parts + 200` when `ctx == 0`. An emission-context image is always BUMP.
