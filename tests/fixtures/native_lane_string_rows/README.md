# String-rule acceptance rows

Acceptance rows for the string rule of the native compiler.
A compiler minted from these sources passes every `run` row. The committed seeds carry the rule
since the seed advance of this release (row `binding` is refused with `lane.string.binding` on both).
In this file "the frozen seed" is the seed before that advance (the `compiler_sha256` line of
`baseline_frozen_seed.tsv`): it has no rule (a refuse row compiles on it), which is why the runner
self-test feeds that seed; CI reads it from history as the seed file before its last change.

## Files

| file | content |
|------|---------|
| `rows.tsv` | one row per program: `id verdict key line col exit profile status clause note` |
| `programs.txt` | the programs, one record each (`=== <id> | text`, `hex` or `recipe`; `=== <id>@carrier | carrier` for a std module) |
| `keys.tsv` | the refusal keys a row may name; a row with any other key is rejected |
| `mutants.tsv` | rule arm -> the rows that must fail when the arm is off |
| `baseline_frozen_seed.tsv` | what the frozen seed does with every row (recorded, nothing asserted) |
| `selftest_judge.tsv` | cases of the runner self-test: a row, what a compiler did, and whether the row holds |
| `selftest_plants.tsv` | planted defects of the runner self-test: an edit of a scratch copy of these files that loading must reject |

Programs are not `.mind` files, so the front-end differential corpus under `tests/fixtures`
does not collect these deliberately refused programs. A program with a control byte is stored
as hex. A program too large to store (the capacity rows) is a recipe: `<count> <json string>`
lines, the program is the concatenation of each string repeated `count` times.

## Columns

- `verdict`: `admit`, `refuse`, `by_run` (the verdict is whatever today's compiler gives) or
  `pending` (capacity rows).
- `key`: the refusal key of a `refuse` row; the recorded key of a `by_run` row, if any.
- `line`, `col`: the refused line (the program text counts its comment lines) and column, where
  known. A column is the character of the refused site (a name, a quote, or the word `String`).
- `exit`: the exit status of the image of an `admit` row, where one is known. Empty: the row
  asserts only that the program compiles.
- `profile`: `both` (default and bounded profile), `default` only or `bounded` only.
- `status`: `run` rows are asserted. `host` rows are refuse rows whose key belongs to the existing check
  of the host, not to the compiler fed directly (below): they run through `mindc build` when the runner is
  given a mindc binary and are otherwise counted in their own count, never as a pass and never as skipped.
  A `host` row of a closure key states no line: the closure check of the host reports by symbol. Its note
  starts with the word `symbol`, the name between backquotes and a colon; the runner asserts the key and
  that symbol on one line of the report.
  The others are kept in the file and skipped, and the runner counts them: `capacity` (the capacity limits are not settled), `by_run` (the expectation is
  taken from a run of today's compiler), `carrier` (the row needs a std module in the std blob,
  the `@carrier` record, and is not asserted yet), `outside` (an `admit` row whose program the frozen seed refuses with
  exit status 3 and no key, and the string rule does not decide it: see below).
- `clause`: the clause of the rule the row tests (below), or a short name for a decision.

## The rule these rows test

- S1: `String` is a parameter type of a function.
- S2: an argument in a String slot is a string literal, or a name that is bound to a String
  parameter of the enclosing function (resolved binding, not spelling). Parentheses are transparent.
  Anything else: `lane.string.argument`.
- S3: `.addr` and `.len` of a String parameter are plain i64 values. No bounds proof.
- S4: a literal's bytes are read-only, exactly `len` bytes are defined, there is no hidden
  terminator. The handle of a literal is built once per program.
- S5: refused by name: String as result, a `let` of type String or with a String value, a const
  of type String, assignment to a String name or field, `==` and `!=`, `.cap`, any other operator,
  unary minus, a cast, a condition or logical use, a mix with i64, print of a String value, any
  other field or a field read on another receiver, String as extern argument or return value. A
  literal anywhere but a String slot or a print argument keeps `lane.source.string`.
- S6: the same for user and std functions. Every user function is walked, called or not; a std
  function only when the image emits it. A std const is not judged: one std const that holds a
  string, or of type String, does not refuse a program (`std_const_lit_unread`, `std_const_string_unread`, arm `m_std_const_region`).
- Named decisions: a raw line feed in a literal is `lane.string.raw_newline`; an unknown escape is
  `lane.string.escape`; of two refused constructs the smaller source position wins, and at the
  same position the inner node. The refused literal is named by the walk itself (it reads the same bytes
  as the lexer, comments and char literals included) at the position of its opening quote, and the walk does
  not judge it as an argument; it takes part in the tie rule like any refused construct.
  A `mindc build` of those programs stops in the host lexer (`error[parse][E1001]`) before the
  compiler. The rows assert the direct feed, which reports `lane.string.escape` or, when `.cap`
  is earlier, `lane.string.cap`.
- Order across families: the string family first (the whole image, the tie rule above inside it), then
  the extern family in declaration order. A string refusal at a later position therefore wins over an
  extern refusal at an earlier one (`order_extern_then_string`; `order_extern_alone` is its control).
- Keys 25 (`lane.type.named`), 26 (`lane.source.binding_type`) and the closure keys stay with the existing
  check of the host in 0.10.4; their five rows (`str_refused_lowercase_type`, `edge_15`, `edge_16`,
  `ptr_cast_len`, `user_collide`) have the status `host`.
- A program that `mindc build` refuses in the host check before the compiler is fed has two rows. The row
  `<id>` (status `run`) asserts the compiler's own key and line on the direct feed; the row `<id>_host`
  (status `host`, a copy of the same program record) asserts the key that the host check prints. 32 rows
  have such a twin (for example `equality` and `equality_host`). The `run` row is the one that the
  compiler arms of `mutants.tsv` name: a `host` row never reaches the compiler's rule. Two twins have a third row on the
  bounded profile: `s5_extern_arg_host_bounded` and `s6_extern_arg_host_bounded` (status `host`, profile `bounded`)
  assert `scope.bounded_extern`, the refusal of every user extern block that the host check gives on that profile
  before the compiler is fed; their twin `_host` rows assert the compiler's key on the default profile.

## Counts

| | rows |
|---|---|
| all | 568 |
| `admit` | 358 (347 `run`, 9 `outside`, 2 `carrier`) |
| `refuse` | 191 (152 `run`, 39 `host`) |
| `by_run` | 14 |
| `pending` (`capacity`) | 5 |
| asserted directly (`run`) | 499 |
| host path, not run without a mindc binary (`host`) | 39 |
| skipped | 30: `by_run` 14, `outside` 9, `capacity` 5, `carrier` 2 |

`refuse` rows by key: lane.source.struct 18, lane.type.mismatch 16, lane.source.string 14, lane.string.use 14,
lane.string.equality 13, lower.materialization_refused 13, lane.call.unknown 12, lane.string.argument 12,
lane.string.cap 12, lane.string.binding 10, lane.string.result 9, lane.string.assign 8,
lane.binop.non_scalar 7, lane.source.cast 5, lane.string.escape 4, lane.string.raw_newline 4,
lane.type.named 3, lane.source.index 2, lane.source.method_call 2, lane.source.neg_type 2,
lane.source.print_argument 2, lane.source.top_level 2, lane.string.const 2, scope.bounded_extern 2,
closure.std_symbol_collision 1, lane.extern.unknown_symbol 1, lane.source.binding_type 1.

The nine `outside` rows hold a construct that is not a string construct, and the frozen seed refuses
each program (the baseline table shows exit status 3 and no key): four hold a unary `~`
(`mv_bit_not_moved`, `mv_bit_not_kept`, `rs_bit_not_moved`, `rs_bit_not_kept`), three hold a top-level
`const` item (`c2_shadow_no_let`, `c4_const_read`, `c5_sibling_scope`) and two hold an `extern`
function declaration (`s3_extern_cast`, `addr_cast_admit`). Set the status to `run` when a compiler
compiles that construct. Until then the arm `m_s3_addr_cast_extern` has no asserted row.
`m_s6_std_walk` asserts `s6_std_reached_cap` and `s6_std_reached_eq`: the runner appends each row's
`@carrier` record (lines whose first non-space character is `#` omitted) to the std blob and counts
those bytes in `user_lo`, so the refusal is in the std region. `m_std_unreached_judged` still has only
`carrier` rows, and `m_r2_exact_name` has only `host` rows: `--string-rows-mutant` exits with status 2
for an arm with no `run` row and says so.

The nine `fcast_*` rows (after `cast_paren`) put `.addr` or `.len` of a String parameter in a pointer
cast (S3): the four `admit` rows read a byte back through the pointer, return the length or pass the
pointer to the `readlink` extern row, so the exit status depends on the cast value (arm
`m_field_ptr_cast`). Parentheses around that receiver are transparent in the same cast:
`paren_recv_ptr` is `((s).len as *mut u8) as i64` and exits 42. `paren_recv_ptr_nest` is
`(((s)).len as *mut u8) as i64` and exits 42. `paren_recv_len` and
`paren_recv_cast` pin the field read; `paren_recv_ptr` pins the cast proof (arm `m_cast_field_paren`).
One unwrap of that receiver is not enough: `paren_recv_ptr_nest` is the row of arm `m_cast_field_paren_one`.
Two unwraps still admit that row. `paren_recv_ptr_nest3` is `((((s))).len as *mut u8) as i64` and exits 42.
`paren_recv_ptr_deep` wraps that receiver in sixteen parentheses and exits 42. A fixed unwrap of two
parentheses is not the proof: the proof strips every parenthesis. Those two rows are arm `m_cast_field_paren_two`.
No row of fixed depth pins every parenthesis, except a row at the nesting ceiling. The nesting budget
(`nb_depth_short`, units in `nb_stack_budget.mind`) charges 4 units for a function body `{` and 16 for each `(` on the
path. It admits at most (stack - 32 MiB) / 1280 B units, and the stack is the one of the RUNNING compiler
(`nb_own_stack_bytes`: a native image reads the memsz of its highest PT_LOAD, of any size; 384 MiB is only the
fallback when it can not read its own headers). Every String receiver sits in a function
body, so it carries at most (units - 4) / 16 parentheses: 18022 on the 384 MiB stack of the bounded compiler (288358
units) and 208076 on the 4 GiB stack of the default-profile compiler (3329228 units). So the ceiling records hold no
fixed count: a count `ceiling`, `ceiling-1` or `ceiling+1` in `programs.txt` gets the ceiling of the compiler under
test (`at_ceiling` in the runner: the same arithmetic on the PT_LOAD of that ELF). `paren_recv_ptr_ceiling` (a recipe)
wraps that receiver in `ceiling-1` parentheses: the deepest receiver of that program shape (an outer parenthesis
around the cast). It exits 42. `paren_recv_ptr_ceiling_bare` (a recipe) drops the outer parenthesis
(`return <ceiling parentheses>s<...>.len as *mut u8 as i64;`) and exits 42. Its `@beyond` record is the same program
one level deeper: `--string-rows` asserts it on both profiles, and it must meet the stack budget refusal (exit status
3, no image, `stack budget` on stderr). So the row is red when the arithmetic of the runner and the ceiling of the
running compiler differ. `--string-rows` also runs the two ceiling rows and the `@beyond` record on the default-profile
seed of the tree (`testdata/selfhost_loop/stage1.elf`, the compiler that `mindc build` embeds for the default
profile), one compile at a time (each holds up to about 3 GiB: maxrss 2.7 to 3.0 GiB measured at the ceiling). Measured on that seed: 208076 parentheses compile and the
image exits 42, 208077 meet the stack budget refusal. A fixed count of unwraps below the ceiling of the compiler that
runs the rows turns that row red: below 18022 (sixteen, 18020 or 18021) on a bounded build, below 208076 on a
default-layout build. A count of 208076 or more acts the same as the proof on every program that the two profiles of
this compiler admit; a compiler with a larger stack gets deeper rows from the same records. Both rows are arm
`m_cast_field_paren_fixed` (sixteen or 18020 unwraps), and `paren_recv_ptr_ceiling_bare` alone is arm
`m_cast_field_paren_last` (a count from 18021). Run that arm on the bounded build of the mutant for 18021, and on the
default-layout build of the mutant for a count from 18022 to 208075.
The frozen seed refuses all nine (exit status 3, no key). `.cap`, an i64 receiver
and a let that shadows the parameter keep the key of the walk; `fcast_struct_recv` (a parameter of
another struct) is `by_run`: the walk names no key and the cast refuses with exit status 3. A `let` of
any type, or a loop binder, that shadows a String parameter ends the String binding in its scope: a field
read of that name is `lane.source.struct` (`fcast_shadow_struct` in the pointer cast, `shadow_struct_read`
as a plain read of another struct's field, `shadow_loop_read` on a loop binder).

The two rows `lim_alloc_over` and `g0_alloc_loop` run on the default profile only: their exit status
132 is the arena-guard trap (the image is killed by signal 4, which the runner reads as 128 plus the
signal), and on the bounded layout the frozen seed ends them with 54 (2^25 allocations of 24 bytes are
805306368 bytes).

## Baseline of the frozen seed

`baseline_frozen_seed.tsv` is the output of `--string-rows-baseline` on the frozen seed
(`examples/mindc_mind/testdata/selfhost_loop/stage1.elf` before the seed advance of this release; the
committed seed names `lane.extern.unknown_symbol` for the two `extern` rows of the nine `outside` rows):
one line per row and profile of the corpus
that table was recorded from. The key column is `none` on every line: the seed names no refusal key,
so a refusal shows only as exit status 3. Both profiles give the same exit status on every recorded row.
Rows added after that recording are not in the table. Nothing in the table is asserted.

| rows | frozen seed |
|---|---|
| `admit`, `run` (326) | 322 compile (exit status 0); the 4 `fcast_*` rows exit with status 3 and no key |
| `refuse`, `run` (130) | 94 compile; 36 exit with status 3 and no key |
| `refuse`, `host` (5) | 4 compile; 1 exits with status 3 and no key |
| `admit`, `outside` (9) | all 9 exit with status 3 |
| `by_run` (14) | 6 compile, 8 exit with status 3 |
| `capacity` (5) | all 5 compile |
| `carrier` (4) | 1 compiles, 3 exit with status 3 |

So a compiler with the rule must turn the 94 compiling `run` refuse rows into exit status 3 and print the key
(and the line) on all 130 `run` refuse rows. The compile status of an `admit` row does not show a wrong exit status of its
image: `--string-rows` runs the images that have an `exit`.

Re-record the table with `--string-rows-baseline STAGE1 --out TABLE` and compare it with this file.

## Runner

The count lines of this file (the Counts table, the keys of the `refuse` rows, the host count and the number of
twins) are checked when the rows load: a count that the rows do not have stops every mode (`readme_counts` in the runner).

All modes belong to `examples/mindc_mind/native_lane_value_shapes_smoke.py`; the code is in
`examples/mindc_mind/_selfhost_string_rows.py`. A program goes straight to the compiler ELF on stdin
(the seed's direct-feed protocol, the std blob of the seed modules in front, no self-mode input magic),
once per profile the row names: the default header and the bounded layout header. A row that has an
`@carrier` record gets that module (every line whose first non-space character is `#` omitted) appended
to the std blob, and `user_lo` counts those bytes, whether the row's status is `run` or `carrier`.
Only a `run` row is fed.

```
python3 examples/mindc_mind/native_lane_value_shapes_smoke.py --string-rows STAGE1 [--string-rows-mindc MINDC]
python3 examples/mindc_mind/native_lane_value_shapes_smoke.py --string-rows-baseline STAGE1 --out TABLE
python3 examples/mindc_mind/native_lane_value_shapes_smoke.py --string-rows-mutant ARM STAGE1
python3 examples/mindc_mind/native_lane_value_shapes_smoke.py --string-rows-selftest STAGE1
```

- `--string-rows`: an `admit` row compiles (exit 0, an image) and, when the row has an `exit`,
  the image exits with it. A `refuse` row exits 3, writes no image, and one stderr line names the
  key and, when the row has a line (and column), that position as `line N`, `:N`, `N:M` or
  `line N, column M`. The summary line counts rows per verdict and skipped rows per status, and
  names the failing ids. Exit status 0 all pass, 1 a failure, 2 not run (bad row files, no rows).
  The `host` rows are not fed to STAGE1: with `--string-rows-mindc MINDC` each runs through
  `MINDC build PROGRAM --backend native --native-profile P` with `MINDC_NATIVE_ELF=STAGE1` (the host's own
  check runs first) and is judged like a refuse row (a failure gives exit status 1); without the option the
  runner prints `host path, not run: 39` with the ids, a count of its own. Such a row is never a pass and is
  not in `skipped`. `--string-rows-mutant` runs only `run` rows: an arm whose rows are all `host`
  (`m_r2_exact_name`) cannot be checked there, like an arm with no `run` row (exit status 2).
- `--string-rows-baseline`: records one line per row and profile (`id profile rc key image`) and
  asserts nothing. `baseline_frozen_seed.tsv` is this table for the frozen seed.
- `--string-rows-mutant ARM STAGE1`: STAGE1 is a compiler built with that rule arm switched off.
  Every `run` row of the arm in `mutants.tsv` must fail. Exit status 1 names the rows that do not.
  Skipped rows of the arm are counted; an arm with no `run` row cannot be checked (exit status 2).
- `--string-rows-selftest STAGE1`: checks the runner itself. These checks need no compiler: the
  judgement of a row on known outputs (`selftest_judge.tsv`); scratch copies of the rows with one
  planted defect (`selftest_plants.tsv`; each is rejected); the runner against four stand-ins
  (a compiler that refuses and reports whether the feed began with the bounded layout header, one
  that accepts and writes an image that does not start, one that never finishes, and a host binary
  that names no key), which pins: a row runs once per profile it names and the bounded feed starts
  with the layout header; `--string-rows` exits 1 on a failing row or a failing host row, 0 when
  every row passes and 2 when no row runs; `--string-rows-mutant` exits 2 for an arm with no `run`
  row and for an unknown arm; a missing compiler exits 2; `--string-rows-baseline` exits 2 when
  there is no row; a compiler that does not finish in time reads as rc 124; an image that does not
  start has no exit value. Then the mutants of the runner: a guard is switched off in a copy of its
  source and a check must turn red (the list is `_selftest_mutants` in
  `_selfhost_string_rows.py`). With STAGE1, a compiler without the rule: a refuse row is red, a
  planted wrong exit is reported, a mutant arm of refuse rows is red and one with a green admit row
  is reported.
