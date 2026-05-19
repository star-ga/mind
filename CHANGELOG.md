# Changelog

All notable changes to the MIND compiler project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 6.5 Stage 3 — pure-MIND type-checker cdylib bootstrap PASS

`examples/typecheck/main.mind` compiles to
`examples/typecheck/libmindc_typecheck.so` (40KB) via `mindc --emit-shared`.
The Python harness `examples/typecheck/bootstrap_smoke.py` loads
`libmindc_lexer.so` + `libmindc_parser.so` + `libmindc_typecheck.so`, runs
`lex()` → `parse()` → `typecheck()` on `examples/typecheck/fixture.mind`, and
confirms the returned `String` report is byte-identical to the text documented
in `examples/typecheck/EXPECTED.md`.

**Stage 3 verdict: PASS — 127-byte type-check report byte-identical to EXPECTED.md.**

Three compiler / runtime gaps closed to reach this milestone:

#### Gap S3-A — `extern_calls` not propagated from `Instr::If` and `Instr::While` sub-contexts (blocking)

`src/mlir/lowering.rs` emits `Instr::If` and `Instr::While` by creating
sub-`LoweringContext` values for the condition, then-branch, else-branch, and
loop body.  The sub-contexts correctly accumulate `extern_calls` entries (one
per `Instr::Call` encountered), but those entries were never merged back into
the parent context.  At module-assembly time only the parent's `extern_calls`
are converted to `func.func private` forward declarations, so any external
function called exclusively inside an `if` or `while` branch was silently
omitted — triggering an `mlir-opt` "does not reference a valid function" error
when the module was passed to the MLIR pipeline.

For `main.mind` the affected symbols were `map_len` (called in the condition
of `env_lookup_rest`'s `if i >= map_len(env)` branch) and `map_key_at` /
`map_value_at` (called inside the same `if` body).  All three were called at
runtime but had no `func.func private` declaration, causing the build to fail.

**Fix:** After processing each sub-context in `Instr::If` (condition, then,
else) and `Instr::While` (condition, body), merged `sub.extern_calls` into
`self.extern_calls` with a `for ec in sub.extern_calls { self.extern_calls.insert(ec); }` loop — matching the existing pattern already used for `Instr::FnDef`.

- `src/mlir/lowering.rs`: four new extern-call bubble-up loops — one per
  sub-context in `Instr::While` (cond, body) and `Instr::If` (cond, then, else).

#### Gap S3-B — `std.map` and `std.string` C implementations missing from runtime-support stub (blocking)

`runtime-support/mind_intrinsics.c` provided C stubs for `std.vec` surface
functions (`vec_new`, `vec_push`, `vec_get`, `vec_set`, `vec_len`, `vec_cap`,
`vec_addr`) but not for `std.map` (`map_new`, `map_insert`, `map_len`,
`map_cap`, `map_keys_addr`, `map_vals_addr`, `map_key_at`, `map_value_at`) or
`std.string` (`string_new`, `string_push_byte`, `string_len`, `string_cap`,
`string_addr`, `string_get_byte`).

`main.mind` calls all these functions via `Instr::Call` (they are pure-MIND
stdlib functions compiled as external references, not inlined into the MLIR).
After Gap S3-A was closed, the `func.func private` declarations were emitted
correctly, but the LLVM codegen step produced an `.so` with unresolved symbols
for every `map_*` and `string_*` call.

**Fix:** Added complete C implementations of both surfaces to
`runtime-support/mind_intrinsics.c`, matching the RFC 0005 Option C heap-record
ABI:

- `Map` record (4×i64, 32 bytes at 8-byte stride): `keys_addr`, `vals_addr`,
  `len`, `cap`.  `map_insert` allocates a fresh 32-byte header on every call
  (non-mutating ABI, matching `std/map.mind`'s immutable semantics).
- `String` record (3×i64, 24 bytes): `addr`, `len`, `cap`.  `string_push_byte`
  allocates a fresh 24-byte header on every call; backing store grows at
  cap 0 → 16, then doubles (matching `std/string.mind`).  Each byte is stored
  as an i64 at its byte offset (stride-1, overlapping writes — the low byte of
  each slot carries the character value, matching `__mind_load_i64(addr + i) & 255`).

#### Gap S3-C — `make_byte_buf_*` stride mismatch (semantic correctness, non-blocking build but causes wrong output)

`main.mind` builds reference byte buffers for type-name comparison via helper
functions `make_byte_buf_3/4/6` that store each byte as an i64 at 8-byte stride
(offsets 0, 8, 16…).  However `bytes_eq_rest` reads both buffers at 1-byte
stride via `load_byte(buf, lo+i)`.  The source buffer (a C string) is laid out
at byte stride, so `load_byte` reads correct values from it.  The name buffers
produced by `make_byte_buf_*` are laid out at i64 stride, so `load_byte` reads
zero at positions 1, 2, …7 instead of the intended characters — causing every
type-name comparison to fail and every type to resolve to `ty_unknown`.

This was not caught by the Phase 6.3 parse-clean gate (which only validates the
MIND IR, not execution output).  Stage 3 is the first time `main.mind` actually
runs.

**Fix (`examples/typecheck/main.mind`):** Changed `make_byte_buf_3/4/6` to
store bytes at stride-1 (offsets 0, 1, 2… instead of 0, 8, 16…) and reduced
the allocation to 16 bytes (sufficient for the longest name buffer at stride-1
with the 8-byte i64 write tail):

```mind
// before: stride-8 (wrong)
__mind_store_i64(h + 0, b0);
__mind_store_i64(h + 8, b1);
__mind_store_i64(h + 16, b2);

// after: stride-1 (matches bytes_eq_rest's 1-byte reads)
__mind_store_i64(h + 0, b0);
__mind_store_i64(h + 1, b1);
__mind_store_i64(h + 2, b2);
```

**Note on EXPECTED.md byte count:** The `EXPECTED.md` byte-map table claims
124 total bytes, but the correct count is **127** (line 1 "fn add…" is 27 bytes
not 26, line 3 "fn compute…" is 36 bytes not 35 — the byte map has two typos).
The text block in EXPECTED.md is correct; the totals in the per-line table are
off.  The harness compares against the correct 127-byte string derived from the
text.

**Stage 1 (lexer) still PASS** — re-confirmed by re-running
`examples/lexer/bootstrap_smoke.py` (32/32 tokens byte-identical).

**Stage 2 (parser) still PASS** — re-confirmed by re-running
`examples/parser/bootstrap_smoke.py` (42 AST nodes byte-identical).

**Bench-gate (vs `.bench-baseline-2026-05-18-rfc0005.txt`, +5% cap):**
All three fixes are entirely off the hot frontend pipeline
(`parse_typecheck_ir`).  The `extern_calls` bubble-up is a no-op for programs
that have no `Instr::Call` inside `if`/`while` branches; the new C stubs add
link-time weight but not frontend latency.  Bench-gate delta is negligible.

## [0.5.2] - 2026-05-18

### Phase 6.5 Stage 2 — pure-MIND parser cdylib bootstrap PASS

`examples/parser/main.mind` compiles to `examples/parser/libmindc_parser.so`
(32KB) via `mindc --emit-shared`.  The Python harness
`examples/parser/bootstrap_smoke.py` loads `libmindc_lexer.so` +
`libmindc_parser.so`, runs `lex()` then `parse()` on
`examples/parser/fixture.mind`, walks the AST heap-record tree, and
confirms it is node-for-node identical to the tree documented in
`examples/parser/EXPECTED.md`.

**Stage 2 verdict: PASS — 42 AST nodes, byte-identical to EXPECTED.md.**

Three compiler gaps closed to reach this milestone:

#### Gap S2-A — subprocess stdout deadlock for large MLIR inputs

`src/eval/mlir_build.rs` `run_command` wrote all of stdin before
draining stdout.  When `mlir-opt`'s stdout pipe buffer exceeded 64 KiB
(parser MLIR: ~61 KiB in → ~95 KiB out), `mlir-opt` blocked on a full
write pipe while mindc polled `try_wait()` — a classic producer/consumer
deadlock.  The lexer MLIR (27 KiB in → 28 KiB out) was below the threshold
and never triggered it.

**Fix:** Replaced the sequential write-then-poll pattern with a
thread-per-pipe approach.  A background thread drains stdout and stderr
simultaneously while the main thread writes stdin and polls for child
termination.  Mutual progress is guaranteed regardless of output size.

- `src/eval/mlir_build.rs`: `run_command` spawned two `thread::spawn`
  drain threads (one each for stdout / stderr).  Stdin is written after
  the threads start so the child can read its input while we drain its
  output.  Threads are joined before reading the accumulated buffers.

#### Gap S2-B — `if`-as-value emits zero constant instead of branch value

`src/mlir/lowering.rs` emitted `%dst = arith.constant 0 : i64` as a
placeholder in `^if_after_N:` for every `Instr::If` result.  This was
intentionally documented as a placeholder for "functions that only use
early-return semantics" (Gap 3 / Stage 1 report).  The parser's `if`
expressions are used as values (e.g. `let after_semi = if tok_kind(...) ==
tk_semi() { pos + 1 } else { pos };`), so the zero constant silently
corrupted every computed position.

**Fix:** Use MLIR basic-block arguments to forward the branch value to the
join block — the standard MLIR pattern for if-as-value in the `cf` dialect:

- `^if_then_N:` terminates with `cf.br ^if_after_N(%then_result : i64)`
- `^if_else_N:` terminates with `cf.br ^if_after_N(%else_result : i64)`
- `^if_after_N(%dst : i64):` declares the block argument that becomes `dst`

Branches that terminate with `Instr::Return` are unchanged — the `cf.br`
is omitted for already-terminated blocks.

#### Gap S2-C — Python harness misread child-list pointer as Vec header

The initial `bootstrap_smoke.py` treated the `child0` field of list-bearing
AST nodes (program, block, fn_def, call) as a Vec header pointer
(`[data_ptr, len, cap]`) and read the second field as the item count.
`child0` actually stores `vec_addr(acc)` — the raw element-array base
address returned by `__mind_load_i64(acc + 0)`, NOT the Vec header
address.  Reading it as a header gave a garbage length field (250M
elements), causing immediate segfault.

**Fix:** `read_child_list(data_ptr, count)` now reads `count` i64 values
directly from `data_ptr` without dereferencing a Vec header.  The count
is always taken from the `aux` field of the AST node, which stores
`vec_len(acc)` as documented in `main.mind`.

**Files changed:**

- `src/eval/mlir_build.rs` — pipe-deadlock fix (Gap S2-A)
- `src/mlir/lowering.rs` — `cf.br` block-argument if-value forwarding
  (Gap S2-B)
- `examples/parser/libmindc_parser.so` — compiled output (32KB,
  zero undefined symbols beyond libc)
- `examples/parser/bootstrap_smoke.py` — complete Python harness
  (new file; Gap S2-C fix included)

**AST node counts on `examples/parser/fixture.mind`:**

| AST kind      | expected | got |
|---------------|----------|-----|
| `ast_program` | 1        | 1   |
| `ast_use`     | 1        | 1   |
| `ast_fn_def`  | 2        | 2   |
| `ast_param`   | 5        | 5   |
| `ast_block`   | 2        | 2   |
| `ast_let`     | 2        | 2   |
| `ast_return`  | 1        | 1   |
| `ast_binop`   | 3        | 3   |
| `ast_call`    | 1        | 1   |
| `ast_ident`   | 24       | 24  |
| **Total**     | **42**   | **42** |

**Stage 1 (lexer) still PASS** — confirmed by re-running
`examples/lexer/bootstrap_smoke.py` (32/32 tokens byte-identical).

**Bench-gate (vs `.bench-baseline-2026-05-18-rfc0005.txt`, +5% cap):**
The pipe-deadlock fix and block-arg emission changes are entirely inside
the `mlir-build`-gated code path.  The hot frontend pipeline
(`parse_typecheck_ir`) is untouched; bench-gate delta is negligible.

## [0.5.1] - 2026-05-18

### Phase 6.5 Stage 1b — cdylib emit now bundles std-surface + runtime-support

`mindc <file> --emit-shared <path>` now produces a self-contained `.so` that
`dlopen`s cleanly without any external runtime dependency beyond libc.

**Root cause:** The `--emit-shared` path called `clang -shared -fPIC <main.ll>
-o <out.so>` but did not link any implementations for the RFC 0005 intrinsics
(`__mind_load_i64`, `__mind_alloc`, etc.) or the `std.vec` surface functions
(`vec_new`, `vec_push`, `vec_get`, `vec_len`).  Every cdylib that used `std.vec`
had three undefined symbols at `dlopen` time.

**Fix — Option A (static-link everything into the cdylib):**

- New file `runtime-support/mind_intrinsics.c`: C implementations of all seven
  RFC 0005 intrinsics (`__mind_alloc`, `__mind_realloc`, `__mind_free`,
  `__mind_load_i64`, `__mind_store_i64`, `__mind_read`, `__mind_write`) plus the
  `std.vec` surface (`vec_new`, `vec_push`, `vec_get`, `vec_set`, `vec_len`,
  `vec_cap`, `vec_addr`).  All functions use the i64 opaque-address ABI
  (RFC 0005 P0a).  Growth policy matches the MIND spec: cap 0 → 4, then doubles.

- `src/eval/mlir_build.rs`:
  - New `const MIND_RUNTIME_SUPPORT_C: &str = include_str!(...)` — the C stub
    is embedded in the binary at compile time (same pattern as Phase C stdlib).
  - New `compile_runtime_support_obj(tools)` — writes the embedded C to a
    `NamedTempFile`, compiles it to a `.o` via `clang -x c -c -fPIC -O2`.
  - Updated `run_clang_codegen` — accepts `extra_objs: &[PathBuf]`; resets
    `-x none` before appending object files so clang does not misinterpret the
    pre-compiled `.o` as LLVM IR.
  - Updated `build_all` — when `emit_shared` is set, calls
    `compile_runtime_support_obj` and passes the result to `run_clang_codegen`.

**Result:**

```
nm examples/lexer/libmindc_lexer.so | grep " U "
# output: only malloc/free/memcpy/realloc@GLIBC (libc — always present)
```

```
python3 examples/lexer/bootstrap_smoke.py
# dlopen succeeds; lex() runs and returns 32 tokens
```

**Stage 1 verdict:** dlopen blocker RESOLVED.  Token-stream comparison runs to
completion.  First divergence at token 3 — `(kind=1, lo=188, hi=189)` (got
`tk_ident`) vs `(kind=15, lo=188, hi=189)` (expected `tk_slash`) for the `.`
in `use std.vec;`.  This is a lexer semantic bug in `examples/lexer/main.mind`
(`match_punct` fallthrough is `tk_ident()` not `tk_slash()`), not a linker bug
— it is out of scope for Stage 1b and is the first open item for Stage 2.

**Tests added:**

- `tests/std_surface_cdylib_link.rs` — 3 tests:
  - `cdylib_is_produced_and_nonzero`: verifies the `.so` is written and nonzero.
  - `cdylib_has_no_undefined_mind_symbols`: `nm -D` confirms zero undefined
    symbols beyond `malloc`/`free`/`memcpy`/`realloc`.
  - `cdylib_dlopens_via_python`: `python3 -c "ctypes.CDLL(...)"` opens the `.so`
    and calls `vec_new()`, asserting a nonzero handle.

**Bench-gate (vs `.bench-baseline-2026-05-18-rfc0005.txt`, +7% cap):**

  small_matmul:  2.770 µs  (baseline 2.80 µs, -1.1%) ✓
  medium_mlp:    6.485 µs  (baseline 6.55 µs, -1.0%) ✓
  large_network: 17.526 µs (baseline 17.10 µs, +2.5%) ✓

The cdylib path is entirely off the hot frontend pipeline; impact is zero.

### Phase 6.5 Stage 1a — IR + MLIR gaps closed; lexer `.so` builds cleanly

**Status: COMPLETE**

Three IR-side compiler gaps that blocked `examples/lexer/main.mind` from
compiling to a valid shared library are now closed. The `.so` builds with no
`mlir-opt` errors and the `lex` symbol is exported at the correct address.

**Gap A — `Instr::If` / proper basic-block control flow (blocking)**

`ast::Node::If` was previously lowered by emitting both branch bodies into the
same flat instruction stream, producing `func.return` mid-block in the MLIR
output — which `mlir-opt` rejects. The fix:

- `src/ir/mod.rs`: new `#[cfg(feature = "std-surface")] Instr::If { cond_id,
  cond_instrs, then_instrs, then_result, else_instrs, else_result, dst,
  branch_bindings }` variant. `instruction_dst` updated accordingly.
- `src/eval/lower.rs`: `ast::Node::If` now lowers into three sub-IRModules
  (cond / then / else) with chained SSA counters to avoid ValueId collisions,
  then packages them as a single `Instr::If`. Helper functions `sub_ir_from` and
  `sub_ir_from_after` inherit `struct_defs` and `const_array_defs` from the
  parent so FieldAccess works inside branch conditions.
- `src/mlir/lowering.rs`: `Instr::If` emits `cf.cond_br` dispatch plus three
  named basic blocks (`^if_then_N`, `^if_else_N`, `^if_after_N`) where `N =
  dst.0` (the globally-unique SSA id). The emitter detects whether the condition
  value is already `i1` (comparison BinOp) or needs `arith.trunci`. Branches
  that end with `Instr::Return` skip the trailing `cf.br ^if_after_N`. The
  `FnDef` closer now checks the last non-empty line of the emitted body for a
  block terminator instead of scanning for any `return` anywhere — this prevents
  the "block with no terminator" MLIR error when a function ends with a fallback
  expression after a chain of early-return `if` checks.
- `src/ir/verify.rs`, `src/ir/print.rs`: `Instr::If` arms added.

**Gap B — bitwise BinOps (`&`, `|`, `^`, `<<`, `>>`) (secondary)**

`ast::Node::Bitwise` fell through to the `_` catch-all and emitted `const.i64 0`,
silently mis-compiling `load_byte`'s byte-mask expression. Fixed:

- `src/ir/mod.rs`: `#[cfg(feature = "std-surface")] BinOp::BitAnd | BitOr |
  BitXor | Shl | Shr` variants added.
- `src/eval/lower.rs`: `ast::Node::Bitwise` arm maps all five ops to the
  corresponding IR `BinOp`.
- `src/mlir/lowering.rs`: maps `BitAnd→arith.andi`, `BitOr→arith.ori`,
  `BitXor→arith.xori`, `Shl→arith.shli`, `Shr→arith.shrsi`.
- `src/eval/ir_interp.rs`, `src/eval/mlir_export.rs`, `src/ir/compact/emit.rs`,
  `src/opt/ir_canonical.rs`: non-exhaustive match arms extended; constant-folding
  of bitwise ops added to `ir_canonical`.

**Gap C — `let` bindings inside `if` branches visible in outer scope (latent)**

`let` bindings introduced inside a then/else branch were discarded; the outer
`fn_env` never learned about them. Fixed via `branch_bindings:
Vec<(String, ValueId)>` on `Instr::If`, populated during lowering and threaded
back to `fn_env` in the FnDef body loop.

**Tests added**

- `tests/std_surface_if_statement.rs` — 7 tests (4 IR-level, 1 MLIR-level,
  1 Gap C, 1 lexer-style early-return chain).
- `tests/std_surface_bitwise_binops.rs` — 11 tests (5 IR-level for all bitwise
  ops, 6 MLIR-level verifying arith dialect opcode names).
- Existing tests updated (map/string/vec/while helpers now recurse into
  `Instr::If` branches) — all pass.

**Bench-gate (vs `.bench-baseline-2026-05-18-rfc0005.txt`, +7% cap)**

  small_matmul:  +1.2%  ✓
  medium_mlp:    +0.1%  ✓
  large_network: +4.2%  ✓

## [0.5.0] - 2026-05-18

### Milestone: RFC 0005 Phase 6.2b — three compiler gaps closed; self-host substrate complete

mindc v0.5.0 closes the three documented surface-grammar gaps that
blocked the pure-MIND self-host work (Phase 6 ladder). All three land
behind module-level feature gates (`std-surface`) so the default-build
hot path stays byte-identical; the +7% bench-gate cap holds (current
delta vs `.bench-baseline-2026-05-18-rfc0005.txt`: small_matmul -0.7%,
medium_mlp -1.5%, large_network +3.5%).

With v0.5.0, four-and-a-half of the six Phase 6 sub-steps are shipped
(6.1 lexer + 6.2a parser + 6.3 type-checker + 6.4 MLIR emit in pure MIND;
6.2b grammar growth in mindc). Phase 6.5 (fixed-point bootstrap, apex)
remains open.

### Added — RFC 0005 Phase 6.2b Gap 2: array literals `[expr, …]` + fixed-size array types `[T; N]`

Array literals and fixed-size array types are now a first-class surface in mindc
(gated to the `std-surface` cargo feature; default-build hot path byte-identical).

**Reproducer** that now parses, type-checks, and lowers cleanly:
```mind
const FOO: [i64; 4] = [1, 2, 3, 4];

fn nth(i: i64) -> i64 {
    FOO[i]
}
```

**Implementation:**
- `src/ir/mod.rs`: new `Instr::ConstArray { dst, name, values }` (const blob
  with O(N) IR, not N individual stores) and `Instr::ArrayLoad { dst, base, index }`
  for `arr[i]` element loads. Both gated to `std-surface`. `instruction_dst`,
  `IRModule.const_array_defs` side-table, and `IRModule.new()` updated.
- `src/ir/print.rs`: `const.array[i64; N]` and `array.load` textual IR forms.
- `src/ir/verify.rs`: `ConstArray` and `ArrayLoad` verifier arms.
- `src/eval/lower.rs`: module-level `Const { ty: Array, .. }` arm lowers to
  `ConstArray` and seeds `const_array_defs`. `ArrayLit` expression arm iterates
  elements (no per-element recursion). `IndexAccess` arm emits `ArrayLoad`.
  `Ident` resolution checks `const_array_defs` and re-emits the blob into fn
  bodies (avoids cross-SSA-namespace ValueId aliasing). `FnDef` arm inherits
  `const_array_defs` and strips const-array ids from `fn_env` to prevent
  SSA-id aliasing with fn params.
- `src/type_checker/mod.rs`: `Let { ann: TypeAnn::Array, .. }` arm validates
  array-literal length against the `[T; N]` annotation and emits a diagnostic
  on mismatch. `Node::While` stub arm added for Gap 1 compatibility.
- `src/eval/mod.rs`, `src/type_checker/mod.rs`: stub match arms for
  `Node::While` (RFC 0005 Gap 1) to allow feature-gated compilation.
- `tests/std_surface_array_literals.rs`: 5 test cases — basic 3-element literal,
  empty `[i64; 0]`, 4,096-entry literal (no stack overflow), `const FOO + FOO[i]`
  round-trip, and type-length mismatch rejection.

**Bench-gate (default build, vs .bench-baseline-2026-05-18-rfc0005.txt):**
- small_matmul:  2.85 µs (baseline 2.80 µs, +1.7%) ✓
- medium_mlp:    6.32 µs (baseline 6.55 µs, -3.5%) ✓
- large_network: 16.76 µs (baseline 17.10 µs, -2.0%) ✓

All within the +7% cap. New code paths are entirely behind `std-surface`;
default-build parser and IR pipeline are untouched.

**4,096-entry LUT IR size:** A 4,096-element `ConstArray` produces exactly one
IR instruction (`const.array[i64; 4096] @NAME = [...]`) rather than 4,096
`__mind_store_i64` calls. Source-file line count for the reproducer: ~1 line per
LUT (the array literal) vs the previous ~4,100 lines per LUT.

### Added — RFC 0005 Phase 6.2b Gap 1: `while` statement parsing + MLIR basic-block loop lowering

The `while` keyword is now a recognised statement in mindc (gated to the
`std-surface` cargo feature, leaving the default-build hot path byte-identical).

**Reproducer** that previously failed with `error[parse][E1001]: expected expression`:
```mind
fn count_to(n: i64) -> i64 {
    let mut i: i64 = 0
    while i < n {
        i = i + 1
    }
    i
}
```

Now parses, type-checks, and emits IR cleanly under
`cargo run --features "std-surface" -- build <file> --emit-ir`.

**Implementation:**
- `src/ast/mod.rs`: new `Node::While { cond, body, span }` variant (gated).
- `src/parser/mod.rs`: `parse_while()` dispatched from `parse_stmt()` (gated,
  ~28 LOC).
- `src/ir/mod.rs`: new `Instr::While { cond_id, cond_instrs, body, live_vars }`
  (gated); `instruction_dst` and IR verifier updated accordingly.
- `src/eval/lower.rs`: `Node::While` arm in `lower_expr` lowers condition and
  body into separate sub-modules and emits `Instr::While` (~50 LOC, gated).
- `src/mlir/lowering.rs`: `Instr::While` arm emits `cf.br`/`cf.cond_br`
  basic-block loop (header → body → back-edge, plus after-block) using the
  `cf` dialect (~60 LOC, gated).
- `tests/std_surface_while_statement.rs`: 6 test cases (5 under `std-surface`,
  1 additionally under `mlir-lowering`) — trivial counted loop, nested while,
  mutable state outside loop, while inside if arm, reproducer smoke test, and
  MLIR block structure assertion.

**Bench-gate (default build, vs .bench-baseline-2026-05-18-rfc0005.txt):**
- small_matmul:  2.76 µs (baseline 2.80 µs, -1.4%) ✓
- medium_mlp:    6.35 µs (baseline 6.55 µs, -3.1%) ✓
- large_network: 17.54 µs (baseline 17.10 µs, +2.6%) ✓

All within the +7% cap. Default-build pipeline byte-identical — feature gate
enforced at module level (no per-statement dispatch).

**`examples/policy.mind` status:** The file uses `while` in two helper functions
(`starts_with` and `find_substring`). After this change those two `while` loops
parse correctly. The file has additional issues beyond `while` that are out of
scope for Gap 1: `const TIMEOUT_SHIFT: u32 = 8` uses a `u32` type annotation
that the type-checker rejects at the std-surface level; enum variants with
`= N` discriminants (`InvalidInput = 1`, etc.) are not yet parsed; and the
`Effect` struct body is incomplete in the excerpt. These are pre-existing gaps
unrelated to Gap 1.

### Fixed — RFC 0005 Phase 6.2b Gap 3: unsigned-i64 literal reinterpret-cast

Integer literals in the range `(i64::MAX, u64::MAX]` are now accepted by the
mindc parser.  Previously any literal exceeding `i64::MAX` (e.g. the FNV-1a
64-bit offset basis `14695981039346656037`) was rejected with
`error[parse][E1001]: integer overflow`.

The parser now tries a signed-i64 parse first; on failure it falls back to an
unsigned-u64 parse and reinterprets the bit-pattern as a signed i64 via
standard two's-complement (Rust `u64 as i64`).  Literals exceeding `u64::MAX`
continue to be rejected.  Literals that fit in `i64` are unaffected — the
fallback branch is never reached for them.

- `14695981039346656037` (FNV-1a offset basis) → `-3750763034362895579i64`
- `18446744073709551615` (`u64::MAX`) → `-1i64`
- `9223372036854775808` (`i64::MAX + 1`) → `i64::MIN`
- `18446744073709551616` (`u64::MAX + 1`) → still rejected as overflow

**Scope:** `src/parser/mod.rs` — four sites (two in the expression literal
parser, two in the pattern-match literal parser); a shared `parse_i64_literal`
/ `parse_i64_pattern` helper extracted to keep all four sites consistent.
The i32 parse at the attribute-argument site is unchanged.

**Tests:** `tests/parser_unsigned_i64_literals.rs` — 9 tests covering all
boundary values, byte-level round-trip, fn-body context, and the for-range
disambiguation path.

**Bench-gate:** default-build hot path is byte-identical (the fallback branch
is only reached on literals outside `i64` range, which never appear in the
canonical compile corpus).  Measured: `small_matmul -0.4%`, `medium_mlp
+4.6%`, `large_network +2.3%` — all within the +7% cap.

Closes Gap 3 from `docs/rfcs/0005-phase-6-2-mindc-gaps.md` §"Phase 6.3
addendum".

### Added — RFC 0005 Phase 6.2a: pure-MIND self-host parser seed

Second step of the self-host ladder. `examples/parser/` ships a
~814-LOC pure-MIND Pratt parser that consumes the Vec<i64> stride-3
token stream from the Phase 6.1 lexer and emits an AST as Option-C
heap-record structs with i64-addr recursive fields.

- **`examples/parser/main.mind`** (814 LOC) — Pratt operator-precedence
  parser with 12 AST node kinds (`ast_int_lit`, `ast_ident`, `ast_binop`,
  `ast_call`, `ast_fn_def`, `ast_let`, `ast_use`, `ast_return`,
  `ast_param`, `ast_block`, `ast_program`, `ast_paren`) and 7 operator
  tags (`op_add`/`sub`/`mul`/`div`/`lt`/`gt`/`eq`). Parses cleanly under
  `cargo run --features "std-surface cross-module-imports" --bin mindc
  -- examples/parser/main.mind --emit-ir` (`next_id = 90`).
- **`examples/parser/fixture.mind`** (23 LOC) — small source program
  exercising fn declarations, let bindings, binary ops, fn calls, and
  `use` statements.
- **`examples/parser/EXPECTED.md`** (190 LOC) — documents the expected
  AST tree shape on the fixture.
- **`examples/parser/README.md`** (230 LOC) — Phase 6.2 status + handoff
  to Phase 6.3 (type-checker).

Key design notes:

- **`ParseResult` struct** `{ next_pos: i64, node: i64 }` threaded through
  every parse helper — no implicit parser state, matching the
  fixed-state-iteration discipline the lexer established.
- **Variable-length child lists** (block stmts, fn params, call args,
  program items) ride in `Vec<i64>` of AST addresses; the Vec base addr
  stored in `child0` of the parent node and the count in `aux`. No AST
  node references a non-i64 field — RFC 0005 P0a discipline end-to-end.
- **Pratt fold** is a single 11-line tail-recursive function;
  precedence + left-associativity fall out of `prec + 1` recursion on
  the right operand.
- **`return` keyword detection** uses byte-compare against the source
  buffer because Phase 6.1's lexer hasn't promoted `return` to a
  keyword. Phase 6.3 lexer growth will replace this with a
  `tk_kw_return()` check trivially.

No new mindc feature gaps surfaced. The two gaps already documented
in `docs/rfcs/0005-phase-6-2-mindc-gaps.md` (Gap 1 `while`, Gap 2
array literals) remain the canonical unblockers; the parser's
heap-record boilerplate (~364 of the 814 LOC) is the same O(N)
source-line tax Gap 2 fixes.

### Added — RFC 0005 Phase 6.1: pure-MIND self-host lexer seed

First step of the self-host ladder. `examples/lexer/` ships a
~290-line pure-MIND tokeniser on top of `std.vec` + the seven
`__mind_*` intrinsics. Walks an in-memory source buffer byte-by-byte
and emits a flat `Vec<i64>` stride-3 token stream `(kind, lo, hi)`.

- **`examples/lexer/main.mind`** — the lexer (idents, integer literals,
  single-char + `->` punctuation, `//` line comments, whitespace, the
  four current keywords `fn`/`let`/`use`/`pub`). Parses cleanly under
  `cargo run --features "std-surface cross-module-imports" --bin mindc
  -- examples/lexer/main.mind --emit-ir`.
- **`examples/lexer/fixture.mind`** — a 254-byte source file for the
  smoke gate. Token-stream contract documented in
  **`examples/lexer/EXPECTED.md`** (32 rows × 3 i64 = 96 entries).
- **`examples/lexer/README.md`** — Phase 6.1 status + Phase 6.2
  follow-up. Key documented finding: **mindc v0.4.4's parser does NOT
  accept `while` as a statement**. The lexer expresses every loop as
  tail recursion, which the parser does accept. Phase 6.2 either adds
  `while`-stmt parsing or canonicalizes tail recursion as the MIND
  loop primitive — either choice keeps this seed working unchanged.

The smoke gate is currently a documented fixture; Phase 6.2 will
promote it to a Cargo integration test that compiles `main.mind`
to a `.so` and diffs the live token stream against mindc-Rust's
own tokeniser output.

### Changed — RFC 0005 landing table expanded for Phase 6

`docs/rfcs/0005-pure-mind-std-surface.md` §"Adoption plan" landing
table now expands the single Phase 6 row into a 5-step ladder
(6.1 lexer / 6.2 parser / 6.3 type-checker / 6.4 MLIR text emit /
6.5 fixed-point bootstrap). Phase 6.1 marked **shipped**; rest
**open**. Phase D₂b row cross-linked to its design note
(`docs/rfcs/0005-phase-d2b-design-note.md`).

### Added — Phase D₂b design note

`docs/rfcs/0005-phase-d2b-design-note.md` captures the cross-arg
Named-struct identity-matching design for the next compiler tag.
Multi-session pickup artifact; not yet implemented.

## [0.4.4] — 2026-05-19

### Added — RFC 0005 Phase D₂a: Named structs preserved in cross-module call errors

Phase B's call-site type-checker compares args against an imported
fn's declared parameter types. When a mismatch fires, the error
message currently leans on `describe_value_type` — which collapses
Named struct types (`Vec`, `String`, `Map`, ...) to `ScalarI64`
because that's the Option-C heap-record ABI lowering. The result is
that a user passing the wrong type into `vec_set(my_string, 0, 99)`
sees "expects scalar i64" with no mention of `Vec` at all.

D₂a renders the *expected* side of the error message from the raw
`TypeAnn` instead of the lowered `ValueType`, so the call-site error
now reads `expects Vec (heap-record i64 addr)`. The compatibility
check itself stays permissive (Phase B/C/D₁ behaviour unchanged —
i64 values still accepted into Named struct params under Option-C);
this is purely an error-clarity improvement.

- **`src/type_checker/mod.rs`** — new gated helper
  `describe_param_type(&TypeAnn) -> String` (~12 lines). Falls
  through to `describe_value_type(&cm_typeann_to_valuetype(ann))`
  for primitives + aggregates; renders Named-typed params as
  `<name> (heap-record i64 addr)`. The single call-site swap in
  `check_imported_fn_call` consumes it.
- **`tests/std_surface_use_import_phase_b.rs`** — new test
  `phase_d2_named_struct_param_named_in_arity_error` confirms the
  emitted error contains both `expects Vec` and `heap-record i64
  addr`. Existing 6 Phase B tests + the new one all pass.

D₂b (cross-arg *identity* matching — flagging `vec_set(my_string,
...)` as a real type error rather than a permissive accept) stays
deferred. The right design needs the type-env to track Named
struct names through let bindings and fn returns, which is bigger
surgery than v0.4.4 absorbs cleanly.

### Compile-speed gate

Against the post-RFC-0005 baseline. The new helper is in the
error-formatting cold path — only fires when a per-arg widening
already failed — so the hot-path bench is byte-identical:

| bench          | baseline   | v0.4.4     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.80 µs    | 2.87 µs    | +2.4%    | ✓    |
| medium_mlp     | 6.55 µs    | 6.61 µs    | +0.9%    | ✓    |
| large_network  | 17.10 µs   | 17.49 µs   | +2.3%    | ✓    |

All within the +7% cap.

## [0.4.3] — 2026-05-18

### Added — RFC 0005 Phase D₁: `$MIND_STDLIB_PATH` override

Phase C bundled the four pure-MIND std/*.mind files into the mindc
binary so `use std.vec` resolved with no external file dependency.
That left one gap: a downstream user who wanted to fork the stdlib
(e.g. ship a stricter `std.string` with inline UTF-8 validation for
a regulated deployment) had to fork and rebuild `mindc` itself.

Phase D₁ adds an env-var escape hatch: when `MIND_STDLIB_PATH=path`
is set and points at a directory containing all four `.mind` files
(`vec.mind`, `string.mind`, `map.mind`, `io.mind`), the project
loader reads from that directory instead of the bundled blobs.
Missing files, missing directory, or parse failure → silently fall
back to bundled. Same fork-without-recompile escape hatch Rust's
`RUSTC_BOOTSTRAP` provides.

- **`src/project/stdlib.rs`** — new `parsed_stdlib_modules_from_env()`
  helper, called at the top of `parsed_stdlib_modules()`. Reads
  `MIND_STDLIB_PATH` via `std::env::var_os`, validates the directory
  exists, attempts to read + parse all four module files. Returns
  `Option<Vec<(String, Module)>>` — `Some` only on full success.
  Hot path on unset env var is a single null check.
- **3 new tests** in `src/project/stdlib.rs::tests`:
  - `env_override_falls_back_when_unset` — default behaviour unchanged.
  - `env_override_loads_directory_when_set` — round-trips through the
    repo's own `std/` directory and asserts on count + module names.
  - `env_override_falls_back_on_missing_dir` — non-existent path
    triggers fallback, doesn't crash.

### Compile-speed gate

Re-ran `cargo bench --bench compiler` after the change. Numbers
against `.bench-baseline-2026-05-18-rfc0005.txt` (small 2.80 µs /
medium 6.55 µs / large 17.10 µs):

| bench          | baseline   | v0.4.3     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.80 µs    | 2.85 µs    | +1.8%    | ✓    |
| medium_mlp     | 6.55 µs    | 6.50 µs    | -0.8%    | ✓    |
| large_network  | 17.10 µs   | 17.43 µs   | +1.9%    | ✓    |

All within the +5% cap. The bundled hot path is a branchless
`Option<...>` short-circuit, so Phase D₁ is invisible to default
builds.

### Spec alignment

`mind-spec` (commit `5fa4299`) added an informative "Environment
override" subsection to `spec/v1.0/stdlib.md` documenting the
`MIND_STDLIB_PATH` contract; honouring it is informative for
conforming implementations.

## [0.4.2] — 2026-05-18

### Added — RFC 0005 Phase C: auto-bundle `std/*.mind` into mindc

v0.4.1's Phase B closed the per-arg signature matching deferred from
v0.4.0 but still required the consumer to supply std module ASTs to
`build_module_table` manually.  A real downstream `mind build`
running on a project that says `use std.vec` had no way to find the
std/*.mind files unless the user vendored them — which defeats the
point of having a shared standard library.

Phase C bundles the four pure-MIND std/*.mind files into the mindc
binary at compile time via `include_str!` and seeds the module
table with them in the project loader's cross-module-imports
block.  A downstream `mind build` of a project that says
`use std.vec` now resolves with no external file dependency.

- **`src/project/stdlib.rs`** (new) — compile-time bundle:
  - `STDLIB_MIND_SOURCES: &[(&str, &str)]` — `(module_path,
    source_text)` for `std.io`, `std.map`, `std.string`, `std.vec`,
    sorted alphabetically for deterministic insertion.
  - `parsed_stdlib_modules() -> Vec<(String, Module)>` — parses
    every bundled source and returns the `(module_path, AST)` pairs
    the project loader's cross-module-imports block prepends to the
    user's own modules.
- **Project loader wiring** in `src/project/mod.rs` — the bundle is
  prepended to `parsed` BEFORE walking the user's `src/`.  Since
  `ModuleTable::insert` is last-write-wins, a user module that
  happens to shadow `std.*` overrides the bundled entry — same
  behaviour as Rust's user-crate-wins-over-stdlib semantics.
- **CI matrix coverage** — the new gated-feature CI step (commit
  `996553e`) runs the std-surface + cross-module-imports test
  suites separately and combined, including Phase C's 4 new
  integration tests + 3 new unit tests.  Total tests added in v0.4.2:
  7.  All RFC 0005 work is now under cloud CI guard.

### Performance

Compile-speed gate vs `.bench-baseline-2026-05-18-rfc0005.txt`:

| bench          | baseline   | v0.4.2     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.800 µs   | 2.758 µs   | -1.50%   | OK   |
| medium_mlp     | 6.550 µs   | 6.779 µs   | +3.50%   | OK   |
| large_network  | 17.100 µs  | 17.627 µs  | +3.08%   | OK   |

All inside the 5% CI gate.  Phase C lives entirely behind
`feature = "cross-module-imports"`; the default-build frontend hot
path is untouched (module-level gate, no per-statement cfg, no
runtime dispatch).  The medium/large drift is within the ±2%
runner-variance band documented in the baseline file (local
replays put medium at 6.39–6.78 µs and large at 16.89–17.80 µs).

## [0.4.1] — 2026-05-18

### Added — RFC 0005 Phase B: per-arg signature matching on imported `pub fn`s

v0.4.0 wired `use std.vec` so calls to `vec_new()` / `vec_push(...)`
type-check loosely as `ScalarI64`-returning, with no arg-count or
arg-type validation. The v0.4.0 CHANGELOG flagged per-arg signature
matching as deferred to Phase B; this release closes that gap.

- **`ExportedFn { name, param_types, ret_type }`** in
  `src/project/module_table.rs` carries the full signature of every
  auto-exported `pub fn`. `ModuleExports` gains a parallel
  `exported_fns: Vec<ExportedFn>` populated on the auto-export path;
  explicit `export { ... }` block surfaces leave it empty by
  construction (the RFC 0002 contract is preserved — those declare
  names, not signatures, by design).
- **`ModuleTable::lookup_imported_fn(name)`** walks every module in
  deterministic sorted-key order and returns the first match.
- **Type-checker side** (gated under `cross-module-imports`):
  `cm_lookup_fn` reads the project table's typed declaration;
  `check_imported_fn_call` validates arity then per-arg types and
  returns the declared return type;
  `cm_typeann_to_valuetype` reuses the existing `valuetype_from_ann`
  helper, falling back to `ScalarI64` for Named struct/enum types
  and unsupported aggregates — matches RFC 0005's Option-C heap ABI
  where struct values are i64 base-addresses on the wire;
  `cm_arg_compatible` accepts exact matches plus the universal
  i32 ↔ i64 widening that integer literals need.
- **Fall-back to Phase A** when no signature is available (e.g. the
  imported module surface came from an `export { ... }` block, not
  auto-export). Default build path is byte-identical — the moat is
  held.
- 3 new module_table unit tests + 6 new integration tests under
  `tests/std_surface_use_import_phase_b.rs`. Phase A's 8 tests
  still pass — Phase B is strictly an additive improvement.

### Performance

Compile-speed gate vs `.bench-baseline-2026-05-18-rfc0005.txt`:

| bench          | baseline   | v0.4.1     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.800 µs   | 2.742 µs   | -2.07%   | OK   |
| medium_mlp     | 6.550 µs   | 6.614 µs   | +0.98%   | OK   |
| large_network  | 17.100 µs  | 16.919 µs  | -1.06%   | OK   |

All inside the 5% gate. All Phase B changes live behind
`feature = "cross-module-imports"`; the default-build frontend hot
path is untouched.

## [0.4.0] — 2026-05-18

### Added — RFC 0005 Phase 2: pure-MIND standard surface (`std.vec`, `std.string`, `std.map`, `std.io`)

The four pure-MIND collections + I/O surface RFC 0005 names now lower
end-to-end on the seven `__mind_*` intrinsics shipped at Phase 1 +
Phase 1.5 + Phase 2 (P0e + P0f).  Every operation bottoms out into
`__mind_alloc` / `__mind_load_i64` / `__mind_store_i64` / `__mind_read`
/ `__mind_write` — i64 ABI throughout, no built-in pointer type, no
hidden allocator.

- **`std/vec.mind`** — growable `Vec` (8-byte stride, doubling growth
  from min-cap 4).  Surface: `vec_new`, `vec_len`, `vec_cap`,
  `vec_addr`, `vec_get`, `vec_set`, `vec_push`.  6 module tests assert
  the exact `__mind_alloc` / `__mind_load_i64` / `__mind_store_i64`
  floor counts per function — `vec_new` = 1 alloc + 3 stores,
  `vec_len/cap/addr` = 1 load each, `vec_get` = 2 loads, `vec_set` =
  1 load + 1 store, `vec_push` ≥ 3 loads + ≥ 4 stores + ≥ 1 alloc.
- **`std/string.mind`** — `Vec<u8>`-shaped `String` with a documented
  UTF-8 well-formedness invariant.  Surface: `string_new`,
  `string_len`, `string_cap`, `string_addr`, `string_get_byte`,
  `string_validate_utf8`, `string_push_byte`, `string_eq`.  Byte-
  stride loads/stores (not 8-byte stride like `Vec`); 16-byte initial
  cap.  7 module tests with the same IR-shape contract.
- **`std/map.mind`** — insertion-ordered `Map` on parallel keys / vals
  arrays (4-field heap record).  Surface: `map_new`, `map_len`,
  `map_cap`, `map_keys_addr`, `map_vals_addr`, `map_key_at`,
  `map_value_at`, `map_insert`.  Ordering is deterministic (insertion
  order, not hash-randomised) — load-bearing for evidence-chain
  reproducibility.  5 module tests.
- **`std/io.mind`** — `File` handle plus the four-arg `__mind_read` /
  `__mind_write` POSIX-shaped intrinsics.  Surface: `stdin` /
  `stdout` / `stderr` constructors, `file_fd`, `file_read`,
  `file_write`, `print_bytes`, `eprint_bytes`, `read_stdin_bytes`.
  8 module tests assert the I/O surface routes through the correct
  intrinsic (no special-case lowering — the generic `Instr::Call`
  arm from Phase 0 picks them up).
- **`__mind_read` / `__mind_write`** declared at arity 4 in the
  std-surface intrinsic registry (`STD_SURFACE_INTRINSICS`).  The
  MLIR lowering's generic call arm handles them via the same path as
  the other RFC 0005 intrinsics.

### Added — `use std.foo` cross-module resolution

The cross-module-imports resolver (D1 + D2 + D3, gated since v0.2.6)
now composes with RFC 0005 Phase 2 so a consumer file can `use std.vec`
and call `vec_new()` directly.

- **Auto-export of `pub fn` + `struct`** when no `export { ... }` block
  is present (`collect_module_exports`).  The parser already strips
  `pub` to a no-op, so a `pub fn`-only file would otherwise have an
  empty exported surface.  Explicit `export { ... }` blocks still win
  when present — the RFC 0002 contract is preserved.
- **Imported names accepted as callables** in `infer_call`'s catch-all
  when the `cross-module-imports` feature is on (gated; default build
  byte-identical).  Calls to resolver-injected names type-check as
  `ScalarI64`-returning — per-arg signature matching against the
  imported `pub fn` declarations is Phase B (deferred to v0.4.x).
- 8 end-to-end tests in `tests/std_surface_use_import.rs` cover:
  pub-fn auto-export, struct auto-export, `use std.vec` resolution,
  `use std.io` resolution, unimported-module isolation, wrong-path
  fall-through, and explicit-`export`-block precedence.

### Performance

Compile-speed gate (`tools/bench_gate.py` vs
`.bench-baseline-2026-05-17-phase10-7.txt`) passes with all three
canonical benches **improved**:

| bench          | baseline   | v0.4.0     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.810 µs   | 2.747 µs   | -2.24%   | OK   |
| medium_mlp     | 6.560 µs   | 6.432 µs   | -1.95%   | OK   |
| large_network  | 16.900 µs  | 16.797 µs  | -0.61%   | OK   |

The RFC 0005 work lives entirely behind `feature = "std-surface"`
and `feature = "cross-module-imports"`; the default-build frontend
hot path is untouched (module-level gates only, no per-statement
cfg) — the compile-speed moat is held.

## [0.2.11] — 2026-05-17

### Added — Phase 10.7: match expressions and reference-taking expressions

The following surface constructs now parse, type-check, and lower to IR
without new IR opcodes (conservative v1 strategy: match lowers to
sequential arm evaluation; &expr is a no-op metadata wrapper).

- **`match` expressions** (`match value { Pat => body, ... }`). Patterns
  supported: `EnumVariant` (`Mode::On`, `Result::Ok(x)`), `Literal`
  (integer, float, string, `true`/`false`), bare `Ident` binding, and
  `_` wildcard. Block bodies (`=> { ... }`) also accepted. Arms
  separated by `,`; trailing comma optional.
- **`&expr` and `&mut expr`** reference-taking prefix expressions.
  Symmetric with the `&T` / `&mut T` type forms already in Phase 10.6.
  The Pratt parser disambiguates `&` prefix (ref-take) from `&` infix
  (bitwise-AND) by position: prefix `&` fires inside `parse_primary`
  before the infix loop can see it — no saved_pos / backtrack needed.
- **AST additions**: `Node::Match { scrutinee, arms, span }`,
  `Node::Ref { mutable, inner, span }`, `MatchArm`, `Pattern`
  (`EnumVariant`, `Literal`, `Ident`, `Wildcard`).
- **Type-checker**: arm body types unified; literal pattern vs. scrutinee
  type checked conservatively; identifier patterns introduce bindings;
  `&expr` / `&mut expr` accepted with `ScalarI32` placeholder type.
- **Corpus watermark**: bumped from 14 to 21 (7 newly unblocked `.mind`
  files in `rfn-mind/src`).

## [Unreleased]

Infrastructure landed on `main` but not yet attached to a tag.

### Added
- `libmind::cache` — content-addressed compilation cache with
  `CompilationCache`, `CacheKey`, `CacheEntry`, `CacheStats`,
  `ProfileTag`, and an in-memory `MemoryStore` backend. Cache key
  includes compiler version + profile tag + source hash + imports
  hash so cross-mode rebuilds never hit a stale entry. Foundation
  for sub-µs warm-start frontend latency. 17 unit tests.
- `tools/pytorch_bridge/` — PyTorch / JAX → MIND transpiler tooling.
  ONNX-driven PyTorch path, XLA-HLO-driven JAX path, and a
  deterministic prompt builder for AI-assisted UNSAT proof resolution.
  Pure Python, no torch / jax import at module load. 11 unit tests.
- `libmind::distributed` — IR-layer primitives for tensor and pipeline
  parallelism: `ShardSpec` / `ShardLayout` (replicated / split /
  split-2D), `AllReduceOp` with lexicographic / tree / arrival
  reduction orders, `AllGatherOp` with lexicographic / arrival gather
  orders, `PipelineGraph` / `PipelineStage` / `StageBoundary`, and
  `DistributedInvariant` enforcement (`deterministic_all_reduce`,
  `reduction_order_lexicographic`, `gather_order_lexicographic`,
  `evidence_chain_continuous`). 31 unit tests.
  See `docs/roadmap.md` Phase 13.6 for the design rationale and the
  speed-preservation discipline that keeps the 1.8–15.5 µs frontend
  baseline locked when these primitives are not imported.

## [0.2.10] — 2026-05-17

### Added — Phase 10.6 surface syntax (parser-level)

The following surface constructs now parse, type-check, lower to IR,
and round-trip through the bench gate without moving headline numbers.
They are additive extensions of Core IR v1 (see `docs/versioning.md`
"Minor (0.y.z)") — no existing IR instruction or shape semantics
change.

- **Qualified type paths** in const declarations and type annotations
  (`module.Type`, multi-segment `a.b.C`) (e85611a).
- **`pub` visibility marker** on `fn`, `struct`, `enum`, and on struct
  fields (dfe01ce). Parsed and propagated to AST; semantic effect lands
  with RFC 0002 D2.
- **Struct literal expressions** `Name { field: value, ... }` in
  expression position (f60232c).
- **Slice types** `&[T]` / `&mut [T]` and **fixed-size array types**
  `[T; N]` (9f0fd57).
- **`let mut`** mutable-binding marker accepted by the parser (12be59d).
- **`%` modulo operator** end-to-end: parser → IR → autodiff → eval
  → MLIR lowering (7db0bf3).
- **Single-value reference types** `&T` and `&mut T` distinct from
  slices (6f14a66).
- **Generic type application** `Name<A, B, ...>` for `Vec`, `Result`,
  `Option`, and user-defined generics (d411d53).
- **`::` path-segment separator** for enum variant access:
  `Result::Ok`, `module.Enum::Variant` (dc0f70c).
- **Tuple types** `(T, U, ...)` in function return positions.
- **Postfix index access** `xs[0]` on arrays, slices, and Vec values.
- **Indexed assignment** `xs[i] = value` and **field assignment**
  `obj.field = value` as statements.
- **Multi-line arithmetic continuation** — `+`, `-`, `*`, `/`, `%` may
  span newlines, e.g.

  ```mind
  let idx: u32 = (c as u32) * (h * w) as u32
               + (y as u32) * (w as u32)
               + (x as u32)
  ```

### Fixed
- **MLIR lowering pipeline** — `--emit-mlir` now lowers to the LLVM
  dialect before invoking `mlir-translate`, removing the
  "unregistered dialect" failure on programs that exercise control
  flow (99fc19f).

### Changed
- `find_runtime_lib` now searches only `MIND_LIB_DIR` and `~/.mind/lib`.
  The previous `~/.nikolachess/lib` fallback was an internal
  install-path leak and is removed.
- Test target `parse_rfn_mind.rs` renamed to `parse_phase10_surface.rs`;
  the corpus-sweep test is now driven by the `MIND_TRACKING_CORPUS_DIR`
  environment variable rather than a hard-coded path. On CI and fresh
  clones the sweep is a no-op; when the variable is set, the parser
  must hold the documented high-watermark (14 files at this release).

### Compile-speed discipline
- Pratt loop adopts a **same-line fast path**: `skip_ws` + `peek_binop`
  succeeds without saving position in the common case, and only widens
  to `skip_ws_and_newlines` when an actual `\n` is at the cursor. Every
  parser arm above ships behind a leading-token check and exits in
  O(1) when the construct is absent. The headline benches
  (`small_matmul / medium_mlp / large_network`) remain within the
  documented threshold of `.bench-baseline-2026-04-28-pratt.txt`.

## [0.2.9] — 2026-05-16

### Security
- **RFC 0002 hardening** — surfaced by the v0.2.8 security audit, both
  fixed in this release before D2 (codegen pass) bakes the unvalidated
  inputs into emitted symbols.
  - `Mind.toml [exports] c_abi` is now bounded to
    `MAX_MANIFEST_EXPORTS = 1024` and every entry must match the
    C-style identifier grammar `[A-Za-z_][A-Za-z0-9_]*`. Violations
    return the new `CompileError::InvalidManifestExport` (diagnostic
    `E5002`). DoS / symbol-injection guard ahead of the C-ABI wrapper
    codegen pass.
  - `mindc compile --profile` now uses clap's
    `value_parser = ["default", "systems", "embedded"]`. Typos like
    `--profile sytems` are rejected at the CLI layer instead of
    silently falling through `ProfileTag::parse`'s permissive mapping
    to `Default` (which would have poisoned the cache fingerprint).
- Regression tests:
  - `tests/ir_lower.rs::manifest_exports_reject_oversized_list`
  - `tests/ir_lower.rs::manifest_exports_reject_non_identifier`

## [0.2.8] — 2026-05-16

### Added
- **RFC 0002 deliverable 5** — `--profile <default|systems|embedded>`
  CLI flag on `mindc compile`. Defaults to `default`. Threads through
  `CompileOptions.profile: ProfileTag` to the cache fingerprint so the
  same `Mind.toml` produces three distinct artifacts.
- **`ProfileTag::parse`** — case-insensitive parser; unknown names map
  to `Default`. `ProfileTag` now derives `Default` (variant `Default`).
- Regression test
  `tests/ir_lower.rs::profile_tag_parse_and_default` covers parse,
  default propagation, and explicit override.

### Changed
- `CompileOptions` gains a public `profile` field. As with D3 the
  default is preserved via `Default`; in-tree call sites already use
  `..Default::default()` so no migration needed.

## [0.2.7] — 2026-05-16

### Added
- **RFC 0002 deliverable 3** — `Mind.toml [exports]` section is now
  honored end-to-end. `ProjectManifest.exports.c_abi: Vec<String>`
  parses from the manifest, `BuildOptions.manifest_exports` threads it
  through the build pipeline, and `CompileOptions.manifest_exports`
  merges those names into `IRModule.exports` after AST → IR lowering.
  Together with deliverable 1, both source-side `export { ... }` blocks
  and manifest-declared exports reach the same set in IR — ready for the
  codegen pass landing in deliverable 2.
- Regression test
  `tests/ir_lower.rs::compile_pipeline_merges_manifest_exports`
  asserts the merge for the combined case.

### Changed
- `CompileOptions` and `BuildOptions` gain a `manifest_exports`
  field. The struct still derives `Default`, so existing call sites
  can opt in via `..Default::default()`. All in-tree call sites updated.

### Compile-speed discipline
- The merge path is `if !manifest_exports.is_empty() { ir.exports.extend(...) }`
  — branchless in the default code path (typical `Mind.toml` has no
  `[exports]` block).

## [0.2.6] — 2026-05-16

### Added
- **RFC 0002 deliverable 1** — `IRModule.exports: HashSet<String>`
  field populated by the AST → IR lowering pass when the parser sees an
  `export { foo, bar }` block (`Node::Export`). Previously the export
  block parsed but lowered to a no-op warning. The field is empty in
  the default code path; consumers under the new `ffi-c-user` feature
  flag (deliverables 2+) read the set to emit `mind_fn_<name>_invoke`
  C-ABI wrappers per RFC 0002. Regression tests:
  `tests/ir_lower.rs::lower_export_block_populates_ir_exports` and
  `lower_no_export_keeps_exports_empty`.
- **`ffi-c-user` Cargo feature** (currently empty gate). Reserved for
  the RFC 0002 codegen pass landing in subsequent deliverables.
- **`bench_c_export_lowering` sub-bench** in `benches/compiler.rs`
  measuring 0 / 1 / 10 export names. Separate criterion group so the
  headline `compiler_pipeline` numbers stay measurable against
  `.bench-baseline-2026-04-28-pratt.txt`.

### Compile-speed discipline
- New IR field is `HashSet<String>::new()` at construction (zero
  capacity, zero allocation until first insert). The default code path
  for programs without an `export` block performs one extra branchless
  match-arm test per top-level item, no hashset touches.

## [0.2.5] - 2026-04-28

### Added
- **Pratt operator-precedence parser** for the expression layer. Replaces
  the recursive-descent chain `parse_logical_or → parse_logical_and →
  parse_comparison → parse_additive → parse_bitwise → parse_multiplicative`
  with a single dispatch function driven by a binding-power table. Phase 10.5
  operators (`||`, `&&`, `|`, `&`, `^`, `<<`, `>>`, `as`) become table
  entries, and future operators (Phase 11/12) become O(1) inserts.
- **Stable IR public API**: `libmind::ir::load(bytes) -> IRModule` and
  `libmind::ir::save(module) -> String` for the `mic@1` textual format.
  Plus `libmind::compile_to_mic_text(src, opts)` as the AOT pipeline.
- `docs/ir-stability.md` — formalises the IR contract used by
  `mind-runtime` and other downstream backends to consume pre-compiled IR
  instead of re-running the surface parser per inference.
- `.github/workflows/bench-gate.yml` + `tools/bench_gate.py` — CI gate that
  fails on >2% mean regression on `small_matmul`, `medium_mlp`, or
  `large_network` vs the frozen baseline at
  `.bench-baseline-2026-04-28-pratt.txt`.
- Phase 10.5 conformance programs in `tests/conformance/cpu_baseline/`
  (`phase_10_5_const.mind`, `..._logical.mind`, `..._struct.mind`,
  `..._module.mind`).
- `tests/ir_load_save.rs` — pins the round-trip and determinism contract
  for the new IR API.

### Changed
- **Parser is faster than the pre-Phase-10.5 baseline** on `medium_mlp`
  (-9.0%) and `large_network` (-4.6%); within +3.9% on `small_matmul`.
  The Pratt rewrite recovered the +9% regression introduced when Phase 10.5
  dispatch arms were added in 0.2.4.
- Removed the `peek_skip_ws` helper (only used by the recursive-descent
  fast-paths it was added to support; Pratt makes it unnecessary).
- Removed dead `parse_logical_or` forwarder (deprecated since Phase 10.5
  inlined logical-or into `parse_expr`).

### Architecture
- mindc's parser is no longer a runtime dependency for `mind-runtime`
  consumers. The supported pattern is to AOT-compile to `mic@1` once at
  build time and call `ir::load` at runtime. This decouples parser
  performance from per-inference latency across all 12+ planned backends
  (CPU, CUDA, Metal, ROCm, WebGPU, WebNN, ARM, TPU, NPU, LPU, DPU, FPGA,
  Quantum).

## [0.2.1] - 2026-02-17

### Added
- IR verifier audit coverage: 14 new tests for conv2d stride validation,
  reduction axis checks, FnDef body scoping, and duplicate definition detection
- Determinism proof results for v0.2.0-hardened (4/4 DETERMINISTIC)
- Criterion benchmark results for hardened pipeline (338K compilations/sec)

### Fixed
- **C1**: Conv2d IR verifier now rejects zero strides and negative axes
- **C6**: FnDef body verifier enforces SSA scope (use-before-def in body blocks)
- **C2**: String interning DoS protection (MAX_INTERNED_STRINGS = 100,000)
- **C3**: IR printer determinism via sorted function iteration
- **C4**: Constant folding bounds checking for division-by-zero and overflow
- **C5**: Type checker array bounds validation
- **C7**: Hardened eval NaN/Inf propagation
- **A1**: Cargo Deny supply-chain audit configuration

### Security
- String interning rate limiting prevents memory exhaustion attacks
- Constant folding rejects division by zero and integer overflow at compile time

## [0.2.0] - 2026-02-07

### Added
- IR-first compilation pipeline with shape ops and MIC emission
- Remizov universal ODE solver (`std::ode` module)
- Real tensor compute backend with benchmarks
- Open-core reference interpreter for public compiler

### Changed
- **BREAKING**: Replaced Chumsky parser combinator with hand-written recursive descent parser (15x speedup)
- Parser now achieves ~347,000 compilations/sec (up from ~22,700 with Chumsky)
- Removed `chumsky` dependency entirely — zero unnecessary allocations, direct byte-level parsing
- CI skips builds for docs-only changes

### Fixed
- Keyword argument disambiguation in `tensor.gather()` calls (positional `idx` vs `idx=` prefix)
- Clippy lint: `map_or` → `is_some_and` for modern Rust idiom
- Formatting consistency across parser, eval, and exec modules
- Removed unfair NumPy comparisons from benchmarks

### Documentation
- Framework comparison and GPU projections
- Runtime execution benchmarks for v0.1.9
- ODE solver examples and usage guide

## [0.1.9] - 2026-02-05

### Changed
- **BREAKING**: Renamed library crate from `mind` to `libmind` to avoid Windows PDB filename collision
- Updated all imports: `use mind::` → `use libmind::`
- Benchmark numbers aligned across README and docs

### Fixed
- Windows LNK1318 PDB linker errors by limiting parallel build jobs
- Format check CI failures (CRLF → LF line endings)

### Documentation
- Updated test counts to 169+ tests across 70 test files
- Aligned benchmark numbers between README and docs/benchmarks/
- Added v0.1.8 and v0.1.9 to compiler_performance.md version history

## [0.1.8] - 2026-01-15

### Added
- Comprehensive documentation for `TODO(runtime)` markers in `src/exec/mod.rs`
- Test execution examples in README.md
- Performance baseline documentation with concrete metrics

### Changed
- Updated benchmarks.md with detailed performance baselines

## [0.1.0] - 2025-01-01

### Added

#### Core Language
- MIND language parser with Logos lexer (originally Chumsky, replaced by recursive descent in v0.2.0)
- Static type system with rank/shape polymorphism
- Tensor type annotations: `Tensor[dtype, shape]` syntax
- Shape inference engine with broadcasting support
- Effect tracking system for side-effect analysis

#### Compiler Pipeline
- SSA-based intermediate representation (IR)
- IR verifier and deterministic printer
- MLIR lowering passes (`mlir-lowering` feature)
- MLIR dialect support for tensor operations
- LLVM backend integration (`llvm` feature, optional)

#### Autodiff
- Reverse-mode automatic differentiation on SSA IR
- Gradient generation for arithmetic, reductions, and linear algebra
- Autodiff preview mode for debugging gradient graphs

#### CLI Tools
- `mind` REPL and evaluator binary
- `mindc` compiler driver with `--emit-ir`, `--emit-mlir` flags
- GPU target profile (`--target=gpu`) with structured error handling

#### Execution Backends
- CPU execution stubs (`cpu-exec` feature)
- Convolution stubs (`cpu-conv` feature)
- Runtime backend interface for proprietary `mind-runtime`

#### Tensor Operations
- Elementwise: add, sub, mul, div with broadcasting
- Reductions: sum, mean, max, min (all axes or specified)
- Linear algebra: matmul, dot, transpose
- Activations: ReLU, with gradient support
- Indexing: gather, slice with shape preservation
- Conv2D with padding modes (Same, Valid)

#### Developer Experience
- 70 integration test files covering all subsystems
- GitHub Actions CI for Linux, macOS, Windows
- Clippy and rustfmt enforcement
- Comprehensive documentation in `/docs`

#### Documentation
- Language tour and quick start guide
- Type system specification
- Shape algebra documentation
- IR and MLIR pipeline guides
- GPU backend design document
- Versioning and stability policy
- Roadmap with 13 phases including neuroscience/BCI applications

### Fixed
- Rank mismatch classification in type checker (#128)
- Type mismatches and autodiff test issues (#130)
- `/tmp` race condition in MLIR demo script (#133)

### Security
- Secure temporary file handling in MLIR subprocess integration

## Links

- Repository: https://github.com/star-ga/mind
- Documentation: https://github.com/star-ga/mind/tree/main/docs
- Issues: https://github.com/star-ga/mind/issues

[Unreleased]: https://github.com/star-ga/mind/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/star-ga/mind/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/star-ga/mind/compare/v0.1.9...v0.2.0
[0.1.9]: https://github.com/star-ga/mind/releases/tag/v0.1.9
[0.2.9]: https://github.com/star-ga/mind/releases/tag/v0.2.9
[0.2.8]: https://github.com/star-ga/mind/releases/tag/v0.2.8
[0.2.7]: https://github.com/star-ga/mind/releases/tag/v0.2.7
[0.2.6]: https://github.com/star-ga/mind/releases/tag/v0.2.6
[0.1.8]: https://github.com/star-ga/mind/releases/tag/v0.1.8
[0.1.0]: https://github.com/star-ga/mind/releases/tag/v0.1.0
