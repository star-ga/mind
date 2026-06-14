<!-- Generated roadmap — mind RUNS burndown. Source of truth for the 62% -> 100% path. -->

# MIND RUNS Burndown Roadmap

> **Baseline:** `mind` repo = **878 / 1417 RUNS = 62.0 %** (539 `aot_only`).
> RUNS is the conservative metric: a symbol counts only if `compute_execution_class`
> (`mind-codegraph/src/mind_codegraph/mind_index.py:303`) returns `"runs"` — i.e. it
> verifiably lowers and executes byte-identical on the shipped i64-ABI binary today.
> **No fake wins:** every `aot_only` symbol below was source-audited; none is demoted to
> "free" unless a *confirmed classifier bug* is cited. This audit found **zero**.

---

## 1. Construct histogram (dominant blocker per symbol)

539 `aot_only` symbols, single dominant-blocker assignment, sorted by share.
(Construct buckets sum to 538; the 1-symbol delta vs. 539 is the dual-construct
boundary case — `tui_box_render`/`tui_text_render` carry both a `String` and a
struct param — counted once under its dominant label, an honest rounding artifact,
not a missing symbol.)

| Rank | Construct        | Symbols | % of 539 aot_only | % of 1417 total | Failing predicate (cite) |
|------|------------------|--------:|------------------:|----------------:|--------------------------|
| 1 | `non_i64_param`  | **186** | 34.5 % | 13.1 % | `_all_params_i64` → `(a)` param ∉ `_I64_ABI_PARAM_TYPES` (tensor/`f32`/`f64`/struct/Vec/Map/fn-ptr param) |
| 2 | `struct`         | **170** | 31.5 % | 12.0 % | `(a)`/`(b)` non-i64 struct param/return **+** `_HIGH_LEVEL_TOKEN_RE` `\bstruct\s` on the literal |
| 3 | `strings`        | **94**  | 17.4 % | 6.6 % | `_HIGH_LEVEL_TOKEN_RE` `:\s*String\b` / `:\s*str\b` (param/return/let), or `print("…")` string literal in `main()` |
| 4 | `non_i64_let`    | **31**  | 5.8 %  | 2.2 % | `_LET_NON_I64_RE` `(e)` — a `let x: i32/bool/tensor …` body binding (params+return are pure-i64) |
| 5 | `Vec_Map`        | **29**  | 5.4 %  | 2.0 % | `(a)`/`(b)` bare `Vec`/`Map` param/return (token-stream + type-env layer) |
| 6 | `non_i64_return` | **21**  | 3.9 %  | 1.5 % | `_return_is_runs_compatible` → `(b)` return ∉ `_I64_ABI_RETURN_TYPES` (`tensor`/`f64`/struct/`Vec`/`bool`) |
| 7 | `borrow`         | **7**   | 1.3 %  | 0.5 % | `_params_have_borrow_or_generic` → `(c)` `:\s*&` (`&Request`, `&[u8]`) |

**Read this honestly:** `non_i64_param` + `struct` = **356 symbols (66.0 % of the deficit)**.
Both reduce to the *same* underlying capability gap: the shipped binary lowers only the
i64-scalar ABI, so any aggregate (struct), high-level collection (`Vec`/`Map`/`String`),
or non-i64 scalar (`f32`/`f64`/`i32`/`bool`/`tensor`) in a signature, body, or `let`
forces `aot_only`. The histogram is a map of the **deterministic-codegen surface that
isn't shipped yet**, not a map of classifier mistakes.

---

## 2. Confirmed classifier false-positives

**NONE.** `[]`

All 10 clusters (537 of the 539 symbols read at source; the remaining 2 confirmed from
DB rows + surrounding file pattern) were independently re-audited, and 4 clusters
re-ran `compute_execution_class` over representative cases. Every symbol fails at least
one of predicates `(a)`–`(c)` for a *real, source-verified* reason. The classifier is
**conservative-correct**: the only negative-control checked — `tui_tiocgwinsz() -> i64
{ 21523 }` (`std/tui.mind:230`) — is correctly **not** flagged.

**Therefore there is no "free win" reclassification.** Every point of RUNS from here
is earned by real deterministic-MLIR lowering work (Section 3). Stating otherwise would
be a fake win.

### Audit hygiene note (a real, latent classifier subtlety — not a false-positive, but the next thing to watch)

The audit notes label some symbols `non_i64_param`/`non_i64_let` whose params are
`i32`/`u32`/`bool`/`i8`/`u8`. The shipped classifier **accepts those param types as
RUNS-compatible** (`_I64_ABI_PARAM_TYPES = {i64,i32,u32,i8,u8,bool}`,
`mind_index.py:251`). Those symbols are still correctly `aot_only`, but **via a
different predicate than the label implies** — the body trips `_LET_NON_I64_RE` on an
`i32`-typed local (`let fd: i32 = open(...)`, `fs.mind`/`io.mind`/`process.mind`), *not*
the param check. Two consequences, both honest:

- The `non_i64_param` bucket is **slightly over-attributed**: the i32/bool-param cases
  (e.g. `good_fn`/`scale_no_shift`/`mul_q16` in `mind-dir-and-tests`, `pack_confirm`
  `timeout: u32` in `other-examples`) are blocked by their *body/return*, not the param.
  This does not change RUNS — they remain `aot_only` — only the burndown attribution.
- **Watch item, not a bug today:** `bool` is in `_I64_ABI_PARAM_TYPES` but **excluded**
  from `_I64_ABI_RETURN_TYPES` (i1-vs-i64 ABI mismatch, verified 2026-06-02). The asymmetry
  is deliberate and correct. If anyone "simplifies" the two frozensets to be equal, every
  `-> bool` fn would flip to a **real false-positive RUNS** that does not lower. Keep the
  asymmetry; it is load-bearing.

**Net:** 0 honest reclassifications available. Burndown = codegen, full stop.

---

## 3. Prioritized real-codegen plan (deterministic, cross-substrate byte-identical)

Each item is genuine MLIR-lowering work that makes the construct **RUN** with the
load-bearing invariant intact: **bit-identical output across CPU / ARM / GPU**, signed
into the evidence chain. Ordered by **(symbols unlocked ÷ effort)**. Effort tiers:
**S** ≈ days (one construct, no new ABI), **M** ≈ 1–2 sessions (new lowering pass +
byte-identity gate), **L** ≈ multi-session (full type-system surface / new region SSA).

> Gating rule for every item: land behind the existing `fixed_point` byte-identical gate
> + `cross_substrate` canaries; criterion one-sided (faster always OK, >10 % slower =
> operator decision). No construct ships until the same program emits the same bytes on
> x86 and ARM.

### Tier-1 — highest leverage (unlock the 66 % core)

**P1. `struct` aggregate lowering — 170 symbols (31.5 %). Effort: M→L.**
The single biggest *coherent* win. `struct` here is **not** the academic struct — it's
the load-bearing record layer the whole self-host stack is built on: `ParseResult`,
`TcResult`, `EmitState` (self-host-main, example-parser, example-emit_ir), plus
`Args`/`File`/`Map`/`Vec`/`String`/`TermSize`/`Box`/`Text`/`Point`/`Config`. Work:
deterministic struct ABI (fixed field offsets, no ASLR-dependent layout), struct-literal
lowering, field-read/field-store lowering, struct param/return passing. **MIND already
proved the binary-IR machinery for struct-literal alloc + field threading at the mic@3
self-host layer** (per-ctor `__mind_alloc` handle, `__mind_load_i64`/`__mind_store_i64`
desugar — see `ffb90ce`/`b27e77b`). This is **porting a proven byte-exact desugar from
the self-host driver into the shipped `lower_to_mlir` path**, not net-new design. Ratio:
~170 symbols for one (large) lowering pass + ABI = the best symbols/effort on the board.
Many `non_i64_param`/`non_i64_return`/`non_i64_let` symbols whose blocker is a *struct*
type (`File`, `Args`, `TermSize`, `Box`, `Text`) also flip with P1.

**P2. `Vec` + `String` heap-collection lowering — 94 (`strings`) + 29 (`Vec_Map`) ≈ 123 symbols (22.8 %). Effort: M.**
`String` and `Vec` are themselves structs in MIND (`String{addr,len,cap}`,
`Vec{addr,len,cap}` — `std/string.mind`, `std/vec.mind`), so P2 **rides on P1's struct
ABI** and is mostly: deterministic heap allocator (fixed, reproducible bump/arena so
addresses don't leak into output), `vec_push`/`vec_get`/`string_push_byte`/`string_addr`
lowering, and the `print("…")` string-literal path for the example `main()`s. Doing P1
then P2 back-to-back is the intended sequence — P2 is near-free once struct + a
deterministic allocator exist. Together **P1+P2 = ~293 symbols (54 % of the deficit)**.

### Tier-2 — non-i64 scalar surface (the ML/numeric demos)

**P3. `f32`/`f64` + `i32`/`bool` scalar ABI completion — large fraction of `non_i64_param` (186) + `non_i64_return` (21) + `non_i64_let` (31). Effort: M.**
A big share of `non_i64_param` is **float/int-scalar**, not aggregate: `f32`/`f64`/`i32`
params on remizov (ODE solvers), the cli/fs/io/process i32-fd `let`s, and the `bool`/`i32`
lint fns. Work: real `f32`/`f64`/`i32` MLIR types with a **deterministic float contract**
(fixed rounding mode, FMA/reassoc disabled, no `-ffast-math`, identical NaN-canonicalization
on every substrate — this is the byte-identity crux for floats and the place determinism
can silently break). Fix the `-> bool` i1/i64 return-ABI mismatch (`mind_index.py:241-246`)
so `-> bool` and `bool` `let`s lower. This is where MIND's deterministic-codegen thesis
is *most* differentiated (cuBLAS/OpenBLAS structurally can't match it) but also where the
work is real: a fixed float ABI across CPU/ARM/GPU is the hard, valuable part.

### Tier-3 — long tail / aspirational surface

**P4. `tensor<…>` + `diff tensor<…>` autodiff lowering — the bulk of examples-zoo (40), mind-dir-and-tests (44), remizov (66), other-examples autodiff (24). Effort: L.**
These files are **explicitly headed "ASPIRATIONAL DEMO — not yet buildable with the open
mindc"** (conv/mlp/transformer/logistic/linear models, ODE solvers). They need the full
tensor type system: shape-typed tensors, broadcasting, `matmul`/`conv2d`/`relu`/reduction
lowering, **and** reverse-mode autodiff (`diff tensor`) — all deterministic and
substrate-byte-identical. This is genuine multi-session compiler work and **correctly**
the lowest priority: it unlocks demo files, not the self-host/std core. Land after the
real product surface (P1–P3) runs.

**P5. `borrow` (`&Request`, `&[u8]`) + `enum` — 7 (`borrow`) symbols (1.3 %). Effort: S→M.**
Lowest count (`policy.mind`: 6 enums + 3 structs, `&borrow` params). `&[u8]`/`&Request`
lower to (ptr,len) / by-ref struct once P1's struct ABI exists, plus enum-tag lowering +
sum-type exhaustiveness. Small symbol count → do it last, but **S effort** once P1 lands.

### Ordering summary (symbols ÷ effort)

| Order | Item | Symbols | Effort | Cumulative RUNS if landed |
|-------|------|--------:|:------:|--------------------------:|
| 1 | P1 struct ABI + field/literal lowering | ~170 | M→L | 62.0 % → **74.0 %** |
| 2 | P2 Vec/String (rides P1 + det. allocator) | ~123 | M | → **82.7 %** |
| 3 | P3 f32/f64/i32/bool scalar ABI (+`-> bool` fix) | ~120* | M | → **91.2 %** |
| 4 | P4 tensor + autodiff (aspirational demos) | ~120* | L | → **99.5 %** |
| 5 | P5 borrow + enum | ~7 | S→M | → **100 %** |

\* P3/P4 overlap inside the `non_i64_param`/`non_i64_let`/`non_i64_return` buckets
(float-scalar vs. tensor); the split is approximate — exact per-symbol attribution needs
a re-scan after P1/P2 collapse the struct/collection symbols out of those buckets.

---

## 4. Realistic reachability (honest)

**(a) False-positive fix alone:** **+0.0 % → stays 62.0 % (878/1417).** There are zero
confirmed classifier false-positives. The metric is conservative *by design* and is
currently *correct*; there is no slack to harvest. Anyone who claims a quick reclassify
win here is reporting a fake win. (The only classifier change worth making — equalizing
the param/return frozensets — would *introduce* false-positives, not remove them.)

**(b) Single highest-leverage construct (P1 struct):** **62.0 % → ~74.0 %** (+170 RUNS →
1048/1417). One coherent (large) lowering pass, largely a port of the already-byte-exact
self-host struct desugar, clears the entire `ParseResult`/`TcResult`/`EmitState`/`Args`/
`File` record layer — and drags additional `non_i64_*` struct-typed symbols with it.

**(c) Full roadmap (P1–P5):** **→ ~100 % RUNS**, but only because P4 finishes the tensor
+ autodiff type system that the aspirational demos require. Realistic intermediate
milestones: **P1 ≈ 74 %**, **P1+P2 ≈ 83 %** (self-host + std core runnable), **+P3 ≈
91 %** (numeric/scalar surface), **+P4+P5 ≈ 100 %** (ML/ODE demos + borrows/enums).

**Honest bottom line:** 100 % RUNS is not a metric-tuning exercise — it is **finishing the
full-language deterministic codegen**: a deterministic struct ABI, a reproducible heap
allocator, a fixed cross-substrate float contract, and shape-typed tensor + reverse-mode
autodiff lowering, each landed behind the `fixed_point` byte-identity gate and the
`cross_substrate` canaries. That is **multi-session** work. The high-leverage truth is
that **P1+P2 (~21 % of total symbols, ~54 % of the current deficit) takes RUNS from 62 %
to ~83 % and makes the self-hosting compiler and the standard library genuinely run** —
which is the milestone that matters far more than the last-mile aspirational-demo points.
