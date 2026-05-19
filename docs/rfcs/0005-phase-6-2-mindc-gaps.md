# RFC 0005 Phase 6.2 — mindc Feature Gaps (Design Note)

> **Status:** Design note. Multi-session pickup artifact captured at
> Phase 6.1 + A1.1 LUT-family completion (2026-05-18).
>
> **Position in landing table:** `docs/rfcs/0005-pure-mind-std-surface.md`
> §"D phases" Phase 6.2 row.
>
> **Predecessor:** Phase 6.1 self-host lexer (shipped at mindc v0.4.4,
> `examples/lexer/`, commit `29bd08b`).
>
> **Drivers:** Both the Phase 6.1 lexer build and the mind-nerve A1.1
> Q16.16 LUT family build surfaced concrete mindc parser feature gaps.
> Either gap blocks the next-level self-host work (parser in MIND) and
> makes Q16.16-substrate work pay an O(N) line-count tax for what
> should be a constant-size literal.

## The two gaps

### Gap 1 — `while` statement support

**Discovered while writing:** `examples/lexer/main.mind`.

**Reproducer:**

```mind
fn count_to(n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        i = i + 1;
    }
    i
}
```

**Result on mindc v0.4.4:**
```
error[parse][E1001]: expected expression
  --> /tmp/while_test.mind:3:18
   |     while i < 10 {
   |                  ^
```

**Diagnosis:** The `while` references in `src/parser/mod.rs` are part of mindc's *own internal Rust scanner* (e.g. `while self.pos < self.b.len()`), **not** the surface grammar mindc accepts. The token `while` is not a recognised statement keyword. The Pratt expression parser also does not register `while` as a prefix token, hence the "expected expression" diagnostic at the condition position.

**Impact:**
- Every loop in `examples/lexer/main.mind` is written as tail recursion.
- Every loop in `mind-nerve/mind/luts/*.mind` and `mind-nerve/mind/kernels/*.mind` is written as tail recursion.
- Existing `examples/policy.mind` is aspirational — it uses `while` and fails to parse on v0.4.4.
- The "loop as canonical iteration" idiom that Rust, Mojo, and Carbon all rely on simply does not exist at the language surface today.

**Two options for Phase 6.2:**

1. **Add `while`-statement parsing to mindc.** Small grammar addition — `while` keyword + condition expression + block body, lowering to a labelled loop in IR. Module-level gated (`std-surface` is the natural home since that's where loops will land first). Bench-gate impact: trivial — one new prefix token, one new lowering path. Compile-speed moat untouched.
2. **Canonicalize tail recursion as the MIND loop primitive.** Document `fn step(state, …) -> …` recursive shape as the idiomatic loop in mind-spec. Every loop must be in explicit fixed-state form. This is **already what RFC 0005 examples do** today; promoting it to canonical means cleaning up `examples/policy.mind` and updating the spec.

**Recommendation:** **Option 1 (add `while`).** Tail recursion forces every loop to allocate a stack frame per iteration, which on x86_64 is fine but on Cerebras / NPU substrates introduces a per-iteration control-flow cost the compiler can't fold. `while` lowers to a basic-block loop in MLIR with zero extra frames. The recursion form should remain *supported* (it's a legitimate idiom) but not be *required*.

**Effort:** ~80 LOC in `src/parser/mod.rs` (statement parser + prefix token), ~50 LOC in `src/mlir/lowering.rs` (basic-block loop lowering), ~30 LOC of tests covering the existing `examples/policy.mind` corpus + new edge cases (nested `while`, `while` with `break`/`continue` — the latter two are their own follow-on items).

### Gap 2 — array literal / const-blob syntax for static data

**Discovered while writing:** `mind-nerve/mind/luts/exp_q16.mind`, `tanh_q16.mind`, `sqrt_q16.mind`.

**Symptom:** A 4096-entry Q16.16 LUT compiled into a `.mind` file requires 4096 lines of `__mind_store_i64(handle + N * 8, VAL)` for each entry. The `exp_q16` LUT file is **4,145 lines** of which **4,096 lines are stores**. Same for `tanh_q16` (4,151 lines, 4,096 stores) and `sqrt_q16` (4,220 lines, 2,048 stores). The actual *logic* is ~50 lines per file; the rest is data.

**Reproducer:**

What we'd like to write:
```mind
const EXP_Q16_LUT: [i64; 4096] = [
    65536, 65488, 65440, 65392, /* ...4092 more entries... */, 0
];
```

What v0.4.4 forces us to write:
```mind
fn _exp_q16_lut_init(handle: i64) -> i64 {
    __mind_store_i64(handle + 0, 65536);
    __mind_store_i64(handle + 8, 65488);
    __mind_store_i64(handle + 16, 65440);
    /* ... 4093 more lines ... */
    __mind_store_i64(handle + 32760, 0);
    handle
}
```

The result is **functionally identical** — both produce the same in-memory blob — but the line count, parse time, and IR size all blow up by O(N) instead of being constant. The compile-speed moat survives because the LUT files are gated under `mind-nerve/` (not in mindc's hot path), but the *source maintainability* and *IR size at link time* are real costs.

**Diagnosis:** mindc v0.4.4's parser supports:
- Scalar constants: `const X: i64 = 42`
- Tensor literals (via `tensor.zeros` and friends, but those allocate runtime tensors, not const blobs)
- Struct literals: `Point { x: 1, y: 2 }` (P0e)

It does **not** support:
- Array literals: `[1, 2, 3]`
- Fixed-size array types: `[i64; N]`
- `include_bytes!`-style file inclusion at the language level (mindc-Rust uses `include_str!` for the bundled stdlib, but that's not exposed in `.mind`)

**Two options for Phase 6.2:**

1. **Add array literals + fixed-size array types.** Parse `[expr, expr, …]` as an `Instr::ArrayLit`; allocate at compile time (const) or alloca (let-mut). Type as `[T; N]` with `T = i64` for the LUT use case. ~200 LOC parser + lowering. Cross-module imports gate (or fold into `std-surface`).
2. **Add `include_bytes!`-style language-level file inclusion.** A statement like `const EXP_Q16_LUT: i64_blob = include_bytes("exp_q16.bin");` reads a precompiled binary blob at compile time and stamps it into the binary as a const section. ~150 LOC + a build-system contract about where the blob files live. Simpler than (1) for this specific use case, but less general.

**Recommendation:** **Option 1 (array literals + fixed-size arrays).** Same shape as Rust's `[T; N]`. Useful far beyond LUTs (lookup tables in general, fixed-size buffers, compile-time tables for parser keyword sets, etc.). `include_bytes!` is a secondary ergonomic ask once arrays exist.

**Effort:** ~200 LOC parser + lowering, ~80 LOC of tests covering the 4096-entry LUT use case + smaller tables (16-entry, 256-entry). Bench-gate impact: zero on default build (gated behind `std-surface`); LUT lowering is straightforward — array literals become const-section blobs.

## Phase 6.2 scope summary

Pick:

- **`while`-statement support** — unblocks ergonomic loop expression in pure-MIND code (the lexer, kernels, and the Phase 6.2 parser-in-MIND all benefit).
- **Array literals + fixed-size array types** — unblocks O(1)-source-cost static tables (Q16.16 LUTs, keyword tables, fixed-size buffers).

Either one alone is a useful Phase 6.2 ship. **Both** is the right scope because they're the two gaps that surfaced naturally during Phase 6.1 + A1.1 production work — they're proven-by-use, not speculative.

**Tag target:** mindc v0.5.0 (the "self-host substrate complete" tag). Phase 6.3 (type-checker in MIND) can then begin in earnest.

**Bench-gate budget:** module-level gating (likely under `std-surface` for `while` and a new `array-literals` feature for the array work — or fold both into `std-surface`). Default-build hot path remains byte-identical to v0.4.4. The `.bench-baseline-2026-05-18-rfc0005.txt` floor and +7% cap stay in effect.

## Test contract for Phase 6.2

Add to `tests/`:

1. `std_surface_while_statement.rs` — `while` parses + lowers + IRs cleanly for: trivial counted loop, nested while, while with mutable state outside, while inside `if`.
2. `std_surface_array_literals.rs` — `[1, 2, 3]` parses + types as `[i64; 3]`; `[i64; 0]` is empty; large array literal (4096-entry hand-written) parses without stack overflow; `const FOO: [i64; 4] = [1, 2, 3, 4]` resolves and is accessible from a fn body.
3. `std_surface_lexer_smoke.rs` — promotes `examples/lexer/EXPECTED.md` from documentation to integration test: compiles `examples/lexer/main.mind` to a `.so`, runs it on `examples/lexer/fixture.mind` as input bytes, diffs the output `Vec<i64>` against the 32-row expected table byte-for-byte.
4. `std_surface_lut_smoke.rs` — promotes `mind-nerve/tests/luts_smoke.mind` to an in-tree mindc test that compiles + runs the 5 LUTs and asserts the documented ULP bounds.

## Decision points for Nikolai

1. **Approve Phase 6.2 scope as `while` + array literals?** Recommended yes.
2. **Tag target — v0.5.0 or v0.4.5?** Recommend v0.5.0 because this is a real surface-grammar growth, not a patch.
3. **Order — `while` first or array literals first?** Recommend `while` first because it unblocks the lexer / parser / kernel work already in flight (mind-nerve A1.2 kernels are all tail-recursive today; landing `while` lets them be progressively re-written in idiomatic loop form). Array literals can land in a follow-on tag without blocking anyone.

## References

- `examples/lexer/main.mind` (Phase 6.1 seed, commit `29bd08b`) — every loop is tail recursion.
- `mind-nerve/mind/luts/exp_q16.mind` (A1.1, commit `bd33d50`) — 4,145 lines, 4,096 of them stores.
- `docs/rfcs/0005-pure-mind-std-surface.md` §"Adoption plan" Phase 6.2 row.
- `mind-internal/plans/encoder-port-audit-2026-05-18.md` §7 ("mindc feature requests") — declared "none required" for A1, which is still true at the *language semantics* level; the gaps documented here are *ergonomic*, not semantic. A1 ships fine without them, but pays the O(N) source-line tax.

## Phase 6.3 addendum (type-checker in MIND)

### Gap 3 — unsigned i64 literals reject as integer overflow

**Discovered while writing:** `examples/typecheck/main.mind`.

**Reproducer:**

```mind
pub fn fnv_offset() -> i64 { 14695981039346656037 }
```

**Result on mindc v0.4.4:**
```
error[parse][E1001]: integer overflow
  --> examples/typecheck/main.mind:232:50
   | pub fn fnv_offset() -> i64 { 14695981039346656037 }
   |                                                  ^
```

**Diagnosis:** mindc v0.4.4's parser interprets every integer
literal as signed i64.  FNV-1a 64-bit's offset basis
(`0xCBF29CE484222325` = `14695981039346656037`) exceeds
`i64::MAX` (`9223372036854775807` = `0x7FFFFFFFFFFFFFFF`) and
trips the overflow guard.  Same shape for any unsigned-i64
constant beyond `i64::MAX` — including useful values like:
- FNV-1a 64-bit offset basis (`14695981039346656037`)
- SipHash round constants (`0x736f6d6570736575` is fine,
  `0xdoddledoodoo`-family rotated constants are not)
- `u64::MAX` (`18446744073709551615`)

**Impact:**
- The Phase 6.3 type-checker swapped FNV-1a for DJB2 (seed 5381,
  prime 33) to stay inside `i64::MAX` — fine for short
  identifiers but a documented narrower hash than FNV.
- Q16.16 LUT work that uses unsigned-i64 sentinels has to
  pre-shift them into signed range.
- Any future hash-based collection (proper `std.map` bucketing,
  Bloom filters, signing-key constants) will hit this.

**Two options for Phase 6.4 (or 6.3b if hash quality
becomes load-bearing first):**

1. **Accept literals as `u64` and reinterpret-cast to i64
   internally.**  Parser change: treat literals in the range
   `[i64::MAX+1, u64::MAX]` as `u64` and store the bit
   pattern as a signed-i64 with the same byte representation
   (i.e. `14695981039346656037` becomes the negative i64
   `-3750762994362895579`).  ~30 LOC parser change; preserves
   the seven-intrinsic surface.
2. **Introduce a `u64` literal suffix (`14695981039346656037u`).**
   Same idea but explicit at the source site; bigger grammar
   change but lets the type-checker distinguish "I want this
   bit pattern" from "I want this number".  ~80 LOC parser
   change including the suffix lexer rule.

**Recommendation:** **Option 1.**  All existing arithmetic
intrinsics are signed-i64 anyway, so the bit-pattern
reinterpretation is the only thing the user actually needs.
The negative-i64 display in tooling is acceptable because
the surface contract is "this constant always has these
bytes", which Option 1 preserves byte-for-byte.

**Effort:** ~30 LOC parser change in `src/parser/mod.rs`
(extend the integer-literal range check), ~20 LOC of tests
covering FNV-1a basis, `u64::MAX`, the i64-MAX boundary.
Bench-gate impact: zero — same parse path, wider range
check.

---

Authored 2026-05-18 alongside the mind-nerve Phase II Q16.16 substrate sprint.
Pickup artifact for the next compiler session.

Updated 2026-05-18 with Phase 6.3 Gap 3 (unsigned-i64 literals).
