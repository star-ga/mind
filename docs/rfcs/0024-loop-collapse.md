# RFC 0024: Loop Collapse — prove-or-fail closed-form replacement of counted loops (`#[collapse]`)

| Field | Value |
|---|---|
| RFC | 0024 |
| Title | Loop Collapse — prove-or-fail closed-form replacement of counted loops (`#[collapse]`) |
| Status | **Draft** — feature UNBUILT. This RFC fixes the `#[collapse]` contract, the `E22xx` diagnostic family, the ring-exact closed-form builder, and the `evidence_chain.collapse.*` receipt. Slice 0 (Z/2^64 affine sums) is the first implementation target; geometric / Q16.16-affine / fixed-point are named follow-on slices. |
| Authors | STARGA Inc. |
| Created | 2026-07-15 |
| Depends | RFC 0015 (cross-substrate bit-identity — the property collapse must preserve), RFC 0016 (evidence-chain emission — the carrier for the collapse receipt), RFC 0017 (`mindc verify` — re-derives the collapse proof), RFC 0021 (canonical mic@3 IR + MAP epilogue — the hashed body collapse rewrites) |

---

## 1. Summary

`#[collapse]` is an opt-in attribute on a counted loop that instructs `mindc` to
**prove the loop equal to a closed form and replace it at compile time, or fail to
compile.** A loop that computes a closed-form-expressible quantity — an affine sum
(`1 + 2 + … + n`), a geometric or power series, a reached fixed point — is turned
from `O(n)` runtime work into `O(1)` evaluation while producing a **byte-identical**
result on every substrate.

The contract is deliberately narrow and unforgiving: `#[collapse]` never silently
keeps the loop. Either the compiler discharges the proof and emits the closed form,
or it emits an `E22xx` diagnostic and the program does not compile. The closed form
is evaluated with **ring-exact arithmetic** — exact in `Z/2^64` (two's-complement
wrapping) or in Q16.16 fixed-point — so no float reassociation is introduced and the
collapsed result is bit-identical across `x86_avx2`, `arm_neon`, and the mic@3
reference emitter. When a loop collapses, a re-derivable **receipt** is recorded in
the artifact's evidence chain (`evidence_chain.collapse.*`) so an auditor can confirm
the collapse without ever re-running the original `O(n)` loop.

## 2. Motivation

MIND's wedge is cross-substrate bit-identity (RFC 0015): the same source produces a
byte-identical artifact and a byte-identical result on every claimed substrate,
because integer and Q16.16 arithmetic are defined operations, not backend accidents
(`docs/determinism.md` §1). That property is what makes a *compile-time* rewrite
safe to trust: if the compiler can prove a loop equals a closed form **within the
loop's own arithmetic ring**, replacing the loop cannot change the observed result
on any substrate — the O(1) value *is* the O(n) value, bit for bit.

Mainstream compilers already do a weak, opportunistic version of this (LLVM's
scalar-evolution / `-freroll`/idiom recognition folds some induction-variable sums),
but it is *best-effort and silent*: the optimizer collapses a loop when it happens to
recognize the shape, leaves it alone otherwise, and gives the programmer no way to
**require** the collapse or to learn that it did not happen. For a language whose
entire value proposition is auditable determinism, "the optimizer might have done it"
is the wrong contract.

`#[collapse]` inverts that. It is an **assertion by the programmer** — "this loop has
a closed form; prove it or tell me why not" — checked the same way MIND checks every
other determinism obligation: define, or reject. The payoff:

- **Latency becomes a compile-time property.** A collapsed loop has no trip count at
  runtime; a governance kernel that sums a bounded ledger, or a numeric routine that
  evaluates a series, runs in constant time regardless of `n`.
- **The result stays exact and portable.** Because the closed form is ring-exact, the
  collapse is invisible to the cross-substrate hash — it changes *how long* the
  program takes, never *what bytes* it produces.
- **The optimization is auditable.** The `evidence_chain.collapse.*` receipt records,
  per collapsed loop, the closed-form class and the witness needed to re-derive the
  proof, so the speedup is a *verifiable* fact and not an unobservable optimizer
  decision.

## 3. Guide-level explanation

### 3.1 The contract

`#[collapse]` attaches to a loop. Its meaning is a single sentence:

> **Prove this loop equals a closed form evaluable exactly within its own arithmetic
> ring, and replace it — or refuse to compile with an `E22xx` diagnostic. Never keep
> the loop silently.**

```mind
pub fn gauss(n: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut i:   u64 = 1;
    #[collapse]
    while i <= n {
        acc = acc + i;   // two's-complement wrapping is DEFINED (docs/determinism.md §1)
        i   = i + 1;
    }
    acc
}
```

The compiler recognizes `acc` as an affine sum over the induction variable `i`,
proves the closed form `acc = n·(n+1)/2 (mod 2^64)`, evaluates it with the ring-exact
builder of §5.2, and emits IR equivalent to a straight-line computation. The loop
body is gone from the artifact; `gauss(1_000_000)` costs the same as `gauss(3)`.

### 3.2 What you get when it succeeds

Nothing changes about the *result*. `gauss(n)` returns exactly the value the loop
would have accumulated, including the two's-complement wraparound for large `n`. What
changes is that the mic@3 body now contains the closed form, and the evidence chain
gains a collapse receipt (§6):

```
$ mindc gauss.mind --emit-evidence out/gauss.mic3
$ mindc verify out/gauss.mic3
artifact:   out/gauss.mic3
trace_hash: <64 hex>  [OK — matches recomputed]
collapse:   1 loop collapsed — [affine_sum / z2_64]  [OK — proof re-derived]
result:     PASS
```

### 3.3 What you get when it fails

`#[collapse]` on a loop the compiler cannot reduce is a **compile error**, with a code
from the `E22xx` family (§5.3) that names *why* the proof failed and how to proceed:

```
error[E2201]: #[collapse] loop is not a recognized closed-form shape
  --> ledger.mind:12:5
   |
12 |     #[collapse]
13 |     while i < len {
14 |         acc = acc + table[i] * table[i];   // not affine in the induction variable
   |               ^^^^^^^^^^^^^^^^^^^^^^^^^^ accumulation depends on data, not on `i`
   |
   = note: `#[collapse]` proves an affine/geometric/fixed-point closed form; a
           data-dependent reduction has none.
   = help: remove `#[collapse]` to keep the loop, or hoist the reducible part.
```

This is the load-bearing difference from an ordinary optimizer: a programmer who wrote
`#[collapse]` gets *told* it did not fire, at compile time, instead of shipping an
`O(n)` loop they believed was `O(1)`.

## 4. Scope and non-goals

### 4.1 In scope

Loops whose accumulated quantity has a closed form that is **exact within the loop's
own ring** — `Z/2^64` (`u64`/`i64` two's-complement) or Q16.16 fixed-point:

- **Affine sums** (slice 0, this RFC): `Σ (p·i + q)` — Gauss and its linear
  generalizations.
- **Power sums / geometric series** (follow-on slice): `Σ i^k`, `Σ a·r^i`, subject to
  the ring-invertibility caveat of §8.2.
- **Reached fixed points** (follow-on slice): an iteration `x ← f(x)` that provably
  attains `x = f(x)` exactly in Q16.16, whose tail iterations collapse to the
  fixed-point constant.

### 4.2 Non-goals

- **Not a general loop optimizer.** `#[collapse]` is a proof obligation on a
  recognized fragment, not a heuristic that "tries hard." Unrecognized loops are
  rejected, not partially transformed.
- **Not a `must_use`-style lint.** A lint is advisory and bypassable; a failed
  `#[collapse]` is a hard `E22xx` error (same discipline as RFC 0004's evidence
  tokens).
- **Not a float optimizer.** Collapse never reassociates floating-point reductions —
  that would break strict-FP byte-identity (`docs/determinism.md`; RFC 0017
  `--require-strict-fp`). A float accumulator in a strict-FP context is rejected
  `E2206`; loops in an explicitly `relaxed` FP mode are out of scope for collapse
  entirely.

### 4.3 Permanently rejected — the arbitrary-precision boundary (`E2205`)

**MIND has no bignum tier, and `#[collapse]` will never introduce one.** Any loop
whose faithful closed-form evaluation would require an intermediate value that does
not fit the loop's own ring (`Z/2^64` or Q16.16) is rejected with `E2205`, and this
rejection is **permanent** — it is a boundary of the feature, not a missing slice.

The canonical example is a Collatz-style `3x+1` iteration (`examples/collatz.mind`).
Two independent reasons place it outside `#[collapse]` forever:

1. **No closed form exists.** Collatz stopping time has no known closed form (the
   conjecture is open ~90 years); the recognizer cannot manufacture one.
2. **Even a bounded orbit overflows the ring.** A faithful compressed-map trajectory
   from a large seed grows through intermediate magnitudes on the order of `3^101`
   (see `examples/collatz.mind`), which do not fit `Z/2^64`. Reproducing such a value
   would require arbitrary precision MIND does not have.

The distinction from the *permitted* Gauss case is precise and is the reason the
even-factor lemma (§5.2) matters: Gauss has a closed form **evaluable entirely within
`Z/2^64`** without ever forming a wider intermediate; Collatz has neither a closed
form nor a ring-bounded evaluation. `#[collapse]` collapses the former and refuses the
latter — it never fabricates a value the loop itself could not have produced.

## 5. Reference-level explanation

Collapse runs as an **IR → IR rewrite** in the lowering pipeline, gated by the
`#[collapse]` attribute, and it runs **before mic@3 emission**. Consequently the
`trace_hash` (RFC 0016 §3.2, SHA-256 over the canonical mic@3 bytes) is computed over
the *collapsed* program: the artifact hashes the closed form, exactly as it would hash
any other straight-line computation. The receipt (§6) is provenance *about* the
collapse and lives in the trailing MAP epilogue, outside the hashed body.

### 5.1 The recognized fragment (slice 0 — Z/2^64 affine sum)

A loop is a **collapsible affine sum** iff all of the following hold; failing any one
yields the indicated diagnostic:

1. It is a `while` with a single integer induction variable `i` of a ring-exact type
   (`u64`/`i64`), initialized to a loop-invariant `a`, guarded by `i <= b` or
   `i < b`, and the **only** update to `i` is a unit step `i = i + 1` as the last
   statement of the body. *(else `E2202`)*
2. The bounds `a` and `b` are loop-invariant — no dependence on loop-carried state.
   *(else `E2203`)*
3. Apart from the induction increment, the body is a single accumulation
   `acc = acc + e(i)`, where `acc` is a ring-exact loop-carried variable and
   `e(i) = p·i + q` is **affine** in `i` with loop-invariant coefficients `p`, `q`.
   *(else `E2201`)*
4. `acc` and `i` are the *only* loop-carried values; the body performs no call with
   effects, no store to escaping memory, and no I/O. *(else `E2204`)*
5. The closed form is evaluable entirely within the loop's ring (§5.2). A form that
   would need a wider-than-64-bit intermediate is rejected. *(else `E2205`)*

The closed form for the fragment is

```
count   = (b >= a) ? (b - a + 1) : 0          // trip count over the integers
S(a,b)  = Σ_{i=a..b} i                          // Gauss partial sum (§5.2)
acc_out = acc_0  +  p · S(a,b)  +  q · count    // all (mod 2^64)
```

For canonical Gauss (`a = 1`, `b = n`, `p = 1`, `q = 0`): `acc_out = S(1,n) =
n·(n+1)/2 (mod 2^64)`.

### 5.2 The closed-form builder — ring-exact arithmetic and the `n(n+1)/2` hazard

The closed form contains a division by 2, and **`/2` is not a well-defined operation
in `Z/2^64`** (2 is not invertible mod `2^64`). Getting this wrong is the classic
Gauss-in-a-fixed-width-ring bug. The builder must therefore compute `S(a,b) (mod
2^64)` *exactly*, using only defined `u64` operations and **without ever forming a
value wider than 64 bits**.

**The hazard (what NOT to do).** The naive form — wrapping-multiply, then halve the
reduced product — is wrong whenever the true product reaches `2^64`, because the
wraparound discards the bit the `/2` needs. Concretely, for `S(1, 2^32)`:

```
true S      = 2^32 · (2^32 + 1) / 2 = 2^63 + 2^31          (mod 2^64)
naive       = ((2^32 · (2^32+1)) mod 2^64) / 2
            = (2^32) / 2 = 2^31                             ✗ WRONG
```

**The fix — the even-factor lemma.** Of the two consecutive integers whose product is
being halved, **exactly one is even**, and halving *that* factor is an exact integer
operation that stays strictly below `2^63`. For `S(a,b)` the product is over the pair
`{ (a+b), (b−a+1) }`, whose sum `(a+b) + (b−a+1) = 2b+1` is **odd**, so exactly one
member of the pair is even — the lemma always applies. Halve the even member *as a
true integer* (never by dividing the already-wrapped increment), then wrapping-multiply
by the odd member:

```
// S(a,b) with a <= b, computed exactly in Z/2^64, no >64-bit intermediate:
let lo = a + b;             // may wrap; used only as a factor
let hi = b - a + 1;         // count; a <= b guaranteed by the guard below
if lo is even  { half = lo >> 1;                 S = half * hi }   // half < 2^63
else /* hi even */ {
    // hi even; compute hi/2 overflow-safe even when (a+b) forced the odd branch:
    half = (hi >> 1);       // hi even  =>  hi>>1 == hi/2 exactly
    S = lo * half
}
```

Because `half < 2^63` and the surviving factor is a `u64`, the true product `half ·
other` may exceed `2^64`, but its **low 64 bits are exactly `S (mod 2^64)`** — which
is precisely what a `u64` wrapping multiply returns. No 128-bit intermediate is ever
materialized. Re-running the `S(1, 2^32)` case: `lo = 2^32+1` (odd), `hi = 2^32`
(even) → `half = 2^31`, `S = (2^32+1)·2^31 mod 2^64 = 2^63 + 2^31` ✓.

For the pure `n(n+1)/2` form the same lemma applies to `{ n, n+1 }` (their sum
`2n+1` is odd): halve whichever of `n`, `n+1` is even, then wrapping-multiply.

This is the entire determinism story of the builder: **every emitted operation is a
defined `Z/2^64` op** (`docs/determinism.md` §1) with no reduction-order or
reassociation freedom, so the closed form lowers to the same bytes on every substrate
and equals the loop's own wrapping accumulation exactly.

### 5.3 The `E22xx` diagnostic family

Collapse proof failures live in the type-check code space (`E2xxx`; `docs/errors.md`).
`E21xx` is taken by shape validation, so collapse claims `E22xx`:

| Code | Meaning |
|---|---|
| `E2201` | `#[collapse]` loop body is not a recognized closed-form shape (accumulation not affine / not a supported reducible series). |
| `E2202` | Induction variable or step is not canonical (non-unit step, `i` mutated off the recurrence, more than one induction variable). |
| `E2203` | Loop bounds are not loop-invariant, or termination over the ring is not provable (e.g. `i <= b` with `b` = ring-max never terminates). |
| `E2204` | Loop body has an observable side effect (effectful call, escaping store, I/O) — collapse would change observable behavior. |
| `E2205` | Closed form would require arbitrary-precision / >64-bit intermediate values — **permanently rejected**; MIND has no bignum tier (§4.3). |
| `E2206` | Accumulator/operand type is not a ring-exact tier (e.g. a strict-FP `f32` accumulator) — collapse would require forbidden float reassociation. |

Every code fails the compile. There is no warning-only or best-effort tier: an
un-provable `#[collapse]` is an error, by design.

### 5.4 Cross-substrate byte-identity (why collapse preserves the wedge)

The collapse decision is a **pure function of the canonical IR**: the recognizer and
the closed-form builder read only the IR and loop-invariant constants, never a clock,
a target flag, or a substrate id. Therefore:

- Two substrates collapse *the same loops* to *the same closed form* → the collapsed
  mic@3 body is byte-identical across substrates → `trace_hash` is identical
  (RFC 0015 §3.1 inspectable identity).
- The closed form is ring-exact (§5.2) → the `O(1)` value equals the `O(n)` value bit
  for bit on every substrate.

Collapse is thus **inside** the cross-substrate invariant, not a threat to it: it is
gated on the keystone byte-identity suite exactly like any lowering change (§7).

### 5.5 Corner cases

- **Empty range (`a > b`).** The closed form guards `count = (b >= a) ? (b−a+1) : 0`
  and returns `acc_0`; it must reproduce the loop's zero-iteration result, not a
  wrapped negative count.
- **`i <= b` at ring-max.** If `b` is the ring maximum, `i` wraps back into range and
  the loop as written does not terminate; collapse must **not** fabricate a finite
  result — it rejects `E2203`. (`i < b` with `b` = ring-max terminates and is
  collapsible.)
- **Wrapping is faithful, not avoided.** The collapse reproduces the loop's
  two's-complement wraparound, since that wraparound is the loop's *defined* result
  (`docs/determinism.md` §1) — collapse is exact-equal to the loop, wrap included.

## 6. Evidence receipt — `evidence_chain.collapse.*`

### 6.1 Receipt keys

When one or more loops collapse, `mindc --emit-evidence` records sibling keys in the
mic@3 MAP epilogue. Like `signature.*` and `evidence_chain.trace_hash`, these keys are
appended **after** the hashed mic@3 body, so they are excluded from the `trace_hash`
preimage **by construction** (RFC 0016 §3.2 self-reference rule) — the receipt never
perturbs byte-identity. When signing is enabled (RFC 0016 Phase C, opt-in), the
signature preimage covers the full MAP-minus-`signature.*`, so the collapse receipt
**is** covered by the signature on a signed artifact.

| Key | Type | Meaning |
|---|---|---|
| `evidence_chain.collapse.count` | `uint` | Number of loops collapsed in this artifact. |
| `evidence_chain.collapse.<k>.class` | `string` | Closed-form class: `affine_sum` (slice 0). `geometric`, `power_sum`, `fixed_point` reserved. |
| `evidence_chain.collapse.<k>.ring` | `string` | `z2_64` \| `q16_16` — the ring the closed form was proven exact in. |
| `evidence_chain.collapse.<k>.witness` | `bytes` | Canonical encoding of the recognized recurrence — the loop-invariant `(a, b, p, q)` as value refs — sufficient to re-derive the closed form. |
| `evidence_chain.collapse.<k>.closed_form` | `bytes` (32) | mic@3 hash of the emitted `O(1)` closed-form sub-IR. |
| `evidence_chain.collapse.<k>.site` | `string` | Stable source site (function ordinal + loop index) for diagnostics — informational; not part of any hash preimage. |

### 6.2 `mindc verify` re-derives the proof — it never re-runs the loop

`mindc verify` (RFC 0017) discharges the collapse receipt with a **re-derivation**,
not a re-execution:

1. Read `class`, `ring`, and `witness` for collapse `<k>`.
2. Run the same deterministic closed-form builder (§5.2) on the witnessed recurrence.
3. Hash the re-derived closed-form sub-IR and compare to the recorded
   `closed_form` hash, byte for byte.
4. Report `collapse: <count> loop(s) collapsed — [class / ring] [OK — proof
   re-derived]`; on mismatch, exit nonzero naming the offending site.

Crucially, the verifier evaluates the *proof* (an `O(1)` re-derivation of the closed
form from its witness), **never the original `O(n) loop** — the loop is not even
present in the collapsed artifact. This mirrors RFC 0017 §4.1, where `verify`
recomputes `trace_hash` rather than re-executing the program: the artifact carries
enough to *re-derive* the claim locally, with no reference database and no second run.

## 7. Determinism gate (hard constraint on every increment)

Every collapse increment must hold all of:

1. Keystone `phase_g_keystone_bootstrap` 7/7 byte-identity stays green.
2. For every collapsible fixture, the collapsed artifact and a reference build with
   collapse disabled produce **the same result bytes** across the full input range
   (collapse is result-preserving, verified by a dedicated
   `tests/collapse_equivalence.rs`), including the wraparound and empty-range corners
   of §5.5.
3. The cross-substrate `trace_hash` for each collapsible Q16.16 fixture is equal on
   `x86_avx2` and `arm_neon` (RFC 0015 gate) — collapse does not perturb identity.
4. `mindc verify` re-derives every recorded `evidence_chain.collapse.*` receipt
   (§6.2) and exits 0.

Any recognizer or builder step whose output depends on a clock, target flag, address
layout, float reassociation, or reduction order is **disqualified** — it would break
both the equivalence gate and the evidence chain.

## 8. Worked examples

### 8.1 Gauss affine sum, `Z/2^64` (slice 0 — fully specified)

```mind
pub fn gauss(n: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut i:   u64 = 1;
    #[collapse]
    while i <= n {
        acc = acc + i;
        i   = i + 1;
    }
    acc
}
```

Recognized: `a=1`, `b=n`, `p=1`, `q=0`. Closed form `acc = S(1,n) = n·(n+1)/2 (mod
2^64)`, evaluated by the even-factor lemma (§5.2): halve whichever of `n`, `n+1` is
even, wrapping-multiply by the other. `gauss(1_000_000)` now costs a handful of `u64`
ops; `gauss(6_074_001_000)` returns the correct **wrapped** value, matching the loop.
Receipt: `class=affine_sum`, `ring=z2_64`.

A linear generalization collapses the same way:

```mind
#[collapse]
while i <= n { acc = acc + (3*i + 1); i = i + 1; }
// recognized p=3, q=1  =>  acc = 3·S(1,n) + 1·n   (mod 2^64)
```

### 8.2 Follow-on slices (specified here, implemented later)

- **Q16.16 affine sum.** Q16.16 is a fixed-point *integer* ring (`i32`/`i64` under the
  hood), so an affine sum of Q16.16 increments reduces by the identical Gauss lemma at
  the raw-integer level, with the Q16.16 `>>16`-per-product rounding rule (RFC 0006)
  applied only where a genuine Q16.16 *multiply* appears. Same `class=affine_sum`,
  `ring=q16_16`.
- **Geometric / power series.** `Σ a·r^i` has closed form `a·(r^n − 1)/(r − 1)`, but
  `(r − 1)` is **not invertible in `Z/2^64`** unless it is odd (i.e. `r` even). The
  geometric slice must prove ring-invertibility of `(r − 1)` (or fall back to a
  power-sum identity) before collapsing; otherwise it rejects rather than emitting a
  wrong quotient. `Σ i^k` (Faulhaber) collapses only when its rational coefficients
  clear to exact `Z/2^64` values.
- **Reached fixed point.** An iteration `x ← f(x)` in Q16.16 that provably attains
  `x = f(x)` exactly (a contraction reaching a representable fixed point — e.g. the
  Q16.16 cosine "Dottie" fixed point) collapses its tail to the fixed-point constant.
  This needs contraction/attainment machinery beyond affine recognition and is its own
  slice; the `fixed_point` receipt class is reserved for it.

## 9. Drawbacks

- **A new proof pass to maintain.** The recognizer and ring-exact builder are new
  compiler surface, gated by an attribute; each new closed-form class widens the
  audited transformation set.
- **Sharp edges by design.** `#[collapse]` fails loudly on loops that "look"
  reducible but are not (data-dependent bodies, non-unit steps). This is intended, but
  it puts the onus on clear diagnostics (§5.3) so the failure is actionable.
- **Small, fixed fragment initially.** Slice 0 recognizes only affine sums; many
  loops a user might hope to collapse (nested, multi-accumulator) are out of scope
  until later slices, and must be rejected rather than half-handled.

## 10. Rationale and alternatives

- **Opportunistic optimizer pass (rejected).** Collapsing loops silently when
  recognized — the LLVM model — gives the programmer no guarantee and no signal.
  MIND's contract is define-or-reject; a *requested* collapse that cannot be proven
  must be an error, not a missed optimization.
- **`must_use`/lint marker (rejected).** Advisory and bypassable; a determinism
  feature needs a type/compile-error-strength guarantee (cf. RFC 0004).
- **Fold at runtime / memoize (rejected).** A runtime shortcut is not
  build-host-independent and adds no auditable artifact; collapse is a *compile-time*
  rewrite whose proof travels in the evidence chain.
- **Symbolic-algebra dependency (rejected for slice 0).** A full CAS could recognize
  more forms but is unbounded work and a large trust surface. The affine fragment is
  decidable by a small, auditable pattern match with an exact ring builder; richer
  classes are added as bounded, individually-gated slices.
- **Impact of not doing this.** MIND keeps paying `O(n)` for quantities it can prove
  are `O(1)`, and the one determinism-native optimization that is *provably*
  result-preserving stays unavailable — while less-auditable compilers do a silent,
  weaker version.

## 11. Prior art

- **Scalar evolution / induction-variable simplification (LLVM, GCC).** Recognizes and
  folds some induction-variable recurrences. MIND's difference is the *contract*
  (opt-in, prove-or-fail) and the *exactness discipline* (ring-exact, evidence-carried)
  rather than best-effort silent folding.
- **Closed-form summation (Gauss; Faulhaber's formula for `Σ i^k`).** The mathematical
  basis for the affine and power-sum slices.
- **Exact fixed-width modular arithmetic.** The even-factor technique for `n(n+1)/2` in
  a two's-complement ring is the standard way to compute the sum exactly without a
  wider intermediate; §5.2 makes it normative for the builder.
- **MIND RFC 0004 (evidence token types).** The same "make a governance/determinism
  obligation a compile-time error, not a runtime surprise" discipline, applied to loops
  instead of attestations.

## 12. Unresolved questions

1. **Attribute placement grammar.** Whether `#[collapse]` attaches only to a `while`
   loop, or also to a `for`/range form and a block, once range loops land. Slice 0
   targets `while`.
2. **Multi-accumulator loops.** A body with several independent affine accumulators is
   collapsible in principle (collapse each); whether to admit it in an early slice or
   reject `E2201` until a dedicated slice is open.
3. **Witness encoding.** The exact canonical byte layout of `evidence_chain.collapse.
   <k>.witness` (value-ref scheme) — to be pinned with the mic@3 codec (RFC 0021)
   before the receipt is stabilized.
4. **Interaction with `mindc verify --cross-substrate`.** Whether a collapse receipt
   participates in the cross-substrate equality check or is verified purely locally
   (lean: local re-derivation is sufficient, §6.2).

## 13. Future possibilities

- **`geometric` / `power_sum` / `fixed_point` classes** as individually-gated slices
  (§8.2), each with its own ring-invertibility or attainment proof obligation.
- **Q16.16 series collapse** feeding numeric kernels (`examples/fft_q16.mind`,
  `examples/mandelbrot.mind` shapes) where a bounded fixed-point series has an exact
  closed form.
- **Governance integration.** A collapsed bounded-ledger sum in a 512-mind DIFC path
  carries its `evidence_chain.collapse.*` receipt into the governance `proof_chain`
  (RFC 0016 §7), so "this constant-time total is the exact sum of the ledger" is an
  auditable fact.
- **`#[collapse(strict)]` / diagnostics-only `mindc explain-collapse`.** A mode that
  reports which loops in a module *would* collapse and why the rest do not, without
  requiring the attribute up front.

## 14. Acceptance

1. `#[collapse]` on the §8.1 Gauss loop compiles to a straight-line closed form; the
   loop body is absent from the emitted mic@3 body.
2. `gauss(n)` returns bit-identical results to the un-collapsed loop for a range of
   `n` including values that force two's-complement wraparound and the empty-range and
   ring-max corners of §5.5.
3. `#[collapse]` on a data-dependent loop fails `E2201`; on a non-unit step fails
   `E2202`; on a non-invariant / non-terminating bound fails `E2203`; on an effectful
   body fails `E2204`; on a Collatz-style / >64-bit-intermediate loop fails `E2205`;
   on a strict-FP `f32` accumulator fails `E2206`. Every case is a compile error, none
   a warning.
4. The closed-form builder computes `n(n+1)/2 (mod 2^64)` exactly via the even-factor
   lemma, matching the loop on the `S(1, 2^32)` counterexample of §5.2.
5. A collapsible Q16.16 fixture has an equal `trace_hash` on `x86_avx2` and
   `arm_neon` (RFC 0015 gate) with collapse enabled.
6. `mindc --emit-evidence` records a well-formed `evidence_chain.collapse.*` receipt;
   `mindc verify` re-derives it (§6.2) and exits 0, and exits nonzero if the recorded
   `closed_form` hash is altered — **without re-running the original loop**.
7. The determinism gate of §7 is green on every increment.

## 15. Acknowledgements

The loop-collapse construct — proving a counted loop equal to a closed form and
replacing it at compile time, **exactly and within the loop's own arithmetic ring** —
is credited to **Valerii Salov**, MIND design partner. His work on exact, deterministic
integer arithmetic and the closed-form reduction of iterative computations is the
origin of this feature and of the even-factor discipline that keeps the collapse
ring-exact.

## 16. References

- RFC 0015 §3.1 — cross-substrate bit-identity (the property collapse preserves).
- RFC 0016 §3.2/§3.3 — the self-reference rule and the mic@3 `trace_hash` anchor
  (why the collapse receipt is outside the hashed body); §6 / Phase C — opt-in signing
  that covers the receipt.
- RFC 0017 §4.1 — `mindc verify` re-derivation model and exit-code discipline.
- RFC 0021 — canonical mic@3 `IRModule` + MAP epilogue (the container the receipt keys
  live in; witness encoding to be pinned here).
- `docs/determinism.md` §1 — defined two's-complement integer semantics and the
  exact-or-skip const-fold rule collapse generalizes to loops.
- `docs/errors.md` — the `E1xxx`–`E5xxx` pipeline code space (`E22xx` is claimed here).
- `examples/collatz.mind` — the `E2205` boundary: no closed form, and a
  `3^101`-magnitude trajectory that does not fit `Z/2^64`.
- `examples/fft_q16.mind`, `examples/mandelbrot.mind` — Q16.16 loop shapes for the
  follow-on slices.
