# The Determinism Contract

> Status legend — **✅ shipped** (implemented and gated in CI) · **📋 specified** (the
> rule is fixed by this contract; enforcement is in progress).

## Definition

MIND is **deterministic**: the same source code, the same inputs, the same
compiler/runtime version, and the same target settings produce the same output —
every time, on every conforming implementation.

This is not a claim that every mathematical question has one truth. It is a claim
that the **language defines one exact behaviour** for every operation and never
leaves the result to accident — to undefined behaviour, backend quirks, hidden
global state, race conditions, or the order in which a parallel runtime happens to
execute.

Every questionable operation falls into exactly one of three buckets:

| Bucket | Meaning |
|--------|---------|
| **define** | The spec picks one rule. The operation always produces that result. |
| **reject** | The operation is a compile error or a defined domain error. |
| **mark non-deterministic** | The operation is explicitly opted into a `fast` / unordered mode that the spec labels non-deterministic. |

The forbidden fourth bucket — *"sometimes 1, sometimes 0, sometimes NaN, depending
on backend / GPU / optimization level"* — does not exist in MIND.

### Determinism is verifiable, not promised

Determinism in MIND is **checkable**. Each compiled artifact embeds an evidence
chain whose `trace_hash = SHA-256` of the canonical `mic@3` bytes. Identical
(source, inputs, version, target) ⇒ identical `trace_hash`. `mind verify ./artifact`
confirms it without trusting the build host. No other toolchain ships a verifiable
determinism contract. ✅

The verifier also re-derives the artifact's **floating-point contract mode**
(`strict` / `relaxed`) directly from the hashed body — so `mind verify
--require-strict-fp ./artifact` fails closed unless the artifact was lowered on
the strict path (no FMA-contraction, no `f32` reduction reassociation). Because
the mode is a pure function of bytes the `trace_hash` already attests, this is
build-host-independent and adds no wire-format surface. The gate fails closed on
`relaxed`, on an `unknown` mode, AND on an **unattested** artifact (no
evidence_chain, hence no `trace_hash` attesting the mode) — it never silently
passes. ✅

---

## 1. Integer semantics — ✅ shipped (v0.10.0)

Integer arithmetic is fully deterministic and byte-identical across substrates.

| Case | Rule |
|------|------|
| `x / 0` | `= 0` (defined; no trap, no UB) |
| `x % 0` | `= 0` (defined) |
| `INT_MIN / -1` | `= INT_MIN` (defined; no overflow trap) |
| Integer overflow | wraps two's-complement (defined; identical on x86 and ARM) |
| Oversized shift (`count ≥ bit-width`) | given a defined result (never UB) |
| Condition truthiness (`if c`) | tests `c != 0` — the whole value, not the low bit |

The normative overflow rule (defined two's-complement wraparound) is honored at
**every** layer: the interpreter, the MLIR/native artifact (plain `arith.addi`, no
`nsw`/`nuw`), **and** compile-time constant folding. Const-folding is **exact-or-skip** —
a constant subtree is folded only when the result is representable; on overflow the
expression is left unfolded so the runtime wraps it identically. It never emits a wrapped
*or* saturated constant, so a const-position expression and the identical runtime-position
expression always produce the same bytes.

The narrow-integer call ABI (`i32`/`u32` across call boundaries) and struct
narrow-field ABI are sound. Gated by the keystone and `cross_substrate` suites.

---

## 2. Floating-point semantics

MIND follows **IEEE 754** and pins every edge case. **Scalar** `f64`/`f32`
arithmetic now compiles and runs on the **strict deterministic path** —
`arith.mulf`/`arith.addf` with no `fmuladd`-contraction, no `fastmath` flag, and
no reassociation, in fixed source order — and is **run-to-run bit-identical** (a
loop-carried `f64` integrator such as a Lorenz–Euler step reproduces a reference
bit-for-bit). Because scalar `+ − × ÷ √` are correctly-rounded IEEE-754
operations, cross-ISA bit-identity follows on any conforming FPU (x86-SSE2 ==
ARM-NEON), and is now verified on real ARM64 (NEON) hardware (2026-07-05)
byte-identical to the x86_64 references. The **Q16.16
fixed-point** tier is fully deterministic and byte-identical across substrates
(x86 == ARM) today. The IEEE edge-case rules below pin the remaining special
values.

| Case | Rule | Status |
|------|------|--------|
| `1.0 / 0.0` | `+Inf` (IEEE) | 📋 |
| `-1.0 / 0.0` | `-Inf` (IEEE) | 📋 |
| `0.0 / 0.0` | `NaN` (IEEE) | 📋 |
| `sqrt(-1.0)` | `NaN` (IEEE); `strict_domain` → defined domain error | 📋 |
| `pow(0.0, 0.0)` | `1.0` — IEEE `pow`: `x^0 == 1` for **all** `x` (including `0` and `NaN`) | 📋 |
| `powr(0.0, 0.0)` | `NaN` — IEEE `powr` (= `exp(0·log 0)`), the strict real-power form | 📋 |
| `limit_form(0^0)` | indeterminate — symbolic/calculus context, not a number | 📋 |
| NaN comparisons | all comparisons `false` except `!=`; `min`/`max`/`sort` use a defined total order (NaN sorts last) so results are deterministic | 📋 |
| Rounding | round-to-nearest-even (IEEE default), fixed | 📋 |
| Scalar `f64`/`f32` arithmetic (`+ − × ÷`, fixed source order, no FMA-contraction) | strict path — run-to-run bit-identical; verified bit-identical across an x86 CPU and an NVIDIA GPU (CUDA, `sm_86`) via the no-FMA-contraction contract (`-ffp-contract=off` ≡ `--fmad=false`) | ✅ |
| Vector `f32` BLAS reductions (`dot` / `L1` / `matmul`, the `*_v` kernels) | strict tier — FMA unfused to separate `mulf`+`addf`, horizontal sum a pinned fixed-order fold, so bit-exact (no `vector.fma` / `vector.reduction <add>`); run-to-run bit-identical, verified `objdump`-clean (0 fused FMA) on x86 | ✅ (x86; ARM re-verify pending) |
| Other vector `f32`/`f64` reductions (tensor `sum`, GPU, `f64`) | ordered reduction trees / superaccumulators — still a documented ~1e-4 relative tolerance, not yet bit-identity | 📋 |
| Transcendentals (`sin`, `exp`, …) | vendored correctly-rounded libm (not host libm) | 📋 |
| Q16.16 fixed-point | fully deterministic, byte-identical x86 == ARM | ✅ |

### `0^0` — worked example

`0^0` is the canonical "vague" case. MIND removes the vagueness by choosing the
function, not the mood:

```mind
pow(0, 0)        // 1     — integer / exact arithmetic, deterministic
pow(0.0, 0.0)    // 1.0   — IEEE pow, deterministic (x^0 == 1 for all x)
powr(0.0, 0.0)   // NaN   — IEEE powr (real power), deterministic
limit_form(0^0)  //        indeterminate — symbolic/calculus, not a runtime number
```

`pow(0,0) = 1` matches the empty-product convention and every mainstream language,
and keeps polynomial / tensor `x^0` well-behaved. `powr` is the honestly-NaN real
power. Both are deterministic — you pick which one. Mathematically honest **and**
never an accident.

---

## 3. Backend must not change meaning

Two execution tiers; the contract is **bit-identity**, never "within tolerance"
(tolerance-equal is a correctness-testing notion, not a determinism guarantee).

- **Strict tier (default).** Integer and Q16.16 results are byte-identical across
  substrates (x86 == ARM), gated by the `cross_substrate` suite (17 gates /
  21 `#[test]` fns). ✅ **Scalar** `f64`/`f32` arithmetic runs on the strict path
  today — fixed source order, no FMA-contraction, no reassociation — and is
  run-to-run bit-identical. The `cross_substrate` suite enforces x86(avx2) ==
  ARM(neon) for the **integer/Q16.16** canaries by construction; the strict
  *scalar-float* legs pin on avx2 + run-to-run and DEFER the neon assertion until
  a real-aarch64 bless (`pin_or_defer_strict_fp`), so float byte-identity is a
  contract we hold, not yet a suite-gated one for every substrate. Separately, and
  **outside** the `cross_substrate` suite (which has no GPU arm), the same `f64`
  Lorenz–Euler integrator has been verified by hand to produce results identical
  to the last bit on an x86 CPU and on an NVIDIA GPU (CUDA, `sm_86`), because the
  same no-FMA-contraction contract
  (`-ffp-contract=off` on the CPU, `--fmad=false` on the GPU) forbids the fused
  multiply-add the hardware would otherwise apply — with FMA fusion left on, the
  chaotic trajectory diverges, worse the longer it runs. Because scalar
  `+ − × ÷ √` are correctly-rounded IEEE-754 operations, the same holds on any
  conforming FPU — now also verified on real ARM64 (NEON) hardware (2026-07-05),
  where all 12 `cross_substrate` canaries reproduced the x86_64 references
  byte-for-byte. ✅ The **f32 vector BLAS reductions** — the `dot` / `L1` / `matmul`
  `*_v` kernels — are now on the **strict tier** too: their per-lane FMA is
  unfused to separate `mulf`+`addf` and the horizontal sum is a pinned
  fixed-order fold, so they emit no `vector.fma` / `vector.reduction <add>` and
  are bit-exact (run-to-run bit-identical, `objdump`-verified free of fused FMA
  on x86; ARM re-verification pending). ✅ What is **not yet** bit-identical:
  broader vector reductions (tensor `sum`, `f64`, GPU) still carry a documented
  ~1e-4 relative tolerance pending canonical reduction trees / superaccumulators,
  and **transcendentals** (`sin`/`exp`/…) await a vendored correctly-rounded
  libm. 📋
- **Fast tier (opt-in).** Explicitly labelled non-deterministic; results may differ
  by substrate. You opt **into** it — you never get it by accident.

GPU and accelerator execution (CUDA, Metal, ROCm, WebGPU) ships in the commercial
`mind-runtime`; bit-identical determinism across those substrates is on the roadmap.
The open-source `mindc` in this repo emits for the CPU.

---

## 4. Parallel execution must not randomise results

For floating-point, `(a + b) + c != a + (b + c)`. A parallel runtime that reorders
a reduction can change the result. MIND's rule:

- **Strict is the default.** Reductions use a defined reduction order (or a stable
  kernel) and are reproducible regardless of thread/lane count.
- **Fast is opt-in and labelled non-deterministic.**

```mind
sum(x)                 // strict: defined reduction order, reproducible
sum(x, mode = "fast")  // explicitly non-deterministic
```

Fixed-reduction-order kernels are the active work (Phase 13.6). 📋

---

## 5. Compiler optimizations cannot change observable behaviour

MIND separates two math modes:

- **`strict_math` (default).** The compiler may not rewrite floating-point in ways
  that change the observable result: **no** `x * 0 → 0` (because `NaN * 0 = NaN`,
  `Inf * 0 = NaN`), **no** reassociation, **no** FMA-contraction. NaN, Inf, and
  rounding are preserved exactly.
- **`fast_math` (opt-in).** Permits those rewrites; the spec labels the result
  non-deterministic.

The native-ELF backend emits an image that is a **pure function of the IR** — there
is no external toolchain whose `-ffast-math` can leak in. ✅ The scalar float path
already realises `strict_math` end-to-end: `arith.mulf`/`arith.addf` lower with
**no `fmuladd`-contraction and no `fastmath` flag**, so a loop-carried `f64`
computation is bit-identical run-to-run. ✅ The full `strict_math` / `fast_math`
opt-in surface (the `fast`-tier toggle) is being finalised. 📋

---

## 6. Randomness must be explicit

There is no implicit `rand()` reading hidden global state. Randomness is always
seeded and explicit:

```mind
let rng = Random(seed = 42)
let x   = rng.normal(shape = [1024])   // same seed ⇒ same tensor, every run
```

The generator is **counter-based** (Philox / Threefry), keyed by
`(seed, element_index)`. Because each element's draw is a stateless function of its
index, parallel generation is reproducible regardless of execution order, and the
result is identical across substrates. This is the basis of MIND's
reproducible-across-hardware `randn` (Phase 11 deterministic intrinsic). 📋

### Determinism is enforced; non-determinism never leaks untraced

MIND programs are **deterministic by default** — but as a systems language MIND
can compile *anything*, including a genuinely non-deterministic operation (an
unseeded PRNG draw `random` / `rand_uniform`, or a wall-clock / stdin read `now` /
`read_line`). Such a program is not silently accepted, nor silently rejected;
non-determinism is a **traced, attested opt-in** across three layers:

1. **Build gate.** Producing a runnable (`--emit-obj` / `--emit-shared`) or
   attested (`--emit-evidence`) artifact from a program that calls such a builtin
   is **rejected fail-loud**, naming the offender and pointing at the seeded
   `Random(seed = 42)` API — *unless* you pass `--allow-nondeterministic`. Hidden
   non-determinism cannot slip into a shipped artifact by accident.
2. **Honest attestation.** With `--allow-nondeterministic` the program compiles,
   and its `evidence_chain.determinism` field — *derived from the IR* — declares
   `nondeterministic`. Every other module (including seeded `randn(shape, seed)`)
   declares `deterministic`. The flag authorises the *build*, never the *label*.
3. **Verify re-derivation (tamper-proof).** The `determinism` field lives in the
   MAP epilogue, outside the `trace_hash` anchor, so on an unsigned artifact it
   would be forgeable. `mind verify` therefore **re-derives** the mode from the
   hashed body (exactly as it re-derives `fp_mode`), reports that authoritative
   value, and **fails closed** if the stored field disagrees — a forged
   `deterministic` label cannot pass. `mind verify --require-deterministic` fails
   closed for a consumer that requires reproducibility.

The result: the attestation can never lie, and non-determinism is always
**opt-in and labelled, never by accident**. ✅

---

## Summary

> MIND does not depend on undefined behaviour, backend quirks, hidden randomness,
> race conditions, or accidental execution order. Every questionable case is either
> precisely **defined**, explicitly **rejected**, or explicitly **marked
> non-deterministic** — and the result is **verifiable** through the artifact's
> `trace_hash`.
