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

The narrow-integer call ABI (`i32`/`u32` across call boundaries) and struct
narrow-field ABI are sound. Gated by the keystone and `cross_substrate` suites.

---

## 2. Floating-point semantics

MIND follows **IEEE 754** and pins every edge case. Scalar `f64`/`f32` follow the
IEEE rules below; the **Q16.16 fixed-point** tier is fully deterministic and
byte-identical across substrates today.

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
  substrates (x86 == ARM), gated by `cross_substrate` (12/12). ✅ For `f32`, the
  strict tier fixes the reduction order and pins one transcendental
  implementation, yielding bit-identical results across substrates. 📋
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
is no external toolchain whose `-ffast-math` can leak in. ✅ The `strict_math` /
`fast_math` surface is being finalised. 📋

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

---

## Summary

> MIND does not depend on undefined behaviour, backend quirks, hidden randomness,
> race conditions, or accidental execution order. Every questionable case is either
> precisely **defined**, explicitly **rejected**, or explicitly **marked
> non-deterministic** — and the result is **verifiable** through the artifact's
> `trace_hash`.
