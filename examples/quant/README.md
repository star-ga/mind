# examples/quant — deterministic derivative pricing

Self-contained MIND programs that price derivatives and measure risk with
bit-reproducible arithmetic, plus their known-answer tests.

| Example | What it is | Why it earns its place |
|---|---|---|
| `black_scholes/` | Black-Scholes-Merton closed form: call, put, and five first-order Greeks, with a continuous dividend yield | The **easy** case for determinism — one pass, no accumulated state. Exercises the full transcendental tier (`exp`, `log`, `sqrt`, `erfc`). |
| `crr_binomial/` | Cox-Ross-Rubinstein lattice: European and American, call and put | The **hard** case. 20100 dependent backward-induction steps at n=200, each consuming the previous step's rounded output, plus a path-dependent early-exercise decision. |
| `greeks_higher_order/` | vanna, volga, charm, veta, speed, zomma, color, ultima | Second and third derivatives. Each differentiation amplifies the kernel's own error by exactly the operation that defines the Greek, so the bands widen visibly with order — the conditioning is readable in the source. |
| `implied_vol/` | implied volatility recovered from a price | An **inverse** problem: a fixed-iteration solve, not a closed form. Fixed trip count on purpose (below). |
| `implied_vol_surface/` | SVI parameterisation, analytic `w'`/`w''`, Durrleman no-butterfly `g(k)`, calendar-spread check | The no-arbitrage conditions are the content. Carries a **known-bad** parameter slice as a fixture, because a no-arbitrage check that never fires is not a check. |
| `barrier_analytic/` | Reiner-Rubinstein closed form, all eight knock types | Worst cancellation in the set — a far knock-in of 5.4e-27 assembled from terms of order 10. Also carries the in-out-parity blind spot documented below. |
| `monte_carlo_gbm/` | deterministic MC: integer-recurrence PRNG, inverse-CDF sampling, antithetic variates | The only path-sampling example. Inverse-CDF rather than Box-Muller or rejection so the draw count is **not** data-dependent, and the accumulation order is pinned. |
| `bond_curve/` | bond price, YTM solve, Macaulay/modified duration, convexity, zero-curve discounting | Not an option. Every quantity here is one a fixed-income desk is asked to reproduce. |
| `portfolio_risk/` | parametric and historical VaR, Expected Shortfall, Euler allocation, portfolio vol | VaR and ES are the numbers reported to a regulator. Demonstrates ES subadditivity **and** the VaR case where subadditivity fails — the Basel argument, executable. |

### Two rules every example follows, for the same reason

**No tolerance loops.** Every iterative solve here — implied vol, YTM, the SVI
round trip, the inverse normal CDF — runs a **fixed** iteration count rather
than looping until a convergence tolerance is met. A tolerance loop's trip
count depends on rounding, so it can differ between substrates; the answer then
differs too, and byte-identity is lost on an input nobody flagged as special.

**Pinned accumulation order.** Sums are accumulated left to right, explicitly,
and never reassociated. This is the property the tolerance bands below are
structurally unable to defend, which is why it is a source-level discipline
rather than a test.

## What these demonstrate, and what they do not

`examples/lorenz_f64.mind` is already a stronger determinism proof than either
of these: a chaotic integrator whose trajectory diverges exponentially, held
bit-identical across x86 `avx2` and ARM `neon`. It is also inert as *evidence*,
because no institution has ever had to defend a Lorenz attractor to an
examiner.

These examples change nothing about the compiler. They change who can read the
result. The gap they close is legibility, not capability.

**The claim, stated exactly.** These modules do **not** claim to match SciPy, a
host libm, or a vendor pricing library in the last bits. They cannot, and they
are not trying to: a re-derived `erfc` rounds differently from Cephes, and both
are within their stated accuracy. The claim is narrower and harder:

> the same source, compiled for two different ISAs, produces byte-identical
> output, with a stated accuracy bound against a 50-digit oracle.

Accuracy and reproducibility are orthogonal. Determinism is what makes an error
analysis meaningful — not a substitute for one.

## Running them

```sh
cd examples/quant/black_scholes && mindc run   # exit 0 == every check passed

# or run the whole corpus, each from a clean build directory:
./scripts/run_quant_examples.sh
```

**Delete `target/` before trusting an exit code.** A stale build directory makes
`mindc run` report the exit status of a *previously* built artifact rather than
the current source — measured, in one session, as three different exit codes (0,
1, 7) from one unmodified file, which reads exactly like nondeterminism and is
not. `run_quant_examples.sh` removes `target/` before and after every example
for this reason.

The exit code is the **count of failed checks**, so a partial failure is never
rounded up to success.

There are no `[test]` functions in any of these files, deliberately. The mindc 0.10.2
`[test]` interpreter mis-evaluates an f64 comparison in a function body (it
takes the then-branch unconditionally), so every guarded function would return
through its leading `if x != x` guard and every interpreted assertion would pass
vacuously — a false green. The native backend evaluates f64 branches correctly,
so the gate is build-and-run.

Every example also duplicates the `std.detmath` kernels rather than importing them,
because mindc 0.10.2 cannot carry an f64 across a module boundary (roadmap
17.1): an imported `fn f(x: f64) -> f64` types as `(i64) -> i64` at the call
site and miscompiles silently. Each duplicated block is delimited by
`KERNEL BEGIN` / `KERNEL END` markers and carries a `deferred:` note with the
upgrade path.

## How the tests are built

Reference values come from an **independent** 50-digit oracle (`mpmath`,
`mp.dps = 50`), rounded once to f64. A KAT generated from the code it tests
proves only that the code is unchanged, not that it is correct.

Three classes of check, in every file:

1. **Value** — against the oracle, within a stated ulp band.
2. **Structural** — theorems that must hold whatever the arithmetic does, and
   which catch whole classes of sign and discounting errors a single value
   check misses:
   - put-call parity, `C - P = S·e^(-qT) - K·e^(-rT)`
   - the delta relation, `Δ_call - Δ_put = e^(-qT)`
   - no-arbitrage bounds on the premium
   - monotonicity in spot and in volatility
   - **an American call on a non-dividend-paying stock equals its European
     counterpart to the last bit** (Merton 1973) — an exact equality, not a
     band, because early exercise is never optimal so the `max()` must never
     bind
   - a deep-ITM American put with no time value equals intrinsic exactly
   - the lattice converges toward the closed form as `n` grows
3. **Adversarial** — a deliberately *wrong* reference must be **rejected**. A
   spurious acceptance is counted as a failure. Without this, every check above
   could be passing merely because a tolerance is too wide.

### The deep-OTM case is the one to look at

`black_scholes` prices S=50, K=100, σ=0.2, T=0.25 — a call worth
**4.955e-12**. That number is the reason `std.detmath`'s `norm_cdf` is built on
`erfc` rather than on the reflected identity `1 - N(-x)`:

- the reflected form **cancels to exactly 0.0** in this regime
- an implementation that returns a hard zero here has not rounded, it has lost
  the number entirely, and will report every deep-OTM book position as exactly
  worthless with no indication of error

The KAT asserts both the value *and* that the premium is strictly positive, so a
flush-to-zero implementation cannot pass by abusing a loose relative band.

## Verified by mutation, not by assertion

A test suite that has never failed is not known to work. Every claim above was
checked by deliberately breaking the code and confirming the suite goes red:

| Mutation | Result |
|---|---|
| Impossibly tight band on the ATM call | caught (exit 1) |
| `norm_cdf` switched to the reflected identity (**the real-world bug**) | caught (exit 2) — fails the deep-OTM value *and* positivity checks while every other case still passes |
| Put drops its dividend discount factor | caught (exit 2) — via put-call parity |
| American `max()` forced to always bind | caught (exit 4) — via the Merton exact-equality check |
| Arbitrage guard loosened from `p > 1.0` to `p > 2.0` | **initially survived** — see below |

That last row was a real hole, found and fixed. The only probe was
`r=5.0, σ=0.01, n=2`, which yields `p ≈ 791` — so far outside `[0,1]` that a
guard loosened to `2.0` still caught it. The test passed for the wrong reason. A
second probe at `r=0.15, σ=0.1, n=2` (`p ≈ 1.0326`) now exercises the actual
bound, and the mutation is caught.

### A structural identity can be a theorem and still be insufficient

Worth its own section, because the lesson generalises past barriers and was
learned the expensive way — by specifying the wrong gate and having the code
catch it.

In-out parity — `knock_in + knock_out == vanilla` — is a genuine theorem, and
it looks like the ideal structural check: it survives a bad oracle, it ties
eight functions to one closed form, and it holds to 1e-49 at 50 digits. It was
specified as *the* check for `barrier_analytic`.

It is not sufficient, and the counterexample is not exotic. Price a **down**-
barrier that sits **above** spot — already breached at inception — at
`S=100, K=100, H=110, r=0.05, σ=0.2, T=1`:

| quantity | value |
|---|---|
| down-and-out call | **−20.78** (a negative option price) |
| down-and-in call | **31.23** (≈3× the vanilla) |
| parity residual | **exactly 0.0** |

Parity constrains only the **sum** of the two halves. Both can be arbitrarily
wrong in opposite directions and it still holds perfectly. Any barrier
implementation gated solely on parity has this hole.

The fix is a domain guard (`H < S` for down-knocks, `H > S` for up-knocks,
`canonical_qnan()` otherwise), plus — and this is the part that matters — an
assertion that parity **accepts** the out-of-domain pair. That records the blind
spot as executable: delete the guard later and the domain section goes red while
the parity section stays green, which localises the regression instead of
merely signalling one.

The general form: **an identity over an aggregate cannot constrain its
components.** Before trusting a structural check, ask what it leaves free.

## What these KATs cannot catch

Stated plainly, because a reader should not over-read a green run.

These are **tolerance bands**, wide enough (~4096 ulp in the lattice) to absorb
the method's own accumulated rounding. That makes them structurally **blind to
reassociation** — a change that is algebraically identical and numerically
different. Measured, not assumed: rewriting the inner recurrence

```
disc * (p * up + pm * dn)   as   disc * p * up + disc * pm * dn
```

changes the result in **5 of 6** test cases by ~500 ulp (~1e-13 relative), and
every band here still accepts it.

Tightening the bands is not the fix — past the accumulated error they would fail
for correct code. Detecting reassociation requires a **bit-exact** reference,
which is the job of the cross-substrate CI fixture: it compares *hashes* across
`avx2` and `neon`, and a hash has no tolerance to hide in.

> These KATs prove the pricing is **correct**.
> The CI fixture proves it is **reproducible**.
> Neither substitutes for the other.

## Status: not yet wired as a CI canary

Every example builds and runs green on x86_64, and the KATs are mutation-verified.
Neither is yet registered in `tests/cross_substrate_identity.rs`.

Wiring one requires, per the contract in
`tests/cross_substrate_identity/scalar-float-f64/`:

1. a kernel function in the harness with a stable exported symbol
2. a `manifest.toml` declaring `substrates`, the input spec, and
   `output_encoding` (here: `f64_bits_i64_le`)
3. a `reference_hashes.toml` with an `avx2` **and** a `neon` hash

Step 3 is the blocker and it is not a formality. Per RFC 0015 §3.1 / RFC 0020
§10, the `neon` hash must be **reproduced on real ARM64 hardware**, not asserted
from x86. The `scalar-float-f64` canary records exactly that provenance (GCP
Ampere Altra aarch64, 2026-07-05). Writing a `neon` hash computed on x86 would
make the fixture assert x86-equals-x86 — a gate that cannot fail, which is worse
than no gate.

## References

Original papers only.

- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate
  Liabilities." *Journal of Political Economy* 81(3), 637–654.
- Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of
  Economics and Management Science* 4(1), 141–183.
- Cox, J. C., Ross, S. A. & Rubinstein, M. (1979). "Option Pricing: A Simplified
  Approach." *Journal of Financial Economics* 7(3), 229–263.
