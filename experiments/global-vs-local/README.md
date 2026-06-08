# Global vs Local — "Closed-form whole-field invariant" experiments

Self-contained numpy/scipy experiments, **separate from any MIND repo**, testing
the claim that a "pre-axiomatic operator" can compute a closed-form topological
invariant of a whole computational field at once, beating step-wise methods.

Run: `python3 exp2.py` · `python3 topo.py` · `python3 chern.py` · `python3 exp3_universal.py`

## Two claims, two verdicts

### 1. Honest "global beats local" (STRUCTURED systems) — TRUE, proven

Where the system has exploitable structure, a whole-system closed form beats
step-wise iteration on accuracy, cost AND reproducibility:

| Experiment | Local (step-wise) | Global (closed-form) |
|---|---|---|
| Stiff linear ODE (`exp2.py` A) | Euler error → **10^84** at 2000 steps | `expm(At)` **exact** |
| Fixed point ρ→1 (`exp2.py` B) | iter error ~ρ^k, 20k iters to 2e-5 | direct solve **one-shot exact** |
| Reproducibility, ρ>1 (`exp2.py` C) | permuted reduction → **16 distinct hashes/16 runs** | spectral apply → **1 hash** |
| Winding number (`topo.py`) | grid scan = resolution-dependent garbage | exact enclosed-zero **integer** |
| Chern number (`chern.py`) | contractible patch = meaningless fraction | exact band Chern **0,±1** |

This is the determinism wedge, demonstrated with hash gates in pure numpy.

### 2. UNIVERSAL closed-form invariant (ARBITRARY systems) — FALSE, proven

`exp3_universal.py` is the decisive test. It uses the **most generous possible
adversary**: the logistic map `x_{n+1}=4x(1−x)`, which is chaotic but
*does* have an exact closed form `x_n = sin²(2ⁿ·arcsin(√x₀))`. So a whole-system
expression genuinely exists — the best case for the universal claim.

It still fails:

```
   n      stepwise (f64)     closed-form (f64)    oracle (400-dig)   CF err vs oracle
  40      0.097874404077        0.097931864176     0.097917028774          1.484e-05
  52      0.189772438429        0.002995482476     0.024442373820          2.145e-02
  60      0.564749697146        0.985899538968     0.368810812672          6.171e-01   <- decorrelated
 1000     0.138223258077        0.750972545389     0.177388796003          5.736e-01
```

**Why:** `2ⁿ` multiplies x₀'s finite precision by 2ⁿ — exactly 1 bit lost per
step (the Lyapunov exponent ln2 made concrete). After ~52 steps (float64 mantissa)
all information in x₀ is gone; the closed form is uncorrelated noise. To get the
right answer you must carry ~n bits = **O(n) work — the same cost as stepping.**
No free lunch.

For a *generic* nonlinear system with no conjugacy, **no closed form exists at
all** (Poincaré non-integrability; undecidability/halting for arbitrary programs).
The logistic case was the steelman, and it still dies.

## Bottom line

- **Global beats local — TRUE, but only with exploitable structure** (linearity,
  topological quantization, integrability). We own this with curves + hashes.
- **A universal closed-form whole-field operator — FALSE.** Forbidden by chaos
  (sensitive dependence / positive Lyapunov), non-integrability, and
  undecidability. Even where a closed form exists, finite precision makes it
  cost the same as stepping.

The honest core of the thesis is real and demonstrated. The universal/mystical
version has no there there — tested as hard as it can be tested.
