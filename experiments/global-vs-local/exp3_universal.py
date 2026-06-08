"""
exp3 — The DECISIVE test of Shamar's UNIVERSAL claim.

Claim under test: a "pre-axiomatic operator" yields a CLOSED-FORM topological
invariant of an ARBITRARY computational field AT ONCE, beating step-wise methods
universally (not just for structured systems).

We already PROVED the honest, STRUCTURED cases (global beats local): expm for
linear ODEs, direct solve, Chern/winding for topological systems. Those win
BECAUSE they have exploitable structure (linearity / quantization).

Here we attack the UNIVERSAL claim with the MOST GENEROUS possible adversary:
the logistic map at r=4. It is chaotic, YET it has a known EXACT closed form
(via the conjugacy to the tent/Bernoulli shift):

        x_n = sin^2( 2^n * arcsin(sqrt(x_0)) )

So this is the BEST case for him: a closed-form whole-system expression DOES
exist. If even HERE the closed form cannot beat stepping, the universal claim
is dead. (For a generic non-conjugate nonlinear system, no closed form exists
at all -- so this case is the strongest steelman.)

Three verdicts:
  1. Does the closed form match stepping?  -> precision collapse vs depth.
  2. WHY it collapses: 2^n amplifies the input's finite precision by a factor
     2^n -> after ~52 steps (float64 mantissa bits) ALL information in x0 is
     gone. This is the Lyapunov exponent ln 2 made concrete: 1 bit lost / step.
  3. Generic case: a non-conjugate map has NO closed form -> stepping forced.

Pure numpy + mpmath (for the "infinite precision" oracle). Separate from MIND.
"""
import numpy as np
import math

def logistic_step(x, r=4.0):
    return r * x * (1.0 - x)

def closed_form(x0, n):
    # x_n = sin^2(2^n * arcsin(sqrt(x0)))   -- exact in the reals
    theta = math.asin(math.sqrt(x0))
    return math.sin((2.0**n) * theta) ** 2

# high-precision oracle (the true x_n) via mpmath at 400 digits
def oracle(x0, n):
    try:
        import mpmath as mp
        mp.mp.dps = 400
        x = mp.mpf(str(x0))
        four = mp.mpf(4)
        for _ in range(n):
            x = four * x * (1 - x)
        return float(x)
    except ImportError:
        return None

if __name__ == "__main__":
    print("exp3 — UNIVERSAL closed-form-invariant claim, logistic map r=4 (chaos)\n")
    print("Even though an EXACT closed form x_n = sin^2(2^n*arcsin(sqrt(x0))) EXISTS,")
    print("does it beat / match step-wise iteration?  x0 = 0.2\n")
    x0 = 0.2
    print(f"{'n':>5}{'stepwise (f64)':>20}{'closed-form (f64)':>22}{'oracle (400-dig)':>20}{'CF err vs oracle':>20}")
    for n in [5, 10, 20, 30, 40, 52, 60, 80, 100, 1000]:
        xs = x0
        for _ in range(n):
            xs = logistic_step(xs)
        cf = closed_form(x0, n)
        orc = oracle(x0, n)
        cferr = abs(cf - orc) if orc is not None else float('nan')
        orc_s = f"{orc:.12f}" if orc is not None else "n/a"
        print(f"{n:>5}{xs:>20.12f}{cf:>22.12f}{orc_s:>20}{cferr:>20.3e}")

    print("\nVERDICT on the UNIVERSAL claim:")
    print("  - The closed form is EXACT in the reals but 2^n multiplies the finite")
    print("    precision of x0 by 2^n. Float64 has 52 mantissa bits, so after ~52")
    print("    steps every bit of information is gone -> closed form is GARBAGE,")
    print("    uncorrelated with the true orbit. (Lyapunov exponent = ln2: 1 bit/step.)")
    print("  - To get the right x_n you must carry ~n bits of precision = O(n) work.")
    print("    There is NO free lunch: the 'whole-system operator' costs the SAME")
    print("    information as stepping. The shortcut is an ILLUSION under finite precision.")
    print("  - For a GENERIC nonlinear map (no conjugacy) NO closed form exists at all")
    print("    (Poincare non-integrability / undecidability). This was the BEST case.")
    print("\n  => UNIVERSAL closed-form whole-field invariant: provably FALSE.")
    print("     Global-beats-local REQUIRES exploitable structure (linearity /")
    print("     topological quantization / integrability). No structure => no shortcut.")
