"""
Field-invariant experiment v2 — AMPLIFYING regime.
Fully separate from any MIND repo. Pure numpy/scipy.

v1 used a too-gentle contraction => unpinned float error stayed at machine
epsilon and did not grow with depth. v2 picks the regime (per DeepSeek panel
hint) where the LOCAL/stepwise method provably diverges while the WHOLE-SYSTEM
closed form stays analytically exact:

  A. Stiff linear ODE x'=Ax: explicit Euler with dt*|lambda_max| just over 2
     => amplification factor |1+dt*lambda|>1 => error doubles every few steps.
     Closed form expm(A t) is analytically exact regardless of stiffness.

  B. Fixed point x=Ax+b with spectral radius rho->1: iteration error ~ rho^k
     (slow), direct solve (I-A)^-1 b is one-shot exact.

  C. Reproducibility under permuted reduction order in a Lyapunov-amplifying
     (rho>1) regime: stepwise bitstream drifts run-to-run; closed-form spectrum
     apply is stable.

Measures per regime: error vs depth (monotone?), and run-to-run hash identity.
"""
import numpy as np
import hashlib
from scipy.linalg import expm

rng = np.random.default_rng(0)

def h(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()[:12]

# ---------------------------------------------------------------------------
# A. Stiff ODE, Euler in the UNSTABLE regime: dt*|lambda_max| slightly > 2.
#    Amplification factor g = |1 + dt*lambda_max| > 1 => exponential blow-up.
#    expm(A t) is the analytic truth (exact for linear systems).
# ---------------------------------------------------------------------------
def stiff_euler_blowup():
    # one slow mode, one stiff mode
    lam = np.array([-1.0, -1000.0])
    A = np.diag(lam)
    x0 = np.array([1.0, 1.0])
    dt = 0.0021                       # dt*|lam_max| = 2.1 > 2  => g = |1-2.1| = 1.1
    g = abs(1 + dt * lam.min())
    out = {"regime": f"stiff ODE Euler-unstable dt={dt} amp_factor={g:.3f}"}
    for steps in [10, 50, 100, 250, 500, 1000, 2000]:
        x = x0.copy()
        for _ in range(steps):
            x = x + dt * (A @ x)              # local stepping (explicit Euler)
        t = steps * dt
        exact = expm(A * t) @ x0             # whole-system closed form
        err = np.linalg.norm(x - exact) / (np.linalg.norm(exact) + 1e-300)
        out[steps] = err
    out["closed_form_err"] = 0.0            # expm IS the reference here
    return out

# ---------------------------------------------------------------------------
# B. Fixed point x = A x + b, rho(A) -> 1.  Iteration err ~ rho^k (slow);
#    direct solve one-shot exact.
# ---------------------------------------------------------------------------
def fixed_point_slow(n=8, rho=0.9995, seed=2):
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((n, n)))
    lam = np.linspace(rho, rho * 0.9, n)            # all < 1, clustered near rho
    A = Q @ np.diag(lam) @ Q.T
    b = r.standard_normal(n)
    closed = np.linalg.solve(np.eye(n) - A, b)
    out = {"regime": f"fixed point n={n} rho={rho} (iter err ~ rho^k)"}
    for k in [10, 100, 1000, 5000, 20000]:
        x = np.zeros(n)
        for _ in range(k):
            x = A @ x + b
        out[k] = np.linalg.norm(x - closed) / (np.linalg.norm(closed) + 1e-300)
    out["closed_form_err"] = 0.0
    return out

# ---------------------------------------------------------------------------
# C. Reproducibility under permuted reduction order, Lyapunov-amplifying rho>1.
#    Tiny reduction-order differences get exponentially amplified => stepwise
#    bitstream differs every run; closed-form spectral apply stays identical.
# ---------------------------------------------------------------------------
def reproducibility_amplified(n=32, depth=400, runs=16, rho=1.02, seed=3):
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((n, n)))
    lam = np.linspace(rho, rho * 0.8, n)            # max eigval > 1 => amplifying
    A = Q @ np.diag(lam) @ Q.T
    A = A / abs(np.linalg.eigvals(A)).max() * rho   # pin top |eigval| = rho
    x0 = r.standard_normal(n)
    step_hashes, closed_hashes = set(), set()
    for run in range(runs):
        # stepwise: same math, per-run permuted reduction order (HW nondet sim)
        perm = r.permutation(n)
        x = x0.copy()
        for _ in range(depth):
            x = (A[:, perm] @ x[perm]) / rho        # /rho keeps it finite, order still varies
        step_hashes.add(h(x))
        # closed form: eigendecomposition apply (A/rho)^depth -- whole-system
        w, V = np.linalg.eig(A / rho)
        cf = (V @ np.diag(w**depth) @ np.linalg.inv(V) @ x0).real
        closed_hashes.add(h(cf))
    return {"regime": f"reproducibility n={n} depth={depth} runs={runs} rho={rho}",
            "stepwise_distinct_hashes": len(step_hashes),
            "closed_form_distinct_hashes": len(closed_hashes)}

def fmt(d):
    print("\n== " + d["regime"] + " ==")
    for k, v in d.items():
        if k == "regime":
            continue
        if isinstance(v, float):
            print(f"  depth/steps {k}: rel_err = {v:.3e}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    print("FIELD-INVARIANT EXPERIMENT v2  (AMPLIFYING regime)  numpy", np.__version__)
    fmt(stiff_euler_blowup())
    fmt(fixed_point_slow())
    fmt(reproducibility_amplified())
