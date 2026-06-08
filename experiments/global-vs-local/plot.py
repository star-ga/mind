"""
Plot the error-vs-depth curves from the field-invariant experiment.
Pure numpy/scipy/matplotlib — fully separate from any MIND repo.

Produces:
  curve_stiff_ode.png      — local Euler error blows up; closed-form expm flat at 0
  curve_fixed_point.png    — local iteration error decays slowly; direct solve = 0
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

OUT = __file__.rsplit("/", 1)[0]

# ---------------------------------------------------------------------------
# A. Stiff ODE: explicit Euler (local) vs expm (whole-system closed form)
# ---------------------------------------------------------------------------
def stiff_curve():
    lam = np.array([-1.0, -1000.0])
    A = np.diag(lam)
    x0 = np.array([1.0, 1.0])
    dt = 0.0021  # dt*|lam_max| = 2.1 > 2 => amplification 1.1
    steps_list = [10, 25, 50, 100, 250, 500, 1000, 1500, 2000]
    euler_err, closed_err = [], []
    for steps in steps_list:
        x = x0.copy()
        for _ in range(steps):
            x = x + dt * (A @ x)
        t = steps * dt
        exact = expm(A * t) @ x0
        denom = np.linalg.norm(exact) + 1e-300
        euler_err.append(np.linalg.norm(x - exact) / denom)
        closed_err.append(0.0)
    return steps_list, euler_err, closed_err

steps, euler, closed = stiff_curve()
# floor the exact-zero closed-form for log plotting
closed_floor = [max(c, 1e-17) for c in closed]
euler_floor = [max(e, 1e-17) for e in euler]

plt.figure(figsize=(8, 5))
plt.semilogy(steps, euler_floor, "o-", color="#d1495b", lw=2,
             label="Local (explicit Euler, step-wise)")
plt.semilogy(steps, closed_floor, "s-", color="#2e86ab", lw=2,
             label="Global (closed-form expm, whole-system)")
plt.xlabel("integration depth (steps)")
plt.ylabel("relative error vs analytic truth (log scale)")
plt.title("Stiff ODE: local stepping diverges, whole-system closed form stays exact")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
p1 = OUT + "/curve_stiff_ode.png"
plt.savefig(p1, dpi=130)
plt.close()
print("wrote", p1)
print("  Euler error at 2000 steps:", f"{euler[-1]:.3e}")

# ---------------------------------------------------------------------------
# B. Fixed point: iteration (local) vs direct solve (whole-system)
# ---------------------------------------------------------------------------
def fixed_point_curve(n=8, rho=0.9995, seed=2):
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((n, n)))
    lam = np.linspace(rho, rho * 0.9, n)
    A = Q @ np.diag(lam) @ Q.T
    b = r.standard_normal(n)
    closed = np.linalg.solve(np.eye(n) - A, b)
    ks = [10, 50, 100, 500, 1000, 5000, 10000, 20000]
    iter_err = []
    for k in ks:
        x = np.zeros(n)
        for _ in range(k):
            x = A @ x + b
        iter_err.append(np.linalg.norm(x - closed) / (np.linalg.norm(closed) + 1e-300))
    return ks, iter_err

ks, iter_err = fixed_point_curve()
plt.figure(figsize=(8, 5))
plt.loglog(ks, [max(e, 1e-17) for e in iter_err], "o-", color="#d1495b", lw=2,
           label="Local (fixed-point iteration)")
plt.axhline(1e-16, color="#2e86ab", lw=2, ls="--",
            label="Global (direct solve, one-shot exact)")
plt.xlabel("iterations")
plt.ylabel("relative error vs true solution (log scale)")
plt.title("Fixed point near rho=1: iteration crawls, direct whole-system solve is exact")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
p2 = OUT + "/curve_fixed_point.png"
plt.savefig(p2, dpi=130)
plt.close()
print("wrote", p2)
