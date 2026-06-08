"""
Visualize the Chern-number result: Berry curvature over the whole Brillouin
zone, with the global integer C annotated. Shows topological (C=+-1) vs
trivial (C=0) phases side by side.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chern import lower_band_vec, chern_number

OUT = __file__.rsplit("/", 1)[0]


def berry_curvature_grid(m, N=64):
    ks = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u = np.empty((N, N, 2), dtype=complex)
    for i in range(N):
        for j in range(N):
            u[i, j] = lower_band_vec(ks[i], ks[j], m)

    def link(a, b):
        ov = np.vdot(a, b)
        return ov / abs(ov)

    F = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ip, jp = (i + 1) % N, (j + 1) % N
            U1 = link(u[i, j], u[ip, j])
            U2 = link(u[ip, j], u[ip, jp])
            U3 = link(u[ip, jp], u[i, jp])
            U4 = link(u[i, jp], u[i, j])
            F[i, j] = np.angle(U1 * U2 * U3 * U4)
    return ks, F


phases = [(-1.0, "topological"), (3.0, "trivial"), (0.5, "topological")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (m, label) in zip(axes, phases):
    ks, F = berry_curvature_grid(m)
    C = int(round(chern_number(m)))
    im = ax.imshow(F.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                   cmap="RdBu_r", aspect="auto")
    ax.set_title(f"m={m}  ({label})\nglobal Chern C = {C}", fontsize=12)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Berry flux / plaquette")
fig.suptitle("Berry curvature over the whole Brillouin zone — the integral is the "
             "exact integer C.  Any local patch sees only a fraction.", fontsize=12)
plt.tight_layout()
p = OUT + "/curve_chern.png"
plt.savefig(p, dpi=120)
plt.close()
print("wrote", p)
