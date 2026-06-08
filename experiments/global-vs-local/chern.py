"""
Chern-number sub-experiment — the panel's pick for the STRONGEST honest
"closed-form invariant of a whole system at once."

Three independent external models (DeepSeek, Mistral, Grok) converged on the
SAME answer: the first Chern number of a 2D Bloch band, computed by the
Fukui-Hatsugai-Suzuki link-variable / plaquette Berry-flux method.

Why it is the strongest honest "global beats local":
  - It is a GLOBAL topological invariant: an integral of Berry curvature over
    the ENTIRE Brillouin-zone torus.
  - It is an INTEGER (quantized) — robust to perturbation, gauge-independent.
  - A LOCAL method (follow the band / Berry connection along any open path,
    or integrate curvature over a patch) provably returns ZERO contribution
    on a contractible patch: the integer is a GLOBAL OBSTRUCTION (the monopole
    charge enclosed by the whole closed manifold). Remove any open set and the
    curvature becomes exact, the integral vanishes — so no finite collection
    of local trajectories recovers C without the complete closed field.

Model used: Qi-Wu-Zhang two-band model
    h(k) = (sin kx, sin ky, m - cos kx - cos ky)
    H(k) = h . sigma
Phase diagram (lower band Chern number, verified vs analytic skyrmion winding):
    |m| > 2      -> C = 0   (trivial)
    0 < m < 2    -> C = +1  (topological)
   -2 < m < 0    -> C = -1  (topological)

Pure numpy. Fully separate from any MIND repo.
"""
import numpy as np

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def qwz_H(kx, ky, m):
    h = np.array([np.sin(kx), np.sin(ky), m - np.cos(kx) - np.cos(ky)])
    return h[0] * sx + h[1] * sy + h[2] * sz


def lower_band_vec(kx, ky, m):
    w, v = np.linalg.eigh(qwz_H(kx, ky, m))
    return v[:, 0]  # lower (occupied) band


def chern_number(m, N=48):
    """Fukui-Hatsugai-Suzuki: gauge-invariant lattice Berry flux summed over
    every plaquette of the discretized BZ torus. Exactly integer."""
    ks = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # precompute occupied eigenvectors on the grid
    u = np.empty((N, N, 2), dtype=complex)
    for i in range(N):
        for j in range(N):
            u[i, j] = lower_band_vec(ks[i], ks[j], m)

    def link(a, b):
        ov = np.vdot(a, b)
        return ov / abs(ov)  # U(1) link variable, unit modulus

    total = 0.0
    for i in range(N):
        for j in range(N):
            ip, jp = (i + 1) % N, (j + 1) % N
            U1 = link(u[i, j], u[ip, j])
            U2 = link(u[ip, j], u[ip, jp])
            U3 = link(u[ip, jp], u[i, jp])
            U4 = link(u[i, jp], u[i, j])
            # plaquette Berry flux in (-pi, pi]
            F = np.angle(U1 * U2 * U3 * U4)
            total += F
    return total / (2 * np.pi)


def local_curvature_on_patch(m, N=48, frac=0.25):
    """LOCAL method: integrate Berry curvature over a CONTRACTIBLE patch only
    (a frac x frac sub-square of the BZ), the way any local/step-wise scheme
    that never closes the manifold would. Returns a continuous, non-integer,
    patch-dependent number — never the topological integer."""
    ks = np.linspace(0, 2 * np.pi, N, endpoint=False)
    npatch = max(2, int(N * frac))
    u = np.empty((npatch + 1, npatch + 1, 2), dtype=complex)
    for i in range(npatch + 1):
        for j in range(npatch + 1):
            u[i, j] = lower_band_vec(ks[i % N], ks[j % N], m)

    def link(a, b):
        ov = np.vdot(a, b)
        return ov / abs(ov)

    total = 0.0
    for i in range(npatch):
        for j in range(npatch):
            U1 = link(u[i, j], u[i + 1, j])
            U2 = link(u[i + 1, j], u[i + 1, j + 1])
            U3 = link(u[i + 1, j + 1], u[i, j + 1])
            U4 = link(u[i, j + 1], u[i, j])
            total += np.angle(U1 * U2 * U3 * U4)
    return total / (2 * np.pi)


if __name__ == "__main__":
    print("CHERN NUMBER  (global topological invariant of the whole BZ torus)")
    print("Qi-Wu-Zhang two-band model, Fukui-Hatsugai-Suzuki lattice method\n")
    print(f"{'mass m':>8}{'expected C':>12}{'global C (whole torus)':>26}"
          f"{'local (contractible patch)':>30}")
    # QWZ lower-band Chern (verified vs analytic skyrmion winding):
    #   |m|>2 -> 0 ;  0<m<2 -> +1 ;  -2<m<0 -> -1
    cases = [(-3.0, 0), (-1.0, -1), (1.0, 1), (3.0, 0), (0.5, 1), (-0.5, -1)]
    for m, expected in cases:
        Cg = chern_number(m)
        Cl = local_curvature_on_patch(m)
        Cg_int = int(round(Cg))
        mark = "OK" if Cg_int == expected else "XX"
        print(f"{m:>8.1f}{expected:>12}{Cg:>17.6f} -> {Cg_int:>2} {mark:<2}"
              f"{Cl:>26.6f}")
    print("\nGlobal Chern number is an EXACT integer from the whole closed torus.")
    print("The local patch integral is a continuous, patch-dependent fraction —")
    print("it can never certify the integer without closing the entire manifold.")
    print("That is the topological obstruction: global beats local, provably.")
