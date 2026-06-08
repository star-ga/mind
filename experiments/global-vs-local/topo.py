"""
Topological-invariant sub-experiment.

His framing: a "closed-form topological invariant of the whole field" that
predicts something local/step-wise methods cannot.

The legitimate, codeable version: the WINDING NUMBER of a closed loop of a
complex field around 0. It is:
  - a GLOBAL property (a single contour integral over the whole loop),
  - an INTEGER (quantized / robust to perturbation),
  - and it predicts the number of zeros enclosed (argument principle) WITHOUT
    locating any of them.

A LOCAL step-wise method (follow the field value point by point, watch the sign)
cannot decide "how many zeros are enclosed" without resolving every zero. The
global invariant answers it with one integral. THAT is "global beats local"
made concrete and honest.
"""
import numpy as np

def winding_number(f, N=4000):
    """Winding number of f around 0 along the unit circle, via discrete
    sum of phase increments (closed-form-ish: one pass over the global loop)."""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.exp(1j*t)
    w = f(z)
    ang = np.angle(w)
    d = np.diff(np.concatenate([ang, ang[:1]]))
    d = (d + np.pi) % (2*np.pi) - np.pi          # unwrap to (-pi,pi]
    return int(round(d.sum() / (2*np.pi)))

def local_zero_hunt(f, grid=200):
    """Local method: scan a grid, count sign-change cells. Misses zeros that
    fall between grid points and double-counts near clusters — i.e. it cannot
    robustly return the enclosed count without infinite resolution."""
    xs = np.linspace(-1, 1, grid)
    X, Y = np.meshgrid(xs, xs)
    Z = X + 1j*Y
    mask = (np.abs(Z) < 1.0)
    W = f(Z)
    # crude local zero detection: magnitude below a threshold
    found = np.sum((np.abs(W) < 0.02) & mask)
    return int(found)

# fields with a KNOWN number of zeros inside the unit disk
fields = {
    "z^3 (3 zeros at origin)":            (lambda z: z**3, 3),
    "z^5 - 0.3 (5 simple zeros)":         (lambda z: z**5 - 0.3, 5),
    "(z-0.1)(z-0.2)(z+0.3) (3 zeros)":    (lambda z: (z-0.1)*(z-0.2)*(z+0.3), 3),
    "z^2+1 (0 zeros inside disk, roots at +-i on boundary)": (lambda z: z**2 + 1, 0),
}

if __name__ == "__main__":
    print("TOPOLOGICAL INVARIANT  (winding number = enclosed zero count)\n")
    print(f"{'field':<52}{'true':>5}{'global(W)':>11}{'local(grid)':>13}")
    for name,(f,true) in fields.items():
        W = winding_number(f)
        loc = local_zero_hunt(f)
        gmark = "OK" if W==true else "X"
        print(f"{name:<52}{true:>5}{W:>9} {gmark:<2}{loc:>11}")
    print("\nGlobal winding number returns the exact enclosed-zero count from one")
    print("contour pass. The local grid scan's count depends entirely on resolution")
    print("and threshold — it cannot certify the integer without resolving every zero.")
