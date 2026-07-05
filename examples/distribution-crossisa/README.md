# Cross-ISA determinism: a piecewise-linear density kernel

This example takes a real third-party C++ quant program — a piecewise
distribution/CDF calculator — and shows the one line in it that is **not**
reproducible across CPU architectures, then shows the same computation in MIND
where it **is** byte-identical across substrates with no special flag.

## The finding

Building the C++ `distribution.cpp` with a matched generic recipe
(`g++ -O2 -std=c++23 -static`, no `-march=native`, no fast-math) and running it
on **x86_64** and **native aarch64** produces output that **diverges at the last
bit (ULP level)** — and only in the piecewise-**linear** columns. The
piecewise-constant and discrete columns (plain sums) stay bit-identical.

The divergent line is `dlPDF`'s interpolation:

```cpp
x==b[i] ? r[i] : (r[i-1]*(b[i]-x) + r[i]*(x-b[i-1])) / (b[i]-b[i-1]);
```

## Root cause — unspecified FMA contraction

The numerator has the shape `a*b + c*d`. On aarch64, the compiler **contracts**
it into a fused multiply-add — one rounding. On x86 it emits a separate multiply
and add — two roundings. IEEE-754 leaves this contraction **unspecified**, so the
same source, same flags, same input, rounds differently per ISA.

Rebuilding *both* sides with `-ffp-contract=off` collapses them to byte-identical
— proving the divergence is exactly the FMA contraction, not chaos and not a bug.

## The MIND version — identical by construction

[`distribution_interp_f64.mind`](./distribution_interp_f64.mind) ports that exact
numerator into MIND's strict f64 lowering path. It lowers to
`arith.mulf` / `arith.mulf` / `arith.addf` with **no fmuladd, no fastmath, no
reassociation, fixed source order**. There is no contraction knob to get wrong:
every substrate rounds the two multiplies and the add the same way. Byte-identical
across x86 and aarch64 **with no flag**.

```bash
# from the mind repo root
mind run examples/distribution_interp_f64.mind        # or compile to a .so and call dl_pdf
```

Verified: two independent compiles of the MIND kernel produce byte-identical raw
f64 output (identical SHA-256), and the values match a Python reference exactly —
including the ULP-visible `0.049999999999999996`.

## The point

`deterministic ≠ predictable`. The distribution being modeled can be as noisy as
you like; what MIND removes is the *second* nondeterminism — the one where the
same formula gives different bits on different hardware because the ISA is allowed
to choose whether to fuse. Name that choice, pin it, and the number stops drifting.

## Honest caveat

The MIND file ports the interpolation **kernel** (the arithmetic that actually
causes the split), not the full array/heap plumbing of `distribution.cpp`. The
sorted-boundary `std::lower_bound` lookup is unrolled into a fixed `if/else`
ladder because the executable subset used here has no array-load codegen yet
(marked `// deferred:` in the source). The `a*b + c*d` numerator and the divide —
the part that diverges — are reproduced verbatim.

## Files

| File | What |
|------|------|
| `distribution_interp_f64.mind` | The MIND port of the divergent kernel (strict f64) |
| `distribution.cpp` | The original third-party C++ program |
| `data1.txt` | A sample P&L input for the C++ program |
