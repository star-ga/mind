# Finite arithmetic circuits — Pure MIND example

The `permanent_2x2.mind` program encodes the 2×2 permanent
`perm([[a,b],[c,d]]) = a*d+b*c` in a tiny arithmetic DAG.
A circuit includes 4 input gates, 2 multiplication gates and 1 addition
gate (7 gates total); the gate-count convention includes input gates.

It **actually evaluates an encoded DAG** using exact signed i64 arithmetic
with restricted inputs, output bounds, strict 1-based backward references,
and fail-closed opcode checks. The independent reference computes `ad+bc`
without consulting the DAG. Positive tests check three input assignments, with independently pinned outputs 10, 57 and 0.
Negative checks: incorrect but structurally valid multiplication edge, forward reference, unknown operation, out-of-bound
input, overflowing output bound, and out-of-range node count.

This is a bounded example of `Statement.Computes`-style arithmetic gates
from Anthropic's 2026 `vp-vnp/Statement.lean`. It does **not** formalize
the permanent for arbitrary dimension, polynomial families, asymptotic
circuit lower bounds, or VP versus VNP. External statements must undergo
independent challenge and axiom audits.

Sources:
- https://github.com/anthropics/formal-math/tree/main/vp-vnp
- https://github.com/openai/math/blob/main/lean/docs/379.md

## Native local replay

```sh
MINDC=/absolute/path/to/mindc bash examples/arithmetic_circuits/replay.sh
```

The shell runner orchestrates the compiler and exact output comparison;
**all circuit calculations and rejection logic are Pure MIND**.
Until a real released `mindc` run completes, `NATIVE_EXECUTED=NO`.
