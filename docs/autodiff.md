<!-- Copyright 2025 STARGA Inc. -->

# Static autodiff (public)

MIND performs static automatic differentiation on the public IR. The autodiff
pipeline is entirely compile-time: it builds a *gradient IR* that mirrors the
primal computation. Execution remains delegated to the public interpreters or
to external runtimes that implement `MindRuntime` in private repositories.

Key properties:

- Deterministic: the gradient IR is built with ordered data structures so the
  same input IR always produces the same gradient IR.
- Separation of concerns: only the public IR is touched; no private runtime
  hooks are referenced.
- IR-level: derivative rules operate on IR instructions (add, sub, mul,
  matmul, dot, transpose, reductions, etc.).

## API surface

Enable the `autodiff` feature and call the entry point:

```rust
use mind::differentiate_function;

let gradients = differentiate_function(&ir, "main")?;
println!("{}", gradients);
```

The result bundles a full gradient IR module plus a deterministic mapping from
primal value IDs to their gradients. Gradients are emitted in canonical form
and, by default, the autodiff engine runs IR verification on both the primal
and gradient modules.

## Supported ops (Phase 1)

- Binary ops: add, sub, mul (div errors explicitly)
- Matrix ops: dot, matmul (naïve transpose-based rules)
- Shape-preserving ops: reshape, expand/squeeze dims, slice/index/gather (see
  [`docs/shapes.md`](./shapes.md) for the exact dimension rules)
- Reduction ops: mean (explicit axes) and sum (passthrough)

Unsupported or ambiguous cases return `AutodiffError` with structured messages
so callers can fall back or report the limitation directly to users.

## Testing determinism

Integration tests under `tests/autodiff.rs` build small IR graphs, differentiate
them twice, and assert the gradient IR and mappings are identical.
