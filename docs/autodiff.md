# Automatic Differentiation

MIND provides reverse-mode automatic differentiation (AD) at the IR level, allowing efficient gradient computation across tensor programs.

## Differentiation Strategy

- **Source Annotation** – Users mark differentiable functions with the `@diff` attribute.
- **IR Expansion** – During lowering, the compiler emits paired forward and backward functions in SSA form.
- **Gradient Propagation** – The backward function walks the IR in reverse topological order, accumulating adjoints per value.

This design keeps differentiation orthogonal to the surface syntax while giving optimization passes explicit control of gradient code.

## Tape & Checkpointing

The differentiator captures intermediate values using SSA references, eliminating the need for an explicit runtime tape. For expensive subgraphs, checkpointing directives allow the compiler to recompute values instead of storing them.

## Supported Operations

- Elementwise arithmetic and activation functions
- Linear algebra primitives (matmul, conv2d, reductions)
- Control flow via structured ops (`if`, `loop`) with differentiable branches
- Custom primitives registered via the `Differentiable` trait

## Higher-Order Gradients

Gradients are first-class functions. Nesting `@diff` triggers another AD pass, generating second-order derivatives as long as the underlying ops provide Jacobian-vector product rules.

## Interfacing with Runtimes

The runtime exposes helpers to bind gradient computations to optimizers. Gradients reuse tensor storage via buffer recycling, and results can be exported through the FFI per [`ffi-runtime.md`](ffi-runtime.md).

## Diagnostics

Common AD errors include:

- **Non-differentiable ops** – Reported with actionable notes and suggested alternatives.
- **Stateful ops without `!state`** – The type system flags these before AD runs.
- **Shape mismatches in gradients** – Emitted after IR validation with references to both primal and gradient nodes.
