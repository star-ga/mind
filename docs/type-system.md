# Type System

The MIND type system models tensor programs with explicit ranks, shapes, and data types while enforcing purity and effect capabilities.

## Goals

- **Predictable performance** – Shapes are statically known, enabling compile-time buffer planning.
- **Expressive generics** – Parametric polymorphism supports reusable operator definitions.
- **Safe effects** – Side effects such as host I/O or stateful ops are opt-in capabilities.

## Primitive Types

| Category       | Examples                                  | Notes                                      |
| -------------- | ------------------------------------------ | ------------------------------------------ |
| Scalars        | `i32`, `f32`, `bool`, `index`              | `index` matches target word size           |
| Tensors        | `Tensor[f32,(2,3)]`, `Tensor[i64,(N,M)]`   | Shapes can be symbolic (compile-time vars) |
| Tuples/Records | `(Tensor[f32,(N)], bool)`                  | Used for multi-value returns               |
| Functions      | `(Tensor[f32,(N)]) -> Tensor[f32,(N)]`     | Pure by default unless effects declared    |

## Shape Variables

Shapes use uppercase identifiers (`N`, `M`, `B`) scoped to the function signature. Constraints propagate through expressions via unification.

```
fn matmul(a: Tensor[f32,(B,N)], b: Tensor[f32,(N,M)]) -> Tensor[f32,(B,M)]
```

The solver ensures output shapes are consistent or emits diagnostic errors pointing to the mismatched dimensions.

## Effects

Effects model capabilities such as:

- `io` – Host input/output (e.g., printing)
- `state` – Mutation of global or captured state
- `ffi` – Crossing the FFI boundary

A function that writes to stdout declares `!io` in its signature. Effect polymorphism allows higher-order functions to thread capabilities without hard-coding them.

## Type Inference

The compiler performs bidirectional inference:

1. Collect constraints from expression contexts.
2. Solve for concrete shapes/types, instantiating generics where needed.
3. Emit constraints into the IR, enabling later passes to reason about buffer layouts and vectorization.

Inference failures surface rich diagnostics with primary spans in the source and secondary notes referencing the conflicting expressions.

## Interop

When interacting with host code via the FFI, types map onto ABI-safe representations described in [`ffi-runtime.md`](ffi-runtime.md).
