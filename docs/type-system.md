# Type System

The MIND type system models tensor programs with explicit ranks, shapes, and data types while enforcing purity and effect capabilities.

## Goals

- **Predictable performance** – Shapes are statically known, enabling compile-time buffer planning.
- **Expressive generics** – Parametric polymorphism supports reusable operator definitions. *(Goal; the current implementation is a bounded subset: a single type parameter over scalar types.)*
- **Safe effects** – Side effects such as host I/O or stateful ops are opt-in capabilities.

## Primitive Types

| Category       | Examples                                  | Notes                                      |
| -------------- | ------------------------------------------ | ------------------------------------------ |
| Scalars        | `i32`, `f32`, `bool`, `index`              | `index` matches target word size           |
| Tensors        | `Tensor[f32,(2,3)]`, `Tensor[i64,(N,M)]`   | Shapes can be symbolic (compile-time vars) |
| Tuples/Records | `(Tensor[f32,(N)], bool)`                  | Used for multi-value returns               |
| Functions      | `(Tensor[f32,(N)]) -> Tensor[f32,(N)]`     | Signature types only; first-class function values and closures are not yet implemented |

## Composite Types

Phase 10.5 / 10.6 added the following composite type forms to the
surface language. They parse, type-check, and lower to the existing
Core IR v1 shape lattice. Lowering depth varies by form: slice calls
support proven dynamic-array handles with `std-surface`, and generic
instantiation is bounded (single type parameter, scalar element types).

| Form                       | Example                          | Notes                                  |
| -------------------------- | -------------------------------- | -------------------------------------- |
| Reference (type)           | `&T`, `&mut T`                   | Single-value borrow; lifetime inferred |
| Reference (expression)     | `&expr`, `&mut expr`             | Phase 10.7 — symmetric with `&T`/`&mut T` types; no-op in v1 IR |
| Slice                      | `&[T]`, `&mut [T]`               | Borrowed dynamic-array call boundary; see restrictions below |
| Fixed-size array           | `[T; N]`                         | `N` is a compile-time integer literal  |
| Tuple                      | `(T, U)`, `(T, U, V)`            | Used in fn returns and destructuring   |
| Qualified type path        | `module.Type`, `crate.Foo`       | Used in const decls and annotations    |
| Generic type instantiation | `Vec<i32>`, `Result<T, E>`       | Syntax accepted; instantiation currently bounded (single type param, scalar types). No full `Vec`/`String`/`HashMap` containers yet — shipped collections are a region allocator plus an insert-only map |

The visibility qualifier `pub` is accepted on `fn`, `struct`, `enum`,
and struct fields. Its semantic effect on the emitted module ABI is
gated by the `ffi-c-user` Cargo feature (see RFC-0002).

## Slice Calls

With `std-surface`, a compatible `array<T>` value or array literal can be
passed directly to a slice parameter:

```mind
fn sum_pair(xs: &[i64]) -> i64 {
    return xs[0] + xs.get(1)
}

pub fn answer() -> i64 {
    return sum_pair([20, 22])
}
```

Slices use the existing dynamic-array handle layout. Read-only slices allow
indexing, `get`, and length queries; mutable slices additionally allow indexed
assignment and `set`. Ownership operations such as `push` remain unavailable
through a slice. Inferred aliases preserve these capabilities, including
across branches, loop iterations, `break`, and `continue`.

The current implementation requires a proven compatible handle at each call.
Element types are compared after type-alias resolution: with `type Q = i32`,
an `array<Q>` value or literal (or an `array<defs.Q>` naming an alias exported
by `defs`) passes to `&[i32]`, and an `array<i32>` passes to `&[Q]`. A type
from an imported signature or struct field is read in its owner module, so a
`Bank` returned by `defs` is the `defs.Bank` the caller names. A literal element
that names a module constant (`[defs.ONE, ONE + 1]`) has the class of the
constant's value. A literal element whose proven integer width is wider than
the slice's element type — a constant, an element or field read, a call
result, or arithmetic over any of these at a wider declared width — is
refused with `E2032` instead of truncating in the native build; a plain
literal element declares no width of its own, so the rule never fires on one.
A call to a module-private helper in a `module m { ... }` block
follows the same rules as a call to an exported one. The slice or collection
annotation itself must be spelled out: a binding, parameter or result typed
through an alias of a slice or of an `array<T>` (`type SL = &[i32]`,
`type V = array<i32>`) is refused with `E2032`.
Opaque integer handles, maps, incompatible element types, and unproven source
expressions are refused with `E2032`. Range views such as `&xs[a..b]` are not
slice arguments and are refused with `E2032` as well. Explicit slice local
bindings are also unsupported. Floating-point, tensor,
fixed-array, and nested-slice elements are outside the current slice ABI.
Capability erasure, read-only mutation, and borrowed handles escaping through
non-slice parameters or returns are refused with `E2033` before emission.
Slice-containing struct fields are refused with `E2033` as well.
General lifetime and alias-exclusivity analysis remains outside this implementation.

Fields and indexed slots whose declared type is `array<T>`, `map<K, V>`, or
`set<T>` retain that exact owner type on replacement. A mutator status, scalar,
different collection kind, or collection with different type arguments is
refused with `E2034` before emission. Replacing an owner with a compatible
field, function result, or literal remains supported, as do scalar element
writes such as `holder.values[0] = value`.

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

A function that writes to stdout declares `!io` in its signature. Effect polymorphism (planned alongside first-class functions, which are not yet implemented) will allow higher-order functions to thread capabilities without hard-coding them.

## Type Inference

The compiler performs bidirectional inference:

1. Collect constraints from expression contexts.
2. Solve for concrete shapes/types, instantiating generics where needed.
3. Emit constraints into the IR, enabling later passes to reason about buffer layouts and vectorization.

Inference failures surface rich diagnostics with primary spans in the source and secondary notes referencing the conflicting expressions.

## Interop

When interacting with host code via the FFI, types map onto ABI-safe representations described in [`ffi-runtime.md`](ffi-runtime.md).
