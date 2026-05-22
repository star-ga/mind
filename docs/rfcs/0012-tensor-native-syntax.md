# RFC 0012: Tensor-Native Surface Syntax — the Differentiation Layer

| Field | Value |
|---|---|
| RFC | 0012 |
| Title | Tensor-native surface syntax — the differentiation layer |
| Status | **Phase B Shipped** |
| Authors | STARGA Inc. |
| Created | 2026-05-22 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0005 (pure-MIND std surface), RFC 0006 (mind-blas), RFC 0007 (Mindcraft), RFC 0010 (memory safety + C ABI) |

---

## 1. Motivation

MIND today has a Rust-family surface — `fn`, `let`, `struct`, `match`, `pub`,
`#[repr(C)]` — and a differentiated substrate: Q16.16 byte-identical
determinism, MLIR-native codegen, a three-tier memory model (RFC 0010), and a
native BLAS surface with cross-arch bit-identity guarantees (RFC 0006 §5.2).

The surface does not express what the substrate is for.

A program that performs a matrix–vector product currently reads:

```mind
// Current: six raw i64 arguments, manual address and dimension threading
let rc: i64 = matmul_rmajor_f32(w_addr, x_addr, y_addr, rows, cols);
```

The call site carries no information about the shapes of `w`, `x`, or `y`.
A dimension mismatch — `rows` and `cols` transposed, a wrong `len` — is a
silent runtime corruption or a segmentation fault. The compiler sees an opaque
function call with integer arguments; it cannot verify, fuse, or place the
operation. The developer must thread six values through every call site and
remember which is which.

The proposed form for the same operation is:

```mind
// Proposed: shape-typed, compiler-visible, determinism-annotated
#[deterministic]
fn encode(W: Tensor<f32, [M, K]>, x: Tensor<f32, [K]>) -> Tensor<f32, [M]> {
    W @ x
}
```

The shape mismatch is a compile error. The compiler owns the operation. The
`#[deterministic]` annotation is a checked property of the function, not a
discipline the author has to remember. The body is one expression.

This RFC specifies the tensor-native surface. Four properties motivate it.

### 1.1 Terseness

The current `matmul_rmajor_f32(w_addr, x_addr, y_addr, rows, cols)` form
requires the caller to:

1. Extract the buffer address from each `Vec` or heap allocation.
2. Extract or compute the integer dimensions.
3. Allocate the output buffer and pass its address.
4. Pass all six values in the correct order without static verification.

The proposed `W @ x` form eliminates all four obligations. The type carries the
shape; the compiler derives the output shape; the allocation is managed by the
type system. The reduction in authoring overhead is not stylistic — it directly
reduces the class of errors the H-01 / #233 incident class represents (shape
arguments silently transposed, output buffer undersized, dimension computed from
a stale binding).

### 1.2 Compile-time shape safety

Shape mismatches that are today runtime crashes (or silent output corruption)
become compile errors under this RFC. The error fires at the call site with a
named diagnostic (`shape::matmul_mismatch`, `shape::broadcast_incompatible`),
citing the conflicting dimensions and their source locations.

This is not a new safety layer — it is the existing determinism contract made
syntactic. The substrate's whole value is "errors caught before execution." The
surface should enforce that at the shape level, not only at the arithmetic level.

### 1.3 Determinism in syntax

The `#[deterministic]` annotation makes the byte-identity guarantee a checked
property of the source. A `#[deterministic]` function that calls a
non-deterministic operation is a compile error (`determinism::nondeterministic_in_deterministic`).

Today, achieving byte-identical execution requires calling the correct
library function — `dot_q16` rather than `dot_f32`, with the correct
sub-backend selected. That is a discipline the compiler cannot enforce. After
this RFC, the annotation elevates the contract into the type system: the compiler
verifies it.

### 1.4 Compiler-fusable first-class operations

Opaque function calls cannot be fused. The expression `W @ x .+ b` is two
separate calls under the current surface; the compiler has no information that
they operate on adjacent memory or that their outputs are consumed immediately.

First-class tensor operations give the compiler the information it needs to fuse
chained operations into a single kernel pass — fewer memory round-trips, fewer
intermediate allocations. The fusion is only possible because the operations are
syntax the compiler analyses, not opaque foreign-function calls.

### 1.5 This RFC is additive

Every existing `.mind` file compiles unchanged. The Rust-family base —
`fn`, `struct`, `match`, `let`, `pub` — is the permanent general-purpose layer
and the lowering target for tensor syntax. Tensor syntax lowers to the same
MLIR the function-call forms already emit (RFC 0006), so the bootstrap
fixed-point and the cross-arch Q16.16 bit-identity gate (task #57) are
unaffected. The function-call forms remain the explicit escape hatch. Both
coexist permanently.

---

## 2. Non-goals

**Not removing the Rust-family base.** `fn`, `struct`, `match`, and `let` are
the general-purpose layer and the lowering substrate. Tensor syntax is sugar
over them. They stay.

**Not a dynamic-shape model.** Shapes are static and checked at compile time.
Dynamic shapes would defeat the determinism-in-syntax argument and eliminate the
compile-time shape-error story. Shape inference defers to the open questions in
§10; the requirement is that shapes are resolved at compile time.

**Not auto-differentiation syntax.** Reverse-mode automatic differentiation
is a distinct feature with its own compiler requirements. This RFC adds
tensor-algebra operators, not differentiation primitives. A future RFC may add
`#[autodiff]` as a sibling annotation to `#[deterministic]`; that is out of
scope here.

**Not replacing std.blas.** Tensor operators lower to the std.blas function-call
surface (RFC 0006). The `@` operator lowers to `matmul_rmajor_f32` or
`matmul_rmajor_q16_v` depending on the tensor dtype. std.blas is the permanent
lowering target; RFC 0012 is the higher-level surface over it.

**Not a GPU kernel authoring DSL.** The `#[target(...)]` placement annotation
(§5) selects which backend lowers the operation, not how the kernel is written.
Per-thread memory, warp-synchronous programming, and shared-memory management
are not expressed in this RFC.

**Not introducing pervasive lifetime annotations.** Tensor allocations are
managed by the existing three-tier memory model (RFC 0010): stack-tier for
temporary results, region-interior for batch computations, `GenRef<T>` for
long-lived tensors. No new annotation form is required at call sites.

---

## 3. Shape-typed tensors

### 3.1 Syntax

A tensor type names a scalar dtype and a static shape:

```mind
Tensor<f32, [M, N]>       // 2-D f32 tensor, M rows, N columns
Tensor<q16, [768]>         // 1-D Q16.16 vector, 768 elements
Tensor<f64, [Batch, 512]>  // 2-D f64 tensor, symbolic batch dimension
Tensor<i32, [3, 3, 3]>    // rank-3 i32 tensor
Tensor<f32, []>            // rank-0 scalar (a single f32 value)
```

The general form is `Tensor<dtype, [dim_0, dim_1, ..., dim_N]>`. The shape is
a bracket-enclosed comma-separated list of dimension expressions. The bracket
list may be empty (rank-0). The dtype and shape together form the type; two
tensors with the same dtype and shape are the same type.

Layout defaults to row-major. Column-major tensors use a third argument:

```mind
Tensor<f32, [M, N], col>   // column-major 2-D f32
```

The layout tag is part of the type. Passing a column-major tensor where a
row-major tensor is expected is a compile-time error
(`shape::layout_mismatch`), not a silent transposition.

### 3.2 Dtype set

| Dtype keyword | Representation | Notes |
|---|---|---|
| `f32` | IEEE-754 single precision | Standard float; non-deterministic reduction order unless `#[deterministic]` constrains it |
| `f64` | IEEE-754 double precision | Same non-determinism caveat; used as oracle reference |
| `i32` | 32-bit signed integer | Integer operations are deterministic by default |
| `i64` | 64-bit signed integer | Integer operations are deterministic by default |
| `q16` | Q16.16 fixed-point | First-class dtype; byte-identical determinism across backends; the substrate-native type |

`q16` is not a library convention in RFC 0012 — it is a keyword dtype on par
with `f32`. A `Tensor<q16, [N]>` carries the byte-identity contract of the
Q16.16 substrate (task #57) without the caller knowing which intrinsic to call.
The compiler selects the correct `dot_q16_v` / `matmul_rmajor_q16_v` path
automatically.

The dtype keyword is lowercase. `Q16_16` is not accepted; `q16` is the
canonical form.

### 3.3 Symbolic dimensions

Dimension expressions are either integer literals or symbolic names:

```mind
Tensor<f32, [128, 64]>     // concrete dimensions
Tensor<f32, [N, 64]>       // N is a symbolic dim resolved by the caller
Tensor<f32, [Batch, D]>    // two symbolic dims
```

Symbolic dimension names are resolved through unification: when `Tensor<f32, [N, K]>`
is passed to a function that expects `Tensor<f32, [N, K]>` and separately
passed as the first argument of a `@` where the right operand is
`Tensor<f32, [K, M]>`, the compiler unifies `N`, `K`, and `M` from the
context. A failed unification — e.g. multiplying a `[N, K]` matrix by a
`[J, M]` matrix where `K ≠ J` — is a compile error naming both dimension
expressions and their values.

Phase A (§7) specifies the scope of unification at launch: symbolic dims unify
within a function body. Cross-function unification and const-generic arithmetic
over dims (`Tensor<f32, [N, N*2]>`) are deferred to Phase A increments after
the basic unification proves stable.

### 3.4 Relationship to the existing i64-address ABI

The underlying storage model is unchanged. A `Tensor<f32, [M, N]>` is a
compile-time view over an i64-address heap record (RFC 0005 Option C ABI): the
runtime value is a base address, a length, and a layout descriptor. The tensor
type is the compiler's lens; the heap record is the runtime representation.

The compiler emits the same load/store sequences into MLIR for tensor operations
that the function-call forms emit. The transformation is entirely in the
front-end. There is no new runtime data structure, no fat-pointer, no RTTI.

A `Tensor<f32, [M, N]>` occupies `M * N * 4` bytes in the backing allocation.
The backing allocation follows the RFC 0010 three-tier model: a tensor whose
lifetime is lexically bounded is region-interior; a tensor that outlives its
enclosing scope is `GenRef<Tensor<...>>`.

### 3.5 Decision: `Tensor<dtype, [dims]>` not shape-prefix or type-suffix form

Three syntactic options were considered:

- `Tensor<f32, [M, N]>` — explicit type-constructor form; parses cleanly in
  the existing generic-bracket grammar; dtype is prominent.
- `[M, N]f32` — shape-prefix form; terse but reads as "array indexed by M and N
  containing f32," which conflicts with MIND's existing bracket-literal grammar.
- `f32[M, N]` — type-suffix form; familiar from C-style array declarations;
  conflates a scalar type and an indexing expression in a way that is ambiguous
  in the MIND grammar when M is a runtime expression.

**Decision: `Tensor<dtype, [dims]>`.**

The explicit generic form parses cleanly against the existing `StructName<...>`
grammar (RFC 0005 Phase 3). The dtype is the first argument and is always
present, which keeps the runtime representation obvious. The shape bracket is
syntactically distinct from both an expression and a type, making disambiguation
unambiguous. No grammar changes are required beyond registering `Tensor` as a
built-in type constructor.

---

## 4. Tensor operators

### 4.1 Matmul: `@`

The binary `@` operator performs matrix multiplication:

```mind
let C: Tensor<f32, [M, N]> = A @ B;
// A: Tensor<f32, [M, K]>, B: Tensor<f32, [K, N]>
```

The shared inner dimension `K` must unify. A mismatch is
`shape::matmul_mismatch`, citing both operand types and the conflicting
dimension values.

Matrix–vector and vector–matrix forms:

```mind
let y: Tensor<f32, [M]>  = W @ x;   // W: Tensor<f32,[M,K]>, x: Tensor<f32,[K]>
let z: Tensor<f32, [N]>  = v @ A;   // v: Tensor<f32,[M]>,   A: Tensor<f32,[M,N]>
```

The `@` operator lowers to:

| Operand shapes | std.blas target |
|---|---|
| `[M,K] @ [K,N]` | `matmul_rmajor_f32` / `matmul_rmajor_f32_v` (f32) |
| `[M,K] @ [K,N]` | `matmul_rmajor_q16_v` (q16) |
| `[M,K] @ [K]` | `matmul_rmajor_f32_v` with N=1 (f32) |
| `[M] @ [M,N]` | transposed matmul, implemented as `(A.T @ v.T).T` |

Every lowering is byte-identity-gated against the equivalent hand-written
std.blas call: for a given `(M, K, N)` and dtype, `A @ B` must produce MLIR
byte-identical to the explicit `matmul_rmajor_f32(a_addr, b_addr, out_addr, M, K)` call.
This gate is enforced in the Phase B test suite (§7).

**Disambiguation: `@` as annotation sigil vs matmul operator.** MIND currently
uses `#[...]` for attributes (`#[test]`, `#[allow(...)]`, `#[repr(C)]`). This
RFC adopts `#[...]` as the canonical annotation form for placement and
determinism directives. The `@` character is thereby available as the binary
infix matmul operator without collision. The grammar distinguishes them by
position: `#[` begins an attribute; `@` in an expression context is the infix
operator. No ambiguity arises. This decision is final; see §11 for the detailed
resolution.

### 4.2 Elementwise operators

Elementwise operators use the dot-prefix convention to distinguish them from
scalar operators:

```mind
let C: Tensor<f32, [M, N]> = A .+ B;   // elementwise add
let D: Tensor<f32, [M, N]> = A .- B;   // elementwise subtract
let E: Tensor<f32, [M, N]> = A .* B;   // elementwise multiply (Hadamard)
let F: Tensor<f32, [M, N]> = A ./ B;   // elementwise divide
```

Scalar broadcast: when one operand is rank-0 (`Tensor<f32, []>`) or a scalar
`f32`/`q16` literal, it is broadcast to the shape of the other operand:

```mind
let G: Tensor<f32, [M, N]> = A .* 2.0;   // broadcast scalar
let H: Tensor<f32, [M, N]> = A .+ bias;  // bias: Tensor<f32, [N]> — broadcast along rows
```

Broadcasting rules follow the standard row-major prefix-alignment convention:
shapes are aligned at their trailing dimension and expanded leftward. A shape
mismatch where no broadcast interpretation is valid is
`shape::broadcast_incompatible`, citing the two operand shapes.

Broadcasting is resolved and verified at compile time. There is no runtime
shape check; if the expression type-checks, the broadcast is valid.

### 4.3 Transpose

The `.T` postfix operator transposes the last two dimensions:

```mind
let AT: Tensor<f32, [N, M]> = A.T;   // A: Tensor<f32, [M, N]>
```

For rank-1 tensors, `.T` is a no-op (a vector has no second dimension to swap):

```mind
let v: Tensor<f32, [N]> = u.T;       // identical to u
```

`.T` on a rank-0 tensor is a compile error (`shape::transpose_of_scalar`).

`.T` is a compile-time type transformation. It does not copy memory; the lowered
MLIR uses a stride adjustment. The output tensor views the same backing
allocation with swapped strides.

### 4.4 Reshape

The `.reshape([dim_0, ..., dim_N])` method produces a view of the tensor data
with a different shape. The total element count must be invariant:

```mind
let flat: Tensor<f32, [M * N]> = matrix.reshape([M * N]);
let square: Tensor<f32, [8, 8]> = flat.reshape([8, 8]);
```

The total-size invariant (`M * N == 8 * 8`) is checked at compile time when
all dimensions are concrete. When symbolic dimensions are involved, the compiler
emits a `shape::reshape_size_mismatch` diagnostic if the constraint is provably
violated; unresolved symbolic cases defer to Phase A's unification scope.

`.reshape` does not copy memory. It produces a new `Tensor` value pointing to
the same backing allocation with updated stride metadata.

### 4.5 Reductions

Reduction operations take an axis argument:

```mind
let s: Tensor<f32, [M]>    = A.sum(axis: 1);    // A: Tensor<f32,[M,N]>; sum along cols
let m: Tensor<f32, [N]>    = A.mean(axis: 0);   // mean along rows
let mx: Tensor<f32, [M]>   = A.max(axis: 1);    // max along cols
let mn: Tensor<f32, [M]>   = A.min(axis: 1);    // min along cols
```

Reducing a rank-1 tensor with `axis: 0` produces a rank-0 scalar:

```mind
let total: Tensor<f32, []> = v.sum(axis: 0);    // v: Tensor<f32, [N]>
```

A full reduction (all axes) is `.sum()` / `.mean()` / `.max()` / `.min()`
without an argument:

```mind
let grand_total: Tensor<f32, []> = A.sum();
```

Reductions lower to the std.blas dot / L1 / L∞ surfaces or to scalar loops
depending on the reduction operation and dtype. `.sum()` on a `q16` tensor
lowers to `dot_q16_v` (via a unit vector), preserving the byte-identity
contract. `.max()` lowers to `dot_linf_f32_v` for f32.

### 4.6 Norm shorthands

Because L1, L2, and L∞ norms correspond directly to the std.blas surface, they
are first-class methods:

```mind
let l2:   f32 = v.norm();        // L2 (Euclidean) — lowers to dot_f32_v then sqrt
let l1:   f32 = v.norm_l1();     // L1 (Manhattan) — lowers to dot_l1_f32_v; no sqrt
let linf: f32 = v.norm_linf();   // L∞ (Chebyshev) — lowers to dot_linf_f32_v; no sqrt
```

`.norm_l1()` and `.norm_linf()` are sqrt-free by definition. On fixed-point
substrates and photonic targets where the square-root is not native, `.norm_l1()`
is the preferred default; the compiler emits a `shape::l2_on_q16_target` info
diagnostic when `.norm()` (L2) is called on a `q16` tensor under a
`#[target(q16)]` or `#[target(cerebras)]` context.

### 4.7 Operator precedence

Tensor operators are added to the expression grammar with the following
precedence, from tightest to loosest:

| Level | Operators | Associativity |
|---|---|---|
| Postfix | `.T`, `.reshape(...)`, `.sum(...)`, `.norm()`, ... | left |
| Matmul | `@` | left |
| Elementwise mul/div | `.*`, `./` | left |
| Elementwise add/sub | `.+`, `.-` | left |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` | left |
| Logical | `&&`, `\|\|` | left |

Scalar operators (`+`, `-`, `*`, `/`) are unchanged and interact with tensor
operators only through broadcast when one operand is a scalar.

### 4.8 Composition example

A single-layer linear transform with bias and activation:

```mind
#[deterministic]
fn linear(W: Tensor<q16, [Out, In]>, b: Tensor<q16, [Out]>, x: Tensor<q16, [In]>)
    -> Tensor<q16, [Out]>
{
    (W @ x .+ b).relu()
}
```

Under the current surface, the equivalent requires 8 explicit steps: allocate
output buffer, call matmul with 5 raw arguments, extract the buffer address for
the bias, call an elementwise-add intrinsic, and apply activation in a separate
loop. The proposed form is three expressions with no address bookkeeping and
compile-time shape verification at every step.

---

## 5. Determinism and placement annotations

### 5.1 `#[deterministic]`

A function marked `#[deterministic]` is subject to the following compile-time
constraints, enforced by the mindc type-checker:

1. Every tensor operation in the function body must lower through a
   deterministic path. For `q16` tensors this is unconditional — Q16.16
   arithmetic is byte-identical across backends. For `f32` tensors, the
   compiler requires a fixed reduction order (the MLIR `arith.addf` sequential
   form, not the tree-reduced `vector.reduction <add>`) unless the function is
   also annotated `#[target(q16)]`, in which case the `q16` path is used
   throughout.

2. A call from a `#[deterministic]` function to a function that is not itself
   `#[deterministic]` is `determinism::nondeterministic_in_deterministic`
   (default severity: `error`). The constraint is transitive: the entire call
   graph reachable from a `#[deterministic]` entry point must be deterministic.

3. Standard library functions in std.blas that operate on `q16` tensors are
   implicitly `#[deterministic]` — the byte-identity gate (task #57) is their
   defining property. std.blas f32 functions are not implicitly deterministic
   (AVX2 tree reduction reorders addition); they require an explicit
   `#[deterministic]` annotation on the calling function combined with a
   target that fixes the reduction order.

`#[deterministic]` functions participate in the evidence-chain reproducibility
guarantee: a binary compiled from `#[deterministic]` source with the same
compiler version and target triple produces byte-identical output for identical
inputs on every machine, every run, forever.

```mind
#[deterministic]
fn score(query: Tensor<q16, [D]>, catalog_row: Tensor<q16, [D]>) -> q16 {
    query .* catalog_row .norm_l1()
}
// Compile error if score calls any non-deterministic function.
// Compile error if the q16 path is not available on the current #[target].
```

### 5.2 `#[target(...)]`

A `#[target(...)]` annotation on a function or a tensor-op expression selects
which MLIR backend dialect the operation lowers to:

```mind
#[target(cpu)]
fn cpu_path(A: Tensor<f32, [M, K]>, B: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    A @ B
}

#[target(cerebras)]
fn cerebras_path(A: Tensor<q16, [M, K]>, B: Tensor<q16, [K, N]>) -> Tensor<q16, [M, N]> {
    A @ B
}
```

The `#[target(...)]` vocabulary is the same backend-target set already defined
in `Mind.toml [build].target` and the existing `parse_target` function in the
mindc CLI (`cpu | gpu | cerebras | tpu | npu | lpu | dpu | fpga`). No new
targets are introduced by this RFC; the annotation surface exposes the existing
backend targets at the language level rather than only at the manifest level.

When `#[target(cerebras)]` is used but the build is invoked without a Cerebras
backend present, the compiler behaviour is governed by the `target_mismatch`
setting in `Mind.toml [mindcraft]`:

- `target_mismatch = "error"` (default) — `shape::target_unavailable` compile
  error.
- `target_mismatch = "warn"` — warning emitted; compiler falls back to `cpu`.
- `target_mismatch = "off"` — silent cpu fallback.

The default is `"error"`. A `#[target(cerebras)]` annotation without a Cerebras
backend is almost always a programming error, not an intentional fallback.

### 5.3 `#[q16]`

`#[q16]` is syntactic shorthand for `#[deterministic] #[target(q16)]`:

```mind
#[q16]
fn fixed_point_dot(a: Tensor<q16, [N]>, b: Tensor<q16, [N]>) -> q16 {
    a .* b .sum()
}
```

A `#[q16]` function is `#[deterministic]` by construction. Every tensor
operation in its body must operate on `q16` tensors or be explicitly cast to
`q16`. A non-`q16` tensor operation in a `#[q16]` function is
`determinism::float_in_q16_fn` (error, not configurable).

### 5.4 Annotation interaction with the type system

Annotations are item-level by default: `#[deterministic]` on a function applies
to the entire function body. An expression-level annotation overrides the
item-level default for one sub-expression:

```mind
#[deterministic]
fn mixed(W: Tensor<q16, [M, K]>, x: Tensor<q16, [K]>) -> Tensor<q16, [M]> {
    // The entire function is #[deterministic].
    // This sub-expression is #[target(cerebras)] for the matmul only:
    #[target(cerebras)](W @ x)
}
```

Expression-level `#[target(...)]` is permitted only when the item-level
annotation permits it — a `#[deterministic]` function may specify
`#[target(cerebras)]` on an expression because the Cerebras target is
deterministic. Specifying `#[target(gpu)]` on a sub-expression inside a
`#[deterministic]` function is `determinism::nondeterministic_in_deterministic`
(GPU floating-point reduction order is not fixed).

---

## 6. Why tensor-native syntax is more efficient

This section states the four efficiency claims precisely and justifiably. These
are not aspirational claims; they describe properties the design guarantees.

### 6.1 Authoring efficiency

**Before (current surface):**

```mind
// Encode a single query vector against a catalog row.
// Manual obligations:
//   1. Extract buffer addresses from Vec values.
//   2. Compute and pass M, K dimensions explicitly.
//   3. Allocate output buffer.
//   4. Pass six arguments in the correct order with no static verification.
let w_addr: i64  = vec_addr(W_vec);
let x_addr: i64  = vec_addr(x_vec);
let y_addr: i64  = __mind_alloc(M * 8);
let rc: i64      = matmul_rmajor_f32(w_addr, x_addr, y_addr, M, K);
```

**After (proposed surface):**

```mind
// The type carries the shape. The compiler derives the output shape.
// The allocation is implicit. No address arithmetic. One expression.
let y: Tensor<f32, [M]> = W @ x;
```

The reduction: 5 statements to 1, 6 raw arguments to 0, 2 manual address
extractions to 0, 1 manual output allocation to 0. The error class — transposed
dimensions, wrong length, uninitialized output buffer — is structurally
eliminated, not caught at runtime.

### 6.2 Compile-time shape safety

Shape mismatches that are today silent runtime corruptions become compile-time
errors. The before/after for a transposed-dimension bug:

**Before:** `matmul_rmajor_f32(w_addr, x_addr, y_addr, K, M)` — `K` and `M`
are swapped. The call succeeds silently, writing `K` rows of garbage into a
buffer allocated for `K` values. The error manifests as incorrect scores
downstream, if at all.

**After:** `W @ x` where `W: Tensor<f32, [M, K]>` and `x: Tensor<f32, [M]>` —
the compiler unifies the inner dimension and emits `shape::matmul_mismatch`:
`expected inner dim K, found K on right operand but M on left operand's second
dimension`. The error fires at the expression, not at a downstream
score-mismatch.

### 6.3 Determinism in syntax

The guarantee that a function is byte-identical across machines was previously a
discipline: the author chose the `dot_q16` function (not `dot_f32`), knew that
Q16.16 was byte-identical (task #57), and maintained that discipline at every
call site. Reviewing a function for determinism required tracing every call in
the body and checking which std.blas variant was chosen.

After this RFC, the guarantee is syntactic: `#[deterministic]` on a function is
a compiler-checked invariant. The review question changes from "did the author
call the right function at every site?" to "does the function have the
`#[deterministic]` annotation and does it compile?" The annotation is the
moat — the governance-substrate property made syntactic.

### 6.4 Compiler fusion

First-class tensor operators give the compiler the semantic information
necessary to fuse chained operations. Consider the common pattern in the
mind-nerve encode path:

```mind
// Today: two separate BLAS calls, two intermediate allocations, two kernel launches
let hidden: i64      = __mind_alloc(Hidden * 8);
let rc1: i64         = matmul_rmajor_f32(W1_addr, x_addr, hidden, M, K);
let output: i64      = __mind_alloc(Out * 8);
let rc2: i64         = matmul_rmajor_f32(W2_addr, hidden, output, Out, M);

// Proposed: one fused expression — the compiler may emit as a single kernel
let y: Tensor<f32, [Out]> = W2 @ (W1 @ x);
```

Because `W1 @ x` is a compiler-visible sub-expression consumed immediately by
`W2 @`, the compiler may fuse both matmuls into a single kernel pass with no
intermediate allocation written to main memory. The fusion is correct by
construction — the compiler has the shape and dtype information to verify that
the intermediate tensor is consumed in full before being discarded.

Opaque function calls cannot be fused. `matmul_rmajor_f32(...)` is an extern
call; the compiler has no information about its semantics or its output's
consumer. First-class syntax is the necessary condition for compiler-driven
fusion. This is the actual runtime efficiency payoff: fewer memory round-trips
on memory-bound workloads, which the V24 paper §9 roofline (RFC 0006 §9.3b)
establishes as the binding bottleneck on commodity-GPU substrates.

---

## 7. Phasing

Each phase is independently shippable. Each phase ends with a byte-identity
gate: the tensor syntax emitted at that phase must produce MLIR that is
byte-identical to the equivalent hand-written std.blas call for the same
operand shapes, dtypes, and values.

| Phase | Deliverable | Byte-identity gate |
|---|---|---|
| A ✓ | Shape-typed `Tensor<dtype, [dims]>` in the type system. Symbolic dimension unification within a function body. Compile-time shape checking with `shape::*` diagnostics. No tensor operators yet — just the type and shape. `q16`, `i64`, `f64` first-class dtypes added. | The type is compile-time-only; no new MLIR is emitted. Gate: existing test suite unchanged; 19 Phase A shape-check tests pass. Bootstrap oracle byte-identical. blas_smoke 12/12. Bench: small_matmul 2.92µs (-3.4% p=0.50), medium_mlp 6.62µs (-6.2% p=0.00), large_network 16.95µs (-11.0% p=0.00). All within +7% cap. |
| B ✓ | Tensor operators `@`, `.+`, `.-`, `.*`, `./`. Each desugars at a single point (`lower_expr`) to existing IR forms: `@` → `Instr::MatMul` (same as `tensor.matmul`), `.+`/`.-`/`.*`/`./` → `Instr::BinOp` (same as scalar `+`/`-`/`*`/`/`). Shape inference: `shape::matmul_mismatch` and `shape::broadcast_mismatch` diagnostics. **B.2 deferred**: `.T`, reductions (`.sum`/`.mean`/`.max`), norm shorthands, `.reshape`, MLIR-level byte-identity with raw `matmul_rmajor_f32_v`/`dot_*_v` intrinsics (requires shape-dim threading from type-checker to lower_expr). | IR-text byte-identity gated (RFC §7.2): `A @ B` ≡ `tensor.matmul(A, B)` via `format_ir_module`; `A .+ B` ≡ `A + B`; same for `.-`/`.*`/`./`. 17/17 Phase B tests pass (`tests/rfc0012_phase_b_operators.rs`). Full suite 0 new failures (3 pre-existing mindcraft_check_cli failures unchanged). Bootstrap oracle 6/6. blas_smoke 12/12. Bench: medium_mlp -0.37% (p=0.05), large_network +3.3% (p=0.08) — all within +7% cap. |
| C | `#[deterministic]`, `#[target(...)]`, and `#[q16]` annotations. Compile-time checks: `determinism::nondeterministic_in_deterministic`, `determinism::float_in_q16_fn`, `shape::target_unavailable`. | `#[deterministic]` functions that operate on `q16` tensors must lower to MLIR byte-identical to the explicit `dot_q16_v` / `matmul_rmajor_q16_v` calls. `#[target(cpu)]` on an f32 function must lower identically to the current default path. |
| D | Tensor-op fusion. The compiler fuses chained tensor expressions (`W2 @ (W1 @ x)`, `(A @ B) .+ c`) into single kernel passes when shape and consumption analysis permits. | The fused lowering must produce a result byte-identical to the sequential unfused lowering for the same inputs. This is the only phase where the MLIR may differ from the hand-written std.blas calls — but the output values must be identical. The Phase D gate is a value-identity check, not an MLIR-byte-identity check, with the unfused Phase B lowering as the oracle. |

### 7.1 Phase A detail

Phase A adds the `Tensor<dtype, [dims]>` type constructor to the mindc
type-checker. No operators, no MLIR changes. The deliverables are:

- Parser: `TypeAnn::Tensor { dtype, shape, layout }` variant.
- Type-checker: shape unification for symbolic dims within function bodies;
  `shape::*` diagnostics for type-mismatch sites.
- `mindc check` (RFC 0007): `shape::*` diagnostics integrated into the check
  pass. The `shape::` namespace is new; it shares the severity model from
  RFC 0007 §5.
- No changes to the MLIR lowering path. `cargo test` and the bootstrap
  fixed-point are byte-identical.

Phase A is a prerequisite for Phase B. Phases C and D are independent of each
other after Phase B lands.

### 7.2 Phase B byte-identity gate (normative)

For Phase B to ship, the following test must pass for every combination in the
gate matrix:

```
gate_matrix = {
  (dtype=f32,   op=matmul,  shape=(M,K)@(K,N))  -> matmul_rmajor_f32_v
  (dtype=q16,   op=matmul,  shape=(M,K)@(K,N))  -> matmul_rmajor_q16_v
  (dtype=f32,   op=dot,     shape=(N,).(N,))     -> dot_f32_v
  (dtype=q16,   op=dot,     shape=(N,).(N,))     -> dot_q16_v
  (dtype=f32,   op=norm_l1, shape=(N,))          -> dot_l1_f32_v
  (dtype=q16,   op=norm_l1, shape=(N,))          -> dot_l1_q16_v
  (dtype=f32,   op=norm_linf, shape=(N,))        -> dot_linf_f32_v
}
```

Test form: compile both the tensor-syntax form and the explicit std.blas form
for a representative set of concrete shapes. Assert that the emitted MLIR is
byte-identical. This is the same discipline used for the mindc bootstrap
fixed-point (RFC 0005 Phase 6) and the Track B vector-path increments (RFC 0006
§9.2).

---

## 8. Backward compatibility and coexistence

### 8.1 Every existing `.mind` file compiles unchanged

Tensor syntax is a strictly additive front-end extension. The lexer, parser, and
type-checker gain new productions; no existing production is removed or
modified. An existing `.mind` file that does not use `Tensor<...>`, `@`, `.+`,
or `#[deterministic]` is parsed and compiled identically to today.

### 8.2 Function-call forms are permanent

The std.blas function-call forms — `dot_q16`, `matmul_rmajor_f32`, etc. — are
the lowering target for tensor operators. They are not deprecated; they are the
explicit, escape-hatch layer. A developer who needs fine-grained control over
buffer addresses, alignment, or sub-backend selection can always drop to the
function-call form. The two layers coexist permanently.

### 8.3 The bootstrap fixed-point is unaffected

The pure-MIND self-host compiler (`examples/mindc_mind/main.mind`, 1,084 LOC)
does not use tensor syntax. It is written in the Rust-family base surface. Phase
A–D additions are front-end productions that the bootstrap source never
exercises. The bootstrap fixed-point (10,889 bytes / 206 SSA, v0.6.2) must be
re-verified to be byte-identical after each phase lands — the same discipline
used throughout the self-hosting ladder. No phase of RFC 0012 may ship without
this verification.

### 8.4 The Q16.16 bit-identity gate is unaffected

Tensor operators on `q16` dtypes lower to the same `dot_q16_v` /
`matmul_rmajor_q16_v` functions that the function-call forms use. The MLIR
emitted for a `q16` tensor operation must be byte-identical to the MLIR emitted
for the explicit function-call form (Phase B gate, §7.2). The cross-arch
bit-identity invariant (task #57) is therefore preserved by construction: the
tensor surface is a different syntax for the same operations, not a new
implementation of them.

### 8.5 Mindcraft lint rules gain `shape::*` and `determinism::*`

`mindc check` (RFC 0007) acquires the new diagnostic namespaces. Shape and
determinism diagnostics share the severity model and reporter infrastructure
from RFC 0007 §6. Existing projects see no new diagnostics unless they use
tensor syntax. The `shape::*` and `determinism::*` namespaces are configurable
in `Mind.toml [mindcraft]` using the same per-rule severity mechanism as the
existing `lint::*` rules.

---

## 9. Relation to other RFCs

**RFC 0005 (pure-MIND std surface)** — `Tensor<dtype, [dims]>` is a
compile-time view over the i64-address heap-record ABI that RFC 0005 establishes.
The underlying storage, allocation, and deallocation are RFC 0005 mechanisms.
RFC 0012 adds the type-level abstraction; RFC 0005 provides the runtime
representation it abstracts over.

**RFC 0006 (mind-blas)** — every tensor operator lowers to a std.blas
function. RFC 0006 is the permanent lowering target; RFC 0012 is the higher
surface over it. The Q16.16 byte-identity gate (task #57) that RFC 0006
establishes is the determinism foundation that `#[q16]` and `#[deterministic]`
build on. Phase B of RFC 0012 cannot ship before RFC 0006 Track B is fully
landed, because the tensor-operator lowering depends on the `*_v` vector-dialect
forms.

**RFC 0007 (Mindcraft)** — `mindc check` gains the `shape::*` and
`determinism::*` diagnostic namespaces in Phase A and Phase C respectively. The
per-rule severity model, the `#[allow(shape::matmul_mismatch, reason="...")]`
suppression form, and the JSON reporter schema are shared with RFC 0007's
existing `lint::*` surface unchanged.

**RFC 0010 (memory safety + C ABI)** — tensor allocations are subject to the
three-tier memory model. A tensor computed inside a `region { }` block is
region-interior; a tensor that must outlive the computation block is
`GenRef<Tensor<...>>`. No new allocation surface is introduced by RFC 0012; it
uses the mechanisms RFC 0010 provides.

**RFC 0008 (mindc build)** — the `#[target(...)]` annotation corresponds to the
`[build].target` vocabulary in `Mind.toml`. A `#[target(cerebras)]` annotation
in source and a `[build] target = "cerebras"` manifest entry select the same
MLIR lowering pipeline. RFC 0012 makes the selection expressible per-function in
source; RFC 0008 provides the manifest-level and CLI-level selection. Both
mechanisms are in effect simultaneously; the more specific (function-level)
annotation wins over the manifest default.

---

## 10. Open questions

### 10.1 Symbolic-dimension unification scope

Phase A specifies unification within a function body. Cross-function unification
— where a function declared as `fn transform<N>(x: Tensor<f32, [N]>) -> Tensor<f32, [N]>`
propagates the symbolic dim `N` into the call site — requires a const-generic-
like mechanism. The question is how much unification to implement in Phase A
versus deferring.

**Proposed resolution:** Phase A implements intra-body unification and
ground-term propagation (symbolic dims unified against concrete values from the
call context). Cross-function symbolic dim propagation is Phase A-extended, not
Phase A. A function that accepts a symbolic dim must receive a ground value at
every call site in Phase A; the compiler rejects ambiguous symbolic dims that
remain unresolved at the function boundary.

### 10.2 Broadcasting: standard row-major prefix rules or a stricter subset?

The standard row-major prefix broadcasting convention permits prefix-1 expansion:
`Tensor<f32, [1, N]>` broadcasts against `Tensor<f32, [M, N]>`. This is useful
but complicates the compile-time shape checker — prefix-1 dimensions can mask
shape bugs.

**Proposed resolution:** Phase B implements a strict subset: broadcast is
permitted only when one operand is rank-0 (scalar) or rank-1 broadcasting along
the last dimension of a rank-2 tensor. Full prefix-expansion broadcasting is a
Phase B extended feature, not Phase B baseline. A future RFC or Phase B extension
may relax this.

### 10.3 `#[target(...)]` without the named backend — error or fallback?

The §5.2 resolution specifies `"error"` as the default. Whether a project should
be able to set `target_mismatch = "warn"` globally and ship a library that uses
`#[target(cerebras)]` for users who don't have the backend is a policy question.

The proposed answer (§5.2) is that `"error"` is the default and `"warn"` is an
explicit opt-in. This means a library distributed via the package layer (RFC 0009)
that uses `#[target(cerebras)]` will fail to compile for users without the
backend by default. The library author opts in to the warn-and-fallback policy
explicitly. This is the correct default for the governance-substrate use case
where silent fallbacks are as dangerous as silent shape mismatches.

### 10.4 Rank-N reshape with unresolved symbolic dims

When `.reshape([N*2, K/2])` involves symbolic arithmetic, the compiler must
verify size invariance symbolically. This requires at minimum a linear
arithmetic solver over dimension expressions.

**Proposed resolution:** Phase A defers symbolic reshape arithmetic. `.reshape`
with concrete dimensions and with a single symbolic dimension that can be
inferred (the "fill" dimension, equivalent to `-1` in other systems) is
Phase A. Reshape with multiple symbolic dimensions and non-trivial constraints
is Phase A-extended or a future RFC.

### 10.5 Interaction between `#[deterministic]` and f32 reduction order

The precise definition of "deterministic" for f32 operations requires
specifying which reduction order is fixed. The Track B MLIR vector path (RFC
0006 §9.1) uses a tree-shaped `vector.reduction <add>` that is reproducible on
a given hardware configuration but not strictly cross-architecture
bit-identical for f32 (floating-point addition is not associative).

**Proposed resolution:** `#[deterministic]` on an f32 function requires
`#[target(q16)]` or explicit `f64` (for oracle use cases) to guarantee cross-
architecture bit-identity. A `#[deterministic]` f32 function without a
fixed-point target emits a `determinism::f32_non_bit_identical` info diagnostic
— not an error, but a signal that the determinism guarantee is
within-substrate-reproducible, not cross-substrate byte-identical. The Mindcraft
configuration can promote this to `warn` or `error` for projects that require
strict cross-architecture identity.

---

## 11. Decision points

This section records the design choices made for this RFC. These decisions are
final; superseding a decision requires a new RFC.

### 11.1 Tensor type syntax: `Tensor<dtype, [dims]>` is selected

Three candidates (§3.5). Decision: `Tensor<dtype, [dims]>`. The generic-bracket
form parses cleanly, makes the dtype prominent, and requires no grammar changes
beyond registering the built-in type constructor. The shape-prefix and
type-suffix forms both introduce ambiguities with the existing bracket-literal
and expression grammar. This decision is final.

### 11.2 `q16` is the dtype keyword (not `Q16_16`, not `q16_16`)

`Q16_16` and `q16_16` both reference the format name in an implementation
notation. `q16` is the surface form: lowercase, two characters after the
qualifier, unambiguous. It is distinct from `i16` (signed 16-bit integer) and
from `f32` by both prefix (`q` vs `f`/`i`) and the implicit scale convention.
The keyword is `q16`; the C intrinsic names (`__mind_blas_dot_q16`) continue to
use their existing identifiers internally. This decision is final.

### 11.3 Annotation sigil: `#[...]` for all annotations; `@` freed for matmul

MIND already uses `#[...]` for attributes: `#[test]`, `#[allow(...)]`,
`#[repr(C)]`, `#[allow(lint::q16_overflow, reason="...")]` (RFC 0007 §5).
Introducing a second annotation sigil `@name(...)` for the placement and
determinism directives would create two annotation systems in the same language
with no semantic distinction.

**Decision: all placement and determinism annotations use `#[...]`.**
`#[deterministic]`, `#[target(cpu)]`, `#[q16]`.

This decision has a direct consequence: `@` is fully available as a binary infix
operator for matrix multiplication. The grammar distinguishes `#[` (attribute
start, always precedes an item or expression at the statement level) from `@`
(binary operator in expression context) by syntactic position. There is no
ambiguity. `@` as a binary operator is unambiguous because MIND has no unary
`@` prefix, no `@` in type position, and no `@` in item-declaration position.

This decision is final. The `@` character is the matmul operator. The `#[...]`
form is the sole annotation sigil in MIND.

### 11.4 Elementwise operator prefix: `.` dot-prefix is selected

Elementwise operators `.+`, `.-`, `.*`, `./` are prefixed with `.` to
distinguish them from their scalar counterparts `+`, `-`, `*`, `/`. The
alternatives — a different character prefix, or overloading `+`/`-`/`*`/`/`
directly — were considered:

- Overloading `+` etc. directly requires the type checker to inspect operand
  types to determine whether a scalar or elementwise path is taken. This is
  workable but means the behavior of `a + b` changes depending on whether `a`
  and `b` are scalars or tensors — a reader cannot determine the operation from
  the syntax alone.
- A different prefix (e.g. `~+`) introduces a new character that has no natural
  mnemonic.

**Decision: `.` dot-prefix.** The dot prefix reads naturally as "elementwise"
and has no conflict with the postfix `.T`, `.norm()`, etc. because the latter
are postfix on a single operand and the elementwise operators are infix between
two operands. Disambiguation in the grammar is positional. This decision is
final.

---

## 12. References

- RFC 0005 — pure-MIND std surface (the i64-address heap-record ABI that tensor
  types are a compile-time view over; the allocation and deallocation model).
- RFC 0006 — mind-blas (the permanent lowering target for tensor operators; the
  Q16.16 byte-identity gate that `#[q16]` and `#[deterministic]` build on).
- RFC 0007 — Mindcraft (the `mindc check` infrastructure that `shape::*` and
  `determinism::*` diagnostics extend; the `#[allow(..., reason="...")]`
  suppression form).
- RFC 0008 — mindc build (the `[build].target` vocabulary that `#[target(...)]`
  annotations correspond to; the backend dispatch table).
- RFC 0010 — memory safety + C ABI (the three-tier memory model governing tensor
  allocation lifetimes; region-interior and `GenRef<T>` as the tensor lifetime
  mechanisms).
- task #57 — cross-arch Q16.16 bit-identity gate (the determinism foundation
  that `#[q16]` lowers to; must hold after every phase of this RFC).
- task #230 — mind-nerve native-encode GEMM latency (the direct performance
  target that tensor-op fusion in Phase D addresses).
- V24 paper §9 roofline (the memory-bound substrate evidence that motivates the
  fusion payoff in §6.4 of this RFC).
