# Sparse Tensor Types

The MIND type system includes a first-class sparse tensor type surface.
Sparse tensors are declared with an explicit storage layout and element type,
giving the compiler visibility into sparsity structure at the type level.

## Syntax

```
tensor<sparse[layout], element_type>
tensor<sparse[layout], element_type[dim0, dim1, ...]>
```

Where `layout` is one of:

| Layout | Description |
|--------|-------------|
| `csr`  | Compressed Sparse Row — efficient row-wise iteration |
| `csc`  | Compressed Sparse Column — efficient column-wise iteration |
| `coo`  | Coordinate format — flexible, unordered |
| `bsr`  | Block Sparse Row — cache-friendly for block-structured matrices |

## Examples

```
fn sparse_matmul(
    a: tensor<sparse[csr], f32[1024, 1024]>,
    b: tensor<sparse[csc], f32[1024, 1024]>,
) -> tensor<sparse[csr], f32[1024, 1024]> {
    return dot(a, b);
}
```

```
let weights: tensor<sparse[csr], q16_16> = load_weights();
```

## Type Annotation Semantics

`tensor<sparse[layout], element_type>` is an opaque aggregate type at the IR
level. The `valuetype_from_ann` function returns `None` for sparse tensor
annotations — the layout is resolved by the execution backend, not the type
checker.

This mirrors the approach of the MLIR `sparse_tensor` dialect (arXiv:2202.04305),
where layout descriptors are carried through the IR without requiring the front
end to know the physical memory representation.

Sparse tensor annotations survive the canonicalization pass without semantic
change. The `SparseAttr` IR instruction carries annotation metadata for
values that have been typed as sparse.

## Wafer-Scale Accelerator Considerations

Wafer-scale accelerators benefit from compile-time sparsity propagation because
routing sparsity decisions to hardware at compile time eliminates the overhead of
runtime layout negotiation. When a function parameter is annotated as
`tensor<sparse[csr], ...>`, the compiler can propagate that annotation through
the dataflow graph and emit backend hints that select the appropriate sparse
kernel at the call site.

This type surface is the scaffold for that propagation. Full sparse autodiff
lowering (CGO 2024, DOI:10.1109/CGO57630.2024.10444787) is deferred to a
follow-on pass.

## Shape Dimensions

Shape dimensions in sparse tensor types follow the same syntax as dense tensor
shapes: integer literals for known sizes, identifiers for symbolic dimensions.

```
tensor<sparse[csr], f32[N, 1024]>   // symbolic batch dimension
tensor<sparse[csr], f32[512, 512]>  // fully concrete shape
```

Omitting the shape `[...]` suffix produces a rank-unknown sparse tensor type.

## IR Representation

At the IR level, a `SparseAttr` instruction annotates an existing SSA value
with its sparse layout:

```
%1 = sparse.attr %0 layout=Csr
```

This instruction is metadata-only: it does not change the value, it only
carries the layout hint downstream for backend consumption.
