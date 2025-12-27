# Tensor shape semantics

MIND tracks tensor ranks and dimensions statically. Shapes are represented as ordered lists of dimensions, where a dimension is either a known integer or a symbolic placeholder.

## Broadcasting

Elementwise operators follow NumPy-style broadcasting:

- Dimensions are aligned from the trailing axis.
- Two dimensions are compatible when they are equal or either side is `1`.
- The result dimension is the non-`1` value when one side provides a singleton.

## Reductions

`tensor.sum` and `tensor.mean` accept an explicit axis list and a `keepdims` flag. An empty axis list reduces over every dimension. When `keepdims` is true the reduced axes become length-`1`; otherwise they are removed.

## Shape transforms

- `tensor.reshape` requires the source and target shapes to have the same element count when all dimensions are known.
- `tensor.transpose` uses an explicit permutation (defaults to full reversal) and requires a valid, duplicate-free axis list.
- `tensor.expand_dims` inserts a length-`1` dimension at the requested position; negative axes count from the end.
- `tensor.squeeze` removes axes that are explicitly listed or, when omitted, every axis whose size is `1`.

## Indexing and slicing

- `tensor.index` removes the selected axis.
- `tensor.slice` keeps the selected axis but updates its size to `end - start` when both bounds are static.
- `tensor.slice_stride` mirrors Python-style slicing with a non-zero `step`, adjusting the output length when bounds and the input dimension are known.
- `tensor.gather` splices the index tensor shape into the target axis, mirroring the runtime layout.

## Convolution

The compiler assumes NHWC inputs and HWCF filters. Strides must be positive and padding can be `valid` or `same`. Channel dimensions must match statically when known. Output heights and widths follow the same formulas used by the MLIR lowering and runtime.
