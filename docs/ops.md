# Core v1 operator coverage

The Core v1 surface exposes a small, auditable set of tensor operators. The
registry lives in `src/ops/core_v1.rs` and is consumable from the CLI via
`mindc ops --core-v1`.

Elementwise ops (`add`, `sub`, `mul`, `div`, `tensor.relu`) and matrix
multiplication (`tensor.matmul`) are tagged with Core v1 shape rule categories
so the shared shape engine can validate broadcasting and inner-dimension
compatibility during compilation.

| name             | arity    | differentiable | notes                                      |
| ---------------- | -------- | -------------- | ------------------------------------------ |
| add              | 2        | yes            | Elementwise add with broadcasting.         |
| sub              | 2        | yes            | Elementwise subtract with broadcasting.    |
| mul              | 2        | yes            | Elementwise multiply with broadcasting.    |
| div              | 2        | yes            | Elementwise divide with broadcasting.      |
| tensor.sum       | 1+       | yes            | Axis reduction with optional keepdims.     |
| tensor.mean      | 1+       | yes            | Mean reduction with optional keepdims.     |
| tensor.reshape   | 2        | yes            | Reshape to a compatible target shape.      |
| tensor.expand_dims | 2      | yes            | Insert a length-1 dimension.               |
| tensor.squeeze   | 1+       | yes            | Remove length-1 dimensions.                |
| tensor.transpose | 2        | yes            | Permute axes.                              |
| tensor.dot       | 2        | yes            | 1D dot product.                            |
| tensor.matmul    | 2        | yes            | Matrix multiplication.                      |
| tensor.conv2d    | 2        | yes            | 2D convolution with stride/padding.        |
| tensor.index     | 2+       | no             | Integer indexing.                          |
| tensor.slice     | 2+       | no             | Half-open slicing.                         |
| tensor.gather    | 3        | yes            | Gather along an axis using indices.        |
| tensor.relu      | 1        | yes            | Elementwise ReLU.                          |

Operators with `differentiable = no` are explicitly excluded from the Core v1
autodiff contract and should surface clear diagnostics when used with gradient
requests.
