"""
Mojo equivalent: Small matrix multiplication
MIND equivalent: benches/simple_benchmarks.rs - small_matmul

Matrix dimensions: [10, 20] × [20, 30] = [10, 30]
"""

from tensor import Tensor, TensorShape
from random import rand

fn matmul_small():
    # Initialize tensors
    let a = Tensor[DType.float32](TensorShape(10, 20))
    let b = Tensor[DType.float32](TensorShape(20, 30))

    # Fill with ones (equivalent to MIND's initialization)
    for i in range(10):
        for j in range(20):
            a[i, j] = 1.0

    for i in range(20):
        for j in range(30):
            b[i, j] = 1.0

    # Matrix multiplication
    let result = a @ b

    print("Result shape:", result.shape())

fn main():
    matmul_small()
