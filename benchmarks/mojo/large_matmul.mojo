"""
Mojo equivalent: Large matrix multiplication
MIND equivalent: benches/simple_benchmarks.rs - large_matmul

Matrix dimensions: [512, 1024] × [1024, 512] = [512, 512]
"""

from tensor import Tensor, TensorShape

fn matmul_large():
    # Initialize tensors
    let a = Tensor[DType.float32](TensorShape(512, 1024))
    let b = Tensor[DType.float32](TensorShape(1024, 512))

    # Fill with ones
    for i in range(512):
        for j in range(1024):
            a[i, j] = 1.0

    for i in range(1024):
        for j in range(512):
            b[i, j] = 1.0

    # Matrix multiplication
    let result = a @ b

    print("Result shape:", result.shape())

fn main():
    matmul_large()
