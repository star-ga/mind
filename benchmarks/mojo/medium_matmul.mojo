"""
Mojo equivalent: Medium matrix multiplication
MIND equivalent: benches/simple_benchmarks.rs - medium_matmul

Matrix dimensions: [128, 256] × [256, 512] = [128, 512]
"""

from tensor import Tensor, TensorShape

fn matmul_medium():
    # Initialize tensors
    let a = Tensor[DType.float32](TensorShape(128, 256))
    let b = Tensor[DType.float32](TensorShape(256, 512))

    # Fill with ones
    for i in range(128):
        for j in range(256):
            a[i, j] = 1.0

    for i in range(256):
        for j in range(512):
            b[i, j] = 1.0

    # Matrix multiplication
    let result = a @ b

    print("Result shape:", result.shape())

fn main():
    matmul_medium()
