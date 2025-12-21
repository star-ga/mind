"""
Mojo equivalent: Small matrix multiplication (10x20 * 20x15)
MIND equivalent: benches/simple_benchmarks.rs - small_matmul
"""

from memory import UnsafePointer

fn matmul(a: UnsafePointer[Float32], b: UnsafePointer[Float32], c: UnsafePointer[Float32], M: Int, N: Int, K: Int):
    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                sum += a[i * K + k] * b[k * N + j]
            c[i * N + j] = sum

fn main():
    var M = 10
    var N = 15
    var K = 20

    var a = UnsafePointer[Float32].alloc(M * K)
    var b = UnsafePointer[Float32].alloc(K * N)
    var c = UnsafePointer[Float32].alloc(M * N)

    # Initialize
    for i in range(M * K):
        a[i] = 1.0
    for i in range(K * N):
        b[i] = 1.0

    matmul(a, b, c, M, N, K)
    print("Done:", c[0])

    a.free()
    b.free()
    c.free()
