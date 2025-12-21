"""
Mojo equivalent: Small matrix multiplication (10x20 * 20x15)
MIND equivalent: benches/simple_benchmarks.rs - small_matmul
"""


fn matmul(a: List[Float32], b: List[Float32], mut c: List[Float32], M: Int, N: Int, K: Int):
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

    var a = List[Float32](capacity=M * K)
    var b = List[Float32](capacity=K * N)
    var c = List[Float32](capacity=M * N)

    # Initialize
    for _ in range(M * K):
        a.append(1.0)
    for _ in range(K * N):
        b.append(1.0)
    for _ in range(M * N):
        c.append(0.0)

    matmul(a, b, c, M, N, K)
    print("Done:", c[0])
