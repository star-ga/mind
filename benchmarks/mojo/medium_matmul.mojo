"""
Mojo equivalent: Medium matrix multiplication (128x256 * 256x512)
MIND equivalent: benches/simple_benchmarks.rs - medium_matmul
"""


fn matmul(a: List[Float32], b: List[Float32], mut c: List[Float32], M: Int, N: Int, K: Int):
    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                sum += a[i * K + k] * b[k * N + j]
            c[i * N + j] = sum

fn main():
    var M = 128
    var N = 512
    var K = 256

    var a = List[Float32](capacity=M * K)
    var b = List[Float32](capacity=K * N)
    var c = List[Float32](capacity=M * N)

    for _ in range(M * K):
        a.append(1.0)
    for _ in range(K * N):
        b.append(1.0)
    for _ in range(M * N):
        c.append(0.0)

    matmul(a, b, c, M, N, K)
    print("Done:", c[0])
