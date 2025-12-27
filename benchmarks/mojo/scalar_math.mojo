"""
Mojo equivalent: Scalar arithmetic
MIND equivalent: benches/simple_benchmarks.rs - scalar_math
"""

fn compute() -> Int:
    return 1 + 2 * 3 - 4 // 2  # Integer division

fn main():
    var result = compute()
    print(result)
