use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mind::{compile_source, CompileOptions};

/// Benchmark source programs that are known to work
const PROGRAMS: &[(&str, &str)] = &[
    ("scalar_math", "1 + 2 * 3 - 4 / 2"),
    (
        "small_matmul",
        r#"
            let a: Tensor[f32,(10,20)] = 1;
            let b: Tensor[f32,(20,30)] = 1;
            tensor.matmul(a, b)
        "#,
    ),
    (
        "medium_matmul",
        r#"
            let a: Tensor[f32,(128,256)] = 1;
            let b: Tensor[f32,(256,512)] = 1;
            tensor.matmul(a, b)
        "#,
    ),
    (
        "large_matmul",
        r#"
            let a: Tensor[f32,(512,1024)] = 1;
            let b: Tensor[f32,(1024,512)] = 1;
            tensor.matmul(a, b)
        "#,
    ),
    (
        "tensor_ops",
        r#"
            let x: Tensor[f32,(64,64)] = 1;
            let y: Tensor[f32,(64,64)] = 2;
            let sum = add(x, y);
            let prod = mul(sum, x);
            tensor.relu(prod)
        "#,
    ),
    (
        "reductions",
        r#"
            let x: Tensor[f32,(128,256,512)] = 1;
            let sum1 = tensor.sum(x, [0]);
            let sum2 = tensor.sum(sum1, [0]);
            tensor.mean(sum2)
        "#,
    ),
    (
        "reshape_ops",
        r#"
            let x: Tensor[f32,(32,64,128)] = 1;
            let reshaped = tensor.reshape(x, [32,8192]);
            let transposed = tensor.transpose(reshaped, [1,0]);
            transposed
        "#,
    ),
];

fn bench_compile_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_small");

    for &(name, source) in &PROGRAMS[0..3] {
        group.bench_with_input(
            BenchmarkId::new("parse_check_lower", name),
            &source,
            |b, src| {
                b.iter(|| {
                    compile_source(black_box(src), &CompileOptions::default())
                        .expect("compilation failed")
                });
            },
        );
    }

    group.finish();
}

fn bench_compile_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_medium");

    for &(name, source) in &PROGRAMS[3..] {
        group.bench_with_input(
            BenchmarkId::new("parse_check_lower", name),
            &source,
            |b, src| {
                b.iter(|| {
                    compile_source(black_box(src), &CompileOptions::default())
                        .expect("compilation failed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_compile_small, bench_compile_medium);

criterion_main!(benches);
