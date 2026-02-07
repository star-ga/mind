use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;

fn parse_and_eval(source: &str) {
    let module = libmind::parser::parse(source).expect("parse failed");
    let mut env: HashMap<String, i64> = HashMap::new();
    let mode = libmind::eval::ExecMode::CpuExec;
    libmind::eval::eval_module_value_with_env_mode(&module, &mut env, Some(source), mode)
        .expect("eval failed");
}

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");

    let cases: &[(&str, &str)] = &[
        ("add_4", "let x: Tensor[f32,(4)] = 1; let y: Tensor[f32,(4)] = 2; x + y"),
        ("add_1000", "let x: Tensor[f32,(1000)] = 1; let y: Tensor[f32,(1000)] = 2; x + y"),
        ("add_10000", "let x: Tensor[f32,(10000)] = 1; let y: Tensor[f32,(10000)] = 2; x + y"),
        ("add_100000", "let x: Tensor[f32,(100000)] = 1; let y: Tensor[f32,(100000)] = 2; x + y"),
        ("mul_10000", "let x: Tensor[f32,(10000)] = 2; let y: Tensor[f32,(10000)] = 3; x * y"),
    ];

    for &(name, source) in cases {
        group.bench_with_input(
            BenchmarkId::new("compute", name),
            &source,
            |b, src| b.iter(|| parse_and_eval(black_box(src))),
        );
    }

    group.finish();
}

fn bench_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction");

    let cases: &[(&str, &str)] = &[
        ("sum_4", "let x: Tensor[f32,(4)] = 1; let y: Tensor[f32,(4)] = 2; tensor.sum(x + y)"),
        ("sum_1000", "let x: Tensor[f32,(1000)] = 1; let y: Tensor[f32,(1000)] = 2; tensor.sum(x + y)"),
        ("sum_10000", "let x: Tensor[f32,(10000)] = 1; let y: Tensor[f32,(10000)] = 2; tensor.sum(x + y)"),
        ("mean_10000", "let x: Tensor[f32,(10000)] = 1; let y: Tensor[f32,(10000)] = 2; tensor.mean(x + y)"),
    ];

    for &(name, source) in cases {
        group.bench_with_input(
            BenchmarkId::new("compute", name),
            &source,
            |b, src| b.iter(|| parse_and_eval(black_box(src))),
        );
    }

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    let cases: &[(&str, &str)] = &[
        ("10x20_20x30", "let a: Tensor[f32,(10,20)] = 1; let b: Tensor[f32,(20,30)] = 1; tensor.matmul(a, b)"),
        ("32x64_64x32", "let a: Tensor[f32,(32,64)] = 1; let b: Tensor[f32,(64,32)] = 1; tensor.matmul(a, b)"),
        ("64x128_128x64", "let a: Tensor[f32,(64,128)] = 1; let b: Tensor[f32,(128,64)] = 1; tensor.matmul(a, b)"),
        ("128x256_256x128", "let a: Tensor[f32,(128,256)] = 1; let b: Tensor[f32,(256,128)] = 1; tensor.matmul(a, b)"),
    ];

    for &(name, source) in cases {
        group.bench_with_input(
            BenchmarkId::new("compute", name),
            &source,
            |b, src| b.iter(|| parse_and_eval(black_box(src))),
        );
    }

    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    let cases: &[(&str, &str)] = &[
        ("relu_1000", "let x: Tensor[f32,(1000)] = 1; let y: Tensor[f32,(1000)] = 2; tensor.relu(x - y)"),
        ("relu_10000", "let x: Tensor[f32,(10000)] = 1; let y: Tensor[f32,(10000)] = 2; tensor.relu(x - y)"),
    ];

    for &(name, source) in cases {
        group.bench_with_input(
            BenchmarkId::new("compute", name),
            &source,
            |b, src| b.iter(|| parse_and_eval(black_box(src))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_elementwise, bench_reduction, bench_matmul, bench_relu);
criterion_main!(benches);
