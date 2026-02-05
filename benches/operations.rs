use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libmind::{compile_source, CompileOptions};

/// Element-wise operations
const ELEMENTWISE_CHAIN: &str = r#"
fn elementwise_ops(
    a: Tensor<F32, [1024, 1024]>,
    b: Tensor<F32, [1024, 1024]>,
    c: Tensor<F32, [1024, 1024]>
) -> Tensor<F32, [1024, 1024]> {
    let ab = add(a, b);
    let abc = mul(ab, c);
    let result = div(abc, b);
    tensor.relu(result)
}
"#;

/// Mixed precision operations
const MIXED_PRECISION: &str = r#"
fn mixed_precision(
    a_f32: Tensor<F32, [256, 256]>,
    b_bf16: Tensor<BF16, [256, 256]>
) -> Tensor<F32, [256, 256]> {
    // Note: This assumes type coercion is handled
    add(a_f32, a_f32)
}
"#;

/// Slice and gather operations
const INDEXING_OPS: &str = r#"
fn indexing_chain(x: Tensor<F32, [128, 256, 512]>) -> Tensor<F32, [64, 128, 256]> {
    let sliced = tensor.slice(x, [0, 0, 0], [64, 128, 256], [1, 1, 1]);
    sliced
}
"#;

/// Deeply nested operations
const DEEP_NEST: &str = r#"
fn deep_computation(x: Tensor<F32, [64, 64]>) -> Tensor<F32, [64, 64]> {
    let x1 = add(x, x);
    let x2 = mul(x1, x1);
    let x3 = add(x2, x);
    let x4 = tensor.relu(x3);
    let x5 = mul(x4, x2);
    let x6 = add(x5, x1);
    let x7 = tensor.relu(x6);
    let x8 = mul(x7, x);
    x8
}
"#;

/// Reduction with various axes
const REDUCTION_AXES: &str = r#"
fn reduce_multiple_axes(x: Tensor<F32, [32, 64, 128, 256]>) -> Tensor<F32, [32, 256]> {
    let reduced = tensor.sum(x, [1, 2]);
    reduced
}
"#;

/// Full neural network layer simulation
const NEURAL_LAYER: &str = r#"
fn dense_layer_with_norm(
    input: Tensor<F32, [128, 512]>,
    weight: Tensor<F32, [512, 256]>,
    bias: Tensor<F32, [256]>
) -> Tensor<F32, [128, 256]> {
    let matmul_out = tensor.matmul(input, weight);
    let biased = add(matmul_out, bias);
    let activated = tensor.relu(biased);

    // Layer normalization simulation
    let mean_val = tensor.mean(activated);
    let centered = sub(activated, mean_val);
    centered
}
"#;

/// Attention-like pattern (simplified)
const ATTENTION_PATTERN: &str = r#"
fn attention_scores(
    query: Tensor<F32, [128, 8, 64]>,
    key: Tensor<F32, [128, 8, 64]>
) -> Tensor<F32, [128, 8, 8]> {
    let key_t = tensor.transpose(key, [0, 2, 1]);
    let scores = tensor.matmul(query, key_t);
    scores
}
"#;

/// ResNet-like residual block
const RESIDUAL_BLOCK: &str = r#"
fn residual_block(
    input: Tensor<F32, [1, 64, 56, 56]>,
    conv1_kernel: Tensor<F32, [64, 64, 3, 3]>,
    conv2_kernel: Tensor<F32, [64, 64, 3, 3]>
) -> Tensor<F32, [1, 64, 54, 54]> {
    let conv1 = tensor.conv2d(input, conv1_kernel, 1, 0);
    let relu1 = tensor.relu(conv1);
    let conv2 = tensor.conv2d(relu1, conv2_kernel, 1, 0);

    // Simplified: no skip connection due to shape mismatch
    tensor.relu(conv2)
}
"#;

fn bench_elementwise_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations_elementwise");

    for (name, source) in [("chain", ELEMENTWISE_CHAIN), ("deep_nest", DEEP_NEST)] {
        group.bench_with_input(BenchmarkId::new("compile", name), source, |b, src| {
            b.iter(|| {
                compile_source(black_box(src), &CompileOptions::default())
                    .expect("compilation failed")
            });
        });
    }

    group.finish();
}

fn bench_indexing_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations_indexing");

    for (name, source) in [("slice", INDEXING_OPS), ("reduction_axes", REDUCTION_AXES)] {
        group.bench_with_input(BenchmarkId::new("compile", name), source, |b, src| {
            b.iter(|| {
                compile_source(black_box(src), &CompileOptions::default())
                    .expect("compilation failed")
            });
        });
    }

    group.finish();
}

fn bench_neural_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations_neural_patterns");

    for (name, source) in [
        ("dense_layer", NEURAL_LAYER),
        ("attention", ATTENTION_PATTERN),
        ("residual_block", RESIDUAL_BLOCK),
    ] {
        group.bench_with_input(BenchmarkId::new("compile", name), source, |b, src| {
            b.iter(|| {
                compile_source(black_box(src), &CompileOptions::default())
                    .expect("compilation failed")
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise_operations,
    bench_indexing_operations,
    bench_neural_patterns
);

criterion_main!(benches);
