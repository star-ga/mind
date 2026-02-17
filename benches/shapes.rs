use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libmind::{compile_source, CompileOptions};

/// Simple broadcasting scenarios
const BROADCAST_SIMPLE: &str = r#"
fn broadcast_add(a: Tensor<F32, [128, 1]>, b: Tensor<F32, [1, 256]>) -> Tensor<F32, [128, 256]> {
    add(a, b)
}
"#;

/// Complex broadcasting with multiple operations
const BROADCAST_COMPLEX: &str = r#"
fn broadcast_chain(
    a: Tensor<F32, [64, 1, 128]>,
    b: Tensor<F32, [1, 32, 128]>,
    c: Tensor<F32, [64, 32, 1]>
) -> Tensor<F32, [64, 32, 128]> {
    let ab = mul(a, b);
    let abc = add(ab, c);
    tensor.relu(abc)
}
"#;

/// High-rank tensor operations
const HIGH_RANK: &str = r#"
fn high_rank_ops(
    x: Tensor<F32, [4, 8, 16, 32, 64]>,
    y: Tensor<F32, [1, 1, 16, 32, 64]>
) -> Tensor<F32, [4, 8, 16, 32, 64]> {
    let sum1 = add(x, y);
    let sum2 = tensor.sum(sum1, [0, 1]);
    let expanded = tensor.expand_dims(sum2, 0);
    expanded
}
"#;

/// Reduction operations
const REDUCTIONS: &str = r#"
fn reduction_chain(x: Tensor<F32, [128, 256, 512]>) -> Tensor<F32, []> {
    let sum1 = tensor.sum(x, [0]);
    let sum2 = tensor.sum(sum1, [0]);
    let sum3 = tensor.sum(sum2, [0]);
    sum3
}
"#;

/// Reshape and transpose operations
const SHAPE_TRANSFORMS: &str = r#"
fn transform_shapes(x: Tensor<F32, [128, 16, 16, 64]>) -> Tensor<F32, [128, 64, 256]> {
    let reshaped = tensor.reshape(x, [128, 256, 64]);
    let transposed = tensor.transpose(reshaped, [0, 2, 1]);
    transposed
}
"#;

/// MatMul with various sizes
const MATMUL_SMALL: &str = r#"
fn matmul_8x8(a: Tensor<F32, [8, 8]>, b: Tensor<F32, [8, 8]>) -> Tensor<F32, [8, 8]> {
    tensor.matmul(a, b)
}
"#;

const MATMUL_MEDIUM: &str = r#"
fn matmul_128x128(a: Tensor<F32, [128, 256]>, b: Tensor<F32, [256, 128]>) -> Tensor<F32, [128, 128]> {
    tensor.matmul(a, b)
}
"#;

const MATMUL_LARGE: &str = r#"
fn matmul_1024x1024(
    a: Tensor<F32, [1024, 2048]>,
    b: Tensor<F32, [2048, 1024]>
) -> Tensor<F32, [1024, 1024]> {
    tensor.matmul(a, b)
}
"#;

const MATMUL_BATCHED: &str = r#"
fn matmul_batched(
    a: Tensor<F32, [32, 128, 256]>,
    b: Tensor<F32, [32, 256, 512]>
) -> Tensor<F32, [32, 128, 512]> {
    tensor.matmul(a, b)
}
"#;

/// Conv2D with various kernel sizes and strides
const CONV_3X3: &str = r#"
fn conv_3x3(
    input: Tensor<F32, [1, 3, 224, 224]>,
    kernel: Tensor<F32, [64, 3, 3, 3]>
) -> Tensor<F32, [1, 64, 222, 222]> {
    tensor.conv2d(input, kernel, 1, 0)
}
"#;

const CONV_5X5: &str = r#"
fn conv_5x5_stride2(
    input: Tensor<F32, [1, 64, 56, 56]>,
    kernel: Tensor<F32, [128, 64, 5, 5]>
) -> Tensor<F32, [1, 128, 26, 26]> {
    tensor.conv2d(input, kernel, 2, 0)
}
"#;

fn bench_shape_inference_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_inference_broadcast");

    for (name, source) in [
        ("simple", BROADCAST_SIMPLE),
        ("complex", BROADCAST_COMPLEX),
        ("high_rank", HIGH_RANK),
    ] {
        group.bench_with_input(BenchmarkId::new("broadcast", name), source, |b, src| {
            b.iter(|| {
                compile_source(black_box(src), &CompileOptions::default())
                    .expect("compilation failed")
            });
        });
    }

    group.finish();
}

fn bench_shape_inference_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_inference_reductions");

    for (name, source) in [("chain", REDUCTIONS), ("transforms", SHAPE_TRANSFORMS)] {
        group.bench_with_input(BenchmarkId::new("reduction", name), source, |b, src| {
            b.iter(|| {
                compile_source(black_box(src), &CompileOptions::default())
                    .expect("compilation failed")
            });
        });
    }

    group.finish();
}

fn bench_shape_inference_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_inference_matmul");

    for (name, source) in [
        ("8x8", MATMUL_SMALL),
        ("128x256", MATMUL_MEDIUM),
        ("1024x2048", MATMUL_LARGE),
        ("batched_32x128x256", MATMUL_BATCHED),
    ] {
        group.bench_with_input(BenchmarkId::new("matmul", name), source, |b, src| {
            b.iter(|| {
                compile_source(black_box(src), &CompileOptions::default())
                    .expect("compilation failed")
            });
        });
    }

    group.finish();
}

fn bench_shape_inference_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_inference_conv");

    for (name, source) in [("3x3_stride1", CONV_3X3), ("5x5_stride2", CONV_5X5)] {
        group.bench_with_input(BenchmarkId::new("conv2d", name), source, |b, src| {
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
    bench_shape_inference_broadcast,
    bench_shape_inference_reductions,
    bench_shape_inference_matmul,
    bench_shape_inference_conv
);

criterion_main!(benches);
