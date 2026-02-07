use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libmind::{compile_source, CompileOptions};

#[cfg(feature = "mlir-lowering")]
use libmind::lower_to_mlir;

#[cfg(feature = "autodiff")]
use libmind::differentiate_function;

/// Small program: Simple matrix multiplication
const SMALL_MATMUL: &str = r#"
    let a: Tensor[f32,(10,20)] = 1;
    let b: Tensor[f32,(20,30)] = 1;
    tensor.matmul(a, b)
"#;

/// Medium program: MatMul with activations
const MEDIUM_MLP: &str = r#"
    let input: Tensor[f32,(128,256)] = 0;
    let weight: Tensor[f32,(256,128)] = 1;
    let bias: Tensor[f32,(128)] = 0;
    let matmul_out = tensor.matmul(input, weight);
    let biased = matmul_out + bias;
    tensor.relu(biased)
"#;

/// Large program: Multi-layer network
const LARGE_NETWORK: &str = r#"
    let input: Tensor[f32,(128,784)] = 0;
    let w1: Tensor[f32,(784,512)] = 1;
    let b1: Tensor[f32,(512)] = 0;
    let w2: Tensor[f32,(512,256)] = 1;
    let b2: Tensor[f32,(256)] = 0;
    let w3: Tensor[f32,(256,10)] = 1;
    let b3: Tensor[f32,(10)] = 0;

    let matmul1 = tensor.matmul(input, w1);
    let h1 = tensor.relu(matmul1 + b1);

    let matmul2 = tensor.matmul(h1, w2);
    let h2 = tensor.relu(matmul2 + b2);

    let matmul3 = tensor.matmul(h2, w3);
    matmul3 + b3
"#;

/// Complex autodiff target: Nested operations
#[cfg(feature = "autodiff")]
const AUTODIFF_TARGET: &str = r#"
    let pred: Tensor[f32,(128,10)] = 0;
    let target: Tensor[f32,(128,10)] = 1;
    let weights: Tensor[f32,(784,512)] = 0;

    let diff = pred - target;
    let squared = diff * diff;
    let loss = tensor.mean(squared);
    let reg = tensor.mean(weights * weights);
    loss + reg
"#;

fn bench_compilation_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiler_pipeline");

    for (name, source) in [
        ("small_matmul", SMALL_MATMUL),
        ("medium_mlp", MEDIUM_MLP),
        ("large_network", LARGE_NETWORK),
    ] {
        group.bench_with_input(
            BenchmarkId::new("parse_typecheck_ir", name),
            source,
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

#[cfg(feature = "mlir-lowering")]
fn bench_mlir_lowering(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlir_lowering");

    for (name, source) in [
        ("small_matmul", SMALL_MATMUL),
        ("medium_mlp", MEDIUM_MLP),
        ("large_network", LARGE_NETWORK),
    ] {
        // Pre-compile to IR
        let products =
            compile_source(source, &CompileOptions::default()).expect("compilation failed");

        group.bench_with_input(
            BenchmarkId::new("ir_to_mlir", name),
            &products.ir,
            |b, ir| {
                b.iter(|| lower_to_mlir(black_box(ir), None).expect("MLIR lowering failed"));
            },
        );
    }

    group.finish();
}

#[cfg(feature = "mlir-lowering")]
fn bench_end_to_end_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");

    for (name, source) in [
        ("small_matmul", SMALL_MATMUL),
        ("medium_mlp", MEDIUM_MLP),
        ("large_network", LARGE_NETWORK),
    ] {
        group.bench_with_input(
            BenchmarkId::new("source_to_mlir", name),
            source,
            |b, src| {
                b.iter(|| {
                    let products = compile_source(black_box(src), &CompileOptions::default())
                        .expect("compilation failed");
                    lower_to_mlir(&products.ir, None).expect("MLIR lowering failed")
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "autodiff")]
fn bench_autodiff_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff");

    for (name, source) in [
        ("simple_matmul", SMALL_MATMUL),
        ("mlp_with_relu", MEDIUM_MLP),
        ("loss_function", AUTODIFF_TARGET),
    ] {
        // Pre-compile to IR
        let products =
            compile_source(source, &CompileOptions::default()).expect("compilation failed");

        group.bench_with_input(
            BenchmarkId::new("generate_gradients", name),
            &products.ir,
            |b, ir| {
                b.iter(|| differentiate_function(black_box(ir), "main").expect("autodiff failed"));
            },
        );
    }

    group.finish();
}

#[cfg(all(feature = "mlir-lowering", feature = "autodiff"))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_mlir_lowering,
    bench_end_to_end_compilation,
    bench_autodiff_generation
);

#[cfg(all(feature = "mlir-lowering", not(feature = "autodiff")))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_mlir_lowering,
    bench_end_to_end_compilation
);

#[cfg(all(not(feature = "mlir-lowering"), feature = "autodiff"))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_autodiff_generation
);

#[cfg(all(not(feature = "mlir-lowering"), not(feature = "autodiff")))]
criterion_group!(benches, bench_compilation_pipeline);

criterion_main!(benches);
