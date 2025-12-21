use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mind::{compile_source, differentiate_function, CompileOptions};

/// Simple linear function
const LINEAR: &str = r#"
fn linear(x: Tensor<F32, [128, 784]>, w: Tensor<F32, [784, 10]>, b: Tensor<F32, [10]>) -> Tensor<F32, [128, 10]> {
    let matmul_out = tensor.matmul(x, w);
    add(matmul_out, b)
}
"#;

/// Two-layer MLP
const TWO_LAYER_MLP: &str = r#"
fn mlp(
    x: Tensor<F32, [128, 784]>,
    w1: Tensor<F32, [784, 512]>,
    b1: Tensor<F32, [512]>,
    w2: Tensor<F32, [512, 10]>,
    b2: Tensor<F32, [10]>
) -> Tensor<F32, [128, 10]> {
    let h1 = tensor.matmul(x, w1);
    let h1_bias = add(h1, b1);
    let h1_relu = tensor.relu(h1_bias);
    let h2 = tensor.matmul(h1_relu, w2);
    add(h2, b2)
}
"#;

/// Deep network (5 layers)
const DEEP_NETWORK: &str = r#"
fn deep_network(
    x: Tensor<F32, [64, 256]>,
    w1: Tensor<F32, [256, 256]>,
    w2: Tensor<F32, [256, 256]>,
    w3: Tensor<F32, [256, 256]>,
    w4: Tensor<F32, [256, 256]>,
    w5: Tensor<F32, [256, 10]>
) -> Tensor<F32, [64, 10]> {
    let h1 = tensor.relu(tensor.matmul(x, w1));
    let h2 = tensor.relu(tensor.matmul(h1, w2));
    let h3 = tensor.relu(tensor.matmul(h2, w3));
    let h4 = tensor.relu(tensor.matmul(h3, w4));
    tensor.matmul(h4, w5)
}
"#;

/// Conv network
const CONV_NETWORK: &str = r#"
fn conv_net(
    input: Tensor<F32, [32, 3, 32, 32]>,
    conv1_w: Tensor<F32, [64, 3, 3, 3]>,
    conv2_w: Tensor<F32, [128, 64, 3, 3]>
) -> Tensor<F32, [32, 128, 28, 28]> {
    let conv1 = tensor.conv2d(input, conv1_w, 1, 0);
    let relu1 = tensor.relu(conv1);
    let conv2 = tensor.conv2d(relu1, conv2_w, 1, 0);
    tensor.relu(conv2)
}
"#;

/// MSE Loss
const MSE_LOSS: &str = r#"
fn mse_loss(pred: Tensor<F32, [128, 10]>, target: Tensor<F32, [128, 10]>) -> Tensor<F32, []> {
    let diff = sub(pred, target);
    let squared = mul(diff, diff);
    tensor.mean(squared)
}
"#;

/// Loss with L2 regularization
const REGULARIZED_LOSS: &str = r#"
fn loss_with_reg(
    pred: Tensor<F32, [128, 10]>,
    target: Tensor<F32, [128, 10]>,
    weights: Tensor<F32, [784, 512]>
) -> Tensor<F32, []> {
    let diff = sub(pred, target);
    let squared = mul(diff, diff);
    let loss = tensor.mean(squared);
    let reg = tensor.mean(mul(weights, weights));
    let scaled_reg = mul(reg, 0.01);
    add(loss, scaled_reg)
}
"#;

/// Complex loss with multiple terms
const COMPLEX_LOSS: &str = r#"
fn complex_loss(
    pred: Tensor<F32, [64, 100]>,
    target: Tensor<F32, [64, 100]>,
    w1: Tensor<F32, [256, 256]>,
    w2: Tensor<F32, [256, 256]>,
    w3: Tensor<F32, [256, 100]>
) -> Tensor<F32, []> {
    let diff = sub(pred, target);
    let squared = mul(diff, diff);
    let mse = tensor.mean(squared);

    let reg1 = tensor.mean(mul(w1, w1));
    let reg2 = tensor.mean(mul(w2, w2));
    let reg3 = tensor.mean(mul(w3, w3));

    let total_reg = add(add(reg1, reg2), reg3);
    let scaled_reg = mul(total_reg, 0.001);

    add(mse, scaled_reg)
}
"#;

/// Attention gradient (simplified QK^T)
const ATTENTION_GRAD: &str = r#"
fn attention_qk(
    query: Tensor<F32, [8, 128, 64]>,
    key: Tensor<F32, [8, 128, 64]>
) -> Tensor<F32, [8, 128, 128]> {
    let key_t = tensor.transpose(key, [0, 2, 1]);
    let scores = tensor.matmul(query, key_t);
    let scaled = mul(scores, 0.125);
    scaled
}
"#;

fn bench_autodiff_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_simple");

    for (name, source) in [("linear", LINEAR), ("mse_loss", MSE_LOSS)] {
        let products =
            compile_source(source, &CompileOptions::default()).expect("compilation failed");

        group.bench_with_input(
            BenchmarkId::new("gradient_gen", name),
            &products.ir,
            |b, ir| {
                b.iter(|| differentiate_function(black_box(ir), "main").expect("autodiff failed"));
            },
        );
    }

    group.finish();
}

fn bench_autodiff_mlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_mlp");

    for (name, source) in [("two_layer", TWO_LAYER_MLP), ("five_layer", DEEP_NETWORK)] {
        let products =
            compile_source(source, &CompileOptions::default()).expect("compilation failed");

        group.bench_with_input(
            BenchmarkId::new("gradient_gen", name),
            &products.ir,
            |b, ir| {
                b.iter(|| differentiate_function(black_box(ir), "main").expect("autodiff failed"));
            },
        );
    }

    group.finish();
}

fn bench_autodiff_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_conv");

    let products =
        compile_source(CONV_NETWORK, &CompileOptions::default()).expect("compilation failed");

    group.bench_with_input(
        BenchmarkId::new("gradient_gen", "conv_net"),
        &products.ir,
        |b, ir| {
            b.iter(|| differentiate_function(black_box(ir), "main").expect("autodiff failed"));
        },
    );

    group.finish();
}

fn bench_autodiff_complex_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_loss");

    for (name, source) in [
        ("regularized", REGULARIZED_LOSS),
        ("complex_multi_term", COMPLEX_LOSS),
        ("attention", ATTENTION_GRAD),
    ] {
        let products =
            compile_source(source, &CompileOptions::default()).expect("compilation failed");

        group.bench_with_input(
            BenchmarkId::new("gradient_gen", name),
            &products.ir,
            |b, ir| {
                b.iter(|| differentiate_function(black_box(ir), "main").expect("autodiff failed"));
            },
        );
    }

    group.finish();
}

fn bench_autodiff_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_end_to_end");

    for (name, source) in [
        ("linear", LINEAR),
        ("two_layer_mlp", TWO_LAYER_MLP),
        ("conv_net", CONV_NETWORK),
    ] {
        group.bench_with_input(
            BenchmarkId::new("compile_and_diff", name),
            source,
            |b, src| {
                b.iter(|| {
                    let products = compile_source(black_box(src), &CompileOptions::default())
                        .expect("compilation failed");
                    differentiate_function(&products.ir, "main").expect("autodiff failed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_autodiff_simple,
    bench_autodiff_mlp,
    bench_autodiff_conv,
    bench_autodiff_complex_loss,
    bench_autodiff_end_to_end
);

criterion_main!(benches);
