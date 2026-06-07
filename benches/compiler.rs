use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use libmind::{CompileOptions, compile_source};

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

/// **Tier T1 — frontend-only** (`docs/benchmarking.md` §1). The clock spans
/// `parse + typecheck + IR` (`compile_source`), in-process. It does **NOT**
/// include MLIR emission, `mlir-opt`, LLVM, the `clang` link, or process
/// startup. This is the source of the published 1.8–15.5 µs numbers; they are
/// frontend-only and must never be compared against another toolchain's full
/// `build` time without the scope note. The tier-matched end-to-end figure is
/// `e2e_compile_shared` (T2) below.
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

// ---------------------------------------------------------------------------
// Tier T2 — end-to-end process wall-clock compile (frontend + MLIR + LLVM).
// ---------------------------------------------------------------------------

/// Minimal cdylib entries the `--emit-shared` path can lower end to end. These
/// are full `pub fn` programs (not the frontend expression snippets above),
/// because the T2 clock has to drive the whole pipeline through to a linked
/// `.so`. Three sizes so the per-shape codegen+LLVM cost is visible.
const E2E_SMALL: &str = r#"
pub fn add(a: i64, b: i64) -> i64 { a + b }
"#;
const E2E_MEDIUM: &str = r#"
pub fn poly(x: i64) -> i64 {
    let a = x * x;
    let b = a * x;
    let c = b + a;
    c + x * 7 - 3
}
"#;
const E2E_LARGE: &str = r#"
pub fn f0(x: i64) -> i64 { x * x + 1 }
pub fn f1(x: i64) -> i64 { let a = f0(x); a * a - x }
pub fn f2(x: i64) -> i64 { let a = f1(x); let b = f0(a); a + b }
pub fn f3(x: i64) -> i64 { let a = f2(x); let b = f1(a); a * 3 + b - x }
pub fn driver(x: i64) -> i64 {
    let a = f3(x);
    let b = f2(a);
    let c = f1(b);
    a + b + c
}
"#;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Locate the built `mindc` binary (debug preferred, release fallback).
fn mindc_path() -> Option<PathBuf> {
    let dbg = manifest_dir().join("target").join("debug").join("mindc");
    if dbg.exists() {
        return Some(dbg);
    }
    let rel = manifest_dir().join("target").join("release").join("mindc");
    if rel.exists() { Some(rel) } else { None }
}

/// Whether the T2 end-to-end path can actually run: the MLIR toolchain is on
/// PATH and `mindc` is built. Same self-skip contract as the GEMM benches —
/// the bench compiles unconditionally; the skip is a runtime check.
fn e2e_available() -> Option<&'static PathBuf> {
    static MINDC: OnceLock<Option<PathBuf>> = OnceLock::new();
    MINDC
        .get_or_init(|| {
            for tool in ["mlir-opt", "mlir-translate", "clang"] {
                if which::which(tool).is_err() {
                    eprintln!(
                        "compiler::e2e_compile_shared: {tool} not on PATH; \
                         skipping the T2 end-to-end sweep (toolchain shadowed)"
                    );
                    return None;
                }
            }
            match mindc_path() {
                Some(p) => Some(p),
                None => {
                    eprintln!(
                        "compiler::e2e_compile_shared: mindc not built; run \
                         `cargo build --features \"mlir-build std-surface cross-module-imports\" \
                         --bin mindc`; skipping the T2 sweep"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// Compile one source to a fresh `.so` via `mindc --emit-shared`, end to end.
/// Returns the exit status' success flag so the timed loop can assert it.
fn emit_shared_once(mindc: &PathBuf, src_path: &PathBuf, so_path: &PathBuf) -> bool {
    Command::new(mindc)
        .args([
            src_path.to_str().unwrap(),
            "--emit-shared",
            so_path.to_str().unwrap(),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// **Tier T2 — process wall-clock, end-to-end** (`docs/benchmarking.md` §1).
/// The clock spans the full `mindc src --emit-shared out.so` pipeline as a
/// spawned process: frontend + MLIR text + `mlir-opt` + LLVM + `clang` link,
/// **including process startup**. This is the tier-matched figure to quote
/// against another toolchain's `build` time — NOT the T1 1.8–15.5 µs frontend
/// number. Distinct from `compiler_pipeline` (T1) and from the in-process
/// `end_to_end::source_to_mlir` group (which stops at MLIR text, no LLVM/link).
///
/// Self-skips cleanly when the MLIR toolchain is shadowed or `mindc` is unbuilt.
fn bench_e2e_compile_shared(c: &mut Criterion) {
    let Some(mindc) = e2e_available() else {
        eprintln!("compiler::e2e_compile_shared: T2 end-to-end sweep unavailable; skipped.");
        return;
    };

    let dir = std::env::temp_dir();
    let mut group = c.benchmark_group("e2e_compile_shared");
    // The whole-pipeline compile is millisecond-scale and spawns a process per
    // sample; a long measurement window here would dominate `cargo bench`. Keep
    // the sample budget modest — this is a wall-clock latency figure, not a
    // tight-CI microbench.
    group.sample_size(20);

    for (name, source) in [
        ("small", E2E_SMALL),
        ("medium", E2E_MEDIUM),
        ("large", E2E_LARGE),
    ] {
        let src_path = dir.join(format!("mind_bench_e2e_{name}.mind"));
        let so_path = dir.join(format!("mind_bench_e2e_{name}.so"));
        if std::fs::write(&src_path, source).is_err() {
            eprintln!("compiler::e2e_compile_shared: could not stage {name} source; skipping it");
            continue;
        }
        // Verify it compiles once before timing — a bench that times a failing
        // compile would report a meaningless number.
        if !emit_shared_once(mindc, &src_path, &so_path) {
            eprintln!(
                "compiler::e2e_compile_shared: {name} did not emit a shared object; \
                 skipping this shape (not a timing result)"
            );
            continue;
        }

        group.bench_with_input(
            BenchmarkId::new("source_to_so", name),
            &(src_path, so_path),
            |b, (src, so)| {
                b.iter(|| {
                    let ok = emit_shared_once(black_box(mindc), black_box(src), black_box(so));
                    assert!(ok, "mindc --emit-shared failed inside the timed loop");
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the WEDGE path, not just the parser: canonical mic@3 serialization
/// (`emit_mic3`) plus the `ir_trace_hash` that anchors the embedded evidence
/// chain. This gates the performance of the moat machinery — the cross-substrate-
/// identical serialization + hash — alongside parse latency.
///
/// Before timing, it asserts the load-bearing property once per input: `emit_mic3`
/// and `ir_trace_hash` are byte-identical across calls. Determinism *is* the wedge;
/// if this ever fails, the bench would be measuring a non-deterministic path that
/// breaks cross-substrate byte-identity.
fn bench_wedge_evidence(c: &mut Criterion) {
    use libmind::ir::compact::emit_mic3;
    use libmind::ir::ir_trace_hash;

    let mut group = c.benchmark_group("wedge_evidence");

    for (name, source) in [
        ("small_matmul", SMALL_MATMUL),
        ("medium_mlp", MEDIUM_MLP),
        ("large_network", LARGE_NETWORK),
    ] {
        let products =
            compile_source(source, &CompileOptions::default()).expect("compilation failed");

        // Wedge invariant, checked once outside the timed loop.
        assert_eq!(
            emit_mic3(&products.ir),
            emit_mic3(&products.ir),
            "mic@3 emission must be byte-identical (wedge determinism) for {name}"
        );
        assert_eq!(
            ir_trace_hash(&products.ir),
            ir_trace_hash(&products.ir),
            "trace_hash must be deterministic (wedge) for {name}"
        );

        group.bench_with_input(BenchmarkId::new("emit_mic3", name), &products.ir, |b, m| {
            b.iter(|| black_box(emit_mic3(black_box(m))));
        });
        group.bench_with_input(
            BenchmarkId::new("ir_trace_hash", name),
            &products.ir,
            |b, m| {
                b.iter(|| black_box(ir_trace_hash(black_box(m))));
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
                b.iter(|| {
                    #[cfg(feature = "autodiff")]
                    let result = lower_to_mlir(black_box(ir), None);
                    #[cfg(not(feature = "autodiff"))]
                    let result = lower_to_mlir(black_box(ir));
                    result.expect("MLIR lowering failed")
                });
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
                    #[cfg(feature = "autodiff")]
                    let result = lower_to_mlir(&products.ir, None);
                    #[cfg(not(feature = "autodiff"))]
                    let result = lower_to_mlir(&products.ir);
                    result.expect("MLIR lowering failed")
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

/// RFC 0002 deliverable 1 sub-bench — exercises the new
/// `Node::Export` -> `IRModule.exports` lowering path with 0, 1, and 10
/// export names. Lives as its own criterion group so the headline
/// `compiler_pipeline` numbers stay measurable against
/// `.bench-baseline-2026-04-28-pratt.txt`.
fn bench_c_export_lowering(c: &mut Criterion) {
    use libmind::eval::lower_to_ir;
    use libmind::parser::parse;

    let mut group = c.benchmark_group("c_export_lowering");

    for (name, source) in [
        ("0_exports", "1 + 2 * 3".to_string()),
        ("1_export", "export { foo }".to_string()),
        (
            "10_exports",
            format!(
                "export {{ {} }}",
                (0..10)
                    .map(|i| format!("name_{i}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ),
    ] {
        group.bench_with_input(BenchmarkId::new("parse_lower", name), &source, |b, src| {
            b.iter(|| {
                let module = parse(black_box(src)).expect("parse failed");
                lower_to_ir(&module)
            })
        });
    }

    group.finish();
}

#[cfg(all(feature = "mlir-lowering", feature = "autodiff"))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_wedge_evidence,
    bench_mlir_lowering,
    bench_end_to_end_compilation,
    bench_e2e_compile_shared,
    bench_autodiff_generation,
    bench_c_export_lowering
);

#[cfg(all(feature = "mlir-lowering", not(feature = "autodiff")))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_wedge_evidence,
    bench_mlir_lowering,
    bench_end_to_end_compilation,
    bench_e2e_compile_shared,
    bench_c_export_lowering
);

#[cfg(all(not(feature = "mlir-lowering"), feature = "autodiff"))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_wedge_evidence,
    bench_e2e_compile_shared,
    bench_autodiff_generation,
    bench_c_export_lowering
);

#[cfg(all(not(feature = "mlir-lowering"), not(feature = "autodiff")))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_wedge_evidence,
    bench_e2e_compile_shared,
    bench_c_export_lowering
);

criterion_main!(benches);
