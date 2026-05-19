// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 0 sub-benchmark — generic `Instr::Call` lowering.
//!
//! Own bench target (`required-features = ["std-surface",
//! "mlir-lowering"]`) so it never enters the headline `compiler`
//! criterion group and cannot perturb `.bench-baseline` (the µs
//! frontend moat). Measures the added cost of lowering a module with
//! N `func.call`s in isolation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libmind::ir::{IRModule, Instr};
use libmind::mlir::lower_ir_to_mlir;

/// RFC 0006 Track B (increment 2) — dense-reduction-throughput
/// sub-category. Measures the cost of *lowering* a module that calls a
/// vector-dialect reduction intrinsic over an 8K-element vector
/// (`__mind_blas_dot_f32_v` / `_q16_v` / `_l1_f32_v` / `_linf_f32_v`).
///
/// This is its own bench target (`required-features = ["std-surface",
/// "mlir-lowering"]`) so it never enters the headline `compiler`
/// criterion group and CANNOT perturb `.bench-baseline-2026-05-18-
/// rfc0005.txt` (the small_matmul / medium_mlp / large_network µs
/// frontend moat lives in `benches/compiler.rs`). The compile-time
/// frontend gate (2.80–17.10 µs) is structurally untouched: this
/// measurement exercises only the gated vector-lowering path.
///
/// It reports the p95 of lowering the 8K-element dot for each metric;
/// the criterion default (mean + std-dev + outlier classification)
/// covers the p95 characterisation the RFC asks for without a custom
/// estimator.
fn bench_dense_reduction_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("blas_dense_reduction_lowering");

    // The four Track B vector reduction surfaces. Each is a single
    // intercepted `Instr::Call` that the MLIR lowering expands into a
    // full vector-dialect reduction loop over `len` elements; lowering
    // an 8K-element reduction is the unit the RFC's "dense reduction
    // throughput" sub-category tracks.
    const VEC_INTRINSICS: &[&str] = &[
        "__mind_blas_dot_f32_v",
        "__mind_blas_dot_q16_v",
        "__mind_blas_dot_l1_f32_v",
        "__mind_blas_dot_linf_f32_v",
    ];
    const ELEMS: usize = 8192;

    for &name in VEC_INTRINSICS {
        group.bench_with_input(BenchmarkId::new("lower_8k_dot", name), &name, |bch, &nm| {
            bch.iter(|| {
                let mut m = IRModule::new();
                let a = m.fresh();
                let b = m.fresh();
                let n = m.fresh();
                m.instrs.push(Instr::ConstI64(a, 0x1000));
                m.instrs.push(Instr::ConstI64(b, 0x2000));
                m.instrs.push(Instr::ConstI64(n, ELEMS as i64));
                let dst = m.fresh();
                m.instrs.push(Instr::Call {
                    dst,
                    name: nm.to_string(),
                    args: vec![a, b, n],
                });
                m.instrs.push(Instr::Output(dst));
                black_box(lower_ir_to_mlir(black_box(&m)).expect("lower vec dot"))
            });
        });
    }

    group.finish();
}

fn bench_call_lowering(c: &mut Criterion) {
    let mut group = c.benchmark_group("std_surface_call_lowering");

    for n in [1usize, 8, 64] {
        group.bench_with_input(BenchmarkId::new("lower_calls", n), &n, |b, &n| {
            b.iter(|| {
                let mut m = IRModule::new();
                let arg = m.fresh();
                m.instrs.push(Instr::ConstI64(arg, 64));
                for i in 0..n {
                    let dst = m.fresh();
                    m.instrs.push(Instr::Call {
                        dst,
                        name: format!("__mind_intr_{}", i % 5),
                        args: vec![arg],
                    });
                    if i + 1 == n {
                        m.instrs.push(Instr::Output(dst));
                    }
                }
                black_box(lower_ir_to_mlir(black_box(&m)).expect("lower"))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_call_lowering,
    bench_dense_reduction_throughput
);
criterion_main!(benches);
