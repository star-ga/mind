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

criterion_group!(benches, bench_call_lowering);
criterion_main!(benches);
