// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Cross-module resolution sub-benchmark (Phase 10.6 item 9 / D2).
//!
//! Its OWN bench target (`required-features = ["cross-module-imports"]`)
//! so it never appears in the headline `compiler` criterion group and
//! cannot perturb the `.bench-baseline` numbers that gate the µs
//! frontend moat. Measures the added cost of building the module table
//! and type-checking a module that imports N symbols, isolated from the
//! base parse/typecheck path.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use libmind::parser::parse;
use libmind::project::module_table::build_module_table;
use libmind::type_checker::check_module_types_with_modules;
use std::collections::HashMap;

fn bench_cross_module_resolve(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_module_resolve");

    for n in [1usize, 8, 64] {
        // Module A exports n symbols; module B imports the module and
        // references the first symbol.
        let exports: Vec<String> = (0..n).map(|i| format!("sym_{i}")).collect();
        let a_src = format!("export {{ {} }}", exports.join(", "));
        let b_src = "use crate.a\nlet x = sym_0\n".to_string();

        group.bench_with_input(BenchmarkId::new("table_build", n), &a_src, |bch, src| {
            let a = parse(src).expect("parse A");
            bch.iter(|| {
                let pairs = [("crate.a".to_string(), &a)];
                black_box(build_module_table(black_box(&pairs)))
            });
        });

        group.bench_with_input(
            BenchmarkId::new("typecheck_resolved", n),
            &b_src,
            |bch, src| {
                let a = parse(&a_src).expect("parse A");
                let b = parse(src).expect("parse B");
                let table = build_module_table(&[("crate.a".to_string(), &a)]);
                bch.iter(|| {
                    black_box(check_module_types_with_modules(
                        black_box(&b),
                        src,
                        None,
                        &HashMap::new(),
                        black_box(&table),
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cross_module_resolve);
criterion_main!(benches);
