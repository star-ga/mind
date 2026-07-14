// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Parser-throughput bench — the workload the `parse_stmt` statement-keyword
//! recogniser actually runs on.
//!
//! The frozen `compiler_pipeline/parse_typecheck_ir` benches
//! (`benches/compiler.rs`, watched by `tools/bench_gate.py`) are 3–7-statement
//! tensor snippets: they are dominated by shape inference and never exercise the
//! statement dispatch more than a handful of times. `parse_stmt` runs once per
//! statement of every compiled file, so the honest instrument is a parse-only
//! clock over a real, large source.
//!
//! `examples/mindc_mind/main.mind` (the self-host compiler, ~29.7k lines) is the
//! real workload. It is checked into the repo, so this bench is reproducible and
//! does not depend on anything generated.
//!
//! This is a NEW bench target on purpose: `tools/bench_gate.py` watches
//! `--bench compiler` only, so adding a measurement instrument here cannot move
//! the frozen baseline.

use std::path::PathBuf;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use libmind::parser;

fn selfhost_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("mindc_mind")
        .join("main.mind");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// A synthetic body made only of statement-leading keywords, so the clock is
/// dominated by the `parse_stmt` dispatch itself rather than by expression
/// parsing. Complements the real-source bench: this one isolates the recogniser,
/// the other one measures the honest end-to-end parse.
fn keyword_dense_source() -> String {
    let mut s = String::with_capacity(256 * 1024);
    s.push_str("fn kw_dense(n: i64) -> i64 {\n");
    s.push_str("    let mut acc: i64 = 0;\n");
    for i in 0..2000 {
        s.push_str("    let x: i64 = ");
        s.push_str(&i.to_string());
        s.push_str(";\n");
        s.push_str("    if x > 0 { acc = acc + x; }\n");
        s.push_str("    while acc > 1000000 { acc = acc - 1000000; }\n");
    }
    s.push_str("    return acc;\n}\n");
    s
}

fn bench_parse_selfhost(c: &mut Criterion) {
    let src = selfhost_source();
    let mut group = c.benchmark_group("parse_only");
    group.throughput(Throughput::Bytes(src.len() as u64));
    group.sample_size(20);
    group.bench_function("selfhost_main_mind", |b| {
        b.iter(|| parser::parse(black_box(src.as_str())).expect("self-host source must parse"));
    });
    group.finish();
}

fn bench_parse_keyword_dense(c: &mut Criterion) {
    let src = keyword_dense_source();
    let mut group = c.benchmark_group("parse_only");
    group.throughput(Throughput::Bytes(src.len() as u64));
    group.bench_function("keyword_dense", |b| {
        b.iter(|| parser::parse(black_box(src.as_str())).expect("keyword-dense source must parse"));
    });
    group.finish();
}

criterion_group!(benches, bench_parse_selfhost, bench_parse_keyword_dense);
criterion_main!(benches);
