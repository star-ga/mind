use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;
use std::path::Path;

// ---------------------------------------------------------------------------
// File readers
// ---------------------------------------------------------------------------

fn read_mind_file(path: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

// ---------------------------------------------------------------------------
// Synthetic 1000-LOC stress corpus
//
// Exercises deeply nested struct definitions + many fn bodies — the two
// constructs that drive worst-case indentation bookkeeping.  Generated
// inline so the bench has no file dependency and is self-contained.
// ---------------------------------------------------------------------------

fn synthetic_1000_loc() -> String {
    let mut out = String::with_capacity(32 * 1024);

    // 10 struct definitions, each with 8 fields.
    for i in 0..10usize {
        out.push_str(&format!("struct Widget{i} {{\n"));
        for f in 0..8usize {
            out.push_str(&format!("    field_{f}: i64,\n"));
        }
        out.push_str("}\n\n");
    }

    // 40 function definitions, each with a nested if/let body (~18 lines).
    for i in 0..40usize {
        out.push_str(&format!("pub fn compute_{i}(x: i64, y: i64) -> i64 {{\n"));
        out.push_str("    let a: i64 = x + y;\n");
        out.push_str("    let b: i64 = a * 2;\n");
        out.push_str("    let c: i64 = if b > 0 {\n");
        out.push_str("        let d: i64 = b - 1;\n");
        out.push_str("        if d > 10 {\n");
        out.push_str("            let e: i64 = d * d;\n");
        out.push_str("            e + 1\n");
        out.push_str("        } else {\n");
        out.push_str("            d + 2\n");
        out.push_str("        }\n");
        out.push_str("    } else {\n");
        out.push_str("        0 - b\n");
        out.push_str("    };\n");
        out.push_str("    c\n");
        out.push_str("}\n\n");
    }

    out
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_fmt_vec_mind(c: &mut Criterion) {
    let src = read_mind_file("std/vec.mind");
    let cfg = MindcraftFormatConfig::default();

    c.bench_with_input(
        BenchmarkId::new("mindcraft_fmt", "vec.mind"),
        &src,
        |b, s| {
            b.iter(|| format_source(black_box(s), black_box(&cfg)).expect("format failed"));
        },
    );
}

fn bench_fmt_mindc_mind(c: &mut Criterion) {
    let src = read_mind_file("examples/mindc_mind/main.mind");
    let cfg = MindcraftFormatConfig::default();

    c.bench_with_input(
        BenchmarkId::new("mindcraft_fmt", "mindc_mind/main.mind"),
        &src,
        |b, s| {
            b.iter(|| format_source(black_box(s), black_box(&cfg)).expect("format failed"));
        },
    );
}

fn bench_fmt_synthetic_stress(c: &mut Criterion) {
    let src = synthetic_1000_loc();
    let cfg = MindcraftFormatConfig::default();

    c.bench_with_input(
        BenchmarkId::new("mindcraft_fmt", "synthetic_1000_loc"),
        &src,
        |b, s| {
            b.iter(|| format_source(black_box(s), black_box(&cfg)).expect("format failed"));
        },
    );
}

criterion_group!(
    benches,
    bench_fmt_vec_mind,
    bench_fmt_mindc_mind,
    bench_fmt_synthetic_stress
);
criterion_main!(benches);
