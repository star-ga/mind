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
/// `build` time without the scope note. The tier-matched end-to-end figures are
/// `e2e_compile_shared/{warm,cold}` (T2) and `e2e_native` (T2-native) below.
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
// End-to-end tiers (docs/benchmarking.md §1): T2 `e2e_compile_shared/{warm,cold}`,
// T2-native `e2e_native`, and the same-session `e2e_reference` toolchains.
// ---------------------------------------------------------------------------

/// Every end-to-end tier, run in one criterion pass so the MIND rows and the
/// reference-toolchain rows share a box and a session.
///
/// Compiled only with `mlir-build` (the build `--emit-shared` needs). The
/// frontend T1 job builds `--no-default-features`, so it neither compiles nor
/// runs any of this: its sweep, its runtime and its frozen floor are unchanged.
fn bench_e2e(c: &mut Criterion) {
    #[cfg(feature = "mlir-build")]
    e2e::run(c);
    #[cfg(not(feature = "mlir-build"))]
    {
        let _ = c;
        eprintln!("compiler::e2e_*: built without `mlir-build`; end-to-end tiers not compiled in.");
    }
}

#[cfg(feature = "mlir-build")]
mod e2e {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::Duration;

    use criterion::{BatchSize, BenchmarkId, Criterion, SamplingMode, black_box};
    use tempfile::TempDir;

    /// T2 cdylib entries the `--emit-shared` path lowers end to end. Full `pub fn`
    /// programs, three sizes so the per-shape codegen+LLVM cost is visible.
    const T2_SHAPES: [(&str, &str); 3] = [
        ("small", "pub fn add(a: i64, b: i64) -> i64 { a + b }\n"),
        (
            "medium",
            "pub fn poly(x: i64) -> i64 {\n    let a = x * x;\n    let b = a * x;\n    \
             let c = b + a;\n    c + x * 7 - 3\n}\n",
        ),
        (
            "large",
            "pub fn f0(x: i64) -> i64 { x * x + 1 }\n\
             pub fn f1(x: i64) -> i64 { let a = f0(x); a * a - x }\n\
             pub fn f2(x: i64) -> i64 { let a = f1(x); let b = f0(a); a + b }\n\
             pub fn f3(x: i64) -> i64 { let a = f2(x); let b = f1(a); a * 3 + b - x }\n\
             pub fn driver(x: i64) -> i64 {\n    let a = f3(x);\n    let b = f2(a);\n    \
             let c = f1(b);\n    a + b + c\n}\n",
        ),
    ];

    /// Native-profile programs: the pinned `backend_native_bridge` corpus. Sources
    /// and byte oracles are read from the tree, never restated here.
    const NATIVE_SHAPES: [&str; 4] = ["const_return", "add", "local_call", "while_sum"];

    /// The equivalent C and Rust program for each shape. Arithmetic is wrapping on
    /// all three sides (MIND i64 is defined-wrap, C `-fwrapv`, Rust
    /// `overflow-checks=off`); a native program's `main` becomes an exported
    /// `entry` because the reference artefact is a shared object.
    fn reference_sources(shape: &str) -> Option<(&'static str, &'static str)> {
        Some(match shape {
            "small" => (
                "long long add(long long a, long long b) { return a + b; }\n",
                "#[no_mangle]\npub extern \"C\" fn add(a: i64, b: i64) -> i64 { a + b }\n",
            ),
            "medium" => (
                "long long poly(long long x) {\n    long long a = x * x;\n    \
                 long long b = a * x;\n    long long c = b + a;\n    return c + x * 7 - 3;\n}\n",
                "#[no_mangle]\npub extern \"C\" fn poly(x: i64) -> i64 {\n    let a = x * x;\n    \
                 let b = a * x;\n    let c = b + a;\n    c + x * 7 - 3\n}\n",
            ),
            "large" => (
                "long long f0(long long x) { return x * x + 1; }\n\
                 long long f1(long long x) { long long a = f0(x); return a * a - x; }\n\
                 long long f2(long long x) { long long a = f1(x); long long b = f0(a); return a + b; }\n\
                 long long f3(long long x) { long long a = f2(x); long long b = f1(a); \
                 return a * 3 + b - x; }\n\
                 long long driver(long long x) {\n    long long a = f3(x);\n    \
                 long long b = f2(a);\n    long long c = f1(b);\n    return a + b + c;\n}\n",
                "#[no_mangle]\npub extern \"C\" fn f0(x: i64) -> i64 { x * x + 1 }\n\
                 #[no_mangle]\npub extern \"C\" fn f1(x: i64) -> i64 { let a = f0(x); a * a - x }\n\
                 #[no_mangle]\npub extern \"C\" fn f2(x: i64) -> i64 { let a = f1(x); let b = f0(a); a + b }\n\
                 #[no_mangle]\npub extern \"C\" fn f3(x: i64) -> i64 { let a = f2(x); let b = f1(a); \
                 a * 3 + b - x }\n\
                 #[no_mangle]\npub extern \"C\" fn driver(x: i64) -> i64 {\n    let a = f3(x);\n    \
                 let b = f2(a);\n    let c = f1(b);\n    a + b + c\n}\n",
            ),
            "const_return" => (
                "long long entry(void) { return 7; }\n",
                "#[no_mangle]\npub extern \"C\" fn entry() -> i64 { 7 }\n",
            ),
            "add" => (
                "static long long add(long long a, long long b) { return a + b; }\n\
                 long long entry(void) { return add(2, 3); }\n",
                "fn add(a: i64, b: i64) -> i64 { a + b }\n\
                 #[no_mangle]\npub extern \"C\" fn entry() -> i64 { add(2, 3) }\n",
            ),
            "local_call" => (
                "static long long helper(long long x) { return x + 1; }\n\
                 long long entry(void) { return helper(6); }\n",
                "fn helper(x: i64) -> i64 { x + 1 }\n\
                 #[no_mangle]\npub extern \"C\" fn entry() -> i64 { helper(6) }\n",
            ),
            "while_sum" => (
                "long long entry(void) {\n    long long s = 0;\n    long long i = 0;\n    \
                 while (i < 10) {\n        s = s + i;\n        i = i + 1;\n    }\n    return s;\n}\n",
                "#[no_mangle]\npub extern \"C\" fn entry() -> i64 {\n    let mut s: i64 = 0;\n    \
                 let mut i: i64 = 0;\n    while i < 10 {\n        s += i;\n        i += 1;\n    }\n    s\n}\n",
            ),
            _ => return None,
        })
    }

    /// Reference flags. clang gets exactly the optimisation/codegen flags `mindc`
    /// passes its own clang link step (`-O3 -march=x86-64-v3 -ffp-contract=off`,
    /// shared PIC, pthread) plus `-fwrapv`; rustc gets `-C opt-level=3`, edition
    /// 2021, overflow checks off, `cdylib`. Same set as the recorded T2 sweep.
    const CLANG_FLAGS: [&str; 8] = [
        "-O3",
        "-march=x86-64-v3",
        "-ffp-contract=off",
        "-fwrapv",
        "-shared",
        "-fPIC",
        "-pthread",
        "-o",
    ];
    const RUSTC_FLAGS: [&str; 9] = [
        "--edition",
        "2021",
        "-C",
        "opt-level=3",
        "-C",
        "overflow-checks=off",
        "--crate-type",
        "cdylib",
        "-o",
    ];

    fn manifest_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// The `mindc` under test: `MINDC` when set (an out-of-tree binary), else the
    /// binary Cargo built for this bench. `CARGO_BIN_EXE_mindc` follows
    /// `CARGO_TARGET_DIR` and the bench profile, so the clock always times the
    /// compiler built from this tree, never a stale `target/debug` leftover.
    fn mindc_path() -> PathBuf {
        match std::env::var_os("MINDC") {
            Some(p) if !p.is_empty() => PathBuf::from(p),
            _ => PathBuf::from(env!("CARGO_BIN_EXE_mindc")),
        }
    }

    /// The real `rustc`, not the rustup proxy: the proxy adds a process hop that
    /// would inflate the reference and flatter MIND.
    fn real_rustc() -> Option<PathBuf> {
        let proxy = which::which("rustc").ok()?;
        let out = Command::new(&proxy)
            .args(["--print", "sysroot"])
            .output()
            .ok()?;
        let sysroot = String::from_utf8(out.stdout).ok()?;
        let real = Path::new(sysroot.trim()).join("bin").join("rustc");
        Some(if real.is_file() { real } else { proxy })
    }

    fn run_ok(cmd: &mut Command) -> bool {
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// One `mindc src --emit-shared out.so` with the runtime-object cache rooted at
    /// `xdg`. `MINDC_RUNTIME_OBJ_CACHE=0` disables that cache on compilers that
    /// honour it; the fresh `XDG_CACHE_HOME` makes the run cold on every compiler.
    fn emit_shared(mindc: &Path, src: &Path, out: &Path, xdg: &Path, cold: bool) -> bool {
        let mut cmd = Command::new(mindc);
        cmd.arg(src)
            .arg("--emit-shared")
            .arg(out)
            .env("XDG_CACHE_HOME", xdg);
        if cold {
            cmd.env("MINDC_RUNTIME_OBJ_CACHE", "0");
        }
        run_ok(&mut cmd)
    }

    /// `mindc build --backend native src --out elf`, from the repo root: the native
    /// bridge resolves `stage1.elf` and `std/` relative to the working directory.
    fn build_native(mindc: &Path, src: &Path, out: &Path) -> bool {
        run_ok(
            Command::new(mindc)
                .current_dir(manifest_dir())
                .args(["build", "--backend", "native"])
                .arg(src)
                .arg("--out")
                .arg(out),
        )
    }

    fn read(path: &Path) -> Vec<u8> {
        std::fs::read(path).unwrap_or_default()
    }

    fn tmp() -> TempDir {
        tempfile::Builder::new()
            .prefix("mind-bench-e2e-")
            .tempdir()
            .expect("compiler::e2e: cannot create a temporary directory")
    }

    /// A process-per-sample group: flat sampling (one spawn per iteration, no
    /// linear ramp), ten samples for the expensive tiers.
    fn group_for<'a>(
        c: &'a mut Criterion,
        name: &str,
        samples: usize,
    ) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
        let mut g = c.benchmark_group(name);
        g.sampling_mode(SamplingMode::Flat);
        g.sample_size(samples);
        g.warm_up_time(Duration::from_secs(3));
        g
    }

    struct Tools {
        mindc: PathBuf,
        t2: bool,
        native: bool,
        clang: Option<PathBuf>,
        rustc: Option<PathBuf>,
    }

    fn probe() -> Option<Tools> {
        let mindc = mindc_path();
        if !mindc.is_file() {
            eprintln!(
                "compiler::e2e: mindc not found at {}; set MINDC or build with \
                 `--features \"mlir-build std-surface cross-module-imports\"`. \
                 End-to-end tiers skipped.",
                mindc.display()
            );
            return None;
        }
        let missing: Vec<&str> = ["mlir-opt", "mlir-translate", "clang"]
            .into_iter()
            .filter(|t| which::which(t).is_err())
            .collect();
        if !missing.is_empty() {
            eprintln!("compiler::e2e_compile_shared: {missing:?} not on PATH; T2 sweep skipped");
        }
        let stage1 = manifest_dir().join("examples/mindc_mind/testdata/selfhost_loop/stage1.elf");
        let native = stage1.is_file() && manifest_dir().join("std").is_dir();
        if !native {
            eprintln!("compiler::e2e_native: stage1.elf or std/ absent; native tier skipped");
        }
        let clang = which::which("clang").ok();
        let rustc = real_rustc();
        if clang.is_none() {
            eprintln!("compiler::e2e_reference: clang absent; clang_c_to_so rows skipped");
        }
        if rustc.is_none() {
            eprintln!("compiler::e2e_reference: rustc absent; rustc_cdylib rows skipped");
        }
        Some(Tools {
            mindc,
            t2: missing.is_empty(),
            native,
            clang,
            rustc,
        })
    }

    pub(super) fn run(c: &mut Criterion) {
        let Some(tools) = probe() else { return };
        // Interleaved by shape: each MIND row is followed immediately by its C and
        // Rust equivalents, so a pair compared by `--require-faster` is measured
        // seconds apart under the same box load.
        for (shape, source) in T2_SHAPES {
            if tools.t2 {
                bench_t2_shape(c, &tools.mindc, shape, source);
            }
            bench_reference(c, &tools, shape);
        }
        for shape in NATIVE_SHAPES {
            if tools.native {
                bench_native_shape(c, &tools.mindc, shape);
            }
            bench_reference(c, &tools, shape);
        }
    }

    /// **T2 — process wall-clock, end to end**: `mindc src --emit-shared out.so`
    /// as a spawned process (frontend + MLIR + `mlir-opt` + LLVM + `clang` link,
    /// process startup included), reported as the mandatory pair:
    /// `warm` (runtime-object cache primed) and `cold` (a fresh cache per sample).
    fn bench_t2_shape(c: &mut Criterion, mindc: &Path, shape: &str, source: &str) {
        let work = tmp();
        let src = work.path().join(format!("{shape}.mind"));
        std::fs::write(&src, source).expect("compiler::e2e: cannot stage the T2 source");
        let warm_xdg = work.path().join("xdg-warm");
        std::fs::create_dir(&warm_xdg).expect("compiler::e2e: cannot create the warm cache");
        let warm_out = work.path().join("warm.so");
        let cold_xdg = tmp();
        let cold_out = work.path().join("cold.so");

        // Untimed pre-check: one cold build, then two warm builds (the first fills
        // the cache, the second hits it). A failing compile is not a timing result.
        // A cache that changes the emitted bytes is a wedge break, not a skip.
        if !emit_shared(mindc, &src, &cold_out, cold_xdg.path(), true)
            || !emit_shared(mindc, &src, &warm_out, &warm_xdg, false)
        {
            eprintln!(
                "compiler::e2e_compile_shared: {shape} did not emit a shared object; skipped"
            );
            return;
        }
        let first_warm = read(&warm_out);
        assert!(
            emit_shared(mindc, &src, &warm_out, &warm_xdg, false),
            "compiler::e2e_compile_shared: {shape} warm rebuild failed"
        );
        let (cold_bytes, warm_bytes) = (read(&cold_out), read(&warm_out));
        assert!(
            !cold_bytes.is_empty() && cold_bytes == first_warm && cold_bytes == warm_bytes,
            "compiler::e2e_compile_shared: {shape}: cold, cache-filling and cache-hit builds \
             emitted different bytes. A cache must never change output."
        );

        let mut g = group_for(c, "e2e_compile_shared", 20);
        g.bench_function(BenchmarkId::new("warm", shape), |b| {
            b.iter(|| {
                assert!(
                    emit_shared(black_box(mindc), &src, &warm_out, &warm_xdg, false),
                    "mindc --emit-shared failed inside the timed loop (warm)"
                );
            });
        });
        g.finish();

        let mut g = group_for(c, "e2e_compile_shared", 10);
        g.bench_function(BenchmarkId::new("cold", shape), |b| {
            b.iter_batched(
                tmp,
                |xdg| {
                    assert!(
                        emit_shared(black_box(mindc), &src, &cold_out, xdg.path(), true),
                        "mindc --emit-shared failed inside the timed loop (cold)"
                    );
                    xdg // dropped (and deleted) outside the timed region
                },
                BatchSize::PerIteration,
            );
        });
        g.finish();
    }

    /// **T2-native**: `mindc build --backend native` to a runnable ELF, no LLVM.
    /// The untimed pre-check pins the output to the committed byte oracle.
    fn bench_native_shape(c: &mut Criterion, mindc: &Path, shape: &str) {
        let corpus = manifest_dir().join("examples/mindc_mind/testdata/backend_native_bridge");
        let src = corpus.join(format!("{shape}.mind"));
        let oracle = read(&corpus.join(format!("{shape}.elf")));
        let work = tmp();
        let out = work.path().join(format!("{shape}.elf"));
        if !build_native(mindc, &src, &out) {
            eprintln!("compiler::e2e_native: {shape} did not build; skipped (not a timing result)");
            return;
        }
        assert!(
            !oracle.is_empty() && read(&out) == oracle,
            "compiler::e2e_native: {shape} is not byte-identical to its committed oracle"
        );

        let mut g = group_for(c, "e2e_native", 10);
        g.bench_function(shape, |b| {
            b.iter(|| {
                assert!(
                    build_native(black_box(mindc), &src, &out),
                    "mindc build --backend native failed inside the timed loop"
                );
            });
        });
        g.finish();
    }

    /// Same-session reference rows: the equivalent C and Rust program for `shape`
    /// through `clang` and `rustc`, each self-skipping when its tool is absent.
    fn bench_reference(c: &mut Criterion, tools: &Tools, shape: &str) {
        let Some((c_src, rs_src)) = reference_sources(shape) else {
            return;
        };
        let work = tmp();
        let rows = [
            (
                "clang_c_to_so",
                tools.clang.as_ref(),
                "c",
                c_src,
                &CLANG_FLAGS[..],
            ),
            (
                "rustc_cdylib",
                tools.rustc.as_ref(),
                "rs",
                rs_src,
                &RUSTC_FLAGS[..],
            ),
        ];
        for (row, tool, ext, text, flags) in rows {
            let Some(tool) = tool else { continue };
            let src = work.path().join(format!("{shape}_{row}.{ext}"));
            std::fs::write(&src, text).expect("compiler::e2e: cannot stage a reference source");
            let out = work.path().join(format!("{shape}_{row}.so"));
            // rustc writes its temporaries next to `-o`; keep them in `work`.
            let compile = |out: &Path| {
                run_ok(
                    Command::new(tool)
                        .current_dir(work.path())
                        .args(&flags[..flags.len() - 1])
                        .arg(&src)
                        .arg(flags[flags.len() - 1])
                        .arg(out),
                )
            };
            if !compile(&out) {
                eprintln!("compiler::e2e_reference: {row}/{shape} did not compile; skipped");
                continue;
            }
            let mut g = group_for(c, "e2e_reference", 10);
            g.bench_function(BenchmarkId::new(row, shape), |b| {
                b.iter(|| assert!(compile(black_box(&out)), "{row} failed in the timed loop"));
            });
            g.finish();
        }
    }
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
                lower_to_ir(&module).expect("lowering")
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
    bench_e2e,
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
    bench_e2e,
    bench_c_export_lowering
);

#[cfg(all(not(feature = "mlir-lowering"), feature = "autodiff"))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_wedge_evidence,
    bench_e2e,
    bench_autodiff_generation,
    bench_c_export_lowering
);

#[cfg(all(not(feature = "mlir-lowering"), not(feature = "autodiff")))]
criterion_group!(
    benches,
    bench_compilation_pipeline,
    bench_wedge_evidence,
    bench_e2e,
    bench_c_export_lowering
);

criterion_main!(benches);
