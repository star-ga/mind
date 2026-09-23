// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! examples/quant/* cross-substrate byte-identity fixtures: kernel ABIs, the
//! per-example `.so` builder, run-to-run stability, and the corpus-level checks.

use super::*;
mod gates;
mod oracle;
use gates::*;

// --- examples/quant/* kernel ABIs -------------------------------------------
// Every quant example is a self-contained package (its own Mind.toml, no
// cross-module f64 import — mindc 0.10.2 cannot carry an f64 across a module
// boundary, roadmap 17.1) built to its OWN `.so` via `build_quant_so`, the
// generic sibling of `build_lorenz_so`/`build_collatz_so` for the examples/
// tree. Each pricing kernel below is a pure scalar function (System V xmm/gpr
// ABI, f64 args before/after i64 args exactly as declared): no pointer, no
// buffer, matching the `ScalarF64Fn`/`ScalarCastFn` precedent above.
type QuantF64x6Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64;
type QuantF64x7Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, f64) -> f64;
/// bond_price(face, coupon_rate, ytm, freq: i64, n_periods: i64) -> f64.
type QuantBondPriceFn = unsafe extern "C" fn(f64, f64, f64, i64, i64) -> f64;
/// two_asset_vol(w1, w2, s1, s2, rho) -> f64.
type QuantF64x5Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64) -> f64;
/// bopm_american_put(s, k, r, q, sigma, t, n: i64) -> f64.
type QuantF64x6I64Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, i64) -> f64;
/// crr_price(s, k, r, q, sigma, t, n: i64, is_call: i64, american: i64) -> f64.
type QuantCrrPriceFn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, i64, i64, i64) -> f64;
/// mc_call(s, k, r, q, sigma, t, npath: i64, seed: i64) -> f64.
type QuantMcCallFn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, i64, i64) -> f64;

/// Compile `examples/quant/<name>/main.mind` to its OWN temp `.so`, memoized
/// per `name` (one `OnceLock`-backed cache shared by every quant fixture, so
/// each of the 10 quant examples is built exactly once for the whole test
/// binary regardless of how many tests reference it). Generic sibling of
/// `build_lorenz_so` for the examples/quant/ tree: same toolchain self-skip /
/// `MIND_BENCH_REQUIRE` fail-closed contract (RFC 0020 §10) — a quant fixture
/// can never pass vacuously off a shadowed MLIR toolchain any more than the
/// integer canaries can.
fn build_quant_so(name: &str) -> Option<PathBuf> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: OnceLock<Mutex<HashMap<String, Option<PathBuf>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("quant .so cache poisoned");
    if let Some(cached) = guard.get(name) {
        return cached.clone();
    }
    let result = (|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                crate::common::gate::skipped(
                    "cross_substrate_identity",
                    &format!(
                        "{tool} not on PATH (quant-{name}); install the MLIR \
                         toolchain (mlir-opt / mlir-translate / clang) on this runner"
                    ),
                );
                return None;
            }
        }
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("quant")
            .join(name)
            .join("main.mind");
        let so_path = scratch().join(format!("mind_xsi_quant_{name}.so"));
        let status = Command::new(mindc_bin())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("spawn mindc --emit-shared for quant/{name}: {e}"));
        assert!(
            status.success(),
            "mindc --emit-shared failed for the quant-{name} workload"
        );
        Some(so_path)
    })();
    guard.insert(name.to_string(), result.clone());
    result
}

/// The quant examples every test below reads, in stable order.
fn quant_example_dirs() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("quant");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("read {}: {e}", root.display()))
        .map(|entry| entry.expect("quant dir entry").path())
        .filter(|dir| dir.join("main.mind").is_file())
        .collect();
    dirs.sort();
    assert!(
        !dirs.is_empty(),
        "no quant examples under {}",
        root.display()
    );
    dirs
}

/// Brace-matched text of `fn name(...) { ... }` in `src`, or None if absent.
/// A `}` inside a nested block must not end the function early.
fn extract_mind_fn<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let start = src
        .lines()
        .scan(0usize, |offset, line| {
            let at = *offset;
            *offset += line.len() + 1;
            Some((at, line))
        })
        .find_map(|(at, line)| {
            let rest = line.strip_prefix("pub ").unwrap_or(line);
            let rest = rest.strip_prefix("fn ")?.strip_prefix(name)?;
            rest.trim_start().starts_with('(').then_some(at)
        })?;
    let open = start + src[start..].find('{')?;
    let mut depth = 0usize;
    for (offset, ch) in src[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[start..=open + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every quant example vendors its own copy of the deterministic math kernels
/// on purpose, so each stays readable and runnable alone. That only holds while
/// the copies are IDENTICAL: an edit to one example's `dm_erfc` would make two
/// examples price the same option differently, each with a green KAT, because
/// each KAT checks its own copy. An example may OMIT a kernel it does not use;
/// it may not define a different one.
#[test]
fn quant_shared_kernels_byte_identical_across_examples() {
    const SHARED_KERNELS: [&str; 11] = [
        "dm_exp",
        "dm_log",
        "dm_sqrt",
        "dm_erf",
        "dm_erfc",
        "dm_norm_cdf",
        "dm_norm_pdf",
        "absf",
        "canonical_qnan",
        "bs_d1",
        "bs_d2",
    ];
    let sources: Vec<(String, String)> = quant_example_dirs()
        .into_iter()
        .map(|dir| {
            let name = dir.file_name().unwrap().to_string_lossy().into_owned();
            let src = std::fs::read_to_string(dir.join("main.mind"))
                .unwrap_or_else(|e| panic!("read quant/{name}/main.mind: {e}"));
            (name, src)
        })
        .collect();

    let mut diverged = Vec::new();
    let mut shared = 0usize;
    for kernel in SHARED_KERNELS {
        let mut variants: std::collections::BTreeMap<&str, Vec<&str>> = Default::default();
        for (name, src) in &sources {
            if let Some(body) = extract_mind_fn(src, kernel) {
                variants.entry(body).or_default().push(name);
            }
        }
        if variants.values().map(Vec::len).sum::<usize>() > 1 {
            shared += 1;
        }
        if variants.len() > 1 {
            let owners: Vec<String> = variants.values().map(|v| v.join(", ")).collect();
            diverged.push(format!(
                "{kernel}: {} definitions [{}]",
                variants.len(),
                owners.join(" | ")
            ));
        }
    }
    assert!(
        diverged.is_empty(),
        "quant examples disagree about a shared kernel -- copy the intended \
         definition verbatim into every example that defines it:\n  {}",
        diverged.join("\n  ")
    );
    // Positive control: the check is vacuous if nothing is actually shared.
    assert!(
        shared >= 5,
        "only {shared} kernels are shared; the extractor found nothing to compare"
    );
}

/// Each quant example's `main` is a known-answer suite whose exit code is its
/// count of failed checks. Run every one from a fresh copy so a stale `target/`
/// can never report an earlier artifact's status.
///
/// x86_64 only: the examples' `mindc run` path links the host CPU runtime, which
/// is not built for aarch64 yet. The neon leg is covered by the byte-identity
/// fixtures above, which compile each kernel to a shared object instead.
/// deferred: would also run on aarch64, skipped because the CPU runtime is
/// x86_64-only -- upgrade path: drop this cfg once that runtime builds for ARM.
#[cfg(target_arch = "x86_64")]
#[test]
fn quant_examples_known_answer_suites_pass() {
    for tool in ["mlir-opt", "mlir-translate", "clang"] {
        if which::which(tool).is_err() {
            crate::common::gate::skipped(
                "cross_substrate_identity",
                &format!("{tool} not on PATH; quant known-answer suites cannot run"),
            );
            return;
        }
    }
    let mut failed = Vec::new();
    for dir in quant_example_dirs() {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let work = scratch().join(format!("quant_kat_{name}"));
        // A fresh copy per run: a stale target/ would report an earlier artifact's status.
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).unwrap_or_else(|e| panic!("create {}: {e}", work.display()));
        for file in ["Mind.toml", "main.mind"] {
            std::fs::copy(dir.join(file), work.join(file))
                .unwrap_or_else(|e| panic!("copy quant/{name}/{file}: {e}"));
        }
        let out = Command::new(mindc_bin())
            .arg("run")
            .current_dir(&work)
            .output()
            .unwrap_or_else(|e| panic!("spawn mindc run for quant/{name}: {e}"));
        if !out.status.success() {
            failed.push(format!(
                "{name}: exit {:?} (count of failed checks, or a build error)\n{}{}",
                out.status.code(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "quant known-answer suites red:\n{}",
        failed.join("\n")
    );
}

/// Canonical hash of a grid of f64 scalar results (manifest `output_encoding =
/// "f64_bits_i64_le_grid"`): each grid point's IEEE-754 bit pattern as an i64,
/// 8 little-endian bytes, in fixed grid order, concatenated, then sha256 — the
/// vector-of-scalars extension of `scalar-float-f64`'s single-value
/// `f64_bits_i64_le` encoding (used by every `quant-*` fixture).
fn canonical_hash_f64_grid(v: &[f64]) -> String {
    let mut h = Sha256::new();
    for &e in v {
        h.update((e.to_bits() as i64).to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// Run-to-run stability for the quant kernels, called from
/// `same_process_run_to_run_determinism` with that test's `stable` checker.
pub(super) fn run_to_run(stable: &dyn Fn(&str, &dyn Fn() -> String)) {
    // --- examples/quant/* pricing kernels — each in its OWN .so (build_quant_so).
    // Every grid point's f64 result is fixed and deterministic (no clock, no
    // OS entropy, no host RNG); a fresh Vec collects the grid each run so a
    // kernel that read prior buffer state or left an output uninitialised
    // would surface as a run-to-run divergence, same rationale as the buffer
    // canaries above.

    if let Some(so) = build_quant_so("black_scholes") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-black-scholes .so") };
        let f: Symbol<QuantF64x6Fn> = unsafe { lib.get(b"bs_call").expect("bs_call symbol") };
        stable("quant-black-scholes", &|| {
            let out: Vec<f64> = QUANT_BLACK_SCHOLES_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t)| unsafe { f(s, k, r, q, sigma, t) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("bond_curve") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-bond-curve .so") };
        let f: Symbol<QuantBondPriceFn> =
            unsafe { lib.get(b"bond_price").expect("bond_price symbol") };
        stable("quant-bond-curve", &|| {
            let out: Vec<f64> = QUANT_BOND_CURVE_GRID
                .iter()
                .map(|&(face, cr, ytm, freq, n)| unsafe { f(face, cr, ytm, freq, n) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("greeks_higher_order") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-greeks-higher-order .so") };
        let f: Symbol<QuantF64x6Fn> = unsafe { lib.get(b"hog_vanna").expect("hog_vanna symbol") };
        stable("quant-greeks-higher-order", &|| {
            let out: Vec<f64> = QUANT_BLACK_SCHOLES_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t)| unsafe { f(s, k, r, q, sigma, t) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("implied_vol_surface") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-implied-vol-surface .so") };
        let f: Symbol<QuantF64x6Fn> = unsafe {
            lib.get(b"svi_total_variance")
                .expect("svi_total_variance symbol")
        };
        stable("quant-implied-vol-surface", &|| {
            let out: Vec<f64> = QUANT_SVI_GRID
                .iter()
                .map(|&(k, a, b, rho, m, p_sigma)| unsafe { f(k, a, b, rho, m, p_sigma) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("portfolio_risk") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-portfolio-risk .so") };
        let f: Symbol<QuantF64x5Fn> =
            unsafe { lib.get(b"two_asset_vol").expect("two_asset_vol symbol") };
        stable("quant-portfolio-risk", &|| {
            let out: Vec<f64> = QUANT_PORTFOLIO_RISK_GRID
                .iter()
                .map(|&(w1, w2, s1, s2, rho)| unsafe { f(w1, w2, s1, s2, rho) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("american_binomial") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-american-binomial .so") };
        let f: Symbol<QuantF64x6I64Fn> = unsafe {
            lib.get(b"bopm_american_put")
                .expect("bopm_american_put symbol")
        };
        stable("quant-american-binomial", &|| {
            let out: Vec<f64> = QUANT_AMERICAN_BINOMIAL_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t, n)| unsafe { f(s, k, r, q, sigma, t, n) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("barrier_analytic") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-barrier-analytic .so") };
        let f: Symbol<QuantF64x7Fn> = unsafe { lib.get(b"bar_doc").expect("bar_doc symbol") };
        stable("quant-barrier-analytic", &|| {
            let out: Vec<f64> = QUANT_BARRIER_ANALYTIC_GRID
                .iter()
                .map(|&(s, k, h, r, q, sigma, t)| unsafe { f(s, k, h, r, q, sigma, t) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("crr_binomial") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-crr-binomial .so") };
        let f: Symbol<QuantCrrPriceFn> =
            unsafe { lib.get(b"crr_price").expect("crr_price symbol") };
        stable("quant-crr-binomial", &|| {
            let out: Vec<f64> = QUANT_CRR_BINOMIAL_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t, n, is_call, american)| unsafe {
                    f(s, k, r, q, sigma, t, n, is_call, american)
                })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("implied_vol") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-implied-vol .so") };
        let f: Symbol<QuantF64x6Fn> = unsafe {
            lib.get(b"implied_vol_call")
                .expect("implied_vol_call symbol")
        };
        stable("quant-implied-vol", &|| {
            let out: Vec<f64> = QUANT_IMPLIED_VOL_GRID
                .iter()
                .map(|&(price, s, k, r, q, t)| unsafe { f(price, s, k, r, q, t) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("monte_carlo_gbm") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-monte-carlo-gbm .so") };
        let f: Symbol<QuantMcCallFn> = unsafe { lib.get(b"mc_call").expect("mc_call symbol") };
        stable("quant-monte-carlo-gbm", &|| {
            let out: Vec<f64> = QUANT_MONTE_CARLO_GBM_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t, npath, seed)| unsafe {
                    f(s, k, r, q, sigma, t, npath, seed)
                })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }
}
