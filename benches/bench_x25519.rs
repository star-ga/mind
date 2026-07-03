// Copyright 2025-2026 STARGA Inc.
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
//
// Part of the MIND project (Machine Intelligence Native Design).

//! X25519 (Curve25519 Montgomery-ladder ECDH) — scalar-multiplication
//! throughput benchmark for the pure-MIND `std/x25519.mind` module.
//!
//! ## What is timed
//!
//! One `x25519(scalar, u, out)` call per iteration — a full 255-bit constant-time
//! Montgomery ladder (the RFC 7748 §5 primitive that serves both directions of
//! ECDH: `pub = X25519(priv, 9)` and `shared = X25519(priv, peer_pub)`). The
//! metric is **operations/second** (one scalar multiply per iter), so this uses
//! plain criterion iters, *not* `Throughput::Bytes` — a scalar-mult is one
//! atomic op regardless of the fixed 32-byte operand size.
//!
//! ## Self-contained, self-skipping, correctness-gated
//!
//! `std/x25519.mind` has **no imports** — only `__mind_*` intrinsics — so the
//! `.so` is built with the plain self-contained recipe (no combine chain):
//! `mindc std/x25519.mind --emit-shared <out>.so`. mindc's `--emit-shared` path
//! links with `-Wl,-Bsymbolic-functions` (src/eval/mlir_build.rs), which makes
//! the module immune to ELF symbol interposition — required here because this
//! criterion binary links libm, whose weak C23 `fmul`/`fadd`/… would otherwise
//! capture the field arithmetic's intra-module calls once the `.so` is dlopen'd
//! (the exact hazard documented in the module header).
//!
//! Like the gated test harnesses, it self-skips (registers no benchmarks, exits
//! clean) when the MLIR toolchain (`mlir-opt` / `mlir-translate` / `clang`) is
//! shadowed or `mindc` is not built — it never panics for a *missing* toolchain.
//! It *does* panic (fails the run) if the built ladder produces the wrong bytes:
//! before timing, it verifies the RFC 7748 §5.2 vector-1 known-answer, so a
//! throughput number for a miscompiled ladder can never be reported as clean.
//!
//! The `.so` is compiled **once** (lazy `OnceLock`); the input buffers are
//! allocated once outside the loop; the timed closure calls only `x25519` in
//! `black_box`.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_x25519 --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds
//! the debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

/// X25519 ABI (std/x25519.mind, "Public API"):
/// `x25519(scalar_addr, u_addr, out_addr) -> 0`.
///   - `scalar_addr`, `u_addr`: 32-byte little-endian input buffers.
///   - `out_addr`: 32-byte output buffer (shared secret / public key).
/// Byte-for-byte the ctypes signature the `x25519_vectors_driver.py` reference
/// driver binds (`restype=c_int64`, `argtypes=[c_int64]*3`).
type X25519Fn = unsafe extern "C" fn(i64, i64, i64) -> i64;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn mindc_path() -> Option<PathBuf> {
    let dbg = manifest_dir().join("target").join("debug").join("mindc");
    if dbg.exists() {
        return Some(dbg);
    }
    let rel = manifest_dir().join("target").join("release").join("mindc");
    if rel.exists() { Some(rel) } else { None }
}

/// Compile `std/x25519.mind` to a temp `.so` once. Returns `None` (self-skip) if
/// the MLIR toolchain is shadowed or `mindc` is not built — same contract as the
/// gated test harnesses. Self-contained: no import stripping / cat-combine chain,
/// because the module has no `import`s.
fn build_x25519_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_x25519: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_x25519: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let src_path = manifest_dir().join("std").join("x25519.mind");
        if !src_path.exists() {
            eprintln!(
                "bench_x25519: source {} not found; skipping",
                src_path.display()
            );
            return None;
        }
        let so_path = std::env::temp_dir().join("mind_bench_x25519.so");
        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status();
        match status {
            Ok(s) if s.success() => Some(so_path),
            _ => {
                eprintln!("bench_x25519: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Decode a lowercase hex string to bytes (run-once helper for the fixed test
/// vectors; not on the timed path).
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex vector"))
        .collect()
}

/// One X25519 scalar multiply: `x25519(scalar, u) -> out` (32 bytes).
fn run_x25519(f: &Symbol<X25519Fn>, scalar: &[u8], u: &[u8], out: &mut [u8]) -> i64 {
    // `**f` copies out the raw `extern "C"` fn pointer (Symbol derefs to it),
    // avoiding any reliance on call-position auto-deref through `&Symbol`.
    let func: X25519Fn = **f;
    unsafe {
        func(
            scalar.as_ptr() as i64,
            u.as_ptr() as i64,
            out.as_mut_ptr() as i64,
        )
    }
}

/// Correctness gate wired to the RFC 7748 §5.2 vector-1 known-answer (the same
/// vector the Python reference driver checks first, itself cross-verified there
/// against pyca/cryptography). Panics (fails the bench run) on any drift, so a
/// lowering regression that changed the shared-secret bytes could never be
/// reported as a clean ops/sec number.
fn assert_rfc7748_vector1(f: &Symbol<X25519Fn>) {
    // RFC 7748 §5.2 vector 1.
    let scalar = unhex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
    let u = unhex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
    let expected = unhex("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552");
    let mut out = vec![0u8; 32];
    let rc = run_x25519(f, &scalar, &u, &mut out);
    assert_eq!(rc, 0, "x25519 returned {rc} (expected 0)");
    assert_eq!(
        out,
        expected,
        "x25519 output drifted from RFC 7748 §5.2 vector 1.\n computed={}\n expected={}\n\
         A benchmark of a miscompiled Montgomery ladder is not a measurement.",
        hex(&out),
        hex(&expected)
    );
    eprintln!(
        "bench_x25519: RFC 7748 §5.2 vector-1 known-answer VERIFIED (shared secret {}…)",
        &hex(&out)[..16]
    );
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn bench_x25519(c: &mut Criterion) {
    let Some(so) = build_x25519_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("bench_x25519: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen x25519 .so") };
    let x25519: Symbol<X25519Fn> = unsafe { lib.get(b"x25519").expect("x25519 symbol") };

    // Correctness gate first — panics on any byte drift.
    assert_rfc7748_vector1(&x25519);

    // Timed inputs: the verified RFC 7748 §5.2 vector-1 operands. The ladder runs
    // the full 255 bits regardless of the scalar/point values, so this is a
    // representative full scalar-multiply. Buffers allocated once, outside the loop.
    let scalar = unhex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
    let u = unhex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
    let mut out = vec![0u8; 32];

    let mut group = c.benchmark_group("bench_x25519");
    // One scalar multiply per iteration — ops/sec axis (no Throughput::Bytes: a
    // scalar-mult is one atomic operation, not a byte-stream transform).
    group.bench_function("scalarmult", |b| {
        b.iter(|| {
            let rc = unsafe {
                x25519(
                    black_box(scalar.as_ptr() as i64),
                    black_box(u.as_ptr() as i64),
                    black_box(out.as_mut_ptr() as i64),
                )
            };
            black_box(rc);
        });
    });
    group.finish();
}

criterion_group! {
    name = bench_x25519_grp;
    // Scalar-mult is ~tens of µs; give it real warm-up + measurement so ops/sec
    // lands with a tight CI rather than a 5s quick-run estimate.
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8))
        .sample_size(100);
    targets = bench_x25519
}
criterion_main!(bench_x25519_grp);
