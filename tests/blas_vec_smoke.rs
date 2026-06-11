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

//! RFC 0006 Track B (increment 1) — native MLIR vector-dialect smoke
//! harness for `__mind_blas_dot_f32_v`.
//!
//! Unlike `blas_smoke.rs` (which clang-compiles the runtime-support C
//! bridge for Track A), this harness drives the *thesis-pure* path: it
//! invokes the `mindc` binary on a `.mind` source that calls the
//! `__mind_blas_dot_f32_v` intrinsic, which lowers to a `vector.load` /
//! `vector.fma` / `vector.reduction <add>` reduction loop *inside mindc*
//! — no runtime-support C shim, no `-fPIC` object, no external symbol.
//! `mlir-opt`'s `convert-vector-to-llvm` legalises the vector ops to the
//! host SIMD width.
//!
//! Numerical contract (mirrors std/blas.mind and Track A's AVX2 path):
//!   * f32: within 1e-4 relative of an f64-accumulating oracle on
//!     1024- and 1,000,000-element vectors (the tree-shaped pairwise
//!     `vector.reduction` reorders summation exactly like AVX2).
//!   * sub-lane lengths (< 8) take the pure scalar-tail path and are
//!     byte-identical to a sequential scalar reference.
//!
//! Gated: `cargo test --features "mlir-build std-surface
//!         cross-module-imports" --test blas_vec_smoke`.
//!
//! Self-skips on Windows to match `blas_smoke.rs` (the mind-blas
//! Windows-MSVC C-shim port is a tracked follow-on; the native vector
//! path is exercised fully on Linux/macOS).

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#![cfg(not(windows))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

/// Direct-intrinsic source: the working Track B codegen entry point.
/// The `Instr::Call` for `__mind_blas_dot_f32_v` is intercepted by the
/// MLIR lowering and emitted as a native `vector`-dialect reduction loop
/// (no `func.call`, no external symbol).
const SRC: &str = r#"
pub fn dotv(a: i64, b: i64, n: i64) -> i64 {
    __mind_blas_dot_f32_v(a, b, n)
}
"#;

type DotFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

fn mindc_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dbg = manifest_dir.join("target").join("debug").join("mindc");
    if dbg.exists() {
        return dbg;
    }
    let rel = manifest_dir.join("target").join("release").join("mindc");
    assert!(
        rel.exists(),
        "mindc binary not found at {dbg:?} or {rel:?}; build with: \
         cargo build --features \"mlir-build std-surface cross-module-imports\" \
         --bin mindc"
    );
    rel
}

/// Compile SRC to a temporary `.so` via the native vector-dialect path,
/// **exactly once** for the whole test binary (the `OnceLock` makes the
/// three tests share one build + one `.so` so they never race the temp-
/// file write/dlopen when `cargo test` runs them on parallel threads —
/// an intermittent "file too short" dlopen failure otherwise surfaces
/// under machine load).
///
/// Returns `None` if the MLIR toolchain (`mlir-opt` / `mlir-translate` /
/// `clang`) is not on PATH, in which case the test self-skips — building
/// mindc already needs clang, but the toolchain may be shadowed in some
/// CI sandboxes and the gated suite must not hard-fail there.
fn build_vec_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                println!("blas_vec_smoke: {tool} not on PATH; skipping");
                return None;
            }
        }

        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_blas_vec_smoke.mind");
        let so_path = dir.join("mind_blas_vec_smoke.so");
        std::fs::write(&src_path, SRC).expect("write test .mind source");

        let status = Command::new(mindc_path())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("spawn mindc");
        assert!(
            status.success(),
            "mindc --emit-shared failed for the Track B vector source"
        );
        Some(so_path)
    })
    .as_ref()
}

fn f32_from_packed(bits_i64: i64) -> f32 {
    f32::from_bits((bits_i64 as u64) as u32)
}

/// Deterministic LCG — identical construction to `blas_smoke.rs` so the
/// two harnesses exercise the same numeric distribution.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 16) as u32
    }
    fn next_f32_unit(&mut self) -> f32 {
        ((self.next_u32() as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    }
}

fn make_pair(len: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut g = Lcg::new(seed);
    let a: Vec<f32> = (0..len).map(|_| g.next_f32_unit()).collect();
    let b: Vec<f32> = (0..len).map(|_| g.next_f32_unit()).collect();
    (a, b)
}

/// f64-accumulating oracle — the true mathematical sum, the same
/// reference `blas_smoke.rs` uses to bound the AVX2 reduction-reorder
/// error.
fn ref_dot_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += (a[i] as f64) * (b[i] as f64);
    }
    acc
}

/// f32 sequential oracle — byte-identical to the scalar-tail path for
/// sub-lane lengths (no SIMD lanes engaged).
fn ref_dot_f32_seq(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn call_dotv(lib: &Library, a: &[f32], b: &[f32]) -> i64 {
    unsafe {
        let f: Symbol<DotFn> = lib
            .get(b"dotv\0")
            .expect("symbol dotv missing from Track B .so");
        f(a.as_ptr() as i64, b.as_ptr() as i64, a.len() as i64)
    }
}

#[test]
fn vec_dot_f32_within_1e4_rel_of_f64_oracle() {
    let Some(so) = build_vec_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen Track B .so") };

    // 1024 and 1,000,000 elements — the RFC-mandated equivalence sizes.
    for &len in &[1024usize, 1_000_000] {
        let (a, b) = make_pair(len, 0xDEAD_BEEF + len as u64);
        let got = f32_from_packed(call_dotv(&lib, &a, &b)) as f64;
        let truth = ref_dot_f64(&a, &b);
        let rel = (got - truth).abs() / truth.abs().max(1e-30);
        assert!(
            rel < 1e-4,
            "len={len}: native vector dot_f32 must be within 1e-4 relative \
             of the f64 oracle (substrate-accuracy floor); \
             rel={rel:e} got={got} truth={truth}"
        );
    }
}

#[test]
fn vec_dot_f32_byte_identical_to_scalar_below_one_lane_group() {
    // Lengths below the 8-lane SIMD width take the pure scalar-tail
    // path; the result is byte-identical to a sequential scalar sum.
    let Some(so) = build_vec_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen Track B .so") };

    for &len in &[0usize, 1, 2, 7] {
        let (a, b) = make_pair(len, 0x1234_5678 + len as u64);
        let got_bits = (call_dotv(&lib, &a, &b) as u64) as u32;
        let ref_bits = ref_dot_f32_seq(&a, &b).to_bits();
        assert_eq!(
            got_bits, ref_bits,
            "len={len}: sub-lane scalar-tail path must be byte-identical \
             to the sequential scalar reference"
        );
    }
}

#[test]
fn vec_dot_f32_handles_ragged_lengths() {
    // Lengths straddling lane-group boundaries exercise the
    // main-loop / scalar-tail split. All must stay within tolerance.
    let Some(so) = build_vec_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen Track B .so") };

    for &len in &[8usize, 9, 15, 16, 17, 31, 32, 33, 4096, 65537] {
        let (a, b) = make_pair(len, 0xBEEF_CAFE + len as u64);
        let got = f32_from_packed(call_dotv(&lib, &a, &b)) as f64;
        let truth = ref_dot_f64(&a, &b);
        let rel = (got - truth).abs() / truth.abs().max(1e-30);
        assert!(
            rel < 1e-4,
            "len={len}: ragged-length vector dot_f32 rel={rel:e} \
             got={got} truth={truth}"
        );
    }
}
