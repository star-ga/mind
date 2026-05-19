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

//! RFC 0006 Track B (increment 2) — native MLIR vector-dialect smoke
//! harness for the Q16.16 and f32 L1/L∞ vector paths.
//!
//! The headline assertion is the **cross-arch Q16.16 bit-identity gate
//! (task #57)** extended to the thesis-pure vector path:
//! `__mind_blas_dot_q16_v` (which lowers, inside mindc, to a
//! `vector<8xi64>` widen-multiply-arithmetic-shift-accumulate loop +
//! associative `vector.reduction <add>`) must return a **byte-identical**
//! i64 to the Track A scalar oracle `mind_blas_dot_q16_scalar`
//! (replicated here in Rust) at every length in
//! {0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1024, 4096, 65537}.
//! Q16.16 integer reduction is associative, so this is exact — not a
//! tolerance.
//!
//! The f32 L1/L∞ vector paths (`__mind_blas_dot_l1_f32_v` /
//! `__mind_blas_dot_linf_f32_v`) are checked within 1e-4 relative of an
//! f64 oracle, mirroring the increment-1 `dot_f32_v` contract (the
//! tree-shaped reduction reorders the f32 summation exactly like Track
//! A's AVX2 path; L∞ max is in fact byte-exact).
//!
//! Gated: `cargo test --features "mlir-build std-surface
//!         cross-module-imports" --test blas_vec_q16_smoke`.
//!
//! Self-skips on Windows to match `blas_smoke.rs` / `blas_vec_smoke.rs`
//! (the mind-blas Windows-MSVC C-shim port is a tracked follow-on; the
//! native vector path is exercised fully on Linux/macOS).

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
/// Each `Instr::Call` is intercepted by the MLIR lowering and emitted as
/// a native `vector`-dialect reduction loop (no `func.call`, no external
/// symbol, no runtime-support C shim).
const SRC: &str = r#"
pub fn dotq(a: i64, b: i64, n: i64) -> i64 {
    __mind_blas_dot_q16_v(a, b, n)
}
pub fn dotl1(a: i64, b: i64, n: i64) -> i64 {
    __mind_blas_dot_l1_f32_v(a, b, n)
}
pub fn dotlinf(a: i64, b: i64, n: i64) -> i64 {
    __mind_blas_dot_linf_f32_v(a, b, n)
}
"#;

type DotFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

/// Every RFC-mandated length for the cross-arch bit-identity gate.
const LENGTHS: &[usize] = &[0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1024, 4096, 65537];

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
/// **exactly once** for the whole test binary. The `OnceLock` is the
/// difference vs the increment-1 harness: all four tests share one
/// build + one `.so`, so they never race the temp-file write/dlopen
/// when criterion / `cargo test` runs them in parallel threads.
///
/// Returns `None` if the MLIR toolchain is not on PATH (the gated suite
/// must self-skip in sandboxes where the toolchain is shadowed, exactly
/// like `blas_vec_smoke.rs`).
fn build_vec_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                println!("blas_vec_q16_smoke: {tool} not on PATH; skipping");
                return None;
            }
        }

        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_blas_vec_q16_smoke.mind");
        let so_path = dir.join("mind_blas_vec_q16_smoke.so");
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
            "mindc --emit-shared failed for the Track B increment-2 vector source"
        );
        Some(so_path)
    })
    .as_ref()
}

fn f32_from_packed(bits_i64: i64) -> f32 {
    f32::from_bits((bits_i64 as u64) as u32)
}

/// Deterministic LCG — identical construction to `blas_smoke.rs` /
/// `blas_vec_smoke.rs` so all three harnesses share a numeric
/// distribution.
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
    /// Q16.16 fixed-point sample in roughly [-8, 8) — small enough that
    /// a 65537-element dot stays inside i32 after the final
    /// truncate-to-low-32 (matching the scalar oracle exactly either
    /// way; this just keeps the magnitudes representative).
    fn next_q16(&mut self) -> i32 {
        (self.next_u32() as i32) >> 12
    }
}

fn make_pair_f32(len: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut g = Lcg::new(seed);
    let a: Vec<f32> = (0..len).map(|_| g.next_f32_unit()).collect();
    let b: Vec<f32> = (0..len).map(|_| g.next_f32_unit()).collect();
    (a, b)
}

fn make_pair_q16(len: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut g = Lcg::new(seed);
    let a: Vec<i32> = (0..len).map(|_| g.next_q16()).collect();
    let b: Vec<i32> = (0..len).map(|_| g.next_q16()).collect();
    (a, b)
}

/// Track A scalar oracle, byte-for-byte. The `>> 16` is an arithmetic
/// shift on a signed i64 (Rust `>>` on `i64` is arithmetic); the final
/// `(i64)(i32)acc` is `acc as i32 as i64`. This is the exact function
/// the cross-arch bit-identity gate (#57) pins.
fn ref_dot_q16_scalar(a: &[i32], b: &[i32]) -> i64 {
    let mut acc: i64 = 0;
    for i in 0..a.len() {
        let prod = (a[i] as i64) * (b[i] as i64);
        acc += prod >> 16;
    }
    (acc as i32) as i64
}

fn ref_dot_l1_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += ((a[i] as f64) - (b[i] as f64)).abs();
    }
    acc
}

fn ref_dot_linf_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..a.len() {
        let d = ((a[i] as f64) - (b[i] as f64)).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn call3(lib: &Library, sym: &[u8], a_addr: i64, b_addr: i64, n: i64) -> i64 {
    unsafe {
        let f: Symbol<DotFn> = lib
            .get(sym)
            .unwrap_or_else(|_| panic!("symbol {sym:?} missing from Track B .so"));
        f(a_addr, b_addr, n)
    }
}

/// THE #57 GATE. Native vector Q16.16 dot == Track A scalar oracle,
/// byte-for-byte, at every RFC-mandated length. Not a tolerance — a
/// bit-exact equality, because Q16.16 integer reduction is associative
/// and the per-element arithmetic `>> 16` is replicated exactly.
#[test]
fn vec_dot_q16_byte_identical_to_scalar_oracle_all_lengths() {
    let Some(so) = build_vec_so() else {
        return;
    };
    let lib = unsafe { Library::new(&so).expect("dlopen Track B .so") };

    for &len in LENGTHS {
        let (a, b) = make_pair_q16(len, 0xC0FF_EE00 + len as u64);
        let got = call3(
            &lib,
            b"dotq\0",
            a.as_ptr() as i64,
            b.as_ptr() as i64,
            len as i64,
        );
        let expect = ref_dot_q16_scalar(&a, &b);
        assert_eq!(
            got, expect,
            "len={len}: native vector Q16.16 dot MUST be byte-identical \
             to the Track A scalar oracle (cross-arch bit-identity gate, \
             task #57); got={got} expect={expect}"
        );
    }
}

/// Empty / null-ish input must not trap and must agree with the oracle
/// (the scalar oracle returns 0 for len 0; the vector loop bounds are
/// `divui`-derived so the main loop and tail both execute zero times).
#[test]
fn vec_dot_q16_zero_length_is_zero() {
    let Some(so) = build_vec_so() else {
        return;
    };
    let lib = unsafe { Library::new(&so).expect("dlopen Track B .so") };
    let got = call3(&lib, b"dotq\0", 0, 0, 0);
    assert_eq!(got, 0, "len=0 Q16.16 vector dot must be 0");
}

#[test]
fn vec_dot_l1_f32_within_1e4_rel_of_f64_oracle() {
    let Some(so) = build_vec_so() else {
        return;
    };
    let lib = unsafe { Library::new(&so).expect("dlopen Track B .so") };

    for &len in &[1usize, 7, 8, 9, 16, 17, 1024, 4096, 65537] {
        let (a, b) = make_pair_f32(len, 0xABCD_0000 + len as u64);
        let got = f32_from_packed(call3(
            &lib,
            b"dotl1\0",
            a.as_ptr() as i64,
            b.as_ptr() as i64,
            len as i64,
        )) as f64;
        let truth = ref_dot_l1_f64(&a, &b);
        let rel = (got - truth).abs() / truth.abs().max(1e-30);
        assert!(
            rel < 1e-4,
            "len={len}: native vector L1 must be within 1e-4 relative of \
             the f64 oracle; rel={rel:e} got={got} truth={truth}"
        );
    }
}

#[test]
fn vec_dot_linf_f32_within_1e4_rel_of_f64_oracle() {
    let Some(so) = build_vec_so() else {
        return;
    };
    let lib = unsafe { Library::new(&so).expect("dlopen Track B .so") };

    for &len in &[1usize, 7, 8, 9, 16, 17, 1024, 4096, 65537] {
        let (a, b) = make_pair_f32(len, 0x9999_0000 + len as u64);
        let got = f32_from_packed(call3(
            &lib,
            b"dotlinf\0",
            a.as_ptr() as i64,
            b.as_ptr() as i64,
            len as i64,
        )) as f64;
        let truth = ref_dot_linf_f64(&a, &b);
        let rel = (got - truth).abs() / truth.abs().max(1e-30);
        assert!(
            rel < 1e-4,
            "len={len}: native vector L∞ must be within 1e-4 relative of \
             the f64 oracle; rel={rel:e} got={got} truth={truth}"
        );
    }
}
