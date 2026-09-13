// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Static-shape tensor-parameter memref-descriptor ABI: field-order EXECUTION
//! controls (RFC 0012 §3.4.1).
//!
//! An LLVM/C signature assertion catches a change in the descriptor's field
//! COUNT or TYPE, but the offset/size/stride fields are all the same integer
//! type, so a PERMUTATION among them (size0<->size1, stride0<->stride1,
//! offset<->a stride) is invisible to a signature check. These controls EXECUTE
//! the boundary through ctypes with:
//!   * unequal sizes (rank-2: 2 x 3, so size0 != size1),
//!   * a non-zero element offset,
//!   * non-unit and mutually-unequal strides,
//!   * element values = powers of two, so the reduced scalar is a bitset
//!     fingerprint of EXACTLY which physical indices were read — any
//!     transposition of the size/stride/offset fields reads a different index
//!     set and yields a different sum.
//! Rank-1 and rank-2 are kept as distinct observations. A scalar positive
//! control proves the ctypes harness + build path are not vacuously green.
//!
//! The descriptor (x86-64 SysV LP64, LLVM/MLIR 20.x), per RFC 0012 §3.4.1:
//!   (ptr alloc, ptr aligned, i64 offset, i64 size_0.., i64 stride_0..) -> f64
//! `aligned` is dereferenced; `offset`/`stride_i` are in ELEMENTS; element
//! [i0..] is read at aligned[offset + Σ i_k*stride_k] — logical indexing, NOT a
//! contiguity assumption.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test tensor_param_descriptor_abi_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

// A rank-1 and a rank-2 static-shape tensor param reduced to a scalar, plus a
// scalar positive control. Runtime-fed descriptors defeat constant-folding.
const SRC: &str = r#"
pub fn rsum1(t: tensor<f64[3]>) -> f64 {
    return t.sum()
}
pub fn rsum2(t: tensor<f64[2,3]>) -> f64 {
    return t.sum()
}
pub fn sc(x: f64) -> f64 {
    return x + x
}
"#;

fn emit(so: &std::path::Path) -> bool {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "tensor_param_descriptor_abi_run",
            "mindc not found; skipping",
        );
        return false;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_tensor_param_descriptor_abi.mind");
    let _ = std::fs::remove_file(so);
    std::fs::write(&src, SRC).expect("write src");
    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !crate::common::gate::compiled("tensor_param_descriptor_abi_run", &out) {
        return false;
    }
    assert!(so.exists(), "no `.so` written despite success");
    true
}

/// Rank-1: offset=2, stride=3 over a 16-wide 2^k buffer reads buf[2],buf[5],buf[8]
/// = 4+32+256 = 292. A contiguity assumption (stride 1) would read
/// buf[2..5]=4+8+16=28; an offset/stride swap reads buf[3],buf[5],buf[7]=168.
#[test]
fn rank1_descriptor_offset_and_stride_are_honored() {
    let so = std::env::temp_dir().join("mind_tpd_rank1.so");
    if !emit(&so) {
        return;
    }
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f = lib.rsum1\n\
         f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong]\n\
         f.restype = ctypes.c_double\n\
         buf = (ctypes.c_double * 16)(*[float(2**k) for k in range(16)])\n\
         p = ctypes.cast(buf, ctypes.c_void_p)\n\
         r = f(p, p, 2, 3, 3)\n\
         assert r == 292.0, 'rsum1=' + repr(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "rank1 descriptor control failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Rank-2: unequal sizes 2x3, offset=1, unequal strides (4,1) over the 2^k
/// buffer reads buf[1,2,3] + buf[5,6,7] = (2+4+8)+(32+64+128) = 238. A size swap
/// (3x2) reads buf up to index 10 => 1638; a stride swap (1,4) => 1638; an
/// offset misread shifts the whole set. 238 is unique to the correct field map.
#[test]
fn rank2_unequal_sizes_offset_and_unequal_strides_are_honored() {
    let so = std::env::temp_dir().join("mind_tpd_rank2.so");
    if !emit(&so) {
        return;
    }
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f = lib.rsum2\n\
         f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong]\n\
         f.restype = ctypes.c_double\n\
         buf = (ctypes.c_double * 16)(*[float(2**k) for k in range(16)])\n\
         p = ctypes.cast(buf, ctypes.c_void_p)\n\
         r = f(p, p, 1, 2, 3, 4, 1)\n\
         assert r == 238.0, 'rsum2=' + repr(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "rank2 descriptor control failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Scalar positive control: the harness + build path produce a correct
/// non-tensor artifact, so a green rank-1/rank-2 above is not a vacuous pass.
#[test]
fn scalar_positive_control_runs() {
    let so = std::env::temp_dir().join("mind_tpd_scalar.so");
    if !emit(&so) {
        return;
    }
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f = lib.sc\n\
         f.argtypes = [ctypes.c_double]\n\
         f.restype = ctypes.c_double\n\
         r = f(21.0)\n\
         assert r == 42.0, 'sc=' + repr(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "scalar positive control failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
