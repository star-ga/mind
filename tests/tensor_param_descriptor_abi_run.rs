// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Static-shape tensor-parameter memref-descriptor ABI: field-order EXECUTION
//! controls (RFC 0012 §3.4.1).
//!
//! An LLVM/C signature assertion catches a change in the descriptor's field
//! COUNT or TYPE, but the offset/size/stride fields are all the same integer
//! type, so a PERMUTATION among them is invisible to a signature check. These
//! controls execute the boundary through ctypes with unequal, in-bounds
//! offset/size/stride values where the rank permits. The rank-2 reduction is
//! retained as a positive strided-read control; reductions are commutative, so
//! it is NOT a universal field-permutation oracle. An index-sensitive rank-2
//! weighted matrix operation distinguishes the tested stride-field mapping.
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
pub fn rpick2(t: tensor<f64[2,3]>, w: tensor<f64[3,1]>) -> f64 {
    return tensor.matmul(t, w).sum()
}
pub fn rmm4(a: tensor<f64[4,3]>, b: tensor<f64[3,4]>) -> f64 {
    return tensor.matmul(a, b).sum()
}
pub fn sc(x: f64) -> f64 {
    return x + x
}
"#;

fn emit(so: &std::path::Path, target: &str) -> bool {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "tensor_param_descriptor_abi_run",
            "mindc not found; skipping",
        );
        return false;
    }
    let dir = crate::common::scratch_dir(target);
    let src = dir.join("descriptor.mind");
    let _ = std::fs::remove_file(so);
    std::fs::write(&src, SRC).expect("write src");
    // GNU coreutils `timeout` is present on the Unix CI runners. Keep a
    // malformed compiler/toolchain from hanging this ABI gate indefinitely.
    let out = Command::new("timeout")
        .arg("600")
        .arg(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !crate::common::gate::compiled("tensor_param_descriptor_abi_run", &out) {
        return false;
    }
    let artifact = std::fs::read(so).expect("read emitted shared artifact");
    assert!(
        artifact.starts_with(b"\x7fELF"),
        "emission succeeded but {} is not an ELF shared artifact",
        so.display()
    );
    true
}

/// Rank-1: offset=2, size=3, stride=4 over a 16-wide 2^k buffer reads
/// buf[2],buf[6],buf[10] = 4+64+1024 = 1092. All three fields are distinct;
/// the alternate offset/stride map remains in bounds but reads different cells.
#[test]
fn rank1_descriptor_offset_and_stride_are_honored() {
    let dir = crate::common::scratch_dir("tensor-param-descriptor-rank1");
    let so = dir.join("artifact.so");
    if !emit(&so, "tensor-param-descriptor-rank1") {
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
         r = f(p, p, 2, 3, 4)\n\
         swapped = f(p, p, 4, 3, 2)\n\
         assert r == 1092.0, 'rsum1=' + repr(r)\n\
         assert swapped == 336.0, 'rsum1(swapped offset/stride)=' + repr(swapped)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("timeout")
        .args(["60", "python3"])
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

/// Rank-2 reduction: offset=1, sizes 2x3 and strides (5,2) reads indices
/// [1,3,5,6,8,10] in a 16-wide 2^k buffer, total 1386. The positive result
/// demonstrates that the declared static shape reaches a real strided read;
/// it does not claim to distinguish axis permutations of a commutative sum.
#[test]
fn rank2_unequal_sizes_offset_and_unequal_strides_are_honored() {
    let dir = crate::common::scratch_dir("tensor-param-descriptor-rank2-sum");
    let so = dir.join("artifact.so");
    if !emit(&so, "tensor-param-descriptor-rank2-sum") {
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
         r = f(p, p, 1, 2, 3, 5, 2)\n\
         assert r == 1386.0, 'rsum2=' + repr(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("timeout")
        .args(["60", "python3"])
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

/// Rank-2 index-sensitive control: a runtime weight tensor makes the one-hot
/// logical [1,2] contribution observable at physical index 10 under
/// (offset=1, strides=5,2). A stride-field permutation with the declared sizes
/// held at (2,3) (strides=2,5) reads physical indices [1,3,6,8,11,13]
/// instead. A mostly-one buffer with an extra unit at physical index 10 makes
/// the correct and permuted weighted sums 18 and 14 respectively, avoiding a
/// zero-result/vacuous control. Both tensor descriptors are passed through the
/// public static-shape C ABI; runtime size-field validation is not asserted.
#[test]
fn rank2_index_sensitive_field_map_is_honored() {
    let dir = crate::common::scratch_dir("tensor-param-descriptor-rank2-index");
    let so = dir.join("artifact.so");
    if !emit(&so, "tensor-param-descriptor-rank2-index") {
        return;
    }
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f = lib.rpick2\n\
         f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong,\n\
         \x20             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong]\n\
         f.restype = ctypes.c_double\n\
         buf = (ctypes.c_double * 16)(*[1.0 + (1.0 if k == 10 else 0.0) for k in range(16)])\n\
         p = ctypes.cast(buf, ctypes.c_void_p)\n\
         weights = (ctypes.c_double * 3)(1.0, 2.0, 4.0)\n\
         wp = ctypes.cast(weights, ctypes.c_void_p)\n\
         good = f(p, p, 1, 2, 3, 5, 2, wp, wp, 0, 3, 1, 1, 1)\n\
         permuted = f(p, p, 1, 2, 3, 2, 5, wp, wp, 0, 3, 1, 1, 1)\n\
         assert good == 18.0, 'rpick2(correct)=' + repr(good)\n\
         assert permuted == 14.0, 'rpick2(stride-permuted)=' + repr(permuted)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("timeout")
        .args(["60", "python3"])
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "rank2 index-sensitive control failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Accumulating-op init control. `linalg.matmul` computes `C += A*B` into its
/// `outs` operand, so that operand must start at zero. A bare `tensor.empty()`
/// init is uninitialised heap memory: whether the result is right then depends
/// on what the allocator hands back, which is why the rank-2 control above was
/// flaky rather than red. This control makes the heap DIRTY on purpose before
/// every call — malloc a spread of chunk sizes, fill them with 0x40 bytes
/// (0x4040404040404040 = 32.50196… as f64), free them — so a reused chunk feeds
/// a non-zero init to the matmul on every call. The 4x4 f64 output (128 bytes)
/// is larger than the allocator's 16-byte free-list header, so the dirty bytes
/// survive into the reused chunk. Measured on an init without the zero fill:
/// 64 of 64 calls wrong (1386.03… / 1321.03… for an expected 866).
#[test]
fn accumulating_matmul_ignores_a_dirty_output_allocation() {
    let dir = crate::common::scratch_dir("tensor-param-descriptor-dirty-init");
    let so = dir.join("artifact.so");
    if !emit(&so, "tensor-param-descriptor-dirty-init") {
        return;
    }
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         libc = ctypes.CDLL(None)\n\
         libc.malloc.restype = ctypes.c_void_p\n\
         libc.malloc.argtypes = [ctypes.c_size_t]\n\
         libc.free.argtypes = [ctypes.c_void_p]\n\
         libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]\n\
         D = ctypes.c_void_p\n\
         L = ctypes.c_longlong\n\
         f = lib.rmm4\n\
         f.argtypes = [D, D, L, L, L, L, L, D, D, L, L, L, L, L]\n\
         f.restype = ctypes.c_double\n\
         a = (ctypes.c_double * 12)(*[float(k + 1) for k in range(12)])\n\
         b = (ctypes.c_double * 12)(*[float((k % 5) + 1) for k in range(12)])\n\
         ap = ctypes.cast(a, D)\n\
         bp = ctypes.cast(b, D)\n\
         want = sum(a[i*3+k] * b[k*4+j] for i in range(4) for j in range(4) for k in range(3))\n\
         assert want == 866.0, 'oracle=' + repr(want)\n\
         sizes = range(16, 1024, 16)\n\
         wrong = []\n\
         for _ in range(64):\n\
         \x20   ps = [libc.malloc(s) for s in sizes]\n\
         \x20   assert all(ps), 'malloc returned NULL'\n\
         \x20   for p, s in zip(ps, sizes):\n\
         \x20       libc.memset(p, 0x40, s)\n\
         \x20   for p in ps:\n\
         \x20       libc.free(p)\n\
         \x20   r = f(ap, ap, 0, 4, 3, 3, 1, bp, bp, 0, 3, 4, 4, 1)\n\
         \x20   if r != want:\n\
         \x20       wrong.append(r)\n\
         assert not wrong, 'rmm4 wrong on ' + str(len(wrong)) + '/64 dirty-heap calls: ' + repr(wrong[:3])\n\
         print('ok calls=64 wrong=0 value=' + repr(want))\n",
        so.to_string_lossy()
    );
    let out = Command::new("timeout")
        .args(["60", "python3"])
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "dirty-init matmul control failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Scalar positive control: the harness + build path produce a correct
/// non-tensor artifact, so a green rank-1/rank-2 above is not a vacuous pass.
#[test]
fn scalar_positive_control_runs() {
    let dir = crate::common::scratch_dir("tensor-param-descriptor-scalar");
    let so = dir.join("artifact.so");
    if !emit(&so, "tensor-param-descriptor-scalar") {
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
    let out = Command::new("timeout")
        .args(["60", "python3"])
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
