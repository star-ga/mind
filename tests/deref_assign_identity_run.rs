// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Deref-assign field-first RUNTIME identity gate (D4 carve-out proof).
//!
//! Proves the ADMITTED subset is not merely accepted by the checker but produces
//! IDENTITY-PRESERVING place replacement at runtime — `*p = new` makes the place
//! DENOTE `new`'s record (a pointer store into the field cell), NOT a field-wise
//! copy. The probe encodes three observations into one i64 so a field-copy
//! miscompile yields a DIFFERENT value and fails loudly:
//!
//!   a = Pair{1,2}; h = H{pt:a}; b = Pair{10,20}
//!   do_replace(h, b)   // replace(&mut h.pt, b): h.pt now denotes b
//!   after = h.pt.x     // 10  (place denotes b)
//!   old   = a.x        // 1   (old record untouched — no clobber)
//!   b.x = 77           // mutate b's record
//!   alias = h.pt.x     // 77  (h.pt IS b — identity, not a copy)
//!   return after*1_000_000 + old*1_000 + alias
//!
//!   identity (correct) -> 10_001_077
//!   field-copy (wrong) -> 10_001_010   (alias would still read 10)
//!
//! `&mut h.pt` is taken on a struct PARAMETER receiver (do_replace's `h`), the
//! shape the D4 carve-out admits; because MIND records are passed by base
//! address, do_replace shares the caller's record, so the replacement is
//! observable back in the probe.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test deref_assign_identity_run`
//! Generic types only — no private consumer source.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
struct Pair {
    x: i64,
    y: i64
}

struct H {
    pt: Pair
}

// Callee: identity place replacement through a &mut <struct> parameter.
fn replace(p: &mut Pair, new: Pair) {
    *p = new
}

// Forwards `&mut h.pt` (depth-one struct field, param receiver) as a direct
// call argument — the admitted caller shape.
fn do_replace(h: H, b: Pair) {
    replace(&mut h.pt, b)
}

pub fn deref_identity_probe() -> i64 {
    let a = Pair { x: 1, y: 2 }
    let h = H { pt: a }
    let mut b = Pair { x: 10, y: 20 }
    do_replace(h, b)
    let after = h.pt.x
    let old = a.x
    b.x = 77
    let alias = h.pt.x
    return after * 1000000 + old * 1000 + alias
}
"#;

#[test]
fn deref_assign_identity_replacement_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "deref_assign_identity_run",
            "deref-assign-identity-run: mindc not found; skipping",
        );
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_deref_assign_identity_run.mind");
    let so = dir.join("mind_deref_assign_identity_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !crate::common::gate::compiled("deref_assign_identity_run", &out) {
        return;
    }

    // identity => 10_001_077; a field copy would give 10_001_010 and fail.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.deref_identity_probe.restype = ctypes.c_int64\n\
         r = lib.deref_identity_probe()\n\
         assert r == 10001077, 'expected identity 10001077 (field-copy would be 10001010); got ' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "deref-assign identity-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
