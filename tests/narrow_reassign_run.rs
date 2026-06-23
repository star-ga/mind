// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Narrow integer let + reassign (including inside if arm) must re-mask
//! to declared width on every Assign so that wrap/overflow is visible to
//! subsequent uses and returns. Without re-record in branch/block paths,
//! an Assign after a narrow let inside an if arm would leave high bits,
//! a silent miscompile (EXIT=0, wrong scalar).
//!
//! Uses i64 ABI wrapper + internal narrow lets + as-casts (u8 here).
//! Gate exercised by --emit-shared + ctypes.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
pub fn narrow_if_re(x: i64) -> i64 {
    if x > 0 {
        let c: u8 = x as u8;
        c = c + 1;
        c as i64
    } else {
        0
    }
}

pub fn narrow_top_re(x: i64) -> i64 {
    let mut c: u8 = x as u8;
    c = c + 100;
    c as i64
}
"#;

#[test]
fn narrow_reassign_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("narrow-reassign-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_narrow_reassign_run.mind");
    let so = dir.join("mind_narrow_reassign_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("narrow-reassign-run: needs mlir-build; skipping");
            return;
        }
        panic!("narrow-reassign-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.narrow_if_re.argtypes = [ctypes.c_int64]; lib.narrow_if_re.restype = ctypes.c_int64\n\
         lib.narrow_top_re.argtypes = [ctypes.c_int64]; lib.narrow_top_re.restype = ctypes.c_int64\n\
         # 255+1 wrap ->0 ; 10+1->11\n\
         r = lib.narrow_if_re(255); assert r == 0, 'if255=' + str(r)\n\
         r = lib.narrow_if_re(10); assert r == 11, 'if10=' + str(r)\n\
         r = lib.narrow_top_re(200); assert r == 44, 'top200=' + str(r)\n\
         r = lib.narrow_top_re(0); assert r == 100, 'top0=' + str(r)\n\
         print('ok narrow reassign')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "narrow-reassign-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
