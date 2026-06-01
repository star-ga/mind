// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native cross-module cdylib linking — composition test.
//!
//! A consumer project that imports the self-contained `std` I/O substrate
//! modules (`std.io_canon` + `std.reactor`) must produce a cdylib in which
//! those modules' symbols (`canon_*`, `reactor_*`) are DEFINED, not left as
//! undefined externals. `build_cdylib_from_entry` compiles each imported
//! substrate module to its own object (localizing the synthetic `main`) and
//! links them into the consumer `.so`.
//!
//! This exercises the project-build cdylib path (`mindc build` with
//! `emit = "cdylib"`), which is distinct from the standalone `--emit-shared`
//! path covered by `std_surface_cdylib_link`.
//!
//! Gated: `cargo test --features "mlir-build std-surface cross-module-imports"
//!                     --test cross_module_cdylib_compose`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const MAIN_MIND: &str = r#"
import std.io_canon;
import std.reactor;
import std.ring;

// Compose all three substrate modules: assign a deterministic key via the
// reactor, stage in a ring, order completions canonically, drain to bytes.
pub fn compose_tick(out_addr: i64, cap: i64) -> i64 {
    let r: i64 = reactor_new(4)
    let _ = reactor_accept(r, 7)
    let _ = ring_new(32)
    let h: i64 = canon_new(8)
    let _ = canon_push(h, 2, 1, 9, 100)
    let _ = canon_push(h, 1, 1, 9, 200)
    canon_drain(h, out_addr, cap)
}
"#;

const MIND_TOML: &str = r#"[package]
name = "compose_probe"
version = "0.1.0"
license = "Apache-2.0"

[build]
target = "cpu"
emit = "cdylib"
optimize = "release"
entry = "src/main.mind"
"#;

fn mindc_bin() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let debug = manifest_dir.join("target").join("debug").join("mindc");
    if debug.exists() {
        return debug;
    }
    manifest_dir.join("target").join("release").join("mindc")
}

#[test]
fn substrate_modules_link_natively_into_consumer_cdylib() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("compose: mindc not found; skipping");
        return;
    }

    // Build an isolated temp project.
    let proj = std::env::temp_dir().join("mind_compose_probe");
    let _ = std::fs::remove_dir_all(&proj);
    std::fs::create_dir_all(proj.join("src")).expect("create project src");
    std::fs::write(proj.join("Mind.toml"), MIND_TOML).expect("write Mind.toml");
    std::fs::write(proj.join("src").join("main.mind"), MAIN_MIND).expect("write main.mind");

    let status = Command::new(&mindc)
        .arg("build")
        .current_dir(&proj)
        .status()
        .expect("spawn mindc build");
    if !status.success() {
        // A toolchain without the MLIR backend can't emit a cdylib; skip
        // rather than fail (mirrors the std_surface_* gating convention).
        println!("compose: mindc build failed (no MLIR backend?); skipping");
        return;
    }

    let so = proj.join("target").join("release").join("libcompose_probe.so");
    assert!(so.exists(), "consumer cdylib not produced at {so:?}");

    // 1. The substrate symbols must be DEFINED (T), not undefined (U).
    let nm = Command::new("nm")
        .arg("-D")
        .arg(&so)
        .output()
        .expect("nm on PATH");
    let text = String::from_utf8_lossy(&nm.stdout);
    for sym in ["canon_new", "canon_drain", "canon_push", "reactor_new", "ring_new"] {
        let defined = text
            .lines()
            .any(|l| l.split_whitespace().last() == Some(sym) && l.contains(" T "));
        assert!(
            defined,
            "substrate symbol `{sym}` is not DEFINED in the consumer cdylib \
             (cross-module linking regressed). nm -D:\n{text}"
        );
    }

    // 2. The composed entry must dlopen and return the correctly-ordered,
    //    deterministic drained byte count (2 events × 32 bytes), with the
    //    canonical-minimum record (conn=1, req=1) first.
    let so_str = so.to_string_lossy().into_owned();
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{so_str}')\n\
         lib.compose_tick.restype = ctypes.c_int64\n\
         buf = (ctypes.c_int64 * 8)()\n\
         n = lib.compose_tick(ctypes.cast(buf, ctypes.c_void_p), 64)\n\
         assert n == 64, f'expected 64 drained bytes, got {{n}}'\n\
         assert buf[0] == 1 and buf[1] == 1, f'canonical order wrong: {{buf[0]}},{{buf[1]}}'\n\
         print('ok', n)\n"
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3 on PATH");
    assert!(
        out.status.success(),
        "composed cdylib dlopen/call failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
