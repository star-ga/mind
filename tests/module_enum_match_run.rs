// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Module-wrapped enum `match` RUNTIME gate (task #271).
//!
//! `module m { enum Mode { On, Off } }` is parsed to a transparent
//! `ast::Node::Block`, so its `EnumDef` never reached the top-level `EnumDef`
//! lowering arm — `enum_variant_tags` stayed unpopulated for `Mode::On`/
//! `Mode::Off`. A `match mode { Mode::On => 1, Mode::Off => 0 }` then found NO
//! tag in `desugar_match_to_if`/`pattern_test`, DEGRADED to the scrutinee-
//! ignoring sequential fallback, and returned the LAST arm for EVERY scrutinee
//! — a SILENT MISCOMPILE (`match Mode::Off { … }` returned 1, not 0). It shipped
//! because `tests/parse_match_and_ref.rs` only asserts `.is_ok()`, never RUNS.
//!
//! This test compiles a module-wrapped-enum program to a `.so`, dlopen-calls it,
//! and asserts BOTH arms take the CORRECT branch (a real discriminant jump on
//! the scrutinee, not the last-arm fallback).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test module_enum_match_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

// A module-wrapped fieldless enum matched BOTH ways. Each fn is 0-arg → i64.
// `on()` matches `Mode::On` and MUST return 1; `off()` matches `Mode::Off` and
// MUST return 0. Under the pre-fix fallback both would return the LAST arm (0),
// so `on()` returning 1 is the direct proof the scrutinee is honoured.
const SRC: &str = r#"
module m {
    enum Mode { On, Off }

    fn classify(mode: Mode) -> i64 {
        match mode {
            Mode::On => 1,
            Mode::Off => 0,
        }
    }

    fn on() -> i64 {
        classify(Mode::On)
    }

    fn off() -> i64 {
        classify(Mode::Off)
    }

    // Direct match on a constructed scrutinee (no fn-param indirection), the
    // exact shape called out in task #271.
    fn direct_off() -> i64 {
        match Mode::Off {
            Mode::On => 1,
            Mode::Off => 0,
        }
    }

    fn direct_on() -> i64 {
        match Mode::On {
            Mode::On => 1,
            Mode::Off => 0,
        }
    }
}
"#;

#[test]
fn module_wrapped_enum_match_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("module-enum-match-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_module_enum_match_run.mind");
    let so = dir.join("mind_module_enum_match_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("module-enum-match-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("module-enum-match-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('on','off','direct_on','direct_off'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.on(); assert r == 1, 'on=' + str(r)\n\
         r = lib.off(); assert r == 0, 'off=' + str(r)\n\
         r = lib.direct_on(); assert r == 1, 'direct_on=' + str(r)\n\
         r = lib.direct_off(); assert r == 0, 'direct_off=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "module-enum-match-run value check failed (last-arm fallback would make on()/direct_on() return 0):\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
