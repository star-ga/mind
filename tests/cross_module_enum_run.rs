// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Cross-module enum dot-variant access RUNTIME gate.
//!
//! An enum declared in one module (`e.mind`: `enum TokKind { … }`) used from
//! another module via the dot form (`TokKind.Eof`) — both as a value and in a
//! `match` — must resolve to the correct ordinal tag, even though the consuming
//! module never lowered the `EnumDef` itself. This exercises the whole-project
//! enum registry (`crate::ir::GlobalEnums`): the project builder collects every
//! declared enum, the parser normalises `Enum.Variant` → `Enum::Variant` for a
//! sibling enum, and the lowering merges the variant tags. Builds a two-file
//! `cdylib` project, dlopen-calls it, and asserts the tag values.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test cross_module_enum_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const E_MIND: &str = "pub enum TokKind { Plus, Minus, Eof }\n";

const MAIN_MIND: &str = r#"
import e

// `TokKind.Eof` is a SIBLING-module enum used via the dot form, as a value and
// in a `match`. Eof is the 3rd variant (ordinal 2) → arm returns 3.
pub fn classify() -> i64 {
    let t = TokKind.Eof
    match t {
        TokKind.Plus => 1,
        TokKind.Minus => 2,
        TokKind.Eof => 3,
    }
}

// A second variant to prove the tag mapping is per-variant, not constant.
pub fn ctor_tag() -> i64 {
    let t = TokKind.Minus
    match t {
        TokKind.Plus => 10,
        TokKind.Minus => 20,
        TokKind.Eof => 30,
    }
}
"#;

fn mindc_bin() -> PathBuf {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d = m.join("target").join("debug").join("mindc");
    if d.exists() {
        d
    } else {
        m.join("target").join("release").join("mindc")
    }
}

#[test]
fn cross_module_enum_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("cross-module-enum-run: mindc not found; skipping");
        return;
    }

    // Isolated project dir (unique per process to avoid cross-test races).
    let proj = std::env::temp_dir().join(format!("mind_xmenum_run_{}", std::process::id()));
    let src = proj.join("src");
    let _ = std::fs::remove_dir_all(&proj);
    std::fs::create_dir_all(&src).expect("mkdir src");
    std::fs::write(
        proj.join("Mind.toml"),
        "[package]\nname = \"xmenum\"\nversion = \"0.1.0\"\n\n[build]\nemit = \"cdylib\"\n",
    )
    .expect("write Mind.toml");
    std::fs::write(src.join("e.mind"), E_MIND).expect("write e.mind");
    std::fs::write(src.join("main.mind"), MAIN_MIND).expect("write main.mind");

    let out = Command::new(&mindc)
        .arg("build")
        .current_dir(&proj)
        .output()
        .expect("run mindc build");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("cross-module-enum-run: needs mlir-build; skipping");
            return;
        }
        panic!("cross-module-enum-run: mindc build failed:\n{stderr}");
    }

    // Locate the produced cdylib under target/.
    let mut so = None;
    for prof in ["debug", "release"] {
        let dir = proj.join("target").join(prof);
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map(|x| x == "so").unwrap_or(false) {
                    so = Some(p);
                    break;
                }
            }
        }
        if so.is_some() {
            break;
        }
    }
    let so = so.expect("cross-module-enum-run: no .so produced");

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('classify','ctor_tag'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.classify(); assert r == 3, 'classify=' + str(r)\n\
         r = lib.ctor_tag(); assert r == 20, 'ctor_tag=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "cross-module-enum-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let _ = std::fs::remove_dir_all(&proj);
}
