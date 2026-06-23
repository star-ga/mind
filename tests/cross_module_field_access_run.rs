// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Cross-module value-position FieldAccess read — RUNTIME gate (regression).
//!
//! A pure CONSUMER module (no local `StructDef`) that reads a field of a
//! struct defined in a SIBLING module must produce the STORED value, not the
//! `ConstI64(0)` placeholder.
//!
//! The concrete miss this guards is a BY-REFERENCE struct PARAMETER
//! (`fn f(p: &Point) -> i64 { p.y }`): the lowering fast-path (`struct_env`,
//! Step 1) only seeds a `TypeAnn::Named` param, so a `&Point` param never
//! seeds Step 1; and the side-table (Step 2) was empty for a struct-less
//! consumer module because:
//!   (a) the resolver gate in `lower_to_ir` only ran the span->struct-name
//!       resolver when the LOCAL module declared a `StructDef`, and
//!   (b) even when run, the resolver only knew struct NAMES from the local
//!       module's own `StructDef` items, so a `&SiblingStruct` param was
//!       dropped from `fn_vars` and its `p.field` span never recorded.
//! Both miss -> the `p.y` read lowered to `ConstI64(0)` (a SILENT miscompile:
//! runnable .so, EXIT=0, wrong value).
//!
//! The by-VALUE form (`fn g(p: Point) -> i64 { p.x }`) already resolved via the
//! `struct_env` Step-1 param seed, so it is included as a control: a fix must
//! not regress it.
//!
//! The fix lives at the resolver layer (`struct_resolver::build_field_access_types`
//! unions the whole-project struct names + field types from the global registry,
//! and `lower_to_ir` widens the gate so the resolver also runs when that registry
//! is non-empty). `unwrap_to_named` already peels `&T`, so the by-ref param now
//! seeds and Step 2 resolves.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test cross_module_field_access_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

// Sibling module: defines the struct. The consumer declares no struct of its own.
const TYPES_SRC: &str = r#"
struct Point { x: i64, y: i64 }
"#;

// Consumer module (the cdylib entry): reads a field of the SIBLING struct via
// a by-reference param (the regressed path) and a by-value param (control).
const COMPUTE_SRC: &str = r#"
// By-REFERENCE struct param field read. Pre-fix: ConstI64(0) -> returns 0.
// Post-fix: resolves the field -> returns p.y.
fn ref_param_read(p: &Point) -> i64 {
    let r = p.y
    return r
}

// By-VALUE struct param field read (control: already worked via struct_env).
fn val_param_read(p: Point) -> i64 {
    let r = p.x
    return r
}
"#;

const MANIFEST: &str = r#"[package]
name = "g41xmod"
version = "0.1.0"

[build]
entry = "src/compute.mind"
output = "g41xmod"

[targets.cpu]
backend = "cpu"

[exports]
c_abi = ["ref_param_read", "val_param_read"]
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
fn cross_module_field_access_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("cross-module-field-access-run: mindc not found; skipping");
        return;
    }

    // Build an isolated 2-file project so the project builder populates the
    // whole-project struct registry (the path that exercises the gap).
    let root = std::env::temp_dir().join("mind_xmod_field_access_run");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src")).expect("mkdir project");
    std::fs::write(root.join("Mind.toml"), MANIFEST).expect("write manifest");
    std::fs::write(root.join("src").join("types.mind"), TYPES_SRC).expect("write types");
    std::fs::write(root.join("src").join("compute.mind"), COMPUTE_SRC).expect("write compute");

    let so = root.join("xmod.so");

    // `--no-cache`: the cdylib object cache is keyed on the entry source and
    // would otherwise serve a stale `.so` from a prior (pre-fix) build, masking
    // the fix. The standing gate's documented footgun.
    let out = Command::new(&mindc)
        .current_dir(&root)
        .args([
            "build",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc build");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("cross-module-field-access-run: needs mlir-build; skipping");
            return;
        }
        panic!(
            "cross-module-field-access-run: mindc build failed:\nstdout: {}\nstderr: {stderr}",
            String::from_utf8_lossy(&out.stdout),
        );
    }

    // Pass &Point{x, y}; assert the cross-module field reads return the stored
    // values, not the ConstI64(0) placeholder.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         class Point(ctypes.Structure):\n\
         \x20   _fields_ = [('x', ctypes.c_int64), ('y', ctypes.c_int64)]\n\
         for fn, exp, pt in (('ref_param_read', 42, (5, 42)), ('val_param_read', 7, (7, 9))):\n\
         \x20   f = getattr(lib, fn); f.restype = ctypes.c_int64\n\
         \x20   f.argtypes = [ctypes.POINTER(Point)]\n\
         \x20   p = Point(pt[0], pt[1])\n\
         \x20   got = f(ctypes.byref(p))\n\
         \x20   assert got == exp, fn + ': got=' + str(got) + ' expected=' + str(exp)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "cross-module-field-access-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
