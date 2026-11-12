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

mod common;
use common::mindc_bin;

use std::process::Command;

// Sibling module: defines the struct. The consumer declares no struct of its own.
const TYPES_SRC: &str = r#"
struct Point { x: i64, y: i64 }
type Elem = i64
type Words = [Elem; 2]
struct ArrayHolder { xs: Words }
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

// The field's fixed-array type is expressed through aliases owned by the
// defining module.  The consumer has no local alias declarations; lowering
// must receive the resolved shared metadata rather than guessing from names.
fn imported_array_alias_read(p: ArrayHolder) -> i64 {
    return p.xs[1]
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
c_abi = ["ref_param_read", "val_param_read", "imported_array_alias_read"]
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn cross_module_field_access_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "cross_module_field_access_run",
            "cross-module-field-access-run: mindc not found; skipping",
        );
        return;
    }

    // Build an isolated 2-file project so the project builder populates the
    // whole-project struct registry (the path that exercises the gap).
    let root = common::scratch_dir("cross-module-field-access-run").join("field-access");
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
            "--backend=mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc build");
    if !crate::common::gate::compiled("cross_module_field_access_run", &out) {
        return;
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

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         class ArrayHolder(ctypes.Structure):\n\
         \x20   _fields_ = [('xs0', ctypes.c_int64), ('xs1', ctypes.c_int64)]\n\
         f = lib.imported_array_alias_read\n\
         f.restype = ctypes.c_int64\n\
         f.argtypes = [ctypes.POINTER(ArrayHolder)]\n\
         p = ArrayHolder(11, 42)\n\
         assert f(ctypes.byref(p)) == 42\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "cross-module-array-alias-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn cross_module_return_alias_scope_and_shape_controls() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "cross_module_return_alias_scope_and_shape_controls",
            "cross-module-return-alias-controls: mindc not found; skipping",
        );
        return;
    }

    let root = common::scratch_dir("cross-module-field-access-run").join("return-alias");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src")).expect("create return-alias project src");
    std::fs::write(
        root.join("Mind.toml"),
        r#"[package]
name = "return_alias_controls"
version = "0.1.0"

[build]
entry = "src/compute.mind"
output = "return_alias_controls"

[targets.cpu]
backend = "cpu"

[exports]
c_abi = ["read"]
"#,
    )
    .expect("write return-alias manifest");
    std::fs::write(
        root.join("src").join("types.mind"),
        "struct Item { value: i64 }\ntype Items = [Item; 2]\npub fn make_items() -> Items { return [Item { value: 11 }, Item { value: 42 }] }\n",
    )
    .expect("write defining module");
    std::fs::write(
        root.join("src").join("compute.mind"),
        "use crate.types\ntype Item = i64\npub fn read() -> i64 { return make_items()[1].value }\n",
    )
    .expect("write consumer module");
    let so = root.join("return_alias_controls.so");
    let out = Command::new(&mindc)
        .current_dir(&root)
        .args([
            "build",
            "--backend=mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run return-alias project");
    assert!(
        out.status.success() && so.is_file(),
        "defining-owner return schema was reinterpreted:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f = lib.read\n\
         f.restype = ctypes.c_int64\n\
         assert f() == 42\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let run = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run return-alias shared library");
    assert!(
        run.status.success(),
        "return-alias runtime result was wrong:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    // Incompatible imported bare names refuse before duplicate-symbol linking.
    let ambiguous = root.join("ambiguous");
    std::fs::create_dir_all(ambiguous.join("src")).expect("create ambiguity project src");
    std::fs::write(
        ambiguous.join("Mind.toml"),
        r#"[package]
name = "return_shape_ambiguity"
version = "0.1.0"

[build]
entry = "src/compute.mind"
output = "return_shape_ambiguity"

[targets.cpu]
backend = "cpu"

[exports]
c_abi = ["read"]
"#,
    )
    .expect("write ambiguity manifest");
    std::fs::write(
        ambiguous.join("src").join("array.mind"),
        "struct Item { value: i64 }\npub fn make_items() -> [Item; 2] { return [Item { value: 11 }, Item { value: 42 }] }\n",
    )
    .expect("write array owner");
    std::fs::write(
        ambiguous.join("src").join("scalar.mind"),
        "pub fn make_items() -> i64 { return 7 }\n",
    )
    .expect("write scalar owner");
    std::fs::write(
        ambiguous.join("src").join("compute.mind"),
        "use crate.array\nuse crate.scalar\npub fn read() -> i64 { return make_items()[1].value }\n",
    )
    .expect("write ambiguity consumer");
    let ambiguous_so = ambiguous.join("return_shape_ambiguity.so");
    let check = Command::new(&mindc)
        .current_dir(&ambiguous)
        .args(["check", "src/compute.mind"])
        .output()
        .expect("check imported return-shape ambiguity");
    let checked = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(
        !check.status.success(),
        "ambiguous check succeeded: {checked}"
    );
    assert!(
        checked.contains("E2003"),
        "ambiguous check lost E2003: {checked}"
    );
    assert!(
        checked.contains("make_items"),
        "ambiguous check did not identify the offending call: {checked}"
    );

    let out = Command::new(&mindc)
        .current_dir(&ambiguous)
        .args([
            "build",
            "--backend=mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            ambiguous_so.to_str().unwrap(),
        ])
        .output()
        .expect("run imported return-shape ambiguity");
    let rendered = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "ambiguous imported return compiled: {rendered}"
    );
    assert!(
        rendered.contains("E2003"),
        "ambiguity lost the documented E2003 refusal: {rendered}"
    );
    assert!(
        rendered.contains("make_items"),
        "ambiguity emit did not identify the offending call: {rendered}"
    );
    assert!(
        !rendered.contains("duplicate symbol"),
        "ambiguity reached linker: {rendered}"
    );
    assert!(
        !rendered.contains("panicked at"),
        "ambiguity panicked: {rendered}"
    );
    assert!(
        !ambiguous_so.exists(),
        "ambiguous imported return emitted an artifact"
    );

    // The same ambiguous bare function must remain a type-check refusal when
    // its result first flows through a local alias. This guards against a
    // direct-shape-only exception being widened into a successful ABI choice.
    std::fs::write(
        ambiguous.join("src").join("compute.mind"),
        "use crate.array\nuse crate.scalar\npub fn read() -> i64 { let xs = make_items(); return xs[1].value }\n",
    )
    .expect("write aliased ambiguity consumer");
    let check = Command::new(&mindc)
        .current_dir(&ambiguous)
        .args(["check", "src/compute.mind"])
        .output()
        .expect("check aliased return-shape ambiguity");
    let checked = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(
        !check.status.success(),
        "aliased ambiguity check succeeded: {checked}"
    );
    assert!(
        checked.contains("E2003"),
        "aliased ambiguity lost E2003: {checked}"
    );
    assert!(
        checked.contains("make_items"),
        "aliased ambiguity check did not identify the offending call: {checked}"
    );
    let aliased_so = ambiguous.join("aliased-return-shape-ambiguity.so");
    let emit = Command::new(&mindc)
        .current_dir(&ambiguous)
        .args([
            "build",
            "--backend=mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            aliased_so.to_str().unwrap(),
        ])
        .output()
        .expect("emit aliased return-shape ambiguity");
    let emitted = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&emit.stdout),
        String::from_utf8_lossy(&emit.stderr)
    );
    assert!(
        !emit.status.success(),
        "aliased ambiguity emit succeeded: {emitted}"
    );
    assert!(
        emitted.contains("E2003"),
        "aliased ambiguity emit lost E2003: {emitted}"
    );
    assert!(
        emitted.contains("make_items"),
        "aliased ambiguity emit did not identify the offending call: {emitted}"
    );
    assert!(
        !emitted.contains("duplicate symbol"),
        "aliased ambiguity reached linker: {emitted}"
    );
    assert!(
        !aliased_so.exists(),
        "aliased ambiguity emitted an artifact"
    );

    // Equal return annotations do not make two defining owners one owner.
    // Their Item schemas intentionally differ, so a resolver comparing only
    // the consumer-visible `[Item; 1]` text must still fail closed at E2003.
    let same_shape = root.join("same-shape-ambiguity");
    std::fs::create_dir_all(same_shape.join("src")).expect("create same-shape project src");
    std::fs::write(
        same_shape.join("Mind.toml"),
        r#"[package]
name = "same_shape_ambiguity"
version = "0.1.0"

[build]
entry = "src/compute.mind"
output = "same_shape_ambiguity"

[targets.cpu]
backend = "cpu"

[exports]
c_abi = ["read"]
"#,
    )
    .expect("write same-shape manifest");
    std::fs::write(
        same_shape.join("src").join("left.mind"),
        "struct Item { value: i64 }\npub fn make_items() -> [Item; 1] { return [Item { value: 11 }] }\n",
    )
    .expect("write left same-shape owner");
    std::fs::write(
        same_shape.join("src").join("right.mind"),
        "struct Item { other: i64 }\npub fn make_items() -> [Item; 1] { return [Item { other: 99 }] }\n",
    )
    .expect("write right same-shape owner");
    std::fs::write(
        same_shape.join("src").join("compute.mind"),
        "use crate.left\nuse crate.right\npub fn read() -> i64 { return make_items()[0].value }\n",
    )
    .expect("write same-shape ambiguity consumer");
    let check = Command::new(&mindc)
        .current_dir(&same_shape)
        .args(["check", "src/compute.mind"])
        .output()
        .expect("check same-shape ambiguity");
    let checked = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(
        !check.status.success(),
        "same-shape ambiguity check succeeded: {checked}"
    );
    assert!(
        checked.contains("E2003"),
        "same-shape ambiguity lost E2003: {checked}"
    );
    assert!(
        checked.contains("make_items"),
        "same-shape check did not identify the offending call: {checked}"
    );
    let same_shape_so = same_shape.join("same_shape_ambiguity.so");
    let emit = Command::new(&mindc)
        .current_dir(&same_shape)
        .args([
            "build",
            "--backend=mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            same_shape_so.to_str().unwrap(),
        ])
        .output()
        .expect("emit same-shape ambiguity");
    let emitted = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&emit.stdout),
        String::from_utf8_lossy(&emit.stderr)
    );
    assert!(
        !emit.status.success(),
        "same-shape ambiguity emit succeeded: {emitted}"
    );
    assert!(
        emitted.contains("E2003"),
        "same-shape ambiguity emit lost E2003: {emitted}"
    );
    assert!(
        emitted.contains("make_items"),
        "same-shape emit did not identify the offending call: {emitted}"
    );
    assert!(
        !emitted.contains("duplicate symbol"),
        "same-shape ambiguity reached linker: {emitted}"
    );
    assert!(
        !same_shape_so.exists(),
        "same-shape ambiguity emitted an artifact"
    );
}
