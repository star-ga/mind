// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Runtime contract for transparent narrow-integer aliases at coercion points.
//!
//! The checker accepted `let y: Byte = if ...` for `type Byte = i8`, but
//! lowering passed the unresolved `Named("Byte")` to the narrow-let mask and
//! retained the full i64 initializer. These checks call shared-library symbols
//! through ctypes so process-exit modulo masking cannot hide a wrong value.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use common::mindc_bin;
use std::path::Path;
use std::process::Command;

const SINGLE_SOURCE: &str = r#"
type Byte = i8;
type Octet = u8;
type ChainedByte = Byte;

pub fn alias_i8_value_if(c: i64) -> i64 {
    let y: Byte = if c == 1 { 300 } else { 300 };
    if y > 255 { return 1; }
    return 0;
}

pub fn builtin_i8_value_if(c: i64) -> i64 {
    let y: i8 = if c == 1 { 300 } else { 300 };
    if y > 255 { return 1; }
    return 0;
}

pub fn alias_u8_value_if(c: i64) -> i64 {
    let y: Octet = if c == 1 { 300 } else { 300 };
    if y == 44 { return 1; }
    return 0;
}

pub fn builtin_u8_value_if(c: i64) -> i64 {
    let y: u8 = if c == 1 { 300 } else { 300 };
    if y == 44 { return 1; }
    return 0;
}

pub fn alias_branch_then(c: i64) -> i64 {
    if c == 1 {
        let y: Byte = 200;
        if y < 0 { return 1; }
        return 0;
    }
    return 9;
}

pub fn alias_branch_else(c: i64) -> i64 {
    if c == 1 {
        return 9;
    } else {
        let y: Byte = 200;
        if y < 0 { return 1; }
        return 0;
    }
}

pub fn builtin_explicit_cast_control() -> i64 {
    let y = 200 as i8;
    if y < 0 { return 1; }
    return 0;
}

pub fn alias_explicit_cast_control() -> i64 {
    let y = 200 as Byte;
    if y < 0 { return 1; }
    return 0;
}

pub fn alias_reassign_control() -> i64 {
    let mut y: ChainedByte = 200;
    y = 300;
    return y;
}

fn alias_narrow_return() -> Byte {
    return 200;
}

fn builtin_narrow_return() -> i8 {
    return 200;
}

pub fn alias_return_control() -> i64 {
    let y = alias_narrow_return();
    if y < 0 { return 1; }
    return 0;
}

pub fn builtin_return_control() -> i64 {
    let y = builtin_narrow_return();
    if y < 0 { return 1; }
    return 0;
}
"#;

fn run_ctypes(so: &Path, assertions: &str) {
    let py = format!(
        "import ctypes\nlib=ctypes.CDLL(r'{}')\n{}\nprint('ok')\n",
        so.display(),
        assertions
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run alias-lowering ctypes checks");
    assert!(
        out.status.success(),
        "alias-lowering runtime checks failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn narrow_aliases_match_builtin_coercions_and_keep_source_owners() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "type_alias_narrow_lowering_run",
            "type-alias narrow lowering: mindc not found; skipping",
        );
        return;
    }

    let scratch = crate::common::scratch_dir("type_alias_narrow_lowering_run");
    let single = scratch.join("single.mind");
    let single_so = scratch.join("single.so");
    std::fs::write(&single, SINGLE_SOURCE).expect("write single-file alias source");
    let compiled = Command::new(&mindc)
        .args([
            single.to_str().unwrap(),
            "--emit-shared",
            single_so.to_str().unwrap(),
        ])
        .output()
        .expect("compile single-file alias source");
    if !crate::common::gate::compiled("type_alias_narrow_lowering_run", &compiled) {
        return;
    }
    run_ctypes(
        &single_so,
        r#"
for name in (
    'alias_i8_value_if', 'builtin_i8_value_if',
    'alias_u8_value_if', 'builtin_u8_value_if',
    'alias_branch_then', 'alias_branch_else',
):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_int64]
    fn.restype = ctypes.c_int64
for name in (
    'builtin_explicit_cast_control', 'alias_explicit_cast_control',
    'alias_reassign_control',
    'alias_return_control', 'builtin_return_control',
):
    getattr(lib, name).restype = ctypes.c_int64
assert lib.alias_i8_value_if(1) == 0
assert lib.builtin_i8_value_if(1) == 0
assert lib.alias_u8_value_if(1) == 1
assert lib.builtin_u8_value_if(1) == 1
assert lib.alias_branch_then(1) == 1
assert lib.alias_branch_else(0) == 1
assert lib.builtin_explicit_cast_control() == 1
assert lib.alias_explicit_cast_control() == 1
assert lib.alias_reassign_control() == 44
assert lib.alias_return_control() == 1
assert lib.builtin_return_control() == 1
"#,
    );

    // Each imported source owns its alias table. The same spelling maps to i8
    // in left.mind and u8 in right.mind; flattening the project aliases would
    // make one of these functions take the other module's coercion.
    let project = scratch.join("owners");
    let project_src = project.join("src");
    std::fs::create_dir_all(&project_src).expect("create owner project");
    std::fs::write(
        project.join("Mind.toml"),
        "[package]\nname = \"alias_owners\"\nversion = \"0.1.0\"\n\n[build]\nemit = \"cdylib\"\n",
    )
    .expect("write owner project manifest");
    std::fs::write(
        project_src.join("dep.mind"),
        "export { DepByte }\ntype DepByte = i8;\n",
    )
    .expect("write transitive alias dependency");
    std::fs::write(
        project_src.join("left.mind"),
        "import dep;\nexport { Byte, left_owner }\ntype PrivateByte = dep.DepByte;\ntype Byte = PrivateByte;\npub fn left_owner() -> i64 { let y: Byte = 200; if y < 0 { return 11; } return 12; }\n",
    )
    .expect("write left alias owner");
    std::fs::write(
        project_src.join("right.mind"),
        "export { Byte, right_owner }\ntype Byte = u8;\npub fn right_owner() -> i64 { let y: Byte = 200; if y > 127 { return 22; } return 23; }\n",
    )
    .expect("write right alias owner");
    std::fs::write(
        project_src.join("main.mind"),
        "import left;\nimport right;\npub fn owner_pair() -> i64 { return left.left_owner() * 100 + right.right_owner(); }\npub fn qualified_left() -> i64 { let y: left.Byte = 200; if y < 0 { return 31; } return 32; }\npub fn qualified_right() -> i64 { let y: right.Byte = 200; if y > 127 { return 41; } return 42; }\n",
    )
    .expect("write alias owner consumer");

    let built = Command::new(&mindc)
        .args(["build", "--backend", "mlir"])
        .current_dir(&project)
        .output()
        .expect("build alias owner project");
    assert!(
        built.status.success(),
        "alias owner project failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&built.stdout),
        String::from_utf8_lossy(&built.stderr)
    );
    run_ctypes(
        &project.join("target/debug/libalias_owners.so"),
        "lib.owner_pair.restype = ctypes.c_int64\nlib.qualified_left.restype = ctypes.c_int64\nlib.qualified_right.restype = ctypes.c_int64\nassert lib.owner_pair() == 1122\nassert lib.qualified_left() == 31\nassert lib.qualified_right() == 41",
    );

    if cfg!(feature = "ffi-c-user") {
        run_ctypes(
            &project.join("target/debug/libalias_owners.so"),
            r#"
lib.owner_pair.restype = ctypes.c_int64
lib.qualified_left.restype = ctypes.c_int64
lib.qualified_right.restype = ctypes.c_int64
assert lib.owner_pair() == 1122
assert lib.qualified_left() == 31
assert lib.qualified_right() == 41

# Type aliases are part of the language export surface but have no callable
# C ABI. Their old wrappers collided across left.mind/right.mind; they must
# be absent while the real exported functions remain linkable ENOSYS stubs.
assert not hasattr(lib, 'mind_fn_Byte_v1_invoke')
for name in ('mind_fn_left_owner_v1_invoke', 'mind_fn_right_owner_v1_invoke'):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    fn.restype = ctypes.c_int32
    assert fn(None, 0, None, 0) == 38
"#,
        );
    }

    std::fs::write(
        project_src.join("main.mind"),
        "import left;\nfn private_alias_access() -> i64 { let y: left.PrivateByte = 200; return y; }\n",
    )
    .expect("write private alias rejection case");
    let checked = Command::new(&mindc)
        .args(["check", "--no-fmt", "--no-lint"])
        .current_dir(&project)
        .output()
        .expect("check private alias access");
    let check_text = format!(
        "{}{}",
        String::from_utf8_lossy(&checked.stdout),
        String::from_utf8_lossy(&checked.stderr)
    );
    assert!(
        !checked.status.success(),
        "private alias access was accepted"
    );
    assert!(check_text.contains("E2002"), "{check_text}");
    assert!(!check_text.contains("panicked"), "{check_text}");
}
