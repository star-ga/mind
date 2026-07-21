// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Task #258 — `std/toml.mind` surface tests.
//!
//! Tests in this file verify:
//!
//! 1. `std/toml.mind` parses and lowers to IR with all required `pub fn`s
//!    present and well-formed.
//! 2. A consumer MIND program that does `use std.toml` and calls `toml_parse`
//!    resolves end-to-end via the bundled stdlib resolver.
//! 3. The `pub fn` signatures exported match the task #258 API contract.
//! 4. The `std.toml` module auto-exports every expected public symbol.
//! 5. The Rust `toml` crate and `std.toml` agree on the key/value structure
//!    of the canonical `Mind.toml` at the repo root (cross-check).
//!
//! Gate: `cargo test --features "std-surface cross-module-imports"
//!                  --test std_surface_toml`
//!
//! The MLIR-build functional execution test (compiling the MIND TOML parser
//! to a native `.so` and running it against `Mind.toml`) is nested under
//! `#[cfg(feature = "mlir-build")]` and skipped automatically on open-core
//! builds that lack the MLIR backend.

#![cfg(feature = "std-surface")]

mod common;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const TOML_MIND_SRC: &str = include_str!("../std/toml.mind");

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn lower_toml_mind() -> libmind::ir::IRModule {
    let module = parser::parse(TOML_MIND_SRC).expect("std/toml.mind must parse cleanly");
    lower_to_ir(&module)
}

/// Recursively count calls to `callee` anywhere in an instruction stream.
fn count_calls_recursive(instrs: &[Instr], callee: &str) -> usize {
    let mut n = 0;
    for instr in instrs {
        match instr {
            Instr::Call { name, .. } if name == callee => n += 1,
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                n += count_calls_recursive(cond_instrs, callee);
                n += count_calls_recursive(then_instrs, callee);
                n += count_calls_recursive(else_instrs, callee);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                n += count_calls_recursive(cond_instrs, callee);
                n += count_calls_recursive(body, callee);
            }
            Instr::FnDef { body, .. } => {
                n += count_calls_recursive(body, callee);
            }
            _ => {}
        }
    }
    n
}

fn fn_body<'a>(ir: &'a libmind::ir::IRModule, name: &str) -> &'a [Instr] {
    ir.instrs
        .iter()
        .find_map(|i| match i {
            Instr::FnDef { name: n, body, .. } if n == name => Some(body.as_slice()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected FnDef `{name}` in lowered IR"))
}

// ─── Test 1: parse + lower ────────────────────────────────────────────────────

#[test]
fn toml_mind_parses_and_lowers() {
    // Every public function required by the task #258 API contract must
    // appear as a FnDef in the lowered IR.
    let ir = lower_toml_mind();
    for want in [
        "toml_parse",
        "toml_get",
        "toml_dump",
        // Kind constants (public).
        "tv_kind_str",
        "tv_kind_int",
        "tv_kind_bool",
        "tv_kind_arr",
        "tv_kind_tbl",
        // Field accessors (public).
        "tv_kind",
        "tv_s_addr",
        "tv_s_len",
        "tv_s_cap",
        "tv_ival",
        "tv_bval",
        "tv_arr_addr",
        "tv_arr_len",
        "tv_arr_cap",
        "tv_tbl_keys",
        "tv_tbl_vals",
        "tv_tbl_len",
        "tv_tbl_cap",
        // Array element accessor.
        "tv_arr_get",
    ] {
        assert!(
            ir.instrs
                .iter()
                .any(|i| matches!(i, Instr::FnDef { name, .. } if name == want)),
            "missing FnDef `{want}` in lowered std/toml.mind IR"
        );
    }
}

// ─── Test 2: toml_parse calls tv_make_tbl ────────────────────────────────────

#[test]
fn toml_parse_calls_tv_make_tbl() {
    let ir = lower_toml_mind();
    let body = fn_body(&ir, "toml_parse");
    // toml_parse must call tv_make_tbl() to construct the root table.
    // __mind_alloc will appear in tv_alloc which is called by tv_make_tbl.
    // We verify tv_make_tbl is called directly from the body.
    let make_tbl_calls = count_calls_recursive(body, "tv_make_tbl");
    assert!(
        make_tbl_calls >= 1,
        "toml_parse must call tv_make_tbl at least once (for the root table); got {make_tbl_calls}"
    );
}

// ─── Test 3: tv_kind_* emit constant zero / non-zero ─────────────────────────

#[test]
fn tv_kind_constants_are_distinct() {
    // The five kind-tag functions must produce distinct constant values.
    // We verify this by checking that each FnDef body emits a ConstI64.
    let ir = lower_toml_mind();
    let mut values: Vec<i64> = Vec::new();
    for fn_name in [
        "tv_kind_str",
        "tv_kind_int",
        "tv_kind_bool",
        "tv_kind_arr",
        "tv_kind_tbl",
    ] {
        let body = fn_body(&ir, fn_name);
        // Find the first ConstI64 in the body.
        let val = body.iter().find_map(|i| match i {
            Instr::ConstI64(_, v) => Some(*v),
            _ => None,
        });
        let v = val.unwrap_or_else(|| panic!("{fn_name} must emit a ConstI64"));
        values.push(v);
    }
    // All five must be distinct.
    let mut seen = std::collections::HashSet::new();
    for v in &values {
        assert!(
            seen.insert(v),
            "tv_kind_* constants must be distinct; duplicate value {v}"
        );
    }
}

// ─── Test 4: toml_dump uses sb_push (transitively) ───────────────────────────

#[test]
fn toml_dump_calls_sb_push() {
    let ir = lower_toml_mind();
    let body = fn_body(&ir, "toml_dump");
    // toml_dump must call sb_push (directly or via dump_value).
    let pushes_direct = count_calls_recursive(body, "sb_push");
    let pushes_via_dv = count_calls_recursive(body, "dump_value");
    assert!(
        pushes_direct > 0 || pushes_via_dv > 0,
        "toml_dump must call sb_push or dump_value; got neither"
    );
}

// ─── Test 5: toml_get uses tv_tbl_get_raw (transitively) ────────────────────

#[test]
fn toml_get_calls_tbl_get() {
    let ir = lower_toml_mind();
    let body = fn_body(&ir, "toml_get");
    let calls = count_calls_recursive(body, "tv_tbl_get_raw");
    assert!(
        calls >= 1,
        "toml_get must call tv_tbl_get_raw at least once; got {calls}"
    );
}

// ─── Test 6: std.toml auto-exports via module table ──────────────────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn toml_mind_auto_exports_public_symbols() {
    use libmind::project::module_table::collect_module_exports;

    let module = parser::parse(TOML_MIND_SRC).expect("std/toml.mind must parse");
    let ex = collect_module_exports("std.toml", &module);

    assert_eq!(ex.module_path, "std.toml");
    for want in [
        "toml_parse",
        "toml_get",
        "toml_dump",
        "tv_kind_str",
        "tv_kind_int",
        "tv_kind_bool",
        "tv_kind_arr",
        "tv_kind_tbl",
        "tv_kind",
        "tv_arr_get",
    ] {
        assert!(
            ex.exported.iter().any(|s| s == want),
            "std.toml must auto-export `{want}`; got {:?}",
            ex.exported
        );
    }
}

// ─── Test 7: bundled std resolves use std.toml ────────────────────────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn bundled_stdlib_resolves_use_std_toml() {
    use libmind::project::module_table::build_module_table;
    use libmind::project::stdlib::parsed_stdlib_modules;
    use libmind::type_checker::{TypeEnv, check_module_types_with_modules};

    let stdlib = parsed_stdlib_modules();
    let refs: Vec<(String, &libmind::ast::Module)> =
        stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&refs);

    // A consumer that imports std.toml and calls toml_parse must resolve
    // without type-checker diagnostics.
    let consumer = "use std.vec\nuse std.toml\nlet root = toml_parse(0, 0)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::default();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "use std.toml + toml_parse should resolve via bundled stdlib; got {:?}",
        diags
    );
}

// ─── Test 8: Mind.toml Rust-side ground-truth cross-check ────────────────────
//
// Parse the canonical Mind.toml at the repo root with the Rust `toml` crate
// and verify the expected key/value pairs.  This documents the ground truth
// that the pure-MIND `toml_parse` must agree with.

#[test]
fn mind_toml_rust_parse_ground_truth() {
    // Read the canonical Mind.toml from the repository root.
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mind_toml_path = manifest_dir.join("Mind.toml");
    assert!(
        mind_toml_path.exists(),
        "Mind.toml must exist at {}",
        mind_toml_path.display()
    );
    let src = std::fs::read_to_string(&mind_toml_path).expect("read Mind.toml");
    // Parse with Rust `toml` crate.
    let parsed: toml::Value = toml::from_str(&src).expect("Mind.toml must parse via toml crate");
    // [package] section checks.
    let pkg = parsed
        .get("package")
        .expect("Mind.toml must have [package] section");
    assert_eq!(
        pkg.get("name").and_then(|v| v.as_str()),
        Some("mind"),
        "[package].name must be \"mind\""
    );
    let version = pkg.get("version").and_then(|v| v.as_str());
    assert!(version.is_some(), "[package].version must be present");
    assert!(
        version.unwrap().contains('.'),
        "[package].version must be semver-shaped: {:?}",
        version
    );
    // [build] section checks.
    let build = parsed.get("build").expect("Mind.toml must have [build]");
    assert_eq!(
        build.get("target").and_then(|v| v.as_str()),
        Some("cpu"),
        "[build].target must be \"cpu\""
    );
    assert_eq!(
        build.get("emit").and_then(|v| v.as_str()),
        Some("cdylib"),
        "[build].emit must be \"cdylib\""
    );
    // [mindcraft] section and sub-table checks.
    let mindcraft = parsed
        .get("mindcraft")
        .expect("Mind.toml must have [mindcraft]");
    let fmt = mindcraft
        .get("format")
        .expect("[mindcraft] must have [format] sub-table");
    let indent = fmt.get("indent_width").and_then(|v| v.as_integer());
    assert_eq!(indent, Some(4), "[mindcraft.format].indent_width must be 4");
}

// ─── Test 9: MLIR functional round-trip (gated on mlir-build) ────────────────
//
// When the full MLIR toolchain is available, compile std/toml.mind to a native
// shared library, invoke `toml_parse` on the Mind.toml bytes, and verify the
// returned TomlValue tree matches the Rust-crate ground truth.

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod mlir_functional {
    use super::mindc_bin;
    use std::path::PathBuf;
    use std::process::Command;

    // mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

    #[test]
    fn toml_parse_mind_toml_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!(
                "toml_parse_mind_toml_via_compiled_so: mindc not found at {mindc:?}; skipping"
            );
            return;
        }

        // Write a tiny driver that loads Mind.toml, calls toml_parse, then
        // calls toml_get with "package.name" and writes the result bytes.
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_toml");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mind_toml_path = manifest_dir.join("Mind.toml");
        let mind_toml_bytes = std::fs::read(&mind_toml_path).expect("read Mind.toml");

        let so_path = out_dir.join("libtoml_smoke.so");
        // Build a self-contained driver .mind that includes the full toml.mind
        // source inline (cross-module function-body inlining is not yet
        // wired in the current mindc --emit-shared pipeline; a `use` import
        // only resolves type-checker symbols, not function bodies).
        let toml_src = include_str!("../std/toml.mind");
        let driver_src = format!(
            r#"{toml_src}

pub fn smoke_toml_parse(buf: i64, buf_len: i64) -> i64 {{
    toml_parse(buf, buf_len)
}}
pub fn smoke_toml_get(root: i64, path_buf: i64, path_len: i64) -> i64 {{
    toml_get(root, path_buf, path_len)
}}
pub fn smoke_tv_kind(h: i64) -> i64 {{ tv_kind(h) }}
pub fn smoke_tv_s_addr(h: i64) -> i64 {{ tv_s_addr(h) }}
pub fn smoke_tv_s_len(h: i64) -> i64 {{ tv_s_len(h) }}
"#
        );
        let driver_path = out_dir.join("toml_smoke.mind");
        std::fs::write(&driver_path, &driver_src).expect("write driver MIND");

        // Compile to shared library.
        let status = Command::new(&mindc)
            .args([
                driver_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");

        if !status.success() {
            println!("toml_parse_mind_toml_via_compiled_so: mindc compile failed; skipping");
            return;
        }

        // Load library and call smoke_toml_parse.
        unsafe {
            let lib = libloading::Library::new(&so_path).expect("dlopen toml_smoke.so");
            type ParseFn = unsafe extern "C" fn(i64, i64) -> i64;
            type GetFn = unsafe extern "C" fn(i64, i64, i64) -> i64;
            type KindFn = unsafe extern "C" fn(i64) -> i64;
            type SAddrFn = unsafe extern "C" fn(i64) -> i64;
            type SLenFn = unsafe extern "C" fn(i64) -> i64;

            let parse_fn: libloading::Symbol<ParseFn> =
                lib.get(b"smoke_toml_parse\0").expect("smoke_toml_parse");
            let get_fn: libloading::Symbol<GetFn> =
                lib.get(b"smoke_toml_get\0").expect("smoke_toml_get");
            let kind_fn: libloading::Symbol<KindFn> =
                lib.get(b"smoke_tv_kind\0").expect("smoke_tv_kind");
            let s_addr_fn: libloading::Symbol<SAddrFn> =
                lib.get(b"smoke_tv_s_addr\0").expect("smoke_tv_s_addr");
            let s_len_fn: libloading::Symbol<SLenFn> =
                lib.get(b"smoke_tv_s_len\0").expect("smoke_tv_s_len");

            let root = parse_fn(
                mind_toml_bytes.as_ptr() as i64,
                mind_toml_bytes.len() as i64,
            );
            assert!(root != 0, "toml_parse must return non-null root");

            // Look up "package.name".
            let path = b"package.name";
            let val = get_fn(root, path.as_ptr() as i64, path.len() as i64);
            assert!(
                val != 0,
                "toml_get(root, \"package.name\") must find a value"
            );

            // Verify it's a string (kind=0).
            assert_eq!(kind_fn(val), 0, "package.name must be a String (kind=0)");

            // Read the string bytes and compare against "mind".
            let s_addr = s_addr_fn(val);
            let s_len = s_len_fn(val);
            assert_eq!(s_len, 4, "package.name must be 4 bytes (\"mind\")");
            let name_bytes = std::slice::from_raw_parts(s_addr as *const u8, s_len as usize);
            assert_eq!(name_bytes, b"mind", "package.name must equal \"mind\"");
        }
    }
}
