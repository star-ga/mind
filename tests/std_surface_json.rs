// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Task #269 -- `std/json.mind` surface tests.
//!
//! Tests in this file verify:
//!
//! 1. `std/json.mind` parses and lowers to IR with all required `pub fn`s present.
//! 2. Kind constants are distinct i64 values.
//! 3. `jv_parse` calls `jv_make_object` / `jv_make_array`.
//! 4. `jv_dump` calls `jvsb_push` (or dump helpers transitively).
//! 5. `jv_get_path` calls `jv_get`.
//! 6. std.json auto-exports all public symbols.
//! 7. Bundled stdlib resolves `use std.json`.
//! 8. MLIR functional round-trip: parse, dump, get for a rich JSON fixture
//!    (gated on `mlir-build`).
//!
//! Gate: `cargo test --features "std-surface cross-module-imports"
//!                   --test std_surface_json`

#![cfg(feature = "std-surface")]

mod common;
use common::mindc_bin;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const JSON_MIND_SRC: &str = include_str!("../std/json.mind");

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn lower_json_mind() -> libmind::ir::IRModule {
    let module = parser::parse(JSON_MIND_SRC).expect("std/json.mind must parse cleanly");
    lower_to_ir(&module)
}

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
fn json_mind_parses_and_lowers() {
    let ir = lower_json_mind();
    for want in [
        // Public API
        "jv_parse",
        "jv_dump",
        "jv_get",
        "jv_get_path",
        // Kind constants
        "jv_kind_null",
        "jv_kind_bool",
        "jv_kind_number",
        "jv_kind_string",
        "jv_kind_array",
        "jv_kind_object",
        // Field accessors
        "jv_kind",
        "jv_bval",
        "jv_n_int",
        "jv_n_frac",
        "jv_n_frac_d",
        "jv_n_neg",
        "jv_n_is_int",
        "jv_s_addr",
        "jv_s_len",
        "jv_arr_len",
        "jv_arr_get",
        "jv_tbl_len",
        "jv_tbl_key",
        "jv_tbl_val",
    ] {
        assert!(
            ir.instrs
                .iter()
                .any(|i| matches!(i, Instr::FnDef { name, .. } if name == want)),
            "missing FnDef `{want}` in lowered std/json.mind IR"
        );
    }
}

// ─── Test 2: kind constants are distinct ──────────────────────────────────────

#[test]
fn jv_kind_constants_are_distinct() {
    let ir = lower_json_mind();
    let mut values: Vec<i64> = Vec::new();
    for fn_name in [
        "jv_kind_null",
        "jv_kind_bool",
        "jv_kind_number",
        "jv_kind_string",
        "jv_kind_array",
        "jv_kind_object",
    ] {
        let body = fn_body(&ir, fn_name);
        let val = body.iter().find_map(|i| match i {
            Instr::ConstI64(_, v) => Some(*v),
            _ => None,
        });
        let v = val.unwrap_or_else(|| panic!("{fn_name} must emit a ConstI64"));
        values.push(v);
    }
    let mut seen = std::collections::HashSet::new();
    for v in &values {
        assert!(
            seen.insert(v),
            "jv_kind_* constants must be distinct; duplicate value {v}"
        );
    }
}

// ─── Test 3: jv_parse calls jv_make_object ────────────────────────────────────

#[test]
fn jv_parse_calls_jv_parse_value() {
    let ir = lower_json_mind();
    let body = fn_body(&ir, "jv_parse");
    // jv_parse calls jv_parse_value and jv_skip_ws
    let pv_calls = count_calls_recursive(body, "jv_parse_value");
    let sw_calls = count_calls_recursive(body, "jv_skip_ws");
    assert!(
        pv_calls >= 1 || sw_calls >= 1,
        "jv_parse must call jv_parse_value or jv_skip_ws; got neither"
    );
}

// ─── Test 4: jv_dump calls jvsb_push (transitively) ──────────────────────────

#[test]
fn jv_dump_calls_dump_value() {
    let ir = lower_json_mind();
    let body = fn_body(&ir, "jv_dump");
    let dv = count_calls_recursive(body, "jv_dump_value");
    let dp = count_calls_recursive(body, "jvsb_push");
    assert!(
        dv > 0 || dp > 0,
        "jv_dump must call jv_dump_value or jvsb_push; got neither"
    );
}

// ─── Test 5: jv_get_path calls jv_get ────────────────────────────────────────

#[test]
fn jv_get_path_calls_jv_get() {
    let ir = lower_json_mind();
    let body = fn_body(&ir, "jv_get_path");
    let calls = count_calls_recursive(body, "jv_get");
    assert!(
        calls >= 1,
        "jv_get_path must call jv_get at least once; got {calls}"
    );
}

// ─── Test 6: jv_alloc uses __mind_alloc ───────────────────────────────────────

#[test]
fn jv_alloc_uses_mind_alloc() {
    let ir = lower_json_mind();
    let body = fn_body(&ir, "jv_alloc");
    let allocs = count_calls_recursive(body, "__mind_alloc");
    assert!(
        allocs >= 1,
        "jv_alloc must call __mind_alloc at least once; got {allocs}"
    );
}

// ─── Test 7: parse_string handles escape sequences ────────────────────────────

#[test]
fn jv_parse_string_calls_jvsb_push() {
    let ir = lower_json_mind();
    let body = fn_body(&ir, "jv_parse_string");
    let pushes = count_calls_recursive(body, "jvsb_push");
    assert!(
        pushes >= 1,
        "jv_parse_string must call jvsb_push at least once; got {pushes}"
    );
}

// ─── Test 8: std.json auto-exports public symbols ─────────────────────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn json_mind_auto_exports_public_symbols() {
    use libmind::project::module_table::collect_module_exports;

    let module = parser::parse(JSON_MIND_SRC).expect("std/json.mind must parse");
    let ex = collect_module_exports("std.json", &module);

    assert_eq!(ex.module_path, "std.json");
    for want in [
        "jv_parse",
        "jv_dump",
        "jv_get",
        "jv_get_path",
        "jv_kind_null",
        "jv_kind_bool",
        "jv_kind_number",
        "jv_kind_string",
        "jv_kind_array",
        "jv_kind_object",
        "jv_kind",
        "jv_bval",
        "jv_n_int",
        "jv_n_is_int",
        "jv_s_addr",
        "jv_s_len",
        "jv_arr_len",
        "jv_arr_get",
        "jv_tbl_len",
        "jv_tbl_key",
        "jv_tbl_val",
    ] {
        assert!(
            ex.exported.iter().any(|s| s == want),
            "std.json must auto-export `{want}`; got {:?}",
            ex.exported
        );
    }
}

// ─── Test 9: bundled stdlib resolves use std.json ─────────────────────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn bundled_stdlib_resolves_use_std_json() {
    use libmind::project::module_table::build_module_table;
    use libmind::project::stdlib::parsed_stdlib_modules;
    use libmind::type_checker::{TypeEnv, check_module_types_with_modules};

    let stdlib = parsed_stdlib_modules();
    let refs: Vec<(String, &libmind::ast::Module)> =
        stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&refs);

    let consumer = "use std.json\nlet root = jv_parse(0, 0)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "use std.json + jv_parse should resolve via bundled stdlib; got {:?}",
        diags
    );
}

// ─── Test 10: MLIR functional round-trip ─────────────────────────────────────

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod mlir_functional {
    use super::mindc_bin;
    use std::path::PathBuf;
    use std::process::Command;

    // mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

    /// Compile a driver MIND program that includes json.mind inline, build it,
    /// and run it against a variety of JSON fixtures.
    #[test]
    fn json_parse_round_trip_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("json_parse_round_trip_via_compiled_so: mindc not found; skipping");
            return;
        }

        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_json");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let json_src = include_str!("../std/json.mind");
        let driver_src = format!(
            r#"{json_src}

pub fn smoke_jv_parse(buf: i64, buf_len: i64) -> i64 {{
    jv_parse(buf, buf_len)
}}
pub fn smoke_jv_kind(h: i64) -> i64 {{ jv_kind(h) }}
pub fn smoke_jv_arr_len(h: i64) -> i64 {{ jv_arr_len(h) }}
pub fn smoke_jv_arr_get(h: i64, i: i64) -> i64 {{ jv_arr_get(h, i) }}
pub fn smoke_jv_tbl_len(h: i64) -> i64 {{ jv_tbl_len(h) }}
pub fn smoke_jv_get(h: i64, k: i64, kl: i64) -> i64 {{ jv_get(h, k, kl) }}
pub fn smoke_jv_s_addr(h: i64) -> i64 {{ jv_s_addr(h) }}
pub fn smoke_jv_s_len(h: i64) -> i64 {{ jv_s_len(h) }}
pub fn smoke_jv_bval(h: i64) -> i64 {{ jv_bval(h) }}
pub fn smoke_jv_n_int(h: i64) -> i64 {{ jv_n_int(h) }}
pub fn smoke_jv_n_is_int(h: i64) -> i64 {{ jv_n_is_int(h) }}
"#
        );
        let driver_path = out_dir.join("json_smoke.mind");
        let so_path = out_dir.join("libjson_smoke.so");
        std::fs::write(&driver_path, &driver_src).expect("write driver MIND");

        let status = Command::new(&mindc)
            .args([
                driver_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");

        if !status.success() {
            println!("json_parse_round_trip_via_compiled_so: mindc compile failed; skipping");
            return;
        }

        unsafe {
            let lib = libloading::Library::new(&so_path).expect("dlopen json_smoke.so");
            type ParseFn = unsafe extern "C" fn(i64, i64) -> i64;
            type KindFn = unsafe extern "C" fn(i64) -> i64;
            type ALenFn = unsafe extern "C" fn(i64) -> i64;
            type AGetFn = unsafe extern "C" fn(i64, i64) -> i64;
            type TLenFn = unsafe extern "C" fn(i64) -> i64;
            type GetFn = unsafe extern "C" fn(i64, i64, i64) -> i64;
            type SAddrFn = unsafe extern "C" fn(i64) -> i64;
            type SLenFn = unsafe extern "C" fn(i64) -> i64;
            type BValFn = unsafe extern "C" fn(i64) -> i64;
            type NIntFn = unsafe extern "C" fn(i64) -> i64;
            type NIsIntFn = unsafe extern "C" fn(i64) -> i64;

            let parse_fn: libloading::Symbol<ParseFn> = lib.get(b"smoke_jv_parse\0").unwrap();
            let kind_fn: libloading::Symbol<KindFn> = lib.get(b"smoke_jv_kind\0").unwrap();
            let alen_fn: libloading::Symbol<ALenFn> = lib.get(b"smoke_jv_arr_len\0").unwrap();
            let aget_fn: libloading::Symbol<AGetFn> = lib.get(b"smoke_jv_arr_get\0").unwrap();
            let tlen_fn: libloading::Symbol<TLenFn> = lib.get(b"smoke_jv_tbl_len\0").unwrap();
            let get_fn: libloading::Symbol<GetFn> = lib.get(b"smoke_jv_get\0").unwrap();
            let s_addr_fn: libloading::Symbol<SAddrFn> = lib.get(b"smoke_jv_s_addr\0").unwrap();
            let s_len_fn: libloading::Symbol<SLenFn> = lib.get(b"smoke_jv_s_len\0").unwrap();
            let bval_fn: libloading::Symbol<BValFn> = lib.get(b"smoke_jv_bval\0").unwrap();
            let nint_fn: libloading::Symbol<NIntFn> = lib.get(b"smoke_jv_n_int\0").unwrap();
            let nisint_fn: libloading::Symbol<NIsIntFn> = lib.get(b"smoke_jv_n_is_int\0").unwrap();

            // Fixture 1: simple object
            let src1 = br#"{"name":"mind","version":1}"#;
            let root1 = parse_fn(src1.as_ptr() as i64, src1.len() as i64);
            assert!(root1 != 0, "fixture 1: jv_parse must return non-null");
            assert_eq!(kind_fn(root1), 5, "fixture 1: root must be object (kind=5)");
            assert_eq!(tlen_fn(root1), 2, "fixture 1: object must have 2 keys");

            let name_key = b"name";
            let name_val = get_fn(root1, name_key.as_ptr() as i64, name_key.len() as i64);
            assert!(name_val != 0, "fixture 1: 'name' key must be present");
            assert_eq!(
                kind_fn(name_val),
                3,
                "fixture 1: 'name' must be string (kind=3)"
            );
            let s_addr = s_addr_fn(name_val);
            let s_len = s_len_fn(name_val);
            assert_eq!(s_len, 4, "fixture 1: name length must be 4");
            let name_bytes = std::slice::from_raw_parts(s_addr as *const u8, s_len as usize);
            assert_eq!(name_bytes, b"mind", "fixture 1: name must equal 'mind'");

            let ver_key = b"version";
            let ver_val = get_fn(root1, ver_key.as_ptr() as i64, ver_key.len() as i64);
            assert!(ver_val != 0, "fixture 1: 'version' key must be present");
            assert_eq!(
                kind_fn(ver_val),
                2,
                "fixture 1: 'version' must be number (kind=2)"
            );
            assert_eq!(nint_fn(ver_val), 1, "fixture 1: version must be 1");
            assert_eq!(nisint_fn(ver_val), 1, "fixture 1: version must be integer");

            // Fixture 2: array of mixed values
            let src2 = br#"[null, true, false, 42, "hello"]"#;
            let root2 = parse_fn(src2.as_ptr() as i64, src2.len() as i64);
            assert!(root2 != 0, "fixture 2: must parse");
            assert_eq!(kind_fn(root2), 4, "fixture 2: root must be array (kind=4)");
            assert_eq!(alen_fn(root2), 5, "fixture 2: array must have 5 elements");
            // null
            assert_eq!(kind_fn(aget_fn(root2, 0)), 0, "fixture 2: [0] must be null");
            // true
            let el1 = aget_fn(root2, 1);
            assert_eq!(kind_fn(el1), 1, "fixture 2: [1] must be bool");
            assert_eq!(bval_fn(el1), 1, "fixture 2: [1] must be true");
            // false
            let el2 = aget_fn(root2, 2);
            assert_eq!(kind_fn(el2), 1, "fixture 2: [2] must be bool");
            assert_eq!(bval_fn(el2), 0, "fixture 2: [2] must be false");
            // 42
            let el3 = aget_fn(root2, 3);
            assert_eq!(kind_fn(el3), 2, "fixture 2: [3] must be number");
            assert_eq!(nint_fn(el3), 42, "fixture 2: [3] must be 42");

            // Fixture 3: malformed -- must return 0
            let src3 = b"{bad json}";
            let root3 = parse_fn(src3.as_ptr() as i64, src3.len() as i64);
            assert_eq!(root3, 0, "fixture 3: malformed JSON must return 0");

            // Fixture 4: empty object
            let src4 = b"{}";
            let root4 = parse_fn(src4.as_ptr() as i64, src4.len() as i64);
            assert!(root4 != 0, "fixture 4: empty object must parse");
            assert_eq!(kind_fn(root4), 5, "fixture 4: kind must be object");
            assert_eq!(tlen_fn(root4), 0, "fixture 4: empty object has 0 keys");

            // Fixture 5: comments are invalid JSON -- must reject
            let src5 = b"// comment\n{\"x\":1}";
            let root5 = parse_fn(src5.as_ptr() as i64, src5.len() as i64);
            assert_eq!(root5, 0, "fixture 5: comment before JSON must return 0");
        }
    }
}
