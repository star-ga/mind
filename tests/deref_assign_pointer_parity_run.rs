// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Deref-assign POINTER PARITY runtime gate: the D4 carve-out must not change the
//! value of any program main already compiles.
//!
//! A `&mut T` parameter is a CELL (the slot holding the record address, needed for
//! `*p = v`) only when its function dereferences it or forwards it to a cell formal
//! (`eval::deref_cells`). Every other `&mut` parameter keeps main's POINTER
//! representation. D3/D4 first made every `&mut <struct>` a cell and refused the
//! pointer uses; eight of the programs below were refused although main compiles
//! and runs them correctly (measured 2026-09-16).
//!
//! The expected values are the values main 5724a6ee produced for this exact module
//! (measured 2026-09-16, `mindc --emit-shared`, ctypes call). They are pinned as
//! numbers, not as "whatever main gives", so the gate stays a gate after this lands.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test deref_assign_pointer_parity_run`
//! Generic types only — no private consumer source.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;

use std::process::Command;

const PREAMBLE: &str =
    "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair,\n    n: i64\n}\n";

/// Pointer-only module: no `*p` anywhere, so it also compiles on main.
const POINTER_MODULE: &str = r#"
fn set5(p: &mut Pair) {
    p.x = 5
}
fn set9(p: &mut Pair) {
    p.x = 9
}
fn set4(p: &mut Pair) {
    p.x = 4
}
fn mid4(p: &mut Pair) {
    set4(p)
}
fn get_mut(p: &mut Pair) -> i64 {
    return p.x + p.y
}
fn get_imm(p: &Pair) -> i64 {
    return p.x + p.y
}
fn get_imm10(p: &Pair) -> i64 {
    return p.x * 10 + p.y
}
fn bump(p: &mut i64) {
    let q = p
}
fn set_by_value(p: Pair) {
    p.x = 8
}
pub fn mut_structfield_write() -> i64 {
    let h = H { pt: Pair { x: 1, y: 2 }, n: 0 }
    set5(&mut h.pt)
    return h.pt.x
}
pub fn mut_structfield_read() -> i64 {
    let h = H { pt: Pair { x: 3, y: 4 }, n: 0 }
    return get_mut(&mut h.pt)
}
pub fn imm_structfield_read() -> i64 {
    let h = H { pt: Pair { x: 3, y: 4 }, n: 0 }
    return get_imm(&h.pt)
}
pub fn mut_wholevar_write() -> i64 {
    let a = Pair { x: 1, y: 2 }
    set9(&mut a)
    return a.x
}
pub fn imm_wholevar_read() -> i64 {
    let a = Pair { x: 1, y: 2 }
    return get_imm10(&a)
}
pub fn let_bind_mut_field() -> i64 {
    let h = H { pt: Pair { x: 1, y: 2 }, n: 0 }
    let r = &mut h.pt
    return h.pt.x
}
pub fn let_bind_then_write() -> i64 {
    let h = H { pt: Pair { x: 1, y: 2 }, n: 0 }
    let r = &mut h.pt
    r.x = 6
    return h.pt.x
}
pub fn forward_mut_param() -> i64 {
    let a = Pair { x: 1, y: 2 }
    mid4(&mut a)
    return a.x
}
pub fn mut_scalarfield() -> i64 {
    let h = H { pt: Pair { x: 1, y: 2 }, n: 3 }
    bump(&mut h.n)
    return h.n
}
pub fn struct_by_value_param_write() -> i64 {
    let a = Pair { x: 1, y: 2 }
    set_by_value(a)
    return a.x
}
"#;

/// Main 5724a6ee's values for `POINTER_MODULE`.
const POINTER_EXPECTED: &[(&str, i64)] = &[
    ("mut_structfield_write", 5),
    ("mut_structfield_read", 7),
    ("imm_structfield_read", 7),
    ("mut_wholevar_write", 9),
    ("imm_wholevar_read", 12),
    ("let_bind_mut_field", 1),
    ("let_bind_then_write", 6),
    ("forward_mut_param", 4),
    ("mut_scalarfield", 3),
    ("struct_by_value_param_write", 8),
];

/// Pointer and cell parameters in ONE module: the inference is per parameter, so a
/// pointer write and a cell replacement must each keep their own meaning.
const MIXED_MODULE: &str = r#"
fn replace(p: &mut Pair, new: Pair) {
    *p = new
}
fn via(q: &mut Pair, new: Pair) {
    replace(q, new)
}
fn set5(p: &mut Pair) {
    p.x = 5
}
fn consume(value: Pair) -> i64 {
    return value.x * 100 + value.y
}
fn snapshot(p: &mut Pair) -> i64 {
    return consume(*p)
}
fn do_replace(h: H, b: Pair) {
    replace(&mut h.pt, b)
}
fn do_via(h: H, b: Pair) {
    via(&mut h.pt, b)
}
fn do_set(h: H) {
    set5(&mut h.pt)
}
fn do_snapshot(h: H) -> i64 {
    return snapshot(&mut h.pt)
}
pub fn pointer_then_cell() -> i64 {
    let h = H { pt: Pair { x: 1, y: 2 }, n: 0 }
    let b = Pair { x: 10, y: 20 }
    do_set(h)
    let first = h.pt.x
    do_replace(h, b)
    return first * 1000 + h.pt.x
}
pub fn cell_then_pointer() -> i64 {
    let h = H { pt: Pair { x: 1, y: 2 }, n: 0 }
    let b = Pair { x: 10, y: 20 }
    do_via(h, b)
    do_set(h)
    return h.pt.x * 1000 + b.x
}
pub fn cell_read() -> i64 {
    let h = H { pt: Pair { x: 3, y: 4 }, n: 0 }
    return do_snapshot(h)
}
"#;

/// * `pointer_then_cell`: the pointer write gives 5, then the replacement makes
///   `h.pt` denote `b` (10) -> 5_010.
/// * `cell_then_pointer`: after the forwarded replacement `h.pt` IS `b`, so the
///   pointer write through `h.pt` is visible in `b` -> 5_005 (a field copy would
///   leave `b.x` at 10 and give 5_010).
/// * `cell_read`: a by-value dereference read -> 304.
const MIXED_EXPECTED: &[(&str, i64)] = &[
    ("pointer_then_cell", 5010),
    ("cell_then_pointer", 5005),
    ("cell_read", 304),
];

/// Compile `src` to a fresh `.so` and call each zero-argument `pub fn`, returning
/// `None` only when `gate::compiled` rules the host lacks the native backend.
fn run_module(target: &str, src: &str, names: &[&str]) -> Option<Vec<i64>> {
    let mindc = common::require_mindc();
    let dir = tempfile::TempDir::new().expect("unique temp dir");
    let mind = dir.path().join(format!("{target}.mind"));
    let so = dir.path().join(format!("{target}.so"));
    std::fs::write(&mind, format!("{PREAMBLE}{src}")).expect("write src");
    assert!(!so.exists(), "temp dir must start without the artifact");
    let out = Command::new(&mindc)
        .args([
            mind.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    if !common::gate::compiled(target, &out) {
        return None;
    }
    assert!(
        so.exists(),
        "{target}: compile produced no fresh `.so`:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let py = "import ctypes, sys\n\
              lib = ctypes.CDLL(sys.argv[1])\n\
              for n in sys.argv[2:]:\n\
              \x20   f = getattr(lib, n)\n\
              \x20   f.restype = ctypes.c_int64\n\
              \x20   print(f())\n";
    let mut args = vec![
        "-c".to_string(),
        py.to_string(),
        so.to_string_lossy().into_owned(),
    ];
    args.extend(names.iter().map(|n| n.to_string()));
    let run = Command::new("python3")
        .args(&args)
        .output()
        .expect("python3");
    assert!(
        run.status.success(),
        "{target}: calling the probes failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
    let values: Vec<i64> = String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(|l| l.trim().parse().expect("probe prints an i64"))
        .collect();
    assert_eq!(values.len(), names.len(), "{target}: one value per probe");
    Some(values)
}

fn assert_values(target: &str, src: &str, expected: &[(&str, i64)]) {
    let names: Vec<&str> = expected.iter().map(|(n, _)| *n).collect();
    let Some(got) = run_module(target, src, &names) else {
        return;
    };
    let got: Vec<(&str, i64)> = names.iter().copied().zip(got).collect();
    assert_eq!(
        got, expected,
        "{target}: values differ from the pinned ones"
    );
}

#[test]
fn pointer_programs_keep_main_values() {
    assert_values("deref_pointer_parity", POINTER_MODULE, POINTER_EXPECTED);
}

#[test]
fn pointer_and_cell_parameters_coexist_in_one_module() {
    assert_values("deref_pointer_cell_mixed", MIXED_MODULE, MIXED_EXPECTED);
}

/// A field read on a let-bound dereference has no struct-field ABI entry yet. It must
/// stay a LOUD refusal (no artifact), never a silent wrong read.
///
/// deferred: `let v = *p; v.x` would record `v`'s owner from `p`'s referent the way
/// a struct-returning call's result is recorded; refused today because the D4 load
/// lowers through the scalar ABI and leaves `v` without an owner — upgrade path:
/// type the `__mind_load_i64` result with the referent struct in the Let arm, then
/// flip this test to a value assertion (3 * 100 + 4 = 304).
#[test]
fn let_bound_dereference_field_read_is_refused_loudly() {
    let mindc = common::require_mindc();
    let dir = tempfile::TempDir::new().expect("unique temp dir");
    let mind = dir.path().join("deref_let_field.mind");
    let so = dir.path().join("deref_let_field.so");
    let src = format!(
        "{PREAMBLE}fn snap(p: &mut Pair) -> i64 {{\n    let v = *p\n    return v.x * 100 + v.y\n}}\nfn outer(h: H) -> i64 {{\n    return snap(&mut h.pt)\n}}\npub fn probe() -> i64 {{\n    let h = H {{ pt: Pair {{ x: 3, y: 4 }}, n: 0 }}\n    return outer(h)\n}}\n"
    );
    std::fs::write(&mind, src).expect("write src");
    let out = Command::new(&mindc)
        .args([
            mind.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() && !so.exists(),
        "expected a refusal with no artifact; status {:?}, stderr:\n{stderr}",
        out.status
    );
    assert!(
        stderr.contains("E6009") && stderr.contains("struct field read"),
        "expected the E6009 struct-field-read refusal; got:\n{stderr}"
    );
}
