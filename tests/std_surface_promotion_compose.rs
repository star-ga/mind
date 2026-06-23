// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! STAGE 1 — compositional interaction tests for the std-surface promotion.
//!
//! The std-surface "default flip" promotes a set of high-level constructs
//! (payload `match`, method dispatch / UFCS desugar, nested-struct field
//! writes, enum-tag matching, `let mut` mutation, nested `while`) from the
//! experimental flag into the default surface. The isolation tests in
//! `std_surface_method_call.rs` prove each construct *lowers* to the right IR.
//! This file is the consensus-required prerequisite to the flip: it proves the
//! promoted constructs **execute correctly when INTERACTING**, not just when
//! lowered in isolation.
//!
//! Two verification strata, mirroring the proven patterns in the repo:
//!   * IR stratum (`std_surface_method_call.rs`): parse -> `lower_to_ir` ->
//!     assert the composed call/site exists in the IR (no const-0 placeholder).
//!   * EXEC stratum (`cross_module_cdylib_compose.rs` / `std_surface_cdylib_link.rs`):
//!     compile via the shipped `mindc --emit-shared`, `dlopen` the `.so` through
//!     python3 + ctypes, call the entry, and assert the CORRECT runtime value.
//!     A const-0 / garbage / pre-mutation result is a FAIL — that is precisely
//!     the silent-miscompile class the flip must not ship.
//!
//! Gated: `cargo test --release --features "mlir-build std-surface
//!         cross-module-imports" --test std_surface_promotion_compose`.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

// ──────────────────────────────────────────────────────────────────────────────
// Harness
// ──────────────────────────────────────────────────────────────────────────────

fn must_parse(src: &str) -> libmind::ast::Module {
    parser::parse(src).unwrap_or_else(|errs| {
        panic!(
            "parse failed with {} error(s):\n{}",
            errs.len(),
            errs.iter()
                .map(|e| format!("  {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    })
}

/// Count `Instr::Call { name == target }` across the whole IR, recursing into
/// function bodies and control-flow blocks. (Same shape as the isolation test.)
fn count_calls_named(instrs: &[Instr], target: &str) -> usize {
    let mut n = 0;
    for instr in instrs {
        match instr {
            Instr::Call { name, .. } if name == target => n += 1,
            Instr::FnDef { body, .. } => n += count_calls_named(body, target),
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                n += count_calls_named(cond_instrs, target);
                n += count_calls_named(then_instrs, target);
                n += count_calls_named(else_instrs, target);
            }
            Instr::While {
                body, cond_instrs, ..
            } => {
                n += count_calls_named(cond_instrs, target);
                n += count_calls_named(body, target);
            }
            _ => {}
        }
    }
    n
}

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// Compile `src` to a temp `.so` via `mindc --emit-shared`, `dlopen` it through
/// python3 + ctypes, call the zero-arg `i64` entry `probe`, and return its
/// value. `None` signals the MLIR backend was unavailable (skip), distinguished
/// from a wrong value (assert-fail). A compile or runtime failure panics.
fn run_probe(tag: &str, src: &str) -> Option<i64> {
    let mindc = mindc_bin();
    if !mindc.exists() {
        eprintln!("compose[{tag}]: mindc not found; skipping");
        return None;
    }
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("mind_compose_{tag}.mind"));
    let so_path = dir.join(format!("mind_compose_{tag}.so"));
    let _ = std::fs::remove_file(&so_path);
    std::fs::write(&src_path, src).expect("write probe source");

    let out = Command::new(&mindc)
        .arg(src_path.to_str().unwrap())
        .arg("--emit-shared")
        .arg(so_path.to_str().unwrap())
        .output()
        .expect("spawn mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        // A toolchain without the MLIR backend cannot emit a shared object;
        // skip rather than fail (the std_surface_* gating convention).
        if stderr.contains("MLIR") || stderr.contains("mlir-opt") || stderr.contains("clang") {
            eprintln!("compose[{tag}]: mindc --emit-shared unavailable; skipping\n{stderr}");
            return None;
        }
        panic!(
            "compose[{tag}]: mindc --emit-shared failed:\nstdout: {}\nstderr: {stderr}",
            String::from_utf8_lossy(&out.stdout)
        );
    }

    let so_str = so_path.to_string_lossy().into_owned();
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{so_str}')\n\
         lib.probe.restype = ctypes.c_int64\n\
         print(lib.probe())\n"
    );
    let py_out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3 on PATH");
    assert!(
        py_out.status.success(),
        "compose[{tag}]: dlopen/call failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&py_out.stdout),
        String::from_utf8_lossy(&py_out.stderr)
    );
    let s = String::from_utf8_lossy(&py_out.stdout);
    let v: i64 = s
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("compose[{tag}]: non-integer probe() output {s:?}"));
    Some(v)
}

/// Assert `probe()` executes to `expected`, or skip if the backend is absent.
fn assert_probe(tag: &str, src: &str, expected: i64) {
    if let Some(got) = run_probe(tag, src) {
        assert_eq!(
            got, expected,
            "compose[{tag}]: promoted constructs miscompiled when interacting — \
             probe() returned {got}, expected {expected}. A wrong/const value here \
             is a silent miscompile and a default-flip blocker."
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. match × Option-payload — build an Option-shaped payload enum, match it,
//    extract + transform the bound value.  Some(41) -> match -> v + 1 = 42.
//
//    (The built-in `Some`/`None` constructors are not yet free symbols in a
//    `--emit-shared` cdylib, so we model the Option payload with a user enum —
//    the construct under test is *payload binding in a match arm*, which is the
//    promoted surface, independent of the std `Option` sugar.)
// ──────────────────────────────────────────────────────────────────────────────

const SRC_MATCH_OPTION: &str = r#"
enum Opt { Some(i64), None }
fn build(x: i64) -> Opt { Opt::Some(x) }
pub fn probe() -> i64 {
    let o = build(41)
    match o {
        Opt::Some(v) => v + 1,
        Opt::None => 0
    }
}
"#;

#[test]
fn match_option_payload_binds_and_transforms_value() {
    // IR stratum: the payload binding must lower (the `build` constructor call
    // is present — no const-0 collapse of the whole pipeline).
    let ir = lower_to_ir(&must_parse(SRC_MATCH_OPTION));
    assert!(
        count_calls_named(&ir.instrs, "build") >= 1,
        "match × Option-payload pipeline collapsed in lowering.\nIR: {:?}",
        ir.instrs
    );
    // EXEC stratum: Some(41) -> v + 1 = 42.
    assert_probe("match_option", SRC_MATCH_OPTION, 42);
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. method-dispatch × String — a zero-arg accessor `s.len()` (lowers to a
//    field load) AND a UFCS method `s.push(b)` (desugars to `string_push`),
//    co-resident on the same String receiver.
//
//    l = s.len() = 5 ; s2 = s.push(65) (len -> 6) ; result = l*100 + s2.len().
//    Correct = 506.  A 500 (s2.len() == 0) means the UFCS return value was lost.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_METHOD_STRING: &str = r#"
struct String { addr: i64, len: i64, cap: i64 }
fn string_push(s: String, b: i64) -> String {
    String { addr: s.addr, len: s.len + 1, cap: s.cap }
}
pub fn probe() -> i64 {
    let s = String { addr: 0, len: 5, cap: 8 }
    let l = s.len()
    let s2 = s.push(65)
    l * 100 + s2.len()
}
"#;

#[test]
fn method_dispatch_accessor_and_ufcs_on_string() {
    let ir = lower_to_ir(&must_parse(SRC_METHOD_STRING));
    // Accessor `s.len()` -> field load; UFCS `s.push(b)` -> `string_push` call.
    assert!(
        count_calls_named(&ir.instrs, "__mind_load_i64") >= 1,
        "`s.len()` accessor did not lower to a field load.\nIR: {:?}",
        ir.instrs
    );
    assert!(
        count_calls_named(&ir.instrs, "string_push") >= 1,
        "`s.push(b)` UFCS did not desugar to `string_push`.\nIR: {:?}",
        ir.instrs
    );
    // EXEC: l*100 + s2.len() = 5*100 + 6 = 506.
    assert_probe("method_string", SRC_METHOD_STRING, 506);
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. nested-struct × field-write — write a field of a nested struct member,
//    then read it back.  o.inner.v = 99 ; o.inner.v == 99.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_NESTED_STRUCT: &str = r#"
struct Inner { v: i64, w: i64 }
struct Outer { inner: Inner, tag: i64 }
pub fn probe() -> i64 {
    let mut o = Outer { inner: Inner { v: 1, w: 2 }, tag: 5 }
    o.inner.v = 99
    o.inner.v
}
"#;

#[test]
fn nested_struct_field_write_then_read() {
    assert_probe("nested_struct", SRC_NESTED_STRUCT, 99);
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. enum-tag-match × struct-field-write — match an enum tag and mutate a
//    struct field in each arm.  apply(On)->x=1, apply(Off)->x=0 ; combine.
//
//    KNOWN FLIP BLOCKER (discovered by this test): a `match` STATEMENT does not
//    forward enclosing-scope mutations made in its arms — the post-match read
//    reverts to the pre-match value (the `if`-statement form does this
//    correctly).  This test asserts the CORRECT result (10) and therefore FAILS
//    until the match region-exit rebinding (F2) is fixed.  See the report.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_ENUM_TAG_FIELD: &str = r#"
enum Mode { On, Off }
fn apply(m: Mode, seed: i64) -> i64 {
    let mut x: i64 = seed
    match m {
        Mode::On => { x = 1 },
        Mode::Off => { x = 0 }
    }
    x
}
pub fn probe() -> i64 {
    apply(Mode::On, 7) * 10 + apply(Mode::Off, 7)
}
"#;

#[test]
fn enum_tag_match_writes_field_in_each_arm() {
    let ir = lower_to_ir(&must_parse(SRC_ENUM_TAG_FIELD));
    assert!(
        count_calls_named(&ir.instrs, "apply") >= 2,
        "enum-tag-match × field-write pipeline collapsed.\nIR: {:?}",
        ir.instrs
    );
    // EXEC: On->1, Off->0 => 1*10 + 0 = 10.
    assert_probe("enum_tag_field", SRC_ENUM_TAG_FIELD, 10);
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. UFCS-method × borrow-read param — a UFCS method whose argument reads
//    another field of the receiver: `f(s)` computes `s.combine(s.b)`.
//    s_combine(s, k) = s.a + k ; with a=10, b=32 -> 42.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_UFCS_BORROW: &str = r#"
struct S { a: i64, b: i64 }
fn s_combine(s: S, k: i64) -> i64 { s.a + k }
fn f(s: S) -> i64 { s.combine(s.b) }
pub fn probe() -> i64 {
    let v = S { a: 10, b: 32 }
    f(v)
}
"#;

#[test]
fn ufcs_method_reads_receiver_field_as_argument() {
    let ir = lower_to_ir(&must_parse(SRC_UFCS_BORROW));
    assert!(
        count_calls_named(&ir.instrs, "s_combine") >= 1,
        "`s.combine(s.b)` UFCS did not desugar to `s_combine`.\nIR: {:?}",
        ir.instrs
    );
    // EXEC: s.a + s.b = 10 + 32 = 42.
    assert_probe("ufcs_borrow", SRC_UFCS_BORROW, 42);
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. string-literal × match-on-int co-resident — two independent promoted
//    lowerings in one function (a string literal binding and a match
//    expression) must coexist with deterministic SSA numbering and not corrupt
//    each other.  t = 2 -> match -> 30.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_CORESIDENT: &str = r#"
pub fn probe() -> i64 {
    let s = "hello"
    let t: i64 = 2
    let m = match t {
        0 => 10,
        1 => 20,
        _ => 30
    }
    m
}
"#;

#[test]
fn string_literal_and_int_match_coresident() {
    assert_probe("coresident", SRC_CORESIDENT, 30);
}

// ──────────────────────────────────────────────────────────────────────────────
// 7. accessor × UFCS chaining — an accessor and a UFCS method composed in one
//    expression with an intermediate receiver-type resolution.
//    obj.a() + obj.scale(7) ; a=6 -> 6 + 6*7 = 48.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_CHAIN: &str = r#"
struct S { a: i64, b: i64 }
fn s_scale(s: S, k: i64) -> i64 { s.a * k }
pub fn probe() -> i64 {
    let v = S { a: 6, b: 7 }
    v.a() + v.scale(7)
}
"#;

#[test]
fn accessor_and_ufcs_chained_in_one_expression() {
    let ir = lower_to_ir(&must_parse(SRC_CHAIN));
    assert!(
        count_calls_named(&ir.instrs, "s_scale") >= 1,
        "`v.scale(7)` UFCS did not desugar to `s_scale`.\nIR: {:?}",
        ir.instrs
    );
    assert!(
        count_calls_named(&ir.instrs, "__mind_load_i64") >= 1,
        "`v.a()` accessor did not lower to a field load.\nIR: {:?}",
        ir.instrs
    );
    // EXEC: v.a + v.a*7 = 6 + 42 = 48.
    assert_probe("chain", SRC_CHAIN, 48);
}

// ──────────────────────────────────────────────────────────────────────────────
// 8. NESTED WHILE regression (f13d570 unique-label fix) — a function with a
//    triple-nested `while` AND a sibling `while` must compile through mlir-opt
//    with NO duplicate-block-label failure, and run correctly.
//    triple(3) = 3^3 (inner triple-nest) + 3 (sibling loop) = 27 + 3 = 30.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_NESTED_WHILE: &str = r#"
fn triple(n: i64) -> i64 {
    let mut total: i64 = 0
    let mut i: i64 = 0
    while i < n {
        let mut j: i64 = 0
        while j < n {
            let mut k: i64 = 0
            while k < n {
                total = total + 1
                k = k + 1
            }
            j = j + 1
        }
        i = i + 1
    }
    let mut a: i64 = 0
    while a < n {
        total = total + 1
        a = a + 1
    }
    total
}
pub fn probe() -> i64 { triple(3) }
"#;

#[test]
fn nested_and_sibling_while_compile_and_run() {
    // Compiling at all (no duplicate-label mlir-opt failure) is half the test;
    // the other half is the correct count.  3^3 + 3 = 30.
    assert_probe("nested_while", SRC_NESTED_WHILE, 30);
}

// ──────────────────────────────────────────────────────────────────────────────
// 9. `match`-STATEMENT arm mutation of an enclosing scalar `let mut` (regression
//    for BLOCKER 1 — the minimal form of test #4 with no enum/field involved).
//
//    A braced match arm (`1 => { x = 100 }`) parses to a `Node::Block`; the
//    desugar-to-if path must SPLICE that block's statements into the if branch
//    so the If lowering records the enclosing-scope `Assign` as a branch write
//    and merges it into the post-match scope.  Before the fix the read after the
//    match reverted to the pre-match value (returned 7 instead of 100/200) — a
//    silent miscompile.  The `if`-statement form was always correct; only the
//    `match`-statement form was broken.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_MATCH_STMT_SCALAR_MUT: &str = r#"
fn pick(t: i64) -> i64 {
    let mut x: i64 = 7
    match t {
        1 => { x = 100 },
        _ => { x = 200 }
    }
    x
}
pub fn probe() -> i64 {
    pick(1) + pick(0)
}
"#;

#[test]
fn match_stmt_arm_mutates_enclosing_scalar() {
    // pick(1) -> 100, pick(0) -> 200 => 300. A value of 14 (== 7 + 7) is the
    // pre-fix miscompile signature (both arms reverting to the initialiser).
    assert_probe("match_stmt_scalar", SRC_MATCH_STMT_SCALAR_MUT, 300);
}

// ──────────────────────────────────────────────────────────────────────────────
// 10. UFCS method call bound to a `let`, then a field read of the result
//     (regression for BLOCKER 2 — the last Stage-1 silent miscompile).
//
//     `let s2 = s.grow(65)` UFCS-desugars to `buf_grow(s, 65)`. The call IS
//     emitted, but the struct-resolver pre-pass had no `MethodCall` arm in
//     `infer_struct`, so `s2` was never recorded as a `Buf` and the later
//     `s2.len` could not resolve to a field load — it collapsed to const-0.
//     The IDENTICAL direct-call form (`let s2 = buf_grow(s, 65)`) always
//     resolved through the `Node::Call` arm and returned 6, isolating the
//     defect to result-binding/type-registration, not the struct-return ABI.
//     The fix teaches `infer_struct` to resolve the UFCS target's return type
//     the same way the lowering arm forms the desugared call name.
// ──────────────────────────────────────────────────────────────────────────────

const SRC_UFCS_LET_FIELD_READ: &str = r#"
struct Buf { addr: i64, len: i64, cap: i64 }
fn buf_grow(s: Buf, b: i64) -> Buf { Buf { addr: s.addr, len: s.len + 1, cap: s.cap } }
pub fn probe() -> i64 {
    let s = Buf { addr: 0, len: 5, cap: 8 }
    let s2 = s.grow(65)
    s2.len
}
"#;

#[test]
fn ufcs_let_binding_field_read() {
    let ir = lower_to_ir(&must_parse(SRC_UFCS_LET_FIELD_READ));
    // UFCS `s.grow(b)` must desugar to the `buf_grow` free function.
    assert!(
        count_calls_named(&ir.instrs, "buf_grow") >= 1,
        "`s.grow(b)` UFCS did not desugar to `buf_grow`.\nIR: {:?}",
        ir.instrs
    );
    // `s2.len` must resolve to a field load — not collapse to const-0.
    assert!(
        count_calls_named(&ir.instrs, "__mind_load_i64") >= 1,
        "`s2.len` did not lower to a field load (UFCS result type lost).\nIR: {:?}",
        ir.instrs
    );
    // EXEC: buf_grow bumps len 5 -> 6; `s2.len` == 6. A 0 is the pre-fix
    // silent miscompile (the UFCS result's struct type was never registered).
    assert_probe("ufcs_let_field", SRC_UFCS_LET_FIELD_READ, 6);
}
