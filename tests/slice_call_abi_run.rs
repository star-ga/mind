// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Issue #234 — native array-to-slice call ABI and capability gate.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const VALID: &str = r#"
fn inspect(xs: &[i64]) -> i64 {
    let ys = xs
    return ys.length * 100 + ys[0] * 10 + ys.get(1)
}

fn mutate(xs: &mut [i64]) -> i64 {
    xs[0] = 4
    xs.set(1, 2)
    return xs[0] * 10 + xs[1]
}

fn identity(xs: &[i64]) -> &[i64] {
    return xs
}

fn dynamic_array(x: i64) -> array<i64> {
    return [x]
}

// Legacy public ABI: an owned Vec handle may pass through an opaque i64.
fn opaque_owned_handle(x: i64) -> i64 {
    let xs: array<i64> = [x]
    return xs
}

pub fn value_slice() -> i64 {
    let xs: array<i64> = [1, 2, 3]
    return inspect(xs)
}

pub fn literal_slice() -> i64 {
    return inspect([3, 4])
}

pub fn mutable_slice() -> i64 {
    let xs: array<i64> = [1, 1]
    return mutate(xs)
}

pub fn array_control() -> i64 {
    let xs: array<i64> = [1, 2, 3]
    return xs.length + xs[2]
}

pub fn returned_slice_view() -> i64 {
    return identity([6, 7]).get(1)
}

pub fn dynamic_named_receiver() -> i64 {
    return dynamic_array(4).get(0)
}

pub fn dynamic_named_local() -> i64 {
    let ys: array<i64> = dynamic_array(4)
    return ys.get(0)
}

pub fn opaque_i64_roundtrip() -> i64 {
    let ys: array<i64> = opaque_owned_handle(8)
    return ys.get(0)
}

fn early_return(xs: &[i64], take: bool) -> i64 {
    let x = 0
    if take {
        let x = xs
        return 1
    }
    return x
}

fn early_assign_return(xs: &[i64], take: bool) -> i64 {
    let x = 0
    if take {
        x = xs
        return 1
    }
    return x
}

fn local_shadow(xs: &[i64], take: bool) -> i64 {
    let x = 3
    if take {
        let x = xs
    }
    return x
}

fn guarded_continue(xs: &[i64]) -> i64 {
    let s = 0
    for i in 0..5 {
        if i == 2 { continue }
        s = s + i
    }
    return s + xs.length
}

fn guarded_break(xs: &[i64]) -> i64 {
    let i = 0
    let s = 0
    loop {
        if i >= 3 { break }
        s = s + i
        i = i + 1
    }
    return s + xs.length
}

fn ordinary_while_condition(xs: &[i64]) -> i64 {
    let i = 0
    while i < 3 { i = i + 1 }
    return i + xs.length
}

pub fn early_return_control() -> i64 {
    return early_return([5], true) * 1000
        + early_return([5], false) * 100
        + early_assign_return([5], true) * 10
        + early_assign_return([5], false)
        + local_shadow([5, 6], true)
}

pub fn continue_control() -> i64 { return guarded_continue([4, 5]) }
pub fn break_control() -> i64 { return guarded_break([4, 5]) }
pub fn while_condition_control() -> i64 { return ordinary_while_condition([4, 5]) }

"#;

fn compile(src: &str, tag: &str, dir: &Path) -> (Output, PathBuf) {
    let source = dir.join(format!("{tag}.mind"));
    let artifact = dir.join(format!("{tag}.so"));
    std::fs::write(&source, src).expect("write source");
    let out = Command::new(common::mindc_bin())
        .args([
            source.to_str().unwrap(),
            "--emit-shared",
            artifact.to_str().unwrap(),
        ])
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run mindc");
    (out, artifact)
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn array_values_literals_indexing_and_mutable_slices_run_natively() {
    let dir = common::scratch_dir("slice_call_abi_run");
    let (out_a, so_a) = compile(VALID, "valid_a", &dir);
    assert!(
        out_a.status.success(),
        "compile failed:\n{}",
        combined(&out_a)
    );

    let py = format!(
        r#"import ctypes
lib=ctypes.CDLL(r'{}')
expected={{'value_slice':312,'literal_slice':234,'mutable_slice':42,'array_control':6,'returned_slice_view':7,'dynamic_named_receiver':4,'dynamic_named_local':4,'opaque_i64_roundtrip':8,'early_return_control':1013,'continue_control':10,'break_control':5,'while_condition_control':5}}
for name,want in expected.items():
    fn=getattr(lib,name); fn.restype=ctypes.c_int64
    got=fn(); assert got==want, f'{{name}}={{got}}, expected {{want}}'
"#,
        so_a.display()
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run native artifact");
    assert!(
        ran.status.success(),
        "native execution failed:\n{}",
        combined(&ran)
    );

    let (out_b, so_b) = compile(VALID, "valid_b", &dir);
    assert!(
        out_b.status.success(),
        "repeat compile failed:\n{}",
        combined(&out_b)
    );
    assert_eq!(
        std::fs::read(so_a).unwrap(),
        std::fs::read(so_b).unwrap(),
        "identical slice source must emit deterministic native bytes"
    );
}

#[test]
fn incompatible_or_overpowered_slice_uses_fail_with_structured_diagnostics() {
    let cases = [
        (
            "set_status_cannot_replace_array",
            "pub fn run() -> i64 { let xs: array<i64> = [1]; xs = xs.set(0, 2); return xs.length }\n",
            "E2032",
        ),
        (
            "push_cannot_create_owner_from_slice",
            "fn bad(xs: &mut [i64]) -> i64 { xs = xs.push(2); return xs.length }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "scalar",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { return take(7) }\n",
            "E2032",
        ),
        (
            "scalar_slice_binding",
            "pub fn run() -> i64 { let xs: &[i64] = 7; return xs.length }\n",
            "E2032",
        ),
        (
            "wrong_element",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { let xs: array<u8> = [1]; return take(xs) }\n",
            "E2032",
        ),
        (
            "float_literal_element",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { return take([1.5]) }\n",
            "E2032",
        ),
        (
            "string_literal_element",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { return take([\"x\"]) }\n",
            "E2032",
        ),
        (
            "named_struct_element",
            "struct Point { x: i64 }\nfn take(xs: &[Point]) -> i64 { return xs.length }\npub fn run() -> i64 { return take([1]) }\n",
            "E2032",
        ),
        (
            "dynamic_float_element",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { let x: f64 = 1.5; return take([x]) }\n",
            "E2032",
        ),
        (
            "unproved_array_return",
            "fn bad() -> array<i64> { return 7 }\nfn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { return take(bad()) }\n",
            "E2032",
        ),
        (
            "readonly_slice_launder",
            "fn bad(xs: &[i64]) -> i64 { let ys: array<i64> = xs; ys.set(0, 9); return ys[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "borrow_erased_through_scalar_return",
            "fn hide(xs: &[i64]) -> i64 { return xs }\nfn bad(xs: &[i64]) -> i64 { let a: array<i64> = hide(xs); a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "if_join_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; if true { x = xs; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "if_scalar_return_borrow",
            "fn hide(xs: &[i64]) -> i64 { let x = 0; if true { x = xs; } return x }\nfn bad(xs: &[i64]) -> i64 { let a: array<i64> = hide(xs); a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "for_join_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; for i in 0..1 { x = xs; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "while_join_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; let i = 0; while i < 1 { x = xs; i = i + 1; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "region_join_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; region { x = xs; 0 }; let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "loop_carried_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; for i in 0..2 { let a: array<i64> = x; a.set(0, 9); x = xs; } return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "while_condition_borrow_from_fallthrough",
            "fn touch(raw: i64, n: i64) -> bool { return true }\nfn bad(xs: &[i64]) -> i64 { let x = 0; let n = 0; while touch(x, n) { x = xs; n = n + 1 } return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "while_condition_borrow_from_continue",
            "fn touch(raw: i64, n: i64) -> bool { return true }\nfn bad(xs: &[i64]) -> i64 { let x = 0; let n = 0; while touch(x, n) { if n == 0 { x = xs; n = n + 1; continue } n = n + 1 } return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "conditional_scalar_overwrite",
            "fn bad(xs: &[i64]) -> i64 { let x = xs; if true { x = 0; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "match_arm_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; match 1 { 1 => x = xs, _ => x = 0 }; let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "if_shadow_after_outer_assignment",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; if true { x = xs; let x = 0; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "loop_shadow_after_outer_assignment",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; for i in 0..1 { x = xs; let x = 0; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "scalar_return_after_shadow",
            "fn hide(xs: &[i64]) -> i64 { let x = 0; if true { x = xs; let x = 0; } return x }\nfn bad(xs: &[i64]) -> i64 { let a: array<i64> = hide(xs); a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "nested_shadow_after_outer_assignment",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; region { x = xs; let x = 0; region { let x = 1; 0 }; 0 }; let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "break_keeps_prior_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; for i in 0..2 { x = xs; break; x = 0; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "continue_keeps_backedge_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; for i in 0..2 { let a: array<i64> = x; a.set(0, 9); x = xs; continue; x = 0; } return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "conditional_break_keeps_prior_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; for i in 0..2 { if true { x = xs; break; } x = 0; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "conditional_continue_keeps_backedge_borrow",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; for i in 0..2 { if true { x = xs; continue; } x = 0; } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "nested_loop_break_targets_inner_loop",
            "fn bad(xs: &[i64]) -> i64 { let x = 0; for i in 0..1 { for j in 0..1 { x = xs; break; x = 0; } } let a: array<i64> = x; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "borrow_to_scalar_formal",
            "fn hide(raw: i64) -> i64 { return raw }\nfn bad(xs: &[i64]) -> i64 { return hide(xs) }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "borrow_to_array_formal",
            "fn own(xs: array<i64>) -> i64 { xs.set(0, 9); return xs[0] }\nfn bad(xs: &[i64]) -> i64 { return own(xs) }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "borrow_into_typed_scalar",
            "fn bad(xs: &[i64]) -> i64 { let raw: i64 = xs; let a: array<i64> = raw; a.set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "borrow_arithmetic",
            "fn bad(xs: &[i64]) -> i64 { return xs + 0 }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "borrow_cast",
            "fn bad(xs: &[i64]) -> i64 { return xs as i64 }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "borrow_implicit_scalar_tail",
            "fn bad(xs: &[i64]) -> i64 { xs }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "nested_borrow_array_storage",
            "fn bad(xs: &[i64]) -> i64 { let a: array<&[i64]> = [xs]; a[0].set(0, 9); return xs[0] }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "nested_borrow_tuple_storage",
            "fn bad(xs: &[i64]) -> i64 { let pair = (xs, 1); return 0 }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "nested_slice_signature",
            "fn bad(xs: array<&[i64]>) -> i64 { return 0 }\npub fn run() -> i64 { return 0 }\n",
            "E2033",
        ),
        (
            "for_rebind",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { let xs: array<i64> = [1]; for i in 0..1 { xs = 7 }; return take(xs) }\n",
            "E2032",
        ),
        (
            "branch_block_rebind",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { let xs: array<i64> = [1]; if 1 { xs = 7 }; return take(xs) }\n",
            "E2032",
        ),
        (
            "while_rebind",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { let xs: array<i64> = [1]; while 0 { xs = 7 }; return take(xs) }\n",
            "E2032",
        ),
        (
            "region_rebind",
            "fn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { let xs: array<i64> = [1]; region { xs = 7; 0 }; return take(xs) }\n",
            "E2032",
        ),
        (
            "immutable_store",
            "fn bad(xs: &[i64]) -> i64 { xs[0] = 7; return 0 }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "slice_push",
            "fn bad(xs: &mut [i64]) -> i64 { xs.push(7); return 0 }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "slice_param_reassigned_array_then_push",
            "fn bad(xs: &[i64]) -> i64 { let ys: array<i64> = [2]; xs = ys; xs.push(3); return xs.length }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "array_reassigned_borrowed_slice",
            "fn bad(xs: &[i64]) -> i64 { let ys: array<i64> = [2]; ys = xs; return ys.length }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "returned_readonly_slice_push",
            "fn id(xs: &[i64]) -> &[i64] { return xs }\npub fn run() -> i64 { return id([1]).push(2) }\n",
            "E2033",
        ),
        (
            "returned_readonly_slice_set",
            "fn id(xs: &[i64]) -> &[i64] { return xs }\npub fn run() -> i64 { id([1]).set(0, 2); return 0 }\n",
            "E2033",
        ),
        (
            "invalid_declared_slice_return",
            "fn bad(xs: &[i64]) -> &[i64] { return 7 }\npub fn run() -> i64 { return bad([1]).length }\n",
            "E2032",
        ),
        (
            "invalid_tail_slice_result",
            "fn bad(xs: &[i64]) -> &[i64] { 7 }\npub fn run() -> i64 { return bad([1]).get(0) }\n",
            "E2032",
        ),
        (
            "missing_slice_result",
            "fn bad(xs: &[i64]) -> &[i64] { let x: i64 = 7 }\npub fn run() -> i64 { return bad([1]).get(0) }\n",
            "E2032",
        ),
        (
            "conditional_only_slice_result",
            "fn bad(xs: &[i64]) -> &[i64] { if 1 { return xs } }\npub fn run() -> i64 { return bad([1]).get(0) }\n",
            "E2032",
        ),
        (
            "stored_readonly_slice_set",
            "struct Boxed { view: &[i64] }\nfn bad(xs: &[i64]) -> i64 { let b: Boxed = Boxed { view: xs }; b.view.set(0, 9); return 0 }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "stored_readonly_slice_index_assign",
            "struct Boxed { view: &[i64] }\nfn bad(xs: &[i64]) -> i64 { let b: Boxed = Boxed { view: xs }; b.view[0] = 9; return 0 }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "stored_readonly_slice_push",
            "struct Boxed { view: &[i64] }\nfn bad(xs: &[i64]) -> i64 { let b: Boxed = Boxed { view: xs }; b.view.push(9); return 0 }\npub fn run() -> i64 { return bad([1]) }\n",
            "E2033",
        ),
        (
            "malformed_stored_slice",
            "struct Boxed { view: &[i64] }\npub fn run() -> i64 { let b: Boxed = Boxed { view: 7 }; return 0 }\n",
            "E2033",
        ),
    ];
    let dir = common::scratch_dir("slice_call_abi_refusal");
    for (tag, src, code) in cases {
        let (out, artifact) = compile(src, tag, &dir);
        let text = combined(&out);
        assert!(!out.status.success(), "{tag} unexpectedly compiled");
        assert!(text.contains(code), "{tag} missed {code}:\n{text}");
        assert!(!text.contains("panicked at"), "{tag} panicked:\n{text}");
        assert!(!artifact.exists(), "{tag} emitted an artifact on refusal");
    }
}

#[test]
fn ordinary_break_and_continue_remain_valid_with_a_slice_in_scope() {
    let src = r#"
fn control(xs: &[i64]) -> i64 {
    let x = 3
    for i in 0..2 { if i == 0 { continue }; break }
    return x + xs.length
}
"#;
    let module = libmind::parser::parse(src).expect("parse break/continue control");
    let diagnostics = libmind::type_checker::check_module_types(&module, src, &Default::default());
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|diag| diag.severity == libmind::diagnostics::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "unexpected diagnostics: {errors:?}");
}

#[test]
fn loop_flow_joins_sequential_branches_without_path_enumeration() {
    let mut src = String::new();
    for width in [8, 16, 32] {
        src.push_str(&format!(
            "fn stress_{width}(xs: &[i64]) -> i64 {{\n  let a = 0\n  let b = 0\n  let c = 0\n  for i in 0..2 {{\n"
        ));
        for branch in 0..width {
            let name = ["a", "b", "c"][branch % 3];
            src.push_str(&format!("    if i == {} {{ {name} = xs }}\n", branch % 2));
        }
        src.push_str(
            "    for j in 0..2 {\n      if j == 0 { b = xs; continue }\n      if j == 1 { c = xs; break }\n    }\n  }\n  let owned: array<i64> = a\n  owned.set(0, 9)\n  return xs[0]\n}\n",
        );
    }
    src.push_str("pub fn run() -> i64 { return stress_32([1]) }\n");

    let dir = common::scratch_dir("slice_call_abi_flow_stress");
    let source = dir.join("stress.mind");
    let artifact = dir.join("stress.so");
    std::fs::write(&source, src).expect("write stress source");
    let out = Command::new("timeout")
        .args([
            "20",
            common::mindc_bin().to_str().unwrap(),
            source.to_str().unwrap(),
            "--emit-shared",
            artifact.to_str().unwrap(),
        ])
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run bounded stress compile");
    let text = combined(&out);
    assert_ne!(out.status.code(), Some(124), "flow analysis timed out");
    assert!(!out.status.success(), "unsafe stress case compiled");
    assert!(
        text.contains("E2033"),
        "missing capability diagnostic:\n{text}"
    );
    assert!(
        !text.contains("panicked at"),
        "stress case panicked:\n{text}"
    );
    assert!(!artifact.exists(), "stress refusal emitted an artifact");
}

#[test]
fn malformed_source_still_uses_the_parser_diagnostic_path() {
    let dir = common::scratch_dir("slice_call_abi_malformed");
    let (out, artifact) = compile(
        "fn take(xs: &[i64]) -> i64 { return xs[0] \n",
        "malformed",
        &dir,
    );
    let text = combined(&out);
    assert!(!out.status.success());
    assert!(text.contains("error"), "missing parser diagnostic:\n{text}");
    assert!(
        !text.contains("panicked at"),
        "parser failure panicked:\n{text}"
    );
    assert!(!artifact.exists());
}

/// A qualified call to an imported `encode` must use that signature.
/// The bare name is also an implicit std function, so a bare lookup is ambiguous.
#[cfg(feature = "cross-module-imports")]
#[test]
fn qualified_imported_slice_call_checks_and_runs() {
    let root = common::scratch_dir("slice_call_abi_qualified").join("flat");
    write_qualified_project(
        &root,
        "pub fn encode(x: &[i64], n: u32) -> i64 {\n    return x[0];\n}\n",
        "import encoder;\nfn fwd(input: &[i64], n: u32) -> i64 {\n    return encoder.encode(input, n);\n}\npub fn run() -> i64 {\n    let a: array<i64> = [4, 5];\n    return fwd(a, 2);\n}\n",
    );
    assert_project_check(&root);
    let artifact = root.join("run.so");
    let built = mindc_in(
        &root,
        &[
            "build",
            "--backend=mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            artifact.to_str().unwrap(),
        ],
    );
    assert!(
        built.status.success(),
        "qualified call did not build:\n{}",
        combined(&built)
    );
    assert_qualified_native(&artifact, "run", 4);

    let blocked = common::scratch_dir("slice_call_abi_qualified").join("module");
    write_qualified_project(
        &blocked,
        "module encoder {\n    fn encode(x: &[i64], n: u32) -> i64 { return x[0] }\n    export fn encode\n}\n",
        "module user {\n    use encoder\n    fn fwd(input: &[i64], n: u32) -> i64 { return encoder.encode(input, n) }\n    pub fn run() -> i64 {\n        let a: array<i64> = [4, 5]\n        return fwd(a, 2)\n    }\n    export fn run\n}\n",
    );
    assert_project_check(&blocked);
}

/// An unresolved qualifier, a non-slice parameter, and an unexported fn stay refused.
#[cfg(feature = "cross-module-imports")]
#[test]
fn qualified_slice_calls_stay_refused() {
    let cases = [
        (
            "missing",
            "pub fn encode(x: &[i64], n: u32) -> i64 { return x[0] }\n",
            "import encoder;\npub fn run(input: &[i64], n: u32) -> i64 {\n    return encoder.missing(input, n)\n}\n",
            "unresolved function `missing`",
        ),
        (
            "scalar",
            "pub fn take(x: i64, n: u32) -> i64 { return x + (n as i64) }\n",
            "import encoder;\npub fn run(input: &[i64], n: u32) -> i64 {\n    return encoder.take(input, n)\n}\n",
            "is not a slice",
        ),
        (
            "unexported",
            "module encoder {\n    fn hide(x: &[i64], n: u32) -> i64 { return x[0] }\n    fn keep() -> i64 { return 1 }\n    export fn keep\n}\n",
            "import encoder;\nfn hide(x: &[i64], n: u32) -> i64 { return x[0] }\npub fn run(input: &[i64], n: u32) -> i64 {\n    return encoder.hide(input, n)\n}\n",
            "unresolved function `hide`",
        ),
    ];
    for (tag, encoder, main_src, needle) in cases {
        let root = common::scratch_dir("slice_call_abi_qualified").join(tag);
        write_qualified_project(&root, encoder, main_src);
        let checked = mindc_in(&root, &["check", "--no-fmt", "--no-lint"]);
        let text = combined(&checked);
        assert!(!checked.status.success(), "{tag} was admitted:\n{text}");
        assert!(text.contains("E2033"), "{tag} missed E2033:\n{text}");
        assert!(text.contains(needle), "{tag} missed `{needle}`:\n{text}");
    }
}

#[cfg(feature = "cross-module-imports")]
fn write_qualified_project(root: &Path, encoder: &str, main_src: &str) {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root.join("src")).expect("src");
    std::fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"qslice\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\noutput = \"qslice\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\n[exports]\nc_abi = [\"run\"]\n",
    )
    .expect("manifest");
    std::fs::write(root.join("src/encoder.mind"), encoder).expect("encoder");
    std::fs::write(root.join("src/main.mind"), main_src).expect("main");
}

#[cfg(feature = "cross-module-imports")]
fn assert_project_check(root: &Path) {
    let checked = mindc_in(root, &["check", "--no-fmt", "--no-lint"]);
    assert!(
        checked.status.success(),
        "qualified slice call refused:\n{}",
        combined(&checked)
    );
}

#[cfg(feature = "cross-module-imports")]
fn assert_qualified_native(artifact: &Path, name: &str, want: i64) {
    let py = format!(
        "import ctypes\nlib = ctypes.CDLL(r'{}')\nfn = getattr(lib, '{name}')\nfn.restype = ctypes.c_int64\ngot = fn()\nassert got == {want}, f'{{got}}'\nprint('native-ok')\n",
        artifact.display()
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run native artifact");
    assert!(
        ran.status.success() && combined(&ran).contains("native-ok"),
        "native run failed:\n{}",
        combined(&ran)
    );
}

#[cfg(feature = "cross-module-imports")]
fn mindc_in(root: &Path, args: &[&str]) -> Output {
    Command::new(common::mindc_bin())
        .args(args)
        .current_dir(root)
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run mindc")
}
