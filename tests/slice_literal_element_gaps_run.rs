// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Review gaps in the slice-argument literal-element admission rules.
//!
//! Four behaviors the review of the alias-element admission work asked for:
//!
//! * a literal element whose proven integer width is wider than the slice's
//!   element type is refused with `E2032` instead of truncating silently in
//!   the native build (`300` stored through a `&[u8]` reads back `44`);
//! * a parameter, result or binding typed through an alias of a collection
//!   type (`type V = array<i32>`) is refused with `E2032` at the check stage,
//!   as docs/type-system.md documents, instead of passing the check and dying
//!   in lowering;
//! * the int/float agreement guard in the `element_class` Binary arm: a mixed
//!   arithmetic element is refused, so one side's class can never admit the
//!   other's data;
//! * a cyclic constant used as literal elements hits the constant nesting
//!   bound and is refused, not recursed to a stack overflow.
//!
//! Each refusal test fails without its fix. Every refusal is asserted at
//! `mindc check`, the stage that admits a slice argument, so no MLIR build is
//! needed and the file runs under the default feature set.

#![cfg(all(unix, feature = "std-surface"))]

mod common;

use std::path::Path;
use std::process::{Command, Output};

fn mindc(args: &[&str], cwd: &Path) -> Output {
    Command::new(common::mindc_bin())
        .args(args)
        .current_dir(cwd)
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run mindc")
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn write_source(tag: &str, src: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = common::scratch_dir("slice_literal_element_gaps_run").join(tag);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join(format!("{tag}.mind"));
    std::fs::write(&path, src).expect("write source");
    (dir, path)
}

/// `mindc check` must refuse `src` with `E2032` and must not panic.
fn assert_check_refused(tag: &str, src: &str) {
    let (dir, path) = write_source(tag, src);
    let out = mindc(
        &["check", "--no-fmt", "--no-lint", path.to_str().unwrap()],
        &dir,
    );
    let text = combined(&out);
    assert!(
        !out.status.success(),
        "{tag} passed `mindc check` and would truncate or fail in lowering:\n{text}"
    );
    assert!(
        text.contains("E2032"),
        "{tag} was not refused with the slice-argument code:\n{text}"
    );
    assert!(
        !text.contains("panicked at"),
        "{tag} panicked instead of refusing:\n{text}"
    );
}

/// `mindc check` must admit `src`: the element rules refuse only a proven
/// mismatch, never a compatible element.
fn assert_check_admitted(tag: &str, src: &str) {
    let (dir, path) = write_source(tag, src);
    let out = mindc(
        &["check", "--no-fmt", "--no-lint", path.to_str().unwrap()],
        &dir,
    );
    assert!(
        out.status.success(),
        "{tag} was refused:\n{}",
        combined(&out)
    );
}

/// A literal element that names a constant, reads a field or an element, or
/// combines either arithmetically is admitted by scalar class alone. Each of
/// these compiled and silently truncated natively before the width rule: the
/// element's proven integer width must fit the slice's element type.
#[test]
fn wide_integer_literal_elements_are_refused_for_narrow_slices() {
    let cases = [
        (
            "wide_const_element",
            "const Z: i64 = 300\nfn take(xs: &[u8]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([Z, Z]) }\n",
        ),
        (
            "wide_field_read_element",
            "struct P { big: i64 }\nfn take(xs: &[u8]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { let p: P = P { big: 300 }; return take([p.big]) }\n",
        ),
        (
            "wide_element_read_element",
            "fn take(xs: &[u8]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { let a: array<i64> = [300]; return take([a[0]]) }\n",
        ),
        (
            "wide_binary_element",
            "const Z: i64 = 4\nfn take(xs: &[i32]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([Z + 1]) }\n",
        ),
    ];
    for (tag, src) in cases {
        assert_check_refused(tag, src);
    }
}

/// The width rule refuses only a provably wider element: a constant at the
/// slice's own width, a narrower constant, and an explicit `as` cast keep
/// passing the check.
#[test]
fn fitting_integer_literal_elements_stay_admitted() {
    let cases = [
        (
            "same_width_const_element",
            "const Z: i32 = 300\nfn take(xs: &[i32]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([Z, Z]) }\n",
        ),
        (
            "narrower_const_element",
            "const Z: i32 = 300\nfn take(xs: &[i64]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([Z, Z]) }\n",
        ),
        (
            "explicit_cast_element",
            "const Z: i64 = 300\nfn take(xs: &[i32]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([Z as i32]) }\n",
        ),
        (
            "plain_literal_elements",
            "fn take(xs: &[u8]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([7, 255]) }\n",
        ),
    ];
    for (tag, src) in cases {
        assert_check_admitted(tag, src);
    }
}

/// docs/type-system.md: a binding, parameter or result typed through an alias
/// of a slice or of an `array<T>` is refused with `E2032`. The parameter and
/// result spellings passed `mindc check` and died in lowering; the binding
/// spelling passed when the value never reached a slice call.
#[test]
fn collection_alias_signatures_and_bindings_are_refused_as_documented() {
    let cases = [
        (
            "array_alias_parameter",
            "type V = array<i32>\nfn g(xs: V) -> i64 { return xs.length }\npub fn run() -> i64 { let a: array<i32> = [5, 6]; return g(a) }\n",
        ),
        (
            "array_alias_result",
            "type V = array<i32>\nfn make() -> V { let a: array<i32> = [5, 6]; return a }\nfn take(xs: &[i32]) -> i64 { return xs.length }\npub fn run() -> i64 { return take(make()) }\n",
        ),
        (
            "array_alias_binding",
            "type V = array<i32>\npub fn run() -> i64 { let v: V = [3, 4]; return v.length }\n",
        ),
    ];
    for (tag, src) in cases {
        assert_check_refused(tag, src);
    }
}

/// The `element_class` Binary arm admits an arithmetic element only when both
/// sides agree on one scalar class. A mixed int/float element is refused with
/// `E2032` instead of taking one side's class and admitting the other's data.
#[test]
fn mixed_class_binary_literal_elements_are_refused() {
    let cases = [
        (
            "mixed_literal_binary",
            "fn take(xs: &[i64]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([1 + 2.5]) }\n",
        ),
        (
            "mixed_const_binary",
            "const A: i64 = 1\nconst B: f64 = 2.5\nfn take(xs: &[i64]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([A + B]) }\n",
        ),
    ];
    for (tag, src) in cases {
        assert_check_refused(tag, src);
    }
}

/// A cyclic constant (`const A = B`, `const B = A`) used as literal elements
/// must hit the constant nesting bound in `const_class_at` and be refused
/// with `E2032`, not recurse to a stack overflow.
#[test]
fn cyclic_constant_elements_are_refused_at_the_nesting_bound() {
    assert_check_refused(
        "cyclic_const_element",
        "const A = B\nconst B = A\nfn take(xs: &[i64]) -> i64 { return xs[0] as i64 }\npub fn run() -> i64 { return take([A, B]) }\n",
    );
}
