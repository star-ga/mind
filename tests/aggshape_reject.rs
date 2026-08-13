// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! corr2 BUG1/BUG2 regression — the aggregate-shape ident checks (E2620/E2621).
//!
//! `ValueType` is scalar-or-tensor only: it collapses a tuple to its last
//! element and an array literal to its first, so once an aggregate flows through
//! a NAME its arity/length is invisible and a wrong-arity destructure or a const
//! out-of-bounds index silently reads adjacent stack / heap garbage. A scoped
//! `aggs` side-table on `ClassCtx` recovers the PROVABLE shape from a literal
//! `let`/`assign` RHS and rejects:
//!   - E2620: `let (a,b,c) = t` where `t` is a provable 2-tuple (BUG2);
//!   - E2621: `a[k]` where `a` is a provable literal array and `k` is a const
//!     index out of bounds (BUG1).
//!
//! Load-bearing zero-over-coverage: the shape is dropped on ANY rebind (assign /
//! let-shadow / let-tuple, nested through branches, loops, blocks, match arms,
//! and rvalue positions) and the checks fire only on a call-free literal-seeded
//! fact — so a conditionally-rebound / loop-rebound / call-returned aggregate
//! must NOT fire. The GREEN controls below pin that no-over-fire boundary; an
//! element write (`a[i] = v`) does NOT change length, so the shape survives it.
//!
//! Both diagnostics are type-checker diagnostics that surface on the compile
//! path (`--emit-shared`), so this gate needs `mlir-build` + `std-surface`; a
//! `cargo test --no-default-features` run compiles it out entirely.
#![cfg(all(unix, feature = "std-surface", feature = "mlir-build"))]

use std::io::Write;
use std::process::Command;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

/// Combined stdout+stderr of a full `mindc <path> --emit-shared <tmp.so>` compile.
fn build_out(name: &str, src: &str) -> String {
    let dir = std::env::temp_dir();
    let mp = dir.join(format!("{name}.mind"));
    let sp = dir.join(format!("{name}.so"));
    let mut f = std::fs::File::create(&mp).expect("write src");
    f.write_all(src.as_bytes()).expect("write src bytes");
    let out = mindc()
        .args([mp.to_str().unwrap(), "--emit-shared", sp.to_str().unwrap()])
        .output()
        .expect("spawn mindc");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

// ---------------------------------------------------------------------------
// RED — the two ident miscompiles must now be rejected fail-closed.
// ---------------------------------------------------------------------------

#[test]
fn bug2_tuple_arity_rejected() {
    // (e) `t` is a provable 2-tuple; destructuring into 3 names reads an
    // adjacent stack slot as `c`. Must reject with E2620, not build a wrong `.so`.
    let out = build_out(
        "aggshape_bug2_tuple_arity",
        r#"pub fn f() -> i64 {
    let t = (11, 22)
    let (a, b, c) = t
    return c
}
"#,
    );
    assert!(
        out.contains("E2620"),
        "tuple-arity destructure NOT rejected (silent slot-garbage read): {out}"
    );
}

#[test]
fn bug1_array_const_oob_rejected() {
    // (f) `a` is a provable length-2 array literal; the const index 2 is out of
    // bounds and reads past the buffer. Must reject with E2621.
    let out = build_out(
        "aggshape_bug1_array_oob",
        r#"pub fn f() -> i64 {
    let a: array<i64> = [11, 22]
    return a[2]
}
"#,
    );
    assert!(
        out.contains("E2621"),
        "const out-of-bounds index NOT rejected (silent past-buffer read): {out}"
    );
}

#[test]
fn element_write_keeps_shape_oob_rejected() {
    // (b, true-positive half) an element write `a[0] = 9` changes a value, NOT
    // the length — so the provable length-3 shape survives it and the const
    // index 3 must still be rejected (E2621). This proves element writes do NOT
    // drop the shape.
    let out = build_out(
        "aggshape_elemwrite_oob",
        r#"pub fn f() -> i64 {
    let mut a: array<i64> = [1, 2, 3]
    a[0] = 9
    return a[3]
}
"#,
    );
    assert!(
        out.contains("E2621"),
        "element write wrongly dropped the array shape (a[3] not rejected): {out}"
    );
}

// ---------------------------------------------------------------------------
// GREEN — the no-over-fire boundary. Each program is runtime-representable and
// MUST NOT emit the aggregate-shape diagnostic (rejecting a build-accepted
// program is a zero-over-coverage violation, strictly worse than shipping).
// ---------------------------------------------------------------------------

#[test]
fn conditional_rebind_not_fired() {
    // (a) `a` MIGHT be reassigned on the `if` path, so its literal shape is no
    // longer provable after the branch — the kill-set drops it and a[5] must NOT
    // fire E2621 (with the naive clone-mirror the outer table kept Array{3} and
    // falsely rejected this).
    let out = build_out(
        "aggshape_cond_rebind",
        r#"fn other() -> array<i64> {
    return [9, 9, 9, 9, 9, 9]
}

pub fn f(c: bool) -> i64 {
    let mut a: array<i64> = [1, 2, 3]
    if c {
        a = other()
    }
    return a[5]
}
"#,
    );
    assert!(
        !out.contains("E2621"),
        "conditional-rebind falsely fired E2621 (over-fire): {out}"
    );
}

#[test]
fn loop_rebind_not_fired() {
    // (d) a name rebound anywhere in a loop body loses its literal shape for the
    // whole loop AND after it: a[5] must NOT fire even though the pre-loop
    // literal was length 3.
    let out = build_out(
        "aggshape_loop_rebind",
        r#"fn other() -> array<i64> {
    return [9, 9, 9, 9, 9, 9]
}

pub fn f(c: bool) -> i64 {
    let mut a: array<i64> = [1, 2, 3]
    while c {
        a = other()
    }
    return a[5]
}
"#,
    );
    assert!(
        !out.contains("E2621"),
        "loop-rebind falsely fired E2621 (over-fire): {out}"
    );
}

#[test]
fn element_write_inbounds_not_fired() {
    // (b, green half) the element write keeps the length-3 shape; the const
    // index 2 is in bounds and must NOT fire.
    let out = build_out(
        "aggshape_elemwrite_inbounds",
        r#"pub fn f() -> i64 {
    let mut a: array<i64> = [1, 2, 3]
    a[0] = 9
    return a[2]
}
"#,
    );
    assert!(
        !out.contains("E2621"),
        "in-bounds index after element write falsely fired E2621: {out}"
    );
}

#[test]
fn call_returned_tuple_not_fired() {
    // (c) the RHS is a call result, not a literal — the shape is NOT provable, so
    // even a mismatched-looking destructure must NOT fire E2620.
    let out = build_out(
        "aggshape_call_tuple",
        r#"fn mkpair() -> (i64, i64) {
    return (1, 2)
}

pub fn f() -> i64 {
    let (a, b, c) = mkpair()
    return a + b + c
}
"#,
    );
    assert!(
        !out.contains("E2620"),
        "call-returned tuple falsely fired E2620 (shape not provable): {out}"
    );
}

#[test]
fn correct_arity_destructure_not_fired() {
    // A matching-arity destructure of a provable 2-tuple must NOT fire E2620.
    let out = build_out(
        "aggshape_correct_arity",
        r#"pub fn f() -> i64 {
    let t = (11, 22)
    let (a, b) = t
    return a + b
}
"#,
    );
    assert!(
        !out.contains("E2620"),
        "correct-arity destructure falsely fired E2620: {out}"
    );
}
