// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! F3 (codex corr-audit #3) — ambiguous BARE enum-payload patterns silently
//! dispatch on the FIRST enum's tag order.
//!
//! When two enums declare the same variant NAMES in DIFFERENT orders, a bare
//! (unqualified) match arm resolved to the lexicographically-first `Enum::V` in
//! the tag registry instead of the SCRUTINEE's enum — a silent wrong-tag
//! dispatch (the type checker DEFERS bare patterns, so the program reaches
//! lowering):
//!
//!   enum A { Y(i64), X(i64) }   // A::Y=0, A::X=1
//!   enum B { X(i64), Y(i64) }   // B::X=0, B::Y=1
//!   let b: B = B::X(7);         // tag 0
//!   match b { X(v) => v + 100, Y(v) => v + 200 }
//!
//! Scrutinee is `B`, so `X(v)` must be B::X (tag 0) -> 107. The bug tested the
//! tag against A's order (A::X=1) and dispatched the `Y` arm -> 207.
//!
//! Fix (correct-scrutinee-dispatch, NOT fail-closed rejection): the scrutinee's
//! enum is recovered at the lowering site from its typed `let b: B` (or a
//! qualified-constructor RHS) via a var->enum side-table, and bare arms anchor
//! to THAT enum. `resolve_bare` binds against the scrutinee's enum, never the
//! first registry key.
//!
//! Adversarial controls prove the fix is not over-applied: an UNAMBIGUOUS bare
//! pattern (variant in exactly one enum) still resolves; fully-qualified
//! `B::X`/`B::Y` are byte-untouched (107/207); an ordinary single-enum
//! one/two-payload match returns 42.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test f3_bare_enum_collision_run`

#![cfg(all(unix, feature = "std-surface", feature = "mlir-build"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// All enums are declared FIRST: enum tags register in declaration order, so a
// match that references an enum declared LATER would fail (unknown-variant),
// unrelated to this fix.
enum A { Y(i64), X(i64) }
enum B { X(i64), Y(i64) }
enum C { Zonk(i64) }
enum Opt { Some(i64), None }

// RED repro: bare `X(v)` over a `B` scrutinee must test B::X (tag 0) -> 107.
// The bug dispatched via A's tag order (A::X=1) -> the `Y` arm -> 207.
pub fn kat_bare_x() -> i64 {
    let b: B = B::X(7);
    return match b { X(v) => v + 100, Y(v) => v + 200, };
}

// The twin: bare `Y(v)` over a `B::Y(7)` scrutinee must be B::Y (tag 1) -> 207.
pub fn kat_bare_y() -> i64 {
    let b: B = B::Y(7);
    return match b { X(v) => v + 100, Y(v) => v + 200, };
}

// CONTROL: an UNAMBIGUOUS bare pattern (Zonk lives in exactly one enum) still
// resolves correctly (9 + 300 -> 309).
pub fn kat_unambiguous_bare() -> i64 {
    let c: C = C::Zonk(9);
    return match c { Zonk(v) => v + 300, };
}

// CONTROL: fully-qualified `B::X`/`B::Y` are byte-untouched and still 107/207.
pub fn kat_qualified_x() -> i64 {
    let b: B = B::X(7);
    return match b { B::X(v) => v + 100, B::Y(v) => v + 200, };
}
pub fn kat_qualified_y() -> i64 {
    let b: B = B::Y(7);
    return match b { B::X(v) => v + 100, B::Y(v) => v + 200, };
}

// CONTROL: an ordinary single-enum one/two-payload match returns 42.
pub fn kat_single_enum() -> i64 {
    let o: Opt = Opt::Some(42);
    return match o { Some(v) => v, None => -1, };
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn f3_bare_enum_collision_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("f3-bare-enum-collision-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_f3_bare_enum_collision_run.mind");
    let so = dir.join("mind_f3_bare_enum_collision_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("f3-bare-enum-collision-run: needs mlir-build; skipping");
            return;
        }
        panic!("f3-bare-enum-collision-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         names = ('kat_bare_x','kat_bare_y','kat_unambiguous_bare',\n\
         \x20        'kat_qualified_x','kat_qualified_y','kat_single_enum')\n\
         for _n in names: getattr(lib, _n).restype = ctypes.c_int64\n\
         r = lib.kat_bare_x();          assert r == 107, 'kat_bare_x=' + str(r)\n\
         r = lib.kat_bare_y();          assert r == 207, 'kat_bare_y=' + str(r)\n\
         r = lib.kat_unambiguous_bare();assert r == 309, 'kat_unambiguous_bare=' + str(r)\n\
         r = lib.kat_qualified_x();     assert r == 107, 'kat_qualified_x=' + str(r)\n\
         r = lib.kat_qualified_y();     assert r == 207, 'kat_qualified_y=' + str(r)\n\
         r = lib.kat_single_enum();     assert r == 42,  'kat_single_enum=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "f3-bare-enum-collision-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
