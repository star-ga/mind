// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Bare enum-variant constructor AMBIGUITY fail-closed gate (Fable finding #8).
//!
//! Two value-position ctor-resolution sites — the `Call`-arm payload ctor and
//! the fieldless `Lit(Ident)` value arm — used to resolve a BARE variant name
//! (`Some(5)`, `V(1)`) to the lexicographically-first `Enum::V` in the
//! `enum_variant_tags` `BTreeMap`. That silently picked the WRONG enum whenever
//! a variant name collided across the Option/Result prelude and a user or
//! sibling-module enum (e.g. a user `Wrapper::Some` losing to `Option::Some`,
//! or a user `Basket::Err` *beating* `Result::Err` and breaking `?`). The fix
//! (`resolve_bare_variant`) fails CLOSED on a cross-enum collision instead of
//! guessing, and QUALIFIES the `?` desugar's synthesized ctor/pattern names so
//! a module that declares a same-named variant does not detonate that panic on
//! every `?`.
//!
//! This gate proves BOTH halves at runtime:
//!   * a genuinely-ambiguous bare ctor fails compilation, naming both owners;
//!   * qualified forms (`Wrapper::Some`, `Option::Some`) still compile + run;
//!   * `f()?` in a module declaring `enum Basket { Err }` compiles + runs.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test bare_variant_ambiguity_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

/// Compile `src` to a `.so` at `stem`; return `Ok(so_path)` on success or
/// `Err(stderr)` on a compile failure. `None` means the toolchain can't build
/// shared objects in this config (skip).
fn compile(stem: &str, src: &str) -> Option<Result<std::path::PathBuf, String>> {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("{stem}: mindc not found; skipping");
        return None;
    }
    let dir = std::env::temp_dir();
    let srcp = dir.join(format!("mind_{stem}.mind"));
    let so = dir.join(format!("mind_{stem}.so"));
    std::fs::write(&srcp, src).expect("write src");
    let out = Command::new(&mindc)
        .args([
            srcp.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if out.status.success() {
        Some(Ok(so))
    } else {
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("{stem}: mindc --emit-shared needs mlir-build; skipping");
            return None;
        }
        Some(Err(stderr))
    }
}

/// dlopen `so` and assert each `(fn, expected_i64)` via ctypes.
fn assert_runs(stem: &str, so: &std::path::Path, cases: &[(&str, i64)]) {
    let decls: String = cases
        .iter()
        .map(|(n, _)| format!("getattr(lib,'{n}').restype = ctypes.c_int64\n"))
        .collect();
    let checks: String = cases
        .iter()
        .map(|(n, v)| format!("r = lib.{n}(); assert r == {v}, '{n}=' + str(r)\n"))
        .collect();
    let py = format!(
        "import ctypes\nlib = ctypes.CDLL(r'{}')\n{decls}{checks}print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "{stem} value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

// ── (1) Ambiguous bare ctor fails closed, naming BOTH candidates ────────────

const AMBIG_PRELUDE: &str = r#"
enum Wrapper { Some(i64), Other }

pub fn f() -> i64 {
    let w = Some(5)
    match w {
        Wrapper::Some(v) => v,
        Wrapper::Other => 0,
    }
}
"#;

#[test]
fn ambiguous_bare_variant_vs_prelude_fails_closed() {
    let Some(res) = compile("bva_prelude_collision", AMBIG_PRELUDE) else {
        return;
    };
    let stderr = res.expect_err(
        "bare `Some(5)` colliding with prelude `Option::Some` MUST fail closed, \
         not silently resolve to the first BTreeMap match",
    );
    assert!(
        stderr.contains("ambiguous bare enum variant") && stderr.contains("`Some`"),
        "expected ambiguity diagnostic, got:\n{stderr}"
    );
    // Both owners named so the author can qualify.
    assert!(
        stderr.contains("Option::Some") && stderr.contains("Wrapper::Some"),
        "diagnostic must name BOTH candidate owners, got:\n{stderr}"
    );
}

// ── (2) Two USER enums sharing a variant name → same fail-closed error ───────

const AMBIG_TWO_USER: &str = r#"
enum A { V(i64), Wa }
enum B { V(i64), Wb }

pub fn f() -> i64 {
    let x = V(1)
    match x {
        A::V(v) => v,
        A::Wa => 0,
    }
}
"#;

#[test]
fn ambiguous_bare_variant_across_two_user_enums_fails_closed() {
    let Some(res) = compile("bva_two_user_collision", AMBIG_TWO_USER) else {
        return;
    };
    let stderr = res.expect_err("bare `V(1)` owned by both `A` and `B` MUST fail closed");
    assert!(
        stderr.contains("ambiguous bare enum variant") && stderr.contains("`V`"),
        "expected ambiguity diagnostic, got:\n{stderr}"
    );
    assert!(
        stderr.contains("A::V") && stderr.contains("B::V"),
        "diagnostic must name BOTH candidate owners, got:\n{stderr}"
    );
}

// ── (3) Qualified forms still compile + run (byte-neutral resolution) ────────

const QUALIFIED_OK: &str = r#"
enum Wrapper { Some(i64), Other }

pub fn via_wrapper() -> i64 {
    let w = Wrapper::Some(5)
    match w {
        Wrapper::Some(v) => v,
        Wrapper::Other => 0,
    }
}

pub fn via_option() -> i64 {
    let o = Option::Some(5)
    match o {
        Option::Some(v) => v,
        Option::None => 0,
    }
}
"#;

#[test]
fn qualified_bare_variant_ctors_compile_and_run() {
    let Some(res) = compile("bva_qualified_ok", QUALIFIED_OK) else {
        return;
    };
    let so = res.expect("qualified `Wrapper::Some` / `Option::Some` must compile");
    assert_runs(
        "bva_qualified_ok",
        &so,
        &[("via_wrapper", 5), ("via_option", 5)],
    );
}

// ── (4) `?` desugar qualified: a module declaring `enum Basket { Err }` ──────
//        compiles + runs `f()?` (proves the desugar's ctor/pattern names are
//        qualified so `Basket::Err` cannot poison the synthesized `Result::Err`).

const TRY_WITH_COLLIDING_ENUM: &str = r#"
enum Basket { Err }

fn f(b: i64) -> Result<i64, i64> {
    if b == 0 {
        return Result::Err(7)
    }
    Result::Ok(b)
}

fn g(b: i64) -> Result<i64, i64> {
    let x = f(b)?
    Result::Ok(x + 1)
}

// Reference Basket so its `Basket::Err` variant is registered and could poison
// the `?` desugar if it synthesized BARE `Err` — it must not.
pub fn touch_basket() -> i64 { Basket::Err }

pub fn run_ok() -> i64 {
    match g(41) {
        Result::Ok(v) => v,
        Result::Err(e) => 0 - 1,
    }
}

pub fn run_err() -> i64 {
    match g(0) {
        Result::Ok(v) => 0 - 1,
        Result::Err(e) => e,
    }
}
"#;

#[test]
fn try_operator_in_module_with_colliding_variant_compiles_and_runs() {
    let Some(res) = compile("bva_try_basket", TRY_WITH_COLLIDING_ENUM) else {
        return;
    };
    let so = res.expect(
        "`f()?` in a module declaring `enum Basket { Err }` MUST compile — the \
         `?` desugar qualifies its synthesized `Result::Err`/`Result::Ok`",
    );
    assert_runs(
        "bva_try_basket",
        &so,
        &[("touch_basket", 0), ("run_ok", 42), ("run_err", 7)],
    );
}
