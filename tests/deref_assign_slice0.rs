// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Deref-assign field-first subset (D3/D4 carve-out). `*p` / `*p = v` / `&mut r.f`
//! PARSE, FORMAT (round-trip), and are carried in the AST. Under std-surface the
//! ADMITTED subset now has executable meaning: `&mut r.f` (a depth-one,
//! struct-typed field of a resolvable-owner receiver) lowers to the field's cell
//! address, and a callee `*p` / `*p = v` on a `&mut <struct>` parameter with an
//! exact-owner value loads/stores through it — identity-preserving place
//! replacement (mind-spec v1.0/types.md:65-99), NOT a field copy.
//!
//! A `&mut` parameter is a CELL only when its function dereferences it or forwards it
//! to a cell formal (`eval::deref_cells`); every other `&mut` parameter, and every
//! `&r.f` / `&mut r.f` / `&a[i]` outside a cell argument, keeps main's POINTER meaning
//! and is admitted exactly as on main 5724a6ee. The deref-assign rules bind cells:
//! `E2028` (unsupported deref), `E2029` (unsupported deref-assign — scalar referent,
//! cross-owner value, `let q = p` / `return p` escape of a cell, `pub` cell function),
//! `E2037` (unsupported cell argument — immutable, scalar, depth-two, region-interior,
//! element, bound or cast reference, unknown owner). Scalar `&mut i64` deref stays
//! refused; it is not in the struct field-first subset.
//!
//! Generic types only — no private consumer (MindLLM/PageTable) source here.

#[cfg(feature = "mlir-build")]
mod common;
#[cfg(feature = "mlir-build")]
use common::mindc_bin;

use libmind::fmt::format_source;
use libmind::parser;
use libmind::project::MindcraftFormatConfig;
use libmind::type_checker::{self, TypeEnv};

#[cfg(feature = "mlir-build")]
use std::process::Command;

/// The exact diagnostic codes emitted for a checked module (empty when clean).
fn codes(src: &str) -> Vec<String> {
    let module = parser::parse(src).expect("Slice 0 source must PARSE");
    let env = TypeEnv::default();
    type_checker::check_module_types(&module, src, &env)
        .iter()
        .map(|d| d.code.to_string())
        .collect()
}

fn fmt(src: &str) -> String {
    format_source(src, &MindcraftFormatConfig::default()).expect("Slice 0 source must FORMAT")
}

#[test]
fn deref_read_round_trips_and_is_refused_with_E2028() {
    let src = "fn f(p: &mut i64) -> i64 {\n    return *p\n}\n";
    let once = fmt(src);
    assert!(
        once.contains("*p"),
        "deref must round-trip through fmt: {once}"
    );
    assert_eq!(once, fmt(&once), "fmt must be idempotent on deref");
    let cs = codes(src);
    assert!(
        cs.contains(&"E2028".to_string()),
        "expected E2028; got {cs:?}"
    );
}

#[test]
fn deref_assign_round_trips_and_is_refused_with_E2029() {
    let src = "fn f(p: &mut i64, v: i64) {\n    *p = v\n}\n";
    let once = fmt(src);
    assert!(
        once.contains("*p = v"),
        "deref-assign must round-trip: {once}"
    );
    assert_eq!(once, fmt(&once), "fmt must be idempotent on deref-assign");
    let cs = codes(src);
    assert!(
        cs.contains(&"E2029".to_string()),
        "expected E2029; got {cs:?}"
    );
}

#[test]
fn scalar_deref_assign_stays_refused_E2029() {
    // A scalar `&mut i64` is NOT in the field-first struct subset: `*p = v`
    // still refuses with E2029 (referent is not a declared struct).
    let src = "fn f(p: &mut i64, v: i64) {\n    *p = v\n}\n";
    assert!(
        codes(src).contains(&"E2029".to_string()),
        "scalar `*p = v` must stay E2029; got {:?}",
        codes(src)
    );
}

#[test]
fn cross_owner_deref_assign_is_refused_E2029() {
    // Owner-exactness: `*p = v` where `v` is a DIFFERENT owner than `p`'s
    // referent is refused (no cross-owner place replacement).
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct Other {\n    z: i64\n}\nfn f(p: &mut Pair, other: Other) {\n    *p = other\n}\n";
    assert!(
        codes(src).contains(&"E2029".to_string()),
        "cross-owner `*p = other` must be E2029; got {:?}",
        codes(src)
    );
}

#[test]
fn valid_multiply_produces_no_diagnostics() {
    // Prefix `*` (deref) must not disturb the infix `*` (multiply): a plain
    // multiply type-checks with ZERO diagnostics — no E2028/E2029, no others.
    let src = "fn f(a: i64, b: i64) -> i64 {\n    return a * b\n}\n";
    assert!(fmt(src).contains("a * b"), "infix multiply must round-trip");
    let cs = codes(src);
    assert!(
        cs.is_empty(),
        "a valid multiply must produce NO diagnostics; got {cs:?}"
    );
}

#[test]
fn deref_nested_in_expressions_is_refused() {
    // `*p` buried inside a call argument and an arithmetic expression is still
    // reached by the whole-module refusal scan.
    let src = "fn g(x: i64) -> i64 {\n    return x\n}\nfn f(p: &mut i64) -> i64 {\n    return g(*p) + 1\n}\n";
    assert!(
        codes(src).contains(&"E2028".to_string()),
        "nested `*p` must be E2028"
    );
}

#[test]
fn deref_assign_inside_control_flow_is_refused() {
    // `*p = v` inside an `if` / loop body is reached by the scan.
    let src = "fn f(p: &mut i64, v: i64, c: i64) {\n    if c == 1 {\n        *p = v\n    }\n}\n";
    assert!(
        codes(src).contains(&"E2029".to_string()),
        "`*p = v` in control flow must be E2029"
    );
}

/// No apparent executable support: through the SAME `--emit-shared` entrypoint,
/// a valid control MUST emit an artifact while a deref-assign program MUST fail
/// with the intended refusal (E2029) and write NO artifact. The valid control
/// is what stops a broken backend/formatter/linker from making the negative
/// case look green. Gated on `mlir-build` (native emission); without it this
/// test is NOT EXECUTED — never counted as proof. Unique per-run temp dir so
/// concurrent runs cannot delete each other's artifacts.
#[cfg(feature = "mlir-build")]
#[test]
fn deref_assign_emits_no_artifact_while_valid_control_does() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped("deref_assign_slice0", "mindc not found; not executed");
        return;
    }
    let dir = tempfile::TempDir::new().expect("unique temp dir");
    // Preserve the FULL Output so a positive compile is classified by
    // `gate::compiled` (recognized capability-unavailable → skip, but under
    // MIND_BENCH_REQUIRE=1 a skip HARD-FAILS; a real compiler/CLI/linker error
    // fails loudly) rather than mislabeling any failure as "toolchain
    // unavailable".
    let run = |name: &str, body: &str| -> (std::process::Output, bool, String) {
        let src = dir.path().join(format!("{name}.mind"));
        let so = dir.path().join(format!("{name}.so"));
        std::fs::write(&src, body).expect("write src");
        let out = Command::new(&mindc)
            .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
            .output()
            .expect("run mindc");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out, so.exists(), text)
    };

    // (a) Valid control through the SAME entrypoint. `gate::compiled` classifies
    //     the Output: a genuine unavailable backend skips (NOT proof), but with
    //     MIND_BENCH_REQUIRE=1 that skip is a hard failure and a real
    //     compiler/linker error always fails — so a broken backend can never
    //     make the negative case below look green.
    let (v_out, v_so, v_text) = run(
        "valid_control",
        "fn f(a: i64, v: i64) -> i64 {\n    return a + v;\n}\n",
    );
    if !crate::common::gate::compiled("deref_assign_slice0", &v_out) {
        // Classified as an unavailable backend AND not enforced — not executed.
        return;
    }
    assert!(
        v_so,
        "valid control compiled but emitted no `.so` through --emit-shared:\n{v_text}"
    );

    // (b) Deref-assign through the SAME entrypoint: must FAIL with the intended
    //     E2029 refusal and write NO artifact.
    let (d_out, d_so, d_text) = run(
        "deref_assign",
        "fn f(p: &mut i64, v: i64) {\n    *p = v;\n}\n",
    );
    let d_ok = d_out.status.success();
    assert!(
        !d_ok,
        "deref-assign must NOT emit an artifact; got success:\n{d_text}"
    );
    assert!(
        d_text.contains("E2029"),
        "the failure must be the deref-assign refusal E2029, not an unrelated CLI/formatter/linker error:\n{d_text}"
    );
    assert!(
        !d_so,
        "no `.so` may be written for a refused deref-assign:\n{d_text}"
    );
}

#[test]
fn whole_var_address_of_is_not_refused_E2037() {
    // The carve: a bare `&name` (whole-variable address-of) is the self-host
    // supported form and OUT of this subset's scope — it must NOT draw E2037.
    // (If this ever fires, the Node::Ref arm has over-reached into `&NAME`.)
    let src = "struct Cell {\n    a: i64,\n    b: i64\n}\nfn f(c: Cell) -> i64 {\n    let p = &c\n    return 0\n}\n";
    assert!(
        !codes(src).contains(&"E2037".to_string()),
        "`&c` (whole variable) must NOT be E2037; got {:?}",
        codes(src)
    );
}

// ── Root U1 D4 rejection regressions: four forbidden forms that had compiled
//    to native artifacts. Each must now REFUSE at type-check (no lowering, no
//    artifact). All share a `replace(p: &mut Pair, new: Pair)` callee so the
//    call-site owner/capability checks have a signature to compare against.

#[test]
fn direct_lowering_api_fails_closed_on_deref() {
    // Root U1 source review: the public DIRECT lowering API (no type-check, so
    // no D3 semantic admission) must FAIL CLOSED on any deref — a place
    // operation may only be lowered through the type-checked pipeline. Even the
    // admitted field-first shape refuses here, because no checker ran.
    use libmind::eval::lower_to_ir;
    let scalar = parser::parse("fn f(p: &mut i64, v: i64) {\n    *p = v\n}\n").expect("parse");
    assert!(
        lower_to_ir(&scalar).is_err(),
        "direct lower_to_ir must refuse `*p = v`"
    );
    let record = parser::parse(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn r(p: &mut Pair, n: Pair) {\n    *p = n\n}\n",
    )
    .expect("parse");
    assert!(
        lower_to_ir(&record).is_err(),
        "direct lower_to_ir must refuse even an admitted-shape deref (no type-check ran)"
    );

    let ordinary = parser::parse("fn inc(a: i64) -> i64 {\n    return a + 1\n}\n")
        .expect("parse ordinary scalar control");
    assert!(
        lower_to_ir(&ordinary).is_ok(),
        "public lowering must retain the ordinary scalar positive control"
    );
}

// ── Post-takeover fixes (root over-refusal + audit finding F1/F2/F4). ─────────────────

#[test]
fn immutable_ref_param_forwarding_is_admitted() {
    // ROOT OVER-REFUSAL REGRESSION GUARD (execution-verified by root): an
    // immutable `&T` parameter forwarded to another `&T` callee is valid,
    // pristine behaviour and MUST NOT be refused. This was wrongly E2037 when
    // the escape/forwarding checks keyed on ALL refs instead of `&mut` only.
    let src = "struct Request {\n    id: i64\n}\nfn validate(r: &Request) {\n}\nfn evaluate(r: &Request) {\n    validate(r)\n}\n";
    let cs = codes(src);
    assert!(
        !cs.iter()
            .any(|c| c == "E2037" || c == "E2028" || c == "E2029"),
        "immutable `&T` forwarding `validate(r)` must be admitted; got {cs:?}"
    );
}

#[test]
fn immutable_ref_param_let_and_return_are_admitted() {
    // Immutable refs are unrestricted: let-binding / returning them is fine.
    let l = "struct Request {\n    id: i64\n}\nfn f(r: &Request) -> i64 {\n    let q = r\n    return 0\n}\n";
    assert!(
        !codes(l).iter().any(|c| c == "E2029" || c == "E2037"),
        "immutable `let q = r` must be admitted; got {:?}",
        codes(l)
    );
}

#[test]
fn immutable_ref_field_read_is_admitted_F2_boundary() {
    // The F2 refusal must NOT touch immutable `&T` field reads — this is exactly
    // the cross_module_field_access shape that must stay green.
    let src = "struct Point {\n    x: i64,\n    y: i64\n}\nfn ry(p: &Point) -> i64 {\n    return p.y\n}\n";
    assert!(
        !codes(src)
            .iter()
            .any(|c| c == "E2028" || c == "E2029" || c == "E2037"),
        "immutable `&Point` field read must be admitted; got {:?}",
        codes(src)
    );
}

#[test]
fn immutable_ref_shadow_is_admitted() {
    // Corrected contract (pristine parity, execution-verified against 460f888c):
    // an immutable `&T` parameter has no `*p` store, so shadowing it (`let p =
    // 1`) is harmless and MUST NOT be refused. Only the `&mut` shadow is a
    // hazard (see immutable_reference_shadow_and_wrappers_are_refused).
    let src =
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &Pair) {\n    let p = 1\n}\n";
    assert!(
        !codes(src).iter().any(|c| c == "E2029" || c == "E2037"),
        "immutable `&Pair` shadow must be admitted; got {:?}",
        codes(src)
    );
}

// ── Adversarial-verify follow-ups (audit findings the exec tier + fixtures missed). ──

#[test]
fn immutable_ref_return_is_admitted() {
    // AUDIT (over-refusal): an immutable `&T` return is read-only and
    // pristine-valid (identity on a borrow). It must NOT be refused — the
    // over-refusal fix previously covered forwarding/binding but not returns.
    let src = "struct Point {\n    x: i64\n}\nfn f(p: &Point) -> &Point {\n    return p\n}\n";
    assert!(
        !codes(src).iter().any(|c| c == "E2037" || c == "E2029"),
        "immutable `&Point` return must be admitted; got {:?}",
        codes(src)
    );
    // The two-borrow selector is also universally safe.
    let sel = "struct Point {\n    x: i64\n}\nfn first(a: &Point, b: &Point) -> &Point {\n    return a\n}\n";
    assert!(
        !codes(sel).iter().any(|c| c == "E2037" || c == "E2029"),
        "immutable two-borrow selector return must be admitted; got {:?}",
        codes(sel)
    );
}

#[test]
fn match_binder_shadow_of_mut_param_is_refused() {
    // AUDIT (HIGH): a `match` arm binder shadowing a `&mut` param launders/
    // mis-stores through the stale param slot. Bare-ident and enum-payload
    // binders both refused (E2029).
    let bare = "struct Owner {\n    a: i64\n}\nfn store(p: &mut Owner, sel: i64) {\n    match sel {\n        p => {\n            *p = Owner { a: 5 }\n        }\n    }\n}\n";
    assert!(
        codes(bare).contains(&"E2029".to_string()),
        "bare match-binder shadow of &mut param must be E2029; got {:?}",
        codes(bare)
    );
    let payload = "struct Owner {\n    a: i64\n}\nenum E {\n    V(i64)\n}\nfn store(p: &mut Owner, e: E) {\n    match e {\n        E::V(p) => {\n            *p = Owner { a: 1 }\n        }\n    }\n}\n";
    assert!(
        codes(payload).contains(&"E2029".to_string()),
        "enum-payload match-binder shadow of &mut param must be E2029; got {:?}",
        codes(payload)
    );
}

/// Without `std-surface`, a mutable field address-of is main's pointer argument and the
/// type check must accept it exactly as main 5724a6ee does (parity measured 2026-09-16:
/// main's `mindc check` is clean for these). Only `*p` / `*p = v` are refused here.
#[cfg(not(feature = "std-surface"))]
#[test]
fn mut_field_address_of_matches_main_without_std_surface() {
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn set(p: &mut Pair) {\n    p.x = 5\n}\nfn f(h: H) {\n    set(&mut h.pt)\n}\n";
    let cs = codes(src);
    assert!(
        !cs.iter()
            .any(|c| c == "E2028" || c == "E2029" || c == "E2037"),
        "got {cs:?}"
    );
}
