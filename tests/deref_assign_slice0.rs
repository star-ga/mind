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
//! Everything OUTSIDE that subset is refused with the exact codes: `E2028`
//! (unsupported deref), `E2029` (unsupported deref-assign — scalar referent,
//! cross-owner value, `let q = p` / `return p` escape), `E2037` (unsupported
//! field/element address-of — immutable, scalar, depth-two, region-interior,
//! unknown owner). Scalar `&mut i64` deref stays refused; it is not in the
//! struct field-first subset.
//!
//! Generic types only — no private consumer (MindLLM/PageTable) source here.

mod common;
use common::mindc_bin;

use libmind::fmt::format_source;
use libmind::parser;
use libmind::project::MindcraftFormatConfig;
use libmind::type_checker::{self, TypeEnv};

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
fn record_deref_assign_on_struct_is_admitted() {
    // Carve-out: the callee-side record place replacement `*p = next`, where `p`
    // is a `&mut <struct>` parameter and `next` is a record of the EXACT owner,
    // is the ADMITTED shape (identity-preserving, not a field copy). It must
    // round-trip AND type-check with no deref diagnostic. (Executable identity
    // semantics are proven by the compiled fixture below.)
    let src = "struct Cell {\n    a: i64,\n    b: i64\n}\nfn r(p: &mut Cell, next: Cell) {\n    *p = next\n}\n";
    assert!(
        fmt(src).contains("*p = next"),
        "record deref-assign must round-trip"
    );
    let cs = codes(src);
    assert!(
        !cs.iter()
            .any(|c| c == "E2028" || c == "E2029" || c == "E2037"),
        "admitted record `*p = next` must have no deref diagnostic; got {cs:?}"
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
fn field_ref_and_forward_call_is_admitted() {
    // The full caller shape: `&mut h.pt` (depth-one struct field) forwarded as a
    // direct call argument to a `&mut Pair` callee. Both the field-ref and the
    // callee store are admitted → no deref diagnostic.
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn caller(h: H, new: Pair) {\n    replace(&mut h.pt, new)\n}\n";
    let cs = codes(src);
    assert!(
        !cs.iter()
            .any(|c| c == "E2028" || c == "E2029" || c == "E2037"),
        "admitted `&mut h.pt` + callee `*p = new` must have no deref diagnostic; got {cs:?}"
    );
}

#[test]
fn immutable_field_ref_is_refused_E2037() {
    // Capability: an IMMUTABLE `&h.pt` (struct field) may not become a writable
    // place — no laundering. Refused E2037 even though the field is a struct.
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn f(h: H) {\n    let c = &h.pt\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "immutable `&h.pt` must be E2037; got {:?}",
        codes(src)
    );
}

#[test]
fn depth_two_field_ref_is_refused_E2037() {
    // `&mut h.pt.x` (depth-two) is refused — only depth-one is admitted.
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn f(h: H) {\n    let c = &mut h.pt.x\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "depth-two `&mut h.pt.x` must be E2037; got {:?}",
        codes(src)
    );
}

#[test]
fn region_interior_field_ref_is_refused_E2037() {
    // A `&mut h.pt` captured inside a region is refused (no lifetime provenance).
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn f(h: H) {\n    region {\n        let c = &mut h.pt\n    }\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "region-interior `&mut h.pt` must be E2037; got {:?}",
        codes(src)
    );
}

#[test]
fn let_bind_and_return_of_ref_param_are_refused_E2029() {
    // Escape: a `&mut` parameter may only be a direct call argument — never
    // let-bound (`let q = p`) or returned (`return p`).
    let alias =
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) {\n    let q = p\n}\n";
    assert!(
        codes(alias).contains(&"E2029".to_string()),
        "`let q = p` (alias escape) must be E2029; got {:?}",
        codes(alias)
    );
    let ret = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) -> &mut Pair {\n    return p\n}\n";
    assert!(
        codes(ret).contains(&"E2029".to_string()),
        "`return p` (escape) must be E2029; got {:?}",
        codes(ret)
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
fn field_address_of_is_refused_E2037() {
    // `&r.f` / `&mut r.f` (field address-of) is the field-first subset's target
    // shape but is REFUSED until the D4 cell-address lowering lands (coupled).
    // Closes the lower.rs Phase-10.7 no-op-value hole.
    let imm = "struct Cell {\n    a: i64,\n    b: i64\n}\nfn f(c: Cell) -> i64 {\n    let p = &c.a\n    return 0\n}\n";
    assert!(
        codes(imm).contains(&"E2037".to_string()),
        "`&c.a` must be E2037; got {:?}",
        codes(imm)
    );
    let mutf = "struct Cell {\n    a: i64,\n    b: i64\n}\nfn f(c: Cell) -> i64 {\n    let p = &mut c.a\n    return 0\n}\n";
    assert!(
        codes(mutf).contains(&"E2037".to_string()),
        "`&mut c.a` must be E2037; got {:?}",
        codes(mutf)
    );
}

#[test]
fn element_address_of_is_refused_E2037() {
    // `&a[i]` (element address-of) is deferred in the subset (no element cell
    // lowering yet) — matches the self-host emitter's deferred set.
    let src = "fn f(a: [i64; 4]) -> i64 {\n    let p = &a[0]\n    return 0\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "`&a[0]` must be E2037; got {:?}",
        codes(src)
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
const PAIR_H_REPLACE: &str = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct Other {\n    z: i64\n}\nstruct H {\n    pt: Pair,\n    other: Other\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\n";

#[test]
fn ref_through_immutable_receiver_is_refused() {
    // (1) `&mut h.pt` through `h: &H` (immutable ref receiver) launders a write
    //     capability — refused.
    let src = format!(
        "{PAIR_H_REPLACE}fn caller(h: &H, new: Pair) {{\n    replace(&mut h.pt, new)\n}}\n"
    );
    assert!(
        codes(&src).contains(&"E2037".to_string()),
        "`&mut h.pt` through `&H` must be E2037; got {:?}",
        codes(&src)
    );
}

#[test]
fn ref_to_wrong_owner_callee_param_is_refused() {
    // (2) `&mut h.other` (owner Other) passed where the callee expects
    //     `&mut Pair` — call-site owner mismatch, refused.
    let src = format!(
        "{PAIR_H_REPLACE}fn caller(h: H, new: Pair) {{\n    replace(&mut h.other, new)\n}}\n"
    );
    // (H has no `other` field; even if it did with type Other, the owner would
    //  mismatch the &mut Pair parameter.) Must refuse, no admit.
    assert!(
        codes(&src).contains(&"E2037".to_string()),
        "cross-owner `&mut h.other` to `&mut Pair` must be E2037; got {:?}",
        codes(&src)
    );
}

#[test]
fn let_bound_field_ref_is_refused() {
    // (3) `let q = &mut h.pt; replace(q, new)` — the field ref is let-bound
    //     (not a direct call argument). Refused at the let.
    let src = format!(
        "{PAIR_H_REPLACE}fn caller(h: H, new: Pair) {{\n    let q = &mut h.pt\n    replace(q, new)\n}}\n"
    );
    assert!(
        codes(&src).contains(&"E2037".to_string()),
        "let-bound `&mut h.pt` must be E2037; got {:?}",
        codes(&src)
    );
}

#[test]
fn cast_laundered_field_ref_is_refused() {
    // (4) `let q = (&mut h.pt) as i64; replace(q, new)` — the field ref is cast
    //     to i64 (capability/owner provenance laundered). Refused at the cast.
    let src = format!(
        "{PAIR_H_REPLACE}fn caller(h: H, new: Pair) {{\n    let q = (&mut h.pt) as i64\n    replace(q, new)\n}}\n"
    );
    assert!(
        codes(&src).contains(&"E2037".to_string()),
        "cast-laundered `(&mut h.pt) as i64` must be E2037; got {:?}",
        codes(&src)
    );
}
