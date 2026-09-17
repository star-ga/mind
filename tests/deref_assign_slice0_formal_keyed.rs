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
//!
//! This file holds the refusals that come from the formal-keyed ADMISSION check
//! (`type_checker::slice_abi::deref_check`, E2037 and its siblings), which is compiled
//! only under `std-surface`. They were split out of `deref_assign_slice0.rs` because
//! that file ran in the `pkg` tier (no `std-surface`) where the check does not exist:
//! 28 tests there reported `got []` — measured 2026-09-16. Measured for the depth-two
//! `&mut h.pt.x` shape only: a non-std-surface `mindc` does not compile it either —
//! lowering ABORTS (a panic, "no IR lowering for `FieldAccess` in value position") —
//! so that shape is refused late and loudly, not admitted. The other 27 shapes were
//! not individually re-measured without `std-surface`. The refusals the BASE type checker owns (E2028/E2029) stay in
//! `deref_assign_slice0.rs`, which runs in every configuration.

#![cfg(feature = "std-surface")]

use libmind::fmt::format_source;
use libmind::parser;
use libmind::project::MindcraftFormatConfig;
use libmind::type_checker::{self, TypeEnv};

// Shared fixture preambles, duplicated from `deref_assign_slice0.rs` (both files are
// separate test crates).
const PAIR_H_REPLACE: &str = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct Other {\n    z: i64\n}\nstruct H {\n    pt: Pair,\n    other: Other\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\n";
const MUT_REPLACE: &str = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct Other {\n    z: i64\n}\nstruct H {\n    pt: Pair,\n    other: Other\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\n";

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
fn reference_return_tail_and_cycle_are_refused_E2037() {
    // A final expression has no Node::Return wrapper, and a mutually recursive
    // call has no finite callee summary. Both still declare reference results,
    // so D4 must reject them before lowering rather than infer a safe lifetime.
    let tail = "struct Pair {
    x: i64,
    y: i64
}
fn leak(p: &mut Pair) -> &mut Pair {
    p
}
";
    let tail_codes = codes(tail);
    assert!(
        tail_codes.contains(&"E2037".to_string()),
        "nominal reference tail return must be refused; got {tail_codes:?}"
    );

    let cycle = "struct Pair {
    x: i64,
    y: i64
}
fn alpha(p: &mut Pair) -> &mut Pair {
    beta(p)
}
fn beta(p: &mut Pair) -> &mut Pair {
    alpha(p)
}
";
    let cycle_codes = codes(cycle);
    assert!(
        cycle_codes.iter().filter(|c| *c == "E2037").count() >= 2,
        "each nominal reference-returning cycle function must be refused; got {cycle_codes:?}"
    );
}

#[test]
fn scalar_reference_return_tail_is_refused_E2037() {
    // The same boundary applies to scalar references; D4's value-producing
    // surface does not gain a scalar-reference return loophole.
    let scalar = "fn leak(p: &mut i64) -> &mut i64 {
    p
}
";
    let scalar_codes = codes(scalar);
    assert!(
        scalar_codes.contains(&"E2037".to_string()),
        "scalar reference tail returns must be refused; got {scalar_codes:?}"
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

#[test]
fn reference_param_to_scalar_callee_is_refused() {
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn caller(p: &mut Pair) {\n    sink(p)\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "a reference parameter passed to a scalar formal must be E2037; got {:?}",
        codes(src)
    );
}

#[test]
fn readonly_reference_param_to_mutable_callee_is_refused() {
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, next: Pair) {\n    *p = next\n}\nfn caller(p: &Pair, next: Pair) {\n    replace(p, next)\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "an immutable reference cannot flow to a mutable formal; got {:?}",
        codes(src)
    );
}

#[test]
fn mutable_reference_forwarding_to_exact_formal_is_admitted() {
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, next: Pair) {\n    *p = next\n}\nfn caller(p: &mut Pair, next: Pair) {\n    replace(p, next)\n}\n";
    let cs = codes(src);
    assert!(
        !cs.iter()
            .any(|c| c == "E2028" || c == "E2029" || c == "E2037"),
        "an exact mutable reference forwarding call must remain admitted; got {cs:?}"
    );
}

#[test]
fn reference_forwarding_to_unknown_callee_is_refused() {
    let src =
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &mut Pair) {\n    unknown(p)\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "an unknown callee cannot establish reference capability/owner; got {:?}",
        codes(src)
    );
}

#[test]
fn reference_forwarding_inside_region_is_refused() {
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, next: Pair) {\n    *p = next\n}\nfn caller(p: &mut Pair, next: Pair) {\n    region {\n        replace(p, next)\n    }\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "reference forwarding inside a region must be E2037; got {:?}",
        codes(src)
    );
}

#[test]
fn immutable_reference_shadow_and_wrappers_are_refused() {
    // The shadow hazard is the MUTABLE case: the flat param table the `*p`
    // admission consults is not updated by a rebind, so shadowing a `&mut`
    // param would let a later `*p` mis-store through the stale slot. An
    // immutable `&T` param has no `*p` store and is unrestricted (pristine
    // parity — see immutable_ref_shadow_is_admitted); so this case is `&mut`.
    let shadow =
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &mut Pair) {\n    let p = 1\n}\n";
    assert!(
        codes(shadow).contains(&"E2029".to_string()),
        "a mutable reference parameter may not be shadowed; got {:?}",
        codes(shadow)
    );

    let wrapped = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, next: Pair) {\n    *p = next\n}\nfn caller(p: &mut Pair, next: Pair) {\n    replace((p), next)\n}\n";
    assert!(
        codes(wrapped).contains(&"E2037".to_string()),
        "a parenthesized reference parameter may not bypass the direct-call rule; got {:?}",
        codes(wrapped)
    );

    let cast = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn caller(p: &mut Pair) {\n    sink(p as i64)\n}\n";
    assert!(
        codes(cast).contains(&"E2037".to_string()),
        "a cast reference parameter may not reach a scalar formal; got {:?}",
        codes(cast)
    );

    let escaped = "struct Pair {
    x: i64,
    y: i64
}
fn caller(p: &mut Pair) {
    let q = (p)
    return (p)
}
";
    assert!(
        codes(escaped).contains(&"E2029".to_string()),
        "a wrapped reference parameter may not escape through bindings or return; got {:?}",
        codes(escaped)
    );

    let consumed = "struct Pair {
    x: i64,
    y: i64
}
fn consume(value: Pair) {
}
fn caller(p: &mut Pair) {
    let q = *p
    consume(*p)
}
";
    let consumed_codes = codes(consumed);
    assert!(
        !consumed_codes
            .iter()
            .any(|code| code == "E2028" || code == "E2029" || code == "E2037"),
        "an admitted dereference consumed by value must not be classified as a reference escape; got {consumed_codes:?}"
    );

    let nested_call = "struct Pair {
    x: i64,
    y: i64
}
fn snapshot(p: &mut Pair) -> Pair {
    return *p
}
fn consume(value: Pair) -> i64 {
    return value.x
}
fn caller(p: &mut Pair) -> i64 {
    return consume(snapshot(p))
}
";
    let nested_codes = codes(nested_call);
    assert!(
        !nested_codes
            .iter()
            .any(|code| code == "E2028" || code == "E2029" || code == "E2037"),
        "a value-producing nested call must consume the reference at its direct formal; got {nested_codes:?}"
    );
}

#[test]
fn value_argument_into_mut_formal_is_refused_F1() {
    // audit finding F1: a by-value record into a `&mut Pair` formal would forge a
    // writable place (field-copy corruption). Refused.
    let src = format!("{MUT_REPLACE}fn c(a: Pair, new: Pair) {{\n    replace(a, new)\n}}\n");
    assert!(
        codes(&src).contains(&"E2037".to_string()),
        "value arg `replace(a,new)` into &mut formal must be E2037; got {:?}",
        codes(&src)
    );
}

#[test]
fn scalar_argument_into_mut_formal_is_refused_F1() {
    // audit finding F1: an i64 into a `&mut Pair` formal = arbitrary-address store.
    let src = format!("{MUT_REPLACE}fn c(addr: i64, new: Pair) {{\n    replace(addr, new)\n}}\n");
    assert!(
        codes(&src).contains(&"E2037".to_string()),
        "scalar arg `replace(addr,new)` into &mut formal must be E2037; got {:?}",
        codes(&src)
    );
}

#[test]
fn immutable_receiver_projection_into_mut_formal_is_refused_F1() {
    // audit finding F1 / root case (1) respelled: `h.pt` from an immutable `&H` into a
    // `&mut Pair` formal launders a read-only receiver into a writable place.
    let src = format!("{MUT_REPLACE}fn c(h: &H, new: Pair) {{\n    replace(h.pt, new)\n}}\n");
    assert!(
        codes(&src).contains(&"E2037".to_string()),
        "`replace(h.pt,new)` (h:&H) must be E2037; got {:?}",
        codes(&src)
    );
}

#[test]
fn field_access_through_mut_param_is_refused_F2() {
    // audit finding F2: `p.x` on a `&mut Pair` param mislowers (cell vs record address).
    // Refused until load-before-field lands.
    let rd = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn getx(p: &mut Pair) -> i64 {\n    return p.x\n}\n";
    assert!(
        codes(rd).contains(&"E2028".to_string()),
        "`p.x` read on &mut param must be E2028; got {:?}",
        codes(rd)
    );
    let wr = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn setx(p: &mut Pair) {\n    p.x = 5\n}\n";
    assert!(
        codes(wr).contains(&"E2029".to_string()),
        "`p.x = 5` on &mut param must be E2029; got {:?}",
        codes(wr)
    );
}

#[test]
fn mut_param_forwarded_to_method_or_scalar_is_refused_F4() {
    // audit finding F4: a `&mut` param forwarded through a scalar formal or a method
    // call escapes; refused.
    let scal = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn f(p: &mut Pair) {\n    sink(p)\n}\n";
    assert!(
        codes(scal).contains(&"E2037".to_string()),
        "`sink(p)` (&mut param into scalar formal) must be E2037; got {:?}",
        codes(scal)
    );
    let tup = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) {\n    let (p, q) = (7, 0)\n}\n";
    assert!(
        codes(tup).contains(&"E2029".to_string()),
        "tuple-let shadow of &mut param must be E2029; got {:?}",
        codes(tup)
    );
}

#[test]
fn mut_ref_return_stays_refused() {
    // The MUTABLE-ref return remains refused (escape).
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) -> &mut Pair {\n    return p\n}\n";
    assert!(
        codes(src).iter().any(|c| c == "E2037" || c == "E2029"),
        "`&mut Pair` return must stay refused; got {:?}",
        codes(src)
    );
}

#[test]
fn mut_field_ref_through_mut_receiver_is_refused_memsafety() {
    // AUDIT (HIGH memory-safety): `&mut h.pt` where h is a `&mut H` PARAMETER was
    // admitted — but a `&mut H` param holds a CELL address, so the field-cell
    // computes addr(h)+offset (wrong-slot/OOB store) with no load-before-field.
    // Must be refused (same hazard as direct `p.f` on a `&mut` param).
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn attack(h: &mut H, new: Pair) {\n    replace(&mut h.pt, new)\n}\n";
    assert!(
        codes(src).contains(&"E2037".to_string()),
        "`&mut h.pt` through a `&mut H` receiver must be E2037 (memory-safety); got {:?}",
        codes(src)
    );
}

#[test]
fn mut_field_ref_through_byvalue_receiver_still_admitted() {
    // The intended positive must SURVIVE the RC1 tightening: `&mut h.pt` where h
    // is a BY-VALUE H (owning place) is admitted (addr(h)+offset is correct).
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn caller(h: H, new: Pair) {\n    replace(&mut h.pt, new)\n}\n";
    assert!(
        !codes(src)
            .iter()
            .any(|c| c == "E2037" || c == "E2028" || c == "E2029"),
        "`&mut h.pt` through a by-value H receiver must stay admitted; got {:?}",
        codes(src)
    );
}
