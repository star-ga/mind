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

/// Pointer parity with main 5724a6ee: no deref-assign refusal. A `&mut` parameter that is
/// never dereferenced is a plain pointer, exactly as on main (see `eval::deref_cells`).
fn assert_pointer_parity(src: &str) {
    let cs = codes(src);
    assert!(
        !cs.iter()
            .any(|c| c == "E2028" || c == "E2029" || c == "E2037"),
        "pointer form must be admitted as on main; got {cs:?}\n{src}"
    );
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

/// Pointer parity: `let c = &h.pt` is main's pointer value and is admitted, as on
/// main 5724a6ee. The capability rule still binds where a CELL is required: that bound
/// reference may not be passed to a cell formal.
#[test]
fn immutable_field_ref_is_refused_E2037() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn f(h: H) {\n    let c = &h.pt\n}\n",
    );
    let cell = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn f(h: H, b: Pair) {\n    let c = &h.pt\n    replace(c, b)\n}\n";
    assert!(
        codes(cell).contains(&"E2037".to_string()),
        "a bound immutable ref into a cell formal must be E2037; got {:?}",
        codes(cell)
    );
}

/// Pointer parity: `let c = &mut h.pt.x` is admitted as on main. A depth-two place is
/// still refused as a CELL argument (only depth-one struct fields have a cell).
#[test]
fn depth_two_field_ref_is_refused_E2037() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn f(h: H) {\n    let c = &mut h.pt.x\n}\n",
    );
    let cell = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nstruct G {\n    h: H\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn f(g: G, b: Pair) {\n    replace(&mut g.h.pt, b)\n}\n";
    assert!(
        codes(cell).contains(&"E2037".to_string()),
        "a depth-two cell argument must be E2037; got {:?}",
        codes(cell)
    );
}

/// Pointer parity: a `&mut h.pt` value taken inside a region is admitted as on main. A
/// cell argument inside a region is still refused (no lifetime provenance).
#[test]
fn region_interior_field_ref_is_refused_E2037() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn f(h: H) {\n    region {\n        let c = &mut h.pt\n    }\n}\n",
    );
    let cell = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct H {\n    pt: Pair\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn f(h: H, b: Pair) {\n    region {\n        replace(&mut h.pt, b)\n    }\n}\n";
    assert!(
        codes(cell).contains(&"E2037".to_string()),
        "a cell argument inside a region must be E2037; got {:?}",
        codes(cell)
    );
}

/// Escape rules bind CELL parameters. A pointer `&mut` parameter may be let-bound or
/// returned exactly as on main; once the function dereferences it, it may not.
#[test]
fn let_bind_and_return_of_ref_param_are_refused_E2029() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) {\n    let q = p\n}\n",
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) -> &mut Pair {\n    return p\n}\n",
    );
    let alias = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) {\n    let v = *p\n    let q = p\n}\n";
    assert!(
        codes(alias).contains(&"E2029".to_string()),
        "`let q = p` of a cell must be E2029; got {:?}",
        codes(alias)
    );
    let ret = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) -> &mut Pair {\n    let v = *p\n    return p\n}\n";
    assert!(
        codes(ret).iter().any(|c| c == "E2029" || c == "E2037"),
        "`return p` of a cell must be refused; got {:?}",
        codes(ret)
    );
}

/// A `&mut` result is refused only for functions with CELL parameters; the pointer
/// tail/cycle forms are admitted as on main.
#[test]
fn reference_return_tail_and_cycle_are_refused_E2037() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn leak(p: &mut Pair) -> &mut Pair {\n    p\n}\n",
    );
    let tail = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn leak(p: &mut Pair) -> &mut Pair {\n    let v = *p\n    p\n}\n";
    assert!(
        codes(tail).contains(&"E2037".to_string()),
        "cell reference tail return must be E2037; got {:?}",
        codes(tail)
    );
    let cycle = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn alpha(p: &mut Pair) -> &mut Pair {\n    let v = *p\n    beta(p)\n}\nfn beta(p: &mut Pair) -> &mut Pair {\n    alpha(p)\n}\n";
    let cycle_codes = codes(cycle);
    assert!(
        cycle_codes.iter().filter(|c| *c == "E2037").count() >= 2,
        "each cell reference-returning cycle function must be refused; got {cycle_codes:?}"
    );
}

/// A scalar `&mut i64` tail return is admitted as on main (pointer). Dereferencing a
/// scalar reference is refused regardless (E2028).
#[test]
fn scalar_reference_return_tail_is_refused_E2037() {
    assert_pointer_parity("fn leak(p: &mut i64) -> &mut i64 {\n    p\n}\n");
    let deref = "fn leak(p: &mut i64) -> i64 {\n    return *p\n}\n";
    assert!(
        codes(deref).contains(&"E2028".to_string()),
        "scalar deref must be E2028; got {:?}",
        codes(deref)
    );
}

/// Pointer parity: `let p = &c.a` / `&mut c.a` are admitted as on main. A scalar field
/// address can never be a cell argument.
#[test]
fn field_address_of_is_refused_E2037() {
    assert_pointer_parity(
        "struct Cell {\n    a: i64,\n    b: i64\n}\nfn f(c: Cell) -> i64 {\n    let p = &c.a\n    return 0\n}\n",
    );
    assert_pointer_parity(
        "struct Cell {\n    a: i64,\n    b: i64\n}\nfn f(c: Cell) -> i64 {\n    let p = &mut c.a\n    return 0\n}\n",
    );
    let cell = "struct Pair {\n    x: i64,\n    y: i64\n}\nstruct Cell {\n    a: i64,\n    b: i64\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn f(c: Cell, b: Pair) {\n    replace(&mut c.a, b)\n}\n";
    assert!(
        codes(cell).contains(&"E2037".to_string()),
        "a scalar field as a cell argument must be E2037; got {:?}",
        codes(cell)
    );
}

/// Pointer parity: `let p = &a[0]` is admitted as on main. An element address is never a
/// cell argument (no element cell lowering).
#[test]
fn element_address_of_is_refused_E2037() {
    assert_pointer_parity("fn f(a: [i64; 4]) -> i64 {\n    let p = &a[0]\n    return 0\n}\n");
    let cell = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn f(a: [Pair; 2], b: Pair) {\n    replace(&mut a[0], b)\n}\n";
    assert!(
        codes(cell).contains(&"E2037".to_string()),
        "an element address as a cell argument must be E2037; got {:?}",
        codes(cell)
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

/// A pointer parameter passed to a scalar formal is admitted as on main; a CELL
/// parameter may only reach an exact `&mut <owner>` cell formal.
#[test]
fn reference_param_to_scalar_callee_is_refused() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn caller(p: &mut Pair) {\n    sink(p)\n}\n",
    );
    let cell = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn caller(p: &mut Pair) {\n    let v = *p\n    sink(p)\n}\n";
    assert!(
        codes(cell).contains(&"E2037".to_string()),
        "a cell into a scalar formal must be E2037; got {:?}",
        codes(cell)
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

/// An unknown callee is refused (E2003) exactly as on main, pointer or cell.
#[test]
fn reference_forwarding_to_unknown_callee_is_refused() {
    let src =
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &mut Pair) {\n    unknown(p)\n}\n";
    assert!(
        codes(src).contains(&"E2003".to_string()),
        "unknown callee must be refused; got {:?}",
        codes(src)
    );
    let cell = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &mut Pair) {\n    let v = *p\n    unknown(p)\n}\n";
    assert!(
        codes(cell).contains(&"E2003".to_string()),
        "unknown callee with a cell must be refused; got {:?}",
        codes(cell)
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

/// Shadow / wrapper / escape rules bind CELL parameters (a later `*p` would store through
/// a stale slot). The pointer forms are admitted as on main; the cell forms stay refused,
/// and an admitted by-value dereference is never mistaken for an escape.
#[test]
fn immutable_reference_shadow_and_wrappers_are_refused() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &mut Pair) {\n    let p = 1\n}\n",
    );
    let shadow = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &mut Pair) {\n    let v = *p\n    let p = 1\n}\n";
    assert!(
        codes(shadow).contains(&"E2029".to_string()),
        "a cell may not be shadowed; got {:?}",
        codes(shadow)
    );
    let wrapped = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, next: Pair) {\n    *p = next\n}\nfn caller(p: &mut Pair, next: Pair) {\n    replace((p), next)\n}\n";
    assert!(
        codes(wrapped).contains(&"E2037".to_string()),
        "a parenthesized cell may not bypass the direct-call rule; got {:?}",
        codes(wrapped)
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn caller(p: &mut Pair) {\n    sink(p as i64)\n}\n",
    );
    let cast = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn caller(p: &mut Pair) {\n    let v = *p\n    sink(p as i64)\n}\n";
    assert!(
        codes(cast).contains(&"E2037".to_string()),
        "a cast cell may not reach a scalar formal; got {:?}",
        codes(cast)
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &mut Pair) {\n    let q = (p)\n    return (p)\n}\n",
    );
    let escaped = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn caller(p: &mut Pair) {\n    let v = *p\n    let q = (p)\n    return (p)\n}\n";
    assert!(
        codes(escaped).contains(&"E2029".to_string()),
        "a wrapped cell may not escape; got {:?}",
        codes(escaped)
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn consume(value: Pair) {\n}\nfn caller(p: &mut Pair) {\n    let q = *p\n    consume(*p)\n}\n",
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn snapshot(p: &mut Pair) -> Pair {\n    return *p\n}\nfn consume(value: Pair) -> i64 {\n    return value.x\n}\nfn caller(p: &mut Pair) -> i64 {\n    return consume(snapshot(p))\n}\n",
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

/// F2 binds CELL parameters (a cell holds the slot address, so `p.x` would read the wrong
/// slot). On a pointer parameter `p.x` / `p.x = 5` compile and run as on main.
#[test]
fn field_access_through_mut_param_is_refused_F2() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn getx(p: &mut Pair) -> i64 {\n    return p.x\n}\n",
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn setx(p: &mut Pair) {\n    p.x = 5\n}\n",
    );
    let rd = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn getx(p: &mut Pair) -> i64 {\n    let v = *p\n    return p.x\n}\n";
    assert!(
        codes(rd).contains(&"E2028".to_string()),
        "`p.x` on a cell must be E2028; got {:?}",
        codes(rd)
    );
    let wr = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn setx(p: &mut Pair) {\n    let v = *p\n    p.x = 5\n}\n";
    assert!(
        codes(wr).contains(&"E2029".to_string()),
        "`p.x = 5` on a cell must be E2029; got {:?}",
        codes(wr)
    );
}

/// F4 binds CELL parameters; the pointer forms are admitted as on main.
#[test]
fn mut_param_forwarded_to_method_or_scalar_is_refused_F4() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn f(p: &mut Pair) {\n    sink(p)\n}\n",
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) {\n    let (p, q) = (7, 0)\n}\n",
    );
    let scal = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn sink(x: i64) {\n}\nfn f(p: &mut Pair) {\n    let v = *p\n    sink(p)\n}\n";
    assert!(
        codes(scal).contains(&"E2037".to_string()),
        "a cell into a scalar formal must be E2037; got {:?}",
        codes(scal)
    );
    let tup = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) {\n    let v = *p\n    let (p, q) = (7, 0)\n}\n";
    assert!(
        codes(tup).contains(&"E2029".to_string()),
        "tuple-let shadow of a cell must be E2029; got {:?}",
        codes(tup)
    );
}

/// A `&mut` return stays refused for a function with a CELL parameter; the pointer form is
/// admitted as on main.
#[test]
fn mut_ref_return_stays_refused() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) -> &mut Pair {\n    return p\n}\n",
    );
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn f(p: &mut Pair) -> &mut Pair {\n    let v = *p\n    return p\n}\n";
    assert!(
        codes(src).iter().any(|c| c == "E2037" || c == "E2029"),
        "cell `&mut Pair` return must stay refused; got {:?}",
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

/// Parentheses must not hide a CELL receiver: `(p).y = v` lowers exactly like `p.y = v`,
/// a wrong-slot write through a cell address, so it is refused like the bare form at any
/// depth, store and read (audit 2026-09-16). On a POINTER parameter `(p).y` is correct and
/// admitted as on main.
#[test]
fn parenthesised_mut_ref_receiver_is_refused_like_the_bare_form() {
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn s(p: &mut Pair, v: i64) {\n    (p).y = v\n}\n",
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\nfn s(p: &mut Pair) -> i64 {\n    return (p).y\n}\n",
    );
    for (src, code) in [
        (
            "struct Pair {\n    x: i64,\n    y: i64\n}\nfn s(p: &mut Pair, v: i64) {\n    let c = *p\n    (p).y = v\n}\n",
            "E2029",
        ),
        (
            "struct Pair {\n    x: i64,\n    y: i64\n}\nfn s(p: &mut Pair, v: i64) {\n    let c = *p\n    ((p)).y = v\n}\n",
            "E2029",
        ),
        (
            "struct Pair {\n    x: i64,\n    y: i64\n}\nfn s(p: &mut Pair) -> i64 {\n    let c = *p\n    return (p).y\n}\n",
            "E2028",
        ),
    ] {
        let cs = codes(src);
        assert!(
            cs.contains(&code.to_string()),
            "{src:?} must be {code}; got {cs:?}"
        );
    }
}

/// A NESTED function's calls are classified with ITS OWN parameters and the
/// enclosing module's callee signatures (audit 2026-09-16). Before, the nested body
/// was classified under the OUTER `&mut` table and its own check ran without the
/// module's signatures, so `replace(p, new)` with an `i64` `p` was admitted.
#[test]
fn nested_fn_calls_are_classified_by_their_own_params() {
    let src = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn outer(p: &mut Pair, new: Pair) {\n    fn inner(p: i64, new: Pair) {\n        replace(p, new)\n    }\n}\n";
    let cs = codes(src);
    assert!(
        cs.contains(&"E2037".to_string()),
        "an i64 `p` into a `&mut Pair` formal inside a nested fn must be E2037; got {cs:?}"
    );
    // Control: the identical call at top level is refused the same way, so the
    // nested verdict is not an artefact of nesting.
    let top = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn inner(p: i64, new: Pair) {\n    replace(p, new)\n}\n";
    assert!(codes(top).contains(&"E2037".to_string()));
}

/// Positive control for the std-surface cell admission (moved from
/// `deref_assign_slice0.rs`, which also runs without std-surface).
#[test]
fn admitted_mut_field_ref_and_param_forwarding_still_pass() {
    // Positive controls the F1 gate must NOT over-refuse: `&mut h.pt` (owning
    // receiver) and a bare `&mut Pair` param forwarded to the exact formal.
    let field = format!("{MUT_REPLACE}fn c(h: H, new: Pair) {{\n    replace(&mut h.pt, new)\n}}\n");
    assert!(
        !codes(&field).iter().any(|c| c == "E2037"),
        "admitted `replace(&mut h.pt,new)` must pass; got {:?}",
        codes(&field)
    );
    let fwd = format!("{MUT_REPLACE}fn c(p: &mut Pair, new: Pair) {{\n    replace(p, new)\n}}\n");
    assert!(
        !codes(&fwd).iter().any(|c| c == "E2037"),
        "admitted `&mut Pair` param forwarding must pass; got {:?}",
        codes(&fwd)
    );
}

/// A `pub` function may not have a CELL parameter: a caller in another module runs its
/// own cell inference and would pass a plain pointer where this body loads a slot. The
/// same function without `pub` is admitted, and a `pub` POINTER function stays admitted.
#[test]
fn pub_fn_with_cell_parameter_is_refused() {
    let public = "struct Pair {\n    x: i64,\n    y: i64\n}\npub fn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\n";
    assert!(
        codes(public).contains(&"E2029".to_string()),
        "a `pub` cell function must be E2029; got {:?}",
        codes(public)
    );
    let private = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\n";
    assert!(
        !codes(private)
            .iter()
            .any(|c| c == "E2028" || c == "E2029" || c == "E2037"),
        "the private cell function must be admitted; got {:?}",
        codes(private)
    );
    assert_pointer_parity(
        "struct Pair {\n    x: i64,\n    y: i64\n}\npub fn set(p: &mut Pair) {\n    p.x = 5\n}\n",
    );
}

/// A CELL must never reach a POINTER formal: the callee would treat the slot address as
/// the record and `p.x = 5` would overwrite the slot (memory corruption). Refused, while
/// the same forwarding from a pointer parameter is admitted as on main.
#[test]
fn cell_parameter_into_pointer_formal_is_refused() {
    let set = "struct Pair {\n    x: i64,\n    y: i64\n}\nfn set(p: &mut Pair) {\n    p.x = 5\n}\n";
    assert_pointer_parity(&format!(
        "{set}fn caller(p: &mut Pair) {{\n    set(p)\n}}\n"
    ));
    let cell = format!("{set}fn caller(p: &mut Pair) {{\n    let v = *p\n    set(p)\n}}\n");
    assert!(
        codes(&cell).contains(&"E2037".to_string()),
        "a cell into a pointer formal must be E2037; got {:?}",
        codes(&cell)
    );
}
