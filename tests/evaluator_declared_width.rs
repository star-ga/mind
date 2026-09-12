// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The `mindc test` ORACLE must wrap at the DECLARED width.
//!
//! The tree-walking evaluator behind `mindc test` is the layer every MIND test
//! is graded against. It carried every scalar at full i64 width, so on ordinary
//! typed `i32`/`u32` code it disagreed with both shipped backends: measured over
//! five witnesses (executed through a compiled `.so` read back at full width by
//! a C probe, with both-direction positive controls) the evaluator answered WIDE
//! 5/5 where the compiled artifact answered NARROW 5/5. The documented contract
//! is two's-complement wrap at the declared width, "identical on every
//! substrate" — so the ORACLE was the layer that was wrong, and a silently wide
//! oracle grades correct code as wrong and wrong code as right.
//!
//! Three DISTINCT backend mechanisms are covered here, one per group:
//!   RETURN `-> i32` — the callee truncates at the return boundary;
//!   PARAM  `(x: i32)` — the CALLER truncates into the narrow signature;
//!   LOCAL  `let a: i32` — the shift pair in the i64 slot, re-applied on every
//!                        later reassignment of that binding.
//!
//! HARNESS NOTE. Every probe is read back through an `-> i64` boundary, because
//! an in-MIND `==` against a wide literal cannot discriminate on a compiled
//! backend (the compiler truncates the LITERAL too). These tests read the
//! evaluator's `Value` directly, so they are falsifiable by construction — an
//! `assert_eq!` on the number, never a `-> i64` MIND test body (which always
//! passes: only an `assert` inside a MIND test can fail it).
//!
//! Gate: `cargo test --test evaluator_declared_width` — no cfg gate, so this
//! file can never compile out to a vacuous zero-test pass.

use std::collections::HashMap;

use libmind::eval;
use libmind::parser;

const SRC: &str = r#"
// --- RETURN: `-> i32` truncates callee-side at the return boundary ----------
pub fn ret_mul() -> i32 { return 50000 * 50000 }
pub fn ret_add() -> i32 { return 2147483647 + 1 }
pub fn ret_u32() -> u32 { return 65536 * 65536 }
pub fn ret_i64() -> i64 { return 5000000000 }
// Sub-32 declared widths narrow at a call boundary too — the compiled
// artifact does it CALLEE-side (the func.func slot stays i64 and the callee
// masks on entry / at each return, `lower.rs::is_named_narrow_sig_ty`), which
// is value-identical to the caller-side trunci i32/u32 take.
pub fn ret_u8() -> u8 { return 300 }
pub fn ret_i8() -> i8 { return 200 }
pub fn ret_i16() -> i16 { return 40000 }
pub fn param_u8(c: u8) -> i64 { return c }

// --- PARAM: the caller truncates into the narrow signature ------------------
pub fn param_i32(x: i32) -> i32 { return x }
pub fn param_u32(x: u32) -> u32 { return x }
pub fn param_i64(x: i64) -> i64 { return x }

// --- LOCAL: `let a: i32` materialises at the declared width -----------------
pub fn local_i32() -> i64 {
    let a: i32 = 5000000000
    return a
}
pub fn local_u32() -> i64 {
    let a: u32 = 5000000000
    return a
}
pub fn local_i64() -> i64 {
    let a: i64 = 5000000000
    return a
}
// A reassignment carries no annotation; the binding's declared width still
// applies (the compiled path re-masks through its narrow-locals registry).
pub fn local_reassigned() -> i64 {
    let mut a: i32 = 1
    a = 50000 * 50000
    return a
}
// A sub-32 PARAM is NOT truncated at the boundary, but it IS a narrow LOCAL
// slot once inside the callee, so a reassignment re-masks (the compiled path
// seeds its narrow params into the same registry `mask_narrow_assign` reads).
pub fn param_reassigned(c: u8) -> i64 {
    let mut d: u8 = c
    d = d + 100
    return d
}
// A narrow local must NOT leak its width onto an enclosing same-named i64
// binding once the block ends.
pub fn block_scope_leak() -> i64 {
    let mut a: i64 = 1
    if 1 == 1 {
        let a: i32 = 7
        a
    }
    a = 5000000000
    return a
}
// A callee's narrow local must not re-type the caller's same-named i64 local.
pub fn callee_narrow() -> i64 {
    let a: i32 = 3
    return a
}
pub fn fn_scope_leak() -> i64 {
    let mut a: i64 = 1
    let ignored: i64 = callee_narrow()
    a = 5000000000
    return a + ignored
}

// --- the `-> i64` re-widening boundary --------------------------------------
pub fn w_ret_mul() -> i64 { return ret_mul() }
pub fn w_ret_add() -> i64 { return ret_add() }
pub fn w_ret_u32() -> i64 { return ret_u32() }
pub fn w_ret_u8() -> i64 { return ret_u8() }
pub fn w_ret_i8() -> i64 { return ret_i8() }
pub fn w_ret_i16() -> i64 { return ret_i16() }
pub fn w_param_u8() -> i64 { return param_u8(300) }
pub fn w_local_u8() -> i64 {
    let a: u8 = 300
    return a
}
pub fn w_param_i32() -> i64 { return param_i32(5000000000) }
pub fn w_param_u32() -> i64 { return param_u32(5000000000) }
pub fn w_param_i64() -> i64 { return param_i64(5000000000) }
pub fn w_ret_i64() -> i64 { return ret_i64() }
pub fn w_param_reassigned() -> i64 { return param_reassigned(200) }
// Sign discipline: `i32` re-widens with extsi, `u32` with extui. -1 through an
// i32 boundary stays -1; the same -1 through a u32 boundary is 4294967295.
pub fn w_neg_through_i32() -> i64 { return param_i32(0 - 1) }
pub fn w_neg_through_u32() -> i64 { return param_u32(0 - 1) }

// --- STRUCT FIELDS: the fourth mechanism ------------------------------------
// A `[u8; N]` field's ELEMENTS wrap at u8 at construction, and a scalar `u8`
// field wraps too; an `i64` field keeps every bit. MEASURED: the compiled
// artifact stores one byte per element, so [0, 128, 255, 300] becomes
// [0, 128, 255, 44] (sum 683 -> 427), i.e. this probe answers 118 not 182.
struct Packet { pre: i64, bytes: [u8; 4], post: i64 }
fn total_bytes(p: Packet) -> i64 {
    let mut i: i64 = 0
    let mut s: i64 = 0
    while i < 4 {
        s = s + p.bytes[i]
        i = i + 1
    }
    return s
}
pub fn struct_u8_array() -> i64 {
    let p: Packet = Packet { pre: 5, bytes: [0, 128, 255, 300], post: 7 }
    return total_bytes(p) / 4 + p.pre + p.post
}
struct Scal { b: u8, wide: i64 }
pub fn struct_u8_scalar() -> i64 {
    let s: Scal = Scal { b: 300, wide: 1 }
    return s.b
}
pub fn struct_i64_field_untouched() -> i64 {
    let s: Scal = Scal { b: 1, wide: 5000000000 }
    return s.wide
}

// --- INTERMEDIATE NARROW ARITHMETIC: the fifth mechanism --------------------
// An 8/16-bit binop/shift RESULT re-wraps at a width the SOURCE never declares,
// inferred from a 3-valued operand lattice (`narrow_arith`, mirror of
// `lower.rs::infer_narrow_arith_ty`). MEASURED against a `.so` emitted by this
// compiler and read back at full width by a C probe: the intermediate wraps
// FIRST — `(a + 10)` with `a: u8 = 250` is 4, not 260, so `(a + 10) / 2` is 2.
struct P { f: u16 }
pub fn arith_u8_wrap() -> i64 {
    let a: u8 = 250
    return (a + 10) / 2
}
// The regression case: a u8 local joined with a u16 struct FIELD. The join
// takes the WIDEST operand (u16), so `a + p.f` wraps at u16: 60250 fits u16 and
// stays 60250. The pre-fix / under-classified-leaf answer collapses the field
// to Neutral, joins at u8, and answers 90.
pub fn arith_u8_plus_u16_field() -> i64 {
    let a: u8 = 250
    let p = P { f: 60000 }
    return a + p.f
}
// Shift COUNT mask: `a: u8 = 1; a << 8` masks the count mod 8 (`8 & 7 == 0`),
// so the result is `1 << 0 == 1`. Masking the RESULT only (the reverted bug)
// gives `1 << 8 == 256`, then `& 0xFF == 0`.
pub fn arith_u8_shift_count() -> i64 {
    let a: u8 = 1
    return a << 8
}
// u16, chained: `b * 2` = 120000 wraps to 54464, then `+ 100` = 54564.
pub fn arith_u16_chain() -> i64 {
    let b: u16 = 60000
    return (b * 2) + 100
}
// Signed i8: `c + 100` = 200 wraps to -56 (sign-extend, not zero-fill), so the
// following `/ 2` is -28, not 100. Collapsing the signedness would give 100.
pub fn arith_i8_signed() -> i64 {
    let c: i8 = 100
    return (c + 100) / 2
}
// A SHIFT result carries the LEFT operand's width: `a(u8) << 1` = 400 wraps to
// 144, then `+ 1` = 145. No narrowing answers 401.
pub fn arith_u8_shift() -> i64 {
    let a: u8 = 200
    return (a << 1) + 1
}

// --- ADVERSARIAL: the re-mask must NOT over-reach --------------------------
// A `Wide` operand (an `i64` var) widens the whole expr — no u8 wrap (6350).
pub fn adv_wide_i64() -> i64 {
    let a: u8 = 250
    let b: i64 = 100
    return (a + b) + 6000
}
// An all-literal expression is `Wide` (no narrow operand) — no wrap (130).
pub fn adv_all_literal() -> i64 {
    return (250 + 10) / 2
}
// `as i64` widens: `(a as i64 + 10) / 2` is 130, not 2.
pub fn adv_as_wide() -> i64 {
    let a: u8 = 250
    return (a as i64 + 10) / 2
}
// i32 is NOT in the 8/16-bit arith re-mask set: 100000*100000 stays
// 10000000000, not a 32-bit wrap.
pub fn adv_i32_intermediate() -> i64 {
    let a: i32 = 100000
    return a * a
}
// `as u8` IS a narrow cast, so the intermediate DOES wrap: 2, not 130.
pub fn adv_as_u8() -> i64 {
    let x: i64 = 250
    return ((x as u8) + 10) / 2
}
"#;

/// Evaluate `<fn>()` through the tree evaluator by appending a top-level
/// `let` of the call and reading the module's last value.
fn eval_call(func: &str) -> i64 {
    let src = format!("{SRC}\nlet __probe: i64 = {func}()\n");
    let module = parser::parse(&src).expect("probe parses");
    let mut env: HashMap<String, i64> = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, None)
        .unwrap_or_else(|e| panic!("tree-eval of {func}() failed: {e}"));
    match value {
        eval::Value::Int(n) => n,
        other => panic!("tree-eval of {func}() gave a non-int {other:?}"),
    }
}

/// `(probe, narrow value the compiled artifact produces, wide value the broken
/// oracle produced)`. The wide column is asserted to DIFFER, so a case that
/// stops discriminating narrow from wide is caught rather than silently passing.
const WITNESSES: &[(&str, i64, i64)] = &[
    // RETURN — witnesses 1, 2 and 5 of the measured set.
    ("w_ret_mul", -1_794_967_296, 2_500_000_000),
    ("w_ret_add", -2_147_483_648, 2_147_483_648),
    ("w_ret_u32", 0, 4_294_967_296),
    // Every other admitted width, at every mechanism.
    ("w_ret_u8", 44, 300),
    ("w_ret_i8", -56, 200),
    ("w_ret_i16", -25_536, 40_000),
    ("w_param_u8", 44, 300),
    ("w_local_u8", 44, 300),
    // PARAM — witness 3.
    ("w_param_i32", 705_032_704, 5_000_000_000),
    ("w_param_u32", 705_032_704, 5_000_000_000),
    ("w_param_reassigned", 44, 300),
    // LOCAL — witness 4, plus the unannotated reassignment.
    ("local_i32", 705_032_704, 5_000_000_000),
    ("local_u32", 705_032_704, 5_000_000_000),
    ("local_reassigned", -1_794_967_296, 2_500_000_000),
    // STRUCT FIELDS — the fourth mechanism. The `[u8; 4]` element wrap turns
    // the pre-fix wide 182 into 118; the scalar `u8` field wraps 300 to 44.
    ("struct_u8_array", 118, 182),
    ("struct_u8_scalar", 44, 300),
    // INTERMEDIATE NARROW ARITHMETIC — the fifth mechanism. Each narrow value
    // is the compiled artifact's answer; the wide value is what the pre-fix
    // evaluator produced (the intermediate carried full width).
    ("arith_u8_wrap", 2, 130),
    // The regression case: u8 local + u16 field. Widest-wins join ⇒ u16 ⇒
    // 60250. The under-classified-leaf bug joined at u8 and answered 90.
    ("arith_u8_plus_u16_field", 60_250, 90),
    // Shift-count mask: `1u8 << 8` == `1 << (8 & 7)` == 1. Masking the result
    // only (the reverted bug) gives `(1 << 8) & 0xFF` == 0.
    ("arith_u8_shift_count", 1, 0),
    ("arith_u16_chain", 54_564, 120_100),
    ("arith_i8_signed", -28, 100),
    ("arith_u8_shift", 145, 401),
];

#[test]
fn oracle_wraps_at_the_declared_width() {
    for (probe, narrow, wide) in WITNESSES {
        assert_ne!(
            narrow, wide,
            "{probe}: the witness stopped discriminating narrow from wide"
        );
        let got = eval_call(probe);
        assert_eq!(
            got, *narrow,
            "{probe}: oracle answered {got}, the compiled artifact answers {narrow} \
             (the pre-fix wide answer was {wide})"
        );
    }
}

#[test]
fn intermediate_arith_acceptance() {
    // The three lane-targeting acceptance cases, artifact-measured values read
    // through the `-> i64` boundary.
    //
    // The regression case — a u8 local joined with a u16 struct FIELD. This is
    // the case the reverted attempt got WRONG (90): it mapped the FieldAccess
    // leaf to Neutral, so the join fell back to u8. The widest-wins join must
    // pick u16.
    //   MUTATION (join): flip `narrow_arith::join`'s `Narrow(a), Narrow(b)` arm
    //   to always pick the LEFT (`Narrow(a)` regardless of width) — this test
    //   goes RED at 90; reverting the flip restores 60250.
    assert_eq!(
        eval_call("arith_u8_plus_u16_field"),
        60_250,
        "u8 + u16-field must join at the WIDEST (u16 ⇒ 60250), not u8 (⇒ 90)"
    );
    // The shift-COUNT mask — masking the result only (the reverted bug) gives 0.
    //   MUTATION (count mask): delete the `narrow_arith::narrow_shift_count`
    //   call in the `Node::Bitwise` arm (or make it return `count` for shifts) —
    //   this test goes RED at 0; restoring the pre-shift count mask restores 1.
    assert_eq!(
        eval_call("arith_u8_shift_count"),
        1,
        "`1u8 << 8` masks the COUNT mod 8 first ⇒ 1, not `(1<<8)&0xFF` ⇒ 0"
    );
    // The original case this lane targets.
    assert_eq!(
        eval_call("arith_u8_wrap"),
        2,
        "`(a+10)/2` with a:u8=250 wraps the intermediate to 4 first ⇒ 2"
    );
}

#[test]
fn intermediate_arith_does_not_over_narrow() {
    // The fifth mechanism must re-mask ONLY where the artifact does.
    //   * a `Wide` i64 operand widens the whole expression (no u8 wrap);
    assert_eq!(
        eval_call("adv_wide_i64"),
        6350,
        "an i64 operand widens the expr"
    );
    //   * an all-literal expression is `Wide` (no narrow operand to re-mask);
    assert_eq!(
        eval_call("adv_all_literal"),
        130,
        "all-literal must not re-mask"
    );
    //   * an `as i64` cast widens;
    assert_eq!(eval_call("adv_as_wide"), 130, "an `as i64` cast widens");
    //   * i32 is NOT in the 8/16-bit arith re-mask set;
    assert_eq!(
        eval_call("adv_i32_intermediate"),
        10_000_000_000,
        "i32 intermediate arithmetic is not re-masked here"
    );
    //   * but an `as u8` cast IS narrow, so it DOES re-mask.
    assert_eq!(eval_call("adv_as_u8"), 2, "an `as u8` cast re-masks");
}

#[test]
fn full_width_declarations_are_not_narrowed() {
    // The fix must not over-reach: `i64` declarations keep every bit.
    assert_eq!(eval_call("w_ret_i64"), 5_000_000_000);
    assert_eq!(eval_call("w_param_i64"), 5_000_000_000);
    assert_eq!(eval_call("local_i64"), 5_000_000_000);
    // …including a full-width struct field alongside a narrow sibling.
    assert_eq!(eval_call("struct_i64_field_untouched"), 5_000_000_000);
}

#[test]
fn sub32_widths_narrow_at_a_call_boundary_too() {
    // MEASURED against a `.so` emitted by this compiler and read back at full
    // width by a C probe, with NON-CONSTANT operands so nothing is literal
    // folding: `-> u8 { 300 }` is 44, `(c: u8)` given 300 is 44, `-> i16`
    // given 40000 is -25536. Reading `type_ann_to_abi_mlir` alone says these
    // slots are i64 and would have left the oracle answering wide — the
    // artifact narrows them CALLEE-side instead (entry mask + mask_narrow_ret).
    assert_eq!(
        eval_call("w_ret_u8"),
        44,
        "an `-> u8` return masks callee-side"
    );
    assert_eq!(
        eval_call("w_ret_i8"),
        -56,
        "an `-> i8` return masks callee-side"
    );
    assert_eq!(eval_call("w_ret_i16"), -25_536);
    assert_eq!(
        eval_call("w_param_u8"),
        44,
        "a `(c: u8)` parameter is materialised at its declared width on entry"
    );
    assert_eq!(eval_call("w_local_u8"), 44);
}

#[test]
fn unsignedness_is_tracked_separately_from_width() {
    // extsi for a signed narrow width, extui for an unsigned one. Collapsing
    // the two would make both of these agree — they must not.
    let signed = eval_call("w_neg_through_i32");
    let unsigned = eval_call("w_neg_through_u32");
    assert_eq!(signed, -1, "an i32 boundary must sign-extend");
    assert_eq!(unsigned, 4_294_967_295, "a u32 boundary must zero-extend");
    assert_ne!(signed, unsigned);
}

#[test]
fn a_narrow_declaration_does_not_leak_out_of_its_scope() {
    // Recording a narrow width must be scoped exactly as the compiled path
    // scopes it, or the fix introduces a NEW class of silent wrongness: an
    // enclosing i64 binding re-masked to 32 bits after an inner declaration.
    assert_eq!(
        eval_call("block_scope_leak"),
        5_000_000_000,
        "a block-local `let a: i32` re-typed the enclosing i64 `a`"
    );
    assert_eq!(
        eval_call("fn_scope_leak"),
        5_000_000_003,
        "a callee's `let a: i32` re-typed the caller's i64 `a`"
    );
}

#[test]
fn a_narrow_declaration_refuses_a_value_it_cannot_materialise() {
    // FAIL CLOSED, never answer wide: a declared narrow width over a value the
    // evaluator cannot materialise at that width is an error naming the
    // declaration, not a silent full-width pass-through.
    // PARAM, a boundary-admitted width.
    let src = r#"
pub fn takes_u32(c: u32) -> i64 { return 1 }
pub fn probe() -> i64 { return takes_u32("not an int") }
let __probe: i64 = probe()
"#;
    let module = parser::parse(src).expect("refusal probe parses");
    let mut env: HashMap<String, i64> = HashMap::new();
    let err = eval::eval_module_value_with_env(&module, &mut env, None)
        .expect_err("a string bound to a `u32` parameter must be refused, not widened");
    let msg = err.to_string();
    assert!(msg.contains("parameter `c`"), "{msg}");
    assert!(msg.contains("`u32`"), "{msg}");
    assert!(msg.contains("refusing"), "{msg}");

    // LOCAL, a slot-only width — the other admitted set, so a refusal on one
    // set is not mistaken for a refusal on both.
    let src = r#"
pub fn probe() -> i64 {
    let a: u8 = "not an int"
    return 1
}
let __probe: i64 = probe()
"#;
    let module = parser::parse(src).expect("refusal probe parses");
    let mut env: HashMap<String, i64> = HashMap::new();
    let err = eval::eval_module_value_with_env(&module, &mut env, None)
        .expect_err("a string bound to a `u8` binding must be refused, not widened");
    let msg = err.to_string();
    assert!(msg.contains("binding `a`"), "{msg}");
    assert!(msg.contains("`u8`"), "{msg}");
    assert!(msg.contains("refusing"), "{msg}");

    // A NON-narrow declaration must not refuse anything: the refusal is keyed
    // on the declared width, never on the value kind alone.
    let src = r#"
pub fn takes_any(c: i64) -> i64 { return 7 }
pub fn probe() -> i64 { return takes_any("carried at full width") }
let __probe: i64 = probe()
"#;
    let module = parser::parse(src).expect("pass-through probe parses");
    let mut env: HashMap<String, i64> = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, None)
        .expect("an `i64` slot is full-width, so nothing is refused");
    assert_eq!(value, eval::Value::Int(7));
}
