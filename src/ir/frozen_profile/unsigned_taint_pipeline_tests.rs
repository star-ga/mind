// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Pipeline tests for the unsigned taint (`unsigned_taint`): real source lowered through
//! the production entry and asked of the fence, plus the compare-taint / MLIR-predicate
//! bijection. Split from `unsigned_taint.rs` for the module-size budget.

use crate::ir::Instr;
use crate::ir::frozen_profile::profile_frozen_admits;

/// Lower real source through the production pipeline and ask the fence. These
/// tests pin the module SHAPE lowering actually produces — the hand-built-IR tests
/// in `unsigned_taint.rs` can insert a signature that lowering never provides (which is exactly how
/// the nested-function door stayed open).
fn fence(src: &str) -> Option<&'static str> {
    let module = crate::parser::parse(src).expect("fixture must parse");
    let ir = crate::eval::lower_to_ir(&module).expect("fixture must lower");
    profile_frozen_admits(&ir).err().map(|r| r.construct)
}

#[test]
fn pipeline_nested_fn_with_u64_param_is_refused() {
    // Before: admitted by fence 1 (no `fn_signatures` entry for `inner`).
    let nested = "fn main() -> i64 {\n    fn inner(x: u64) -> i64 {\n        return x / 2\n    }\n    return 0\n}\n";
    assert!(fence(nested).is_some(), "nested u64 div must be refused");
    // Undecidable => refuse also covers a nested i64 body (its signature is equally
    // unknown). No capability is lost: the frozen ELF refuses nested fns anyway.
    let nested_i64 = "fn main() -> i64 {\n    fn inner(x: i64) -> i64 {\n        return x / 2\n    }\n    return 0\n}\n";
    assert!(fence(nested_i64).is_some());
    // Control: the same helper at TOP level has a known i64 signature and stays
    // admitted, so the refusal above is about nesting, not about `/`.
    let top = "fn inner(x: i64) -> i64 {\n    return x / 2\n}\nfn main() -> i64 {\n    return inner(8)\n}\n";
    assert_eq!(
        fence(top),
        None,
        "top-level signed division must stay admitted"
    );
    let top_u64 =
        "fn inner(x: u64) -> i64 {\n    return x / 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(top_u64), Some("binop.div_mod_unsigned"));
}

#[test]
fn pipeline_u64_inside_an_array_param_taints_the_body() {
    let arr = "fn f(a: [u64; 2]) -> i64 {\n    return a[0] / a[1]\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(arr), Some("binop.div_mod_unsigned"));
    // ...but only for division: MLIR compares a `[u64; N]` element SIGNED, and main
    // 5724a6ee compiles this compare natively.
    let arr_lt = "fn f(a: [u64; 2]) -> i64 {\n    if a[0] < a[1] {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(arr_lt), None);
    let arr_i64 = "fn f(a: [i64; 2]) -> i64 {\n    return a[0] / a[1]\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(arr_i64), None, "the i64 twin must stay admitted");
}

/// The `let x: u64` / `as u64` door is shut by `admit_call` refusing the synthesised
/// `__mind_conv_u64`, INDEPENDENTLY of the native bridge's closure check. Pinned by
/// name because nothing else asserts it: an intrinsic allowlist added to
/// `admit_call` by analogy with `__mind_conv_i64` would silently reopen the
/// unsigned-division hole (audit 2026-09-16).
#[test]
fn conv_u64_call_is_refused_by_the_fence_itself() {
    let cast =
        "fn main() -> i64 {\n    let y: i64 = 84\n    let x: u64 = y as u64\n    return 0\n}\n";
    let module = crate::parser::parse(cast).expect("parse");
    let ir = crate::eval::lower_to_ir(&module).expect("fixture must lower");
    fn has_conv_u64(instrs: &[Instr]) -> bool {
        instrs.iter().any(|i| {
            matches!(i, Instr::Call { name, .. } if name == "__mind_conv_u64")
                || crate::ir::instr_bodies(i).iter().any(|b| has_conv_u64(b))
        })
    }
    assert!(
        has_conv_u64(&ir.instrs),
        "positive control: lowering emits __mind_conv_u64"
    );
    assert_eq!(fence(cast), Some("call.undefined_or_builtin"));
}

/// PARITY with main 5724a6ee (measured 2026-09-16: native and MLIR both return 3):
/// an i64 loop guard in a function with a `u64` parameter is not an unsigned
/// compare. The body-level taint refused it.
#[test]
fn pipeline_i64_compare_beside_a_u64_param_is_admitted() {
    let src = "fn f(a: u64) -> i64 {\n    let i = 0\n    while i < 3 {\n        i = i + 1\n    }\n    return i\n}\nfn main() -> i64 {\n    return f(7)\n}\n";
    assert_eq!(fence(src), None);
}

/// `usize` is signed on BOTH backends today, so it is not tainted (main compiles
/// `a < b` on `usize` natively, returning 1 for `f(1, 2)`). If MLIR ever selects an
/// unsigned predicate for `usize`, this fails and `is_unsigned_wide_type` must add it.
#[test]
fn usize_is_signed_on_the_mlir_backend_too() {
    let src = "fn f(a: usize, b: usize) -> i64 {\n    if a < b {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(1, 2)\n}\n";
    assert_eq!(fence(src), None);
    // The MLIR half needs the lowering module; the `mlir-build` tiers run it.
    #[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
    {
        let module = crate::parser::parse(src).expect("parse");
        let mut ir = crate::eval::lower_to_ir(&module).expect("lower");
        let text = crate::mlir::compile_ir_to_mlir_text(&mut ir).expect("mlir text");
        assert!(
            text.contains("cmpi \"slt\""),
            "usize compare must be signed:\n{text}"
        );
        assert!(
            !text.contains("cmpi \"ult\""),
            "usize compare turned unsigned:\n{text}"
        );
    }
}

/// The miscompile the compare refusal exists for, measured on main 5724a6ee: with
/// `b = 18446744073709551615`, `a < b` gives 0 natively and 1 on MLIR, and
/// `big() > 5` gives 0 natively and 1 on MLIR.
#[test]
fn pipeline_unsigned_operand_compares_are_refused() {
    let param = "fn f(a: u64, b: u64) -> i64 {\n    if a < b {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(1, 18446744073709551615)\n}\n";
    assert_eq!(fence(param), Some("binop.ordered_compare_unsigned"));
    let ret = "fn big() -> u64 {\n    return 18446744073709551615\n}\nfn main() -> i64 {\n    if big() > 5 {\n        return 1\n    }\n    return 0\n}\n";
    assert_eq!(fence(ret), Some("binop.ordered_compare_unsigned"));
}

/// Probe programs for the compare taint, each with ONE ordered compare:
/// `(label, source, MLIR selects an unsigned predicate, the fence refuses it)`.
const COMPARE_SHAPES: &[(&str, &str, bool, bool)] = &[
    (
        "u64 params",
        "fn f(a: u64, b: u64) -> i64 {\n    if a < b {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "u64 call result",
        "fn big() -> u64 {\n    return 18446744073709551615\n}\nfn main() -> i64 {\n    if big() > 5 {\n        return 1\n    }\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "arithmetic on a u64",
        "fn f(a: u64) -> i64 {\n    let s = a + 1\n    if s > 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "if merge absorbs u64",
        "fn f(a: u64, c: i64) -> i64 {\n    let m = 0\n    if c == 1 {\n        m = a\n    }\n    if m > 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "loop-carried accumulator keeps its i64 init kind",
        "fn f(a: u64) -> i64 {\n    let s = 0\n    while s < 100 {\n        s = s + a\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "accumulator compared after the loop",
        "fn f(a: u64) -> i64 {\n    let s = 0\n    let i = 0\n    while i < 3 {\n        s = s + a\n        i = i + 1\n    }\n    if s > 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "u64 array element loads as i64",
        "fn f(a: [u64; 2]) -> i64 {\n    if a[0] < a[1] {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "i64 guard beside a u64 param",
        "fn f(a: u64) -> i64 {\n    let i = 0\n    while i < 3 {\n        i = i + 1\n    }\n    return i\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "u64 masked by a non-negative constant",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    if m < 10 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "`a & b` of two u64 params is not a mask",
        "fn f(a: u64, b: u64) -> i64 {\n    let m = a & b\n    if m < 10 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "arithmetic on a masked value re-taints it",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let t = m - 256\n    if t < 10 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "an all-ones constant is not a mask",
        "fn f(a: u64) -> i64 {\n    let m = a & 18446744073709551615\n    if m < 10 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "an if-join of two masks stays a mask",
        "fn f(a: u64, c: i64) -> i64 {\n    let m = a & 15\n    if c == 1 {\n        m = a & 255\n    }\n    if m < 10 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "xor of a mask with a non-negative constant stays a mask",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let t = m ^ 9223372036854775807\n    if t < 10 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "or of a mask with an unmasked u64 is not a mask",
        "fn f(a: u64, b: u64) -> i64 {\n    let m = a & 255\n    let t = m | b\n    if t < 10 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "a masked loop variable modified in the body is not a mask (compare in body)",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let i = 0\n    let r = 0\n    while i < 2 {\n        if m < 0 {\n            r = 1\n        }\n        m = m - 256\n        i = i + 1\n    }\n    return r\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "a masked loop variable compared in the loop CONDITION is not a mask",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let i = 0\n    while m < 300 {\n        m = m - 256\n        i = i + 1\n        if i > 5 {\n            m = 1000\n        }\n    }\n    return i\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "a loop-carried constant is not a non-negative constant",
        "fn f(a: u64) -> i64 {\n    let c = 255\n    let i = 0\n    let r = 0\n    while i < 2 {\n        let m = a & c\n        if m < 0 {\n            r = 1\n        }\n        c = -1\n        i = i + 1\n    }\n    return r\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "an if-join with a loop variable inside the loop is not a mask",
        "fn f(a: u64, c: i64) -> i64 {\n    let m = a & 255\n    let i = 0\n    let r = 0\n    while i < 2 {\n        if c == 1 {\n            m = a & 15\n        }\n        if m < 0 {\n            r = 1\n        }\n        m = m - 256\n        i = i + 1\n    }\n    return r\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "or with a loop variable is not a mask",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let i = 0\n    let r = 0\n    while i < 2 {\n        let t = m | (a & 15)\n        if t < 0 {\n            r = 1\n        }\n        m = m - 256\n        i = i + 1\n    }\n    return r\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "a mask read inside a loop but never reassigned stays a mask",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let i = 0\n    let r = 0\n    while i < 2 {\n        if m < 0 {\n            r = 1\n        }\n        i = i + 1\n    }\n    return r\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "a masked operand beside a negative literal",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    if m < -1 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "a masked operand beside an i64 param",
        "fn f(a: u64, b: i64) -> i64 {\n    let m = a & 255\n    if b < m {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "a masked operand beside a loop counter",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let k = 1\n    let i = 0\n    let r = 0\n    while i < 3 {\n        if m > k {\n            r = r + 1\n        }\n        k = k - 1\n        i = i + 1\n    }\n    return r\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "two masked operands",
        "fn f(a: u64, b: u64) -> i64 {\n    let m = a & 255\n    let n = b & 15\n    if m < n {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "a masked operand beside a non-negative literal",
        "fn f(a: u64) -> i64 {\n    let m = a & 255\n    if m < 10 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "u32 beside a negative literal (i32 ult on the truncated literal)",
        "fn f(a: u32) -> i64 {\n    if a < -1 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "u32 beside a literal at or above 2^32 (truncated)",
        "fn f(a: u32) -> i64 {\n    if a < 4294967296 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "u32 beside a literal in [0, 2^32) agrees",
        "fn f(a: u32) -> i64 {\n    if a < 3000000000 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "u32 beside a u32 agrees (both zero-extend)",
        "fn f(a: u32, b: u32) -> i64 {\n    if a < b {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "u32 beside a non-literal i64 widens signed",
        "fn f(a: u32, b: i64) -> i64 {\n    if a < b {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "u32 plus a non-literal i64 widens to signed i64",
        "fn f(a: u32, b: i64) -> i64 {\n    let t = a + b\n    if t < -1 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "u32 plus an i32 widens to signed i64",
        "fn f(a: u32, b: i32) -> i64 {\n    let t = a + b\n    if t < -1 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "an if-join of a literal and a u32 widens",
        "fn f(a: u32, c: i64) -> i64 {\n    let s = 5\n    if c > 0 {\n        s = a\n    }\n    if s < -1 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "a loop accumulating a u32 into an i64 init widens",
        "fn f(a: u32) -> i64 {\n    let s = 0\n    let i = 0\n    while i < 2 {\n        s = s + a\n        i = i + 1\n    }\n    if s < -1 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "a u32 call result plus an i64 widens",
        "fn g() -> u32 {\n    return 7\n}\nfn f(b: i64) -> i64 {\n    let x = g()\n    let t = x + b\n    if t <= -1 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "u32 & literal is narrow but not wrapped (round 6)",
        "fn f(a: u32) -> i64 {\n    let m = a & 255\n    if m < 3 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "mask then + 1 is wrapped again",
        "fn f(a: u32) -> i64 {\n    let m = a & 255\n    let n = m + 1\n    if n < 300 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
    (
        "u32 loop counter seeded from a param (MLIR cannot build it; main correct)",
        "fn f(a: u32, n: u32) -> i64 {\n    let i = a\n    let acc = 0\n    while i < n {\n        acc = acc + 1\n        i = i + 1\n    }\n    return acc\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "u8 compare in range agrees (MLIR masks, native zero-extends)",
        "fn f(a: u8, b: u8) -> i64 {\n    if a < b {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        false,
        false,
    ),
    (
        "u32 join of two u32 params compared in range",
        "fn f(a: u32, b: u32, c: i64) -> i64 {\n    let x = b\n    if c == 1 {\n        x = a\n    }\n    if x < 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        false,
    ),
    (
        "an i32 ⊔ u32 join absorbs to u32 on MLIR (ult on a negative)",
        "fn f(a: i32, b: u32, c: i64) -> i64 {\n    let x = b\n    if c == 1 {\n        x = a\n    }\n    if x < 0 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        true,
        true,
    ),
];

/// The fence refuses an ordered compare ONLY where MLIR emits an unsigned predicate
/// (refusing more loses programs main compiles natively with the same result as MLIR —
/// audit 2026-09-16), and refuses EVERY such compare unless the operand provably has
/// bit 63 clear (then signed and unsigned agree, so the native signed `setcc` is
/// correct). The MLIR half re-derives the predicate from the emitted text, so a change
/// to MLIR's kinds turns this red.
#[test]
fn compare_taint_refuses_only_mlir_unsigned_predicates() {
    for (label, src, unsigned, refuses) in COMPARE_SHAPES {
        assert!(
            *unsigned || !*refuses,
            "{label}: the fence may not refuse a compare MLIR emits signed"
        );
        let want = refuses.then_some("binop.ordered_compare_unsigned");
        assert_eq!(fence(src), want, "fence verdict for {label}");
        #[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
        {
            let module = crate::parser::parse(src).expect("parse");
            let mut ir = crate::eval::lower_to_ir(&module).expect("lower");
            let text = crate::mlir::compile_ir_to_mlir_text(&mut ir).expect("mlir text");
            let has = |p: &str| text.contains(&format!("cmpi \"{p}\""));
            let emitted_unsigned = ["ult", "ule", "ugt", "uge"].iter().any(|p| has(p));
            let emitted_signed = ["slt", "sle", "sgt", "sge"].iter().any(|p| has(p));
            assert!(
                emitted_unsigned || emitted_signed,
                "{label}: no ordered compare in the MLIR:\n{text}"
            );
            assert_eq!(
                emitted_unsigned, *unsigned,
                "{label}: MLIR predicate signedness disagrees with the fence:\n{text}"
            );
        }
    }
}

/// A narrow operand that may have WRAPPED (u32 + u32 is 32-bit on MLIR, 64-bit
/// natively: `s = a + b; s < 5` with `f(4294967295, 1)` — main native 0 vs MLIR 1) or
/// a literal MLIR truncates (`a == 4294967296` is `a == 0` there — main native 0 vs
/// MLIR 1) is refused for every compare, `==`/`!=` included; a bare narrow parameter
/// compared in range is admitted (it is exact on both backends for in-range callers).
#[test]
fn narrow_wrapped_or_truncated_compares_are_refused() {
    let wrapped_lt = "fn f(a: u32, b: u32) -> i64 {\n    let s = a + b\n    if s < 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(wrapped_lt), Some("binop.ordered_compare_unsigned"));
    let truncated_eq = "fn f(a: u32) -> i64 {\n    if a == 4294967296 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(truncated_eq), Some("binop.compare_narrow_width"));
    let wrapped_eq = "fn f(a: u32, b: u32) -> i64 {\n    let s = a + b\n    if s == 0 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(wrapped_eq), Some("binop.compare_narrow_width"));
    let i32_wrapped = "fn f(a: i32, b: i32) -> i64 {\n    let s = a + b\n    if s < 0 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(i32_wrapped), Some("binop.ordered_compare_unsigned"));
    let i32_far_literal = "fn f(a: i32) -> i64 {\n    if a < 2147483648 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(
        fence(i32_far_literal),
        Some("binop.ordered_compare_unsigned")
    );
    // The call boundary: a literal outside the formal's range is refused at the CALL
    // (`f(-1)` into u32: main native 1 vs MLIR 0); an in-range literal is admitted.
    let bad_arg = "fn f(a: u32) -> i64 {\n    if a < 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(-1)\n}\n";
    assert_eq!(fence(bad_arg), Some("call.narrow_arg_out_of_range"));
    let big_arg =
        "fn f(a: u32) -> i64 {\n    return 0\n}\nfn main() -> i64 {\n    return f(4294967296)\n}\n";
    assert_eq!(fence(big_arg), Some("call.narrow_arg_out_of_range"));
    let bool_arg =
        "fn f(b: bool) -> i64 {\n    return 0\n}\nfn main() -> i64 {\n    return f(2)\n}\n";
    assert_eq!(fence(bool_arg), Some("call.narrow_arg_out_of_range"));
    let good_arg = "fn f(a: u32) -> i64 {\n    if a < 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(4294967295)\n}\n";
    assert_eq!(fence(good_arg), None);
    // `bool < bool`: MLIR compares i1 SIGNED (true is -1 there), so `false < true` is 0
    // on MLIR and 1 natively — the one refusal of a compare MLIR emits signed.
    let bool_lt = "fn f(b: bool, c: bool) -> i64 {\n    if b < c {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(bool_lt), Some("binop.ordered_compare_unsigned"));
    let bare_param = "fn f(a: u32) -> i64 {\n    if a < 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(bare_param), None, "in range on both backends");
    let i32_in_range = "fn f(a: i32) -> i64 {\n    if a < -5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(i32_in_range), None);
    let eq_in_range = "fn f(a: u32) -> i64 {\n    if a == 4294967295 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(eq_in_range), None);
    // Round 6: bitwise results are exact on both backends; `x - y == 0` on two
    // unwrapped same-kind narrows is exact too (|x - y| < 2^width); an out-of-range
    // literal in a bitwise op still diverges (`a & 4294967295` on i32 is `a & -1` there).
    for (label, src) in [
        (
            "u32 & literal == 2",
            "fn f(a: u32) -> i64 {\n    let m = a & 255\n    if m == 2 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "bool & bool == 1",
            "fn f(b: bool, c: bool) -> i64 {\n    let m = b & c\n    if m == 1 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "u32 | u32 == 7",
            "fn f(a: u32, b: u32) -> i64 {\n    let m = a | b\n    if m == 7 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "u32 - u32 == 0",
            "fn f(a: u32, b: u32) -> i64 {\n    let s = a - b\n    if s == 0 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "i32 - i32 == 0",
            "fn f(a: i32, b: i32) -> i64 {\n    let s = a - b\n    if s == 0 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
    ] {
        assert_eq!(fence(src), None, "{label}");
    }
    let diff_eq_five = "fn f(a: u32, b: u32) -> i64 {\n    let s = a - b\n    if s == 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(
        fence(diff_eq_five),
        Some("binop.compare_narrow_width"),
        "only == 0 is exact"
    );
    let i32_and_far = "fn f(a: i32) -> i64 {\n    let m = a & 4294967295\n    return m\n}\nfn main() -> i64 {\n    let r = f(-1)\n    if r == -1 {\n        return 1\n    }\n    return 0\n}\n";
    assert_eq!(fence(i32_and_far), Some("narrow.wrapped_value_escapes"));
    // u8/u16 are masked at 8/16 bits by the Rust lowering (invisible to the native
    // front end): `a + b == 0` at 255 + 1 is main native 0 vs MLIR 1.
    let u8_wrapped = "fn f(a: u8, b: u8) -> i64 {\n    let s = a + b\n    if s == 0 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(u8_wrapped), Some("binop.compare_narrow_width"));
    let u8_arg = "fn f(a: u8) -> i64 {\n    if a < 100 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(300)\n}\n";
    assert_eq!(fence(u8_arg), Some("call.narrow_arg_out_of_range"));
}

/// A WRAPPED narrow value may not leave the narrow world: MLIR carries the wrapped
/// value on, native the 64-bit one (`u32 + u32` returned as i64: native 0 vs MLIR 1 at
/// 2^32-1 + 1; `bool + bool` returned: native 2 vs MLIR 0 — round 6).
#[test]
fn wrapped_narrow_values_may_not_escape() {
    for (label, src) in [
        (
            "u32 sum returned",
            "fn f(a: u32, b: u32) -> i64 {\n    let s = a + b\n    return s\n}\nfn main() -> i64 {\n    let r = f(4294967295, 1)\n    if r == 0 {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "bool sum returned",
            "fn f(b: bool, c: bool) -> i64 {\n    let s = b + c\n    return s\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "bool sum returned as bool",
            "fn g(b: bool, c: bool) -> bool {\n    return b + c\n}\nfn main() -> i64 {\n    let x = g(true, true)\n    if x == 0 {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "u32 sum into an i64 if-join",
            "fn f(a: u32, b: u32, c: i64) -> i64 {\n    let s = a + b\n    let x = 5\n    if c == 1 {\n        x = s\n    }\n    if x == 0 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(4294967295, 1, 1)\n}\n",
        ),
        (
            "u32 sum passed to a call",
            "fn h(x: i64) -> i64 {\n    return x\n}\nfn f(a: u32, b: u32) -> i64 {\n    let s = a + b\n    return h(s)\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "u32 sum widened by an i64",
            "fn f(a: u32, b: u32, c: i64) -> i64 {\n    let s = a + b\n    let t = s + c\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
    ] {
        assert_eq!(fence(src), Some("narrow.wrapped_value_escapes"), "{label}");
    }
    let bool_cond = "fn f(b: bool, c: bool) -> i64 {\n    let s = b + c\n    if s {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(
        fence(bool_cond),
        Some("narrow.wrapped_value_escapes"),
        "wrapped bool condition"
    );
    // Native truncates a `-> u32` / `-> i32` return, so a wrapped value of that exact
    // kind returned through it is exact (measured round 6); `-> bool` is not truncated.
    let ret_u32 =
        "fn g(a: u32, b: u32) -> u32 {\n    return a + b\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(ret_u32), None, "wrapped u32 through -> u32");
    let ret_i32 =
        "fn g(a: i32) -> i32 {\n    return a * 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(ret_i32), None, "wrapped i32 through -> i32");
    let ret_bool = "fn g(b: bool, c: bool) -> bool {\n    return b + c\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(
        fence(ret_bool),
        Some("narrow.wrapped_value_escapes"),
        "-> bool is not truncated natively"
    );
    // An UNWRAPPED narrow value escapes freely (a bare param returned, a mask returned):
    // native truncates `-> u32` returns, and a u32 param fits an i64.
    for (label, src) in [
        (
            "u32 param returned",
            "fn f(a: u32) -> i64 {\n    return a\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "mask returned",
            "fn f(a: u32) -> i64 {\n    let m = a & 255\n    return m\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "u32 param passed to a call",
            "fn h(x: i64) -> i64 {\n    return x\n}\nfn f(a: u32) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
    ] {
        assert_eq!(fence(src), None, "{label}");
    }
}

/// Division stays on the WIDE set: a loop-carried accumulator of a `u64` is refused
/// for `/` even though its compare is admitted (main refused every native division).
#[test]
fn pipeline_division_uses_the_wide_taint() {
    let after_loop = "fn f(a: u64) -> i64 {\n    let s = 0\n    let i = 0\n    while i < 3 {\n        s = s + a\n        i = i + 1\n    }\n    return s / 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(after_loop), Some("binop.div_mod_unsigned"));
    let merge_div = "fn f(a: u64, c: i64) -> i64 {\n    let m = 1\n    if c == 1 {\n        m = a\n    }\n    return 10 / m\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(merge_div), Some("binop.div_mod_unsigned"));
    // A mask clears bit 63, so `(a & 255) / 2` is admitted; subtracting afterwards makes
    // the value negative again and MLIR divides it `divui` (audit 2026-09-16: native 0 vs
    // MLIR 255 for `((a & 255) - 256) / 2` while the mask exemption leaked into `/`).
    let masked_div = "fn f(a: u64) -> i64 {\n    let m = a & 255\n    return m / 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(masked_div), None);
    let resigned_div = "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let t = m - 256\n    return t / 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(resigned_div), Some("binop.div_mod_unsigned"));
    let resigned_mod = "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let t = m - 256\n    return t % 7\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(resigned_mod), Some("binop.div_mod_unsigned"));
    // A loop init id names the header argument, which holds the POST-body value from the
    // second iteration on (audit 2026-09-16: native 0 vs MLIR divui 255 for both).
    let loop_masked_div = "fn f(a: u64) -> i64 {\n    let m = a & 255\n    let i = 0\n    let r = 0\n    while i < 2 {\n        r = m / 2\n        m = m - 256\n        i = i + 1\n    }\n    return r\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(loop_masked_div), Some("binop.div_mod_unsigned"));
    // Every operand must be non-negative: MLIR goes `divui` when ANY operand is u64-kinded
    // (audit 2026-09-16: `(a & 255) / -2` native -127 vs MLIR 0, main refused).
    let masked_by_neg = "fn f(a: u64) -> i64 {\n    let m = a & 255\n    return m / -2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(masked_by_neg), Some("binop.div_mod_unsigned"));
    let masked_mod_param = "fn f(a: u64, b: i64) -> i64 {\n    let m = a & 255\n    return m % b\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(masked_mod_param), Some("binop.div_mod_unsigned"));
    let param_by_masked = "fn f(a: u64, b: i64) -> i64 {\n    let m = a & 255\n    return b / m\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(param_by_masked), Some("binop.div_mod_unsigned"));
    let masked_by_masked = "fn f(a: u64, b: u64) -> i64 {\n    let m = a & 255\n    let n = b & 15\n    return m / n\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(masked_by_masked), None);
    // u32 meets a LITERAL in i32 width, unsigned, truncated (`a / -2` native -127 vs MLIR
    // divui 0, main refused); a non-literal i64 divisor widens signed and agrees.
    let u32_by_neg =
        "fn f(a: u32) -> i64 {\n    return a / -2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(u32_by_neg), Some("binop.div_mod_unsigned"));
    let u32_mod_neg =
        "fn f(a: u32) -> i64 {\n    return a % -3\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(u32_mod_neg), Some("binop.div_mod_unsigned"));
    let u32_by_param =
        "fn f(a: u32, b: i64) -> i64 {\n    return a / b\n}\nfn main() -> i64 {\n    return 0\n}\n";
    // MLIR truncates the u32 ARGUMENT at the call site (`f(-1)` arrives as 4294967295)
    // where native passes it raw, and computes u32 arithmetic at 32 bits: every narrow
    // operand of `/` `%` is refused (audit 2026-09-18; main refused every native division).
    assert_eq!(fence(u32_by_param), Some("binop.div_mod_unsigned"));
    let u32_by_pos =
        "fn f(a: u32) -> i64 {\n    return a / 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(u32_by_pos), Some("binop.div_mod_unsigned"));
    let u8_by_neg =
        "fn f(a: u8) -> i64 {\n    return a / -2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(
        fence(u8_by_neg),
        Some("binop.div_mod_unsigned"),
        "u8 is masked to 8 bits by the Rust lowering (round 6)"
    );
    // 32-bit wrap on MLIR, 64-bit on native (audit 2026-09-18, main refused all five).
    for (label, src) in [
        (
            "u32 sum",
            "fn f(a: u32, b: u32) -> i64 {\n    let s = a + b\n    let q = s / 2\n    if q == 0 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "i32 sum",
            "fn f(a: i32, b: i32) -> i64 {\n    let s = a + b\n    return s / 3\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "u32 minus literal",
            "fn f(a: u32) -> i64 {\n    let s = a - 1\n    return s % 1000\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
    ] {
        assert_eq!(fence(src), Some("binop.div_mod_unsigned"), "{label}");
    }
    let bool_div =
        "fn f(b: bool) -> i64 {\n    return b / 1\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(
        fence(bool_div),
        Some("binop.div_mod_unsigned"),
        "bool is 1-bit in MLIR"
    );
    let loop_const_div = "fn f(a: u64) -> i64 {\n    let c = 255\n    let i = 0\n    let r = 0\n    while i < 2 {\n        r = (a & c) / 2\n        c = -1\n        i = i + 1\n    }\n    return r\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(loop_const_div), Some("binop.div_mod_unsigned"));
}

/// The literal band the frozen native ELF mis-encodes (measured: `9223372036854775808`
/// became `movabs 0x7fffffffffffff00`) is refused, at both edges; the values just
/// outside it, and `i64::MIN` COMPUTED at run time, stay admitted.
#[test]
fn misencoded_literal_band_is_refused() {
    let prog = |lit: &str| {
        format!(
            "fn g(a: u64) -> i64 {{\n    return 0\n}}\nfn main() -> i64 {{\n    return g({lit})\n}}\n"
        )
    };
    for lit in ["9223372036854775808", "9223372036854776062"] {
        assert_eq!(
            fence(&prog(lit)),
            Some("const.i64_min_band_literal"),
            "{lit}"
        );
    }
    for lit in [
        "9223372036854775807",
        "9223372036854776063",
        "18446744073709551615",
    ] {
        assert_eq!(fence(&prog(lit)), None, "{lit}");
    }
    let computed = "fn g(a: i64) -> i64 {\n    return 0\n}\nfn main() -> i64 {\n    return g(0 - 9223372036854775807 - 1)\n}\n";
    assert_eq!(
        fence(computed),
        None,
        "a run-time i64::MIN is encoded correctly natively"
    );
}
