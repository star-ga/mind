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

/// Division stays on the WIDE set: a loop-carried accumulator of a `u64` is refused
/// for `/` even though its compare is admitted (main refused every native division).
#[test]
fn pipeline_division_uses_the_wide_taint() {
    let after_loop = "fn f(a: u64) -> i64 {\n    let s = 0\n    let i = 0\n    while i < 3 {\n        s = s + a\n        i = i + 1\n    }\n    return s / 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(after_loop), Some("binop.div_mod_unsigned"));
    let merge_div = "fn f(a: u64, c: i64) -> i64 {\n    let m = 1\n    if c == 1 {\n        m = a\n    }\n    return 10 / m\n}\nfn main() -> i64 {\n    return 0\n}\n";
    assert_eq!(fence(merge_div), Some("binop.div_mod_unsigned"));
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
