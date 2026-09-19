// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Pipeline tests for the NARROW-width model (`unsigned_taint/narrow.rs`): wrapped and
//! truncated compares, escapes, and the audit round 7-8 shapes. Split from
//! `unsigned_taint_pipeline_tests.rs` for the module-size budget.

use super::unsigned_taint_pipeline_tests::fence;

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

/// Round 7. `-> bool` RESULTS are i64 on both backends (MLIR collapses the bool return
/// kind at the call result, native does not truncate it), so predicate arithmetic and
/// compares are exact and admitted (main compiles them). `u8`/`u16` are i64-physical on
/// MLIR with masks after arithmetic only, so a literal beside them is never truncated
/// (`a == 256` on u8 agrees) and `-> u8` / `-> u16` returns mask natively too; a wrapped
/// value is exact through any return not wider than its kind. A narrow VALUE passed to
/// a narrower or other-signedness narrow formal is changed on MLIR only (call-site
/// truncation / callee-entry mask) and is refused; `u8 + u16` is masked at 16 on MLIR.
#[test]
fn narrow_round_7_shapes() {
    for (label, src) in [
        (
            "bool results added and returned",
            "fn g(a: i64, b: i64) -> bool {\n    return a < b\n}\nfn main() -> i64 {\n    let x = g(1, 2)\n    let y = g(1, 2)\n    let s = x + y\n    return s\n}\n",
        ),
        (
            "bool results ordered-compared",
            "fn g(a: i64, b: i64) -> bool {\n    return a < b\n}\nfn main() -> i64 {\n    let x = g(2, 1)\n    let y = g(1, 2)\n    if x < y {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "bool results ==",
            "fn g(a: i64, b: i64) -> bool {\n    return a < b\n}\nfn main() -> i64 {\n    let x = g(1, 2)\n    let y = g(1, 2)\n    if x == y {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "u8 == literal 256",
            "fn f(a: u8) -> i64 {\n    if a == 256 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(0)\n}\n",
        ),
        (
            "u16 < literal 70000",
            "fn f(a: u16) -> i64 {\n    if a < 70000 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(5)\n}\n",
        ),
        (
            "u8 | 256 returned",
            "fn f(a: u8) -> i64 {\n    let m = a | 256\n    return m\n}\nfn main() -> i64 {\n    let r = f(1)\n    if r == 257 {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "u16 sum through -> u16",
            "fn g(a: u16, b: u16) -> u16 {\n    return a + b\n}\nfn main() -> i64 {\n    let s = g(65535, 2)\n    if s == 1 {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "u8 sum through -> u8",
            "fn g(a: u8, b: u8) -> u8 {\n    return a + b\n}\nfn main() -> i64 {\n    let s = g(255, 2)\n    if s == 1 {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "u32 sum through -> i32",
            "fn g(a: u32, b: u32) -> i32 {\n    return a + b\n}\nfn main() -> i64 {\n    let s = g(4294967295, 2)\n    if s == 1 {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "u8 param passed to an i64 call",
            "fn h(x: i64) -> i64 {\n    return x\n}\nfn f(a: u8) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
        (
            "u8 param into a u16 formal (wider)",
            "fn h(x: u16) -> i64 {\n    return x\n}\nfn f(a: u8) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
    ] {
        assert_eq!(fence(src), None, "{label}");
    }
    for (label, src, want) in [
        (
            "u32 value into a u8 formal",
            "fn h(x: u8) -> i64 {\n    return x\n}\nfn f(a: u32) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    let r = f(300)\n    if r == 44 {\n        return 1\n    }\n    return 0\n}\n",
            "call.narrow_arg_out_of_range",
        ),
        (
            "i32 value into a u32 formal",
            "fn h(x: u32) -> i64 {\n    if x == 4294967295 {\n        return 1\n    }\n    return 0\n}\nfn f(a: i32) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    return f(-1)\n}\n",
            "call.narrow_arg_out_of_range",
        ),
        (
            "u32 value into a bool formal",
            "fn h(x: bool) -> i64 {\n    if x == 0 {\n        return 1\n    }\n    return 0\n}\nfn f(a: u32) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    return f(2)\n}\n",
            "call.narrow_arg_out_of_range",
        ),
        (
            "u8 + u16 returned (masked at 16 on MLIR)",
            "fn f(a: u8, b: u16) -> i64 {\n    let s = a + b\n    return s\n}\nfn main() -> i64 {\n    let r = f(255, 65535)\n    if r == 254 {\n        return 1\n    }\n    return 0\n}\n",
            "narrow.wrapped_value_escapes",
        ),
        (
            "literal 300 into a u8 formal",
            "fn f(a: u8) -> i64 {\n    return a\n}\nfn main() -> i64 {\n    let r = f(300)\n    if r == 44 {\n        return 1\n    }\n    return 0\n}\n",
            "call.narrow_arg_out_of_range",
        ),
        (
            "u8 sum through -> i32 (wider)",
            "fn g(a: u8, b: u8) -> i32 {\n    return a + b\n}\nfn main() -> i64 {\n    return 0\n}\n",
            "narrow.wrapped_value_escapes",
        ),
    ] {
        assert_eq!(fence(src), Some(want), "{label}");
    }
}

/// Round 8. A WRAPPED `u8` was masked at 8 on MLIR already; promoting it into `u16`
/// mode and returning through `-> u16` cannot reconcile that with native's 64-bit value
/// (`a + b + c` with a,b: u8, c: u16 at 255+1+0: main 256 vs MLIR 0) — that mix widens
/// and the escape rule refuses. `u8 | 256` is exact everywhere on MLIR except at a
/// `u8`/`u16` FORMAL, whose entry mask MLIR applies and native does not. An `i32` formal
/// is sign-extended natively at entry, so any literal and a `u32` value are exact there
/// (only `i32` -> `u32` diverges). `b & 255` on a bool is the identity (the lowering's
/// own `-> u8` return mask), not a wrap.
#[test]
fn narrow_round_8_shapes() {
    for (label, src, want) in [
        (
            "wrapped u8 promoted to u16 returned as u16",
            "fn f(a: u8, b: u8, c: u16) -> u16 {\n    let s = a + b\n    return s + c\n}\nfn main() -> i64 {\n    let r = f(255, 1, 0)\n    if r == 256 {\n        return 1\n    }\n    if r == 0 {\n        return 2\n    }\n    return 0\n}\n",
            "narrow.wrapped_value_escapes",
        ),
        (
            "u8 | 256 into a u8 formal",
            "fn h(x: u8) -> i64 {\n    return x\n}\nfn f(a: u8) -> i64 {\n    let m = a | 256\n    return h(m)\n}\nfn main() -> i64 {\n    let r = f(1)\n    if r == 257 {\n        return 1\n    }\n    if r == 1 {\n        return 2\n    }\n    return 0\n}\n",
            "call.narrow_arg_out_of_range",
        ),
        (
            "u16 | 65536 into a u16 formal",
            "fn h(x: u16) -> i64 {\n    return x\n}\nfn f(a: u16) -> i64 {\n    let m = a | 65536\n    return h(m)\n}\nfn main() -> i64 {\n    let r = f(1)\n    if r == 65537 {\n        return 1\n    }\n    if r == 1 {\n        return 2\n    }\n    return 0\n}\n",
            "call.narrow_arg_out_of_range",
        ),
        (
            "i32 value into a u32 formal",
            "fn k(y: i64) -> i64 {\n    if y == -1 {\n        return 1\n    }\n    if y == 4294967295 {\n        return 2\n    }\n    return 0\n}\nfn h(x: u32) -> i64 {\n    return k(x)\n}\nfn f(a: i32) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    return f(-1)\n}\n",
            "call.narrow_arg_out_of_range",
        ),
        (
            "u8 sum through -> u16 (wider)",
            "fn g(a: u8, b: u8) -> u16 {\n    return a + b\n}\nfn main() -> i64 {\n    let r = g(255, 2)\n    if r == 1 {\n        return 1\n    }\n    if r == 257 {\n        return 2\n    }\n    return 0\n}\n",
            "narrow.wrapped_value_escapes",
        ),
    ] {
        assert_eq!(fence(src), Some(want), "{label}");
    }
    for (label, src) in [
        (
            "wrapped u8 plus u16 through -> u8",
            "fn f(a: u8, b: u8, c: u16) -> u8 {\n    let s = a + b\n    return s + c\n}\nfn main() -> i64 {\n    let r = f(255, 1, 0)\n    if r == 0 {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "u32 value into an i32 formal",
            "fn h(x: i32) -> i64 {\n    return x\n}\nfn f(a: u32) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    return f(5)\n}\n",
        ),
        (
            "u32 value into an i32 formal, compared",
            "fn h(x: i32) -> i64 {\n    if x < 0 {\n        return 1\n    }\n    return 0\n}\nfn f(a: u32) -> i64 {\n    return h(a)\n}\nfn main() -> i64 {\n    return f(4294967295)\n}\n",
        ),
        (
            "literal 4294967295 into an i32 formal",
            "fn k(y: i64) -> i64 {\n    if y == -1 {\n        return 1\n    }\n    if y == 4294967295 {\n        return 2\n    }\n    return 0\n}\nfn h(x: i32) -> i64 {\n    return k(x)\n}\nfn main() -> i64 {\n    return h(4294967295)\n}\n",
        ),
        (
            "bool returned as u8 (identity mask)",
            "fn f(b: bool) -> u8 {\n    return b\n}\nfn main() -> i64 {\n    let r = f(true)\n    if r == 1 {\n        return 1\n    }\n    return 0\n}\n",
        ),
        (
            "u8 | 256 returned as i64",
            "fn f(a: u8) -> i64 {\n    let m = a | 256\n    return m\n}\nfn main() -> i64 {\n    return 0\n}\n",
        ),
    ] {
        assert_eq!(fence(src), None, "{label}");
    }
}
