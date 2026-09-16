// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Signed i64 integer division for the tree-walking evaluator: TOTAL, and identical to
//! what both compiled backends emit. This evaluator is also the conformance VALUE oracle
//! (`conformance::VALUE_ORACLE_ENGINE == AstEvaluator`), so a disagreement here is a
//! wrong expected value, not just a wrong `mindc test` result.
//!
//! * `x / 0 == 0`, `x % 0 == 0` — the native emitter's branchless zero-guard
//!   (`nb_div_guarded`, pinned by `examples/mindc_mind/div_shift_cmp_edge_smoke.py`) and
//!   the MLIR `div_zero_guard` (divisor substituted to 1, result forced to 0) both yield 0
//!   rather than trapping on `#DE`. This used to be `Err(DivZero)` here.
//! * `i64::MIN / -1 == i64::MIN`, `i64::MIN % -1 == 0` — the defined two's-complement
//!   wrap both backends produce by substituting divisor 1. Rust's `/` and `%` PANIC on
//!   exactly that pair, so `wrapping_div`/`wrapping_rem` are load-bearing.
//!
//! `apply_int_op_u64` implements the same zero-divisor contract for the unsigned
//! dispatcher (#99); `opt::comptime` and `opt::fold` leave a zero divisor unfolded and fold
//! `i64::MIN / -1` to the same wrapped value. Floats are a different contract
//! (`apply_float_op` keeps `DivZero`: IEEE division has no guard in either backend).
//!
//! Narrow widths: this divides at 64 bits and `declared_width` then wraps the result to
//! the declared type, so `fn f(a: i32, b: i32) -> i32 { a / b }` gives
//! `f(i32::MIN, -1) == i32::MIN`, matching the MLIR narrow arm (measured 2026-09-16).
//!
//! Cross-tier structural check (the evaluator running the same corpus as the native gate
//! and the MLIR parity test): `tests/signed_div_mod_interpreter_corpus_run.rs`.

/// `left / right` under the total signed contract.
pub(super) fn total_div(left: i64, right: i64) -> i64 {
    if right == 0 {
        0
    } else {
        left.wrapping_div(right)
    }
}

/// `left % right` under the total signed contract.
pub(super) fn total_rem(left: i64, right: i64) -> i64 {
    if right == 0 {
        0
    } else {
        left.wrapping_rem(right)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    #[cfg(feature = "std-surface")]
    use crate::parser;

    /// Signed i64 division is TOTAL and matches both compiled backends. Unit-level
    /// check of the arithmetic helper with its own operand list (including cases
    /// the program corpus cannot express, like `MIN / 0`). The STRUCTURAL
    /// cross-tier check — the interpreter running the very corpus the native gate
    /// and the MLIR parity test run — is
    /// `tests/signed_div_mod_interpreter_corpus_run.rs`; this test is not coupled
    /// to that corpus and does not claim to be. The MLIR tier emits the same values via the
    /// `div_zero_guard` in `mlir::lowering` (divisor substituted to 1 on both
    /// `== 0` and the `INT_MIN / -1` overflow, result forced to 0 on the zero
    /// case).
    ///
    /// Mutation-sensitive by construction: restoring `Err(DivZero)` for a zero
    /// divisor, or swapping `wrapping_div` back to `/`, fails this test (the
    /// latter by panicking on the `INT_MIN / -1` pair rather than returning).
    #[test]
    fn signed_int_div_mod_are_total_and_match_the_compiled_backends() {
        // `EvalError` is not `PartialEq`, and the point of this test is that the
        // Err arm is GONE — so unwrap loudly and compare the value.
        let div = |a: i64, b: i64| {
            apply_int_op(BinOp::Div, a, b)
                .unwrap_or_else(|e| panic!("{a}/{b} must be total, got error {e:?}"))
        };
        let rem = |a: i64, b: i64| {
            apply_int_op(BinOp::Mod, a, b)
                .unwrap_or_else(|e| panic!("{a}%{b} must be total, got error {e:?}"))
        };

        // Zero divisor: a VALUE, never an error — the artifact cannot report one.
        assert_eq!(div(7, 0), 0, "7/0 == 0 (zero-guard)");
        assert_eq!(rem(7, 0), 0, "7%0 == 0 (zero-guard)");
        assert_eq!(div(0, 0), 0, "0/0 == 0");
        assert_eq!(div(i64::MIN, 0), 0, "MIN/0 == 0");

        // INT_MIN / -1: the quotient is unrepresentable. Rust's `/` PANICS here;
        // x86 `idiv` raises #DE; both backends substitute divisor 1 and so yield
        // the defined two's-complement wrap.
        assert_eq!(
            div(i64::MIN, -1),
            i64::MIN,
            "INT_MIN/-1 == INT_MIN (defined wrap, no trap)"
        );
        assert_eq!(
            rem(i64::MIN, -1),
            0,
            "INT_MIN%-1 == 0 (INT_MIN%1 after the divisor substitution)"
        );

        // Ordinary signed edges are UNCHANGED: truncation toward zero, and a
        // remainder that takes the sign of the DIVIDEND.
        assert_eq!(div(-17, 5), -3, "-17/5 == -3");
        assert_eq!(rem(-17, 5), -2, "-17%5 == -2");
        assert_eq!(rem(17, -5), 2, "17%-5 == 2");
        assert_eq!(div(-17, -5), 3, "-17/-5 == 3");
        assert_eq!(div(100, 7), 14, "100/7 == 14");
        assert_eq!(rem(100, 7), 2, "100%7 == 2");

        // The signed and unsigned dispatchers agree wherever signedness cannot
        // matter — the zero divisor is exactly such a case, and the unsigned
        // side has shipped this contract since issue #99.
        assert_eq!(
            div(7, 0),
            apply_int_op_u64(BinOp::Div, 7, 0).unwrap(),
            "signed and unsigned zero-divisor contracts must not diverge"
        );
        assert_eq!(
            rem(7, 0),
            apply_int_op_u64(BinOp::Mod, 7, 0).unwrap(),
            "signed and unsigned zero-modulus contracts must not diverge"
        );
    }

    /// The same contract reached through the WHOLE interpreter path (parse →
    /// eval), not just the arithmetic helper — `apply_int_op` being right is
    /// worth nothing if the expression path never reaches it.
    #[cfg(feature = "std-surface")]
    #[test]
    fn interpreter_end_to_end_div_by_zero_yields_zero_not_an_error() {
        let src = "pub fn f(a: i64, b: i64) -> i64 { return a / b }\n\
                       let __probe: i64 = f(7, 0)\n";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        match eval_module_value_with_env(&module, &mut env, Some(src)) {
            Ok(Value::Int(n)) => assert_eq!(n, 0, "`7 / 0` evaluates to 0, as the artifact does"),
            Ok(other) => panic!("expected Int(0), got {other:?}"),
            Err(e) => panic!(
                "`7 / 0` must not be an interpreter error (got {e:?}) — the compiled artifact returns 0, so an error here is a cross-tier divergence"
            ),
        }
    }
}
