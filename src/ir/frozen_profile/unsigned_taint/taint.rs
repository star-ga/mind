// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! The per-body unsigned taint and its USE-SITE rules (`Taint::any_cmp` / `any_div`).
//! Split from `unsigned_taint.rs` for the module-size budget.

#[cfg(feature = "std-surface")]
use super::narrow::Narrow;
use crate::ir::{BinOp, ValueId};

/// The values of ONE body that may hold a full-width unsigned value at run time, at two
/// precisions, one per refused operator family.
///
/// * `cmp` — ordered compares. EXACTLY the values the MLIR backend kinds `ScalarU64`
///   (its `ult`/`slt` choice): seeded by `u64` parameters and `u64` call results,
///   propagated through non-compare arithmetic and `if` joins (signedness absorption),
///   but NOT into a loop-carried variable (MLIR's header/exit block args keep the init
///   kind) and NOT into an array element (`[u64; N]` loads as `ScalarI64`). Refusing
///   more than this refuses programs main 5724a6ee compiles natively with the same
///   result as MLIR (audit 2026-09-16: `let s = 0 while .. { s = s + a }` then
///   `s < 100` — native == MLIR == 1 on main). Each rule is pinned against the emitted
///   MLIR by the `mlir_*_signedness` tests, so a change to MLIR's kinds turns them red.
/// * `div` — `/` and `%`. A conservative SUPERSET: also array elements and loop-carried
///   variables. Main refused every native division, so over-refusing here loses nothing
///   main had, and it keeps the newly admitted division away from values whose DECLARED
///   type is unsigned even where MLIR itself divides them signed.
///
/// It used to be one BODY-level bit — any `u64` in the signature refused every `/`, `%`
/// and ordered compare in the body — which refused `fn f(a: u64) -> i64 { let i = 0
/// while i < 3 { i = i + 1 } return i }` (3 on main, native and MLIR alike).
pub(crate) enum Taint {
    /// No signature table: every value is treated as possibly unsigned.
    #[cfg_attr(feature = "std-surface", allow(dead_code))]
    All,
    /// Without `std-surface` there is no `While`/`If` IR and no signature table, so
    /// this variant is never built there (`body_taint` returns `All`).
    #[cfg(feature = "std-surface")]
    Values {
        cmp: std::collections::BTreeSet<ValueId>,
        div: std::collections::BTreeSet<ValueId>,
        /// Values MLIR computes at 32/1 bits (see `narrow.rs`).
        narrow: Narrow,
        /// Values provably in `[0, 2^63)`: masks (`unsigned_taint/mask.rs`) and
        /// non-negative literals of this body. A masked value stays in `cmp`/`div`, so
        /// arithmetic on it is tainted again.
        nonneg: std::collections::BTreeSet<ValueId>,
        /// This body's literals (loop init ids excluded — they are header arguments).
        consts: std::collections::BTreeMap<ValueId, i64>,
    },
}

impl Taint {
    /// The taint of a scope with no unsigned value in it.
    #[cfg(feature = "std-surface")]
    pub(crate) fn empty() -> Self {
        Taint::Values {
            cmp: Default::default(),
            div: Default::default(),
            narrow: Default::default(),
            nonneg: Default::default(),
            consts: Default::default(),
        }
    }

    /// Without a signature table nothing is decidable.
    #[cfg(not(feature = "std-surface"))]
    pub(crate) fn empty() -> Self {
        Taint::All
    }

    /// May an ORDERED COMPARE over `ids` diverge between native and MLIR?
    pub(crate) fn any_cmp(&self, op: &BinOp, ids: &[ValueId]) -> bool {
        #[cfg(not(feature = "std-surface"))]
        let _ = (op, ids);
        match self {
            Taint::All => true,
            #[cfg(feature = "std-surface")]
            Taint::Values { cmp, .. } => self.diverges(op, ids, cmp),
        }
    }

    /// May an `==` / `!=` over `ids` diverge? Bit compares are sign-agnostic, but MLIR
    /// truncates a literal beside a narrow operand and wraps narrow arithmetic.
    pub(crate) fn any_eq(&self, op: &BinOp, ids: &[ValueId]) -> bool {
        #[cfg(not(feature = "std-surface"))]
        let _ = (op, ids);
        match self {
            // Main 5724a6ee admitted `==`/`!=` in every configuration; without a
            // signature table there is no narrow value to see.
            Taint::All => false,
            #[cfg(feature = "std-surface")]
            Taint::Values { narrow, consts, .. } => narrow.diverges(op, ids, consts),
        }
    }

    /// Is a literal argument of this call outside its narrow formal's range?
    pub(crate) fn narrow_arg_out_of_range(&self, callee: &str, args: &[ValueId]) -> bool {
        #[cfg(not(feature = "std-surface"))]
        let _ = (callee, args);
        match self {
            Taint::All => false,
            #[cfg(feature = "std-surface")]
            Taint::Values { narrow, consts, .. } => narrow.arg_out_of_range(callee, args, consts),
        }
    }

    /// May a `/` or `%` over `ids` diverge between native and MLIR?
    pub(crate) fn any_div(&self, op: &BinOp, ids: &[ValueId]) -> bool {
        #[cfg(not(feature = "std-surface"))]
        let _ = (op, ids);
        match self {
            Taint::All => true,
            #[cfg(feature = "std-surface")]
            Taint::Values { div, .. } => self.diverges(op, ids, div),
        }
    }

    /// Signed and unsigned `/ % < <= > >=` agree iff EVERY operand is in `[0, 2^63)`.
    /// MLIR goes unsigned when ANY operand is `u64`-kinded, so one masked operand is not
    /// enough (audit 2026-09-16: `(a & 255) / -2` native -127 vs MLIR `divui` 0). A
    /// narrow (`u32`/`i32`/`bool`) operand is decided by `narrow.rs`: MLIR computes it at
    /// 32/1 bits and truncates literals and `u32` call arguments to that width, native
    /// computes at 64 bits.
    #[cfg(feature = "std-surface")]
    fn diverges(
        &self,
        op: &BinOp,
        ids: &[ValueId],
        set: &std::collections::BTreeSet<ValueId>,
    ) -> bool {
        let Taint::Values {
            narrow,
            nonneg,
            consts,
            ..
        } = self
        else {
            return true;
        };
        let wide =
            ids.iter().any(|id| set.contains(id)) && !ids.iter().all(|id| nonneg.contains(id));
        wide || narrow.diverges(op, ids, consts)
    }
}
