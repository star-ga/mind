// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! The per-body unsigned taint and its USE-SITE rules (`Taint::any_cmp` / `any_div`).
//! Split from `unsigned_taint.rs` for the module-size budget.

use crate::ir::ValueId;

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
    #[cfg_attr(not(feature = "std-surface"), allow(dead_code))]
    Values {
        cmp: std::collections::BTreeSet<ValueId>,
        div: std::collections::BTreeSet<ValueId>,
        /// Over-approximation of MLIR's `ScalarU32` values.
        narrow: std::collections::BTreeSet<ValueId>,
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
    pub(crate) fn empty() -> Self {
        Taint::Values {
            cmp: Default::default(),
            div: Default::default(),
            narrow: Default::default(),
            nonneg: Default::default(),
            consts: Default::default(),
        }
    }

    /// May an ORDERED COMPARE over `ids` diverge between native and MLIR?
    pub(crate) fn any_cmp(&self, ids: &[ValueId]) -> bool {
        match self {
            Taint::All => true,
            Taint::Values { cmp, .. } => self.diverges(ids, cmp),
        }
    }

    /// May a `/` or `%` over `ids` diverge between native and MLIR?
    pub(crate) fn any_div(&self, ids: &[ValueId]) -> bool {
        match self {
            Taint::All => true,
            Taint::Values { div, .. } => self.diverges(ids, div),
        }
    }

    /// Signed and unsigned `/ % < <= > >=` agree iff EVERY operand is in `[0, 2^63)`.
    /// MLIR goes unsigned when ANY operand is `u64`-kinded, so one masked operand is not
    /// enough (audit 2026-09-16: `(a & 255) / -2` native -127 vs MLIR `divui` 0). A
    /// `u32`-kinded operand meets a literal in i32 width, unsigned, with the literal
    /// TRUNCATED to 32 bits; native compares the zero-extended `u32` against the exact
    /// literal as signed i64. Those agree iff the literal is in `[0, 2^32)` (`a / -2`:
    /// native -127 vs MLIR 0; `a < 4294967296`: native true, MLIR compares against 0).
    /// Beside a non-literal `i64` MLIR widens and stays signed, which agrees.
    fn diverges(&self, ids: &[ValueId], set: &std::collections::BTreeSet<ValueId>) -> bool {
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
        let bad_literal = |id: &ValueId| consts.get(id).is_some_and(|v| !(0..1 << 32).contains(v));
        let narrow_hit = ids.iter().any(|id| narrow.contains(id)) && ids.iter().any(bad_literal);
        wide || narrow_hit
    }
}
