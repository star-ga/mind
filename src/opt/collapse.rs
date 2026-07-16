// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Salov loop-collapse pass — Slice S1 (affine sum/count → O(1) closed form).
//!
//! An AST→AST rewrite that runs BEFORE lowering, so it never touches the SSA /
//! block-arg machinery (keeping the bit-identity hot path and the keystone
//! self-host gate untouched). It fires ONLY on a range `for` loop annotated
//! `#[collapse]`:
//!
//! ```text
//! #[collapse]
//! for i in LO..HI { acc = acc + (A*i + B) }
//!   ==>  acc = acc + ( A * Σi + B * n )        // n = HI-LO, Σi = n·p/2
//! ```
//!
//! When `A`, `B`, `LO`, `HI` are all compile-time constants the whole added
//! term folds to a single `ConstI64`; otherwise the branchless ring-exact form
//! (see [`crate::opt::scev`]) is emitted. No float, ring-exact in `Z/2^64`,
//! byte-identical across substrates.
//!
//! ## Contract (`#[collapse]` = prove-or-fail)
//! An annotated loop that is NOT a provable affine sum FAILS to compile with a
//! specific `E2201` diagnostic — never a silent pass-through. An unannotated
//! affine loop is left INTACT (opt-in only in S1): collapsing every affine loop
//! unconditionally would change the emitted artifact of existing sources and
//! move the pinned cross-substrate canaries, so S1 keeps the optimization
//! strictly opt-in. The general unannotated-collapse optimization is a
//! follow-on slice.
//!
//! deferred: unannotated-affine auto-collapse (an optimization, not a
//! correctness requirement) — upgrade path: gate it behind a flag and re-bless
//! the affected canaries, since it changes emitted bytes for existing sources.

use crate::ast::BinOp;
use crate::ast::Literal;
use crate::ast::Node;
use crate::ast::Span;
use crate::ast::TypeAnn;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::Span as DiagSpan;

use super::comptime::{CtFnTable, Q16Outcome, eval_const_i64, iterate_user_fixed_point};
use super::scev::{
    AffineSum, GeometricPow, Q16Map, closed_form_i64, geometric_pow_i64, recognize_for,
    recognize_geometric, recognize_q16_map,
};

const COLLAPSE_ATTR: &str = "collapse";
/// Affine-sum reject (Slice S1).
const E_COLLAPSE: &str = "E2201";
/// Geometric-powering reject (Slice S2).
const E_COLLAPSE_GEO: &str = "E2202";
/// Q16.16 fixed-point iteration rejects (Slice S3).
/// No period-1 fixed point within fuel.
const E_Q16_NOFIX: &str = "E2210";
/// A cycle of length `k >= 2` (value depends on `N mod k`).
const E_Q16_CYCLE: &str = "E2211";
/// Divergence / overflow (rejected, never saturated).
const E_Q16_DIVERGE: &str = "E2212";
/// Fixed point depends on `N` (stabilisation step `M > N`).
const E_Q16_DEPENDS_N: &str = "E2213";
/// Map is not purely comptime-expressible (unknown call / references `i` /
/// division by zero).
const E_Q16_NONCOMPTIME: &str = "E2214";
/// Non-constant seed or bounds.
const E_Q16_NONCONST: &str = "E2215";

/// Rewrite every `#[collapse]`-annotated affine loop in `module` to its closed
/// form. Returns `E2201` diagnostics for any annotated loop that cannot be
/// proven (empty on success). Mutates `module` in place.
pub fn collapse_module(
    module: &mut crate::ast::Module,
    source: &str,
    file: Option<&str>,
) -> Vec<Diagnostic> {
    // Snapshot the module's user-defined scalar functions so the S3 Q16.16 fold
    // can run the user's REAL bodies (collapse == loop by construction). Only
    // paid when a `#[collapse]` loop is actually present — the clone is skipped
    // for every ordinary source, so the bit-identity hot path is unmoved.
    let fns = if module_has_collapse(&module.items) {
        build_fn_table(&module.items)
    } else {
        CtFnTable::new()
    };
    let mut ctx = Ctx {
        source,
        file,
        diags: Vec::new(),
        fns,
    };
    for item in &mut module.items {
        rewrite_node(item, &mut ctx);
    }
    ctx.diags
}

struct Ctx<'a> {
    source: &'a str,
    file: Option<&'a str>,
    diags: Vec<Diagnostic>,
    /// User function bodies for the S3 comptime fold (empty unless a
    /// `#[collapse]` loop is present).
    fns: CtFnTable,
}

/// Does any node in `items` contain a `#[collapse]`-annotated `for` loop?
fn module_has_collapse(items: &[Node]) -> bool {
    fn scan(node: &Node) -> bool {
        if is_collapse_for(node) {
            return true;
        }
        // Immutable mirror of `child_stmt_lists`.
        match node {
            Node::FnDef { body, .. }
            | Node::Block { stmts: body, .. }
            | Node::For { body, .. }
            | Node::ForEach { body, .. } => body.iter().any(scan),
            Node::If {
                then_branch,
                else_branch,
                ..
            } => {
                then_branch.iter().any(scan)
                    || else_branch.as_ref().is_some_and(|eb| eb.iter().any(scan))
            }
            #[cfg(feature = "std-surface")]
            Node::While { body, .. } | Node::Region { body, .. } => body.iter().any(scan),
            _ => false,
        }
    }
    items.iter().any(scan)
}

/// Snapshot every top-level `fn name(params) { body }` as `(param names, body)`
/// for the comptime evaluator.
fn build_fn_table(items: &[Node]) -> CtFnTable {
    let mut table = CtFnTable::new();
    for item in items {
        if let Node::FnDef {
            name, params, body, ..
        } = item
        {
            let pnames = params.iter().map(|p| p.name.clone()).collect();
            table.insert(name.clone(), (pnames, body.clone()));
        }
    }
    table
}

/// Recurse into a node's statement lists, collapsing annotated `for` loops.
///
/// A non-collapse node just forwards into its child statement lists (via
/// `rewrite_stmt_list`, which is index-aware so the S3 Q16.16 path can read a
/// loop's preceding siblings for the constant seed).
fn rewrite_node(node: &mut Node, ctx: &mut Ctx) {
    for list in child_stmt_lists(node) {
        rewrite_stmt_list(list, ctx);
    }
}

/// Walk a statement list left-to-right, collapsing each `#[collapse]` `for`
/// loop. Iterating by index (rather than mapping over the nodes) lets the S3
/// Q16.16 path scan the loop's preceding siblings for the accumulator's
/// compile-time seed.
fn rewrite_stmt_list(list: &mut [Node], ctx: &mut Ctx) {
    for idx in 0..list.len() {
        if is_collapse_for(&list[idx]) {
            // `try_collapse` reads `list[..idx]` (immutable) for the S3 seed and
            // `list[idx]` for the loop; on success it returns the replacement,
            // on reject it pushes a diagnostic and leaves the loop intact.
            if let Some(replacement) = try_collapse(list, idx, ctx) {
                list[idx] = replacement;
            }
        } else {
            rewrite_node(&mut list[idx], ctx);
        }
    }
}

/// Is `node` a `#[collapse]`-annotated `for` loop?
fn is_collapse_for(node: &Node) -> bool {
    matches!(node, Node::For { attrs, .. } if attrs.iter().any(|a| a.name == COLLAPSE_ATTR))
}

/// Attempt to collapse the `#[collapse]` `for` loop at `list[idx]`, trying the
/// three recognised shapes IN ORDER:
///   - S1 affine sum      `acc = acc + (A*i + B)`   -> E2201 on reject
///   - S2 geometric power  `acc = acc * R`           -> E2202 on reject
///   - S3 Q16.16 map       `x = f(x)`                -> E2210..E2215 on reject
///
/// Each is prove-or-fail: an annotated loop matching no shape (or matching but
/// unprovable) is a compile error, never a silent pass-through. Returns the
/// closed-form replacement on success, or `None` after pushing a diagnostic.
fn try_collapse(list: &[Node], idx: usize, ctx: &mut Ctx) -> Option<Node> {
    let Node::For {
        var,
        start,
        end,
        body,
        span,
        ..
    } = &list[idx]
    else {
        return None;
    };
    let span = *span;

    match recognize_for(var, start, end, body, span) {
        Ok(affine) => match build_closed_form(&affine) {
            Ok(replacement) => Some(replacement),
            Err(reason) => {
                ctx.diags.push(collapse_error(
                    ctx.source, ctx.file, span, E_COLLAPSE, reason,
                ));
                None
            }
        },
        Err(affine_reject) => {
            // Not an affine sum — try the S2 geometric-powering shape.
            match recognize_geometric(var, start, end, body, span) {
                Ok(geo) => match build_geometric(&geo) {
                    Ok(replacement) => Some(replacement),
                    Err(reason) => {
                        ctx.diags.push(collapse_error(
                            ctx.source,
                            ctx.file,
                            span,
                            E_COLLAPSE_GEO,
                            reason,
                        ));
                        None
                    }
                },
                Err(geo_reject) => {
                    // Not geometric — try the S3 Q16.16 fixed-point map.
                    match recognize_q16_map(var, start, end, body, span) {
                        Ok(map) => build_q16_collapse(&map, &list[..idx], ctx),
                        Err(q16_reject) => {
                            // No shape matched. Route to the family the body most
                            // resembles: a call-RHS body is a Q16.16 attempt
                            // (E2214), a `*` accumulation is geometric (E2202),
                            // everything else affine (E2201).
                            let (code, reason) = if body_is_q16_map_attempt(body) {
                                (E_Q16_NONCOMPTIME, q16_reject)
                            } else if body_is_mul_accumulation(body) {
                                (E_COLLAPSE_GEO, geo_reject)
                            } else {
                                (E_COLLAPSE, affine_reject)
                            };
                            ctx.diags
                                .push(collapse_error(ctx.source, ctx.file, span, code, reason));
                            None
                        }
                    }
                }
            }
        }
    }
}

/// Mutable references to every statement list a node owns.
fn child_stmt_lists(node: &mut Node) -> Vec<&mut Vec<Node>> {
    match node {
        Node::FnDef { body, .. } => vec![body],
        Node::Block { stmts, .. } => vec![stmts],
        Node::For { body, .. } | Node::ForEach { body, .. } => vec![body],
        Node::If {
            then_branch,
            else_branch,
            ..
        } => match else_branch {
            Some(eb) => vec![then_branch, eb],
            None => vec![then_branch],
        },
        #[cfg(feature = "std-surface")]
        Node::While { body, .. } => vec![body],
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => vec![body],
        _ => Vec::new(),
    }
}

/// Build the closed-form replacement for a recognised loop.
///
/// - **Fully-const** bounds+coeffs -> a single `Assign { acc = acc + ConstI64 }`.
///   `closed_form_i64` already applies the trip-count guard (`hi <= lo -> 0`),
///   so the constant is correct for reversed/empty ranges.
/// - **Symbolic** bounds -> a statement-level `if LO < HI { acc = acc + <sum> }`.
///   The guard mirrors the loop's own semantics EXACTLY: `for i in LO..HI` runs
///   iff `LO < HI`, so when `LO >= HI` the `if` is skipped and `acc` is left
///   untouched — matching the zero-iteration loop. Inside the branch `n = HI-LO`
///   is guaranteed `>= 1`, so `build_symbolic_sum`'s ring-exact form is valid
///   with no in-expression guard (which would need a deep-position value-if).
fn build_closed_form(affine: &AffineSum) -> Result<Node, super::scev::Reject> {
    let sp = affine.span;
    let acc = affine.acc.clone();

    match (
        eval_const_i64(&affine.a),
        eval_const_i64(&affine.b),
        eval_const_i64(&affine.lo),
        eval_const_i64(&affine.hi),
    ) {
        (Some(a), Some(b), Some(lo), Some(hi)) => {
            // Fully-const loop -> a single ConstI64 (guard applied inside
            // closed_form_i64, so reversed/empty ranges fold to `+ 0`).
            //
            // Feasibility: for a forward range the ring-exact Σi divides `n·p`
            // by 2, which is only exact when `n = HI-LO` and `p = LO+(HI-1)` are
            // the TRUE values (no i64 overflow). A span > i64::MAX iterates more
            // than 2^63 times (non-terminating) and the division would be
            // inexact, so we cannot prove a closed form — fail (E2201) rather
            // than fold to a wrong constant.
            if hi > lo {
                let n_ok = hi.checked_sub(lo).is_some();
                let p_ok = hi
                    .checked_sub(1)
                    .and_then(|h1| lo.checked_add(h1))
                    .is_some();
                if !n_ok || !p_ok {
                    return Err(
                        "loop bound span overflows i64 (range too large to prove an exact closed form)",
                    );
                }
            }
            Ok(assign_acc_plus(
                &acc,
                int(closed_form_i64(a, b, lo, hi), sp),
                sp,
            ))
        }
        _ => {
            // Symbolic bounds: guard the whole update behind `LO < HI` so a
            // reversed/empty range (LO >= HI) leaves acc unchanged, exactly as
            // the zero-iteration loop would. WITHOUT this guard the closed form
            // is a silent miscompile for `for i in a..b` with a > b at runtime.
            //
            // Feasibility caveat (documented, not compile-time checkable here):
            // the closed form is exact for every FEASIBLE range — trip count
            // n = HI-LO in [1, i64::MAX] with p = LO+(HI-1) non-overflowing. A
            // symbolic range spanning > 2^63 elements iterates more than 2^63
            // times (non-terminating in any real execution), so there is no
            // finite reference value to diverge from; it is out of the
            // equivalence contract, matching the E2205 >64-bit boundary.
            let update = assign_acc_plus(&acc, build_symbolic_sum(affine, sp), sp);
            Ok(Node::If {
                cond: Box::new(bin(BinOp::Lt, affine.lo.clone(), affine.hi.clone(), sp)),
                then_branch: vec![update],
                else_branch: None,
                span: sp,
            })
        }
    }
}

/// `acc = acc + <added>`.
fn assign_acc_plus(acc: &str, added: Node, sp: Span) -> Node {
    Node::Assign {
        name: acc.to_string(),
        value: Box::new(bin(
            BinOp::Add,
            Node::Lit(Literal::Ident(acc.to_string()), sp),
            added,
            sp,
        )),
        span: sp,
    }
}

/// Branchless ring-exact `A * Σi + B * n`, `Σi = (n/2)*p + (n%2)*(p/2)`,
/// `n = hi-lo`, `p = lo + (hi-1)` — the exact op structure of
/// [`crate::opt::scev::closed_form_i64`].
fn build_symbolic_sum(affine: &AffineSum, sp: Span) -> Node {
    let lo = &affine.lo;
    let hi = &affine.hi;
    let a = &affine.a;
    let b = &affine.b;

    // n = hi - lo
    let n = || bin(BinOp::Sub, hi.clone(), lo.clone(), sp);
    // p = lo + (hi - 1)
    let p = || {
        bin(
            BinOp::Add,
            lo.clone(),
            bin(BinOp::Sub, hi.clone(), int(1, sp), sp),
            sp,
        )
    };
    // Σi = (n / 2) * p + (n % 2) * (p / 2)
    let sum_i = bin(
        BinOp::Add,
        bin(BinOp::Mul, bin(BinOp::Div, n(), int(2, sp), sp), p(), sp),
        bin(
            BinOp::Mul,
            bin(BinOp::Mod, n(), int(2, sp), sp),
            bin(BinOp::Div, p(), int(2, sp), sp),
            sp,
        ),
        sp,
    );
    // A * Σi + B * n
    bin(
        BinOp::Add,
        bin(BinOp::Mul, a.clone(), sum_i, sp),
        bin(BinOp::Mul, b.clone(), n(), sp),
        sp,
    )
}

/// Build the closed-form replacement for a recognised geometric-powering loop
/// (Slice S2): `for i in LO..HI { acc = acc * R }` -> `acc = acc * R^n`,
/// `n = HI - LO` the trip count.
///
/// - **Fully-const** `R`/`LO`/`HI` -> a single `acc = acc * ConstI64`, the const
///   factor computed by [`geometric_pow_i64`] (which yields `R^0 = 1` for a
///   reversed/empty range, leaving `acc` unchanged).
/// - **Symbolic** -> a statement-level `if LO < HI { <64-step ladder>; acc = acc * pow }`.
///   The `LO < HI` guard mirrors the loop's own semantics exactly: `for i in
///   LO..HI` runs iff `LO < HI`, so `LO >= HI` skips the update and leaves `acc`
///   untouched — matching the zero-iteration loop. Inside the guard the exponent
///   `HI - LO >= 1`, so the ladder's signed `%2`/`/2` never see a negative
///   exponent (which would corrupt the masked multiply).
fn build_geometric(geo: &GeometricPow) -> Result<Node, super::scev::Reject> {
    let sp = geo.span;
    let acc = geo.acc.clone();

    match (
        eval_const_i64(&geo.r),
        eval_const_i64(&geo.lo),
        eval_const_i64(&geo.hi),
    ) {
        (Some(r), Some(lo), Some(hi)) => {
            // Feasibility: a forward range whose span `HI - LO` overflows i64
            // iterates more than 2^63 times (non-terminating); we cannot prove
            // a finite closed form, so reject (E2202) rather than fold a value
            // computed from a wrapped, wrong trip count.
            if hi > lo && hi.checked_sub(lo).is_none() {
                return Err(
                    "loop bound span overflows i64 (range too large to prove an exact closed form)",
                );
            }
            Ok(assign_acc_mul(
                &acc,
                int(geometric_pow_i64(r, lo, hi), sp),
                sp,
            ))
        }
        _ => Ok(build_symbolic_pow(geo, sp)),
    }
}

/// `acc = acc * <factor>`.
fn assign_acc_mul(acc: &str, factor: Node, sp: Span) -> Node {
    Node::Assign {
        name: acc.to_string(),
        value: Box::new(bin(
            BinOp::Mul,
            Node::Lit(Literal::Ident(acc.to_string()), sp),
            factor,
            sp,
        )),
        span: sp,
    }
}

/// The fixed 64-step square-and-multiply ladder, guarded by `LO < HI`, computing
/// `R^(HI-LO)` into a fresh local and multiplying it into `acc`. Constant
/// iteration count (no data-dependent branch count) → deterministic and
/// constant-time; the per-step multiply is BRANCHLESS
/// (`pow * (1 + bit*(base-1))`), mirroring [`geometric_pow_i64`]'s op structure
/// exactly, so the emitted code and the const-fold agree by construction. Ring-
/// exact in `Z/2^64`, no float, byte-identical across substrates.
fn build_symbolic_pow(geo: &GeometricPow, sp: Span) -> Node {
    // Fresh, span-disambiguated locals so multiple collapsed loops in one fn
    // never collide.
    let disc = sp.start();
    let base = format!("__mind_collapse_base_{disc}");
    let exp = format!("__mind_collapse_exp_{disc}");
    let pow = format!("__mind_collapse_pow_{disc}");
    let kvar = format!("__mind_collapse_k_{disc}");

    // pow = pow * (1 + (exp % 2) * (base - 1))
    let bit = bin(BinOp::Mod, ident(&exp, sp), int(2, sp), sp);
    let masked = bin(
        BinOp::Add,
        int(1, sp),
        bin(
            BinOp::Mul,
            bit,
            bin(BinOp::Sub, ident(&base, sp), int(1, sp), sp),
            sp,
        ),
        sp,
    );
    let step_pow = assign(&pow, bin(BinOp::Mul, ident(&pow, sp), masked, sp), sp);
    // base = base * base
    let step_base = assign(
        &base,
        bin(BinOp::Mul, ident(&base, sp), ident(&base, sp), sp),
        sp,
    );
    // exp = exp / 2
    let step_exp = assign(&exp, bin(BinOp::Div, ident(&exp, sp), int(2, sp), sp), sp);

    let ladder = Node::For {
        var: kvar,
        start: Box::new(int(0, sp)),
        end: Box::new(int(64, sp)),
        body: vec![step_pow, step_base, step_exp],
        attrs: Vec::new(),
        span: sp,
    };

    let stmts = vec![
        // let mut base = R
        let_i64(&base, geo.r.clone(), sp),
        // let mut exp = HI - LO   (>= 1 inside the `LO < HI` guard)
        let_i64(
            &exp,
            bin(BinOp::Sub, geo.hi.clone(), geo.lo.clone(), sp),
            sp,
        ),
        // let mut pow = 1
        let_i64(&pow, int(1, sp), sp),
        ladder,
        // acc = acc * pow
        assign_acc_mul(&geo.acc, ident(&pow, sp), sp),
    ];

    Node::If {
        cond: Box::new(bin(BinOp::Lt, geo.lo.clone(), geo.hi.clone(), sp)),
        then_branch: stmts,
        else_branch: None,
        span: sp,
    }
}

fn ident(name: &str, sp: Span) -> Node {
    Node::Lit(Literal::Ident(name.to_string()), sp)
}
fn assign(name: &str, value: Node, sp: Span) -> Node {
    Node::Assign {
        name: name.to_string(),
        value: Box::new(value),
        span: sp,
    }
}
fn let_i64(name: &str, value: Node, sp: Span) -> Node {
    Node::Let {
        name: name.to_string(),
        mutable: true,
        ann: Some(TypeAnn::ScalarI64),
        value: Box::new(value),
        span: sp,
    }
}

/// Build the closed-form replacement for a recognised Q16.16 fixed-point map
/// (Slice S3): `for i in LO..HI { x = f(x) }` -> `x = <fixed point>`.
///
/// S3 requires the whole fixed point to be determined at COMPILE TIME: `LO`/`HI`
/// must be constants (E2215 otherwise) and the accumulator's seed must be a
/// compile-time constant found in the loop's preceding siblings (E2215
/// otherwise). The map is then iterated over the user's real function bodies
/// (see `iterate_user_fixed_point`) to its bit-exact period-1 fixed point; each
/// non-fixed outcome is a specific reject:
///
/// - `Cycle` (period `k >= 2`) -> E2211 (value depends on `N`)
/// - `Diverged` (`|x| > 2^30` / overflow) -> E2212 (reject, never saturate)
/// - `DependsOnN` (no fixed point within `N` iters) -> E2213
/// - `NonComptime` (unresolved/looping call, unsupported construct, div-by-zero) -> E2214
/// - `NoFixedPoint` (fuel exhausted) -> E2210
///
/// A reversed/empty range collapses to the seed (the map is iterated zero
/// times, so `x` is unchanged — bug-class 1).
fn build_q16_collapse(map: &Q16Map, preceding: &[Node], ctx: &mut Ctx) -> Option<Node> {
    let sp = map.span;

    // Bounds must be compile-time constants (S3 determines the fixed point at
    // compile time).
    let (lo, hi) = match (eval_const_i64(&map.lo), eval_const_i64(&map.hi)) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => {
            ctx.diags.push(collapse_error(
                ctx.source,
                ctx.file,
                sp,
                E_Q16_NONCONST,
                "loop bounds are not compile-time constants \
                 (S3 requires const LO/HI to determine the fixed point at compile time)",
            ));
            return None;
        }
    };

    // The seed is the nearest preceding constant binding of the accumulator.
    let seed = match find_const_seed(preceding, &map.acc) {
        Some(v) => v,
        None => {
            ctx.diags.push(collapse_error(
                ctx.source,
                ctx.file,
                sp,
                E_Q16_NONCONST,
                "accumulator seed is not a compile-time constant \
                 (S3 requires a `let <acc> = <const>` before the loop)",
            ));
            return None;
        }
    };

    // Trip count = iterations of `for i in lo..hi` (0 for a reversed/empty
    // range). `saturating_sub` avoids an i64 overflow for a huge const span;
    // the iteration then caps at the fuel and rejects if no fixed point is
    // reached (E2210).
    let trip = if hi > lo { hi.saturating_sub(lo) } else { 0 };

    // Fold by evaluating the user's REAL function bodies (collapse == loop by
    // construction — no name-trust). `ctx.fns` holds the module's fn bodies.
    match iterate_user_fixed_point(&map.f, &map.acc, &ctx.fns, seed, trip) {
        Q16Outcome::Fixed(v) => Some(assign_const(&map.acc, v, sp)),
        Q16Outcome::Cycle(_) => {
            ctx.diags.push(collapse_error(
                ctx.source,
                ctx.file,
                sp,
                E_Q16_CYCLE,
                "map enters a cycle of length >= 2 (the value depends on N mod k, not a fixed point)",
            ));
            None
        }
        Q16Outcome::Diverged => {
            ctx.diags.push(collapse_error(
                ctx.source,
                ctx.file,
                sp,
                E_Q16_DIVERGE,
                "map diverges / overflows the Q16.16 range (rejected, never saturated)",
            ));
            None
        }
        Q16Outcome::DependsOnN => {
            ctx.diags.push(collapse_error(
                ctx.source,
                ctx.file,
                sp,
                E_Q16_DEPENDS_N,
                "map does not reach a fixed point within the N iterations (the result depends on N)",
            ));
            None
        }
        Q16Outcome::NonComptime => {
            ctx.diags.push(collapse_error(
                ctx.source,
                ctx.file,
                sp,
                E_Q16_NONCOMPTIME,
                "map is not purely comptime-evaluable \
                 (an unresolved/looping call, an unsupported construct, or a division by zero)",
            ));
            None
        }
        Q16Outcome::NoFixedPoint => {
            ctx.diags.push(collapse_error(
                ctx.source,
                ctx.file,
                sp,
                E_Q16_NOFIX,
                "map does not reach a fixed point within the iteration fuel",
            ));
            None
        }
    }
}

/// `acc = <const>` — the S3 fixed-point replacement.
fn assign_const(acc: &str, v: i64, sp: Span) -> Node {
    Node::Assign {
        name: acc.to_string(),
        value: Box::new(int(v, sp)),
        span: sp,
    }
}

/// Find the accumulator's compile-time seed: the nearest PRECEDING `let`/assign
/// of `acc` whose value is a constant. Returns `None` if no binding is found or
/// the nearest binding is non-constant (both -> E2215).
fn find_const_seed(preceding: &[Node], acc: &str) -> Option<i64> {
    for node in preceding.iter().rev() {
        match node {
            Node::Let { name, value, .. } if name == acc => return eval_const_i64(value),
            Node::Assign { name, value, .. } if name == acc => return eval_const_i64(value),
            _ => {}
        }
    }
    None
}

/// Does the loop body look like a Q16.16 map attempt (`acc = <call-expr>`)?
/// Used only to route a total-recognition failure to the S3 `E2214` diagnostic
/// (rather than the affine E2201) when the body's RHS is a function call.
fn body_is_q16_map_attempt(body: &[Node]) -> bool {
    matches!(body, [Node::Assign { value, .. }] if rhs_is_call(value))
}

/// Is `node` (after unwrapping parens / unary negation) a function call?
fn rhs_is_call(node: &Node) -> bool {
    match node {
        Node::Call { .. } => true,
        Node::Paren(inner, _) => rhs_is_call(inner),
        Node::Neg { operand, .. } => rhs_is_call(operand),
        _ => false,
    }
}

/// Does the loop body look like a `*` accumulation (`acc = <expr> * <expr>`)?
/// Used only to route a total-recognition failure to the E2202 (geometric)
/// diagnostic instead of E2201 (affine).
fn body_is_mul_accumulation(body: &[Node]) -> bool {
    matches!(
        body,
        [Node::Assign { value, .. }]
            if matches!(**value, Node::Binary { op: BinOp::Mul, .. })
    )
}

fn collapse_error(
    source: &str,
    file: Option<&str>,
    span: Span,
    code: &'static str,
    reason: &str,
) -> Diagnostic {
    let dspan = DiagSpan::from_offsets(source, span.start(), span.end(), file);
    Diagnostic::error(
        "collapse",
        code,
        format!("`#[collapse]` cannot prove a closed form: {reason}"),
    )
    .with_span(dspan)
    .with_help(
        "`#[collapse]` requires `for i in LO..HI { acc = acc + (A*i + B) }` (affine sum), \
         `for i in LO..HI { acc = acc * R }` (geometric), or `for i in LO..HI { x = f(x) }` \
         where f is a Q16.16 contraction over qmul/qadd/qsub/qdiv/cos_q16 with const bounds \
         and a const seed (fixed-point iteration)",
    )
}

fn int(v: i64, sp: Span) -> Node {
    Node::Lit(Literal::Int(v), sp)
}
fn bin(op: BinOp, l: Node, r: Node, sp: Span) -> Node {
    Node::Binary {
        op,
        left: Box::new(l),
        right: Box::new(r),
        span: sp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(s: &str) -> Node {
        Node::Lit(Literal::Ident(s.into()), Span::new(0, 0))
    }
    fn lit(v: i64) -> Node {
        Node::Lit(Literal::Int(v), Span::new(0, 0))
    }

    /// Evaluate the emitted `added` sub-expression (the RHS after `acc +`) for
    /// concrete `lo`/`hi` (as `n` bound) by substituting and const-folding.
    /// Used to prove the EMITTED symbolic AST == `closed_form_i64` (gate #3).
    fn eval_emitted(a: i64, b: i64, lo: i64, hi: i64) -> i64 {
        let affine = AffineSum {
            acc: "s".into(),
            a: lit(a),
            b: lit(b),
            lo: lit(lo),
            hi: lit(hi),
            span: Span::new(0, 0),
        };
        // Force the SYMBOLIC path even though bounds are const, then const-fold
        // the emitted tree to prove op-structure equality with closed_form_i64.
        let emitted = build_symbolic_sum(&affine, Span::new(0, 0));
        eval_const_i64(&emitted).expect("emitted symbolic form must be const-foldable")
    }

    #[test]
    fn emitted_symbolic_matches_closed_form_at_wrap_boundary() {
        let big = 1i64 << 30;
        let cases = [
            (1i64, 0i64, 0i64, 10i64),
            (3, 5, 2, 9),
            (-2, 7, -4, 6),
            (i64::MAX / 2, 0, 0, 10),
            (i64::MIN, 3, 0, 9),
            (1, 0, 0, big),
            (1, 0, 0, big + 1),
            (i64::MAX, i64::MIN, 0, big),
        ];
        for (a, b, lo, hi) in cases {
            assert_eq!(
                eval_emitted(a, b, lo, hi),
                closed_form_i64(a, b, lo, hi),
                "emitted-vs-closed_form mismatch a={a} b={b} lo={lo} hi={hi}"
            );
        }
    }

    // ---- S2 geometric powering -------------------------------------------

    #[test]
    fn geometric_const_folds_to_acc_times_pow() {
        // for i in 0..10 { acc = acc * 2 }  -> acc = acc * 1024
        let geo = GeometricPow {
            acc: "acc".into(),
            r: lit(2),
            lo: lit(0),
            hi: lit(10),
            span: Span::new(0, 0),
        };
        let node = build_geometric(&geo).expect("const geometric must collapse");
        let Node::Assign { value, .. } = node else {
            panic!("expected `acc = acc * <const>`");
        };
        let Node::Binary {
            op: BinOp::Mul,
            right,
            ..
        } = *value
        else {
            panic!("expected a `* <const>` factor");
        };
        assert_eq!(*right, lit(1024));
    }

    #[test]
    fn geometric_reversed_range_folds_to_identity() {
        // for i in 10..3 { acc = acc * 7 } -> 0 iterations -> acc = acc * 1.
        let geo = GeometricPow {
            acc: "acc".into(),
            r: lit(7),
            lo: lit(10),
            hi: lit(3),
            span: Span::new(0, 0),
        };
        let node = build_geometric(&geo).expect("empty geometric must collapse");
        let Node::Assign { value, .. } = node else {
            panic!("expected `acc = acc * 1`");
        };
        let Node::Binary { right, .. } = *value else {
            panic!("expected factor");
        };
        assert_eq!(*right, lit(1));
    }

    #[test]
    fn geometric_const_span_overflow_is_rejected() {
        // for i in i64::MIN..i64::MAX { acc = acc * 2 } — span overflows i64.
        let geo = GeometricPow {
            acc: "acc".into(),
            r: lit(2),
            lo: lit(i64::MIN),
            hi: lit(i64::MAX),
            span: Span::new(0, 0),
        };
        assert!(
            build_geometric(&geo).is_err(),
            "overflowing span must be rejected"
        );
    }

    #[test]
    fn geometric_symbolic_emits_guarded_ladder() {
        // for i in 0..n { acc = acc * r } -> `if 0 < n { <lets><for 0..64>acc=acc*pow }`.
        let geo = GeometricPow {
            acc: "acc".into(),
            r: ident("r"),
            lo: lit(0),
            hi: ident("n"),
            span: Span::new(0, 0),
        };
        let node = build_geometric(&geo).expect("symbolic geometric must collapse");
        let Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } = node
        else {
            panic!("expected a guarded `if LO < HI {{ ... }}`");
        };
        assert!(
            matches!(&*cond, Node::Binary { op: BinOp::Lt, .. }),
            "guard must be `LO < HI`"
        );
        assert!(else_branch.is_none());
        // 3 lets + the 64-step ladder for-loop + the final `acc = acc * pow`.
        assert_eq!(then_branch.len(), 5);
        let Node::For { start, end, .. } = &then_branch[3] else {
            panic!("4th stmt must be the ladder for-loop");
        };
        assert_eq!(**start, lit(0));
        assert_eq!(**end, lit(64), "ladder must be a fixed 64-step loop");
    }

    #[test]
    fn const_bounds_fold_to_single_literal() {
        // for i in 0..100 { s = s + i }  -> s = s + 4950
        let affine = AffineSum {
            acc: "s".into(),
            a: lit(1),
            b: lit(0),
            lo: lit(0),
            hi: lit(100),
            span: Span::new(0, 0),
        };
        let node = build_closed_form(&affine).expect("const range must collapse");
        if let Node::Assign { value, .. } = node {
            if let Node::Binary { right, .. } = *value {
                assert_eq!(*right, lit(4950));
                return;
            }
        }
        panic!("expected `s = s + 4950`");
    }

    #[test]
    fn const_span_overflow_is_rejected() {
        // for i in i64::MIN..i64::MAX { s = s + i } — n = HI-LO overflows i64
        // (a > 2^63-iteration, non-terminating loop). Cannot prove an exact
        // closed form -> E2201, never a wrong constant.
        let affine = AffineSum {
            acc: "s".into(),
            a: lit(1),
            b: lit(0),
            lo: lit(i64::MIN),
            hi: lit(i64::MAX),
            span: Span::new(0, 0),
        };
        assert!(
            build_closed_form(&affine).is_err(),
            "overflowing span must be rejected"
        );
    }

    #[test]
    fn symbolic_bounds_emit_guarded_branchless_form() {
        // for i in 0..n { s = s + i } -> `if 0 < n { s = s + <symbolic> }`.
        // The `LO < HI` guard mirrors the zero-iteration loop for reversed
        // ranges; without it the closed form silently miscompiles when n < 0.
        let affine = AffineSum {
            acc: "s".into(),
            a: lit(1),
            b: lit(0),
            lo: lit(0),
            hi: ident("n"),
            span: Span::new(0, 0),
        };
        let node = build_closed_form(&affine).expect("symbolic range must collapse");
        let Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } = node
        else {
            panic!("expected a guarded `if LO < HI {{ ... }}`");
        };
        // Guard must be exactly `LO < HI`.
        assert!(
            matches!(&*cond, Node::Binary { op: BinOp::Lt, .. }),
            "guard must be `LO < HI`"
        );
        assert!(else_branch.is_none(), "guard must have no else branch");
        assert_eq!(then_branch.len(), 1);
        // The single guarded stmt is `s = s + <non-const symbolic tree>`.
        let Node::Assign { value, .. } = &then_branch[0] else {
            panic!("guarded body must be an assignment");
        };
        let Node::Binary { right, .. } = &**value else {
            panic!("assignment RHS must be `s + <expr>`");
        };
        assert!(matches!(**right, Node::Binary { .. }));
        assert!(eval_const_i64(right).is_none());
    }

    // ---- S3 Q16.16 fixed-point collapse ----------------------------------
    //
    // The fold runs the USER'S real function bodies (option B), so every test
    // installs a fn table via `ctx_with_fns` — collapse == loop by construction.

    /// The canonical Q16.16 cos map bodies (== the `dottie_collapse.mind` example).
    const CANON_COS: &str = r#"
fn qmul(a: i64, b: i64) -> i64 {
    let p: i64 = a * b;
    let neg: bool = p < 0;
    let mut m: i64 = p;
    if neg { m = 0 - p; }
    let mut q: i64 = m >> 16;
    let rem: i64 = m & 65535;
    if rem > 32768 { q = q + 1; } else { if rem == 32768 { if (q & 1) == 1 { q = q + 1; } } }
    if neg { return 0 - q; }
    return q;
}
fn cos_q16(x: i64) -> i64 {
    let x2: i64 = qmul(x, x);
    let mut acc: i64 = 2;
    acc = qmul(acc, x2) - 91;
    acc = qmul(acc, x2) + 2731;
    acc = qmul(acc, x2) - 32768;
    acc = qmul(acc, x2) + 65536;
    return acc;
}
"#;

    fn ctx_with_fns(src: &str) -> Ctx<'static> {
        let module = crate::parser::parse(src).expect("parse fn table source");
        Ctx {
            source: "",
            file: None,
            diags: Vec::new(),
            fns: build_fn_table(&module.items),
        }
    }

    fn let_const(name: &str, v: i64) -> Node {
        Node::Let {
            name: name.into(),
            mutable: true,
            ann: Some(TypeAnn::ScalarI64),
            value: Box::new(lit(v)),
            span: Span::new(0, 0),
        }
    }

    /// `x = <callee>(x)` as a raw RHS map.
    fn call_map(callee: &str, lo: i64, hi: i64) -> Q16Map {
        Q16Map {
            acc: "x".into(),
            f: Node::Call {
                callee: callee.into(),
                args: vec![ident("x")],
                span: Span::new(0, 0),
            },
            lo: lit(lo),
            hi: lit(hi),
            span: Span::new(0, 0),
        }
    }

    #[test]
    fn q16_cos_collapse_folds_to_dottie_constant() {
        // `let mut x = 0; #[collapse] for i in 0..1000 { x = cos_q16(x) }`
        // -> `x = 48437` (the Q16.16 Dottie fixed point, 0x0000BD35), computed by
        // running the user's REAL cos_q16.
        let preceding = [let_const("x", 0)];
        let mut ctx = ctx_with_fns(CANON_COS);
        let node = build_q16_collapse(&call_map("cos_q16", 0, 1000), &preceding, &mut ctx)
            .expect("cos map must collapse to a constant");
        assert!(ctx.diags.is_empty());
        let Node::Assign { name, value, .. } = node else {
            panic!("expected `x = <const>`");
        };
        assert_eq!(name, "x");
        assert_eq!(*value, lit(48437));
    }

    #[test]
    fn q16_redefined_cos_never_folds_to_dottie_constant() {
        // THE hole-is-closed proof at the collapse layer: a module that redefines
        // `cos_q16(x) = x + 1` must NOT fold to 48437. x+1 has no Q16.16 fixed
        // point within N -> E2213, never the compiler's cos constant.
        let preceding = [let_const("x", 0)];
        let mut ctx = ctx_with_fns("fn cos_q16(x: i64) -> i64 { return x + 1; }");
        let folded = build_q16_collapse(&call_map("cos_q16", 0, 1000), &preceding, &mut ctx);
        assert!(folded.is_none(), "must NOT fold a non-contraction");
        assert_eq!(ctx.diags[0].code, E_Q16_DEPENDS_N);
        // And prove it is never the cos-Dottie constant.
        assert!(!matches!(folded, Some(Node::Assign { value, .. }) if *value == lit(48437)));
    }

    #[test]
    fn q16_non_const_seed_is_rejected() {
        // Seed comes from `let mut x = seed` (a non-const identifier) -> E2215.
        let preceding = [Node::Let {
            name: "x".into(),
            mutable: true,
            ann: Some(TypeAnn::ScalarI64),
            value: Box::new(ident("seed")),
            span: Span::new(0, 0),
        }];
        let mut ctx = ctx_with_fns(CANON_COS);
        assert!(build_q16_collapse(&call_map("cos_q16", 0, 1000), &preceding, &mut ctx).is_none());
        assert_eq!(ctx.diags.len(), 1);
        assert_eq!(ctx.diags[0].code, E_Q16_NONCONST);
    }

    #[test]
    fn q16_non_const_bound_is_rejected() {
        // Symbolic upper bound -> E2215.
        let preceding = [let_const("x", 0)];
        let mut map = call_map("cos_q16", 0, 0);
        map.hi = ident("n");
        let mut ctx = ctx_with_fns(CANON_COS);
        assert!(build_q16_collapse(&map, &preceding, &mut ctx).is_none());
        assert_eq!(ctx.diags[0].code, E_Q16_NONCONST);
    }

    #[test]
    fn q16_reversed_range_folds_to_seed() {
        // Reversed range -> 0 iterations -> x stays the seed (bug-class 1).
        let preceding = [let_const("x", 12345)];
        let mut ctx = ctx_with_fns(CANON_COS);
        let node = build_q16_collapse(&call_map("cos_q16", 10, 3), &preceding, &mut ctx)
            .expect("reversed range collapses to the seed");
        let Node::Assign { value, .. } = node else {
            panic!("expected `x = <seed>`");
        };
        assert_eq!(*value, lit(12345));
    }

    #[test]
    fn q16_two_cycle_map_is_rejected() {
        // `x = negate(x)` -> 2-cycle -> E2211.
        let preceding = [let_const("x", 0x1_0000)];
        let mut ctx = ctx_with_fns("fn negate(x: i64) -> i64 { return 0 - x; }");
        assert!(build_q16_collapse(&call_map("negate", 0, 1000), &preceding, &mut ctx).is_none());
        assert_eq!(ctx.diags[0].code, E_Q16_CYCLE);
    }

    #[test]
    fn q16_divergent_map_is_rejected() {
        // `x = grow(x)` (x + 0.5) diverges -> E2212 (reject, never saturate).
        let preceding = [let_const("x", 0x2000_0000)];
        let mut ctx = ctx_with_fns("fn grow(x: i64) -> i64 { return x + 536870912; }");
        let map = call_map("grow", 0, 1000);
        assert!(build_q16_collapse(&map, &preceding, &mut ctx).is_none());
        assert_eq!(ctx.diags[0].code, E_Q16_DIVERGE);
    }

    #[test]
    fn q16_unresolved_call_is_non_comptime() {
        // A call to a function absent from the module cannot be folded -> E2214.
        let preceding = [let_const("x", 0)];
        let mut ctx = ctx_with_fns("fn other(x: i64) -> i64 { return x; }");
        assert!(build_q16_collapse(&call_map("missing", 0, 1000), &preceding, &mut ctx).is_none());
        assert_eq!(ctx.diags[0].code, E_Q16_NONCOMPTIME);
    }

    #[test]
    fn q16_too_few_iters_depends_on_n_is_rejected() {
        // cos converges at ~step 30; N=5 doesn't reach it -> E2213.
        let preceding = [let_const("x", 0)];
        let mut ctx = ctx_with_fns(CANON_COS);
        assert!(build_q16_collapse(&call_map("cos_q16", 0, 5), &preceding, &mut ctx).is_none());
        assert_eq!(ctx.diags[0].code, E_Q16_DEPENDS_N);
    }
}
