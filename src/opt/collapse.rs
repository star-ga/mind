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

use super::comptime::eval_const_i64;
use super::scev::{
    AffineSum, GeometricPow, closed_form_i64, geometric_pow_i64, recognize_for, recognize_geometric,
};

const COLLAPSE_ATTR: &str = "collapse";
/// Affine-sum reject (Slice S1).
const E_COLLAPSE: &str = "E2201";
/// Geometric-powering reject (Slice S2).
const E_COLLAPSE_GEO: &str = "E2202";

/// Rewrite every `#[collapse]`-annotated affine loop in `module` to its closed
/// form. Returns `E2201` diagnostics for any annotated loop that cannot be
/// proven (empty on success). Mutates `module` in place.
pub fn collapse_module(
    module: &mut crate::ast::Module,
    source: &str,
    file: Option<&str>,
) -> Vec<Diagnostic> {
    let mut ctx = Ctx {
        source,
        file,
        diags: Vec::new(),
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
}

/// Recurse into a node's statement lists, collapsing annotated `for` loops.
fn rewrite_node(node: &mut Node, ctx: &mut Ctx) {
    // A `#[collapse]` range-for is handled here (replaced or diagnosed); every
    // other node just forwards into its child statement lists.
    if let Node::For {
        var,
        start,
        end,
        body,
        attrs,
        span,
    } = node
    {
        if attrs.iter().any(|a| a.name == COLLAPSE_ATTR) {
            // Two recognised shapes, tried in order:
            //   S1 affine sum       `acc = acc + (A*i + B)`  -> E2201 on reject
            //   S2 geometric power   `acc = acc * R`          -> E2202 on reject
            // Each is prove-or-fail: an annotated loop matching neither shape
            // (or matching but unprovable, e.g. an overflowing const span) is a
            // compile error, never a silent pass-through.
            match recognize_for(var, start, end, body, *span) {
                Ok(affine) => {
                    match build_closed_form(&affine) {
                        Ok(replacement) => *node = replacement,
                        Err(reason) => ctx.diags.push(collapse_error(
                            ctx.source, ctx.file, *span, E_COLLAPSE, reason,
                        )),
                    }
                    return;
                }
                Err(affine_reject) => {
                    // Not an affine sum — try the S2 geometric-powering shape.
                    match recognize_geometric(var, start, end, body, *span) {
                        Ok(geo) => match build_geometric(&geo) {
                            Ok(replacement) => *node = replacement,
                            Err(reason) => ctx.diags.push(collapse_error(
                                ctx.source,
                                ctx.file,
                                *span,
                                E_COLLAPSE_GEO,
                                reason,
                            )),
                        },
                        Err(geo_reject) => {
                            // Neither shape. Surface the reject for the family
                            // the body most resembles: a `*` accumulation is a
                            // geometric attempt (E2202), everything else affine
                            // (E2201). Keeps `acc = acc * acc` a specific E2202.
                            if body_is_mul_accumulation(body) {
                                ctx.diags.push(collapse_error(
                                    ctx.source,
                                    ctx.file,
                                    *span,
                                    E_COLLAPSE_GEO,
                                    geo_reject,
                                ));
                            } else {
                                ctx.diags.push(collapse_error(
                                    ctx.source,
                                    ctx.file,
                                    *span,
                                    E_COLLAPSE,
                                    affine_reject,
                                ));
                            }
                        }
                    }
                    return;
                }
            }
        }
    }
    for list in child_stmt_lists(node) {
        for child in list.iter_mut() {
            rewrite_node(child, ctx);
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
        "`#[collapse]` requires `for i in LO..HI { acc = acc + (A*i + B) }` (affine sum) \
         or `for i in LO..HI { acc = acc * R }` (geometric) with loop-invariant \
         coefficients/multiplier and pure bounds LO/HI",
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
}
