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
use crate::diagnostics::Diagnostic;
use crate::diagnostics::Span as DiagSpan;

use super::comptime::eval_const_i64;
use super::scev::{AffineSum, closed_form_i64, recognize_for};

const COLLAPSE_ATTR: &str = "collapse";
const E_COLLAPSE: &str = "E2201";

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
            // recognize_for validates the shape; build_closed_form additionally
            // rejects a const range whose span overflows i64 (the ring-exact Σi
            // division would be inexact). Both surface as E2201 — prove-or-fail.
            let outcome = recognize_for(var, start, end, body, *span)
                .and_then(|affine| build_closed_form(&affine));
            match outcome {
                Ok(replacement) => {
                    *node = replacement;
                }
                Err(reason) => {
                    ctx.diags
                        .push(collapse_error(ctx.source, ctx.file, *span, reason));
                    // Leave the loop intact; compilation fails on the diagnostic.
                }
            }
            return;
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

fn collapse_error(source: &str, file: Option<&str>, span: Span, reason: &str) -> Diagnostic {
    let dspan = DiagSpan::from_offsets(source, span.start(), span.end(), file);
    Diagnostic::error(
        "collapse",
        E_COLLAPSE,
        format!("`#[collapse]` cannot prove a closed form: {reason}"),
    )
    .with_span(dspan)
    .with_help(
        "`#[collapse]` requires `for i in LO..HI { acc = acc + (A*i + B) }` \
         with loop-invariant A/B and pure bounds LO/HI",
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
