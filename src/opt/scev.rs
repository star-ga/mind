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

//! Scalar-evolution recognition for the Salov loop-collapse pass (S1).
//!
//! Recognises the single affine-accumulator shape
//!
//! ```text
//! for i in LO..HI { acc = acc + (A*i + B) }
//! ```
//!
//! where `A`/`B` are loop-invariant (ivar-free) and `LO`/`HI` are pure,
//! duplicable bound expressions. The recognised loop is described by
//! [`AffineSum`], which the collapse pass ([`crate::opt::collapse`]) rewrites to
//! the closed form
//!
//! ```text
//! acc = acc + ( A * Σi + B * n )
//! ```
//!
//! with `n = HI - LO`, `Σi = Σ_{i=LO}^{HI-1} i`.
//!
//! ## The `/2` ring hazard — solved by halving the even factor
//! `Σi = n·p/2` where `p = LO + (HI-1)`. In `Z/2^64` you cannot naively divide
//! a product by 2: `(n·p mod 2^64)/2 ≠ (n·p/2) mod 2^64` in general (e.g.
//! `n·p = 2^64 + 2`). Exactly one of `n`, `p` is even (their parities differ
//! because `n + p = 2·HI - 1` is odd), so we use the sign-robust identity
//!
//! ```text
//! Σi = (n / 2) · p + (n % 2) · (p / 2)         // trunc-toward-zero div/rem
//! ```
//!
//! which equals `n·p/2` as an EXACT integer for every signed `n, p` (the two
//! truncations compensate — verified for n<0, n>0, even/odd in the unit
//! tests), and therefore equals `(n·p/2) mod 2^64` under wrapping ops. No
//! float, ring-exact, byte-identical across substrates.
//!
//! Trip-count contract: `closed_form_i64` returns 0 for a reversed/empty range
//! (`HI <= LO`), and the symbolic emitter in `opt::collapse` guards the update
//! with `if LO < HI`, so both paths match the zero-iteration loop exactly. The
//! formula is ring-exact for every FEASIBLE range — `n = HI-LO` in `[1,
//! i64::MAX]` with `p = LO+(HI-1)` non-overflowing; `opt::collapse` rejects a
//! const span that overflows (E2201) and documents the symbolic >2^63 boundary.

use crate::ast::BinOp;
use crate::ast::Literal;
use crate::ast::Node;
use crate::ast::Span;

/// A recognised affine-accumulator loop.
#[derive(Debug, Clone)]
pub struct AffineSum {
    /// Name of the accumulator local updated as `acc = acc + (A*i + B)`.
    pub acc: String,
    /// Coefficient `A` (ivar-free, pure). `A*i`.
    pub a: Node,
    /// Constant term `B` (ivar-free, pure).
    pub b: Node,
    /// Lower bound `LO` (inclusive, pure, duplicable).
    pub lo: Node,
    /// Upper bound `HI` (exclusive, pure, duplicable).
    pub hi: Node,
    /// Span of the originating `for` loop (for diagnostics).
    pub span: Span,
}

/// Ring-exact closed form for `Σ_{i=lo}^{hi-1} (a*i + b)` in `Z/2^64`.
///
/// THIS IS THE SINGLE SOURCE OF TRUTH for the collapse formula: the comptime
/// const-fold path calls it, and the emitted symbolic AST (`build_closed_form`
/// in `crate::opt::collapse`) mirrors its exact op structure, so the two are
/// equal by construction (gate #3). Valid for a forward range `lo <= hi`.
#[inline]
pub fn closed_form_i64(a: i64, b: i64, lo: i64, hi: i64) -> i64 {
    // Trip count MUST equal MIND's `for i in lo..hi` iteration count: a SIGNED
    // reversed/empty range (`hi <= lo`) runs zero iterations, so the sum is 0.
    // Using a raw wrapping `hi - lo` here would be WRONG for every reversed
    // range (huge bogus n -> garbage instead of 0).
    if hi <= lo {
        return 0;
    }
    let n = hi.wrapping_sub(lo);
    let p = lo.wrapping_add(hi.wrapping_sub(1));
    // Σi = (n/2)*p + (n%2)*(p/2), ring-exact by halving the EVEN factor first
    // (exactly one of n, p is even). Signed trunc-toward-zero div/rem; the two
    // truncations compensate for both signs. Exact for every feasible trip
    // count n in [1, 2^63); for n >= 2^63 (a non-terminating loop that cannot
    // run) the signed-div high bit is off by 2^63 — see the module note /
    // `build_symbolic_sum` deferred marker.
    let sum_i = n
        .wrapping_div(2)
        .wrapping_mul(p)
        .wrapping_add(n.wrapping_rem(2).wrapping_mul(p.wrapping_div(2)));
    a.wrapping_mul(sum_i).wrapping_add(b.wrapping_mul(n))
}

/// A recognised geometric-powering loop (Slice S2).
///
/// ```text
/// for i in LO..HI { acc = acc * R }
/// ```
///
/// where `R` is loop-invariant (ivar-free AND acc-free — see `recognize_geometric`)
/// and `LO`/`HI` are pure, duplicable bounds. The collapse pass rewrites it to the
/// closed form `acc = acc * R^n` (`n = HI - LO`, the trip count), computed by a
/// fixed 64-step square-and-multiply ladder that is EXACT in `Z/2^64` (multiplication
/// is associative/commutative in the ring, so `R^n` by square-and-multiply equals
/// `R` multiplied `n` times — byte-identical to the un-annotated loop).
#[derive(Debug, Clone)]
pub struct GeometricPow {
    /// Name of the accumulator local updated as `acc = acc * R`.
    pub acc: String,
    /// Loop-invariant multiplier `R` (ivar-free, acc-free, pure).
    pub r: Node,
    /// Lower bound `LO` (inclusive, pure, duplicable).
    pub lo: Node,
    /// Upper bound `HI` (exclusive, pure, duplicable).
    pub hi: Node,
    /// Span of the originating `for` loop (for diagnostics).
    pub span: Span,
}

/// Ring-exact `R^n mod 2^64` via a FIXED 64-step square-and-multiply ladder,
/// where `n` is the trip count of `for i in lo..hi` (0 for a reversed/empty
/// range, so the factor is `R^0 = 1` and `acc` is unchanged).
///
/// THIS IS THE SINGLE SOURCE OF TRUTH for the S2 collapse factor: the comptime
/// const-fold path calls it, and the emitted symbolic ladder (`build_symbolic_pow`
/// in `crate::opt::collapse`) mirrors its exact op structure (branchless masked
/// multiply `pow * (1 + bit*(base-1))`, `%2` / `/2`), so the two are equal by
/// construction. Constant iteration count (64) means no
/// data-dependent branch count → deterministic and constant-time.
#[inline]
pub fn geometric_pow_i64(r: i64, lo: i64, hi: i64) -> i64 {
    // Trip count MUST equal MIND's `for i in lo..hi` iteration count: a reversed
    // or empty range (`hi <= lo`) runs zero iterations, so the factor is R^0 = 1.
    if hi <= lo {
        return 1;
    }
    let n = hi.wrapping_sub(lo); // caller (opt::collapse) rejects a const span that overflows.
    pow_wrap(r, n)
}

/// `R^n mod 2^64` for `n >= 1` via the fixed 64-step ladder. Uses the SAME
/// branchless masked-multiply and `%2`/`/2` op structure as the emitted AST, so
/// the const-fold and the symbolic ladder produce byte-identical values.
fn pow_wrap(r: i64, n: i64) -> i64 {
    let mut result: i64 = 1;
    let mut base: i64 = r;
    let mut e: i64 = n; // n >= 1, and `e/2` keeps it non-negative through all 64 steps.
    let mut k: i64 = 0;
    while k < 64 {
        // bit = e % 2 (0 or 1 since e >= 0). result *= (bit==1 ? base : 1),
        // written branchlessly as result * (1 + bit*(base-1)).
        let bit = e.wrapping_rem(2);
        result = result.wrapping_mul(1i64.wrapping_add(bit.wrapping_mul(base.wrapping_sub(1))));
        base = base.wrapping_mul(base);
        e = e.wrapping_div(2);
        k += 1;
    }
    result
}

/// Reject reason for a `#[collapse]`-annotated loop that is not a provable
/// affine sum. Carried into the `E2201` diagnostic verbatim.
pub type Reject = &'static str;

/// Attempt to recognise `for var in start..end { body }` as an [`AffineSum`].
pub fn recognize_for(
    var: &str,
    start: &Node,
    end: &Node,
    body: &[Node],
    span: Span,
) -> Result<AffineSum, Reject> {
    // Bounds must be pure and safe to duplicate (the closed form references
    // them several times); an impure or opaque bound could change behaviour or
    // be evaluated a different number of times.
    if !is_pure_dupable(start) || !is_pure_dupable(end) {
        return Err("loop bounds are not a pure, duplicable expression");
    }
    // The body must be exactly one statement: `acc = acc + <affine-in-i>`.
    if body.len() != 1 {
        return Err("loop body is not a single accumulator assignment");
    }
    let (acc, value) = match &body[0] {
        Node::Assign { name, value, .. } => (name.clone(), value.as_ref()),
        _ => return Err("loop body is not an assignment"),
    };
    // value must be `acc + <affine>` or `<affine> + acc`.
    let affine = match value {
        Node::Binary {
            op: BinOp::Add,
            left,
            right,
            ..
        } => {
            if is_ident(left, &acc) {
                right.as_ref()
            } else if is_ident(right, &acc) {
                left.as_ref()
            } else {
                return Err("accumulator update is not `acc = acc + <expr>`");
            }
        }
        _ => return Err("accumulator update is not `acc = acc + <expr>`"),
    };
    // The added expression must be affine in the induction variable with
    // ivar-free, pure coefficient A and constant B.
    let (a, b) = match affine_of(affine, var) {
        Some(pair) => pair,
        None => return Err("added expression is not affine in the loop variable"),
    };
    if !is_pure_dupable(&a) || !is_pure_dupable(&b) {
        return Err("affine coefficients are not pure expressions");
    }
    // The accumulator is written every iteration, so a coefficient that
    // references it is NOT loop-invariant: the loop carries `acc` through
    // A/B (e.g. `s = s + s*i`, a product recurrence — not an affine sum). The
    // closed form would read `acc` once and silently miscompile. Reject.
    if references_ident(&a, &acc) || references_ident(&b, &acc) {
        return Err(
            "affine coefficient references the accumulator (loop-carried, not a closed-form affine sum)",
        );
    }
    // The bounds must be invariant across the loop. A bound that references the
    // accumulator (`for i in 0..acc { acc = acc + i }`) or the induction
    // variable makes the trip count depend on mutated state — its
    // capture-once-vs-re-evaluate semantics and the collapsed snapshot can
    // diverge — so it is not a provable closed-form affine sum. Reject.
    if references_ident(start, &acc)
        || references_ident(end, &acc)
        || references_ident(start, var)
        || references_ident(end, var)
    {
        return Err(
            "loop bound references the accumulator or induction variable (not loop-invariant)",
        );
    }
    Ok(AffineSum {
        acc,
        a,
        b,
        lo: start.clone(),
        hi: end.clone(),
        span,
    })
}

/// Attempt to recognise `for var in start..end { acc = acc * R }` as a
/// [`GeometricPow`] (Slice S2). Mirrors [`recognize_for`]'s validation exactly.
pub fn recognize_geometric(
    var: &str,
    start: &Node,
    end: &Node,
    body: &[Node],
    span: Span,
) -> Result<GeometricPow, Reject> {
    if !is_pure_dupable(start) || !is_pure_dupable(end) {
        return Err("loop bounds are not a pure, duplicable expression");
    }
    if body.len() != 1 {
        return Err("loop body is not a single accumulator assignment");
    }
    let (acc, value) = match &body[0] {
        Node::Assign { name, value, .. } => (name.clone(), value.as_ref()),
        _ => return Err("loop body is not an assignment"),
    };
    // value must be `acc * R` or `R * acc`.
    let r = match value {
        Node::Binary {
            op: BinOp::Mul,
            left,
            right,
            ..
        } => {
            if is_ident(left, &acc) {
                right.as_ref()
            } else if is_ident(right, &acc) {
                left.as_ref()
            } else {
                return Err("accumulator update is not `acc = acc * R`");
            }
        }
        _ => return Err("accumulator update is not `acc = acc * R`"),
    };
    if !is_pure_dupable(r) {
        return Err("multiplier R is not a pure expression");
    }
    // R must be a CONSTANT ratio across the loop. `acc = acc * acc` (R references
    // the accumulator) is a double-exponential recurrence (acc^(2^n)), NOT a
    // constant-ratio geometric — collapsing it to acc*R^n would silently
    // miscompile. `acc = acc * i` (R references the ivar) is a factorial-like
    // varying ratio, also not geometric. Reject both.
    if references_ident(r, &acc) {
        return Err(
            "multiplier references the accumulator (double-exponential recurrence, not a constant-ratio geometric)",
        );
    }
    if references_ident(r, var) {
        return Err("multiplier references the induction variable (not a constant ratio)");
    }
    // Bounds must be loop-invariant (same reasoning as the affine recogniser).
    if references_ident(start, &acc)
        || references_ident(end, &acc)
        || references_ident(start, var)
        || references_ident(end, var)
    {
        return Err(
            "loop bound references the accumulator or induction variable (not loop-invariant)",
        );
    }
    Ok(GeometricPow {
        acc,
        r: r.clone(),
        lo: start.clone(),
        hi: end.clone(),
        span,
    })
}

fn is_ident(node: &Node, name: &str) -> bool {
    matches!(node, Node::Lit(Literal::Ident(x), _) if x == name)
}

/// Does `node` reference the identifier `name` anywhere? Complete over the
/// pure-dupable expression shapes (`is_pure_dupable`): literals, idents, parens,
/// unary neg, and binary arithmetic.
fn references_ident(node: &Node, name: &str) -> bool {
    match node {
        Node::Lit(Literal::Ident(x), _) => x == name,
        Node::Lit(_, _) => false,
        Node::Paren(inner, _) => references_ident(inner, name),
        Node::Neg { operand, .. } => references_ident(operand, name),
        Node::Binary { left, right, .. } => {
            references_ident(left, name) || references_ident(right, name)
        }
        _ => false,
    }
}

/// A pure, duplicable expression: literals, identifiers, parens, unary neg, and
/// arithmetic over the same. Rejects calls / field / index / method access —
/// anything that could carry a side effect or be non-idempotent to duplicate.
fn is_pure_dupable(node: &Node) -> bool {
    match node {
        Node::Lit(Literal::Int(_), _) | Node::Lit(Literal::Ident(_), _) => true,
        Node::Paren(inner, _) => is_pure_dupable(inner),
        Node::Neg { operand, .. } => is_pure_dupable(operand),
        Node::Binary {
            op, left, right, ..
        } => {
            matches!(
                op,
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
            ) && is_pure_dupable(left)
                && is_pure_dupable(right)
        }
        _ => false,
    }
}

/// Extract `(A, B)` such that `node ≡ A*ivar + B`, with `A`, `B` ivar-free
/// Node expressions. Returns `None` if `node` is non-affine in `ivar`
/// (e.g. `i*i`).
fn affine_of(node: &Node, ivar: &str) -> Option<(Node, Node)> {
    let sp = node.span();
    match node {
        Node::Lit(Literal::Int(v), _) => Some((int(0, sp), int(*v, sp))),
        Node::Lit(Literal::Ident(x), _) => {
            if x == ivar {
                Some((int(1, sp), int(0, sp)))
            } else {
                // Loop-invariant symbol: coefficient 0, constant = the symbol.
                Some((int(0, sp), node.clone()))
            }
        }
        Node::Paren(inner, _) => affine_of(inner, ivar),
        Node::Neg { operand, .. } => {
            let (a, b) = affine_of(operand, ivar)?;
            Some((neg(a, sp), neg(b, sp)))
        }
        Node::Binary {
            op: BinOp::Add,
            left,
            right,
            ..
        } => {
            let (al, bl) = affine_of(left, ivar)?;
            let (ar, br) = affine_of(right, ivar)?;
            Some((bin(BinOp::Add, al, ar, sp), bin(BinOp::Add, bl, br, sp)))
        }
        Node::Binary {
            op: BinOp::Sub,
            left,
            right,
            ..
        } => {
            let (al, bl) = affine_of(left, ivar)?;
            let (ar, br) = affine_of(right, ivar)?;
            Some((bin(BinOp::Sub, al, ar, sp), bin(BinOp::Sub, bl, br, sp)))
        }
        Node::Binary {
            op: BinOp::Mul,
            left,
            right,
            ..
        } => {
            let (al, bl) = affine_of(left, ivar)?;
            let (ar, br) = affine_of(right, ivar)?;
            // affine * affine is affine only when one side is ivar-free
            // (its ivar-coefficient folds to a literal 0).
            let l_invariant = super::comptime::eval_const_i64(&al) == Some(0);
            let r_invariant = super::comptime::eval_const_i64(&ar) == Some(0);
            if l_invariant {
                // left ≡ bl (invariant); result = bl*(ar*i + br).
                Some((
                    bin(BinOp::Mul, bl.clone(), ar, sp),
                    bin(BinOp::Mul, bl, br, sp),
                ))
            } else if r_invariant {
                // right ≡ br (invariant); result = br*(al*i + bl).
                Some((
                    bin(BinOp::Mul, br.clone(), al, sp),
                    bin(BinOp::Mul, br, bl, sp),
                ))
            } else {
                // Both depend on i (e.g. i*i) — non-affine.
                None
            }
        }
        _ => None,
    }
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
fn neg(operand: Node, sp: Span) -> Node {
    Node::Neg {
        operand: Box::new(operand),
        span: sp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Ring-exactness of the closed form (the /2 hazard) ----------------

    /// i128 oracle: exact Σ(a*i+b) with no wrap, then reduced mod 2^64.
    fn oracle_i128(a: i64, b: i64, lo: i64, hi: i64) -> i64 {
        let mut s: i128 = 0;
        // Iterate in i128 exactly; only used for feasibly-small ranges.
        let mut i = lo as i128;
        while i < hi as i128 {
            s += (a as i128) * i + (b as i128);
            i += 1;
        }
        s as i64 // truncate low 64 bits == wrapping semantics
    }

    /// Closed-form oracle in i128 (no per-term loop) for the WRAP-BOUNDARY
    /// witnesses where an actual loop would be infeasible. Exact for the bounds
    /// used below (|hi| <= 2^30, |a|,|b| <= i64 range keep the product < 2^127).
    fn oracle_i128_closed(a: i64, b: i64, lo: i64, hi: i64) -> i64 {
        let n = (hi as i128) - (lo as i128);
        let sum_i = (n * ((lo as i128) + (hi as i128) - 1)) / 2; // exact: product even
        ((a as i128) * sum_i + (b as i128) * n) as i64
    }

    #[test]
    fn closed_form_matches_small_loop_oracle() {
        // Cross-check against a real wrapping loop for feasibly-small ranges,
        // including odd/even counts, negative coefficients and negative lo.
        let cases = [
            (1i64, 0i64, 0i64, 10i64),
            (1, 0, 0, 11),
            (3, 5, 2, 9),
            (-2, 7, -4, 6),
            (1, 0, -5, 5),
            (7, -3, 0, 1),
            (0, 4, 0, 100),
            (5, 0, 0, 0), // empty range -> 0
        ];
        for (a, b, lo, hi) in cases {
            assert_eq!(
                closed_form_i64(a, b, lo, hi),
                oracle_i128(a, b, lo, hi),
                "loop-oracle mismatch for a={a} b={b} lo={lo} hi={hi}"
            );
        }
    }

    #[test]
    fn closed_form_is_ring_exact_at_wrap_boundary() {
        // Witnesses chosen so the true (unbounded) sum FAR exceeds i64::MAX, so
        // the result genuinely wraps — proving the /2 halve-even trick is
        // ring-exact, not merely correct for small n. `hi` kept <= 2^30 so the
        // i128 closed-form oracle stays exact.
        let big = 1i64 << 30;
        let cases = [
            // huge coefficient -> a*Σi overflows within a few terms
            (i64::MAX / 2, 0, 0, 10),
            (i64::MAX, 1, 0, 7),
            (i64::MIN, 3, 0, 8), // n even
            (i64::MIN, 3, 0, 9), // n odd -> exercises (n%2)*(p/2)
            // large n so Σi itself overflows i64 (Σi ~ 2^59) -> tests /2 path
            (1, 0, 0, big),
            (1, 0, 0, big + 1),   // n odd
            (2, 0, 1, big),       // lo != 0, n odd
            (-1, 0, 0, big),      // negative coefficient
            (3, -7, -(big), big), // symmetric-ish range, large n
            (i64::MAX, i64::MIN, 0, big),
        ];
        for (a, b, lo, hi) in cases {
            assert_eq!(
                closed_form_i64(a, b, lo, hi),
                oracle_i128_closed(a, b, lo, hi),
                "WRAP-BOUNDARY mismatch for a={a} b={b} lo={lo} hi={hi}"
            );
        }
    }

    // ---- Recognition ------------------------------------------------------

    fn ident(s: &str) -> Node {
        Node::Lit(Literal::Ident(s.into()), Span::new(0, 0))
    }
    fn lit(v: i64) -> Node {
        Node::Lit(Literal::Int(v), Span::new(0, 0))
    }
    fn add(l: Node, r: Node) -> Node {
        bin(BinOp::Add, l, r, Span::new(0, 0))
    }
    fn mul(l: Node, r: Node) -> Node {
        bin(BinOp::Mul, l, r, Span::new(0, 0))
    }
    fn assign(name: &str, value: Node) -> Node {
        Node::Assign {
            name: name.into(),
            value: Box::new(value),
            span: Span::new(0, 0),
        }
    }

    // ---- S2 geometric powering ladder (ring-exactness) -------------------

    /// i128 oracle: R multiplied `n` times, reduced mod 2^64 (== the
    /// un-annotated loop's wrapping semantics). `n` kept small enough to loop.
    fn pow_oracle_i128(r: i64, n: i64) -> i64 {
        let mut acc: i128 = 1;
        let modulus: i128 = 1i128 << 64;
        let mut k = 0i64;
        while k < n {
            acc = (acc * (r as i128)).rem_euclid(modulus);
            k += 1;
        }
        // rem_euclid keeps acc in [0, 2^64); reinterpret the low 64 bits as i64.
        acc as u64 as i64
    }

    #[test]
    fn geometric_pow_matches_repeated_multiply_oracle() {
        // Includes n=0 (empty), n=1, n=64 (== ladder width), n=65 (past width),
        // negative R, R=0, R=1, R near i64::MAX, and a larger n so the product
        // genuinely wraps (proving the ladder is ring-exact, not just small-n).
        let cases = [
            (2i64, 0i64),
            (2, 1),
            (2, 64),
            (2, 65),
            (2, 200),
            (-3, 7),
            (-3, 64),
            (0, 5),
            (0, 0),
            (1, 1_000_000),
            (i64::MAX, 3),
            (i64::MAX - 1, 5),
            (i64::MIN, 4),
            (7, 130), // > 2*width
            (-1, 63),
            (-1, 64),
        ];
        for (r, n) in cases {
            // geometric_pow_i64 with lo=0, hi=n exercises the trip-count path.
            assert_eq!(
                geometric_pow_i64(r, 0, n),
                pow_oracle_i128(r, n.max(0)),
                "ladder mismatch for r={r} n={n}"
            );
        }
    }

    #[test]
    fn geometric_pow_reversed_range_is_identity() {
        // hi <= lo -> 0 iterations -> factor R^0 = 1 (acc unchanged).
        assert_eq!(geometric_pow_i64(7, 10, 3), 1);
        assert_eq!(geometric_pow_i64(7, 5, 5), 1);
        assert_eq!(geometric_pow_i64(-9, 0, -4), 1);
    }

    #[test]
    fn recognizes_geometric_pow() {
        // for i in 0..n { acc = acc * r }
        let body = vec![assign("acc", mul(ident("acc"), ident("r")))];
        let g = recognize_geometric("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).unwrap();
        assert_eq!(g.acc, "acc");
        assert!(is_ident(&g.r, "r"));
        // Commuted form `r * acc` is recognised too.
        let body2 = vec![assign("acc", mul(ident("r"), ident("acc")))];
        assert!(recognize_geometric("i", &lit(0), &ident("n"), &body2, Span::new(0, 0)).is_ok());
    }

    #[test]
    fn rejects_multiplier_referencing_accumulator() {
        // for i in 0..n { acc = acc * acc } — double-exponential, NOT geometric.
        let body = vec![assign("acc", mul(ident("acc"), ident("acc")))];
        assert!(recognize_geometric("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).is_err());
    }

    #[test]
    fn rejects_multiplier_referencing_ivar() {
        // for i in 0..n { acc = acc * i } — factorial-like varying ratio.
        let body = vec![assign("acc", mul(ident("acc"), ident("i")))];
        assert!(recognize_geometric("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).is_err());
    }

    #[test]
    fn recognizes_simple_sum() {
        // for i in 0..n { s = s + i }
        let body = vec![assign("s", add(ident("s"), ident("i")))];
        let r = recognize_for("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).unwrap();
        assert_eq!(r.acc, "s");
        assert_eq!(super::super::comptime::eval_const_i64(&r.a), Some(1));
        assert_eq!(super::super::comptime::eval_const_i64(&r.b), Some(0));
    }

    #[test]
    fn recognizes_affine_with_coeffs() {
        // for i in 0..n { s = s + (3*i + 5) }
        let body = vec![assign(
            "s",
            add(ident("s"), add(mul(lit(3), ident("i")), lit(5))),
        )];
        let r = recognize_for("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).unwrap();
        assert_eq!(super::super::comptime::eval_const_i64(&r.a), Some(3));
        assert_eq!(super::super::comptime::eval_const_i64(&r.b), Some(5));
    }

    #[test]
    fn rejects_quadratic() {
        // for i in 0..n { s = s + i*i }  -> non-affine
        let body = vec![assign("s", add(ident("s"), mul(ident("i"), ident("i"))))];
        assert!(recognize_for("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).is_err());
    }

    #[test]
    fn rejects_non_accumulator() {
        // for i in 0..n { s = i }
        let body = vec![assign("s", ident("i"))];
        assert!(recognize_for("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).is_err());
    }

    #[test]
    fn rejects_accumulator_in_coefficient_a() {
        // for i in 0..n { s = s + s*i } — A = s (the accumulator). This is a
        // product recurrence, NOT an affine sum; recognising it would silently
        // miscompile (loop = 120, naive closed form = 11 for n=5, s0=1).
        let body = vec![assign("s", add(ident("s"), mul(ident("s"), ident("i"))))];
        assert!(recognize_for("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).is_err());
    }

    #[test]
    fn rejects_accumulator_in_coefficient_b() {
        // for i in 0..n { s = s + (i + s) } — B = s (the accumulator).
        let body = vec![assign("s", add(ident("s"), add(ident("i"), ident("s"))))];
        assert!(recognize_for("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).is_err());
    }

    #[test]
    fn rejects_bound_referencing_accumulator() {
        // for i in 0..s { s = s + i } — HI = s (the accumulator). Trip count
        // depends on mutated state; not a provable closed-form affine sum.
        let body = vec![assign("s", add(ident("s"), ident("i")))];
        assert!(recognize_for("i", &lit(0), &ident("s"), &body, Span::new(0, 0)).is_err());
    }

    #[test]
    fn rejects_multi_statement_body() {
        let body = vec![
            assign("s", add(ident("s"), ident("i"))),
            assign("t", ident("i")),
        ];
        assert!(recognize_for("i", &lit(0), &ident("n"), &body, Span::new(0, 0)).is_err());
    }
}
