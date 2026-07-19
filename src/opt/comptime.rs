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

//! Minimal compile-time i64 const evaluator for the Salov loop-collapse pass
//! (`opt::collapse`, S1).
//!
//! This is deliberately tiny and self-contained: the collapse pass needs its
//! OWN const-bound discovery (`opt::fold` is leaf-only and not pipeline-wired),
//! so this module folds an integer subtree to an `i64` iff every leaf is an
//! integer literal reachable through `+ - * / %`, unary negation, or parens.
//! A non-integer leaf (an identifier, a call, a float) makes the whole subtree
//! non-const and the fold returns `None` — the caller then emits the symbolic
//! branchless closed form instead.
//!
//! ## Ring semantics (the load-bearing invariant)
//! Arithmetic is EXACT in `Z/2^64` using `wrapping_*`, matching the language's
//! defined-wrap i64 semantics (== the emitted MLIR artifact). There is NO
//! saturation and NO float, ever, so a const-folded bound is byte-identical to
//! the value the same expression would produce at runtime.
//!
//! `Div`/`Mod` by a zero literal returns `None` (the loop is left intact / the
//! collapse refuses) rather than trapping at compile time.

use crate::ast::BinOp;
use crate::ast::Literal;
use crate::ast::Node;

/// Fold `node` to an `i64` iff it is a fully-constant integer expression under
/// wrapping (`Z/2^64`) semantics. Returns `None` for any non-constant or
/// division-by-zero subtree.
pub fn eval_const_i64(node: &Node) -> Option<i64> {
    match node {
        Node::Lit(Literal::Int(v), _) => Some(*v),
        Node::Paren(inner, _) => eval_const_i64(inner),
        Node::Neg { operand, .. } => Some(0i64.wrapping_sub(eval_const_i64(operand)?)),
        Node::Binary {
            op, left, right, ..
        } => {
            let a = eval_const_i64(left)?;
            let b = eval_const_i64(right)?;
            match op {
                BinOp::Add => Some(a.wrapping_add(b)),
                BinOp::Sub => Some(a.wrapping_sub(b)),
                BinOp::Mul => Some(a.wrapping_mul(b)),
                // `wrapping_div`/`wrapping_rem` are exact for every operand pair
                // except division by zero (returns None) and i64::MIN / -1 (whose
                // wrapping result is i64::MIN — the defined-wrap answer). Both
                // mirror `arith.divsi`/`arith.remsi` at runtime.
                BinOp::Div => {
                    if b == 0 {
                        None
                    } else {
                        Some(a.wrapping_div(b))
                    }
                }
                BinOp::Mod => {
                    if b == 0 {
                        None
                    } else {
                        Some(a.wrapping_rem(b))
                    }
                }
                // Comparisons/other ops are not part of an affine bound.
                _ => None,
            }
        }
        _ => None,
    }
}

// ===========================================================================
// Q16.16 fixed-point comptime evaluation (Slice S3).
// ===========================================================================
//
// The marquee loop-collapse slice iterates a contractive Q16.16 map `x = f(x)`
// AT COMPILE TIME to its bit-exact period-1 fixed point, then replaces the loop
// with that constant. Q16.16 is a signed 32-bit value (stored in an `i64`) with
// 16 fractional bits.
//
// SOUNDNESS — collapse == loop BY CONSTRUCTION (no name-trust). The map is
// folded by evaluating the USER'S ACTUAL function bodies (`cos_q16`, `qmul`, …
// resolved from the module) with the SAME defined i64 semantics the shipped
// backend uses (wrapping add/sub/mul, trunc-toward-zero div/rem == `arith`), so
// the compile-time orbit is bit-identical to the runtime orbit. A program that
// redefines `cos_q16` to something else folds to THAT function's behaviour (and
// rejects if it has no fixed point) — never a compiler-assumed constant.

use crate::ast::BitOp;
use crate::ast::LogicalOp;
use std::collections::BTreeMap;

/// Representable range (signed 32-bit) — a valid Q16.16 seed.
const Q16_MAX: i64 = 0x7FFF_FFFF;
const Q16_MIN: i64 = -0x8000_0000;
/// Divergence guard: `|x| > this` ⇒ the orbit is escaping the Q16.16 contraction
/// range, reject (never saturate — saturation would mask divergence).
const Q16_DIVERGE: i64 = 0x4000_0000;
/// Comptime iteration fuel (max steps evaluated regardless of trip count).
const Q16_FUEL: i64 = 65536;
/// Comptime call-recursion depth guard (a runaway/mutually-recursive user map
/// is rejected as non-comptime rather than blowing the stack).
const CT_MAX_DEPTH: usize = 256;

/// The module's user-defined scalar functions available to the comptime
/// evaluator: `name -> (param names, body statements)`. Built by `opt::collapse`
/// (only when a `#[collapse]` loop is present) so the fold can run the user's
/// real map — the key to collapse == loop by construction.
pub type CtFnTable = BTreeMap<String, (Vec<String>, Vec<Node>)>;

/// A subtree the comptime evaluator cannot reduce to an `i64` ⇒ the map is not
/// purely comptime-evaluable and the loop is left intact (E2214).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CtErr {
    /// An unsupported construct, an unresolved variable/function, or a recursion
    /// past `CT_MAX_DEPTH`.
    NotComptime,
    /// Division / remainder by zero.
    DivByZero,
}

/// Control-flow result of evaluating a function body: a value that fell out of
/// the last statement, or a value produced by an early `return`.
enum CtFlow {
    Fell(i64),
    Returned(i64),
}

#[inline]
fn q_in_range(v: i64) -> bool {
    (Q16_MIN..=Q16_MAX).contains(&v)
}

/// Evaluate a pure scalar expression to an `i64` using the SAME defined i64
/// semantics as the shipped backend (wrapping arithmetic, trunc-toward-zero
/// div/rem, arithmetic shift). `env` maps in-scope locals/params to values;
/// `fns` resolves user function calls. Any unsupported node ⇒ `NotComptime`.
fn eval_ct_expr(
    node: &Node,
    env: &BTreeMap<String, i64>,
    fns: &CtFnTable,
    depth: usize,
) -> Result<i64, CtErr> {
    match node {
        Node::Lit(Literal::Int(v), _) => Ok(*v),
        Node::Lit(Literal::Ident(name), _) => env.get(name).copied().ok_or(CtErr::NotComptime),
        Node::Paren(inner, _) => eval_ct_expr(inner, env, fns, depth),
        Node::Neg { operand, .. } => Ok(0i64.wrapping_sub(eval_ct_expr(operand, env, fns, depth)?)),
        Node::Binary {
            op, left, right, ..
        } => {
            let a = eval_ct_expr(left, env, fns, depth)?;
            let b = eval_ct_expr(right, env, fns, depth)?;
            match op {
                BinOp::Add => Ok(a.wrapping_add(b)),
                BinOp::Sub => Ok(a.wrapping_sub(b)),
                BinOp::Mul => Ok(a.wrapping_mul(b)),
                BinOp::Div => {
                    if b == 0 {
                        Err(CtErr::DivByZero)
                    } else {
                        Ok(a.wrapping_div(b))
                    }
                }
                BinOp::Mod => {
                    if b == 0 {
                        Err(CtErr::DivByZero)
                    } else {
                        Ok(a.wrapping_rem(b))
                    }
                }
                BinOp::Lt => Ok((a < b) as i64),
                BinOp::Le => Ok((a <= b) as i64),
                BinOp::Gt => Ok((a > b) as i64),
                BinOp::Ge => Ok((a >= b) as i64),
                BinOp::Eq => Ok((a == b) as i64),
                BinOp::Ne => Ok((a != b) as i64),
            }
        }
        Node::Bitwise {
            op, left, right, ..
        } => {
            let a = eval_ct_expr(left, env, fns, depth)?;
            let b = eval_ct_expr(right, env, fns, depth)?;
            match op {
                BitOp::Or => Ok(a | b),
                BitOp::And => Ok(a & b),
                BitOp::Xor => Ok(a ^ b),
                // Shift amount out of `[0, 63]` is not defined by the scalar
                // backend; reject rather than guess. Signed `>>` is arithmetic.
                BitOp::Shl => {
                    if (0..64).contains(&b) {
                        Ok(a.wrapping_shl(b as u32))
                    } else {
                        Err(CtErr::NotComptime)
                    }
                }
                BitOp::Shr => {
                    if (0..64).contains(&b) {
                        Ok(a.wrapping_shr(b as u32))
                    } else {
                        Err(CtErr::NotComptime)
                    }
                }
            }
        }
        Node::Logical {
            op, left, right, ..
        } => {
            let a = eval_ct_expr(left, env, fns, depth)? != 0;
            let b = eval_ct_expr(right, env, fns, depth)? != 0;
            let r = match op {
                LogicalOp::And => a && b,
                LogicalOp::Or => a || b,
            };
            Ok(r as i64)
        }
        Node::Call { callee, args, .. } => {
            if depth >= CT_MAX_DEPTH {
                return Err(CtErr::NotComptime);
            }
            let (params, body) = fns.get(callee).ok_or(CtErr::NotComptime)?;
            if params.len() != args.len() {
                return Err(CtErr::NotComptime);
            }
            // Args are evaluated in the CALLER's env; params bind in a FRESH env.
            let mut call_env: BTreeMap<String, i64> = BTreeMap::new();
            for (p, a) in params.iter().zip(args.iter()) {
                let v = eval_ct_expr(a, env, fns, depth + 1)?;
                call_env.insert(p.clone(), v);
            }
            match eval_ct_body(body, &mut call_env, fns, depth + 1)? {
                CtFlow::Fell(v) | CtFlow::Returned(v) => Ok(v),
            }
        }
        _ => Err(CtErr::NotComptime),
    }
}

/// Evaluate a straight-line-plus-`if` function/branch body. `env` is threaded
/// in place so `let`/assignments (including those inside an `if`) are visible to
/// later statements. An early `return` (directly or bubbled up from an `if`)
/// short-circuits as `CtFlow::Returned`; otherwise the last statement's value
/// is `CtFlow::Fell`. A loop or any other unsupported statement ⇒ `NotComptime`.
fn eval_ct_body(
    body: &[Node],
    env: &mut BTreeMap<String, i64>,
    fns: &CtFnTable,
    depth: usize,
) -> Result<CtFlow, CtErr> {
    let mut last = 0i64;
    for stmt in body {
        match stmt {
            Node::Let { name, value, .. } | Node::Assign { name, value, .. } => {
                let v = eval_ct_expr(value, env, fns, depth)?;
                env.insert(name.clone(), v);
                last = v;
            }
            Node::Return { value, .. } => {
                let v = match value {
                    Some(e) => eval_ct_expr(e, env, fns, depth)?,
                    None => 0,
                };
                return Ok(CtFlow::Returned(v));
            }
            Node::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                let c = eval_ct_expr(cond, env, fns, depth)?;
                let branch: &[Node] = if c != 0 {
                    then_branch
                } else {
                    match else_branch {
                        Some(eb) => eb,
                        None => &[],
                    }
                };
                match eval_ct_body(branch, env, fns, depth)? {
                    CtFlow::Returned(v) => return Ok(CtFlow::Returned(v)),
                    CtFlow::Fell(v) => last = v,
                }
            }
            other => {
                // A bare expression statement (e.g. a trailing `acc`) is the
                // body's value; anything else (a loop, a tensor op, …) is not
                // comptime-evaluable here.
                last = eval_ct_expr(other, env, fns, depth)?;
            }
        }
    }
    Ok(CtFlow::Fell(last))
}

/// Outcome of iterating a Q16.16 map to a fixed point (Slice S3). Each non-
/// `Fixed` variant maps to a specific `E22xx` reject in `opt::collapse`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q16Outcome {
    /// Reached a bit-exact period-1 fixed point (`f(x) == x`); collapse to this.
    Fixed(i64),
    /// Exhausted fuel (`N > FUEL`) without any fixed point → `E2210`.
    NoFixedPoint,
    /// Entered a cycle of length `k >= 2` (value depends on `N mod k`) → `E2211`.
    Cycle(i64),
    /// Overflow / divergence (`|x| > 2^30`) at some step → `E2212`.
    Diverged,
    /// Did all `N` iterations without stabilising → the value depends on `N`
    /// (stabilisation step `M > N`) → `E2213`.
    DependsOnN,
    /// The map is not purely comptime-evaluable (unresolved call / unsupported
    /// construct / division by zero) → `E2214`.
    NonComptime,
}

/// Iterate `x = f(x)` from `seed` for `min(trip, FUEL)` steps, evaluating the
/// map `rhs` (in terms of the accumulator `acc`) by running the user's real
/// function bodies in `fns` — so the fixed point is bit-identical to the loop.
///
/// - period-1 (`f(x) == x`) at step `M <= trip` → [`Q16Outcome::Fixed`] (the
///   loop result after `N = trip` iters is that fixed point, N-independent).
/// - a revisited value (cycle `k >= 2`) → [`Q16Outcome::Cycle`].
/// - `|x| > 2^30` → [`Q16Outcome::Diverged`] (reject, never saturate).
/// - a non-comptime subtree / division by zero → [`Q16Outcome::NonComptime`].
/// - all `trip` steps done with no fixed point (`trip <= FUEL`) →
///   [`Q16Outcome::DependsOnN`]; fuel exhausted (`trip > FUEL`) →
///   [`Q16Outcome::NoFixedPoint`].
///
/// `trip <= 0` (reversed/empty range) yields `Fixed(seed)` — the map is iterated
/// zero times, so `x` is unchanged.
pub fn iterate_user_fixed_point(
    rhs: &Node,
    acc: &str,
    fns: &CtFnTable,
    seed: i64,
    trip: i64,
) -> Q16Outcome {
    if !q_in_range(seed) {
        return Q16Outcome::Diverged;
    }
    if trip <= 0 {
        return Q16Outcome::Fixed(seed);
    }
    let steps = trip.min(Q16_FUEL);
    let mut x = seed;
    // A BTreeSet (not HashMap/HashSet) keeps the analysis free of any hasher-seed
    // nondeterminism; the outcome is deterministic regardless, but this is the
    // defensively-clean choice for the bit-identity hot path.
    let mut seen: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    seen.insert(x);
    let mut j = 0i64;
    while j < steps {
        let mut env: BTreeMap<String, i64> = BTreeMap::new();
        env.insert(acc.to_string(), x);
        let y = match eval_ct_expr(rhs, &env, fns, 0) {
            Ok(v) => v,
            Err(_) => return Q16Outcome::NonComptime,
        };
        if y.abs() > Q16_DIVERGE {
            return Q16Outcome::Diverged;
        }
        if y == x {
            // period-1: M = j + 1 <= steps <= trip, so M <= N. The loop result
            // after N iters is this fixed point.
            return Q16Outcome::Fixed(x);
        }
        if seen.contains(&y) {
            return Q16Outcome::Cycle(y);
        }
        seen.insert(y);
        x = y;
        j += 1;
    }
    if trip <= Q16_FUEL {
        // Did all N real iterations, never period-1 ⇒ M would exceed N ⇒ the
        // value depends on N. Reject rather than fold an N-dependent constant.
        Q16Outcome::DependsOnN
    } else {
        Q16Outcome::NoFixedPoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;

    fn int(v: i64) -> Node {
        Node::Lit(Literal::Int(v), Span::new(0, 0))
    }
    fn bin(op: BinOp, l: Node, r: Node) -> Node {
        Node::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
            span: Span::new(0, 0),
        }
    }

    #[test]
    fn folds_wrapping_arithmetic() {
        // (2 + 3) * 4 = 20
        let e = bin(BinOp::Mul, bin(BinOp::Add, int(2), int(3)), int(4));
        assert_eq!(eval_const_i64(&e), Some(20));
    }

    #[test]
    fn wraps_at_i64_boundary() {
        // i64::MAX + 1 wraps to i64::MIN (defined-wrap, not None).
        let e = bin(BinOp::Add, int(i64::MAX), int(1));
        assert_eq!(eval_const_i64(&e), Some(i64::MIN));
    }

    #[test]
    fn ident_is_not_const() {
        let e = Node::Lit(Literal::Ident("n".into()), Span::new(0, 0));
        assert_eq!(eval_const_i64(&e), None);
    }

    #[test]
    fn div_by_zero_is_none() {
        let e = bin(BinOp::Div, int(10), int(0));
        assert_eq!(eval_const_i64(&e), None);
    }

    // ---- S3 comptime evaluation over the USER'S real function bodies -----

    /// Build a comptime fn table from `.mind` source (mirrors what
    /// `opt::collapse` does — the fold runs the user's ACTUAL bodies).
    fn fn_table(src: &str) -> CtFnTable {
        let module = crate::parser::parse(src).expect("parse");
        let mut table = CtFnTable::new();
        for item in &module.items {
            if let Node::FnDef(fd, _) = item {
                let pnames = fd.params.iter().map(|p| p.name.clone()).collect();
                table.insert(fd.name.clone(), (pnames, fd.body.clone()));
            }
        }
        table
    }

    /// The canonical Q16.16 cos map (magnitude-based RNE qmul + degree-8 Taylor)
    /// — the same bodies the `dottie_collapse.mind` example ships.
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

    /// `x = f(x)` where `f` is the single call `callee(x)`.
    fn call_x(callee: &str) -> Node {
        Node::Call {
            callee: callee.into(),
            args: vec![Node::Lit(Literal::Ident("x".into()), Span::new(0, 0))],
            span: Span::new(0, 0),
        }
    }

    #[test]
    fn qmul_rne_is_bit_exact_including_negative_trap() {
        // Evaluate the USER'S qmul body directly (the RNE, magnitude-based pin).
        let fns = fn_table(CANON_COS);
        let call = |a: i64, b: i64| {
            let node = Node::Call {
                callee: "qmul".into(),
                args: vec![int(a), int(b)],
                span: Span::new(0, 0),
            };
            eval_ct_expr(&node, &BTreeMap::new(), &fns, 0).unwrap()
        };
        // 0.5 * 0.5 = 0.25 exact; 1.0 * v = v.
        assert_eq!(call(0x8000, 0x8000), 0x4000);
        assert_eq!(call(0x1_0000, 0x1234), 0x1234);
        // Positive half ties round to even.
        assert_eq!(call(1, 0x8000), 0); // q=0 even -> stays
        assert_eq!(call(3, 0x8000), 2); // q=1 odd -> up to 2
        // NEGATIVE half tie: magnitude-based RNE gives -2, NOT the
        // (p+0x8000)>>16 arithmetic-shift trap value of -1.
        assert_eq!(call(-3, 0x8000), -2);
        assert_ne!(call(-3, 0x8000), -1);
        assert_eq!(call(-1, 0x8000), 0);
    }

    #[test]
    fn cos_orbit_reaches_dottie_fixed_point() {
        // THE marquee: iterate x = cos_q16(x) from seed 0 for 1000 steps, running
        // the user's real cos_q16 -> the Q16.16 Dottie fixed point 0x0000BD35.
        let fns = fn_table(CANON_COS);
        let outcome = iterate_user_fixed_point(&call_x("cos_q16"), "x", &fns, 0, 1000);
        assert_eq!(outcome, Q16Outcome::Fixed(0x0000_BD35));
        assert_eq!(outcome, Q16Outcome::Fixed(48437));
    }

    #[test]
    fn cos_with_too_few_iters_depends_on_n() {
        // Converges at ~step 30; N = 5 does not reach it -> depends on N (E2213).
        let fns = fn_table(CANON_COS);
        assert_eq!(
            iterate_user_fixed_point(&call_x("cos_q16"), "x", &fns, 0, 5),
            Q16Outcome::DependsOnN
        );
    }

    #[test]
    fn redefined_cos_never_folds_to_the_dottie_constant() {
        // THE hole-is-closed proof: a program that redefines `cos_q16(x) = x + 1`
        // must fold to THAT behaviour (no fixed point -> reject), NEVER 48437.
        let fns = fn_table("fn cos_q16(x: i64) -> i64 { return x + 1; }");
        let outcome = iterate_user_fixed_point(&call_x("cos_q16"), "x", &fns, 0, 1000);
        assert_ne!(outcome, Q16Outcome::Fixed(48437));
        assert_eq!(outcome, Q16Outcome::DependsOnN); // x+1 has no Q16.16 fixed point
    }

    #[test]
    fn user_map_with_real_fixed_point_folds() {
        // A user map that genuinely has a fixed point folds to it, by evaluating
        // the real body: `halve(x) = x / 2` from seed 0 is already at 0 (0/2==0).
        let fns = fn_table("fn halve(x: i64) -> i64 { return x / 2; }");
        assert_eq!(
            iterate_user_fixed_point(&call_x("halve"), "x", &fns, 0, 100),
            Q16Outcome::Fixed(0)
        );
    }

    #[test]
    fn negate_map_is_a_two_cycle() {
        // x = negate(x) -> x, -x, x, -x ... period-2 (E2211). Seed 1.0.
        let fns = fn_table("fn negate(x: i64) -> i64 { return 0 - x; }");
        assert_eq!(
            iterate_user_fixed_point(&call_x("negate"), "x", &fns, 0x1_0000, 1000),
            Q16Outcome::Cycle(0x1_0000)
        );
    }

    #[test]
    fn divergent_map_is_rejected_not_saturated() {
        // x = grow(x) = x + 0.5 -> |x| exceeds 2^30 (E2212), never saturated.
        let fns = fn_table("fn grow(x: i64) -> i64 { return x + 536870912; }");
        assert_eq!(
            iterate_user_fixed_point(&call_x("grow"), "x", &fns, 0x2000_0000, 1000),
            Q16Outcome::Diverged
        );
    }

    #[test]
    fn division_by_zero_map_is_rejected() {
        // x = boom(x) = 1 / (x - x) -> divisor always 0 (E2214, NonComptime).
        let fns = fn_table("fn boom(x: i64) -> i64 { return 65536 / (x - x); }");
        assert_eq!(
            iterate_user_fixed_point(&call_x("boom"), "x", &fns, 0x1_0000, 1000),
            Q16Outcome::NonComptime
        );
    }

    #[test]
    fn unresolved_call_is_not_comptime() {
        // A call to a function absent from the table cannot be folded (E2214).
        let fns = CtFnTable::new();
        assert_eq!(
            iterate_user_fixed_point(&call_x("missing"), "x", &fns, 0, 1000),
            Q16Outcome::NonComptime
        );
    }

    #[test]
    fn reversed_range_leaves_seed_unchanged() {
        // trip <= 0 -> f iterated zero times -> the seed (bug-class 1).
        let fns = fn_table(CANON_COS);
        assert_eq!(
            iterate_user_fixed_point(&call_x("cos_q16"), "x", &fns, 12345, 0),
            Q16Outcome::Fixed(12345)
        );
    }
}
