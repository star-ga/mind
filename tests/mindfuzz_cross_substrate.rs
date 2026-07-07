// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

// Part of the MIND project (Machine Intelligence Native Design).

// The generator/harness (dlopen + FFI, sha2, temp .so build) is only exercised
// by the Linux-gated test below; on macOS/Windows the test no-ops, leaving these
// items unused. Silence the resulting dead-code warnings off-Linux rather than
// cfg-gating every helper individually.
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]

//! Differential DETERMINISM fuzzer over the executable MIND subset (issue #72).
//!
//! This is the in-tree, CI-runnable sibling of `tests/cross_substrate_identity.rs`.
//! Where that gate pins a fixed corpus of canary kernels to committed
//! per-substrate hashes, this test **deterministically generates** a bounded
//! batch of random-but-valid MIND programs from a seeded PRNG and drives each
//! through THREE independent oracles that must all agree:
//!
//!   1. **mic@3 byte-identity (x86 == ARM).** The mic@3 artifact is the compact
//!      serialisation of the *canonical IR* — it is emitted BEFORE any substrate
//!      lowering, so it carries no avx2/neon machine bytes and is
//!      substrate-independent by construction. This is exactly why the sibling
//!      `reference_hashes.toml` files carry the SAME hash on both their `avx2`
//!      and `neon` lines (RFC 0015 §3.1). Here we prove the artifact is a byte
//!      fixed point (compile twice → identical bytes) and fold its sha256 into a
//!      batch digest. **That batch digest is now ASSERTED across runners in CI**
//!      (job `mindfuzz_cross_runner_identity`): the `ubuntu-24.04` (avx2) and
//!      `ubuntu-24.04-arm` (neon) runners each write their batch digest to an
//!      artifact and a dependent job fails RED if the two differ — a real
//!      x86 != ARM divergence in the fuzzed corpus can no longer ship green.
//!
//!   2. **Lowered execution == substrate-invariant interpreter.** We emit a real
//!      ELF `.so` via `mindc --emit-shared`, dlopen it, call the generated
//!      `f(a)` over a fixed input vector, and assert the result equals an
//!      independent in-process AST interpreter over the SAME generated program.
//!      The interpreter uses two's-complement wrapping i64 arithmetic — exactly
//!      MIND's integer overflow semantics AND bit-identical on x86 and ARM.
//!      Because integer add is associative and every per-element operation is
//!      exact, the value the avx2-lowered ELF computes here is the identical
//!      value the neon lowering computes (MIND-CONSTITUTION §III).
//!
//!   3. **mic@3 execution == substrate-invariant interpreter (the third oracle).**
//!      Oracle 1 only proves the mic@3 bytes are DETERMINISTIC; it never RUNS
//!      them, so an operator wired correctly in the native-ELF path but wrong in
//!      the mic@3 encoding (or vice-versa — the `%`-modulo class, where an
//!      operator shipped correct in only ONE lowering path) stays invisible. So
//!      we decode the emitted mic@3 bytes with the compiler's own
//!      `compact::v3::parse_mic3`, execute the recovered IR with a small
//!      bytecode VM, and assert IT too equals the interpreter. All three
//!      agreeing — native-ELF == interpreter == mic@3-VM — is what makes an
//!      operator that is correct in only one lowering path turn the fuzzer RED.
//!
//! The generator emits `+ - * / % << >> |` over i64 plus `if/else` expressions
//! with the full comparison set (`< <= > >= == !=`) as branch conditions, so the
//! `%`-class of single-path miscompiles is inside the family the fuzzer can both
//! GENERATE and differentially EXECUTE. Divisors are forced non-zero (`x | 1`)
//! and shift amounts are bounded to `0..=5`, so no generated program can trap or
//! invoke shift-overflow UB; magnitudes stay far inside i64 so every oracle
//! wraps identically.
//!
//! The generator is SEEDED from a fixed constant (`0xDEADBEEF`, the manifest
//! seed contract) with a pure LCG — no wall-clock, no OS RNG — so the same run
//! always produces the same programs and the same verdict. A fixed program
//! count bounds wall-time deterministically.
//!
//! On ANY divergence the test fails loud, printing the seed and the exact
//! offending program, and stages a minimal reproducer under
//! `tests/mindfuzz_cross_substrate/staged/` for regression.
//!
//! Fail-closed: with `MIND_BENCH_REQUIRE=1` (set in CI) a shadowed MLIR
//! toolchain or a stub `.so` is a HARD FAIL, never a silent skip — a fuzzer
//! that did not execute the real byte comparison proves nothing (RFC 0020 §10).
//!
//! Run:
//! ```
//! MIND_BENCH_REQUIRE=1 cargo test --release \
//!     --features "mlir-build std-surface cross-module-imports" \
//!     --test mindfuzz_cross_substrate -- --nocapture
//! ```

mod common;
use common::mindc_bin;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Deterministic PRNG — byte-for-byte the LCG the canary gate uses so the whole
// suite shares one reproducible input engine (tests/cross_substrate_identity.rs).
// ---------------------------------------------------------------------------

/// Fixed generator seed. `0xDEADBEEF` == 3735928559, the manifest seed contract
/// pinned across the cross-substrate fixtures (RFC 0020 §4.3).
const FUZZ_SEED: u64 = 0xDEAD_BEEF;

/// Number of programs generated per run. Fixed so wall-time is bounded and the
/// verdict is deterministic. Override upward locally with `MINDFUZZ_ITERS` for a
/// heavier soak; CI uses this default.
const DEFAULT_ITERS: usize = 32;

/// Fixed input vector every generated `f(a)` is probed over. Chosen to hit the
/// loop-not-taken path (negative / zero) and a spread of positive bounds. All
/// magnitudes stay far inside i64 with the bounded generator, so results never
/// overflow — the wrapping arithmetic below is defence-in-depth, not reliance.
const PROBE_INPUTS: &[i64] = &[0, 1, 2, 3, 5, 7, 11, 13, 20, 33, 40, -1, -5, -40];

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 16) as u32
    }
    /// Uniform-ish integer in `[0, n)` (n must be > 0).
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

// ---------------------------------------------------------------------------
// AST for the executable integer subset — one source of truth for BOTH the
// MIND emitter and the reference interpreter, so they can never drift apart.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Op {
    Add,
    Sub,
    Mul,
    /// `/` — signed, truncating toward zero. Divisor forced non-zero at
    /// generation time (`rhs | 1`), so no divide-by-zero trap.
    Div,
    /// `%` — signed remainder, sign of the dividend. The single-path-miscompile
    /// class this fuzzer exists to catch. Divisor forced non-zero (`rhs | 1`).
    Mod,
    /// `<<` — logical left shift. Amount forced to a `0..=5` literal so it is
    /// always inside `[0, 63]` (a shift `>= 64` is UB in the ELF path).
    Shl,
    /// `>>` — ARITHMETIC right shift (sign-extending), matching MIND's
    /// `BinOp::Shr` and Rust's `i64 >>`. Amount forced to a `0..=5` literal.
    Shr,
    /// `|` — bitwise OR. Also the mechanism that makes a divisor provably
    /// non-zero (`x | 1` is always odd).
    BitOr,
}

/// Comparison operators — only ever emitted as an `if` condition, never as a
/// free i64-valued expression (a bare `(x < y)` need not coerce bool→i64 in the
/// surface language), so the generated source always type-checks.
#[derive(Clone, Copy)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Clone)]
enum Expr {
    /// The function parameter `a`.
    Param,
    /// The loop induction variable `i` (only valid inside a loop body).
    Induction,
    /// A small integer literal.
    Const(i64),
    /// A reference to a prior `let v{idx}` binding.
    Var(usize),
    /// A binary integer op.
    Bin(Op, Box<Expr>, Box<Expr>),
    /// `if (cl CMP cr) { then_e } else { else_e }` as an i64-valued expression.
    /// Only generated at binding / return top level (never nested inside a
    /// `Bin`), so the emitted `let x = if ... { .. } else { .. };` always parses.
    /// Only the TAKEN branch is evaluated (matching the ELF `scf.if` lowering).
    IfElse {
        cmp: CmpOp,
        cl: Box<Expr>,
        cr: Box<Expr>,
        then_e: Box<Expr>,
        else_e: Box<Expr>,
    },
}

/// A generated program: a chain of `let v{i}` bindings, an OPTIONAL while-loop
/// accumulator, and a return expression. `pub fn f(a: i64) -> i64`.
struct Program {
    lets: Vec<Expr>,
    /// `Some((bound, body))` adds `let mut sum=0; let mut i=0; while i<bound {
    /// sum = sum + body; i = i+1; }`. `body` may reference `Induction`.
    loop_accum: Option<(Expr, Expr)>,
    ret: Expr,
}

// ---------------------------------------------------------------------------
// Generator — bounded-depth expressions over the in-scope variable environment.
// ---------------------------------------------------------------------------

/// Generate one ARITHMETIC expression of at most `depth` nesting (never an
/// `if/else` — those are chosen only at binding top level by `gen_binding`).
/// `n_lets` = number of `let` bindings currently in scope; `allow_induction`
/// gates the `i` leaf.
fn gen_expr(g: &mut Lcg, depth: u32, n_lets: usize, allow_induction: bool) -> Expr {
    // Leaf when out of depth budget, else branch (~45% of the time).
    if depth == 0 || g.below(100) < 55 {
        return gen_leaf(g, n_lets, allow_induction);
    }
    let op = match g.below(8) {
        0 => Op::Add,
        1 => Op::Sub,
        2 => Op::Mul,
        3 => Op::Div,
        4 => Op::Mod,
        5 => Op::Shl,
        6 => Op::Shr,
        _ => Op::BitOr,
    };
    let lhs = gen_expr(g, depth - 1, n_lets, allow_induction);
    let rhs = match op {
        // Force a provably-non-zero divisor: `(expr | 1)` is always odd. This
        // removes divide-by-zero traps outright; magnitudes stay small so the
        // MIN/-1 overflow trap can never occur either.
        Op::Div | Op::Mod => {
            let inner = gen_expr(g, depth - 1, n_lets, allow_induction);
            Expr::Bin(Op::BitOr, Box::new(inner), Box::new(Expr::Const(1)))
        }
        // Bound the shift amount to a `0..=5` literal: always inside [0,63] and
        // the shifted result stays far from overflow.
        Op::Shl | Op::Shr => Expr::Const(g.below(6) as i64),
        _ => gen_expr(g, depth - 1, n_lets, allow_induction),
    };
    Expr::Bin(op, Box::new(lhs), Box::new(rhs))
}

fn gen_leaf(g: &mut Lcg, n_lets: usize, allow_induction: bool) -> Expr {
    // Weighted leaf: param, induction (if allowed), a prior let, or a const.
    let roll = g.below(100);
    if allow_induction && roll < 25 {
        Expr::Induction
    } else if n_lets > 0 && roll < 60 {
        Expr::Var(g.below(n_lets as u32) as usize)
    } else if roll < 80 {
        Expr::Param
    } else {
        // Small non-negative constant 0..=9 keeps magnitudes bounded.
        Expr::Const(g.below(10) as i64)
    }
}

/// Generate the RHS of a `let` binding: ~30% an `if/else` expression (branch /
/// comparison coverage), otherwise a plain arithmetic expression.
fn gen_binding(g: &mut Lcg, n_lets: usize, allow_induction: bool) -> Expr {
    if g.below(100) < 30 {
        let cmp = match g.below(6) {
            0 => CmpOp::Lt,
            1 => CmpOp::Le,
            2 => CmpOp::Gt,
            3 => CmpOp::Ge,
            4 => CmpOp::Eq,
            _ => CmpOp::Ne,
        };
        Expr::IfElse {
            cmp,
            cl: Box::new(gen_expr(g, 2, n_lets, allow_induction)),
            cr: Box::new(gen_expr(g, 2, n_lets, allow_induction)),
            then_e: Box::new(gen_expr(g, 3, n_lets, allow_induction)),
            else_e: Box::new(gen_expr(g, 3, n_lets, allow_induction)),
        }
    } else {
        gen_expr(g, 3, n_lets, allow_induction)
    }
}

/// Generate a full program. Two families, chosen by the PRNG:
///   * `scalar` — pure let-chain + return (SSA / lowering stress).
///   * `accum`  — let-chain + while-loop accumulator + return (reduction-order
///     stress: a loop the backend must NOT silently reorder/vectorise wrongly).
fn gen_program(g: &mut Lcg) -> Program {
    let n_lets = 1 + g.below(4) as usize; // 1..=4 bindings
    let mut lets = Vec::with_capacity(n_lets);
    for k in 0..n_lets {
        // Binding k sees params/consts and the k earlier bindings; no induction
        // (these bindings are computed before any loop).
        lets.push(gen_binding(g, k, false));
    }

    // ~55% of programs get the loop accumulator.
    let loop_accum = if g.below(100) < 55 {
        // Bound: a small positive literal (1..=12) so iteration count is
        // bounded and > 0 for positive inputs; the loop body may use `i`.
        let bound = Expr::Const(1 + g.below(12) as i64);
        let body = gen_expr(g, 3, n_lets, true);
        Some((bound, body))
    } else {
        None
    };

    // Return combines the last binding and (if present) the accumulator.
    let last_var = Expr::Var(n_lets - 1);
    let ret = if loop_accum.is_some() {
        // `sum` is bound after the lets as v{n_lets}; reference it by index.
        Expr::Bin(Op::Add, Box::new(Expr::Var(n_lets)), Box::new(last_var))
    } else {
        Expr::Bin(
            Op::Add,
            Box::new(last_var),
            Box::new(gen_leaf(g, n_lets, false)),
        )
    };

    Program {
        lets,
        loop_accum,
        ret,
    }
}

/// True if any binding is an `if/else` expression (branch-coverage bookkeeping).
fn program_has_if(p: &Program) -> bool {
    p.lets.iter().any(|e| matches!(e, Expr::IfElse { .. }))
}

// ---------------------------------------------------------------------------
// MIND source emitter — AST → valid `.mind` text.
// ---------------------------------------------------------------------------

fn emit_op(op: Op) -> &'static str {
    match op {
        Op::Add => "+",
        Op::Sub => "-",
        Op::Mul => "*",
        Op::Div => "/",
        Op::Mod => "%",
        Op::Shl => "<<",
        Op::Shr => ">>",
        Op::BitOr => "|",
    }
}

fn emit_cmp(cmp: CmpOp) -> &'static str {
    match cmp {
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
    }
}

fn emit_expr(e: &Expr, out: &mut String) {
    match e {
        Expr::Param => out.push('a'),
        Expr::Induction => out.push('i'),
        Expr::Const(c) => out.push_str(&c.to_string()),
        Expr::Var(idx) => {
            out.push('v');
            out.push_str(&idx.to_string());
        }
        Expr::Bin(op, l, r) => {
            out.push('(');
            emit_expr(l, out);
            out.push(' ');
            out.push_str(emit_op(*op));
            out.push(' ');
            emit_expr(r, out);
            out.push(')');
        }
        Expr::IfElse {
            cmp,
            cl,
            cr,
            then_e,
            else_e,
        } => {
            out.push_str("if ");
            emit_expr(cl, out);
            out.push(' ');
            out.push_str(emit_cmp(*cmp));
            out.push(' ');
            emit_expr(cr, out);
            out.push_str(" { ");
            emit_expr(then_e, out);
            out.push_str(" } else { ");
            emit_expr(else_e, out);
            out.push_str(" }");
        }
    }
}

/// Render the program to MIND source. `sum` (when a loop is present) is bound as
/// `v{n_lets}` so return-expression `Var` indices line up with the interpreter.
fn emit_program(p: &Program) -> String {
    let mut s = String::new();
    s.push_str("// MIND-Fuzz #72 generated program — deterministic, seed 0xDEADBEEF.\n");
    s.push_str("// One AST, three consumers: this source, the in-test oracle, the mic@3 VM.\n");
    s.push_str("pub fn f(a: i64) -> i64 {\n");
    for (k, e) in p.lets.iter().enumerate() {
        s.push_str(&format!("    let v{k}: i64 = "));
        emit_expr(e, &mut s);
        s.push_str(";\n");
    }
    if let Some((bound, body)) = &p.loop_accum {
        let sum_idx = p.lets.len();
        s.push_str(&format!("    let mut v{sum_idx}: i64 = 0;\n"));
        s.push_str("    let mut i: i64 = 0;\n");
        s.push_str("    while i < ");
        emit_expr(bound, &mut s);
        s.push_str(" {\n");
        s.push_str(&format!("        v{sum_idx} = v{sum_idx} + "));
        emit_expr(body, &mut s);
        s.push_str(";\n");
        s.push_str("        i = i + 1;\n");
        s.push_str("    }\n");
    }
    s.push_str("    return ");
    emit_expr(&p.ret, &mut s);
    s.push_str(";\n}\n");
    s
}

// ---------------------------------------------------------------------------
// Reference interpreter — the substrate-invariant oracle.
//
// Two's-complement WRAPPING i64 arithmetic: this is MIND's integer overflow
// semantics (verified: `i64::MAX + 1` wraps to `i64::MIN` in the compiled ELF)
// AND it is bit-identical on x86 and ARM. That dual property is what makes the
// interpreter a valid cross-substrate reference, not just an x86 reference.
// ---------------------------------------------------------------------------

fn apply_op(op: Op, lv: i64, rv: i64) -> i64 {
    match op {
        Op::Add => lv.wrapping_add(rv),
        Op::Sub => lv.wrapping_sub(rv),
        Op::Mul => lv.wrapping_mul(rv),
        // `rv` is guaranteed non-zero (`x | 1`) so wrapping_div/rem never panic.
        Op::Div => lv.wrapping_div(rv),
        Op::Mod => lv.wrapping_rem(rv),
        // Shift amount is a bounded `0..=5` literal; mask is defence-in-depth.
        Op::Shl => lv.wrapping_shl((rv as u64 & 63) as u32),
        Op::Shr => lv.wrapping_shr((rv as u64 & 63) as u32),
        Op::BitOr => lv | rv,
    }
}

fn eval_cmp(cmp: CmpOp, lv: i64, rv: i64) -> bool {
    match cmp {
        CmpOp::Lt => lv < rv,
        CmpOp::Le => lv <= rv,
        CmpOp::Gt => lv > rv,
        CmpOp::Ge => lv >= rv,
        CmpOp::Eq => lv == rv,
        CmpOp::Ne => lv != rv,
    }
}

fn eval_expr(e: &Expr, a: i64, i: i64, vars: &[i64]) -> i64 {
    match e {
        Expr::Param => a,
        Expr::Induction => i,
        Expr::Const(c) => *c,
        Expr::Var(idx) => vars[*idx],
        Expr::Bin(op, l, r) => {
            let lv = eval_expr(l, a, i, vars);
            let rv = eval_expr(r, a, i, vars);
            apply_op(*op, lv, rv)
        }
        Expr::IfElse {
            cmp,
            cl,
            cr,
            then_e,
            else_e,
        } => {
            let lv = eval_expr(cl, a, i, vars);
            let rv = eval_expr(cr, a, i, vars);
            // Only the taken branch is evaluated — matches the ELF `scf.if`.
            if eval_cmp(*cmp, lv, rv) {
                eval_expr(then_e, a, i, vars)
            } else {
                eval_expr(else_e, a, i, vars)
            }
        }
    }
}

/// Evaluate the whole program at input `a`. Mirrors the emitted control flow
/// exactly: bind the lets in order, run the loop accumulator (fixed forward
/// order — the order the ELF must preserve), then the return expression.
fn eval_program(p: &Program, a: i64) -> i64 {
    let mut vars: Vec<i64> = Vec::with_capacity(p.lets.len() + 1);
    for e in &p.lets {
        let v = eval_expr(e, a, 0, &vars);
        vars.push(v);
    }
    if let Some((bound, body)) = &p.loop_accum {
        let bound_v = eval_expr(bound, a, 0, &vars);
        let mut sum: i64 = 0;
        let mut i: i64 = 0;
        while i < bound_v {
            // Body may reference `i`; `sum` (Var(n_lets)) is not yet pushed, so
            // the body cannot reference it — matching the generator, which only
            // ever lets the loop body see the lets + param + induction.
            let inc = eval_expr(body, a, i, &vars);
            sum = sum.wrapping_add(inc);
            i = i.wrapping_add(1);
        }
        vars.push(sum);
    }
    eval_expr(&p.ret, a, 0, &vars)
}

// ---------------------------------------------------------------------------
// Oracle 3 — the mic@3 bytecode VM.
//
// Decodes the EMITTED mic@3 artifact with the compiler's own
// `compact::v3::parse_mic3` and executes the recovered IR over the probe inputs.
// This is what distinguishes oracle 3 from oracle 1: oracle 1 only proves the
// mic@3 bytes are a deterministic fixed point, never that they COMPUTE the right
// value. Running the decoded IR closes the `%`-class hole where an operator is
// correct in one lowering path (e.g. mic@3) but wrong in another (native-ELF):
// native-ELF == interpreter == mic@3-VM must all hold.
//
// Gated to `std-surface`: the `While`/`If`/bitwise `BinOp` variants only exist
// with that feature (the fuzzer only runs meaningfully with it enabled). Off
// the feature the whole module is compiled out and oracle 3 is not run — the
// test soft-skips at runtime in that configuration anyway.
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "linux", feature = "std-surface"))]
mod mic3vm {
    use libmind::ir::{BinOp, IRModule, Instr, ValueId};
    use std::collections::HashMap;

    /// Look up `fname` in the parsed module and evaluate it at `arg`.
    pub fn eval_fn(module: &IRModule, fname: &str, arg: i64) -> Result<i64, String> {
        for instr in &module.instrs {
            if let Instr::FnDef {
                name, body, ret_id, ..
            } = instr
            {
                if name == fname {
                    let mut env: HashMap<usize, i64> = HashMap::new();
                    if let Some(v) = exec_seq(body, &mut env, &[arg])? {
                        return Ok(v);
                    }
                    // No explicit `Return` hit — fall back to the fn's ret_id.
                    if let Some(rid) = ret_id {
                        return get(&env, *rid);
                    }
                    return Err("mic@3-VM: fn reached end with no return".into());
                }
            }
        }
        Err(format!("mic@3-VM: fn `{fname}` not found in module"))
    }

    fn get(env: &HashMap<usize, i64>, v: ValueId) -> Result<i64, String> {
        env.get(&v.0)
            .copied()
            .ok_or_else(|| format!("mic@3-VM: unbound value %{}", v.0))
    }

    /// Execute a straight-line instruction sequence, threading the SSA env.
    /// Returns `Ok(Some(v))` when a `Return` is hit, `Ok(None)` at the end.
    fn exec_seq(
        instrs: &[Instr],
        env: &mut HashMap<usize, i64>,
        args: &[i64],
    ) -> Result<Option<i64>, String> {
        for instr in instrs {
            match instr {
                Instr::Param { dst, index, .. } => {
                    let a = *args
                        .get(*index)
                        .ok_or_else(|| format!("mic@3-VM: param index {index} out of range"))?;
                    env.insert(dst.0, a);
                }
                Instr::ConstI64(dst, val) => {
                    env.insert(dst.0, *val);
                }
                Instr::BinOp { dst, op, lhs, rhs } => {
                    let l = get(env, *lhs)?;
                    let r = get(env, *rhs)?;
                    env.insert(dst.0, apply(*op, l, r)?);
                }
                Instr::Return { value } => {
                    return Ok(Some(match value {
                        Some(v) => get(env, *v)?,
                        None => 0,
                    }));
                }
                Instr::While {
                    cond_id,
                    cond_instrs,
                    body,
                    live_vars,
                    init_ids,
                    exit_ids,
                    ..
                } => {
                    // Loop-carried values live in the `init_ids` SSA slots; each
                    // iteration re-binds those slots to the current carried value,
                    // re-evaluates the condition, runs the body, and captures the
                    // post-body `live_vars` ids as the next carried value.
                    let mut carried: Vec<i64> = init_ids
                        .iter()
                        .map(|id| get(env, *id))
                        .collect::<Result<_, _>>()?;
                    loop {
                        for (k, id) in init_ids.iter().enumerate() {
                            env.insert(id.0, carried[k]);
                        }
                        if let Some(v) = exec_seq(cond_instrs, env, args)? {
                            return Ok(Some(v));
                        }
                        if get(env, *cond_id)? == 0 {
                            break;
                        }
                        if let Some(v) = exec_seq(body, env, args)? {
                            return Ok(Some(v));
                        }
                        for (k, (_, post)) in live_vars.iter().enumerate() {
                            carried[k] = get(env, *post)?;
                        }
                    }
                    // Post-loop references use the `exit_ids` SSA slots.
                    for (k, eid) in exit_ids.iter().enumerate() {
                        env.insert(eid.0, carried[k]);
                    }
                }
                Instr::If {
                    cond_id,
                    cond_instrs,
                    then_instrs,
                    then_result,
                    else_instrs,
                    else_result,
                    dst,
                    ..
                } => {
                    if let Some(v) = exec_seq(cond_instrs, env, args)? {
                        return Ok(Some(v));
                    }
                    // Evaluate only the taken branch (matches the ELF `scf.if`,
                    // and keeps an untaken branch's arithmetic from being run).
                    if get(env, *cond_id)? != 0 {
                        if let Some(v) = exec_seq(then_instrs, env, args)? {
                            return Ok(Some(v));
                        }
                        let r = get(env, *then_result)?;
                        env.insert(dst.0, r);
                    } else {
                        if let Some(v) = exec_seq(else_instrs, env, args)? {
                            return Ok(Some(v));
                        }
                        let r = get(env, *else_result)?;
                        env.insert(dst.0, r);
                    }
                }
                other => {
                    return Err(format!(
                        "mic@3-VM: instruction outside the generated subset: {other:?}"
                    ));
                }
            }
        }
        Ok(None)
    }

    fn apply(op: BinOp, l: i64, r: i64) -> Result<i64, String> {
        Ok(match op {
            BinOp::Add => l.wrapping_add(r),
            BinOp::Sub => l.wrapping_sub(r),
            BinOp::Mul => l.wrapping_mul(r),
            BinOp::Div => l.wrapping_div(r),
            BinOp::Mod => l.wrapping_rem(r),
            BinOp::Lt => (l < r) as i64,
            BinOp::Le => (l <= r) as i64,
            BinOp::Gt => (l > r) as i64,
            BinOp::Ge => (l >= r) as i64,
            BinOp::Eq => (l == r) as i64,
            BinOp::Ne => (l != r) as i64,
            BinOp::BitAnd => l & r,
            BinOp::BitOr => l | r,
            BinOp::BitXor => l ^ r,
            BinOp::Shl => l.wrapping_shl((r as u64 & 63) as u32),
            BinOp::Shr => l.wrapping_shr((r as u64 & 63) as u32),
        })
    }
}

// ---------------------------------------------------------------------------
// Toolchain guard — fail-closed under MIND_BENCH_REQUIRE (RFC 0020 §10).
// ---------------------------------------------------------------------------

/// Returns `true` if the MLIR toolchain is present. Under `MIND_BENCH_REQUIRE`
/// a missing tool is a HARD FAIL (the fuzzer must never self-skip into a vacuous
/// green); without the var it soft-skips like the blas smoke tests.
fn toolchain_ready() -> bool {
    for tool in ["mlir-opt", "mlir-translate", "clang"] {
        if which::which(tool).is_err() {
            assert!(
                std::env::var_os("MIND_BENCH_REQUIRE").is_none(),
                "MIND_BENCH_REQUIRE is set but '{tool}' is not on PATH: the \
                 differential determinism fuzzer cannot run. Install the MLIR \
                 toolchain (mlir-opt / mlir-translate / clang) on this runner."
            );
            println!("mindfuzz_cross_substrate: {tool} not on PATH; skipping");
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Compile helpers.
// ---------------------------------------------------------------------------

/// Emit mic@3 for `src_path` to `out`. Returns the artifact bytes, or a String
/// error on a non-zero compile.
fn emit_mic3(bin: &Path, src_path: &Path, out: &Path) -> Result<Vec<u8>, String> {
    let r = Command::new(bin)
        .args([
            src_path.to_str().unwrap(),
            "--emit-mic3",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("spawn mindc --emit-mic3");
    if !r.status.success() {
        return Err(format!(
            "mindc --emit-mic3 exit {}: {}",
            r.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&r.stderr)
        ));
    }
    fs::read(out).map_err(|e| format!("read mic3 artifact: {e}"))
}

/// Emit a real ELF `.so` for `src_path`. Fails hard (never skips) on a stub or a
/// non-ELF artifact — a stub `.so` (#306) makes the executed byte comparison
/// vacuous, so it must not pass as green.
fn emit_shared(bin: &Path, src_path: &Path, out: &Path) -> Result<(), String> {
    let r = Command::new(bin)
        .args([
            src_path.to_str().unwrap(),
            "--emit-shared",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("spawn mindc --emit-shared");
    if !r.status.success() {
        return Err(format!(
            "mindc --emit-shared exit {}: {}",
            r.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&r.stderr)
        ));
    }
    let bytes = fs::read(out).map_err(|e| format!("read .so: {e}"))?;
    if !bytes.starts_with(b"\x7fELF") {
        return Err(format!(
            "emitted .so is not an ELF ({} bytes) — refusing a vacuous comparison",
            bytes.len()
        ));
    }
    // A real lowered kernel .so is many KiB; the ~1245-byte launcher stub (#306)
    // is not a valid substrate under test.
    if bytes.len() < 2048 {
        return Err(format!(
            "emitted .so is only {} bytes — looks like a launcher stub, not a \
             real kernel; refusing a vacuous byte comparison",
            bytes.len()
        ));
    }
    Ok(())
}

/// dlopen `so_path`, resolve `f`, and evaluate it at each probe input.
fn run_compiled_f(so_path: &Path, inputs: &[i64]) -> Result<Vec<i64>, String> {
    type FFn = unsafe extern "C" fn(i64) -> i64;
    // SAFETY: so_path was just verified to be a real ELF above.
    let lib = unsafe { libloading::Library::new(so_path).map_err(|e| format!("dlopen: {e}"))? };
    let f: libloading::Symbol<FFn> =
        unsafe { lib.get(b"f\0").map_err(|e| format!("dlsym f: {e}"))? };
    let mut out = Vec::with_capacity(inputs.len());
    for &a in inputs {
        // SAFETY: matching (i64) -> i64 System V AMD64 C ABI; `f` is
        // `pub fn f(a: i64) -> i64` exported as the C symbol `f`.
        out.push(unsafe { f(a) });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Reproducer staging.
// ---------------------------------------------------------------------------

fn staged_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("mindfuzz_cross_substrate")
        .join("staged")
}

/// Write a minimal reproducer for a divergent program so it becomes a permanent
/// regression fixture. Named by seed + program index for reproducibility.
fn stage_reproducer(idx: usize, header: &str, src: &str) -> PathBuf {
    let dir = staged_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(format!("fuzz_repro_seed_deadbeef_prog{idx:03}.mind"));
    let body = format!("// DIVERGENCE REPRODUCER (issue #72 fuzzer)\n// {header}\n{src}");
    let _ = fs::write(&path, body);
    path
}

// ---------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "linux"))]
#[test]
fn mindfuzz_cross_substrate_determinism() {
    println!(
        "mindfuzz_cross_substrate: SKIP — dlopen()s a Linux ELF; gated to \
         #[cfg(target_os = \"linux\")]"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn mindfuzz_cross_substrate_determinism() {
    let bin = mindc_bin();
    assert!(
        bin.exists(),
        "mindc binary not found at {} — run `cargo build --release --features \
         mlir-build,std-surface,cross-module-imports` first",
        bin.display()
    );

    if !toolchain_ready() {
        // Soft-skip only reachable without MIND_BENCH_REQUIRE (else the guard
        // already panicked). Honest: identity is UNVERIFIED on this run.
        return;
    }

    let iters = std::env::var("MINDFUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ITERS);

    let tmp = std::env::temp_dir().join("mindfuzz_xsi");
    let _ = fs::create_dir_all(&tmp);

    let mut g = Lcg::new(FUZZ_SEED);
    let mut n_scalar = 0usize;
    let mut n_accum = 0usize;
    let mut n_if = 0usize;
    let mut mic3_digest = Sha256::new(); // running digest over all mic@3 bytes

    // Whether oracle 3 (the mic@3 VM) is compiled in this feature configuration.
    let vm_enabled = cfg!(feature = "std-surface");

    println!(
        "\n=== MIND-Fuzz #72 differential determinism — seed 0x{FUZZ_SEED:08X}, \
         {iters} programs, {} probe inputs, oracles: mic@3-fixed-point + ELF + \
         mic@3-VM{} ===",
        PROBE_INPUTS.len(),
        if vm_enabled {
            ""
        } else {
            " (VM DISABLED: no std-surface)"
        }
    );

    for idx in 0..iters {
        let prog = gen_program(&mut g);
        if prog.loop_accum.is_some() {
            n_accum += 1;
        } else {
            n_scalar += 1;
        }
        if program_has_if(&prog) {
            n_if += 1;
        }
        let src = emit_program(&prog);
        let src_path = tmp.join(format!("prog{idx:03}.mind"));
        fs::write(&src_path, &src).expect("write generated program");

        // The substrate-invariant reference: the AST interpreter over the probes.
        // Both the ELF oracle and the mic@3-VM oracle are compared against this.
        let expected: Vec<i64> = PROBE_INPUTS
            .iter()
            .map(|&a| eval_program(&prog, a))
            .collect();

        // --- Oracle 1: mic@3 byte-identity (the substrate-invariant artifact) ---
        let mic3_a = tmp.join(format!("prog{idx:03}_a.mic3"));
        let mic3_b = tmp.join(format!("prog{idx:03}_b.mic3"));
        let bytes_a = emit_mic3(&bin, &src_path, &mic3_a).unwrap_or_else(|e| {
            let repro = stage_reproducer(idx, &format!("mic3 compile failed: {e}"), &src);
            panic!(
                "MIND-Fuzz PROG {idx} (seed 0x{FUZZ_SEED:08X}): a generated program \
                 that should compile FAILED --emit-mic3.\n{e}\n\
                 reproducer staged at {}\n--- source ---\n{src}",
                repro.display()
            );
        });
        let bytes_b = emit_mic3(&bin, &src_path, &mic3_b).expect("second mic3 emit");
        if bytes_a != bytes_b {
            let repro = stage_reproducer(
                idx,
                "mic@3 NON-DETERMINISM: two compiles of one source differ",
                &src,
            );
            panic!(
                "MIND-Fuzz PROG {idx} (seed 0x{FUZZ_SEED:08X}): mic@3 is NOT a byte \
                 fixed point — codegen non-determinism ({} vs {} bytes).\n\
                 reproducer staged at {}\n--- source ---\n{src}",
                bytes_a.len(),
                bytes_b.len(),
                repro.display()
            );
        }
        let mic3_hash = {
            let mut h = Sha256::new();
            h.update(&bytes_a);
            format!("{:x}", h.finalize())
        };
        mic3_digest.update(&bytes_a);

        // --- Oracle 3: decode + EXECUTE the mic@3 bytes (third oracle) ---
        // Only compiled with std-surface (While/If/bitwise IR variants). The
        // decode uses the compiler's own parser, so this proves the emitted
        // mic@3 both round-trips AND computes the substrate-invariant value.
        #[cfg(feature = "std-surface")]
        {
            let module = libmind::ir::compact::v3::parse_mic3(&bytes_a).unwrap_or_else(|e| {
                let repro = stage_reproducer(idx, &format!("mic@3 re-parse failed: {e:?}"), &src);
                panic!(
                    "MIND-Fuzz PROG {idx} (seed 0x{FUZZ_SEED:08X}): emitted mic@3 did \
                     not parse back with compact::v3::parse_mic3.\n{e:?}\n\
                     reproducer staged at {}\n--- source ---\n{src}",
                    repro.display()
                );
            });
            let vm: Vec<i64> = PROBE_INPUTS
                .iter()
                .map(|&a| {
                    mic3vm::eval_fn(&module, "f", a).unwrap_or_else(|e| {
                        let repro = stage_reproducer(idx, &format!("mic@3-VM error: {e}"), &src);
                        panic!(
                            "MIND-Fuzz PROG {idx} (seed 0x{FUZZ_SEED:08X}): mic@3-VM \
                             failed to execute the decoded IR at a={a}.\n{e}\n\
                             reproducer staged at {}\n--- source ---\n{src}",
                            repro.display()
                        );
                    })
                })
                .collect();
            if vm != expected {
                let mut diverge = String::new();
                for (k, &a) in PROBE_INPUTS.iter().enumerate() {
                    if vm[k] != expected[k] {
                        diverge = format!("f({a}): mic@3-VM={} oracle={}", vm[k], expected[k]);
                        break;
                    }
                }
                let repro = stage_reproducer(idx, &format!("mic@3-VM DIVERGENCE: {diverge}"), &src);
                panic!(
                    "MIND-Fuzz PROG {idx} (seed 0x{FUZZ_SEED:08X}): the mic@3 lowering \
                     path computed a DIFFERENT value than the substrate-invariant \
                     oracle.\n  {diverge}\n\
                     (an operator correct in only one lowering path — the `%`-class \
                     silent miscompile the third oracle exists to catch).\n\
                     reproducer staged at {}\n--- source ---\n{src}",
                    repro.display()
                );
            }
        }

        // --- Oracle 2: lowered ELF execution == substrate-invariant interpreter ---
        let so_path = tmp.join(format!("prog{idx:03}.so"));
        emit_shared(&bin, &src_path, &so_path).unwrap_or_else(|e| {
            let repro = stage_reproducer(idx, &format!("emit-shared failed: {e}"), &src);
            panic!(
                "MIND-Fuzz PROG {idx} (seed 0x{FUZZ_SEED:08X}): --emit-shared did not \
                 produce a real kernel .so.\n{e}\n\
                 reproducer staged at {}\n--- source ---\n{src}",
                repro.display()
            );
        });

        let actual = run_compiled_f(&so_path, PROBE_INPUTS).expect("dlopen + call generated f");
        if actual != expected {
            // Find the first diverging probe for a crisp message.
            let mut diverge = String::new();
            for (k, &a) in PROBE_INPUTS.iter().enumerate() {
                if actual[k] != expected[k] {
                    diverge = format!("f({a}): compiled={} oracle={}", actual[k], expected[k]);
                    break;
                }
            }
            let repro = stage_reproducer(idx, &format!("EXECUTION DIVERGENCE: {diverge}"), &src);
            panic!(
                "MIND-Fuzz PROG {idx} (seed 0x{FUZZ_SEED:08X}): compiled ELF output \
                 diverged from the substrate-invariant oracle.\n  {diverge}\n\
                 (a lowering / reduction-order bug: the value differs, so avx2 and \
                 neon can no longer agree — MIND-CONSTITUTION §III).\n\
                 reproducer staged at {}\n--- source ---\n{src}",
                repro.display()
            );
        }

        println!(
            "  PROG {idx:03} [{}{}] mic3={}… ({} bytes) probes={} OK (ELF{})",
            if prog.loop_accum.is_some() {
                "accum"
            } else {
                "scalar"
            },
            if program_has_if(&prog) { "+if" } else { "" },
            &mic3_hash[..12],
            bytes_a.len(),
            PROBE_INPUTS.len(),
            if vm_enabled { " == mic@3-VM" } else { "" },
        );
    }

    // A batch-level digest over every mic@3 artifact. mic@3 is
    // substrate-independent by construction (RFC 0015 §3.1), so the digest the
    // `ubuntu-24.04` (avx2) runner computes MUST equal the one the
    // `ubuntu-24.04-arm` (neon) runner computes over the identical seeded
    // corpus. When `MINDFUZZ_DIGEST_OUT` is set (CI), the digest is written
    // there so the `mindfuzz_cross_runner_identity` job can download both
    // runners' files and FAIL RED on any mismatch — the cross-runner assertion,
    // not just a print.
    let batch = format!("{:x}", mic3_digest.finalize());
    if let Some(path) = std::env::var_os("MINDFUZZ_DIGEST_OUT") {
        fs::write(&path, &batch).unwrap_or_else(|e| {
            panic!(
                "MIND-Fuzz: failed to write batch digest to MINDFUZZ_DIGEST_OUT \
                 ({path:?}): {e}"
            )
        });
        println!(
            "mindfuzz_cross_substrate: wrote batch mic@3 digest to {path:?} for the \
             cross-runner identity assertion"
        );
    }
    println!(
        "\nmindfuzz_cross_substrate PASS: {iters} programs \
         ({n_scalar} scalar / {n_accum} accum, {n_if} with if/else), {} probes \
         each, 0 mic@3 non-determinism, 0 mic@3-VM divergence, 0 ELF divergence.\n\
         batch mic@3 digest (x86; ARM must match) = {batch}",
        PROBE_INPUTS.len()
    );
}
