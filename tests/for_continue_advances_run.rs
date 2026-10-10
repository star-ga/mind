// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! `continue` inside a `for` / `for-each` must still advance the loop — RUNTIME
//! gate (audit rank 5).
//!
//! `for`/`for-each` desugar to a `while` whose counter increment sits at the
//! body TAIL. A `continue` lowers to a jump straight to the `while` header,
//! SKIPPING that tail increment — so the counter never advances and the loop
//! spins forever (NET-verified: `mindc run` on `for i in 0..5 { if i==2 {
//! continue; } ... }` hangs, `rc=124`). The fix injects the loop's step before
//! every in-scope `continue` (`inject_step_before_continue` in lower.rs), so
//! both the fall-through and continue paths increment exactly once.
//!
//! These functions would HANG on the buggy compiler; the whole check runs under
//! `timeout` so a regression fails loudly (rc=124) instead of stalling CI.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test for_continue_advances_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// Three range-`for` shapes (for-each cannot yet lower to a runnable `.so` — the
// array-literal `vec_len` aggregate-call ABI is a separate RFC 0005 phase-2+
// gap — so the for-each step-injection is proven at the IR level instead, in
// `for_continue_step_injection.rs`):
//  * for_cont         — `continue` guarded by a single `if`.
//  * for_cont_deep    — `continue` nested two levels deep (if→if) to exercise
//                       the rewriter's recursion into nested control flow.
//  * nested           — an inner `for` whose `continue` targets the INNER loop,
//                       plus an outer `continue`: proves the boundary (the outer
//                       step is NOT grafted onto the inner `continue`, and both
//                       loops still terminate).
//  * block_cont       — `continue` nested inside a bare braced block `{ … }` in
//                       the loop body (a `Node::Block` the walker must descend
//                       into; NET-verified it hung before the Block arm was
//                       added). A bare `continue;` as a DIRECT block statement
//                       does not parse, so the continue is always inside an `if`.
//  * block_deep       — the same through two nested bare blocks.
//
// Nested-loop `continue` shapes — the MLIR leg of the nested-continue gate whose
// native-ELF leg is `examples/mindc_mind/self_host_continue_smoke.py` (the native
// self-host emitter hung on every one of these; this backend must keep running
// them with the same values):
//  * nest_bare        — an UNCONDITIONAL `continue` as the inner `for`'s whole body,
//                       then an outer statement after the inner loop.
//  * nest_both        — unconditional `continue` at BOTH levels.
//  * nest_triple      — three nested `for`s, the innermost ending in `continue`.
//  * for_in_while / while_in_for — a range-`for` inside a `while` and back.
//  * nest_while_tail  — nested `while`s whose bodies carry an UNREACHABLE tail
//                       after `continue` that writes an OUTER var (must not run).
//  * nest_break_tail  — the same with `break`: the post-loop value of the inner
//                       counter must be the LIVE one (1), never the dead tail's.
const SRC: &str = r#"
fn nest_bare(n: i64, m: i64) -> i64 {
    let mut s = 0;
    for i in 0..n {
        for j in 0..m { continue; }
        s = s + 1;
    }
    return s;
}
fn nest_both(n: i64, m: i64) -> i64 {
    let mut s = 0;
    for i in 0..n {
        for j in 0..m { s = s + 1; continue; }
        s = s + 10;
        continue;
    }
    return s;
}
fn nest_triple(a: i64, b: i64, c: i64) -> i64 {
    let mut s = 0;
    for i in 0..a {
        for j in 0..b {
            for k in 0..c { s = s + 1; continue; }
        }
    }
    return s;
}
fn for_in_while(n: i64, m: i64) -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < n {
        for j in 0..m { s = s + j; continue; }
        i = i + 1;
    }
    return s;
}
fn while_in_for(n: i64, m: i64) -> i64 {
    let mut s = 0;
    for i in 0..n {
        let mut j = 0;
        while j < m { j = j + 1; s = s + 1; continue; }
    }
    return s;
}
fn nest_while_tail(n: i64, m: i64) -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < n {
        let mut j = 0;
        while j < m { j = j + 1; s = s + 1; continue; s = s + 100; }
        i = i + 1;
    }
    return s;
}
fn nest_break_tail(n: i64, m: i64) -> i64 {
    let mut i = 0;
    let mut t = 0;
    while i < n {
        let mut j = 0;
        while j < m { j = j + 1; break; j = j + 1; }
        t = t + j;
        i = i + 1;
    }
    return t * 10 + i;
}
fn for_cont(n: i64) -> i64 {
    let mut s = 0;
    for i in 0..n {
        if i == 2 { continue; }
        s = s + i;
    }
    return s;
}
fn for_cont_deep(n: i64) -> i64 {
    let mut s = 0;
    for i in 0..n {
        if i >= 2 {
            if i <= 3 { continue; }
        }
        s = s + i;
    }
    return s;
}
fn nested(n: i64, m: i64) -> i64 {
    let mut s = 0;
    for i in 0..n {
        for j in 0..m {
            if j == 1 { continue; }
            s = s + 1;
        }
        if i == 0 { continue; }
        s = s + 100;
    }
    return s;
}
fn block_cont(n: i64) -> i64 {
    let mut s = 0;
    for i in 0..n {
        {
            if i == 2 { continue; }
        }
        s = s + i;
    }
    return s;
}
fn block_deep(n: i64) -> i64 {
    let mut s = 0;
    for i in 0..n {
        {
            {
                if i == 2 { continue; }
            }
        }
        s = s + i;
    }
    return s;
}
fn main() -> i64 { return 0; }
"#;

#[test]
fn for_continue_advances_loop() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "for_continue_advances_run",
            "for-continue-advances-run: mindc not found; skipping",
        );
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_for_continue_advances_run.mind");
    let so = dir.join("mind_for_continue_advances_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !crate::common::gate::compiled("for_continue_advances_run", &out) {
        return;
    }

    // Expected (first-match / real `for` semantics):
    //  for_cont(5)      = 0+1+3+4                     = 8
    //  for_cont(6)      = 0+1+3+4+5                   = 13
    //  for_cont_deep(6) = 0+1 + (2,3 skipped) + 4+5   = 10
    //  nested(2,3)      = i0:{j0,j2}=2, i0 continue;
    //                     i1:{j0,j2}=2, +100          = 104
    //  block_cont(5)    = 0+1+3+4                     = 8
    //  block_deep(5)    = 0+1+3+4                     = 8
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         def f(name, nargs):\n\
         \x20   fn = getattr(lib, name); fn.restype = ctypes.c_int64\n\
         \x20   fn.argtypes = [ctypes.c_int64]*nargs; return fn\n\
         fc = f('for_cont', 1); fd = f('for_cont_deep', 1); ns = f('nested', 2)\n\
         bc = f('block_cont', 1); bd = f('block_deep', 1)\n\
         assert fc(5) == 8,  'for_cont(5)='+str(fc(5))\n\
         assert fc(6) == 13, 'for_cont(6)='+str(fc(6))\n\
         assert fd(6) == 10, 'for_cont_deep(6)='+str(fd(6))\n\
         assert ns(2,3) == 104, 'nested(2,3)='+str(ns(2,3))\n\
         assert bc(5) == 8,  'block_cont(5)='+str(bc(5))\n\
         assert bd(5) == 8,  'block_deep(5)='+str(bd(5))\n\
         nb = f('nest_bare', 2); nh = f('nest_both', 2); nt = f('nest_triple', 3)\n\
         fw = f('for_in_while', 2); wf = f('while_in_for', 2)\n\
         wt = f('nest_while_tail', 2); bt = f('nest_break_tail', 2)\n\
         assert nb(3,2) == 3,     'nest_bare(3,2)='+str(nb(3,2))\n\
         assert nh(3,2) == 36,    'nest_both(3,2)='+str(nh(3,2))\n\
         assert nt(2,3,4) == 24,  'nest_triple(2,3,4)='+str(nt(2,3,4))\n\
         assert fw(4,3) == 12,    'for_in_while(4,3)='+str(fw(4,3))\n\
         assert wf(3,4) == 12,    'while_in_for(3,4)='+str(wf(3,4))\n\
         assert wt(3,2) == 6,     'nest_while_tail(3,2)='+str(wt(3,2))\n\
         assert bt(3,2) == 33,    'nest_break_tail(3,2)='+str(bt(3,2))\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    // Guard against a regression hang: `timeout` kills a spinning call (rc=124).
    let out = Command::new("timeout")
        .args(["60", "python3", "-c", &py])
        .output()
        .expect("timeout python3");
    assert!(
        out.status.success(),
        "for-continue-advances-run check failed (rc={:?}; 124=HANG regression):\n\
         stdout: {}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
