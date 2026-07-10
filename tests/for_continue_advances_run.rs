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
const SRC: &str = r#"
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
        println!("for-continue-advances-run: mindc not found; skipping");
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
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("for-continue-advances-run: needs mlir-build; skipping");
            return;
        }
        panic!("for-continue-advances-run: mindc --emit-shared failed:\n{stderr}");
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
