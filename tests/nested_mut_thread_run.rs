// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Nested-mutation threading RUNTIME gate (fix: batch A — findings #6 + #5(ii)).
//!
//! `stmt_exit_rebindings` (src/eval/lower.rs) threads a nested `while`/`if`
//! statement's outer-var mutation back into the enclosing env at the three
//! statement contexts that previously dropped it (module-item, value-`Block`,
//! `lower_stmt_seq`/`region`). The helper emits ZERO instructions — a program
//! without a nested-mutation-then-read shape lowers byte-identically; only the
//! silent-stale-read (buggy) shapes change.
//!
//! Covered here (all executing, compiled output == expected):
//!   [s2] value-`Block` else-stmt arm — a block-valued `match` arm whose body
//!        mutates a block-local `x` inside an `if`, then reads it → 5 (was 0).
//!   [s3] `region { }` (`lower_stmt_seq` other arm) — a `while` mutates `x`,
//!        read after the loop inside the region → 4 (was 0).
//!   [ctrl] genuine loop-carried `while` var is still threaded (the #5(ii)
//!        `env.contains_key` filter keeps names that pre-exist in env) → 3.
//!
//! #5(ii) (the `for`-counter scope-leak half) and #9 (undefined-identifier
//! fail-closed panic) are asserted separately — the leak is rejected at
//! type-check (`return j` where `j` is a synthesized for-counter → E2002), and
//! the #9 panic is a `#[should_panic]` unit test in `src/eval/lower.rs` that
//! constructs the module AST directly (source-level tests cannot reach it past
//! the front-end unknown-identifier check).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test nested_mut_thread_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// [s2] value-Block else-stmt arm: the `_ =>` arm body is a Node::Block; the
// inner `if c != 0 { x = 5 }` statement's branch binding must thread back into
// the block env so the trailing `x` reads 5, not the pre-if 0.
pub fn s2(c: i64) -> i64 {
    match c {
        0 => 0,
        _ => {
            let mut x = 0
            if c != 0 {
                x = 5
            }
            x
        }
    }
}

// [s3] region { } / lower_stmt_seq other arm: the while mutates x; after the
// loop, x must read the loop-exit id (4), not the pre-loop 0.
pub fn s3() -> i64 {
    region {
        let mut x = 0
        let mut i = 0
        while i < 4 {
            x = x + 1
            i = i + 1
        }
        x
    }
}

// [ctrl] the #5(ii) env.contains_key filter must KEEP a genuine loop-carried
// var (it pre-exists in the enclosing env) — this still threads → 3.
pub fn ctrl() -> i64 {
    let mut x = 0
    while x < 3 {
        x = x + 1
    }
    return x
}
"#;

#[test]
fn nested_mutation_threads_at_all_sites() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("nested-mut-thread-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_nested_mut_thread_run.mind");
    let so = dir.join("mind_nested_mut_thread_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("nested-mut-thread-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("nested-mut-thread-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.s2.restype = ctypes.c_int64; lib.s2.argtypes = [ctypes.c_int64]\n\
         lib.s3.restype = ctypes.c_int64; lib.s3.argtypes = []\n\
         lib.ctrl.restype = ctypes.c_int64; lib.ctrl.argtypes = []\n\
         r = lib.s2(1);  assert r == 5, '[s2] value-block if-mutation: got ' + str(r) + ' (stale-read bug -> 0)'\n\
         r = lib.s2(0);  assert r == 0, '[s2] guard arm: got ' + str(r)\n\
         r = lib.s3();   assert r == 4, '[s3] region while-mutation: got ' + str(r) + ' (stale-read bug -> 0)'\n\
         r = lib.ctrl(); assert r == 3, '[ctrl] loop-carried filter kept genuine var: got ' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "nested-mut-thread-run value check failed (silent stale-read regression?):\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// #5(ii): a synthesized `for`-counter must NOT leak a usable binding into the
/// enclosing scope. Reading it post-loop is rejected at type-check (E2002),
/// consistent with the interpreter erroring on the same program — never a
/// silent stale value.
#[test]
fn for_counter_does_not_leak() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("for-counter-leak: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_for_counter_leak.mind");
    std::fs::write(
        &src,
        "pub fn leaky() -> i64 {\n    for j in 0..2 {\n    }\n    return j\n}\n",
    )
    .expect("write src");
    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-ir"])
        .output()
        .expect("run mindc");
    assert!(
        !out.status.success(),
        "for-counter-leak: reading a for-counter post-loop must FAIL (leak regression)"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unknown identifier `j`") || stderr.contains("undefined identifier `j`"),
        "for-counter-leak: expected a loud unbound-`j` diagnostic, got:\n{stderr}"
    );
}
