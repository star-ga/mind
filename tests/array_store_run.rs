// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Run-test for fixed `[T; N]` / const-literal-tensor `a[i] = v`
//! (`Instr::ArrayStore`, #320 Step D aggregate mutation, the fix-the-write).
//!
//! Two guarantees, both proven by COMPILE + RUN (not a byte-probe), because
//! `main.mind` self-uses zero surface `a[i]=v` — so no keystone / oracle-parity /
//! cross_substrate gate exercises this path, and a mis-lower would otherwise ship:
//!
//!  1. A straight-line store RUNS correctly: a later read observes the write
//!     (value-semantic — the fresh post-store aggregate incarnation is rebound to
//!     the receiver name), and an untouched sibling element is unchanged.
//!  2. A store inside a loop / branch body FAILS CLOSED (loud), never silently
//!     drops the write. The F2 region-exit rebind that would make loop/branch
//!     `a[i]=v` correct is the tracked follow-on; until then the compiler must
//!     REJECT it, not emit a store whose effect is silently lost (the exact
//!     silent-miscompile the whole #320 line exists to prevent).

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

/// Compile a `fn main() -> i64` and run it; returns its real exit code (the
/// masked return value). Panics if no real artifact is produced (a masked
/// compile failure that emits a launcher stub is caught, never a clean run).
fn build_and_run(mindc: &Path, src: &str) -> i32 {
    let dir = std::env::temp_dir().join(format!("mind_arrstore_{}", src.len()));
    let _ = std::fs::create_dir_all(&dir);
    let srcf = dir.join("main.mind");
    let out = dir.join("p.bin");
    let _ = std::fs::remove_file(&out);
    std::fs::write(&srcf, src).expect("write src");
    let c = Command::new(mindc)
        .arg("build")
        .arg("--emit=binary")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(format!("--out={}", out.display()))
        .arg(&srcf)
        .output()
        .expect("run mindc build");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&c.stderr),
        String::from_utf8_lossy(&c.stdout)
    );
    assert!(
        !blob.contains("not natively compiled")
            && out.exists()
            && out.metadata().unwrap().len() > 0,
        "compile did not produce a real binary:\n{blob}"
    );
    let mut perm = std::fs::metadata(&out).unwrap().permissions();
    perm.set_mode(0o755);
    std::fs::set_permissions(&out, perm).unwrap();
    Command::new(&out)
        .output()
        .expect("run artifact")
        .status
        .code()
        .unwrap_or(-1)
}

/// True iff the build FAILS to produce a runnable artifact (fail-closed): a
/// non-zero mindc exit or no/empty output binary. Used to prove that an
/// unsupported `a[i]=v` position is rejected LOUDLY, never silently miscompiled.
fn build_fails(mindc: &Path, src: &str) -> bool {
    let dir = std::env::temp_dir().join(format!("mind_arrstore_fc_{}", src.len()));
    let _ = std::fs::create_dir_all(&dir);
    let srcf = dir.join("main.mind");
    let out = dir.join("p.bin");
    let _ = std::fs::remove_file(&out);
    std::fs::write(&srcf, src).expect("write src");
    let c = Command::new(mindc)
        .arg("build")
        .arg("--emit=binary")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(format!("--out={}", out.display()))
        .arg(&srcf)
        .output()
        .expect("run mindc build");
    !c.status.success() || !out.exists() || out.metadata().map(|m| m.len() == 0).unwrap_or(true)
}

#[test]
fn array_store_straight_line_runs_correctly() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        eprintln!("skip: release mindc not built");
        return;
    }
    // a[0] = 9; read it back -> 9 (the write is observed, not dropped).
    assert_eq!(
        build_and_run(
            &mindc,
            "fn main() -> i64 { let mut a: [i64; 3] = [1, 2, 3]; a[0] = 9; return a[0]; }"
        ),
        9
    );
    // a[1] = 9 -> 9 (non-zero slot).
    assert_eq!(
        build_and_run(
            &mindc,
            "fn main() -> i64 { let mut a: [i64; 3] = [1, 2, 3]; a[1] = 9; return a[1]; }"
        ),
        9
    );
    // two independent writes then sum -> 90 (both observed).
    assert_eq!(
        build_and_run(
            &mindc,
            "fn main() -> i64 { let mut a: [i64; 3] = [1, 2, 3]; a[0] = 40; a[1] = 50; return a[0] + a[1]; }"
        ),
        90
    );
    // const-literal `tensor<i64[N]>` store -> 9 (same value-semantic path).
    assert_eq!(
        build_and_run(
            &mindc,
            "fn main() -> i64 { let mut t: tensor<i64[3]> = [1, 2, 3]; t[0] = 9; return t[0]; }"
        ),
        9
    );
    // write a[0], read the UNTOUCHED a[1] -> 2 (no aliasing bleed: the store
    // rebinds `a` but leaves other elements at their initializer value).
    assert_eq!(
        build_and_run(
            &mindc,
            "fn main() -> i64 { let mut a: [i64; 3] = [1, 2, 3]; a[0] = 9; return a[1]; }"
        ),
        2
    );
}

#[test]
fn array_store_in_loop_or_branch_fails_closed_never_silent() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        eprintln!("skip: release mindc not built");
        return;
    }
    // for-loop body: without the F2 region-exit rebind the post-loop read would
    // see the PRE-loop value (verified: it returned 0, a silent miscompile). The
    // compiler must REJECT it instead — fail-closed, never a silent 0.
    assert!(
        build_fails(
            &mindc,
            "fn main() -> i64 { let mut a: [i64; 3] = [0, 0, 0]; for k in 0..3 { a[k] = k * 10; } return a[2]; }"
        ),
        "a[i]=v inside a for-loop body must fail closed, not silently drop the write"
    );
    // while-loop body: same guarantee.
    assert!(
        build_fails(
            &mindc,
            "fn main() -> i64 { let mut a: [i64; 3] = [0, 0, 0]; let mut k: i64 = 0; while k < 3 { a[k] = 7; k = k + 1; } return a[0]; }"
        ),
        "a[i]=v inside a while-loop body must fail closed"
    );
}
