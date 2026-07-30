// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Static-dispatch trait/impl RUNTIME + fail-closed gate (#268 Phase 1).
//!
//! Worked example: a `trait Speak { fn val(self) -> i64 }` with
//! `impl Speak for Foo { fn val(self) -> i64 { self.x + self.y } }` is desugared
//! to an ordinary free fn `foo_val(self: Foo)`; a `foo.val()` call on a
//! statically-known `Foo` receiver resolves through the EXISTING UFCS method
//! lowering to a DIRECT call — no vtable, no dyn, no trait object. This proves
//! the value roundtrips (returns 42), not merely that it parses.
//!
//! Fail-closed witness: a `foo.val()` call where `val` is a declared trait method
//! but `Foo` has NO `impl` is REJECTED at compile time with the `E2708`
//! diagnostic — never a silent const-0 miscompile.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test trait_static_dispatch_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC_OK: &str = r#"
struct Foo { x: i64, y: i64 }

trait Speak {
    fn val(self) -> i64
}

impl Speak for Foo {
    fn val(self) -> i64 {
        return self.x + self.y
    }
}

// Static dispatch: `foo.val()` on a known-`Foo` receiver resolves to the impl
// method `foo_val(foo)` — 40 + 2 = 42.
pub fn run() -> i64 {
    let foo = Foo { x: 40, y: 2 }
    return foo.val()
}
"#;

// A trait method `val` is declared, `Foo` is constructed, but there is NO
// `impl Speak for Foo`. Calling `foo.val()` must be rejected fail-closed.
const SRC_NO_IMPL: &str = r#"
struct Foo { x: i64 }

trait Speak {
    fn val(self) -> i64
}

pub fn run() -> i64 {
    let foo = Foo { x: 7 }
    return foo.val()
}
"#;

#[test]
fn trait_static_dispatch_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("trait-static-dispatch-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_trait_static_dispatch_run.mind");
    let so = dir.join("mind_trait_static_dispatch_run.so");
    std::fs::write(&src, SRC_OK).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("trait-static-dispatch-run: needs mlir-build; skipping");
            return;
        }
        panic!("trait-static-dispatch-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 42, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "trait-static-dispatch-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn trait_call_without_impl_is_rejected() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("trait-no-impl: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_trait_no_impl.mind");
    let so = dir.join("mind_trait_no_impl.so");
    std::fs::write(&src, SRC_NO_IMPL).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    // Must FAIL to compile — never a silent const-0.
    assert!(
        !out.status.success(),
        "trait-no-impl: expected compile failure for a trait-method call with no impl, but it succeeded"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("E2708") || stderr.contains("no implementation of trait"),
        "trait-no-impl: expected an E2708 no-impl diagnostic, got:\n{stderr}"
    );
}
