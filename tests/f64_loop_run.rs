// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Loop-carried `f64` `while`-loop RUNTIME gate.
//!
//! A `while` loop whose carried variable is `f64` (or a mix of `f64` and `i64`)
//! must thread the REAL loop-carried type through the `cf.br`/`cf.cond_br` edge
//! operand lists, the header/body/after block-arg declarations, AND the exit-id
//! kind that types a post-loop `return`. A hardcoded `i64` on any of those makes
//! mlir-opt reject the edge with `i64 vs f64`. This compiles a program with
//! f64-carried loops to a `.so`, dlopen-calls it, and asserts the float results.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test f64_loop_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// Loop-carried f64 ADD: y starts 0.0, adds 1.5 n times.
pub fn f64sum(n: i64) -> f64 {
    let mut y: f64 = 0.0
    let mut i: i64 = 0
    while i < n { y = y + 1.5  i = i + 1 }
    return y
}

// Loop-carried f64 MULTIPLY: p starts 1.0, doubles n times.
pub fn fprod(n: i64) -> f64 {
    let mut p: f64 = 1.0
    let mut i: i64 = 0
    while i < n { p = p * 2.0  i = i + 1 }
    return p
}

// Mixed f64/i64 loop shaped like a Lorenz Euler step: x,y,z f64 carried
// plus an i64 step counter.
pub fn lorenz(steps: i64) -> f64 {
    let mut x: f64 = 1.0
    let mut y: f64 = 1.0
    let mut z: f64 = 1.0
    let mut i: i64 = 0
    let dt: f64 = 0.01
    let sigma: f64 = 10.0
    let rho: f64 = 28.0
    let beta: f64 = 2.666
    while i < steps {
        let dx: f64 = sigma * (y - x)
        let dy: f64 = x * (rho - z) - y
        let dz: f64 = x * y - beta * z
        x = x + dt * dx
        y = y + dt * dy
        z = z + dt * dz
        i = i + 1
    }
    return x + y + z
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn f64_loop_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("f64-loop-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_f64_loop_run.mind");
    let so = dir.join("mind_f64_loop_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("f64-loop-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("f64-loop-run: mindc --emit-shared failed:\n{stderr}");
    }

    // Euler-integrated Lorenz reference, computed the same way the .mind does,
    // so the assertion is an exact f64 bit-match (single-substrate).
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.f64sum.restype = ctypes.c_double; lib.f64sum.argtypes = [ctypes.c_int64]\n\
         lib.fprod.restype = ctypes.c_double; lib.fprod.argtypes = [ctypes.c_int64]\n\
         lib.lorenz.restype = ctypes.c_double; lib.lorenz.argtypes = [ctypes.c_int64]\n\
         r = lib.f64sum(4); assert r == 6.0, 'f64sum(4)=' + repr(r)\n\
         r = lib.f64sum(10); assert r == 15.0, 'f64sum(10)=' + repr(r)\n\
         r = lib.fprod(10); assert r == 1024.0, 'fprod(10)=' + repr(r)\n\
         def ref(steps):\n\
        \x20   x=y=z=1.0; dt=0.01; sigma=10.0; rho=28.0; beta=2.666\n\
        \x20   for _ in range(steps):\n\
        \x20       dx=sigma*(y-x); dy=x*(rho-z)-y; dz=x*y-beta*z\n\
        \x20       x=x+dt*dx; y=y+dt*dy; z=z+dt*dz\n\
        \x20   return x+y+z\n\
         for s in (0, 1, 50, 200):\n\
        \x20   g = lib.lorenz(s); e = ref(s)\n\
        \x20   assert g == e, 'lorenz(' + str(s) + ')=' + repr(g) + ' expected ' + repr(e)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "f64-loop-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
