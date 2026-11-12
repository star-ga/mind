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
        crate::common::gate::skipped("f64_loop_run", "f64-loop-run: mindc not found; skipping");
        return;
    }
    let dir = common::scratch_dir("f64_loop_run");
    let src = dir.join("mind_f64_loop_run.mind");
    let so = dir.join("mind_f64_loop_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !crate::common::gate::compiled("f64_loop_run", &out) {
        return;
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

/// A float literal bound to a `let` declared `f32` takes the declared width
/// (before: it stayed an `f64` constant, so `return a` in a `-> f32` fn was an
/// `mlir-opt` type error). `t_f32_widen` reads the rounded `f32` value back
/// through `f64`, so a value that only changed its label cannot pass. The
/// narrow integer lets wrap at the declared width (docs/errors.md).
#[test]
fn typed_let_takes_the_declared_width() {
    const LETS: &str = "pub fn t_f32_let() -> f32 {\n    let a: f32 = 1.5\n    return a + a\n}\n\
        pub fn t_f32_neg() -> f32 {\n    let a: f32 = -0.1\n    return a\n}\n\
        pub fn t_f32_expr() -> f32 {\n    let a: f32 = 0.1 * 3.0\n    return a\n}\n\
        pub fn t_f32_widen() -> f64 {\n    let a: f32 = 0.1\n    return a as f64\n}\n\
        pub fn t_f64_let() -> f64 {\n    let b: f64 = 0.1\n    return b\n}\n\
        pub fn t_i8_wrap() -> i64 {\n    let a: i8 = 300\n    return a\n}\n\
        pub fn t_u16_wrap() -> i64 {\n    let b: u16 = 70000\n    return b\n}\n\
        pub fn t_i8_neg_wrap() -> i64 {\n    let c: i8 = -129\n    return c\n}\n";
    let dir = common::scratch_dir("typed-let-width");
    let (src, so) = (dir.join("t.mind"), dir.join("t.so"));
    std::fs::write(&src, LETS).expect("write src");
    let out = Command::new(mindc_bin())
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !crate::common::gate::compiled("typed_let_width", &out) {
        return;
    }
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f, d, q = ctypes.c_float, ctypes.c_double, ctypes.c_int64\n\
         rows = [('t_f32_let', f, 3.0), ('t_f32_neg', f, f(-0.1).value),\n\
         \x20   ('t_f32_expr', f, f(0.1 * 3.0).value), ('t_f32_widen', d, f(0.1).value),\n\
         \x20   ('t_f64_let', d, 0.1), ('t_i8_wrap', q, 44), ('t_u16_wrap', q, 4464),\n\
         \x20   ('t_i8_neg_wrap', q, 127)]\n\
         bad = []\n\
         for name, ty, want in rows:\n\
         \x20   fn = getattr(lib, name); fn.restype = ty; got = fn()\n\
         \x20   bad += [] if got == want else [(name, got, want)]\n\
         print('ok' if not bad else bad)\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "ok",
        "typed-let widths; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Build `src` to a `.so` and call each row through ctypes. A row is
/// `(fn, restype, [ctypes args], want)` in Python; `f`/`d`/`q` are `c_float`,
/// `c_double` and `c_int64`, and `W` is `0.1` rounded to `f32` and widened back
/// (`0.10000000149011612`), so a value that kept its `f64` width cannot pass.
fn so_rows(target: &str, src: &str, rows: &str) {
    let dir = common::scratch_dir(target);
    let (mind, so) = (dir.join("t.mind"), dir.join("t.so"));
    std::fs::write(&mind, src).expect("write src");
    let out = Command::new(mindc_bin())
        .args([
            mind.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    if !crate::common::gate::compiled(target, &out) {
        return;
    }
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f, d, q = ctypes.c_float, ctypes.c_double, ctypes.c_int64\n\
         W = f(0.1).value\n\
         assert W == 0.10000000149011612\n\
         bad = []\n\
         for name, ty, args, want in [{rows}]:\n\
         \x20   fn = getattr(lib, name); fn.restype = ty\n\
         \x20   fn.argtypes = [type(a) for a in args]; got = fn(*args)\n\
         \x20   bad += [] if got == want else [(name, got, want)]\n\
         print('ok' if not bad else bad)\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "ok",
        "{target}; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The declared width governs a `let`: an `f64` value that is not a literal (a
/// parameter, a call result, a cast, an expression over them) bound to an `f32`
/// let converts too. Before: only a literal converted, and these were `mlir-opt`
/// type errors.
#[test]
fn f32_let_of_an_f64_value_converts() {
    const SRC: &str = "fn g64() -> f64 {\n    return 0.1\n}\n\
        pub fn l_param(x: f64) -> f64 {\n    let a: f32 = x\n    return a as f64\n}\n\
        pub fn l_call() -> f64 {\n    let a: f32 = g64()\n    return a as f64\n}\n\
        pub fn l_conv(n: i64) -> f64 {\n    let a: f32 = (n as f64) / 3.0\n    return a as f64\n}\n\
        pub fn l_expr(x: f64) -> f64 {\n    let a: f32 = x * 3.0\n    return a as f64\n}\n";
    so_rows(
        "f32-let-f64-value",
        SRC,
        "('l_param', d, [d(0.1)], W), ('l_call', d, [], W), \
         ('l_conv', d, [q(1)], f(1 / 3).value), ('l_expr', d, [d(0.1)], f(0.1 * 3.0).value)",
    );
}

/// The declared width governs a `return` in a `-> f32` fn: an `f64` value is
/// converted (`arith.truncf`) at the return, in a branch too. Before: `mlir-opt`
/// refused the module (return operand `f64` vs result `f32`).
#[test]
fn f32_return_takes_the_declared_width() {
    const SRC: &str = "fn g64() -> f64 {\n    return 0.1\n}\n\
        pub fn r_lit() -> f32 {\n    return 0.1\n}\n\
        pub fn r_param(x: f64) -> f32 {\n    return x\n}\n\
        pub fn r_call() -> f32 {\n    return g64()\n}\n\
        pub fn r_branch(x: f64, c: i64) -> f32 {\n    if c > 0 {\n        return x\n    }\n    return 0.2\n}\n\
        pub fn r_wide() -> f64 {\n    return r_lit() as f64\n}\n";
    so_rows(
        "f32-return",
        SRC,
        "('r_lit', f, [], W), ('r_param', f, [d(0.1)], W), ('r_call', f, [], W), \
         ('r_branch', f, [d(0.1), q(1)], W), ('r_branch', f, [d(0.1), q(0)], f(0.2).value), \
         ('r_wide', d, [], 0.10000000149011612)",
    );
}

/// The declared width governs an argument: an `f64` literal, variable or
/// expression passed to an `f32` parameter of a MIND fn is converted at the call
/// (`arith.truncf`). Before: `mlir-opt` refused the call (`f32` vs `f64`).
#[test]
fn f32_argument_takes_the_declared_width() {
    const SRC: &str = "fn h(x: f32) -> f64 {\n    return x as f64\n}\n\
        pub fn a_lit() -> f64 {\n    return h(0.1)\n}\n\
        pub fn a_param(x: f64) -> f64 {\n    return h(x)\n}\n\
        pub fn a_expr(x: f64) -> f64 {\n    return h(x * 3.0)\n}\n";
    so_rows(
        "f32-argument",
        SRC,
        "('a_lit', d, [], W), ('a_param', d, [d(0.1)], W), \
         ('a_expr', d, [d(0.1)], f(0.1 * 3.0).value)",
    );
}

/// The declared width governs an assignment: `a = <f64 value>` on an `f32`
/// binding converts, in a branch and a loop body too.
/// Before: the assigned value stayed `f64` (`mlir-opt` type error).
#[test]
fn f32_assignment_takes_the_declared_width() {
    const SRC: &str = "pub fn s_lit() -> f64 {\n    let mut a: f32 = 1.5\n    a = 0.1\n    return a as f64\n}\n\
        pub fn s_param(x: f64) -> f64 {\n    let mut a: f32 = 1.5\n    a = x\n    return a as f64\n}\n\
        pub fn s_expr(x: f64) -> f64 {\n    let mut a: f32 = 1.5\n    a = x * 3.0\n    return a as f64\n}\n\
        pub fn s_branch(c: i64) -> f64 {\n    let mut a: f32 = 1.5\n    if c > 0 {\n        a = 0.1\n    }\n    return a as f64\n}\n\
        pub fn s_loop(n: i64) -> f64 {\n    let mut a: f32 = 1.5\n    let mut i: i64 = 0\n    while i < n {\n        a = 0.1\n        i = i + 1\n    }\n    return a as f64\n}\n";
    so_rows(
        "f32-assign",
        SRC,
        "('s_lit', d, [], W), ('s_param', d, [d(0.1)], W), \
         ('s_expr', d, [d(0.1)], f(0.1 * 3.0).value), \
         ('s_branch', d, [q(1)], W), ('s_branch', d, [q(0)], 1.5), \
         ('s_loop', d, [q(3)], W), ('s_loop', d, [q(0)], 1.5)",
    );
}

/// A top-level `let g: f32 = 0.1` type-checks and builds, as the same `let` in a
/// fn body does (before: E2001, annotation `f32` vs inferred `f64`). An integer
/// value is still refused (no implicit int to float conversion).
#[test]
fn top_level_f32_let_type_checks_and_builds() {
    let codes = |src: &str| -> Vec<String> {
        let module = libmind::parser::parse(src).expect("parses");
        let env = libmind::type_checker::TypeEnv::default();
        libmind::type_checker::check_module_types_in_file(&module, src, None, &env)
            .into_iter()
            .map(|d| d.code.to_string())
            .collect()
    };
    let top = codes("let g: f32 = 0.1\n");
    assert!(
        !top.iter().any(|c| c == "E2001"),
        "f64 into a top-level f32 let: {top:?}"
    );
    let int = codes("let g: f32 = 1\n");
    assert!(
        int.iter().any(|c| c == "E2001"),
        "an integer is refused: {int:?}"
    );
    const SRC: &str = "let g: f32 = 0.1\n\
        pub fn t() -> f64 {\n    let a: f32 = 0.1\n    return a as f64\n}\n";
    so_rows("f32-top-level-let", SRC, "('t', d, [], W)");
}

/// `mindc test` (the tree evaluator) and a built `.so` agree on an `f32` binding:
/// both hold the `f32` nearest the value (`0.10000000149011612` for `0.1`) at a
/// `let`, a parameter and a return. Before: the evaluator kept the `f64` (`0.1`).
#[test]
fn evaluator_and_artifact_agree_on_an_f32_binding() {
    const SRC: &str = "pub fn e_let() -> f64 {\n    let a: f32 = 0.1\n    return a as f64\n}\n\
        fn p(x: f32) -> f64 {\n    return x as f64\n}\n\
        pub fn e_param() -> f64 {\n    return p(0.1)\n}\n\
        fn r() -> f32 {\n    return 0.1\n}\n\
        pub fn e_ret() -> f64 {\n    return r() as f64\n}\n";
    for probe in ["e_let", "e_param", "e_ret"] {
        let src = format!("{SRC}let __probe: f64 = {probe}()\n");
        let module = libmind::parser::parse(&src).expect("parses");
        let mut env = std::collections::HashMap::new();
        let got = libmind::eval::eval_module_value_with_env(&module, &mut env, None)
            .unwrap_or_else(|e| panic!("tree-eval of {probe}() failed: {e}"));
        match got {
            libmind::eval::Value::Float(x) => assert_eq!(x, 0.10000000149011612, "{probe}"),
            other => panic!("{probe}: not a float: {other:?}"),
        }
    }
    so_rows(
        "f32-eval-parity",
        SRC,
        "('e_let', d, [], 0.10000000149011612), ('e_param', d, [], 0.10000000149011612), \
         ('e_ret', d, [], 0.10000000149011612)",
    );
}
