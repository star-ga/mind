// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native admission of the frozen emitter's own intrinsics (W1) and std seed
//! linkage (W3/W3b), reduced from the two refusal shapes the MIND-Lab native
//! matrix measured on e9737689:
//!
//! * `decision` / `numeric`: `main calls __mind_alloc, which has no definition
//!   in the closure (closure.unresolved_edge)` — the program only uses the
//!   memory / I-O intrinsics `stage1.elf` already expands inline;
//! * `evidence` and seven more: `bsig0 collides with a std seed-blob symbol
//!   (closure.std_symbol_collision)` — the program carries its own SHA-256
//!   compression helpers, whose names std.sha256 also uses, and never imports
//!   std.sha256.
//!
//! Every positive case runs the artifact and checks its exit code; every negative
//! case pins the refusal label, so a widened admission cannot pass silently.
#![cfg(all(feature = "cross-module-imports", feature = "std-surface"))]

mod native_bridge_support;
use native_bridge_support::{Project, assert_native_result};

fn build_ok(name: &str, src: &str, expected: i32) {
    let p = Project::new(name);
    p.write("src/main.mind", src);
    let (code, stderr, bytes) = p.build_native("src/main.mind");
    assert_eq!(code, 0, "{name}: expected a native build:\n{stderr}");
    let bytes = bytes.expect("artifact");
    assert_native_result(&p, &bytes, expected, name);
}

fn refused(name: &str, src: &str, kind: &str) {
    let p = Project::new(name);
    p.write("src/main.mind", src);
    let (code, stderr, bytes) = p.build_native("src/main.mind");
    assert_eq!(code, 3, "{name}: expected a refusal (exit 3):\n{stderr}");
    assert!(
        stderr.contains(kind),
        "{name}: expected `{kind}`, got:\n{stderr}"
    );
    assert!(bytes.is_none(), "{name}: a refused build wrote an artifact");
}

// ---- W1: the decision / numeric shape --------------------------------------

/// Allocate, write bytes, write them to stdout with the `-1` offset sentinel,
/// read them back and exit with a value computed from all of it.
const IO_SHAPE: &str = "\
fn fill(p: i64, n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        __mind_store_i8(p + i, 65 + i);
        i = i + 1;
    }
    0
}

fn main() -> i64 {
    let p: i64 = __mind_alloc(32);
    fill(p, 4);
    __mind_store_i8(p + 4, 10);
    let mut sent: i64 = 0;
    while sent < 5 {
        let n: i64 = __mind_write(1, p + sent, 5 - sent, 0 - 1);
        sent = sent + n;
    }
    __mind_store_i64(p + 8, 1000);
    let q: i64 = __mind_realloc(p, 64);
    let kept: i64 = __mind_load_i64(q + 8) + __mind_load_i8(q + 3);
    let freed: i64 = __mind_free(q);
    return kept - 1000 + sent + freed;
}
";

#[test]
fn the_memory_and_io_intrinsics_build_and_run() {
    // 'D' (68) preserved across realloc, 5 bytes sent, free returns 0.
    build_ok("w1_io_shape", IO_SHAPE, 68 + 5);
}

/// The narrow loads ZERO-extend, the narrow stores truncate, and the wrap
/// intrinsics sign-extend from their width.
#[test]
fn narrow_memory_and_wrap_intrinsics_build_and_run() {
    build_ok(
        "w1_narrow",
        "fn main() -> i64 {
            let p: i64 = __mind_alloc(16);
            __mind_store_i64(p, 0 - 1);
            __mind_store_i16(p, 70000);
            __mind_store_i32(p + 8, 0 - 2);
            let a: i64 = __mind_load_i16(p);
            let b: i64 = __mind_load_i32(p + 8);
            let w: i64 = __mind_wrap_i8(300) + __mind_wrap_i16(65535) + __mind_wrap_i32(4294967297);
            if a != 4464 { return 1; }
            if b != 4294967294 { return 2; }
            if __mind_load_i8(p + 2) != 255 { return 3; }
            return w + 100;
        }\n",
        // wrap_i8(300) = 44, wrap_i16(65535) = -1, wrap_i32(2^32 + 1) = 1.
        144,
    );
}

#[test]
fn reading_a_store_result_is_refused() {
    refused(
        "w1_discard",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); \
         let r: i64 = __mind_store_i64(p, 5); return r + 3; }\n",
        "call.discard_only_result_used",
    );
}

#[test]
fn an_io_offset_not_proven_minus_one_is_refused() {
    refused(
        "w1_offset",
        "fn rd(p: i64, off: i64) -> i64 { return __mind_read(0, p, 8, off); }\n\
         fn main() -> i64 { let p: i64 = __mind_alloc(8); return rd(p, 0 - 1); }\n",
        "call.io_offset_not_minus_one",
    );
    refused(
        "w1_offset_zero",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); \
         return __mind_write(1, p, 1, 0); }\n",
        "call.io_offset_not_minus_one",
    );
}

#[test]
fn a_program_definition_of_an_intrinsic_is_refused() {
    refused(
        "w1_redefine",
        "fn __mind_alloc(x: i64) -> i64 { return 0; }\n\
         fn main() -> i64 { let p: i64 = __mind_alloc(8); return 1; }\n",
        "closure.intrinsic_redefinition",
    );
}

/// A name `nb_intrinsic_id` does not know stays out of scope (seed item W8c).
#[test]
fn an_intrinsic_outside_the_frozen_table_stays_refused() {
    refused(
        "w1_conv",
        "fn main() -> i64 { let x: i64 = 7; return __mind_bits_to_f64(x) as i64; }\n",
        "__mind_bits_to_f64",
    );
}

/// Round-2 blocker 1: `__mind_alloc(n <= 0)` is 0 on the C runtime and a fresh
/// arena pointer natively, so a size that is not proven positive is refused.
#[test]
fn an_alloc_size_not_proven_positive_is_refused() {
    for (name, src) in [
        (
            "w1_alloc_zero",
            "fn main() -> i64 {\n    let p: i64 = __mind_alloc(0);\n    if p == 0 {\n        \
             return 1;\n    }\n    return 2;\n}\n",
        ),
        (
            "w1_alloc_negative",
            "fn main() -> i64 {\n    let p: i64 = __mind_alloc(16);\n    __mind_store_i64(p, 7);\n    \
             let q: i64 = __mind_alloc(0 - 16);\n    if q == 0 {\n        return 1;\n    }\n    \
             __mind_store_i64(q, 99);\n    return __mind_load_i64(p);\n}\n",
        ),
        (
            "w1_alloc_variable",
            "fn mk(n: i64) -> i64 {\n    return __mind_alloc(n);\n}\n\n\
             fn main() -> i64 {\n    let p: i64 = mk(0);\n    if p == 0 {\n        return 1;\n    }\n    \
             return 2;\n}\n",
        ),
        (
            "w1_realloc_zero",
            "fn main() -> i64 {\n    let p: i64 = __mind_alloc(16);\n    \
             let q: i64 = __mind_realloc(p, 0);\n    if q == 0 {\n        return 1;\n    }\n    \
             return 2;\n}\n",
        ),
    ] {
        refused(name, src, "call.alloc_size_not_proven_positive");
    }
}

/// Round-2 blocker 3: an f64 operand reached the frozen emitter as raw IEEE bits
/// (`wrap_i8(300.0)` ran and exited 0; `alloc(16.0)` exhausted the arena) while
/// MLIR refuses the type mismatch. An operand must be proven i64.
#[test]
fn a_float_operand_to_an_intrinsic_is_refused() {
    for (name, src, kind) in [
        (
            "w1_wrap_f64",
            "fn main() -> i64 {\n    let x: f64 = 300.0;\n    return __mind_wrap_i8(x);\n}\n",
            "call.intrinsic_operand_not_i64",
        ),
        (
            // The size is a float, so it is not a proven positive integer constant
            // either; the size check runs first and names itself.
            "w1_alloc_f64",
            "fn main() -> i64 {\n    let n: f64 = 16.0;\n    let p: i64 = __mind_alloc(n);\n    \
             __mind_store_i64(p, 3);\n    return __mind_load_i64(p);\n}\n",
            "call.alloc_size_not_proven_positive",
        ),
        (
            "w1_store_f64",
            "fn main() -> i64 {\n    let p: i64 = __mind_alloc(8);\n    let x: f64 = 1.5;\n    \
             __mind_store_i64(p, x);\n    let v: i64 = __mind_load_i64(p);\n    \
             if v == 1 {\n        return 1;\n    }\n    if v == 0 {\n        return 0;\n    }\n    \
             return 2;\n}\n",
            "call.intrinsic_operand_not_i64",
        ),
        (
            "w1_load_f64_addr",
            "fn main() -> i64 {\n    let p: f64 = 8.0;\n    return __mind_load_i8(p);\n}\n",
            "call.intrinsic_operand_not_i64",
        ),
    ] {
        refused(name, src, kind);
    }
}

/// The result side of the same rule: `let y: f64 = __mind_load_i64(p)` compared
/// the loaded bits as a double natively (exit 1) while the MLIR backend refuses the
/// program (`mlir-opt`: i64 vs f64). An intrinsic's result must stay an i64.
#[test]
fn an_intrinsic_result_used_as_a_float_is_refused() {
    refused(
        "w1_load_as_f64",
        "fn main() -> i64 {\n    let p: i64 = __mind_alloc(8);\n    \
         __mind_store_i64(p, 4609434218613702656);\n    let y: f64 = __mind_load_i64(p);\n    \
         if y == 1.5 {\n        return 1;\n    }\n    if y > 1000.0 {\n        return 3;\n    }\n    \
         return 2;\n}\n",
        "call.intrinsic_result_not_i64",
    );
}

/// Builds `src` with the MLIR backend and returns the artifact's exit code.
#[cfg(all(feature = "mlir-build", target_os = "linux", target_arch = "x86_64"))]
fn mlir_exit(p: &Project, name: &str) -> Option<i32> {
    let out = p.root().join("mlir.out");
    let res = std::process::Command::new(native_bridge_support::MINDC)
        .args(["build", "src/main.mind", "--out"])
        .arg(&out)
        .current_dir(p.root())
        .env("MINDC_STD_DIR", native_bridge_support::repo_path("std"))
        .output()
        .expect("spawn mindc (mlir)");
    if !res.status.success() {
        let stderr = String::from_utf8_lossy(&res.stderr);
        // A host without the MLIR toolchain cannot produce the reference; that is a
        // capability gap, reported loudly, and a hard failure when the gate demands it.
        assert!(
            std::env::var_os("MIND_BENCH_REQUIRE").is_none() && stderr.contains("PATH"),
            "{name}: the MLIR reference build failed:\n{stderr}"
        );
        eprintln!("{name}: MLIR toolchain unavailable, parity leg skipped:\n{stderr}");
        return None;
    }
    let bytes = std::fs::read(&out).expect("mlir artifact");
    Some(native_bridge_support::run(&bytes))
}

/// Round-2 blocker 2: on failure the frozen emitter's raw syscalls return
/// `-errno`, and so must the C runtime the MLIR backend links. Each program builds
/// on both backends and exits with the same value, equal to an oracle derived from
/// the Linux errno numbers (ENOENT 2, EBADF 9, EFAULT 14).
#[cfg(all(feature = "mlir-build", target_os = "linux", target_arch = "x86_64"))]
#[test]
fn io_error_results_match_the_mlir_backend() {
    let missing_path: String = b"/nonexistent-mind-dir/no/such/file\0"
        .iter()
        .enumerate()
        .map(|(i, b)| format!("    __mind_store_i8(path + {i}, {b});\n"))
        .collect();
    let dev_zero: String = b"/dev/zero\0"
        .iter()
        .enumerate()
        .map(|(i, b)| format!("    __mind_store_i8(path + {i}, {b});\n"))
        .collect();
    let open_missing = format!(
        "fn main() -> i64 {{\n    let path: i64 = __mind_alloc(64);\n{missing_path}    \
         let fd: i64 = __mind_open(path);\n    return fd + 100;\n}}\n"
    );
    let cases: [(&str, String, i32); 3] = [
        ("w1_open_missing", open_missing, 100 - 2),
        (
            "w1_bad_fd",
            "fn main() -> i64 {\n    let buf: i64 = __mind_alloc(8);\n    __mind_store_i8(buf, 65);\n    \
             let w: i64 = __mind_write(99, buf, 1, 0 - 1);\n    \
             let r: i64 = __mind_read(99, buf, 1, 0 - 1);\n    \
             let z: i64 = __mind_write(1, buf, 0, 0 - 1);\n    return w + 100 + r * 3 + z;\n}\n"
                .to_string(),
            -9 + 100 - 27,
        ),
        (
            "w1_null_and_empty",
            format!(
                "fn main() -> i64 {{\n    let path: i64 = __mind_alloc(16);\n{dev_zero}    \
                 let fd: i64 = __mind_open(path);\n    if fd < 3 {{\n        return 1;\n    }}\n    \
                 let buf: i64 = __mind_alloc(8);\n    \
                 let a: i64 = __mind_read(fd, 0, 8, 0 - 1);\n    \
                 let b: i64 = __mind_write(99, buf, 0, 0 - 1);\n    \
                 let c: i64 = __mind_write(1, buf, 0 - 5, 0 - 1);\n    \
                 let d: i64 = if c < 0 {{ 0 }} else {{ 50 }};\n    \
                 return 200 + a + b * 2 + d;\n}}\n"
            ),
            // read(fd, NULL, 8) on /dev/zero = -EFAULT; write(99, buf, 0) = -EBADF
            // (the fd is checked before the count); a negative count fails.
            200 - 14 - 18,
        ),
    ];
    for (name, src, expected) in cases {
        let p = Project::new(name);
        p.write("src/main.mind", &src);
        let (code, stderr, bytes) = p.build_native("src/main.mind");
        assert_eq!(code, 0, "{name}: expected a native build:\n{stderr}");
        let native = native_bridge_support::run(&bytes.expect("artifact"));
        assert_eq!(native, expected, "{name}: native exit");
        if let Some(mlir) = mlir_exit(&p, name) {
            assert_eq!(mlir, native, "{name}: MLIR exit != native exit");
        }
    }
}

// ---- W3 / W3b: the evidence shape and calls into std ----------------------

/// The evidence programs define their own SHA-256 helpers under names std.sha256
/// also uses, without importing std.sha256. The program's definitions win, and
/// no retained std code calls them.
#[test]
fn a_program_shadowing_an_unimported_std_name_builds_and_runs() {
    build_ok(
        "w3_shadow",
        "fn rotr32(x: i64, n: i64) -> i64 { return x + n; }\n\
         fn bsig0(x: i64) -> i64 { return rotr32(x, 2) + 1; }\n\
         fn main() -> i64 { return bsig0(39); }\n",
        42,
    );
}

/// A call into the seed std resolves as an edge and the std bodies run.
#[test]
fn a_call_into_seed_std_resolves_and_runs() {
    build_ok(
        "w3b_vec",
        "import std.vec;\n\n\
         fn main() -> i64 {\n    let v = vec_new();\n    \
         return vec_len(v) + vec_cap(v) + 42;\n}\n",
        42,
    );
}

/// `vec_push` grows through `__mind_realloc(v.addr, new_cap * 8)`, a size the
/// fence cannot prove positive, so the linked body is refused (the frozen arena
/// returns a pointer where the C runtime returns 0 for a size <= 0).
#[test]
fn a_linked_std_body_with_an_unproven_alloc_size_is_refused() {
    refused(
        "w3b_vec_push",
        "import std.vec;\n\n\
         fn main() -> i64 {\n    let v = vec_new();\n    let w = vec_push(v, 7);\n    \
         return vec_get(w, 0);\n}\n",
        "call.alloc_size_not_proven_positive",
    );
}

/// The program imports std.vec AND defines one of its names: which one a call
/// means is ambiguous, so it is refused with both names in the message.
#[test]
fn defining_a_name_of_an_imported_std_module_is_refused() {
    refused(
        "w3_import_collision",
        "import std.vec;\n\nfn vec_len(v: i64) -> i64 {\n    return 3;\n}\n\n\
         fn main() -> i64 {\n    return vec_len(0);\n}\n",
        "closure.std_symbol_collision",
    );
}

/// A retained std function calls a name the program redefines: the frozen
/// compiler would bind that call to the program's definition. Refused.
#[test]
fn a_std_function_calling_a_shadowed_name_is_refused() {
    refused(
        "w3_shadowed_callee",
        "import std.io;\n\nfn string_push_byte(s: i64, b: i64) -> i64 {\n    return b;\n}\n\n\
         fn main() -> i64 {\n    let s = string_push_ansi_sgr(string_push_byte(0, 1), 31);\n    \
         return 0;\n}\n",
        "closure.std_callee_shadowed",
    );
}

/// A linked std body is fenced like a user body: std.sha256's `rotr32` shifts.
#[test]
fn a_linked_std_body_outside_the_profile_is_refused() {
    refused(
        "w3_std_fence",
        "import std.sha256;\n\nfn main() -> i64 {\n    return bsig0(5);\n}\n",
        "links std seed function `rotr32`",
    );
}

// ---- Round 3: facts that hold only across a call ---------------------------

/// A declared `i64` parameter proves nothing on its own: the native bridge runs
/// no type checker, so an f64 passed through a wrapper reached the frozen
/// intrinsic as its IEEE bits (`w8(300.0)` exited 50 instead of being refused;
/// MLIR refuses it). The argument at every call site must be proven, and a
/// declared `i64` return must return proven values.
#[test]
fn a_float_reaching_an_intrinsic_through_a_wrapper_is_refused() {
    for (name, src, kind) in [
        (
            "r3_wrap_through_fn",
            "fn w8(x: i64) -> i64 { return __mind_wrap_i8(x); }\n\
             fn main() -> i64 { let f: f64 = 300.0; return w8(f) + 50; }\n",
            "call.intrinsic_operand_not_i64",
        ),
        (
            "r3_store_through_fn",
            "fn put(p: i64, v: i64) -> i64 { __mind_store_i64(p, v); return 0; }\n\
             fn main() -> i64 { let p: i64 = __mind_alloc(8); let f: f64 = 2.0; put(p, f); \
             return __mind_load_i64(p) & 255; }\n",
            "call.intrinsic_operand_not_i64",
        ),
        (
            "r3_write_through_fn",
            "fn emit(p: i64, n: i64) -> i64 { return __mind_write(1, p, n, 0 - 1); }\n\
             fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i8(p, 65); \
             let n: f64 = 1.0; let r: i64 = emit(p, n); if r == 1 { return 1; } return 2; }\n",
            "call.intrinsic_operand_not_i64",
        ),
        (
            "r3_f64_returned_as_i64",
            "fn g() -> i64 { let f: f64 = 2.5; return f; }\n\
             fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, g()); \
             if __mind_load_i64(p) == 2 { return 1; } return 3; }\n",
            "call.intrinsic_operand_not_i64",
        ),
        (
            // Through a linked std wrapper: std.io's print_bytes writes `count`.
            "r3_std_wrapper_f64",
            "import std.io;\nfn main() -> i64 { let p: i64 = __mind_alloc(8); \
             __mind_store_i8(p, 65); let n: f64 = 1.0; return print_bytes(p, n); }\n",
            "call.intrinsic_operand_not_i64",
        ),
        (
            // A linked std function with no intrinsic in it still takes an i64.
            "r3_std_arg_f64",
            "import std.sha256;\nfn main() -> i64 { let x: f64 = 1.5; \
             if mask32(x) == 0 { return 1; } return 2; }\n",
            "call.std_argument_not_i64",
        ),
    ] {
        refused(name, src, kind);
    }
}

/// Positive controls: the same wrappers called with integers build and run.
#[test]
fn integers_through_wrappers_build_and_run() {
    build_ok(
        "r3_wrap_int",
        "fn w8(x: i64) -> i64 { return __mind_wrap_i8(x); }\n\
         fn main() -> i64 { let a: i64 = 300; return w8(a) + 50; }\n",
        94,
    );
    build_ok(
        "r3_std_wrapper_int",
        "import std.io;\nfn main() -> i64 { let p: i64 = __mind_alloc(8); \
         __mind_store_i8(p, 10); let n: i64 = 1; return print_bytes(p, n) + 40; }\n",
        41,
    );
}

/// `__mind_free` is a no-op on the frozen bump arena, so native memory is the
/// total ever allocated while the C runtime's is the peak: a 2000 x 1 MB
/// alloc/free loop exited 70 ("native arena exhausted") natively and 33 under
/// MLIR. Admission now requires a static bound on the total.
#[test]
fn an_unbounded_or_oversized_allocation_total_is_refused() {
    refused(
        "r3_free_loop",
        "fn main() -> i64 { let mut i: i64 = 0; let mut a: i64 = 0; while i < 2000 { \
         let p: i64 = __mind_alloc(1000000); __mind_store_i64(p, i); \
         a = a + __mind_load_i64(p) - i; let f: i64 = __mind_free(p); a = a + f; \
         i = i + 1; } return a + 33; }\n",
        "alloc.in_loop",
    );
    refused(
        "r3_big_alloc",
        "fn main() -> i64 { let p: i64 = __mind_alloc(900000000); if p == 0 { return 5; } \
         __mind_store_i64(p + 899999992, 9); return __mind_load_i64(p + 899999992); }\n",
        "alloc.total_over_budget",
    );
    build_ok(
        "r3_free_once",
        "fn main() -> i64 { let p: i64 = __mind_alloc(1000000); __mind_store_i64(p, 30); \
         let v: i64 = __mind_load_i64(p); let f: i64 = __mind_free(p); return v + f + 3; }\n",
        33,
    );
}
