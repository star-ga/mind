// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Unit tests for the whole-program half of the W1 fence: operand proofs that
//! cross a call, and the static allocation bound.
#![cfg(feature = "std-surface")]

use crate::ir::frozen_profile::{FrozenProfileRejection, profile_frozen_admits};

fn lowered_admits(src: &str) -> Result<(), FrozenProfileRejection> {
    let module = crate::parser::parse(src).expect("test source parses");
    let ir = crate::eval::lower_to_ir(&module).expect("lowers");
    profile_frozen_admits(&ir)
}

fn refused_as(construct: &str, sources: &[&str]) {
    for src in sources {
        assert_eq!(
            lowered_admits(src).map_err(|r| r.construct),
            Err(construct),
            "{src}"
        );
    }
}

fn admitted(sources: &[&str]) {
    for src in sources {
        assert_eq!(lowered_admits(src), Ok(()), "{src}");
    }
}

/// A declared `i64` parameter is not a proof: the native bridge runs no type
/// checker, so `w8(300.0)` hands the frozen emitter the IEEE bits of 300.0
/// (wrap_i8 of them is 0; MLIR refuses the program). The argument at EVERY call
/// site must be proven, through any depth of wrappers, and a declared `i64`
/// return must return proven values.
#[test]
fn an_i64_parameter_or_return_is_proven_at_the_other_end_of_the_call() {
    refused_as(
        "call.intrinsic_operand_not_i64",
        &[
            // The verifier's a08 / a10 / a11 / a12.
            "fn w8(x: i64) -> i64 { return __mind_wrap_i8(x); } \
             fn main() -> i64 { let f: f64 = 300.0; return w8(f) + 50; }",
            "fn put(p: i64, v: i64) -> i64 { __mind_store_i64(p, v); return 0; } \
             fn main() -> i64 { let p: i64 = __mind_alloc(8); let f: f64 = 2.0; put(p, f); \
             return __mind_load_i64(p) & 255; }",
            "fn emit(p: i64, n: i64) -> i64 { return __mind_write(1, p, n, 0 - 1); } \
             fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i8(p, 65); \
             let n: f64 = 1.0; let r: i64 = emit(p, n); if r == 1 { return 1; } return 2; }",
            "fn g() -> i64 { let f: f64 = 2.5; return f; } \
             fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, g()); \
             if __mind_load_i64(p) == 2 { return 1; } return 3; }",
            // Two wrappers deep; one good call site does not cover a bad one.
            "fn w8(x: i64) -> i64 { return __mind_wrap_i8(x); } fn w(y: i64) -> i64 { return w8(y); } \
             fn main() -> i64 { let f: f64 = 1.5; return w(7) + w(f); }",
            // A tail value and a returned value in a branch both count.
            "fn g(c: i64) -> i64 { if c == 1 { return 4; } let f: f64 = 2.5; f } \
             fn main() -> i64 { return __mind_wrap_i8(g(1)); }",
            // Recursion does not prove itself.
            "fn r(n: i64) -> i64 { if n == 0 { let f: f64 = 1.5; return f; } return r(n - 1); } \
             fn main() -> i64 { return __mind_wrap_i8(r(3)); }",
        ],
    );
    admitted(&[
        "fn w8(x: i64) -> i64 { return __mind_wrap_i8(x); } \
         fn main() -> i64 { let a: i64 = 300; return w8(a) + 50; }",
        "fn w8(x: i64) -> i64 { return __mind_wrap_i8(x); } fn w(y: i64) -> i64 { return w8(y); } \
         fn main() -> i64 { return w(7) + w(300); }",
        "fn g(c: i64) -> i64 { if c == 1 { return 4; } 5 } \
         fn main() -> i64 { return __mind_wrap_i8(g(1)); }",
        "fn r(n: i64) -> i64 { if n == 0 { return 1; } return r(n - 1) + 1; } \
         fn main() -> i64 { return __mind_wrap_i8(r(3)); }",
        // An unrelated function returning an unproven value is not inspected.
        "fn h() -> i64 { let f: f64 = 2.5; return f; } \
         fn main() -> i64 { return __mind_wrap_i8(300); }",
    ]);
}

/// The frozen bump arena never reuses memory (`__mind_free` is a no-op), so
/// native memory is the total ever allocated while `malloc`'s is the peak. A
/// program is admitted only with a static proof that its total fits the budget.
#[test]
fn total_allocation_must_be_statically_bounded() {
    refused_as(
        "alloc.in_loop",
        &[
            // The verifier's r18: 2000 x (alloc 1 MB, free) peaks at 1 MB.
            "fn main() -> i64 { let mut i: i64 = 0; let mut a: i64 = 0; while i < 2000 { \
             let p: i64 = __mind_alloc(1000000); __mind_store_i64(p, i); \
             a = a + __mind_load_i64(p) - i; let f: i64 = __mind_free(p); a = a + f; \
             i = i + 1; } return a + 33; }",
            // Through a helper, and a struct literal (an alloc) in a loop.
            "fn mk() -> i64 { return __mind_alloc(64); } fn main() -> i64 { let mut i: i64 = 0; \
             while i < 3 { let p: i64 = mk(); i = i + 1; } return 0; }",
            "struct P { a: i64, b: i64 } fn main() -> i64 { let mut i: i64 = 0; let mut s: i64 = 0; \
             while i < 3 { let p = P { a: i, b: 1 }; s = s + p.a; i = i + 1; } return s; }",
            "fn main() -> i64 { let p: i64 = __mind_alloc(8); let mut i: i64 = 0; let mut q: i64 = p; \
             while i < 3 { q = __mind_realloc(q, 64); i = i + 1; } return 0; }",
        ],
    );
    refused_as(
        "alloc.total_over_budget",
        &[
            // The verifier's r17: one block above the bounded arena.
            "fn main() -> i64 { let p: i64 = __mind_alloc(900000000); if p == 0 { return 5; } \
             __mind_store_i64(p + 899999992, 9); return __mind_load_i64(p + 899999992); }",
            // Multiplicity along the call tree: 3 calls x 200 MB.
            "fn mk() -> i64 { return __mind_alloc(200000000); } \
             fn main() -> i64 { let a: i64 = mk(); let b: i64 = mk(); let c: i64 = mk(); return 0; }",
        ],
    );
    refused_as(
        "alloc.recursive",
        &[
            "fn r(n: i64) -> i64 { let p: i64 = __mind_alloc(64); if n == 0 { return 0; } \
           return r(n - 1); } fn main() -> i64 { return r(3); }",
        ],
    );
    admitted(&[
        "fn main() -> i64 { let p: i64 = __mind_alloc(1000000); let f: i64 = __mind_free(p); \
         return f + 33; }",
        "fn mk() -> i64 { return __mind_alloc(100000000); } \
         fn main() -> i64 { let a: i64 = mk(); let b: i64 = mk(); return 0; }",
        // A loop that uses memory allocated outside it.
        "fn main() -> i64 { let p: i64 = __mind_alloc(64); let mut i: i64 = 0; \
         while i < 8 { __mind_store_i8(p + i, 65); i = i + 1; } return __mind_load_i8(p); }",
        // Recursion that does not allocate.
        "fn r(n: i64) -> i64 { if n == 0 { return 0; } return r(n - 1) + 1; } \
         fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, r(3)); \
         return __mind_load_i64(p); }",
    ]);
}

/// The budget counts what the arena takes: the data rounded up to 8 bytes plus
/// an 8-byte header per block. One block that fits exactly is admitted; one byte
/// more is not.
#[test]
fn the_budget_boundary_counts_headers() {
    let budget = super::alloc_budget::NATIVE_ALLOC_BUDGET;
    let fits = format!(
        "fn main() -> i64 {{ let p: i64 = __mind_alloc({}); return 0; }}",
        budget - 8
    );
    let over = format!(
        "fn main() -> i64 {{ let p: i64 = __mind_alloc({}); return 0; }}",
        budget - 7
    );
    admitted(&[&fits]);
    refused_as("alloc.total_over_budget", &[&over]);
}

/// An intrinsic's i64 reaches only slots declared exactly `i64` (or a struct
/// handle). A parameter or return declared `u64` compares unsigned under MLIR and
/// signed natively (`fn big(y: u64)` given `load_i64(p) == -1`: MLIR 1, native 2);
/// a `u8`/`u16`/`u32` one is masked by MLIR and not natively; `bool`, `isize`,
/// `usize` and a raw pointer are not the slot the corpus proves. Judged
/// on the IR alone here (the source fence is not consulted), so the operand proof
/// itself must refuse each one.
#[test]
fn an_intrinsic_result_reaches_only_exact_i64_slots() {
    let load = "let p: i64 = __mind_alloc(8); __mind_store_i64(p, 0 - 1);";
    let mut refused_srcs = Vec::new();
    for ty in [
        "u64", "u32", "u16", "u8", "isize", "usize", "bool", "*mut i64",
    ] {
        refused_srcs.push(format!(
            "fn f(y: {ty}) -> i64 {{ return 2; }} \
             fn main() -> i64 {{ {load} return f(__mind_load_i64(p)); }}"
        ));
        refused_srcs.push(format!(
            "fn g(p: i64) -> {ty} {{ return __mind_load_i64(p); }} \
             fn main() -> i64 {{ {load} let v = g(p); return 2; }}"
        ));
    }
    let refused_refs: Vec<&str> = refused_srcs.iter().map(String::as_str).collect();
    refused_as("call.intrinsic_result_not_i64", &refused_refs);
    // A narrow i64-proven value passed to such a slot is not an operand proof either.
    refused_as(
        "call.intrinsic_operand_not_i64",
        &["fn f(y: u64) -> i64 { return __mind_wrap_i8(y); } \
           fn main() -> i64 { return f(300); }"],
    );
    admitted(&[
        &format!(
            "fn f(y: i64) -> i64 {{ return 2; }} fn main() -> i64 {{ {load} return f(__mind_load_i64(p)); }}"
        ),
        &format!(
            "fn g(p: i64) -> i64 {{ return __mind_load_i64(p); }} fn main() -> i64 {{ {load} let v = g(p); return 2; }}"
        ),
    ]);
}
