// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unit tests for the f64 slot proof.

use super::*;

fn admit(src: &str) -> Result<(), &'static str> {
    let ast = crate::parser::parse(src).expect("parse");
    admit_f64_slots(&ast).map_err(|r| r.construct)
}

/// Every slot shape a value reaches: `let`, assignment, parameter (call and
/// method), return (explicit and trailing), struct field (literal and store),
/// const. Each fed an unproven value is refused.
#[test]
fn an_unproven_value_in_any_f64_slot_is_refused() {
    for src in [
        "fn main() -> i64 { let x: f64 = __mind_load_i64(8); return 0; }",
        "fn main() -> i64 { let y = __mind_load_i64(8); let x: f64 = y; return 0; }",
        "fn main() -> i64 { let mut x: f64 = 1.0; x = __mind_load_i64(8); return 0; }",
        "fn big(x: f64) -> i64 { return 1; } fn main() -> i64 { return big(7); }",
        "fn big(x: f64) -> i64 { return 1; } fn main() -> i64 { let a: i64 = 3; return big(a); }",
        "fn g(a: i64) -> f64 { return a; }",
        "fn g(p: i64) -> f64 { __mind_load_i64(p) }",
        "fn g(c: i64) -> f64 { if c == 1 { 1.0 } else { 2 } }",
        "struct S { v: f64 } fn main() -> i64 { let s = S { v: 3 }; return 0; }",
        "struct S { v: f64 } fn f(s: S) -> i64 { s.v = 3; return 0; }",
        "const C: f64 = 7;",
        "fn k(s: i64, x: f64) -> i64 { return 0; } fn main() -> i64 { let a: i64 = 2; return a.k(5); }",
        // Shadowing: one non-f64 binding of the name withholds the proof.
        "fn big(x: f64) -> i64 { return 1; } \
         fn main() -> i64 { let x: f64 = 1.0; let x: i64 = 2; return big(x); }",
        // Two unannotated names feeding each other prove nothing.
        "fn big(x: f64) -> i64 { return 1; } \
         fn main() -> i64 { let mut a = b; let mut b = a; return big(a); }",
        // The declared-f64 invariant holds only if every assignment keeps it.
        "fn big(x: f64) -> i64 { return 1; } \
         fn main() -> i64 { let mut x: f64 = 0.0; x = __mind_load_i64(8); return big(x); }",
        "fn main() -> i64 { let mut x: f64 = 0.0; x = x + 1; return 0; }",
        // A nested function's slot is checked against its own declaration.
        "fn main() -> i64 { fn g() -> f64 { return 3; } return 0; }",
        // A field read is not proven, even of an i64 field; a bool is not an f64.
        "struct S { v: i64 } fn main() -> i64 { let s = S { v: 7 }; let x: f64 = s.v; return 0; }",
        "fn main() -> i64 { let x: f64 = 1 < 2; return 0; }",
    ] {
        assert_eq!(admit(src), Err(UNPROVEN), "{src}");
    }
}

#[test]
fn a_proven_f64_in_a_non_f64_slot_is_refused() {
    for src in [
        "fn f() -> i64 { let x: f64 = 2.5; return x; }",
        "fn f() -> i64 { 2.5 }",
        "fn main() -> i64 { let b: f64 = 2.5; let a: i64 = b; return a; }",
        "fn g(y: i64) -> i64 { return y; } fn main() -> i64 { return g(2.5); }",
        "fn h() -> f64 { return 1.0; } fn main() -> i64 { let a: i64 = h(); return a; }",
        "struct S { v: i64 } fn main() -> i64 { let s = S { v: 1.5 }; return 0; }",
        // A name or field declared both ways: a write must satisfy both slots.
        "fn main() -> i64 { let mut x: i64 = 1; let mut x: f64 = 1.0; x = 2.5; return 0; }",
        "struct A { v: f64 } struct B { v: i64 } fn f(b: B) -> i64 { b.v = 2.5; return 0; }",
        // An f64 that is not PROVEN one (a branch value, a match value, a name
        // bound to one, a field read) still may not reach a non-f64 slot.
        "fn main() -> i64 { let c: i64 = 1; let a: i64 = if c == 1 { 1.5 } else { 2.5 }; return a; }",
        "fn main() -> i64 { let c: i64 = 1; let a: i64 = match c { 1 => 1.5, _ => 2.5 }; return a; }",
        "fn g(c: i64) -> i64 { let y = if c == 1 { 1.5 } else { 2.5 }; return y; }",
        "fn g(c: i64) -> i64 { let y = if c == 1 { 1.5 } else { 2.5 }; let z = y; z }",
        "struct S { v: f64 } fn g(s: S) -> i64 { return s.v; }",
        "fn g(c: i64) -> i64 { let (a, b) = (1.5, 2); return a; }",
        "fn g(c: i64) -> i64 { let m = match c { 1 => 1.5, _ => 2.5 }; return match m { x => x }; }",
        // A callee the program does not declare (an intrinsic, a std seed fn)
        // takes only i64 parameters.
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, 2.5); return 0; }",
        "fn main() -> i64 { let x: f64 = 2.5; let p: i64 = __mind_alloc(8); \
         __mind_store_i64(p, x); return 0; }",
        "fn main() -> i64 { let x: f64 = 2.5; return vec_len(x); }",
        // A function with no declared return type returns an i64 slot.
        "fn h() { return 1.5; } fn main() -> i64 { let a: i64 = h(); return a; }",
        "fn h() { let x: f64 = 1.5; x }",
    ] {
        assert_eq!(admit(src), Err(MISMATCH), "{src}");
    }
}

#[test]
fn an_f64_in_an_aggregate_or_alias_is_refused() {
    for src in [
        "type R = f64;",
        "fn f(xs: [f64; 2]) -> i64 { return 0; }",
        "fn f() -> (f64, i64) { return (1.0, 2); }",
        "struct S { v: [f64; 2] }",
        "enum E { A(f64), B }",
        "fn main() -> i64 { let t: Option<f64> = None; return 0; }",
    ] {
        assert_eq!(admit(src), Err(AGGREGATE), "{src}");
    }
}

/// Positive control: proven values in f64 slots, and integer programs, admit.
#[test]
fn proven_f64_slots_and_integer_programs_are_admitted() {
    for src in [
        "fn k(x: f64, y: f64) -> i64 { if x < y { return 1; } return 2; } \
         fn main() -> i64 { let a: f64 = 1.5; let b: f64 = a * 2.0; \
         return k(a, b) + k(3.0, 0.0 - 1.0) + k(-a, (b)); }",
        "fn h(x: f64) -> f64 { return x * 2.0; } fn main() -> i64 { let a: f64 = h(1.5); return 0; }",
        "fn h(x: f64) -> f64 { x * 2.0 }",
        "fn k(x: f64) -> i64 { return 1; } fn main() -> i64 { let a = 1.5; let b = a + 1.0; return k(b); }",
        "fn main() -> i64 { let mut x: f64 = 0.0; x = x + 1.5; if x > 1.0 { return 1; } return 2; }",
        "struct S { v: f64, n: i64 } fn main() -> i64 { let s = S { v: 2.5, n: 3 }; return s.n; }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, 7); \
         return __mind_load_i64(p); }",
        "fn main() -> i64 { let x: f64 = __mind_load_i64(8) as f64; return 0; }",
        "const C: f64 = 2.5;\nfn main() -> i64 { let x: f64 = C; return 0; }",
        // Integer results of float operands stay admitted in integer slots.
        "fn main() -> i64 { let x: f64 = 2.5; let a: i64 = x > 1.0; let b: i64 = (x * 2.0) as i64; \
         let c: i64 = !(x < 0.5) && x != 3.0; return a + b + c; }",
        "fn main() -> i64 { let mut t: i64 = 0; for i in 0..3 { t = t + i; } return t; }",
        // An unannotated float keeps its slot across a later float write, including
        // a loop-carried `x = x + 1.0`.
        "fn main() -> i64 { let mut x = 0.5; x = 1.5; if x > 2.0 { return 1; } return 2; }",
        "fn main() -> i64 { let mut x = 0.5; let mut i: i64 = 0; while i < 3 { x = x + 1.0; \
         i = i + 1; } if x > 2.0 { return 1; } return 2; }",
        "fn main() -> i64 { let mut x = 1; x = x + 2; return x; }",
    ] {
        assert_eq!(admit(src), Ok(()), "{src}");
    }
}

/// A binding introduced by an unannotated float is an f64 slot, and one
/// introduced by an integer is not. A later write under an `if` or a loop is
/// the same slot the straight-line write is. MLIR refuses the mismatch
/// (`f64` vs `i64`); the frozen compiler used to build it.
#[test]
fn an_unannotated_binding_refuses_a_later_write_of_the_other_kind() {
    for src in [
        "fn main() -> i64 { let mut x = 0.5; let c: i64 = 1; if c > 0 { x = 5; } \
         if x > 2.0 { return 1; } return 2; }",
        "fn main() -> i64 { let mut x = 0.5; let mut i: i64 = 0; while i < 3 { x = i; \
         i = i + 1; } if x > 2.0 { return 1; } return 2; }",
        "fn main() -> i64 { let a: i64 = 5; let mut x = 0.5; let mut i: i64 = 0; \
         while i < 6 { x = a; i = i + 1; } return 2; }",
        "fn main() -> i64 { let mut x = 0.5; x = 5; return 2; }",
        "fn main() -> i64 { let mut x = 0.5; let mut i: i64 = 0; while i < 6 { x = 5; \
         i = i + 1; } return 2; }",
    ] {
        assert_eq!(admit(src), Err(UNPROVEN), "{src}");
    }
    for src in [
        "fn main() -> i64 { let mut x = 1; let c: i64 = 1; if c > 0 { x = 1.5; } return 2; }",
        "fn main() -> i64 { let mut x = 5; x = 0.5; if x > 2 { return 1; } return 2; }",
        "fn main() -> i64 { let mut x = 1; let mut i: i64 = 0; while i < 6 { x = 3.5; \
         i = i + 1; } return x; }",
    ] {
        assert_eq!(admit(src), Err(MISMATCH), "{src}");
    }
}
