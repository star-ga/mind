// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! The allocation-size positivity proof (`alloc_size`): what it proves, and
//! the adversarial shapes it must not.

use super::*;

fn admits(source: &str) -> Check<()> {
    let ast = crate::parser::parse(source).expect("fixture parses");
    let ir = crate::eval::lower_to_ir(&ast).expect("fixture lowers");
    profile_lane_admits(&ast, &ir, true)
}

#[test]
fn a_size_behind_a_non_positive_exit_test_is_proven() {
    for source in [
        // The bridge's own guard, verbatim in shape.
        "fn g(n: i64) -> i64 { if n <= 0 { return 0; } return __mind_alloc(n); } \
         fn main() -> i64 { return g(8); }",
        "fn g(n: i64) -> i64 { if n < 1 { return 0; } return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { if 0 >= n { return 0; } return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { if 1 > n { return 0; } return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { if (n <= 0) { return 0; } return __mind_alloc((n)); }",
        // The fact holds in every nested block after the test.
        "fn g(n: i64) -> i64 { if n <= 0 { return 0; } if 1 { let p = __mind_alloc(n); } return 0; }",
        // break and continue leave the rest of the iteration unreachable too.
        "fn g(n: i64) -> i64 { let i = 0; while i < 2 { if n <= 0 { break; } \
         __mind_alloc(n); i = i + 1; } return 0; }",
        "fn g(n: i64) -> i64 { for i in 0..2 { if n <= 0 { continue; } __mind_alloc(n); } return 0; }",
        "fn g(p: i64, n: i64) -> i64 { if n < 1 { return 0; } return __mind_realloc(p, n); }",
        // The realloc guard, verbatim. The null-and-zero edge allocates a
        // positive literal; the call after the test is what the guard proves.
        "fn __lane_realloc(p: i64, n: i64) -> i64 { if n <= 0 { if p == 0 { \
         if n == 0 { return __mind_alloc(24); } } return 0; } return __mind_realloc(p, n); } \
         fn main() -> i64 { return __lane_realloc(0, 8); }",
        "fn main() -> i64 { let n = 4; if n <= 0 { return 0; } return __mind_alloc(n) + __mind_alloc((8)); }",
    ] {
        assert_eq!(admits(source), Ok(()), "{source}");
    }
}

#[test]
fn adversarial_sizes_stay_refused() {
    for source in [
        // Assigned after the test, or anywhere in the function.
        "fn g(n: i64) -> i64 { if n <= 0 { return 0; } n = n - 5; return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { if n <= 0 { return 0; } if 1 { n = 0; } return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { let x = if 1 { n = 0; 1 } else { 2 }; \
         if n <= 0 { return 0; } return __mind_alloc(n); }",
        // A test that admits zero, a bound that is not recognised, no exit, an else.
        "fn g(n: i64) -> i64 { if n < 0 { return 0; } return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { if n <= 0 - 1 { return 0; } return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { if n <= 0 { let z = 1; } return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { if n <= 0 { return 0; } else { } return __mind_alloc(n); }",
        "fn g(n: i64) -> i64 { if n > 0 { return __mind_alloc(n); } return 0; }",
        // A fact does not leak out of the block that proved it.
        "fn g(c: i64, n: i64) -> i64 { if c { if n <= 0 { return 0; } } return __mind_alloc(n); }",
        // A later binding of the name hides the fact.
        "fn g(n: i64) -> i64 { if n <= 0 { return 0; } let n = 0 - 4; return __mind_alloc(n); }",
        "fn g(i: i64) -> i64 { if i <= 0 { return 0; } for i in 0..3 { __mind_alloc(i); } return 0; }",
        // Arithmetic on a proven name can wrap; another name is not proven.
        "fn g(n: i64) -> i64 { if n <= 0 { return 0; } return __mind_alloc(n * 8); }",
        "fn g(m: i64, n: i64) -> i64 { if m <= 0 { return 0; } return __mind_alloc(n); }",
        // A use before the test, also inside a loop body.
        "fn g(n: i64) -> i64 { while 1 { __mind_alloc(n); if n <= 0 { return 0; } } return 0; }",
        // The fact belongs to one function.
        "fn h(n: i64) -> i64 { if n <= 0 { return 0; } return n; } \
         fn g(n: i64) -> i64 { return __mind_alloc(n); }",
        "fn g(p: i64, n: i64) -> i64 { if n < 0 { return 0; } return __mind_realloc(p, n); }",
    ] {
        assert_eq!(
            admits(source).unwrap_err().construct,
            "lane.alloc.size_positive",
            "{source}"
        );
    }
}
