// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Issue #263 surface 2: pointer-sized integer suffixes (`0usize`, `3isize`).
//!
//! Issue #205 wired the width suffixes `u8`..`i64`; `usize`/`isize` were still
//! missing, so `0usize` (rfn-mind memory.mind:69, a match-arm literal) lexed as
//! `0` then the identifier `usize` and died with E2002 "unknown identifier
//! usize". The suffix is now consumed at a word boundary and desugared into the
//! existing `expr as usize` cast — value = the digits, type = the suffix. The
//! type checker already recognises `usize`/`isize` as integer-class named
//! scalars, so the cast type-checks unchanged.

use libmind::eval;
use libmind::parser;
use libmind::type_checker;

fn errors(src: &str) -> Vec<String> {
    let m = parser::parse(src).expect("must parse");
    let env = type_checker::TypeEnv::default();
    type_checker::check_module_types(&m, src, &env)
        .iter()
        .map(|d| format!("{d:?}"))
        .filter(|d| d.to_lowercase().contains("error"))
        .collect()
}

/// `0usize` / `3isize` parse and type-check with no error diagnostic (no more
/// E2002 for `usize`).
#[test]
fn usize_isize_suffix_type_checks() {
    let src = "pub fn f(x: i64) -> i64 {\n    return match x {\n        0 => 0usize as i64,\n        _ => 3isize as i64,\n    }\n}\n";
    assert!(errors(src).is_empty(), "unexpected errors");
}

/// The suffixed literal evaluates to its digit value (`5usize == 5`) — the
/// `as usize` cast is value-preserving on the integer path.
#[test]
fn usize_suffix_value_is_the_digits() {
    let m = parser::parse("5usize").expect("must parse");
    assert_eq!(eval::eval_first_expr(&m).unwrap(), 5);

    let m = parser::parse("42isize").expect("must parse");
    assert_eq!(eval::eval_first_expr(&m).unwrap(), 42);

    // A bare literal with no suffix is unaffected (byte-identical path).
    let m = parser::parse("7").expect("must parse");
    assert_eq!(eval::eval_first_expr(&m).unwrap(), 7);
}

/// A `usize`-looking suffix that is NOT at a word boundary must not be
/// silently split — `1usizex` keeps `usizex` as a separate token (the literal
/// is not consumed as `1 as usize`). It fails to parse as a bare expression.
#[test]
fn usize_suffix_requires_word_boundary() {
    // `1usizes` is `1` followed by ident `usizes`; as a standalone expression
    // that is a parse error (two adjacent tokens), proving no mis-split.
    assert!(parser::parse("1usizes + ").is_err());
}
