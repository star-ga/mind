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

//! rfn-mind #246 surface: the `~` unary bitwise-NOT operator.
//!
//! `~` was not in the parser, so `seen_mask & ~required` (bundle.mind:345)
//! failed with E1001 "unexpected `~`". It is now a prefix operator:
//! parser (`~expr`), type-checker (int -> int), interpreter (two's-complement
//! `!n`), formatter (`~`), and native lowering desugars `~x` to `(-1) - x`
//! (no new opcode) so the byte-identity gate is unaffected.

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

/// `~n` evaluates to the two's-complement bitwise complement (`~0 == -1`,
/// `~5 == -6`), and composes as an operand of `&` (`12 & ~5 == 8`).
#[test]
fn bitnot_evaluates_twos_complement() {
    let m = parser::parse("~0").expect("must parse");
    assert_eq!(eval::eval_first_expr(&m).unwrap(), -1);

    let m = parser::parse("~5").expect("must parse");
    assert_eq!(eval::eval_first_expr(&m).unwrap(), -6);

    let m = parser::parse("12 & ~5").expect("must parse");
    assert_eq!(eval::eval_first_expr(&m).unwrap(), 8);
}

/// `~` on an integer operand type-checks with no error diagnostic.
#[test]
fn bitnot_type_checks_int() {
    let src = "pub fn f(x: i64) -> i64 {\n    return x & ~1\n}\n";
    assert!(
        errors(src).is_empty(),
        "unexpected errors: {:?}",
        errors(src)
    );
}
