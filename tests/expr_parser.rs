// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use libmind::eval;
use libmind::parser;

#[test]
fn precedence_and_parens() {
    let module = parser::parse("1 + 2 * 3").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 7);

    let module = parser::parse("(1 + 2) * 3").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 9);
}

#[test]
fn division_and_zero_guard() {
    let module = parser::parse("8 / 2").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 4);

    let module = parser::parse("1 / 0").unwrap();
    assert!(eval::eval_first_expr(&module).is_err());
}
