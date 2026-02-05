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
fn let_and_use_variable() {
    let m = parser::parse("let x = 2; x * 3 + 1").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 7);
}

#[test]
fn assign_updates_value() {
    let m = parser::parse("let x = 1; x = x + 4; x * 2").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 10);
}
