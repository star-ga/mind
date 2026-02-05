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

use libmind::parser;

#[test]
fn shows_pretty_error_for_unexpected_paren() {
    let src = ")";
    let Err(diags) = parser::parse_with_diagnostics(src) else {
        panic!("expected error");
    };
    let joined = diags
        .iter()
        .map(|d| libmind::diagnostics::render(src, d))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("error"));
    assert!(joined.contains("line 1"));
    assert!(joined.contains("^")); // caret present
}

#[test]
fn shows_error_for_unclosed_paren() {
    let src = "(";
    let Err(diags) = parser::parse_with_diagnostics(src) else {
        panic!("expected error");
    };
    let s = libmind::diagnostics::render(src, &diags[0]);
    assert!(s.contains("line 1"));
    assert!(s.contains("^"));
}
