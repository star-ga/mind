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

use std::collections::HashMap;

use libmind::diagnostics;

use libmind::parser;

use libmind::type_checker;

#[test]
fn unknown_ident_points_to_name() {
    let src = "let n: i32 = x + 1";
    let module = parser::parse_with_diagnostics(src).expect("parse failed");
    let diags = type_checker::check_module_types(&module, src, &HashMap::new());
    assert!(!diags.is_empty(), "expected type error diagnostic");
    let rendered = diagnostics::render(src, &diags[0]);
    assert!(
        rendered.contains("x + 1"),
        "diagnostic missing offending line: {rendered}"
    );
    let line = "let n: i32 = x + 1";
    let x_idx = line.find('x').unwrap();
    let caret_line = rendered.lines().last().unwrap_or("");
    let caret_pos = caret_line.find('^').unwrap_or(usize::MAX);
    assert!(
        caret_pos >= x_idx.saturating_sub(2),
        "caret not near identifier: {rendered}"
    );
}
