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

use libmind::parser;
use libmind::type_checker;
use libmind::types::ValueType;
#[test]
fn scalars_ok() {
    let src = "1 + 2 * 3";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    assert!(diags.is_empty());
}

#[test]
fn unknown_ident_reports_error() {
    let src = "y + 1";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    assert!(!diags.is_empty());
}
