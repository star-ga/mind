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

//! rfn-mind #246 surface: `Tensor<NamedType>` element type.
//!
//! The Tensor dtype parser accepted only builtin dtype keywords, so
//! `Tensor<Q16_16>` (bitlinear.mind, an imported/aliased named type) — and
//! even `Tensor<i8>` — failed with E1001 "expected dtype". The dtype parser
//! now falls through to an identifier / qualified-path name after the builtin
//! table, carried opaquely and accepted by the type-checker.

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

/// `Tensor<Q16_16>` over a user-defined named type parses + type-checks clean.
#[test]
fn tensor_named_element_type_checks() {
    let src = "struct Q16_16 { v: i64 }\nstruct Holder { pub bias: Tensor<Q16_16> }\n";
    assert!(
        errors(src).is_empty(),
        "unexpected errors: {:?}",
        errors(src)
    );
}

/// `Tensor<i8>` (a narrow builtin that was also missing from the dtype table)
/// parses + type-checks clean.
#[test]
fn tensor_i8_element_type_checks() {
    let src = "struct Holder { pub w: Tensor<i8> }\n";
    assert!(
        errors(src).is_empty(),
        "unexpected errors: {:?}",
        errors(src)
    );
}
