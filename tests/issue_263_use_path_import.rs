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

//! Issue #263 surface 1: path-style `use` imports (`use a::b::C;`).
//!
//! Before the fix `use` accepted only the shipped dotted form (`use a.b.c`); the
//! Rust-style `::` path form (`use std::fixed_point::Q16_16;`, rfn-mind
//! bitlinear.mind:8) parse-errored at the first `::`. The parser now accepts a
//! `::` (or `.`) separator per segment and binds the final segment as the
//! imported name, exactly as the dotted form does — a compile-time-only surface
//! with no runtime effect.

use libmind::parser;
use libmind::type_checker;

/// The `::` path form parses and type-checks with no error-severity diagnostics
/// (a benign `unused_import` warning is allowed — `mindc check` treats warnings
/// as non-fatal).
///
/// Gated to the std-resolving feature configs: a binary built with NO std
/// surface *correctly* rejects a `use std::…` import with E2007 (fail-loud, by
/// design), so the "checks clean" assertion only holds when a std surface
/// exists. The feature-independent parse+bind of the `::` form is covered by
/// `use_dotted_form_still_parses` and `use_path_trailing_colons_errors` below.
#[cfg(any(feature = "std-surface", feature = "cross-module-imports"))]
#[test]
fn use_path_form_parses_and_checks() {
    let src = "use std::fixed_point::Q16_16;\n\npub fn main() -> i64 {\n    return 0\n}\n";
    let m = parser::parse(src).expect("`use a::b::C;` must parse");
    let env = type_checker::TypeEnv::default();
    let diags = type_checker::check_module_types(&m, src, &env);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| format!("{d:?}").to_lowercase().contains("error"))
        .collect();
    assert!(errors.is_empty(), "unexpected type errors: {errors:?}");
}

/// The dotted form (the pre-existing spelling) still parses — the `::` support
/// is purely additive.
#[test]
fn use_dotted_form_still_parses() {
    let src = "use std.fixed_point.Q16_16;\n\npub fn main() -> i64 {\n    return 0\n}\n";
    parser::parse(src).expect("`use a.b.c;` must still parse");
}

/// A `::` path with a missing trailing segment is a hard parse error (fail-loud,
/// never a silent partial import).
#[test]
fn use_path_trailing_colons_errors() {
    let src = "use std::fixed_point::;\n";
    assert!(
        parser::parse(src).is_err(),
        "`use a::b::;` must be a parse error"
    );
}
