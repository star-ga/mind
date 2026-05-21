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

//! RFC 0007 Phase 4 — `lint::naming_convention` rule tests.
//!
//! Fixtures under `tests/mindcraft/lint/naming_convention/`:
//! - `positive_bad_fn.mind`    — fn `BadFn` → fires (not lower_snake_case)
//! - `positive_bad_struct.mind`— struct `badStruct` → fires (not UpperCamelCase)
//! - `positive_bad_const.mind` — const `bad_const` → fires (not SCREAMING_SNAKE_CASE)
//! - `negative.mind`           — all canonical names → silent

use std::path::Path;

use libmind::lint::rules::NamingConvention;
use libmind::lint::rule::LintRule;
use libmind::lint::rule::LintCtx;
use libmind::parser::parse_with_trivia;
use libmind::project::MindcraftConfig;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mindcraft/lint/naming_convention")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read fixture {name}: {e}"))
}

fn run(src: &str, file: &str) -> Vec<libmind::lint::Diagnostic> {
    let (module, trivia) = parse_with_trivia(src)
        .unwrap_or_else(|e| panic!("parse failed for {file}: {e:?}"));
    let config = MindcraftConfig::default();
    let path = Path::new(file);
    let ctx = LintCtx { module: &module, trivia: &trivia, source: src, file: path, config: &config };
    NamingConvention.check(&ctx)
}

// ---------------------------------------------------------------------------
// Positive 1: fn `BadFn` — should fire (not lower_snake_case)
// ---------------------------------------------------------------------------

#[test]
fn naming_fires_on_bad_fn_name() {
    let src = fixture("positive_bad_fn.mind");
    let diags = run(&src, "positive_bad_fn.mind");
    assert!(
        !diags.is_empty(),
        "expected lint::naming_convention for fn `BadFn`, got none"
    );
    for d in &diags {
        assert_eq!(d.rule_id, "lint::naming_convention");
    }
    assert!(
        diags.iter().any(|d| d.message.contains("BadFn")),
        "diagnostic should mention the bad name `BadFn`"
    );
}

// ---------------------------------------------------------------------------
// Positive 2: struct `badStruct` — should fire (not UpperCamelCase)
// ---------------------------------------------------------------------------

#[test]
fn naming_fires_on_bad_struct_name() {
    let src = fixture("positive_bad_struct.mind");
    let diags = run(&src, "positive_bad_struct.mind");
    assert!(
        !diags.is_empty(),
        "expected lint::naming_convention for struct `badStruct`, got none"
    );
    assert!(
        diags.iter().any(|d| d.message.contains("badStruct")),
        "diagnostic should mention the bad name `badStruct`"
    );
}

// ---------------------------------------------------------------------------
// Positive 3: const `bad_const` — should fire (not SCREAMING_SNAKE_CASE)
// ---------------------------------------------------------------------------

#[test]
fn naming_fires_on_bad_const_name() {
    let src = fixture("positive_bad_const.mind");
    let diags = run(&src, "positive_bad_const.mind");
    assert!(
        !diags.is_empty(),
        "expected lint::naming_convention for const `bad_const`, got none"
    );
    assert!(
        diags.iter().any(|d| d.message.contains("bad_const")),
        "diagnostic should mention the bad name `bad_const`"
    );
}

// ---------------------------------------------------------------------------
// Negative: canonical names — silent
// ---------------------------------------------------------------------------

#[test]
fn naming_silent_on_canonical_names() {
    let src = fixture("negative.mind");
    let diags = run(&src, "negative.mind");
    assert!(
        diags.is_empty(),
        "expected no naming_convention diagnostics on negative fixture, got: {diags:?}"
    );
}
