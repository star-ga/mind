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

//! RFC 0007 Phase 4 — `lint::shadowing` rule tests.
//!
//! Fixtures under `tests/mindcraft/lint/shadowing/`:
//! - `positive.mind` — two `let x` in the same fn body → fires.
//! - `negative.mind` — two different names → silent.

use std::path::Path;

use libmind::lint::rule::LintCtx;
use libmind::lint::rule::LintRule;
use libmind::lint::rules::Shadowing;
use libmind::parser::parse_with_trivia;
use libmind::project::MindcraftConfig;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mindcraft/lint/shadowing")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("could not read fixture {name}: {e}"))
}

fn run(src: &str, file: &str) -> Vec<libmind::lint::Diagnostic> {
    let (module, trivia) =
        parse_with_trivia(src).unwrap_or_else(|e| panic!("parse failed for {file}: {e:?}"));
    let config = MindcraftConfig::default();
    let path = Path::new(file);
    let ctx = LintCtx {
        module: &module,
        trivia: &trivia,
        source: src,
        file: path,
        config: &config,
    };
    Shadowing.check(&ctx)
}

// ---------------------------------------------------------------------------
// Positive: two `let x` in the same fn body — should fire
// ---------------------------------------------------------------------------

#[test]
fn shadowing_fires_on_same_scope_rebind() {
    let src = fixture("positive.mind");
    let diags = run(&src, "positive.mind");
    assert!(
        !diags.is_empty(),
        "expected lint::shadowing diagnostic on positive fixture, got none"
    );
    for d in &diags {
        assert_eq!(d.rule_id, "lint::shadowing");
    }
    assert!(
        diags.iter().any(|d| d.message.contains('x')),
        "diagnostic should mention the shadowed name `x`"
    );
}

// ---------------------------------------------------------------------------
// Negative: two different names — should be silent
// ---------------------------------------------------------------------------

#[test]
fn shadowing_silent_on_distinct_names() {
    let src = fixture("negative.mind");
    let diags = run(&src, "negative.mind");
    assert!(
        diags.is_empty(),
        "expected no lint::shadowing diagnostic on negative fixture, got: {diags:?}"
    );
}

// ---------------------------------------------------------------------------
// Inline: same-scope rebind detected in inline source
// ---------------------------------------------------------------------------

#[test]
fn shadowing_inline_same_scope() {
    let src = "fn f() -> i32 {\n    let v: i32 = 1\n    let v: i32 = 2\n    v\n}\n";
    let diags = run(src, "inline.mind");
    assert!(
        !diags.is_empty(),
        "expected shadowing diagnostic for inline same-scope rebind"
    );
}

// ---------------------------------------------------------------------------
// Inline: distinct names are not flagged
// ---------------------------------------------------------------------------

#[test]
fn shadowing_inline_distinct_names_silent() {
    let src = "fn f() -> i32 {\n    let a: i32 = 1\n    let b: i32 = 2\n    a + b\n}\n";
    let diags = run(src, "inline.mind");
    assert!(
        diags.is_empty(),
        "expected no shadowing diagnostic for distinct names, got: {diags:?}"
    );
}
