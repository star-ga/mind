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

//! RFC 0007 Phase 4 — `lint::q16_overflow` rule tests.
//!
//! Fixtures under `tests/mindcraft/lint/q16_overflow/`:
//! - `positive.mind` — bare `i32 * i32`, no `>>16`, should fire.
//! - `negative.mind` — proper `((a as i64 * b as i64) >> 16) as i32`, should NOT fire.
//! - `edge_constant.mind` — `x * 65536` (i32 * literal int), should fire.

use std::path::Path;

use libmind::lint::rule::LintCtx;
use libmind::lint::rule::LintRule;
use libmind::lint::rules::Q16Overflow;
use libmind::parser::parse_with_trivia;
use libmind::project::MindcraftConfig;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mindcraft/lint/q16_overflow")
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
    Q16Overflow.check(&ctx)
}

// ---------------------------------------------------------------------------
// Positive: bare i32 * i32 without >>16 — must fire
// ---------------------------------------------------------------------------

#[test]
fn q16_overflow_fires_on_bare_mul() {
    let src = fixture("positive.mind");
    let diags = run(&src, "positive.mind");
    assert!(
        !diags.is_empty(),
        "expected lint::q16_overflow diagnostic on positive fixture, got none"
    );
    for d in &diags {
        assert_eq!(d.rule_id, "lint::q16_overflow");
    }
}

// ---------------------------------------------------------------------------
// Negative: proper >>16 narrowing — must NOT fire
// ---------------------------------------------------------------------------

#[test]
fn q16_overflow_silent_on_proper_shift() {
    let src = fixture("negative.mind");
    let diags = run(&src, "negative.mind");
    assert!(
        diags.is_empty(),
        "expected no lint::q16_overflow diagnostic on negative fixture, got: {diags:?}"
    );
}

// ---------------------------------------------------------------------------
// Edge: i32 * literal constant still fires (no >>16)
// ---------------------------------------------------------------------------

#[test]
fn q16_overflow_fires_on_mul_with_literal() {
    let src = fixture("edge_constant.mind");
    let diags = run(&src, "edge_constant.mind");
    assert!(
        !diags.is_empty(),
        "expected lint::q16_overflow diagnostic on edge_constant fixture, got none"
    );
    for d in &diags {
        assert_eq!(d.rule_id, "lint::q16_overflow");
    }
}
