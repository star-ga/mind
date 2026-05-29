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

//! RFC 0007 Phase 4 — `lint::unused_import` rule tests.
//!
//! Fixtures under `tests/mindcraft/lint/unused_import/`:
//! - `positive.mind` — `use std.vec` with no `vec` reference → fires.
//! - `negative.mind` — `use std.vec` with `vec_new()` call → does NOT fire.

use std::path::Path;

use libmind::lint::rule::LintCtx;
use libmind::lint::rule::LintRule;
use libmind::lint::rules::UnusedImport;
use libmind::parser::parse_with_trivia;
use libmind::project::MindcraftConfig;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mindcraft/lint/unused_import")
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
    UnusedImport.check(&ctx)
}

// ---------------------------------------------------------------------------
// Positive: `use std.vec` with no usage → fires
// ---------------------------------------------------------------------------

#[test]
fn unused_import_fires_when_not_referenced() {
    let src = fixture("positive.mind");
    let diags = run(&src, "positive.mind");
    assert!(
        !diags.is_empty(),
        "expected lint::unused_import diagnostic on positive fixture, got none"
    );
    for d in &diags {
        assert_eq!(d.rule_id, "lint::unused_import");
    }
}

// ---------------------------------------------------------------------------
// Negative: `use std.vec` with `vec_new()` call → silent
// ---------------------------------------------------------------------------

#[test]
fn unused_import_silent_when_referenced() {
    let src = fixture("negative.mind");
    let diags = run(&src, "negative.mind");
    assert!(
        diags.is_empty(),
        "expected no lint::unused_import diagnostic on negative fixture, got: {diags:?}"
    );
}
