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

//! Phase 2A formatter fixture tests.
//!
//! Each fixture is a `.in.mind` / `.out.mind` pair under
//! `tests/mindcraft/fmt/`.  This test:
//!   1. Formats the `.in.mind` source and asserts byte-equality with `.out.mind`.
//!   2. Applies `format_source` a second time (round-trip idempotence gate).

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

/// Read fixture file relative to the crate root.
fn read_fixture(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mindcraft/fmt")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read fixture {name}: {e}"))
}

/// Produce a simple line-by-line diff for failure messages.
fn diff_lines(expected: &str, got: &str) -> String {
    let exp_lines: Vec<&str> = expected.lines().collect();
    let got_lines: Vec<&str> = got.lines().collect();
    let max = exp_lines.len().max(got_lines.len());
    let mut out = String::new();
    for i in 0..max {
        let e = exp_lines.get(i).copied().unwrap_or("<missing>");
        let g = got_lines.get(i).copied().unwrap_or("<missing>");
        if e != g {
            out.push_str(&format!("line {}: expected {:?}\n", i + 1, e));
            out.push_str(&format!("line {}:      got {:?}\n", i + 1, g));
        }
    }
    out
}

fn run_fixture(name: &str, cfg: &MindcraftFormatConfig) {
    let in_src = read_fixture(&format!("{name}.in.mind"));
    let expected = read_fixture(&format!("{name}.out.mind"));

    // Pass 1: format the input.
    let formatted = format_source(&in_src, cfg)
        .unwrap_or_else(|e| panic!("fixture {name}: format pass 1 failed: {e}"));

    assert_eq!(
        formatted,
        expected,
        "fixture {name}: pass-1 output differs from expected\n{}",
        diff_lines(&expected, &formatted),
    );

    // Pass 2: idempotence — format the already-formatted output.
    let formatted2 = format_source(&formatted, cfg)
        .unwrap_or_else(|e| panic!("fixture {name}: format pass 2 failed: {e}"));

    assert_eq!(
        formatted2,
        formatted,
        "fixture {name}: idempotence violated (pass 2 != pass 1)\n{}",
        diff_lines(&formatted, &formatted2),
    );
}

fn default_cfg() -> MindcraftFormatConfig {
    MindcraftFormatConfig::default()
}

// ---------------------------------------------------------------------------
// Fixture 01 — nested if/else indentation
// ---------------------------------------------------------------------------

#[test]
fn fixture_01_indent_if_else() {
    run_fixture("01_indent_if_else", &default_cfg());
}

// ---------------------------------------------------------------------------
// Fixture 02 — struct literal single-line
// ---------------------------------------------------------------------------

#[test]
fn fixture_02_struct_literal_multiline() {
    run_fixture("02_struct_literal_multiline", &default_cfg());
}

// ---------------------------------------------------------------------------
// Fixture 03 — fn call args whitespace normalisation
// ---------------------------------------------------------------------------

#[test]
fn fixture_03_fn_args_multiline() {
    run_fixture("03_fn_args_multiline", &default_cfg());
}

// ---------------------------------------------------------------------------
// Fixture 04 — trailing comma toggle (default = true)
// ---------------------------------------------------------------------------

#[test]
fn fixture_04_trailing_comma_toggle() {
    run_fixture("04_trailing_comma_toggle", &default_cfg());
}

// ---------------------------------------------------------------------------
// Fixture 05 — internal whitespace normalisation
// ---------------------------------------------------------------------------

#[test]
fn fixture_05_internal_whitespace() {
    run_fixture("05_internal_whitespace", &default_cfg());
}

// ---------------------------------------------------------------------------
// Fixture 06 — comment attachment (leading, doc, copyright header)
// ---------------------------------------------------------------------------

#[test]
fn fixture_06_comment_attachment() {
    run_fixture("06_comment_attachment", &default_cfg());
}

// ---------------------------------------------------------------------------
// Fixture 07 — simple fn (string passthrough is implicit in AST)
// ---------------------------------------------------------------------------

#[test]
fn fixture_07_string_literal_passthrough() {
    run_fixture("07_string_literal_passthrough", &default_cfg());
}
