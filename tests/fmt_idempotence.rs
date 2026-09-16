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

//! Formatter idempotence gate — Phase 2A acceptance test (Step 3 of PR #3).
//!
//! For every source file in scope, asserts:
//!   `format_source(format_source(src)) == format_source(src)`
//!
//! This is the hard contract: round-trip stability must hold at 100% with
//! zero skips.  A formatter that fails idempotence is structurally broken.
//!
//! Scope:
//!   - `std/*.mind` (5 files, canonical stdlib)
//!   - `examples/**/*.mind` (all discoverable .mind sources)
//!   - `tests/mindcraft/fmt/*.in.mind` (7 Phase-2A fixture inputs)
//!
//! All files that parse successfully must be idempotent.  Files that fail to
//! parse (e.g. because they use features the parser doesn't yet support) are
//! skipped with a note — parse failures are not idempotence failures.

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

fn default_cfg() -> MindcraftFormatConfig {
    MindcraftFormatConfig::default()
}

fn manifest_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Produce a compact line-by-line diff for failure messages.
fn diff_lines(a: &str, b: &str) -> String {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();
    let max = a_lines.len().max(b_lines.len());
    let mut out = String::new();
    for i in 0..max {
        let la = a_lines.get(i).copied().unwrap_or("<missing>");
        let lb = b_lines.get(i).copied().unwrap_or("<missing>");
        if la != lb {
            out.push_str(&format!("  line {}: pass1={la:?}\n", i + 1));
            out.push_str(&format!("  line {}:  pass2={lb:?}\n", i + 1));
        }
    }
    out
}

/// Assert idempotence for a single source string.
///
/// Returns `true` if the file was exercised (either passed or failed),
/// `false` if skipped (parse error on pass 1 — not a formatter bug).
fn check_idempotence(label: &str, src: &str, cfg: &MindcraftFormatConfig) -> bool {
    let pass1 = match format_source(src, cfg) {
        Ok(s) => s,
        Err(_) => {
            // Parse failures are expected for some examples using tensor/
            // autodiff syntax not in the current formatter scope.
            return false;
        }
    };
    let pass2 = format_source(&pass1, cfg)
        .unwrap_or_else(|e| panic!("idempotence: pass-2 parse failed for {label}: {e}"));
    assert_eq!(
        pass1,
        pass2,
        "idempotence violated for {label}:\n{}",
        diff_lines(&pass1, &pass2),
    );
    true
}

// ---------------------------------------------------------------------------
// std/*.mind — 6 files (toml requires std-surface for while-loop parsing)
// ---------------------------------------------------------------------------

#[test]
fn idempotence_stdlib() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let mut passed = 0usize;
    let mut skipped = 0usize;

    // The three files below use only if/let/expression syntax — no while loops —
    // and must parse under the default build. `string.mind`, `toml.mind` and
    // `map.mind` use `while` loops and are tested separately under std-surface
    // (idempotence_stdlib_string / idempotence_stdlib_toml / idempotence_stdlib_map).
    for name in &["vec", "io", "blas"] {
        let path = base.join(format!("{name}.mind"));
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if check_idempotence(name, &src, &cfg) {
            passed += 1;
        } else {
            skipped += 1;
        }
    }

    assert_eq!(
        skipped, 0,
        "unexpected parse failures in std/ ({skipped} file(s))"
    );
    assert_eq!(passed, 3, "expected 3 stdlib files, got {passed}");
}

/// `string.mind` uses `while` loops (the byte-compare bodies of
/// `string_eq` / `string_starts_with`), which require the `std-surface`
/// feature for the formatter's parser. Gated separately, like toml.
#[test]
#[cfg(feature = "std-surface")]
fn idempotence_stdlib_string() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let path = base.join("string.mind");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let exercised = check_idempotence("string", &src, &cfg);
    assert!(
        exercised,
        "string.mind failed to parse under std-surface — unexpected"
    );
}

/// `sha256.mind` uses `while` loops in compress_block and sha256, which require
/// the `std-surface` feature for the formatter's parser.
#[test]
#[cfg(feature = "std-surface")]
fn idempotence_stdlib_sha256() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let path = base.join("sha256.mind");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let exercised = check_idempotence("sha256", &src, &cfg);
    assert!(
        exercised,
        "sha256.mind failed to parse under std-surface — unexpected"
    );
}

/// `map.mind` uses `while` loops, which require the `std-surface` feature to be
/// recognised by the formatter's parser.  This test is separately gated.
#[test]
#[cfg(feature = "std-surface")]
fn idempotence_stdlib_map() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let path = base.join("map.mind");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let exercised = check_idempotence("map", &src, &cfg);
    assert!(
        exercised,
        "map.mind failed to parse under std-surface — unexpected"
    );
}

/// `toml.mind` uses `while` loops, which require the `std-surface` feature to
/// be recognised by the formatter's parser.  This test is separately gated.
#[test]
#[cfg(feature = "std-surface")]
fn idempotence_stdlib_toml() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let path = base.join("toml.mind");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let exercised = check_idempotence("toml", &src, &cfg);
    assert!(
        exercised,
        "toml.mind failed to parse under std-surface — unexpected"
    );
}

// ---------------------------------------------------------------------------
// examples/**/*.mind
// ---------------------------------------------------------------------------

/// Ratchet floor for `idempotence_examples`: the number of files under examples/
/// that round-trip through the formatter today. Raising it is free; lowering it is a
/// deliberate, explained change.
// Ratchet floor for `idempotence_examples`, measured in a CLEAN `git worktree`.
//
// THE FLOOR IS FEATURE-DEPENDENT and must be, because how many examples PARSE depends on
// which language surface is compiled in:
//
//   --features std-surface …    50 passed,  8 skipped, 58 total
//   --no-default-features       31 passed, 27 skipped, 58 total   (ci.yml runs this)
//
// The 27 extra skips without `std-surface` are not a regression: those files use
// std-surface constructs that a default build genuinely cannot parse.
//
// Two calibration mistakes are baked into this comment so they are not repeated:
//   1. The first value (52) was measured in a working tree carrying two UNTRACKED fuzzer
//      artifacts under examples/mindc_mind/mindfuzz_self_host_staged/, inflating the
//      corpus 58 -> 60. Unreachable from any fresh clone. Calibrate ONLY from a clean
//      worktree — `git worktree add --detach /tmp/x HEAD`.
//   2. The second value (50) was measured with the FULL feature set only, and would still
//      have redded the `--no-default-features` tier at 31. Calibrate EVERY configuration
//      CI builds, not the one that happens to be in your shell history.
#[cfg(feature = "std-surface")]
const EXAMPLES_IDEMPOTENCE_FLOOR: usize = 50;
#[cfg(not(feature = "std-surface"))]
const EXAMPLES_IDEMPOTENCE_FLOOR: usize = 25;

// Measured at 68d8e172. These are an allow-list, not a skip target: any entry
// may start passing, while a new skip or a changed cause fails at that fixture.
#[cfg(not(feature = "std-surface"))]
const BARE_E1042_REFUSALS: &[&str] = &[
    "examples/dottie_collapse.mind",
    "examples/emit_ir/main.mind",
    "examples/grammar_mask/main.mind",
    "examples/lexer/main.mind",
    "examples/mindc_mind/main.mind",
    // The codec mirror uses bitwise masking/shifts; the bare parser must
    // classify those operators as the feature-specific E1042 refusal.
    "examples/mind_mirror_v04/mic3_v04_prefix_mirror.mind",
    "examples/parser/main.mind",
    "examples/remizov_feynman.mind",
    "examples/typecheck/main.mind",
    "examples/mindc_mind/testdata/native_record_array/computed_i64_bitwise.mind",
];

#[cfg(not(feature = "std-surface"))]
const BARE_LEGACY_PARSE_SKIPS: &[&str] = &[
    "examples/anthropobrot.mind",
    "examples/collatz.mind",
    "examples/columnar/structural_scan_json.mind",
    "examples/columnar/tiled_fold.mind",
    "examples/compliance/auditable_model.mind",
    "examples/cos_dottie.mind",
    "examples/detmath_kat/main.mind",
    "examples/fft_q16.mind",
    "examples/fft_signal.mind",
    "examples/galperin_pi.mind",
    "examples/halbach_q16/main.mind",
    "examples/halbach_q16.mind",
    "examples/lorenz_f64.mind",
    "examples/lorenz_q16.mind",
    "examples/mandelbrot.mind",
    "examples/mandelbrot_strict.mind",
    "examples/mindc_mind/testdata/rh_f64_aggregate_canary.mind",
    // The RH Tier-0 totality drivers use `while`, which is std-surface grammar.
    "examples/mindc_mind/testdata/rh_totality_tier0_budget.mind",
    "examples/mindc_mind/testdata/rh_totality_tier0_digest.mind",
    "examples/mindc_mind/testdata/rh_totality_tier0_verdict.mind",
    "examples/native/ci_kernel.mind",
    "examples/native/loop.mind",
    // The native bridge corpus includes a while loop.  `while` belongs to the
    // std-surface grammar, so the bare formatter must classify this fixture as
    // a measured legacy parse skip rather than silently treating it as an
    // unexpected corpus regression.
    "examples/mindc_mind/testdata/backend_native_bridge/while_sum.mind",
    "examples/mindc_mind/testdata/native_record_array/extent_overflow_refuse.mind",
    "examples/mindc_mind/testdata/native_record_array/record_param_loop.mind",
    "examples/mindc_mind/testdata/native_record_array/reference.mind",
    // These loop fixtures also require std-surface; the enabled-profile
    // formatter pass below must parse and format them successfully.
    "examples/mindc_mind/testdata/native_record_array/reference_u8.mind",
    "examples/mindc_mind/testdata/native_record_array/u8_direct_loop.mind",
    "examples/policy.mind",
    "examples/remizov_benchmark.mind",
    "examples/remizov_gpu.mind",
    "examples/remizov_inverse.mind",
    "examples/remizov_solver.mind",
    "examples/remizov_verify.mind",
];

#[cfg(not(feature = "std-surface"))]
fn expected_bare_skip_cause(label: &str) -> Option<&'static str> {
    let normalized = label.replace('\\', "/");
    let label = normalized.as_str();
    if BARE_E1042_REFUSALS.contains(&label) {
        Some("E1042")
    } else if BARE_LEGACY_PARSE_SKIPS.contains(&label) {
        Some("legacy-parse")
    } else {
        None
    }
}

#[test]
fn idempotence_examples() {
    let base = manifest_dir();
    let cfg = default_cfg();
    let mut passed = 0usize;
    let mut skipped = 0usize;
    #[cfg(not(feature = "std-surface"))]
    let mut bitwise_refused = 0usize;
    #[cfg(not(feature = "std-surface"))]
    let mut legacy_parse_skipped = 0usize;
    #[cfg(not(feature = "std-surface"))]
    let mut skip_cause_mismatches = Vec::new();

    // Collect all .mind files under examples/
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let examples_dir = base.join("examples");
    collect_mind_files(&examples_dir, &mut paths);
    paths.sort();

    assert!(!paths.is_empty(), "no .mind files found under examples/");

    for path in &paths {
        let label = path
            .strip_prefix(&base)
            .unwrap_or(path)
            .display()
            .to_string();
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if check_idempotence(&label, &src, &cfg) {
            passed += 1;
        } else {
            skipped += 1;
            #[cfg(not(feature = "std-surface"))]
            {
                let cause = match libmind::parser::parse(&src) {
                    Err(errors) if errors.iter().any(|e| e.cause_code == Some("E1042")) => {
                        bitwise_refused += 1;
                        "E1042"
                    }
                    Err(_) => {
                        legacy_parse_skipped += 1;
                        "legacy-parse"
                    }
                    Ok(_) => "formatter-only",
                };
                let expected = expected_bare_skip_cause(&label);
                if expected != Some(cause) {
                    skip_cause_mismatches.push(format!(
                        "{label}: expected {}, got {cause}",
                        expected.unwrap_or("no skip")
                    ));
                }
            }
        }
    }

    // Parse skips are permitted (tensor/autodiff examples), but the
    // idempotence assertion inside check_idempotence must hold for every
    // file that does parse.
    //
    // The counts used to be discarded here (`let _ = (passed, skipped)`), which made
    // this a test that could not fail: `check_idempotence` returns false on a pass-1
    // parse/format error, so if EVERY example stopped parsing, `passed` was 0,
    // `skipped` was everything, and the only surviving assertion -- that the file list
    // is non-empty -- still held. A formatter or parser regression severe enough to
    // make the entire examples corpus unparseable was reported as ok.
    //
    // The sibling tests already demand positive counts (`idempotence_stdlib` requires
    // skipped == 0 && passed == 3; `idempotence_fmt_fixtures` requires passed == 7).
    // This one cannot demand zero skips -- some examples legitimately do not parse --
    // so it ratchets on what the corpus proves TODAY instead.
    eprintln!(
        "idempotence_examples: {passed} passed, {skipped} skipped, {} total",
        paths.len()
    );
    #[cfg(not(feature = "std-surface"))]
    eprintln!("bare_skip_causes: E1042={bitwise_refused}, legacy-parse={legacy_parse_skipped}");
    #[cfg(not(feature = "std-surface"))]
    assert!(
        skip_cause_mismatches.is_empty(),
        "bare formatter skip classification changed:\n{}",
        skip_cause_mismatches.join("\n")
    );
    #[cfg(not(feature = "std-surface"))]
    assert_eq!(
        skipped,
        bitwise_refused + legacy_parse_skipped,
        "every bare skip must have a measured parse-cause classification"
    );
    #[cfg(not(feature = "std-surface"))]
    assert!(
        bitwise_refused >= 6,
        "expected at least 6 std-surface examples refused with E1042, got {bitwise_refused}"
    );
    assert!(
        passed >= EXAMPLES_IDEMPOTENCE_FLOOR,
        "idempotence floor breached: {passed} of {} examples round-tripped, floor is \
         {EXAMPLES_IDEMPOTENCE_FLOOR}. Either a formatter/parser regression stopped \
         files parsing, or examples were removed -- if the drop is intentional, lower \
         the floor IN THE SAME CHANGE and say why.",
        paths.len(),
    );
}

// ---------------------------------------------------------------------------
// tests/mindcraft/fmt/*.in.mind — Phase-2A fixture inputs
// ---------------------------------------------------------------------------

#[test]
fn idempotence_fmt_fixtures() {
    let fixture_dir = manifest_dir().join("tests/mindcraft/fmt");
    let cfg = default_cfg();
    let mut passed = 0usize;

    for entry in
        std::fs::read_dir(&fixture_dir).unwrap_or_else(|e| panic!("cannot read fixture dir: {e}"))
    {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("mind")
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".in.mind"))
                .unwrap_or(false)
        {
            let label = path.file_name().unwrap().to_string_lossy().to_string();
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            let exercised = check_idempotence(&label, &src, &cfg);
            assert!(exercised, "fixture {label} failed to parse — unexpected");
            passed += 1;
        }
    }

    assert_eq!(passed, 7, "expected 7 fixture .in.mind files, got {passed}");
}

// ---------------------------------------------------------------------------
// Bare `[T]` dynamic-slice canonicalisation (parser accepts `[T]` as sugar
// for `&[T]`; both share the `Slice` AST node). The formatter has one
// surface form for that node — `&[T]` — so `[T]` canonicalises to `&[T]`.
// This must be a DECIDED, stable rewrite (lossless: MIND draws no
// borrow/owned distinction on slices), not an accidental drift. Asserts the
// canonical target and that a second pass is a fixpoint.
// ---------------------------------------------------------------------------

#[test]
fn bare_slice_canonicalises_to_borrowed_and_is_idempotent() {
    let cfg = default_cfg();
    let src = "struct S { items: [u32] }\n";

    let once = format_source(src, &cfg).expect("bare `[T]` must format");
    assert!(
        once.contains("&[u32]"),
        "bare `[T]` must canonicalise to `&[T]`, got:\n{once}"
    );
    assert!(
        !once.contains(": [u32]"),
        "no bare `[T]` should survive formatting, got:\n{once}"
    );

    let twice = format_source(&once, &cfg).expect("second pass must format");
    assert_eq!(once, twice, "`[T]` canonicalisation must be a fixpoint");
}

#[test]
fn fixed_array_is_not_canonicalised_to_slice() {
    let cfg = default_cfg();
    let src = "struct S { lut: [u32; 4] }\n";
    let once = format_source(src, &cfg).expect("`[T; N]` must format");
    assert!(
        once.contains("[u32; 4]"),
        "`[T; N]` must stay a fixed-size array, got:\n{once}"
    );
    assert!(
        !once.contains("&["),
        "`[T; N]` must not become a slice, got:\n{once}"
    );
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn collect_mind_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_mind_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("mind") {
            out.push(path);
        }
    }
}
