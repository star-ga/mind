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

//! RFC 0007 Phase 3 — lint infrastructure acceptance tests.
//!
//! Covers:
//!   1. `unknown_rules` reports an id in `config.rules` that isn't registered.
//!   2. `effective_severity` returns the configured value when set.
//!   3. `effective_severity` returns `default_severity` when not configured.
//!   4. A glob override layer changes severity for matching paths.
//!   5. An `Off` severity skips the rule (check is never called).
//!   6. `trailing_whitespace` emits diagnostics on a file with trailing whitespace.
//!   7. `trailing_whitespace` emits NO diagnostics on a clean file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use libmind::ast::Module;
use libmind::lint::rule::{LintCtx, LintRule, RuleRegistry};
use libmind::lint::rules::{register_defaults, TrailingWhitespace};
use libmind::lint::{run_lint, Diagnostic};
use libmind::parser::{parse_with_trivia, TriviaStream};
use libmind::project::{MindcraftConfig, MindcraftOverride, RuleSeverity};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn empty_config() -> MindcraftConfig {
    MindcraftConfig::default()
}

fn config_with_rule(id: &str, sev: RuleSeverity) -> MindcraftConfig {
    let mut cfg = empty_config();
    cfg.rules.insert(id.to_string(), sev);
    cfg
}

/// Parse `src` into (Module, TriviaStream), panicking on error.
fn parse(src: &str) -> (Module, TriviaStream) {
    parse_with_trivia(src).expect("test source must parse")
}

/// Build a LintCtx from parts.
fn ctx<'a>(
    module: &'a Module,
    trivia: &'a TriviaStream,
    source: &'a str,
    file: &'a Path,
    config: &'a MindcraftConfig,
) -> LintCtx<'a> {
    LintCtx { module, trivia, source, file, config }
}

// ---------------------------------------------------------------------------
// Test 1 — unknown_rules reports unregistered id
// ---------------------------------------------------------------------------

#[test]
fn unknown_rules_reports_unregistered_id() {
    let mut registry = RuleRegistry::new();
    register_defaults(&mut registry);

    // "lint::does_not_exist" is referenced in config but has no registered rule.
    let config = config_with_rule("lint::does_not_exist", RuleSeverity::Warn);

    let unknown = registry.unknown_rules(&config);
    assert!(
        unknown.contains(&"lint::does_not_exist".to_string()),
        "expected 'lint::does_not_exist' in unknown list, got: {unknown:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — effective_severity returns configured value
// ---------------------------------------------------------------------------

#[test]
fn effective_severity_returns_configured_value() {
    let mut registry = RuleRegistry::new();
    register_defaults(&mut registry);

    let config = config_with_rule("lint::trailing_whitespace", RuleSeverity::Error);
    let file = Path::new("src/main.mind");

    let sev = registry.effective_severity("lint::trailing_whitespace", &config, file);
    assert_eq!(sev, RuleSeverity::Error);
}

// ---------------------------------------------------------------------------
// Test 3 — effective_severity returns default when not configured
// ---------------------------------------------------------------------------

#[test]
fn effective_severity_returns_default_when_unconfigured() {
    let mut registry = RuleRegistry::new();
    register_defaults(&mut registry);

    let config = empty_config(); // no rules configured
    let file = Path::new("src/main.mind");

    let sev = registry.effective_severity("lint::trailing_whitespace", &config, file);
    // TrailingWhitespace::default_severity() == Warn
    assert_eq!(sev, RuleSeverity::Warn);
}

// ---------------------------------------------------------------------------
// Test 4 — glob override changes severity for matching paths
// ---------------------------------------------------------------------------

#[test]
fn glob_override_changes_severity_for_matching_path() {
    let mut registry = RuleRegistry::new();
    register_defaults(&mut registry);

    // Base config: warn for trailing_whitespace.
    // Override for tests/**: upgrade to Error.
    let mut ov_rules = HashMap::new();
    ov_rules.insert("lint::trailing_whitespace".to_string(), RuleSeverity::Error);

    let mut config = empty_config();
    config.overrides.push(MindcraftOverride {
        includes: vec!["tests/**".to_string()],
        excludes: vec![],
        rules: ov_rules,
        format: None,
    });

    let matching_file = Path::new("tests/foo/bar.mind");
    let non_matching_file = Path::new("src/lib.mind");

    let sev_match = registry.effective_severity(
        "lint::trailing_whitespace",
        &config,
        matching_file,
    );
    let sev_no_match = registry.effective_severity(
        "lint::trailing_whitespace",
        &config,
        non_matching_file,
    );

    assert_eq!(sev_match, RuleSeverity::Error, "override should apply for tests/**");
    assert_eq!(sev_no_match, RuleSeverity::Warn, "override must not apply outside tests/**");
}

// ---------------------------------------------------------------------------
// Test 5 — Off severity skips the rule entirely
// ---------------------------------------------------------------------------

/// A spy rule that records whether `check` was called.
struct SpyRule {
    called: std::sync::atomic::AtomicBool,
}

impl SpyRule {
    fn new() -> Self {
        Self { called: std::sync::atomic::AtomicBool::new(false) }
    }

    fn was_called(&self) -> bool {
        self.called.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl LintRule for SpyRule {
    fn id(&self) -> &'static str {
        "lint::spy_rule"
    }
    fn default_severity(&self) -> RuleSeverity {
        RuleSeverity::Warn
    }
    fn description(&self) -> &'static str {
        "spy rule for testing"
    }
    fn check(&self, _ctx: &LintCtx<'_>) -> Vec<Diagnostic> {
        self.called.store(true, std::sync::atomic::Ordering::SeqCst);
        vec![]
    }
}

#[test]
fn off_severity_skips_rule_entirely() {
    // We need the spy to outlive the registry so we can inspect it.
    // Use a shared Arc to observe from outside.
    use std::sync::Arc;

    let spy = Arc::new(SpyRule::new());
    let spy_clone = Arc::clone(&spy);

    // Wrap in a newtype so we can box it.
    struct ArcSpy(Arc<SpyRule>);
    impl LintRule for ArcSpy {
        fn id(&self) -> &'static str { self.0.id() }
        fn default_severity(&self) -> RuleSeverity { self.0.default_severity() }
        fn description(&self) -> &'static str { self.0.description() }
        fn check(&self, ctx: &LintCtx<'_>) -> Vec<Diagnostic> { self.0.check(ctx) }
    }

    let mut registry = RuleRegistry::new();
    registry.register(ArcSpy(spy_clone));

    // Set severity to Off — check must not be called.
    let config = config_with_rule("lint::spy_rule", RuleSeverity::Off);
    let src = "fn noop() {}\n";
    let (module, trivia) = parse(src);
    let file = PathBuf::from("dummy.mind");
    let lctx = ctx(&module, &trivia, src, &file, &config);

    let diags = run_lint(&lctx, &registry);
    assert!(diags.is_empty(), "Off rule should produce no diagnostics");
    assert!(!spy.was_called(), "Off rule's check() must not be invoked");
}

// ---------------------------------------------------------------------------
// Test 6 — trailing_whitespace emits diagnostics on dirty fixture
// ---------------------------------------------------------------------------

#[test]
fn trailing_whitespace_emits_on_dirty_file() {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mindcraft/lint/trailing_ws_dirty.mind");
    let src = std::fs::read_to_string(&fixture_path)
        .expect("trailing_ws_dirty.mind must exist");

    let (module, trivia) = parse(&src);
    let config = empty_config();
    let lctx = ctx(&module, &trivia, &src, &fixture_path, &config);

    let rule = TrailingWhitespace;
    let diags = rule.check(&lctx);

    assert!(
        !diags.is_empty(),
        "expected at least one trailing_whitespace diagnostic on dirty fixture"
    );
    // Every diagnostic must reference the correct rule id.
    for d in &diags {
        assert_eq!(d.rule_id, "lint::trailing_whitespace");
    }
    // The span must be a valid byte range within the source.
    for d in &diags {
        assert!(d.span.start <= d.span.end);
        assert!(d.span.end <= src.len());
    }
}

// ---------------------------------------------------------------------------
// Test 7 — trailing_whitespace emits NO diagnostics on clean file
// ---------------------------------------------------------------------------

#[test]
fn trailing_whitespace_clean_on_clean_file() {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mindcraft/lint/trailing_ws_clean.mind");
    let src = std::fs::read_to_string(&fixture_path)
        .expect("trailing_ws_clean.mind must exist");

    let (module, trivia) = parse(&src);
    let config = empty_config();
    let lctx = ctx(&module, &trivia, &src, &fixture_path, &config);

    let rule = TrailingWhitespace;
    let diags = rule.check(&lctx);

    assert!(
        diags.is_empty(),
        "expected zero diagnostics on clean fixture, got: {diags:?}"
    );
}
