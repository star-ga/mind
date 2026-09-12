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

//! The formatter must not delete surface spelling that a parser desugar threw away.
//!
//! ONE PATTERN, SIX DEFECTS (traced 2026-09-11). The parser performs semantic desugars and
//! discards how the source was written. The formatter round-trips through that same AST, so
//! every discarded spelling was text `mindc fmt` deleted:
//!
//!   1. `true` / `false`        → `Literal::Int(1)` / `Int(0)`   (covered in
//!                                 `fmt_bool_literal_roundtrip.rs`)
//!   2. `module NAME { … }`     → name dropped, wrapper printed away
//!   3. `use X as Y`            → `as Y` never consumed; left as stray `as;` / `Y;` statements
//!   4. `q.fn(a)` / `q.CONST`   → desugared to bare `fn(a)` / `CONST`, qualifier gone
//!   5. `export struct X`       → category keyword dropped, reprinted as `export { X }`
//!   6. `#[attr] module X { }`   → `parse_module_block` took `_attrs` and dropped them
//!
//! MEASURED IMPACT. Every one of these is caught by the identifier-loss guard, which refuses to
//! write rather than corrupt the file — so the symptom was always identical and always total: a
//! file that can never clear `fmt::drift`, and therefore can never pass `mindc check`. Across
//! MindLLM's 27 modules the count went from **0 formatting** to **27 formatting** once all six
//! were fixed. Before: 80 lost qualifiers in `rfn_bridge.mind` alone.
//!
//! WHY ASSERTIONS ON OUTPUT TEXT, NOT IDEMPOTENCE. `format(format(x)) == format(x)` held
//! throughout every one of these bugs — a dropped word stays dropped, stably. Idempotence
//! cannot see a lossy desugar, which is exactly why these survived an idempotence gate.

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

fn fmt(src: &str) -> String {
    format_source(src, &MindcraftFormatConfig::default())
        .unwrap_or_else(|e| panic!("format_source refused:\n{src}\nerror: {e:?}"))
}

/// Assert every listed fragment survives, AND that the output is a stable, re-readable fixed
/// point.
///
/// THE REPARSE GATE IS THE POINT, and its absence is how a worse defect than the six shipped.
/// A `contains` check over ONE pass cannot see a formatter that emits a form its own parser
/// cannot read: the alias fix made the printer emit `import X as Y;` while only `parse_use`
/// understood `as`, so a SECOND `fmt` rewrote that line into three statements — `import X;`,
/// `as`, `Y` — and WROTE IT TO DISK with exit 0. The identifier-loss guard is blind to it
/// because nothing is deleted; the tokens are merely reparsed as separate statements. Pass 3
/// is a fixed point of the corrupted form, so idempotence alone goes green from pass 2 onward.
///
/// So every case asserts three things, and all three are needed:
///   1. the spelling survives the first pass (what the six fixes are about);
///   2. `fmt(fmt(x)) == fmt(x)` — the output is a fixed point, which catches a lossy RE-emit;
///   3. `parse(fmt(x))` succeeds — the output is readable at all, which catches a form the
///      printer can produce and the parser cannot accept.
fn assert_kept(what: &str, src: &str, expect: &[&str]) {
    let once = fmt(src);
    for frag in expect {
        assert!(
            once.contains(frag),
            "{what}: {frag:?} missing from output.\n--- in ---\n{src}\n--- out ---\n{once}"
        );
    }

    // (3) The formatter's own output must re-parse. A printer that emits a form its parser
    // rejects has corrupted the file, whatever the identifier census says.
    libmind::parser::parse(&once).unwrap_or_else(|e| {
        panic!("{what}: formatter output does not re-parse.\n--- out ---\n{once}\nerror: {e:?}")
    });

    // (2) …and formatting it again must change nothing.
    let twice = fmt(&once);
    assert_eq!(
        once, twice,
        "{what}: formatting is not a fixed point — pass 2 differs, which means pass 1 emitted \
         something pass 2 reads differently.\n--- pass 1 ---\n{once}\n--- pass 2 ---\n{twice}"
    );
}

#[test]
fn module_block_header_survives() {
    assert_kept(
        "single-segment module wrapper",
        "module model_lock {\n    fn f() -> i64 {\n        1\n    }\n}\n",
        &["module model_lock {", "fn f()", "}"],
    );
    // Dotted paths too — the parser consumed `.segment` continuations and discarded them
    // separately from the first segment, so both halves needed reassembling.
    assert_kept(
        "dotted module path",
        "module backends.tool {\n    fn f() -> i64 {\n        1\n    }\n}\n",
        &["module backends.tool {"],
    );
}

#[test]
fn module_block_attributes_survive() {
    // `parse_module_block` took its attribute list as `_attrs` and dropped it, so
    // `#[protection] module X { … }` was reprinted without the attribute — silently deleting it
    // from all 24 attributed MindLLM modules.
    assert_kept(
        "attributed module block",
        "#[protection]\nmodule governance_bridge {\n    fn f() -> i64 {\n        1\n    }\n}\n",
        &["#[protection]", "module governance_bridge {"],
    );
    // A module block with NO attribute must not acquire one — the span-keyed lookup has to miss.
    let out = fmt("module plain {\n    fn f() -> i64 {\n        1\n    }\n}\n");
    assert!(
        !out.contains("#["),
        "an unattributed module block invented an attribute.\n{out}"
    );
}

#[test]
fn use_alias_survives() {
    assert_kept(
        "aliased import",
        "use mindllm_config as config\n\npub fn f() -> i64 {\n    1\n}\n",
        // `use` -> `import` IS a sanctioned canonicalisation (CANONICAL_KEYWORD_REWRITES);
        // `as config` is not, and is what this pins.
        &["as config"],
    );
    assert_kept(
        "dotted path with alias",
        "use mind.core.bytes as bytes\n\npub fn f() -> i64 {\n    1\n}\n",
        &["mind.core.bytes", "as bytes"],
    );
}

#[test]
fn module_qualified_call_survives() {
    assert_kept(
        "aliased single-segment qualifier",
        "use mind.core.bytes as bytes\n\npub fn f(b: i64) -> i64 {\n    bytes.append(b, 1)\n}\n",
        &["bytes.append(b, 1)"],
    );
    assert_kept(
        "unaliased single-segment qualifier",
        "use inference_auth_hash\n\npub fn f(x: i64) -> i64 {\n    inference_auth_hash.compute(x)\n}\n",
        &["inference_auth_hash.compute(x)"],
    );
    assert_kept(
        "multi-segment dotted qualifier",
        "import mind.core.hash\n\npub fn f(x: i64) -> i64 {\n    mind.core.hash.sha256(x)\n}\n",
        &["mind.core.hash.sha256(x)"],
    );
}

#[test]
fn module_qualified_value_read_survives() {
    // `q.CONST` desugars to a bare Ident rather than a Call — a separate parser site, and it
    // needed its own record.
    assert_kept(
        "qualified constant read",
        "use mindllm_config as config\n\npub fn f() -> i64 {\n    config.MAX_DEPTH\n}\n",
        &["config.MAX_DEPTH"],
    );
}

#[test]
fn export_category_survives() {
    assert_kept(
        "export struct",
        "struct S {\n    a: i64,\n}\n\nexport struct S\n",
        &["export struct S"],
    );
    assert_kept(
        "export fn list",
        "fn a() -> i64 {\n    1\n}\n\nfn b() -> i64 {\n    2\n}\n\nexport fn a, b\n",
        &["export fn a, b"],
    );
}

/// `use X as Y` must make BOTH `Y.f()` and `X.f()` resolve as module-qualified calls.
///
/// THIS IS THE TEST TWO INDEPENDENT AUDITS ASKED FOR, and its absence was the fair complaint:
/// the alias fix changed NAME RESOLUTION inside a commit titled as a formatting fix. The first
/// version registered only the alias, which silently broke any file that aliased an import and
/// still called it by its original last segment — that spelling fell through to a `MethodCall`
/// whose receiver has no struct type, tripping the fail-closed lowering guard.
///
/// Both names are now registered. Asserted through the FORMATTER rather than by reading parser
/// state, because the qualifier is only observable in output: a receiver the parser did not
/// recognise as a module is never desugared, so its qualifier is never recorded, and the
/// identifier-loss guard then refuses the file. A passing round-trip here therefore means the
/// name resolved.
#[test]
fn an_alias_registers_both_names_as_qualifiers() {
    assert_kept(
        "alias spelling resolves",
        "use mind.core.bytes as bytes\n\npub fn f(b: i64) -> i64 {\n    bytes.append(b, 1)\n}\n",
        &["bytes.append(b, 1)"],
    );
    // The ORIGINAL last segment must keep working alongside the alias. This is the regression
    // the alias-only registration introduced.
    assert_kept(
        "original spelling still resolves after an alias",
        "use mindllm_config as cfg\n\npub fn f() -> i64 {\n    mindllm_config.MAX\n}\n",
        &["mindllm_config.MAX"],
    );
    // And both spellings in one file.
    assert_kept(
        "both spellings in one file",
        "use mindllm_config as cfg\n\npub fn f() -> i64 {\n    cfg.A\n}\n\npub fn g() -> i64 {\n    mindllm_config.B\n}\n",
        &["cfg.A", "mindllm_config.B"],
    );
}

/// POSITIVE CONTROLS for every assertion above.
///
/// `assert_kept` only ever checks that a fragment IS present, and a `contains` check on a
/// formatter that returned its input verbatim would pass every test in this file. These pin the
/// two things that must still be true for the checks above to mean anything: the formatter does
/// transform its input, and an export with NO category still prints the brace form rather than
/// acquiring a spurious one.
#[test]
fn positive_controls() {
    // The formatter really does rewrite: `use` is canonicalised to `import`, so an output equal
    // to its input would mean nothing was formatted at all.
    let out = fmt("use mindllm_config as config\n\npub fn f() -> i64 {\n    1\n}\n");
    assert!(
        out.contains("import ") && !out.contains("use "),
        "formatter did not apply the use->import canonicalisation; \
         every `contains` assertion in this file would be vacuous.\n{out}"
    );

    // An uncategorised export must NOT gain a category word.
    let out = fmt("fn a() -> i64 {\n    1\n}\n\nexport { a }\n");
    assert!(
        out.contains("export { a }"),
        "uncategorised export lost its brace form.\n{out}"
    );
    for kw in ["export fn", "export struct", "export const", "export enum", "export type"] {
        assert!(
            !out.contains(kw),
            "uncategorised export invented the category {kw:?}.\n{out}"
        );
    }

    // An UNQUALIFIED call must not acquire a qualifier — the span-keyed lookup must miss when
    // nothing was recorded, or every bare call would grow a prefix.
    let out = fmt("fn g(x: i64) -> i64 {\n    x\n}\n\npub fn f(x: i64) -> i64 {\n    g(x)\n}\n");
    assert!(
        out.contains("g(x)") && !out.contains(".g(x)"),
        "an unqualified call acquired a qualifier.\n{out}"
    );
}
