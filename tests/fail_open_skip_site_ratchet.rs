// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The fail-OPEN skip-and-return prohibition. The ratchet's floor is now ZERO.
//!
//! # What this gate forbids
//!
//! A wide class of integration-test sites carried the shape
//! `if !something_available() { println!("... skipping"); return; }` without
//! ever consulting `MIND_BENCH_REQUIRE`, so `MIND_BENCH_REQUIRE=1` could not
//! turn them into a hard failure and a tier could pass vacuously. Worse, cargo
//! captures the stdout of a PASSING test, so the announcement was not merely
//! tolerated, it was *unobservable*: the tier gate could not tell "asserted"
//! from "executed".
//!
//! The fail-CLOSED helper every such site must route through is `common::gate`,
//! whose contract is proven in `tests/fail_closed_capability_skip.rs`.
//!
//! # Why there is no backlog table any more
//!
//! The first version pinned the CARDINALITY (`n == 256`), which forbids
//! nothing. The second froze a per-FILE table — 136 hand-copied names beside a
//! scanner deriving its scope from disk, two scopes that must agree with
//! nothing asserting they do. Both are gone: the backlog is drained to zero and
//! the gate is a flat prohibition over `tests/**/*.rs`.
//!
//! # The shapes this gate RECOGNISES — and the residuals it does not
//!
//! This enumeration is the claim; `.github/workflows/ci.yml` and
//! `tests/common/gate.rs` point here instead of restating it. The earlier
//! wording — "every skip-and-return site routes through `common::gate`" — was
//! wider than any detector backing it and so read as coverage of exactly the
//! gap it was blind to. Every entry names the escape it closed, each found by
//! mutating the tree under the built scanner, each with LIVE sites riding it
//! while this gate reported clean:
//!
//! 1. An ANNOUNCED skip — `println!`/`eprintln!` naming a skip or a deferral,
//!    possibly wrapped over lines — followed by a `return` OR by a bare `None`
//!    ([`is_bare_none`]). The tail used to require the token `return`, so a
//!    probe that printed its skip and handed the caller `None` was unseen: 4
//!    live sites (`mindc_cache_phase_f`, `g2_differential_mlir`,
//!    `mindc_build_phase_a`, `mlir_build`).
//! 2. A SILENT capability probe: an `if` / `match` / `let ... else` whose
//!    scrutinee is derived from a call in [`skip_shape_scan::AVAILABILITY_PROBES`]
//!    or [`skip_shape_scan::INPUT_READS`], and whose block (or its `else`
//!    branch) exits early without a verdict. Decided over the BRACE-BALANCED
//!    BLOCK by [`skip_shape_scan`], not a line window. The scan was once keyed
//!    on a print macro, so `if !bin.exists() { return; }` was invisible: 17 live
//!    sites. The vocabulary then had no `.success()`, so the FAILED-COMPILE skip
//!    this helper exists for was invisible too: 3 live sites in
//!    `mindc_cache_phase_f`, each grading `ok` on a build that printed
//!    `error[build][E5003]`. And the shape was pinned as TEXT — one probe line
//!    plus a two-line tail — so five ordinary Rust spellings of it passed green
//!    while the gate printed `12 passed`. The vocabulary was then an ALLOW-LIST
//!    OF FOUR calls while this header claimed a residual of three named shapes,
//!    so every other ordinary spelling of the same question — `.is_file()`,
//!    `.is_dir()`, `metadata()`, `try_exists()`, the non-`_os` `var()`,
//!    `.output()`, `.status()`, and a swallowed `read_to_string()` — walked past
//!    it: 3 live sites, one of them on the TRACKED self-host source
//!    (`stmt_keyword_recognizer.rs` graded a missing `main.mind` as a pass) and
//!    one an unrouted ELF-magic skip in `g2_differential_mlir.rs` that
//!    `MIND_BENCH_REQUIRE=1` could not turn into a failure. The vocabulary and
//!    its per-spelling controls are now a BIJECTION, asserted by
//!    [`skip_shape_scan::controls::the_vocabulary_and_its_controls_agree`], so a
//!    spelling can no longer be added without a control proving it fires.
//! 3. The superstring capability test, matched variable-agnostically. It used
//!    to be one literal spelled `stderr.`, so the same predicate under any
//!    other receiver was invisible: 8 live sites.
//! 4. A second reader of `MIND_BENCH_REQUIRE`. `ROUTED_MARKERS` used to accept
//!    the bare strings `MIND_BENCH_REQUIRE` / `enforce_real_backend`, so a
//!    comment merely naming the variable bought a pass; only `gate::`-qualified
//!    call syntax counts now. 17 sites rode it, each hand-rolling
//!    `var_os("MIND_BENCH_REQUIRE").is_some()` — which made
//!    `MIND_BENCH_REQUIRE=0` ENFORCE. They call `gate::skipped`.
//! 5. A `skipped_optional` call site that does not name what is optional.
//! 6. A skip decided by a child process's EXIT STATUS that never decides the
//!    CAUSE. `gate::skipped` names the absence class at the call site and never
//!    reads the failure, so under `MIND_BENCH_REQUIRE=1` it fails closed but on
//!    every other run it grades a compiler regression as a capability gap. The
//!    marker was in `ROUTED_MARKERS`, so the block scan above walked straight
//!    past it: 6 live sites in `phase_g_keystone_bootstrap.rs`, the WEDGE gate,
//!    where a `[E1001]` parse error in the tracked self-host source printed
//!    `ran=0 class=toolchain` and four keystone tests reported `ok`. An
//!    exit-status head must reach `gate::compiled` or `gate::is_capability_gap`;
//!    the subtraction that decides which markers count is owned by
//!    `skip_shape_scan::classifies_text`.
//!
//! The shapes below are the ones this scanner still cannot see, each knowingly
//! deferred. Each was MEASURED escaping the shipped scanner — handed to
//! `skip_shape_scan::skip_block_sites` beside a control that the same run
//! reported — rather than guessed; a residual nobody names is the hole the last
//! escapes came through. What this list does NOT claim is that the tree holds
//! none of them: that is precisely the question a blind detector cannot answer,
//! and the previous wording ("None exists in the tree today") asserted it anyway
//! while `stmt_keyword_recognizer.rs` was riding the fourth bullet below.
//!
//! * A skip announced ONLY in a comment above a bare early-out
//!   (`// LLVM not available; skip corruption check`). 3 such sites were
//!   measured when this bullet was written, all legitimate (2 documented
//!   `STABILITY_SKIP_LIST` gaps in `fmt_stdlib_stability.rs`, 1 guarded in
//!   `mindfuzz_cross_substrate.rs`), so adding it lands red on non-defects.
//!   Upgrade path: route those three.
//! * A COLLAPSED `Option` probe — `if b.exists() { Some(b) } else { None }` on
//!   ONE line. The bare-`None` early-out is decided per LINE because a
//!   token-level `None` matches every `None =>` match PATTERN in the tree.
//!   Upgrade path: require `-> Option<_>` probes to return `gate::compiled`.
//! * A probe behind an INDIRECTION — `if !have_toolchain() { return; }`. Taint
//!   follows a `let` inside one function, never across a call. Upgrade path:
//!   seed the taint set from the return expression of same-file helpers.
//! * An INPUT READ bound to a local whose failure is tested later —
//!   `let r = fs::read_to_string(p); if r.is_err() { return; }`.
//!   [`skip_shape_scan::INPUT_READS`] deliberately do not TAINT: their `Ok` is
//!   payload, not an availability answer, and tainting them flagged the SUCCESS
//!   path of `fmt_stdlib_stability::check_or_skip`. Read failures still count in
//!   a head scrutinee, which is where every live instance sat.
//! * An EXIT STATUS bound to a local before it is tested —
//!   `let ok = out.status.success(); if !ok { gate::skipped(..); return; }`. The
//!   taint set carries the NAME but not the CLASS, so the head reads as an
//!   ordinary probe and the sixth rule above never applies to it. No live site
//!   spells it that way today (pinned as EV5, which is caught as a plain
//!   fail-open skip only because it routes nowhere). Upgrade path: taint with
//!   the probe entry that produced the answer, not with a bare name.
//!
//! The first four share ONE upgrade path, and it is the one worth taking:
//! invert the rule so an UNROUTED bare early-out anywhere inside a `#[test]`
//! body under `tests/` is red REGARDLESS of scrutinee, with an explicit
//! `// not-a-skip:` annotation for the legitimate cases. That fails closed on
//! probe spellings nobody has thought of yet, which is the only version of
//! this gate whose claim cannot outrun its detector.
//!
//! # Scope is read out of the thing being checked
//!
//! The scanned set is `tests/**/*.rs` walked from disk (minus this scanner's
//! own source, whose positive controls must quote the shape it forbids), and
//! the routed-detection is the same `ROUTED_MARKERS` list applied to each of
//! them. Neither is written down twice, so they cannot drift apart.
//!
//! `tests/skip_shape_scan/mod.rs`, which OWNS the vocabulary and its positive
//! controls, is deliberately NOT exempted: it is walked like every other source
//! and comes back clean, because it reads masked CODE and both its vocabulary
//! and its specimens are spelled as string literals. A two-entry exemption list
//! would be the second hand-copied scope this file's doctrine is against. The
//! controls live beside the rule rather than here for the same reason: the
//! vocabulary and the proof it can SEE each spelling are one diff, not two.
//!
mod skip_shape_scan;

use std::path::{Path, PathBuf};

/// Does any CODE line in `window` call the shared helper?
///
/// Comment lines are excluded, and that exclusion is load-bearing: the marker
/// test used to run over the raw window text, so a doc comment mentioning
/// `gate::skipped` within eight lines above an UNROUTED skip bought it a pass —
/// the doc comment written for `mlir_build::resolve_or_skip` exempted the very
/// site it described. The marker LIST, and why only `gate::`-qualified call
/// syntax counts, are owned by [`skip_shape_scan::ROUTED_MARKERS`].
fn routed(window: &[&str]) -> bool {
    window
        .iter()
        .filter(|l| !l.trim_start().starts_with("//"))
        .any(|l| skip_shape_scan::routed_text(l))
}

/// The last line index of the print macro starting at `lines[i]`, or `None` if
/// `lines[i]` does not open one outside a comment.
///
/// A single-line match is NOT enough: the announcement this scanner exists to
/// find is routinely wrapped across lines, e.g.
///
/// ```text
/// println!(
///     "target: rebuilt .so is not an ELF - \
///      toolchain unavailable, skipping test"
/// );
/// ```
///
/// where the word "skipping" sits three lines below the macro name. Measured
/// while building this scanner: two such sites read as clean under a
/// line-anchored match. A scanner blind to the wrapped form is the same
/// false-green it is meant to catch, so the span is followed to its `);`.
fn print_macro_span(lines: &[&str], i: usize) -> Option<usize> {
    let t = lines[i].trim_start();
    if t.starts_with("//") {
        return None; // prose quoting the bad shape is not the bad shape
    }
    if !t.contains("println!(") && !t.contains("eprintln!(") {
        return None;
    }
    let last = (i + MAX_MACRO_SPAN).min(lines.len() - 1);
    for (n, line) in lines.iter().enumerate().take(last + 1).skip(i) {
        if line.contains(");") {
            return Some(n);
        }
    }
    Some(last)
}

/// A wrapped print macro is followed at most this many lines to its `);`.
const MAX_MACRO_SPAN: usize = 5;

/// Does the print macro spanning `lines[i..=end]` announce a skip?
///
/// "defer" counts too. A deferral is a legitimate, honestly-named outcome (the
/// VNNI rung is deferred by design), but it is still a gate that did not run —
/// if the word were outside this vocabulary, renaming a skip to a deferral
/// would be a one-word escape from this prohibition.
fn is_skip_announcement(lines: &[&str], i: usize, end: usize) -> bool {
    let text = lines[i..=end].join("\n").to_ascii_lowercase();
    text.contains("skip") || text.contains("defer")
}

/// This scanner's OWN source. It is excluded from the scan because its positive
/// control must quote the very shape it forbids; a scanner that flagged its own
/// specimen jar would be unable to prove it can still see the bad shape.
const SCANNER_SELF: &str = "fail_open_skip_site_ratchet.rs";

/// Every `tests/**/*.rs` source except this scanner, with its path RELATIVE to
/// `tests/` — the identity a failure is reported by, so two same-named files in
/// different directories can never be confused for one entry.
fn test_sources() -> Vec<(String, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).expect("read tests dir") {
            let p = e.expect("dir entry").path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut paths = Vec::new();
    walk(&root, &mut paths);
    paths.sort();
    paths
        .into_iter()
        .filter(|p| p.file_name().and_then(|s| s.to_str()) != Some(SCANNER_SELF))
        .map(|p| {
            let text = std::fs::read_to_string(&p).expect("read test source");
            let rel = p
                .strip_prefix(&root)
                .expect("test source under tests/")
                .to_string_lossy()
                .replace('\\', "/");
            (rel, text)
        })
        .collect()
}

/// Sites that ANNOUNCE a skip, return, and never consult the fail-closed
/// helper, as `<path relative to tests/>:<line>`.
fn announced_skip_sites(rel: &str, lines: &[&str]) -> Vec<String> {
    let mut open = Vec::new();
    for i in 0..lines.len() {
        let Some(end) = print_macro_span(lines, i) else {
            continue;
        };
        if !is_skip_announcement(lines, i, end) {
            continue;
        }
        let tail_end = (end + 4).min(lines.len());
        if !lines[i..tail_end]
            .iter()
            .any(|l| l.contains("return") || is_bare_none(l))
            && !falls_off_the_fn(lines, i, end)
        {
            continue;
        }
        let head = i.saturating_sub(8);
        if routed(&lines[head..tail_end]) {
            continue;
        }
        open.push(format!("{}:{} (announced skip)", rel, i + 1));
    }
    open
}

/// Does the announcement spanning `lines[start..=end]` end its top-level fn
/// without a verdict?
///
/// Two shapes exit as surely as a `return`: the announcement inside a block
/// that closes straight into the fn's own column-0 `}` (`if missing {
/// println!("skipped"); }` as a test's last statement), and a fn whose WHOLE
/// body is the announcement (a `#[cfg(not(..))]` stub). The tail rule above,
/// keyed on `return` / `None`, saw neither. A summary print as the last
/// top-level statement after real work is neither shape.
fn falls_off_the_fn(lines: &[&str], start: usize, end: usize) -> bool {
    let whole_body = start > 0
        && lines[start - 1].trim_end().ends_with('{')
        && lines[start - 1].starts_with("fn ");
    let mut nested = false;
    for line in lines.iter().skip(end + 1).take(MAX_MACRO_SPAN) {
        match line.trim() {
            "" => {}
            "}" if line.starts_with('}') => return nested || whole_body,
            "}" => nested = true,
            _ => return false,
        }
    }
    false
}

/// The identifier run starting at byte 0 of `s`, as a byte length.
///
/// Rust identifiers are ASCII here, so a byte count is exact and lets the
/// caller index straight back into the slice.
fn ident_len(s: &str) -> usize {
    s.bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
        .count()
}

/// An early-out spelled as a bare `None` value rather than a `return`.
///
/// THE BLIND SPOT THIS CLOSES: the announced-skip detector required the token
/// `return` under the announcement, so `fn require_mindc() -> Option<PathBuf>`
/// — `eprintln!("SKIP: ...")` then `None`, every caller writing
/// `None => return` — was invisible while ANNOUNCING its skip in plain text.
/// Handing back `None` and returning are one decision, written two ways.
fn is_bare_none(line: &str) -> bool {
    matches!(line.trim(), "None" | "None," | "None;")
}

fn open_skip_sites() -> Vec<String> {
    let mut open = Vec::new();
    for (rel, text) in test_sources() {
        let lines: Vec<&str> = text.lines().collect();
        open.extend(announced_skip_sites(&rel, &lines));
        open.extend(skip_shape_scan::skip_block_sites(&rel, &text));
    }
    open
}

#[test]
fn the_scan_scope_is_not_empty() {
    // A prohibition over an empty set is the `ran=0` defect it exists to
    // forbid. This asserts the scanner actually read the tree before the
    // negative assertions below are allowed to mean anything.
    let sources = test_sources();
    assert!(
        sources.len() > 250,
        "the scan walked only {} tests/**/*.rs sources; the tree holds far more, \
         so the prohibition below would be vacuous",
        sources.len()
    );
    assert!(
        sources.iter().any(|(p, _)| p == "common/gate.rs"),
        "the scan must reach tests/common/, the module every routed site names"
    );
}

#[test]
fn no_test_source_holds_a_fail_open_skip_site() {
    // THE DEFECT THIS PINS: `println!("... skipping"); return;` exits 0, so a
    // missing prerequisite — or a FAILING compile — is reported as a passing
    // test. There is no backlog and no tolerated file: every skip decision goes
    // through common::gate, which panics under MIND_BENCH_REQUIRE=1 and
    // otherwise emits `SDLC-GATE <target> ran=0 fail=0`.
    let open = open_skip_sites();
    assert!(
        open.is_empty(),
        "{} skip-and-return site(s) do not consult the shared fail-closed \
         helper, or consult it without deciding the cause. A skip that never \
         reads MIND_BENCH_REQUIRE cannot be turned into a hard failure, so the \
         tier can pass vacuously; a skip decided by an EXIT STATUS that never \
         classifies the failure grades a compiler regression as a capability \
         gap on every unenforced run. Route a probe through \
         common::gate::{{skipped, skipped_optional}} and a compile through \
         common::gate::compiled.\n  {}",
        open.len(),
        open.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// ONE OWNER for the fail-closed predicate itself.
// ---------------------------------------------------------------------------

/// The single file allowed to read `MIND_BENCH_REQUIRE` out of the environment.
const PREDICATE_OWNER: &str = "common/gate.rs";

/// The env accessors that READ a variable's value.
///
/// `Command::env` / `Command::env_remove` SET a child's environment and are how
/// the enforcement path is tested, so they are deliberately absent: writing the
/// variable for a child is not a second reader of the rule.
const ENV_READS: &[&str] = &["var_os(", "var("];

/// Does `line` read the fail-closed variable straight from the environment?
///
/// Matched by ACCESSOR plus the variable's identity — its literal name or the
/// `REQUIRE_VAR` constant that spells it — never by the surrounding shape. The
/// site this closes on is an `assert!(var_os(..).is_none(), ..)` inside a
/// `-> bool` helper: it announces no skip and returns no bare `return`, so it
/// was structurally invisible to BOTH detectors above while implementing a
/// competing copy of the very rule they enforce routing to.
fn reads_require_var_from_env(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with("//") {
        return false; // prose naming the variable is not a second reader
    }
    let names_it = t.contains("MIND_BENCH_REQUIRE") || t.contains("REQUIRE_VAR");
    names_it && ENV_READS.iter().any(|r| t.contains(r))
}

#[test]
fn the_fail_closed_predicate_has_exactly_one_owner() {
    // THE DEFECT THIS PINS: two readers of one variable drifted into two
    // DIFFERENT rules. `gate::enforce_real_backend` requires the value `1`; the
    // hand-rolled `assert!(var_os("MIND_BENCH_REQUIRE").is_none())` made ANY
    // value enforce, so a value that reads as "off" hard-failed one gate while
    // its neighbour skipped cleanly under the identical environment. Measured
    // before this arm existed, MLIR tools off PATH and MIND_BENCH_REQUIRE=0:
    //
    //   mindfuzz_cross_substrate  -> panicked "MIND_BENCH_REQUIRE is set but
    //                                'mlir-opt' is not on PATH" -> FAILED
    //   cross_substrate_identity  -> SDLC-GATE ... ran=0 fail=0 -> ok, 1 passed
    //
    // One variable, one rule, one reader.
    let mut second_owners = Vec::new();
    for (rel, text) in test_sources() {
        if rel == PREDICATE_OWNER {
            continue;
        }
        for (i, line) in text.lines().enumerate() {
            if reads_require_var_from_env(line) {
                second_owners.push(format!("{}:{}", rel, i + 1));
            }
        }
    }
    assert!(
        second_owners.is_empty(),
        "these sites read MIND_BENCH_REQUIRE from the environment instead of \
         asking common::gate, which owns the rule. A second reader is a second \
         RULE: the two spellings measured here disagreed about the value `0`. \
         Route the decision through common::gate::{{skipped, skipped_optional, \
         compiled, enforce_real_backend}}.\n  {}",
        second_owners.join("\n  ")
    );
}

#[test]
fn the_scanner_can_see_a_second_owner_of_the_predicate() {
    // Positive control for the FIFTH detector. A prohibition that matches
    // nothing is the ran=0 defect this file exists to forbid, in its active
    // form — and this detector's whole reason to exist is that the two above it
    // reported a clean tree while a competing predicate was live.
    let var = "MIND_BENCH_REQUIRE";
    for spelling in [
        format!("            std::env::var_os(\"{var}\").is_none(),"),
        format!("    if std::env::var(\"{var}\").is_ok() {{"),
        format!("    let on = env::var_os(\"{var}\").is_some();"),
        // Reading the OWNER's constant is still a second reader of the rule.
        "    let on = std::env::var_os(gate::REQUIRE_VAR).is_some();".to_string(),
    ] {
        assert!(reads_require_var_from_env(&spelling), "missed: {spelling}");
    }
    // Routing through the owner, SETTING a child's environment, and prose that
    // merely names the variable are all legitimate and must not be flagged.
    for ok in [
        "    if crate::common::gate::enforce_real_backend() {".to_string(),
        "        cmd.env(gate::REQUIRE_VAR, v);".to_string(),
        "            .env_remove(gate::REQUIRE_VAR);".to_string(),
        format!("    // NOTE: this gate does not honour {var}"),
        format!("        \"{var} is set but the artifact is a stub\","),
    ] {
        assert!(!reads_require_var_from_env(&ok), "false positive: {ok}");
    }
}

/// The exact fail-OPEN string a prior finding named, ASSEMBLED at run time.
///
/// Spelling it as one literal would make this scanner a hit on its own scan and
/// on the shell gate that greps the same text, so the specimen is built from
/// halves. It is still one definition, used by both the assertion and the
/// scanner's positive control.
fn banned_literal() -> String {
    format!("{}{}", "compile failed", "; skipping")
}

/// Does `line` re-implement the banned two-substring capability test?
///
/// The banned predicate ANDed two floating `contains(...)` probes for the
/// feature name and the word "requires". It is superstring-satisfiable: ANY
/// compiler failure whose stderr happens to carry both tokens anywhere graded as
/// a capability gap. The classifier that replaced it matches the stable
/// diagnostic CODE, owned by `libmind::diagnostics::capability`.
///
/// VARIABLE-AGNOSTIC BY CONSTRUCTION. The needle used to be one literal spelled
/// with the receiver name `stderr.`, so the identical predicate written over
/// `e.`, `err.` or `es_err.` was invisible — measured, eight live sites kept it
/// while this gate reported the shape gone. Only the two method calls and the
/// `&&` between them are matched; the receivers are read as identifier runs of
/// any name. The tokens are assembled at run time so a repo-wide grep for the
/// bad shape returns call sites only.
fn is_superstring_capability_test(line: &str) -> bool {
    let feature = format!(".contains(\"{}\")", "mlir-build");
    let word = format!(".contains(\"{}\")", "requires");
    let Some(p) = line.find(&feature) else {
        return false;
    };
    // A receiver of ANY name must sit left of the first probe.
    if !line.as_bytes()[..p]
        .last()
        .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
    {
        return false;
    }
    let rest = line[p + feature.len()..].trim_start();
    let Some(rest) = rest.strip_prefix("&&") else {
        return false; // an `||` of unrelated tokens is not the banned AND-pair
    };
    let rest = rest.trim_start();
    let n = ident_len(rest);
    n > 0 && rest[n..].starts_with(&word)
}

#[test]
fn the_converted_compile_sites_are_gone() {
    // The exact fail-OPEN string the finding named. Zero is the whole point.
    let hits: Vec<String> = test_sources()
        .into_iter()
        .filter(|(_, t)| t.contains(&banned_literal()))
        .map(|(p, _)| p)
        .collect();
    assert!(hits.is_empty(), "fail-open compile skips remain: {hits:?}");
}

#[test]
fn the_two_substring_capability_test_is_gone() {
    // A re-worded diagnostic must not silently widen or close the skip hole, and
    // a program that merely NAMES the tokens must not be able to buy a pass.
    // The prose is not the contract; the code is.
    let mut hits = Vec::new();
    for (rel, text) in test_sources() {
        for (i, line) in text.lines().enumerate() {
            if is_superstring_capability_test(line) {
                hits.push(format!("{}:{}", rel, i + 1));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "these sites re-implement the superstring-satisfiable capability test \
         instead of calling common::gate::is_capability_gap (which delegates to \
         libmind::diagnostics::capability).\n  {}",
        hits.join("\n  ")
    );
}

/// The literal token every `skipped_optional` call site must carry overhead.
///
/// The check used to accept any comment containing the substring "opt", which
/// `// needs mlir-opt` satisfies — a comment naming a TOOLCHAIN gap bought a
/// pass for the one outcome `MIND_BENCH_REQUIRE` does not close. The token is
/// deliberately shouty and unlikely to appear by accident; all six current call
/// sites already carry it.
const OPTIONAL_INPUT_TOKEN: &str = "OPTIONAL INPUT";

/// Does a comment in the eight lines above `lines[i]` carry the token?
fn names_optional_input(lines: &[&str], i: usize) -> bool {
    let head = i.saturating_sub(8);
    lines[head..i]
        .iter()
        .any(|l| l.trim_start().starts_with("//") && l.contains(OPTIONAL_INPUT_TOKEN))
}

#[test]
fn every_optional_input_skip_names_what_is_optional() {
    // `skipped_optional` is the ONE outcome MIND_BENCH_REQUIRE does not close,
    // so it is the only place a vacuous pass could re-enter. It stays honest by
    // being greppable and self-documenting: each call site must carry a comment
    // in the eight lines above it naming what is optional and who supplies it.
    let mut undocumented = Vec::new();
    for (rel, text) in test_sources() {
        if rel == "common/gate.rs" {
            continue; // the definition, not a call site
        }
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains("gate::skipped_optional") || line.trim_start().starts_with("//") {
                continue;
            }
            if !names_optional_input(&lines, i) {
                undocumented.push(format!("{}:{}", rel, i + 1));
            }
        }
    }
    assert!(
        undocumented.is_empty(),
        "these opt-in skips carry no `{OPTIONAL_INPUT_TOKEN}` comment naming \
         what is optional and who supplies it; an undocumented one is \
         indistinguishable from a fail-open escape hatch.\n  {}",
        undocumented.join("\n  ")
    );
}

#[test]
fn the_scanner_itself_can_see_the_bad_shape() {
    // Positive control: a scanner that matches nothing would pass silently.
    fn one(l: &str) -> bool {
        let lines = vec![l];
        print_macro_span(&lines, 0).is_some_and(|e| is_skip_announcement(&lines, 0, e))
    }
    let specimen = format!("    println!(\"arena: mindc {}\");", banned_literal());
    assert!(one(&specimen));
    assert!(one(
        r#"        eprintln!("SKIP: MLIR tools not available");"#
    ));
    // Renaming a skip to a deferral must not escape the prohibition.
    assert!(one(r#"        println!("DEFER vnni: rung not run here");"#));
    assert!(!one(
        r#"    //! println!("... build failed -> skipped"); return;"#
    ));
    assert!(!one(r#"    println!("all good");"#));

    // The wrapped form: the word sits three lines below the macro name.
    let wrapped = vec![
        r#"        println!("#,
        r#"            "target: rebuilt .so is not an ELF - \"#,
        r#"             toolchain unavailable, skipping test""#,
        r#"        );"#,
    ];
    let end = print_macro_span(&wrapped, 0).expect("span must be found");
    assert_eq!(end, 3);
    assert!(is_skip_announcement(&wrapped, 0, end));
}

#[test]
fn the_scanner_reads_code_and_not_the_text_around_it() {
    // The masker is what makes a brace-balanced block possible in THIS tree:
    // test sources embed MIND programs, MLIR and JSON in string literals, all
    // full of braces. Without it the block after a probe runs to the wrong `}`.
    let masked = skip_shape_scan::mask_code("let s = \"fn f() { let x = 1; }\"; // {{{");
    assert_eq!(masked.len(), 1);
    assert!(
        !masked[0].contains('{'),
        "literal braces leaked: {}",
        masked[0]
    );
    assert!(masked[0].starts_with("let s = "));

    // A file that QUOTES the bad shape inside a string is not the bad shape —
    // which is what lets `skip_shape_scan` itself be scanned rather than
    // exempted. A raw string, a char literal and a block comment must all be
    // as inert as a plain one.
    for quoted in [
        "let bad = \"if !p.exists() { return; }\";",
        "let bad = r#\"if !p.exists() { return; }\"#;",
        "let brace = '{'; let done = p.exists();",
        "/* if !p.exists() { return; } */",
    ] {
        assert!(
            skip_shape_scan::skip_block_sites("specimen.rs", quoted).is_empty(),
            "quoted prose read as code: {quoted}"
        );
    }
}

#[test]
fn the_scanner_can_see_an_announced_skip_that_hands_back_none() {
    // Positive control. A `-> Option<T>` capability probe that PRINTS its skip
    // and yields `None` was invisible while every call site wrote
    // `None => return`: four live, against one sibling already routing it.
    let probe = [
        "    if bin.exists() {",
        "        Some(bin)",
        "    } else {",
        "        eprintln!(\"SKIP: mindc binary not found\");",
        "        None",
        "    }",
    ];
    assert_eq!(
        announced_skip_sites("specimen.rs", &probe),
        vec!["specimen.rs:4 (announced skip)".to_string()]
    );

    // Routing the same decision through the owner clears it.
    let routed = [
        "    if bin.exists() {",
        "        Some(bin)",
        "    } else {",
        "        gate::skipped(\"t\", \"no mindc\");",
        "        None",
        "    }",
    ];
    assert!(announced_skip_sites("specimen.rs", &routed).is_empty());

    // A doc comment MENTIONING the helper is prose, not routing. This exact
    // shape returned GREEN from the mutation written to prove the detector
    // bites: the comment describing the fix exempted the unrouted site.
    let commented = [
        "    /// `gate::skipped` panics under enforcement; otherwise it counts.",
        "    if bin.exists() {",
        "        Some(bin)",
        "    } else {",
        "        eprintln!(\"SKIP: mindc binary not found\");",
        "        None",
        "    }",
    ];
    assert_eq!(
        announced_skip_sites("specimen.rs", &commented),
        vec!["specimen.rs:5 (announced skip)".to_string()]
    );

    // A print-only skip as the test's LAST statement exits as surely as a
    // `return`; a print followed by more work does not.
    let falls_off = [
        "fn t() {",
        "    if stdout.contains(\"unavailable\") {",
        "        println!(\"round-trip skipped\");",
        "    }",
        "}",
    ];
    assert_eq!(
        announced_skip_sites("specimen.rs", &falls_off),
        vec!["specimen.rs:3 (announced skip)".to_string()]
    );
    let stub = [
        "#[test]",
        "fn t() {",
        "    println!(\"t: SKIP -- gated to linux\");",
        "}",
    ];
    assert_eq!(
        announced_skip_sites("specimen.rs", &stub),
        vec!["specimen.rs:3 (announced skip)".to_string()]
    );
    let summary = [
        "fn t() {",
        "    let n = run();",
        "    eprintln!(\"summary: {n} passed, 2 skipped\");",
        "}",
    ];
    assert!(announced_skip_sites("specimen.rs", &summary).is_empty());
    let continues = [
        "fn t() {",
        "    if stdout.contains(\"unavailable\") {",
        "        println!(\"round-trip skipped\");",
        "    }",
        "    assert_eq!(n, 3);",
        "}",
    ];
    assert!(announced_skip_sites("specimen.rs", &continues).is_empty());

    // Every early-out spelling, and a `None` that is a real value, not an exit.
    assert!(is_bare_none("        None"));
    assert!(is_bare_none("        None,"));
    assert!(is_bare_none("    None;"));
    assert!(!is_bare_none("        Some(bin)"));
    assert!(!is_bare_none("        None => return,"));
}

#[test]
fn the_scanner_can_see_the_capability_test_under_any_receiver_name() {
    // Positive control for the THIRD detector. The old needle was one literal
    // spelled `stderr.`; these four receivers are the ones measured live.
    for name in ["stderr", "e", "err", "es_err"] {
        let line = format!(
            "    if {name}.contains(\"{}\") && {name}.contains(\"{}\") {{",
            "mlir-build", "requires"
        );
        assert!(is_superstring_capability_test(&line), "missed: {line}");
    }
    // Mixed receivers, and the negated form, are the same banned predicate.
    assert!(is_superstring_capability_test(&format!(
        "    !(e.contains(\"{}\") && stderr.contains(\"{}\"))",
        "mlir-build", "requires"
    )));
    // An `||` of unrelated tokens is a different, legitimate assertion.
    assert!(!is_superstring_capability_test(&format!(
        "        stderr.contains(\"tool not found\") || stderr.contains(\"{}\"),",
        "mlir-build"
    )));
    // The routed replacement must not read as the banned shape.
    assert!(!is_superstring_capability_test(
        "        if crate::common::gate::is_capability_gap(&e) {"
    ));
}

#[test]
fn the_optional_input_check_rejects_a_merely_opt_shaped_comment() {
    // Positive control for the FOURTH check. `// needs mlir-opt` contains "opt"
    // and used to buy a pass for the one outcome MIND_BENCH_REQUIRE cannot
    // close — while naming a TOOLCHAIN gap, the class that must fail closed.
    let weak = [
        "        // needs mlir-opt",
        "        gate::skipped_optional(",
    ];
    assert!(!names_optional_input(&weak, 1));
    let strong = [
        "        // OPTIONAL INPUT: supplied by the operator via an env var.",
        "        gate::skipped_optional(",
    ];
    assert!(names_optional_input(&strong, 1));
}
