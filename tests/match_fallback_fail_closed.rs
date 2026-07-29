// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! IR-audit finding #3 (the last HIGH silent-miscompile): the `match` sequential
//! fallback (`src/eval/lower.rs`) lowers every arm body and returns the LAST
//! arm's id, IGNORING the scrutinee — so `match x { … }` yields the last arm for
//! EVERY `x`. `abi_gate::check_match_runnable` rejects shapes that would hit that
//! fallback, but it used to walk ONLY `FnDef` bodies, and two lowering-site bails
//! dropped through to the fallback UNGATED. Three routes reached it silently;
//! this session lands two and DEFERS one:
//!
//!   1. MODULE-TOP-LEVEL match (`let y = match x { 1.0 => 10, _ => 20 }`) — a
//!      float pattern type-checks against a float scrutinee, so it bypassed the
//!      FnDef-only gate. FIXED: the gate now walks every top-level item + nested
//!      positions and REJECTS it with a loud blocker.
//!   2. UNKNOWN VARIANT TAG — an `EnumVariant` arm whose path is not in the
//!      merged tag registry used to bail the desugar to the fallback. DEFERRED:
//!      a poison here cannot distinguish a genuinely-unknown tag from a VALID
//!      module-wrapped variant (`module m { enum Mode { On, Off } }`), which is
//!      likewise absent from `enum_variant_tags` at lowering time. Upgrade path:
//!      register module-qualified variant tags before match lowering, then the
//!      registry miss is unambiguous. (See the `deferred:` marker in `lower.rs`.)
//!   3. VARIANT-NAME COLLISION — a non-final bare-identifier arm whose name
//!      equals a registered variant (`Some`, or a user variant) is neither a
//!      catch-all nor a discriminant test; it used to bail to the fallback.
//!      FIXED: it now REFUSES (panic, 0-byte artifact).
//!
//! Correctly-gated matches (exhaustive enum match, real `_`/final-bind catch-all)
//! are UNCHANGED — they still lower + run, byte-identity preserved.
//!
//! Gate: `cargo test --features "std-surface cross-module-imports"
//!                   --test match_fallback_fail_closed`

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::parser;
use libmind::pipeline::{CompileOptions, compile_source_with_name};

fn must_parse(src: &str) -> libmind::ast::Module {
    parser::parse(src).unwrap_or_else(|errs| {
        panic!(
            "parse failed with {} error(s):\n{}",
            errs.len(),
            errs.iter()
                .map(|e| format!("  {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    })
}

// ── Route 1: module-top-level degenerate match is now GATED, not miscompiled ──

#[test]
fn module_level_float_match_is_gated_not_return_last_arm() {
    // `match x { 1.0 => 10, _ => 20 }` at MODULE scope: the float-literal arm
    // bails `desugar_match_to_if`, and the sequential fallback returns 20 for
    // EVERY x. The FnDef-only gate missed this module-level position. It must now
    // be rejected as a runnable blocker (loud, no artifact) rather than compiling
    // to a scrutinee-ignoring constant.
    let p = compile_source_with_name(
        "let x: f64 = 1.0\nlet y = match x { 1.0 => 10, _ => 20 }",
        None,
        &CompileOptions::default(),
    )
    .expect("should parse + type-check");
    assert!(
        p.runnable_blockers
            .iter()
            .any(|d| d.code == "lower::match_pattern_unsupported"),
        "a module-level float-literal match must be gated (route 1), got: {:?}",
        p.runnable_blockers
    );
}

// Route 2 (unknown variant tag) is DEFERRED — see the module doc comment and the
// `deferred:` marker in `src/eval/lower.rs`. A regression test is intentionally
// omitted: with module-wrapped enums also absent from the tag registry, there is
// no source shape that isolates a genuinely-unknown tag from a valid one, so any
// assertion would either be a false-positive or duplicate route 3.

// ── Route 3: variant-name collision in a non-final arm REFUSES (0-byte) ──

#[test]
fn nonfinal_bare_ident_named_like_variant_refuses_not_fallback() {
    // `Flag` is a bare-identifier arm whose name equals the registered variant
    // `E::Flag`, so `is_catch_all` rejects it as a catch-all; sitting in a
    // non-final (test) slot it is neither a discriminant test nor a catch-all.
    // The old `(None, None)` bail dropped the match to the return-last-arm
    // fallback; it must now REFUSE (panic during lowering = 0-byte artifact).
    let module = must_parse(
        "enum E { Flag, Other }\nfn f(x: i64) -> i64 {\n    match x { Flag => 1, _ => 2 }\n}",
    );
    let result = std::panic::catch_unwind(|| lower_to_ir(&module));
    assert!(
        result.is_err(),
        "a non-final bare-ident arm colliding with a registered variant must \
         fail-closed (panic), NOT drop to the sequential fallback (route 3)."
    );
}

#[test]
fn nonfinal_prelude_variant_ident_refuses_not_fallback() {
    // Same shape using a PRELUDE variant name (`Some`). Referencing `Some(..)`
    // registers the Option prelude tags, so bare `Some` in a non-final test slot
    // collides with `Option::Some` and must REFUSE, not fall back.
    let module = must_parse(
        "fn f(x: i64) -> i64 {\n    let _o = Some(1)\n    match x { Some => 1, _ => 2 }\n}",
    );
    let result = std::panic::catch_unwind(|| lower_to_ir(&module));
    assert!(
        result.is_err(),
        "a non-final bare-ident arm colliding with a prelude variant (`Some`) must \
         fail-closed (panic), NOT drop to the sequential fallback (route 3)."
    );
}

// ── Correctly-gated matches are UNCHANGED (byte-identity preserved) ──

#[test]
fn correct_exhaustive_and_catch_all_matches_still_lower() {
    // (d) The must-not-regress twin: a genuinely-exhaustive fieldless enum match
    // and an int match with a REAL `_` catch-all must still lower with NO blocker
    // and NO panic — the sequential fallback's known-safe path (a lone catch-all)
    // and the well-formed If-chain path are untouched.
    let src = "enum Mode { On, Off }\n\
               fn pick(m: Mode) -> i64 { match m { Mode::On => 1, Mode::Off => 0 } }\n\
               fn classify(n: i64) -> i64 { match n { 0 => 100, 1 => 200, _ => 300 } }\n\
               fn ident_catch(n: i64) -> i64 { match n { 0 => 7, k => k } }";
    let p = compile_source_with_name(src, None, &CompileOptions::default())
        .expect("should parse + type-check");
    assert!(
        p.runnable_blockers.is_empty(),
        "correctly-gated matches must NOT be blocked, got: {:?}",
        p.runnable_blockers
    );
    // And lowering must NOT panic for any of these well-formed shapes.
    let module = must_parse(src);
    let result = std::panic::catch_unwind(|| lower_to_ir(&module));
    assert!(
        result.is_ok(),
        "a well-formed exhaustive/catch-all match must lower cleanly, never refuse."
    );
}
