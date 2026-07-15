// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `#[bimap]` single-source bijection derive — Phase 1 fixtures.
//!
//! Every module-consuming front-end obtains its `Module` through the single
//! expansion chokepoint (`parser::parse` / `parser::parse_with_diagnostics`),
//! so these tests exercise the WIRED pass, not an isolated helper. The must-fail
//! cases assert the exact E-code; the positive cases assert the three derived
//! functions are synthesised (and are `pub`, hence cross-module exportable).
//!
//! Run: `cargo test --features "std-surface cross-module-imports" --test bimap_derive`

use libmind::ast::{Module, Node};
use libmind::parser::{parse, parse_with_diagnostics};

/// Names of every function defined in the module.
fn fn_names(m: &Module) -> Vec<String> {
    m.items
        .iter()
        .filter_map(|it| match it {
            Node::FnDef { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn has_fn(m: &Module, name: &str) -> bool {
    fn_names(m).iter().any(|n| n == name)
}

/// Collect the E-codes of the diagnostics a source rejects with.
fn reject_codes(src: &str) -> Vec<String> {
    match parse_with_diagnostics(src) {
        Ok(_) => Vec::new(),
        Err(diags) => diags.iter().map(|d| d.code.to_string()).collect(),
    }
}

// ── Positive ──────────────────────────────────────────────────────────────

#[test]
fn derives_the_three_functions() {
    let src = "#[bimap]\nenum Currency { AUD, JPY, USD }\n";
    let m = parse(src).expect("bimap enum must parse+expand");
    assert!(has_fn(&m, "currency_count"), "missing currency_count");
    assert!(has_fn(&m, "currency_to_str"), "missing currency_to_str");
    assert!(has_fn(&m, "currency_from_str"), "missing currency_from_str");
}

#[test]
fn generated_functions_are_pub() {
    let src = "#[bimap]\nenum Currency { AUD, JPY, USD }\n";
    let m = parse(src).expect("parse");
    for it in &m.items {
        if let Node::FnDef { name, is_pub, .. } = it {
            if name.starts_with("currency_") {
                assert!(
                    *is_pub,
                    "generated `{name}` must be pub (cross-module exportable)"
                );
            }
        }
    }
}

#[test]
fn explicit_pair_override_is_used() {
    // `= "euro"` overrides the default variant-name pairing.
    let src = "#[bimap]\nenum Currency { AUD, EUR = \"euro\" }\n";
    let m = parse(src).expect("parse");
    assert!(has_fn(&m, "currency_to_str"));
}

#[test]
fn snake_case_multiword_enum() {
    let src = "#[bimap]\nenum HttpStatus { Ok, NotFound }\n";
    let m = parse(src).expect("parse");
    assert!(has_fn(&m, "http_status_count"), "names: {:?}", fn_names(&m));
    assert!(has_fn(&m, "http_status_to_str"));
    assert!(has_fn(&m, "http_status_from_str"));
}

#[test]
fn non_bimap_enum_is_untouched() {
    // No `#[bimap]` ⇒ zero synthesised fns, byte-for-byte the old behaviour.
    let src = "enum Currency { AUD, JPY, USD }\n";
    let m = parse(src).expect("parse");
    assert!(
        fn_names(&m).is_empty(),
        "no fns should be synthesised: {:?}",
        fn_names(&m)
    );
}

// ── Coexistence (incremental adoption) ─────────────────────────────────────

#[test]
fn coexists_with_hand_written_converter() {
    let src = "import std.string;\n\
               #[bimap]\n\
               enum Currency { AUD, JPY, USD }\n\
               fn legacy_currency_name(k: i64) -> String { return \"AUD\"; }\n";
    let m = parse(src).expect("bimap + legacy converter must coexist");
    assert!(has_fn(&m, "currency_to_str"), "derived fn present");
    assert!(has_fn(&m, "legacy_currency_name"), "legacy fn kept");
}

// ── Must-fail, one per E-code ──────────────────────────────────────────────

#[test]
fn e2017_duplicate_key_rejected_via_dup_variant_guard() {
    // E2017 (dup key) is subsumed by the #165 parse-time dup-variant guard for
    // the enum form: a duplicate variant name is rejected BEFORE expansion, so
    // the module never parses. (E2017 is allocated + defensively checked in the
    // pass for the Phase-3 const-table form.)
    let src = "#[bimap]\nenum E { A, A }\n";
    assert!(parse(src).is_err(), "duplicate variant must be rejected");
}

#[test]
fn e2018_duplicate_value_rejected() {
    let src = "#[bimap]\nenum E { A = \"X\", B = \"X\" }\n";
    assert!(
        reject_codes(src).contains(&"E2018".to_string()),
        "expected E2018"
    );
}

#[test]
fn e2019_non_total_payload_variant_rejected() {
    let src = "#[bimap]\nenum E { A, B(i64) }\n";
    assert!(
        reject_codes(src).contains(&"E2019".to_string()),
        "expected E2019"
    );
}

#[test]
fn e2019_empty_enum_rejected() {
    let src = "#[bimap]\nenum E {}\n";
    assert!(
        reject_codes(src).contains(&"E2019".to_string()),
        "expected E2019"
    );
}

#[test]
fn e2020_non_string_discriminant_rejected() {
    let src = "#[bimap]\nenum E { A = 5, B }\n";
    assert!(
        reject_codes(src).contains(&"E2020".to_string()),
        "expected E2020"
    );
}

#[test]
fn e2021_generated_name_collision_rejected() {
    let src = "import std.string;\n\
               #[bimap]\n\
               enum Currency { AUD, JPY, USD }\n\
               fn currency_to_str(k: i64) -> String { return \"nope\"; }\n";
    assert!(
        reject_codes(src).contains(&"E2021".to_string()),
        "expected E2021"
    );
}

// ── Escape parity (checker compares DECODED bytes, one shared unescape) ─────

#[test]
fn escape_collision_is_a_duplicate_value() {
    // A real TAB byte and the `\t` escape decode to the same 0x09 — DIFFERENT
    // source spellings, SAME bytes ⇒ E2018. Proves the value check compares
    // decoded `Literal::Str` bytes, not raw source text.
    let src = "#[bimap]\nenum E { A = \"a\tb\", B = \"a\\tb\" }\n";
    assert!(
        reject_codes(src).contains(&"E2018".to_string()),
        "expected E2018 on decoded-byte collision"
    );
}

// ── Marker forgery + same-base collision (audit remediation) ───────────────

#[test]
fn e2022_forged_marker_does_not_suppress_derive() {
    // A user-authored `#[__bimap_generated]` marker on a `<base>_count` fn must
    // NOT make expansion skip the enum (which would suppress the whole derive
    // with no E2021, no synthesis). The reserved marker is rejected fail-loud.
    let src = "#[bimap]\n\
               enum Currency { AUD, JPY, USD }\n\
               #[__bimap_generated]\n\
               fn currency_count() -> i64 { return 999; }\n";
    assert!(
        reject_codes(src).contains(&"E2022".to_string()),
        "forged marker must be rejected E2022; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn e2021_same_snake_base_second_enum_rejected() {
    // `E` and `e` both snake_case-normalise to base `e`, so both derives want
    // `e_count`/`e_to_str`/`e_from_str`. The second is a generated-name collision
    // (E2021), NOT a silent skip that drops its derive.
    let src = "#[bimap]\nenum E { A, B }\n#[bimap]\nenum e { X, Y, Z }\n";
    assert!(
        reject_codes(src).contains(&"E2021".to_string()),
        "same-base second enum must be rejected E2021; got: {:?}",
        reject_codes(src)
    );
}

// ── Two-module: importer resolves the generated fns ────────────────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn generated_fns_resolve_across_modules() {
    use libmind::project::module_table::build_module_table;
    use libmind::type_checker::check_module_types_with_modules;
    use std::collections::HashMap;

    // Module A derives `currency_to_str` (pub) from a `#[bimap]` enum.
    let a_src = "#[bimap]\nenum Currency { AUD, JPY, USD }\n";
    let a = parse(a_src).expect("parse A");
    assert!(has_fn(&a, "currency_to_str"), "A must carry the derived fn");

    // Module B imports A and calls the DERIVED fn.
    let b_src = "use crate.a\nlet x = currency_to_str\n";
    let b = parse(b_src).expect("parse B");

    let table = build_module_table(&[("crate.a".to_string(), &a)]);
    let errs = check_module_types_with_modules(&b, b_src, Some("b.mind"), &HashMap::new(), &table);
    assert!(
        !errs
            .iter()
            .any(|e| format!("{e:?}").contains("currency_to_str")),
        "importer must resolve the derived `currency_to_str`; got: {errs:?}"
    );
}

// ── Formatter round-trip ────────────────────────────────────────────────────

/// The `= "string"` variant pairing MUST survive `mindc fmt`. The formatter runs
/// on the un-expanded module (via `parse_with_trivia`), so it sees the enum with
/// its `paired` values, not the synthesised functions — and it has to re-emit the
/// pairing or the bijection table is silently dropped and the file no longer
/// carries the bimap. This covers the whole-string decode→re-escape path
/// (`\t` must round-trip as the two chars `\t`, never a raw tab byte).
#[test]
fn formatter_round_trips_string_pairing() {
    use libmind::fmt::format_source;
    use libmind::project::MindcraftFormatConfig;

    // `Euro`'s pairing carries an escaped tab; the printer must re-escape it.
    let src = "#[bimap]\nenum Money {\n    Aud = \"aud\",\n    Euro = \"e\\tu\",\n}\n\nfn main() -> i64 {\n    return 0;\n}\n";
    let cfg = MindcraftFormatConfig::default();

    let once = format_source(src, &cfg).expect("bimap source must format");
    assert!(
        once.contains("= \"aud\""),
        "plain pairing dropped by formatter:\n{once}"
    );
    assert!(
        once.contains("= \"e\\tu\""),
        "escaped pairing not re-escaped by formatter:\n{once}"
    );
    assert!(
        !once.contains("e\tu"),
        "a RAW tab byte leaked into the formatted pairing (escape lost):\n{once}"
    );

    // Idempotent: formatting the formatted output is a fixed point.
    let twice = format_source(&once, &cfg).expect("re-format");
    assert_eq!(once, twice, "formatter is not idempotent on bimap pairings");

    // The formatted output is still valid MIND that expands (parse runs the pass).
    let m = parse(&once).expect("formatted bimap source must re-parse+expand");
    assert!(
        has_fn(&m, "money_to_str"),
        "derived fn lost after round-trip"
    );
    assert!(
        has_fn(&m, "money_from_str"),
        "derived fn lost after round-trip"
    );
}

#[test]
fn fmt_does_not_launder_invalid_bimap_table() {
    // `A = 5` is an E2020-invalid `#[bimap]` table. The formatter MUST re-emit the
    // non-string discriminant verbatim, otherwise `mindc fmt` drops the `= 5`,
    // turning an erroring program into a passing one (invalid → valid different).
    use libmind::fmt::format_source;
    use libmind::project::MindcraftFormatConfig;
    let src = "#[bimap]\nenum E {\n    A = 5,\n    B,\n}\n";
    let out = format_source(src, &MindcraftFormatConfig::default()).expect("format");
    assert!(
        out.contains("A = 5"),
        "fmt dropped the non-string discriminant:\n{out}"
    );
    assert!(
        reject_codes(&out).contains(&"E2020".to_string()),
        "fmt laundered an E2020-invalid table into a valid one: {:?}",
        reject_codes(&out)
    );
}

#[test]
fn malformed_discriminant_is_a_parse_error() {
    // A consumed `=` whose expression fails to parse must error, not silently
    // swallow — `A = ,` previously parsed as a bare `A`.
    assert!(
        parse("enum E { A = , B }\n").is_err(),
        "malformed discriminant must be a parse error"
    );
}
