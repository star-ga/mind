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
            Node::FnDef(fd, _) => Some(fd.name.clone()),
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
        if let Node::FnDef(fd, _) = it {
            let name = &fd.name;
            let is_pub = &fd.is_pub;
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

// ── Inverse mode stamps (Phase 2, std-surface) ──────────────────────────────

/// The second marker-attribute arg of `<base>_from_str` — the stamped inverse
/// strategy (`phf-v1` / `binsearch-v1`).
#[cfg(feature = "std-surface")]
fn from_str_mode(m: &Module, fn_name: &str) -> Option<String> {
    m.items.iter().find_map(|it| match it {
        Node::FnDef(fd, _) if fd.name == fn_name => {
            let name = &fd.name;
            let attrs = &fd.attrs;
            attrs.first().and_then(|a| a.args.get(1).cloned())
        }
        _ => None,
    })
}

#[cfg(feature = "std-surface")]
#[test]
fn in_envelope_from_str_stamps_phf_mode() {
    let src = "#[bimap]\nenum Currency { AUD, JPY, USD }\n";
    let m = parse(src).expect("parse");
    assert_eq!(
        from_str_mode(&m, "currency_from_str").as_deref(),
        Some("phf-v1"),
        "in-envelope key set must take the perfect-hash inverse"
    );
}

#[cfg(feature = "std-surface")]
#[test]
fn out_of_envelope_130_keys_stamps_binsearch_mode() {
    // 130 keys exceeds the PHF envelope (MAX_KEYS = 128), so the derive must
    // fall back to the O(log n) sorted-binary-search inverse — and say so.
    let mut src = String::from("#[bimap]\nenum Big {\n");
    for i in 0..130 {
        src.push_str(&format!("    V{i:03},\n"));
    }
    src.push_str("}\n");
    let m = parse(&src).expect("130-key bimap enum must parse+expand");
    assert_eq!(
        from_str_mode(&m, "big_from_str").as_deref(),
        Some("binsearch-v1"),
        "out-of-envelope key set must take the binary-search inverse"
    );
}

// ── E2023: reserved `__mind_` intrinsic prefix ─────────────────────────────

#[test]
fn e2023_user_mind_prefixed_fn_rejected() {
    // A user `fn __mind_*` would shadow a reserved intrinsic on the interpreter
    // fn-table oracle; it is rejected fail-loud at check time.
    use libmind::type_checker::check_module_types;
    let src = "fn __mind_load_i8(a: i64) -> i64 { return 7 }\n";
    let m = parse(src).expect("parse");
    let diags = check_module_types(&m, src, &Default::default());
    assert!(
        diags.iter().any(|d| d.code == "E2023"),
        "expected E2023 for `__mind_`-prefixed fn; got: {:?}",
        diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );
}

#[test]
fn e2023_ordinary_name_is_fine() {
    // The prefix guard must not fire for a normal name that merely contains
    // `mind` or a single leading underscore.
    use libmind::type_checker::check_module_types;
    let src = "fn _helper() -> i64 { return 1 }\nfn remind_me() -> i64 { return 2 }\n";
    let m = parse(src).expect("parse");
    let diags = check_module_types(&m, src, &Default::default());
    assert!(
        !diags.iter().any(|d| d.code == "E2023"),
        "E2023 must not fire on ordinary names; got: {:?}",
        diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );
}

// ── Two-module: importer resolves the generated fns ────────────────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn generated_fns_resolve_across_modules() {
    use libmind::project::module_table::build_module_table;
    use libmind::type_checker::check_module_types_with_modules;

    // Module A derives `currency_to_str` (pub) from a `#[bimap]` enum.
    let a_src = "#[bimap]\nenum Currency { AUD, JPY, USD }\n";
    let a = parse(a_src).expect("parse A");
    assert!(has_fn(&a, "currency_to_str"), "A must carry the derived fn");

    // Module B imports A and calls the DERIVED fn.
    let b_src = "use crate.a\nlet x = currency_to_str\n";
    let b = parse(b_src).expect("parse B");

    let table = build_module_table(&[("crate.a".to_string(), &a)]);
    let errs =
        check_module_types_with_modules(&b, b_src, Some("b.mind"), &Default::default(), &table);
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

// ── Phase 3: const pair-table forms (string<->string / number<->string) ─────

const SS_TABLE: &str =
    "#[bimap]\nconst COUNTRY = [(\"US\", \"United States\"), (\"JP\", \"Japan\")];\n";
const NS_TABLE: &str = "#[bimap]\nconst HTTP_STATUS = [(200, \"OK\"), (404, \"Not Found\")];\n";

#[test]
fn string_string_derives_count_value_key() {
    let m = parse(SS_TABLE).expect("string<->string const table must parse+expand");
    assert!(has_fn(&m, "country_count"), "missing country_count");
    assert!(
        has_fn(&m, "country_value"),
        "missing country_value (forward)"
    );
    assert!(has_fn(&m, "country_key"), "missing country_key (inverse)");
    // The hidden per-direction index fns carry the reused PHF/binsearch machinery.
    assert!(
        has_fn(&m, "__bimap_country_key_idx"),
        "missing key index fn"
    );
    assert!(
        has_fn(&m, "__bimap_country_value_idx"),
        "missing value index fn"
    );
}

#[test]
fn number_string_derives_count_to_str_from_str() {
    let m = parse(NS_TABLE).expect("number<->string const table must parse+expand");
    assert!(has_fn(&m, "http_status_count"), "missing http_status_count");
    assert!(
        has_fn(&m, "http_status_to_str"),
        "missing http_status_to_str (forward)"
    );
    assert!(
        has_fn(&m, "http_status_from_str"),
        "missing http_status_from_str (inverse)"
    );
}

#[test]
fn const_table_is_consumed_by_the_derive() {
    // A validated `#[bimap]` const has no independent lowering; the expansion
    // removes it from the item list so lowering never sees an array-of-tuples
    // const that would otherwise fail to lower.
    let m = parse(SS_TABLE).expect("parse");
    assert!(
        !m.items
            .iter()
            .any(|it| matches!(it, Node::Const { name, .. } if name == "COUNTRY")),
        "validated #[bimap] const must be consumed (removed) after expansion"
    );
}

#[test]
fn const_form_generated_fns_are_pub() {
    let m = parse(SS_TABLE).expect("parse");
    for it in &m.items {
        if let Node::FnDef(fd, _) = it {
            let name = &fd.name;
            let is_pub = &fd.is_pub;
            if name == "country_count" || name == "country_value" || name == "country_key" {
                assert!(*is_pub, "generated `{name}` must be pub");
            }
        }
    }
}

#[test]
fn ss_duplicate_key_rejected() {
    let src = "#[bimap]\nconst C = [(\"US\", \"a\"), (\"US\", \"b\")];\n";
    assert!(
        reject_codes(src).contains(&"E2017".to_string()),
        "string<->string dup key must be E2017; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn ss_duplicate_value_rejected() {
    let src = "#[bimap]\nconst C = [(\"US\", \"same\"), (\"GB\", \"same\")];\n";
    assert!(
        reject_codes(src).contains(&"E2018".to_string()),
        "string<->string dup value must be E2018; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn ns_duplicate_key_rejected() {
    let src = "#[bimap]\nconst H = [(404, \"a\"), (404, \"b\")];\n";
    assert!(
        reject_codes(src).contains(&"E2017".to_string()),
        "number<->string dup key must be E2017; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn ns_duplicate_value_rejected() {
    let src = "#[bimap]\nconst H = [(404, \"same\"), (410, \"same\")];\n";
    assert!(
        reject_codes(src).contains(&"E2018".to_string()),
        "number<->string dup value must be E2018; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn const_mixed_keys_rejected() {
    // A table that mixes a string key and a number key has no single derived
    // signature — reject E2020, do not silently pick one form.
    let src = "#[bimap]\nconst M = [(\"US\", \"a\"), (200, \"b\")];\n";
    assert!(
        reject_codes(src).contains(&"E2020".to_string()),
        "mixed-key table must be E2020; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn const_negative_number_key_rejected() {
    // Negative keys fall outside the 0..=2^32-1 field the emitted tables encode
    // ordinals in — reject loudly rather than silently truncate.
    let src = "#[bimap]\nconst H = [(0 - 1, \"a\"), (2, \"b\")];\n";
    assert!(
        !reject_codes(src).is_empty(),
        "negative number key must be rejected; got no diagnostics"
    );
}

#[test]
fn const_non_string_value_rejected() {
    let src = "#[bimap]\nconst C = [(\"US\", 5)];\n";
    assert!(
        reject_codes(src).contains(&"E2020".to_string()),
        "non-string value column must be E2020; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn const_empty_table_rejected() {
    let src = "#[bimap]\nconst C = [];\n";
    assert!(
        reject_codes(src).contains(&"E2019".to_string()),
        "empty const table must be E2019; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn const_and_enum_same_base_collides() {
    // An enum and a const that normalise to the same base both claim
    // `<base>_count`; the second is an E2021 collision, not a silent drop.
    let src = "#[bimap]\nenum Country { US, JP }\n#[bimap]\nconst COUNTRY = [(\"US\", \"a\")];\n";
    assert!(
        reject_codes(src).contains(&"E2021".to_string()),
        "enum+const same-base must be E2021; got: {:?}",
        reject_codes(src)
    );
}

#[test]
fn non_bimap_const_is_untouched() {
    // A const WITHOUT `#[bimap]` is not a table — it is left exactly as authored
    // and no fns are synthesised.
    let src = "const C = [(\"US\", \"a\")];\n";
    let m = parse(src).expect("parse");
    assert!(
        m.items
            .iter()
            .any(|it| matches!(it, Node::Const { name, .. } if name == "C")),
        "non-#[bimap] const must be preserved"
    );
    assert!(!has_fn(&m, "c_value"), "no derive on a plain const");
}

#[cfg(feature = "std-surface")]
#[test]
fn const_form_inverse_reuses_the_phf_ladder() {
    // The const-form inverses reuse the enum-form `make_from_str_fn` ladder, so
    // they stamp one of the two real inverse strategies — never a bespoke path.
    // string<->string: both hidden index fns hash over their column's byte keys
    // (ordinals are the dense 0..n-1 row indices, in the PHF one-byte envelope),
    // so a small table lands on the O(1) perfect hash.
    let ss = parse(SS_TABLE).expect("parse ss");
    assert_eq!(
        from_str_mode(&ss, "__bimap_country_key_idx").as_deref(),
        Some("phf-v1"),
        "string<->string key-index inverse must reuse phf-v1"
    );
    assert_eq!(
        from_str_mode(&ss, "__bimap_country_value_idx").as_deref(),
        Some("phf-v1"),
        "string<->string value-index inverse must reuse phf-v1"
    );

    // number<->string: the inverse's ordinals ARE the authored number keys.
    // Keys 200/404 exceed the PHF single-byte ordinal envelope (ORD_EMPTY=255),
    // so construction correctly falls to the O(log n) binary search whose
    // 4-byte little-endian ordinal field carries the wide keys — a deterministic
    // pure function of the table, proven by the example's inverse returning 500.
    let ns = parse(NS_TABLE).expect("parse ns");
    assert_eq!(
        from_str_mode(&ns, "http_status_from_str").as_deref(),
        Some("binsearch-v1"),
        "wide-ordinal number<->string inverse must fall to binsearch-v1"
    );

    // A number<->string table whose keys fit the one-byte ordinal envelope
    // lands on the O(1) perfect hash, same ladder.
    let small = parse("#[bimap]\nconst K = [(1, \"a\"), (2, \"bb\"), (3, \"ccc\")];\n")
        .expect("parse small ns");
    assert_eq!(
        from_str_mode(&small, "k_from_str").as_deref(),
        Some("phf-v1"),
        "in-envelope number<->string inverse must reuse phf-v1"
    );
}
