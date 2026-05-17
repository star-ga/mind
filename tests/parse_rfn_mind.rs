//! Parse-target tests for rfn-mind syntax acceptance (Phase 10.5).
//!
//! Each test asserts that public `mindc` parses a rfn-mind-style construct
//! without error. Tests are added in dependency order; failing tests at the
//! end of the file are RED markers for parser additions still in flight.
//!
//! Reference grammar: `/home/n/mind-spec/spec/v1.0/grammar-syntax.ebnf`.
//! Reference rfn-mind file: `/home/n/rfn-mind/src/fixed_point.mind`.

use libmind::{compile_source, CompileOptions};

fn parses(src: &str) -> bool {
    compile_source(src, &CompileOptions::default()).is_ok()
}

// ──────────────────────────────────────────────────────────────────────
//  Sequencing per architect review:
//    1. use dispatch wiring  (parse_use exists; just unreachable)
//    2. const NAME: T = expr
//    3. type X = Y  (+ TypeAnn::Named)
//    4. module {} unwrap
//    5. [attribute] pre-pass
//    6. export {}
//    7. struct
//    8. enum
//    9. fixed_point.mind end-to-end
// ──────────────────────────────────────────────────────────────────────

// Step 1 — `use` dispatch
#[test]
fn parses_use_directive() {
    assert!(parses("use math.utils;\n"), "use directive must parse");
    assert!(parses("use a.b.c;\n"), "multi-segment use path must parse");
}

// Step 2 — const declarations
#[test]
fn parses_const_decl() {
    assert!(
        parses("const X: i32 = 5;\n"),
        "const NAME: type = value must parse"
    );
}

#[test]
fn parses_const_no_semicolon() {
    // rfn-mind style: no trailing semicolon at module scope
    assert!(
        parses("const Q16_ONE: i32 = 65536\n"),
        "const without trailing ; must parse at module scope"
    );
}

// Step 3 — type aliases
#[test]
fn parses_type_alias() {
    assert!(parses("type Q16_16 = i32\n"), "type alias must parse");
}

#[test]
fn parses_type_alias_then_const() {
    let src = "type Q16_16 = i32\nconst Q16_ONE: i32 = 65536\n";
    assert!(parses(src), "type alias + const using primitive must parse");
}

// Step 4 — module unwrap
#[test]
fn parses_module_unwrap_empty() {
    assert!(
        parses("module foo {}\n"),
        "empty module block must parse and unwrap"
    );
}

#[test]
fn parses_module_unwrap_with_const() {
    assert!(
        parses("module fixed_point { const X: i32 = 1 }\n"),
        "module with single const must unwrap to flat module"
    );
}

#[test]
fn parses_module_unwrap_with_fn() {
    let src = "module fp {\n  fn add(a: i32, b: i32) -> i32 { a + b }\n}\n";
    assert!(parses(src), "module with fn must parse");
}

// Step 5 — [attribute] pre-pass
#[test]
fn parses_attribute_on_module() {
    assert!(
        parses("[protection]\nmodule foo {}\n"),
        "[protection] on module must parse"
    );
}

#[test]
fn parses_attribute_on_fn() {
    assert!(
        parses("[test]\nfn t() -> i32 { 1 }\n"),
        "[test] on fn must parse"
    );
}

// Step 6 — export blocks
#[test]
fn parses_export_block() {
    assert!(parses("export { foo, bar }\n"), "export block must parse");
}

// Step 7 — struct
#[test]
fn parses_struct_decl() {
    let src = "struct Point { x: i32, y: i32 }\n";
    assert!(parses(src), "struct with two fields must parse");
}

// Step 8 — enum
#[test]
fn parses_enum_decl() {
    let src = "enum AddressingMode { Positional, Content }\n";
    assert!(parses(src), "enum with unit variants must parse");
}

// Step 8b — qualified type paths in type annotations (Phase 10.6 / RFC 0003).
// Required for every rfn-mind source file that uses `use fixed_point`
// followed by `fixed_point.Q16_16` in const, fn signatures, struct fields,
// type aliases, and elsewhere.
#[test]
fn parses_qualified_type_in_const() {
    let src = "module foo { type Q = i32 }\n\
               module bar { use foo\n const X: foo.Q = 1 }\n";
    assert!(parses(src), "qualified type `foo.Q` in const must parse");
}

#[test]
fn parses_qualified_type_in_fn_signature() {
    let src = "module foo { type Q = i32 }\n\
               module bar { use foo\n fn f(x: foo.Q) -> foo.Q { x } }\n";
    assert!(
        parses(src),
        "qualified type in fn param + return must parse"
    );
}

#[test]
fn parses_multi_segment_qualified_type() {
    // Forward-compat: nested module paths `a.b.C` must accumulate cleanly.
    let src = "module bar { fn f(x: a.b.C) -> a.b.C { x } }\n";
    assert!(parses(src), "multi-segment qualified type must parse");
}

// Step 9 — full rfn-mind file end-to-end (Tier-1 milestone)
#[test]
fn parses_fixed_point_mind_end_to_end() {
    let path = "/home/n/rfn-mind/src/fixed_point.mind";
    let Ok(src) = std::fs::read_to_string(path) else {
        eprintln!("{} not present; skipping milestone gate", path);
        return;
    };
    assert!(parses(&src), "fixed_point.mind must parse end-to-end");
}

// Step 10 — full rfn-mind src/ sweep
#[test]
#[ignore = "Tier-2 milestone — enable after struct + enum land"]
fn parses_all_rfn_mind_src() {
    let dir = std::path::Path::new("/home/n/rfn-mind/src");
    if !dir.exists() {
        eprintln!("/home/n/rfn-mind/src not present; skipping sweep");
        return;
    }
    let mut failed = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("mind") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read .mind");
        if !parses(&src) {
            failed.push(path.display().to_string());
        }
    }
    assert!(
        failed.is_empty(),
        "{} rfn-mind file(s) failed to parse: {:#?}",
        failed.len(),
        failed
    );
}
