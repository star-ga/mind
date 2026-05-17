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

// Step 8c — `pub` visibility marker (Phase 10.6). mindc treats `pub` as a
// no-op (module-level visibility is via `export` blocks); accepting it
// keeps rfn-mind/src/ternary.mind and src/bitlinear.mind syntax-compatible
// without forcing a source rewrite.
#[test]
fn parses_pub_struct_decl() {
    let src = "module m { pub struct Pair { a: i32, b: i32 } }\n";
    assert!(parses(src), "`pub struct` at module scope must parse");
}

#[test]
fn parses_pub_enum_decl() {
    let src = "module m { pub enum Side { Left, Right } }\n";
    assert!(parses(src), "`pub enum` at module scope must parse");
}

#[test]
fn parses_pub_fn_def() {
    let src = "module m { pub fn id(x: i32) -> i32 { x } }\n";
    assert!(parses(src), "`pub fn` at module scope must parse");
}

#[test]
fn parses_pub_field_in_struct() {
    let src = "module m { struct Pair { pub a: i32, pub b: i32 } }\n";
    assert!(parses(src), "`pub field` inside struct body must parse");
}

#[test]
fn parses_mixed_pub_fields() {
    // Mix of pub and non-pub fields must coexist cleanly.
    let src = "module m { struct R { pub x: i32, y: i32, pub z: u32 } }\n";
    assert!(parses(src), "mixed pub/non-pub fields must parse");
}

// Step 8d — struct literal expressions (Phase 10.6). Required by
// rfn-mind/src/fixed_point_bwd.mind (returns PartialPair { da, db }) and
// rfn-mind/src/field_step.mind (constructs StepBuffers, etc).
#[test]
fn parses_struct_literal_in_return() {
    let src = "module m {\n\
                 struct Pair { a: i32, b: i32 }\n\
                 fn make() -> Pair { return Pair { a: 1, b: 2 } }\n\
               }\n";
    assert!(parses(src), "struct literal in return must parse");
}

#[test]
fn parses_struct_literal_in_let() {
    let src = "module m {\n\
                 struct Pair { a: i32, b: i32 }\n\
                 fn f() -> i32 { let p = Pair { a: 10, b: 20 }\n p.a + p.b }\n\
               }\n";
    assert!(parses(src), "struct literal in let must parse");
}

#[test]
fn parses_empty_struct_literal() {
    let src = "module m {\n\
                 struct Unit {}\n\
                 fn f() -> Unit { return Unit { } }\n\
               }\n";
    assert!(parses(src), "empty struct literal must parse");
}

#[test]
fn parses_struct_literal_with_qualified_field_type() {
    // The combined case rfn-mind uses everywhere:
    // qualified types in the field types and struct literal in the body.
    let src = "module foo { type Q = i32 }\n\
               module bar {\n\
                 use foo\n\
                 struct Pair { a: foo.Q, b: foo.Q }\n\
                 fn mk(x: foo.Q, y: foo.Q) -> Pair { return Pair { a: x, b: y } }\n\
               }\n";
    assert!(
        parses(src),
        "qualified field types + struct literal must parse together"
    );
}

// Step 8e — slice and array types (Phase 10.6). Required by rfn-mind/src/
// reduce.mind (&[T]), groupnorm.mind (&mut [T]), and lut.mind ([T; N]).
#[test]
fn parses_slice_type_in_param() {
    let src = "module m { fn s(xs: &[i32]) -> i32 { 0 } }\n";
    assert!(parses(src), "&[T] in fn param must parse");
}

#[test]
fn parses_mut_slice_type_in_param() {
    let src = "module m { fn s(xs: &mut [i32]) -> i32 { 0 } }\n";
    assert!(parses(src), "&mut [T] in fn param must parse");
}

#[test]
fn parses_array_type_in_fn_param() {
    let src = "module m { fn lut(t: [i32; 256]) -> i32 { 0 } }\n";
    assert!(parses(src), "[T; N] array type in fn param must parse");
}

#[test]
fn parses_slice_of_qualified_type() {
    let src = "module foo { type Q = i32 }\n\
               module bar { use foo\n fn s(xs: &[foo.Q]) -> foo.Q { 0 } }\n";
    assert!(parses(src), "&[module.Type] must parse");
}

// Step 8f — `let mut` mutable binding (Phase 10.6). Used for accumulator
// loops in rfn-mind/src/reduce.mind, conv.mind, groupnorm.mind. mindc
// treats `mut` as informational; the eval env always allows reassignment.
#[test]
fn parses_let_mut_binding() {
    let src = "module m { fn f() -> i32 { let mut x: i32 = 0\n x = x + 1\n x } }\n";
    assert!(parses(src), "`let mut x = ...` must parse");
}

#[test]
fn parses_let_mut_without_annotation() {
    let src = "module m { fn f() -> i32 { let mut x = 0\n x = x + 1\n x } }\n";
    assert!(parses(src), "`let mut x = ...` without type ann must parse");
}

// Step 8g — modulo `%` (Phase 10.6). rfn-mind/src/groupnorm.mind uses
// `c_count % num_groups` to validate channel-group divisibility; memory.mind
// uses it for wrap-around address arithmetic.
#[test]
fn parses_modulo_binop() {
    let src = "module m { fn f() -> i32 { 10 % 3 } }\n";
    assert!(parses(src), "`%` modulo must parse");
}

#[test]
fn parses_modulo_in_assert() {
    let src = "module m { fn f(c: u32, g: u32) -> i32 { assert c % g == 0, \"divisible\"\n 0 } }\n";
    assert!(
        parses(src),
        "`%` inside assert condition must parse"
    );
}

// Step 8h — single-value `&T` / `&mut T` and generic `Name<A, B>` types
// (Phase 10.6).
#[test]
fn parses_single_value_reference_type() {
    let src = "module m { struct Bank { n: u32 } fn f(b: &Bank) -> u32 { 0 } }\n";
    assert!(parses(src), "`&Type` single-value reference must parse");
}

#[test]
fn parses_mut_single_value_reference() {
    let src = "module m { struct Bank { n: u32 } fn f(b: &mut Bank) { } }\n";
    assert!(parses(src), "`&mut Type` must parse");
}

#[test]
fn parses_generic_one_arg() {
    let src = "module m { struct S { xs: Vec<i32> } }\n";
    assert!(parses(src), "`Vec<i32>` in struct field must parse");
}

#[test]
fn parses_generic_two_args() {
    let src = "module m { fn f() -> Result<i32, u32> { 0 as i32 } }\n";
    assert!(parses(src), "`Result<i32, u32>` in fn return must parse");
}

#[test]
fn parses_nested_generic_with_qualified_arg() {
    let src = "module foo { type Q = i32 }\n\
               module m { use foo\n struct S { xs: Vec<foo.Q> } }\n";
    assert!(
        parses(src),
        "`Vec<module.Type>` (nested generic + qualified arg) must parse"
    );
}

#[test]
fn does_not_parse_struct_literal_for_arbitrary_block() {
    // Lookahead must reject `IDENT { stmt }` shape (no `field: value` pair).
    // Even though this is invalid at expression position regardless, we
    // want the parser to NOT misparse it as a struct literal first.
    // Here `Pair { let x = 1 }` is not a valid struct literal because
    // the body doesn't start with `field:`.
    let src = "module m {\n\
                 struct Pair { a: i32 }\n\
                 fn f() -> i32 { let p = Pair { let x = 1 }\n 0 }\n\
               }\n";
    assert!(
        !parses(src),
        "non-`field:value` body must not be accepted as struct literal"
    );
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
