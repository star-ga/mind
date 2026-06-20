//! Parse-target tests for Phase 10.5 / 10.6 surface-syntax acceptance.
//!
//! Each test asserts that public `mindc` parses a real-world MIND
//! construct without error. Tests are added in dependency order; failing
//! tests at the end of the file are RED markers for parser additions
//! still in flight.
//!
//! Reference grammar: `mind-spec/spec/v1.0/grammar-syntax.ebnf`.
//!
//! The corpus-sweep test (`parses_tracking_corpus_watermark`) is driven
//! by the `MIND_TRACKING_CORPUS_DIR` environment variable. When unset
//! the sweep is a no-op so CI and fresh clones stay green; on
//! development machines where the variable points at a directory of
//! `.mind` files, the sweep asserts the documented high-watermark.

use libmind::{CompileOptions, compile_source, parser};

// This file is a PARSER-acceptance suite (see the module docstring): each test
// asserts a Phase-10 surface construct *parses*. It deliberately stops at the
// parser — using a full `compile_source` here conflated parser coverage with
// compile-to-MIC lowering coverage, so constructs that parse fine but are not
// lowered yet (TypeAlias, qualified types, …) produced false RED markers. IR
// lowering is gated by `fmt_ir_preservation` instead.
fn parses(src: &str) -> bool {
    parser::parse(src).is_ok()
}

// A handful of tests assert COMPILE-level rejection (a semantic/lowering
// guard), which is distinct from a parse rejection — those run the full
// pipeline. Example: `-> (T, U)` parses fine but the compiler rejects it
// (no aggregate-return ABI).
fn compiles(src: &str) -> bool {
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
    // No trailing semicolon at module scope is allowed.
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
        parses("#[protection]\nmodule foo {}\n"),
        "#[protection] on module must parse"
    );
}

#[test]
fn parses_attribute_on_fn() {
    assert!(
        parses("#[test]\nfn t() -> i32 { 1 }\n"),
        "#[test] on fn must parse"
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
// Required wherever `use <module>` is followed by `<module>.Type` in
// const, fn signatures, struct fields, type aliases, and elsewhere.
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
// keeps pub-prefixed source compatible without forcing a rewrite.
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

// Step 8d — struct literal expressions (Phase 10.6). Required for fns
// that return aggregate values such as `Pair { a, b }` directly.
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
    // Combined case: qualified types in the field declarations and
    // struct literal in the body.
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

// Step 8e — slice and array types (Phase 10.6). Used wherever fn
// signatures pass contiguous buffers (`&[T]`, `&mut [T]`) or compile-
// time LUT tables (`[T; N]`).
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

// Step 8e' — bare `[T]` dynamic slice (no `; N`). Distinct from the
// borrowed `&[T]` and from the fixed-size `[T; N]`: the length is not
// part of the type. Reuses the `Slice` representation so the type
// checker treats it as a contiguous run of `T`. (mindc parser gap that
// blocked ~half of the 512-mind modules from parsing.)
#[test]
fn parses_bare_dynamic_slice_in_param() {
    let src = "module m { fn s(xs: [i32]) -> i32 { 0 } }\n";
    assert!(
        parses(src),
        "bare `[T]` dynamic slice in fn param must parse"
    );
}

#[test]
fn parses_bare_dynamic_slice_in_struct_field() {
    let src = "module m { struct Holder { items: [u32] } }\n";
    assert!(
        parses(src),
        "bare `[T]` dynamic slice in struct field must parse"
    );
}

#[test]
fn parses_bare_dynamic_slice_of_qualified_type() {
    let src = "module foo { type Q = i32 }\n\
               module bar { use foo\n struct S { xs: [foo.Q] } }\n";
    assert!(parses(src), "bare `[module.Type]` dynamic slice must parse");
}

#[test]
fn parses_fixed_array_still_distinct_from_slice() {
    // `[T; N]` must keep parsing after the dynamic-slice path was added.
    let src = "module m { fn lut(t: [i32; 8]) -> i32 { 0 } }\n";
    assert!(parses(src), "`[T; N]` array must still parse");
}

#[test]
fn rejects_array_type_missing_length() {
    // `[T; ]` (a `;` with no length) is still a hard parse error — the
    // dynamic-slice path only fires when there is no `;` at all.
    let src = "module m { struct B { x: [u32; ] } }\n";
    assert!(!parses(src), "`[T; ]` with empty length must not parse");
}

#[test]
fn parses_nested_dynamic_slice() {
    // `[[T]]` — element type recurses, so a slice-of-slices parses.
    let src = "module m { struct S { grid: [[u32]] } }\n";
    assert!(parses(src), "nested `[[T]]` dynamic slice must parse");
}

#[test]
fn parses_dynamic_slice_as_generic_arg() {
    // `Map<K, [V]>` — the bare slice appears as a generic type argument.
    let src = "module m { struct S { m: Map<u32, [u256]> } }\n";
    assert!(parses(src), "bare `[T]` as a generic arg must parse");
}

// Step 8f — `let mut` mutable binding (Phase 10.6). Used for
// accumulator loops. mindc treats `mut` as informational; the eval
// env always allows reassignment.
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

// Step 8g — modulo `%` (Phase 10.6). Used for divisibility checks
// (channel / group counts) and for wrap-around address arithmetic.
#[test]
fn parses_modulo_binop() {
    let src = "module m { fn f() -> i32 { 10 % 3 } }\n";
    assert!(parses(src), "`%` modulo must parse");
}

#[test]
fn parses_modulo_in_assert() {
    let src = "module m { fn f(c: u32, g: u32) -> i32 { assert c % g == 0, \"divisible\"\n 0 } }\n";
    assert!(parses(src), "`%` inside assert condition must parse");
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

// Step 8i — `::` path-segment separator for enum variants (Phase 10.6).
// Examples: `config.AddressingMode::Content`, `Side::Left`.
#[test]
fn parses_double_colon_enum_variant() {
    let src = "module m { enum Side { Left, Right }\n fn f() -> i32 { let x = Side::Left\n 0 } }\n";
    assert!(parses(src), "`Side::Left` enum variant access must parse");
}

#[test]
fn parses_qualified_double_colon_path() {
    let src = "module config { enum Mode { On, Off } }\n\
               module m { use config\n fn f() -> i32 { let x = config.Mode::On\n 0 } }\n";
    assert!(parses(src), "`module.Enum::Variant` must parse");
}

// Step 8j — postfix indexing + index-LHS + field-LHS assignment (Phase 10.6).
#[test]
fn parses_index_access_postfix() {
    let src = "module m { fn f(xs: &[i32]) -> i32 { xs[0] } }\n";
    assert!(parses(src), "`xs[0]` postfix index access must parse");
}

#[test]
fn parses_index_assign_lhs() {
    let src = "module m { fn f(xs: &mut [i32]) { xs[0] = 1 } }\n";
    assert!(parses(src), "`xs[0] = 1` indexed assignment must parse");
}

#[test]
fn parses_field_assign_lhs() {
    let src = "module m { struct S { x: i32 } fn f(s: &mut S) { s.x = 1 } }\n";
    assert!(parses(src), "`s.x = 1` field assignment must parse");
}

// Step 8k — multi-line arithmetic continuation (Phase 10.6). Index
// math that spans newlines uses `+` and `*` on continuation lines; the
// Pratt parser must skip newlines when peeking for an infix operator.
#[test]
fn parses_multiline_arithmetic() {
    let src = "module m { fn f(c: u32, h: u32, w: u32, x: u32, y: u32) -> u32 {\n\
                 let idx: u32 = (c as u32) * (h * w) as u32\n\
                              + (y as u32) * (w as u32)\n\
                              + (x as u32)\n\
                 idx\n\
               } }\n";
    assert!(parses(src), "multi-line `+` continuation must parse");
}

// Step 8l — tuple return types `-> (T, U)` (Phase 10.6 + real tuple aggregates).
// Once tuples became real heap aggregates, an ALL-I64 tuple return is legal: the
// callee returns the `__mind_alloc` base pointer (i64) — exactly like a struct
// return — and the caller takes it apart with `let (a, b) = …`. A FLOAT element
// still cannot ride the all-i64 ABI (its bits would be reinterpreted), so a
// float-bearing tuple return stays rejected fail-loud
// (safety::tuple_return_unsupported), accepted only under
// `--features std-surface-experimental`.
#[test]
fn all_i64_tuple_return_is_legal() {
    let src = "module m { fn pair() -> (i32, u32) { (0, 0) } }\n";
    assert!(
        compiles(src),
        "`-> (i32, u32)` all-i64 tuple return must be accepted (heap aggregate, base-pointer ABI)"
    );
}

#[test]
fn float_tuple_return_fails_loud() {
    let src = "module m { fn pair() -> (f64, i32) { (0.0, 0) } }\n";
    assert!(
        !compiles(src),
        "`-> (f64, i32)` float-bearing tuple return must be rejected fail-loud"
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

// Step 9 — corpus-driven watermark sweep. Acts as a regression gate
// against an external tracking corpus of `.mind` files that exercise
// every Phase-10.5/10.6 surface. The corpus directory is opt-in via
// the `MIND_TRACKING_CORPUS_DIR` environment variable; when unset, the
// sweep is a no-op so CI and fresh clones stay green. When set, the
// parser must hold the documented high-watermark count: any drop is a
// regression in the parser, never a "spec gap" to ignore.
//
// Current watermark: 21 files parse cleanly. Bumped in the commit
// that lands match expressions (Phase 10.7) and `&expr` / `&mut expr`
// reference-taking expressions (Phase 10.7), unblocking 7 previously
// failing corpus files (laplacian, memory, bundle for match;
// field_step, memory_bwd, readout, rfn for &expr).
const TRACKING_CORPUS_WATERMARK: usize = 21;

#[test]
fn parses_tracking_corpus_watermark() {
    let Some(dir_str) = std::env::var_os("MIND_TRACKING_CORPUS_DIR") else {
        eprintln!("MIND_TRACKING_CORPUS_DIR not set; skipping sweep");
        return;
    };
    let dir = std::path::PathBuf::from(&dir_str);
    if !dir.exists() {
        eprintln!("{} not present; skipping sweep", dir.display());
        return;
    }
    let mut passed = 0usize;
    let mut failed = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("mind") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read .mind");
        if parses(&src) {
            passed += 1;
        } else {
            failed.push(path.display().to_string());
        }
    }
    assert!(
        passed >= TRACKING_CORPUS_WATERMARK,
        "tracking-corpus parse watermark regression: {} files parsed, expected >= {}. \
         Failing files (informational, gated on pending language features):\n{:#?}",
        passed,
        TRACKING_CORPUS_WATERMARK,
        failed
    );
}
