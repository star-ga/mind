// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Strings that reached an integer slot without a recorded store, string
//! literal `match` patterns, and string length / slice / trim.
//!
//! A review measured these shapes on the MLIR build and under `mindc test`:
//!
//! * A string reached an `i64` slot through a value whose type the equality
//!   pass did not establish (a tuple destructured from a tuple result, a value
//!   block that reassigns a local, a local assigned in a loop), through raw
//!   reads of a std collection's storage, or through another module of a
//!   project. `mindc check` accepted `==` on two such values and the MLIR build
//!   returned 0 for equal text. Each is now refused with `E2038` by `mindc
//!   check`, `mindc build` and `mindc test` (the project shapes by `mindc
//!   check` of the importing module and by `mindc build`).
//! * `match s { "abc" => 1, _ => 0 }` returned 1 on the MLIR build and 0
//!   under `mindc test`, and on an integer scrutinee it crashed the MLIR build.
//!   Both backends now compare bytes; a scrutinee not known to be a string is
//!   refused at check time.
//! * `.len()`, `.slice(a, b)` and `.trim()` on a string computed a value on the
//!   MLIR build and failed under `mindc test`; `.length` passed `mindc check`
//!   and was refused only by the build. The methods now give the same value
//!   on both backends, and `.length` is refused at check time.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test string_flows_backend_parity`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::{mindc_bin, scratch_dir};

use std::path::Path;
use std::process::{Command, Output};

const EQH: &str =
    "fn eqh(a: i64, b: i64) -> i64 {\n    if a == b { return 1 }\n    return 0\n}\n\n";

/// A value block whose result is a local it reassigns.
const BLOCK: &str = "if true { let mut t = \"x\"; t = \"abc\"; t } else { let mut u = \"y\"; \
                     u = \"abc\"; u }";

/// `(case, source, code)`: each source compares two strings held in integer
/// slots (or reads a member no string has) in `probe()`.
fn refused_cases() -> Vec<(&'static str, String, &'static str)> {
    vec![
        (
            "tuple_destructured",
            format!(
                "{EQH}fn pair() -> (i64, i64) {{ return (\"abc\", \"abc\") }}\n\n\
                 pub fn probe() -> i64 {{\n    let (a, b) = pair()\n    return eqh(a, b)\n}}\n"
            ),
            "E2038",
        ),
        (
            "block_value_into_param",
            format!(
                "{EQH}pub fn probe() -> i64 {{\n    let a = {BLOCK}\n    let b = {BLOCK}\n    \
                 return eqh(a, b)\n}}\n"
            ),
            "E2038",
        ),
        (
            "block_value_returned",
            format!(
                "fn mk() -> i64 {{\n    let a = {BLOCK}\n    return a\n}}\n\n\
                 pub fn probe() -> i64 {{\n    if mk() == mk() {{ return 1 }}\n    return 0\n}}\n"
            ),
            "E2038",
        ),
        (
            "block_value_in_field",
            format!(
                "struct Box {{ v: i64 }}\n\npub fn probe() -> i64 {{\n    \
                 let x = Box {{ v: {BLOCK} }}\n    let y = Box {{ v: {BLOCK} }}\n    \
                 if x.v == y.v {{ return 1 }}\n    return 0\n}}\n"
            ),
            "E2038",
        ),
        (
            "loop_local_into_param",
            format!(
                "{EQH}pub fn probe() -> i64 {{\n    let mut a = 0\n    let mut b = 0\n    \
                 let mut i = 0\n    while i < 1 {{\n        let s = \"abc\"\n        a = s\n        \
                 b = s\n        i = i + 1\n    }}\n    return eqh(a, b)\n}}\n"
            ),
            "E2038",
        ),
        (
            "block_value_pushed",
            format!(
                "use std.vec;\n\npub fn probe() -> i64 {{\n    let a = {BLOCK}\n    \
                 let b = {BLOCK}\n    let v = vec_push(vec_push(vec_new(), a), b)\n    \
                 if vec_get(v, 0) == vec_get(v, 1) {{ return 1 }}\n    return 0\n}}\n"
            ),
            "E2038",
        ),
        (
            "vec_storage_read_raw",
            "use std.vec;\n\npub fn probe() -> i64 {\n    \
             let v = vec_push(vec_push(vec_new(), \"abc\"), \"abc\")\n    let p = vec_addr(v)\n    \
             if __mind_load_i64(p) == __mind_load_i64(p + 8) { return 1 }\n    return 0\n}\n"
                .to_string(),
            "E2038",
        ),
        (
            "vec_set_storage_read_raw",
            "use std.vec;\n\npub fn probe() -> i64 {\n    \
             let v = vec_push(vec_push(vec_new(), 0), 0)\n    vec_set(v, 0, \"abc\")\n    \
             vec_set(v, 1, \"abc\")\n    let p = vec_addr(v)\n    \
             if __mind_load_i64(p) == __mind_load_i64(p + 8) { return 1 }\n    return 0\n}\n"
                .to_string(),
            "E2038",
        ),
        (
            "map_storage_read_raw",
            "use std.map;\n\npub fn probe() -> i64 {\n    \
             let m = map_insert(map_insert(map_new(), 1, \"abc\"), 2, \"abc\")\n    \
             let p = map_vals_addr(m)\n    \
             if __mind_load_i64(p) == __mind_load_i64(p + 8) { return 1 }\n    return 0\n}\n"
                .to_string(),
            "E2038",
        ),
        (
            "raw_store_read_by_vec_get",
            "use std.vec;\n\npub fn probe() -> i64 {\n    \
             let v = vec_push(vec_push(vec_new(), 0), 0)\n    let p = vec_addr(v)\n    \
             __mind_store_i64(p, \"abc\")\n    __mind_store_i64(p + 8, \"abc\")\n    \
             if vec_get(v, 0) == vec_get(v, 1) { return 1 }\n    return 0\n}\n"
                .to_string(),
            "E2038",
        ),
        (
            "literal_pattern_on_an_integer",
            "fn is_abc(s: i64) -> i64 {\n    match s {\n        \"abc\" => 1,\n        \
             _ => 0,\n    }\n}\n\npub fn probe() -> i64 { return is_abc(5) }\n"
                .to_string(),
            "E2038",
        ),
        (
            "length_member",
            "fn slen(s: string) -> i64 { return s.length }\n\n\
             pub fn probe() -> i64 { return slen(\"abc\") }\n"
                .to_string(),
            "E6009",
        ),
    ]
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn mindc(args: &[&str]) -> Output {
    Command::new(mindc_bin())
        .args(args)
        .output()
        .expect("spawn mindc")
}

#[test]
fn strings_reaching_integer_slots_are_refused_by_check_build_and_test() {
    let dir = scratch_dir("string_flows_backend_parity_refused");
    for (case, src, code) in refused_cases() {
        let path = dir.join(format!("{case}.mind"));
        std::fs::write(&path, &src).expect("write case source");
        let check = mindc(&["check", "--no-fmt", path.to_str().unwrap()]);
        let check_text = text(&check);
        assert!(
            !check.status.success(),
            "{case}: check accepted:\n{check_text}"
        );
        assert_eq!(
            check_text.matches(code).count(),
            1,
            "{case}: one {code} for the one refused form:\n{check_text}"
        );
        assert!(!check_text.contains("panicked"), "{case}:\n{check_text}");

        let so = dir.join(format!("{case}.so"));
        let build = mindc(&[
            path.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ]);
        let build_text = text(&build);
        assert!(
            !build.status.success(),
            "{case}: build accepted:\n{build_text}"
        );
        assert!(build_text.contains(code), "{case}:\n{build_text}");
        assert!(!build_text.contains("panicked"), "{case}:\n{build_text}");
        assert!(!so.exists(), "{case}: a refused build left an artifact");

        let test_path = dir.join(format!("{case}_test.mind"));
        let test_src =
            format!("{src}\n#[test]\nfn probed() {{ assert probe() == 1, \"value\" }}\n");
        std::fs::write(&test_path, test_src).expect("write case test source");
        let test = mindc(&["test", test_path.to_str().unwrap()]);
        let test_text = text(&test);
        assert!(
            !test.status.success(),
            "{case}: mindc test passed:\n{test_text}"
        );
        assert!(
            test_text.contains(&format!("[{code}]")),
            "{case}:\n{test_text}"
        );
    }
}

/// `(case, provider, main)`: a two-module project whose `main` stores strings
/// into `provider`'s integer slots, or reads them from its integer result.
const PROJECT_CASES: &[(&str, &str, &str)] = &[
    (
        "param",
        "export { eqh }\n\nfn eqh(a: i64, b: i64) -> i64 {\n    if a == b { return 1 }\n    \
         return 0\n}\n",
        "import provider;\n\npub fn probe() -> i64 { return provider.eqh(\"abc\", \"abc\") }\n",
    ),
    (
        "result",
        "export { mk }\n\nfn mk() -> i64 { return \"abc\" }\n",
        "import provider;\n\npub fn probe() -> i64 {\n    \
         if provider.mk() == provider.mk() { return 1 }\n    return 0\n}\n",
    ),
    // A collection of a type the pass does not model (`Vec<i64>`) given
    // strings, then moved into the other module's typed slot: a parameter,
    // a result, a field; or a typed collection the other module stores
    // strings into through such an alias. Only the last reaches `main`
    // through an element fact, across a typed parameter.
    (
        "unknown_collection_param",
        "export { same }\n\nfn same(xs: array<i64>) -> i64 {\n    \
         if xs[0] == xs[1] { return 1 }\n    return 0\n}\n",
        "import provider;\n\npub fn probe(n: i64) -> i64 {\n    \
         let es: Vec<i64> = vec_zeroed(n);\n    es[0] = \"abc\";\n    es[1] = \"abc\";\n    \
         return provider.same(es)\n}\n",
    ),
    (
        "unknown_collection_result",
        "export { strings }\n\nfn strings(n: i64) -> array<i64> {\n    \
         let es: Vec<i64> = vec_zeroed(n);\n    es[0] = \"abc\";\n    es[1] = \"abc\";\n    \
         return es\n}\n",
        "import provider;\n\npub fn probe() -> i64 {\n    \
         let xs: array<i64> = provider.strings(2);\n    \
         if xs[0] == xs[1] { return 1 }\n    return 0\n}\n",
    ),
    (
        "unknown_collection_field",
        "export { hold, Holder }\n\nstruct Holder {\n    items: array<i64>,\n}\n\n\
         fn hold(n: i64) -> Holder {\n    let es: Vec<i64> = vec_zeroed(n);\n    \
         es[0] = \"abc\";\n    es[1] = \"abc\";\n    return Holder { items: es }\n}\n",
        "import provider;\n\npub fn probe() -> i64 {\n    let h: Holder = provider.hold(2);\n    \
         let xs: array<i64> = h.items;\n    if xs[0] == xs[1] { return 1 }\n    return 0\n}\n",
    ),
    (
        "unknown_alias_of_a_typed_param",
        "export { poison }\n\nfn poison(xs: array<i64>) -> i64 {\n    \
         let ys: Vec<i64> = xs;\n    ys[0] = \"abc\";\n    ys[1] = \"abc\";\n    return 0\n}\n",
        "import provider;\n\npub fn probe() -> i64 {\n    let xs: array<i64> = [1, 2];\n    \
         provider.poison(xs);\n    if xs[0] == xs[1] { return 1 }\n    return 0\n}\n",
    ),
];

const MANIFEST: &str = "[package]\nname = \"xmod\"\nversion = \"0.1.0\"\n\n[build]\n\
                        entry = \"main.mind\"\nemit = \"cdylib\"\n\n[targets.cpu]\n\
                        backend = \"cpu\"\nsources = [\"main.mind\", \"provider.mind\"]\n";

#[cfg(feature = "cross-module-imports")]
#[test]
fn strings_stored_across_project_modules_are_refused_by_check_and_build() {
    for (case, provider, main) in PROJECT_CASES {
        let dir = scratch_dir(&format!("string_flows_backend_parity_project_{case}"));
        std::fs::write(dir.join("Mind.toml"), MANIFEST).expect("write manifest");
        std::fs::write(dir.join("provider.mind"), provider).expect("write provider");
        let main_path = dir.join("main.mind");
        std::fs::write(&main_path, main).expect("write main");
        let check = mindc(&["check", "--no-fmt", main_path.to_str().unwrap()]);
        let check_text = text(&check);
        assert!(
            !check.status.success(),
            "{case}: check of the importing module accepted:\n{check_text}"
        );
        assert!(check_text.contains("E2038"), "{case}:\n{check_text}");
        let so = dir.join("x.so");
        let build = Command::new(mindc_bin())
            .args([
                "build",
                "--backend",
                "mlir",
                "--emit",
                "cdylib",
                "--out",
                so.to_str().unwrap(),
            ])
            .current_dir(&dir)
            .output()
            .expect("spawn mindc build");
        let build_text = text(&build);
        assert!(
            !build.status.success(),
            "{case}: project build accepted:\n{build_text}"
        );
        assert!(build_text.contains("E2038"), "{case}:\n{build_text}");
        assert!(!so.exists(), "{case}: a refused build left an artifact");
    }
}

/// `mindc test` decides a comparison with the declarations `mindc check`
/// reads: `mk` is the one `t` imports (`alpha.mk -> i64`), not the other
/// project function of that name (`zeta.mk -> string`). Without the
/// project's module table, `mindc test` read the registry's `string` result
/// and refused the integer compare the check accepted.
#[cfg(feature = "cross-module-imports")]
#[test]
fn mindc_test_reads_the_declarations_its_check_reads() {
    let dir = scratch_dir("string_flows_backend_parity_table");
    let _ = std::fs::remove_dir_all(&dir);
    let files = [
        (
            "Mind.toml",
            "[package]\nname = \"tbl\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\n\
             [targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\", \"src/alpha.mind\", \
             \"src/zeta.mind\", \"src/t.mind\"]\n",
        ),
        (
            "main.mind",
            "import t;\n\nfn main() -> i64 {\n    return 0;\n}\n",
        ),
        (
            "src/alpha.mind",
            "export { mk }\n\nfn mk() -> i64 {\n    return 7;\n}\n",
        ),
        (
            "src/zeta.mind",
            "export { mk }\n\nfn mk() -> string {\n    return \"seven\";\n}\n",
        ),
        (
            "src/t.mind",
            "import alpha;\n\nfn same(x: i64) -> i64 {\n    if mk() == x {\n        return 1;\n    \
             }\n    return 0;\n}\n\n#[test]\nfn t_same() {\n    assert same(7) == 1, \"seven\";\n}\n",
        ),
    ];
    for (path, src) in files {
        let at = dir.join(path);
        std::fs::create_dir_all(at.parent().unwrap()).expect("create fixture dir");
        std::fs::write(at, src).expect("write fixture");
    }
    let run = |args: &[&str]| {
        Command::new(mindc_bin())
            .args(args)
            .current_dir(&dir)
            .output()
            .expect("spawn mindc")
    };
    let check = run(&["check", "--no-fmt", "--no-lint", "src/t.mind"]);
    let check_text = text(&check);
    assert!(check.status.success(), "check refused:\n{check_text}");
    let test = run(&["test", "src/t.mind"]);
    let test_text = text(&test);
    assert!(
        test.status.success() && test_text.contains("1 passed"),
        "mindc test decided otherwise than its check:\n{test_text}"
    );
    // Positive control: importing `zeta` makes `mk()` a string, refused by both.
    let t = dir.join("src/t.mind");
    let source = std::fs::read_to_string(&t).expect("read t");
    std::fs::write(&t, source.replace("import alpha;", "import zeta;")).expect("write t");
    for args in [
        &["check", "--no-fmt", "--no-lint", "src/t.mind"][..],
        &["test", "src/t.mind"],
    ] {
        let out = run(args);
        let out_text = text(&out);
        assert!(
            !out.status.success() && out_text.contains("E2038"),
            "{args:?} accepted a string compared with an integer:\n{out_text}"
        );
    }
}

/// A check or a test of `main.mind` decides over the program it links
/// (`main` and `provider`; a cdylib build links the same modules): `other`,
/// which `main` does not link, hands strings to `provider.same`. Once `main`
/// links `other`, its check, test and build refuse.
#[cfg(feature = "cross-module-imports")]
#[test]
fn a_module_the_program_does_not_link_stores_nothing_it_reads() {
    let dir = scratch_dir("string_flows_backend_parity_program");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create project");
    let main = |imports: &str| {
        format!(
            "{imports}\npub fn probe() -> i64 {{ return provider.same(1, 2) }}\n\n\
             #[test]\nfn t_probe() {{\n    assert probe() == 0, \"unequal\";\n}}\n"
        )
    };
    let files = [
        (
            "Mind.toml",
            "[package]\nname = \"prog\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
             emit = \"cdylib\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
             sources = [\"main.mind\", \"provider.mind\", \"other.mind\"]\n"
                .to_string(),
        ),
        (
            "provider.mind",
            "export { same }\n\nfn same(a: i64, b: i64) -> i64 {\n    if a == b { return 1 }\n    \
             return 0\n}\n"
                .to_string(),
        ),
        (
            "other.mind",
            "import provider;\n\nexport { other }\n\n\
             pub fn other() -> i64 { return provider.same(\"abc\", \"abc\") }\n"
                .to_string(),
        ),
        ("main.mind", main("import provider;\n")),
    ];
    for (path, src) in &files {
        std::fs::write(dir.join(path), src).expect("write fixture");
    }
    let run = |args: &[&str]| {
        Command::new(mindc_bin())
            .args(args)
            .current_dir(&dir)
            .output()
            .expect("spawn mindc")
    };
    let check = run(&["check", "--no-fmt", "--no-lint", "main.mind"]);
    let check_text = text(&check);
    assert!(
        check.status.success(),
        "check of main refused:\n{check_text}"
    );
    let test = run(&["test", "main.mind"]);
    let test_text = text(&test);
    assert!(
        test.status.success() && test_text.contains("1 passed"),
        "mindc test of main refused:\n{test_text}"
    );
    // Positive controls: the module that stores, and `main` once it links it.
    let other = run(&["check", "--no-fmt", "--no-lint", "other.mind"]);
    let other_text = text(&other);
    assert!(
        !other.status.success() && other_text.contains("E2038"),
        "check of other accepted:\n{other_text}"
    );
    std::fs::write(
        dir.join("main.mind"),
        main("import provider;\nimport other;\n"),
    )
    .expect("write main");
    let so = dir.join("prog.so");
    let build_args = [
        "build",
        "--backend",
        "mlir",
        "--emit",
        "cdylib",
        "--out",
        so.to_str().unwrap(),
    ];
    for args in [
        &["check", "--no-fmt", "--no-lint", "main.mind"][..],
        &["test", "main.mind"],
        &build_args,
    ] {
        let out = run(args);
        let out_text = text(&out);
        assert!(
            !out.status.success() && out_text.contains("E2038"),
            "{args:?} accepted a program that links `other`:\n{out_text}"
        );
    }
    assert!(!so.exists(), "a refused build left an artifact");
}

/// Copy fixture directory `from` into `to`, recursively.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create fixture copy");
    for entry in std::fs::read_dir(from).expect("read fixture") {
        let entry = entry.expect("fixture entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("fixture entry type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy fixture file");
        }
    }
}

/// A store of a value of unknown type into a collection element of one
/// project module (`src/z.mind`) refuses the typed element compares of the
/// modules that can exchange a collection with it (`src/j.mind` imports it),
/// and leaves the typed compares of the others (`src/k.mind`) integer
/// compares. Element facts are keyed by fact, not by slot: before they were
/// kept per module, `z`'s store made every `[u8; N]` element of the project
/// maybe a string.
#[cfg(feature = "cross-module-imports")]
#[test]
fn element_stores_of_an_unrelated_module_keep_typed_compares() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/string_eq_module_scope");
    let dir = scratch_dir("string_flows_backend_parity_module_scope").join("project");
    let _ = std::fs::remove_dir_all(&dir);
    copy_tree(&fixture, &dir);
    let check = |file: &str| {
        Command::new(mindc_bin())
            .args(["check", "--no-fmt", "--no-lint", file])
            .current_dir(&dir)
            .output()
            .expect("spawn mindc check")
    };
    let unrelated = check("src/k.mind");
    let unrelated_text = text(&unrelated);
    assert!(
        unrelated.status.success() && !unrelated_text.contains("E2038"),
        "typed compares of a module no importer shares with `z` refused:\n{unrelated_text}"
    );
    // Positive control: the module that imports `z` is still refused.
    let related = check("src/j.mind");
    let related_text = text(&related);
    assert!(
        !related.status.success(),
        "check of the module importing `z` accepted:\n{related_text}"
    );
    assert!(
        related_text.contains("src/j.mind:9:12") && related_text.contains("E2038"),
        "{related_text}"
    );
}

/// A string literal arm lowers to the runtime byte-wise comparison, which
/// reads no record through the null handle; an inline record decode read
/// the length at address 8 for it.
#[test]
fn a_string_literal_arm_lowers_to_the_byte_wise_comparison() {
    let dir = scratch_dir("string_flows_backend_parity_arm");
    let src = dir.join("arm.mind");
    std::fs::write(
        &src,
        "fn is_abc(s: string) -> i64 {\n    match s {\n        \"abc\" => 1,\n        \
         _ => 0,\n    }\n}\n",
    )
    .expect("write arm source");
    let out = mindc(&[src.to_str().unwrap(), "--emit-mlir"]);
    let ir = text(&out);
    assert!(out.status.success(), "lowering failed:\n{ir}");
    assert!(ir.contains("__mind_string_eq"), "no byte-wise test:\n{ir}");
    assert!(
        !ir.contains("__mind_load_i8"),
        "inline record decode:\n{ir}"
    );
}

/// String match arms and string methods: each case is a
/// zero-argument `pub fn`, and both backends must give the expected value.
const DEFINED: &str = r#"fn is_abc(s: string) -> i64 {
    match s {
        "abc" => 1,
        _ => 0,
    }
}

fn tlen(s: string) -> i64 { return string_len(s.trim()) }

fn slen(s: string) -> i64 { return string_len(s.slice(1, 3)) }

fn mlen(s: string) -> i64 { return s.len() }

fn tails(a: string, b: string) -> i64 {
    if a.slice(1, 3) == b.slice(1, 3) { return 1 }
    return 0
}

"#;

const DEFINED_CASES: &[(&str, &str, i64)] = &[
    ("match_equal", "is_abc(\"abc\")", 1),
    ("match_unequal", "is_abc(\"abd\")", 0),
    ("match_built", "is_abc(string_slice(\"xabc\", 1, 4))", 1),
    ("match_null", "is_abc(0)", 0),
    ("trim", "tlen(\" ab \")", 2),
    ("trim_all_space", "tlen(\"  \")", 0),
    ("slice", "slen(\"abcd\")", 2),
    ("len_method", "mlen(\"abc\")", 3),
    ("literal_len_method", "\"abcd\".len()", 4),
    ("bare_string_len", "string_len(\"abc\")", 3),
    ("slices_equal", "tails(\"xbcd\", \"ybce\")", 1),
    ("slices_unequal", "tails(\"xbcd\", \"ybde\")", 0),
];

fn defined_source() -> String {
    let mut src = String::from(DEFINED);
    for (case, expr, _) in DEFINED_CASES {
        src.push_str(&format!("\npub fn {case}() -> i64 {{ return {expr} }}\n"));
    }
    src
}

/// The value each case produces under `mindc test` (the interpreter), one
/// module per case. Integer controls of these shapes are unit tests
/// (`string_equality::flow_tests`): a `mindc test` module that holds these
/// helpers and a loop or tuple helper calling `eqh` overflows the
/// interpreter's stack on the previous head too, which this test does not
/// measure.
fn interpreter_values(dir: &Path) -> Vec<i64> {
    DEFINED_CASES
        .iter()
        .map(|(case, expr, _)| {
            let mut src = format!(
                "{DEFINED}\npub fn {case}() -> i64 {{ return {expr} }}\n\n#[test]\n\
                 fn probed() {{\n    let v: i64 = {case}()\n"
            );
            for candidate in 0..=4 {
                src.push_str(&format!(
                    "    assert v != {candidate}, \"value of {case} is {candidate}.\"\n"
                ));
            }
            src.push_str("}\n");
            let path = dir.join(format!("interp_{case}.mind"));
            std::fs::write(&path, src).expect("write interpreter source");
            let log = text(&mindc(&["test", path.to_str().unwrap()]));
            (0..=4)
                .find(|candidate| log.contains(&format!("value of {case} is {candidate}.")))
                .unwrap_or_else(|| panic!("mindc test gave no value for `{case}`:\n{log}"))
        })
        .collect()
}

/// The value each case produces from the MLIR-built shared library, one
/// child process per case so a crash names its case.
fn mlir_values(dir: &Path) -> Option<Vec<i64>> {
    let src = dir.join("defined_mlir.mind");
    let so = dir.join("defined_mlir.so");
    std::fs::write(&src, defined_source()).expect("write build source");
    let check = mindc(&["check", "--no-fmt", src.to_str().unwrap()]);
    assert!(check.status.success(), "check refused:\n{}", text(&check));
    let out = mindc(&[src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()]);
    if !common::gate::compiled("string_flows_backend_parity", &out) {
        return None;
    }
    let values = DEFINED_CASES
        .iter()
        .map(|(case, _, _)| {
            let py = format!(
                "import ctypes\nlib = ctypes.CDLL(r'{}')\nf = lib.{case}\n\
                 f.restype = ctypes.c_int64\nprint(f())\n",
                so.display()
            );
            let ran = Command::new("python3")
                .args(["-c", &py])
                .output()
                .expect("spawn python3");
            assert!(ran.status.success(), "`{case}` crashed:\n{}", text(&ran));
            String::from_utf8_lossy(&ran.stdout)
                .trim()
                .parse()
                .expect("an i64")
        })
        .collect();
    Some(values)
}

#[test]
fn string_matches_and_methods_agree_between_mindc_test_and_the_mlir_build() {
    let dir = scratch_dir("string_flows_backend_parity_defined");
    let expected: Vec<i64> = DEFINED_CASES.iter().map(|(_, _, v)| *v).collect();
    let interp = interpreter_values(&dir);
    assert_eq!(
        interp, expected,
        "mindc test values, in DEFINED_CASES order"
    );
    if let Some(built) = mlir_values(&dir) {
        assert_eq!(built, expected, "MLIR build values, in DEFINED_CASES order");
    }
}
