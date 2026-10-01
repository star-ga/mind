// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Another module's enum variant must lower to its tag, never panic.
//!
//! When two project modules declare an enum with the same name, the
//! whole-project registry keys each one by its owner (`crate.table::Table`).
//! A module that imports ONE of them and writes the bare spelling
//! `Table::RawResponses` type-checked, but the spelling was never resolved to
//! the owner-qualified key, so native lowering found no tag and aborted with
//! "undefined identifier `Table::RawResponses` reached lowering" (and, for a
//! `match` arm, the dangling-variant panic). The parser now resolves the bare
//! head through the importing module's imports, so the variant lowers to the
//! same discriminant its own module uses.
//!
//! The run gate builds and executes a four-module project: `table.mind` and
//! `events.mind` own the imported enums, `shadow.mind` declares PRIVATE enums
//! with the same names (the collision), and `main.mind` uses the imported
//! variants as values, payload constructors, `match` patterns, the dot form,
//! and in `==` tag comparisons against values built inside the owning module.
//!
//! Negative controls: a bare variant of an enum the module does not import,
//! an ambiguous bare head, and an unknown variant are refused with a located
//! E2002 diagnostic instead of a lowering panic. `let x = null;` (`null` is the
//! std JSON constructor FUNCTION, not a literal) is refused at check time with
//! a located E2037 (function used as a value) instead of reaching the
//! undefined-identifier panic.
//!
//! Gate: `cargo test --features "mlir-build cross-module-imports"
//!                   --test cross_module_enum_variant_lowering`

#![cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]

mod common;
use common::mindc_bin;

use std::path::Path;
use std::process::{Command, Output};

const MANIFEST: &str = "[package]\nname = \"xmvariant\"\nversion = \"0.1.0\"\n\n\
                        [build]\nentry = \"src/main.mind\"\n";

const TABLE: &str = r#"pub enum Table { Other, RawResponses }

pub enum Scope { Load, Extract }

pub fn raw() -> Table {
    Table::RawResponses
}

pub fn extract() -> Scope {
    Scope::Extract
}

pub fn table_code(t: Table) -> i64 {
    match t {
        Table::Other => 1,
        Table::RawResponses => 2,
    }
}
"#;

const EVENTS: &str = r#"pub enum XmlEvent { Start, Text(i64), End }

pub fn event_code(e: XmlEvent) -> i64 {
    match e {
        XmlEvent::Start => 1,
        XmlEvent::Text(n) => 100 + n,
        XmlEvent::End => 3,
    }
}
"#;

// Same enum names, kept private by the explicit export list, with different
// variant lists. Their presence makes every `Table` / `Scope` / `XmlEvent`
// registry key owner-qualified. The module's own bare spellings must keep
// resolving to its own declarations.
const SHADOW: &str = r#"export { shadow_code }

enum Table { A, B, C }

enum Scope { Only }

enum XmlEvent { Open, Close }

fn shadow_code() -> i64 {
    let t = Table::C
    let e = XmlEvent::Close
    let s = Scope::Only
    let a = match t {
        Table::A => 1,
        Table::B => 2,
        Table::C => 3,
    }
    let b = match e {
        XmlEvent::Open => 10,
        XmlEvent::Close => 20,
    }
    let c = match s {
        Scope::Only => 100,
    }
    a + b + c
}
"#;

// Each check returns a distinct failure code; 42 means every check passed.
const MAIN: &str = r#"import table
import events
import shadow

fn local_table(t: Table) -> i64 {
    match t {
        Table::Other => 5,
        Table::RawResponses => 6,
    }
}

fn local_event(e: XmlEvent) -> i64 {
    match e {
        XmlEvent::Start => 7,
        XmlEvent::Text(n) => n,
        XmlEvent::End => 8,
    }
}

fn main() -> i64 {
    let r = Table::RawResponses
    let o = Table.Other
    let s = Scope::Extract
    let x = XmlEvent::Text(9)
    if table_code(r) != 2 {
        return 11
    }
    if table_code(o) != 1 {
        return 12
    }
    if local_table(r) != 6 {
        return 13
    }
    if r != raw() {
        return 14
    }
    if s != extract() {
        return 15
    }
    if Scope::Load == extract() {
        return 16
    }
    if local_event(x) != 9 {
        return 17
    }
    if event_code(x) != 109 {
        return 18
    }
    if event_code(XmlEvent::End) != 3 {
        return 19
    }
    if shadow_code() != 123 {
        return 20
    }
    return 42
}
"#;

fn write_project(root: &Path, main: &str) {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root.join("src")).expect("mkdir project");
    std::fs::write(root.join("Mind.toml"), MANIFEST).expect("write manifest");
    std::fs::write(root.join("src/table.mind"), TABLE).expect("write table");
    std::fs::write(root.join("src/events.mind"), EVENTS).expect("write events");
    std::fs::write(root.join("src/shadow.mind"), SHADOW).expect("write shadow");
    std::fs::write(root.join("src/main.mind"), main).expect("write main");
}

fn build(root: &Path, artifact: &Path) -> Output {
    Command::new(mindc_bin())
        .args(["build", "--no-cache", "--out", artifact.to_str().unwrap()])
        .current_dir(root)
        .output()
        .expect("run mindc build")
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// `check` and `build` must both refuse `main` with a located `code`
/// diagnostic carrying `needle`, without a panic and without writing an
/// artifact.
fn assert_refused(root: &Path, case: &str, main: &str, code: &str, needle: &str, location: &str) {
    write_project(root, main);
    let checked = Command::new(mindc_bin())
        .args(["check", "--no-fmt", "--no-lint"])
        .current_dir(root)
        .output()
        .expect("run mindc check");
    let check_text = text(&checked);
    assert!(!checked.status.success(), "{case}: check accepted it");
    assert!(check_text.contains(code), "{case}: {check_text}");
    assert!(check_text.contains(needle), "{case}: {check_text}");
    assert!(check_text.contains(location), "{case}: {check_text}");
    assert!(!check_text.contains("panicked"), "{case}: {check_text}");

    let artifact = root.join(format!("{case}.bin"));
    let built = build(root, &artifact);
    let build_text = text(&built);
    assert!(!built.status.success(), "{case}: build accepted it");
    assert!(build_text.contains(code), "{case}: {build_text}");
    assert!(build_text.contains(needle), "{case}: {build_text}");
    assert!(!build_text.contains("panicked"), "{case}: {build_text}");
    assert!(!artifact.exists(), "{case}: build left an artifact");
}

#[test]
fn imported_enum_variants_lower_to_their_owner_tags() {
    let root = common::scratch_dir("cross_module_enum_variant_lowering").join("run");
    write_project(&root, MAIN);
    let artifact = root.join("xmvariant.bin");
    let out = build(&root, &artifact);
    assert!(
        !text(&out).contains("panicked"),
        "lowering panicked on another module's enum variant:\n{}",
        text(&out)
    );
    if !crate::common::gate::compiled("cross_module_enum_variant_lowering", &out) {
        return;
    }
    let run = Command::new(&artifact)
        .output()
        .expect("run the built program");
    assert_eq!(
        run.status.code(),
        Some(42),
        "an imported enum variant lowered to the wrong tag (exit code names the \
         failed check; 42 = all passed):\n{}",
        text(&run)
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn unresolvable_bare_variants_are_refused_with_a_located_diagnostic() {
    let root = common::scratch_dir("cross_module_enum_variant_lowering").join("refuse");
    // The enum lives in `table.mind`, which this module does not import.
    assert_refused(
        &root,
        "not_imported",
        "fn main() -> i64 {\n    let t = Table::RawResponses\n    return 0\n}\n",
        "E2002",
        "enum `Table` in `Table::RawResponses` is not visible in this module",
        "main.mind:2:13",
    );
    // Two imported modules export an enum named `Table`: the bare head cannot
    // pick an owner. (`shadow`'s `Table` is private and never competes.)
    let ambiguous = "import table\nimport events\nfn main() -> i64 {\n    match Table::Other {\n        Table::Other => 0,\n        _ => 1,\n    }\n}\n";
    write_project(&root, ambiguous);
    std::fs::write(
        root.join("src/events.mind"),
        format!("{EVENTS}\npub enum Table {{ Other }}\n"),
    )
    .expect("write duplicate exporter");
    let checked = Command::new(mindc_bin())
        .args(["check", "--no-fmt", "--no-lint"])
        .current_dir(&root)
        .output()
        .expect("run mindc check");
    let check_text = text(&checked);
    assert!(!checked.status.success(), "ambiguous: {check_text}");
    assert!(
        check_text.contains("ambiguous enum `Table` in `Table::Other`"),
        "ambiguous: {check_text}"
    );
    assert!(
        check_text.contains("main.mind:4:11"),
        "ambiguous: {check_text}"
    );
    assert!(!check_text.contains("panicked"), "ambiguous: {check_text}");

    assert_refused(
        &root,
        "unknown_variant",
        "import table\nfn main() -> i64 {\n    let t = Table::Missing\n    return 0\n}\n",
        "E2002",
        "unknown variant `Missing` of imported enum `Table`",
        "main.mind:3:13",
    );
    // `null` is std.json's constructor FUNCTION; MIND has no null literal and
    // no first-class functions, so it has no value to lower. The project
    // builder must refuse it outright rather than embed it as a runtime
    // fallback (that path lowers the module too).
    assert_refused(
        &root,
        "null_value",
        "import table\nfn main() -> i64 {\n    let x = null;\n    return 0\n}\n",
        "E2037",
        "`null` is a function, not a value",
        "main.mind:3:13",
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// The single-translation-unit pipeline (the API `mindc build file.mind` and
/// the interpreter drive) refuses `let x = null;` with a structured, located
/// diagnostic before lowering instead of panicking inside it. A local binding
/// named `null` is an ordinary value and still compiles.
#[test]
fn null_value_is_a_structured_refusal_not_a_lowering_panic() {
    use libmind::pipeline::{CompileError, CompileOptions, compile_source};

    let src = "fn main() -> i64 {\n    let x = null;\n    return 0\n}\n";
    let result = std::panic::catch_unwind(|| compile_source(src, &CompileOptions::default()))
        .expect("compiling `let x = null;` panicked");
    let diags = match result {
        Err(CompileError::TypeError(diags)) => diags,
        Err(other) => panic!("expected a type-check refusal, got {other:?}"),
        Ok(_) => panic!("`let x = null;` compiled; `null` names a function"),
    };
    let null = diags
        .iter()
        .find(|d| d.message.contains("`null` is a function, not a value"))
        .unwrap_or_else(|| panic!("no function-as-value diagnostic in {diags:?}"));
    assert_eq!(null.code, "E2037");
    let span = null.span.as_ref().expect("diagnostic carries a location");
    assert_eq!((span.line, span.column), (2, 13), "{null:?}");
    // `null` IS a resolvable name and nothing is called: the unknown-identifier
    // and function-value-call rules keep their verdict at this position.
    assert!(
        !diags.iter().any(|d| matches!(d.code, "E2002" | "E2012")),
        "{diags:?}"
    );

    // The module's own function used as a value is refused the same way.
    let local_fn = "fn helper() -> i64 {\n    return 1\n}\nfn main() -> i64 {\n    let f = helper\n    return 0\n}\n";
    match compile_source(local_fn, &CompileOptions::default()) {
        Err(CompileError::TypeError(diags)) => assert!(
            diags
                .iter()
                .any(|d| d.code == "E2037" && d.message.contains("`helper` is a function")),
            "{diags:?}"
        ),
        other => panic!("`let f = helper` was not refused: {:?}", other.err()),
    }

    // Controls: a local binding and a module const named like a std function
    // are values and keep compiling.
    for ok in [
        "fn main() -> i64 {\n    let null = 5\n    let x = null\n    return x\n}\n",
        "const null: i64 = 5\nfn main() -> i64 {\n    let x = null\n    return x\n}\n",
    ] {
        if let Err(e) = compile_source(ok, &CompileOptions::default()) {
            panic!("value named `null` was refused: {e:?}\n{ok}");
        }
    }
}
