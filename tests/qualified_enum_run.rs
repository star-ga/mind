// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native gate for module-qualified enum constructors, annotations, patterns,
//! and same-name module ownership (issue #239).

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;
use std::process::Command;

fn assert_e2002_refusal(mindc: &std::path::Path, proj: &std::path::Path, case: &str, src: &str) {
    std::fs::write(proj.join("src/main.mind"), src).expect("write negative main");
    let checked = Command::new(mindc)
        .args(["check", "--no-fmt", "--no-lint"])
        .current_dir(proj)
        .output()
        .expect("run negative qualified enum check");
    let check_text = String::from_utf8_lossy(&checked.stdout);
    assert!(
        !checked.status.success(),
        "{case}: check accepted invalid source"
    );
    assert!(check_text.contains("E2002"), "{case}: {check_text}");
    assert!(!check_text.contains("panicked"), "{case}: {check_text}");

    let artifact = proj.join(format!("target/debug/{case}"));
    let built = Command::new(mindc)
        .args([
            "build",
            "--backend",
            "mlir",
            "--out",
            artifact.to_str().unwrap(),
            "--no-cache",
        ])
        .current_dir(proj)
        .output()
        .expect("run negative qualified enum build");
    let build_text = String::from_utf8_lossy(&built.stderr);
    assert!(
        !built.status.success(),
        "{case}: build accepted invalid source"
    );
    assert!(build_text.contains("E2002"), "{case}: {build_text}");
    assert!(!build_text.contains("panicked"), "{case}: {build_text}");
    assert!(!artifact.exists(), "{case}: build left an artifact");
}

const DEFS: &str = "pub enum Color { Red(i64), Blue }\n\
                    pub fn make() -> Color { Color.Red(41) }\n\
                    pub enum Predicate { DependsOn, Cites }\n\
                    pub fn make_predicate() -> Predicate { Predicate::DependsOn }\n\
                    pub struct Record { pub value: i64 }\n\
                    pub fn make_record() -> Record { Record { value: 41 } }\n\
                    type Count = i64;\n\
                    pub fn make_count() -> Count { 41 }\n";
const OTHER: &str = "pub enum Color { Red(i64), Blue }\n\
                     pub fn make_other() -> Color { Color.Red(99) }\n\
                     pub struct Record { pub value: i64 }\n\
                     type Count = i64;\n";
const MAIN: &str = r#"
import defs;
import other;

type ImportedRecordRef = &defs.Record;
type ImportedColorArray = [defs.Color; 1];
type ImportedTypes = (defs.Record, defs.Count);
type ImportedOption = Option<defs.Record>;
type ImportedPointer = *const defs.Record;
type ImportedCallback = extern "C" fn(defs.Record) -> defs.Count;

pub fn qualified() -> i64 {
    let c: defs.Color = defs.Color.Red(41);
    match c {
        defs.Color.Red(v) => v + 1,
        defs.Color.Blue => 0,
    }
}

pub fn other_qualified() -> i64 {
    let c = other.make_other();
    match c {
        other.Color.Red(v) => v + 1,
        other.Color.Blue => 0,
    }
}

pub fn other_inferred_bare() -> i64 {
    let c = other.make_other();
    match c {
        Red(v) => v + 1,
        Blue => 0,
    }
}

pub fn qualified_unit() -> i64 {
    let p = defs.make_predicate();
    match p {
        defs.Predicate::DependsOn => 1,
        defs.Predicate::Cites => 0,
    }
}

pub fn ownership() -> i64 {
    let c = defs.make();
    match c {
        other.Color.Red(v) => 99,
        defs.Color.Red(v) => v + 1,
        defs.Color.Blue => 0,
    }
}

pub fn qualified_struct_field() -> i64 {
    let record: defs.Record = defs.make_record();
    record.value + 1
}

pub fn qualified_alias_value() -> i64 {
    let count: defs.Count = defs.make_count();
    count + 1
}
"#;

#[test]
fn qualified_enum_paths_run_and_unknown_paths_refuse() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "qualified_enum_run",
            "qualified-enum-run: mindc not found; skipping",
        );
        return;
    }
    let proj = crate::common::scratch_dir("qualified_enum_run").join("qualified");
    let src = proj.join("src");
    let _ = std::fs::remove_dir_all(&proj);
    std::fs::create_dir_all(&src).expect("mkdir src");
    std::fs::write(
        proj.join("Mind.toml"),
        "[package]\nname = \"qualified_enum\"\nversion = \"0.1.0\"\n\n[build]\nemit = \"cdylib\"\n",
    )
    .expect("write manifest");
    std::fs::write(src.join("defs.mind"), DEFS).expect("write defs");
    std::fs::write(src.join("other.mind"), OTHER).expect("write other");
    std::fs::write(src.join("main.mind"), MAIN).expect("write main");

    let out = Command::new(&mindc)
        .args(["build", "--backend", "mlir"])
        .current_dir(&proj)
        .output()
        .expect("run qualified enum build");
    if !crate::common::gate::compiled("qualified_enum_run", &out) {
        return;
    }
    let so = proj.join("target/debug/libqualified_enum.so");
    assert!(
        so.exists(),
        "qualified enum gate did not emit {}",
        so.display()
    );
    let py = format!(
        "import ctypes\nlib=ctypes.CDLL(r'{}')\nfor n,e in (('qualified',42),('other_qualified',100),('other_inferred_bare',100),('qualified_unit',1),('ownership',42),('qualified_struct_field',42),('qualified_alias_value',42)):\n f=getattr(lib,n); f.restype=ctypes.c_int64; r=f(); assert r==e,(n,r)\nprint('ok')\n",
        so.display()
    );
    let run = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run qualified enum artifact");
    assert!(
        run.status.success(),
        "qualified enum native checks failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let first = std::fs::read(&so).expect("read first qualified enum artifact");
    let repeat = proj.join("target/debug/qualified_enum_repeat.so");
    let rebuilt = Command::new(&mindc)
        .args([
            "build",
            "--backend",
            "mlir",
            "--out",
            repeat.to_str().unwrap(),
            "--no-cache",
        ])
        .current_dir(&proj)
        .output()
        .expect("repeat qualified enum build");
    assert!(rebuilt.status.success(), "repeat build failed");
    assert_eq!(
        first,
        std::fs::read(&repeat).expect("read repeated artifact"),
        "qualified enum artifact changed across identical builds"
    );

    assert_e2002_refusal(
        &mindc,
        &proj,
        "unknown_owner",
        r#"import defs;
fn main() -> i32 {
    let c = defs.make();
    match c {
        missing.Color.Red(v) => return v,
        defs.Color.Red(v) => return v + 1,
        defs.Color.Blue => return 0,
    }
}
"#,
    );
    assert_e2002_refusal(
        &mindc,
        &proj,
        "unknown_let_type",
        "import defs;\nfn main() -> i32 { let c: defs.Missing = 0; return 0; }\n",
    );
    assert_e2002_refusal(
        &mindc,
        &proj,
        "unknown_return_type",
        "import defs;\nfn bad() -> missing.Color { defs.Color.Red(1) }\nfn main() -> i32 { return 0; }\n",
    );
    for (case, main) in [
        (
            "unknown_array_element",
            "import defs; fn main() -> i32 { let x: [defs.Missing; 1] = [1]; 0 }",
        ),
        (
            "unknown_slice_element",
            "import defs; fn unused(x: &[defs.Missing]) -> i32 { 0 } fn main() -> i32 { 0 }",
        ),
        (
            "unknown_ref_target",
            "import defs; fn unused(x: &defs.Missing) -> i32 { 0 } fn main() -> i32 { 0 }",
        ),
        (
            "unknown_tuple_element",
            "import defs; fn unused(x: (i64, defs.Missing)) -> i32 { 0 } fn main() -> i32 { 0 }",
        ),
        (
            "unknown_generic_argument",
            "import defs; fn unused(x: Option<defs.Missing>) -> i32 { 0 } fn main() -> i32 { 0 }",
        ),
        (
            "unknown_raw_pointer",
            "import defs; fn unused(x: *const defs.Missing) -> i32 { 0 } fn main() -> i32 { 0 }",
        ),
        (
            "unknown_fn_pointer",
            "import defs; fn unused(x: extern \"C\" fn(defs.Missing) -> i32) -> i32 { 0 } fn main() -> i32 { 0 }",
        ),
        (
            "unknown_struct_field",
            "import defs; struct Bad { value: defs.Missing } fn main() -> i32 { 0 }",
        ),
        (
            "unknown_alias_target",
            "import defs; type Bad = defs.Missing; fn main() -> i32 { 0 }",
        ),
        (
            "unknown_enum_payload",
            "import defs; enum Bad { Value(defs.Missing) } fn main() -> i32 { 0 }",
        ),
        (
            "unknown_const_type",
            "import defs; const BAD: defs.Missing = 0; fn main() -> i32 { 0 }",
        ),
        (
            "unknown_cast_type",
            "import defs; fn main() -> i32 { let x = 0 as defs.Missing; 0 }",
        ),
    ] {
        assert_e2002_refusal(&mindc, &proj, case, main);
    }
    assert_e2002_refusal(
        &mindc,
        &proj,
        "unknown_imported_unit",
        r#"import defs;
fn main() -> i32 {
    let c = defs.make();
    match c {
        defs.Color.Missing => return 7,
        defs.Color.Red(v) => return v + 1,
        defs.Color.Blue => return 0,
    }
}
"#,
    );
    assert_e2002_refusal(
        &mindc,
        &proj,
        "unknown_owner_unit",
        r#"import defs;
fn main() -> i32 {
    let c = defs.make();
    match c {
        missing.Color.Blue => return 7,
        defs.Color.Red(v) => return v + 1,
        defs.Color.Blue => return 0,
    }
}
"#,
    );

    assert_e2002_refusal(
        &mindc,
        &proj,
        "unimported_owner",
        r#"import defs;
fn main() -> i32 {
    let c = defs.make();
    match c {
        other.Color::Red(v) => return 99,
        defs.Color::Red(v) => return v + 1,
        defs.Color::Blue => return 0,
    }
}
"#,
    );
    assert_e2002_refusal(
        &mindc,
        &proj,
        "unimported_struct_owner",
        "import defs; fn unused(x: other.Record) -> i32 { 0 } fn main() -> i32 { 0 }",
    );
    assert_e2002_refusal(
        &mindc,
        &proj,
        "unimported_alias_owner",
        "import defs; fn unused(x: other.Count) -> i32 { 0 } fn main() -> i32 { 0 }",
    );

    std::fs::write(
        src.join("defs.mind"),
        "export { make }\nenum Color { Red(i64), Blue }\nfn make() -> Color { Color.Red(41) }\n",
    )
    .expect("write private enum module");
    assert_e2002_refusal(
        &mindc,
        &proj,
        "private_enum",
        r#"import defs;
fn main() -> i32 {
    let c: defs.Color = defs.Color.Red(41);
    match c {
        defs.Color.Red(v) => return v,
        defs.Color.Blue => return 0,
    }
}
"#,
    );
    assert_e2002_refusal(
        &mindc,
        &proj,
        "private_enum_bare_variant",
        r#"import defs;
fn main() -> i32 {
    let c = defs.make();
    match c {
        Red(v) => return v,
        x => return 0,
    }
}
"#,
    );
    std::fs::write(
        src.join("defs.mind"),
        "export { make_record, make_count }\n\
         struct Record { pub value: i64 }\n\
         type Count = i64;\n\
         fn make_record() -> Record { Record { value: 41 } }\n\
         fn make_count() -> Count { 41 }\n",
    )
    .expect("write private named types module");
    assert_e2002_refusal(
        &mindc,
        &proj,
        "private_struct_type",
        "import defs; fn unused(x: defs.Record) -> i32 { 0 } fn main() -> i32 { 0 }",
    );
    assert_e2002_refusal(
        &mindc,
        &proj,
        "private_alias_type",
        "import defs; fn unused(x: defs.Count) -> i32 { 0 } fn main() -> i32 { 0 }",
    );
    std::fs::write(
        src.join("defs.mind"),
        "export { Record, Count, make_record, make_count }\n\
         struct Record { pub value: i64 }\n\
         type Count = i64;\n\
         fn make_record() -> Record { Record { value: 41 } }\n\
         fn make_count() -> Count { 1 }\n",
    )
    .expect("write explicitly exported named types module");
    std::fs::write(
        src.join("main.mind"),
        "import defs; pub fn explicit_types() -> i64 { let r: defs.Record = defs.make_record(); let c: defs.Count = defs.make_count(); r.value + c }",
    )
    .expect("write explicitly exported named types main");
    let explicit = proj.join("target/debug/explicit_types.so");
    let built = Command::new(&mindc)
        .args([
            "build",
            "--backend",
            "mlir",
            "--out",
            explicit.to_str().unwrap(),
            "--no-cache",
        ])
        .current_dir(&proj)
        .output()
        .expect("build explicitly exported named types");
    assert!(
        built.status.success(),
        "explicitly exported named types failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&built.stdout),
        String::from_utf8_lossy(&built.stderr)
    );
    let py = format!(
        "import ctypes\nl=ctypes.CDLL(r'{}'); f=l.explicit_types; f.restype=ctypes.c_int64; assert f()==42, f()\n",
        explicit.display()
    );
    assert!(
        Command::new("python3")
            .args(["-c", &py])
            .status()
            .expect("run explicitly exported named types")
            .success()
    );
    let _ = std::fs::remove_dir_all(&proj);
    nested_import_terminal_alias_keeps_the_defining_owner();
}

fn nested_import_terminal_alias_keeps_the_defining_owner() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "qualified_enum_nested",
            "qualified-enum-nested: mindc not found; skipping",
        );
        return;
    }
    let proj = crate::common::scratch_dir("qualified_enum_run").join("nested");
    let nested = proj.join("src/nested");
    let _ = std::fs::remove_dir_all(&proj);
    std::fs::create_dir_all(&nested).expect("mkdir nested src");
    std::fs::write(
        proj.join("Mind.toml"),
        "[package]\nname = \"qualified_nested\"\nversion = \"0.1.0\"\n\n[build]\nemit = \"cdylib\"\n",
    )
    .expect("write manifest");
    std::fs::write(
        nested.join("defs.mind"),
        "pub enum Color { Red(i64), Blue }\npub fn make() -> Color { Color.Red(41) }\n",
    )
    .expect("write nested defs");
    std::fs::write(
        proj.join("src/main.mind"),
        r#"import nested.defs;
pub fn terminal_alias() -> i64 {
    let c: defs.Color = defs.Color.Red(41);
    match c {
        nested.defs.Color.Red(v) => v + 1,
        nested.defs.Color.Blue => 0,
    }
}
pub fn crate_alias() -> i64 {
    let c: crate.nested.defs.Color = crate.nested.defs.Color::Red(41);
    match c {
        defs.Color::Red(v) => v + 1,
        defs.Color::Blue => 0,
    }
}
"#,
    )
    .expect("write nested main");
    let checked = Command::new(&mindc)
        .args(["check", "--no-fmt", "--no-lint"])
        .current_dir(&proj)
        .output()
        .expect("check nested aliases");
    assert!(checked.status.success(), "nested alias check failed");
    let built = Command::new(&mindc)
        .args(["build", "--backend", "mlir"])
        .current_dir(&proj)
        .output()
        .expect("build nested aliases");
    assert!(built.status.success(), "nested alias build failed");
    let so = proj.join("target/debug/libqualified_nested.so");
    let py = format!(
        "import ctypes\nl=ctypes.CDLL(r'{}')\nfor n in ('terminal_alias','crate_alias'):\n f=getattr(l,n); f.restype=ctypes.c_int64; assert f()==42,(n,f())\n",
        so.display()
    );
    assert!(
        Command::new("python3")
            .args(["-c", &py])
            .status()
            .expect("run nested aliases")
            .success()
    );
    let _ = std::fs::remove_dir_all(&proj);
}
