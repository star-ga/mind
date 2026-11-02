// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Every module shipped under `std/` resolves through the bundled table, and an
//! import of a `std` module that does not ship fails at the import.
//!
//! The defect: `std/keccak.mind` shipped, but the bundled table in
//! `src/project/stdlib.rs` had no `std.keccak` row (19 of the 43 shipped
//! modules had none). `import std.keccak;` was accepted, and every call was
//! then refused: `mindc check` reported E2003 `unsupported call to
//! keccak_sha3_512` and `mindc test` reported `unsupported operation`. An
//! import of a module that does not exist at all was accepted the same way.
//!
//! The enumeration below reads `std/*.mind` from disk, so a new std module is
//! red here until it has a row in the table.

#![cfg(feature = "std-surface")]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use libmind::project::stdlib::{STDLIB_MIND_SOURCES, parsed_stdlib_modules};

fn std_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("std")
}

/// `(module path, file)` for every `std/*.mind`, sorted by module path.
fn shipped_std_modules() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(std_dir()).expect("read std/") {
        let path = entry.expect("read a std/ entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("mind") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("std module file names are UTF-8")
            .to_string();
        out.push((format!("std.{stem}"), path));
    }
    out.sort();
    out
}

#[test]
fn every_shipped_std_module_is_in_the_bundled_table() {
    let shipped = shipped_std_modules();
    // Positive control: the walk saw the real tree, not an empty directory.
    assert!(
        shipped.iter().any(|(m, _)| m == "std.sha256"),
        "the std/ walk must see std/sha256.mind; saw {shipped:?}"
    );
    let missing: Vec<&str> = shipped
        .iter()
        .filter(|(m, _)| !STDLIB_MIND_SOURCES.iter().any(|(k, _)| k == m))
        .map(|(m, _)| m.as_str())
        .collect();
    assert!(
        missing.is_empty(),
        "shipped std modules with no row in STDLIB_MIND_SOURCES \
         (src/project/stdlib.rs); `import` of these is accepted and every call \
         is then refused: {missing:?}"
    );
    assert_eq!(
        STDLIB_MIND_SOURCES.len(),
        shipped.len(),
        "every table row must name a shipped std/*.mind file"
    );
    for (module, text) in STDLIB_MIND_SOURCES {
        let stem = module
            .strip_prefix("std.")
            .unwrap_or_else(|| panic!("table row `{module}` is not a std.* path"));
        let file = std_dir().join(format!("{stem}.mind"));
        let on_disk = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("table row `{module}`: read {}: {e}", file.display()));
        assert!(
            *text == on_disk,
            "table row `{module}` does not carry the text of {}",
            file.display()
        );
    }
    assert!(
        STDLIB_MIND_SOURCES.windows(2).all(|w| w[0].0 < w[1].0),
        "STDLIB_MIND_SOURCES must stay sorted by module path (deterministic insertion order)"
    );
}

#[test]
fn every_bundled_std_module_parses() {
    // `parsed_stdlib_modules` drops a source that does not parse. A dropped
    // module is accepted at the import and then refuses every call, so a
    // parse failure must be red here rather than silent.
    let parsed = parsed_stdlib_modules();
    let names: Vec<&str> = parsed.iter().map(|(p, _)| p.as_str()).collect();
    let dropped: Vec<&str> = STDLIB_MIND_SOURCES
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| !names.contains(k))
        .collect();
    assert!(
        dropped.is_empty(),
        "bundled std modules that do not parse: {dropped:?}"
    );
    assert!(names.contains(&"std.keccak"), "std.keccak must be bundled");
}

/// Every `pub fn` and `extern` function a shipped module declares resolves
/// through the module table the project loader builds from the bundle.
#[cfg(feature = "cross-module-imports")]
#[test]
fn every_public_function_of_every_shipped_module_resolves() {
    use libmind::ast::{Module, Node};
    use libmind::project::module_table::build_module_table;

    fn public_names(items: &[Node], out: &mut Vec<String>) {
        for item in items {
            match item {
                Node::FnDef(fd, _) if fd.is_pub => out.push(fd.name.clone()),
                Node::ExternBlock { fns, .. } => out.extend(fns.iter().map(|f| f.name.clone())),
                Node::Block { stmts, .. } => public_names(stmts, out),
                _ => {}
            }
        }
    }

    let parsed = parsed_stdlib_modules();
    let refs: Vec<(String, &Module)> = parsed.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&refs);
    let mut unresolved = Vec::new();
    let mut checked = 0usize;
    for (module, file) in shipped_std_modules() {
        let source = std::fs::read_to_string(&file).expect("read shipped std module");
        let ast = libmind::parser::parse(&source)
            .unwrap_or_else(|e| panic!("{module} does not parse: {e:?}"));
        let mut names = Vec::new();
        public_names(&ast.items, &mut names);
        assert!(!names.is_empty(), "{module} declares no public function");
        let path: Vec<String> = module.split('.').map(str::to_string).collect();
        for name in names {
            checked += 1;
            if !table.resolves(&path, &name) {
                unresolved.push(format!("{module}.{name}"));
            }
        }
    }
    assert!(checked > 0, "no public function was checked");
    assert!(
        unresolved.is_empty(),
        "{} of {checked} public std functions do not resolve through the bundled \
         table; first: {:?}",
        unresolved.len(),
        unresolved.iter().take(8).collect::<Vec<_>>()
    );
}

// ── CLI: the import and the call, end to end ───────────────────────────────

fn write_program(case: &str, source: &str) -> PathBuf {
    let dir = common::scratch_dir("std_bundled_module_table").join(case);
    std::fs::create_dir_all(dir.join(".git")).expect("create project boundary");
    std::fs::write(
        dir.join("Mind.toml"),
        format!("[package]\nname = \"{case}\"\nversion = \"0.0.1\"\n"),
    )
    .expect("write Mind.toml");
    let file = dir.join("p.mind");
    std::fs::write(&file, source).expect("write p.mind");
    file
}

fn mindc(sub: &str, file: &Path) -> Output {
    Command::new(common::mindc_bin())
        .arg(sub)
        .arg(file.file_name().expect("file name"))
        .current_dir(file.parent().expect("program dir"))
        .output()
        .unwrap_or_else(|e| panic!("spawn mindc {sub}: {e}"))
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

const MAIN_ONLY: &str = "\n\nfn main() -> i64 {\n    return 0;\n}\n";

#[test]
fn an_import_of_a_std_module_that_does_not_ship_fails_at_the_import() {
    // Control: the same program with a shipped module checks clean, so the
    // failure below is about the module name and nothing else.
    let control = write_program("unknown_control", &format!("import std.keccak;{MAIN_ONLY}"));
    let out = mindc("check", &control);
    assert!(
        out.status.success(),
        "control must check clean:\n{}",
        text(&out)
    );

    let file = write_program(
        "unknown_check",
        &format!("import std.no_such_module;{MAIN_ONLY}"),
    );
    let out = mindc("check", &file);
    let all = text(&out);
    assert!(
        !out.status.success(),
        "an import of a std module that does not ship must fail `mindc check`:\n{all}"
    );
    assert!(
        all.contains("p.mind:1:") && all.contains("std.no_such_module") && all.contains("E2002"),
        "the error must be E2002 at the import (line 1) and name the module:\n{all}"
    );

    let file = write_program(
        "unknown_test",
        "import std.no_such_module;\n\n#[test]\nfn t() {\n    assert 1 == 1;\n}\n",
    );
    let out = mindc("test", &file);
    let all = text(&out);
    assert!(
        !out.status.success() && all.contains("std.no_such_module"),
        "`mindc test` must refuse the import and name the module:\n{all}"
    );
}

/// The native backend admits sources without the type checker on the
/// single-file path, and its frozen compiler treats an import as a no-op, so it
/// refuses the unknown module at the import itself.
#[test]
fn the_native_backend_refuses_an_unknown_std_import_at_the_import() {
    let file = write_program(
        "unknown_native",
        &format!("import std.no_such_module;{MAIN_ONLY}"),
    );
    let out = Command::new(common::mindc_bin())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["build", "--backend", "native"])
        .arg(&file)
        .arg("--out")
        .arg(file.with_extension("elf"))
        .output()
        .expect("spawn mindc build --backend native");
    let all = text(&out);
    assert!(
        !out.status.success()
            && all.contains("scope.unknown_std_module")
            && all.contains("std.no_such_module"),
        "the native backend must refuse the unknown std module at the import:\n{all}"
    );
}

#[test]
fn the_builtin_std_modules_named_by_the_spec_still_import() {
    // spec/v1.0/stdlib.md "Module organization": core, math, tensor, diff and
    // io are standard-library modules whose entities need no import. Naming one
    // is not an unknown module.
    for module in ["std.core", "std.diff", "std.io", "std.math", "std.tensor"] {
        let case = format!("builtin_{}", module.replace('.', "_"));
        let file = write_program(&case, &format!("import {module};{MAIN_ONLY}"));
        let out = mindc("check", &file);
        assert!(
            out.status.success(),
            "`import {module};` must check clean:\n{}",
            text(&out)
        );
    }
}

#[cfg(feature = "cross-module-imports")]
fn scalar_call(name: &str, params: &[libmind::ast::Param]) -> Option<String> {
    use libmind::ast::TypeAnn;
    let args: Option<Vec<&str>> = params
        .iter()
        .map(|p| match p.ty {
            TypeAnn::ScalarI32 | TypeAnn::ScalarI64 | TypeAnn::ScalarU32 => Some("0"),
            TypeAnn::ScalarF32 | TypeAnn::ScalarF64 => Some("0.0"),
            TypeAnn::ScalarBool => Some("false"),
            _ => None,
        })
        .collect();
    args.map(|a| format!("{name}({})", a.join(", ")))
}

/// The first public function of `items` callable with scalar literals, as a
/// call expression; `externs` also admits `extern "C"` declarations.
#[cfg(feature = "cross-module-imports")]
fn first_call(items: &[libmind::ast::Node], externs: bool) -> Option<String> {
    use libmind::ast::Node;
    for item in items {
        let found = match item {
            Node::FnDef(fd, _) if fd.is_pub => scalar_call(&fd.name, &fd.params),
            Node::ExternBlock { fns, .. } if externs => {
                fns.iter().find_map(|f| scalar_call(&f.name, &f.params))
            }
            Node::Block { stmts, .. } => first_call(stmts, externs),
            _ => None,
        };
        if found.is_some() {
            return found;
        }
    }
    None
}

/// Every shipped module imports, and one of its public functions is callable
/// at `mindc check`. Before the fix the 19 modules without a table row were
/// accepted at the import and refused at the call (E2003).
#[cfg(feature = "cross-module-imports")]
#[test]
fn every_shipped_std_module_imports_and_a_public_function_checks() {
    let mut failures = Vec::new();
    let shipped = shipped_std_modules();
    for (module, file) in &shipped {
        let source = std::fs::read_to_string(file).expect("read shipped std module");
        let ast = libmind::parser::parse(&source).expect("shipped std module parses");
        let Some(call) = first_call(&ast.items, true) else {
            failures.push(format!(
                "{module}: no public function with scalar parameters"
            ));
            continue;
        };
        let case = format!("call_{}", module.replace('.', "_"));
        let program = format!(
            "import {module};\n\nfn probe() -> i64 {{\n    {call};\n    return 0;\n}}{MAIN_ONLY}"
        );
        let out = mindc("check", &write_program(&case, &program));
        if !out.status.success() {
            failures.push(format!("{module}: `{call}`:\n{}", text(&out)));
        }
    }
    assert!(
        shipped.iter().any(|(m, _)| m == "std.keccak"),
        "the std/ walk must see std/keccak.mind"
    );
    assert!(
        failures.is_empty(),
        "{} of {} shipped std modules are not importable-and-callable:\n{}",
        failures.len(),
        shipped.len(),
        failures.join("\n")
    );
}

// ── std.keccak known answers (FIPS 202) ────────────────────────────────────

/// SHA3-512 known answers from FIPS 202 (the empty message, "abc", and the
/// 1600-bit message of 200 bytes 0xa3), each digest as sixteen big-endian
/// 32-bit words. `kat_misses` counts the words that differ; `main` returns it,
/// so the interpreter and the native build report the same number.
#[cfg(feature = "cross-module-imports")]
fn keccak_kat_program(mutate: bool) -> String {
    // A one-word change in the expected table is the negative control: both
    // engines must then report exactly one miss.
    let first = if mutate { 2795467725u64 } else { 2795467724u64 };
    format!(
        r#"import std.keccak;

fn be32(p: i64, i: i64) -> i64 {{
    let b0: i64 = __mind_load_i8(p + i) & 255;
    let b1: i64 = __mind_load_i8(p + i + 1) & 255;
    let b2: i64 = __mind_load_i8(p + i + 2) & 255;
    let b3: i64 = __mind_load_i8(p + i + 3) & 255;
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}}

fn miss(out: i64, word: i64, want: i64) -> i64 {{
    if be32(out, word * 4) == want {{
        return 0;
    }}
    return 1;
}}

fn misses_empty() -> i64 {{
    let msg: i64 = __mind_alloc(8);
    let out: i64 = __mind_alloc(64);
    let rc: i64 = keccak_sha3_512(msg, 0, out);
    let mut n: i64 = rc;
    n = n + miss(out, 0, {first}) + miss(out, 1, 2721749701) + miss(out, 2, 3367331804) + miss(out, 3, 408581486);
    n = n + miss(out, 4, 2546565654) + miss(out, 5, 1340233817) + miss(out, 6, 3771849921) + miss(out, 7, 1197244582);
    n = n + miss(out, 8, 363991610) + miss(out, 9, 4059429196) + miss(out, 10, 300149056) + miss(out, 11, 742049112);
    n = n + miss(out, 12, 4110424477) + miss(out, 13, 2511786979) + miss(out, 14, 24479110) + miss(out, 15, 673041702);
    return n;
}}

fn misses_abc() -> i64 {{
    let msg: i64 = __mind_alloc(8);
    __mind_store_i8(msg, 97);
    __mind_store_i8(msg + 1, 98);
    __mind_store_i8(msg + 2, 99);
    let out: i64 = __mind_alloc(64);
    let rc: i64 = keccak_sha3_512(msg, 3, out);
    let mut n: i64 = rc;
    n = n + miss(out, 0, 3075573003) + miss(out, 1, 441915018) + miss(out, 2, 1452526994) + miss(out, 3, 1265305966);
    n = n + miss(out, 4, 150348162) + miss(out, 5, 1950676749) + miss(out, 6, 2286902530) + miss(out, 7, 1087533358);
    n = n + miss(out, 8, 283186921) + miss(out, 9, 422245321) + miss(out, 10, 444515702) + miss(out, 11, 1206096704);
    n = n + miss(out, 12, 1463028556) + miss(out, 13, 4094219685) + miss(out, 14, 1704130599) + miss(out, 15, 1324110832);
    return n;
}}

fn misses_a3x200() -> i64 {{
    let msg: i64 = __mind_alloc(200);
    let mut i: i64 = 0;
    while i < 200 {{
        __mind_store_i8(msg + i, 163);
        i = i + 1;
    }}
    let out: i64 = __mind_alloc(64);
    let rc: i64 = keccak_sha3_512(msg, 200, out);
    let mut n: i64 = rc;
    n = n + miss(out, 0, 3882744530) + miss(out, 1, 545564849) + miss(out, 2, 1182781231) + miss(out, 3, 4200085019);
    n = n + miss(out, 4, 3967166701) + miss(out, 5, 4126408128) + miss(out, 6, 3833617860) + miss(out, 7, 2360274088);
    n = n + miss(out, 8, 461116355) + miss(out, 9, 182407843) + miss(out, 10, 1704297587) + miss(out, 11, 2586702955);
    n = n + miss(out, 12, 3851011356) + miss(out, 13, 2711922753) + miss(out, 14, 1844859994) + miss(out, 15, 485014016);
    return n;
}}

pub fn kat_misses() -> i64 {{
    return misses_empty() + misses_abc() + misses_a3x200();
}}

#[test]
fn sha3_512_fips202_known_answers() {{
    assert kat_misses() == 0;
}}

fn main() -> i64 {{
    return kat_misses();
}}
"#
    )
}

#[cfg(feature = "cross-module-imports")]
#[test]
fn keccak_sha3_512_known_answers_check_and_pass_under_mindc_test() {
    let file = write_program("keccak_kat_test", &keccak_kat_program(false));
    let out = mindc("check", &file);
    assert!(
        out.status.success(),
        "`mindc check` must accept the std.keccak calls:\n{}",
        text(&out)
    );
    let out = mindc("test", &file);
    let all = text(&out);
    assert!(
        out.status.success() && all.contains("1 passed; 0 failed"),
        "the FIPS 202 SHA3-512 vectors must pass under `mindc test`:\n{all}"
    );

    // Negative control: one wrong expected word turns the run red.
    let file = write_program("keccak_kat_test_mutated", &keccak_kat_program(true));
    let out = mindc("test", &file);
    let all = text(&out);
    assert!(
        !out.status.success() && all.contains("0 passed; 1 failed"),
        "a wrong expected word must fail `mindc test`:\n{all}"
    );
}

/// The native build computes the same digests as the interpreter: zero
/// misses on the real vectors and exactly one on the mutated table.
#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
#[test]
fn keccak_known_answers_agree_between_the_interpreter_and_the_native_build() {
    for (case, mutate, want) in [
        ("keccak_native", false, 0),
        ("keccak_native_mutated", true, 1),
    ] {
        let dir = common::scratch_dir("std_bundled_module_table_native").join(case);
        std::fs::create_dir_all(&dir).expect("create native project dir");
        std::fs::write(
            dir.join("Mind.toml"),
            "[package]\nname = \"kat\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
             output = \"kat\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
             sources = [\"main.mind\"]\n",
        )
        .expect("write Mind.toml");
        std::fs::write(dir.join("main.mind"), keccak_kat_program(mutate)).expect("write main.mind");
        let out = Command::new(common::mindc_bin())
            .arg("build")
            .current_dir(&dir)
            .output()
            .expect("run mindc build");
        if !common::gate::compiled("std_bundled_module_table", &out) {
            return;
        }
        let run = Command::new(dir.join("target").join("debug").join("kat"))
            .output()
            .expect("run the built executable");
        assert_eq!(
            run.status.code(),
            Some(want),
            "{case}: the native build must report {want} missed digest word(s)"
        );
        let interp = mindc("test", &dir.join("main.mind"));
        assert_eq!(
            interp.status.success(),
            want == 0,
            "{case}: the interpreter and the native build disagree:\n{}",
            text(&interp)
        );
    }
}

// ── The build path links every shipped module ──────────────────────────────

/// A cpu-executable project in `dir` whose only source is `main.mind`.
#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
fn build_project(dir: &Path, main: &str) -> Output {
    std::fs::create_dir_all(dir).expect("create build project dir");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"p\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
         output = \"p\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
         sources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(dir.join("main.mind"), main).expect("write main.mind");
    Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(dir)
        .output()
        .expect("run mindc build")
}

/// Bundled modules a call into which `mindc build` cannot link yet: the
/// module, the call, and the text the failure names. Each was bundled, and
/// failed the same way, before the table became exhaustive; a module that
/// starts linking turns this test red until its row is removed.
#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
const KNOWN_UNLINKED: &[(&str, &str, &str)] = &[
    // `__mind_argc` / `__mind_argv` are not registered for the Rust/MLIR
    // path, so std.cli is not a substrate member and its symbols stay
    // undefined.
    ("std.cli", "args_new()", "undefined reference to `args_new'"),
    // The runtime-support shim defines `print_bytes`, so std.io is not a
    // member; its other functions stay undefined.
    ("std.io", "sgr_bold()", "undefined reference to `sgr_bold'"),
    // A member whose object cannot compile until `__mind_now_ns` is
    // registered (tests/std_substrate_native_link.rs, KNOWN_UNCOMPILABLE).
    ("std.time", "", "__mind_now_ns"),
];

/// Every shipped module that `mindc check` admits also links: a program that
/// imports it and calls one of its public functions builds. Before the fix 18
/// newly bundled modules checked clean and then failed in the linker with an
/// untyped `undefined reference` (`x25519`, `aes128_encrypt_block`, ...).
/// A module that declares only `extern "C"` bindings (`std.llvm`, `std.mlir`)
/// is built with the import alone: its functions are C symbols that the
/// program links like any other extern.
#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
#[test]
fn every_shipped_std_module_imports_and_a_public_function_builds() {
    let root = common::scratch_dir("std_bundled_module_table_build");
    let probe = build_project(&root.join("probe"), MAIN_ONLY.trim_start());
    if !common::gate::compiled("std_bundled_module_table_build", &probe) {
        return;
    }
    let (mut built, mut failures) = (0usize, Vec::new());
    for (module, file) in shipped_std_modules() {
        let source = std::fs::read_to_string(&file).expect("read shipped std module");
        let ast = libmind::parser::parse(&source).expect("shipped std module parses");
        let known = KNOWN_UNLINKED.iter().find(|(m, _, _)| *m == module);
        let call = match known {
            Some((_, call, _)) => (*call).to_string(),
            None => first_call(&ast.items, false).unwrap_or_default(),
        };
        let stmt = if call.is_empty() {
            String::new()
        } else {
            format!("    {call};\n")
        };
        let main = format!("import {module};\n\nfn main() -> i64 {{\n{stmt}    return 0;\n}}\n");
        let out = build_project(&root.join(module.replace('.', "_")), &main);
        let all = text(&out);
        match known {
            Some((_, _, cause)) if out.status.success() || !all.contains(cause) => failures.push(
                format!("{module} is in KNOWN_UNLINKED but no longer fails with `{cause}`:\n{all}"),
            ),
            Some(_) => {}
            None if out.status.success() => built += 1,
            None => failures.push(format!("{module}: `{call}`\n{all}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} shipped std module(s) check but do not build:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(built >= 38, "only {built} shipped modules were built");
}

/// The native backend sees only the std modules its frozen compiler was
/// minted from. An import of any other shipped module is refused at the
/// import, not accepted and then refused at each call as an unresolved
/// closure edge.
#[test]
fn the_native_backend_refuses_a_std_module_outside_its_seed_at_the_import() {
    let native = |file: &Path| {
        Command::new(common::mindc_bin())
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args(["build", "--backend", "native"])
            .arg(file)
            .arg("--out")
            .arg(file.with_extension("elf"))
            .output()
            .expect("spawn mindc build --backend native")
    };
    // Control: a seed module imports and builds.
    let control = write_program("seed_native", &format!("import std.sha256;{MAIN_ONLY}"));
    let out = native(&control);
    assert!(
        out.status.success(),
        "an import of a seed module must build natively:\n{}",
        text(&out)
    );
    let file = write_program(
        "unseeded_native",
        "import std.keccak;\n\nfn main() -> i64 {\n    return keccak_sha3_512(0, 0, 0);\n}\n",
    );
    let out = native(&file);
    let all = text(&out);
    assert!(
        !out.status.success()
            && all.contains("scope.std_module_not_in_native_seed")
            && all.contains("std.keccak")
            && !all.contains("closure.unresolved_edge"),
        "an import of a shipped module outside the native seed must be refused at the import:\n{all}"
    );
}

// ── Which modules a compile loads ──────────────────────────────────────────

/// A non-implicit module's names come in through its import only; an implicit
/// module's names still resolve with no import, as before the table grew.
#[cfg(feature = "cross-module-imports")]
#[test]
fn a_non_implicit_std_function_resolves_through_its_import_only() {
    let call = "\n\nfn probe() -> i64 {\n    return keccak_sha3_512(0, 0, 0);\n}";
    let with = write_program(
        "keccak_import",
        &format!("import std.keccak;{call}{MAIN_ONLY}"),
    );
    let out = mindc("check", &with);
    assert!(out.status.success(), "with its import:\n{}", text(&out));
    let without = write_program("keccak_no_import", &format!("{call}{MAIN_ONLY}"));
    let out = mindc("check", &without);
    let all = text(&out);
    assert!(
        !out.status.success() && all.contains("keccak_sha3_512"),
        "without its import a non-implicit std function must not resolve:\n{all}"
    );
    let implicit = write_program(
        "sha256_no_import",
        &format!("fn probe(a: i64) -> i64 {{\n    return sha256(a, 0, a);\n}}{MAIN_ONLY}"),
    );
    let out = mindc("check", &implicit);
    assert!(
        out.status.success(),
        "an implicit std function resolves with no import:\n{}",
        text(&out)
    );
}

/// spec/v1.0/stdlib.md, "use std.<name> resolution", rule 2: a project MAY
/// supply its own module at `std.<name>`. Its import is not an unknown std
/// module, though no bundled module has that path: `mindc check`, `mindc
/// test` and `mindc build` all admit it.
#[cfg(feature = "cross-module-imports")]
#[test]
fn a_std_module_the_project_supplies_is_not_an_unknown_import() {
    let main = "import std.own_extra;\n\n#[test]\nfn t() {\n    assert 1 == 1;\n}\n\n\
                fn main() -> i64 {\n    return 0;\n}\n";
    let project = |case: &str, supplied: bool| {
        let dir = common::scratch_dir("std_bundled_module_table").join(case);
        std::fs::create_dir_all(dir.join("std")).expect("create the project's std dir");
        let sources = if supplied {
            "\"main.mind\", \"std/own_extra.mind\""
        } else {
            "\"main.mind\""
        };
        std::fs::write(
            dir.join("Mind.toml"),
            format!(
                "[package]\nname = \"p\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
                 output = \"p\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
                 sources = [{sources}]\n"
            ),
        )
        .expect("write Mind.toml");
        if supplied {
            std::fs::write(
                dir.join("std").join("own_extra.mind"),
                "pub fn own_extra_value() -> i64 {\n    return 3;\n}\n",
            )
            .expect("write the project's std.own_extra");
        }
        let file = dir.join("main.mind");
        std::fs::write(&file, main).expect("write main.mind");
        file
    };
    let file = project("project_std_module", true);
    let out = mindc("check", &file);
    assert!(
        out.status.success(),
        "`mindc check` must resolve a project-supplied std.own_extra:\n{}",
        text(&out)
    );
    let out = mindc("test", &file);
    let all = text(&out);
    assert!(
        out.status.success() && all.contains("1 passed; 0 failed"),
        "`mindc test` must admit a project-supplied std.own_extra:\n{all}"
    );
    #[cfg(all(unix, feature = "mlir-build"))]
    {
        let out = Command::new(common::mindc_bin())
            .arg("build")
            .current_dir(file.parent().expect("project dir"))
            .output()
            .expect("run mindc build");
        if common::gate::compiled("std_bundled_module_table_project_std", &out) {
            assert!(out.status.success(), "`mindc build` must admit it");
        }
    }
    // Control: the same import with no project module is an unknown module.
    let out = mindc("check", &project("project_std_module_absent", false));
    let all = text(&out);
    assert!(
        !out.status.success() && all.contains("E2002") && all.contains("std.own_extra"),
        "without the project module the import must be E2002:\n{all}"
    );
}

/// spec/v1.0/stdlib.md, "Environment override": a `MIND_STDLIB_PATH`
/// directory holding the four specified files is used. Each module it holds
/// replaces the bundled one for `mindc check` and `mindc test` alike; the
/// others keep their bundled source. The directory of every module the bundle
/// carried before it became exhaustive (the implicit set) is used too.
#[cfg(feature = "cross-module-imports")]
#[test]
fn a_stdlib_override_with_the_specified_files_applies_to_check_and_test() {
    use libmind::project::stdlib::IMPLICIT_STD_MODULES;
    let marker = "\npub fn ovr_marker() -> i64 {\n    return 4242;\n}\n";
    let check = "import std.vec;\n\nfn main() -> i64 {\n    return ovr_marker();\n}\n";
    let test = "import std.vec;\n\n#[test]\nfn t() {\n    assert ovr_marker() == 4242;\n}\n";
    let run = |sub: &str, file: &Path, root: Option<&Path>| {
        let mut cmd = Command::new(common::mindc_bin());
        cmd.arg(sub)
            .arg(file.file_name().expect("file name"))
            .current_dir(file.parent().expect("program dir"));
        match root {
            Some(root) => cmd.env("MIND_STDLIB_PATH", root),
            None => cmd.env_remove("MIND_STDLIB_PATH"),
        };
        cmd.output().expect("spawn mindc")
    };
    let four: &[&str] = &["std.io", "std.map", "std.string", "std.vec"];
    for (case, modules) in [("four", four), ("implicit", IMPLICIT_STD_MODULES)] {
        let root = common::scratch_dir("std_bundled_module_table").join(format!("override_{case}"));
        std::fs::create_dir_all(&root).expect("create override dir");
        for module in modules {
            let name = module.trim_start_matches("std.");
            let mut text = std::fs::read_to_string(std_dir().join(format!("{name}.mind")))
                .expect("read std module");
            if name == "vec" {
                text.push_str(marker);
            }
            std::fs::write(root.join(format!("{name}.mind")), text).expect("write override file");
        }
        let file = write_program(&format!("override_check_{case}"), check);
        let out = run("check", &file, Some(&root));
        assert!(
            out.status.success(),
            "{case}: `mindc check` must use the override:\n{}",
            text(&out)
        );
        let out = run("check", &file, None);
        assert!(
            !out.status.success(),
            "{case}: without the override `ovr_marker` must not resolve"
        );
        let file = write_program(&format!("override_test_{case}"), test);
        let out = run("test", &file, Some(&root));
        let all = text(&out);
        assert!(
            out.status.success() && all.contains("1 passed; 0 failed"),
            "{case}: `mindc test` must use the override:\n{all}"
        );
    }
}
