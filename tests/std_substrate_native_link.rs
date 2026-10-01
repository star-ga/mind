// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Every substrate `std` module links into a native executable, and linking
//! them never breaks a program that built before.
//!
//! A project that imported `std.regex`, `std.net`, `std.json`, `std.toml` and
//! other bundled modules type-checked and then failed the native link with
//! "undefined reference to `rx_kind_match`" and the like: only seven modules
//! were compiled and linked, and the executable path kept its own copy of that
//! list. Both paths now read `substrate_link::SUBSTRATE_MODULES`.
//!
//! Linking more std code must not collide with the program's own names, so the
//! substrate objects go into one archive (a module is linked only when the
//! program references it) and a non-`pub` fn that shares a name with another
//! module of the build gets internal linkage on either side. The collision
//! cases below built before the list grew and must keep building.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::path::Path;
use std::process::{Command, Output};

use libmind::project::substrate_link::SUBSTRATE_MODULES;

/// Members whose object cannot be compiled yet, and the diagnostic that says
/// why. `std.time` reads the clock through `__mind_now_ns`, which the
/// Rust/MLIR intrinsic table does not register.
const KNOWN_UNCOMPILABLE: &[(&str, &str)] = &[("std.time", "__mind_now_ns")];

/// Write a project with `files` (the first is the entry), build it, and return
/// the build output.
fn build(dir: &Path, files: &[(&str, &str)]) -> Output {
    build_with(dir, files, "")
}

/// [`build`] with `manifest_tail` appended to the generated `Mind.toml`.
fn build_with(dir: &Path, files: &[(&str, &str)], manifest_tail: &str) -> Output {
    write_project(dir, files, manifest_tail);
    Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(dir)
        .output()
        .expect("run mindc build")
}

/// Write `Mind.toml` (entry = the first of `files`, all of them as sources)
/// and the sources.
fn write_project(dir: &Path, files: &[(&str, &str)], manifest_tail: &str) {
    let sources: Vec<String> = files.iter().map(|(n, _)| format!("\"{n}\"")).collect();
    std::fs::write(
        dir.join("Mind.toml"),
        format!(
            "[package]\nname = \"sub\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"{}\"\noutput = \"sub\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [{}]\n{manifest_tail}",
            files[0].0,
            sources.join(", ")
        ),
    )
    .expect("write Mind.toml");
    for (name, text) in files {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create source dir");
        }
        std::fs::write(path, text).expect("write source");
    }
}

/// Build and run a project; `None` when the build was routed through the
/// capability-skip gate.
fn build_and_run(target: &str, files: &[(&str, &str)]) -> Option<i32> {
    let dir = common::scratch_dir(target);
    let out = build(&dir, files);
    if !common::gate::compiled(target, &out) {
        return None;
    }
    let run = Command::new(dir.join("target").join("debug").join("sub"))
        .output()
        .expect("run the built executable");
    Some(run.status.code().expect("exit code"))
}

#[test]
fn every_substrate_module_compiles_and_links_into_one_executable() {
    let imports: String = SUBSTRATE_MODULES
        .iter()
        .filter(|m| !KNOWN_UNCOMPILABLE.iter().any(|(k, _)| k == *m))
        .map(|m| format!("import {m};\n"))
        .collect();
    // Calls into modules that used to stay undefined at link; the exit code
    // checks two of the values (af_inet = 2, TIOCGWINSZ = 21523).
    let main = format!(
        "{imports}\nfn main() -> i64 {{\n    let mut r: i64 = net.af_inet();\n    if tui.tui_tiocgwinsz() == 21523 {{\n        r = r + 40;\n    }}\n    if regex.rx_kind_match() != 0 {{\n        r = 1;\n    }}\n    let d: i64 = json.jv_max_depth() + toml.toml_max_depth();\n    if d <= 0 {{\n        r = 2;\n    }}\n    return r;\n}}\n"
    );
    let Some(code) = build_and_run("std_substrate_native_link", &[("main.mind", &main)]) else {
        return;
    };
    assert_eq!(
        code, 42,
        "the linked substrate functions must return their values"
    );
}

#[test]
fn an_imported_but_unused_module_does_not_collide_with_a_program_fn() {
    // std.async defines `pub fn run`; the program imports the module without
    // calling it and defines its own `run`. The archive leaves async unlinked.
    let main = "import std.async;\n\npub fn run(a: i64) -> i64 {\n    return a + 11;\n}\n\nfn main() -> i64 {\n    return run(1);\n}\n";
    let Some(code) = build_and_run("std_substrate_unused_import", &[("main.mind", main)]) else {
        return;
    };
    assert_eq!(code, 12);
}

#[test]
fn a_private_helper_sharing_a_name_with_a_used_module_still_links() {
    // std.toml has a private `is_digit` and std.json a `pub fn push`; the
    // program uses both modules and defines its own private `is_digit` and
    // `push`. Each colliding non-`pub` fn gets internal linkage.
    let main = "import std.toml;\nimport std.json;\n\nfn is_digit(c: i64) -> i64 {\n    return c + 1;\n}\n\nfn push(a: i64) -> i64 {\n    return a * 2;\n}\n\nfn main() -> i64 {\n    let d: i64 = toml.toml_max_depth() + json.jv_max_depth();\n    if d <= 0 {\n        return 1;\n    }\n    return push(is_digit(4));\n}\n";
    let Some(code) = build_and_run("std_substrate_private_collision", &[("main.mind", main)])
    else {
        return;
    };
    assert_eq!(
        code, 10,
        "the program's own `is_digit` and `push` must be the ones called"
    );
}

#[test]
fn a_project_std_module_shadows_the_bundled_one() {
    // The project supplies `std/regex.mind`; it wins in the module table, and
    // the bundled std.regex must not be linked next to it.
    let shadow = "pub fn rx_kind_match() -> i64 {\n    return 7;\n}\n";
    let main = "import std.regex;\n\nfn main() -> i64 {\n    return regex.rx_kind_match();\n}\n";
    let Some(code) = build_and_run(
        "std_substrate_shadow",
        &[("main.mind", main), ("std/regex.mind", shadow)],
    ) else {
        return;
    };
    assert_eq!(
        code, 7,
        "the project's own std.regex must be the one linked"
    );
    let bundled = common::scratch_dir("std_substrate_shadow")
        .join("target")
        .join("obj")
        .join("__std_regex.o");
    assert!(
        !bundled.exists(),
        "the bundled std.regex must not be compiled when the project supplies its own"
    );
}

#[test]
fn a_program_helper_named_like_a_std_dependency_stays_local() {
    // std.toml's object calls `vec_push`, which the runtime shim defines. A
    // program's own non-`pub` `vec_push` must not stand in for it, so it gets
    // internal linkage: never a global `T` symbol (a local one may be inlined
    // away entirely).
    let main = "import std.toml;\n\nfn vec_push(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n\nfn main() -> i64 {\n    let d: i64 = toml.toml_max_depth();\n    if d <= 0 {\n        return 1;\n    }\n    return vec_push(3, 4);\n}\n";
    let Some(code) = build_and_run("std_substrate_dependency_name", &[("main.mind", main)]) else {
        return;
    };
    assert_eq!(code, 7);
    let obj = common::scratch_dir("std_substrate_dependency_name")
        .join("target")
        .join("obj")
        .join("main.o");
    let nm = Command::new("nm").arg(&obj).output().expect("run nm");
    let symbols = String::from_utf8_lossy(&nm.stdout);
    assert!(
        !symbols.lines().any(|l| l.ends_with(" T vec_push")),
        "the program's vec_push must not be a global symbol; nm:\n{symbols}"
    );
}

#[test]
fn emit_shared_links_the_substrate_archive() {
    let dir = common::scratch_dir("std_substrate_emit_shared");
    let src = dir.join("lib.mind");
    std::fs::write(
        &src,
        "import std.toml;\n\nfn is_digit(c: i64) -> i64 {\n    return c;\n}\n\npub fn depth() -> i64 {\n    return toml.toml_max_depth() + is_digit(0);\n}\n",
    )
    .expect("write source");
    let so = dir.join("lib.so");
    let out = Command::new(common::mindc_bin())
        .arg(&src)
        .arg("--emit-shared")
        .arg(&so)
        .output()
        .expect("run mindc --emit-shared");
    if !common::gate::compiled("std_substrate_emit_shared", &out) {
        return;
    }
    assert!(so.exists(), "the shared library must be written");
    assert!(
        !undefined_symbols(&so).contains(&"toml_max_depth".to_string()),
        "std.toml must be linked into the shared library, not left undefined"
    );
    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .expect("read scratch dir")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("__std_"))
        .map(|e| e.file_name())
        .collect();
    assert!(
        leftovers.is_empty(),
        "substrate objects must be built in a private directory, not next to the output: {leftovers:?}"
    );
}

#[test]
fn known_uncompilable_members_still_fail_with_their_diagnostic() {
    for (module, cause) in KNOWN_UNCOMPILABLE {
        assert!(
            SUBSTRATE_MODULES.contains(module),
            "{module} is listed as a known exception but is not a member"
        );
        let short = module.rsplit('.').next().unwrap_or(module);
        let dir = common::scratch_dir(&format!("std_substrate_known_{short}"));
        // A plain program first: on a host without the MLIR toolchain it is
        // routed through the capability gate, and this check is skipped.
        let probe = build(
            &dir,
            &[("main.mind", "fn main() -> i64 {\n    return 0;\n}\n")],
        );
        if !common::gate::compiled("std_substrate_known_uncompilable", &probe) {
            return;
        }
        let main = format!("import {module};\n\nfn main() -> i64 {{\n    return 0;\n}}\n");
        let out = build(&dir, &[("main.mind", &main)]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success() && stderr.contains(cause),
            "{module} now builds (or fails for another reason); remove it from \
             KNOWN_UNCOMPILABLE. status: {:?}, stderr: {stderr}",
            out.status.code()
        );
    }
}

/// The undefined dynamic symbols of a shared library (`nm -D --undefined-only`).
fn undefined_symbols(so: &Path) -> Vec<String> {
    let out = Command::new("nm")
        .args(["-D", "--undefined-only"])
        .arg(so)
        .output()
        .expect("run nm");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_whitespace().last().map(str::to_string))
        .collect()
}

/// The dynamic symbol table of a shared library (`nm -D`).
fn dynamic_symbols(so: &Path) -> String {
    let out = Command::new("nm")
        .arg("-D")
        .arg(so)
        .output()
        .expect("run nm");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Build a manifest cdylib from `src/<name>` files with `[exports] c_abi`.
fn build_cdylib(
    target: &str,
    c_abi: &[&str],
    files: &[(&str, &str)],
) -> Option<std::path::PathBuf> {
    let root = common::scratch_dir(target);
    let _ = std::fs::remove_dir_all(root.join("src"));
    let exports: Vec<String> = c_abi.iter().map(|n| format!("\"{n}\"")).collect();
    std::fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"lib\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\noutput = \"lib\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\n[exports]\nc_abi = [{}]\n",
            exports.join(", ")
        ),
    )
    .expect("write Mind.toml");
    for (name, text) in files {
        let path = root.join("src").join(name);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create src dir");
        std::fs::write(path, text).expect("write source");
    }
    let so = root.join("lib.so");
    let out = Command::new(common::mindc_bin())
        .current_dir(&root)
        .args(["build", "--emit", "cdylib", "--no-cache", "--out"])
        .arg(&so)
        .output()
        .expect("run mindc build --emit cdylib");
    if !common::gate::compiled(target, &out) {
        return None;
    }
    Some(so)
}

#[test]
fn a_program_fn_named_like_a_libc_call_in_std_still_builds() {
    // std.fs declares libc `read(fd, buf, n)` in an `extern "C"` block. The
    // program's own one-argument `read` must not stand in for it while std.fs
    // is type-checked (it did, failing std.fs with E2005).
    let main = "import std.fs;\n\nfn read(a: i64) -> i64 {\n    return a + 1;\n}\n\nfn main() -> i64 {\n    return read(41);\n}\n";
    let Some(code) = build_and_run("std_substrate_libc_name", &[("main.mind", main)]) else {
        return;
    };
    assert_eq!(code, 42);
}

#[test]
fn a_c_abi_export_colliding_with_a_std_helper_stays_exported() {
    // `is_digit` is listed in [exports] c_abi and also a private helper of the
    // linked std.toml: the program's export must stay global (std's yields).
    let main = "import std.toml;\n\nfn is_digit(c: i64) -> i64 {\n    return c + 1000;\n}\n\npub fn go() -> i64 {\n    return toml.toml_max_depth() + is_digit(0);\n}\n";
    let Some(so) = build_cdylib(
        "std_substrate_c_abi",
        &["is_digit", "go"],
        &[("main.mind", main)],
    ) else {
        return;
    };
    assert!(
        dynamic_symbols(&so)
            .lines()
            .any(|l| l.ends_with(" T is_digit")),
        "the c_abi export must stay a global symbol; nm -D:\n{}",
        dynamic_symbols(&so)
    );
}

#[test]
fn a_sibling_helper_named_like_a_std_dependency_stays_local_in_a_cdylib() {
    // std.toml calls the runtime's `vec_push`; a sibling module's own private
    // `vec_push` must not be exported in its place.
    let main = "import util;\nimport std.toml;\n\npub fn go() -> i64 {\n    return util.u() + toml.toml_max_depth();\n}\n";
    let util = "fn vec_push(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n\npub fn u() -> i64 {\n    return vec_push(1, 2);\n}\n";
    let Some(so) = build_cdylib(
        "std_substrate_cdylib_sibling",
        &["go"],
        &[("main.mind", main), ("util.mind", util)],
    ) else {
        return;
    };
    assert!(
        !dynamic_symbols(&so)
            .lines()
            .any(|l| l.ends_with(" T vec_push")),
        "the sibling's vec_push must not be a global definition; nm -D:\n{}",
        dynamic_symbols(&so)
    );
}

#[test]
fn a_project_std_copy_leaves_no_undefined_symbol_in_a_cdylib() {
    // The cdylib links siblings by import from the entry, so a project's own
    // `std/regex.mind` is not linked there; the bundled module must still be.
    let main = "import std.regex;\n\npub fn go() -> i64 {\n    return regex.rx_kind_match();\n}\n";
    let shadow = "pub fn rx_kind_match() -> i64 {\n    return 7;\n}\n";
    let Some(so) = build_cdylib(
        "std_substrate_cdylib_shadow",
        &["go"],
        &[("main.mind", main), ("std/regex.mind", shadow)],
    ) else {
        return;
    };
    assert!(
        !undefined_symbols(&so).contains(&"rx_kind_match".to_string()),
        "rx_kind_match must not be left undefined"
    );
}

#[test]
fn a_native_source_can_call_into_a_linked_std_module() {
    // A `native_sources` C object that calls a std.regex function: the std
    // archive must come after it on the link line, or that reference is left
    // undefined (an archive only serves objects before it).
    let dir = common::scratch_dir("std_substrate_native_source");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"sub\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"sub\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\nnative_sources = [\"shim.c\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(
        dir.join("main.mind"),
        "import std.regex;\n\nfn main() -> i64 {\n    return __mind_nerve_lut_exp_h();\n}\n",
    )
    .expect("write main.mind");
    std::fs::write(
        dir.join("shim.c"),
        "long rx_kind_char(void);\nlong __mind_nerve_lut_exp_h(void) { return rx_kind_char() + 40; }\n",
    )
    .expect("write shim.c");
    let out = Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(&dir)
        .output()
        .expect("run mindc build");
    if !common::gate::compiled("std_substrate_native_source", &out) {
        return;
    }
    let run = Command::new(dir.join("target").join("debug").join("sub"))
        .output()
        .expect("run the built executable");
    assert_eq!(run.status.code(), Some(42), "rx_kind_char() + 40");
}

/// Flat `mindc <entry> --emit-shared` of the project `files` (the first is the
/// entry); the shared library, or `None` when the build was routed through the
/// capability-skip gate.
fn emit_shared(target: &str, files: &[(&str, &str)]) -> Option<std::path::PathBuf> {
    let dir = common::scratch_dir(target);
    write_project(&dir, files, "");
    let so = dir.join("lib.so");
    let out = Command::new(common::mindc_bin())
        .arg(dir.join(files[0].0))
        .arg("--emit-shared")
        .arg(&so)
        .output()
        .expect("run mindc --emit-shared");
    if !common::gate::compiled(target, &out) {
        return None;
    }
    assert!(so.exists(), "the shared library must be written");
    Some(so)
}

fn exports_text_symbol(so: &Path, name: &str) -> bool {
    dynamic_symbols(so)
        .lines()
        .any(|l| l.ends_with(&format!(" T {name}")))
}

#[test]
fn a_shared_library_keeps_a_fn_named_like_a_std_helper() {
    // std.toml has a private `is_digit`, std.json a `pub fn push`. The
    // library's own non-`pub` fns of those names stay exported: std.toml's
    // helper is the one made internal, and std.json is never linked because
    // nothing calls into it.
    let cases = [
        (
            "std_substrate_shared_helper_unused",
            "import std.toml;\n\nfn is_digit(c: i64) -> i64 {\n    return c + 1000;\n}\n\npub fn go() -> i64 {\n    return is_digit(1);\n}\n",
            "is_digit",
        ),
        (
            "std_substrate_shared_helper_used",
            "import std.toml;\n\nfn is_digit(c: i64) -> i64 {\n    return c + 1000;\n}\n\npub fn go() -> i64 {\n    return toml.toml_max_depth() + is_digit(1);\n}\n",
            "is_digit",
        ),
        (
            "std_substrate_shared_pub_unused",
            "import std.json;\n\nfn push(c: i64) -> i64 {\n    return c * 2;\n}\n\npub fn go() -> i64 {\n    return push(5);\n}\n",
            "push",
        ),
    ];
    for (target, lib, name) in cases {
        let Some(so) = emit_shared(target, &[("lib.mind", lib)]) else {
            return;
        };
        assert!(
            exports_text_symbol(&so, name),
            "{target}: `{name}` must stay exported; nm -D:\n{}",
            dynamic_symbols(&so)
        );
    }
}

#[test]
fn a_native_source_can_call_a_program_fn_named_like_a_std_helper() {
    // The C object calls the program's private `is_digit`; std.toml, imported
    // but unused, has a private helper of the same name.
    let dir = common::scratch_dir("std_substrate_native_calls_program");
    std::fs::write(
        dir.join("shim.c"),
        "long is_digit(long);\nlong __mind_nerve_lut_exp_h(void) { return is_digit(41); }\n",
    )
    .expect("write shim.c");
    let main = "import std.toml;\n\nfn is_digit(c: i64) -> i64 {\n    return c + 1;\n}\n\nfn main() -> i64 {\n    return __mind_nerve_lut_exp_h();\n}\n";
    let out = build_with(
        &dir,
        &[("main.mind", main)],
        "native_sources = [\"shim.c\"]\n",
    );
    if !common::gate::compiled("std_substrate_native_calls_program", &out) {
        return;
    }
    let run = Command::new(dir.join("target").join("debug").join("sub"))
        .output()
        .expect("run the built executable");
    assert_eq!(run.status.code(), Some(42), "is_digit(41)");
}

/// main imports `util` and std.sha256; `other.mind` is a declared source that
/// main never imports. It imports std.time (which cannot compile yet) and
/// defines its own private `helper`.
const LINKED_MAIN: &str = "import util;\nimport std.sha256;\n\nfn helper(x: i64) -> i64 {\n    return x + 7;\n}\n\npub fn go() -> i64 {\n    return helper(util.one());\n}\n";
const LINKED_UTIL: &str = "pub fn one() -> i64 {\n    return 1;\n}\n";
const UNLINKED_OTHER: &str = "import std.time;\n\nfn helper(x: i64) -> i64 {\n    return x;\n}\n\npub fn other() -> i64 {\n    return helper(2);\n}\n";

#[test]
fn an_unlinked_project_source_does_not_join_a_shared_library() {
    // Neither its std imports nor its fn names concern the `.so`: the build
    // succeeds and the entry's `helper` stays exported.
    let Some(so) = emit_shared(
        "std_substrate_unlinked_flat",
        &[
            ("main.mind", LINKED_MAIN),
            ("util.mind", LINKED_UTIL),
            ("other.mind", UNLINKED_OTHER),
        ],
    ) else {
        return;
    };
    assert!(
        exports_text_symbol(&so, "helper"),
        "nm -D:\n{}",
        dynamic_symbols(&so)
    );
    let Some(so) = build_cdylib(
        "std_substrate_unlinked_cdylib",
        &["go"],
        &[
            ("main.mind", LINKED_MAIN),
            ("util.mind", LINKED_UTIL),
            ("other.mind", UNLINKED_OTHER),
        ],
    ) else {
        return;
    };
    assert!(
        exports_text_symbol(&so, "helper"),
        "nm -D:\n{}",
        dynamic_symbols(&so)
    );
}

#[test]
fn export_lists_do_not_keep_colliding_fns_global_in_an_executable() {
    // Three private `helper`s. An executable has no exported surface, so
    // neither `[exports] c_abi` nor `export { }` may keep one of them global:
    // all three get internal linkage and the program links.
    let a = "fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n\npub fn fa(x: i64) -> i64 {\n    return helper(x);\n}\n";
    let b = "fn helper(x: i64) -> i64 {\n    return x * 10;\n}\n\npub fn fb(x: i64) -> i64 {\n    return helper(x);\n}\n";
    let main = "import a;\nimport b;\n\nfn helper(x: i64) -> i64 {\n    return x + 100;\n}\n\nfn main() -> i64 {\n    return helper(0) + a.fa(1) + b.fb(10);\n}\n";
    let listed_a = format!("export {{ helper, fa }}\n\n{a}");
    let mut cases: Vec<(&str, &str, &str)> = vec![("std_substrate_exe_export_list", &listed_a, "")];
    // `ffi-c-user` emits a `mind_fn_<name>_v1_invoke` C-ABI wrapper for each
    // `[exports] c_abi` name in every module that defines it, so three
    // `helper`s give three global wrappers whatever the fns' linkage; that
    // build can only export a name one module defines.
    if !cfg!(feature = "ffi-c-user") {
        cases.push((
            "std_substrate_exe_c_abi",
            a,
            "\n[exports]\nc_abi = [\"helper\"]\n",
        ));
    }
    for (target, a_src, tail) in cases {
        let dir = common::scratch_dir(target);
        let out = build_with(
            &dir,
            &[("main.mind", main), ("a.mind", a_src), ("b.mind", b)],
            tail,
        );
        if !common::gate::compiled(target, &out) {
            return;
        }
        let run = Command::new(dir.join("target").join("debug").join("sub"))
            .output()
            .expect("run the built executable");
        // 100 + (1 + 1) + 10 * 10
        assert_eq!(run.status.code(), Some(202), "{target}");
    }
}
