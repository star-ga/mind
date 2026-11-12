// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Which bundled std modules a compile loads, and from which source text.
//!
//! The bundled table is exhaustive over `std/*.mind`, but a compile parses
//! only what it can use: the [`IMPLICIT_STD_MODULES`] (whose names resolve
//! with no import) plus every module the program imports, closed over the
//! imports of the loaded std modules. Parsing the whole table on every
//! compile made an import-free warm `--emit-shared` 6% slower and put the
//! names of every new std module into programs that never asked for them.
//!
//! The source of each module is the bundled text, unless `MIND_STDLIB_PATH`
//! names an override directory (spec/v1.0/stdlib.md, "Environment override"):
//! the directory must hold the four specified files (`vec`, `string`, `map`,
//! `io`) and every bundled module file it holds must parse; each module it
//! supplies replaces the bundled text, every other module keeps its bundled
//! text. A set but unusable value (not a directory, a required file missing,
//! a module file unreadable or not parsing) is an error that names the path
//! and the missing file or the parse error. It never falls back to the
//! bundled text: a build against a std the user did not ask for is a silent
//! wrong result. Unset uses the bundled text.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::OsString;
use std::path::Path;
use std::sync::{Mutex, OnceLock, PoisonError};

use super::STDLIB_MIND_SOURCES;
use crate::ast::{Module, Node, Span};

/// The standard-library modules whose names resolve with no import: the
/// modules the bundle carried before it became exhaustive over `std/*.mind`.
/// A program that called one of them without an import keeps compiling.
/// Every other bundled module is loaded only for a program that imports it,
/// directly or through another std module. This set is closed under std
/// imports (a unit test checks it), so loading it never pulls in another
/// module.
pub const IMPLICIT_STD_MODULES: &[&str] = &[
    "std.arena",
    "std.async",
    "std.blas",
    "std.cli",
    "std.fs",
    "std.http",
    "std.io",
    "std.io_canon",
    "std.iouring",
    "std.json",
    "std.json_canon",
    "std.map",
    "std.net",
    "std.process",
    "std.reactor",
    "std.regex",
    "std.ring",
    "std.sha256",
    "std.sha512",
    "std.string",
    "std.time",
    "std.toml",
    "std.tui",
    "std.vec",
];

/// The files spec/v1.0/stdlib.md ("Environment override") requires an
/// override directory to contain.
const OVERRIDE_REQUIRED: [&str; 4] = ["std.io", "std.map", "std.string", "std.vec"];

const STDLIB_COUNT: usize = STDLIB_MIND_SOURCES.len();

type SourceTable = &'static [(&'static str, &'static str)];

/// `(module path, source)` for every bundled module as this process sees it:
/// the bundled text, or the override directory's file for each module it
/// supplies. Every reader of std source text goes through here, so `check`,
/// `test` and the linked substrate objects see the same code.
///
/// A set but unusable `MIND_STDLIB_PATH` is a configuration error: this
/// prints it and exits with status 2 before any compile reads std text.
pub fn std_sources() -> SourceTable {
    match override_sources() {
        Ok(table) => table.unwrap_or(STDLIB_MIND_SOURCES),
        Err(message) => {
            eprintln!("error: {message}");
            std::process::exit(2)
        }
    }
}

/// Why the directory `root` is not a usable `MIND_STDLIB_PATH` override, or
/// `None` when it is usable. Reads and parses the directory on every call.
pub fn stdlib_override_error(root: &Path) -> Option<String> {
    load_override(root).err()
}

/// The source of one bundled module (see [`std_sources`]).
pub fn std_source(module: &str) -> Option<&'static str> {
    std_sources()
        .iter()
        .find(|(name, _)| *name == module)
        .map(|(_, source)| *source)
}

/// The override table for the current `MIND_STDLIB_PATH` (`Ok(None)` when it
/// is unset), read and validated once per distinct value. The texts live for
/// the process: every table entry is `&'static`, whether bundled or read from
/// the override.
fn override_sources() -> Result<Option<SourceTable>, String> {
    type Seen = Vec<(OsString, Result<SourceTable, String>)>;
    static SEEN: Mutex<Seen> = Mutex::new(Vec::new());
    let Some(root) = std::env::var_os("MIND_STDLIB_PATH") else {
        return Ok(None);
    };
    let mut seen = SEEN.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, table)) = seen.iter().find(|(value, _)| *value == root) {
        return table.clone().map(Some);
    }
    let table: Result<SourceTable, String> =
        load_override(Path::new(&root)).map(|rows| &*Box::leak(rows.into_boxed_slice()));
    seen.push((root, table.clone()));
    table.map(Some)
}

/// Read an override directory, or the reason it is not usable: not a
/// directory, a required file missing, a module file unreadable or not
/// parsing. The reason names the path and the file.
fn load_override(root: &Path) -> Result<Vec<(&'static str, &'static str)>, String> {
    let shown = root.display();
    if !root.is_dir() {
        return Err(format!(
            "MIND_STDLIB_PATH `{shown}` is not a directory (unset it to use the bundled std)"
        ));
    }
    let mut supplied: Vec<Option<String>> = Vec::with_capacity(STDLIB_COUNT);
    for (module, _) in STDLIB_MIND_SOURCES {
        let name = format!("{}.mind", module.trim_start_matches("std."));
        let file = root.join(&name);
        match std::fs::read_to_string(&file) {
            Ok(text) => {
                if let Err(errors) = crate::parser::parse(&text) {
                    let first = errors.first().map(ToString::to_string).unwrap_or_default();
                    return Err(format!(
                        "MIND_STDLIB_PATH `{shown}`: `{name}` ({module}) does not parse: {first}"
                    ));
                }
                supplied.push(Some(text));
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::NotFound
                    && !OVERRIDE_REQUIRED.contains(module) =>
            {
                supplied.push(None);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(format!(
                    "MIND_STDLIB_PATH `{shown}` lacks `{name}` ({module}): an override \
                     directory must hold io.mind, map.mind, string.mind and vec.mind"
                ));
            }
            Err(e) => {
                return Err(format!(
                    "MIND_STDLIB_PATH `{shown}`: cannot read `{name}` ({module}): {e}"
                ));
            }
        }
    }
    Ok(STDLIB_MIND_SOURCES
        .iter()
        .zip(supplied)
        .map(|(&(module, bundled), text)| {
            let source: &'static str = match text {
                Some(text) => Box::leak(text.into_boxed_str()),
                None => bundled,
            };
            (module, source)
        })
        .collect())
}

/// The `std.<m>` import paths `module` declares at module scope (top level
/// and transparent `module NAME { ... }` blocks), dotted. A `std` import
/// anywhere else is refused (see [`misplaced_std_imports`]).
pub fn std_import_roots(module: &Module) -> Vec<String> {
    fn walk(items: &[Node], out: &mut Vec<String>) {
        for item in items {
            match item {
                Node::Import { path, .. } if path.first().is_some_and(|s| s == "std") => {
                    out.push(path.join("."));
                }
                Node::Block { stmts, .. } => walk(stmts, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(&module.items, &mut out);
    out
}

/// Every `std` import of `module` that is not a module item: one inside a
/// function, closure or method body, or inside a statement. The grammar
/// (spec/v1.0/grammar-syntax.ebnf) admits an import only as a `ModuleItem`,
/// so [`std_import_roots`] and every other reader of a program's std imports
/// see module scope only. Such an import bound nothing, and every call into
/// it was refused far from the cause; `mindc check` and `mindc test` refuse
/// it at the import instead.
pub fn misplaced_std_imports(module: &Module) -> Vec<(&[String], Span)> {
    fn module_scope<'a>(items: &'a [Node], out: &mut Vec<(&'a [String], Span)>) {
        for item in items {
            match item {
                Node::Import { .. } => {}
                Node::Block { stmts, .. } => module_scope(stmts, out),
                body => crate::type_checker::nerve_walk::walk(body, &mut |node| match node {
                    Node::Import { path, span, .. } if path.first().is_some_and(|s| s == "std") => {
                        out.push((path, *span));
                    }
                    _ => {}
                }),
            }
        }
    }
    let mut out = Vec::new();
    module_scope(&module.items, &mut out);
    out
}

/// The refusal of a `std` import that is not a module item (see
/// [`misplaced_std_imports`]).
pub fn misplaced_std_import(path: &[String]) -> String {
    format!(
        "`import {}` is not a module item: an import belongs at the top level of the file \
         or in a `module` block, never inside a function or statement body, where it binds \
         nothing (spec/v1.0/grammar-syntax.ebnf, `ModuleItem`)",
        path.join(".")
    )
}

/// Parse [`IMPLICIT_STD_MODULES`] plus `roots` and every bundled module they
/// import, transitively, into `(module path, AST)` pairs in table order (the
/// module table's deterministic-insertion contract). Each module is parsed
/// once per call; one that does not parse contributes nothing, so its import
/// stays unresolved at the use site rather than crashing the loader.
pub fn parsed_std_modules(roots: impl IntoIterator<Item = String>) -> Vec<(String, Module)> {
    let sources = std_sources();
    let mut queue: VecDeque<String> = IMPLICIT_STD_MODULES
        .iter()
        .map(|module| (*module).to_string())
        .chain(roots)
        .collect();
    let mut seen = BTreeSet::new();
    let mut loaded: BTreeMap<usize, (String, Module)> = BTreeMap::new();
    while let Some(module) = queue.pop_front() {
        let Some(index) = sources.iter().position(|(name, _)| *name == module) else {
            continue;
        };
        if !seen.insert(index) {
            continue;
        }
        if let Ok(ast) = crate::parser::parse(sources[index].1) {
            queue.extend(std_import_roots(&ast));
            loaded.insert(index, (module, ast));
        }
    }
    loaded.into_values().collect()
}

/// [`parsed_std_modules`] for the imports of `programs`: the std modules a
/// compile of those sources loads.
pub fn parsed_stdlib_modules_for<'a>(
    programs: impl IntoIterator<Item = &'a Module>,
) -> Vec<(String, Module)> {
    parsed_std_modules(programs.into_iter().flat_map(std_import_roots))
}

/// The AST of bundled module `module`, parsed once per process on first use.
/// For caches of name-level facts (declared names, imports) that already
/// outlive one compile; a module table is built from [`parsed_std_modules`].
pub fn bundled_module_ast(module: &str) -> Option<&'static Module> {
    static CACHE: [OnceLock<Option<Module>>; STDLIB_COUNT] =
        [const { OnceLock::new() }; STDLIB_COUNT];
    let index = STDLIB_MIND_SOURCES
        .iter()
        .position(|(name, _)| *name == module)?;
    CACHE[index]
        .get_or_init(|| std_source(module).and_then(|source| crate::parser::parse(source).ok()))
        .as_ref()
}

/// The bundled modules `module` loads beyond [`IMPLICIT_STD_MODULES`]: the
/// non-implicit modules it imports and, transitively, the non-implicit
/// modules those import. Empty (and nothing parsed) for a program that
/// imports no such module.
pub fn imported_std_extras(module: &Module) -> BTreeSet<&'static str> {
    let mut queue: VecDeque<String> = std_import_roots(module)
        .into_iter()
        .filter(|path| !IMPLICIT_STD_MODULES.contains(&path.as_str()))
        .collect();
    let mut out = BTreeSet::new();
    while let Some(path) = queue.pop_front() {
        let Some(&(name, _)) = STDLIB_MIND_SOURCES.iter().find(|(n, _)| *n == path) else {
            continue;
        };
        if IMPLICIT_STD_MODULES.contains(&name) || !out.insert(name) {
            continue;
        }
        if let Some(ast) = bundled_module_ast(name) {
            queue.extend(std_import_roots(ast));
        }
    }
    out
}

/// True when bundled module `name` is loaded for a program with these
/// `extras` (see [`imported_std_extras`]).
pub fn is_loaded(name: &str, extras: &BTreeSet<&'static str>) -> bool {
    IMPLICIT_STD_MODULES.contains(&name) || extras.contains(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(src: &str) -> Module {
        crate::parser::parse(src).expect("test program parses")
    }

    fn loaded_names(src: &str) -> Vec<String> {
        parsed_stdlib_modules_for([&program(src)])
            .into_iter()
            .map(|(path, _)| path)
            .collect()
    }

    #[test]
    fn every_implicit_module_is_bundled_and_the_set_is_closed_under_imports() {
        for module in IMPLICIT_STD_MODULES {
            let ast = bundled_module_ast(module)
                .unwrap_or_else(|| panic!("{module} is implicit but not a parsing bundled module"));
            for import in std_import_roots(ast) {
                assert!(
                    IMPLICIT_STD_MODULES.contains(&import.as_str()),
                    "{module} imports {import}, which is not implicit: an import-free \
                     compile would load it"
                );
            }
        }
    }

    #[test]
    fn an_import_free_program_loads_only_the_implicit_modules() {
        let names = loaded_names("fn main() -> i64 {\n    return 0;\n}\n");
        let want: Vec<String> = IMPLICIT_STD_MODULES.iter().map(|m| m.to_string()).collect();
        assert_eq!(names, want);
        assert!(!names.iter().any(|m| m == "std.keccak"));
    }

    #[test]
    fn an_import_loads_the_module_and_its_std_imports_in_table_order() {
        let names =
            loaded_names("import std.tls13_handshake;\nfn main() -> i64 {\n    return 0;\n}\n");
        for module in [
            "std.tls13_handshake",
            "std.hkdf",
            "std.x509_bignum",
            "std.tls13_keyschedule",
            "std.tls13_finished",
            "std.rsa_pss",
        ] {
            assert!(
                names.iter().any(|m| m == module),
                "{module} not loaded: {names:?}"
            );
        }
        assert!(!names.iter().any(|m| m == "std.keccak"), "{names:?}");
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "loaded modules must keep table order");

        let extras = imported_std_extras(&program("import std.tls13_handshake;\n"));
        assert_eq!(extras.len(), 6, "{extras:?}");
        assert!(
            !extras.contains("std.sha256"),
            "sha256 is implicit: {extras:?}"
        );
    }

    #[test]
    fn only_a_std_import_outside_module_scope_is_misplaced() {
        let ast = program(
            "import std.keccak;\nmodule inner {\n    import std.x25519;\n}\n\
             fn f() -> i64 {\n    import std.hkdf;\n    if 1 == 1 {\n        import std.x509;\n    }\n    \
             import helper;\n    return 0;\n}\n\
             struct Foo {\n    v: i64,\n}\n\
             impl Show for Foo {\n    fn show(self) -> i64 {\n        import std.mlkem768;\n        \
             return 0;\n    }\n}\n",
        );
        let misplaced: Vec<String> = misplaced_std_imports(&ast)
            .iter()
            .map(|(path, _)| path.join("."))
            .collect();
        assert_eq!(misplaced, ["std.hkdf", "std.x509", "std.mlkem768"]);
        assert_eq!(std_import_roots(&ast), ["std.keccak", "std.x25519"]);
    }

    #[test]
    fn an_override_directory_with_the_four_specified_files_is_used() {
        let dir =
            std::env::temp_dir().join(format!("mind-std-override-four-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create override dir");
        for module in OVERRIDE_REQUIRED {
            let file = module.trim_start_matches("std.");
            let bundled = STDLIB_MIND_SOURCES
                .iter()
                .find(|(n, _)| *n == module)
                .map(|(_, s)| *s)
                .expect("required module is bundled");
            std::fs::write(
                dir.join(format!("{file}.mind")),
                format!("{bundled}\npub fn ovr_{file}() -> i64 {{\n    return 7;\n}}\n"),
            )
            .expect("write override file");
        }
        let table = load_override(&dir).expect("a directory with the four files is used");
        let vec = table
            .iter()
            .find(|(n, _)| *n == "std.vec")
            .expect("vec row")
            .1;
        assert!(
            vec.contains("ovr_vec"),
            "the override text replaces std.vec"
        );
        let row = |rows: &[(&str, &'static str)]| {
            rows.iter()
                .find(|(n, _)| *n == "std.keccak")
                .expect("keccak row")
                .1
        };
        assert!(
            std::ptr::eq(row(&table), row(STDLIB_MIND_SOURCES)),
            "a module the directory does not supply keeps its bundled text"
        );

        // A module file that does not parse is an error naming the file.
        std::fs::write(dir.join("keccak.mind"), "fn (").expect("write broken keccak");
        let err = load_override(&dir).expect_err("a module that does not parse is refused");
        assert!(
            err.contains(&dir.display().to_string())
                && err.contains("`keccak.mind` (std.keccak) does not parse"),
            "{err}"
        );
        std::fs::remove_file(dir.join("keccak.mind")).expect("remove keccak.mind");

        // Without `io.mind` the directory is an error naming the path and the file.
        std::fs::remove_file(dir.join("io.mind")).expect("remove io.mind");
        let err = load_override(&dir).expect_err("a missing required file is refused");
        assert!(
            err.contains(&dir.display().to_string()) && err.contains("lacks `io.mind` (std.io)"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
