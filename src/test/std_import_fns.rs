// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Which bundled std functions `mindc test` admits.
//!
//! An explicit import admits that module and every module it reaches. An
//! ambient module, present with no import, admits only the names the
//! runtime support links. Kept in its own file so the test runner stays
//! inside the source-size ceiling.

use crate::ast::Node;
use crate::parser;

use super::top_level;

/// The std modules a program calls WITHOUT an import: the compiled build
/// resolves their `string_*` / `vec_*` / `map_*` functions through the
/// runtime-support library it links into every artifact, and the std modules
/// themselves call each other this way (std.io builds strings with
/// `string_push_byte` and no import line). The test evaluator runs the same
/// definitions from their bundled MIND source.
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
const AMBIENT_STD: [&str; 3] = ["std.string", "std.vec", "std.map"];

/// The runtime-support source every artifact links — the authority on which
/// std names the compiled build can actually call without an import.
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
const MIND_RUNTIME_SUPPORT_C: &str = include_str!("../../runtime-support/mind_intrinsics.c");

/// The function names the runtime-support library exports (`MIND_EXPORT` /
/// `MIND_EXPORT_WEAK` definitions in [`MIND_RUNTIME_SUPPORT_C`]). An ambient
/// std fn is admissible exactly when its name is exported here: the ambient
/// set is "the names the compiled build links", so admitting anything else
/// made `mindc test` pass a program whose `mindc build` cannot link —
/// `vec_free` was callable with no import while the linker answered
/// `undefined reference to 'vec_free'`.
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
fn runtime_support_exports() -> &'static std::collections::HashSet<&'static str> {
    static EXPORTS: std::sync::OnceLock<std::collections::HashSet<&'static str>> =
        std::sync::OnceLock::new();
    EXPORTS.get_or_init(|| {
        MIND_RUNTIME_SUPPORT_C
            .lines()
            .filter_map(|line| {
                // `MIND_EXPORT int64_t name(...)` / `MIND_EXPORT_WEAK ... name(...)`.
                // The `#  define MIND_EXPORT ...` macro lines start with `#`
                // and never match.
                let rest = line.trim_start().strip_prefix("MIND_EXPORT")?;
                let rest = rest.strip_prefix("_WEAK").unwrap_or(rest);
                let paren = rest.find('(')?;
                rest[..paren]
                    .trim()
                    .rsplit(char::is_whitespace)
                    .next()
                    .filter(|name| !name.is_empty())
            })
            .collect()
    })
}

/// Resolve `import std.X` / `use std.X` declarations of a test module (and,
/// transitively, of the modules they import) against the stdlib sources baked
/// into the binary (RFC 0005 Phase C, `project::stdlib`), returning the
/// imported modules' `FnDef` items so the interpreter's fn table can dispatch
/// cross-module calls — e.g. a `[test]` running the std.sha256 KAT — plus
/// their module-level `Const` items, so a fn body that reads a module const
/// (std.json's `MAX_DEPTH`) resolves it through the interpreter's const
/// table.
///
/// Deterministic: imports are processed in source-encounter order (FIFO
/// worklist, `BTreeSet` visited guard against cycles). A non-`std` import
/// contributes nothing here, nor does a `std.<name>` the project supplies
/// (`project_std`); any other `std` import that names no shipped module is
/// refused at the import, as `mindc check` refuses it, and so is a `std`
/// import inside a body (not a module item); a bundled module that fails to
/// parse is a fail-loud error. An EXPLICIT import admits that module and
/// every module it imports, in full. An AMBIENT module (present with no
/// import, and not reached from an explicit one) admits only the fns its
/// runtime support links ([`runtime_support_exports`]).
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
pub(super) fn resolve_std_import_fns(
    module: &crate::ast::Module,
    project_std: Option<&std::collections::BTreeSet<String>>,
) -> Result<Vec<Node>, String> {
    use std::collections::VecDeque;

    let import_key = |item: &Node| -> Result<Option<String>, String> {
        if let Node::Import { path, .. } = item {
            if project_std.is_some_and(|own| own.contains(&path.join("."))) {
                return Ok(None);
            }
            if let Some(refusal) = crate::project::stdlib::unknown_std_import(path) {
                return Err(format!("error[type-check][E2002]: {refusal}"));
            }
            if path.len() == 2 && path[0] == "std" {
                return Ok(Some(path.join(".")));
            }
        }
        Ok(None)
    };

    // A `std` import inside a body binds nothing; `mindc check` refuses it at
    // the import, and so does this runner, before any test body runs.
    if let Some((path, _)) = crate::project::stdlib::misplaced_std_imports(module).first() {
        let supplied = project_std.is_some_and(|own| own.contains(&path.join(".")));
        let refusal = crate::project::stdlib::unknown_std_import(path)
            .filter(|_| !supplied)
            .unwrap_or_else(|| crate::project::stdlib::misplaced_std_import(path));
        return Err(format!("error[type-check][E2002]: {refusal}"));
    }

    let mut out: Vec<Node> = Vec::new();
    let mut visited: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // The file's OWN imports come first, in source-encounter order (the
    // documented FIFO discipline); the ambient modules follow. An explicit
    // import admits that module's every definition, and every module it
    // imports: the compiled build compiles that graph from source. Only an
    // ambient module (no import in the file, and not reached from one) is
    // limited to the names the runtime support links.
    let mut explicit: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    for item in top_level::refs(&module.items) {
        if let Some(key) = import_key(item)? {
            explicit.insert(key.clone());
            queue.push_back(key);
        }
    }
    queue.extend(AMBIENT_STD.iter().map(|key| key.to_string()));
    // Transitive imports of an explicit module join this set when they are
    // seen, before they are processed. Ambient modules never add to it.
    let mut admit_all: std::collections::BTreeSet<String> = explicit.clone();
    // A name the file binds through its own `extern "C"` block calls that
    // symbol, never the ambient std definition of the same name.
    let extern_names: std::collections::BTreeSet<&str> = top_level::refs(&module.items)
        .into_iter()
        .filter_map(|item| match item {
            Node::ExternBlock { fns, .. } => Some(fns.iter().map(|f| f.name.as_str())),
            _ => None,
        })
        .flatten()
        .collect();
    while let Some(key) = queue.pop_front() {
        if !visited.insert(key.clone()) {
            continue;
        }
        let Some(src) = crate::project::stdlib::std_source(&key) else {
            continue;
        };
        let imported = parser::parse(src).map_err(|errs| {
            format!(
                "failed to parse bundled std module '{key}': {}",
                errs.iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        })?;
        let admit_module = admit_all.contains(&key);
        for item in top_level::refs(&imported.items) {
            match item {
                Node::FnDef(fd, _) if extern_names.contains(fd.name.as_str()) => {}
                Node::FnDef(fd, _)
                    if !admit_module && !runtime_support_exports().contains(fd.name.as_str()) => {}
                Node::FnDef(..) | Node::Const { .. } => out.push(item.clone()),
                Node::Import { .. } => {
                    if let Some(child) = import_key(item)? {
                        if admit_module {
                            admit_all.insert(child.clone());
                        }
                        queue.push_back(child);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(out)
}

/// An explicit `import std.x509` compiles `std.sha256` from source, including
/// the helpers `sha256` calls. Those helpers are not runtime-support exports.
/// The ambient `std.vec` name `vec_free` stays refused: nothing imported it.
#[cfg(all(test, any(feature = "cross-module-imports", feature = "std-surface")))]
#[test]
fn explicit_import_admits_the_modules_it_imports() {
    let src = "import std.x509;\nfn t() -> i64 {\n    return 0;\n}\n";
    let module = parser::parse(src).expect("parse");
    let items = resolve_std_import_fns(&module, None).expect("resolve");
    let names: std::collections::BTreeSet<&str> = items
        .iter()
        .filter_map(|item| match item {
            Node::FnDef(fd, _) => Some(fd.name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        names.contains("sha256") && names.contains("rotr32"),
        "std.sha256 and its helpers must be admitted through import std.x509, got {names:?}"
    );
    assert!(
        !names.contains("vec_free"),
        "ambient vec_free must stay refused"
    );
}
