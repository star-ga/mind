// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Std modules the `mindc test` runner merges into a test module.
//!
//! Kept in its own file so the test runner stays inside the source-size
//! ceiling. An explicit import admits that module's functions, constants,
//! extern blocks and structs. An ambient module admits only the functions
//! the runtime support library links, plus the structs and extern blocks
//! those functions' bodies name. A runtime string function stays with the
//! evaluator, and a struct the test file declares itself keeps its own
//! declaration.

use std::collections::VecDeque;

use crate::ast::Node;
use crate::parser;

use super::top_level;

/// The std modules a program calls WITHOUT an import: the compiled build
/// resolves their `string_*` / `vec_*` / `map_*` functions through the
/// runtime-support library it links into every artifact, and the std modules
/// themselves call each other this way (std.io builds strings with
/// `string_push_byte` and no import line). The test evaluator runs the same
/// definitions from their bundled MIND source.
const AMBIENT_STD: [&str; 3] = ["std.string", "std.vec", "std.map"];

/// The runtime-support source every artifact links — the authority on which
/// std names the compiled build can actually call without an import.
const MIND_RUNTIME_SUPPORT_C: &str = include_str!("../../runtime-support/mind_intrinsics.c");

/// The function names the runtime-support library exports (`MIND_EXPORT` /
/// `MIND_EXPORT_WEAK` definitions in [`MIND_RUNTIME_SUPPORT_C`]). An ambient
/// std fn is admissible exactly when its name is exported here: the ambient
/// set is "the names the compiled build links", so admitting anything else
/// made `mindc test` pass a program whose `mindc build` cannot link —
/// `vec_free` was callable with no import while the linker answered
/// `undefined reference to 'vec_free'`.
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
/// table, plus their `extern "C"` blocks and `struct` declarations (inert
/// when evaluated), so the string-equality pass reads the declared result of
/// a libc call in std code (std.fs `read(...) -> i64`) and a std record's
/// field (`v.cap` in std.vec) as `mindc check` and the build do. A struct the
/// test module declares itself keeps its own declaration. A function the
/// evaluator runs as the runtime support library does (`string_eq`, which the
/// build links) is not taken: a call of it computes what the build computes.
///
/// Deterministic: imports are processed in source-encounter order (FIFO
/// worklist, `BTreeSet` visited guard against cycles). A non-`std` import
/// contributes nothing here, nor does a `std.<name>` the project supplies
/// (`project_std`); any other `std` import that names no shipped module is
/// refused at the import, as `mindc check` refuses it, and so is a `std`
/// import inside a body (not a module item); a bundled module that fails to
/// parse is a fail-loud error. An explicit import admits the module's every
/// definition; an ambient module admits only the fns its runtime support
/// links ([`runtime_support_exports`]).
pub(super) fn resolve_std_import_fns(
    module: &crate::ast::Module,
    project_std: Option<&std::collections::BTreeSet<String>>,
) -> Result<Vec<Node>, String> {
    use crate::eval::extern_c_intrinsics;

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

    let own_structs: std::collections::BTreeSet<&str> = top_level::refs(&module.items)
        .into_iter()
        .filter_map(|item| match item {
            Node::StructDef { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    let mut out: Vec<Node> = Vec::new();
    let mut visited: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // The file's OWN imports come first, in source-encounter order (the
    // documented FIFO discipline); the ambient modules follow. An explicit
    // import keeps admitting the module's every definition (that is the
    // documented cross-module behavior), while an AMBIENT module admits only
    // the names the compiled build links from its runtime support.
    let mut explicit: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    for item in top_level::refs(&module.items) {
        if let Some(key) = import_key(item)? {
            explicit.insert(key.clone());
            queue.push_back(key);
        }
    }
    queue.extend(AMBIENT_STD.iter().map(|key| key.to_string()));
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
    queue.extend(bare_std_modules(module));
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
        let ambient = !explicit.contains(&key);
        for item in top_level::refs(&imported.items) {
            match item {
                Node::FnDef(fd, _) if extern_names.contains(fd.name.as_str()) => {}
                Node::FnDef(fd, _) if extern_c_intrinsics::is_runtime_string_fn(&fd.name) => {}
                Node::FnDef(fd, _)
                    if ambient && !runtime_support_exports().contains(fd.name.as_str()) => {}
                Node::StructDef { name, .. } if own_structs.contains(name.as_str()) => {}
                Node::FnDef(..)
                | Node::Const { .. }
                | Node::ExternBlock { .. }
                | Node::StructDef { .. } => out.push(item.clone()),
                Node::Import { .. } => queue.extend(import_key(item)?),
                _ => {}
            }
        }
    }
    Ok(out)
}

/// The bundled std modules that define a std.string function (`string_*`)
/// `module` calls by its bare name and does not define itself. `mindc check`
/// and the build see every bundled std declaration with no `import` (the
/// bundled std signatures), so `mindc test` loads the module that defines
/// the callee too. A method call on a string names its `string_<method>`
/// function, which is the call the build makes for it. Callees the evaluator
/// runs itself (the runtime-support emulations) load nothing.
fn bare_std_modules(module: &crate::ast::Module) -> Vec<String> {
    use crate::eval::extern_c_intrinsics;
    use crate::project::stdlib::STDLIB_MIND_SOURCES;
    use std::collections::BTreeSet;

    let mut defined = BTreeSet::new();
    let mut wanted = BTreeSet::new();
    let mut pending: Vec<&Node> = module.items.iter().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::FnDef(fd, _) => {
                defined.insert(fd.name.clone());
                pending.extend(fd.body.iter());
                continue;
            }
            Node::Call { callee, .. } if !callee.contains('.') && !callee.contains("::") => {
                wanted.insert(callee.clone());
            }
            Node::MethodCall { method, .. } => {
                wanted.insert(format!("string_{method}"));
            }
            _ => {}
        }
        pending.extend(crate::eval::closures::node_children_ref(node));
    }
    let wanted: Vec<String> = wanted
        .into_iter()
        .filter(|name| {
            name.starts_with("string_")
                && !defined.contains(name)
                && !extern_c_intrinsics::is_extern_c_intrinsic(name)
                && !extern_c_intrinsics::is_runtime_string_fn(name)
        })
        .collect();
    STDLIB_MIND_SOURCES
        .iter()
        .filter(|(_, src)| {
            wanted
                .iter()
                .any(|name| src.contains(&format!("fn {name}(")))
        })
        .map(|(path, _)| (*path).to_string())
        .collect()
}
