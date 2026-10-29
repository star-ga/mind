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

//! Manifest-bounded dependency capture for the tree-evaluated test runner.

use std::path::Path;

use crate::ast::{Module, Node};
#[cfg(feature = "cross-module-imports")]
use crate::eval::EvalBindingKind;
use crate::eval::{EvalBindings, eval_owned_symbol};

#[derive(Default)]
pub(super) struct EvalSupport {
    pub(super) imported_items: Vec<Node>,
    pub(super) bindings: EvalBindings,
    pub(super) entry_owner: Option<String>,
    pub(super) setup_error: Option<String>,
}

pub(super) fn qualify_module_declarations(module: &mut Module, owner: &str, globals: bool) {
    fn qualify(items: &mut [Node], owner: &str, globals: bool) {
        for item in items {
            match item {
                Node::FnDef(function, _) => {
                    function.name = eval_owned_symbol(owner, &function.name);
                }
                Node::Const { name, .. } => {
                    *name = eval_owned_symbol(owner, name);
                }
                Node::Let { name, .. } if globals => {
                    *name = eval_owned_symbol(owner, name);
                }
                Node::Block { stmts, .. } => qualify(stmts, owner, globals),
                _ => {}
            }
        }
    }
    qualify(&mut module.items, owner, globals);
}

#[cfg(feature = "cross-module-imports")]
fn resolve_typecheck_imports(
    module: &Module,
    owner: &str,
    scope: &crate::project::single_file_scope::ProjectScope,
) -> Module {
    fn resolve(
        items: &mut [Node],
        owner: &str,
        scope: &crate::project::single_file_scope::ProjectScope,
    ) {
        for item in items {
            match item {
                Node::Import { path, .. } => {
                    if let Some(target) = scope.resolve_import_path(owner, path) {
                        *path = target.split('.').map(str::to_string).collect();
                    }
                }
                Node::Block { stmts, .. } => resolve(stmts, owner, scope),
                _ => {}
            }
        }
    }

    let mut resolved = module.clone();
    resolve(&mut resolved.items, owner, scope);
    resolved
}

#[cfg(feature = "cross-module-imports")]
fn bind_import_refs(
    owner: &str,
    imports: &[Vec<String>],
    references: Vec<crate::parser::EvalImportRef>,
    scope: &crate::project::single_file_scope::ProjectScope,
    bindings: &mut EvalBindings,
    qualified_refs: &mut std::collections::BTreeMap<(String, usize, usize, String, bool), String>,
) -> Result<(), String> {
    use crate::parser::EvalImportRefKind;

    for reference in references {
        let matches = imports
            .iter()
            .filter(|import| {
                import.as_slice() == reference.qualifier.as_slice()
                    || (reference.qualifier.len() == 1
                        && import.last() == reference.qualifier.first())
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            let code = match reference.kind {
                EvalImportRefKind::Call => "E2003",
                EvalImportRefKind::Value => "E2002",
            };
            return Err(format!(
                "error[type-check][{code}]: ambiguous evaluator import qualifier `{}` in module `{owner}`",
                reference.qualifier.join(".")
            ));
        }
        let import = matches[0];
        let Some(target_owner) = scope.resolve_import_path(owner, import) else {
            let code = match reference.kind {
                EvalImportRefKind::Call => "E2003",
                EvalImportRefKind::Value => "E2002",
            };
            return Err(format!(
                "error[type-check][{code}]: unresolved evaluator import `{}` in module `{owner}`",
                import.join(".")
            ));
        };
        if !scope.resolves_imported_symbol(owner, import, &reference.symbol) {
            let code = match reference.kind {
                EvalImportRefKind::Call => "E2003",
                EvalImportRefKind::Value => "E2002",
            };
            return Err(format!(
                "error[type-check][{code}]: imported symbol `{}.{}` is private or unresolved in module `{owner}`",
                import.join("."),
                reference.symbol
            ));
        }
        bindings.insert(
            owner,
            reference.span,
            match reference.kind {
                EvalImportRefKind::Call => EvalBindingKind::Call,
                EvalImportRefKind::Value => EvalBindingKind::Value,
            },
            target_owner,
            &reference.symbol,
        );
        qualified_refs.insert(
            (
                owner.to_string(),
                reference.span.start(),
                reference.span.end(),
                reference.symbol,
                matches!(reference.kind, EvalImportRefKind::Call),
            ),
            target_owner.to_string(),
        );
    }
    Ok(())
}

/// Bind the bare calls `owner` may make into the exported functions of the
/// project modules its own imports select, mirroring the type checker's
/// lexical-then-imported order for unqualified callees. Bundled `std`
/// modules keep their established resolution and are not bound here.
#[cfg(feature = "cross-module-imports")]
fn bind_bare_imported_calls(
    owner: &str,
    imports: &[Vec<String>],
    scope: &crate::project::single_file_scope::ProjectScope,
    linked_owners: &std::collections::BTreeSet<String>,
    bindings: &mut EvalBindings,
) {
    for import in imports {
        let Some(target) = scope.resolve_import_path(owner, import) else {
            continue;
        };
        if target == owner || target.starts_with("std.") || !linked_owners.contains(target) {
            continue;
        }
        let Some(source) = scope
            .sources()
            .iter()
            .find(|source| source.module_path() == target)
        else {
            continue;
        };
        for item in super::top_level::refs(&source.module().items) {
            if let Node::FnDef(function, _) = item {
                if scope.resolves_imported_symbol(owner, import, &function.name) {
                    bindings.insert_imported_call(owner, target, &function.name);
                }
            }
        }
    }
}

/// Capture the same manifest-bounded import closure used by `check` and native
/// builds. Declarations retain an internal lexical owner, while import
/// references are bound by `(owner file, source span)` to the exact captured
/// module selected by that file's own import declaration.
#[cfg(feature = "cross-module-imports")]
pub(super) fn prepare_eval_support(path: &Path, source: &str) -> EvalSupport {
    use std::collections::{BTreeMap, BTreeSet};

    use crate::project::single_file_scope::{Discovery, discover_with_source};

    let discovery = match discover_with_source(path, crate::project::DEFAULT_TARGET_BLOCK, source) {
        Ok(discovery) => discovery,
        Err(error) => {
            return EvalSupport {
                setup_error: Some(format!("project import resolution failed: {error}")),
                ..EvalSupport::default()
            };
        }
    };

    match discovery {
        Discovery::SingleTranslationUnit => EvalSupport::default(),
        Discovery::MissingProject(imports) => EvalSupport {
            setup_error: Some(format!(
                "error[type-check][E2003]: local import(s) {} require an enclosing Mind.toml project that declares the source set",
                imports.join(", ")
            )),
            ..EvalSupport::default()
        },
        Discovery::Project(scope) => {
            use std::collections::VecDeque;

            use crate::project::stdlib::STDLIB_MIND_SOURCES;

            let _scope_guard = scope.install();
            let entry = scope
                .entry()
                .canonicalize()
                .unwrap_or_else(|_| scope.entry().to_path_buf());
            let linked_owners = scope
                .linked_sources()
                .map(|captured| captured.module_path().to_string())
                .collect::<BTreeSet<_>>();
            let mut imported_items = Vec::new();
            let mut bindings = EvalBindings::default();
            let mut entry_owner = None;
            let mut linked_modules = BTreeMap::<String, Module>::new();
            let mut module_dependencies = BTreeMap::<String, BTreeSet<String>>::new();
            let mut std_queue = VecDeque::<String>::new();
            let mut qualified_refs = std::collections::BTreeMap::new();

            for captured in scope.linked_sources() {
                let parsed = match crate::parser::parse_for_eval(captured.source()) {
                    Ok(parsed) => parsed,
                    Err(errors) => {
                        return EvalSupport {
                            setup_error: Some(format!(
                                "project import evaluation parse failed for `{}`: {}",
                                captured.module_path(),
                                errors
                                    .iter()
                                    .map(ToString::to_string)
                                    .collect::<Vec<_>>()
                                    .join("; ")
                            )),
                            ..EvalSupport::default()
                        };
                    }
                };
                let _module_guard =
                    crate::qualified_enums::ModuleGuard::install(captured.module_path());
                let imports = super::top_level::refs(&parsed.module.items)
                    .into_iter()
                    .filter_map(|item| match item {
                        Node::Import { path, .. } => Some(path.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if let Err(error) = bind_import_refs(
                    captured.module_path(),
                    &imports,
                    parsed.import_refs.clone(),
                    &scope,
                    &mut bindings,
                    &mut qualified_refs,
                ) {
                    return EvalSupport {
                        setup_error: Some(error),
                        ..EvalSupport::default()
                    };
                }
                bind_bare_imported_calls(
                    captured.module_path(),
                    &imports,
                    &scope,
                    &linked_owners,
                    &mut bindings,
                );
                let _qualified_guard =
                    crate::project::active_module_table::QualifiedImportsGuard::install(
                        qualified_refs.clone(),
                    );
                let captured_path = captured
                    .path()
                    .canonicalize()
                    .unwrap_or_else(|_| captured.path().to_path_buf());
                let is_entry = captured_path == entry;
                // Type validation must see the same exact import owner selected
                // during manifest discovery. A source-level `import schema`
                // can resolve to `crate.src.schema`; reconstructing it as
                // `crate.schema` loses qualified exported types before test
                // evaluation starts.
                let typecheck_module =
                    resolve_typecheck_imports(&parsed.module, captured.module_path(), &scope);
                let unresolved = crate::type_checker::check_module_types(
                    &typecheck_module,
                    captured.source(),
                    &crate::type_checker::TypeEnv::default(),
                )
                .into_iter()
                .filter(|diag| {
                    // The evaluator deliberately defers an unresolved bare
                    // call in a linked module to execution, where it reports
                    // `unsupported operation` unless one of that module's own
                    // imports exports the callee (`bind_bare_imported_calls`). Explicit namespace references
                    // were validated by `bind_import_refs` above, so this
                    // narrow defer cannot allow a private imported symbol or
                    // an arbitrary consumer-local capture.
                    matches!(diag.code, "E2002" | "E2003") && (is_entry || diag.code != "E2003")
                })
                .map(|diag| crate::diagnostics::render(captured.source(), &diag))
                .collect::<Vec<_>>();
                if !unresolved.is_empty() {
                    return EvalSupport {
                        setup_error: Some(unresolved.join("\n")),
                        ..EvalSupport::default()
                    };
                }
                for import in &imports {
                    if let Some(target) = scope.resolve_import_path(captured.module_path(), import)
                    {
                        if linked_owners.contains(target) || target.starts_with("std.") {
                            module_dependencies
                                .entry(captured.module_path().to_string())
                                .or_default()
                                .insert(target.to_string());
                            if target.starts_with("std.") {
                                std_queue.push_back(target.to_string());
                            }
                        }
                    }
                }

                if captured_path == entry {
                    entry_owner = Some(captured.module_path().to_string());
                    continue;
                }

                let mut linked_module = parsed.module;
                qualify_module_declarations(&mut linked_module, captured.module_path(), true);
                linked_modules.insert(captured.module_path().to_string(), linked_module);
            }

            let mut loaded_std = BTreeSet::new();
            while let Some(owner) = std_queue.pop_front() {
                if !loaded_std.insert(owner.clone()) {
                    continue;
                }
                let Some((_, source)) = STDLIB_MIND_SOURCES
                    .iter()
                    .find(|(module_path, _)| *module_path == owner)
                else {
                    return EvalSupport {
                        setup_error: Some(format!("unresolved bundled evaluator module `{owner}`")),
                        ..EvalSupport::default()
                    };
                };
                let parsed = match crate::parser::parse_for_eval(source) {
                    Ok(parsed) => parsed,
                    Err(errors) => {
                        return EvalSupport {
                            setup_error: Some(format!(
                                "bundled evaluator parse failed for `{owner}`: {}",
                                errors
                                    .iter()
                                    .map(ToString::to_string)
                                    .collect::<Vec<_>>()
                                    .join("; ")
                            )),
                            ..EvalSupport::default()
                        };
                    }
                };
                let imports = super::top_level::refs(&parsed.module.items)
                    .into_iter()
                    .filter_map(|item| match item {
                        Node::Import { path, .. } => Some(path.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                for import in &imports {
                    let Some(target) = scope.resolve_import_path(&owner, import) else {
                        return EvalSupport {
                            setup_error: Some(format!(
                                "unresolved evaluator import `{}` in bundled module `{owner}`",
                                import.join(".")
                            )),
                            ..EvalSupport::default()
                        };
                    };
                    module_dependencies
                        .entry(owner.clone())
                        .or_default()
                        .insert(target.to_string());
                    std_queue.push_back(target.to_string());
                }
                if let Err(error) = bind_import_refs(
                    &owner,
                    &imports,
                    parsed.import_refs,
                    &scope,
                    &mut bindings,
                    &mut qualified_refs,
                ) {
                    return EvalSupport {
                        setup_error: Some(error),
                        ..EvalSupport::default()
                    };
                }
                let mut module = parsed.module;
                qualify_module_declarations(&mut module, &owner, true);
                linked_modules.insert(owner, module);
            }

            fn visit(
                owner: &str,
                dependencies: &BTreeMap<String, BTreeSet<String>>,
                modules: &BTreeMap<String, Module>,
                visiting: &mut BTreeSet<String>,
                visited: &mut BTreeSet<String>,
                order: &mut Vec<String>,
            ) {
                if visited.contains(owner) || !modules.contains_key(owner) {
                    return;
                }
                // A module-import cycle is ordered deterministically here. Any
                // cyclic const initializer still fails when its unavailable
                // peer is evaluated; function-only cycles remain executable.
                if !visiting.insert(owner.to_string()) {
                    return;
                }
                if let Some(required) = dependencies.get(owner) {
                    for dependency in required {
                        visit(dependency, dependencies, modules, visiting, visited, order);
                    }
                }
                visiting.remove(owner);
                if visited.insert(owner.to_string()) {
                    order.push(owner.to_string());
                }
            }

            let mut order = Vec::new();
            let mut visiting = BTreeSet::new();
            let mut visited = BTreeSet::new();
            for owner in linked_modules.keys() {
                visit(
                    owner,
                    &module_dependencies,
                    &linked_modules,
                    &mut visiting,
                    &mut visited,
                    &mut order,
                );
            }
            for owner in order {
                append_declarations(&linked_modules[&owner].items, &mut imported_items);
            }

            EvalSupport {
                imported_items,
                bindings,
                entry_owner,
                setup_error: None,
            }
        }
    }
}

#[cfg(feature = "cross-module-imports")]
fn append_declarations(items: &[Node], out: &mut Vec<Node>) {
    for item in super::top_level::refs(items) {
        match item {
            Node::FnDef(..) | Node::Const { .. } | Node::Let { .. } => out.push(item.clone()),
            _ => {}
        }
    }
}

#[cfg(not(feature = "cross-module-imports"))]
pub(super) fn prepare_eval_support(_path: &Path, _source: &str) -> EvalSupport {
    EvalSupport::default()
}
