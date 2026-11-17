// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Runnable-artifact preflight for a local function that shares an imported name.

use std::collections::BTreeSet;

use crate::ast::{Module, Node};
use crate::diagnostics::{Diagnostic, Span};

const HELP: &str = "the shipped backend cannot emit owner-colliding native symbols yet; rename the local or imported function";

/// On the executable route, refuse local function `N` only when this module
/// references the imported function `N`. A reference is a qualified call, or
/// the last segment or alias of a `use` or `import`, including a one-segment
/// item. A local that merely shares an unused export name still builds.
/// The shared-library route has a scope map and keeps the name check.
pub fn check_imported_local_fn_collisions(
    module: &Module,
    src: &str,
    file: Option<&str>,
) -> Vec<Diagnostic> {
    let owner = crate::qualified_enums::current_module_path();
    let references = imported_fn_references(module);
    let executable = crate::project::active_module_table::executable_collision_record();
    module
        .items
        .iter()
        .filter_map(|item| {
            let Node::FnDef(fd, span) = item else {
                return None;
            };
            let conflicts = crate::project::active_module_table::imported_fn_conflicts_with_local(
                owner.as_deref(),
                &fd.name,
            );
            if !conflicts {
                return None;
            }
            if executable && !references.contains(fd.name.as_str()) {
                return None;
            }
            Some(
                Diagnostic::error(
                    "lower",
                    "lower::imported_local_fn_collision",
                    format!(
                        "local function `{}` conflicts with a captured imported function; owner-aware native linkage cannot emit this shadow yet — rename one",
                        fd.name
                    ),
                )
                .with_span(Span::from_offsets(src, span.start(), span.end(), file))
                .with_help(HELP),
            )
        })
        .collect()
}

/// Names this module uses as an imported function: the callee of a qualified
/// call, or the last segment and alias of each `use` or `import`. The shared
/// collision resolver then decides whether that name is an imported function.
fn imported_fn_references(module: &Module) -> BTreeSet<String> {
    let mut names = qualified_import_callees(module);
    for item in &module.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            let Node::Import { path, alias, .. } = node else {
                return;
            };
            if let Some(last) = path.last() {
                names.insert(last.clone());
            }
            if let Some(alias) = alias {
                names.insert(alias.clone());
            }
        });
    }
    names
}

/// Bare names of `import.fn(...)` calls. The parser folds those calls to the
/// bare callee and keeps the qualifier on the span.
fn qualified_import_callees(module: &Module) -> BTreeSet<String> {
    let qualified: BTreeSet<(usize, usize)> = module
        .spelling
        .call_qualifiers
        .iter()
        .map(|(span, _)| (span.start(), span.end()))
        .collect();
    let mut names = BTreeSet::new();
    for item in &module.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            if let Node::Call { callee, span, .. } = node {
                if qualified.contains(&(span.start(), span.end())) {
                    names.insert(callee.clone());
                }
            }
        });
    }
    names
}
