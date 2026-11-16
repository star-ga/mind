// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Runnable-artifact preflight for owner collisions in imported functions.

use std::collections::BTreeSet;

use crate::ast::{Module, Node};
use crate::diagnostics::{Diagnostic, Span};

const HELP: &str = "the shipped backend cannot emit owner-colliding native symbols yet; rename the local or imported function";

/// Refuse a local function that the link would emit as a global symbol
/// duplicating a captured import. A private function the link plan marks
/// internal is not that symbol. A qualified call of the same name still
/// lowers onto the local definition (the import's forward declaration is
/// skipped), so that call stays refused.
pub fn check_imported_local_fn_collisions(
    module: &Module,
    src: &str,
    file: Option<&str>,
) -> Vec<Diagnostic> {
    let owner = crate::qualified_enums::current_module_path();
    let imported_calls = qualified_import_callees(module);
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
            let internal =
                crate::project::active_module_table::local_fn_emitted_internal(&fd.name);
            if internal && !imported_calls.contains(fd.name.as_str()) {
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
