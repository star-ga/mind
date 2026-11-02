// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! An `import std.X` whose path names no shipped standard-library module is
//! refused at the import.
//!
//! Such an import used to be accepted while resolving nothing, so each call
//! into it failed on its own (`unsupported call`, E2003) far from the cause.
//! E2002 is the unresolved-name/owner code: the import names a module that
//! does not exist. A `std` import inside a function or statement body binds
//! nothing (an import is a module item), so it is refused at the import too.

use super::{AstSpan, Pretty, diag_from_span};

const UNKNOWN_STD_MODULE_CODE: &str = "E2002";

/// Push the refusal for `path` when it is an unknown `std` import.
pub(super) fn check(
    path: &[String],
    span: AstSpan,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    if let Some(refusal) = refusal(path) {
        errs.push(diag_from_span(
            src,
            file,
            refusal,
            span,
            UNKNOWN_STD_MODULE_CODE,
        ));
    }
}

/// Push the refusal for every `std` import of `module` that is not a module
/// item: the unknown-module refusal when the path names no module, the
/// placement refusal otherwise. Either way the error is at the import.
pub(super) fn check_placement(
    module: &crate::ast::Module,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    for (path, span) in crate::project::stdlib::misplaced_std_imports(module) {
        let message =
            refusal(path).unwrap_or_else(|| crate::project::stdlib::misplaced_std_import(path));
        errs.push(diag_from_span(
            src,
            file,
            message,
            span,
            UNKNOWN_STD_MODULE_CODE,
        ));
    }
}

/// The refusal for an unknown `std` import, or `None` when the import resolves.
/// A project MAY supply its own module at `std.<name>` (spec/v1.0/stdlib.md,
/// "use std.<name> resolution", rule 2), so with a module table installed a
/// path the table resolves is accepted even when no bundled module has it.
fn refusal(path: &[String]) -> Option<String> {
    let refusal = crate::project::stdlib::unknown_std_import(path)?;
    #[cfg(feature = "cross-module-imports")]
    if crate::project::active_module_table::with(|table| {
        table.is_some_and(|table| table.get_import(path).is_some())
    }) {
        return None;
    }
    Some(refusal)
}
