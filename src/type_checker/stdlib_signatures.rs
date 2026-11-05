// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Cached return metadata for the bundled pure-MIND standard library.

use std::sync::OnceLock;

use crate::project::stdlib::{STDLIB_MIND_SOURCES, imported_std_extras, is_loaded, std_source};

/// One bundled fn's `(name, param types, return type)`.
pub(crate) type StdSignature = (
    String,
    Vec<crate::ast::TypeAnn>,
    Option<crate::ast::TypeAnn>,
);

const STDLIB_COUNT: usize = STDLIB_MIND_SOURCES.len();

/// The single-file resolver exposes bundled std names through its cached
/// name-only surface, but it deliberately has no active project module table.
/// Keep the return metadata on parsed declarations so an unannotated
/// `let s = string_push_byte(...);` can retain String ownership for a later
/// method call. Under `cross-module-imports`, the existing `ExportedFn`
/// collector supplies the declarations; the std-surface-only build walks the
/// same parsed `FnDef`s because its module table is cfg-gated off.
///
/// Only the std modules `program` loads contribute (the implicit set plus
/// what it imports), in table order. Each module's signatures are parsed once
/// per process, so an import-free compile never parses a module it cannot
/// call.
pub(crate) fn bundled_std_fn_signatures(
    program: &crate::ast::Module,
) -> impl Iterator<Item = &'static StdSignature> {
    let extras = imported_std_extras(program);
    STDLIB_MIND_SOURCES
        .iter()
        .enumerate()
        .filter(move |(_, (path, _))| is_loaded(path, &extras))
        .flat_map(|(index, (path, _))| module_signatures(index, path).iter())
}

/// The signatures bundled module `path` (at table position `index`) declares.
fn module_signatures(index: usize, path: &str) -> &'static [StdSignature] {
    static CACHE: [OnceLock<Vec<StdSignature>>; STDLIB_COUNT] =
        [const { OnceLock::new() }; STDLIB_COUNT];
    CACHE[index].get_or_init(|| {
        let mut out = Vec::new();
        let Some(module) = std_source(path).and_then(|src| crate::parser::parse(src).ok()) else {
            return out;
        };
        #[cfg(feature = "cross-module-imports")]
        out.extend(
            crate::project::module_table::collect_module_exports(path, &module)
                .exported_fns
                .into_iter()
                .map(|f| (f.name, f.param_types, f.ret_type)),
        );
        #[cfg(not(feature = "cross-module-imports"))]
        collect_std_fn_signatures(&module.items, &mut out);
        out
    })
}

#[cfg(not(feature = "cross-module-imports"))]
fn collect_std_fn_signatures(items: &[crate::ast::Node], out: &mut Vec<StdSignature>) {
    for item in items {
        match item {
            crate::ast::Node::FnDef(fd, _) => out.push((
                fd.name.clone(),
                fd.params.iter().map(|param| param.ty.clone()).collect(),
                fd.ret_type.clone(),
            )),
            crate::ast::Node::Block { stmts, .. } => collect_std_fn_signatures(stmts, out),
            _ => {}
        }
    }
}
