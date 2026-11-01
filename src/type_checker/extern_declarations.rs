// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

use super::{check_extern_type_with_repr_c, diag_from_span, nerve_walk};
use crate::ast::{CallConv, Node, TypeAnn};
use crate::diagnostics::Diagnostic;
use crate::eval::{lower::resolve_callconv, type_aliases::LocalTypeAliases};
use std::collections::{BTreeMap, BTreeSet};

type Signature = (CallConv, bool, Vec<TypeAnn>, Option<TypeAnn>);

/// A source module owns one extern namespace: lowering hoists declarations
/// from function and control-flow bodies into the same foreign-symbol table.
/// Repeated declarations may rename parameters, but must agree on the complete
/// ABI. A separate public checker call creates a separate source-owner table.
pub(super) fn check(items: &[Node], src: &str, file: Option<&str>, errs: &mut Vec<Diagnostic>) {
    let aliases = LocalTypeAliases::new(items);
    let mut seen = BTreeMap::new();
    fn visit<'a>(
        item: &'a Node,
        scope: (&LocalTypeAliases, bool),
        seen: &mut BTreeMap<&'a str, Signature>,
        src: &str,
        file: Option<&str>,
        errs: &mut Vec<Diagnostic>,
    ) {
        let (aliases, nested) = scope;
        if let Node::ExternBlock { callconv, fns, .. } = item {
            for function in fns {
                if nested {
                    // Nested lowering consumes raw annotations and has no
                    // reviewed alias/aggregate-layout ownership contract.
                    // Reuse the existing C-type admission rule; never let an
                    // unknown Named/aggregate annotation fall back to i64.
                    let no_layouts = BTreeSet::new();
                    for param in &function.params {
                        if let Err(reason) = check_extern_type_with_repr_c(&param.ty, &no_layouts) {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "nested extern `{}` parameter `{}`: {reason}; move layout-dependent declarations to module scope",
                                    function.name, param.name
                                ),
                                param.span,
                                "safety::extern_non_repr_c",
                            ));
                        }
                    }
                    if let Some(ret) = &function.ret_type {
                        if let Err(reason) = check_extern_type_with_repr_c(ret, &no_layouts) {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "nested extern `{}` return type: {reason}; move layout-dependent declarations to module scope",
                                    function.name
                                ),
                                function.span,
                                "safety::extern_non_repr_c",
                            ));
                        }
                    }
                }
                let canonical_type = |ty: &TypeAnn| {
                    if nested {
                        ty.clone()
                    } else {
                        aliases.resolve(ty)
                    }
                };
                let signature = (
                    resolve_callconv(*callconv),
                    function.is_varargs,
                    function
                        .params
                        .iter()
                        .map(|param| canonical_type(&param.ty))
                        .collect(),
                    function.ret_type.as_ref().map(canonical_type),
                );
                if let Some(previous) = seen.get(function.name.as_str()) {
                    if previous != &signature {
                        errs.push(diag_from_span(
                                    src,
                                    file,
                                    format!(
                                        "conflicting extern declaration `{}` in the same source module; parameter types, return type, variadic status and calling convention must agree",
                                        function.name
                                    ),
                                    function.span,
                                    "safety::extern_conflicting_declaration",
                                ));
                    }
                } else {
                    seen.insert(function.name.as_str(), signature);
                }
            }
        }
        // Only a root-level transparent module block preserves declaration
        // scope. Functions and executable expressions share symbol ownership
        // but must not inherit unimplemented nested alias/layout semantics.
        let child_nested = nested || !matches!(item, Node::Block { .. });
        nerve_walk::for_each_child(item, &mut |child| {
            visit(child, (aliases, child_nested), seen, src, file, errs)
        });
    }
    for item in items {
        visit(item, (&aliases, false), &mut seen, src, file, errs);
    }
    #[cfg(feature = "cross-module-imports")]
    refuse_non_primitive_imports(src, file, errs, &seen);
}

/// An imported extern may not name a struct or other non-primitive type.
/// The importer does not own that layout, so classifying it would be a guess.
#[cfg(feature = "cross-module-imports")]
pub(crate) fn imported_type_is_primitive(ty: &TypeAnn) -> bool {
    check_extern_type_with_repr_c(ty, &BTreeSet::new()).is_ok()
}

#[cfg(feature = "cross-module-imports")]
fn refuse_non_primitive_imports(
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
    local_names: &BTreeMap<&str, Signature>,
) {
    let owner = crate::qualified_enums::current_module_path();
    for ext in crate::project::active_module_table::visible_externs(owner.as_deref()) {
        if local_names.contains_key(ext.name.as_str()) {
            continue;
        }
        let mut bad = Vec::new();
        for ty in ext.param_types.iter().chain(ext.ret_type.as_ref()) {
            if !imported_type_is_primitive(ty) {
                bad.push(match ty {
                    TypeAnn::Named(name) => name.clone(),
                    other => format!("{other:?}"),
                });
            }
        }
        if bad.is_empty() {
            continue;
        }
        errs.push(diag_from_span(
            src,
            file,
            format!(
                "imported extern `{}` names a non-primitive type ({}); the imported C signature is refused",
                ext.name,
                bad.join(", ")
            ),
            crate::ast::Span::new(0, 0),
            "safety::imported_extern_non_primitive",
        ));
    }
}

#[cfg(all(test, feature = "cross-module-imports"))]
mod imported_primitive_tests {
    use super::check;
    use crate::project::active_module_table::{Guard, ResolvedImportsGuard};
    use crate::project::module_table::build_module_table;
    use std::collections::BTreeMap;

    #[test]
    fn imported_extern_refuses_a_non_primitive_type() {
        let provider = crate::parser::parse(
            "#[repr(C)] struct Pair { x: i64, y: i64 } extern \"C\" { fn take(p: Pair) -> i64; fn plain(n: i64) -> i64; }",
        )
        .unwrap();
        let importer_src = "import ffi;\npub fn go() -> i64 { return take(plain(1)); }\n";
        let importer = crate::parser::parse(importer_src).unwrap();
        let _table = Guard::install(build_module_table(&[
            ("crate.ffi".into(), &provider),
            ("crate.user".into(), &importer),
        ]));
        let _owner = crate::qualified_enums::ModuleGuard::install("crate.user");
        let _imports = ResolvedImportsGuard::install(BTreeMap::from([(
            ("crate.user".to_string(), vec!["ffi".to_string()]),
            "crate.ffi".to_string(),
        )]));
        let mut diags = Vec::new();
        check(&importer.items, importer_src, Some("user.mind"), &mut diags);
        let hits: Vec<_> = diags
            .iter()
            .filter(|d| d.code == "safety::imported_extern_non_primitive")
            .collect();
        assert_eq!(hits.len(), 1, "{diags:?}");
        assert!(hits[0].message.contains("take"), "{diags:?}");
        assert!(hits[0].message.contains("Pair"), "{diags:?}");
        let ir = crate::eval::lower::lower_to_ir(&importer).expect("lower");
        assert!(
            ir.instrs.iter().any(|instr| {
                matches!(instr, crate::ir::Instr::ExternFnDecl { name, .. } if name == "plain")
            }),
            "a called primitive import must still be classified"
        );
        assert!(
            ir.instrs.iter().all(|instr| {
                !matches!(instr, crate::ir::Instr::ExternFnDecl { name, .. } if name == "take")
            }),
            "a refused imported extern must not be classified"
        );
    }
}
