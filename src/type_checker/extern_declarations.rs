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
}
