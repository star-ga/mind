// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
//! An extern declaration is a request for a particular std.posix signature,
//! never permission to call an arbitrary seed function with the same bare name.
use super::{ClosureRejection, StdSeed};
use crate::ast::{CallConv, Module, Node};

pub(super) fn admit_externs(ast: &Module, seed: &StdSeed) -> Result<(), ClosureRejection> {
    let mut result = Ok(());
    let user_definitions = super::ast_definition_counts(ast);
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            let Node::ExternBlock { callconv, fns, .. } = node else {
                return;
            };
            for external in fns {
                if result.is_err() {
                    return;
                }
                let posix_name = seed
                    .modules
                    .iter()
                    .any(|(module, names)| module == "posix" && names.contains(&external.name));
                let source = seed.fn_source(&external.name);
                let matched = posix_name
                    && !user_definitions.contains_key(&external.name)
                    && *callconv == CallConv::C
                    && !external.is_varargs
                    && source.as_ref().is_some_and(|m| {
                        m.items.iter().any(|n| {
                            matches!(n, Node::FnDef(f, _)
                        if f.name == external.name && f.type_params.is_empty()
                        && f.ret_type == external.ret_type
                        && f.params.len() == external.params.len()
                        && f.params.iter().zip(&external.params).all(|(a, b)| a.ty == b.ty))
                        })
                    })
                    && seed.fn_ir(&external.name).is_some();
                if !matched {
                    result = Err(ClosureRejection {
                        kind: "closure.extern_signature_mismatch",
                        detail: format!(
                            "extern `{}` must match a non-variadic std.posix function's C signature and available body",
                            external.name
                        ),
                    });
                }
            }
        });
    }
    result
}
