// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Enum-variant payload pattern matching in the tree-walking evaluator.

use super::{Value, result_variants};

/// Phase 10.7: bind an enum-variant pattern's payload sub-patterns against the
/// matched payload values. Returns `false` (no match) if any sub-pattern fails;
/// successful bindings are appended to `out`. Ident sub-patterns bind the
/// corresponding payload element; literals must compare equal; nested
/// enum-variant sub-patterns recurse; wildcard always matches.
pub(super) fn match_enum_payload(
    patterns: &[crate::ast::Pattern],
    values: &[Value],
    out: &mut Vec<(String, Value)>,
) -> bool {
    for (pat, value) in patterns.iter().zip(values.iter()) {
        match pat {
            crate::ast::Pattern::Wildcard => {}
            crate::ast::Pattern::Ident(name) => out.push((name.clone(), value.clone())),
            crate::ast::Pattern::Literal(lit) => {
                let eq = match (lit, value) {
                    (crate::ast::Literal::Int(m), Value::Int(n)) => n == m,
                    (crate::ast::Literal::Float(a), Value::Float(b)) => a == b,
                    (crate::ast::Literal::Str(s), Value::Str(v)) => v == s,
                    _ => false,
                };
                if !eq {
                    return false;
                }
            }
            crate::ast::Pattern::EnumVariant { path, args } => match value {
                Value::Enum { variant, payload }
                    if result_variants::same_variant(variant, path)
                        && payload.len() == args.len() =>
                {
                    if !match_enum_payload(args, payload, out) {
                        return false;
                    }
                }
                _ => return false,
            },
            // Tuple sub-pattern `(a, b)` destructures a tuple value of equal
            // arity, recursing each element (enum_match #9, e.g.
            // `Ok((p1, decorators))`).
            crate::ast::Pattern::Tuple(elems) => match value {
                Value::Tuple(items) if items.len() == elems.len() => {
                    if !match_enum_payload(elems, items, out) {
                        return false;
                    }
                }
                _ => return false,
            },
            // Struct-variant sub-pattern `E.V { f, g }`. The interpreter is the
            // const-eval path (mind-flow uses the RUNS path), and lacks the enum
            // decl's field_names here, so it binds each named field's sub-pattern
            // positionally in source order against the enum payload.
            crate::ast::Pattern::EnumStruct { fields, .. } => match value {
                Value::Enum { payload, .. } if payload.len() >= fields.len() => {
                    let subs: Vec<crate::ast::Pattern> =
                        fields.iter().map(|(_, p)| p.clone()).collect();
                    if !match_enum_payload(&subs, payload, out) {
                        return false;
                    }
                }
                _ => return false,
            },
        }
    }
    true
}
