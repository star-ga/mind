// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Deref-assign track D3 — admit/refuse decision for the field-only mutable
//! reference subset. This pass decides, per function, whether each `*p` / `*p =
//! v` / `&mut r.f` occurrence is the ONE supported shape (a depth-one
//! struct-typed field place reached through a `&mut Named` parameter) or an
//! unsupported form that must be REFUSED with a spanned diagnostic
//! (E2028 dereference, E2029 deref-assign). Fail-closed: a shape is admitted
//! ONLY when every condition below holds; anything unresolved or unrecognized
//! is refused.
//!
//! Target scope (root field-first ruling + FABLE_FIELD_FIRST_CONTRACT §2.3-§2.7
//! + LUNA F1-F3) — the full contract this pass is being built toward:
//!   * owner exactness — `&mut r.f` target is the field's declared struct type,
//!     canonicalized; a same-named struct of a different owner does not match;
//!   * read-only capability survival — a `&R` (immutable) reference may not
//!     drive `*p = v` / `p.f = v` / projection / cast / by-value coercion;
//!   * explicit region refusal — a deref / `&mut r.f` inside an explicit
//!     `region { .. }` is refused until lifetime provenance exists;
//!   * escape — a reference is only a parameter type / a direct call argument;
//!     never let-bound, returned, stored, captured, or formed at module level;
//!   * depth-one struct field only — `&mut h.pt.a`, `&mut xs[i].pt`,
//!     `&mut mk().pt`, `&mut h.n` (scalar field) are refused.
//!
//! STATUS — this pass is INCOMPLETE and runs ONLY additively; the blanket
//! Slice-0 refusal (type_checker/mod.rs) is what actually rejects every deref
//! today, so nothing below is yet a load-bearing admission gate. IMPLEMENTED so
//! far: the region refusal, and a refusal for a `*p` / `*p = v` whose operand is
//! not a bare `&mut <struct>` parameter. NOT YET implemented (tracked, must land
//! WITH the D4 lowering before the blanket refusal is removed): owner-exact
//! target comparison (canonical_struct_name only clones today; ref_target_name
//! returns None), caller-side `Node::Ref` classification (`&mut r.f` admit +
//! depth/index/call-result/scalar-field/escape refusals — `admitted_field_place`
//! is not yet called), `*p = v` value-owner validation, and shadowing-aware
//! parameter tracking. Do NOT carve the admitted shape out of the blanket
//! refusal until these are complete AND the D4 lowering for the same shape
//! exists (root D3->D4 direction, 2026-09-13).

#![cfg(feature = "std-surface")]

use crate::ast::{FnDefData, Node, TypeAnn};
use crate::diagnostics::Diagnostic;

use super::provenance;
use super::{Env, StructFieldTypes};

const DEREF_CODE: &str = "E2028";
const DEREF_ASSIGN_CODE: &str = "E2029";

/// Canonical (owner-qualified) name of a declared struct type annotation, or
/// `None` if it is not a nominal struct type. Applies the `crate.`-prefix
/// canonicalization used elsewhere so an intra-module owner and its
/// `crate.`-qualified form compare equal.
fn canonical_struct_name(ty: &TypeAnn) -> Option<String> {
    match ty {
        // Canonicalize the owner exactly as `provenance::field_type` does: an
        // intra-module dotted owner without the `crate.` prefix is normalized to
        // `crate.<owner>` so a same-named struct from a *different* module does
        // NOT compare equal (owner exactness). A bare (undotted) name is a
        // local struct and stays as-is.
        TypeAnn::Named(name) => Some(if name.contains('.') && !name.starts_with("crate.") {
            format!("crate.{name}")
        } else {
            name.clone()
        }),
        _ => None,
    }
}

/// The set of parameter names that are `&mut Named` references in this fn (the
/// only bindings a `*p` / `*p = v` may dereference), mapped to the canonical
/// target struct name.
fn mut_ref_params(fd: &FnDefData) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    for p in &fd.params {
        if let TypeAnn::Ref {
            mutable: true,
            target,
        } = &p.ty
        {
            if let Some(name) = canonical_struct_name(target) {
                out.insert(p.name.clone(), name);
            }
        }
    }
    out
}

/// Is `node` the admitted receiver-field place `r.f` where `f` is a depth-one
/// struct-typed field of an admitted receiver? Returns the field's canonical
/// struct-type name when admitted.
fn admitted_field_place(node: &Node, env: &Env) -> Option<String> {
    let Node::FieldAccess {
        receiver, field, ..
    } = node
    else {
        return None;
    };
    // Depth-one only: the receiver must NOT itself be a field access / index /
    // call — it is a plain place (a local, a by-value param, or `*p` of a
    // &mut param, handled by expr_type). `r.f.g` (depth two) is refused.
    if matches!(
        receiver.as_ref(),
        Node::FieldAccess { .. } | Node::IndexAccess { .. }
    ) {
        return None;
    }
    let ft = provenance::field_type(receiver, field, env)?;
    canonical_struct_name(&ft)
}

/// Report a refusal.
fn refuse(
    errs: &mut Vec<Diagnostic>,
    src: &str,
    file: Option<&str>,
    span: crate::ast::Span,
    code: &'static str,
    msg: &str,
) {
    errs.push(
        Diagnostic::error("type-check", code, msg.to_string()).with_span(
            crate::diagnostics::Span::from_offsets(src, span.start(), span.end(), file),
        ),
    );
}

/// D3 entry: classify every deref occurrence in `fd`, emitting refusals for the
/// unsupported forms. Admitted forms produce no diagnostic. (The caller keeps
/// the blanket Slice-0 refusal until the D4 carve-out; see the module note.)
pub(crate) fn check_fn(
    fd: &FnDefData,
    struct_fields: &std::rc::Rc<StructFieldTypes>,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    let mut env = Env {
        struct_fields: std::rc::Rc::clone(struct_fields),
        ..Env::default()
    };
    // Seed parameter types so `provenance::field_type`/`expr_type` can resolve a
    // receiver's declared struct type.
    for p in &fd.params {
        env.types.insert(p.name.clone(), p.ty.clone());
    }
    let mrefs = mut_ref_params(fd);
    // in_region tracks explicit `region { .. }` nesting for the region-refusal.
    walk(&fd.body, &env, &mrefs, false, src, file, errs);
}

#[allow(clippy::too_many_arguments)]
fn walk(
    stmts: &[Node],
    env: &Env,
    mrefs: &std::collections::BTreeMap<String, String>,
    in_region: bool,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    for node in stmts {
        classify(node, env, mrefs, in_region, src, file, errs);
    }
}

#[allow(clippy::too_many_arguments)]
fn classify(
    node: &Node,
    env: &Env,
    mrefs: &std::collections::BTreeMap<String, String>,
    in_region: bool,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    match node {
        // `*p` / `*p = v`: admitted only when `p` is a `&mut Named` param of
        // this fn and not inside a region.
        Node::Deref { operand, span } => {
            let admitted = !in_region && deref_operand_is_mut_ref_param(operand, mrefs);
            if !admitted {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_CODE,
                    "dereference `*expr` is not supported here: only `*p` where `p` is a `&mut <struct>` parameter, outside any region, is admitted (deref-assign field-first subset).",
                );
            }
        }
        Node::DerefAssign {
            target,
            value,
            span,
        } => {
            let admitted = !in_region && deref_operand_is_mut_ref_param(target, mrefs);
            if !admitted {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "assignment through a dereference `*p = value` is not supported here: only `*p = v` where `p` is a `&mut <struct>` parameter, outside any region, is admitted; record-identity place replacement, not a field copy.",
                );
            }
            classify(value, env, mrefs, in_region, src, file, errs);
        }
        // Recurse into children for nested occurrences. Region bodies flip
        // in_region so any deref / &mut r.f inside is refused (§2.7).
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => walk(body, env, mrefs, true, src, file, errs),
        Node::Block { stmts, .. } => walk(stmts, env, mrefs, in_region, src, file, errs),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            classify(cond, env, mrefs, in_region, src, file, errs);
            walk(then_branch, env, mrefs, in_region, src, file, errs);
            if let Some(eb) = else_branch {
                walk(eb, env, mrefs, in_region, src, file, errs);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            classify(cond, env, mrefs, in_region, src, file, errs);
            walk(body, env, mrefs, in_region, src, file, errs);
        }
        Node::For {
            var, body, span, ..
        }
        | Node::ForEach {
            var, body, span, ..
        } => {
            if mrefs.contains_key(var) {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `for` loop variable may not shadow a `&mut` reference parameter in the deref-assign subset; rename it.",
                );
            }
            walk(body, env, mrefs, in_region, src, file, errs);
        }
        Node::Return { value: Some(v), .. } => classify(v, env, mrefs, in_region, src, file, errs),
        // Shadowing / reassignment of a `&mut` parameter is refused (§2.5): the
        // lowering env is string-keyed, so a `let p = ..` / `p = ..` naming a
        // `&mut` param would make the flat param map unsound. Refuse rather than
        // add binding-identity machinery in this slice.
        Node::Let {
            name, value, span, ..
        } => {
            if mrefs.contains_key(name) {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `let` binding may not shadow a `&mut` reference parameter in the deref-assign subset; rename it.",
                );
            }
            classify(value, env, mrefs, in_region, src, file, errs);
        }
        Node::Assign { name, value, span } => {
            if mrefs.contains_key(name) {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `&mut` reference parameter may not be reassigned in the deref-assign subset.",
                );
            }
            classify(value, env, mrefs, in_region, src, file, errs);
        }
        _ => {
            // Generic child recursion for everything else (call args, binops,
            // etc.) so a deref buried anywhere is still classified. `region`
            // status does not change here (only an explicit Region body flips
            // it). `nerve_walk::for_each_child` covers fn/closure/statement
            // bodies exhaustively.
            super::super::nerve_walk::for_each_child(node, &mut |child| {
                classify(child, env, mrefs, in_region, src, file, errs)
            });
        }
    }
}

/// Is `operand` a bare reference to a `&mut Named` parameter (`*p` where `p` is
/// that param)? A `*x.y` or `*(expr)` is not this shape.
fn deref_operand_is_mut_ref_param(
    operand: &Node,
    mrefs: &std::collections::BTreeMap<String, String>,
) -> bool {
    matches!(operand, Node::Lit(crate::ast::Literal::Ident(name), _) if mrefs.contains_key(name))
}
