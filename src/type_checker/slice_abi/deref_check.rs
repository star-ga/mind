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
//! today, so no admission below is yet a load-bearing gate. IMPLEMENTED:
//!   * region refusal (a deref / ref-take inside `region { .. }`);
//!   * `*p` / `*p = v` refusal when the operand is not a bare `&mut <struct>`
//!     parameter;
//!   * owner canonicalization in `canonical_struct_name` (crate.-prefix, so a
//!     same-named struct of a different module does not compare equal);
//!   * shadowing / reassignment refusal (a `let` / `=` / `for` var naming a
//!     `&mut` reference parameter);
//!   * caller-side `Node::Ref` refusal for a field address-of (`&r.f` /
//!     `&mut r.f`) and an element address-of (`&a[i]`) — closing the lower.rs
//!     Phase-10.7 no-op-value hole; a bare `&NAME` is untouched.
//! NOT YET (tracked; must land WITH the D4 cell-address lowering, in the SAME
//! increment that carves the admitted shape out of the blanket refusal):
//!   * `ref_target_name` real resolution (still returns None) — so owner-EXACT
//!     comparison is not yet exercised, because nothing is admitted yet;
//!   * the depth-one struct-field carve-out (`admitted_field_place` is defined
//!     but NOT called — every field address-of refuses today; the carve-out and
//!     its `&mut r.f` cell lowering land together);
//!   * `*p = v` value-owner validation;
//!   * a distinct call-result address-of refusal (`&mk()` is currently recursed
//!     as a temporary, not yet a named refusal).
//! Do NOT carve the admitted shape out of the blanket refusal until the
//! predicates above are complete AND the D4 lowering for the same shape exists
//! (root D3->D4 direction, 2026-09-13).

#![cfg(feature = "std-surface")]

use crate::ast::{FnDefData, Node, TypeAnn};
use crate::diagnostics::Diagnostic;

use super::provenance;
use super::{Env, StructFieldTypes};

const DEREF_CODE: &str = "E2028";
const DEREF_ASSIGN_CODE: &str = "E2029";
/// Unsupported reference-take form: a field address-of (`&r.f` / `&mut r.f`) or
/// an element address-of (`&a[i]`). Both are the field-first subset's eventual
/// target shapes, but they are REFUSED until the D4 cell-address lowering for
/// the same shape exists — matching the self-host emitter (main.mind), which
/// already defers `&p.f` / `&a[i]` in its `&NAME` address-of subset. Without
/// this refusal the Rust path silently lowered `&mut r.f` to the *loaded field
/// value* (lower.rs Phase 10.7 no-op wrapper), a latent miscompile whenever the
/// reference is not immediately consumed by an (E2028/E2029-refused) callee
/// deref. A bare `&NAME` (whole-variable address-of) is NOT refused here — it is
/// the self-host supported form and out of this subset's scope.
const REF_FORM_CODE: &str = "E2037";

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

/// Bare (crate.-stripped) form of a canonical struct name, for looking it up in
/// the `(struct, field)` layout map (whose keys are the declared bare names).
fn bare_struct_name(canonical: &str) -> &str {
    canonical.strip_prefix("crate.").unwrap_or(canonical)
}

/// Is `name` a declared struct (i.e. it owns at least one field in the layout
/// map)? A primitive/scalar field type (`i64`, …) is NOT, so `&mut r.scalar`
/// fails this and is refused.
fn is_declared_struct(name: &str, env: &Env) -> bool {
    let bare = bare_struct_name(name);
    env.struct_fields.keys().any(|(s, _)| s == bare)
}

/// Is `node` the admitted receiver-field place `r.f` where `f` is a depth-one
/// STRUCT-typed field of an admitted receiver with a resolvable declared owner?
/// Returns the field's canonical struct-type name when admitted; `None` (→
/// refuse) for depth-2, index/call receivers, unknown owner/layout, or a
/// scalar field. NO `idx*8` or scalar fallback — an unresolved owner refuses.
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
    // The field must have a resolvable declared type (owner/layout known) …
    let ft = provenance::field_type(receiver, field, env)?;
    let owner = canonical_struct_name(&ft)?;
    // … and that type must be a DECLARED STRUCT (identity-carrying record),
    // never a scalar/primitive field. This is what makes the cell a pointer
    // slot whose replacement is identity-preserving (see D4 lowering).
    if !is_declared_struct(&owner, env) {
        return None;
    }
    Some(owner)
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
            // Admit `*p` only when `p` is a bare `&mut <struct>` parameter whose
            // referent is a DECLARED STRUCT (a scalar `&mut i64` deref is NOT in
            // the field-first subset), outside any region.
            let referent = deref_param_referent_owner(operand, mrefs);
            let admitted = !in_region
                && referent
                    .as_deref()
                    .is_some_and(|owner| is_declared_struct(owner, env));
            if !admitted {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_CODE,
                    "dereference `*expr` is not supported here: only `*p` where `p` is a `&mut <struct>` parameter (referent a declared struct), outside any region, is admitted (deref-assign field-first subset).",
                );
            }
        }
        Node::DerefAssign {
            target,
            value,
            span,
        } => {
            // Admit `*p = v` only when: (1) not in a region, (2) `p` is a bare
            // `&mut <struct>` parameter, AND (3) `v` is a record of the EXACT
            // canonical owner `p` points to — identity place replacement, never
            // a field copy or a cross-owner / scalar store. Value-owner is
            // resolved from the seeded env (params + struct literals); a value
            // whose owner cannot be proven equal fails closed.
            // deferred: a `let`-bound value's owner is not tracked in this
            // pass's flat env, so `let n = Pair{..}; *p = n` refuses
            // conservatively — upgrade path: thread let-binding types into env.
            let referent_owner = deref_param_referent_owner(target, mrefs);
            let value_owner = provenance::expr_type(value, env)
                .as_ref()
                .and_then(canonical_struct_name);
            // The referent must be a DECLARED STRUCT and the value must be a
            // record of that EXACT canonical owner (scalar `&mut i64` stores and
            // cross-owner stores are refused).
            let owner_ok = match (&referent_owner, &value_owner) {
                (Some(r), Some(v)) => r == v && is_declared_struct(r, env),
                _ => false,
            };
            let admitted = !in_region && owner_ok;
            if !admitted {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "assignment through a dereference `*p = value` is not supported here: only `*p = v` where `p` is a `&mut <struct>` parameter (outside any region) and `v` is a record of the EXACT owner `p` points to is admitted; record-identity place replacement, not a field copy or a cross-owner/scalar store.",
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
        Node::Return {
            value: Some(v),
            span,
        } => {
            // Escape: `return p` lets a `&mut` parameter's cell address outlive
            // the frame. Refuse (a reference may only be a direct call arg).
            if ident_names_mut_ref_param(v, mrefs) {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `&mut` reference parameter may not be returned in the deref-assign subset: a reference may only appear as a direct call argument.",
                );
            }
            classify(v, env, mrefs, in_region, src, file, errs);
        }
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
            // Escape: `let q = p` aliases a `&mut` parameter into a new binding
            // the flat param map does not track. A reference may only appear as
            // a direct call argument, never let-bound. Refuse.
            if ident_names_mut_ref_param(value, mrefs) {
                refuse(
                    errs,
                    src,
                    file,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `&mut` reference parameter may not be let-bound (`let q = p`) in the deref-assign subset: a reference may only appear as a direct call argument.",
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
        // Caller-side reference-take classification (gaps 2/4/6). A field
        // address-of (`&r.f` / `&mut r.f`, any depth, struct- OR scalar-field)
        // and an element address-of (`&a[i]`) are the field-first subset's
        // eventual target shapes, but ADMISSION is coupled to the D4 cell-address
        // lowering (root D3->D4 direction): until that lowering exists for the
        // same shape, they are REFUSED — closing the lower.rs Phase-10.7 no-op
        // hole and matching the self-host emitter's deferred `&p.f` / `&a[i]`.
        // A bare `&NAME` (whole-variable address-of) is the self-host supported
        // form and stays untouched: recurse into the operand only.
        //
        // NOTE: `admitted_field_place` (depth-one struct-field discrimination)
        // is intentionally NOT called yet — every field-address-of refuses in
        // this increment, so the depth-one carve-out lands together with D4.
        Node::Ref {
            inner,
            mutable,
            span,
        } => {
            match inner.as_ref() {
                // `&mut r.f` — the ONE admitted reference shape: a mutable,
                // depth-one, STRUCT-typed field of a resolvable-owner receiver,
                // outside any region. Admitted → NO diagnostic (D4 lowers it to
                // the field's cell address). Everything else about a field
                // address-of refuses: `&r.f` (immutable — no capability
                // laundering into a writable place), `&mut r.f.a` (depth-two),
                // `&mut r.scalar` (scalar field), `&mut r.f` in a region, or an
                // unresolved owner/layout.
                Node::FieldAccess { .. } => {
                    let admitted =
                        !in_region && *mutable && admitted_field_place(inner, env).is_some();
                    if !admitted {
                        refuse(
                            errs,
                            src,
                            file,
                            *span,
                            REF_FORM_CODE,
                            "unsupported field address-of: only `&mut r.f` where `f` is a depth-one struct-typed field of a resolvable-owner receiver, outside any region, is admitted; `&r.f` (immutable), a depth-two / scalar field, an unknown owner, or a region-interior capture is refused.",
                        );
                    }
                }
                Node::IndexAccess { .. } => refuse(
                    errs,
                    src,
                    file,
                    *span,
                    REF_FORM_CODE,
                    "taking the address of an element (`&a[i]`) is not supported here: element address-of is deferred in the field-first mutable-reference subset (no element cell-address lowering); a bare `&name` (whole variable) is unaffected.",
                ),
                // `&NAME` whole-variable address-of, `&(expr)` temporaries, etc.:
                // not this subset's concern. Fall through to operand recursion so
                // a deref buried inside (`&(*p)`) is still classified.
                _ => {}
            }
            classify(inner, env, mrefs, in_region, src, file, errs);
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

/// The canonical referent-owner name a `*p`/`*p = v` target points to, when the
/// target is a bare `&mut <struct>` parameter; `None` otherwise.
fn deref_param_referent_owner(
    target: &Node,
    mrefs: &std::collections::BTreeMap<String, String>,
) -> Option<String> {
    match target {
        Node::Lit(crate::ast::Literal::Ident(name), _) => mrefs.get(name).cloned(),
        _ => None,
    }
}

/// Is `node` a bare identifier naming a `&mut` reference parameter? Used to
/// refuse the escaping forms `let q = p` and `return p`.
fn ident_names_mut_ref_param(
    node: &Node,
    mrefs: &std::collections::BTreeMap<String, String>,
) -> bool {
    matches!(node, Node::Lit(crate::ast::Literal::Ident(name), _) if mrefs.contains_key(name))
}
