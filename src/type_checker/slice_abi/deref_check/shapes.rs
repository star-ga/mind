// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0

//! Node-shape predicates for the deref-assign admission check (`deref_check`): which
//! place a node denotes, whether it names a reference parameter, and whether an
//! argument is an admitted cell. Split from `deref_check.rs` for the module-size budget.

use crate::ast::Node;
use std::collections::BTreeMap;

use super::{Ctx, RefParamSigs, admitted_field_place, receiver_capability_ok};

/// The receiver of a depth-one field/index place, if `node` is one.
pub(super) fn receiver_field_receiver(node: &Node) -> Option<&Node> {
    match node {
        Node::FieldAccess { receiver, .. } | Node::IndexAccess { receiver, .. } => Some(receiver),
        _ => None,
    }
}

/// The canonical referent-owner name a `*p`/`*p = v` target points to, when the
/// target is a bare `&mut <struct>` parameter; `None` otherwise.
pub(super) fn deref_param_referent_owner(
    target: &Node,
    mrefs: &BTreeMap<String, String>,
) -> Option<String> {
    match target {
        Node::Lit(crate::ast::Literal::Ident(name), _) => mrefs.get(name).cloned(),
        _ => None,
    }
}

/// Is `node` a bare identifier naming a MUTABLE reference parameter? Only a
/// `&mut` parameter carries write capability + identity concerns, so only it is
/// subject to the escape/forwarding refusals. An immutable `&T` parameter is
/// read-only and unrestricted (pristine behaviour) — restricting it was the
/// over-refusal root found (`fn evaluate(r: &Request) { validate(r) }`).
pub(super) fn ident_names_ref_param(node: &Node, refs: &RefParamSigs) -> bool {
    // Refusal-side, so parentheses are peeled: `(p)` escapes exactly as `p` does.
    matches!(strip_parens(node), Node::Lit(crate::ast::Literal::Ident(name), _)
        if refs.get(name).is_some_and(|(mutable, _)| *mutable))
}

/// Does a `match` arm pattern bind any name that shadows a `&mut` reference
/// parameter? Walks Ident / Tuple / EnumVariant / EnumStruct sub-patterns. A
/// binder colliding with a `&mut` param name would rebind it without updating
/// the flat param table, so `*p` / forwarding inside the arm mis-stores through
/// the stale slot — refuse such an arm.
pub(super) fn pattern_binds_mut_ref(
    pat: &crate::ast::Pattern,
    mrefs: &BTreeMap<String, String>,
) -> bool {
    use crate::ast::Pattern as P;
    match pat {
        P::Ident(name) => mrefs.contains_key(name),
        P::Tuple(ps) | P::EnumVariant { args: ps, .. } => {
            ps.iter().any(|p| pattern_binds_mut_ref(p, mrefs))
        }
        P::EnumStruct { fields, .. } => fields.iter().any(|(_, p)| pattern_binds_mut_ref(p, mrefs)),
        P::Literal(_) | P::Wildcard => false,
    }
}

/// Is `receiver` a bare identifier naming a `&mut` reference parameter? Used to
/// refuse field access/assignment through a `&mut` parameter until load-before-
/// field lowering exists (a `&mut Pair` param holds a CELL address, not a record
/// address, so `p.f` would mislower). Immutable `&T` receivers are unaffected.
pub(super) fn receiver_is_mut_ref_param(receiver: &Node, ctx: &Ctx) -> bool {
    // Parentheses must not HIDE a `&mut` parameter from this refusal: the struct
    // resolver walks through `Paren`, so `(p).y = v` lowers to exactly the same
    // `store addr(p)+8` as `p.y = v` — a wrong-slot / out-of-bounds write through
    // a CELL address (audit 2026-09-16). Only the refusal side strips parens; every
    // admission check stays keyed on the bare identifier, so a wrapped form is
    // never ADMITTED.
    matches!(strip_parens(receiver), Node::Lit(crate::ast::Literal::Ident(name), _)
        if ctx.mrefs.contains_key(name))
}

/// Peel any number of `( … )` wrappers.
pub(super) fn strip_parens(mut node: &Node) -> &Node {
    while let Node::Paren(inner, _) = node {
        node = inner;
    }
    node
}

/// Is `arg` an ADMITTED argument for a callee `&mut <want>` formal? Either a
/// depth-one `&mut r.f` field address-of of owner `want` (capable receiver), or
/// a bare `&mut <want>` reference parameter forwarded. Everything else — a
/// value, a scalar, an immutable reference, a wrong owner, an element `&a[i]`,
/// or a wrapped/cast form — is NOT admitted (closes the non-reference-into-
/// `&mut`-formal hole, audit finding F1 / root case (1) respelled).
pub(super) fn arg_is_admitted_mut_ref_to_owner(
    arg: &Node,
    want: &str,
    ctx: &Ctx,
    in_region: bool,
) -> bool {
    if in_region {
        return false;
    }
    match arg {
        Node::Ref {
            mutable: true,
            inner,
            ..
        } if matches!(inner.as_ref(), Node::FieldAccess { .. }) => {
            admitted_field_place(inner, ctx.env).as_deref() == Some(want)
                && receiver_field_receiver(inner)
                    .is_some_and(|r| receiver_capability_ok(r, ctx.env))
        }
        Node::Lit(crate::ast::Literal::Ident(name), _) => {
            ctx.mrefs.get(name).map(String::as_str) == Some(want)
        }
        _ => false,
    }
}

/// Find an unconsumed reference-parameter occurrence inside an expression.
/// Only a bare identifier can be forwarded: parentheses/casts must not launder
/// a capability into an untyped call expression. A dereference consumes the
/// capability as a by-value result; its own owner/region admission is checked
/// by the `Node::Deref` arm instead of being rejected as a reference escape.
/// A call is also a value boundary: its arguments are classified recursively
/// by the call walker, while the call's return type is checked by the ordinary
/// type checker. D4 functions that return a reference are rejected at their
/// own return site, so a value-producing nested call cannot carry `p` out.
pub(super) fn contains_unconsumed_ref_param(node: &Node, refs: &RefParamSigs) -> bool {
    // A field/index projection reads a value from the referent; it does not
    // carry the reference parameter itself into the enclosing expression.
    // Keep reference-taking nodes subject to the direct-call checks above,
    // while treating ordinary value projections like dereference/call results.
    if matches!(
        node,
        Node::Deref { .. }
            | Node::Call { .. }
            | Node::FieldAccess { .. }
            | Node::IndexAccess { .. }
    ) {
        return false;
    }
    if ident_names_ref_param(node, refs) {
        return true;
    }
    let mut found = false;
    super::super::super::nerve_walk::for_each_child(node, &mut |child| {
        if contains_unconsumed_ref_param(child, refs) {
            found = true;
        }
    });
    found
}
