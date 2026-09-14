// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Deref-assign D3/D4 — admit/refuse decision for the field-only mutable
//! reference subset. This pass is the SOLE gate under std-surface (the blanket
//! Slice-0 refusal is compiled out). It decides, per function, whether each
//! `*p` / `*p = v` / `&mut r.f` occurrence is an ADMITTED shape or an
//! unsupported form REFUSED with a spanned diagnostic (E2028 dereference, E2029
//! deref-assign, E2037 reference form). Fail-closed: a shape is admitted ONLY
//! when every condition holds; anything unresolved is refused.
//!
//! Admission is CONTEXT-SENSITIVE and CAPABILITY-AWARE (root U1 D4 rejection,
//! 2026-09-13 — four forbidden forms had compiled to artifacts):
//!
//!   * `*p` (read) / `*p = v` (store) — admitted only when `p` is a bare
//!     `&mut <struct>` PARAMETER whose referent is a DECLARED STRUCT, outside
//!     any region; for `*p = v`, `v` must be a record of the EXACT canonical
//!     owner `p` points to (identity place replacement, never a field copy /
//!     cross-owner / scalar store).
//!
//!   * `&mut r.f` (a mutable, depth-one, struct-typed field address-of) —
//!     admitted ONLY as a DIRECT CALL ARGUMENT, and only when ALL hold:
//!       (a) the ref is `&mut` (an immutable `&r.f` cannot launder into a
//!           writable place);
//!       (b) `r.f` is depth-one, struct-typed, with a resolvable declared owner
//!           (no idx*8 / scalar fallback);
//!       (c) the receiver `r` has mutable/owning capability — it is NOT an
//!           immutable `&T` reference (capability provenance);
//!       (d) the callee's corresponding parameter is `&mut <owner>` with an
//!           EXACT owner match (call-site owner exactness).
//!     A field/element address-of in ANY other position — let-bound
//!     (`let q = &mut r.f`), cast (`(&mut r.f) as i64`), returned, stored, an
//!     arithmetic/index operand, a mismatched-owner or non-`&mut` call
//!     parameter — is REFUSED. Capability/owner provenance therefore cannot
//!     survive a cast / projection / by-value / forwarding launder: such forms
//!     fail closed.
//!
//!   * escapes — a reference parameter may not be let-bound (`let q = p`),
//!     reassigned, shadowed by a let/for var, or returned.
//!
//! `&NAME` (whole-variable address-of) and `&(expr)` are NOT this subset's
//! concern and are left untouched (measured: zero real ref-take expressions in
//! main.mind + std code, so this pass never fires on keystone source).
//!
//! D4 lowering for the admitted `&mut r.f` (field cell = base + declared offset)
//! lives in eval/lower.rs + eval/field_access.rs and is coupled to this gate.

#![cfg(feature = "std-surface")]

use crate::ast::{FnDefData, Node, TypeAnn};
use crate::diagnostics::Diagnostic;

use super::provenance;
use super::{Env, StructFieldTypes};

use std::collections::BTreeMap;

const DEREF_CODE: &str = "E2028";
const DEREF_ASSIGN_CODE: &str = "E2029";
/// Unsupported reference-take form (field/element address-of that is not the
/// admitted direct-`&mut`-call-argument shape).
const REF_FORM_CODE: &str = "E2037";

/// Map of `fn name -> declared parameter types`, for validating that a
/// `&mut r.f` call argument targets a callee parameter that is `&mut <owner>`.
pub(crate) type FnParamSigs = BTreeMap<String, Vec<TypeAnn>>;

/// Every reference parameter in the current function, including immutable and
/// non-nominal references. A call-site must prove both capability and nominal
/// owner before forwarding one of these values.
type RefParamSigs = BTreeMap<String, (bool, Option<String>)>;

/// Invariant per-function context threaded through the walk (everything except
/// the region flag and the diagnostic sink).
struct Ctx<'a> {
    env: &'a Env,
    mrefs: &'a BTreeMap<String, String>,
    refs: &'a RefParamSigs,
    fn_sigs: &'a FnParamSigs,
    src: &'a str,
    file: Option<&'a str>,
}

/// Collect `fn name -> param types` for every function in `items` (recursively,
/// so nested definitions are covered). Used to check call-site reference owner.
pub(crate) fn fn_param_sigs(items: &[Node]) -> FnParamSigs {
    fn walk(items: &[Node], out: &mut FnParamSigs) {
        for item in items {
            match item {
                Node::FnDef(fd, _) => {
                    out.insert(
                        fd.name.clone(),
                        fd.params.iter().map(|p| p.ty.clone()).collect(),
                    );
                    walk(&fd.body, out);
                }
                Node::Block { stmts, .. } => walk(stmts, out),
                _ => {}
            }
        }
    }
    let mut out = FnParamSigs::new();
    walk(items, &mut out);
    out
}

/// Canonical (owner-qualified) name of a declared struct type annotation, or
/// `None` if it is not a nominal struct type. An intra-module dotted owner
/// without the `crate.` prefix is normalized to `crate.<owner>` so a same-named
/// struct from a *different* module does not compare equal (owner exactness). A
/// bare (undotted) name is a local struct and stays as-is — which is sound
/// because MIND's transparent module namespace forbids two same-name structs in
/// one compilation (a duplicate is rejected before this pass matters).
fn canonical_struct_name(ty: &TypeAnn) -> Option<String> {
    match ty {
        TypeAnn::Named(name) => Some(if name.contains('.') && !name.starts_with("crate.") {
            format!("crate.{name}")
        } else {
            name.clone()
        }),
        _ => None,
    }
}

/// Bare (crate.-stripped) form of a canonical struct name, for looking it up in
/// the `(struct, field)` layout map (whose keys are the declared bare names).
fn bare_struct_name(canonical: &str) -> &str {
    canonical.strip_prefix("crate.").unwrap_or(canonical)
}

/// Is `name` a declared struct (owns at least one field in the layout map)? A
/// primitive/scalar field type (`i64`, …) is NOT, so `&mut r.scalar` fails.
fn is_declared_struct(name: &str, env: &Env) -> bool {
    let bare = bare_struct_name(name);
    env.struct_fields.keys().any(|(s, _)| s == bare)
}

/// The set of parameter names that are `&mut Named` references in this fn,
/// mapped to the canonical target struct name.
fn mut_ref_params(fd: &FnDefData) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
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

fn ref_params(fd: &FnDefData) -> RefParamSigs {
    let mut out = BTreeMap::new();
    for p in &fd.params {
        if let TypeAnn::Ref { mutable, target } = &p.ty {
            out.insert(p.name.clone(), (*mutable, canonical_struct_name(target)));
        }
    }
    out
}

/// The admitted receiver-field place `r.f`: a depth-one, STRUCT-typed field of a
/// resolvable-owner receiver. Returns the field's canonical struct-type name, or
/// `None` (→ refuse) for depth-2, index/call receivers, unknown owner/layout, or
/// a scalar field. NO idx*8 / scalar fallback.
fn admitted_field_place(node: &Node, env: &Env) -> Option<String> {
    let Node::FieldAccess {
        receiver, field, ..
    } = node
    else {
        return None;
    };
    if matches!(
        receiver.as_ref(),
        Node::FieldAccess { .. } | Node::IndexAccess { .. }
    ) {
        return None;
    }
    let ft = provenance::field_type(receiver, field, env)?;
    let owner = canonical_struct_name(&ft)?;
    if !is_declared_struct(&owner, env) {
        return None;
    }
    Some(owner)
}

/// Does the receiver of a `&mut r.f` have mutable/owning capability? True for a
/// by-value struct place or a `&mut T` receiver; FALSE for an immutable `&T`
/// reference (capability provenance — no laundering `&H` into a writable field)
/// and for an unresolvable receiver (fail closed).
fn receiver_capability_ok(receiver: &Node, env: &Env) -> bool {
    match receiver {
        Node::Lit(crate::ast::Literal::Ident(name), _) => match env.types.get(name) {
            // An immutable reference receiver cannot yield a writable field.
            Some(TypeAnn::Ref { mutable: false, .. }) => false,
            // A `&mut T` receiver, or a by-value owning place of known type.
            Some(_) => true,
            // Unknown binding (e.g. an untracked local) — fail closed.
            None => false,
        },
        // Any non-identifier receiver (`*p`, a call result, etc.) is not an
        // owning place we can prove capability for here — fail closed.
        _ => false,
    }
}

/// Is the `arg_index`-th parameter of `callee` a `&mut <owner>` matching the
/// borrowed field's canonical owner? Unknown callee / arity / non-`&mut` /
/// wrong-owner → false (fail closed). This is call-site owner exactness.
fn callee_param_is_mut_owner(
    fn_sigs: &FnParamSigs,
    callee: &str,
    arg_index: usize,
    field_owner: &str,
) -> bool {
    let Some(params) = fn_sigs.get(callee) else {
        return false;
    };
    let Some(ty) = params.get(arg_index) else {
        return false;
    };
    match ty {
        TypeAnn::Ref {
            mutable: true,
            target,
        } => canonical_struct_name(target).as_deref() == Some(field_owner),
        _ => false,
    }
}

/// Report a refusal.
fn refuse(
    errs: &mut Vec<Diagnostic>,
    ctx: &Ctx,
    span: crate::ast::Span,
    code: &'static str,
    msg: &str,
) {
    errs.push(
        Diagnostic::error("type-check", code, msg.to_string()).with_span(
            crate::diagnostics::Span::from_offsets(ctx.src, span.start(), span.end(), ctx.file),
        ),
    );
}

/// D3/D4 entry: classify every deref / reference occurrence in `fd`.
pub(crate) fn check_fn(
    fd: &FnDefData,
    struct_fields: &std::rc::Rc<StructFieldTypes>,
    fn_sigs: &FnParamSigs,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    let mut env = Env {
        struct_fields: std::rc::Rc::clone(struct_fields),
        ..Env::default()
    };
    for p in &fd.params {
        env.types.insert(p.name.clone(), p.ty.clone());
    }
    let mrefs = mut_ref_params(fd);
    let refs = ref_params(fd);
    let ctx = Ctx {
        env: &env,
        mrefs: &mrefs,
        refs: &refs,
        fn_sigs,
        src,
        file,
    };
    walk(&fd.body, &ctx, false, errs);
}

fn walk(stmts: &[Node], ctx: &Ctx, in_region: bool, errs: &mut Vec<Diagnostic>) {
    for node in stmts {
        classify(node, ctx, in_region, errs);
    }
}

fn classify(node: &Node, ctx: &Ctx, in_region: bool, errs: &mut Vec<Diagnostic>) {
    match node {
        // `*p` (read): admitted only when `p` is a bare `&mut <struct>` param
        // whose referent is a declared struct, outside any region.
        Node::Deref { operand, span } => {
            let referent = deref_param_referent_owner(operand, ctx.mrefs);
            let admitted = !in_region
                && referent
                    .as_deref()
                    .is_some_and(|owner| is_declared_struct(owner, ctx.env));
            if !admitted {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_CODE,
                    "dereference `*expr` is not supported here: only `*p` where `p` is a `&mut <struct>` parameter (referent a declared struct), outside any region, is admitted (deref-assign field-first subset).",
                );
            }
        }
        // `*p = v` (store): identity place replacement. Referent a declared
        // struct; value a record of the EXACT canonical owner.
        Node::DerefAssign {
            target,
            value,
            span,
        } => {
            let referent_owner = deref_param_referent_owner(target, ctx.mrefs);
            let value_owner = provenance::expr_type(value, ctx.env)
                .as_ref()
                .and_then(canonical_struct_name);
            let owner_ok = match (&referent_owner, &value_owner) {
                (Some(r), Some(v)) => r == v && is_declared_struct(r, ctx.env),
                _ => false,
            };
            let admitted = !in_region && owner_ok;
            if !admitted {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "assignment through a dereference `*p = value` is not supported here: only `*p = v` where `p` is a `&mut <struct>` parameter (outside any region) and `v` is a record of the EXACT owner `p` points to is admitted; record-identity place replacement, not a field copy or a cross-owner/scalar store.",
                );
            }
            classify(value, ctx, in_region, errs);
        }
        // A call: the ONLY position where a `&mut r.f` borrow is admitted. Each
        // reference argument is validated against the callee's parameter here;
        // any reference reaching the standalone `Node::Ref` arm below is NOT a
        // direct call argument and is refused.
        Node::Call { callee, args, .. } => {
            for (i, arg) in args.iter().enumerate() {
                match arg {
                    Node::Lit(crate::ast::Literal::Ident(name), span)
                        if ctx.refs.contains_key(name) =>
                    {
                        let (mutable, owner) = ctx.refs.get(name).expect("reference checked");
                        let admitted = *mutable
                            && !in_region
                            && owner.as_deref().is_some_and(|o| {
                                callee_param_is_mut_owner(ctx.fn_sigs, callee, i, o)
                            });
                        if !admitted {
                            refuse(
                                errs,
                                ctx,
                                *span,
                                REF_FORM_CODE,
                                "unsupported reference argument: a reference parameter may only be forwarded as a mutable, nominal `&mut <owner>` to a known callee parameter of the EXACT same `&mut <owner>` type; scalar, immutable, wrong-owner, non-reference, or unknown-signature calls are refused.",
                            );
                        }
                    }
                    Node::Ref {
                        mutable,
                        inner,
                        span,
                    } if matches!(
                        inner.as_ref(),
                        Node::FieldAccess { .. } | Node::IndexAccess { .. }
                    ) =>
                    {
                        let owner = admitted_field_place(inner, ctx.env);
                        let admitted = !in_region
                            && *mutable
                            && owner.as_deref().is_some_and(|o| {
                                receiver_field_receiver(inner)
                                    .is_some_and(|r| receiver_capability_ok(r, ctx.env))
                                    && callee_param_is_mut_owner(ctx.fn_sigs, callee, i, o)
                            });
                        if !admitted {
                            refuse(
                                errs,
                                ctx,
                                *span,
                                REF_FORM_CODE,
                                "unsupported reference argument: only `&mut r.f` — a mutable, depth-one, struct-typed field of an owning/`&mut` receiver, passed to a callee parameter of the EXACT `&mut <owner>` type — is admitted; an immutable receiver, a scalar/depth-two/element field, an owner mismatch, or a non-reference callee parameter is refused.",
                            );
                        }
                        // Recurse into the receiver for any nested deref.
                        if let Some(r) = receiver_field_receiver(inner) {
                            classify(r, ctx, in_region, errs);
                        }
                    }
                    _ if contains_unconsumed_ref_param(arg, ctx.refs) => {
                        refuse(
                            errs,
                            ctx,
                            arg.span(),
                            REF_FORM_CODE,
                            "unsupported reference argument: a reference parameter must be a bare identifier at a known exact mutable-reference formal; parentheses, casts, nested expressions, and unknown signatures are refused.",
                        );
                    }
                    _ => classify(arg, ctx, in_region, errs),
                }
            }
        }
        // Region bodies flip in_region so any deref / &mut r.f inside is refused.
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => walk(body, ctx, true, errs),
        Node::Block { stmts, .. } => walk(stmts, ctx, in_region, errs),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            classify(cond, ctx, in_region, errs);
            walk(then_branch, ctx, in_region, errs);
            if let Some(eb) = else_branch {
                walk(eb, ctx, in_region, errs);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            classify(cond, ctx, in_region, errs);
            walk(body, ctx, in_region, errs);
        }
        Node::For {
            var, body, span, ..
        }
        | Node::ForEach {
            var, body, span, ..
        } => {
            if ctx.refs.contains_key(var) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `for` loop variable may not shadow a reference parameter in the deref-assign subset; rename it.",
                );
            }
            walk(body, ctx, in_region, errs);
        }
        Node::Return {
            value: Some(v),
            span,
        } => {
            if contains_unconsumed_ref_param(v, ctx.refs) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a reference parameter may not be returned in the deref-assign subset: a reference may only appear as a direct call argument.",
                );
            }
            classify(v, ctx, in_region, errs);
        }
        Node::Let {
            name, value, span, ..
        } => {
            if ctx.refs.contains_key(name) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `let` binding may not shadow a reference parameter in the deref-assign subset; rename it.",
                );
            }
            if contains_unconsumed_ref_param(value, ctx.refs) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a reference parameter may not be let-bound (`let q = p`) in the deref-assign subset: a reference may only appear as a direct call argument.",
                );
            }
            classify(value, ctx, in_region, errs);
        }
        Node::Assign { name, value, span } => {
            if ctx.refs.contains_key(name) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a reference parameter may not be reassigned in the deref-assign subset.",
                );
            }
            if contains_unconsumed_ref_param(value, ctx.refs) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a reference parameter may not be assigned into another binding in the deref-assign subset: a reference may only appear as a direct call argument.",
                );
            }
            classify(value, ctx, in_region, errs);
        }
        // A reference-take reaching here is NOT a direct call argument (the Call
        // arm handles those). A field/element address-of in any such position —
        // let value, cast operand, return, arithmetic, index — is refused; this
        // closes the let-bind / cast launder holes. `&NAME` (whole-variable) and
        // `&(expr)` are out of this subset's scope: recurse only.
        Node::Ref { inner, span, .. } => {
            if matches!(
                inner.as_ref(),
                Node::FieldAccess { .. } | Node::IndexAccess { .. }
            ) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    REF_FORM_CODE,
                    "a field/element address-of (`&mut r.f`, `&a[i]`) is only admitted as a DIRECT call argument in the deref-assign subset; it may not be let-bound, cast, returned, stored, or used in an expression (capability/owner provenance would be laundered).",
                );
            }
            classify(inner, ctx, in_region, errs);
        }
        _ => {
            super::super::nerve_walk::for_each_child(node, &mut |child| {
                classify(child, ctx, in_region, errs)
            });
        }
    }
}

/// The receiver of a depth-one field/index place, if `node` is one.
fn receiver_field_receiver(node: &Node) -> Option<&Node> {
    match node {
        Node::FieldAccess { receiver, .. } | Node::IndexAccess { receiver, .. } => Some(receiver),
        _ => None,
    }
}

/// The canonical referent-owner name a `*p`/`*p = v` target points to, when the
/// target is a bare `&mut <struct>` parameter; `None` otherwise.
fn deref_param_referent_owner(target: &Node, mrefs: &BTreeMap<String, String>) -> Option<String> {
    match target {
        Node::Lit(crate::ast::Literal::Ident(name), _) => mrefs.get(name).cloned(),
        _ => None,
    }
}

/// Is `node` a bare identifier naming a reference parameter? Used to refuse
/// the escaping forms `let q = p` and `return p` for either capability.
fn ident_names_ref_param(node: &Node, refs: &RefParamSigs) -> bool {
    matches!(node, Node::Lit(crate::ast::Literal::Ident(name), _) if refs.contains_key(name))
}

/// Find an unconsumed reference-parameter occurrence inside an expression.
/// Only a bare identifier can be forwarded: parentheses/casts must not launder
/// a capability into an untyped call expression. A dereference consumes the
/// capability as a by-value result; its own owner/region admission is checked
/// by the `Node::Deref` arm instead of being rejected as a reference escape.
fn contains_unconsumed_ref_param(node: &Node, refs: &RefParamSigs) -> bool {
    if matches!(node, Node::Deref { .. }) {
        return false;
    }
    if ident_names_ref_param(node, refs) {
        return true;
    }
    let mut found = false;
    super::super::nerve_walk::for_each_child(node, &mut |child| {
        if contains_unconsumed_ref_param(child, refs) {
            found = true;
        }
    });
    found
}
