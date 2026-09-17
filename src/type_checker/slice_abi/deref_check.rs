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
//!       (c) the receiver `r` is a BY-VALUE owning place — NOT a reference of
//!           any kind: an immutable `&T` can't launder write capability, and a
//!           `&mut T` param holds a CELL address so `addr(r)+offset` would
//!           OOB-store (load-before-field is unimplemented);
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

use std::collections::{BTreeMap, BTreeSet};

mod shapes;
use shapes::{
    arg_is_admitted_mut_ref_to_owner, contains_unconsumed_ref_param, deref_param_referent_owner,
    pattern_binds_mut_ref, receiver_field_receiver, receiver_is_mut_ref_param,
};

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
    /// Which callee parameters are CELLS (see `crate::eval::deref_cells`). Every
    /// deref-assign restriction below applies to cell parameters only; a `&mut`
    /// parameter that is never dereferenced keeps main's pointer semantics.
    cells: &'a crate::eval::deref_cells::CellParams,
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
            // A REFERENCE receiver (immutable OR mutable) cannot yield an
            // admitted field cell: an immutable `&T` can't launder write
            // capability, and a `&mut T` PARAMETER holds a CELL address (a
            // pointer to the record), so `&mut recv.f` -> addr(recv)+offset is a
            // wrong-slot / out-of-bounds store — the record needs load-before-
            // field, which is unimplemented (same hazard the direct `p.f` F2
            // refusal guards). Admit ONLY a by-value owning nominal place.
            Some(TypeAnn::Ref { .. }) => false,
            Some(TypeAnn::Named(_)) => true,
            // Other by-value types (scalars, generics, unknown binding) — fail closed.
            _ => false,
        },
        // Any non-identifier receiver (`*p`, a call result, etc.) is not an
        // owning place we can prove capability for here — fail closed.
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
    cells: &crate::eval::deref_cells::CellParams,
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
    // Only CELL parameters (dereferenced, or forwarded to a cell formal) are subject
    // to the deref-assign rules. A `&mut` parameter this function never dereferences
    // is a plain POINTER exactly as on main 5724a6ee, where `p.x`, `p.x = v`,
    // forwarding, `let q = p` and `return p` all compile and run correctly; refusing
    // them regressed eight measured main-correct programs (2026-09-16).
    let cell_names: BTreeSet<String> = cells
        .get(&fd.name)
        .map(|idxs| {
            idxs.iter()
                .filter_map(|i| fd.params.get(*i).map(|p| p.name.clone()))
                .collect()
        })
        .unwrap_or_default();
    let mrefs: BTreeMap<String, String> = mut_ref_params(fd)
        .into_iter()
        .filter(|(name, _)| cell_names.contains(name))
        .collect();
    let refs: RefParamSigs = ref_params(fd)
        .into_iter()
        .filter(|(name, (mutable, _))| !*mutable || cell_names.contains(name))
        .collect();
    let ctx = Ctx {
        env: &env,
        mrefs: &mrefs,
        refs: &refs,
        fn_sigs,
        cells,
        src,
        file,
    };
    // A CELL parameter must not be reachable from another module: callers there run
    // their own inference and would pass a POINTER where this body loads a cell. A
    // non-`pub` function is not callable across modules (E2003); a `pub` one is, so
    // it may not use cells. (New capability only — `*p` did not parse on main.)
    if fd.is_pub && !cell_names.is_empty() {
        let span = fd
            .body
            .first()
            .map(Node::span)
            .unwrap_or_else(|| crate::ast::Span::new(0, 0));
        refuse(
            errs,
            &ctx,
            span,
            DEREF_ASSIGN_CODE,
            "a `pub` function may not dereference a `&mut` parameter (`*p` / `*p = v`) in the deref-assign subset: callers in other modules would pass a plain reference, not the place cell this body replaces through. Keep the dereferencing function private and call it from this module.",
        );
    }
    // A MUTABLE reference return is an escape even when its expression is a tail
    // value or a call: those forms do not pass through the explicit
    // `Node::Return` escape check, and call summaries are too weak to prove
    // lifetime safety across cycles. Refuse a declared `&mut` result up front.
    // An IMMUTABLE `&T` return is read-only and pristine-valid (identity on a
    // borrow, e.g. `fn f(p: &Point) -> &Point { return p }`) — it must NOT be
    // refused (immutable refs are unrestricted; over-refusing returns was an
    // audit finding). Only `&mut` results are refused here.
    // Scoped to functions WITH cell parameters: a pointer-only `fn f(p: &mut Pair) ->
    // &mut Pair { return p }` compiles and runs on main and stays admitted.
    if !cell_names.is_empty()
        && matches!(
            fd.ret_type.as_ref(),
            Some(TypeAnn::Ref { mutable: true, .. })
        )
    {
        let span = fd
            .body
            .first()
            .map(Node::span)
            .unwrap_or_else(|| crate::ast::Span::new(0, 0));
        refuse(
            errs,
            &ctx,
            span,
            REF_FORM_CODE,
            "reference-returning functions are outside the D4 deref-assign subset: references may be forwarded only as direct arguments to known exact mutable-reference formals, never returned.",
        );
    }
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
        //
        // deferred: no LIFETIME check on `v`. Identity replacement stores the record's
        // address, so a record allocated inside a `region { }` and passed (via a
        // by-value parameter) into a store whose referent outlives the region would
        // dangle once the region frees. Unreachable today — measured 2026-09-16, any
        // `region` containing a struct literal fails at mlir-opt ("redefinition of SSA
        // value") on main 5724a6ee, so no such program builds. Upgrade path: the fix
        // that makes region+struct lowering compile MUST land with a refusal of
        // region-local records flowing into a `&mut`-formal call (or an escape analysis
        // over by-value record params); do not fix one without the other.
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
        // A call: the only position where a mutable reference may appear. The
        // check is FORMAL-keyed (not argument-shape-keyed): for each argument we
        // look at what the callee's parameter DENOTES.
        //   * formal `&mut <O>`  -> the argument MUST be an admitted `&mut r.f`
        //     (owner O, capable receiver) or a bare `&mut O` parameter; anything
        //     else (a value, scalar, immutable reference, wrong owner, element
        //     `&a[i]`, wrapped/cast) is refused (audit finding F1 / root case (1): a
        //     non-reference argument into a `&mut` formal is an arbitrary write).
        //   * any other formal (`&T` immutable / value / unknown callee) imposes
        //     NO restriction on immutable references or values -- pristine
        //     behaviour (restricting immutable `&T` forwarding was root's
        //     over-refusal). Only a MUTABLE reference parameter or a field
        //     address-of may not escape here; both travel solely to a `&mut O`
        //     formal, handled above.
        Node::Call { callee, args, .. } => {
            for (i, arg) in args.iter().enumerate() {
                let formal = ctx.fn_sigs.get(callee).and_then(|params| params.get(i));
                let cell_formal = crate::eval::deref_cells::is_cell(ctx.cells, callee, i);
                if let (
                    true,
                    Some(TypeAnn::Ref {
                        mutable: true,
                        target,
                    }),
                ) = (cell_formal, formal)
                {
                    let admitted = canonical_struct_name(target)
                        .as_deref()
                        .is_some_and(|o| arg_is_admitted_mut_ref_to_owner(arg, o, ctx, in_region));
                    if !admitted {
                        refuse(
                            errs,
                            ctx,
                            arg.span(),
                            REF_FORM_CODE,
                            "unsupported argument at a `&mut <owner>` formal: only `&mut r.f` (that owner, owning/`&mut` receiver) or a bare `&mut <owner>` parameter is admitted; a value, scalar, immutable reference, wrong owner, element address, or wrapped/cast argument would forge a writable place and is refused.",
                        );
                    }
                    if let Node::Ref { inner, .. } = arg {
                        if let Some(r) = receiver_field_receiver(inner) {
                            classify(r, ctx, in_region, errs);
                        }
                    } else {
                        classify(arg, ctx, in_region, errs);
                    }
                    continue;
                }
                // A field/element address-of passed to a NON-cell formal is main's
                // pointer argument (`set(&mut h.pt)` with `p.x = 5` inside returns 5 on
                // main) — no longer refused; its operand is still classified.
                match arg {
                    _ if contains_unconsumed_ref_param(arg, ctx.refs) => {
                        refuse(
                            errs,
                            ctx,
                            arg.span(),
                            REF_FORM_CODE,
                            "a `&mut` reference parameter may only be forwarded as a bare identifier to a known callee parameter of the EXACT `&mut <owner>` type; a non-`&mut`-owner formal, an unknown signature, or a wrapped/cast form is refused.",
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
            if ctx.mrefs.contains_key(var) {
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
            if ctx.mrefs.contains_key(name) {
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
            if ctx.mrefs.contains_key(name) {
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
        // F2: field access / assignment THROUGH a `&mut` parameter is refused
        // until load-before-field lowering exists (a `&mut Pair` param holds a
        // CELL address, not a record address, so `p.f` / `p.f = v` mislower to
        // the wrong slot). Immutable `&T` receivers are unaffected (a `&Point`
        // field read stays admitted); non-`&mut`-param receivers recurse below.
        Node::FieldAccess { receiver, span, .. } if receiver_is_mut_ref_param(receiver, ctx) => {
            refuse(
                errs,
                ctx,
                *span,
                DEREF_CODE,
                "field access `p.f` through a `&mut` parameter is not supported yet: dereference the parameter first (`*p`); field-cell addressing for `&mut` receivers is unimplemented.",
            );
        }
        Node::FieldAssign {
            receiver,
            value,
            span,
            ..
        } => {
            if receiver_is_mut_ref_param(receiver, ctx) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "field assignment `p.f = v` through a `&mut` parameter is not supported yet: dereference the parameter first (`*p`).",
                );
            } else {
                classify(receiver, ctx, in_region, errs);
            }
            if contains_unconsumed_ref_param(value, ctx.refs) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `&mut` reference parameter may not be stored into a field (escape): a reference may only appear as a direct call argument.",
                );
            }
            classify(value, ctx, in_region, errs);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            span,
        } => {
            classify(receiver, ctx, in_region, errs);
            classify(index, ctx, in_region, errs);
            if contains_unconsumed_ref_param(value, ctx.refs) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `&mut` reference parameter may not be stored into an array element (escape): a reference may only appear as a direct call argument.",
                );
            }
            classify(value, ctx, in_region, errs);
        }
        // F4: a method-call argument is not a checkable exact `&mut <owner>`
        // formal (UFCS resolution is not modeled here), so a `&mut` parameter
        // may not be forwarded through one; refuse fail-closed. Receiver and
        // value args are classified normally.
        Node::MethodCall {
            receiver,
            args,
            span,
            ..
        } => {
            classify(receiver, ctx, in_region, errs);
            for arg in args {
                if contains_unconsumed_ref_param(arg, ctx.refs) {
                    refuse(
                        errs,
                        ctx,
                        *span,
                        REF_FORM_CODE,
                        "a `&mut` reference parameter may not be forwarded through a method call: a reference may only be a direct argument to a known exact `&mut <owner>` free-function formal.",
                    );
                }
                classify(arg, ctx, in_region, errs);
            }
        }
        // F3: a tuple-`let` binder may not shadow a `&mut` parameter (the flat
        // param table the `*p` admission consults is not updated by a rebind, so
        // the shadow would make `*p` mis-store through the stale param slot). The
        // `match`-arm-binder analogue is handled in the `Node::Match` arm below.
        Node::LetTuple {
            names, value, span, ..
        } => {
            if names.iter().any(|n| ctx.mrefs.contains_key(n)) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a tuple-`let` binder may not shadow a `&mut` reference parameter in the deref-assign subset; rename it.",
                );
            }
            if contains_unconsumed_ref_param(value, ctx.refs) {
                refuse(
                    errs,
                    ctx,
                    *span,
                    DEREF_ASSIGN_CODE,
                    "a `&mut` reference parameter may not be tuple-let-bound (escape): a reference may only appear as a direct call argument.",
                );
            }
            classify(value, ctx, in_region, errs);
        }
        // F3 (match analogue): a `match` arm-pattern binder that shadows a `&mut`
        // parameter is refused. A binder rebinds the param name, but the flat
        // param table the `*p` / forwarding admission consults is NOT updated, so
        // a `*p` / `*p = v` / forward inside the arm mis-stores or launders
        // through the stale param slot (verified: bare-ident, tuple, and
        // enum-payload binders all do). Refusing the whole match covers the read
        // (Deref), store (DerefAssign) and alias (forward) facets at once.
        Node::Match {
            scrutinee,
            arms,
            span,
        } => {
            classify(scrutinee, ctx, in_region, errs);
            for arm in arms {
                if pattern_binds_mut_ref(&arm.pattern, ctx.mrefs) {
                    refuse(
                        errs,
                        ctx,
                        *span,
                        DEREF_ASSIGN_CODE,
                        "a `match` arm pattern may not bind a name that shadows a `&mut` reference parameter in the deref-assign subset; rename the binder.",
                    );
                }
                if let Some(g) = &arm.guard {
                    classify(g, ctx, in_region, errs);
                }
                classify(&arm.body, ctx, in_region, errs);
            }
        }

        // Outside a direct cell-formal argument, `&r.f` / `&mut r.f` / `&a[i]` is main's
        // POINTER value (`let r = &mut h.pt; r.x = 6` gives 6 on main) and is not
        // refused. It can never become a cell: the only cell producer is the admitted
        // direct argument in the Call arm, and a bound or cast reference passed to a
        // cell formal is refused there. The operand is still classified (so `&mut p.x`
        // through a CELL `p` is refused by the F2 arm).
        Node::Ref { inner, .. } => {
            classify(inner, ctx, in_region, errs);
        }
        // A NESTED function is its own scope with its own parameters: classifying its
        // body under THIS function's `&mut` parameter table would admit or refuse by
        // the wrong names (audit 2026-09-16: `fn outer(p: &mut Pair, ..) { fn inner(p:
        // i64) { replace(p, new) } }` admitted `replace(p, new)` against the OUTER `p`).
        // The nested body gets its own `check_fn` pass from the type checker's
        // per-FnDef recursion, whose deref diagnostics are kept (see the body-errors
        // whitelist in `type_checker::mod`).
        Node::FnDef(..) => {}
        _ => {
            super::super::nerve_walk::for_each_child(node, &mut |child| {
                classify(child, ctx, in_region, errs)
            });
        }
    }
}
