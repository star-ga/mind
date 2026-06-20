// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 P0f Step 2 — side-table that resolves the struct-type of
//! every `FieldAccess` receiver in a module, so lowering can emit the
//! correct `__mind_load_i64` even for receivers that aren't a plain
//! `Ident` (`a.b.c`, `foo().x`, struct-typed parameters).
//!
//! Internal review (2026-05-18) picked **β — type-checker struct
//! annotation** over the post-lowering IR-pass alternative. We implement that as a
//! lightweight pre-pass that runs once per module before lowering and
//! builds a `HashMap<Span, String>` keyed on each `FieldAccess`
//! receiver-position span. The full type-checker is not invoked; we
//! track only what `FieldAccess` lowering needs (struct-name bindings),
//! which keeps the pass O(n) over the AST with a small constant.
//!
//! Gated under `feature = "std-surface"` — non-feature builds construct
//! the empty map for free and never enter this module.

#![cfg(feature = "std-surface")]

use std::collections::{BTreeMap, HashMap};

use crate::ast::{Literal, Module, Node, Span, TypeAnn};

/// Maps `(struct_name, field_name)` → the **struct-type name of that
/// field**, for every struct field whose declared type names a known
/// struct. Built once from the module's `StructDef` nodes. Empty for any
/// field that is scalar / slice-of-scalar / not a struct.
///
/// This is what lets [`infer_struct`] chain through a nested
/// `FieldAccess` receiver: given `o.inner` where `o: Outer`, we look up
/// `(Outer, inner)` and learn the receiver resolves to `Inner`, so the
/// outer `.v` can then resolve. `BTreeMap` keeps construction
/// deterministic for byte-identity (the map is lookup-only at
/// resolution time, but we keep the deterministic container regardless).
type FieldTypes = BTreeMap<(String, String), String>;

/// The side-table produced by [`build_field_access_types`]. Maps each
/// `FieldAccess` node's span to the **struct-type name of its receiver**
/// — *not* the type of the field itself. Lowering uses the receiver-side
/// name to look up `ir.struct_defs[T]` and compute the field's 8-byte
/// offset.
pub type FieldAccessTypes = HashMap<Span, String>;

/// Walk a module and build the side-table.
///
/// Resolution rules (Step 2 scope, per internal review 2026-05-18):
///
/// - (1) **Chained access** `a.b.c`: `a`'s struct gives `b`'s type; that
///   type gives `c`'s offset. Implemented as recursive descent — every
///   level of nesting records its own entry in the table.
/// - (2) **Function return** `foo().x`: a first sub-pass collects
///   `fn name → return-type-struct-name` from `FnDef.ret_type`
///   annotations that name a known `StructDef`. Resolution looks up
///   the callee and uses that.
/// - (3) **Struct parameter** `fn read(c: Cfg) { c.max }`: when we
///   descend into a fn body, the per-fn binding map seeds with the
///   parameter list (any `Param` whose `ty` is `Named(T)` for a known
///   `StructDef` gets `param_name → T`).
///
/// - (4) **Nested struct-typed field** `a.b.c` where field `b` is itself
///   a struct: the per-field type table (built from `StructDef.fields[i].ty`)
///   lets `infer_struct` chain `a → S`, `(S, b) → Inner`, so `.c`
///   resolves against `Inner`. Both reads (`o.inner.v`) and writes
///   (`o.inner.v = x`) light up — the lowering re-lowers the inner
///   `FieldAccess` receiver to its base address and adds the field offset.
///
/// Out of scope (intentionally — Step 3 / generics):
/// - Generic / type-parameterised struct fields. Scalar fields stay
///   i64-shaped (heap-record ABI Option C, P0e); the table only records
///   fields whose declared type names a concrete known struct.
pub fn build_field_access_types(module: &Module) -> FieldAccessTypes {
    let mut types = FieldAccessTypes::new();

    // First pass — collect known struct names + fn-return mappings +
    // per-field struct types (for nested-access chaining).
    let mut struct_defs: Vec<String> = Vec::new();
    let mut field_types: FieldTypes = FieldTypes::new();
    let mut fn_returns: HashMap<String, String> = HashMap::new();
    for item in &module.items {
        match item {
            Node::StructDef { name, fields, .. } => {
                struct_defs.push(name.clone());
                // Record `(struct, field) → inner-struct-name` for every
                // field whose declared type names a struct (peeling
                // `&T` / `&mut T` / `[T]` / `&[T]`). The "is this a real
                // struct?" filter runs in the second pass below, after
                // all struct names are known, since a field's struct may
                // be declared textually later.
                for f in fields {
                    if let Some(t) = unwrap_to_named(&f.ty) {
                        field_types.insert((name.clone(), f.name.clone()), t.to_string());
                    }
                }
            }
            Node::FnDef {
                name,
                ret_type: Some(ret),
                ..
            } => {
                // Peel `&T` / `&mut T` / `[T]` / `&[T]` to the inner
                // `Named(T)` so a `-> &Cfg` return type resolves its
                // struct just like a `-> Cfg` one. Defer the "is this t
                // a struct?" check to the second pass, since StructDef
                // may appear textually below the FnDef.
                if let Some(t) = unwrap_to_named(ret) {
                    fn_returns.insert(name.clone(), t.to_string());
                }
            }
            _ => {}
        }
    }
    // Filter fn_returns to only those whose return-type is a real struct.
    fn_returns.retain(|_, t| struct_defs.iter().any(|s| s == t));
    // Likewise keep only field types that name a real struct — scalar
    // fields and slices-of-scalar drop out, so `infer_struct`'s
    // FieldAccess arm only ever returns a known struct name.
    field_types.retain(|_, t| struct_defs.iter().any(|s| s == t));

    // Second pass — walk every expression, tracking var→struct bindings.
    let mut var_to_struct: HashMap<String, String> = HashMap::new();
    for item in &module.items {
        match item {
            Node::Let { name, value, .. } => {
                walk_expr(
                    value,
                    &var_to_struct,
                    &fn_returns,
                    &struct_defs,
                    &field_types,
                    &mut types,
                );
                if let Some(t) = infer_struct(value, &var_to_struct, &fn_returns, &field_types) {
                    if struct_defs.iter().any(|s| s == &t) {
                        var_to_struct.insert(name.clone(), t);
                    }
                }
            }
            Node::Assign { name, value, .. } => {
                walk_expr(
                    value,
                    &var_to_struct,
                    &fn_returns,
                    &struct_defs,
                    &field_types,
                    &mut types,
                );
                if let Some(t) = infer_struct(value, &var_to_struct, &fn_returns, &field_types) {
                    if struct_defs.iter().any(|s| s == &t) {
                        var_to_struct.insert(name.clone(), t);
                    }
                }
            }
            Node::FnDef { params, body, .. } => {
                // Seed fn scope with module-level bindings, then add params.
                let mut fn_vars = var_to_struct.clone();
                for p in params {
                    // Peel `&T` / `&mut T` / `[T]` / `&[T]` down to the
                    // inner `Named(T)` so by-reference struct params
                    // (`c: &Cfg`) seed the same way by-value ones do.
                    if let Some(t) = unwrap_to_named(&p.ty) {
                        if struct_defs.iter().any(|s| s == t) {
                            fn_vars.insert(p.name.clone(), t.to_string());
                        }
                    }
                }
                for stmt in body {
                    walk_stmt(
                        stmt,
                        &mut fn_vars,
                        &fn_returns,
                        &struct_defs,
                        &field_types,
                        &mut types,
                    );
                }
            }
            // Items that are themselves expressions used for their side
            // effect (rare at module scope, but the catch-all in lower.rs
            // does invoke lower_expr on them).
            other => walk_expr(
                other,
                &var_to_struct,
                &fn_returns,
                &struct_defs,
                &field_types,
                &mut types,
            ),
        }
    }

    types
}

/// Walk a function-body statement, mutating `fn_vars` for inner Let/Assign.
fn walk_stmt(
    stmt: &Node,
    fn_vars: &mut HashMap<String, String>,
    fn_returns: &HashMap<String, String>,
    struct_defs: &[String],
    field_types: &FieldTypes,
    types: &mut FieldAccessTypes,
) {
    match stmt {
        Node::Let { name, value, .. } | Node::Assign { name, value, .. } => {
            walk_expr(value, fn_vars, fn_returns, struct_defs, field_types, types);
            if let Some(t) = infer_struct(value, fn_vars, fn_returns, field_types) {
                if struct_defs.iter().any(|s| s == &t) {
                    fn_vars.insert(name.clone(), t);
                }
            }
        }
        Node::Return { value: Some(v), .. } => {
            walk_expr(v, fn_vars, fn_returns, struct_defs, field_types, types);
        }
        other => walk_expr(other, fn_vars, fn_returns, struct_defs, field_types, types),
    }
}

/// Walk an expression, recording every `FieldAccess` whose receiver
/// resolves to a known struct type.
fn walk_expr(
    expr: &Node,
    vars: &HashMap<String, String>,
    fn_returns: &HashMap<String, String>,
    struct_defs: &[String],
    field_types: &FieldTypes,
    types: &mut FieldAccessTypes,
) {
    match expr {
        Node::FieldAccess { receiver, span, .. } => {
            // Recurse first so nested FieldAccess entries get recorded
            // before we look up the outer one.
            walk_expr(receiver, vars, fn_returns, struct_defs, field_types, types);
            if let Some(t) = infer_struct(receiver, vars, fn_returns, field_types) {
                if struct_defs.iter().any(|s| s == &t) {
                    types.insert(*span, t);
                }
            }
        }
        Node::Binary { left, right, .. } => {
            walk_expr(left, vars, fn_returns, struct_defs, field_types, types);
            walk_expr(right, vars, fn_returns, struct_defs, field_types, types);
        }
        Node::Logical { left, right, .. } => {
            walk_expr(left, vars, fn_returns, struct_defs, field_types, types);
            walk_expr(right, vars, fn_returns, struct_defs, field_types, types);
        }
        Node::Neg { operand, .. } | Node::Not { operand, .. } => {
            walk_expr(operand, vars, fn_returns, struct_defs, field_types, types)
        }
        Node::Paren(inner, _) => {
            walk_expr(inner, vars, fn_returns, struct_defs, field_types, types)
        }
        Node::Ref { inner, .. } => {
            walk_expr(inner, vars, fn_returns, struct_defs, field_types, types)
        }
        Node::Call { args, .. } => {
            for a in args {
                walk_expr(a, vars, fn_returns, struct_defs, field_types, types);
            }
        }
        Node::MethodCall {
            receiver,
            args,
            span,
            ..
        } => {
            walk_expr(receiver, vars, fn_returns, struct_defs, field_types, types);
            for a in args {
                walk_expr(a, vars, fn_returns, struct_defs, field_types, types);
            }
            // RFC 0005 method-as-field brick — record the receiver's
            // struct type at the MethodCall's own span so the lowering
            // arm can resolve a zero-arg accessor method (`s.len()`) to
            // the same field load the `FieldAccess` arm emits for `s.len`.
            // Mirror of the `FieldAccess` arm: lowering keys the side-table
            // on the node span and looks up the field offset in
            // `ir.struct_defs[T]`. Purely additive — method calls are absent
            // from the keystone and all of std, so no emitted bytes change.
            if let Some(t) = infer_struct(receiver, vars, fn_returns, field_types) {
                if struct_defs.iter().any(|s| s == &t) {
                    types.insert(*span, t);
                }
            }
        }
        Node::Block { stmts, .. } => {
            // Inner Let/Assign in a block can shadow — use a snapshot.
            let mut local = vars.clone();
            for stmt in stmts {
                walk_stmt(
                    stmt,
                    &mut local,
                    fn_returns,
                    struct_defs,
                    field_types,
                    types,
                );
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expr(cond, vars, fn_returns, struct_defs, field_types, types);
            let mut local = vars.clone();
            for stmt in then_branch {
                walk_stmt(
                    stmt,
                    &mut local,
                    fn_returns,
                    struct_defs,
                    field_types,
                    types,
                );
            }
            if let Some(else_b) = else_branch {
                let mut local = vars.clone();
                for stmt in else_b {
                    walk_stmt(
                        stmt,
                        &mut local,
                        fn_returns,
                        struct_defs,
                        field_types,
                        types,
                    );
                }
            }
        }
        Node::StructLit { fields, .. } => {
            for f in fields {
                walk_expr(&f.value, vars, fn_returns, struct_defs, field_types, types);
            }
        }
        // RFC 0005 P0g — `receiver.field = value`. Two records are needed
        // for a nested write (`o.inner.v = x`):
        //   * walking the receiver records the inner `o.inner` FieldAccess
        //     span, so the write arm's `lower_expr(receiver, …)` resolves
        //     `o.inner` to the inner record's base address; and
        //   * the FieldAssign's *own* span must map to the receiver's
        //     struct type (`Inner`), since the write arm's Step-2 looks up
        //     `receiver_types[FieldAssign.span]` to index the field being
        //     stored. This mirrors how a FieldAccess read keys its own
        //     span on the receiver's struct name.
        // The RHS is walked too for any field accesses it contains.
        #[cfg(feature = "std-surface")]
        Node::FieldAssign {
            receiver,
            span,
            value,
            ..
        } => {
            walk_expr(receiver, vars, fn_returns, struct_defs, field_types, types);
            if let Some(t) = infer_struct(receiver, vars, fn_returns, field_types) {
                if struct_defs.iter().any(|s| s == &t) {
                    types.insert(*span, t);
                }
            }
            walk_expr(value, vars, fn_returns, struct_defs, field_types, types);
        }
        // RFC 0010 Phase J-A: region body — same walk as Block.
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => {
            let mut local = vars.clone();
            for stmt in body {
                walk_stmt(
                    stmt,
                    &mut local,
                    fn_returns,
                    struct_defs,
                    field_types,
                    types,
                );
            }
        }
        // Other nodes either don't contain expressions or are
        // declaration-shaped (StructDef, EnumDef, …) — nothing to walk.
        _ => {}
    }
}

/// Peel borrow / slice wrappers off a type annotation and return the
/// inner `Named` type name, if any. `&Cfg` / `&mut Cfg` / `[Cfg]` /
/// `&[Cfg]` all resolve to `Cfg`; a bare `Named(Cfg)` passes through.
///
/// This is what lets a by-reference struct parameter (`c: &Cfg`) or a
/// `-> &Cfg` return type seed the same field-access machinery that a
/// by-value `c: Cfg` parameter already used. It is purely additive:
/// non-`Ref`/`Slice` annotations that aren't `Named` still return
/// `None`, so no previously-emitted byte changes.
fn unwrap_to_named(ty: &TypeAnn) -> Option<&str> {
    match ty {
        TypeAnn::Named(t) => Some(t.as_str()),
        TypeAnn::Ref { target, .. } => unwrap_to_named(target),
        TypeAnn::Slice { element, .. } => unwrap_to_named(element),
        _ => None,
    }
}

/// Infer the struct-type name of an expression, if known. Returns
/// `None` for scalar / unresolvable / not-a-struct expressions.
fn infer_struct(
    expr: &Node,
    vars: &HashMap<String, String>,
    fn_returns: &HashMap<String, String>,
    field_types: &FieldTypes,
) -> Option<String> {
    match expr {
        // The most common cases first — direct StructLit + Ident lookup.
        Node::StructLit { name, .. } => Some(name.clone()),
        Node::Lit(Literal::Ident(v), _) => vars.get(v).cloned(),
        Node::Call { callee, .. } => fn_returns.get(callee).cloned(),
        Node::Paren(inner, _) => infer_struct(inner, vars, fn_returns, field_types),
        Node::Ref { inner, .. } => infer_struct(inner, vars, fn_returns, field_types),
        // Chained access — `a.b` resolves to the struct type of field `b`
        // when that field is itself struct-typed. We first resolve the
        // receiver `a` to its struct name `S` (recursively, so deeper
        // chains `a.b.c` work), then look up `(S, b)` in the per-field
        // type table built from the `StructDef` declarations. The table
        // already peeled `&T` / `[T]` wrappers and dropped scalar fields,
        // so a hit is always a known inner struct name and a scalar field
        // (`o.tag`) returns `None` exactly as before — keeping this arm
        // purely additive (nested struct-typed fields are absent from the
        // keystone and all of std, so no currently-emitted byte changes).
        Node::FieldAccess {
            receiver, field, ..
        } => {
            let recv_struct = infer_struct(receiver, vars, fn_returns, field_types)?;
            field_types.get(&(recv_struct, field.clone())).cloned()
        }
        // BLOCKER 2 — UFCS method call bound to a `let`. `let s2 = s.grow(b)`
        // desugars in lowering to the free function `{lowercase(T)}_{method}`
        // (here `buf_grow`) with the receiver threaded first. For a later
        // `s2.field` access to resolve, `s2` must be recorded as the desugared
        // target's *return* struct type — exactly as the direct-call form
        // (`let s2 = buf_grow(s, b)`) already resolves through the `Node::Call`
        // arm above. We resolve the receiver's struct `T`, form the same
        // `{T.to_lowercase()}_{method}` name the lowering arm emits, and look
        // up its return type in `fn_returns`. A zero-arg accessor that names a
        // field of `T` (`s.len()`) is a scalar field read, not a struct value,
        // so it correctly returns `None` (the UFCS target `{t}_len` will not be
        // a registered struct-returning free function).
        Node::MethodCall {
            receiver, method, ..
        } => {
            let recv_struct = infer_struct(receiver, vars, fn_returns, field_types)?;
            let fn_name = format!("{}_{}", recv_struct.to_lowercase(), method);
            fn_returns.get(&fn_name).cloned()
        }
        _ => None,
    }
}
