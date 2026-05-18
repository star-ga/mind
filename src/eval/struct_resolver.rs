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
//! Multi-LLM consensus on 2026-05-18 (grok-4.3 / glm-5.1 / mistral-large
//! 3/3 unanimous) picked **β — type-checker struct annotation** over
//! the post-lowering IR-pass alternative. We implement that as a
//! lightweight pre-pass that runs once per module before lowering and
//! builds a `HashMap<Span, String>` keyed on each `FieldAccess`
//! receiver-position span. The full type-checker is not invoked; we
//! track only what `FieldAccess` lowering needs (struct-name bindings),
//! which keeps the pass O(n) over the AST with a small constant.
//!
//! Gated under `feature = "std-surface"` — non-feature builds construct
//! the empty map for free and never enter this module.

#![cfg(feature = "std-surface")]

use std::collections::HashMap;

use crate::ast::{Literal, Module, Node, Span, TypeAnn};

/// The side-table produced by [`build_field_access_types`]. Maps each
/// `FieldAccess` node's span to the **struct-type name of its receiver**
/// — *not* the type of the field itself. Lowering uses the receiver-side
/// name to look up `ir.struct_defs[T]` and compute the field's 8-byte
/// offset.
pub type FieldAccessTypes = HashMap<Span, String>;

/// Walk a module and build the side-table.
///
/// Resolution rules (Step 2 scope, per multi-LLM consensus 2026-05-18):
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
/// Out of scope for Step 2 (intentionally — Step 3 / generics):
/// - Fields *of* struct types being themselves structs that we'd then
///   want to load further. Today every struct field is i64-shaped
///   (heap-record ABI Option C, P0e). When std-surface adds nested
///   struct fields, the recursion in (1) needs to consult
///   `StructDef.fields[i].ty` to find the inner struct's name —
///   straightforward extension, but not needed for `std.vec`'s shape.
pub fn build_field_access_types(module: &Module) -> FieldAccessTypes {
    let mut types = FieldAccessTypes::new();

    // First pass — collect known struct names + fn-return mappings.
    let mut struct_defs: Vec<String> = Vec::new();
    let mut fn_returns: HashMap<String, String> = HashMap::new();
    for item in &module.items {
        match item {
            Node::StructDef { name, .. } => struct_defs.push(name.clone()),
            Node::FnDef { name, ret_type, .. } => {
                if let Some(TypeAnn::Named(t)) = ret_type {
                    // Defer "is this t a struct?" check to the second
                    // pass, since StructDef may appear textually below
                    // the FnDef. Just record the candidate name.
                    fn_returns.insert(name.clone(), t.clone());
                }
            }
            _ => {}
        }
    }
    // Filter fn_returns to only those whose return-type is a real struct.
    fn_returns.retain(|_, t| struct_defs.iter().any(|s| s == t));

    // Second pass — walk every expression, tracking var→struct bindings.
    let mut var_to_struct: HashMap<String, String> = HashMap::new();
    for item in &module.items {
        match item {
            Node::Let { name, value, .. } => {
                walk_expr(value, &var_to_struct, &fn_returns, &struct_defs, &mut types);
                if let Some(t) = infer_struct(value, &var_to_struct, &fn_returns) {
                    if struct_defs.iter().any(|s| s == &t) {
                        var_to_struct.insert(name.clone(), t);
                    }
                }
            }
            Node::Assign { name, value, .. } => {
                walk_expr(value, &var_to_struct, &fn_returns, &struct_defs, &mut types);
                if let Some(t) = infer_struct(value, &var_to_struct, &fn_returns) {
                    if struct_defs.iter().any(|s| s == &t) {
                        var_to_struct.insert(name.clone(), t);
                    }
                }
            }
            Node::FnDef { params, body, .. } => {
                // Seed fn scope with module-level bindings, then add params.
                let mut fn_vars = var_to_struct.clone();
                for p in params {
                    if let TypeAnn::Named(t) = &p.ty {
                        if struct_defs.iter().any(|s| s == t) {
                            fn_vars.insert(p.name.clone(), t.clone());
                        }
                    }
                }
                for stmt in body {
                    walk_stmt(stmt, &mut fn_vars, &fn_returns, &struct_defs, &mut types);
                }
            }
            // Items that are themselves expressions used for their side
            // effect (rare at module scope, but the catch-all in lower.rs
            // does invoke lower_expr on them).
            other => walk_expr(other, &var_to_struct, &fn_returns, &struct_defs, &mut types),
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
    types: &mut FieldAccessTypes,
) {
    match stmt {
        Node::Let { name, value, .. } | Node::Assign { name, value, .. } => {
            walk_expr(value, fn_vars, fn_returns, struct_defs, types);
            if let Some(t) = infer_struct(value, fn_vars, fn_returns) {
                if struct_defs.iter().any(|s| s == &t) {
                    fn_vars.insert(name.clone(), t);
                }
            }
        }
        Node::Return { value: Some(v), .. } => {
            walk_expr(v, fn_vars, fn_returns, struct_defs, types);
        }
        other => walk_expr(other, fn_vars, fn_returns, struct_defs, types),
    }
}

/// Walk an expression, recording every `FieldAccess` whose receiver
/// resolves to a known struct type.
fn walk_expr(
    expr: &Node,
    vars: &HashMap<String, String>,
    fn_returns: &HashMap<String, String>,
    struct_defs: &[String],
    types: &mut FieldAccessTypes,
) {
    match expr {
        Node::FieldAccess { receiver, span, .. } => {
            // Recurse first so nested FieldAccess entries get recorded
            // before we look up the outer one.
            walk_expr(receiver, vars, fn_returns, struct_defs, types);
            if let Some(t) = infer_struct(receiver, vars, fn_returns) {
                if struct_defs.iter().any(|s| s == &t) {
                    types.insert(*span, t);
                }
            }
        }
        Node::Binary { left, right, .. } => {
            walk_expr(left, vars, fn_returns, struct_defs, types);
            walk_expr(right, vars, fn_returns, struct_defs, types);
        }
        Node::Logical { left, right, .. } => {
            walk_expr(left, vars, fn_returns, struct_defs, types);
            walk_expr(right, vars, fn_returns, struct_defs, types);
        }
        Node::Neg { operand, .. } => walk_expr(operand, vars, fn_returns, struct_defs, types),
        Node::Paren(inner, _) => walk_expr(inner, vars, fn_returns, struct_defs, types),
        Node::Ref { inner, .. } => walk_expr(inner, vars, fn_returns, struct_defs, types),
        Node::Call { args, .. } => {
            for a in args {
                walk_expr(a, vars, fn_returns, struct_defs, types);
            }
        }
        Node::MethodCall { receiver, args, .. } => {
            walk_expr(receiver, vars, fn_returns, struct_defs, types);
            for a in args {
                walk_expr(a, vars, fn_returns, struct_defs, types);
            }
        }
        Node::Block { stmts, .. } => {
            // Inner Let/Assign in a block can shadow — use a snapshot.
            let mut local = vars.clone();
            for stmt in stmts {
                walk_stmt(stmt, &mut local, fn_returns, struct_defs, types);
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expr(cond, vars, fn_returns, struct_defs, types);
            let mut local = vars.clone();
            for stmt in then_branch {
                walk_stmt(stmt, &mut local, fn_returns, struct_defs, types);
            }
            if let Some(else_b) = else_branch {
                let mut local = vars.clone();
                for stmt in else_b {
                    walk_stmt(stmt, &mut local, fn_returns, struct_defs, types);
                }
            }
        }
        Node::StructLit { fields, .. } => {
            for f in fields {
                walk_expr(&f.value, vars, fn_returns, struct_defs, types);
            }
        }
        // Other nodes either don't contain expressions or are
        // declaration-shaped (StructDef, EnumDef, …) — nothing to walk.
        _ => {}
    }
}

/// Infer the struct-type name of an expression, if known. Returns
/// `None` for scalar / unresolvable / not-a-struct expressions.
fn infer_struct(
    expr: &Node,
    vars: &HashMap<String, String>,
    fn_returns: &HashMap<String, String>,
) -> Option<String> {
    match expr {
        // The most common cases first — direct StructLit + Ident lookup.
        Node::StructLit { name, .. } => Some(name.clone()),
        Node::Lit(Literal::Ident(v), _) => vars.get(v).cloned(),
        Node::Call { callee, .. } => fn_returns.get(callee).cloned(),
        Node::Paren(inner, _) => infer_struct(inner, vars, fn_returns),
        Node::Ref { inner, .. } => infer_struct(inner, vars, fn_returns),
        // Chained access — `a.b` returns whatever field `b`'s type is.
        // For Step 2 we don't track nested struct fields (every field is
        // i64-shaped under the P0e Option-C ABI); when `std-surface`
        // grows nested struct fields, extend this arm to look up
        // `StructDef.fields[idx].ty` and return the inner struct name.
        Node::FieldAccess { .. } => None,
        _ => None,
    }
}
