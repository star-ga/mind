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
//! Gated under `feature = "std-surface"` by its `mod` declaration in
//! `src/eval/mod.rs` — non-feature builds construct
//! the empty map for free and never enter this module.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ast::{Module, Node, Span, TypeAnn};

mod inference;
use inference::{
    declared_array_element_named, infer_struct, insert_unique_array_return, unwrap_to_named,
    update_binding,
};

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

/// Direct struct returns and full array returns have distinct meanings. Keep
/// the array schema separate so a physical i64/tensor carrier is never used
/// to infer a record element.
struct ResolverReturns<'a> {
    direct: &'a HashMap<String, String>,
    array_elements: &'a HashMap<String, TypeAnn>,
}

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
/// - (3) **Returned fixed-array element** `foo()[i].x`: a separate first-pass
///   map records only an exact declared `TypeAnn::Array` whose element is a
///   known struct. Scalar, unknown, and ambiguous return schemas stay absent,
///   so the physical i64/tensor carrier cannot authorize a field read.
/// - (4) **Struct parameter** `fn read(c: Cfg) { c.max }`: when we
///   descend into a fn body, the per-fn binding map seeds with the
///   parameter list (any `Param` whose `ty` is `Named(T)` for a known
///   `StructDef` gets `param_name → T`).
///
/// - (5) **Nested struct-typed field** `a.b.c` where field `b` is itself
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
    let mut fn_array_returns: HashMap<String, TypeAnn> = HashMap::new();
    let mut ambiguous_array_returns: HashSet<String> = HashSet::new();
    let mut local_fn_names: HashSet<String> = HashSet::new();
    #[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
    let mut imported_return_schemas: HashMap<String, Option<TypeAnn>> = HashMap::new();
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
            Node::FnDef(fd, _) if fd.ret_type.is_some() => {
                let name = &fd.name;
                local_fn_names.insert(name.clone());
                let ret = fd.ret_type.as_ref().unwrap();
                // Peel `&T` / `&mut T` / `[T]` / `&[T]` to the inner
                // `Named(T)` so a `-> &Cfg` return type resolves its
                // struct just like a `-> Cfg` one. Defer the "is this t
                // a struct?" check to the second pass, since StructDef
                // may appear textually below the FnDef.
                if let Some(t) = unwrap_to_named(ret) {
                    fn_returns.insert(name.clone(), t.to_string());
                }
                let resolved_ret = crate::eval::type_aliases::resolve_active(ret);
                if declared_array_element_named(&resolved_ret).is_some() {
                    insert_unique_array_return(
                        &mut fn_array_returns,
                        &mut ambiguous_array_returns,
                        name,
                        resolved_ret,
                    );
                }
            }
            Node::FnDef(fd, _) => {
                local_fn_names.insert(fd.name.clone());
            }
            _ => {}
        }
    }
    // Cross-module struct names + field types + fn returns. A pure CONSUMER
    // module (no local `StructDef`, e.g. mind-flow's compile.mind reading a
    // sibling module's `AnalyzedFlow`) declares none of these locally, so
    // without this the span->struct-name resolver below has nothing to anchor
    // on and a value-position FieldAccess on a sibling struct falls through to
    // the `ConstI64(0)` placeholder (a silent miscompile). The most common
    // concrete miss is a BY-REFERENCE struct param (`p: &Point`): the lowering
    // fast-path (struct_env) only seeds a `TypeAnn::Named` param, so a `&Point`
    // param never seeds Step 1, and Step 2 is empty unless this resolver runs —
    // `p.y` then reads `ConstI64(0)`.
    //
    // The field-name->offset + field-type data for these sibling structs is
    // ALREADY merged into `ir.struct_defs` / `ir.struct_field_types` in
    // `lower_to_ir` (via `with_global_enums`); here we only teach the resolver
    // that these names ARE structs so it records the right `receiver_types[span]`
    // entry. `unwrap_to_named` below already peels `&T` so a `&Point` param seeds
    // correctly.
    //
    // Outside a project the registry is empty, so this inserts nothing and the
    // resolver behaves byte-identically to the local-only path (the keystone and
    // every single-file compile see an empty registry). All inserts use
    // `or_insert`-style guards so a module's OWN `StructDef`/`FnDef` wins on a
    // name collision — the local declaration is authoritative.
    //
    // This populates the SAME `receiver_types` side-table consulted by the
    // FieldAccess-read arm, the MethodCall accessor arm, and the FieldAssign
    // write arm in `lower.rs` (all key on the node span via `infer_struct`), so
    // the cross-module fix is uniform across all three.
    // deferred: only the value-position FieldAccess-READ form (by-ref param) is
    // covered by a *_run.rs regression here; cross-module MethodCall-accessor and
    // FieldAssign resolution flow through the identical machinery but their
    // end-to-end run-tests are a follow-up — upgrade path: extend
    // tests/cross_module_field_access_run.rs with a sibling-struct `s.len()`
    // accessor and an `o.f = x` write, ctypes-asserting both.
    #[cfg(feature = "std-surface")]
    crate::ir::with_global_enums(|g| {
        for name in g.structs.keys().chain(g.qualified_structs.keys()) {
            for key in std::iter::once(name.as_str()).chain(name.strip_prefix("crate.")) {
                if !struct_defs.iter().any(|known| known == key) {
                    struct_defs.push(key.to_string());
                }
            }
        }
        // Seed cross-module `(struct, field) -> inner-struct-name` so a CHAINED
        // read `a.b.c` on sibling structs resolves: `infer_struct` walks
        // `a -> S`, `(S, b) -> Inner`, then `.c` resolves against `Inner`. The
        // retain below drops any whose inner name isn't a real struct.
        for (sname, (fnames, ftypes)) in g.structs.iter().chain(&g.qualified_structs) {
            for owner in std::iter::once(sname.as_str()).chain(sname.strip_prefix("crate.")) {
                for (fname, fty) in fnames.iter().zip(ftypes) {
                    if let Some(t) = unwrap_to_named(fty) {
                        field_types
                            .entry((owner.to_string(), fname.clone()))
                            .or_insert_with(|| t.to_string());
                    }
                }
            }
        }
        // Cross-module fn returns, for `let x = sibling_fn(); x.field`.
        for (fname, rt) in &g.fn_returns {
            if let Some(t) = unwrap_to_named(rt) {
                fn_returns
                    .entry(fname.clone())
                    .or_insert_with(|| t.to_string());
            }
            if !local_fn_names.contains(fname) {
                // Global return metadata was resolved in the defining module
                // by `build_global_enums`; applying the consumer aliases here
                // would reinterpret a defining `Item` as the consumer's
                // same-named scalar alias.
                if declared_array_element_named(rt).is_some() {
                    insert_unique_array_return(
                        &mut fn_array_returns,
                        &mut ambiguous_array_returns,
                        fname,
                        rt.clone(),
                    );
                }
            }
        }
    });

    #[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
    for (fname, _params, ret) in crate::type_checker::cm_all_imported_fn_signatures() {
        if local_fn_names.contains(&fname) {
            continue;
        }
        // `cm_all_imported_fn_signatures` publishes owner-resolved types from
        // the module table. Keep them in that owner scope; resolving again
        // under the consumer aliases can turn `[Item; N]` into `[i64; N]`.
        // Record every return shape, including scalar/no-return shapes, so a
        // duplicate imported bare name cannot silently select its array
        // declaration when another owner publishes a conflicting scalar.
        if let Some(previous) = imported_return_schemas.get(&fname) {
            if previous != &ret {
                fn_array_returns.remove(&fname);
                ambiguous_array_returns.insert(fname.clone());
            }
        } else {
            imported_return_schemas.insert(fname.clone(), ret.clone());
        }
        if !ambiguous_array_returns.contains(&fname) {
            if let Some(rt) = ret {
                if declared_array_element_named(&rt).is_some() {
                    insert_unique_array_return(
                        &mut fn_array_returns,
                        &mut ambiguous_array_returns,
                        &fname,
                        rt,
                    );
                }
            }
        }
    }

    // Filter fn_returns to only those whose return-type is a real struct.
    fn_returns.retain(|_, t| struct_defs.iter().any(|s| s == t));
    for name in ambiguous_array_returns {
        fn_array_returns.remove(&name);
    }
    fn_array_returns.retain(|_, t| {
        declared_array_element_named(t)
            .is_some_and(|name| struct_defs.iter().any(|known| known == name))
    });
    let returns = ResolverReturns {
        direct: &fn_returns,
        array_elements: &fn_array_returns,
    };
    // Likewise keep only field types that name a real struct — scalar
    // fields and slices-of-scalar drop out, so `infer_struct`'s
    // FieldAccess arm only ever returns a known struct name.
    field_types.retain(|_, t| struct_defs.iter().any(|s| s == t));

    // Second pass — walk every expression, tracking var→struct bindings.
    let mut var_to_struct: HashMap<String, String> = HashMap::new();
    for item in &module.items {
        match item {
            Node::Let {
                name, ann, value, ..
            }
            | Node::Const {
                name,
                ty: ann,
                value,
                ..
            } => {
                walk_expr(
                    value,
                    &var_to_struct,
                    &returns,
                    &struct_defs,
                    &field_types,
                    &mut types,
                );
                update_binding(
                    name,
                    ann.as_ref(),
                    Some(value),
                    &struct_defs,
                    &returns,
                    &field_types,
                    &mut var_to_struct,
                );
            }
            Node::Assign { name, value, .. } => {
                walk_expr(
                    value,
                    &var_to_struct,
                    &returns,
                    &struct_defs,
                    &field_types,
                    &mut types,
                );
                if let Some(t) = infer_struct(value, &var_to_struct, &returns, &field_types) {
                    if struct_defs.iter().any(|s| s == &t) {
                        var_to_struct.insert(name.clone(), t);
                    }
                }
            }
            Node::FnDef(fd, _) => {
                let (params, body) = (&fd.params, &fd.body);
                // Seed fn scope with module-level bindings, then add params.
                let mut fn_vars = var_to_struct.clone();
                for p in params {
                    update_binding(
                        &p.name,
                        Some(&p.ty),
                        None,
                        &struct_defs,
                        &returns,
                        &field_types,
                        &mut fn_vars,
                    );
                }
                for stmt in body {
                    walk_stmt(
                        stmt,
                        &mut fn_vars,
                        &returns,
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
                &returns,
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
    returns: &ResolverReturns,
    struct_defs: &[String],
    field_types: &FieldTypes,
    types: &mut FieldAccessTypes,
) {
    match stmt {
        Node::Let {
            name, ann, value, ..
        } => {
            walk_expr(value, fn_vars, returns, struct_defs, field_types, types);
            update_binding(
                name,
                ann.as_ref(),
                Some(value),
                struct_defs,
                returns,
                field_types,
                fn_vars,
            );
        }
        Node::Assign { name, value, .. } => {
            walk_expr(value, fn_vars, returns, struct_defs, field_types, types);
            update_binding(
                name,
                None,
                Some(value),
                struct_defs,
                returns,
                field_types,
                fn_vars,
            );
        }
        Node::Return { value: Some(v), .. } => {
            walk_expr(v, fn_vars, returns, struct_defs, field_types, types);
        }
        other => walk_expr(other, fn_vars, returns, struct_defs, field_types, types),
    }
}

/// Walk an expression, recording every `FieldAccess` whose receiver
/// resolves to a known struct type.
fn walk_expr(
    expr: &Node,
    vars: &HashMap<String, String>,
    returns: &ResolverReturns,
    struct_defs: &[String],
    field_types: &FieldTypes,
    types: &mut FieldAccessTypes,
) {
    match expr {
        Node::FieldAccess { receiver, span, .. } => {
            // Recurse first so nested FieldAccess entries get recorded
            // before we look up the outer one.
            walk_expr(receiver, vars, returns, struct_defs, field_types, types);
            if let Some(t) = infer_struct(receiver, vars, returns, field_types) {
                if struct_defs.iter().any(|s| s == &t) {
                    types.insert(*span, t);
                }
            }
        }
        Node::Binary { left, right, .. } => {
            walk_expr(left, vars, returns, struct_defs, field_types, types);
            walk_expr(right, vars, returns, struct_defs, field_types, types);
        }
        Node::Logical { left, right, .. } => {
            walk_expr(left, vars, returns, struct_defs, field_types, types);
            walk_expr(right, vars, returns, struct_defs, field_types, types);
        }
        Node::Neg { operand, .. } | Node::Not { operand, .. } | Node::BitNot { operand, .. } => {
            walk_expr(operand, vars, returns, struct_defs, field_types, types)
        }
        Node::Paren(inner, _) => walk_expr(inner, vars, returns, struct_defs, field_types, types),
        Node::Ref { inner, .. } => walk_expr(inner, vars, returns, struct_defs, field_types, types),
        // A cast `<expr> as <ty>` is transparent to field-access resolution:
        // walk the operand so any FieldAccess nested under the cast (e.g. the
        // `jv_n_int(f.h) as u64` tail of std/json's `get_u64`) records its
        // span. Without this arm the catch-all skipped the whole subtree, the
        // lowering's Step-2 side-table lookup missed, and the read fell to the
        // `ConstI64(0)` placeholder — a NULL receiver address whose field load
        // SEGFAULTs at runtime (`__mind_load_i64(0 + off)`). The cast target
        // type is scalar-only surface and never a struct value, so only the
        // operand is walked; `infer_struct` correctly keeps returning `None`
        // for the cast expression itself.
        Node::As { expr, .. } => walk_expr(expr, vars, returns, struct_defs, field_types, types),
        Node::Assert { cond, .. } => {
            walk_expr(cond, vars, returns, struct_defs, field_types, types)
        }
        Node::Call { args, .. } => {
            for a in args {
                walk_expr(a, vars, returns, struct_defs, field_types, types);
            }
        }
        Node::MethodCall {
            receiver,
            args,
            span,
            ..
        } => {
            walk_expr(receiver, vars, returns, struct_defs, field_types, types);
            for a in args {
                walk_expr(a, vars, returns, struct_defs, field_types, types);
            }
            // RFC 0005 method-as-field brick — record the receiver's
            // struct type at the MethodCall's own span so the lowering
            // arm can resolve a zero-arg accessor method (`s.len()`) to
            // the same field load the `FieldAccess` arm emits for `s.len`.
            // Mirror of the `FieldAccess` arm: lowering keys the side-table
            // on the node span and looks up the field offset in
            // `ir.struct_defs[T]`. Purely additive — method calls are absent
            // from the keystone and all of std, so no emitted bytes change.
            if let Some(t) = infer_struct(receiver, vars, returns, field_types) {
                if struct_defs.iter().any(|s| s == &t) {
                    types.insert(*span, t);
                }
            }
        }
        Node::Block { stmts, .. } => {
            // Inner Let/Assign in a block can shadow — use a snapshot.
            let mut local = vars.clone();
            for stmt in stmts {
                walk_stmt(stmt, &mut local, returns, struct_defs, field_types, types);
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expr(cond, vars, returns, struct_defs, field_types, types);
            let mut local = vars.clone();
            for stmt in then_branch {
                walk_stmt(stmt, &mut local, returns, struct_defs, field_types, types);
            }
            if let Some(else_b) = else_branch {
                let mut local = vars.clone();
                for stmt in else_b {
                    walk_stmt(stmt, &mut local, returns, struct_defs, field_types, types);
                }
            }
        }
        Node::StructLit { fields, .. } => {
            for f in fields {
                walk_expr(&f.value, vars, returns, struct_defs, field_types, types);
            }
        }
        Node::ArrayLit { elements, .. } => {
            for element in elements {
                walk_expr(element, vars, returns, struct_defs, field_types, types);
            }
        }
        Node::IndexAccess {
            receiver, index, ..
        } => {
            walk_expr(receiver, vars, returns, struct_defs, field_types, types);
            walk_expr(index, vars, returns, struct_defs, field_types, types);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            walk_expr(receiver, vars, returns, struct_defs, field_types, types);
            walk_expr(index, vars, returns, struct_defs, field_types, types);
            walk_expr(value, vars, returns, struct_defs, field_types, types);
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
            walk_expr(receiver, vars, returns, struct_defs, field_types, types);
            if let Some(t) = infer_struct(receiver, vars, returns, field_types) {
                if struct_defs.iter().any(|s| s == &t) {
                    types.insert(*span, t);
                }
            }
            walk_expr(value, vars, returns, struct_defs, field_types, types);
        }
        // RFC 0010 Phase J-A: region body — same walk as Block.
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => {
            let mut local = vars.clone();
            for stmt in body {
                walk_stmt(stmt, &mut local, returns, struct_defs, field_types, types);
            }
        }
        // A `while` loop body is a scope, like a Block/If branch. Without this
        // arm a `FieldAccess` inside the loop body — `make().field`,
        // `a.b.c`, or a method accessor — was never recorded in the side-table,
        // so lowering's Step 2 could not resolve it and it fell through to the
        // `ConstI64(0)` placeholder (a SILENT wrong-value read). Snapshot `vars`
        // so an in-loop `let` shadow does not leak past the loop.
        Node::While { cond, body, .. } => {
            walk_expr(cond, vars, returns, struct_defs, field_types, types);
            let mut local = vars.clone();
            for stmt in body {
                walk_stmt(stmt, &mut local, returns, struct_defs, field_types, types);
            }
        }
        // A `for x in coll` body is likewise a scope. The element binding `x`
        // is a per-iteration collection element whose struct type this
        // side-table does not track (collections are i64-handle shaped here),
        // so we only walk the collection expression and the body statements —
        // enough for an in-body `make().field` / `a.b.c` to resolve.
        Node::ForEach {
            collection, body, ..
        } => {
            walk_expr(collection, vars, returns, struct_defs, field_types, types);
            let mut local = vars.clone();
            for stmt in body {
                walk_stmt(stmt, &mut local, returns, struct_defs, field_types, types);
            }
        }
        // A counted `for` body has the same lexical scope as `while` and
        // `for-each`. Preserve the existing receiver metadata for field
        // accesses in the body; otherwise lowering can only see the loop
        // node and a fixed-array field write may evade its capability gate.
        Node::For {
            start, end, body, ..
        } => {
            walk_expr(start, vars, returns, struct_defs, field_types, types);
            walk_expr(end, vars, returns, struct_defs, field_types, types);
            let mut local = vars.clone();
            for stmt in body {
                walk_stmt(stmt, &mut local, returns, struct_defs, field_types, types);
            }
        }
        // Other nodes either don't contain expressions or are
        // declaration-shaped (StructDef, EnumDef, …) — nothing to walk.
        _ => {}
    }
}
