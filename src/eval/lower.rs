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

use std::collections::HashMap;

use crate::ast;
use crate::ast::Literal;
use crate::ast::TensorElemOp;
use crate::ast::TypeAnn;

use crate::ir::BinOp;
use crate::ir::IRModule;
use crate::ir::IndexSpec;
use crate::ir::Instr;
use crate::ir::SliceSpec;
use crate::ir::ValueId;
use crate::types::DType;
use crate::types::ShapeDim;

// ---------------------------------------------------------------------------
// Codegen monomorphization (bounded slice) — generic fns reach the IR backend.
//
// A generic `fn id<T>(x: T) -> T { x }` is a TEMPLATE: it is never emitted as
// an `Instr::FnDef` itself. Instead, every concrete call site (`id(5)`) records
// a monomorphization request keyed on the deterministic mangled instance name
// (`id$i64`); after the module body is lowered, each distinct instance is
// emitted once, in sorted-by-mangled-name order, so `emit_mic3`/`trace_hash`
// stay byte-identical across runs (no HashMap iteration, no clocks, no rng).
//
// WEDGE: non-generic code is untouched — a function with empty `type_params`
// flows through the existing FnDef path verbatim, and a call whose callee is
// not a registered generic flows through the existing Call path verbatim. The
// state below lives in a `thread_local!` scoped to a single `lower_to_ir` call,
// so no existing `lower_expr` signature changes (and the non-generic lowering
// is literally the same code).
//
// Bounded to: a single type parameter, scalar concrete arg types inferred from
// a literal call argument (`Int -> i64`, `Float -> f64`), identity/min shape.
// Multi-param / nested generics / trait bounds are a later slice.
// ---------------------------------------------------------------------------

/// One pending monomorphization: the generic fn name plus the concrete type
/// substituted for its single type parameter, both already reflected in the
/// mangled instance name used as the map key.
// `generic_name` resolves the template and `concrete` re-derives the instance
// signature in the monomorphization drain (`lower_to_ir`); the registry keys on
// the mangled name.
#[derive(Clone)]
struct MonoRequest {
    /// Source-level generic fn name (e.g. `id`).
    generic_name: String,
    /// Concrete type bound to the (single) type parameter.
    concrete: TypeAnn,
}

#[derive(Default)]
struct MonoCtx {
    /// Generic fn templates by source name (only fns with non-empty
    /// `type_params`). Read-only after the pre-pass.
    templates: std::collections::BTreeMap<String, ast::Node>,
    /// Requested instances keyed on mangled name (e.g. `id$i64`). `BTreeMap`
    /// so draining is deterministic.
    requests: std::collections::BTreeMap<String, MonoRequest>,
}

thread_local! {
    static MONO: std::cell::RefCell<MonoCtx> = std::cell::RefCell::new(MonoCtx::default());
    /// Part 1 (generics): the ENCLOSING fn's `param name -> declared TypeAnn`,
    /// so a generic call `id(n)` with `n` a scalar parameter monomorphizes to
    /// `id$i64` instead of leaving a bare `@id` reference — which was a SILENT
    /// MISCOMPILE (an EXIT=0 `.so` with an undefined symbol). Seeded by the
    /// FnDef-lowering arm ONLY when the module declares generic templates, so a
    /// non-generic module never touches it and its IR bytes stay byte-identical.
    /// Restored on scope exit via `ParamTypesGuard` so it never leaks across fns.
    static PARAM_TYPES: std::cell::RefCell<std::collections::HashMap<String, TypeAnn>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// Generic-arg inference (0-fail-closed): every module fn's declared scalar
    /// RETURN TypeAnn (`name -> ret`), so a generic call over a NESTED call
    /// `id(g(3))` resolves to `g`'s return type. Populated once up-front in
    /// `lower_to_ir` ONLY when the module declares generic templates (cleared +
    /// left empty otherwise, so a non-generic module never reads/builds it and
    /// its IR bytes stay byte-identical).
    static FN_RETURNS: std::cell::RefCell<std::collections::HashMap<String, TypeAnn>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// RAII guard restoring the previous `PARAM_TYPES` map when a fn's lowering
/// scope ends. `None` => seeding was skipped (non-generic module) — a pure
/// no-op, so the byte-identity hot path is untouched.
struct ParamTypesGuard(Option<std::collections::HashMap<String, TypeAnn>>);
impl Drop for ParamTypesGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.0.take() {
            PARAM_TYPES.with(|p| *p.borrow_mut() = prev);
        }
    }
}

/// Seed `PARAM_TYPES` with this fn's parameters for generic-arg inference, ONLY
/// when the module has generic templates (otherwise a no-op guard). The returned
/// guard restores the prior map on drop.
fn seed_param_types(params: &[ast::Param]) -> ParamTypesGuard {
    let has_templates = MONO.with(|c| !c.borrow().templates.is_empty());
    if !has_templates {
        return ParamTypesGuard(None);
    }
    PARAM_TYPES.with(|p| {
        let prev = p.borrow().clone();
        {
            let mut m = p.borrow_mut();
            m.clear();
            for prm in params {
                m.insert(prm.name.clone(), prm.ty.clone());
            }
        }
        ParamTypesGuard(Some(prev))
    })
}

/// Short, deterministic mangle suffix for a concrete scalar type.
/// Returns `None` for shapes outside the bounded slice (the call then lowers
/// through the ordinary, non-monomorphized path — no behavior change).
/// `pub(crate)` so the abi_gate fail-closed gate builds its `fn_returns` map with
/// the IDENTICAL scalar filter the lowering uses (lockstep).
pub(crate) fn mangle_suffix(ty: &TypeAnn) -> Option<&'static str> {
    match ty {
        TypeAnn::ScalarI64 => Some("i64"),
        TypeAnn::ScalarI32 => Some("i32"),
        TypeAnn::ScalarF64 => Some("f64"),
        TypeAnn::ScalarF32 => Some("f32"),
        TypeAnn::ScalarBool => Some("bool"),
        _ => None,
    }
}

/// Infer the concrete scalar type of a call argument so a generic call
/// monomorphizes (`id(arg)` -> `id$<suffix>`). Recognises: literals; a bare
/// variable bound to an enclosing-fn scalar PARAMETER or a top-level Let-local
/// (via `bindings`); an explicit `as` cast (the target type); a nested call (the
/// callee's declared return type, via `fn_returns`); arithmetic/bitwise over two
/// same-typed operands; parenthesised/negated inner expressions. Anything else
/// returns `None` so the call fail-closes rather than emitting a dangling
/// reference. `pub(crate)` so the abi_gate fail-closed check shares this exact
/// predicate — gate and lowering cannot drift.
pub(crate) fn infer_concrete_arg_type(
    arg: &ast::Node,
    bindings: &std::collections::HashMap<String, TypeAnn>,
    fn_returns: &std::collections::HashMap<String, TypeAnn>,
) -> Option<TypeAnn> {
    let scalar = |t: TypeAnn| -> Option<TypeAnn> { Some(t).filter(|t| mangle_suffix(t).is_some()) };
    match arg {
        ast::Node::Lit(Literal::Int(_), _) => Some(TypeAnn::ScalarI64),
        ast::Node::Lit(Literal::Float(_), _) => Some(TypeAnn::ScalarF64),
        // A bare variable bound to an enclosing-fn parameter OR a top-level
        // Let-local resolves to its declared/inferred scalar type.
        ast::Node::Lit(Literal::Ident(name), _) => bindings
            .get(name)
            .cloned()
            .filter(|t| mangle_suffix(t).is_some()),
        // An explicit cast: the target type IS the concrete type (`id(x as i64)`).
        ast::Node::As { ty, .. } => scalar(ty.clone()),
        // A nested call: the callee's declared scalar return type (`id(g(3))`).
        ast::Node::Call { callee, .. } => fn_returns
            .get(callee)
            .cloned()
            .filter(|t| mangle_suffix(t).is_some()),
        // Parenthesised / negated: the inner expression's type.
        ast::Node::Paren(inner, _) | ast::Node::Neg { operand: inner, .. } => {
            infer_concrete_arg_type(inner, bindings, fn_returns)
        }
        // Arithmetic / comparison over two same-typed operands. A comparison op
        // yields `bool`; an arithmetic op yields the (shared) operand type. Mixed
        // operand types are ambiguous to mangle -> None (fail-closed).
        ast::Node::Binary {
            op, left, right, ..
        } => {
            let l = infer_concrete_arg_type(left, bindings, fn_returns)?;
            let r = infer_concrete_arg_type(right, bindings, fn_returns)?;
            if l != r {
                return None;
            }
            match op {
                crate::ast::BinOp::Lt
                | crate::ast::BinOp::Le
                | crate::ast::BinOp::Gt
                | crate::ast::BinOp::Ge
                | crate::ast::BinOp::Eq
                | crate::ast::BinOp::Ne => Some(TypeAnn::ScalarBool),
                _ => Some(l),
            }
        }
        // Bitwise (`& | ^ << >>`) yields the (shared) operand type.
        ast::Node::Bitwise { left, right, .. } => {
            let l = infer_concrete_arg_type(left, bindings, fn_returns)?;
            let r = infer_concrete_arg_type(right, bindings, fn_returns)?;
            if l == r { Some(l) } else { None }
        }
        _ => None,
    }
}

/// Whether a call's argument shape can be monomorphized in the bounded slice
/// (exactly one arg whose concrete scalar type is inferable). THE single source
/// of truth shared by the lowering (`try_register_mono_instance`) and the
/// abi_gate fail-closed gate, so the gate flags exactly the calls the lowering
/// would leave as a dangling bare-template reference — they cannot drift.
pub(crate) fn is_monomorphizable(
    args: &[ast::Node],
    bindings: &std::collections::HashMap<String, TypeAnn>,
    fn_returns: &std::collections::HashMap<String, TypeAnn>,
) -> bool {
    args.len() == 1 && infer_concrete_arg_type(&args[0], bindings, fn_returns).is_some()
}

/// Resolve and record a top-level `Let`'s binding type into `bindings` for
/// generic-arg inference: the explicit annotation (when scalar) else the inferred
/// type of the value expression. Non-scalar lets are not recorded (a generic call
/// over them then fail-closes, in lockstep). THE single Let-handling
/// implementation, called by BOTH the lowering and the abi_gate gate so they
/// never drift. The RHS is resolved against the bindings BEFORE this insert, so a
/// `let z = z` cannot see its own binding (matches SSA lowering order).
pub(crate) fn bind_let(
    bindings: &mut std::collections::HashMap<String, TypeAnn>,
    let_node: &ast::Node,
    fn_returns: &std::collections::HashMap<String, TypeAnn>,
) {
    if let ast::Node::Let {
        name, ann, value, ..
    } = let_node
    {
        let ty = match ann {
            Some(a) if mangle_suffix(a).is_some() => Some(a.clone()),
            _ => infer_concrete_arg_type(value, bindings, fn_returns),
        };
        if let Some(t) = ty {
            bindings.insert(name.clone(), t);
        }
    }
}

/// If `callee` names a registered generic and the call's single argument has an
/// inferable concrete scalar type, register the instance and return its mangled
/// name. Returns `None` (caller keeps the original name) when the callee is not
/// generic or the shape is outside the bounded slice.
fn try_register_mono_instance(callee: &str, args: &[ast::Node]) -> Option<String> {
    MONO.with(|cell| {
        // Fast path: a module with no generic templates (the overwhelmingly common
        // case) pays only this O(1) is_empty check on the lowering hot path —
        // avoids a borrow_mut + per-call HashMap probe for every non-generic call.
        if cell.borrow().templates.is_empty() {
            return None;
        }
        let mut ctx = cell.borrow_mut();
        if !ctx.templates.contains_key(callee) {
            return None;
        }
        // Bounded slice: exactly one argument whose concrete scalar type is
        // inferable — a literal, or an enclosing-fn scalar parameter resolved via
        // the PARAM_TYPES map the FnDef arm seeded (shared with the abi_gate
        // fail-closed gate through `infer_concrete_arg_type`).
        if args.len() != 1 {
            return None;
        }
        let concrete = PARAM_TYPES.with(|p| {
            FN_RETURNS.with(|fr| infer_concrete_arg_type(&args[0], &p.borrow(), &fr.borrow()))
        })?;
        let suffix = mangle_suffix(&concrete)?;
        let mangled = format!("{callee}${suffix}");
        ctx.requests
            .entry(mangled.clone())
            .or_insert_with(|| MonoRequest {
                generic_name: callee.to_string(),
                concrete: concrete.clone(),
            });
        Some(mangled)
    })
}

/// Synthesize the concrete (non-generic) `FnDef` AST for one instance:
/// clone the template, rename it to `mangled`, drop `type_params`, and rewrite
/// every parameter typed by the (single) type parameter to the concrete type.
// Called by the monomorphization drain in `lower_to_ir` to synthesize each
// requested concrete instance before lowering it through the FnDef path.
fn instantiate_template(
    template: &ast::Node,
    mangled: &str,
    concrete: &TypeAnn,
) -> Option<ast::Node> {
    if let ast::Node::FnDef {
        is_pub,
        is_test,
        type_params,
        params,
        ret_type,
        body,
        reap_threshold,
        attrs,
        span,
        ..
    } = template
    {
        // Bounded slice: a single type parameter.
        let tp = type_params.first()?.clone();
        let new_params: Vec<ast::Param> = params
            .iter()
            .map(|p| {
                let ty = if matches!(&p.ty, TypeAnn::Named(n) if *n == tp) {
                    concrete.clone()
                } else {
                    p.ty.clone()
                };
                ast::Param {
                    name: p.name.clone(),
                    ty,
                    span: p.span,
                }
            })
            .collect();
        let new_ret = ret_type.as_ref().map(|r| {
            if matches!(r, TypeAnn::Named(n) if *n == tp) {
                concrete.clone()
            } else {
                r.clone()
            }
        });
        Some(ast::Node::FnDef {
            is_pub: *is_pub,
            is_test: *is_test,
            name: mangled.to_string(),
            type_params: Vec::new(),
            params: new_params,
            ret_type: new_ret,
            body: body.clone(),
            reap_threshold: *reap_threshold,
            attrs: attrs.clone(),
            span: *span,
        })
    } else {
        None
    }
}

/// True if a type annotation references the type-parameter name `tp` — checks
/// `Named(tp)` and the element/target of slice/array/ref wrappers.
fn type_ann_mentions(ty: &TypeAnn, tp: &str) -> bool {
    match ty {
        TypeAnn::Named(n) => n == tp,
        TypeAnn::Slice { element, .. } | TypeAnn::Array { element, .. } => {
            type_ann_mentions(element, tp)
        }
        TypeAnn::Ref { target, .. } => type_ann_mentions(target, tp),
        _ => false,
    }
}

/// True if any type annotation in `node` (recursively) names the type parameter
/// `tp`. The monomorphization drain uses this to refuse an instance whose body
/// still carries the type parameter in a type position (`let r: T = ...`) — the
/// signature-only rewrite leaves such a binding to default silently to the i64
/// ABI. Covers the type-annotation-bearing nodes (`let` / `const` / `as`) and
/// the statement / expression containers a function body is built from.
fn node_mentions_type_name(node: &ast::Node, tp: &str) -> bool {
    use ast::Node as N;
    let ann_hit = |o: &Option<TypeAnn>| o.as_ref().is_some_and(|t| type_ann_mentions(t, tp));
    let any = |ns: &[ast::Node]| ns.iter().any(|n| node_mentions_type_name(n, tp));
    match node {
        N::Let { ann, value, .. } => ann_hit(ann) || node_mentions_type_name(value, tp),
        N::Const { ty, value, .. } => ann_hit(ty) || node_mentions_type_name(value, tp),
        N::As { expr, ty, .. } => type_ann_mentions(ty, tp) || node_mentions_type_name(expr, tp),
        N::Block { stmts, .. } => any(stmts),
        N::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            node_mentions_type_name(cond, tp)
                || any(then_branch)
                || else_branch.as_ref().is_some_and(|e| any(e))
        }
        #[cfg(feature = "std-surface")]
        N::While { cond, body, .. } => node_mentions_type_name(cond, tp) || any(body),
        N::Match {
            scrutinee, arms, ..
        } => {
            node_mentions_type_name(scrutinee, tp)
                || arms.iter().any(|a| node_mentions_type_name(&a.body, tp))
        }
        N::Return { value, .. } => value
            .as_ref()
            .is_some_and(|v| node_mentions_type_name(v, tp)),
        N::Assign { value, .. } => node_mentions_type_name(value, tp),
        N::FieldAssign {
            receiver, value, ..
        } => node_mentions_type_name(receiver, tp) || node_mentions_type_name(value, tp),
        N::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            node_mentions_type_name(receiver, tp)
                || node_mentions_type_name(index, tp)
                || node_mentions_type_name(value, tp)
        }
        N::Paren(inner, _) | N::Neg { operand: inner, .. } | N::Ref { inner, .. } => {
            node_mentions_type_name(inner, tp)
        }
        N::Binary { left, right, .. }
        | N::Logical { left, right, .. }
        | N::Bitwise { left, right, .. } => {
            node_mentions_type_name(left, tp) || node_mentions_type_name(right, tp)
        }
        N::Call { args, .. } => args.iter().any(|a| node_mentions_type_name(a, tp)),
        N::MethodCall { receiver, args, .. } => {
            node_mentions_type_name(receiver, tp)
                || args.iter().any(|a| node_mentions_type_name(a, tp))
        }
        N::FieldAccess { receiver, .. } => node_mentions_type_name(receiver, tp),
        N::IndexAccess {
            receiver, index, ..
        } => node_mentions_type_name(receiver, tp) || node_mentions_type_name(index, tp),
        _ => false,
    }
}

pub fn lower_to_ir(module: &ast::Module) -> IRModule {
    // Pre-pass: rewrite statement-position collection mutations (`m.insert(k,v)`,
    // `v.push(x)`) into assignments so the non-mutating std handle is rebound.
    // A no-op (byte-identical) for any module with no collection-mutation
    // statement — notably the keystone main.mind.
    #[cfg(feature = "std-surface")]
    let preprocessed = preprocess_collection_mutations(module);
    #[cfg(feature = "std-surface")]
    let module = preprocessed.as_ref().unwrap_or(module);
    // Install this module's top-level `const NAME = value` table so a reference
    // `Lit(Ident(NAME))` inlines the value at its use site (the read path in the
    // `Lit(Ident)` arm). Overwrites any prior pass's table — a const-free module
    // (the keystone) installs an empty table and is byte-identical.
    #[cfg(feature = "std-surface")]
    {
        crate::ir::clear_module_consts();
        let mut consts = std::collections::BTreeMap::new();
        let mut const_types = std::collections::BTreeMap::new();
        for item in &module.items {
            if let ast::Node::Const { name, value, ty, .. } = item {
                consts.insert(name.clone(), (**value).clone());
                if let Some(t) = ty {
                    const_types.insert(name.clone(), t.clone());
                }
            }
        }
        crate::ir::set_module_consts(consts);
        crate::ir::set_module_const_types(const_types);
    }
    let mut ir = IRModule::new();
    // Built-in Result/Option PRELUDE. MIND has no source-level prelude, but
    // Rust-style code expects `Result<T,E>` / `Option<T>` with bare `Ok`/`Err`/
    // `Some`/`None`. Register them in the boxed-enum side-tables (NOT as emitted
    // `EnumDef` instrs) so a construction `Ok(v)` / `Err(e)` / `Some(v)` and the
    // matching `match` resolve via the bare-constructor path. All-i64 ABI: each
    // payload is one i64 handle. These are lowering-only side-tables that never
    // serialise into mic@3, and the keystone constructs/matches no
    // Result/Option, so its emit is byte-identical. A user module that DEFINES
    // its own `Result`/`Option` overwrites these via last-write-wins in the
    // `EnumDef` arm below.
    #[cfg(feature = "std-surface")]
    {
        use crate::ast::TypeAnn::ScalarI64;
        ir.enum_variant_tags.insert("Result::Ok".to_string(), 0);
        ir.enum_variant_tags.insert("Result::Err".to_string(), 1);
        ir.enum_variant_tags.insert("Option::Some".to_string(), 0);
        ir.enum_variant_tags.insert("Option::None".to_string(), 1);
        ir.boxed_enums.insert("Result".to_string());
        ir.boxed_enums.insert("Option".to_string());
        ir.enum_payload_slots.insert("Result".to_string(), 2);
        ir.enum_payload_slots.insert("Option".to_string(), 2);
        ir.enum_payload_types
            .insert("Result::Ok".to_string(), vec![ScalarI64]);
        ir.enum_payload_types
            .insert("Result::Err".to_string(), vec![ScalarI64]);
        ir.enum_payload_types
            .insert("Option::Some".to_string(), vec![ScalarI64]);
        // Cross-module enum propagation: merge the whole-project enum registry
        // (collected by the project builder from EVERY parsed source) so a
        // variant defined in a SIBLING module — e.g. `TokKind::Eof` from another
        // file — resolves to its tag / boxed record here, even though this
        // module never lowered its `EnumDef`. The per-module `EnumDef` arm below
        // still overwrites any same-name entry via last-write-wins, so a locally
        // defined enum wins. Outside a project the registry is empty, so this
        // inserts nothing and the keystone emit stays byte-identical.
        crate::ir::with_global_enums(|g| {
            for (k, v) in &g.variant_tags {
                ir.enum_variant_tags.insert(k.clone(), *v);
            }
            for (k, v) in &g.payload_types {
                ir.enum_payload_types.insert(k.clone(), v.clone());
            }
            for (k, v) in &g.slots {
                ir.enum_payload_slots.insert(k.clone(), *v);
            }
            for name in &g.boxed {
                ir.boxed_enums.insert(name.clone());
            }
            for (k, v) in &g.struct_field_names {
                ir.enum_struct_field_names.insert(k.clone(), v.clone());
            }
            // Cross-module struct field names + types, so a module can resolve a
            // sibling-module struct's field (e.g. compile.mind reading
            // `analyzed.determinism` where `AnalyzedFlow` lives in sema.mind). The
            // module's OWN StructDef arm below re-inserts via last-write-wins.
            for (name, (field_names, field_types)) in &g.structs {
                ir.struct_defs
                    .entry(name.clone())
                    .or_insert_with(|| field_names.clone());
                ir.struct_field_types
                    .entry(name.clone())
                    .or_insert_with(|| field_types.clone());
            }
        });
    }
    // Pre-size the instruction buffer. `IRModule::new()` starts `instrs` at
    // capacity 0, so the AST→IR builder below grows it 0→4→8→16…, and the
    // profiler attributes the resulting `RawVec::finish_grow` realloc +
    // `memmove` chain to this hot path. Reserving once caps the realloc count
    // for small/medium modules. This is a CAPACITY hint only — it never
    // changes the instruction content or ordering, so emitted mic@1/mic@3
    // bytes (and cross-substrate identity) are byte-for-byte unchanged. The
    // estimate is O(1) (each top-level item expands to a handful of instrs);
    // any underestimate simply falls back to the existing growth path.
    ir.instrs
        .reserve(module.items.len().saturating_mul(8).max(16));
    // Codegen monomorphization pre-pass — collect generic fn TEMPLATES (those
    // with a non-empty `type_params`) so call sites can route to a concrete
    // instance. The thread-local is reset at entry so a prior `lower_to_ir`
    // call on this thread can never leak templates/requests into this one
    // (which would perturb `emit_mic3`/`trace_hash`). Functions with empty
    // `type_params` are never registered, so the non-generic path is untouched.
    MONO.with(|cell| {
        let mut ctx = cell.borrow_mut();
        *ctx = MonoCtx::default();
        for item in &module.items {
            if let ast::Node::FnDef {
                name, type_params, ..
            } = item
            {
                if !type_params.is_empty() {
                    ctx.templates.insert(name.clone(), item.clone());
                }
            }
        }
    });
    // Generic-arg inference pre-pass (0-fail-closed): record each NON-generic
    // fn's declared scalar return type so a generic call over a NESTED call
    // (`id(g(3))`) resolves to `g`'s return type. Reset each call (like MONO).
    // Gated behind templates-present so a non-generic module never builds it —
    // the byte-identity hot path executes zero extra work.
    FN_RETURNS.with(|fr| {
        let mut m = fr.borrow_mut();
        m.clear();
        if MONO.with(|c| !c.borrow().templates.is_empty()) {
            for item in &module.items {
                if let ast::Node::FnDef {
                    name,
                    ret_type,
                    type_params,
                    ..
                } = item
                {
                    if type_params.is_empty()
                        && let Some(rt) = ret_type
                        && mangle_suffix(rt).is_some()
                    {
                        m.insert(name.clone(), rt.clone());
                    }
                }
            }
        }
    });
    let mut env: HashMap<String, ValueId> = HashMap::new();
    // RFC 0005 P0f Step 1 — track `let x = Foo { ... }` so a later
    // `x.field` can resolve `Foo`'s canonical field-name order from
    // `ir.struct_defs` and emit the correct heap-record load offset.
    // Stays empty in non-std-surface builds; the FieldAccess arm and
    // the Let-side insert below are gated identically so the
    // side-table is dead-code-eliminated. `mut` is unused without the
    // feature, so silence the unused-mut lint instead of duplicating
    // the binding under a second cfg.
    #[allow(unused_mut)]
    let mut struct_env: HashMap<String, String> = HashMap::new();
    // RFC 0005 P0f Step 2 — module-wide side-table that maps every
    // `FieldAccess` span to its receiver's struct-type name. Built by
    // a single AST pre-pass so the FieldAccess arm in `lower_expr` can
    // resolve chained access (`a.b.c`), function-return receivers
    // (`foo().x`), and struct-typed parameters even when struct_env
    // doesn't have a direct Ident binding for the receiver. Internal
    // review (2026-05-18) picked this "type-checker annotation" approach
    // over a post-lowering IR rewrite. The builder lives in
    // src/eval/struct_resolver.rs; in non-feature builds the table is
    // empty and never queried.
    //
    // PROFILED HOT PATH: `build_field_access_types` walks the ENTIRE module AST
    // on every compile, yet every one of its `types.insert` sites is gated by
    // `struct_defs.iter().any(...)`, and `struct_defs` is collected EXCLUSIVELY
    // from top-level `Node::StructDef` items (struct_resolver.rs). A module with
    // no `StructDef` therefore PROVABLY yields an empty map. Guard the call on a
    // cheap O(items) shallow scan so struct-free modules (scalar_math, the
    // keystone, every canary) skip the whole-AST walk entirely. When skipped the
    // FieldAccess / MethodCall / FieldAssign arms simply query an empty table and
    // resolve to `None` exactly as they would against the empty map the walk
    // would have returned — so emitted mic@1/mic@3 bytes and cross-substrate
    // identity are byte-for-byte unchanged.
    //
    // CONSUMER-MODULE WIDENING: the shallow `any(StructDef)` scan is sufficient
    // ONLY for a self-contained module. A pure consumer module (no local
    // `StructDef`) that reads a SIBLING-module struct field still needs the
    // resolver to run — the cross-module struct names/field-types now live in
    // the global registry (and were merged into `ir.struct_defs` above). So also
    // run the resolver when that registry holds any struct. When the registry is
    // empty (the keystone, scalar_math, every single-file compile, every canary)
    // BOTH disjuncts are false for a struct-less module and the walk is skipped
    // exactly as before, so emitted mic@1/mic@3 bytes + cross-substrate identity
    // are byte-for-byte unchanged.
    #[cfg(feature = "std-surface")]
    let receiver_types_owned: HashMap<crate::ast::Span, String> = if module
        .items
        .iter()
        .any(|it| matches!(it, ast::Node::StructDef { .. }))
        || crate::ir::with_global_enums(|g| !g.structs.is_empty())
    {
        crate::eval::struct_resolver::build_field_access_types(module)
    } else {
        HashMap::new()
    };
    #[cfg(not(feature = "std-surface"))]
    let receiver_types_owned: HashMap<crate::ast::Span, String> = HashMap::new();
    let receiver_types: &HashMap<crate::ast::Span, String> = &receiver_types_owned;

    // RFC 0010 Phase B fix: two-pass repr_c collection.
    // Pass 0: collect ALL #[repr(C)] struct field types before processing any
    // ExternBlock nodes.  A single top-to-bottom pass caused a declaration-order
    // hazard: structs defined after the extern block were not in repr_c_structs
    // when the ExternBlock lowering ran, causing silent fallback to "i64" for
    // mixed-type structs that should classify to MEMORY (!llvm.ptr).
    #[cfg(feature = "std-surface")]
    for item in &module.items {
        if let ast::Node::StructDef {
            name,
            fields,
            attrs,
            ..
        } = item
        {
            if attrs
                .iter()
                .any(|a| a.name == "repr" && a.args.iter().any(|arg| arg == "C"))
            {
                let field_types: Vec<crate::ast::TypeAnn> =
                    fields.iter().map(|f| f.ty.clone()).collect();
                ir.repr_c_structs.insert(name.clone(), field_types);
            }
        }
    }

    for item in &module.items {
        match item {
            ast::Node::Let {
                name, ann, value, ..
            } => {
                let id = match ann {
                    Some(TypeAnn::Tensor { dtype, dims }) => lower_tensor_binding(
                        &mut ir,
                        value,
                        dtype,
                        dims,
                        &env,
                        &struct_env,
                        receiver_types,
                    ),
                    // `array<T>` binding whose RHS is an array literal `[..]`:
                    // lower onto the std.vec heap runtime (vec_new + vec_push
                    // chain) instead of the const-array/tensor path.
                    #[cfg(feature = "std-surface")]
                    _ if (is_array_surface_type(ann)
                        && matches!(value.as_ref(), ast::Node::ArrayLit { .. }))
                        || is_growable_bytes_init(ann, value) =>
                    {
                        let elements = match value.as_ref() {
                            ast::Node::ArrayLit { elements, .. } => elements.as_slice(),
                            _ => &[],
                        };
                        lower_array_surface_lit(
                            elements,
                            &mut ir,
                            &env,
                            &struct_env,
                            receiver_types,
                        )
                    }
                    _ => lower_expr(value, &mut ir, &env, &struct_env, receiver_types),
                };
                // Narrow-typed binding (`let c: u8 = a * b`, i8/u8/i16/u16/u32):
                // truncate/sign-adjust the lowered init to the declared width, so
                // later arithmetic on `c` sees the real-width value (no-op for
                // i64/u64/handles/collections — preserves byte-identity).
                #[cfg(feature = "std-surface")]
                let id = mask_narrow_let(&mut ir, ann, id);
                // Track a module-level narrow local so a later top-level `c = c + …`
                // reassignment re-masks (no-op for non-narrow; per-fn scopes take +
                // restore this map, so the entry never leaks into a fn body).
                #[cfg(feature = "std-surface")]
                record_narrow_let(name, ann);
                env.insert(name.clone(), id);
                // Dynamic `bytes = [..]` tracks as the vec surface (growable u8).
                #[cfg(feature = "std-surface")]
                if is_growable_bytes_init(ann, value) {
                    struct_env.insert(name.clone(), ARRAY_VEC_SENTINEL.to_string());
                }
                // P0f Step 1: if the RHS is a StructLit, record the var→type
                // binding so a later FieldAccess on this name resolves the
                // correct offset out of `ir.struct_defs`.
                #[cfg(feature = "std-surface")]
                if let ast::Node::StructLit {
                    name: struct_name, ..
                } = value.as_ref()
                {
                    struct_env.insert(name.clone(), struct_name.clone());
                }
                // `let x = y` where `y` is a tracked struct/collection local or
                // param: `x` aliases `y`'s type (and element tracking), so a
                // field/method/index on `x` resolves — e.g. `let next = t;
                // next.scopes.push(..)` with `t: SomeStruct`.
                #[cfg(feature = "std-surface")]
                if let ast::Node::Lit(Literal::Ident(src), _) = value.as_ref() {
                    if let Some(t) = struct_env.get(src).cloned() {
                        struct_env.entry(name.clone()).or_insert(t);
                    }
                    if let Some(e) = struct_env.get(&format!("__elem__{src}")).cloned() {
                        struct_env
                            .entry(format!("__elem__{name}"))
                            .or_insert(e);
                    }
                }
                // `array<T>` binding: record the vec sentinel so a later
                // `arr.push/get/set/len/length` or `arr[i]` resolves to the
                // std.vec runtime. Pure metadata (never serialized into mic@3).
                #[cfg(feature = "std-surface")]
                if is_array_surface_type(ann) {
                    struct_env.insert(name.clone(), ARRAY_VEC_SENTINEL.to_string());
                    if let Some(__e) = ann.as_ref().and_then(array_element_track) {
                        struct_env.insert(format!("__elem__{}", name), __e);
                    }
                }
                // `set<T>` binding: record the set sentinel so `.contains/.add/.len`
                // resolve to the std.map runtime.
                #[cfg(feature = "std-surface")]
                if let Some(s) = set_sentinel_for_opt(ann) {
                    struct_env.insert(name.clone(), s.to_string());
                }
                // `let x = f(...)` / `let x = s.field` — infer x's type from the
                // RHS so a method on x resolves without an annotation (e.g.
                // `let raw = decorator_arg_string(d); raw.split(…)`). Annotations
                // (handled above) win via or_insert.
                #[cfg(feature = "std-surface")]
                if let Some((__s, __e)) =
                    let_rhs_collection_track(value, &ir, &struct_env, receiver_types)
                {
                    struct_env.entry(name.clone()).or_insert(__s);
                    if let Some(__el) = __e {
                        struct_env
                            .entry(format!("__elem__{}", name))
                            .or_insert(__el);
                    }
                }
                // `map<K, V>` binding: record the map sentinel (str-key vs i64-key)
                // so `m.insert/.get/.contains_key/.len` resolve to std.map.
                #[cfg(feature = "std-surface")]
                if let Some(s) = map_sentinel_for_opt(ann) {
                    struct_env.insert(name.clone(), s.to_string());
                }
                ir.instrs.push(Instr::Output(id));
            }
            ast::Node::Assign { name, value, .. } => {
                let id = lower_expr(value, &mut ir, &env, &struct_env, receiver_types);
                // Reassigning a module-level narrow local re-masks to its declared
                // width (no-op for non-narrow names — the all-i64 hot path).
                #[cfg(feature = "std-surface")]
                let id = mask_narrow_assign(&mut ir, name, id);
                env.insert(name.clone(), id);
                ir.instrs.push(Instr::Output(id));
            }
            ast::Node::Export { names, .. } => {
                // RFC 0002, deliverable 1: lower the parsed `export { ... }`
                // block into `IRModule.exports`. Verification that named
                // functions exist lands in deliverable 2 under
                // `feature = "ffi-c-user"` together with the codegen pass.
                ir.exports.extend(names.iter().cloned());
            }
            // RFC 0005 P0e Step 1 — record the struct's field-name order in
            // the schema registry so a later `StructLit` can reorder
            // literal fields into canonical order before emitting stores.
            // The placeholder `Output(ConstI64(0))` is preserved to keep
            // the IR-shape contract that downstream consumers (verifier,
            // canonicaliser, MLIR emitter) rely on for declaration-only
            // modules — a struct declaration is still a no-op at the
            // value level, the side-table is pure metadata.
            #[cfg(feature = "std-surface")]
            ast::Node::StructDef {
                name,
                fields,
                attrs,
                ..
            } => {
                let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
                ir.struct_defs.insert(name.clone(), field_names);
                // Record EVERY struct's declared field types (declaration order)
                // for the width-aware struct ABI lowering. Serialization-exempt
                // (see `struct_field_types` in ir/mod.rs); unused until the
                // lowering arms consume it, so an all-i64 struct is unchanged.
                let all_field_types: Vec<crate::ast::TypeAnn> =
                    fields.iter().map(|f| f.ty.clone()).collect();
                ir.struct_field_types.insert(name.clone(), all_field_types);
                // RFC 0010 Phase B: if the struct carries `#[repr(C)]`, register
                // its field types in `repr_c_structs` so extern_type_to_mlir can
                // classify Named types that appear in `extern "C"` signatures.
                let is_repr_c = attrs
                    .iter()
                    .any(|a| a.name == "repr" && a.args.iter().any(|arg| arg == "C"));
                if is_repr_c {
                    let field_types: Vec<crate::ast::TypeAnn> =
                        fields.iter().map(|f| f.ty.clone()).collect();
                    ir.repr_c_structs.insert(name.clone(), field_types);
                }
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                ir.instrs.push(Instr::Output(id));
            }
            // RFC 0005 Phase 6.2b Gap 2 — module-level `const NAME: [i64; N] = [...]`.
            // Lowers to a named ConstArray IR node and also registers the
            // element data in `ir.const_array_defs` so that fn bodies (which
            // use a fresh SSA namespace) can re-emit the blob on demand.
            #[cfg(feature = "std-surface")]
            ast::Node::Const {
                name,
                ty: Some(TypeAnn::Array { .. }),
                value,
                ..
            } => {
                let values = extract_array_lit_values(value);
                ir.const_array_defs.insert(name.clone(), values.clone());
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstArray {
                    dst: id,
                    name: Some(name.clone()),
                    values,
                });
                env.insert(name.clone(), id);
                ir.instrs.push(Instr::Output(id));
            }
            // "finish MIND" Step 2 — record each enum variant's ordinal i64
            // tag (`0, 1, 2, …` in declaration order) under its fully-qualified
            // path (`"Mode::On"`) so that (a) the `Lit(Ident)` arm can lower a
            // variant used as a value to its tag and (b) `desugar_match_to_if`
            // can compare a scrutinee against a fieldless variant's tag. An
            // enum declaration is a no-op at the value level, so the
            // `ConstI64(0)`/`Output` placeholder is preserved for the
            // declaration-only IR-shape contract (mirrors `StructDef`).
            #[cfg(feature = "std-surface")]
            ast::Node::EnumDef { name, variants, .. } => {
                for (ordinal, variant) in variants.iter().enumerate() {
                    let path = format!("{name}::{}", variant.name);
                    ir.enum_variant_tags.insert(path, ordinal as i64);
                }
                // A "boxed" enum carries a payload on ≥1 variant. Record it so
                // EVERY constructor of this enum (including its fieldless
                // variants) lowers to the uniform heap record, keeping the
                // match's `__mind_load_i64(scrutinee + 0)` tag-read valid. The
                // record is `1 + max payload arity` i64 slots (tag + the widest
                // variant's fields), used for every variant so any arm's
                // field-load addresses valid memory.
                let max_arity = variants.iter().map(|v| v.payload.len()).max().unwrap_or(0);
                if max_arity > 0 {
                    ir.boxed_enums.insert(name.clone());
                    ir.enum_payload_slots.insert(name.clone(), 1 + max_arity);
                    // Record each variant's declared field types so the ctor and
                    // match desugar can coerce a non-i64 field across the i64 slot.
                    for variant in variants {
                        if !variant.payload.is_empty() {
                            ir.enum_payload_types.insert(
                                format!("{name}::{}", variant.name),
                                variant.payload.clone(),
                            );
                        }
                        // Struct-variant field-name order, so a named construction
                        // / match resolves each field to its declared slot.
                        if !variant.field_names.is_empty() {
                            ir.enum_struct_field_names.insert(
                                format!("{name}::{}", variant.name),
                                variant.field_names.clone(),
                            );
                        }
                    }
                }
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                ir.instrs.push(Instr::Output(id));
            }
            other => {
                let id = lower_expr(other, &mut ir, &env, &struct_env, receiver_types);
                ir.instrs.push(Instr::Output(id));
            }
        }
    }

    // ── Monomorphization drain (codegen generics) ──────────────────────────
    // Emit a concrete `FnDef` body for every generic instance requested during
    // the body lowering above (via `try_register_mono_instance`). A non-generic
    // module registers zero templates, hence zero requests, so this loop never
    // runs and its IR — and therefore the mic@3 fixed point, the keystone, and
    // the cross-substrate canaries — is byte-identical. Instances drain in
    // BTreeMap mangled-name (lexicographic) order: deterministic, with no
    // HashMap iteration / clock / rng / address bits, so avx2 == neon.
    let (templates, mut pending) = MONO.with(|cell| {
        let ctx = cell.borrow();
        (ctx.templates.clone(), ctx.requests.clone())
    });
    let mut emitted: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    while let Some(mangled) = pending.keys().next().cloned() {
        let req = pending
            .remove(&mangled)
            .expect("key just taken from pending");
        if !emitted.insert(mangled.clone()) {
            continue;
        }
        let Some(template) = templates.get(&req.generic_name) else {
            continue;
        };
        let Some(instance) = instantiate_template(template, &mangled, &req.concrete) else {
            continue;
        };
        // Body-ABI safety net: `instantiate_template` rewrites only the
        // signature. If the template body still names the type parameter in a
        // type position (`let r: T = ...`), the concrete instance would silently
        // classify that binding at the default i64 ABI — a silent mis-ABI
        // miscompile. Refuse such an instance: leaving its symbol body-less
        // surfaces a LOUD undefined-symbol link error instead of a confidently-
        // wrong body. The current literal-inferred slice never produces such a
        // body, so this only guards future, out-of-subset shapes.
        // deferred: full body type-substitution is a later monomorphization slice.
        if let ast::Node::FnDef { type_params, .. } = template {
            if let (Some(tp), ast::Node::FnDef { body, .. }) = (type_params.first(), &instance) {
                if body.iter().any(|n| node_mentions_type_name(n, tp)) {
                    continue;
                }
            }
        }
        // Lower the concrete instance through the ordinary FnDef path: it records
        // the instance signature and pushes an `Instr::FnDef` carrying a real
        // body (the empty `type_params` skips the template short-circuit).
        let _ = lower_expr(&instance, &mut ir, &env, &struct_env, receiver_types);
        // Closure: lowering an instance may itself register further generic
        // instances (a generic body that calls another generic). Merge any
        // not-yet-emitted requests; the idempotent `or_insert` on the fixed
        // mangled key makes the fixed point order-independent.
        MONO.with(|cell| {
            for (k, v) in &cell.borrow().requests {
                if !emitted.contains(k) {
                    pending.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        });
    }

    ir
}

fn lower_tensor_binding(
    ir: &mut IRModule,
    value: &ast::Node,
    dtype: &str,
    dims: &[String],
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> ValueId {
    if let Some((dtype, shape)) = parse_tensor_ann(dtype, dims) {
        match value {
            ast::Node::Lit(Literal::Int(n), _) => {
                let id = ir.fresh();
                ir.instrs
                    .push(Instr::ConstTensor(id, dtype, shape, Some(*n as f64)));
                return id;
            }
            ast::Node::Lit(Literal::Float(f), _) => {
                let id = ir.fresh();
                ir.instrs
                    .push(Instr::ConstTensor(id, dtype, shape, Some(*f)));
                return id;
            }
            ast::Node::Lit(Literal::Ident(name), _) => {
                if let Some(id) = env.get(name) {
                    return *id;
                }
            }
            // Negated literal tensor fill (`let t: f32[4] = -1.0`). Without
            // this, the negative fill value fell through to `lower_expr` and
            // lost its tensor shape; fold the sign into the fill scalar.
            ast::Node::Neg { operand, .. } => match operand.as_ref() {
                ast::Node::Lit(Literal::Int(n), _) => {
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstTensor(
                        id,
                        dtype,
                        shape,
                        Some(n.wrapping_neg() as f64),
                    ));
                    return id;
                }
                ast::Node::Lit(Literal::Float(f), _) => {
                    let id = ir.fresh();
                    ir.instrs
                        .push(Instr::ConstTensor(id, dtype, shape, Some(-*f)));
                    return id;
                }
                _ => {}
            },
            // Dense tensor literal `let t: tensor<f32[3]> = [1.0, 2.0, 3.0]`.
            // Materialise the EXACT per-element bits via ConstDenseTensor (the
            // generic i64 ConstArray path coerced floats to 0 and registered no
            // tensor type, so `t + u` failed "missing type information"). Only
            // emit when the element count matches a fully-known shape; otherwise
            // fall through so the type checker reports the shape/literal mismatch.
            ast::Node::ArrayLit { elements, .. } => {
                let expected: usize = shape
                    .iter()
                    .map(|d| match d {
                        ShapeDim::Known(n) => *n as usize,
                        _ => 0,
                    })
                    .product();
                if expected == elements.len() {
                    let data: Vec<u64> = elements
                        .iter()
                        .map(|e| dense_elem_bits(e, &dtype))
                        .collect();
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstDenseTensor {
                        dst: id,
                        dtype,
                        shape,
                        data,
                    });
                    return id;
                }
            }
            _ => {}
        }
    }

    lower_expr(value, ir, env, struct_env, receiver_types)
}

/// Signed-integer bit-width of a scalar `as`-cast *target* type, if the target
/// is a known fixed-width integer that the scalar i64 ABI must materialise at
/// its real width. Returns `None` for non-integer / pointer / alias / unsigned
/// targets, which lower transparently (the i64 SSA value is left unchanged).
///
/// MIND integers are signed, so a narrowing cast to one of these widths must
/// truncate + sign-extend (the caller emits the `(x << k) >> k` shift pair).
/// Only the canonical signed widths are mapped; `u32`/`u64`/`i64` and aliases
/// stay full-width (no narrowing op), preserving the prior pass-through for
/// pointers and same-width casts.
#[cfg(feature = "std-surface")]
fn scalar_int_cast_width(ty: &TypeAnn) -> Option<u32> {
    match ty {
        TypeAnn::ScalarI32 => Some(32),
        TypeAnn::ScalarI64 => Some(64),
        TypeAnn::Named(name) => match name.as_str() {
            "i8" => Some(8),
            "i16" => Some(16),
            "i32" => Some(32),
            "i64" => Some(64),
            _ => None,
        },
        _ => None,
    }
}

/// Unsigned-integer bit-width of a scalar `as`-cast (or call-form `uN(x)`)
/// *target* type, if the target is a narrowing unsigned width (`u8`/`u16`/`u32`)
/// the scalar i64 ABI must materialise at its real width by *zero*-extending.
///
/// MIND's scalar values are carried in i64. A narrowing cast to an unsigned
/// width must keep only the low `width` bits with the high bits CLEARED (zero
/// extension), unlike the signed path which sign-extends. The caller emits a
/// `BitAnd` against the i64 const mask `(1 << width) - 1`. Returns `None` for
/// full-width (`u64`) and any non-narrowing-unsigned target, which lower
/// transparently (the i64 SSA value is left unchanged).
///
/// `u32` reaches this both as `TypeAnn::ScalarU32` (postfix `as u32` and the
/// `u32(x)` call-form) and as `TypeAnn::Named("u32")`; `u8`/`u16` arrive as
/// `TypeAnn::Named`. All map to their mask width here.
#[cfg(feature = "std-surface")]
fn scalar_uint_cast_width(ty: &TypeAnn) -> Option<u32> {
    match ty {
        TypeAnn::ScalarU32 => Some(32),
        TypeAnn::Named(name) => match name.as_str() {
            "u8" => Some(8),
            "u16" => Some(16),
            "u32" => Some(32),
            _ => None,
        },
        _ => None,
    }
}

/// Truncate/sign-adjust the lowered init value `val` of a `let` binding to its
/// declared narrow integer width, mirroring the `as`-cast scalar path.
///
/// MIND scalars are carried full-width in i64 SSA. A `let c: u8 = a * b` (or
/// `i8`/`u8`/`i16`/`u16`/`u32`) must materialise `c` at its real width — without
/// this, the un-truncated i64 product flows into subsequent arithmetic on `c`,
/// a silent miscompile (e.g. `200 * 2 == 400` instead of `144`). The catch-all
/// `_ => lower_expr(value, ..)` Let arms previously dropped the declared width.
///
///   * narrow SIGNED (`i8`/`i16`/`i32`) → the `(x << (64-W)) >> (64-W)` shift
///     pair with an arithmetic (signed) right shift — clears high bits AND
///     sign-extends in one i64-carried value (same as `scalar_int_cast_width`).
///   * narrow UNSIGNED (`u8`/`u16`/`u32`) → a single `BitAnd` against the i64
///     const mask `(1 << W) - 1` (zero-extend, same as `scalar_uint_cast_width`).
///   * `i64`/`u64`/pointers/floats/handles/aliases/no annotation → unchanged
///     (`val` returned verbatim, so i64 locals and the keystone are byte-identical).
///
/// No new IR opcode and no mic@1/mic@3 layout change (only `ConstI64`/`BinOp`),
/// so no version bump. Gated to `std-surface` because the narrow scalar types and
/// the `Shl`/`Shr`/`BitAnd` ops only exist there.
#[cfg(feature = "std-surface")]
fn mask_narrow_let(ir: &mut IRModule, ann: &Option<TypeAnn>, val: ValueId) -> ValueId {
    let ty = match ann {
        Some(t) => t,
        None => return val,
    };
    // Signed narrow widths: shift-pair sign-extend.
    if let Some(width) = scalar_int_cast_width(ty) {
        if width < 64 {
            let shift = 64 - width as i64;
            let shift_id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(shift_id, shift));
            let shl_id = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: shl_id,
                op: BinOp::Shl,
                lhs: val,
                rhs: shift_id,
            });
            let shr_id = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: shr_id,
                op: BinOp::Shr,
                lhs: shl_id,
                rhs: shift_id,
            });
            return shr_id;
        }
        // width == 64 (i64): leave the value untouched.
        return val;
    }
    // Unsigned narrow widths: BitAnd zero-extend mask.
    if let Some(width) = scalar_uint_cast_width(ty) {
        if width < 64 {
            let mask: i64 = if width == 32 {
                0xFFFF_FFFF
            } else {
                (1i64 << width) - 1
            };
            let mask_id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(mask_id, mask));
            let and_id = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: and_id,
                op: BinOp::BitAnd,
                lhs: val,
                rhs: mask_id,
            });
            return and_id;
        }
    }
    val
}

/// True when `ty` is a NARROW integer scalar that `mask_narrow_let` actually
/// re-materialises at sub-i64 width (`i8`/`i16`/`i32`/`u8`/`u16`/`u32`). `i64`,
/// `u64`, pointers, handles, floats and aliases all return `false` — they carry
/// their full i64 representation unchanged, so they never need re-masking on
/// reassignment and never enter the narrow-locals registry (keeping it empty for
/// any all-i64 module, the keystone included).
#[cfg(feature = "std-surface")]
fn is_narrow_scalar_ty(ty: &TypeAnn) -> bool {
    matches!(scalar_int_cast_width(ty), Some(w) if w < 64)
        || matches!(scalar_uint_cast_width(ty), Some(w) if w < 64)
}

thread_local! {
    /// Per-fn registry of NARROW-typed locals/params (`name -> declared TypeAnn`).
    /// A `let c: u8 = …` masks its initializer to 8 bits (`mask_narrow_let`), but a
    /// later REASSIGNMENT `c = c + 100` (and the `c += 100` that desugars to it)
    /// carries no annotation — without re-masking, `c` silently keeps the full i64
    /// value (`300` instead of `44`), a SILENT MISCOMPILE that spans top-level,
    /// branch and loop bodies. `Assign` lowering consults this map and re-applies
    /// `mask_narrow_let`. Populated ONLY with genuinely narrow scalars (see
    /// `is_narrow_scalar_ty`), so an all-i64 module leaves it empty and the masking
    /// is a no-op (zero extra IR, byte-identical hot path). Scoped per fn body and
    /// restored on exit via `NarrowLocalsGuard`, so it never leaks across fns.
    #[cfg(feature = "std-surface")]
    static NARROW_LOCALS: std::cell::RefCell<std::collections::HashMap<String, TypeAnn>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// RAII guard restoring the previous `NARROW_LOCALS` map when a fn body's lowering
/// scope ends, mirroring `ParamTypesGuard`. Seeds the map with this fn's narrow
/// PARAMS so a reassigned narrow param (`fn f(c: u8) { c = c + 100 }`) re-masks too.
#[cfg(feature = "std-surface")]
struct NarrowLocalsGuard(std::collections::HashMap<String, TypeAnn>);

#[cfg(feature = "std-surface")]
impl Drop for NarrowLocalsGuard {
    fn drop(&mut self) {
        NARROW_LOCALS.with(|n| *n.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Begin a fresh narrow-locals scope for a fn body, seeded with its narrow
/// params. Returns the restore guard. Cheap (one HashMap swap) and only inserts
/// narrow-typed params, so a non-narrow fn keeps an empty map.
#[cfg(feature = "std-surface")]
fn enter_narrow_scope(params: &[ast::Param]) -> NarrowLocalsGuard {
    NARROW_LOCALS.with(|n| {
        let prev = std::mem::take(&mut *n.borrow_mut());
        {
            let mut m = n.borrow_mut();
            for prm in params {
                if is_narrow_scalar_ty(&prm.ty) {
                    m.insert(prm.name.clone(), prm.ty.clone());
                }
            }
        }
        NarrowLocalsGuard(prev)
    })
}

/// Record a narrow-typed `let` so a later reassignment of `name` re-masks.
/// No-op for non-narrow annotations (keeps the registry empty for i64 modules).
#[cfg(feature = "std-surface")]
fn record_narrow_let(name: &str, ann: &Option<TypeAnn>) {
    if let Some(ty) = ann {
        if is_narrow_scalar_ty(ty) {
            NARROW_LOCALS.with(|n| n.borrow_mut().insert(name.to_string(), ty.clone()));
        }
    }
}

/// Re-apply the declared narrow-width mask after a reassignment `name = …`.
/// Looks `name` up in the per-fn narrow-locals registry; emits the same
/// truncate/sign-extend `mask_narrow_let` uses, or returns `val` unchanged when
/// `name` is not a tracked narrow local (the byte-identical all-i64 path).
#[cfg(feature = "std-surface")]
fn mask_narrow_assign(ir: &mut IRModule, name: &str, val: ValueId) -> ValueId {
    let ann = NARROW_LOCALS.with(|n| n.borrow().get(name).cloned());
    match ann {
        Some(ty) => mask_narrow_let(ir, &Some(ty), val),
        None => val,
    }
}

/// The byte width and signedness of a struct field type for the canonical
/// width-aware struct ABI. Returns `(width_bytes, signed)`:
///   * `i64`/`u64`/struct-handle (`Named` non-narrow)/pointer  → 8, signed
///   * `i32`/`u32`                                             → 4
///   * `i16`/`u16`                                             → 2
///   * `i8`/`u8`/`bool`                                        → 1
///
/// `signed` is true only for the signed integer scalars (`i64`/`i32`/`i16`/`i8`);
/// `u*`/`bool`/handles are unsigned (zero-extended on load). Any field that is
/// not a recognised scalar (a nested struct handle, a `Vec`/`String`/`Map`
/// handle, etc.) is an i64-wide handle.
#[cfg(feature = "std-surface")]
fn struct_field_width(ty: &TypeAnn) -> (i64, bool) {
    match ty {
        TypeAnn::ScalarI64 => (8, true),
        TypeAnn::ScalarI32 => (4, true),
        TypeAnn::ScalarU32 => (4, false),
        TypeAnn::ScalarBool => (1, false),
        TypeAnn::Named(n) => match n.as_str() {
            "i8" => (1, true),
            "u8" => (1, false),
            "i16" => (2, true),
            "u16" => (2, false),
            "i32" => (4, true),
            "u32" => (4, false),
            "i64" => (8, true),
            // u64 and every other Named type (nested struct / Vec / String /
            // Map handle, type alias) is an i64-wide value.
            _ => (8, false),
        },
        // Floats are handled by the existing loud lowering error, not here;
        // anything else is treated as an i64-wide handle (8 bytes).
        _ => (8, false),
    }
}

/// The `__mind_store_i{N}` intrinsic name for a field byte width.
#[cfg(feature = "std-surface")]
fn store_helper_for_width(width: i64) -> &'static str {
    match width {
        1 => "__mind_store_i8",
        2 => "__mind_store_i16",
        4 => "__mind_store_i32",
        _ => "__mind_store_i64",
    }
}

/// The `__mind_load_i{N}` intrinsic name for a field byte width.
#[cfg(feature = "std-surface")]
fn load_helper_for_width(width: i64) -> &'static str {
    match width {
        1 => "__mind_load_i8",
        2 => "__mind_load_i16",
        4 => "__mind_load_i32",
        _ => "__mind_load_i64",
    }
}

/// Canonical per-field layout for a struct: `(offset, width_bytes, signed)` in
/// declaration order, plus the total allocation size. Offsets are a pure
/// function of the declared field widths (self-aligned: each field starts at the
/// next multiple of its own width), so the layout is identical on every
/// substrate — no host `sizeof`/`alignof`, no target-dependent padding. Returns
/// `None` when the field-type side-table has no entry for `name` (an unknown /
/// forward-referenced struct), so callers fall back to the legacy 8-byte-stride
/// path. `all_i64` is true when every field is 8 bytes wide AND tightly packed
/// at `8*i` — the case where the legacy `__mind_alloc(8*n)` + `store_i64` IR is
/// byte-identical and must be preserved verbatim.
/// One field's resolved placement within a struct: `(byte_offset, width_bytes,
/// signed)`. Offsets are self-aligned and substrate-independent (see
/// `struct_layout`).
#[cfg(feature = "std-surface")]
type FieldPlacement = (i64, i64, bool);

/// A struct's fully-resolved layout: each field's placement in declaration
/// order, the total allocation size in bytes, and `all_i64` (every field is an
/// 8-byte tightly-packed slot — the legacy byte-identical `__mind_alloc(8*n)`
/// path).
#[cfg(feature = "std-surface")]
type StructLayout = (Vec<FieldPlacement>, i64, bool);

#[cfg(feature = "std-surface")]
fn struct_layout(ir: &IRModule, name: &str) -> Option<StructLayout> {
    let field_types = ir.struct_field_types.get(name)?;
    let mut layout = Vec::with_capacity(field_types.len());
    let mut running: i64 = 0;
    let mut all_i64 = true;
    for ty in field_types {
        let (w, signed) = struct_field_width(ty);
        // Self-aligned offset: round `running` up to a multiple of `w`.
        let offset = (running + (w - 1)) / w * w;
        if w != 8 || offset != (layout.len() as i64) * 8 {
            all_i64 = false;
        }
        layout.push((offset, w, signed));
        running = offset + w;
    }
    Some((layout, running, all_i64))
}

/// Sentinel "struct type name" recorded in `struct_env` for an `array<T>`-typed
/// binding. It is deliberately the lowercase string `"vec"` so the existing UFCS
/// method-call desugar (`{lowercase(T)}_{method}`) resolves `arr.push(x)` to the
/// `vec_push` free function in `std/vec.mind` with no special-case branch. It is
/// NOT a real struct (`ir.struct_defs` has no `"vec"` entry — the runtime struct
/// is `Vec`), so the zero-arg field-accessor fast path never matches it and every
/// `array<T>` method/index falls through to the vec runtime mapping.
#[cfg(feature = "std-surface")]
const ARRAY_VEC_SENTINEL: &str = "vec";

/// True when `ann` is the dynamic-array surface type `array<T>` (RFC 0005 vec
/// surface). Parsed as `TypeAnn::Generic { name: "array", .. }`. The fixed-size
/// `[T; N]` LUT type (`TypeAnn::Array`) and the slice `[T]` (`TypeAnn::Slice`)
/// are distinct and intentionally NOT matched — those keep the const-array /
/// tensor lowering, so the keystone (which uses neither `array<T>`) is untouched.
#[cfg(feature = "std-surface")]
fn is_array_surface_type(ann: &Option<TypeAnn>) -> bool {
    matches!(ann, Some(t) if is_array_surface_ty(t))
}

/// A `let x: bytes = [..]` binding — dynamic `bytes` INITIALISED from an array
/// literal is a growable `Vec<u8>` (`buf.push(b)`), so it lowers onto the std.vec
/// runtime exactly like `array<u8>`. Gated on the `[..]` initialiser so a raw
/// byte VIEW (a `bytes` struct field / param read from data, NOT freshly built)
/// is left untouched — conflating the two would miscompile the view's indexing.
#[cfg(feature = "std-surface")]
fn is_growable_bytes_init(ann: &Option<TypeAnn>, value: &ast::Node) -> bool {
    matches!(ann, Some(TypeAnn::Named(n)) if n == "bytes")
        && matches!(value, ast::Node::ArrayLit { .. })
}

/// `&TypeAnn` form of [`is_array_surface_type`], for params/fields whose type is
/// a bare `TypeAnn` (not `Option<TypeAnn>`).
#[cfg(feature = "std-surface")]
fn is_array_surface_ty(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::Generic { name, .. } if name == "array")
}

/// Lower a dynamic-array literal `[a, b, c]` onto the `std.vec` heap runtime:
///
/// ```text
///   let _v = vec_new();
///   _v = vec_push(_v, a);   // vec_push returns the (possibly realloc'd) handle
///   _v = vec_push(_v, b);
///   _v = vec_push(_v, c);
///   _v                       // final handle is the array value
/// ```
///
/// `[]` lowers to a bare `vec_new()`. The handle is an opaque i64 — no pointer
/// bits, no const-tensor, no `tensor.extract` (which does not bufferize in the
/// build pipeline). Returns the SSA id holding the final vec handle.
#[cfg(feature = "std-surface")]
fn lower_array_surface_lit(
    elements: &[ast::Node],
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> ValueId {
    let mut handle = ir.fresh();
    ir.instrs.push(Instr::Call {
        dst: handle,
        name: "vec_new".to_string(),
        args: vec![],
    });
    for elem in elements {
        let val = lower_expr(elem, ir, env, struct_env, receiver_types);
        let next = ir.fresh();
        ir.instrs.push(Instr::Call {
            dst: next,
            name: "vec_push".to_string(),
            args: vec![handle, val],
        });
        handle = next;
    }
    handle
}

/// Lower a `StructLit` FIELD value. A field declared `array<T>` whose literal is
/// `[..]` must lower onto the std.vec heap runtime (a registered i64 handle), not
/// the generic `ArrayLit` const-array/tensor path — whose result is a non-i64
/// aggregate the field's `__mind_store_i64` cannot accept (it surfaces as the
/// "non-i64 argument to call" aggregate-ABI error). Every other field value
/// lowers normally.
#[cfg(feature = "std-surface")]
fn lower_struct_field_value(
    struct_name: &str,
    field_name: &str,
    value: &ast::Node,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> ValueId {
    if let ast::Node::ArrayLit { elements, .. } = value {
        let is_arr_field = ir
            .struct_defs
            .get(struct_name)
            .and_then(|names| names.iter().position(|n| n == field_name))
            .and_then(|idx| {
                ir.struct_field_types
                    .get(struct_name)
                    .and_then(|ts| ts.get(idx))
            })
            .map(is_array_surface_ty)
            .unwrap_or(false);
        if is_arr_field {
            return lower_array_surface_lit(elements, ir, env, struct_env, receiver_types);
        }
    }
    lower_expr(value, ir, env, struct_env, receiver_types)
}

// `map<K, V>` surface over the std.map heap runtime (i64 handles). Two
// sentinels distinguish the KEY type so a lookup picks the correct comparison:
// `MAP_SENTINEL` (i64-identity keys → map_get / map_contains_key) lowercases to
// `map`, so the UFCS method desugar resolves `m.insert/.get/.len` for free;
// `MAP_STR_SENTINEL` (String keys → map_get_str / map_contains_key_str, content
// equality) is routed explicitly in the MethodCall arm. A handle `==` on two
// String keys would compare pointers, not bytes — the wrong answer — so the
// key-type split is load-bearing for correctness, not an optimization.
#[cfg(feature = "std-surface")]
const MAP_SENTINEL: &str = "map";
#[cfg(feature = "std-surface")]
const MAP_STR_SENTINEL: &str = "mapstr";

/// True when `ann` is the dynamic-map surface type `map<K, V>`
/// (`TypeAnn::Generic { name: "map", .. }`).
#[cfg(feature = "std-surface")]
fn is_map_surface_ty(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::Generic { name, .. } if name == "map")
}

/// The struct-env sentinel for a `map<K, V>`-typed binding: `MAP_STR_SENTINEL`
/// when the key type `K` is `string` (content-equality lookups), else
/// `MAP_SENTINEL` (i64-identity lookups).
#[cfg(feature = "std-surface")]
fn map_sentinel_for(ty: &TypeAnn) -> &'static str {
    let key_is_string = matches!(
        ty,
        TypeAnn::Generic { name, args } if name == "map"
            && matches!(args.first(), Some(TypeAnn::Named(n)) if n == "string")
    );
    if key_is_string {
        MAP_STR_SENTINEL
    } else {
        MAP_SENTINEL
    }
}

/// `Option<TypeAnn>` form for let-binding annotations.
#[cfg(feature = "std-surface")]
fn map_sentinel_for_opt(ann: &Option<TypeAnn>) -> Option<&'static str> {
    match ann {
        Some(t) if is_map_surface_ty(t) => Some(map_sentinel_for(t)),
        _ => None,
    }
}

/// Lower a map literal `{}` / `{ k: v, … }` onto the std.map heap runtime:
/// `let _m = map_new(); _m = map_insert(_m, k, v); …; _m`. `map_insert` returns
/// the (possibly grown) handle, threaded through each entry. Returns the SSA id
/// holding the final map handle.
#[cfg(feature = "std-surface")]
fn lower_map_surface_lit(
    entries: &[(ast::Node, ast::Node)],
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> ValueId {
    let mut handle = ir.fresh();
    ir.instrs.push(Instr::Call {
        dst: handle,
        name: "map_new".to_string(),
        args: vec![],
    });
    for (key, value) in entries {
        let k = lower_expr(key, ir, env, struct_env, receiver_types);
        let v = lower_expr(value, ir, env, struct_env, receiver_types);
        let next = ir.fresh();
        ir.instrs.push(Instr::Call {
            dst: next,
            name: "map_insert".to_string(),
            args: vec![handle, k, v],
        });
        handle = next;
    }
    handle
}

// `set<T>` is a map keyed by its elements (value 1). Two sentinels mirror the
// map ones for the element-type comparison: `SET_SENTINEL` (i64 identity) and
// `SET_STR_SENTINEL` (String content equality). The MethodCall arm routes
// `.contains`→map_contains_key(_str), `.add`/`.insert`→map_insert(recv, x, 1),
// `.len`→map_len.
#[cfg(feature = "std-surface")]
const SET_SENTINEL: &str = "set";
#[cfg(feature = "std-surface")]
const SET_STR_SENTINEL: &str = "setstr";

/// True when `ann` is the `set<T>` surface type.
#[cfg(feature = "std-surface")]
fn is_set_surface_ty(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::Generic { name, .. } if name == "set")
}

/// The sentinel for a `set<T>`-typed binding: `SET_STR_SENTINEL` when the
/// element type `T` is `string`, else `SET_SENTINEL`.
#[cfg(feature = "std-surface")]
fn set_sentinel_for(ty: &TypeAnn) -> &'static str {
    let elem_is_string = matches!(
        ty,
        TypeAnn::Generic { name, args } if name == "set"
            && matches!(args.first(), Some(TypeAnn::Named(n)) if n == "string")
    );
    if elem_is_string {
        SET_STR_SENTINEL
    } else {
        SET_SENTINEL
    }
}

/// `Option<TypeAnn>` form for let-binding annotations.
#[cfg(feature = "std-surface")]
fn set_sentinel_for_opt(ann: &Option<TypeAnn>) -> Option<&'static str> {
    match ann {
        Some(t) if is_set_surface_ty(t) => Some(set_sentinel_for(t)),
        _ => None,
    }
}

/// Lower a set literal `{ a, b, c }` onto the std.map runtime: `map_new()` then
/// a `map_insert(m, elem, 1)` chain (the value is the unit sentinel 1). Returns
/// the SSA id holding the final handle.
#[cfg(feature = "std-surface")]
fn lower_set_surface_lit(
    elements: &[ast::Node],
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> ValueId {
    let mut handle = ir.fresh();
    ir.instrs.push(Instr::Call {
        dst: handle,
        name: "map_new".to_string(),
        args: vec![],
    });
    for elem in elements {
        let k = lower_expr(elem, ir, env, struct_env, receiver_types);
        let one = ir.fresh();
        ir.instrs.push(Instr::ConstI64(one, 1));
        let next = ir.fresh();
        ir.instrs.push(Instr::Call {
            dst: next,
            name: "map_insert".to_string(),
            args: vec![handle, k, one],
        });
        handle = next;
    }
    handle
}

/// Collection sentinel for a `TypeAnn` (`array<T>`/`map<K,V>`/`set<T>`), or None.
#[cfg(feature = "std-surface")]
fn collection_sentinel_for_ty(ty: &TypeAnn) -> Option<&'static str> {
    if is_array_surface_ty(ty) {
        Some(ARRAY_VEC_SENTINEL)
    } else if is_map_surface_ty(ty) {
        Some(map_sentinel_for(ty))
    } else if is_set_surface_ty(ty) {
        Some(set_sentinel_for(ty))
    } else {
        None
    }
}

/// Resolve the collection sentinel of a method-call / index RECEIVER, covering
/// BOTH an Ident bound to a collection (via `struct_env`) AND a struct-FIELD
/// access whose declared field type is a collection
/// (`analyzed.determinism.contains_key(...)`). For the field case the base
/// struct type comes from the `receiver_types` side-table (the same source the
/// FieldAccess read path uses), then the field's declared type is looked up in
/// `struct_field_types`. None for a non-collection receiver (the caller then
/// falls through to the normal struct / UFCS path).
#[cfg(feature = "std-surface")]
fn receiver_collection_sentinel(
    receiver: &ast::Node,
    ir: &IRModule,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> Option<&'static str> {
    match receiver {
        ast::Node::Lit(Literal::Ident(v), _) => match struct_env.get(v).map(|s| s.as_str()) {
            Some(ARRAY_VEC_SENTINEL) => Some(ARRAY_VEC_SENTINEL),
            Some(MAP_SENTINEL) => Some(MAP_SENTINEL),
            Some(MAP_STR_SENTINEL) => Some(MAP_STR_SENTINEL),
            Some(SET_SENTINEL) => Some(SET_SENTINEL),
            Some(SET_STR_SENTINEL) => Some(SET_STR_SENTINEL),
            // Not a local/param — may be a module-level `const` of collection
            // type (`BACKEND_AVAILABILITY.get(p)` where `const … : map<…>`).
            _ => crate::ir::module_const_type(v).and_then(|t| collection_sentinel_for_ty(&t)),
        },
        ast::Node::FieldAccess {
            receiver: base,
            field,
            span,
        } => {
            // The base struct type: the `receiver_types` side-table when the
            // struct_resolver populated it, else struct_env recursion (covers a
            // method-receiver field access the resolver skips, and struct params).
            let sname = receiver_types
                .get(span)
                .cloned()
                .or_else(|| receiver_struct_type(base, ir, struct_env))?;
            let fields = ir.struct_defs.get(&sname)?;
            let idx = fields.iter().position(|f| f == field)?;
            let field_ty = ir.struct_field_types.get(&sname)?.get(idx)?;
            collection_sentinel_for_ty(field_ty)
        }
        // `coll[i]` — the sentinel of the ELEMENT type of `coll`. Resolves a
        // struct array FIELD indexed to a nested collection element
        // (`next.scopes[i]` where `scopes: array<map<K,V>>` → the map sentinel),
        // so a method on `coll[i]` desugars correctly.
        ast::Node::IndexAccess { receiver: base, .. } => {
            if let ast::Node::FieldAccess {
                receiver: obj,
                field,
                span,
            } = base.as_ref()
            {
                let sname = receiver_types
                    .get(span)
                    .cloned()
                    .or_else(|| receiver_struct_type(obj, ir, struct_env))?;
                let fields = ir.struct_defs.get(&sname)?;
                let idx = fields.iter().position(|f| f == field)?;
                let field_ty = ir.struct_field_types.get(&sname)?.get(idx)?;
                let elem = match field_ty {
                    TypeAnn::Generic { name, args } if name == "array" => args.first(),
                    _ => None,
                }?;
                collection_sentinel_for_ty(elem)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Resolve the STRUCT-TYPE NAME a receiver evaluates to: an Ident via
/// `struct_env` (struct-typed param/let), or a nested `s.field` whose declared
/// field type is itself a struct. None when not a known struct.
#[cfg(feature = "std-surface")]
fn receiver_struct_type(
    receiver: &ast::Node,
    ir: &IRModule,
    struct_env: &HashMap<String, String>,
) -> Option<String> {
    match receiver {
        ast::Node::Lit(Literal::Ident(v), _) => {
            let s = struct_env.get(v)?;
            if ir.struct_defs.contains_key(s) {
                Some(s.clone())
            } else {
                None
            }
        }
        ast::Node::FieldAccess {
            receiver: base,
            field,
            ..
        } => {
            let sname = receiver_struct_type(base, ir, struct_env)?;
            let fields = ir.struct_defs.get(&sname)?;
            let idx = fields.iter().position(|f| f == field)?;
            match ir.struct_field_types.get(&sname)?.get(idx)? {
                TypeAnn::Named(n) if ir.struct_defs.contains_key(n) => Some(n.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// True when a method-call receiver is the std `String` type — an Ident bound to
/// a string (`struct_env` sentinel `"String"`, set for string lets/params and
/// for-each elements over `array<string>`) OR a struct-FIELD whose declared type
/// is `string`. A string receiver's methods route to the `string_<method>` std
/// free functions (`.split`→string_split, `.trim`→string_trim, etc.).
#[cfg(feature = "std-surface")]
fn receiver_is_string(
    receiver: &ast::Node,
    ir: &IRModule,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> bool {
    match receiver {
        ast::Node::Lit(Literal::Ident(v), _) => struct_env
            .get(v)
            .map(|s| s == "String" || s == "string")
            .unwrap_or(false),
        ast::Node::FieldAccess {
            receiver: base,
            field,
            span,
        } => {
            let sname = receiver_types
                .get(span)
                .cloned()
                .or_else(|| receiver_struct_type(base, ir, struct_env));
            sname
                .and_then(|sn| {
                    let idx = ir.struct_defs.get(&sn)?.iter().position(|f| f == field)?;
                    ir.struct_field_types.get(&sn)?.get(idx).cloned()
                })
                .map(|ty| matches!(ty, TypeAnn::Named(n) if n == "string" || n == "String"))
                .unwrap_or(false)
        }
        _ => false,
    }
}

/// The struct_env value tracking an ELEMENT of declared type `ty`: a string
/// element → `"String"`; a struct element → its type name; a nested
/// collection element → its collection sentinel. None for a plain scalar.
#[cfg(feature = "std-surface")]
fn element_type_sentinel(ty: &TypeAnn) -> Option<String> {
    match ty {
        TypeAnn::Named(n) if n == "string" || n == "String" => Some("String".to_string()),
        _ if is_map_surface_ty(ty) => Some(map_sentinel_for(ty).to_string()),
        _ if is_set_surface_ty(ty) => Some(set_sentinel_for(ty).to_string()),
        _ if is_array_surface_ty(ty) => Some(ARRAY_VEC_SENTINEL.to_string()),
        TypeAnn::Named(n) => Some(n.clone()),
        _ => None,
    }
}

/// Element tracking value for an `array<T>` annotation (the `T` sentinel), or
/// None. Stored under the `__elem__<name>` struct_env key so a for-each over an
/// IDENT array recovers the element type the bare `"vec"` sentinel drops.
#[cfg(feature = "std-surface")]
fn array_element_track(ty: &TypeAnn) -> Option<String> {
    if let TypeAnn::Generic { name, args } = ty {
        if name == "array" {
            return element_type_sentinel(args.first()?);
        }
    }
    None
}

/// Resolve the struct_env tracking value for a for-each ELEMENT, from the
/// collection expression's element type: `coll.split(...)` → String elements; a
/// struct FIELD of type `array<T>` → element type T; an IDENT `array<T>` via its
/// recorded `__elem__` tracking. So `for d in decs` tracks `d` as `Decorator`,
/// `for d in flow.decorators` likewise, and `for part in s.split("+")` tracks
/// `part` as `String`, letting their methods resolve.
#[cfg(feature = "std-surface")]
fn foreach_element_sentinel(
    collection: &ast::Node,
    ir: &IRModule,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> Option<String> {
    if let ast::Node::MethodCall { method, .. } = collection {
        if method == "split" {
            return Some("String".to_string());
        }
    }
    if let ast::Node::Lit(Literal::Ident(v), _) = collection {
        if let Some(elem) = struct_env.get(&format!("__elem__{v}")) {
            return Some(elem.clone());
        }
    }
    if let ast::Node::FieldAccess {
        receiver: base,
        field,
        span,
    } = collection
    {
        let sname = receiver_types
            .get(span)
            .cloned()
            .or_else(|| receiver_struct_type(base, ir, struct_env))?;
        let idx = ir
            .struct_defs
            .get(&sname)?
            .iter()
            .position(|f| f == field)?;
        if let TypeAnn::Generic { name, args } = ir.struct_field_types.get(&sname)?.get(idx)? {
            if name == "array" {
                return element_type_sentinel(args.first()?);
            }
        }
    }
    None
}

/// Infer the struct_env tracking for a `let x = <rhs>` from the RHS TYPE, so a
/// method on `x` resolves: `let raw = decorator_arg_string(d)` (a string-returning
/// call) → `x` is `String`; `let s = analyzed.flags` (a `set<T>` field) → the set
/// sentinel; `let d = decorator_new(...)` (a struct-returning call) → the struct
/// name. Returns `(sentinel, optional __elem__ element)` for an `array<T>` RHS so
/// a later for-each over `x` recovers `T`. None for a scalar/unknown RHS.
#[cfg(feature = "std-surface")]
fn let_rhs_collection_track(
    value: &ast::Node,
    ir: &IRModule,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> Option<(String, Option<String>)> {
    // A string method whose result is itself a string (`text.slice(..)`,
    // `s.trim()`) or an `array<string>` (`s.split(..)`) — track the binding so a
    // chained method on the result (`text.slice(0, n).byte_at(i)` via a let)
    // resolves. Only fires when the receiver is statically known to be a string.
    if let ast::Node::MethodCall {
        receiver, method, ..
    } = value
    {
        if receiver_is_string(receiver, ir, struct_env, receiver_types) {
            return match method.as_str() {
                "slice" | "substring" | "trim" | "trim_start" | "trim_end" | "to_lowercase"
                | "to_uppercase" | "to_lower" | "to_upper" | "replace" | "concat" | "repeat" => {
                    Some(("String".to_string(), None))
                }
                "split" | "split_whitespace" | "lines" => {
                    Some(("vec".to_string(), Some("String".to_string())))
                }
                _ => None,
            };
        }
        // `m.get(k)` on a `map<K, V>` FIELD → the binding takes the VALUE type V,
        // so a method on it resolves (`let deps = n.edges.get(x); deps.push(..)`
        // where `edges: map<_, array<_>>` makes `deps` a vec). Resolves V from the
        // map field's declared type.
        if method == "get" {
            // Resolve the receiver map's declared `map<K, V>` type: a struct map
            // FIELD (`n.edges.get`) or a module-level const map
            // (`BACKEND_AVAILABILITY.get`).
            let map_ty: Option<TypeAnn> = match receiver.as_ref() {
                ast::Node::FieldAccess {
                    receiver: obj,
                    field,
                    span,
                } => receiver_types
                    .get(span)
                    .cloned()
                    .or_else(|| receiver_struct_type(obj, ir, struct_env))
                    .and_then(|sname| {
                        let idx = ir.struct_defs.get(&sname)?.iter().position(|f| f == field)?;
                        ir.struct_field_types.get(&sname)?.get(idx).cloned()
                    }),
                ast::Node::Lit(Literal::Ident(v), _) => crate::ir::module_const_type(v),
                _ => None,
            };
            if let Some(TypeAnn::Generic { name, args }) = map_ty {
                if name == "map" {
                    let v = args.get(1)?;
                    let sentinel = element_type_sentinel(v)?;
                    let elem = match v {
                        TypeAnn::Generic { name, args } if name == "array" => {
                            args.first().and_then(element_type_sentinel)
                        }
                        _ => None,
                    };
                    return Some((sentinel, elem));
                }
            }
        }
        return None;
    }
    let ty: TypeAnn = match value {
        ast::Node::Call { callee, .. } => {
            crate::ir::with_global_enums(|g| g.fn_returns.get(callee).cloned())?
        }
        ast::Node::FieldAccess {
            receiver: base,
            field,
            span,
        } => {
            let sname = receiver_types
                .get(span)
                .cloned()
                .or_else(|| receiver_struct_type(base, ir, struct_env))?;
            let idx = ir.struct_defs.get(&sname)?.iter().position(|f| f == field)?;
            ir.struct_field_types.get(&sname)?.get(idx)?.clone()
        }
        _ => return None,
    };
    let sentinel = element_type_sentinel(&ty)?;
    let elem = match &ty {
        TypeAnn::Generic { name, args } if name == "array" => {
            args.first().and_then(element_type_sentinel)
        }
        _ => None,
    };
    Some((sentinel, elem))
}

/// The collection mutating-method names whose std implementation returns a
/// FRESH handle on realloc (`vec_push`, `map_insert`, …). A bare-statement call
/// is rebound (see [`rewrite_collection_mutations`]); the same call in
/// expression position cannot rebind its realloc'd handle and is rejected.
/// Kept IN SYNC with the statement-rebind matcher's method list so a mutator is
/// rejected in expr position iff it would be rebound as a statement.
#[cfg(feature = "std-surface")]
const COLLECTION_MUTATORS: &[&str] = &["insert", "push", "set", "add"];

/// Does `receiver` name a tracked collection (a local/param of `array<T>` /
/// `map<K,V>` / `set<T>`, a struct collection FIELD, or a struct
/// `array<collection>` ELEMENT)? Mirrors the rebind matcher in
/// [`rewrite_collection_mutations`] so the expr-position reject below fires on
/// exactly the same receiver shapes that statement-position rebinding handles.
#[cfg(feature = "std-surface")]
fn receiver_is_tracked_collection(
    receiver: &ast::Node,
    scope: &std::collections::HashSet<String>,
    struct_collection_fields: &std::collections::HashSet<(String, String)>,
    struct_collection_element_fields: &std::collections::HashSet<(String, String)>,
    vtypes: &std::collections::HashMap<String, String>,
) -> bool {
    match receiver {
        ast::Node::Lit(Literal::Ident(v), _) => scope.contains(v),
        ast::Node::FieldAccess {
            receiver: base,
            field,
            ..
        } => matches!(base.as_ref(), ast::Node::Lit(Literal::Ident(obj), _)
            if vtypes.get(obj).is_some_and(|s| {
                struct_collection_fields.contains(&(s.clone(), field.clone()))
            })),
        ast::Node::IndexAccess { receiver: base, .. } => matches!(
            base.as_ref(),
            ast::Node::FieldAccess { receiver: obj, field, .. }
                if matches!(obj.as_ref(), ast::Node::Lit(Literal::Ident(ov), _)
                    if vtypes.get(ov).is_some_and(|s| {
                        struct_collection_element_fields.contains(&(s.clone(), field.clone()))
                    }))
        ),
        _ => false,
    }
}

/// True when `node` (an expression or statement subtree) READS any identifier
/// in `targets` — i.e. references it as a *value*, not as an assignment target
/// or a fresh binding name.
///
/// Used by the `While` arm to decide whether a loop-carried variable's alias
/// SOURCE (`let mut j = start` makes `j` and `start` share one ValueId) is
/// actually referenced inside the loop. That is the exact condition under which
/// the MLIR While emitter's purely-numeric `substitute_ids` rewrite (first-match
/// on the shared `%init_id`) would clobber the source into the loop counter — a
/// silent miscompile. When no source name is read the alias is harmless and the
/// binding is left untouched, preserving byte-identity for correct programs.
#[cfg(feature = "std-surface")]
fn ast_reads_ident(node: &ast::Node, targets: &std::collections::HashSet<String>) -> bool {
    use ast::Node as N;
    match node {
        N::Lit(Literal::Ident(name), _) => targets.contains(name),
        N::Lit(..) => false,
        N::Binary { left, right, .. }
        | N::Logical { left, right, .. }
        | N::Bitwise { left, right, .. } => {
            ast_reads_ident(left, targets) || ast_reads_ident(right, targets)
        }
        N::Paren(inner, _)
        | N::Neg { operand: inner, .. }
        | N::Not { operand: inner, .. }
        | N::Ref { inner, .. }
        | N::As { expr: inner, .. } => ast_reads_ident(inner, targets),
        N::Tuple { elements, .. }
        | N::ArrayLit { elements, .. }
        | N::SetLit { elements, .. }
        | N::Print { args: elements, .. } => {
            elements.iter().any(|e| ast_reads_ident(e, targets))
        }
        N::Call { args, .. } => args.iter().any(|a| ast_reads_ident(a, targets)),
        N::CallGrad { loss, .. } => ast_reads_ident(loss, targets),
        N::CallTensorSum { x, .. }
        | N::CallTensorMean { x, .. }
        | N::CallReshape { x, .. }
        | N::CallExpandDims { x, .. }
        | N::CallSqueeze { x, .. }
        | N::CallTranspose { x, .. }
        | N::CallIndex { x, .. }
        | N::CallSlice { x, .. }
        | N::CallSliceStride { x, .. }
        | N::CallTensorRelu { x, .. } => ast_reads_ident(x, targets),
        N::CallGather { x, idx, .. } => {
            ast_reads_ident(x, targets) || ast_reads_ident(idx, targets)
        }
        N::CallDot { a, b, .. } | N::CallMatMul { a, b, .. } => {
            ast_reads_ident(a, targets) || ast_reads_ident(b, targets)
        }
        N::TensorMatmul { lhs, rhs, .. } | N::TensorElemwise { lhs, rhs, .. } => {
            ast_reads_ident(lhs, targets) || ast_reads_ident(rhs, targets)
        }
        N::CallTensorConv2d { x, w, .. } => {
            ast_reads_ident(x, targets) || ast_reads_ident(w, targets)
        }
        // Binding introductions / assignment targets: the LHS `name` is NOT a
        // read (counting it would over-fire and drift correct programs); only
        // the RHS value is a read.
        N::Let { value, .. }
        | N::LetTuple { value, .. }
        | N::Assign { value, .. }
        | N::Const { value, .. } => ast_reads_ident(value, targets),
        N::Return { value, .. } => {
            value.as_ref().is_some_and(|v| ast_reads_ident(v, targets))
        }
        N::Block { stmts, .. } => stmts.iter().any(|s| ast_reads_ident(s, targets)),
        N::Region { body, .. } => body.iter().any(|s| ast_reads_ident(s, targets)),
        N::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            ast_reads_ident(cond, targets)
                || then_branch.iter().any(|s| ast_reads_ident(s, targets))
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| eb.iter().any(|s| ast_reads_ident(s, targets)))
        }
        N::For {
            start, end, body, ..
        } => {
            ast_reads_ident(start, targets)
                || ast_reads_ident(end, targets)
                || body.iter().any(|s| ast_reads_ident(s, targets))
        }
        N::ForEach {
            collection, body, ..
        } => {
            ast_reads_ident(collection, targets)
                || body.iter().any(|s| ast_reads_ident(s, targets))
        }
        N::While { cond, body, .. } => {
            ast_reads_ident(cond, targets) || body.iter().any(|s| ast_reads_ident(s, targets))
        }
        N::MethodCall { receiver, args, .. } => {
            ast_reads_ident(receiver, targets)
                || args.iter().any(|a| ast_reads_ident(a, targets))
        }
        N::FieldAccess { receiver, .. } => ast_reads_ident(receiver, targets),
        N::IndexAccess { receiver, index, .. } => {
            ast_reads_ident(receiver, targets) || ast_reads_ident(index, targets)
        }
        N::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            ast_reads_ident(receiver, targets)
                || ast_reads_ident(index, targets)
                || ast_reads_ident(value, targets)
        }
        N::FieldAssign {
            receiver, value, ..
        } => ast_reads_ident(receiver, targets) || ast_reads_ident(value, targets),
        N::Match {
            scrutinee, arms, ..
        } => {
            ast_reads_ident(scrutinee, targets)
                || arms.iter().any(|a| ast_reads_ident(&a.body, targets))
        }
        N::Assert { cond, .. } => ast_reads_ident(cond, targets),
        N::StructLit { fields, .. } => {
            fields.iter().any(|f| ast_reads_ident(&f.value, targets))
        }
        N::MapLit { entries, .. } => entries
            .iter()
            .any(|(k, v)| ast_reads_ident(k, targets) || ast_reads_ident(v, targets)),
        // Items / declarations / non-read leaves (imports, fn/struct/enum defs,
        // tensor.rand, etc.) contain no loop-body value reads relevant here.
        _ => false,
    }
}

/// Collect the names of variables ASSIGNED (`x = ...`) anywhere in `stmts`
/// that are visible in the outer loop iteration — descending through the
/// nested control flow that executes within a single iteration (`if`/`block`/
/// `match`/`region`) but NOT into nested `while`/`for`/`for-each` loops, which
/// manage their own loop-carried scope (and get their own alias-break when
/// lowered). Used by the `While` arm to find loop-carried candidates whose
/// alias source may be clobbered — including assignments buried in a branch
/// (e.g. `cur = new_tbl` inside `if b == 91 { .. }`, std/toml.mind:1185).
#[cfg(feature = "std-surface")]
fn collect_assign_targets(stmts: &[ast::Node], out: &mut Vec<String>) {
    use ast::Node as N;
    for stmt in stmts {
        match stmt {
            N::Assign { name, .. } => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            N::If {
                then_branch,
                else_branch,
                ..
            } => {
                collect_assign_targets(then_branch, out);
                if let Some(eb) = else_branch {
                    collect_assign_targets(eb, out);
                }
            }
            N::Block { stmts, .. } | N::Region { body: stmts, .. } => {
                collect_assign_targets(stmts, out);
            }
            N::Match { arms, .. } => {
                for arm in arms {
                    collect_assign_targets(std::slice::from_ref(&arm.body), out);
                }
            }
            // Do NOT descend into nested loops — separate loop-carried scope.
            _ => {}
        }
    }
}

/// Walk an EXPRESSION (non-statement) sub-tree and FAIL LOUD (#306) on any
/// collection mutating-method call on a tracked collection receiver. Such a call
/// (`w.push(v.push(5))`, `f(a.push(x))`, `let n = a.push(x)`) cannot rebind its
/// realloc'd handle in expression position — only the bare-statement form is
/// rebound — so it would silently lose the mutation. Refuse it at compile time
/// rather than emit a silent miscompile. A normal value-returning method
/// (`a.length`, `a.get(i)`, a non-collection `.add`) is NOT a tracked-collection
/// mutator, so it passes through untouched. Does NOT descend into nested
/// statement bodies (`If`/`Block`/closures) — the main loop recurses into those
/// with the correct lexical scope.
#[cfg(feature = "std-surface")]
fn reject_collection_mutation_in_expr(
    expr: &ast::Node,
    scope: &std::collections::HashSet<String>,
    struct_collection_fields: &std::collections::HashSet<(String, String)>,
    struct_collection_element_fields: &std::collections::HashSet<(String, String)>,
    vtypes: &std::collections::HashMap<String, String>,
) {
    use ast::Node as N;
    if let N::MethodCall {
        receiver, method, ..
    } = expr
    {
        if COLLECTION_MUTATORS.contains(&method.as_str())
            && receiver_is_tracked_collection(
                receiver,
                scope,
                struct_collection_fields,
                struct_collection_element_fields,
                vtypes,
            )
        {
            panic!(
                "collection mutation `{}.{}(...)` in expression position is not \
                 supported: the std `{}` returns a fresh handle on realloc that \
                 cannot be rebound here, so the mutation would be silently lost. \
                 Use it as its own statement (`{}.{}(...)`) instead. (#306: \
                 refusing a known silent miscompile.)",
                describe_receiver(receiver),
                method,
                method,
                describe_receiver(receiver),
                method,
            );
        }
    }
    // Recurse into expression children only. The walk mirrors `lower_expr`'s
    // value-position descent; it deliberately stops at nodes that open a new
    // statement scope (`If`/`Block`/`Match`/`FnDef`), which the caller's main
    // loop handles with the correct collection scope.
    let recur = |e: &ast::Node| {
        reject_collection_mutation_in_expr(
            e,
            scope,
            struct_collection_fields,
            struct_collection_element_fields,
            vtypes,
        )
    };
    match expr {
        N::Binary { left, right, .. } => {
            recur(left);
            recur(right);
        }
        N::Paren(inner, _) => recur(inner),
        N::Tuple { elements, .. } | N::ArrayLit { elements, .. } => {
            elements.iter().for_each(recur)
        }
        N::Call { args, .. } => args.iter().for_each(recur),
        N::MethodCall {
            receiver, args, ..
        } => {
            recur(receiver);
            args.iter().for_each(recur);
        }
        N::FieldAccess { receiver, .. } => recur(receiver),
        N::IndexAccess { receiver, index, .. } => {
            recur(receiver);
            recur(index);
        }
        N::MapLit { entries, .. } => {
            for (k, v) in entries {
                recur(k);
                recur(v);
            }
        }
        _ => {}
    }
}

/// Rewrite STATEMENT-position collection mutations into assignments so the
/// non-mutating std handle is rebound. `m.insert(k,v)` / `v.push(x)` /
/// `v.set(i,x)` as a bare statement (result discarded) → `m = m.insert(k,v)`.
/// std.map's `map_insert` and (on realloc) std.vec's `vec_push` return a FRESH
/// handle, so without the rebind the change is silently lost. Emitting a real
/// `Node::Assign` — rather than rebinding the SSA env after lowering — keeps the
/// existing loop-carried-SSA detection intact (the `while`/`for` lowering scans
/// the body for `Node::Assign` to find loop-carried vars; an env-only rebind
/// would be invisible to it). Only applied when the receiver is a local KNOWN to
/// be a collection (`array<T>` / `map<K,V>` from its let/param annotation), so a
/// non-collection `.insert`/`.push`/`.set` is untouched and the keystone (no
/// collections) is byte-identical.
#[cfg(feature = "std-surface")]
fn rewrite_collection_mutations(
    stmts: &mut [ast::Node],
    collections: &std::collections::HashSet<String>,
    struct_collection_fields: &std::collections::HashSet<(String, String)>,
    struct_collection_element_fields: &std::collections::HashSet<(String, String)>,
    var_types: &std::collections::HashMap<String, String>,
) {
    let mut scope = collections.clone();
    let mut vtypes = var_types.clone();
    for stmt in stmts.iter_mut() {
        // A `let x: array<T> | map<K,V>` introduces a new collection local; a
        // `let x: StructName` (or `let x = y` aliasing a tracked struct) introduces
        // a struct-typed local (so `x.field.push` / `x.field[i].insert` resolve).
        if let ast::Node::Let {
            name, ann, value, ..
        } = stmt
        {
            let growable = is_growable_bytes_init(ann, value);
            if is_array_surface_type(ann)
                || map_sentinel_for_opt(ann).is_some()
                || set_sentinel_for_opt(ann).is_some()
                || growable
            {
                scope.insert(name.clone());
            }
            // A `bytes`-typed name is a struct alias only when NOT a growable
            // `[..]` buffer (which is a vec, tracked above).
            if !growable {
                if let Some(ast::TypeAnn::Named(sname)) = ann {
                    vtypes.insert(name.clone(), sname.clone());
                }
            }
            if let ast::Node::Lit(Literal::Ident(src), _) = value.as_ref() {
                if let Some(t) = vtypes.get(src).cloned() {
                    vtypes.entry(name.clone()).or_insert(t);
                }
            }
        }
        // FAIL LOUD (#306): a collection mutating call used in EXPRESSION
        // position (`let n = a.push(x)`, `f(a.push(x))`, `w.push(v.push(5))`)
        // cannot rebind its realloc'd handle and would silently lose the
        // mutation. Scan every expression-position child of this statement and
        // refuse it. The statement-level mutating call itself is rebound below
        // (allowed) — so for a bare `MethodCall` statement only its receiver's
        // sub-expressions and ARGUMENTS are scanned, never the top call.
        {
            let reject = |e: &ast::Node| {
                reject_collection_mutation_in_expr(
                    e,
                    &scope,
                    struct_collection_fields,
                    struct_collection_element_fields,
                    &vtypes,
                );
            };
            match stmt {
                // A SAME-NAME functional-update rebind `let m = m.insert(..)`
                // SHADOWS the receiver `m` with the realloc'd fresh handle: the
                // old handle becomes unreachable, so nothing dangles and the
                // mutation is NOT lost. This is the safe, idiomatic map<K,V> /
                // vec update form (the new `let` binds the fresh handle), so it
                // must be allowed — unlike a DIFFERENT-name `let n = m.insert(..)`
                // which leaves `m` pointing at the freed handle and IS rejected.
                ast::Node::Let { name, value, .. } => {
                    let same_name_rebind = matches!(
                        value.as_ref(),
                        ast::Node::MethodCall { receiver, method, .. }
                            if COLLECTION_MUTATORS.contains(&method.as_str())
                                && matches!(
                                    receiver.as_ref(),
                                    ast::Node::Lit(Literal::Ident(r), _) if r == name
                                )
                    );
                    if !same_name_rebind {
                        reject(value);
                    }
                }
                ast::Node::Assign { value, .. } | ast::Node::LetTuple { value, .. } => {
                    reject(value)
                }
                ast::Node::Return { value: Some(v), .. } => reject(v),
                ast::Node::FieldAssign { receiver, value, .. } => {
                    reject(receiver);
                    reject(value);
                }
                ast::Node::IndexAssign {
                    receiver,
                    index,
                    value,
                    ..
                } => {
                    reject(receiver);
                    reject(index);
                    reject(value);
                }
                ast::Node::Call { args, .. } => args.iter().for_each(&reject),
                // A bare `recv.method(args)` statement: the top call is either a
                // rebound mutation (allowed) or a value-returning method (no
                // mutation). Either way scan its receiver's sub-expressions and
                // its arguments for nested expr-position mutations.
                ast::Node::MethodCall {
                    receiver, args, ..
                } => {
                    match receiver.as_ref() {
                        ast::Node::FieldAccess { receiver: b, .. } => reject(b),
                        ast::Node::IndexAccess { receiver: b, index, .. } => {
                            reject(b);
                            reject(index);
                        }
                        other => reject(other),
                    }
                    args.iter().for_each(&reject);
                }
                ast::Node::If { cond, .. } | ast::Node::While { cond, .. } => reject(cond),
                _ => {}
            }
        }
        // A bare collection-mutation statement whose fresh-on-realloc std handle
        // would otherwise be discarded is rebound to write the handle back:
        //   `x.push(a)`              → `x = x.push(a)`                 (Var)
        //   `obj.field.push(a)`      → `obj.field = obj.field.push(a)` (Field)
        //   `obj.field[i].insert(a)` → `obj.field[i] = obj.field[i].insert(a)` (Index)
        enum Rebind {
            Var(String),
            Field,
            Index,
        }
        let rebind: Option<Rebind> = if let ast::Node::MethodCall {
            receiver, method, ..
        } = stmt
        {
            if matches!(method.as_str(), "insert" | "push" | "set" | "add") {
                match receiver.as_ref() {
                    ast::Node::Lit(Literal::Ident(v), _) if scope.contains(v) => {
                        Some(Rebind::Var(v.clone()))
                    }
                    ast::Node::FieldAccess {
                        receiver: base,
                        field,
                        ..
                    } => match base.as_ref() {
                        ast::Node::Lit(Literal::Ident(obj), _)
                            if vtypes
                                .get(obj)
                                .map(|s| {
                                    struct_collection_fields.contains(&(s.clone(), field.clone()))
                                })
                                .unwrap_or(false) =>
                        {
                            Some(Rebind::Field)
                        }
                        _ => None,
                    },
                    // `obj.field[i].method(..)` — a collection ELEMENT of a struct
                    // `array<collection>` field.
                    ast::Node::IndexAccess { receiver: base, .. } => match base.as_ref() {
                        ast::Node::FieldAccess {
                            receiver: obj,
                            field,
                            ..
                        } => match obj.as_ref() {
                            ast::Node::Lit(Literal::Ident(ov), _)
                                if vtypes
                                    .get(ov)
                                    .map(|s| {
                                        struct_collection_element_fields
                                            .contains(&(s.clone(), field.clone()))
                                    })
                                    .unwrap_or(false) =>
                            {
                                Some(Rebind::Index)
                            }
                            _ => None,
                        },
                        _ => None,
                    },
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        };
        match rebind {
            Some(Rebind::Var(var)) => {
                let sp = stmt.span();
                let mc = std::mem::replace(stmt, ast::Node::Lit(Literal::Int(0), sp));
                *stmt = ast::Node::Assign {
                    name: var,
                    value: Box::new(mc),
                    span: sp,
                };
                continue; // an Assign has no nested statement lists to recurse into
            }
            Some(Rebind::Field) => {
                let sp = stmt.span();
                let mc = std::mem::replace(stmt, ast::Node::Lit(Literal::Int(0), sp));
                // Reconstruct the FieldAssign LHS from the method receiver
                // (`obj.field`), reusing the same receiver/field nodes.
                if let ast::Node::MethodCall { receiver, .. } = &mc {
                    if let ast::Node::FieldAccess {
                        receiver: base,
                        field,
                        ..
                    } = receiver.as_ref()
                    {
                        *stmt = ast::Node::FieldAssign {
                            receiver: base.clone(),
                            field: field.clone(),
                            value: Box::new(mc),
                            span: sp,
                        };
                        continue;
                    }
                }
                // Shouldn't happen (we matched the shape above); restore.
                *stmt = mc;
            }
            Some(Rebind::Index) => {
                let sp = stmt.span();
                let mc = std::mem::replace(stmt, ast::Node::Lit(Literal::Int(0), sp));
                // Reconstruct the IndexAssign LHS from the method receiver
                // (`obj.field[i]`), reusing the same receiver/index nodes.
                if let ast::Node::MethodCall { receiver, .. } = &mc {
                    if let ast::Node::IndexAccess {
                        receiver: base,
                        index,
                        ..
                    } = receiver.as_ref()
                    {
                        *stmt = ast::Node::IndexAssign {
                            receiver: base.clone(),
                            index: index.clone(),
                            value: Box::new(mc),
                            span: sp,
                        };
                        continue;
                    }
                }
                *stmt = mc;
            }
            None => {}
        }
        // Recurse into nested statement bodies, threading the collection scope
        // and struct-type env.
        match stmt {
            ast::Node::For { body, .. }
            | ast::Node::ForEach { body, .. }
            | ast::Node::While { body, .. }
            | ast::Node::Block { stmts: body, .. } => {
                rewrite_collection_mutations(body, &scope, struct_collection_fields, struct_collection_element_fields, &vtypes);
            }
            ast::Node::If {
                then_branch,
                else_branch,
                ..
            } => {
                rewrite_collection_mutations(
                    then_branch,
                    &scope,
                    struct_collection_fields,
                    struct_collection_element_fields,
                    &vtypes,
                );
                if let Some(eb) = else_branch {
                    rewrite_collection_mutations(eb, &scope, struct_collection_fields, struct_collection_element_fields, &vtypes);
                }
            }
            ast::Node::Match { arms, .. } => {
                for arm in arms.iter_mut() {
                    if let ast::Node::Block { stmts: body, .. } = &mut arm.body {
                        rewrite_collection_mutations(
                            body,
                            &scope,
                            struct_collection_fields,
                            struct_collection_element_fields,
                            &vtypes,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

/// Cheap bool pre-scan: does the module declare ANY `array<T>` / `map<K,V>`
/// binding or parameter? If not, there is nothing to rewrite and the expensive
/// module clone in [`preprocess_collection_mutations`] is skipped entirely. An
/// allocation-free, early-exit walk — keeps the collection-free hot path (the
/// keystone, the compile_small bench) at zero added cost.
#[cfg(feature = "std-surface")]
fn module_declares_collection(module: &ast::Module) -> bool {
    fn ann_is_collection(ann: &Option<TypeAnn>) -> bool {
        matches!(ann, Some(t) if is_array_surface_ty(t) || is_map_surface_ty(t) || is_set_surface_ty(t))
    }
    fn walk(stmts: &[ast::Node]) -> bool {
        for s in stmts {
            match s {
                ast::Node::Let { ann, .. } if ann_is_collection(ann) => return true,
                ast::Node::FnDef { params, body, .. } => {
                    if params.iter().any(|p| {
                        is_array_surface_ty(&p.ty)
                            || is_map_surface_ty(&p.ty)
                            || is_set_surface_ty(&p.ty)
                    }) {
                        return true;
                    }
                    if walk(body) {
                        return true;
                    }
                }
                ast::Node::For { body, .. }
                | ast::Node::ForEach { body, .. }
                | ast::Node::While { body, .. }
                | ast::Node::Block { stmts: body, .. } => {
                    if walk(body) {
                        return true;
                    }
                }
                ast::Node::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    if walk(then_branch) || else_branch.as_deref().is_some_and(walk) {
                        return true;
                    }
                }
                ast::Node::Match { arms, .. } => {
                    for arm in arms {
                        if let ast::Node::Block { stmts: body, .. } = &arm.body {
                            if walk(body) {
                                return true;
                            }
                        }
                    }
                }
                // A struct with a collection FIELD (`struct Bag { items: array<T> }`)
                // also needs the rewrite pass — `obj.field.push(x)` is rebound to
                // `obj.field = obj.field.push(x)` so the fresh-on-realloc handle
                // persists.
                ast::Node::StructDef { fields, .. }
                    if fields.iter().any(|f| {
                        is_array_surface_ty(&f.ty)
                            || is_map_surface_ty(&f.ty)
                            || is_set_surface_ty(&f.ty)
                    }) =>
                {
                    return true;
                }
                _ => {}
            }
        }
        false
    }
    walk(&module.items)
}

/// Apply [`rewrite_collection_mutations`] to every top-level function body,
/// seeding each with its `array<T>` / `map<K,V>` parameters. Returns `Some` of a
/// rewritten clone only when the module declares a collection; `None` (no clone)
/// otherwise — so a collection-free module (e.g. the keystone main.mind, the
/// compile_small bench fixture) pays only the cheap pre-scan, not a full clone.
#[cfg(feature = "std-surface")]
fn preprocess_collection_mutations(module: &ast::Module) -> Option<ast::Module> {
    if !module_declares_collection(module) {
        return None;
    }
    let mut m = module.clone();
    // (struct, field) pairs whose field is itself a collection — so a
    // `obj.field.push(x)` statement (a FieldAccess receiver) can be rebound to
    // `obj.field = obj.field.push(x)` (the std handle is fresh on realloc).
    let mut struct_collection_fields: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    // (struct, field) pairs whose field is `array<collection>` — so an indexed
    // element mutation `obj.field[i].insert(x)` can be rebound to
    // `obj.field[i] = obj.field[i].insert(x)`.
    let mut struct_collection_element_fields: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    for item in &m.items {
        if let ast::Node::StructDef { name, fields, .. } = item {
            for f in fields {
                if is_array_surface_ty(&f.ty)
                    || is_map_surface_ty(&f.ty)
                    || is_set_surface_ty(&f.ty)
                {
                    struct_collection_fields.insert((name.clone(), f.name.clone()));
                }
                // `array<COLLECTION>` — the element is itself a collection.
                if let TypeAnn::Generic { name: g, args } = &f.ty {
                    if g == "array" {
                        if let Some(elem) = args.first() {
                            if is_array_surface_ty(elem)
                                || is_map_surface_ty(elem)
                                || is_set_surface_ty(elem)
                            {
                                struct_collection_element_fields
                                    .insert((name.clone(), f.name.clone()));
                            }
                        }
                    }
                }
            }
        }
    }
    for item in m.items.iter_mut() {
        if let ast::Node::FnDef { params, body, .. } = item {
            let mut collections = std::collections::HashSet::new();
            // var -> struct-type name, so `obj.field` can resolve `obj`'s struct.
            let mut var_types: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();
            for p in params.iter() {
                if is_array_surface_ty(&p.ty)
                    || is_map_surface_ty(&p.ty)
                    || is_set_surface_ty(&p.ty)
                {
                    collections.insert(p.name.clone());
                }
                if let ast::TypeAnn::Named(sname) = &p.ty {
                    var_types.insert(p.name.clone(), sname.clone());
                }
            }
            rewrite_collection_mutations(
                body,
                &collections,
                &struct_collection_fields,
                &struct_collection_element_fields,
                &var_types,
            );
        }
    }
    Some(m)
}

fn lower_expr(
    node: &ast::Node,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    // RFC 0005 P0f Step 1 — per-fn binding from variable name to its
    // struct-type name. Populated at Let sites whose RHS is a
    // `StructLit`; consumed by the FieldAccess read-path arm below
    // to look up the canonical field-name list from `ir.struct_defs`
    // and emit `__mind_load_i64` at the correct 8-byte offset.
    struct_env: &HashMap<String, String>,
    // RFC 0005 P0f Step 2 — module-wide side-table keyed on each
    // FieldAccess span, mapping to the receiver's struct-type name.
    // Built once per `lower_to_ir` call by `struct_resolver`. Lets
    // the FieldAccess arm resolve chained access (`a.b.c`), fn
    // returns (`foo().x`), and struct-typed parameters that Step 1
    // can't see via a direct `Ident` lookup.
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> ValueId {
    match node {
        ast::Node::Lit(Literal::Int(n), _) => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, *n));
            id
        }
        ast::Node::Lit(Literal::Float(f), _) => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstF64(id, *f));
            id
        }
        // Unary negation `-expr`. Without this arm a bare negative literal
        // (`-65536`) — or any unary minus — fell through to the catch-all
        // `_ =>` and was silently lowered to `const.i64 0`. `-N` must be
        // identical to `(0 - N)` for every i64 N. Literal operands fold to
        // a single negated constant; runtime operands lower as `0 - operand`
        // so the type-driven IR→MLIR path picks `arith.subi`/`arith.subf`
        // exactly as the binary-subtraction source form already does.
        ast::Node::Neg { operand, .. } => match operand.as_ref() {
            ast::Node::Lit(Literal::Int(n), _) => {
                let id = ir.fresh();
                // `wrapping_neg` keeps INT64_MIN well-defined: `-INT64_MIN`
                // wraps back to INT64_MIN, matching two's-complement
                // `0 - INT64_MIN` via `arith.subi`.
                ir.instrs.push(Instr::ConstI64(id, n.wrapping_neg()));
                id
            }
            ast::Node::Lit(Literal::Float(f), _) => {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstF64(id, -*f));
                id
            }
            _ => {
                let zero = ir.fresh();
                ir.instrs.push(Instr::ConstI64(zero, 0));
                let rhs = lower_expr(operand, ir, env, struct_env, receiver_types);
                let dst = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst,
                    op: BinOp::Sub,
                    lhs: zero,
                    rhs,
                });
                dst
            }
        },
        // Unary logical NOT `!expr`. Desugars to `operand == 0` so it produces
        // the exact same IR as the binary `==` source form already does — 1 when
        // the operand is falsy (0), else 0 — reusing the keystone-stable
        // comparison lowering and its i1→bool widening verbatim (enum_match #9).
        ast::Node::Not { operand, .. } => match operand.as_ref() {
            ast::Node::Lit(Literal::Int(n), _) => {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, (*n == 0) as i64));
                id
            }
            _ => {
                let lhs = lower_expr(operand, ir, env, struct_env, receiver_types);
                let zero = ir.fresh();
                ir.instrs.push(Instr::ConstI64(zero, 0));
                let dst = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst,
                    op: BinOp::Eq,
                    lhs,
                    rhs: zero,
                });
                dst
            }
        },
        #[cfg(feature = "std-surface")]
        ast::Node::Lit(Literal::Str(s), _) => {
            // Materialize the literal into a real `String { addr, len, cap }`
            // value, reusing the StructLit machinery verbatim:
            //
            //   addr = __mind_alloc(n)               // n = UTF-8 byte length
            //   __mind_store_i8(addr + i, byte_i)    // for each UTF-8 byte
            //   rec  = __mind_alloc(24)              // 3-field i64 String record
            //   __mind_store_i64(rec + 0,  addr)     // field 0: addr
            //   __mind_store_i64(rec + 8,  n)        // field 1: len
            //   __mind_store_i64(rec + 16, n)        // field 2: cap
            //   rec                                  // ← the String value
            //
            // The literal then IS a normal String value that string_len /
            // string_slice_from / string_eq operate on with zero further
            // change. No new Instr / MLIR codegen / ABI surface — every
            // intrinsic here is already used by std (the same __mind_store_i8
            // that string_push_byte calls at std/string.mind:99).
            let bytes = s.as_bytes();
            let n = bytes.len() as i64;

            // n_const = number of UTF-8 bytes
            let n_const = ir.fresh();
            ir.instrs.push(Instr::ConstI64(n_const, n));

            // addr = __mind_alloc(n)  — backing buffer for the bytes
            let addr = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: addr,
                name: "__mind_alloc".to_string(),
                args: vec![n_const],
            });

            // __mind_store_i8(addr + i, byte_i) for each UTF-8 byte.
            for (i, &b) in bytes.iter().enumerate() {
                let byte_addr = if i == 0 {
                    addr
                } else {
                    let offset = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(offset, i as i64));
                    let sum = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: sum,
                        op: BinOp::Add,
                        lhs: addr,
                        rhs: offset,
                    });
                    sum
                };
                let byte_val = ir.fresh();
                ir.instrs.push(Instr::ConstI64(byte_val, b as i64));
                let store_ret = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst: store_ret,
                    name: "__mind_store_i8".to_string(),
                    args: vec![byte_addr, byte_val],
                });
            }

            // Build the 3-field i64 String record (24 bytes), exactly as the
            // StructLit arm does: rec = __mind_alloc(24); store addr/len/cap.
            let rec_bytes = ir.fresh();
            ir.instrs.push(Instr::ConstI64(rec_bytes, 24));
            let rec = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: rec,
                name: "__mind_alloc".to_string(),
                args: vec![rec_bytes],
            });

            // field 0 (addr) at offset 0
            let store0 = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: store0,
                name: "__mind_store_i64".to_string(),
                args: vec![rec, addr],
            });

            // field 1 (len) at offset 8
            let off8 = ir.fresh();
            ir.instrs.push(Instr::ConstI64(off8, 8));
            let rec8 = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: rec8,
                op: BinOp::Add,
                lhs: rec,
                rhs: off8,
            });
            let len_val = ir.fresh();
            ir.instrs.push(Instr::ConstI64(len_val, n));
            let store1 = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: store1,
                name: "__mind_store_i64".to_string(),
                args: vec![rec8, len_val],
            });

            // field 2 (cap) at offset 16
            let off16 = ir.fresh();
            ir.instrs.push(Instr::ConstI64(off16, 16));
            let rec16 = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: rec16,
                op: BinOp::Add,
                lhs: rec,
                rhs: off16,
            });
            let cap_val = ir.fresh();
            ir.instrs.push(Instr::ConstI64(cap_val, n));
            let store2 = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: store2,
                name: "__mind_store_i64".to_string(),
                args: vec![rec16, cap_val],
            });

            rec
        }
        #[cfg(not(feature = "std-surface"))]
        ast::Node::Lit(Literal::Str(_), _) => {
            // Strings don't have IR representation yet; emit placeholder
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Lit(Literal::Ident(name), _) => {
            // Fast path: SSA binding from env (params, let-bindings).
            if let Some(id) = env.get(name).copied() {
                return id;
            }
            // Phase 6.2b Gap 2: const-array identifier — re-emit the
            // ConstArray blob into the current IR (fn body or module level)
            // so the ArrayLoad that follows has a valid base in this
            // IR's SSA namespace.
            #[cfg(feature = "std-surface")]
            if let Some(values) = ir.const_array_defs.get(name).cloned() {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstArray {
                    dst: id,
                    name: Some(name.clone()),
                    values,
                });
                return id;
            }
            // Module-level `const NAME = value` (scalar / string / collection):
            // inline the value expression in the current SSA namespace. `env` is
            // checked first above, so a local of the same name shadows the const.
            // `const A = B` (const-references-const) resolves by re-entering this
            // arm for `B`; a self-cycle is caught by the resolving guard and
            // fails loud rather than recursing forever.
            #[cfg(feature = "std-surface")]
            if let Some(cval) = crate::ir::module_const_value(name) {
                if !crate::ir::begin_resolving_const(name) {
                    panic!(
                        "const `{name}` is defined in terms of itself (cyclic \
                         const reference) — refusing to inline (that would loop)."
                    );
                }
                let id = lower_expr(&cval, ir, env, struct_env, receiver_types);
                crate::ir::end_resolving_const(name);
                return id;
            }
            // "finish MIND" Step 2 (Defect B fix): a fully-qualified enum
            // variant used as a VALUE (e.g. `Mode::On`) lowers to its ordinal
            // i64 discriminant tag instead of falling through to the
            // placeholder `const.i64 0`. The path string (`"Mode::On"`) is the
            // exact key the parser accumulates for a `Type::Variant`
            // identifier, matching the `enum_variant_tags` registry built when
            // the `EnumDef` was lowered.
            #[cfg(feature = "std-surface")]
            {
                // Resolve a fieldless variant value used bare (`None`, `Nothing`)
                // OR qualified (`Mode::On`): a bare name with no `::` is matched
                // against any `Enum::V` in the registry, mirroring the bare
                // payload-ctor resolution in the `Node::Call` arm.
                let vkey: Option<String> = if ir.enum_variant_tags.contains_key(name) {
                    Some(name.clone())
                } else if !name.contains("::") {
                    ir.enum_variant_tags
                        .keys()
                        .find(|k| k.rsplit_once("::").map(|(_, v)| v == name).unwrap_or(false))
                        .cloned()
                } else {
                    None
                };
                if let Some(vkey) = vkey {
                    let tag = ir.enum_variant_tags[&vkey];
                    // A FIELDLESS variant of a BOXED enum (one with a payload
                    // sibling, e.g. `Opt::None`) must lower to the SAME heap
                    // record `[tag, 0…]` its payload siblings use, so the match's
                    // `__mind_load_i64(scrutinee + 0)` tag-read dereferences a
                    // valid record instead of a bare ordinal (`*1` → SEGFAULT). A
                    // purely fieldless (C-like) enum is not boxed and keeps the
                    // bare tag.
                    let enum_name = vkey.rsplit_once("::").map(|(e, _)| e);
                    if let Some(&total_slots) = enum_name.and_then(|e| ir.enum_payload_slots.get(e))
                    {
                        return emit_boxed_enum_record(ir, tag, &[], total_slots);
                    }
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(id, tag));
                    return id;
                }
            }
            // Undefined — emit placeholder.
            #[cfg(debug_assertions)]
            eprintln!("[WARN] lower_expr: undefined identifier `{name}`, defaulting to 0");
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Binary {
            op, left, right, ..
        } => {
            let lhs = lower_expr(left, ir, env, struct_env, receiver_types);
            let rhs = lower_expr(right, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let op = match op {
                ast::BinOp::Add => BinOp::Add,
                ast::BinOp::Sub => BinOp::Sub,
                ast::BinOp::Mul => BinOp::Mul,
                ast::BinOp::Div => BinOp::Div,
                ast::BinOp::Mod => BinOp::Mod,
                ast::BinOp::Lt => BinOp::Lt,
                ast::BinOp::Le => BinOp::Le,
                ast::BinOp::Gt => BinOp::Gt,
                ast::BinOp::Ge => BinOp::Ge,
                ast::BinOp::Eq => BinOp::Eq,
                ast::BinOp::Ne => BinOp::Ne,
            };
            ir.instrs.push(Instr::BinOp { dst, op, lhs, rhs });
            dst
        }
        // Logical `&&` / `||` (Phase 10.5 `Node::Logical`, kept separate from
        // `BinOp`). The IR has no logical-and/or instruction, so we DESUGAR to
        // the existing, keystone-stable `Node::If` lowering — the same proven
        // technique as `desugar_match_to_if`. Without this arm `a && b` falls
        // through the master catch-all below and silently lowers to `const 0`
        // (a release-silent miscompile).
        //
        // The desugar is interpreter-faithful (src/eval/mod.rs `Node::Logical`):
        // short-circuit + 0/1 normalisation. Crucially every BRANCH RESULT is a
        // literal i64 `0`/`1` (never a bare comparison), so no `i1` value ever
        // reaches an If-merge block-arg (which would mis-type as i64); the
        // conditions are `e != 0` comparisons — the i1 fast-path of the
        // If/While `cond_already_i1` check.
        //
        //   a && b  ->  if a != 0 { if b != 0 { 1 } else { 0 } } else { 0 }
        //   a || b  ->  if a != 0 { 1 } else { if b != 0 { 1 } else { 0 } }
        //
        // Additive: only code that previously degenerated to `const 0` (i.e.
        // was already broken) changes emitted text, so existing artifacts and
        // the keystone byte-identity are unaffected.
        ast::Node::Logical {
            op,
            left,
            right,
            span,
        } => {
            let span = *span;
            let lit = |n: i64| ast::Node::Lit(ast::Literal::Int(n), span);
            // Turn an operand into a boolean If-condition. A comparison already
            // produces the MLIR `i1` the If lowering wants on its fast path, so
            // it is used DIRECTLY — wrapping it in `e != 0` would emit
            // `cmpi ne <i1>, 0 : i64` (an i1 used as i64, which mlir-opt
            // rejects). Any non-comparison operand is an i64, so `e != 0` gives
            // correct truthiness (matching the interpreter) and lowers cleanly.
            let is_cmp = |e: &ast::Node| {
                matches!(
                    e,
                    ast::Node::Binary {
                        op: ast::BinOp::Lt
                            | ast::BinOp::Le
                            | ast::BinOp::Gt
                            | ast::BinOp::Ge
                            | ast::BinOp::Eq
                            | ast::BinOp::Ne,
                        ..
                    }
                )
            };
            let as_cond = |e: ast::Node| {
                if is_cmp(&e) {
                    e
                } else {
                    ast::Node::Binary {
                        op: ast::BinOp::Ne,
                        left: Box::new(e),
                        right: Box::new(ast::Node::Lit(ast::Literal::Int(0), span)),
                        span,
                    }
                }
            };
            // `if b { 1 } else { 0 }` — normalise the RHS to a true i64 0/1.
            let norm_right = ast::Node::If {
                cond: Box::new(as_cond((**right).clone())),
                then_branch: vec![lit(1)],
                else_branch: Some(vec![lit(0)]),
                span,
            };
            let desugared = match op {
                ast::LogicalOp::And => ast::Node::If {
                    cond: Box::new(as_cond((**left).clone())),
                    then_branch: vec![norm_right],
                    else_branch: Some(vec![lit(0)]),
                    span,
                },
                ast::LogicalOp::Or => ast::Node::If {
                    cond: Box::new(as_cond((**left).clone())),
                    then_branch: vec![lit(1)],
                    else_branch: Some(vec![norm_right]),
                    span,
                },
            };
            lower_expr(&desugared, ir, env, struct_env, receiver_types)
        }
        // Phase 6.5 Stage 1a — bitwise binary operators.
        // `ast::Node::Bitwise` is kept separate from `Node::Binary` by design
        // (see ast/mod.rs comments). Map each BitOp to its IR BinOp variant.
        // Gated to `std-surface`.
        #[cfg(feature = "std-surface")]
        ast::Node::Bitwise {
            op, left, right, ..
        } => {
            let lhs = lower_expr(left, ir, env, struct_env, receiver_types);
            let rhs = lower_expr(right, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let ir_op = match op {
                ast::BitOp::And => BinOp::BitAnd,
                ast::BitOp::Or => BinOp::BitOr,
                ast::BitOp::Xor => BinOp::BitXor,
                ast::BitOp::Shl => BinOp::Shl,
                ast::BitOp::Shr => BinOp::Shr,
            };
            ir.instrs.push(Instr::BinOp {
                dst,
                op: ir_op,
                lhs,
                rhs,
            });
            dst
        }
        ast::Node::CallTensorSum {
            x, axes, keepdims, ..
        } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let axes = axes.iter().map(|a| *a as i64).collect();
            ir.instrs.push(Instr::Sum {
                dst,
                src,
                axes,
                keepdims: *keepdims,
            });
            dst
        }
        ast::Node::CallTensorMean {
            x, axes, keepdims, ..
        } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let axes = axes.iter().map(|a| *a as i64).collect();
            ir.instrs.push(Instr::Mean {
                dst,
                src,
                axes,
                keepdims: *keepdims,
            });
            dst
        }
        ast::Node::CallTensorRelu { x, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Relu { dst, src });
            dst
        }
        ast::Node::CallReshape { x, dims, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let new_shape = dims.iter().map(|dim| parse_dim(dim)).collect();
            ir.instrs.push(Instr::Reshape {
                dst,
                src,
                new_shape,
            });
            dst
        }
        ast::Node::CallExpandDims { x, axis, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::ExpandDims {
                dst,
                src,
                axis: *axis as i64,
            });
            dst
        }
        ast::Node::CallSqueeze { x, axes, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let axes = axes.iter().map(|a| *a as i64).collect();
            ir.instrs.push(Instr::Squeeze { dst, src, axes });
            dst
        }
        ast::Node::CallTranspose { x, axes, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let perm = axes
                .as_ref()
                .map(|axes| axes.iter().map(|a| *a as i64).collect())
                .unwrap_or_default();
            ir.instrs.push(Instr::Transpose { dst, src, perm });
            dst
        }
        ast::Node::CallIndex { x, axis, i, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let indices = vec![IndexSpec {
                axis: (*axis).max(0) as i64,
                index: (*i).max(0) as i64,
            }];
            ir.instrs.push(Instr::Index { dst, src, indices });
            dst
        }
        ast::Node::CallMatMul { a, b, .. } => {
            let lhs = lower_expr(a, ir, env, struct_env, receiver_types);
            let rhs = lower_expr(b, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::MatMul {
                dst,
                a: lhs,
                b: rhs,
            });
            dst
        }
        // RFC 0012 Phase B — `A @ B` matmul operator.
        //
        // DESUGAR POINT (single, well-defined): `A @ B` lowers here to
        // `Instr::MatMul { a, b }` — the same IR node that `CallMatMul`
        // (the explicit `tensor.matmul(A, B)` form) produces.  This
        // guarantees byte-identical IR text between `A @ B` and
        // `tensor.matmul(A, B)`.
        //
        // MLIR-level byte-identity with `matmul_rmajor_f32_v` (the RFC
        // 0012 §7.2 gate-matrix target) requires threading shape dims
        // (M, K) through from the type-checker to emit the correct
        // `Instr::Call` args — deferred to Phase B.2.  At the IR text
        // level (`format_ir_module`) both forms emit `matmul %A, %B`,
        // which is byte-identical and sufficient for the Phase B gate
        // as implemented in this test suite.
        ast::Node::TensorMatmul { lhs, rhs, .. } => {
            let a = lower_expr(lhs, ir, env, struct_env, receiver_types);
            let b = lower_expr(rhs, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::MatMul { dst, a, b });
            dst
        }
        // RFC 0012 Phase B — elementwise `.+ .- .* ./` operators.
        //
        // DESUGAR POINT (single, well-defined): desugars to `Instr::BinOp`
        // — the same IR node that `Node::Binary` (scalar `+`, `-`, `*`, `/`)
        // produces for tensor operands.  The IR-level representation is
        // identical: both forms emit `add %L, %R` (or sub/mul/div).
        ast::Node::TensorElemwise { op, lhs, rhs, .. } => {
            let l = lower_expr(lhs, ir, env, struct_env, receiver_types);
            let r = lower_expr(rhs, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let ir_op = match op {
                TensorElemOp::Add => BinOp::Add,
                TensorElemOp::Sub => BinOp::Sub,
                TensorElemOp::Mul => BinOp::Mul,
                TensorElemOp::Div => BinOp::Div,
            };
            ir.instrs.push(Instr::BinOp {
                dst,
                op: ir_op,
                lhs: l,
                rhs: r,
            });
            dst
        }
        ast::Node::CallTensorRand { shape, .. } => {
            let dst = ir.fresh();
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            ir.instrs.push(Instr::ConstTensor(
                dst,
                crate::types::DType::F32,
                dims.iter()
                    .map(|s| crate::types::ShapeDim::Known(s.parse().unwrap()))
                    .collect(),
                None, // None = random fill, forces GPU materialization
            ));
            dst
        }
        ast::Node::CallDot { a, b, .. } => {
            let lhs = lower_expr(a, ir, env, struct_env, receiver_types);
            let rhs = lower_expr(b, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Dot {
                dst,
                a: lhs,
                b: rhs,
            });
            dst
        }
        ast::Node::CallSlice {
            x,
            axis,
            start,
            end,
            ..
        } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let dims = vec![SliceSpec {
                axis: (*axis).max(0) as i64,
                start: (*start).max(0) as i64,
                end: Some((*end).max(0) as i64),
                stride: 1,
            }];
            ir.instrs.push(Instr::Slice { dst, src, dims });
            dst
        }
        ast::Node::CallSliceStride {
            x,
            axis,
            start,
            end,
            step,
            ..
        } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let dims = vec![SliceSpec {
                axis: (*axis).max(0) as i64,
                start: (*start).max(0) as i64,
                end: Some((*end).max(0) as i64),
                stride: (*step).max(1) as i64,
            }];
            ir.instrs.push(Instr::Slice { dst, src, dims });
            dst
        }
        ast::Node::CallGather { x, axis, idx, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let indices = lower_expr(idx, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Gather {
                dst,
                src,
                indices,
                axis: (*axis).max(0) as i64,
            });
            dst
        }
        ast::Node::Paren(inner, _) => lower_expr(inner, ir, env, struct_env, receiver_types),
        ast::Node::Tuple { elements, .. } => {
            // A tuple is an anonymous all-i64 product type, lowered with the
            // exact machinery of an all-i64 `StructLit` / multi-payload enum
            // variant: `addr = __mind_alloc(8*n)`, then store each element at
            // offset `8*i`, and the tuple VALUE is the base pointer. This lets a
            // tuple flow through bindings, enum payloads (`Ok((a, b))`) and
            // returns, and be read back by a destructuring `let (a, b) = …`.
            // A 0-tuple is unit `0`; a 1-tuple `(x)` is just `x` (grouping),
            // matching the parser's 1-element collapse — so single values never
            // pay the alloc and nothing that worked before regresses. The
            // keystone source contains no tuple literals, so its emit is
            // byte-identical.
            let n = elements.len();
            if n <= 1 {
                return elements
                    .first()
                    .map(|e| lower_expr(e, ir, env, struct_env, receiver_types))
                    .unwrap_or_else(|| {
                        let id = ir.fresh();
                        ir.instrs.push(Instr::ConstI64(id, 0));
                        id
                    });
            }
            // bytes = 8 * n
            let eight = ir.fresh();
            ir.instrs.push(Instr::ConstI64(eight, 8));
            let count = ir.fresh();
            ir.instrs.push(Instr::ConstI64(count, n as i64));
            let bytes = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: bytes,
                op: BinOp::Mul,
                lhs: eight,
                rhs: count,
            });
            // addr = __mind_alloc(bytes)
            let addr = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: addr,
                name: "__mind_alloc".to_string(),
                args: vec![bytes],
            });
            // Store each element at offset 8*i (lowered in source order, after
            // the alloc — same left-to-right evaluation as `StructLit`).
            for (i, element) in elements.iter().enumerate() {
                let value = lower_expr(element, ir, env, struct_env, receiver_types);
                let field_addr = if i == 0 {
                    addr
                } else {
                    let offset = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(offset, (i as i64) * 8));
                    let sum = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: sum,
                        op: BinOp::Add,
                        lhs: addr,
                        rhs: offset,
                    });
                    sum
                };
                let store_ret = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst: store_ret,
                    name: "__mind_store_i64".to_string(),
                    args: vec![field_addr, value],
                });
            }
            addr
        }
        ast::Node::FnDef {
            name,
            type_params,
            params,
            ret_type,
            body,
            reap_threshold,
            ..
        } => {
            // Codegen monomorphization: a generic fn is a TEMPLATE, never
            // emitted as an `Instr::FnDef` here. It was registered in the
            // pre-pass; concrete instances are emitted after the module body
            // is lowered (sorted by mangled name). Like every other declaration
            // arm, a template produces no value — return the unit placeholder.
            if !type_params.is_empty() {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                return id;
            }
            // RFC 0012 §5.1 — record this fn's ABI signature (param `.ty`s +
            // declared return type) in the IRModule side-table so the MLIR
            // FnDef emitter can type each `func.func` param / return as
            // `f64`/`f32` where the source declared a scalar float, rather than
            // defaulting to the i64 ABI. Pure metadata; an i64-only fn records
            // all-`ScalarI64` here and lowers byte-identically as before.
            #[cfg(feature = "std-surface")]
            {
                let param_types: Vec<crate::ast::TypeAnn> =
                    params.iter().map(|p| p.ty.clone()).collect();
                ir.fn_signatures
                    .insert(name.clone(), (param_types, ret_type.clone()));
            }
            // `ret_type` only feeds the std-surface signature table above; touch
            // it in the default build so the destructured binding isn't flagged
            // unused.
            #[cfg(not(feature = "std-surface"))]
            let _ = ret_type;
            // Lower function definition
            let mut fn_ir = IRModule::new();
            // RFC 0005 P0f Step 1 — the FieldAccess read-path resolves
            // a field offset via `fn_ir.struct_defs[T]`; without
            // inheriting the parent module's schema registry, every
            // struct used inside a fn body would silently fall through
            // to the placeholder. Schema is metadata only — cloning
            // does not duplicate any IR instructions and is gated to
            // std-surface so non-feature builds incur zero cost.
            #[cfg(feature = "std-surface")]
            {
                fn_ir.struct_defs = ir.struct_defs.clone();
                // Phase 6.2b Gap 2: inherit const-array data so that
                // fn bodies can re-emit ConstArray nodes on demand.
                fn_ir.const_array_defs = ir.const_array_defs.clone();
                // "finish MIND" Step 2: inherit the enum-discriminant table so
                // a variant referenced inside a fn body (as a value or a match
                // arm) resolves to its tag rather than the placeholder 0.
                fn_ir.enum_variant_tags = ir.enum_variant_tags.clone();
                // Inherit the boxed-enum set so a fieldless variant of a
                // payload-carrying enum constructed inside a fn body lowers to
                // the uniform heap record (not a bare ordinal the match would
                // dereference as a pointer).
                fn_ir.boxed_enums = ir.boxed_enums.clone();
                fn_ir.enum_payload_slots = ir.enum_payload_slots.clone();
                fn_ir.enum_payload_types = ir.enum_payload_types.clone();
                fn_ir.enum_struct_field_names = ir.enum_struct_field_names.clone();
                // Width-aware struct ABI: inherit the per-struct field-type table
                // so StructLit/FieldAccess/FieldAssign inside a fn body compute
                // the canonical offset/width (sub-i64 fields), not the legacy
                // 8-byte stride. Metadata only; an all-i64 struct still takes the
                // byte-identical fast path.
                fn_ir.struct_field_types = ir.struct_field_types.clone();
            }
            // Build fn_env from env, but do NOT carry over const-array
            // SSA ids from the outer module — those ids are only valid in
            // the outer ir's SSA namespace.  Const-array identifiers will
            // be re-resolved in the Ident arm below via const_array_defs.
            let mut fn_env: HashMap<String, ValueId> = env
                .iter()
                .filter(|(name, _)| {
                    #[cfg(feature = "std-surface")]
                    {
                        !ir.const_array_defs.contains_key(*name)
                    }
                    #[cfg(not(feature = "std-surface"))]
                    {
                        // `name` is only consulted under `std-surface`
                        // (const-array shadowing); touch it here so the
                        // binding isn't flagged unused in the default build.
                        let _ = name;
                        true
                    }
                })
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            // RFC 0005 P0f Step 1 — fresh per-fn struct binding map.
            // Inherits outer module-scope bindings (so an outer
            // `let cfg = Config { ... }` is visible to inner field
            // reads) but additions inside this fn body do not leak
            // back out to siblings or to module scope. `mut` is
            // unused without std-surface; silence the lint here too.
            #[allow(unused_mut)]
            let mut fn_struct_env = struct_env.clone();

            // Create parameters
            let mut param_pairs = Vec::new();
            for (idx, param) in params.iter().enumerate() {
                let param_id = fn_ir.fresh();
                fn_ir.instrs.push(Instr::Param {
                    dst: param_id,
                    name: param.name.clone(),
                    index: idx,
                });
                fn_env.insert(param.name.clone(), param_id);
                param_pairs.push((param.name.clone(), param_id));
                // `array<T>` param → vec sentinel so `p.push/get/len/length` and
                // `p[i]` in the body resolve to the std.vec runtime (e.g.
                // mind-flow `fn assign_ids(order: array<string>)`).
                #[cfg(feature = "std-surface")]
                if is_array_surface_ty(&param.ty) {
                    fn_struct_env.insert(param.name.clone(), ARRAY_VEC_SENTINEL.to_string());
                    if let Some(__e) = array_element_track(&param.ty) {
                        fn_struct_env.insert(format!("__elem__{}", param.name), __e);
                    }
                }
                // `map<K, V>` param → map sentinel (str-key vs i64-key).
                #[cfg(feature = "std-surface")]
                if is_map_surface_ty(&param.ty) {
                    fn_struct_env
                        .insert(param.name.clone(), map_sentinel_for(&param.ty).to_string());
                }
                // `set<T>` param → set sentinel.
                #[cfg(feature = "std-surface")]
                if is_set_surface_ty(&param.ty) {
                    fn_struct_env
                        .insert(param.name.clone(), set_sentinel_for(&param.ty).to_string());
                }
                // Struct-typed param → record its struct TYPE NAME so a field
                // access on it (`analyzed.determinism`) resolves the field's type
                // for the collection-method desugar. Just the Named name; the
                // struct may be cross-module (merged into struct_defs from the
                // global registry), so no struct_defs check here.
                #[cfg(feature = "std-surface")]
                if let TypeAnn::Named(n) = &param.ty {
                    fn_struct_env
                        .entry(param.name.clone())
                        .or_insert_with(|| n.clone());
                }
            }

            // Part 1 (generics): expose this fn's params as inferable concrete
            // types so a generic call `id(p)` over a scalar param monomorphizes.
            // No-op (no allocation) unless the module declares templates; the
            // guard restores the prior map when this fn's body finishes lowering.
            let _param_types_guard = seed_param_types(params);
            // Fresh narrow-locals scope (seeded with this fn's narrow params) so a
            // REASSIGNMENT of a narrow local/param re-masks to its declared width.
            // Empty (and a no-op) for any fn with no narrow scalars — the keystone
            // included — so the byte-identity hot path is untouched. Guard restores
            // the prior map on body-scope exit, so narrow types never leak across fns.
            #[cfg(feature = "std-surface")]
            let _narrow_guard = enter_narrow_scope(params);
            // Whether the module declares generic templates (computed once):
            // gates the per-Let binding-type recording below so a non-generic
            // module records nothing on its byte-identity hot path.
            let gen_active = MONO.with(|c| !c.borrow().templates.is_empty());

            // Lower function body.
            //
            // `Return` is unique to fn scope and handled inline.
            // `Let` / `Assign` / expression stmts share the same
            // Let→tensor-binding + Assign→bind + expr pattern that is
            // extracted in `lower_stmt_seq` (used by `Node::Region`).
            // FnDef-specific extras — P0f struct-env tracking and Gap-C
            // branch-binding propagation — are layered on top after each
            // stmt is lowered.
            let mut ret_id = None;
            for stmt in body {
                match stmt {
                    ast::Node::Return { value, .. } => {
                        if let Some(val) = value {
                            ret_id = Some(lower_expr(
                                val,
                                &mut fn_ir,
                                &fn_env,
                                &fn_struct_env,
                                receiver_types,
                            ));
                        }
                        fn_ir.instrs.push(Instr::Return { value: ret_id });
                    }
                    ast::Node::Let {
                        name, ann, value, ..
                    } => {
                        let id = match ann {
                            Some(TypeAnn::Tensor { dtype, dims })
                            | Some(TypeAnn::DiffTensor { dtype, dims }) => lower_tensor_binding(
                                &mut fn_ir,
                                value,
                                dtype,
                                dims,
                                &fn_env,
                                &fn_struct_env,
                                receiver_types,
                            ),
                            // `array<T>` binding with an array-literal RHS:
                            // lower onto the std.vec heap runtime.
                            #[cfg(feature = "std-surface")]
                            _ if (is_array_surface_type(ann)
                                && matches!(value.as_ref(), ast::Node::ArrayLit { .. }))
                                || is_growable_bytes_init(ann, value) =>
                            {
                                let elements = match value.as_ref() {
                                    ast::Node::ArrayLit { elements, .. } => elements.as_slice(),
                                    _ => &[],
                                };
                                lower_array_surface_lit(
                                    elements,
                                    &mut fn_ir,
                                    &fn_env,
                                    &fn_struct_env,
                                    receiver_types,
                                )
                            }
                            _ => lower_expr(
                                value,
                                &mut fn_ir,
                                &fn_env,
                                &fn_struct_env,
                                receiver_types,
                            ),
                        };
                        // Narrow-typed binding: mask/sign-adjust to declared width
                        // (no-op for i64/u64/handles/collections).
                        #[cfg(feature = "std-surface")]
                        let id = mask_narrow_let(&mut fn_ir, ann, id);
                        // Remember a narrow declared type so a LATER `c = c + …`
                        // reassignment (incl. the desugared `c += …`) re-masks.
                        #[cfg(feature = "std-surface")]
                        record_narrow_let(name, ann);
                        fn_env.insert(name.clone(), id);
                        // Generic-arg inference: record this top-level Let's
                        // scalar type so a later `id(z)` over `z` monomorphizes.
                        // Resolved AFTER the value lowers (the RHS cannot see its
                        // own binding) and ONLY when templates are present —
                        // exactly mirroring the abi_gate gate's forward walk
                        // (lockstep). No-op for a non-generic module.
                        if gen_active {
                            PARAM_TYPES.with(|p| {
                                FN_RETURNS
                                    .with(|fr| bind_let(&mut p.borrow_mut(), stmt, &fr.borrow()))
                            });
                        }
                        // P0f Step 1: track fn-scoped var→struct binding for
                        // FieldAccess inside this fn body.
                        #[cfg(feature = "std-surface")]
                        if let ast::Node::StructLit {
                            name: struct_name, ..
                        } = value.as_ref()
                        {
                            fn_struct_env.insert(name.clone(), struct_name.clone());
                        }
                        // `let x = y` where `y` is a tracked struct/collection
                        // local or param: `x` aliases `y`'s type + element
                        // tracking (e.g. `let next = t; next.scopes.push(..)`).
                        #[cfg(feature = "std-surface")]
                        if let ast::Node::Lit(Literal::Ident(src), _) = value.as_ref() {
                            if let Some(t) = fn_struct_env.get(src).cloned() {
                                fn_struct_env.entry(name.clone()).or_insert(t);
                            }
                            if let Some(e) = fn_struct_env.get(&format!("__elem__{src}")).cloned() {
                                fn_struct_env
                                    .entry(format!("__elem__{name}"))
                                    .or_insert(e);
                            }
                        }
                        // `array<T>` binding: record the vec sentinel so a later
                        // method/index on this name resolves to the std.vec runtime.
                        #[cfg(feature = "std-surface")]
                        if is_array_surface_type(ann) {
                            fn_struct_env.insert(name.clone(), ARRAY_VEC_SENTINEL.to_string());
                            if let Some(__e) = ann.as_ref().and_then(array_element_track) {
                                fn_struct_env.insert(format!("__elem__{}", name), __e);
                            }
                        }
                        // Dynamic `bytes = [..]` tracks as the vec surface.
                        #[cfg(feature = "std-surface")]
                        if is_growable_bytes_init(ann, value) {
                            fn_struct_env.insert(name.clone(), ARRAY_VEC_SENTINEL.to_string());
                        }
                        #[cfg(feature = "std-surface")]
                        if let Some(s) = map_sentinel_for_opt(ann) {
                            fn_struct_env.insert(name.clone(), s.to_string());
                        }
                        #[cfg(feature = "std-surface")]
                        if let Some(s) = set_sentinel_for_opt(ann) {
                            fn_struct_env.insert(name.clone(), s.to_string());
                        }
                        // RHS type inference (`let raw = f(...); raw.split(…)`).
                        #[cfg(feature = "std-surface")]
                        if let Some((__s, __e)) =
                            let_rhs_collection_track(value, &fn_ir, &fn_struct_env, receiver_types)
                        {
                            fn_struct_env.entry(name.clone()).or_insert(__s);
                            if let Some(__el) = __e {
                                fn_struct_env
                                    .entry(format!("__elem__{}", name))
                                    .or_insert(__el);
                            }
                        }
                    }
                    ast::Node::LetTuple { names, value, .. } => {
                        // Tuple-destructuring `let (a, b) = expr` in a fn body:
                        // lower the RHS to the tuple base pointer, then bind each
                        // name to `__mind_load_i64(addr + 8*i)` in `fn_env` — the
                        // read side of the `Node::Tuple` aggregate. A tuple-free fn
                        // body never reaches here, so the keystone is byte-identical.
                        let addr =
                            lower_expr(value, &mut fn_ir, &fn_env, &fn_struct_env, receiver_types);
                        for (i, nm) in names.iter().enumerate() {
                            let elem_addr = if i == 0 {
                                addr
                            } else {
                                let offset = fn_ir.fresh();
                                fn_ir.instrs.push(Instr::ConstI64(offset, (i as i64) * 8));
                                let sum = fn_ir.fresh();
                                fn_ir.instrs.push(Instr::BinOp {
                                    dst: sum,
                                    op: BinOp::Add,
                                    lhs: addr,
                                    rhs: offset,
                                });
                                sum
                            };
                            let loaded = fn_ir.fresh();
                            fn_ir.instrs.push(Instr::Call {
                                dst: loaded,
                                name: "__mind_load_i64".to_string(),
                                args: vec![elem_addr],
                            });
                            fn_env.insert(nm.clone(), loaded);
                        }
                    }
                    ast::Node::Assign { name, value, .. } => {
                        let id =
                            lower_expr(value, &mut fn_ir, &fn_env, &fn_struct_env, receiver_types);
                        // Reassigning a narrow local/param re-masks to its declared
                        // width (no-op for non-narrow names — the all-i64 hot path).
                        #[cfg(feature = "std-surface")]
                        let id = mask_narrow_assign(&mut fn_ir, name, id);
                        fn_env.insert(name.clone(), id);
                    }
                    other => {
                        let id =
                            lower_expr(other, &mut fn_ir, &fn_env, &fn_struct_env, receiver_types);
                        ret_id = Some(id);
                        // Gap C: if the emitted statement was an `Instr::If`,
                        // thread its branch_bindings back into `fn_env` so
                        // subsequent statements in this fn body can reference
                        // let bindings declared inside either branch.
                        #[cfg(feature = "std-surface")]
                        if let Some(Instr::If {
                            branch_bindings, ..
                        }) = fn_ir.instrs.last()
                        {
                            for (bname, bid) in branch_bindings.clone() {
                                fn_env.insert(bname, bid);
                            }
                        }
                        // RFC 0005 Gap 1: if the emitted statement was an
                        // `Instr::While`, thread live_vars back into `fn_env`
                        // so code after the loop uses the post-loop SSA ids.
                        // The While emitter appends a trailing ConstI64(unit,0)
                        // after the While instr itself, so we check the
                        // second-to-last instruction for the While node.
                        #[cfg(feature = "std-surface")]
                        {
                            let n = fn_ir.instrs.len();
                            if n >= 2 {
                                if let Instr::While {
                                    live_vars,
                                    exit_ids,
                                    ..
                                } = &fn_ir.instrs[n - 2]
                                {
                                    // F2: rebind to the loop EXIT id (dominating
                                    // ^while_after block arg), not the
                                    // body-internal post_id.
                                    for (k, (vname, _post)) in live_vars.iter().enumerate() {
                                        let exit =
                                            exit_ids.get(k).copied().unwrap_or(live_vars[k].1);
                                        fn_env.insert(vname.clone(), exit);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Add function definition to IR, propagating the REAP threshold
            // from the AST attribute if present.
            ir.instrs.push(Instr::FnDef {
                name: name.clone(),
                params: param_pairs,
                ret_id,
                body: fn_ir.instrs,
                reap_threshold: *reap_threshold,
            });

            // Function definitions don't produce a value
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Return { value, .. } => {
            let ret_val = value
                .as_ref()
                .map(|v| lower_expr(v, ir, env, struct_env, receiver_types));
            ir.instrs.push(Instr::Return { value: ret_val });
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Block { stmts, .. } => {
            // A `let` binding inside a block must be visible to the statements
            // that follow it in the SAME block — e.g. a block-valued `match` arm
            // `{ let x = 1\n x }`. `lower_expr`'s `env` is immutable, so thread a
            // block-scoped LOCAL clone and bind each `let` into it (mirroring the
            // fn-body loop); without this a `let` reaching here would route into
            // the fail-loud catch-all and panic. The clone is block-scoped, so
            // bindings do not leak to the enclosing scope, and for a let-free
            // block `local_env` stays equal to `env` — so every other statement
            // lowers against an identical env and the emitted IR (and the
            // keystone) is byte-identical to the old simple loop.
            #[allow(unused_mut)]
            let mut local_env = env.clone();
            #[allow(unused_mut)]
            let mut local_struct_env = struct_env.clone();
            let mut last_id = None;
            for stmt in stmts {
                if let ast::Node::Let {
                    name, ann, value, ..
                } = stmt
                {
                    let id = match ann {
                        Some(TypeAnn::Tensor { dtype, dims })
                        | Some(TypeAnn::DiffTensor { dtype, dims }) => lower_tensor_binding(
                            ir,
                            value,
                            dtype,
                            dims,
                            &local_env,
                            &local_struct_env,
                            receiver_types,
                        ),
                        // `array<T>` binding with an array-literal RHS: lower
                        // onto the std.vec heap runtime.
                        #[cfg(feature = "std-surface")]
                        _ if is_array_surface_type(ann)
                            && matches!(value.as_ref(), ast::Node::ArrayLit { .. }) =>
                        {
                            let elements = match value.as_ref() {
                                ast::Node::ArrayLit { elements, .. } => elements.as_slice(),
                                _ => &[],
                            };
                            lower_array_surface_lit(
                                elements,
                                ir,
                                &local_env,
                                &local_struct_env,
                                receiver_types,
                            )
                        }
                        _ => lower_expr(value, ir, &local_env, &local_struct_env, receiver_types),
                    };
                    // Narrow-typed block-local: mask/sign-adjust to declared width.
                    #[cfg(feature = "std-surface")]
                    let id = mask_narrow_let(ir, ann, id);
                    local_env.insert(name.clone(), id);
                    #[cfg(feature = "std-surface")]
                    record_narrow_let(name, ann);
                    // P0f Step 1: track var→struct binding so a later FieldAccess
                    // inside this block resolves the canonical field offset.
                    #[cfg(feature = "std-surface")]
                    if let ast::Node::StructLit {
                        name: struct_name, ..
                    } = value.as_ref()
                    {
                        local_struct_env.insert(name.clone(), struct_name.clone());
                    }
                    // `array<T>` binding: record the vec sentinel for later
                    // method/index resolution onto the std.vec runtime.
                    #[cfg(feature = "std-surface")]
                    if is_array_surface_type(ann) {
                        local_struct_env.insert(name.clone(), ARRAY_VEC_SENTINEL.to_string());
                        if let Some(__e) = ann.as_ref().and_then(array_element_track) {
                            local_struct_env.insert(format!("__elem__{}", name), __e);
                        }
                    }
                    #[cfg(feature = "std-surface")]
                    if let Some(s) = set_sentinel_for_opt(ann) {
                        local_struct_env.insert(name.clone(), s.to_string());
                    }
                    #[cfg(feature = "std-surface")]
                    if let Some(s) = map_sentinel_for_opt(ann) {
                        local_struct_env.insert(name.clone(), s.to_string());
                    }
                    // RHS type inference (`let raw = f(...); raw.split(…)`).
                    #[cfg(feature = "std-surface")]
                    if let Some((__s, __e)) =
                        let_rhs_collection_track(value, ir, &local_struct_env, receiver_types)
                    {
                        local_struct_env.entry(name.clone()).or_insert(__s);
                        if let Some(__el) = __e {
                            local_struct_env
                                .entry(format!("__elem__{}", name))
                                .or_insert(__el);
                        }
                    }
                    last_id = Some(id);
                } else if let ast::Node::LetTuple { names, value, .. } = stmt {
                    // Tuple-destructuring `let (a, b) = expr` inside a block: lower
                    // the RHS to the tuple base pointer, then bind each name to
                    // `__mind_load_i64(addr + 8*i)` in the block-local env — the
                    // read side of the `Node::Tuple` aggregate. Tuple-free blocks
                    // never reach here, so the keystone stays byte-identical.
                    let addr = lower_expr(value, ir, &local_env, &local_struct_env, receiver_types);
                    for (i, nm) in names.iter().enumerate() {
                        let elem_addr = if i == 0 {
                            addr
                        } else {
                            let offset = ir.fresh();
                            ir.instrs.push(Instr::ConstI64(offset, (i as i64) * 8));
                            let sum = ir.fresh();
                            ir.instrs.push(Instr::BinOp {
                                dst: sum,
                                op: BinOp::Add,
                                lhs: addr,
                                rhs: offset,
                            });
                            sum
                        };
                        let loaded = ir.fresh();
                        ir.instrs.push(Instr::Call {
                            dst: loaded,
                            name: "__mind_load_i64".to_string(),
                            args: vec![elem_addr],
                        });
                        local_env.insert(nm.clone(), loaded);
                        last_id = Some(loaded);
                    }
                } else {
                    last_id = Some(lower_expr(
                        stmt,
                        ir,
                        &local_env,
                        &local_struct_env,
                        receiver_types,
                    ));
                }
            }
            last_id.unwrap_or_else(|| {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                id
            })
        }
        // Phase 6.5 Stage 1a — `if cond { then } else { else }` lowering.
        //
        // The condition, then-branch, and else-branch each lower into separate
        // sub-IRModule scratch buffers so that `Instr::Return` nodes inside a
        // branch do not appear as mid-block terminators in the parent flat
        // instruction stream. The MLIR lowerer converts `Instr::If` into an
        // `scf.if` or a `cf.cond_br`+basic-block structure, placing each
        // branch's instructions in its own MLIR basic block.
        //
        // Gap C: `let` bindings produced inside either branch are collected in
        // `branch_bindings` and re-inserted into the outer `env` after the
        // `Instr::If` is emitted so subsequent statements in the same scope can
        // reference them. This replicates the pattern `Instr::While` uses for
        // `live_vars`.
        //
        // Gated to `std-surface`.
        #[cfg(feature = "std-surface")]
        ast::Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            // ── 1. Lower the condition into a scratch sub-module ──────────────
            //
            // Sub-modules must inherit `struct_defs` and `const_array_defs`
            // from the parent IR so that `FieldAccess` and const-array
            // references inside branch conditions and bodies resolve correctly.
            //
            // Chain `next_id`: cond_ir starts at ir.next_id, then_ir starts
            // at cond_ir.next_id, else_ir starts at then_ir.next_id. This
            // ensures all ValueIds across all three sub-modules are globally
            // unique and disjoint from the parent scope's ids (especially fn
            // parameters which occupy the lowest ids).
            let mut cond_ir = sub_ir_from(ir);
            let cond_env = env.clone();
            let cond_id = lower_expr(cond, &mut cond_ir, &cond_env, struct_env, receiver_types);

            // ── 2. Lower the then-branch into a scratch sub-module ────────────
            //      Starts from cond_ir's highest id.
            let mut then_ir = sub_ir_from_after(&cond_ir, ir);
            let mut then_env = env.clone();
            // F2: names this branch writes — outer-var Assigns, branch-local
            // Lets, and any outer var rebound by a NESTED region (loop/if).
            // The union of then/else writes becomes the merge phi set, and each
            // merged var's per-branch value is taken from this env (dominating
            // at the branch's exit).
            let mut then_writes: Vec<String> = Vec::new();
            let record_then_write = |name: &str, writes: &mut Vec<String>| {
                if !writes.iter().any(|n| n == name) {
                    writes.push(name.to_owned());
                }
            };
            // Branch-local struct-type env so a collection/string `let` inside the
            // then-branch is tracked for later methods in the same branch
            // (`let raw = f(); for p in raw.split(…)`). Outer `struct_env` is
            // shared-immutable.
            #[cfg(feature = "std-surface")]
            let mut then_struct_env = struct_env.clone();
            #[cfg(not(feature = "std-surface"))]
            let then_struct_env = struct_env;
            let mut then_result = then_ir.fresh();
            then_ir.instrs.push(Instr::ConstI64(then_result, 0));
            for stmt in then_branch {
                match stmt {
                    ast::Node::Return { value, .. } => {
                        let ret_val = value.as_ref().map(|v| {
                            lower_expr(v, &mut then_ir, &then_env, &then_struct_env, receiver_types)
                        });
                        then_ir.instrs.push(Instr::Return { value: ret_val });
                        if let Some(rv) = ret_val {
                            then_result = rv;
                        }
                    }
                    ast::Node::Let {
                        name, ann, value, ..
                    } => {
                        let id = match ann {
                            Some(TypeAnn::Tensor { dtype, dims })
                            | Some(TypeAnn::DiffTensor { dtype, dims }) => lower_tensor_binding(
                                &mut then_ir,
                                value,
                                dtype,
                                dims,
                                &then_env,
                                &then_struct_env,
                                receiver_types,
                            ),
                            _ => lower_expr(
                                value,
                                &mut then_ir,
                                &then_env,
                                &then_struct_env,
                                receiver_types,
                            ),
                        };
                        // Narrow-typed branch-local: mask/sign-adjust to width.
                        #[cfg(feature = "std-surface")]
                        let id = mask_narrow_let(&mut then_ir, ann, id);
                        then_env.insert(name.clone(), id);
                        #[cfg(feature = "std-surface")]
                        record_narrow_let(name, ann);
                        #[cfg(feature = "std-surface")]
                        {
                            let _ = ann;
                            if is_array_surface_type(ann) {
                                then_struct_env
                                    .insert(name.clone(), ARRAY_VEC_SENTINEL.to_string());
                            }
                            if let Some(s) = map_sentinel_for_opt(ann) {
                                then_struct_env.insert(name.clone(), s.to_string());
                            }
                            if let Some(s) = set_sentinel_for_opt(ann) {
                                then_struct_env.insert(name.clone(), s.to_string());
                            }
                            if let Some((__s, __e)) = let_rhs_collection_track(
                                value,
                                &then_ir,
                                &then_struct_env,
                                receiver_types,
                            ) {
                                then_struct_env.entry(name.clone()).or_insert(__s);
                                if let Some(__el) = __e {
                                    then_struct_env
                                        .entry(format!("__elem__{}", name))
                                        .or_insert(__el);
                                }
                            }
                        }
                        record_then_write(name, &mut then_writes);
                        then_result = id;
                    }
                    ast::Node::Assign { name, value, .. } => {
                        let id =
                            lower_expr(value, &mut then_ir, &then_env, &then_struct_env, receiver_types);
                        // Re-mask a narrow local reassigned inside the then-branch.
                        #[cfg(feature = "std-surface")]
                        let id = mask_narrow_assign(&mut then_ir, name, id);
                        then_env.insert(name.clone(), id);
                        record_then_write(name, &mut then_writes);
                        then_result = id;
                    }
                    ast::Node::LetTuple { names, value, .. } => {
                        then_result = lower_lettuple_stmt(
                            names,
                            value,
                            &mut then_ir,
                            &mut then_env,
                            struct_env,
                            receiver_types,
                        );
                        for nm in names {
                            record_then_write(nm, &mut then_writes);
                        }
                    }
                    other => {
                        then_result =
                            lower_expr(other, &mut then_ir, &then_env, &then_struct_env, receiver_types);
                        // F2: thread a nested region's EXIT/merge ids upward so
                        // an outer var mutated inside it is visible (and
                        // dominating) at this branch's exit.
                        for (nm, eid) in last_region_exit_rebindings(&then_ir.instrs) {
                            then_env.insert(nm.clone(), eid);
                            record_then_write(&nm, &mut then_writes);
                        }
                    }
                }
            }

            // ── 3. Lower the else-branch (or synthesise a unit zero) ──────────
            //      Starts from then_ir's highest id.
            let mut else_ir = sub_ir_from_after(&then_ir, ir);
            let mut else_env = env.clone();
            let mut else_writes: Vec<String> = Vec::new();
            let record_else_write = |name: &str, writes: &mut Vec<String>| {
                if !writes.iter().any(|n| n == name) {
                    writes.push(name.to_owned());
                }
            };
            #[cfg(feature = "std-surface")]
            let mut else_struct_env = struct_env.clone();
            #[cfg(not(feature = "std-surface"))]
            let else_struct_env = struct_env;
            let mut else_result = else_ir.fresh();
            else_ir.instrs.push(Instr::ConstI64(else_result, 0));
            if let Some(else_stmts) = else_branch {
                for stmt in else_stmts {
                    match stmt {
                        ast::Node::Return { value, .. } => {
                            let ret_val = value.as_ref().map(|v| {
                                lower_expr(v, &mut else_ir, &else_env, &else_struct_env, receiver_types)
                            });
                            else_ir.instrs.push(Instr::Return { value: ret_val });
                            if let Some(rv) = ret_val {
                                else_result = rv;
                            }
                        }
                        ast::Node::Let {
                            name, ann, value, ..
                        } => {
                            let id = match ann {
                                Some(TypeAnn::Tensor { dtype, dims })
                                | Some(TypeAnn::DiffTensor { dtype, dims }) => {
                                    lower_tensor_binding(
                                        &mut else_ir,
                                        value,
                                        dtype,
                                        dims,
                                        &else_env,
                                        &else_struct_env,
                                        receiver_types,
                                    )
                                }
                                _ => lower_expr(
                                    value,
                                    &mut else_ir,
                                    &else_env,
                                    &else_struct_env,
                                    receiver_types,
                                ),
                            };
                            // Narrow-typed branch-local: mask/sign-adjust to width.
                            #[cfg(feature = "std-surface")]
                            let id = mask_narrow_let(&mut else_ir, ann, id);
                            else_env.insert(name.clone(), id);
                            #[cfg(feature = "std-surface")]
                            record_narrow_let(name, ann);
                            #[cfg(feature = "std-surface")]
                            {
                                let _ = ann;
                                if is_array_surface_type(ann) {
                                    else_struct_env
                                        .insert(name.clone(), ARRAY_VEC_SENTINEL.to_string());
                                }
                                if let Some(s) = map_sentinel_for_opt(ann) {
                                    else_struct_env.insert(name.clone(), s.to_string());
                                }
                                if let Some(s) = set_sentinel_for_opt(ann) {
                                    else_struct_env.insert(name.clone(), s.to_string());
                                }
                                if let Some((__s, __e)) = let_rhs_collection_track(
                                    value,
                                    &else_ir,
                                    &else_struct_env,
                                    receiver_types,
                                ) {
                                    else_struct_env.entry(name.clone()).or_insert(__s);
                                    if let Some(__el) = __e {
                                        else_struct_env
                                            .entry(format!("__elem__{}", name))
                                            .or_insert(__el);
                                    }
                                }
                            }
                            record_else_write(name, &mut else_writes);
                            else_result = id;
                        }
                        ast::Node::Assign { name, value, .. } => {
                            let id = lower_expr(
                                value,
                                &mut else_ir,
                                &else_env,
                                &else_struct_env,
                                receiver_types,
                            );
                            // Re-mask a narrow local reassigned inside the else-branch.
                            #[cfg(feature = "std-surface")]
                            let id = mask_narrow_assign(&mut else_ir, name, id);
                            else_env.insert(name.clone(), id);
                            record_else_write(name, &mut else_writes);
                            else_result = id;
                        }
                        ast::Node::LetTuple { names, value, .. } => {
                            else_result = lower_lettuple_stmt(
                                names,
                                value,
                                &mut else_ir,
                                &mut else_env,
                                struct_env,
                                receiver_types,
                            );
                            for nm in names {
                                record_else_write(nm, &mut else_writes);
                            }
                        }
                        other => {
                            else_result = lower_expr(
                                other,
                                &mut else_ir,
                                &else_env,
                                &else_struct_env,
                                receiver_types,
                            );
                            for (nm, eid) in last_region_exit_rebindings(&else_ir.instrs) {
                                else_env.insert(nm.clone(), eid);
                                record_else_write(&nm, &mut else_writes);
                            }
                        }
                    }
                }
            }

            // ── 4. Build the merge phi set ────────────────────────────────────
            //
            // F2 dominance fix. For every variable written in EITHER branch,
            // allocate a fresh merge id (declared as an `^if_after` block arg)
            // and record, per branch, the value of that variable at the branch
            // EXIT (`then_env`/`else_env`). These per-branch values dominate the
            // branch's `cf.br ^if_after` because they are either the incoming
            // value, a top-level branch value, or a nested region's exit id
            // (threaded above) — never a raw value defined in a deeper branch.
            //
            // A branch that does not write the variable passes its incoming
            // value (`env[name]`). If the variable does not exist in the outer
            // env either (a branch-local `let`), that branch synthesises a unit
            // 0 inside its own block so both edges still pass a dominating
            // value of matching type.
            //
            // `branch_bindings[i].1` is set to the merge id so post-if code and
            // upward threading (`region_exit_rebindings`) pick up the
            // dominating merge value, never a branch-internal id.
            ir.next_id = ir.next_id.max(else_ir.next_id);
            // If exactly one branch is EMPTY — so its if-value `*_result` is the
            // bare `ConstI64(0)` placeholder — while the other yields an `f64`,
            // re-type the empty branch's placeholder to `ConstF64(0.0)` so the
            // if-VALUE column types `f64` instead of the i64 default (the same
            // hazard the one-sided merge placeholder had, but on the if-value
            // column). This propagates through a desugared `match`'s nested-if
            // chain (the innermost arm's empty else). ADDITIVE: an all-i64 or
            // both-non-empty `if` is unchanged → byte-identical.
            let then_empty = then_branch.is_empty();
            let else_empty = match else_branch {
                Some(s) => s.is_empty(),
                None => true,
            };
            // Allocate the placeholder id from `ir` (the canonical space, already
            // synced past both branch IRs above) — NOT from `then_ir`/`else_ir`,
            // whose sequential id spaces overlap the merge block-arg ids and would
            // collide (exactly as the merge-phi placeholders below do).
            if then_empty
                && !else_empty
                && branch_value_is_f64(&else_ir.instrs, else_result, &ir.fn_signatures)
            {
                let z = ir.fresh();
                then_ir.instrs.push(Instr::ConstF64(z, 0.0));
                then_result = z;
            }
            if else_empty
                && !then_empty
                && branch_value_is_f64(&then_ir.instrs, then_result, &ir.fn_signatures)
            {
                let z = ir.fresh();
                else_ir.instrs.push(Instr::ConstF64(z, 0.0));
                else_result = z;
            }
            let mut merged_names: Vec<String> = Vec::new();
            for n in then_writes.iter().chain(else_writes.iter()) {
                if !merged_names.iter().any(|m| m == n) {
                    merged_names.push(n.clone());
                }
            }
            // A branch that ends in a block terminator does not fall through to
            // `^if_after`; its `cf.br` is omitted and it must not pass a merge
            // value (and must not get a dead const pushed after its terminator).
            // `return` always terminates; under `std-surface`, `break` /
            // `continue` (a `match` arm body that exits/re-tests the enclosing
            // `while`) terminate too. Recognizing only `Return` here wrongly
            // deemed a `break`/`continue` branch to fall through, pushing dead
            // merge-placeholder consts AFTER the terminator — which then made
            // the MLIR if-lowering's `.last()` terminator check (correctly
            // returns true for Break/Continue) miss, so it appended a second
            // `cf.br ^if_after`, yielding a mid-block + trailing `cf.br` that
            // mlir-opt rejects. ADDITIVE: a non-std-surface build only matches
            // `Return`, so the keystone stays byte-identical.
            fn branch_terminates(instrs: &[Instr]) -> bool {
                match instrs.last() {
                    Some(Instr::Return { .. }) => true,
                    #[cfg(feature = "std-surface")]
                    Some(Instr::Break { .. }) | Some(Instr::Continue { .. }) => true,
                    _ => false,
                }
            }
            let then_falls_through = !branch_terminates(&then_ir.instrs);
            let else_falls_through = !branch_terminates(&else_ir.instrs);
            let mut branch_bindings: Vec<(String, ValueId)> = Vec::new();
            let mut merges: Vec<(ValueId, ValueId, ValueId)> = Vec::new();
            for name in &merged_names {
                let then_has = then_env.get(name).copied();
                let else_has = else_env.get(name).copied();
                // Type the synthesized absent-side ZERO placeholder by the side
                // that DEFINES the binding (a one-sided let/assign): an f64
                // binding gets an f64 placeholder so the merge phi types f64
                // rather than clashing with the i64 default. i64 stays
                // `ConstI64(0)` → every all-i64 program is byte-identical.
                let placeholder_f64 = match (then_has, else_has) {
                    (Some(tid), None) => {
                        branch_value_is_f64(&then_ir.instrs, tid, &ir.fn_signatures)
                    }
                    (None, Some(eid)) => {
                        branch_value_is_f64(&else_ir.instrs, eid, &ir.fn_signatures)
                    }
                    _ => false,
                };
                // then-edge value (only meaningful if then falls through).
                let then_val = if then_falls_through {
                    match then_has {
                        Some(id) => id,
                        None => {
                            let z = ir.fresh();
                            then_ir.instrs.push(if placeholder_f64 {
                                Instr::ConstF64(z, 0.0)
                            } else {
                                Instr::ConstI64(z, 0)
                            });
                            z
                        }
                    }
                } else {
                    // No then-edge; reuse the else value as the placeholder so
                    // the tuple is well-formed (the then `cf.br` is not emitted).
                    else_has.unwrap_or(ValueId(usize::MAX))
                };
                // else-edge value (only meaningful if else falls through).
                let else_val = if else_falls_through {
                    match else_has {
                        Some(id) => id,
                        None => {
                            let z = ir.fresh();
                            else_ir.instrs.push(if placeholder_f64 {
                                Instr::ConstF64(z, 0.0)
                            } else {
                                Instr::ConstI64(z, 0)
                            });
                            z
                        }
                    }
                } else {
                    then_has.unwrap_or(ValueId(usize::MAX))
                };
                let merge_id = ir.fresh();
                merges.push((merge_id, then_val, else_val));
                branch_bindings.push((name.clone(), merge_id));
            }

            // ── 5. Emit Instr::If into the parent IR stream ───────────────────
            let dst = ir.fresh();
            ir.instrs.push(Instr::If {
                cond_id,
                cond_instrs: cond_ir.instrs,
                then_instrs: then_ir.instrs,
                then_result,
                else_instrs: else_ir.instrs,
                else_result,
                dst,
                branch_bindings,
                merges,
            });

            // Gap C: branch_bindings are stored on the Instr::If node so
            // callers that own a mutable env (e.g. the fn-body loop below)
            // can thread them back after the if.  `lower_expr` takes `env`
            // as a shared reference and cannot mutate the outer scope here.

            dst
        }
        // Non-gated fallback for `ast::Node::If` when `std-surface` is off.
        // Retains the old sequential-flatten behaviour so the default build
        // compiles and the existing `if_expr` tests continue to pass.
        #[cfg(not(feature = "std-surface"))]
        ast::Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            let _cond_id = lower_expr(cond, ir, env, struct_env, receiver_types);
            let mut last_id = None;
            for stmt in then_branch {
                last_id = Some(lower_expr(stmt, ir, env, struct_env, receiver_types));
            }
            if let Some(else_stmts) = else_branch {
                for stmt in else_stmts {
                    last_id = Some(lower_expr(stmt, ir, env, struct_env, receiver_types));
                }
            }
            last_id.unwrap_or_else(|| {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                id
            })
        }
        ast::Node::Call { callee, args, .. } => {
            // "finish MIND" Step 5 — payload-carrying enum constructor.
            // When the callee resolves to an enum variant in
            // `enum_variant_tags` AND the call has at least one argument
            // (e.g. `Opt::Some(42)`), the variant is a payload-carrying
            // constructor, NOT a runtime function call. Build the same
            // 2-field heap record the StructLit arm builds —
            // `[tag @ +0, payload @ +8]` — so the matching
            // `Opt::Some(v) => …` arm (desugared below) can load the tag
            // from `scrutinee + 0` and the bound payload from
            // `scrutinee + 8`. A FIELDLESS variant is never a `Call`
            // (it parses to `Lit(Ident)` and lowers to its bare tag in the
            // `Lit(Ident)` arm above), so this only fires for tuple
            // variants. The record shape is identical to StructLit's
            // (`__mind_alloc` + `__mind_store_i64` at 8-byte offsets), so
            // no new Instr/intrinsic/ABI is introduced.
            #[cfg(feature = "std-surface")]
            if !args.is_empty() {
                // Resolve the variant key: a qualified `Enum::V` callee matches
                // directly; a BARE `V` (`Some(x)`, `Ok(v)`, `Err(e)`) is matched
                // against any `Enum::V` in the registry so UNQUALIFIED
                // constructors resolve (one global link unit). A bare name that
                // collides across enums takes the first in deterministic BTreeMap
                // order — distinct in practice (Ok/Err/Some/None).
                let vkey: Option<String> = if ir.enum_variant_tags.contains_key(callee) {
                    Some(callee.clone())
                } else if !callee.contains("::") {
                    ir.enum_variant_tags
                        .keys()
                        .find(|k| {
                            k.rsplit_once("::")
                                .map(|(_, v)| v == callee)
                                .unwrap_or(false)
                        })
                        .cloned()
                } else {
                    None
                };
                if let Some(vkey) = vkey {
                    let tag = ir.enum_variant_tags[&vkey];
                    // Lower every payload field in declaration order (mirrors the
                    // StructLit per-field lowering order), coercing a non-i64
                    // field (e.g. `f64`) to its raw bits so the i64 slot holds it.
                    let field_types = ir.enum_payload_types.get(&vkey).cloned();
                    let payloads: Vec<ValueId> = args
                        .iter()
                        .enumerate()
                        .map(|(i, a)| {
                            let ty = field_types.as_ref().and_then(|ts| ts.get(i));
                            let coerced = coerce_enum_field_to_bits(a.clone(), ty, a.span());
                            lower_expr(&coerced, ir, env, struct_env, receiver_types)
                        })
                        .collect();
                    // Record size = the enum's uniform `1 + max arity` (recovered
                    // from the RESOLVED qualified key so a bare ctor is sized for
                    // its whole enum, not just this variant's arity).
                    let enum_name = vkey.rsplit_once("::").map(|(e, _)| e);
                    let total_slots = enum_name
                        .and_then(|e| ir.enum_payload_slots.get(e).copied())
                        .unwrap_or(1 + payloads.len());
                    return emit_boxed_enum_record(ir, tag, &payloads, total_slots);
                }
            }
            let arg_ids: Vec<ValueId> = args
                .iter()
                .map(|a| lower_expr(a, ir, env, struct_env, receiver_types))
                .collect();
            // Codegen monomorphization: if the callee is a registered generic
            // and the concrete arg type is inferable, route this call to the
            // mangled instance (`id$i64`) and queue that instance for emission.
            // A non-generic callee (or a shape outside the bounded slice) keeps
            // the original name, so non-generic call lowering is byte-identical.
            let name = try_register_mono_instance(callee, args).unwrap_or_else(|| callee.clone());
            let dst = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst,
                name,
                args: arg_ids,
            });
            dst
        }
        // Phase 10.7 / "finish MIND" Step 1: `match scrutinee { arms }` —
        // DESUGAR to a right-nested chain of `Instr::If`. Each integer/bool
        // (`Literal::Int`) arm becomes `if scrutinee == <lit> { body } else
        // { rest }`; a `Wildcard` (`_`) or bare `Ident` arm becomes the
        // terminal `else` (an `Ident` first binds the scrutinee under that
        // name). The desugar is purely at the AST level and recurses into the
        // existing `ast::Node::If` lowering, so it reuses the keystone-
        // protected `exit_ids`/`merges`/`region_exit_rebindings` machinery
        // untouched. Enum-discriminant and payload-binding patterns are a
        // later step and fall back to the old sequential lowering.
        #[cfg(feature = "std-surface")]
        ast::Node::Match {
            scrutinee, arms, ..
        } => {
            match desugar_match_to_if(
                scrutinee,
                arms,
                &ir.enum_variant_tags,
                &ir.boxed_enums,
                &ir.enum_payload_types,
                &ir.enum_struct_field_names,
            ) {
                Some(if_node) => lower_expr(&if_node, ir, env, struct_env, receiver_types),
                None => {
                    // Unsupported pattern kind (enum variant / non-int
                    // literal) — preserve the prior sequential behaviour so
                    // those matches are not regressed by this step.
                    let _scrut_id = lower_expr(scrutinee, ir, env, struct_env, receiver_types);
                    let mut last_id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(last_id, 0));
                    for arm in arms {
                        last_id = lower_expr(&arm.body, ir, env, struct_env, receiver_types);
                    }
                    last_id
                }
            }
        }
        // Non-gated fallback: default builds have no branching `If` lowering,
        // so retain the sequential-flatten behaviour.
        #[cfg(not(feature = "std-surface"))]
        ast::Node::Match {
            scrutinee, arms, ..
        } => {
            let _scrut_id = lower_expr(scrutinee, ir, env, struct_env, receiver_types);
            let mut last_id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(last_id, 0));
            for arm in arms {
                last_id = lower_expr(&arm.body, ir, env, struct_env, receiver_types);
            }
            last_id
        }
        // Phase 10.7: `&expr` / `&mut expr` — no-op metadata wrapper in
        // v1. The inner expression lowers directly; the ref tag is only
        // meaningful to the type-checker.
        ast::Node::Ref { inner, .. } => lower_expr(inner, ir, env, struct_env, receiver_types),
        // A cast `<expr> as <ty>`. Scalars and raw pointers are all carried as
        // i64 SSA values, so for pointers / f-types / aliases the target type is
        // purely a type-checker concern and the operand lowers transparently
        // (mirrors `Ref` / `Paren`). Without an explicit arm the cast fell
        // through to the catch-all and was silently lowered to `const.i64 0`,
        // dropping the operand entirely — e.g. `memset(sa as *mut u8, 0, 16)`
        // lost `sa`, then the FFI bridge `inttoptr`-ed a zero, producing a
        // NULL-pointer memset and an `!llvm.ptr` vs `i64` mlir-opt type error.
        //
        // BUT a cast to a *narrow signed integer* (`i8`/`i16`/`i32`) must
        // actually narrow: scalars live full-width in i64 SSA, so `70000 as i16`
        // has to truncate to the low 16 bits and sign-extend — `(70000 & 0xFFFF)`
        // sign-extended = 4464, not 70000 (the no-op result). The BLAS vector
        // paths already do this with `arith.trunci`/`arith.extsi`; for the scalar
        // i64 ABI the equivalent is the shift pair `(x << (64-W)) >> (64-W)` with
        // an *arithmetic* (signed) right shift (`BinOp::Shr` -> `arith.shrsi`),
        // which both clears the high bits and sign-extends in a single
        // i64-carried value. The shift width is materialised as an i64 const so
        // the lowering stays within the existing scalar instruction set (no new
        // IR opcode, no mic@1/mic@3 layout change, no version bump). Gated to
        // `std-surface` because `BinOp::Shl`/`Shr` only exist there.
        #[cfg(feature = "std-surface")]
        ast::Node::As { expr, ty, .. } => {
            let val = lower_expr(expr, ir, env, struct_env, receiver_types);
            match scalar_int_cast_width(ty) {
                // Narrowing to a known signed integer narrower than 64 bits:
                // truncate to the low `width` bits and sign-extend via the
                // arithmetic shift pair. Same-width (i64) and unknown types
                // (pointers, floats, aliases, u32/u64 widening, etc.) fall
                // through to the transparent pass-through below.
                Some(width) if width < 64 => {
                    let shift = 64 - width as i64;
                    let shift_id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(shift_id, shift));
                    let shl_id = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: shl_id,
                        op: BinOp::Shl,
                        lhs: val,
                        rhs: shift_id,
                    });
                    let shr_id = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: shr_id,
                        op: BinOp::Shr,
                        lhs: shl_id,
                        rhs: shift_id,
                    });
                    shr_id
                }
                // Narrowing to a known unsigned integer narrower than 64 bits
                // (`u8`/`u16`/`u32`, in `as` and call form): ZERO-extend by
                // masking off the high bits with `val & ((1 << width) - 1)`.
                // Unlike the signed path this must NOT sign-extend, so it uses a
                // single `BitAnd` against an i64 const mask (no new IR opcode, no
                // mic@1/mic@3 layout change). `u64`/full-width stays transparent.
                _ => match scalar_uint_cast_width(ty) {
                    Some(width) if width < 64 => {
                        let mask: i64 = if width == 32 {
                            0xFFFF_FFFF
                        } else {
                            (1i64 << width) - 1
                        };
                        let mask_id = ir.fresh();
                        ir.instrs.push(Instr::ConstI64(mask_id, mask));
                        let and_id = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: and_id,
                            op: BinOp::BitAnd,
                            lhs: val,
                            rhs: mask_id,
                        });
                        and_id
                    }
                    _ => val,
                },
            }
        }
        #[cfg(not(feature = "std-surface"))]
        ast::Node::As { expr, .. } => lower_expr(expr, ir, env, struct_env, receiver_types),
        // RFC 0005 Gap 1: `while cond { body }` lowering.
        //
        // The condition and body each lower into their own sub-modules so
        // the MLIR stage can place them in separate basic blocks (header and
        // body blocks, respectively).  Mutable variables that are written in
        // the body are collected as `live_vars` and threaded as block
        // arguments in the MLIR lowering.
        //
        // Gated to `std-surface` — default builds never reach this arm.
        #[cfg(feature = "std-surface")]
        ast::Node::While { cond, body, .. } => {
            // Lower the condition expression into a scratch sub-module to
            // capture the instructions that produce it without polluting the
            // parent IR stream.  The resulting ValueIds are local to the
            // sub-module; MLIR lowering re-emits them verbatim in the header
            // block so the numbering is stable.
            //
            // Use sub_ir_from so the condition sub-module's ValueIds start
            // above the parent's current next_id.  Without this, a constant
            // emitted in the condition (e.g. ConstI64(ValueId(0), 16)) would
            // collide with the function's first parameter (%0: i64) when both
            // are serialised into the same MLIR func.func body — the same
            // fix already applied to Instr::If (see sub_ir_from comment).
            // --- Alias-break for loop-carried variables (silent-miscompile fix) ---
            // `let mut j = start` lowers as pure env aliasing: `j` and `start`
            // share ONE ValueId. The MLIR While emitter rewrites every textual
            // `%<init_id>` to the loop-carried block-arg (substitute_ids, purely
            // numeric, first-match-wins), so any read of the alias SOURCE inside
            // the condition or body gets clobbered into the loop counter — the
            // documented mlkem/toml silent miscompile. Mirror the If path's
            // env-level rebinding: give the carried variable a FRESH copy id so
            // it no longer shares an id with any still-referenced source.
            //
            // Guarded so it fires ONLY when a differently-named env variable
            // sharing the id is actually READ in cond/body (the exact clobber
            // condition). When the source is unread the alias is harmless and
            // the binding is left untouched — currently-correct programs stay
            // byte-identical.
            #[cfg(feature = "std-surface")]
            let seed_env: HashMap<String, ValueId> = {
                let mut seed = env.clone();
                // Loop-carried candidates: outer-scope vars assigned anywhere in
                // the body (including inside a branch/block/match executed in a
                // single iteration), matching the set the arm below records as
                // loop-carried (direct + nested-region rebindings).
                let mut assigned: Vec<String> = Vec::new();
                collect_assign_targets(body, &mut assigned);
                let mut candidates: Vec<String> = assigned
                    .into_iter()
                    .filter(|name| env.contains_key(name.as_str()))
                    .collect();
                // Deterministic processing order (no HashMap iteration order).
                candidates.sort();
                for name in &candidates {
                    let id = match seed.get(name) {
                        Some(v) => *v,
                        None => continue,
                    };
                    // Other env names currently sharing this ValueId — the alias
                    // sources whose reads substitute_ids would clobber.
                    let sources: std::collections::HashSet<String> = seed
                        .iter()
                        .filter(|(n, v)| n.as_str() != name.as_str() && **v == id)
                        .map(|(n, _)| n.clone())
                        .collect();
                    if sources.is_empty() {
                        continue;
                    }
                    let read = ast_reads_ident(cond, &sources)
                        || body.iter().any(|s| ast_reads_ident(s, &sources));
                    if !read {
                        continue;
                    }
                    // Materialise `%copy = name + 0` in the PARENT ir (dominates
                    // the loop entry) so `name` gets a distinct id from every
                    // source. i64 is sound: loop-carried scalars are i64 in this
                    // block-arg machinery.
                    let zero = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(zero, 0));
                    let copy = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: copy,
                        op: BinOp::Add,
                        lhs: id,
                        rhs: zero,
                    });
                    seed.insert(name.clone(), copy);
                }
                seed
            };
            #[cfg(not(feature = "std-surface"))]
            let seed_env = env.clone();

            #[cfg(feature = "std-surface")]
            let mut cond_ir = sub_ir_from(ir);
            #[cfg(not(feature = "std-surface"))]
            let mut cond_ir = IRModule::new();
            // Seed the condition sub-module's env with the current bindings
            // so identifiers in the condition (e.g. `i`, `n`) resolve.
            let cond_env = seed_env.clone();
            let cond_id = lower_expr(cond, &mut cond_ir, &cond_env, struct_env, receiver_types);

            // Lower the body into a scratch sub-module.  Track every Assign
            // target — those are the variables that are live across the
            // back-edge and must become block arguments in MLIR.
            //
            // Chain from cond_ir so body ValueIds are disjoint from both
            // parent scope and condition scope (mirrors sub_ir_from_after
            // in the Instr::If path).
            #[cfg(feature = "std-surface")]
            let mut body_ir = sub_ir_from_after(&cond_ir, ir);
            #[cfg(not(feature = "std-surface"))]
            let mut body_ir = IRModule::new();
            let mut body_env = seed_env.clone();
            let mut mutated: Vec<(String, ValueId)> = Vec::new();
            // Pre-loop ValueId for each mutated variable (parallel to mutated).
            // Captures the ValueId from env BEFORE the while loop so the MLIR
            // emitter can produce `cf.br ^while_header(init_0, init_1, ...)`.
            let mut init_ids: Vec<ValueId> = Vec::new();

            // Record that `name` is loop-carried with post-body value `new_id`.
            // The first time the loop sees a variable mutated, capture its
            // pre-loop init id from `body_env` (parallel to `mutated`).
            fn record_loop_mut(
                name: &str,
                new_id: ValueId,
                mutated: &mut Vec<(String, ValueId)>,
                init_ids: &mut Vec<ValueId>,
                pre_init: Option<ValueId>,
            ) {
                if let Some(pos) = mutated.iter().position(|(n, _)| n == name) {
                    mutated[pos].1 = new_id;
                } else {
                    init_ids.push(pre_init.unwrap_or(ValueId(usize::MAX)));
                    mutated.push((name.to_owned(), new_id));
                }
            }

            // The loop body gets its OWN mutable struct-type env clone so a
            // collection/string-typed `let` inside the body is tracked for later
            // methods in the same body (`let raw = f(); for p in raw.split(…)`).
            // The outer `struct_env` is shared-immutable, so without this a
            // loop-body let's type was lost.
            #[cfg(feature = "std-surface")]
            let mut body_struct_env = struct_env.clone();
            #[cfg(not(feature = "std-surface"))]
            let body_struct_env = struct_env;
            for stmt in body {
                match stmt {
                    ast::Node::Let {
                        name, ann, value, ..
                    } => {
                        // `let` inside the loop body introduces a new SSA binding
                        // scoped to the body.  Emit the RHS and update body_env so
                        // subsequent body statements can reference the binding.
                        // These are NOT live_vars (they don't survive across the
                        // back-edge) unless a later Assign overwrites them.
                        let new_id = lower_expr(
                            value,
                            &mut body_ir,
                            &body_env,
                            &body_struct_env,
                            receiver_types,
                        );
                        // Narrow-typed loop-body local: mask/sign-adjust to width.
                        #[cfg(feature = "std-surface")]
                        let new_id = mask_narrow_let(&mut body_ir, ann, new_id);
                        body_env.insert(name.clone(), new_id);
                        #[cfg(feature = "std-surface")]
                        record_narrow_let(name, ann);
                        #[cfg(feature = "std-surface")]
                        {
                            let _ = ann;
                            if is_array_surface_type(ann) {
                                body_struct_env
                                    .insert(name.clone(), ARRAY_VEC_SENTINEL.to_string());
                            }
                            if let Some(s) = map_sentinel_for_opt(ann) {
                                body_struct_env.insert(name.clone(), s.to_string());
                            }
                            if let Some(s) = set_sentinel_for_opt(ann) {
                                body_struct_env.insert(name.clone(), s.to_string());
                            }
                            // A `let p = T { .. }` declared INSIDE the loop body
                            // must record `p`'s struct type so a later `p.field`
                            // resolves its 8-byte offset (Step 1) — exactly as
                            // the module/fn-scope `Let` handler already does. Without
                            // this the in-loop field read fell through to the
                            // `ConstI64(0)` placeholder and SILENTLY read 0 instead
                            // of the stored value.
                            if let ast::Node::StructLit {
                                name: struct_name, ..
                            } = value.as_ref()
                            {
                                body_struct_env.insert(name.clone(), struct_name.clone());
                            }
                            // `let q = p` inside the loop aliases `p`'s tracked
                            // struct/collection type (and element tracking), matching
                            // the outer-scope alias rule.
                            if let ast::Node::Lit(Literal::Ident(src), _) = value.as_ref() {
                                if let Some(t) = body_struct_env.get(src).cloned() {
                                    body_struct_env.entry(name.clone()).or_insert(t);
                                }
                                if let Some(e) =
                                    body_struct_env.get(&format!("__elem__{src}")).cloned()
                                {
                                    body_struct_env
                                        .entry(format!("__elem__{name}"))
                                        .or_insert(e);
                                }
                            }
                            if let Some((__s, __e)) = let_rhs_collection_track(
                                value,
                                &body_ir,
                                &body_struct_env,
                                receiver_types,
                            ) {
                                body_struct_env.entry(name.clone()).or_insert(__s);
                                if let Some(__el) = __e {
                                    body_struct_env
                                        .entry(format!("__elem__{}", name))
                                        .or_insert(__el);
                                }
                            }
                        }
                    }
                    ast::Node::Assign { name, value, .. } => {
                        let pre_init = body_env.get(name.as_str()).copied();
                        let new_id = lower_expr(
                            value,
                            &mut body_ir,
                            &body_env,
                            &body_struct_env,
                            receiver_types,
                        );
                        // Re-mask a narrow loop-carried local to its declared width;
                        // the masked id is what gets recorded as the carried value.
                        #[cfg(feature = "std-surface")]
                        let new_id = mask_narrow_assign(&mut body_ir, name, new_id);
                        body_env.insert(name.clone(), new_id);
                        // Only record as loop-carried (crossing the back-edge) when
                        // `name` is a genuine outer-scope variable present in the
                        // pre-loop env. A body-local `let mut` reassigned inside the
                        // body (e.g. a byte-copy counter `j` reset between sibling
                        // inner loops) is re-initialised each iteration and must NOT
                        // cross the back-edge — mirroring the guard in the nested-
                        // region (`other =>`) arm below. Without this guard, `j`'s
                        // pre_init could be an id defined DEEP in the body (a prior
                        // sibling loop's exit arg), which becomes a non-dominating
                        // outer-loop init and drives substitute_ids to rewrite that
                        // inner after-block arg into the outer `%wbod_0_0`, tripping
                        // `redefinition of SSA value '%wbod_0_0'` at mlir-opt.
                        if env.contains_key(name.as_str()) {
                            record_loop_mut(name, new_id, &mut mutated, &mut init_ids, pre_init);
                        }
                    }
                    ast::Node::LetTuple { names, value, .. } => {
                        lower_lettuple_stmt(
                            names,
                            value,
                            &mut body_ir,
                            &mut body_env,
                            struct_env,
                            receiver_types,
                        );
                    }
                    other => {
                        lower_expr(other, &mut body_ir, &body_env, &body_struct_env, receiver_types);
                        // F2: a nested region (if/while) inside the loop body may
                        // mutate an OUTER (loop-carried) variable. Thread the
                        // nested region's EXIT/merge id into body_env AND record
                        // the variable as loop-carried, so the back-edge passes a
                        // dominating value and the header re-feeds it next
                        // iteration. Without this, mutations buried in a nested
                        // branch (e.g. `while c { if p { x = 0 } }`) are invisible
                        // to the loop and it never makes progress.
                        {
                            for (nm, eid) in last_region_exit_rebindings(&body_ir.instrs) {
                                // Update body_env for any variable the nested
                                // region modified that is visible in the current
                                // outer loop body scope (including `let mut`
                                // bindings declared earlier in this body that are
                                // NOT pre-loop vars, e.g. `let mut min = i`).
                                // This ensures reads AFTER the inner loop within
                                // the same outer-loop iteration see the updated id.
                                //
                                // Only call record_loop_mut (make it loop-carried
                                // across the back-edge) for variables that exist in
                                // the pre-loop env (`env`) — genuine outer vars.
                                // Body-local `let` bindings are re-initialised each
                                // iteration and must not cross the back-edge.
                                if body_env.contains_key(&nm) {
                                    let pre_init = body_env.get(nm.as_str()).copied();
                                    body_env.insert(nm.clone(), eid);
                                    if env.contains_key(&nm) {
                                        record_loop_mut(
                                            &nm,
                                            eid,
                                            &mut mutated,
                                            &mut init_ids,
                                            pre_init,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Note: we cannot update the parent `env` here because `lower_expr`
            // takes `env: &HashMap` (immutable).  The FnDef body loop detects
            // the emitted `Instr::While` and propagates `live_vars` back to its
            // own mutable `fn_env`, mirroring the `branch_bindings` pattern used
            // for `Instr::If`.  See the `other =>` arm in the FnDef lowering.

            // Advance the parent next_id past all IDs used in cond and body
            // so subsequent instructions in the parent fn body stay disjoint.
            #[cfg(feature = "std-surface")]
            {
                ir.next_id = ir.next_id.max(cond_ir.next_id).max(body_ir.next_id);
            }

            // F2 region-scoped exit env: one fresh SSA id per loop-carried var.
            // `^while_after_N` declares these as block args (fed by the
            // header→after cond_br edge with header args, which dominate), and
            // code AFTER the loop is rebound to these exit ids instead of the
            // body-internal `post_id`s — guaranteeing dominance for every
            // post-loop reference, at every nesting level.
            #[cfg(feature = "std-surface")]
            let exit_ids: Vec<ValueId> = mutated.iter().map(|_| ir.fresh()).collect();
            #[cfg(not(feature = "std-surface"))]
            let exit_ids: Vec<ValueId> = Vec::new();

            ir.instrs.push(Instr::While {
                cond_id,
                cond_instrs: cond_ir.instrs,
                body: body_ir.instrs,
                live_vars: mutated,
                init_ids,
                exit_ids,
            });

            // `while` is a statement; produce a unit i64 placeholder.
            let unit = ir.fresh();
            ir.instrs.push(Instr::ConstI64(unit, 0));
            unit
        }
        // RFC 0005 P0e Step 1 — `Foo { f1: v1, f2: v2, ... }` lowers to a
        // heap record. Layout = one `i64` slot per field, packed at
        // 8-byte stride. The struct value is the `i64` base address from
        // `__mind_alloc`; field reads are deferred to P0f (FieldAccess
        // needs the receiver's struct name threaded through env first).
        //
        //   addr = __mind_alloc(8 * N)
        //   __mind_store_i64(addr + 0,        v_for_field_0)
        //   __mind_store_i64(addr + 8,        v_for_field_1)
        //   ...
        //   addr            ← the struct's value
        //
        // Field order is canonical (from `StructDef`) — literals can
        // appear out of order and we reorder here. Unknown struct names
        // (no matching `StructDef` was lowered) fall through to literal
        // order so a forward-reference doesn't lose data.
        #[cfg(feature = "std-surface")]
        ast::Node::StructLit { name, fields, .. } => {
            // A StructLit whose name resolves to an enum VARIANT is a struct-variant
            // CONSTRUCTION `E.V { f: a, g: b }` (or `E::V { … }`), not a plain
            // struct. Build the boxed enum record `[tag, <fields in DECLARED
            // order>]`, reordering the provided fields by the variant's declared
            // `field_names` and coercing each across the i64 slot — the identical
            // machinery the tuple-variant ctor `Call` uses (emit_boxed_enum_record).
            #[cfg(feature = "std-surface")]
            {
                let dotnorm = name.rsplit_once('.').map(|(a, b)| format!("{a}::{b}"));
                let vkey = if ir.enum_variant_tags.contains_key(name) {
                    Some(name.clone())
                } else {
                    dotnorm.filter(|k| ir.enum_variant_tags.contains_key(k))
                };
                if let Some(vkey) = vkey {
                    let tag = ir.enum_variant_tags[&vkey];
                    let order = ir.enum_struct_field_names.get(&vkey).cloned();
                    let field_types = ir.enum_payload_types.get(&vkey).cloned();
                    if let Some(order) = order {
                        // Lower each declared field by NAME, in declaration order.
                        let payloads: Vec<ValueId> = order
                            .iter()
                            .enumerate()
                            .map(
                                |(i, fname)| match fields.iter().find(|f| &f.name == fname) {
                                    Some(f) => {
                                        let ty = field_types.as_ref().and_then(|ts| ts.get(i));
                                        let coerced = coerce_enum_field_to_bits(
                                            f.value.clone(),
                                            ty,
                                            f.value.span(),
                                        );
                                        lower_expr(&coerced, ir, env, struct_env, receiver_types)
                                    }
                                    None => {
                                        // A field omitted in the literal — zero-fill its
                                        // slot (the record is always fully initialised).
                                        let z = ir.fresh();
                                        ir.instrs.push(Instr::ConstI64(z, 0));
                                        z
                                    }
                                },
                            )
                            .collect();
                        let enum_name = vkey.rsplit_once("::").map(|(e, _)| e);
                        let total_slots = enum_name
                            .and_then(|e| ir.enum_payload_slots.get(e).copied())
                            .unwrap_or(1 + payloads.len());
                        return emit_boxed_enum_record(ir, tag, &payloads, total_slots);
                    }
                }
            }
            // Canonical field order, if the schema is known.
            let canonical = ir.struct_defs.get(name).cloned();
            let order: Vec<&ast::StructLitField> = match canonical {
                Some(names) => names
                    .iter()
                    .filter_map(|fname| fields.iter().find(|f| &f.name == fname))
                    .collect(),
                None => fields.iter().collect(),
            };
            let n = order.len() as i64;

            // Width-aware canonical layout, when the field-type side-table
            // knows this struct. `all_i64 == true` (every field 8 bytes, tightly
            // packed at 8*i) routes to the IDENTICAL legacy IR below so the
            // self-host records (all i64) stay byte-identical.
            let layout = struct_layout(ir, name).filter(|(l, _, _)| l.len() == order.len());
            let all_i64 = layout.as_ref().map(|(_, _, a)| *a).unwrap_or(true);

            if all_i64 {
                // bytes = 8 * n  — emit two consts + a Mul rather than a
                // precomputed literal so the IR matches what a future
                // arbitrary-N codegen path will produce.
                let eight = ir.fresh();
                ir.instrs.push(Instr::ConstI64(eight, 8));
                let count = ir.fresh();
                ir.instrs.push(Instr::ConstI64(count, n));
                let bytes = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst: bytes,
                    op: BinOp::Mul,
                    lhs: eight,
                    rhs: count,
                });

                // addr = __mind_alloc(bytes)
                let addr = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst: addr,
                    name: "__mind_alloc".to_string(),
                    args: vec![bytes],
                });

                // Per-field store at offset 8*i.
                for (i, f) in order.iter().enumerate() {
                    #[cfg(feature = "std-surface")]
                    let value = lower_struct_field_value(
                        name,
                        &f.name,
                        &f.value,
                        ir,
                        env,
                        struct_env,
                        receiver_types,
                    );
                    #[cfg(not(feature = "std-surface"))]
                    let value = lower_expr(&f.value, ir, env, struct_env, receiver_types);
                    let field_addr = if i == 0 {
                        addr
                    } else {
                        let offset = ir.fresh();
                        ir.instrs.push(Instr::ConstI64(offset, (i as i64) * 8));
                        let sum = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: sum,
                            op: BinOp::Add,
                            lhs: addr,
                            rhs: offset,
                        });
                        sum
                    };
                    let store_ret = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst: store_ret,
                        name: "__mind_store_i64".to_string(),
                        args: vec![field_addr, value],
                    });
                }

                return addr;
            }

            // Width-aware path: a struct with at least one sub-i64 field.
            // bytes = total (a single const — the size is fully determined by
            // the declared widths, no runtime Mul needed).
            let (layout, total, _) = layout.expect("non-all_i64 => layout is Some");
            let bytes = ir.fresh();
            ir.instrs.push(Instr::ConstI64(bytes, total));
            let addr = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: addr,
                name: "__mind_alloc".to_string(),
                args: vec![bytes],
            });
            for (i, f) in order.iter().enumerate() {
                #[cfg(feature = "std-surface")]
                let value = lower_struct_field_value(
                    name,
                    &f.name,
                    &f.value,
                    ir,
                    env,
                    struct_env,
                    receiver_types,
                );
                #[cfg(not(feature = "std-surface"))]
                let value = lower_expr(&f.value, ir, env, struct_env, receiver_types);
                let (offset, width, _signed) = layout[i];
                let field_addr = if offset == 0 {
                    addr
                } else {
                    let off = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(off, offset));
                    let sum = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: sum,
                        op: BinOp::Add,
                        lhs: addr,
                        rhs: off,
                    });
                    sum
                };
                let store_ret = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst: store_ret,
                    name: store_helper_for_width(width).to_string(),
                    args: vec![field_addr, value],
                });
            }
            addr
        }
        // RFC 0005 P0f — `receiver.field` reads from the heap record
        // produced by P0e StructLit lowering.
        //
        //   offset      = index_of(field, struct_defs[T]) * 8
        //   field_addr  = addr + offset    (or addr itself when offset == 0)
        //   result      = __mind_load_i64(field_addr)
        //
        // Step 1 — fast path: receiver is a plain `Ident` bound to a
        // `StructLit` via `Let` in this (or an enclosing) scope. The
        // receiver's struct name lives in `struct_env[var_name]`;
        // we look it up without re-lowering the receiver.
        //
        // Step 2 — general path: receiver type is precomputed by the
        // pre-pass in `src/eval/struct_resolver.rs` and stored in the
        // `receiver_types` side-table keyed on this FieldAccess's
        // span. Covers chained access (`a.b.c`), function-return
        // receivers (`foo().x`), and struct-typed parameters. We
        // lower the receiver expression for its base-address value
        // and then add the field's 8-byte offset as before.
        //
        // Unresolved receivers fall through to a `ConstI64(0)`
        // placeholder so the IR shape is stable and older modules
        // still compile.
        #[cfg(feature = "std-surface")]
        ast::Node::FieldAccess {
            receiver,
            field,
            span,
        } => {
            // `array<T>` length: `arr.len` / `arr.length` (no parens) on a
            // vec-sentinel receiver lowers to the std.vec `vec_len` free
            // function. mind-flow writes `.length`; std.vec exposes `vec_len`,
            // so the name is normalised here. The sentinel is only set for
            // `array<T>`-annotated bindings/params, so non-array `.len`/`.length`
            // field reads are unaffected and the keystone never hits this path.
            #[cfg(feature = "std-surface")]
            if field == "len" || field == "length" {
                // `arr.len`/`.length` → vec_len; `m.len`/`s.len` → map_len.
                // mind-flow writes `.length`; std exposes `vec_len`/`map_len`, so
                // the name is normalised. Resolves an Ident-bound collection AND a
                // struct-FIELD collection (`a.ids.len`) via the unified resolver,
                // so non-collection field reads are unaffected.
                #[cfg(feature = "std-surface")]
                {
                    let len_fn = match receiver_collection_sentinel(
                        receiver,
                        &ir,
                        struct_env,
                        receiver_types,
                    ) {
                        Some(ARRAY_VEC_SENTINEL) => Some("vec_len"),
                        Some(MAP_SENTINEL) | Some(MAP_STR_SENTINEL) => Some("map_len"),
                        Some(SET_SENTINEL) | Some(SET_STR_SENTINEL) => Some("map_len"),
                        _ => None,
                    };
                    if let Some(len_fn) = len_fn {
                        let recv_id = lower_expr(receiver, ir, env, struct_env, receiver_types);
                        let dst = ir.fresh();
                        ir.instrs.push(Instr::Call {
                            dst,
                            name: len_fn.to_string(),
                            args: vec![recv_id],
                        });
                        return dst;
                    }
                }
            }
            // ── Step 1: cheap Ident-bound lookup ─────────────────────
            let step1 = match receiver.as_ref() {
                ast::Node::Lit(Literal::Ident(var_name), _) => {
                    struct_env.get(var_name).and_then(|struct_name| {
                        ir.struct_defs
                            .get(struct_name)
                            .and_then(|fields| fields.iter().position(|f| f == field))
                            .map(|idx| (Some(var_name.clone()), idx, struct_name.clone()))
                    })
                }
                _ => None,
            };
            // ── Step 2: side-table fallback (general path) ───────────
            // Only consulted when Step 1 fast-path failed.
            let step2 = if step1.is_none() {
                receiver_types.get(span).and_then(|struct_name| {
                    ir.struct_defs
                        .get(struct_name)
                        .and_then(|fields| fields.iter().position(|f| f == field))
                        .map(|idx| (None::<String>, idx, struct_name.clone()))
                })
            } else {
                None
            };

            let resolved = step1.or(step2);

            match resolved {
                Some((var_name_opt, idx, struct_name)) => {
                    // Width-aware field offset/load. `(offset, width, signed)`;
                    // an all-i64 struct yields `(idx*8, 8, _)` so the emitted IR
                    // is byte-identical to the legacy path. Falls back to the
                    // legacy `idx*8` / i64 when the layout is unknown.
                    let fld =
                        struct_layout(ir, &struct_name).and_then(|(l, _, _)| l.get(idx).copied());
                    let (offset, width, signed) = fld.unwrap_or(((idx as i64) * 8, 8, true));
                    // Step 1 path can take addr from env without re-lowering.
                    // Step 2 path must lower the receiver expression to
                    // get its base address (it may be a Call, FieldAccess,
                    // or anything else that evaluates to an i64 heap addr).
                    let addr = match var_name_opt {
                        Some(var_name) => match env.get(&var_name) {
                            Some(id) => *id,
                            None => {
                                let id = ir.fresh();
                                ir.instrs.push(Instr::ConstI64(id, 0));
                                return id;
                            }
                        },
                        None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                    };
                    let field_addr = if offset == 0 {
                        addr
                    } else {
                        let off = ir.fresh();
                        ir.instrs.push(Instr::ConstI64(off, offset));
                        let sum = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: sum,
                            op: BinOp::Add,
                            lhs: addr,
                            rhs: off,
                        });
                        sum
                    };
                    let loaded = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst: loaded,
                        name: load_helper_for_width(width).to_string(),
                        args: vec![field_addr],
                    });
                    // The typed load zero-extends; a SIGNED narrow field needs a
                    // sign-extend (shl then arithmetic shr by 64-bits). i64 and
                    // unsigned/bool fields use the zero-extended value directly,
                    // so this is byte-identical for the all-i64 path.
                    if signed && width < 8 {
                        let shift = 64 - width * 8;
                        let sh = ir.fresh();
                        ir.instrs.push(Instr::ConstI64(sh, shift));
                        let shl = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: shl,
                            op: BinOp::Shl,
                            lhs: loaded,
                            rhs: sh,
                        });
                        let sar = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: sar,
                            op: BinOp::Shr,
                            lhs: shl,
                            rhs: sh,
                        });
                        sar
                    } else {
                        loaded
                    }
                }
                None => {
                    // Receiver type still unresolvable even after the
                    // side-table — emit placeholder so the module
                    // produces a stable IR shape. Step 3 will lift the
                    // remaining cases (heap-allocated fields of struct
                    // type, generics) when std.vec needs them.
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(id, 0));
                    id
                }
            }
        }
        // RFC 0005 P0g — `receiver.field = value` writes IN PLACE to the
        // heap record produced by P0e StructLit lowering. Exact inverse of
        // the FieldAccess read arm above: the same Step-1 (struct_env Ident
        // lookup) + Step-2 (receiver_types side-table) resolver computes the
        // field index, the same idx==0 fast path computes the field address,
        // and the RHS is lowered to an SSA value — but we emit a
        // `__mind_store_i64(field_addr, value)` instead of a load.
        //
        //   offset      = index_of(field, struct_defs[T]) * 8
        //   field_addr  = base + offset      (or base itself when offset == 0)
        //   __mind_store_i64(field_addr, value)
        //
        // The base address is the SAME allocation already bound in `env`
        // (Step 1) or produced by re-lowering the receiver (Step 2), so this
        // is a pure in-place mutation — no new struct-value SSA id is created
        // or threaded, and no exit_ids/merges/region rebinding changes are
        // needed. Both flat fields (`s.f = v`) and nested struct-typed field
        // writes (`o.inner.v = x`) are supported: the Step-2 side-table now
        // resolves a `FieldAccess` receiver (struct_resolver chains through
        // the inner field's declared type), so `lower_expr(receiver, …)`
        // re-lowers `o.inner` to the inner record's base address and the
        // store targets `base + idx*8` exactly like the flat case.
        // Unresolved receivers fall through to a `ConstI64(0)` placeholder,
        // matching the read arm, so older modules still compile.
        #[cfg(feature = "std-surface")]
        ast::Node::FieldAssign {
            receiver,
            field,
            value,
            span,
        } => {
            // ── Step 1: cheap Ident-bound lookup ─────────────────────
            let step1 = match receiver.as_ref() {
                ast::Node::Lit(Literal::Ident(var_name), _) => {
                    struct_env.get(var_name).and_then(|struct_name| {
                        ir.struct_defs
                            .get(struct_name)
                            .and_then(|fields| fields.iter().position(|f| f == field))
                            .map(|idx| (Some(var_name.clone()), idx, struct_name.clone()))
                    })
                }
                _ => None,
            };
            // ── Step 2: side-table fallback (general path) ───────────
            // Only consulted when Step 1 fast-path failed.
            let step2 = if step1.is_none() {
                receiver_types.get(span).and_then(|struct_name| {
                    ir.struct_defs
                        .get(struct_name)
                        .and_then(|fields| fields.iter().position(|f| f == field))
                        .map(|idx| (None::<String>, idx, struct_name.clone()))
                })
            } else {
                None
            };

            let resolved = step1.or(step2);

            match resolved {
                Some((var_name_opt, idx, struct_name)) => {
                    // Width-aware field offset/store; all-i64 yields (idx*8, 8)
                    // so the IR is byte-identical to the legacy path.
                    let fld =
                        struct_layout(ir, &struct_name).and_then(|(l, _, _)| l.get(idx).copied());
                    let (offset, width, _signed) = fld.unwrap_or(((idx as i64) * 8, 8, true));
                    // Step 1 takes the base addr straight from env (no
                    // re-lowering); Step 2 must lower the receiver to get it.
                    let addr = match var_name_opt {
                        Some(var_name) => match env.get(&var_name) {
                            Some(id) => *id,
                            None => {
                                let id = ir.fresh();
                                ir.instrs.push(Instr::ConstI64(id, 0));
                                return id;
                            }
                        },
                        None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                    };
                    let field_addr = if offset == 0 {
                        addr
                    } else {
                        let off = ir.fresh();
                        ir.instrs.push(Instr::ConstI64(off, offset));
                        let sum = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: sum,
                            op: BinOp::Add,
                            lhs: addr,
                            rhs: off,
                        });
                        sum
                    };
                    let rhs = lower_expr(value, ir, env, struct_env, receiver_types);
                    let store_ret = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst: store_ret,
                        name: store_helper_for_width(width).to_string(),
                        args: vec![field_addr, rhs],
                    });
                    // A field assignment is a statement; the store's return
                    // (unit) id is the value this expression yields.
                    store_ret
                }
                None => {
                    // Receiver type unresolvable — emit a stable placeholder,
                    // mirroring the read arm, so older modules still compile.
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(id, 0));
                    id
                }
            }
        }
        // RFC 0005 Phase 6.2b Gap 2 — anonymous array literal `[v0, v1, …]`
        // in expression position.  Elements are extracted iteratively
        // (not by recursing once per element) so a 4,096-entry literal
        // does not grow the Rust call-stack linearly.
        #[cfg(feature = "std-surface")]
        ast::Node::ArrayLit { elements, .. } => {
            let values: Vec<i64> = elements
                .iter()
                .map(|e| extract_const_i64(e).unwrap_or(0))
                .collect();
            let dst = ir.fresh();
            ir.instrs.push(Instr::ConstArray {
                dst,
                name: None,
                values,
            });
            dst
        }
        // Map literal `{}` / `{ k: v, … }` → std.map heap runtime (map_new +
        // map_insert chain). Unlike ArrayLit there is no const-map path, so this
        // one arm handles every position (binding RHS, arg, return). main.mind
        // has no map literal, so the keystone is unaffected.
        #[cfg(feature = "std-surface")]
        ast::Node::MapLit { entries, .. } => {
            lower_map_surface_lit(entries, ir, env, struct_env, receiver_types)
        }
        // Set literal `{ a, b, c }` → std.map runtime (map_new + map_insert(_,_,1)
        // chain — a set is a map keyed by its elements). main.mind has no set
        // literal, so the keystone is unaffected.
        #[cfg(feature = "std-surface")]
        ast::Node::SetLit { elements, .. } => {
            lower_set_surface_lit(elements, ir, env, struct_env, receiver_types)
        }
        // RFC 0005 Phase 6.2b Gap 2 — `receiver[index]`.  When the receiver
        // resolves to a ConstArray base address, this emits `ArrayLoad`.
        #[cfg(feature = "std-surface")]
        ast::Node::IndexAccess {
            receiver, index, ..
        } => {
            // `arr[i]` on a vec-sentinel (`array<T>`) receiver → std.vec
            // `vec_get` (the receiver is an i64 heap handle, not a const array,
            // so the `ArrayLoad` LUT path below would misinterpret it).
            // The vec-sentinel receiver may be an Ident (`arr[i]`) OR a struct
            // FIELD of `array<T>` (`b.items[i]`) — `receiver_collection_sentinel`
            // resolves both. A const-array Ident keeps the `ArrayLoad` path below.
            #[cfg(feature = "std-surface")]
            if receiver_collection_sentinel(receiver, &ir, struct_env, receiver_types)
                == Some(ARRAY_VEC_SENTINEL)
            {
                let base = lower_expr(receiver, ir, env, struct_env, receiver_types);
                let index_id = lower_expr(index, ir, env, struct_env, receiver_types);
                let dst = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst,
                    name: "vec_get".to_string(),
                    args: vec![base, index_id],
                });
                return dst;
            }
            let base = lower_expr(receiver, ir, env, struct_env, receiver_types);
            let index_id = lower_expr(index, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::ArrayLoad {
                dst,
                base,
                index: index_id,
            });
            dst
        }
        // RFC 0005 Phase 6.2b Gap 2 — `receiver[index] = value`.
        #[cfg(feature = "std-surface")]
        ast::Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            // `arr[i] = v` on a vec-sentinel (`array<T>`) receiver → std.vec
            // `vec_set`. A const array stays read-only (placeholder), preserving
            // the prior IR shape for the non-array path.
            #[cfg(feature = "std-surface")]
            if receiver_collection_sentinel(receiver, &ir, struct_env, receiver_types)
                == Some(ARRAY_VEC_SENTINEL)
            {
                let base = lower_expr(receiver, ir, env, struct_env, receiver_types);
                let index_id = lower_expr(index, ir, env, struct_env, receiver_types);
                let val_id = lower_expr(value, ir, env, struct_env, receiver_types);
                let dst = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst,
                    name: "vec_set".to_string(),
                    args: vec![base, index_id, val_id],
                });
                return dst;
            }
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        // RFC 0010 Phase A — `extern "C" { fn decls }` block.
        //
        // Emits one `Instr::ExternFnDecl` per declared symbol so the MLIR
        // lowerer knows to emit `llvm.func @name(...)` declarations and to
        // use `llvm.call` (not `func.call`) for calls to those names.
        //
        // Phase A lowers all integer/pointer parameter types to `i64` and
        // f64 to `f64`; raw pointer types lower to `i64` (opaque address
        // under the Option-C ABI convention already in use across the
        // std-surface runtime bridge).
        //
        // Gated to `std-surface` — default builds never construct this.
        #[cfg(feature = "std-surface")]
        ast::Node::ExternBlock { fns, callconv, .. } => {
            // RFC 0010 Phase B/C: use the repr_c_structs registry (populated by
            // any preceding StructDef nodes with `#[repr(C)]`) to emit correct
            // ABI-classified types for struct-valued parameters.
            // Phase C: dispatch to Win64 or SysV classifier based on callconv.
            let repr_c_snapshot = ir.repr_c_structs.clone();
            let effective_callconv = resolve_callconv(*callconv);
            for efn in fns {
                let param_types: Vec<String> = efn
                    .params
                    .iter()
                    .flat_map(|p| {
                        extern_type_to_mlir_multi_for(&p.ty, &repr_c_snapshot, effective_callconv)
                    })
                    .collect();
                let ret_type = efn.ret_type.as_ref().map(|t| {
                    // Return types: structs >8B returned via hidden pointer;
                    // use first ABI slot as the declared return type (single register).
                    extern_type_to_mlir_multi_for(t, &repr_c_snapshot, effective_callconv)
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| "i64".to_string())
                });
                ir.instrs.push(Instr::ExternFnDecl {
                    name: efn.name.clone(),
                    param_types,
                    ret_type,
                    is_varargs: efn.is_varargs,
                    vararg_hints: Vec::new(),
                    callconv: effective_callconv,
                });
            }
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        // RFC 0010 Phase J-A — `region { ... }` block lowering.
        //
        // Strategy:
        //   1. Lower the body statements into a scratch sub-IRModule so that
        //      alloc ids are collected in a fresh SSA namespace.
        //   2. Walk the sub-module's instructions to record every SSA id that
        //      was produced by a `__mind_alloc` call (region-interior allocs).
        //   3. Perform the escape check: if the body's result value (last SSA
        //      id) is in `alloc_ids`, emit a `safety::region_escape` diagnostic
        //      and continue (we don't abort — diagnostics are advisory at the
        //      IR level; the runtime will safely free the ptr at region exit).
        //   4. Emit `Instr::Region { body, result, alloc_ids }` into the
        //      parent IR. The MLIR backend emits the enter/track/exit calls.
        //
        // Gated to `std-surface`.
        #[cfg(feature = "std-surface")]
        ast::Node::Region { body, .. } => {
            let mut body_ir = sub_ir_from(ir);
            let mut body_env = env.clone();
            let mut alloc_ids: Vec<crate::ir::ValueId> = Vec::new();

            // Lower the body using the shared statement-sequence helper.
            // It handles Let / Assign / expression statements and appends
            // every __mind_alloc call id to `alloc_ids` so the runtime
            // track-calls and the type-checker escape check can both act on
            // the same information.
            let last_id = lower_stmt_seq(
                body,
                &mut body_ir,
                &mut body_env,
                struct_env,
                receiver_types,
                Some(&mut alloc_ids),
            );

            // Determine the result value (last expression in body).
            let result = last_id.unwrap_or_else(|| {
                let id = body_ir.fresh();
                body_ir.instrs.push(Instr::ConstI64(id, 0));
                id
            });

            // The escape check (safety::region_escape) is now performed at
            // the type-checker level (Node::Region arm in
            // check_module_types_in_file) so that it flows through the
            // structured diagnostic surface (--reporter json / lsp) and is
            // consistent with the Phase A/B pattern.  No eprintln here.

            // Advance the parent IR's next_id past everything allocated in
            // the body sub-module so all SSA ids remain globally unique.
            ir.next_id = body_ir.next_id;

            // Allocate unique SSA ids for the enter/exit call results.
            // These must be globally unique (from the parent IR's counter)
            // so that nested regions do not emit duplicate MLIR value names.
            let enter_id = ir.fresh();
            let exit_id = ir.fresh();

            ir.instrs.push(Instr::Region {
                body: body_ir.instrs,
                result,
                enter_id,
                exit_id,
                alloc_ids,
            });

            result
        }
        // RFC 0005 method brick — a `recv.method(args)` call. Two cases,
        // both keyed off the receiver's resolved struct type `T`:
        //
        //   1. ZERO-ARG FIELD ACCESSOR — a zero-arg method whose name matches
        //      a field of `T` is a field read (`s.len()` on a String == `s.len`).
        //      We emit the identical `__mind_load_i64(base + idx*8)` the
        //      `FieldAccess` arm emits. No new IR, no new ABI. (Already shipped.)
        //
        //   2. UFCS DESUGAR — any other resolved method (`v.push(x)`,
        //      `s.push_byte(b)`) is sugar for the free function
        //      `{lowercase(T)}_{method}(recv, args…)` that std declares by
        //      convention (struct `Vec` -> `vec_push`, struct `String` ->
        //      `string_push_byte`). We lower the receiver + each arg and emit a
        //      plain `Instr::Call` with the receiver threaded as the first
        //      argument — the SAME machinery the `Node::Call` arm uses for a
        //      direct free-function call. If the named function does not exist
        //      it fails LOUD at link time (an undefined `func.call` symbol), it
        //      never silently returns 0. This is purely additive desugar onto
        //      existing `Instr::Call`; no new IR/intrinsic/ABI.
        //
        // The receiver's struct type is resolved exactly like the `FieldAccess`
        // arm: Step 1 is the `struct_env` Ident fast-path, Step 2 is the
        // `receiver_types` side-table keyed on this MethodCall's span (populated
        // by the struct_resolver pre-pass). When the receiver type cannot be
        // resolved AT ALL, a method-with-args call would otherwise be a silent
        // const-0 miscompile — so we emit a clear diagnostic and a poison value
        // (panics in debug, returns a loud sentinel call in release) rather than
        // const-0. Method calls are absent from the keystone and all of std, so
        // the keystone artifact is unaffected.
        #[cfg(feature = "std-surface")]
        ast::Node::MethodCall {
            receiver,
            method,
            args,
            span,
        } => {
            // `c.byte()` — the byte (low 8 bits) of a char/int receiver, lowered
            // as `recv & 0xFF`. A char literal is its codepoint, so this extracts
            // the byte (identity for ASCII). Intercepted here because the receiver
            // has no struct type, so the UFCS path below cannot resolve it; the
            // type-check accepts `byte` as a 1-arg intrinsic. mind-flow lexer idiom.
            #[cfg(feature = "std-surface")]
            if method == "byte" && args.is_empty() {
                let recv_id = lower_expr(receiver, ir, env, struct_env, receiver_types);
                let mask = ir.fresh();
                ir.instrs.push(Instr::ConstI64(mask, 0xFF));
                let dst = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst,
                    op: BinOp::BitAnd,
                    lhs: recv_id,
                    rhs: mask,
                });
                return dst;
            }
            // `bytes[N].zero()` — a zeroed N-byte heap buffer (i64 handle). The
            // receiver parses as `IndexAccess(Ident("bytes"), N)`; lower it to
            // `__mind_calloc(N)`. mind-flow uses this for fixed-size hash buffers
            // (`bytes[32]` / `bytes[8]`).
            #[cfg(feature = "std-surface")]
            if method == "zero" && args.is_empty() {
                if let ast::Node::IndexAccess {
                    receiver: base,
                    index,
                    ..
                } = receiver.as_ref()
                {
                    if matches!(base.as_ref(), ast::Node::Lit(Literal::Ident(n), _) if n == "bytes")
                    {
                        if let Some(n) = extract_const_i64(index) {
                            let size = ir.fresh();
                            ir.instrs.push(Instr::ConstI64(size, n));
                            let dst = ir.fresh();
                            ir.instrs.push(Instr::Call {
                                dst,
                                name: "__mind_calloc".to_string(),
                                args: vec![size],
                            });
                            return dst;
                        }
                    }
                }
            }
            // `map<K, V>` methods on a map-sentinel receiver → std.map runtime.
            // The str-key sentinel routes lookups to the content-equality
            // variants (map_get_str / map_contains_key_str); the i64-key sentinel
            // uses identity. Intercepted here (not via UFCS) so `.length`→map_len
            // and the str-key split both resolve correctly.
            #[cfg(feature = "std-surface")]
            {
                let sentinel =
                    receiver_collection_sentinel(receiver, &ir, struct_env, receiver_types);
                if sentinel == Some(MAP_SENTINEL) || sentinel == Some(MAP_STR_SENTINEL) {
                    let is_str = sentinel == Some(MAP_STR_SENTINEL);
                    let fname = match method.as_str() {
                        "insert" => Some("map_insert"),
                        "get" => Some(if is_str { "map_get_str" } else { "map_get" }),
                        "contains_key" => Some(if is_str {
                            "map_contains_key_str"
                        } else {
                            "map_contains_key"
                        }),
                        "len" | "length" => Some("map_len"),
                        _ => None,
                    };
                    if let Some(fname) = fname {
                        let recv_id = lower_expr(receiver, ir, env, struct_env, receiver_types);
                        let mut call_args = vec![recv_id];
                        for a in args {
                            call_args.push(lower_expr(a, ir, env, struct_env, receiver_types));
                        }
                        let dst = ir.fresh();
                        ir.instrs.push(Instr::Call {
                            dst,
                            name: fname.to_string(),
                            args: call_args,
                        });
                        return dst;
                    }
                }
            }
            // `set<T>` methods on a set-sentinel receiver → std.map runtime (a set
            // is a map keyed by its elements). `.contains`/`.has` → map_contains_key
            // (_str for string elements); `.add`/`.insert` → map_insert(recv, x, 1)
            // (the unit value is synthesized); `.len` → map_len.
            #[cfg(feature = "std-surface")]
            {
                let sentinel =
                    receiver_collection_sentinel(receiver, &ir, struct_env, receiver_types);
                if sentinel == Some(SET_SENTINEL) || sentinel == Some(SET_STR_SENTINEL) {
                    let is_str = sentinel == Some(SET_STR_SENTINEL);
                    let recv_id = lower_expr(receiver, ir, env, struct_env, receiver_types);
                    match method.as_str() {
                        "contains" | "has" if args.len() == 1 => {
                            let x = lower_expr(&args[0], ir, env, struct_env, receiver_types);
                            let dst = ir.fresh();
                            ir.instrs.push(Instr::Call {
                                dst,
                                name: if is_str {
                                    "map_contains_key_str".to_string()
                                } else {
                                    "map_contains_key".to_string()
                                },
                                args: vec![recv_id, x],
                            });
                            return dst;
                        }
                        "add" | "insert" if args.len() == 1 => {
                            let x = lower_expr(&args[0], ir, env, struct_env, receiver_types);
                            let one = ir.fresh();
                            ir.instrs.push(Instr::ConstI64(one, 1));
                            let dst = ir.fresh();
                            ir.instrs.push(Instr::Call {
                                dst,
                                name: "map_insert".to_string(),
                                args: vec![recv_id, x, one],
                            });
                            return dst;
                        }
                        "len" | "length" if args.is_empty() => {
                            let dst = ir.fresh();
                            ir.instrs.push(Instr::Call {
                                dst,
                                name: "map_len".to_string(),
                                args: vec![recv_id],
                            });
                            return dst;
                        }
                        _ => {}
                    }
                }
            }
            // String methods on a `String` receiver (Ident or struct field) →
            // the `string_<method>` std free functions: `.split`→string_split,
            // `.trim`→string_trim, `.starts_with`→string_starts_with, etc. The
            // receiver is threaded as arg 0 (UFCS), but routed here so a
            // FIELD-typed string receiver (whose type the receiver_types side
            // table may not carry) also resolves.
            // STATIC/associated string fn: `string.from_utf8_bytes(buf)` — the
            // receiver is the bare TYPE name `string`/`String`, NOT a value, so it
            // routes to `string_<method>(args…)` with NO receiver arg (distinct
            // from the instance-method UFCS path below which threads the receiver
            // as arg 0). A local actually named `string`/`String` would be in
            // `struct_env`/`env`; the static path only fires when it is not.
            #[cfg(feature = "std-surface")]
            if let ast::Node::Lit(Literal::Ident(tn), _) = receiver.as_ref() {
                if (tn == "string" || tn == "String")
                    && !env.contains_key(tn)
                    && !struct_env.contains_key(tn)
                {
                    let mut call_args = Vec::with_capacity(args.len());
                    for a in args {
                        call_args.push(lower_expr(a, ir, env, struct_env, receiver_types));
                    }
                    let dst = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst,
                        name: format!("string_{method}"),
                        args: call_args,
                    });
                    return dst;
                }
            }
            #[cfg(feature = "std-surface")]
            if receiver_is_string(receiver, &ir, struct_env, receiver_types) {
                let recv_id = lower_expr(receiver, ir, env, struct_env, receiver_types);
                let mut call_args = vec![recv_id];
                for a in args {
                    call_args.push(lower_expr(a, ir, env, struct_env, receiver_types));
                }
                let dst = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst,
                    name: format!("string_{method}"),
                    args: call_args,
                });
                return dst;
            }
            // Resolve the receiver's struct type name `T` and, for the cheap
            // Ident path, the bound variable name so we can reuse its SSA id.
            let (struct_name, var_name_opt): (Option<String>, Option<String>) =
                match receiver.as_ref() {
                    ast::Node::Lit(Literal::Ident(var_name), _) => match struct_env.get(var_name) {
                        Some(t) => (Some(t.clone()), Some(var_name.clone())),
                        None => (receiver_types.get(span).cloned(), None),
                    },
                    // A FieldAccess receiver whose field is itself an `array<T>`
                    // (`next.scopes.push(x)` where `scopes: array<T>`) resolves to
                    // the vec sentinel so `.push`/`.get`/`.set`/`.length` route to
                    // the std.vec free functions — mirroring the map/set field
                    // receivers handled in the intercept above. The mutating forms
                    // are rebound to a FieldAssign in `rewrite_collection_mutations`
                    // so the fresh-on-realloc handle persists. Falls back to the
                    // struct-type side table for a struct-typed field receiver.
                    _ => {
                        #[cfg(feature = "std-surface")]
                        {
                            let sentinel = receiver_collection_sentinel(
                                receiver,
                                &ir,
                                struct_env,
                                receiver_types,
                            )
                            .map(|s| s.to_string());
                            (sentinel.or_else(|| receiver_types.get(span).cloned()), None)
                        }
                        #[cfg(not(feature = "std-surface"))]
                        {
                            (receiver_types.get(span).cloned(), None)
                        }
                    }
                };

            // Case 1 — zero-arg accessor whose name is a field of `T`.
            let field_idx = if args.is_empty() {
                struct_name.as_ref().and_then(|t| {
                    ir.struct_defs
                        .get(t)
                        .and_then(|fields| fields.iter().position(|f| f == method))
                })
            } else {
                None
            };

            if let Some(idx) = field_idx {
                let addr = match &var_name_opt {
                    Some(var_name) => match env.get(var_name) {
                        Some(id) => *id,
                        None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                    },
                    None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                };
                let field_addr = if idx == 0 {
                    addr
                } else {
                    let offset = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(offset, (idx as i64) * 8));
                    let sum = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: sum,
                        op: BinOp::Add,
                        lhs: addr,
                        rhs: offset,
                    });
                    sum
                };
                let result = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst: result,
                    name: "__mind_load_i64".to_string(),
                    args: vec![field_addr],
                });
                return result;
            }

            // Case 2 — UFCS desugar to `{lowercase(T)}_{method}(recv, args…)`.
            match &struct_name {
                Some(t) => {
                    let recv_id = match &var_name_opt {
                        Some(var_name) => match env.get(var_name) {
                            Some(id) => *id,
                            None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                        },
                        None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                    };
                    let mut call_args = vec![recv_id];
                    for a in args {
                        call_args.push(lower_expr(a, ir, env, struct_env, receiver_types));
                    }
                    // `array<T>` (the `vec` sentinel) method-name aliasing onto
                    // the std.vec free functions. The only surface/runtime name
                    // mismatch is `.length` → `vec_len` (mind-flow spells the
                    // length accessor `.length`; the runtime exports `vec_len`).
                    // `.push/.get/.set/.len` already match `vec_*` 1:1.
                    let method_lc: &str = if t == ARRAY_VEC_SENTINEL && method == "length" {
                        "len"
                    } else {
                        method.as_str()
                    };
                    let fn_name = format!("{}_{}", t.to_lowercase(), method_lc);
                    let dst = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst,
                        name: fn_name,
                        args: call_args,
                    });
                    dst
                }
                None => {
                    // Receiver type unresolved. A zero-arg unresolved call could
                    // be a never-defined accessor; a with-args unresolved call is
                    // a guaranteed silent miscompile under the old const-0
                    // fallthrough. Fail LOUD — never emit a silent const-0 at a
                    // method-with-args call site (#306 fail-closed philosophy).
                    if !args.is_empty() {
                        panic!(
                            "method call `{}.{}(...)` could not be resolved: the \
                             receiver's struct type is unknown, so it cannot desugar \
                             to a `<type>_{}` free function. Annotate the receiver's \
                             type or use the free-function form directly. (Emitting a \
                             const-0 placeholder here would be a silent miscompile.)",
                            describe_receiver(receiver),
                            method,
                            method,
                        );
                    }
                    // Zero-arg, unresolved type, not a known field — preserve the
                    // historical const-0 placeholder (it returns the receiver's
                    // identity for opaque accessors; unchanged behaviour).
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(id, 0));
                    id
                }
            }
        }
        // A `use`/import statement carries no runtime value — it is resolved at
        // module-load time. When it reaches `lower_expr` (e.g. a top-level
        // `use` routed through the module loop) emit the unit placeholder
        // EXPLICITLY — a documented compile-time no-op, not a mystery const-0
        // from the silent catch-all. Same bytes as the old catch-all path, so
        // emitted artifacts (and the keystone) are byte-identical.
        ast::Node::Import { .. } => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        // `print(...)` is NOT yet lowered in the compiled (codegen) path — the
        // side effect is dropped (KNOWN GAP, tracked #54; the tree-walking
        // interpreter DOES execute print). Emit the unit placeholder explicitly
        // rather than via the silent catch-all. Byte-identical to the old path.
        ast::Node::Print { .. } => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        // `assert cond, "msg"` (#203) — a STATEMENT-form runtime check that
        // lowers to a deterministic conditional trap. The frontend reaches it
        // through the module/block statement loop (which routes every statement
        // through `lower_expr`); without this arm it fell into the value-position
        // fail-closed panic below. It produces no meaningful value (it is a
        // statement), so the discarded merge id returned here is correct.
        //
        // Desugar to `if cond { } else { __mind_assert_fail(<msg-len>); }` and
        // reuse the existing, battle-tested `If` region-SSA lowering verbatim:
        //   * cond true  → empty then-branch, falls through (unit 0);
        //   * cond false → else-branch calls the runtime trap intrinsic, which
        //     `abort()`s deterministically (the same `abort()` path the region/
        //     genref runtime already relies on — see `runtime-support/
        //     mind_intrinsics.c`). `__mind_assert_fail` is auto-declared as a
        //     `func.func private @__mind_assert_fail(i64) -> i64` extern via the
        //     standard `extern_calls` collection, so no new IR variant, MLIR arm,
        //     or wire-format change is introduced — keystone bytes are untouched
        //     (no program in the bootstrap uses `assert`).
        //
        // The message is diagnostic-only; we pass its byte length as a single
        // deterministic i64 argument (no pointer bits, no float, identical across
        // substrates). The trap is unconditional once reached, so the exact
        // argument value does not affect control flow.
        #[cfg(feature = "std-surface")]
        ast::Node::Assert { cond, msg, span } => {
            let msg_len = msg.as_ref().map_or(0, |m| m.len()) as i64;
            let trap_call = ast::Node::Call {
                callee: "__mind_assert_fail".to_string(),
                args: vec![ast::Node::Lit(Literal::Int(msg_len), *span)],
                span: *span,
            };
            let if_node = ast::Node::If {
                cond: cond.clone(),
                then_branch: Vec::new(),
                else_branch: Some(vec![trap_call]),
                span: *span,
            };
            lower_expr(&if_node, ir, env, struct_env, receiver_types)
        }
        // A `struct`/`enum` type definition carries no runtime value — it is a
        // compile-time declaration collected in an earlier pass (see the
        // struct/enum item-collection arms above). When a top-level type def is
        // walked through the module statement loop into `lower_expr`, emit the
        // unit placeholder EXPLICITLY — a documented compile-time no-op, same as
        // the `use`/import arm. Same bytes as the old silent catch-all, so
        // emitted artifacts (and the keystone) stay byte-identical.
        ast::Node::StructDef { .. } | ast::Node::EnumDef { .. } => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        // `break` / `continue` — emit a loop-control marker carrying a snapshot
        // of every in-scope var → its CURRENT ValueId at this point. The
        // `Instr::While` MLIR arm forwards each loop-carried var's current value
        // (looked up by name) as the `^while_after` (break) / `^while_header`
        // (continue) block-arg. `env` here is the live body/branch env, so for a
        // break/continue nested inside an `if` we capture the correct
        // mid-iteration values. The snapshot is sorted by name so the IR (and
        // its mic@3 serialization / trace_hash) is deterministic. The cf.br
        // terminates the block, so the trailing const-0 "value" is unreachable.
        #[cfg(feature = "std-surface")]
        ast::Node::Break { .. } | ast::Node::Continue { .. } => {
            let mut live: Vec<(String, ValueId)> =
                env.iter().map(|(k, v)| (k.clone(), *v)).collect();
            live.sort_by(|a, b| a.0.cmp(&b.0));
            // Emit the (unreachable) unit value FIRST, then the terminator
            // marker, so the MLIR block ends on the `cf.br` — never a stray
            // instruction after the terminator.
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            match node {
                ast::Node::Break { .. } => ir.instrs.push(Instr::Break { live }),
                _ => ir.instrs.push(Instr::Continue { live }),
            }
            id
        }
        // `for VAR in START..END { BODY }` — desugar to the equivalent `while`
        // loop and reuse the existing, battle-tested `While` lowering above.
        // This is the mic@1 / value-position (`--emit-ir`) path; the desugared
        // shape is identical to the build path's loop semantics so the emitted
        // IR reflects exactly what is built (#4).
        //
        // The parser only accepts an exclusive range (`START..END`, see
        // `parse_for`) and the interpreter iterates `START..END` (exclusive),
        // so the desugared condition is `VAR < END` (never `<=`). The desugar:
        //
        //     let VAR = START;
        //     while VAR < END { BODY; VAR = VAR + 1; }
        //
        // The synthesized `VAR = VAR + 1` Assign makes `VAR` loop-carried, so
        // the `While` arm captures its pre-loop init (= START's ValueId) and
        // threads it through the header/back-edge exactly like a hand-written
        // counter. We do NOT emit a const-0 placeholder (the panic below guards
        // against that silent miscompile).
        #[cfg(feature = "std-surface")]
        ast::Node::For {
            var,
            start,
            end,
            body,
            span,
        } => {
            // `let VAR = START;` — lower START into the parent IR and bind VAR
            // so the synthesized `while` condition and body resolve it. The
            // While arm seeds its body/cond envs from this env, so VAR's
            // pre-loop init id is captured as the loop-carried init.
            let start_id = lower_expr(start, ir, env, struct_env, receiver_types);
            let mut loop_env = env.clone();
            loop_env.insert(var.clone(), start_id);

            // Build `VAR = VAR + 1` and append it to the loop body so the var
            // is detected as loop-carried (mirrors a hand-written `i = i + 1`).
            let ident_var = ast::Node::Lit(Literal::Ident(var.clone()), *span);
            let one = ast::Node::Lit(Literal::Int(1), *span);
            let incr = ast::Node::Assign {
                name: var.clone(),
                value: Box::new(ast::Node::Binary {
                    op: ast::BinOp::Add,
                    left: Box::new(ident_var.clone()),
                    right: Box::new(one),
                    span: *span,
                }),
                span: *span,
            };
            let mut while_body = body.clone();
            while_body.push(incr);

            // Condition `VAR < END`.
            let cond = ast::Node::Binary {
                op: ast::BinOp::Lt,
                left: Box::new(ident_var),
                right: end.clone(),
                span: *span,
            };

            let while_node = ast::Node::While {
                cond: Box::new(cond),
                body: while_body,
                span: *span,
            };

            // Lower the synthesized `while` with VAR in scope. Reuses the
            // `While` arm verbatim (cond/body sub-modules, loop-carried vars,
            // F2 region-scoped exit ids).
            lower_expr(&while_node, ir, &loop_env, struct_env, receiver_types)
        }
        // For-each `for x in coll { body }` over an `array<T>` (std.vec handle).
        // Flat-desugared to an indexed `while` so the loop-carried index gets the
        // same region-scoped SSA the `For`/`While` arms provide — no nested Block
        // (which would break loop-carried value detection). The collection and
        // its length are pre-lowered once into hidden span-unique bindings so a
        // NESTED for-each never collides; the hidden collection binding carries
        // the vec sentinel so `coll[idx]` lowers to `vec_get`.
        #[cfg(feature = "std-surface")]
        ast::Node::ForEach {
            var,
            collection,
            body,
            span,
        } => {
            let uniq = span.start();
            let coll_var = format!("__fe_coll_{uniq}");
            let idx_var = format!("__fe_i_{uniq}");
            let len_var = format!("__fe_len_{uniq}");

            // Pre-lower the collection (i64 vec handle) and its length once.
            let coll_id = lower_expr(collection, ir, env, struct_env, receiver_types);
            let len_id = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: len_id,
                name: "vec_len".to_string(),
                args: vec![coll_id],
            });
            let zero_id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(zero_id, 0));

            let mut loop_env = env.clone();
            loop_env.insert(coll_var.clone(), coll_id);
            loop_env.insert(len_var.clone(), len_id);
            loop_env.insert(idx_var.clone(), zero_id);

            // Mark the hidden collection binding as a vec sentinel so the
            // `coll[idx]` element read lowers to `vec_get`.
            let mut fe_struct_env = struct_env.clone();
            fe_struct_env.insert(coll_var.clone(), ARRAY_VEC_SENTINEL.to_string());
            // Track the element var's type (struct / String / nested collection)
            // so methods on it resolve — `for d in flow.decorators` makes `d` a
            // Decorator, `for part in s.split(...)` makes `part` a String.
            #[cfg(feature = "std-surface")]
            if let Some(elem) =
                foreach_element_sentinel(collection, &ir, struct_env, receiver_types)
            {
                fe_struct_env.insert(var.clone(), elem);
            }

            let idx_ident = ast::Node::Lit(Literal::Ident(idx_var.clone()), *span);
            // `let VAR = coll[idx]` — the per-iteration element binding.
            let elem_bind = ast::Node::Let {
                name: var.clone(),
                mutable: false,
                ann: None,
                value: Box::new(ast::Node::IndexAccess {
                    receiver: Box::new(ast::Node::Lit(Literal::Ident(coll_var.clone()), *span)),
                    index: Box::new(idx_ident.clone()),
                    span: *span,
                }),
                span: *span,
            };
            // `idx = idx + 1` — makes idx loop-carried (mirrors the For arm).
            let incr = ast::Node::Assign {
                name: idx_var.clone(),
                value: Box::new(ast::Node::Binary {
                    op: ast::BinOp::Add,
                    left: Box::new(idx_ident.clone()),
                    right: Box::new(ast::Node::Lit(Literal::Int(1), *span)),
                    span: *span,
                }),
                span: *span,
            };
            let mut while_body = Vec::with_capacity(body.len() + 2);
            while_body.push(elem_bind);
            while_body.extend(body.iter().cloned());
            while_body.push(incr);

            // Condition `idx < len`.
            let cond = ast::Node::Binary {
                op: ast::BinOp::Lt,
                left: Box::new(idx_ident),
                right: Box::new(ast::Node::Lit(Literal::Ident(len_var.clone()), *span)),
                span: *span,
            };
            let while_node = ast::Node::While {
                cond: Box::new(cond),
                body: while_body,
                span: *span,
            };
            lower_expr(&while_node, ir, &loop_env, &fe_struct_env, receiver_types)
        }
        // A `const NAME = value` DECLARATION is a no-op at the value level — the
        // value is inlined at each `Lit(Ident(NAME))` use site (see the
        // `module_const_value` read path above), exactly as `StructDef`/`EnumDef`
        // declarations emit a unit placeholder. (The top-level `array`-const arm
        // keeps its dedicated `ConstArray` path; this catches scalar / string /
        // collection consts and any const reaching a block position.)
        ast::Node::Const { .. } => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        // Genuinely-unhandled value-position node. After the explicit Import /
        // Print arms above, std/ + examples/ no longer reach here (verified by
        // the #54 sweep). FAIL CLOSED (#306 philosophy): a const-0 placeholder
        // here would be a release-silent miscompile, and a future `ast::Node`
        // variant added without a lowering arm must surface loudly — never as a
        // wrong runtime value.
        node => {
            let dbg = format!("{node:?}");
            let kind = dbg.split(['(', ' ', '{']).next().unwrap_or("<node>");
            panic!(
                "lower_expr: no IR lowering for `{kind}` in value position — \
                 refusing to emit a const-0 placeholder (that would be a silent \
                 miscompile). Add an explicit lowering arm or handle it upstream."
            );
        }
    }
}

/// Best-effort textual description of a method-call receiver, used only to
/// build a clear diagnostic when a `recv.method(args)` call cannot resolve its
/// receiver's struct type (so it cannot UFCS-desugar to a `<type>_method` free
/// function). Falls back to a generic placeholder for non-Ident receivers.
#[cfg(feature = "std-surface")]
fn describe_receiver(receiver: &ast::Node) -> String {
    match receiver {
        ast::Node::Lit(Literal::Ident(name), _) => name.clone(),
        _ => "<expr>".to_string(),
    }
}

/// Collect all SSA ids produced by `__mind_alloc` calls in `instrs` into
/// `out`. Called after lowering each region body statement so that alloc
/// sites introduced by nested calls (vec_new, struct literals, etc.) are
/// also recorded.
///
/// Only looks one level deep — the Phase J-A escape check is conservative
/// (flags direct-return of an alloc result; aliasing through fields is
/// Phase J-B).
#[cfg(feature = "std-surface")]
fn collect_alloc_ids(instrs: &[Instr], out: &mut Vec<crate::ir::ValueId>) {
    for instr in instrs {
        if let Instr::Call { dst, name, .. } = instr {
            if name == "__mind_alloc" && !out.contains(dst) {
                out.push(*dst);
            }
        }
    }
}

/// Wrap a constructor payload `value` in the to-bits coercion when its declared
/// field type does not natively fit the i64 record slot, so the stored value is
/// always i64. v1 handles `f64` (a same-width `arith.bitcast` via
/// `__mind_f64_to_bits`); an i64 field passes through unchanged, and any other
/// non-i64 field is left unwrapped and fails loud at the i64 store (honest,
/// no silent miscompile).
#[cfg(feature = "std-surface")]
fn coerce_enum_field_to_bits(
    value: ast::Node,
    ty: Option<&ast::TypeAnn>,
    span: ast::Span,
) -> ast::Node {
    match ty {
        Some(ast::TypeAnn::ScalarF64) => ast::Node::Call {
            callee: "__mind_f64_to_bits".to_string(),
            args: vec![value],
            span,
        },
        _ => value,
    }
}

/// Inverse of [`coerce_enum_field_to_bits`]: wrap the i64 slot `load` in the
/// from-bits coercion when the field's declared type is non-i64, so a match
/// binding has the declared type. v1 handles `f64` (`__mind_bits_to_f64`).
#[cfg(feature = "std-surface")]
fn coerce_enum_field_from_bits(
    load: ast::Node,
    ty: Option<&ast::TypeAnn>,
    span: ast::Span,
) -> ast::Node {
    match ty {
        Some(ast::TypeAnn::ScalarF64) => ast::Node::Call {
            callee: "__mind_bits_to_f64".to_string(),
            args: vec![load],
            span,
        },
        _ => load,
    }
}

/// Emit the uniform boxed-enum heap record `[tag @ +0, field0 @ +8, field1 @
/// +16, …]` (`total_slots` i64 slots) and return its address. Used by BOTH the
/// payload-carrying constructor (`Opt::Some(v)`, `Pair::P(a, b)`) and the
/// fieldless constructor of a boxed enum (`Opt::None`, no fields), so every
/// variant of a boxed enum has the IDENTICAL record SIZE (`total_slots` = the
/// enum's `1 + max payload arity`) and a `match` can always read the tag from
/// `+0` and any of the widest variant's fields from `+8*(i+1)`. Slots past the
/// supplied payloads are zero-filled (a narrower variant's unused fields), so
/// the record is fully initialised. The `__mind_alloc` + `__mind_store_i64`
/// sequence mirrors the StructLit heap-record build — no new `Instr`/intrinsic.
#[cfg(feature = "std-surface")]
fn emit_boxed_enum_record(
    ir: &mut IRModule,
    tag: i64,
    payloads: &[ValueId],
    total_slots: usize,
) -> ValueId {
    // bytes = 8 * total_slots.
    let eight = ir.fresh();
    ir.instrs.push(Instr::ConstI64(eight, 8));
    let count = ir.fresh();
    ir.instrs.push(Instr::ConstI64(count, total_slots as i64));
    let bytes = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst: bytes,
        op: BinOp::Mul,
        lhs: eight,
        rhs: count,
    });
    // addr = __mind_alloc(bytes)
    let addr = ir.fresh();
    ir.instrs.push(Instr::Call {
        dst: addr,
        name: "__mind_alloc".to_string(),
        args: vec![bytes],
    });
    // __mind_store_i64(addr + 0, tag)
    let tag_id = ir.fresh();
    ir.instrs.push(Instr::ConstI64(tag_id, tag));
    let store_tag = ir.fresh();
    ir.instrs.push(Instr::Call {
        dst: store_tag,
        name: "__mind_store_i64".to_string(),
        args: vec![addr, tag_id],
    });
    // Store each field at `addr + 8*(i+1)`, then zero-fill the remaining slots
    // (so a narrower variant's unused fields are deterministically 0).
    for slot in 1..total_slots {
        let value = if slot - 1 < payloads.len() {
            payloads[slot - 1]
        } else {
            let z = ir.fresh();
            ir.instrs.push(Instr::ConstI64(z, 0));
            z
        };
        let offset = ir.fresh();
        ir.instrs.push(Instr::ConstI64(offset, (slot * 8) as i64));
        let field_addr = ir.fresh();
        ir.instrs.push(Instr::BinOp {
            dst: field_addr,
            op: BinOp::Add,
            lhs: addr,
            rhs: offset,
        });
        let store = ir.fresh();
        ir.instrs.push(Instr::Call {
            dst: store,
            name: "__mind_store_i64".to_string(),
            args: vec![field_addr, value],
        });
    }
    addr
}

/// "finish MIND" Step 1/2: desugar a `match` expression into a right-nested
/// chain of `ast::Node::If` so the existing branching `If` lowering executes
/// the match (instead of evaluating every arm unconditionally).
///
/// Supported: integer/bool arms (`Pattern::Literal(Literal::Int)`, which is
/// also how `true`/`false` patterns parse) and — Step 2 — FIELDLESS
/// enum-variant arms (`Pattern::EnumVariant` with no payload), which compare
/// the scrutinee against the variant's ordinal discriminant tag looked up in
/// `enum_tags`. Both forms may be followed by a single terminal catch-all
/// (`Wildcard` or bare `Ident`); an `Ident` catch-all binds the scrutinee
/// under that name before its arm body via a synthetic `let`.
///
/// Returns `None` for any unsupported pattern kind (payload-binding variants
/// such as `Some(v)`, string/float literals, or an enum-variant path absent
/// from `enum_tags`) so the caller can fall back to the prior behaviour; those
/// are handled by later steps.
///
/// The result is a pure AST rewrite: it constructs standard `Node::If` /
/// `Node::Binary(Eq)` nodes and is lowered through the unchanged
/// `ast::Node::If` arm, so none of the dominance/merge machinery is touched.
#[cfg(feature = "std-surface")]
fn desugar_match_to_if(
    scrutinee: &ast::Node,
    arms: &[ast::MatchArm],
    enum_tags: &std::collections::BTreeMap<String, i64>,
    boxed_enums: &std::collections::BTreeSet<String>,
    payload_types: &std::collections::BTreeMap<String, Vec<ast::TypeAnn>>,
    struct_field_names: &std::collections::BTreeMap<String, Vec<String>>,
) -> Option<ast::Node> {
    let span = scrutinee.span();
    // Disambiguate BARE variant patterns (`Foo(v)`, `Bar`) to a single owning
    // enum BEFORE the per-arm normalisation. All variant arms of one match
    // belong to the same enum, so a bare name must resolve within THAT enum —
    // the prior code resolved each bare arm INDEPENDENTLY to the first
    // lexicographic `Enum::V` in the registry, which silently picks the WRONG
    // enum (wrong tag → wrong arm) whenever a variant name collides across enums
    // (e.g. `Foo` in both `Alpha` and `Zeta` — a `Zeta` scrutinee then tested
    // `Foo` against `Alpha::Foo`'s tag). Determine one owning enum:
    //   1. If ANY arm is qualified (`Zeta::Bar`), that enum owns the match; every
    //      bare arm resolves within it.
    //   2. Otherwise pick the UNIQUE enum whose variant set contains every bare
    //      variant name. If 0 or ≥2 enums qualify, leave `owning_enum = None`
    //      and `resolve_bare` falls back to the prior first-match resolution
    //      (deterministic, and correct for the non-colliding Option/Result-style
    //      programs where the user enum sorts before the builtin registry copy).
    let bare_variant_names: Vec<&str> = arms
        .iter()
        .filter_map(|a| match &a.pattern {
            ast::Pattern::EnumVariant { path, .. } if !path.contains("::") => Some(path.as_str()),
            _ => None,
        })
        .collect();
    let owning_enum: Option<String> = if bare_variant_names.is_empty() {
        None
    } else {
        // (1) A qualified arm pins the enum.
        let qualified_enum: Option<String> = arms.iter().find_map(|a| match &a.pattern {
            ast::Pattern::EnumVariant { path, .. } => path
                .rsplit_once("::")
                .map(|(e, _)| e.to_string()),
            ast::Pattern::EnumStruct { path, .. } => path
                .rsplit_once("::")
                .or_else(|| path.rsplit_once('.'))
                .map(|(e, _)| e.to_string()),
            _ => None,
        });
        if let Some(e) = qualified_enum {
            Some(e)
        } else {
            // (2) Candidate enums = those declaring EVERY bare variant name.
            let mut candidates: std::collections::BTreeSet<String> = enum_tags
                .keys()
                .filter_map(|k| k.rsplit_once("::"))
                .filter(|(_, v)| bare_variant_names.contains(v))
                .map(|(e, _)| e.to_string())
                .collect();
            candidates.retain(|e| {
                bare_variant_names
                    .iter()
                    .all(|v| enum_tags.contains_key(&format!("{e}::{v}")))
            });
            // Exactly one enum owns all the bare names → unambiguous.
            if candidates.len() == 1 {
                candidates.into_iter().next()
            } else {
                None
            }
        }
    };
    // Resolve a bare variant name. When an owning enum was determined, anchor to
    // it (the collision fix). Otherwise reproduce the prior deterministic
    // first-in-BTreeMap resolution so non-colliding programs are byte-identical.
    let resolve_bare = |path: &str| -> String {
        if path.contains("::") || enum_tags.contains_key(path) {
            return path.to_string();
        }
        if let Some(e) = &owning_enum {
            let qualified = format!("{e}::{path}");
            if enum_tags.contains_key(&qualified) {
                return qualified;
            }
        }
        enum_tags
            .keys()
            .find(|k| k.rsplit_once("::").map(|(_, v)| v == path).unwrap_or(false))
            .cloned()
            .unwrap_or_else(|| path.to_string())
    };
    // Normalise any STRUCT-variant pattern `E.V { f, g }` into the equivalent
    // POSITIONAL variant pattern `E::V(<f-slot>, <g-slot>, …)` using the enum's
    // declared `field_names`, so the rest of the desugar (tag compare + slot
    // binding) reuses the tuple-variant machinery verbatim. A field omitted in
    // the pattern binds a `Wildcard` for its slot; the path is normalised
    // dot→`::` to match the tag registry. (enum_match #9 struct variants.)
    let mut arms_owned: Vec<ast::MatchArm> = Vec::with_capacity(arms.len());
    for arm in arms {
        let converted = match &arm.pattern {
            ast::Pattern::EnumStruct { path, fields } => {
                let key = match path.rsplit_once('.') {
                    Some((a, b)) => {
                        let dotted = format!("{a}::{b}");
                        if enum_tags.contains_key(&dotted) {
                            dotted
                        } else {
                            path.clone()
                        }
                    }
                    None => path.clone(),
                };
                // A named `{ … }` pattern is ONLY valid on a struct variant —
                // i.e. a variant with declared field_names. If the variant has
                // none (it is a unit/tuple variant, or the path is unknown),
                // resolving names to slots is impossible; bail the WHOLE match to
                // the fail-loud fallback rather than binding by source order,
                // which would silently mis-map `{ y, x }` (adversarial-review fix).
                let order = struct_field_names.get(&key)?;
                let args: Vec<ast::Pattern> = order
                    .iter()
                    .map(|fname| {
                        fields
                            .iter()
                            .find(|(n, _)| n == fname)
                            .map(|(_, p)| p.clone())
                            .unwrap_or(ast::Pattern::Wildcard)
                    })
                    .collect();
                ast::MatchArm {
                    pattern: ast::Pattern::EnumVariant { path: key, args },
                    body: arm.body.clone(),
                    span: arm.span,
                }
            }
            // Resolve a BARE variant pattern (`Some(v)`, `Ok(e)`, `Err(e)`) to
            // its qualified `Enum::V` so the tag compare below finds it — the
            // pattern-side mirror of the bare-constructor resolution in the
            // `Node::Call` arm. Resolution is anchored to the match's OWNING enum
            // (computed above) so a variant name shared by two enums resolves to
            // the scrutinee's enum, never the first lexicographic registry hit.
            // An unknown bare name is left as-is.
            ast::Pattern::EnumVariant { path, args } if !path.contains("::") => {
                let key = if enum_tags.contains_key(path) {
                    path.clone()
                } else {
                    resolve_bare(path)
                };
                ast::MatchArm {
                    pattern: ast::Pattern::EnumVariant {
                        path: key,
                        args: args.clone(),
                    },
                    body: arm.body.clone(),
                    span: arm.span,
                }
            }
            // A fieldless enum variant written with DOT notation
            // (`MindTypeKind.Scalar`) parses as a bare `Ident` pattern (no `::`,
            // no args). Resolve it to its `Enum::Variant` form when the dotted
            // path is a known tag so it becomes a discriminant TEST arm rather
            // than being misread as a catch-all binding. A plain ident that is
            // NOT a known variant (a real binding/catch-all) is left untouched.
            ast::Pattern::Ident(name) if name.contains('.') => {
                let dotted = name.replacen('.', "::", 1);
                if enum_tags.contains_key(&dotted) {
                    ast::MatchArm {
                        pattern: ast::Pattern::EnumVariant {
                            path: dotted,
                            args: Vec::new(),
                        },
                        body: arm.body.clone(),
                        span: arm.span,
                    }
                } else {
                    arm.clone()
                }
            }
            _ => arm.clone(),
        };
        arms_owned.push(converted);
    }
    let arms = &arms_owned[..];
    // An arm body written with braces (`1 => { x = 100 }`) parses to a single
    // `Node::Block` wrapping the arm's statements, whereas a parsed `if { … }`
    // produces a FLAT `Vec<Node>` of statements. The If lowering only treats
    // top-level `Assign`/`Let` nodes as branch writes that merge into the
    // post-`if` scope (its `then_writes`/`else_writes` → `branch_bindings`);
    // a `Block` falls through to the expression arm and its enclosing-scope
    // mutations are dropped at the merge. So splice a braced body's statements
    // directly into the branch list to mirror the parsed-`if` shape exactly.
    let flatten_body = |body: ast::Node| -> Vec<ast::Node> {
        match body {
            ast::Node::Block { stmts, .. } => stmts,
            other => vec![other],
        }
    };
    // Split the arms into the leading test arms and the terminal else.
    // Only the FINAL arm may be a catch-all (`_` / bare ident); a catch-all
    // in a non-final position would shadow the rest — leave such (malformed)
    // matches to the fallback path.
    let mut else_branch: Option<Vec<ast::Node>> = None;
    let mut last_idx = arms.len();
    if let Some(last) = arms.last() {
        match &last.pattern {
            ast::Pattern::Wildcard => {
                else_branch = Some(flatten_body(last.body.clone()));
                last_idx = arms.len() - 1;
            }
            ast::Pattern::Ident(name) => {
                // Bind the scrutinee under `name` before the arm body.
                let bind = ast::Node::Let {
                    name: name.clone(),
                    mutable: false,
                    ann: None,
                    value: Box::new(scrutinee.clone()),
                    span,
                };
                let mut stmts = vec![bind];
                stmts.extend(flatten_body(last.body.clone()));
                else_branch = Some(stmts);
                last_idx = arms.len() - 1;
            }
            _ => {}
        }
    }

    // "finish MIND" Step 5 — does this match scrutinise a PAYLOAD-carrying
    // enum value? If ANY arm binds a payload (`Some(v)` — a non-empty
    // `EnumVariant.args`), the scrutinee is a 2-field heap record
    // `[tag @ +0, payload @ +8]` (built by the enum-constructor path in the
    // `Node::Call` arm), NOT a bare discriminant. In that case every
    // enum-variant comparison must test the LOADED tag
    // (`__mind_load_i64(scrutinee + 0)`) rather than the scrutinee value
    // itself, and a payload-binding arm prepends a synthetic
    // `let <ident> = __mind_load_i64(scrutinee + 8)` so the payload binds —
    // exactly the synthetic-let shape the terminal `Ident` catch-all above
    // already uses. Pure fieldless (C-like) enums keep comparing the bare
    // scrutinee value, so this never perturbs Step-2 fieldless matches.
    // The scrutinee is a boxed heap record when EITHER an arm binds a payload
    // (`Some(v)`) OR the matched enum is in `boxed_enums` (so even a match that
    // names ONLY fieldless variants of a boxed enum — e.g. `Res::Err`/`Res::Ok`
    // with no binding — loads the tag from the record rather than comparing the
    // record POINTER against an ordinal, which would never match). A purely
    // fieldless enum is not boxed, so its match still compares the bare value.
    let scrutinee_carries_payload = arms.iter().any(|a| match &a.pattern {
        ast::Pattern::EnumVariant { args, .. } if !args.is_empty() => true,
        ast::Pattern::EnumVariant { path, .. } => path
            .rsplit_once("::")
            .is_some_and(|(e, _)| boxed_enums.contains(e)),
        _ => false,
    });

    // Build the comparison LHS node: for a payload-carrying scrutinee this is
    // `__mind_load_i64(scrutinee + 0)` (mirrors the FieldAccess +0 fast path,
    // which passes the base address with no offset); otherwise the bare
    // scrutinee.
    let load_i64 = |addr: ast::Node| ast::Node::Call {
        callee: "__mind_load_i64".to_string(),
        args: vec![addr],
        span,
    };
    let cmp_lhs: ast::Node = if scrutinee_carries_payload {
        load_i64(scrutinee.clone())
    } else {
        scrutinee.clone()
    };

    // Each remaining arm becomes a `<cmp_lhs> == <rhs>` comparison. Build the
    // RHS literal node per pattern kind: an integer/bool literal compares
    // against itself; an enum variant (fieldless OR payload-carrying)
    // compares against its ordinal discriminant tag. Any other pattern
    // (unknown variant path, non-int literal) bails the whole match to the
    // fallback.
    let test_arms = &arms[..last_idx];
    let mut rhs_nodes: Vec<ast::Node> = Vec::with_capacity(test_arms.len());
    for arm in test_arms {
        let rhs = match &arm.pattern {
            ast::Pattern::Literal(Literal::Int(_)) => {
                let lit = match &arm.pattern {
                    ast::Pattern::Literal(l) => l.clone(),
                    _ => unreachable!(),
                };
                ast::Node::Lit(lit, span)
            }
            // Step 2/5: enum-discriminant arm. Fieldless variants compare the
            // bare scrutinee against the tag; payload variants compare the
            // loaded tag and bind their payload (handled below when the arm
            // body is assembled).
            ast::Pattern::EnumVariant { path, .. } => {
                let tag = enum_tags.get(path).copied()?;
                ast::Node::Lit(Literal::Int(tag), span)
            }
            _ => return None,
        };
        rhs_nodes.push(rhs);
    }
    // Need at least one test arm to form a branch; a lone catch-all is
    // already handled fine by the fallback (and has no comparison to make).
    if test_arms.is_empty() {
        return None;
    }

    // Build the chain from the tail backwards so the first arm ends up
    // outermost. `else_stmts` is the body of the current innermost `else`:
    // the terminal catch-all arm initially, then each enclosing `If`.
    let mut else_stmts: Option<Vec<ast::Node>> = else_branch;
    for (arm, rhs) in test_arms.iter().zip(rhs_nodes.iter()).rev() {
        let cond = ast::Node::Binary {
            op: ast::BinOp::Eq,
            left: Box::new(cmp_lhs.clone()),
            right: Box::new(rhs.clone()),
            span,
        };
        // Step 5: for a payload-binding arm, PREPEND a synthetic
        // `let <name> = __mind_load_i64(scrutinee + 8*(i+1))` for each `Ident`
        // sub-pattern (so `Pair::P(a, b)` binds `a` from `+8` and `b` from
        // `+16`); a `Wildcard` sub-pattern binds nothing (the tag comparison
        // already discriminates). Field offsets are POSITIONAL — `i` is the
        // sub-pattern's index — so a `_` does not shift later fields. A nested /
        // literal sub-pattern is unsupported: it bails the whole match to `None`
        // and `check_match_runnable` turns that into a loud fail-closed error on
        // the emit path (never a silent sequential miscompile).
        let then_branch: Vec<ast::Node> = match &arm.pattern {
            ast::Pattern::EnumVariant { path, args } if !args.is_empty() => {
                // Each payload sub-pattern must be an `Ident`, a `Wildcard`, or a
                // single-level `Tuple` of Idents/Wildcards (`Ok((a, b))`). A
                // nested struct/literal/deeper-tuple sub-pattern bails the whole
                // match to `None` (loud fail-closed downstream), never a silent
                // sequential miscompile.
                let sub_ok = |p: &ast::Pattern| {
                    matches!(p, ast::Pattern::Ident(_) | ast::Pattern::Wildcard)
                        || matches!(p, ast::Pattern::Tuple(inner)
                            if inner.iter().all(|q| matches!(q,
                                ast::Pattern::Ident(_) | ast::Pattern::Wildcard)))
                };
                if !args.iter().all(sub_ok) {
                    return None;
                }
                let field_types = payload_types.get(path);
                let mut stmts = Vec::new();
                for (i, sub) in args.iter().enumerate() {
                    // field_addr = scrutinee + 8*(i+1)
                    let offset = ast::Node::Lit(Literal::Int(((i + 1) * 8) as i64), span);
                    let field_addr = ast::Node::Binary {
                        op: ast::BinOp::Add,
                        left: Box::new(scrutinee.clone()),
                        right: Box::new(offset),
                        span,
                    };
                    match sub {
                        ast::Pattern::Ident(name) => {
                            // Reinterpret the i64 slot back to the field's declared
                            // type (e.g. `f64`) so the binding has the right type.
                            let ty = field_types.and_then(|ts| ts.get(i));
                            let value =
                                coerce_enum_field_from_bits(load_i64(field_addr), ty, span);
                            stmts.push(ast::Node::Let {
                                name: name.clone(),
                                mutable: false,
                                ann: None,
                                value: Box::new(value),
                                span,
                            });
                        }
                        // `Ok((a, b))` — the payload slot holds a handle to a
                        // tuple aggregate ([elem@+0, elem@+8, …], all-i64). Bind
                        // the handle to a hidden let, then load each element. The
                        // hidden name is span+index-unique so sibling arms/nested
                        // matches never collide.
                        ast::Pattern::Tuple(inner) => {
                            let tp_name =
                                format!("__mind_tp_{}_{}", span.start(), i);
                            stmts.push(ast::Node::Let {
                                name: tp_name.clone(),
                                mutable: false,
                                ann: None,
                                value: Box::new(load_i64(field_addr)),
                                span,
                            });
                            for (j, inner_p) in inner.iter().enumerate() {
                                if let ast::Pattern::Ident(name) = inner_p {
                                    let base = ast::Node::Lit(
                                        Literal::Ident(tp_name.clone()),
                                        span,
                                    );
                                    let elem_addr = if j == 0 {
                                        base
                                    } else {
                                        ast::Node::Binary {
                                            op: ast::BinOp::Add,
                                            left: Box::new(base),
                                            right: Box::new(ast::Node::Lit(
                                                Literal::Int((j * 8) as i64),
                                                span,
                                            )),
                                            span,
                                        }
                                    };
                                    stmts.push(ast::Node::Let {
                                        name: name.clone(),
                                        mutable: false,
                                        ann: None,
                                        value: Box::new(load_i64(elem_addr)),
                                        span,
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }
                stmts.extend(flatten_body(arm.body.clone()));
                stmts
            }
            _ => flatten_body(arm.body.clone()),
        };
        let if_node = ast::Node::If {
            cond: Box::new(cond),
            then_branch,
            else_branch: else_stmts.take(),
            span,
        };
        else_stmts = Some(vec![if_node]);
    }
    // `else_stmts` now holds the single outermost `If` (test_arms is
    // non-empty, so exactly one node).
    else_stmts.and_then(|mut v| v.pop())
}

/// Lower a sequence of `Let` / `Assign` / expression body statements into
/// `ir`, updating `env` with new name→id bindings.
///
/// Returns the `ValueId` of the last statement, or `None` when `stmts` is
/// empty.  Callers that need a unit value (region, fn body) synthesise a
/// `ConstI64(0)` when `None` is returned.
///
/// When `alloc_ids` is `Some`, every `__mind_alloc` call id produced during
/// lowering is appended to it (used by `Node::Region` to build the escape-
/// check set passed to `Instr::Region`).
///
/// This is the shared body-lowering core used by both `Node::Region` and
/// `Node::FnDef`.  The `FnDef` arm adds `Return`-statement handling and
/// `std-surface`-gated struct-env / branch-binding tracking on top.
#[cfg(feature = "std-surface")]
fn lower_stmt_seq(
    stmts: &[ast::Node],
    ir: &mut IRModule,
    env: &mut HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
    mut alloc_ids: Option<&mut Vec<ValueId>>,
) -> Option<ValueId> {
    let mut last_id: Option<ValueId> = None;
    for stmt in stmts {
        let id = match stmt {
            ast::Node::Let {
                name, ann, value, ..
            } => {
                let id = match ann {
                    Some(TypeAnn::Tensor { dtype, dims })
                    | Some(TypeAnn::DiffTensor { dtype, dims }) => lower_tensor_binding(
                        ir,
                        value,
                        dtype,
                        dims,
                        env,
                        struct_env,
                        receiver_types,
                    ),
                    _ => lower_expr(value, ir, env, struct_env, receiver_types),
                };
                // Narrow-typed Region-local: mask to declared width AND record it
                // so a later `c = c + …` reassignment in this region re-masks.
                #[cfg(feature = "std-surface")]
                let id = mask_narrow_let(ir, ann, id);
                #[cfg(feature = "std-surface")]
                record_narrow_let(name, ann);
                env.insert(name.clone(), id);
                id
            }
            ast::Node::LetTuple { names, value, .. } => {
                // Lower the RHS to the tuple's base pointer, then bind each name
                // to `__mind_load_i64(addr + 8*i)` — the read side of the
                // `Node::Tuple` aggregate above (all-i64 layout, 8-byte slots).
                let addr = lower_expr(value, ir, env, struct_env, receiver_types);
                let mut last = addr;
                for (i, nm) in names.iter().enumerate() {
                    let elem_addr = if i == 0 {
                        addr
                    } else {
                        let offset = ir.fresh();
                        ir.instrs.push(Instr::ConstI64(offset, (i as i64) * 8));
                        let sum = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: sum,
                            op: BinOp::Add,
                            lhs: addr,
                            rhs: offset,
                        });
                        sum
                    };
                    let loaded = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst: loaded,
                        name: "__mind_load_i64".to_string(),
                        args: vec![elem_addr],
                    });
                    env.insert(nm.clone(), loaded);
                    last = loaded;
                }
                last
            }
            ast::Node::Assign { name, value, .. } => {
                let id = lower_expr(value, ir, env, struct_env, receiver_types);
                // Reassigning a narrow Region-local re-masks to its declared width.
                #[cfg(feature = "std-surface")]
                let id = mask_narrow_assign(ir, name, id);
                env.insert(name.clone(), id);
                id
            }
            other => lower_expr(other, ir, env, struct_env, receiver_types),
        };
        if let Some(ref mut out) = alloc_ids {
            collect_alloc_ids(&ir.instrs, out);
        }
        last_id = Some(id);
    }
    last_id
}

/// Lower a tuple-destructuring `let (a, b, …) = expr` as a *statement* into a
/// mutable env: lower the RHS to the tuple base pointer, then bind each name to
/// `__mind_load_i64(addr + 8*i)` (all-i64, 8-byte slots — the read side of the
/// `Node::Tuple` aggregate). Used by the if-then/else and while-body block loops
/// where statements are otherwise routed through value-position `lower_expr`,
/// which cannot mutate the caller's binding env.
fn lower_lettuple_stmt(
    names: &[String],
    value: &ast::Node,
    ir: &mut IRModule,
    env: &mut HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> ValueId {
    let addr = lower_expr(value, ir, env, struct_env, receiver_types);
    let mut last = addr;
    for (i, nm) in names.iter().enumerate() {
        let elem_addr = if i == 0 {
            addr
        } else {
            let offset = ir.fresh();
            ir.instrs.push(Instr::ConstI64(offset, (i as i64) * 8));
            let sum = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: sum,
                op: BinOp::Add,
                lhs: addr,
                rhs: offset,
            });
            sum
        };
        let loaded = ir.fresh();
        ir.instrs.push(Instr::Call {
            dst: loaded,
            name: "__mind_load_i64".to_string(),
            args: vec![elem_addr],
        });
        env.insert(nm.clone(), loaded);
        last = loaded;
    }
    last
}

/// Extract a compile-time i64 value from a literal expression node.
/// Returns `None` for non-literal (runtime) expressions.
fn extract_const_i64(node: &ast::Node) -> Option<i64> {
    match node {
        ast::Node::Lit(Literal::Int(n), _) => Some(*n),
        ast::Node::Neg { operand, .. } => extract_const_i64(operand).map(|v| -v),
        _ => None,
    }
}

/// Extract a constant numeric element as f64 (int literals widen to float),
/// honouring a leading unary `-`. Used to materialise dense tensor literals.
fn extract_const_f64(node: &ast::Node) -> Option<f64> {
    match node {
        ast::Node::Lit(Literal::Float(f), _) => Some(*f),
        ast::Node::Lit(Literal::Int(n), _) => Some(*n as f64),
        ast::Node::Neg { operand, .. } => extract_const_f64(operand).map(|v| -v),
        _ => None,
    }
}

/// Encode one dense-tensor-literal element to the `u64` bit pattern stored in
/// `Instr::ConstDenseTensor` (must mirror `render_dense_elem` in the MLIR
/// backend): f32 → its u32 IEEE bits, f64 → all 64 bits, i32 → two's-complement
/// low-32, i64 as-is, Q16.16 → `round(v * 2^16)` as i32. Preserves the EXACT
/// element bits (the old i64 ConstArray path coerced floats to 0).
fn dense_elem_bits(node: &ast::Node, dtype: &DType) -> u64 {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => {
            (extract_const_f64(node).unwrap_or(0.0) as f32).to_bits() as u64
        }
        DType::F64 => extract_const_f64(node).unwrap_or(0.0).to_bits(),
        DType::I32 => ((extract_const_i64(node).unwrap_or(0) as i32) as u32) as u64,
        DType::Q16 => {
            let q = (extract_const_f64(node).unwrap_or(0.0) * 65536.0).round() as i32;
            (q as u32) as u64
        }
        DType::I64 => extract_const_i64(node).unwrap_or(0) as u64,
    }
}

/// Extract the element value list from an `ArrayLit` node iteratively.
/// Non-literal elements default to 0.  Returns an empty Vec for non-ArrayLit RHS.
#[cfg(feature = "std-surface")]
fn extract_array_lit_values(node: &ast::Node) -> Vec<i64> {
    match node {
        ast::Node::ArrayLit { elements, .. } => {
            let mut out = Vec::with_capacity(elements.len());
            for elem in elements {
                out.push(extract_const_i64(elem).unwrap_or(0));
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Create a sub-IRModule for branch/body lowering that inherits the metadata
/// tables from `parent` (struct schemas, const-array data) and starts its SSA
/// counter at `parent.next_id`.
///
/// Starting at the parent's current `next_id` ensures that every ValueId
/// allocated inside the sub-module is disjoint from all ValueIds already
/// visible in the enclosing function scope (parameters, outer lets, etc.).
/// Without this, a constant emitted in a condition sub-IR (e.g.
/// `ConstI64(ValueId(0), 32)`) would collide with the function's first
/// parameter (`%0: i64`) when both are serialised into the same MLIR block.
///
/// Used by `Instr::If` lowering and any future control-flow arms that lower
/// branches into separate scratch IRModules.
///
/// Best-effort: does SSA value `id`, defined within branch `instrs`, carry an
/// `f64`? Consulted ONLY to TYPE the absent-branch zero placeholder of a
/// one-sided merge (a `let`/assign present in one branch, absent in the other),
/// so an `f64` one-sided binding gets an `f64` placeholder and the merge phi
/// types `f64` instead of clashing with the i64 default. ADDITIVE: an i64 value
/// returns `false` → the placeholder stays `ConstI64(0)` exactly as before, so
/// every all-i64 program is byte-identical. A conservative `false` is always
/// SSA-safe for an i64 value; the only value it can mistype is a genuinely-f64
/// one-sided binding, which is already broken (the bug this fixes).
#[cfg(feature = "std-surface")]
fn branch_value_is_f64(
    instrs: &[Instr],
    id: ValueId,
    fn_signatures: &std::collections::BTreeMap<String, (Vec<ast::TypeAnn>, Option<ast::TypeAnn>)>,
) -> bool {
    for instr in instrs.iter().rev() {
        match instr {
            Instr::ConstF64(d, _) if *d == id => return true,
            Instr::ConstI64(d, _) if *d == id => return false,
            Instr::Call { dst, name, .. } if *dst == id => {
                if name == "__mind_bits_to_f64" {
                    return true;
                }
                return matches!(
                    fn_signatures.get(name).and_then(|(_, r)| r.as_ref()),
                    Some(ast::TypeAnn::ScalarF64) | Some(ast::TypeAnn::ScalarF32)
                );
            }
            Instr::BinOp { dst, lhs, rhs, .. } if *dst == id => {
                return branch_value_is_f64(instrs, *lhs, fn_signatures)
                    || branch_value_is_f64(instrs, *rhs, fn_signatures);
            }
            // A nested value-`if` produces its merge outputs (`dst` + each
            // `merge_id`) into THIS scope, but their defining values live in the
            // nested then/else instruction lists. Recurse into the matching
            // column so an f64 merged up through an inner `if` (a desugared match
            // arm) is recognised — otherwise a one-sided let whose defined side
            // is such a merge would mistype as i64.
            Instr::If {
                dst,
                then_result,
                else_result,
                merges,
                then_instrs,
                else_instrs,
                ..
            } => {
                if *dst == id {
                    return branch_value_is_f64(then_instrs, *then_result, fn_signatures)
                        || branch_value_is_f64(else_instrs, *else_result, fn_signatures);
                }
                if let Some((_, tv, ev)) = merges.iter().find(|(m, _, _)| *m == id) {
                    return branch_value_is_f64(then_instrs, *tv, fn_signatures)
                        || branch_value_is_f64(else_instrs, *ev, fn_signatures);
                }
            }
            _ => {}
        }
    }
    false
}

#[cfg(feature = "std-surface")]
fn sub_ir_from(parent: &IRModule) -> IRModule {
    let mut m = IRModule::new();
    m.next_id = parent.next_id;
    m.struct_defs = parent.struct_defs.clone();
    m.const_array_defs = parent.const_array_defs.clone();
    m.enum_variant_tags = parent.enum_variant_tags.clone();
    // Boxed-enum metadata MUST be inherited too: a variant constructed inside a
    // control-flow body (`if cond { e = E.V { .. } }`) needs `boxed_enums` +
    // `enum_payload_slots` to size the uniform heap record, `enum_payload_types`
    // for f64-slot coercion, and `enum_struct_field_names` to even RECOGNISE a
    // struct-variant `StructLit` (without it the ctor falls through to a plain
    // struct with no tag, and the enclosing match never matches). Found by
    // adversarial review — happy-path tests only constructed at fn-body top level.
    m.boxed_enums = parent.boxed_enums.clone();
    m.enum_payload_slots = parent.enum_payload_slots.clone();
    m.enum_payload_types = parent.enum_payload_types.clone();
    m.enum_struct_field_names = parent.enum_struct_field_names.clone();
    m.struct_field_types = parent.struct_field_types.clone();
    m
}

/// Like `sub_ir_from`, but chains the SSA counter from `prev` (the previously
/// built sub-module) so that each successive sub-module's ids are disjoint from
/// all predecessors.  Metadata is still copied from `meta_src` (the original
/// parent scope).
#[cfg(feature = "std-surface")]
fn sub_ir_from_after(prev: &IRModule, meta_src: &IRModule) -> IRModule {
    let mut m = IRModule::new();
    m.next_id = prev.next_id;
    m.struct_defs = meta_src.struct_defs.clone();
    m.const_array_defs = meta_src.const_array_defs.clone();
    m.enum_variant_tags = meta_src.enum_variant_tags.clone();
    // Inherit the boxed-enum metadata for variants constructed in this scope —
    // see the note in `sub_ir_from`.
    m.boxed_enums = meta_src.boxed_enums.clone();
    m.enum_payload_slots = meta_src.enum_payload_slots.clone();
    m.enum_payload_types = meta_src.enum_payload_types.clone();
    m.enum_struct_field_names = meta_src.enum_struct_field_names.clone();
    m.struct_field_types = meta_src.struct_field_types.clone();
    m
}

/// F2: extract the variable rebindings produced by a nested control-flow
/// instruction so they can be threaded into the enclosing region's env at
/// EVERY nesting level (fn-body, while-body, then-branch, else-branch).
///
/// Returns `(name, exit_id)` pairs where `exit_id` is a DOMINATING value for
/// the enclosing region — the loop's `^while_after` exit block-arg id
/// (`While.exit_ids`) or the if's `^if_after` merge block-arg id (the value in
/// `If.branch_bindings`, which the F2 If lowering sets to the merge id).
///
/// This is the recursion crux: when an if-branch or loop-body contains a nested
/// loop/if that mutates an outer variable, the enclosing region picks up the
/// nested region's EXIT id (not a raw value defined inside the deeper branch),
/// so any value later passed on a branch/back edge is guaranteed to dominate.
#[cfg(feature = "std-surface")]
fn region_exit_rebindings(instr: &Instr) -> Vec<(String, ValueId)> {
    match instr {
        Instr::While {
            live_vars,
            exit_ids,
            ..
        } => live_vars
            .iter()
            .enumerate()
            .map(|(k, (name, post))| {
                let exit = exit_ids.get(k).copied().unwrap_or(*post);
                (name.clone(), exit)
            })
            .collect(),
        Instr::If {
            branch_bindings, ..
        } => branch_bindings.clone(),
        _ => Vec::new(),
    }
}

/// F2: the rebindings produced by the LAST control-flow statement pushed into
/// `instrs`. A `While` arm pushes the `Instr::While` followed by a trailing
/// unit `ConstI64`, so a bare last-instruction check would miss it — we look
/// past that trailing placeholder. An `If` arm pushes only the `Instr::If`
/// (its `dst` is the value), so the last instruction is the node itself.
#[cfg(feature = "std-surface")]
fn last_region_exit_rebindings(instrs: &[Instr]) -> Vec<(String, ValueId)> {
    let n = instrs.len();
    if n == 0 {
        return Vec::new();
    }
    // Direct match (If, or While with no trailing placeholder).
    let direct = region_exit_rebindings(&instrs[n - 1]);
    if !direct.is_empty() {
        return direct;
    }
    // While pushes a trailing unit ConstI64 after the loop node; the loop is
    // the second-to-last instruction.
    if n >= 2 {
        if let Instr::ConstI64(..) = &instrs[n - 1] {
            return region_exit_rebindings(&instrs[n - 2]);
        }
    }
    Vec::new()
}

fn parse_tensor_ann(dtype: &str, dims: &[String]) -> Option<(DType, Vec<ShapeDim>)> {
    let dtype = dtype.parse().ok()?;
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        shape.push(parse_dim(dim));
    }
    Some((dtype, shape))
}

fn parse_dim(dim: &str) -> ShapeDim {
    if let Ok(n) = dim.parse::<usize>() {
        ShapeDim::Known(n)
    } else {
        ShapeDim::Sym(crate::types::intern::intern_str(dim))
    }
}

/// RFC 0010 Phase A/B — map an `extern "C"` parameter/return `TypeAnn` to
/// the MLIR type string(s) used in `llvm.func` declarations and `llvm.call`
/// ops.
///
/// Returns a `Vec<String>` of MLIR type tokens because a single MIND type
/// can expand to multiple MLIR types under the SysV x86_64 ABI (e.g. a
/// 16-byte all-integer `#[repr(C)]` struct expands to two `i64` parameters).
/// For all non-struct types the Vec always has exactly one element.
///
/// `repr_c` is the `repr_c_structs` registry from `IRModule` — a map from
/// struct name to field types. Pass an empty map when no struct types are
/// expected (Phase A callers).
#[cfg(feature = "std-surface")]
pub(crate) fn extern_type_to_mlir_multi(
    ty: &crate::ast::TypeAnn,
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> Vec<String> {
    use crate::ast::TypeAnn;
    match ty {
        TypeAnn::ScalarF32 => vec!["f32".to_string()],
        TypeAnn::ScalarF64 => vec!["f64".to_string()],
        TypeAnn::RawPtr { .. } => vec!["!llvm.ptr".to_string()],
        // RFC 0010 Phase B: callback function pointer -> opaque !llvm.ptr.
        TypeAnn::FnPtr { .. } => vec!["!llvm.ptr".to_string()],
        TypeAnn::Named(name) => {
            match name.as_str() {
                "f32" => return vec!["f32".to_string()],
                "f64" => return vec!["f64".to_string()],
                "i8" | "i16" | "i32" | "u8" | "u16" | "u32" | "bool" => {
                    return vec!["i64".to_string()];
                }
                "i64" | "u64" | "usize" | "isize" => return vec!["i64".to_string()],
                _ => {}
            }
            // Check for repr(C) struct — apply SysV classification.
            if let Some(fields) = repr_c.get(name.as_str()) {
                sysv_classify_struct(fields, repr_c)
            } else {
                vec!["i64".to_string()]
            }
        }
        // Built-in scalar integer/bool types all lower to i64.
        TypeAnn::ScalarI32 | TypeAnn::ScalarI64 | TypeAnn::ScalarBool | TypeAnn::ScalarU32 => {
            vec!["i64".to_string()]
        }
        // Fallback: any aggregate that slipped past the type-checker becomes i64.
        _ => vec!["i64".to_string()],
    }
}

/// RFC 0010 Phase A compatibility shim — single-type version of
/// `extern_type_to_mlir_multi`. Used by Phase A callers. Struct types are
/// passed by pointer (!llvm.ptr) when the registry is empty.
#[cfg(feature = "std-surface")]
#[allow(dead_code)]
pub(crate) fn extern_type_to_mlir(ty: &crate::ast::TypeAnn) -> String {
    let empty = std::collections::BTreeMap::new();
    extern_type_to_mlir_multi(ty, &empty)
        .into_iter()
        .next()
        .unwrap_or_else(|| "i64".to_string())
}

/// RFC 0010 Phase B — System V AMD64 ABI struct field classification.
/// Classifies a scalar type into Integer or Float class and returns its byte size.
/// Returns `(None, 0)` for types that cannot be classified (nested aggregates, etc.).
#[cfg(feature = "std-surface")]
pub fn classify_scalar_field(
    ty: &crate::ast::TypeAnn,
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> (Option<SysVClass>, usize) {
    use crate::ast::TypeAnn;
    match ty {
        TypeAnn::ScalarI32 | TypeAnn::ScalarBool | TypeAnn::ScalarU32 => {
            (Some(SysVClass::Integer), 4)
        }
        TypeAnn::ScalarI64 => (Some(SysVClass::Integer), 8),
        TypeAnn::ScalarF32 => (Some(SysVClass::Float), 4),
        TypeAnn::ScalarF64 => (Some(SysVClass::Float), 8),
        TypeAnn::RawPtr { .. } | TypeAnn::FnPtr { .. } => (Some(SysVClass::Integer), 8),
        TypeAnn::Named(name) => match name.as_str() {
            "i8" | "u8" => (Some(SysVClass::Integer), 1),
            "i16" | "u16" => (Some(SysVClass::Integer), 2),
            "i32" | "u32" | "bool" => (Some(SysVClass::Integer), 4),
            "i64" | "u64" | "usize" | "isize" => (Some(SysVClass::Integer), 8),
            "f32" => (Some(SysVClass::Float), 4),
            "f64" => (Some(SysVClass::Float), 8),
            _other => {
                // Nested repr(C) struct or unknown — Phase B defers to MEMORY class.
                let _ = repr_c; // used in future phases
                (None, 0)
            }
        },
        _ => (None, 0),
    }
}

/// RFC 0010 Phase B — SysV AMD64 struct parameter class.
/// Used by `sysv_classify_struct` and exposed for tests.
#[cfg(feature = "std-surface")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SysVClass {
    /// Integer/pointer fields — passed in general-purpose registers.
    Integer,
    /// Floating-point fields — passed in XMM registers.
    Float,
    /// Aggregate too large or mixed — passed via pointer (caller allocates).
    Memory,
}

/// RFC 0010 Phase B — SysV AMD64 ABI struct-passing classification.
///
/// Given the field types of a `#[repr(C)]` struct (up to 4 fields, all Copy),
/// returns the list of MLIR type strings that represent how the struct is
/// passed in a function call under the SysV AMD64 ABI:
///
/// - All-integer/pointer, total ≤ 8 B → `["i64"]` (one eightbyte)
/// - All-integer/pointer, total ≤ 16 B → `["i64", "i64"]` (two eightbytes)
/// - All-float, total ≤ 8 B → single float type
/// - All-float, total ≤ 16 B → two float types
/// - Mixed int+float or > 16 B → `["!llvm.ptr"]` (MEMORY class)
///
/// This is a pure function with no I/O; it is `pub` so Phase B tests can
/// invoke it directly to verify the classification logic.
#[cfg(feature = "std-surface")]
pub fn sysv_classify_struct(
    fields: &[crate::ast::TypeAnn],
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> Vec<String> {
    if fields.is_empty() {
        return vec!["i64".to_string()];
    }
    if fields.len() > 4 {
        return vec!["!llvm.ptr".to_string()];
    }

    let mut total_bytes: usize = 0;
    let mut classes: Vec<SysVClass> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();

    for field_ty in fields {
        match classify_scalar_field(field_ty, repr_c) {
            (Some(cls), sz) => {
                total_bytes += sz;
                classes.push(cls);
                sizes.push(sz);
            }
            (None, _) => return vec!["!llvm.ptr".to_string()],
        }
    }

    if total_bytes > 16 {
        return vec!["!llvm.ptr".to_string()];
    }

    let all_integer = classes.iter().all(|c| *c == SysVClass::Integer);
    let all_float = classes.iter().all(|c| *c == SysVClass::Float);

    if all_integer {
        if total_bytes <= 8 {
            vec!["i64".to_string()]
        } else {
            vec!["i64".to_string(), "i64".to_string()]
        }
    } else if all_float {
        // Determine the dominant float type per eightbyte slot (0..8, 8..16).
        // f64 (8 bytes) dominates f32 (4 bytes) within a slot.
        // Walk fields in order, tracking byte offset to assign each to a slot.
        let mut slot0_has_f64 = false;
        let mut slot1_has_f64 = false;
        let mut byte_off: usize = 0;
        for &sz in &sizes {
            if byte_off < 8 {
                if sz == 8 {
                    slot0_has_f64 = true;
                }
            } else {
                if sz == 8 {
                    slot1_has_f64 = true;
                }
            }
            byte_off += sz;
        }
        let first_slot = if slot0_has_f64 { "f64" } else { "f32" };
        if total_bytes <= 8 {
            vec![first_slot.to_string()]
        } else {
            let second_slot = if slot1_has_f64 { "f64" } else { "f32" };
            vec![first_slot.to_string(), second_slot.to_string()]
        }
    } else {
        // Mixed integer + float -> MEMORY class.
        vec!["!llvm.ptr".to_string()]
    }
}

/// RFC 0010 Phase C — Win64 struct parameter class.
/// Used by `win64_classify_struct` and exposed for tests.
#[cfg(feature = "std-surface")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Win64Class {
    /// Struct fits in one general-purpose register (size ∈ {1, 2, 4, 8}).
    Register,
    /// Struct passed by pointer (caller-allocated; size not in {1,2,4,8}).
    Memory,
}

/// RFC 0010 Phase C — Microsoft x64 ABI struct-passing classification.
///
/// Microsoft x64 ABI §4 (struct/union passing rules):
/// - Structs of size exactly 1, 2, 4, or 8 bytes: passed by value in one
///   general-purpose register as the matching integer type (i8, i16, i32, i64).
/// - All other sizes: passed by pointer (caller allocates on the stack).
///   Sizes 3, 5, 6, 7 technically "round up" but since there is no canonical
///   way to represent a 3-byte integer in LLVM IR, we classify them as MEMORY
///   (the caller passes a pointer to the aligned copy, which is the safe and
///   correct ABI implementation).
///
/// This function returns the `Vec<String>` of MLIR type tokens, matching the
/// calling convention of `sysv_classify_struct`.
#[cfg(feature = "std-surface")]
pub fn win64_classify_struct(
    fields: &[crate::ast::TypeAnn],
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> Vec<String> {
    if fields.is_empty() {
        return vec!["i64".to_string()];
    }

    // Compute total byte size using the same scalar classifier as SysV.
    let mut total_bytes: usize = 0;
    for field_ty in fields {
        match classify_scalar_field(field_ty, repr_c) {
            (Some(_), sz) => total_bytes += sz,
            (None, _) => return vec!["!llvm.ptr".to_string()],
        }
    }

    // Win64: pass by value only for sizes {1, 2, 4, 8}.
    match total_bytes {
        1 => vec!["i8".to_string()],
        2 => vec!["i16".to_string()],
        4 => vec!["i32".to_string()],
        8 => vec!["i64".to_string()],
        _ => vec!["!llvm.ptr".to_string()],
    }
}

/// RFC 0010 Phase C — resolve `CallConv::C` to the platform-default ABI.
///
/// On Linux/macOS x86_64: resolves to `CallConv::SysV`.
/// On Windows x86_64: resolves to `CallConv::Win64`.
/// `CallConv::Aapcs` is passed through (Phase D will handle it; Phase C
/// callers fall back to SysV for the MLIR emission).
#[cfg(feature = "std-surface")]
pub(crate) fn resolve_callconv(cc: crate::ast::CallConv) -> crate::ast::CallConv {
    use crate::ast::CallConv;
    match cc {
        CallConv::C => {
            if cfg!(target_os = "windows") {
                CallConv::Win64
            } else {
                CallConv::SysV
            }
        }
        other => other,
    }
}

/// RFC 0010 Phase C — ABI-aware type classifier dispatcher.
///
/// Routes to `extern_type_to_mlir_multi` (SysV) or
/// `extern_type_to_mlir_multi_win64` (Win64) based on the resolved callconv.
/// `CallConv::Aapcs` is not yet implemented (Phase D); it falls back to SysV
/// with a runtime note so callers can test the dispatch path today.
#[cfg(feature = "std-surface")]
pub(crate) fn extern_type_to_mlir_multi_for(
    ty: &crate::ast::TypeAnn,
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
    callconv: crate::ast::CallConv,
) -> Vec<String> {
    use crate::ast::CallConv;
    match callconv {
        CallConv::Win64 => extern_type_to_mlir_multi_win64(ty, repr_c),
        CallConv::SysV | CallConv::C => extern_type_to_mlir_multi(ty, repr_c),
        CallConv::Aapcs => {
            // Phase D deferred. Fall back to SysV for now.
            extern_type_to_mlir_multi(ty, repr_c)
        }
    }
}

/// RFC 0010 Phase C — Win64 variant of `extern_type_to_mlir_multi`.
///
/// Maps a MIND `TypeAnn` to the MLIR LLVM type string(s) using the
/// Microsoft x64 ABI struct-passing rules instead of SysV.
///
/// For non-struct types the result is identical to `extern_type_to_mlir_multi`
/// (scalars, pointers, function pointers all have the same representation
/// under both ABIs on x86_64). The difference appears only for `#[repr(C)]`
/// struct types: Win64 passes them by value when they are exactly {1,2,4,8}
/// bytes, and by pointer otherwise.
#[cfg(feature = "std-surface")]
pub fn extern_type_to_mlir_multi_win64(
    ty: &crate::ast::TypeAnn,
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> Vec<String> {
    use crate::ast::TypeAnn;
    match ty {
        TypeAnn::ScalarF32 => vec!["f32".to_string()],
        TypeAnn::ScalarF64 => vec!["f64".to_string()],
        TypeAnn::RawPtr { .. } => vec!["!llvm.ptr".to_string()],
        TypeAnn::FnPtr { .. } => vec!["!llvm.ptr".to_string()],
        TypeAnn::Named(name) => {
            match name.as_str() {
                "f32" => return vec!["f32".to_string()],
                "f64" => return vec!["f64".to_string()],
                "i8" | "u8" => return vec!["i8".to_string()],
                "i16" | "u16" => return vec!["i16".to_string()],
                "i32" | "u32" | "bool" => return vec!["i32".to_string()],
                "i64" | "u64" | "usize" | "isize" => return vec!["i64".to_string()],
                _ => {}
            }
            // Check for repr(C) struct — apply Win64 classification.
            if let Some(fields) = repr_c.get(name.as_str()) {
                win64_classify_struct(fields, repr_c)
            } else {
                vec!["i64".to_string()]
            }
        }
        TypeAnn::ScalarI32 | TypeAnn::ScalarBool | TypeAnn::ScalarU32 => {
            vec!["i32".to_string()]
        }
        TypeAnn::ScalarI64 => vec!["i64".to_string()],
        _ => vec!["i64".to_string()],
    }
}
