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

//! Trait / interface desugaring (#268 Phase 1) — the byte-identity-safe,
//! STATIC-dispatch form.
//!
//! A `trait T { fn m(self, ...) -> R }` + `impl T for Foo { fn m(self, ...) { .. } }`
//! is rewritten, BEFORE type-check and IR emit, into ordinary constructs that
//! already round-trip through the mic@3 codec:
//!
//! - each `impl T for Foo` method becomes a top-level free `fn` named by the
//!   EXISTING UFCS convention `{lowercase(Foo)}_{method}` — the SAME free
//!   function a `foo.method(args)` call already resolves to in the `MethodCall`
//!   lowering arm (its `self` receiver threaded as arg 0), so static dispatch on
//!   a statically-known concrete receiver falls out of the existing machinery
//!   with NO new IR, NO wire change; and
//! - the `trait` declaration is DROPPED after it is used to check that every
//!   `impl` covers its method set.
//!
//! There is NO vtable, NO `dyn`, NO trait object, NO fn-pointer — only
//! fns/structs/calls reach `emit_mic3`. A module with no trait/impl is left
//! byte-identical (early no-op — no clone, no allocation), so the keystone
//! `main.mind` (zero traits) and all of `std/` are untouched.
//!
//! # Phase 1 support boundary (fail-closed outside it)
//! - method call `x.m(args)` where `x`'s CONCRETE type is statically known
//!   resolves to the impl method — static dispatch only;
//! - a generic bound `fn use<X: T>(x: X)`, associated types, default methods,
//!   and MULTI-trait method-name overlap on one type are Phase 2, rejected here.
//!
//! # Fail-closed diagnostics (E27xx) — the #1 MIND risk is a silent miscompile
//! - `impl` of an UNDECLARED trait;
//! - `impl` method NOT declared in the trait;
//! - `impl` MISSING a trait method;
//! - DUPLICATE `impl T for Foo`;
//! - two impls providing the SAME method name for the SAME type (multi-trait
//!   ambiguity — Phase 2);
//! - a synthesized `{lowercase(Foo)}_{method}` name COLLIDING with an existing
//!   top-level free fn;
//! - a call to a TRAIT method on a statically-known type with NO matching impl
//!   (and no matching free fn) — the fail-closed witness, rejected rather than
//!   folding to a silent const-0.
//!
//! // deferred: generic trait bounds + associated types + default methods +
//! // multi-trait overlap + self-host `main.mind` mirror — upgrade path: #268
//! // Phase 2 (monomorphize a `fn use<X: T>` per concrete X; mangle
//! // `__impl_{Type}_{Trait}_{method}` to disambiguate overlapping trait methods).

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{FnDefData, Literal, Module, Node, Param, Span, TypeAnn};
use crate::diagnostics::{Diagnostic, Span as DiagSpan};

/// Desugar every trait/impl in `module` in place: lift each impl method to a
/// top-level free fn and drop the trait/impl declarations. Returns any
/// fail-closed diagnostics. A module with no trait/impl is left byte-identical
/// (early no-op).
pub fn desugar_traits(module: &mut Module, source: &str, file: Option<&str>) -> Vec<Diagnostic> {
    let has_any = module
        .items
        .iter()
        .any(|it| matches!(it, Node::TraitDef { .. } | Node::ImplBlock { .. }));
    if !has_any {
        return Vec::new();
    }

    let mut diags: Vec<Diagnostic> = Vec::new();

    // ── 1. Collect trait declarations (name -> method-name set). ────────────
    // `BTreeMap`/`BTreeSet` keep every subsequent iteration in a fixed order —
    // the byte-identity discipline (never a HashMap walk in a code path that
    // emits items).
    let mut traits: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // method-name -> the (sole, Phase-1) trait that declares it, for the
    // call-site fail-closed check.
    let mut method_to_trait: BTreeMap<String, String> = BTreeMap::new();
    for item in &module.items {
        if let Node::TraitDef {
            name,
            methods,
            span,
            ..
        } = item
        {
            if traits.contains_key(name) {
                diags.push(err(
                    source,
                    file,
                    *span,
                    "E2700",
                    format!("duplicate trait declaration `{name}`"),
                    "Each trait may be declared at most once.",
                ));
                continue;
            }
            let mut mset: BTreeSet<String> = BTreeSet::new();
            for m in methods {
                mset.insert(m.name.clone());
                // Multi-trait overlap of a method NAME is deferred (Phase 2): the
                // `{lowercase(type)}_{method}` mangling cannot disambiguate two
                // traits that both declare `m` for the same type. Record only the
                // first; the impl-coverage + call-site checks stay sound because a
                // second impl providing the same (type, method) is rejected below.
                method_to_trait
                    .entry(m.name.clone())
                    .or_insert_with(|| name.clone());
            }
            traits.insert(name.clone(), mset);
        }
    }

    // ── 2. Collect + validate impls; synthesize free fns. ───────────────────
    // (type_name, method) actually implemented — used by the call-site check.
    let mut impls: BTreeSet<(String, String)> = BTreeSet::new();
    // (trait, type) seen — duplicate-impl guard.
    let mut impl_pairs: BTreeSet<(String, String)> = BTreeSet::new();
    // Existing top-level free-fn names — collision guard for synthesized names,
    // and the call-site "a matching free fn already exists" escape hatch.
    let mut existing_fns: BTreeSet<String> = BTreeSet::new();
    for item in &module.items {
        if let Node::FnDef(fd, _) = item {
            existing_fns.insert(fd.name.clone());
        }
    }

    let mut synthesized: Vec<Node> = Vec::new();
    for item in &module.items {
        let Node::ImplBlock {
            trait_name,
            type_name,
            methods,
            span,
        } = item
        else {
            continue;
        };

        let Some(trait_methods) = traits.get(trait_name) else {
            diags.push(err(
                source,
                file,
                *span,
                "E2701",
                format!("`impl {trait_name} for {type_name}`: trait `{trait_name}` is not declared"),
                "Declare the trait with `trait Name { fn m(self, ...) -> R }` before implementing it.",
            ));
            continue;
        };

        if !impl_pairs.insert((trait_name.clone(), type_name.clone())) {
            diags.push(err(
                source,
                file,
                *span,
                "E2702",
                format!("duplicate `impl {trait_name} for {type_name}`"),
                "A trait may be implemented at most once for a given type.",
            ));
            continue;
        }

        // Which trait methods this impl provides.
        let mut provided: BTreeSet<String> = BTreeSet::new();
        for m in methods {
            let Node::FnDef(fd, m_span) = m else {
                continue;
            };
            let m_span = *m_span;
            if !trait_methods.contains(&fd.name) {
                diags.push(err(
                    source,
                    file,
                    m_span,
                    "E2703",
                    format!(
                        "method `{}` is not declared in trait `{trait_name}`",
                        fd.name
                    ),
                    "An impl may only define methods that the trait declares.",
                ));
                continue;
            }
            if !provided.insert(fd.name.clone()) {
                diags.push(err(
                    source,
                    file,
                    m_span,
                    "E2704",
                    format!("method `{}` implemented twice in this impl block", fd.name),
                    "Define each trait method exactly once per impl.",
                ));
                continue;
            }

            // Synthesized free-fn name = the EXISTING UFCS convention a
            // `receiver.method(args)` call already lowers to.
            let fn_name = format!("{}_{}", type_name.to_lowercase(), fd.name);

            // Multi-trait ambiguity (Phase 2): another impl already produced this
            // exact (type, method). The `{type}_{method}` name cannot carry the
            // trait, so reject rather than emit two colliding fns.
            if impls.contains(&(type_name.clone(), fd.name.clone())) {
                diags.push(err(
                    source,
                    file,
                    m_span,
                    "E2705",
                    format!(
                        "method `{}` is implemented for type `{type_name}` by more than one trait",
                        fd.name
                    ),
                    "Multi-trait method-name overlap on one type is not supported in Phase 1.",
                ));
                continue;
            }
            // Collision with a hand-written free fn of the same mangled name.
            if existing_fns.contains(&fn_name) {
                diags.push(err(
                    source,
                    file,
                    m_span,
                    "E2706",
                    format!(
                        "impl method `{}.{}` would synthesize free fn `{fn_name}`, which already exists",
                        type_name, fd.name
                    ),
                    "Rename the conflicting free function, or remove it and rely on the impl.",
                ));
                continue;
            }

            impls.insert((type_name.clone(), fd.name.clone()));

            // Lift to a top-level free fn: `fn {type}_{method}(self: Type, ...) { body }`.
            // The parsed `self` param already carries `Named(type_name)`, so
            // field access `self.x` resolves through the normal struct-param
            // path in lowering — no new machinery.
            let lifted = FnDefData {
                is_pub: fd.is_pub,
                is_test: false,
                name: fn_name.clone(),
                type_params: Vec::new(),
                params: fd.params.clone(),
                ret_type: fd.ret_type.clone(),
                body: fd.body.clone(),
                reap_threshold: None,
                attrs: Vec::new(),
            };
            synthesized.push(Node::FnDef(Box::new(lifted), m_span));
        }

        // Coverage: every trait method must be provided.
        for want in trait_methods.iter() {
            if !provided.contains(want) {
                diags.push(err(
                    source,
                    file,
                    *span,
                    "E2707",
                    format!("`impl {trait_name} for {type_name}` is missing method `{want}`"),
                    "An impl must define every method declared by the trait.",
                ));
            }
        }
    }

    // ── 3. Call-site fail-closed check: a trait-method call on a statically ─
    // known type with no matching impl (and no matching free fn) → E2708.
    // Scoped to modules that declare a trait (this pass only runs then), so std /
    // keystone are untouched. Conservative: only Ident receivers whose concrete
    // struct type is locally obvious (a `let x = T{..}` / `let x: T = ..` binding
    // or a `x: T` param) are resolved — never a false positive on an unresolved
    // receiver.
    for item in &module.items {
        if let Node::FnDef(fd, _) = item {
            check_calls_in_fn(
                &fd.params,
                &fd.body,
                &method_to_trait,
                &impls,
                &existing_fns,
                source,
                file,
                &mut diags,
            );
        }
    }

    // ── 4. Drop trait/impl declarations; append synthesized free fns. ───────
    module
        .items
        .retain(|it| !matches!(it, Node::TraitDef { .. } | Node::ImplBlock { .. }));
    module.items.extend(synthesized);

    diags
}

/// Walk one fn body, tracking a locally-obvious receiver-type map, and flag any
/// trait-method call with no matching impl (and no matching free fn).
#[allow(clippy::too_many_arguments)]
fn check_calls_in_fn(
    params: &[Param],
    body: &[Node],
    method_to_trait: &BTreeMap<String, String>,
    impls: &BTreeSet<(String, String)>,
    existing_fns: &BTreeSet<String>,
    source: &str,
    file: Option<&str>,
    diags: &mut Vec<Diagnostic>,
) {
    let mut types: BTreeMap<String, String> = BTreeMap::new();
    for p in params {
        if let TypeAnn::Named(t) = &p.ty {
            types.insert(p.name.clone(), t.clone());
        }
    }
    for stmt in body {
        check_calls_in_stmt(
            stmt,
            &mut types,
            method_to_trait,
            impls,
            existing_fns,
            source,
            file,
            diags,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_calls_in_stmt(
    node: &Node,
    types: &mut BTreeMap<String, String>,
    method_to_trait: &BTreeMap<String, String>,
    impls: &BTreeSet<(String, String)>,
    existing_fns: &BTreeSet<String>,
    source: &str,
    file: Option<&str>,
    diags: &mut Vec<Diagnostic>,
) {
    // Seed the receiver-type map from `let x = T{..}` / `let x: T = ..`.
    if let Node::Let {
        name, ann, value, ..
    } = node
    {
        if let Some(TypeAnn::Named(t)) = ann {
            types.insert(name.clone(), t.clone());
        } else if let Node::StructLit { name: t, .. } = value.as_ref() {
            types.insert(name.clone(), t.clone());
        }
    }

    // Flag a trait-method call on a statically-known concrete receiver.
    if let Node::MethodCall {
        receiver,
        method,
        span,
        ..
    } = node
    {
        if let Node::Lit(Literal::Ident(var), _) = receiver.as_ref() {
            if let Some(t) = types.get(var) {
                if let Some(trait_name) = method_to_trait.get(method) {
                    let mangled = format!("{}_{}", t.to_lowercase(), method);
                    let has_impl = impls.contains(&(t.clone(), method.clone()));
                    let has_free_fn = existing_fns.contains(&mangled);
                    if !has_impl && !has_free_fn {
                        diags.push(err(
                            source,
                            file,
                            *span,
                            "E2708",
                            format!(
                                "no implementation of trait `{trait_name}` (method `{method}`) \
                                 for type `{t}`"
                            ),
                            "Add `impl <Trait> for <Type> { fn <method>(self, ...) { .. } }`, \
                             or call a defined method. (Lowering would otherwise fail to resolve \
                             this method — never silently return 0.)",
                        ));
                    }
                }
            }
        }
    }

    for child in super::closures::node_children_ref(node) {
        check_calls_in_stmt(
            child,
            types,
            method_to_trait,
            impls,
            existing_fns,
            source,
            file,
            diags,
        );
    }
}

/// Fail-closed diagnostic constructor (#268 Phase 1).
fn err(
    source: &str,
    file: Option<&str>,
    span: Span,
    code: &'static str,
    message: String,
    help: &str,
) -> Diagnostic {
    let dspan = DiagSpan::from_offsets(source, span.start(), span.end(), file);
    Diagnostic::error("trait", code, message)
        .with_span(dspan)
        .with_help(help.to_string())
}
