// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Values stored into declared slots that cannot hold them.
//!
//! `mindc check` accepts a value of another type where a declaration names
//! one: a call argument for a parameter, a value returned through a result, a
//! struct field, a variant payload, a collection element (by index
//! assignment or `push` / `set` / `insert` / `add`), a module-level binding
//! assigned in a function, and a local binding assigned in its own scope. A
//! value whose type the pass did not establish may be a string, so storing
//! it counts as storing one. The declaration then no longer proves what
//! the slot holds (see [`super::slots`]), and `==` on two such slots would
//! compare string handles, which is false for equal text, or read an integer
//! as a string record. A walk records each such store here; the next walk
//! reads every recorded slot as [`Fact::MaybeStr`], so a comparison of two of
//! them is refused like any other pair of operands of unknown type, and
//! storing one of them into another declared slot is recorded in turn. A
//! function with no declared result type that returns a string is recorded
//! the same way.
//!
//! A value that may be a string (one whose type the pass did not establish
//! included) handed to a callee whose body this module does not analyse (an
//! imported or std function, an extern, an intrinsic), through a parameter
//! not declared `string`, may be kept there and given back as an integer:
//! `vec_push(v, "abc")` and `vec_get(v, 0)`, `__mind_store_i64(p, s)` and
//! `__mind_load_i64(p)`. Such a store is recorded by the record types the
//! callee's declaration names (or as a raw store when it names none), and
//! the integer result of a later call to such a callee whose declaration
//! names one of those record types (or, after a raw store, any opaque
//! callee) is maybe a string. Of the intrinsics only `__mind_store_i64`
//! keeps a handle whole, and only `__mind_load_i64` gives one back, from raw
//! memory or from a collection's storage (`vec_addr(v)`).

use std::collections::BTreeSet;

use crate::ast::{FnDefData, Node, StructLitField, TypeAnn};

use super::classify::lowered_args;
use super::facts::{Decls, Env, Fact, Ret, SCALAR_NAMES, join, tail_value};
use super::slots::{Contradicted, holds_str, struct_key};

/// Collection methods that store their last argument as an element.
const STORE_METHODS: &[&str] = &["push", "set", "insert", "add"];

/// The prefix of compiler intrinsics.
pub(super) const INTRINSIC_PREFIX: &str = "__mind_";

/// The intrinsic that stores a 64-bit value, and the one that loads it back.
const RAW_STORE: &str = "__mind_store_i64";
pub(super) const RAW_LOAD: &str = "__mind_load_i64";

impl Decls {
    /// Record in `out` every declared slot `node` stores a contradicting
    /// value into. `owner` is the function whose body holds `node` (`None`
    /// at module level and in a closure).
    pub(super) fn record_flows(
        &self,
        node: &Node,
        env: &Env,
        owner: Option<&str>,
        out: &mut Contradicted,
    ) {
        match node {
            Node::Call { callee, args, .. } => {
                let args: Vec<&Node> = args.iter().collect();
                self.call_flows(callee, &args, env, out);
            }
            Node::MethodCall {
                receiver,
                method,
                args,
                ..
            } => self.method_flows(receiver, method, args, env, out),
            Node::Return {
                value: Some(value), ..
            } => {
                if let Some(owner) = owner {
                    self.return_flow(owner, value, &self.value_fact(value, env), out);
                }
            }
            Node::StructLit { name, fields, .. } => self.struct_flows(name, fields, env, out),
            Node::FieldAssign {
                receiver,
                field,
                value,
                ..
            } => self.field_assign_flow(receiver, field, value, env, out),
            Node::IndexAssign {
                receiver, value, ..
            } => match self.value_fact(receiver, env) {
                // A byte stored into a string.
                Fact::Str => {}
                Fact::Elems(element) | Fact::Map(element) => {
                    if self.stores_contradiction(&element, value, env) {
                        out.add_element(*element);
                    }
                }
                _ => {
                    if holds_str(&self.value_fact(value, env)) {
                        out.add_element(Fact::Unknown);
                    }
                }
            },
            _ => {}
        }
    }

    /// Record `name = value` on a local binding in scope (see
    /// `Decls::local_fact`): a value that contradicts a declared binding, or
    /// whose fact an unannotated binding does not have, read in the scope of
    /// the assignment.
    pub(super) fn record_local_assign(
        &self,
        name: &str,
        value: &Node,
        env: &Env,
        out: &mut Contradicted,
    ) {
        let Some(bound) = env.get(name) else {
            return;
        };
        let Some((site, declared)) = bound.site else {
            return;
        };
        if declared {
            if self.stores_contradiction(&bound.fact, value, env) {
                out.add_local(site, Fact::MaybeStr);
            }
        } else {
            let assigned = self.value_fact(value, env);
            if join(bound.fact.clone(), assigned.clone()) != bound.fact {
                out.add_local(site, assigned);
            }
        }
    }

    /// Record `name = value` on a module-level binding.
    pub(super) fn record_global_assign(
        &self,
        name: &str,
        value: &Node,
        env: &Env,
        out: &mut Contradicted,
    ) {
        if env.contains_key(name) {
            return;
        }
        let Some(declared) = self.global_declared(name) else {
            return;
        };
        if self.stores_contradiction(&declared, value, env) {
            out.globals.insert(name.to_string());
        }
    }

    /// Record a function body's tail expression, which the function returns.
    pub(super) fn record_tail_flow(&self, fd: &FnDefData, env: &Env, out: &mut Contradicted) {
        if tail_value(&fd.body) {
            let tail = self.block_fact(&fd.body, env, 0);
            if let Some(last) = fd.body.last() {
                self.return_flow(super::facts::fn_key(&fd.name), last, &tail, out);
            }
        }
    }

    /// Record `value` (with fact `fact`) returned from `owner`. With no
    /// declared result type, a returned string makes the result one on some
    /// path.
    fn return_flow(&self, owner: &str, value: &Node, fact: &Fact, out: &mut Contradicted) {
        let declared = match self.returns.get(owner) {
            Some(Some(Ret::Ty(ty))) => self.typed_fact(ty),
            Some(Some(Ret::Undeclared)) => Fact::NotStr,
            _ => return,
        };
        let null = declared == Fact::Str && super::slots::is_null(value);
        if !null && super::slots::contradicts(&declared, fact) {
            out.returns.insert(owner.to_string());
        }
    }

    /// Whether `callee` is not a function this module analyses: not a local
    /// binding (a closure value), not a local function with one declaration,
    /// not a variant constructor.
    pub(super) fn is_opaque(&self, callee: &str, env: &Env) -> bool {
        !env.contains_key(callee)
            && !matches!(self.params.get(callee), Some(Some(_)))
            && !self.is_variant(callee)
    }

    /// Record the arguments of a call to a local function, a variant
    /// constructor or an opaque callee.
    fn call_flows(&self, callee: &str, args: &[&Node], env: &Env, out: &mut Contradicted) {
        if env.contains_key(callee) {
            return;
        }
        if let Some(declared) = self.params.get(callee) {
            let Some(types) = declared else {
                return self.escape_flows(callee, args, env, out);
            };
            for (index, (ty, arg)) in types.iter().zip(args).enumerate() {
                if self.stores_contradiction(&self.typed_fact(ty), arg, env) {
                    let at = out.params.entry(callee.to_string()).or_default();
                    at.insert(index);
                }
            }
        } else if let Some(name) = self.project_fn(callee) {
            // Its body is analysed in its own module, which reads this store
            // (see [`super::project`]).
            let types = self.project_fns.get(name).cloned().flatten();
            for (index, arg) in args.iter().enumerate() {
                let stored = match types.as_ref().and_then(|t| t.get(index)) {
                    Some(ty) => self.stores_contradiction(&self.typed_fact(ty), arg, env),
                    None => holds_str(&self.value_fact(arg, env)),
                };
                if stored {
                    out.params
                        .entry(name.to_string())
                        .or_default()
                        .insert(index);
                }
            }
        } else if self.is_variant(callee) {
            if let Some(payload) = self.variant_payload(callee) {
                for (index, (ty, arg)) in payload.types.iter().zip(args).enumerate() {
                    if self.stores_contradiction(&self.typed_fact(ty), arg, env) {
                        let at = out.payloads.entry(payload.key.clone()).or_default();
                        at.insert(index);
                    }
                }
            }
        } else {
            self.escape_flows(callee, args, env, out);
        }
    }

    /// Record a method call: an element stored into a collection, or the
    /// arguments of the function the method lowers to.
    fn method_flows(
        &self,
        receiver: &Node,
        method: &str,
        args: &[Node],
        env: &Env,
        out: &mut Contradicted,
    ) {
        let receiver_fact = self.value_fact(receiver, env);
        if let Fact::Elems(element) | Fact::Map(element) = &receiver_fact {
            if let (true, Some(value)) = (STORE_METHODS.contains(&method), args.last()) {
                if self.stores_contradiction(element, value, env) {
                    out.add_element((**element).clone());
                }
            }
            return;
        }
        match self.method_callee(receiver, &receiver_fact, method, env) {
            Some(callee) => {
                let lowered = lowered_args(receiver, &receiver_fact, args);
                self.call_flows(&callee, &lowered, env, out);
            }
            // The callee is not known: whatever it is, it is not analysed.
            None => {
                let args: Vec<&Node> = args.iter().collect();
                self.escape_flows("", &args, env, out);
            }
        }
    }

    /// The declared parameter types of opaque `callee`, when declared.
    fn opaque_types(&self, callee: &str) -> Option<&Vec<TypeAnn>> {
        let tail = callee.rsplit_once('.').map_or(callee, |(_, tail)| tail);
        self.opaque_params
            .get(callee)
            .or_else(|| self.opaque_params.get(tail))
    }

    /// The record types opaque `callee`'s declaration names (parameters and
    /// result, other than scalars and strings): what it can keep a value in.
    /// Read off the declaration, not the arguments, so the answer does not
    /// depend on which records this module's registry knows.
    pub(super) fn callee_records(&self, callee: &str) -> BTreeSet<String> {
        let ret = match self.call_ret(callee) {
            Some(Ret::Ty(ty)) => Some(ty),
            _ => None,
        };
        let params = self.opaque_types(callee).into_iter().flatten();
        let mut records = BTreeSet::new();
        for ty in params.chain(ret.as_ref()) {
            let mut ty = self.resolve(ty);
            while let TypeAnn::Ref { target, .. } = ty {
                ty = self.resolve(&target);
            }
            if let TypeAnn::Named(name) = ty {
                let scalar = SCALAR_NAMES.contains(&name.as_str());
                if !scalar && name != "string" && name != "String" {
                    records.insert(name);
                }
            }
        }
        records
    }

    /// Record a call to an opaque callee (see [`Self::is_opaque`]) that is
    /// handed a value that may be a string ([`holds_str`], so also a value
    /// whose type the pass did not establish) through a parameter not
    /// declared `string`: by the record types its declaration names, or as a
    /// raw store when it names none. Of the compiler intrinsics only the
    /// 64-bit store keeps a value whole for a later load to give back.
    fn escape_flows(&self, callee: &str, args: &[&Node], env: &Env, out: &mut Contradicted) {
        if callee.starts_with(INTRINSIC_PREFIX) {
            let value = (callee == RAW_STORE).then(|| args.get(1)).flatten();
            if value.is_some_and(|v| holds_str(&self.value_fact(v, env))) {
                out.escaped_raw = true;
            }
            return;
        }
        let types = self.opaque_types(callee);
        let escapes = args.iter().enumerate().any(|(index, arg)| {
            // A string or record parameter (`string_len(s: String)`,
            // `vec_get(v: Vec, i)`) keeps no handle as an integer.
            let declared = types.and_then(|types| types.get(index));
            let kept = declared.is_none_or(|ty| !self.names_record(ty));
            kept && holds_str(&self.value_fact(arg, env))
        });
        if !escapes {
            return;
        }
        let records = self.callee_records(callee);
        if records.is_empty() {
            out.escaped_raw = true;
        }
        out.escaped_records.extend(records);
    }

    /// Whether `ty` names a string or a record type (not a scalar, a
    /// collection, a tensor or a type of unknown shape): a parameter of that
    /// type does not keep an integer handle.
    fn names_record(&self, ty: &TypeAnn) -> bool {
        let mut ty = self.resolve(ty);
        while let TypeAnn::Ref { target, .. } = ty {
            ty = self.resolve(&target);
        }
        matches!(ty, TypeAnn::Named(name) if !SCALAR_NAMES.contains(&name.as_str()))
    }

    /// Record `receiver.field = value`. A receiver whose struct is not known
    /// records the field of every struct declaring one of that name.
    fn field_assign_flow(
        &self,
        receiver: &Node,
        field: &str,
        value: &Node,
        env: &Env,
        out: &mut Contradicted,
    ) {
        let owners = match self.value_fact(receiver, env) {
            Fact::Record(owner) => vec![owner],
            _ => self.structs_with_field(field),
        };
        for owner in owners {
            let Some(declared) = self.declared_field_fact(&owner, field) else {
                continue;
            };
            if self.stores_contradiction(&declared, value, env) {
                out.fields
                    .entry(struct_key(&owner).to_string())
                    .or_default()
                    .insert(field.to_string());
            }
        }
    }

    /// Record the fields of a struct literal, or of a struct variant's
    /// literal.
    fn struct_flows(
        &self,
        name: &str,
        fields: &[StructLitField],
        env: &Env,
        out: &mut Contradicted,
    ) {
        if self.knows_struct(name) {
            for field in fields {
                let Some(declared) = self.declared_field_fact(name, &field.name) else {
                    continue;
                };
                if self.stores_contradiction(&declared, &field.value, env) {
                    let at = out.fields.entry(struct_key(name).to_string()).or_default();
                    at.insert(field.name.clone());
                }
            }
        } else if let Some(payload) = self.variant_payload(name) {
            for field in fields {
                let index = payload.names.iter().position(|n| *n == field.name);
                let Some((index, ty)) = index.and_then(|i| Some((i, payload.types.get(i)?))) else {
                    continue;
                };
                if self.stores_contradiction(&self.typed_fact(ty), &field.value, env) {
                    let at = out.payloads.entry(payload.key.clone()).or_default();
                    at.insert(index);
                }
            }
        }
    }
}
