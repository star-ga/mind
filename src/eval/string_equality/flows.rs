// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Strings stored into declared non-string slots.
//!
//! `mindc check` accepts a string where a declaration names another type: a
//! call argument for an `i64` parameter, a value returned through an `i64`
//! result, a struct field or a variant payload. The declaration then no
//! longer proves the slot holds no string, and `==` on two such slots would
//! compare string handles, which is false for equal text. A walk records each
//! such store here; the next walk reads every recorded slot as
//! [`Fact::MaybeStr`], so a comparison of two of them is refused like any
//! other pair of operands of unknown type, and storing one of them into
//! another declared non-string slot is recorded in turn. A function with no
//! declared result type that returns a string is recorded the same way.
//!
//! Only declarations of this module are recorded: an imported function's
//! body is analysed with its own module.

use crate::ast::{FnDefData, Node, StructLitField};

use super::facts::{Contradicted, Decls, Env, Fact, Ret, contradicts, tail_value};

impl Decls {
    /// Record in `out` every declared non-string slot `node` stores a string
    /// into. `owner` is the function whose body holds `node` (`None` at
    /// module level and in a closure).
    pub(super) fn record_flows(
        &self,
        node: &Node,
        env: &Env,
        owner: Option<&str>,
        out: &mut Contradicted,
    ) {
        match node {
            Node::Call { callee, args, .. } if !env.contains_key(callee) => {
                self.call_flows(callee, args.iter(), env, out);
            }
            Node::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let receiver_fact = self.value_fact(receiver, env);
                if let Some(callee) = self.method_callee(receiver, &receiver_fact, method, env) {
                    // A value receiver is the lowered call's first argument; a
                    // type-name receiver (`String.from(x)`) is not an argument.
                    let value_receiver = matches!(receiver_fact, Fact::Str | Fact::Record(_));
                    let first = value_receiver.then_some(&**receiver);
                    self.call_flows(&callee, first.into_iter().chain(args), env, out);
                }
            }
            Node::Return {
                value: Some(value), ..
            } => {
                if let Some(owner) = owner {
                    self.return_flow(owner, &self.value_fact(value, env), out);
                }
            }
            Node::StructLit { name, fields, .. } => self.struct_flows(name, fields, env, out),
            Node::FieldAssign {
                receiver,
                field,
                value,
                ..
            } => {
                if let Fact::Record(owner) = self.value_fact(receiver, env) {
                    if let Some(declared) = self.declared_field_fact(&owner, field) {
                        if contradicts(&declared, &self.value_fact(value, env)) {
                            let fields = out.fields.entry(owner).or_default();
                            fields.insert(field.clone());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Record a function body's tail expression, which the function returns.
    pub(super) fn record_tail_flow(&self, fd: &FnDefData, env: &Env, out: &mut Contradicted) {
        if tail_value(&fd.body) {
            let tail = self.block_fact(&fd.body, env, 0);
            self.return_flow(&fd.name, &tail, out);
        }
    }

    /// Record a value with fact `value` returned from `owner`. With no
    /// declared result type, a returned string makes the result one on some
    /// path.
    fn return_flow(&self, owner: &str, value: &Fact, out: &mut Contradicted) {
        let declared = match self.returns.get(owner) {
            Some(Some(Ret::Ty(ty))) => self.typed_fact(ty),
            Some(Some(Ret::Undeclared)) => Fact::NotStr,
            _ => return,
        };
        if contradicts(&declared, value) {
            out.returns.insert(owner.to_string());
        }
    }

    /// Record the arguments of a call to a local function or a variant
    /// constructor.
    fn call_flows<'n>(
        &self,
        callee: &str,
        args: impl Iterator<Item = &'n Node>,
        env: &Env,
        out: &mut Contradicted,
    ) {
        if let Some(declared) = self.params.get(callee) {
            let Some(types) = declared else {
                return;
            };
            for (index, (ty, arg)) in types.iter().zip(args).enumerate() {
                if contradicts(&self.typed_fact(ty), &self.value_fact(arg, env)) {
                    let at = out.params.entry(callee.to_string()).or_default();
                    at.insert(index);
                }
            }
        } else if self.is_variant(callee) {
            if let Some(payload) = self.variant_payload(callee) {
                for (index, (ty, arg)) in payload.types.iter().zip(args).enumerate() {
                    if contradicts(&self.typed_fact(ty), &self.value_fact(arg, env)) {
                        let at = out.payloads.entry(payload.key.clone()).or_default();
                        at.insert(index);
                    }
                }
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
                if contradicts(&declared, &self.value_fact(&field.value, env)) {
                    let at = out.fields.entry(name.to_string()).or_default();
                    at.insert(field.name.clone());
                }
            }
        } else if let Some(payload) = self.variant_payload(name) {
            for field in fields {
                let index = payload.names.iter().position(|n| *n == field.name);
                let Some((index, ty)) = index.and_then(|i| Some((i, payload.types.get(i)?))) else {
                    continue;
                };
                if contradicts(&self.typed_fact(ty), &self.value_fact(&field.value, env)) {
                    let at = out.payloads.entry(payload.key.clone()).or_default();
                    at.insert(index);
                }
            }
        }
    }
}
