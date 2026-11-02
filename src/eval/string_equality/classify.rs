// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The fact of an expression's value, for typed string equality.
//!
//! Every expression form gets a fact from what is declared: a literal, a
//! binding, a callee's declared result (including the `string_<method>` and
//! `<type>_<method>` functions a method call lowers to), a struct field, a
//! collection element, or both branches of an `if` / every arm of a `match`.
//! A form the pass cannot type is [`Fact::Unknown`], never "not a string":
//! that default is what lets two strings of unknown type be refused instead
//! of compiled to a handle compare.

use crate::ast::{Literal, Node, Pattern, TypeAnn};

use super::facts::{Class, Decls, Env, Fact, Ret, fact_class, is_numeric_literal, join};

/// Nesting beyond this depth is not analysed: the value is `Unknown`. The
/// bound keeps the recursive walk off deep `else if` chains, whose analysis
/// would otherwise grow the compiler thread stack.
const MAX_DEPTH: usize = 64;

/// Collection methods whose result is one element (a map's value).
const ELEMENT_METHODS: &[&str] = &["get", "get_str"];

/// Collection methods whose result is a count, a flag, an address or the
/// collection handle: never one element.
const SCALAR_COLLECTION_METHODS: &[&str] = &[
    "len",
    "length",
    "cap",
    "addr",
    "is_empty",
    "contains",
    "contains_key",
    "contains_key_str",
    "has",
    "push",
    "set",
    "add",
    "insert",
    "free",
];

/// Fields of the std `String` record, all `i64`.
const STRING_RECORD_FIELDS: &[&str] = &["addr", "len", "cap", "length"];

impl Decls {
    pub(super) fn classify(&self, node: &Node, env: &Env) -> Class {
        match node {
            Node::Lit(Literal::Int(_), _) => Class::IntLiteral,
            Node::Neg { operand, .. } if matches!(**operand, Node::Lit(Literal::Int(_), _)) => {
                Class::IntLiteral
            }
            Node::Paren(inner, _) => self.classify(inner, env),
            _ => fact_class(&self.value_fact(node, env)),
        }
    }

    pub(super) fn value_fact(&self, node: &Node, env: &Env) -> Fact {
        self.fact_at(node, env, 0)
    }

    /// The fact of a `let` binding. An annotation is trusted unless the value
    /// contradicts it: no string is a numeric literal, and a string value
    /// under a non-string annotation is still a string handle.
    pub(super) fn binding_fact(
        &self,
        ann: Option<&TypeAnn>,
        value: &Node,
        env: &Env,
        depth: usize,
    ) -> Fact {
        let Some(ty) = ann else {
            return self.fact_at(value, env, depth);
        };
        match self.typed_fact(ty) {
            Fact::Str if is_numeric_literal(value) => Fact::Unknown,
            Fact::Str => Fact::Str,
            _ if fact_class(&self.fact_at(value, env, depth)) == Class::Str => Fact::Unknown,
            fact => fact,
        }
    }

    /// The element fact of a collection expression, `Unknown` when unknown.
    pub(super) fn element_fact(&self, collection: &Node, env: &Env) -> Fact {
        match self.value_fact(collection, env) {
            Fact::Elems(element) => *element,
            _ => Fact::Unknown,
        }
    }

    fn fact_at(&self, node: &Node, env: &Env, depth: usize) -> Fact {
        if depth > MAX_DEPTH {
            return Fact::Unknown;
        }
        let d = depth + 1;
        match node {
            Node::Lit(Literal::Str(_), _) => Fact::Str,
            Node::Lit(Literal::Int(_) | Literal::Float(_), _) => Fact::NotStr,
            Node::Lit(Literal::Ident(name), _) => match env.get(name) {
                Some(fact) => fact.clone(),
                None => self.global_fact(name),
            },
            Node::Paren(inner, _) | Node::Ref { inner, .. } => self.fact_at(inner, env, d),
            Node::Binary { .. }
            | Node::Logical { .. }
            | Node::Bitwise { .. }
            | Node::Neg { .. }
            | Node::Not { .. }
            | Node::BitNot { .. } => Fact::NotStr,
            Node::StructLit { name, .. } => Fact::Record(name.clone()),
            Node::ArrayLit { elements, .. } | Node::SetLit { elements, .. } => {
                let element = elements
                    .iter()
                    .map(|e| self.fact_at(e, env, d))
                    .reduce(join)
                    .unwrap_or(Fact::Unknown);
                Fact::Elems(Box::new(element))
            }
            Node::MapLit { entries, .. } => {
                let value = entries
                    .iter()
                    .map(|(_, v)| self.fact_at(v, env, d))
                    .reduce(join)
                    .unwrap_or(Fact::Unknown);
                Fact::Map(Box::new(value))
            }
            Node::Tuple { elements, .. } => {
                Fact::Tuple(elements.iter().map(|e| self.fact_at(e, env, d)).collect())
            }
            Node::Call { callee, .. } => self.call_fact(callee, env),
            Node::MethodCall {
                receiver,
                method,
                args,
                ..
            } => self.method_fact(receiver, method, args.is_empty(), env, d),
            Node::FieldAccess {
                receiver, field, ..
            } => self.field_fact(receiver, field, env, d),
            Node::IndexAccess { receiver, .. } => match self.fact_at(receiver, env, d) {
                Fact::Elems(element) | Fact::Map(element) => *element,
                // A byte of a string.
                Fact::Str => Fact::NotStr,
                _ => Fact::Unknown,
            },
            Node::SliceRange { receiver, .. } => match self.fact_at(receiver, env, d) {
                fact @ Fact::Elems(_) => fact,
                _ => Fact::Unknown,
            },
            Node::If {
                then_branch,
                else_branch,
                ..
            } => {
                let then = self.block_fact(then_branch, env, d);
                let other = match else_branch {
                    Some(branch) => self.block_fact(branch, env, d),
                    None => Fact::NotStr,
                };
                join(then, other)
            }
            Node::Match {
                scrutinee, arms, ..
            } => {
                let scrutinee = self.fact_at(scrutinee, env, d);
                arms.iter()
                    .map(|arm| {
                        let mut arm_env = env.clone();
                        self.bind_pattern(&arm.pattern, scrutinee.clone(), &mut arm_env);
                        self.fact_at(&arm.body, &arm_env, d)
                    })
                    .reduce(join)
                    .unwrap_or(Fact::Never)
            }
            Node::Block { stmts, .. } => self.block_fact(stmts, env, d),
            Node::Region { body, .. } => self.block_fact(body, env, d),
            Node::As { ty, .. } => self.typed_fact(ty),
            Node::Try { inner, .. } => self.try_fact(inner, env, d),
            Node::Return { .. } | Node::Break { .. } | Node::Continue { .. } => Fact::Never,
            Node::Closure(..)
            | Node::CallGrad { .. }
            | Node::CallTensorSum { .. }
            | Node::CallTensorMean { .. }
            | Node::CallReshape { .. }
            | Node::CallExpandDims { .. }
            | Node::CallSqueeze { .. }
            | Node::CallTranspose { .. }
            | Node::CallIndex { .. }
            | Node::CallSlice { .. }
            | Node::CallSliceStride { .. }
            | Node::CallGather { .. }
            | Node::CallDot { .. }
            | Node::CallMatMul { .. }
            | Node::TensorMatmul { .. }
            | Node::TensorElemwise { .. }
            | Node::CallTensorRelu { .. }
            | Node::CallTensorRand { .. } => Fact::NotStr,
            // Statements produce unit.
            Node::Let { .. }
            | Node::LetTuple { .. }
            | Node::Assign { .. }
            | Node::IndexAssign { .. }
            | Node::FieldAssign { .. }
            | Node::For { .. }
            | Node::ForEach { .. }
            | Node::While { .. }
            | Node::Print { .. }
            | Node::Assert { .. } => Fact::NotStr,
            _ => Fact::Unknown,
        }
    }

    /// The fact of the value a statement list produces: its last statement,
    /// after the list's own bindings.
    fn block_fact(&self, stmts: &[Node], env: &Env, depth: usize) -> Fact {
        let Some((last, init)) = stmts.split_last() else {
            return Fact::NotStr;
        };
        let mut env = env.clone();
        for stmt in init {
            match stmt {
                Node::Let {
                    name, ann, value, ..
                } => {
                    let fact = self.binding_fact(ann.as_ref(), value, &env, depth);
                    self.bind(&mut env, name, fact);
                }
                Node::LetTuple { names, .. } => {
                    for name in names {
                        self.bind(&mut env, name, Fact::Unknown);
                    }
                }
                // A diverging statement: the list produces no value.
                Node::Return { .. } | Node::Break { .. } | Node::Continue { .. } => {
                    return Fact::Never;
                }
                _ => {}
            }
        }
        // A name assigned inside the list may hold anything at its end.
        for name in super::assigned_in(stmts) {
            self.bind(&mut env, &name, Fact::Unknown);
        }
        self.fact_at(last, &env, depth)
    }

    /// Bind every name `pattern` introduces, matched against a value with
    /// `fact`: a variant payload binding gets the payload's declared type.
    pub(super) fn bind_pattern(&self, pattern: &Pattern, fact: Fact, env: &mut Env) {
        let mut pending = vec![(pattern, fact)];
        while let Some((pattern, fact)) = pending.pop() {
            match pattern {
                Pattern::Ident(name) => self.bind(env, name, fact),
                Pattern::Tuple(args) => {
                    let facts = match fact {
                        Fact::Tuple(facts) if facts.len() == args.len() => facts,
                        _ => vec![Fact::Unknown; args.len()],
                    };
                    pending.extend(args.iter().zip(facts));
                }
                Pattern::EnumVariant { path, args } => {
                    let payload = self.variant_payload(path).map(|(types, _)| types);
                    for (i, arg) in args.iter().enumerate() {
                        let ty = payload.as_ref().and_then(|types| types.get(i));
                        pending.push((arg, ty.map_or(Fact::Unknown, |t| self.typed_fact(t))));
                    }
                }
                Pattern::EnumStruct { path, fields } => {
                    let payload = self.variant_payload(path);
                    for (field, arg) in fields {
                        let ty = payload.as_ref().and_then(|(types, names)| {
                            names
                                .iter()
                                .position(|n| n == field)
                                .and_then(|i| types.get(i))
                        });
                        pending.push((arg, ty.map_or(Fact::Unknown, |t| self.typed_fact(t))));
                    }
                }
                Pattern::Literal(_) | Pattern::Wildcard => {}
            }
        }
    }

    fn call_fact(&self, callee: &str, env: &Env) -> Fact {
        // A local binding called as a function (a closure value).
        if env.contains_key(callee) {
            return Fact::Unknown;
        }
        if let Some(ret) = self.call_ret(callee) {
            return self.ret_fact(&ret);
        }
        if self.returns.contains_key(callee) {
            // Two local declarations disagree.
            return Fact::Unknown;
        }
        // A module-qualified call names the imported function by its tail.
        if let Some((_, tail)) = callee.rsplit_once('.') {
            if let Some(ret) = self.call_ret(tail) {
                return self.ret_fact(&ret);
            }
        }
        // Compiler intrinsics return integers or floats; a `tensor.` builtin is
        // a tensor operation; a variant constructor builds an enum record.
        if callee.starts_with("__mind_") || callee.starts_with("tensor.") || self.is_variant(callee)
        {
            return Fact::NotStr;
        }
        Fact::Unknown
    }

    /// The callee a method call lowers to, as lowering resolves it, given the
    /// receiver's fact.
    fn method_callee(
        &self,
        receiver: &Node,
        receiver_fact: &Fact,
        method: &str,
        env: &Env,
    ) -> Option<String> {
        if let Node::Lit(Literal::Ident(name), _) = receiver {
            if !env.contains_key(name) && *receiver_fact == Fact::Unknown {
                // A static call on a type name: `string.from_utf8_bytes(b)`.
                if name == "string" || name == "String" {
                    return Some(format!("string_{method}"));
                }
                if self.knows_struct(name) || self.knows_enum(name) {
                    return Some(format!("{}_{method}", name.to_lowercase()));
                }
            }
        }
        match receiver_fact {
            Fact::Str => Some(format!("string_{method}")),
            Fact::Record(owner) => {
                let bare = owner.rsplit('.').next().unwrap_or(owner);
                Some(format!("{}_{method}", bare.to_lowercase()))
            }
            _ => None,
        }
    }

    fn method_fact(
        &self,
        receiver: &Node,
        method: &str,
        no_args: bool,
        env: &Env,
        depth: usize,
    ) -> Fact {
        let receiver_fact = self.fact_at(receiver, env, depth);
        match &receiver_fact {
            Fact::Elems(element) | Fact::Map(element) => {
                return if ELEMENT_METHODS.contains(&method) {
                    (**element).clone()
                } else if SCALAR_COLLECTION_METHODS.contains(&method) {
                    Fact::NotStr
                } else {
                    Fact::Unknown
                };
            }
            // A zero-argument accessor named like a field reads the field.
            Fact::Record(owner) if no_args => {
                if let Some(ty) = self.field_ty(owner, method) {
                    return self.typed_fact(&ty);
                }
            }
            _ => {}
        }
        match self.method_callee(receiver, &receiver_fact, method, env) {
            Some(callee) => self.call_fact(&callee, env),
            None => Fact::Unknown,
        }
    }

    fn field_fact(&self, receiver: &Node, field: &str, env: &Env, depth: usize) -> Fact {
        // `Enum.Variant` (a dotted variant access) is an enum value.
        if let Node::Lit(Literal::Ident(name), _) = receiver {
            if !env.contains_key(name) && self.knows_enum(name) {
                return Fact::NotStr;
            }
        }
        match self.fact_at(receiver, env, depth) {
            Fact::Record(owner) => self
                .field_ty(&owner, field)
                .map_or(Fact::Unknown, |ty| self.typed_fact(&ty)),
            Fact::Str if STRING_RECORD_FIELDS.contains(&field) => Fact::NotStr,
            Fact::Elems(_) | Fact::Map(_) if field == "length" || field == "len" => Fact::NotStr,
            Fact::Tuple(facts) => field
                .parse::<usize>()
                .ok()
                .and_then(|i| facts.get(i).cloned())
                .unwrap_or(Fact::Unknown),
            _ => Fact::Unknown,
        }
    }

    /// The fact of `inner?`: the payload of the `Result<T, E>` / `Option<T>`
    /// the inner call or method call is declared to return.
    fn try_fact(&self, inner: &Node, env: &Env, depth: usize) -> Fact {
        let callee = match inner {
            Node::Call { callee, .. } if !env.contains_key(callee) => Some(callee.clone()),
            Node::MethodCall {
                receiver, method, ..
            } => {
                let receiver_fact = self.fact_at(receiver, env, depth);
                self.method_callee(receiver, &receiver_fact, method, env)
            }
            _ => None,
        };
        let ret = callee.and_then(|callee| self.call_ret(&callee));
        match ret {
            Some(Ret::Ty(ty)) => match self.resolve(&ty) {
                TypeAnn::Generic { name, args } if name == "Result" || name == "Option" => {
                    args.first().map_or(Fact::Unknown, |t| self.typed_fact(t))
                }
                _ => Fact::Unknown,
            },
            _ => Fact::Unknown,
        }
    }
}
