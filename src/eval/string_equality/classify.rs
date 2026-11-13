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

use super::facts::{Class, Decls, Env, Fact, Ret, fact_class, join, tuple_parts};
use super::flows::{INTRINSIC_PREFIX, RAW_LOAD};
use super::slots::{contradicts, is_null};

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

/// Collection methods whose result is a count or a flag, never a handle:
/// the result of one on a parameter declared a collection, whatever a call
/// stores into it.
const COUNT_METHODS: &[&str] = &[
    "len",
    "length",
    "cap",
    "is_empty",
    "contains",
    "contains_key",
    "contains_key_str",
    "has",
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

    /// Whether `expr as ty` casts an integer literal other than the null
    /// handle `0` to `string`: no string record is at that address, so the
    /// cast is refused.
    pub(super) fn refused_cast(&self, expr: &Node, ty: &TypeAnn, env: &Env) -> bool {
        self.typed_fact(ty) == Fact::Str
            && self.classify(expr, env) == Class::IntLiteral
            && !is_null(expr)
    }

    /// The fact of `expr as ty`. A cast to `string` stores its operand in a
    /// string slot: an operand not proven a string, other than the null
    /// handle, is no string record there (see `Decls::stores_contradiction`).
    /// A refused cast keeps its declared fact, so it is reported once.
    fn cast_fact(&self, expr: &Node, ty: &TypeAnn, env: &Env, depth: usize) -> Fact {
        let declared = self.typed_fact(ty);
        if declared != Fact::Str || is_null(expr) || self.refused_cast(expr, ty, env) {
            return declared;
        }
        if contradicts(&declared, &self.fact_at(expr, env, depth)) {
            Fact::MaybeStr
        } else {
            declared
        }
    }

    /// The fact of a `let` binding. An annotation is trusted unless the value
    /// contradicts it (see `Decls::stores_contradiction`): a value that may
    /// be a string under a non-string annotation is still a string handle,
    /// and an integer other than the null handle under a `string` annotation
    /// is no string record.
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
        let declared = self.typed_fact(ty);
        if self.stores_contradiction(&declared, value, env) {
            Fact::MaybeStr
        } else {
            declared
        }
    }

    /// The element fact of a collection expression, `Unknown` when unknown.
    pub(super) fn element_fact(&self, collection: &Node, env: &Env) -> Fact {
        match self.value_fact(collection, env) {
            Fact::Elems(element) => self.contradicted.element(*element),
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
                Some(bound) => bound.fact.clone(),
                None => self.global_fact(name),
            },
            Node::Paren(inner, _) | Node::Ref { inner, .. } => self.fact_at(inner, env, d),
            Node::Binary { .. }
            | Node::Logical { .. }
            | Node::Bitwise { .. }
            | Node::Neg { .. }
            | Node::Not { .. }
            | Node::BitNot { .. } => Fact::NotStr,
            // A literal of the std `String` record is a string.
            Node::StructLit { name, .. } if is_string_record(name) => Fact::Str,
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
            Node::Call { callee, args, .. } => match self.builtin_variant(callee) {
                Some(variant) if !env.contains_key(callee) => variant.build(
                    args.first()
                        .map_or(Fact::Never, |a| self.fact_at(a, env, d)),
                ),
                _ => self.call_fact(callee, env),
            },
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
                Fact::Elems(element) | Fact::Map(element) => self.contradicted.element(*element),
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
            Node::As { expr, ty, .. } => self.cast_fact(expr, ty, env, d),
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
    pub(super) fn block_fact(&self, stmts: &[Node], env: &Env, depth: usize) -> Fact {
        let Some((last, init)) = stmts.split_last() else {
            return Fact::NotStr;
        };
        // A binding of this list has the fact every recorded store into it
        // leaves it (`Decls::local_fact`).
        let mut env = env.clone();
        for stmt in init {
            match stmt {
                Node::Let {
                    name, ann, value, ..
                } => {
                    let fact = self.binding_fact(ann.as_ref(), value, &env, depth);
                    self.bind_local(&mut env, name, fact, ann.is_some());
                }
                Node::LetTuple { names, value, .. } => {
                    let parts = tuple_parts(self.fact_at(value, &env, depth), names.len());
                    let mut bound = env.clone();
                    for (name, fact) in names.iter().zip(parts) {
                        self.bind_local(&mut bound, name, fact, false);
                    }
                    env = bound;
                }
                // A diverging statement: the list produces no value.
                Node::Return { .. } | Node::Break { .. } | Node::Continue { .. } => {
                    return Fact::Never;
                }
                _ => {}
            }
        }
        self.fact_at(last, &env, depth)
    }

    /// Bind every name `pattern` introduces, matched against a value with
    /// `fact`: a variant payload binding gets the payload's declared type.
    pub(super) fn bind_pattern(&self, pattern: &Pattern, fact: Fact, env: &mut Env) {
        let mut pending = vec![(pattern, fact)];
        while let Some((pattern, fact)) = pending.pop() {
            match pattern {
                Pattern::Ident(name) => self.bind_local(env, name, fact, false),
                Pattern::Tuple(args) => {
                    pending.extend(args.iter().zip(tuple_parts(fact, args.len())));
                }
                Pattern::EnumVariant { path, args } => {
                    // A built-in `Option` / `Result` payload has the scrutinee's
                    // payload fact; the prelude's registered payload type is
                    // generic and proves nothing about it.
                    if let Some(variant) = self.builtin_variant(path) {
                        let fact = variant.payload_of(&fact).unwrap_or(Fact::Unknown);
                        pending.extend(args.iter().map(|arg| (arg, fact.clone())));
                        continue;
                    }
                    let payload = self.variant_payload(path);
                    for (i, arg) in args.iter().enumerate() {
                        let fact = payload
                            .as_ref()
                            .map_or(Fact::Unknown, |p| self.payload_fact(p, i));
                        pending.push((arg, fact));
                    }
                }
                Pattern::EnumStruct { path, fields } => {
                    let payload = self.variant_payload(path);
                    for (field, arg) in fields {
                        let fact = payload.as_ref().map_or(Fact::Unknown, |p| {
                            let index = p.names.iter().position(|n| n == field);
                            index.map_or(Fact::Unknown, |i| self.payload_fact(p, i))
                        });
                        pending.push((arg, fact));
                    }
                }
                Pattern::Literal(_) | Pattern::Wildcard => {}
            }
        }
    }

    /// The fact of the result of `callee(args)`. A callee whose body is not
    /// analysed here may give back, as an integer, a string an earlier call
    /// handed it (see `Decls::escape_flows`).
    fn call_fact(&self, callee: &str, env: &Env) -> Fact {
        let fact = self.declared_call_fact(callee, env);
        if fact == Fact::NotStr && self.returns_escaped(callee, env) {
            Fact::MaybeStr
        } else {
            fact
        }
    }

    /// Whether `callee` may return a string handed to an opaque callee
    /// earlier: it is opaque itself, and its declaration names a record type
    /// such a string was handed over with, or a string was stored to raw
    /// memory (which may be any record's storage), or it names no record
    /// while a string was handed over with none (see `Decls::escape_flows`).
    /// The raw 64-bit load reads any record's storage (`vec_addr(v)`,
    /// `map_vals_addr(m)`), so it may give back a string handed over by
    /// either route.
    fn returns_escaped(&self, callee: &str, env: &Env) -> bool {
        let c = &self.contradicted;
        if (!c.escaped_raw && c.escaped_records.is_empty()) || !self.is_opaque(callee, env) {
            return false;
        }
        if callee.starts_with(INTRINSIC_PREFIX) {
            return callee == RAW_LOAD;
        }
        let records = self.callee_records(callee);
        c.escaped_raw || !records.is_disjoint(&c.escaped_records)
    }

    /// The fact a callee's declaration gives its result.
    fn declared_call_fact(&self, callee: &str, env: &Env) -> Fact {
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
        // a tensor operation; a bare math / tensor builtin (`sqrt`, `exp`)
        // returns a number or a tensor; a variant constructor builds an enum
        // record.
        if callee.starts_with("__mind_")
            || callee.starts_with("tensor.")
            || crate::type_checker::resolve::BARE_BUILTINS
                .binary_search(&callee)
                .is_ok()
            || self.is_variant(callee)
        {
            return Fact::NotStr;
        }
        Fact::Unknown
    }

    /// The callee a method call lowers to, as lowering resolves it, given the
    /// receiver's fact.
    pub(super) fn method_callee(
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

    /// Whether the method call `receiver.method(..)` is a method on a string,
    /// which lowering makes a call of `string_<method>`: `Some(true)` for a
    /// string value receiver (passed first), `Some(false)` for the type name
    /// `string` / `String` (not passed). `None` when the evaluator has no
    /// `string_<method>` to run: the module does not define it (`mindc test`
    /// merges the std module that does) and the evaluator does not supply it
    /// itself, so the call stays a method call the evaluator runs as before.
    pub(super) fn string_method(&self, receiver: &Node, method: &str, env: &Env) -> Option<bool> {
        use crate::eval::extern_c_intrinsics::{is_extern_c_intrinsic, is_runtime_string_fn};
        let function = format!("string_{method}");
        let runs = self.params.contains_key(&function)
            || self.supplied.contains(&function)
            || is_runtime_string_fn(&function)
            || is_extern_c_intrinsic(&function);
        if !runs {
            return None;
        }
        let fact = self.value_fact(receiver, env);
        if fact == Fact::Str {
            return Some(true);
        }
        let callee = self.method_callee(receiver, &fact, method, env)?;
        (callee == function).then_some(false)
    }

    fn method_fact(
        &self,
        receiver: &Node,
        method: &str,
        no_args: bool,
        env: &Env,
        depth: usize,
    ) -> Fact {
        // Lowering resolves the method by the parameter's declared type, so a
        // value stored into the parameter that its type does not hold does
        // not change what a count or a flag of the collection is.
        if let Node::Lit(Literal::Ident(name), _) = receiver
            && let Some(Fact::Elems(_) | Fact::Map(_)) =
                env.get(name).and_then(|b| b.shape.as_ref())
            && COUNT_METHODS.contains(&method)
        {
            return Fact::NotStr;
        }
        let receiver_fact = self.fact_at(receiver, env, depth);
        match &receiver_fact {
            Fact::Elems(element) | Fact::Map(element) => {
                return if ELEMENT_METHODS.contains(&method) {
                    self.contradicted.element((**element).clone())
                } else if SCALAR_COLLECTION_METHODS.contains(&method) {
                    Fact::NotStr
                } else {
                    Fact::Unknown
                };
            }
            // A zero-argument accessor named like a field reads the field.
            Fact::Record(owner) if no_args => {
                if let Some(fact) = self.declared_field_fact(owner, method) {
                    return fact;
                }
            }
            _ => {}
        }
        match self.method_callee(receiver, &receiver_fact, method, env) {
            // A method that names no declared function (the runtime library
            // may supply one, as it does `string_trim`): its result type is
            // not visible here.
            Some(callee) if self.call_ret(&callee).is_none() => Fact::Any,
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
                .declared_field_fact(&owner, field)
                .unwrap_or(Fact::Unknown),
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
        let ret = callee.as_deref().and_then(|callee| self.call_ret(callee));
        let payload = match ret {
            Some(Ret::Ty(ty)) => match self.resolve(&ty) {
                TypeAnn::Generic { name, args } if name == "Result" || name == "Option" => {
                    args.first().map_or(Fact::Unknown, |t| self.typed_fact(t))
                }
                _ => Fact::Unknown,
            },
            _ => Fact::Unknown,
        };
        let escaped = callee.is_some_and(|c| self.returns_escaped(&c, env));
        if payload == Fact::NotStr && escaped {
            Fact::MaybeStr
        } else {
            payload
        }
    }
}

/// Whether `name` names the std `String` record (a `string` value).
pub(super) fn is_string_record(name: &str) -> bool {
    matches!(name.rsplit('.').next(), Some("String" | "string"))
}

/// The arguments a method call lowers to: a value receiver first; a
/// type-name receiver (`String.from(x)`) is not an argument.
pub(super) fn lowered_args<'n>(
    receiver: &'n Node,
    receiver_fact: &Fact,
    args: &'n [Node],
) -> Vec<&'n Node> {
    let value_receiver = matches!(receiver_fact, Fact::Str | Fact::Record(_));
    value_receiver
        .then_some(receiver)
        .into_iter()
        .chain(args)
        .collect()
}
