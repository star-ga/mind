// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Declaration facts and operand classes for typed string equality.
//!
//! A value is a string when its declared type says so: a `string`/`String`
//! literal, parameter, annotated binding, constant, call result or struct
//! field. The interpreter and every lowering read the same facts, so a
//! comparison is byte-wise on every backend or refused on every backend.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::ast::{FnDefData, Literal, Module, Node, Param, TypeAnn};
use crate::eval::type_aliases::LocalTypeAliases;

/// What the pass knows about one lexical binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Fact {
    /// The binding holds a `string`/`String` record.
    Str,
    /// The binding holds a record of the named struct.
    Record(String),
    /// The binding holds an `array<string>`, `[string; N]` or `&[string]`.
    StrArray,
    /// A binding with no string or record fact. It is recorded only to shadow
    /// a module-level string constant of the same name (see [`Decls::bind`]).
    Opaque,
}

/// The equality class of one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Class {
    Str,
    /// An integer literal: a handle compared with it is a null or sentinel test.
    IntLiteral,
    Other,
}

/// Facts in scope, shared copy-on-write: a statement's sub-walks clone the
/// pointer, and only a binding that changes a fact copies the map.
pub(super) type Env = Rc<BTreeMap<String, Fact>>;

/// Methods whose result is a string when their receiver is a string.
fn string_result_method(name: &str) -> bool {
    matches!(
        name,
        "slice"
            | "substring"
            | "trim"
            | "trim_start"
            | "trim_end"
            | "to_lowercase"
            | "to_uppercase"
            | "to_lower"
            | "to_upper"
            | "replace"
            | "concat"
            | "repeat"
    )
}

fn is_numeric_literal(node: &Node) -> bool {
    match node {
        Node::Lit(Literal::Int(_) | Literal::Float(_), _) => true,
        Node::Neg { operand, .. } | Node::Paren(operand, _) => is_numeric_literal(operand),
        _ => false,
    }
}

/// Record `value` for `name`; two declarations that disagree prove nothing.
fn merge<T: PartialEq>(map: &mut BTreeMap<String, Option<T>>, name: &str, value: Option<T>) {
    match map.get_mut(name) {
        Some(slot) if *slot != value => *slot = None,
        Some(_) => {}
        None => {
            map.insert(name.to_string(), value);
        }
    }
}

/// Declarations visible to the comparisons of one module, collected once.
pub(super) struct Decls {
    aliases: LocalTypeAliases,
    returns: BTreeMap<String, Option<TypeAnn>>,
    structs: BTreeMap<String, Option<Vec<(String, TypeAnn)>>>,
    /// Module-level `const` and `let` names: whether each holds a string.
    string_consts: BTreeMap<String, Option<bool>>,
}

impl Decls {
    pub(super) fn new(module: &Module) -> Self {
        let mut decls = Self {
            aliases: LocalTypeAliases::new(&module.items),
            returns: BTreeMap::new(),
            structs: BTreeMap::new(),
            string_consts: BTreeMap::new(),
        };
        let mut pending: Vec<&Node> = module.items.iter().collect();
        while let Some(item) = pending.pop() {
            match item {
                Node::FnDef(fd, _) => merge(&mut decls.returns, &fd.name, fd.ret_type.clone()),
                Node::StructDef { name, fields, .. } => {
                    let fields = fields.iter().map(|f| (f.name.clone(), f.ty.clone()));
                    merge(&mut decls.structs, name, Some(fields.collect()));
                }
                Node::Const {
                    name, ty, value, ..
                } => {
                    let is_string = match ty {
                        Some(ty) => decls.is_string(ty),
                        None => matches!(**value, Node::Lit(Literal::Str(_), _)),
                    };
                    merge(&mut decls.string_consts, name, Some(is_string));
                }
                // A module-level binding is visible in every function body.
                Node::Let {
                    name, ann, value, ..
                } => {
                    let is_string = match ann {
                        Some(ty) => decls.is_string(ty) && !is_numeric_literal(value),
                        None => matches!(**value, Node::Lit(Literal::Str(_), _)),
                    };
                    merge(&mut decls.string_consts, name, Some(is_string));
                }
                Node::Block { stmts, .. } => pending.extend(stmts),
                _ => {}
            }
        }
        // Imported and bundled std callees, after local ones, as lowering
        // registers `IRModule::fn_signatures`: a local definition shadows them.
        #[cfg(feature = "cross-module-imports")]
        let imported = crate::type_checker::cm_all_imported_fn_signatures();
        #[cfg(not(feature = "cross-module-imports"))]
        let imported = crate::type_checker::bundled_std_fn_signatures().clone();
        for (name, _, ret) in imported {
            decls.returns.entry(name).or_insert(ret);
        }
        decls
    }

    pub(super) fn is_string(&self, ty: &TypeAnn) -> bool {
        matches!(self.aliases.resolve(ty), TypeAnn::Named(n) if n == "string" || n == "String")
    }

    /// A callee's declared return type. A module-local declaration wins over
    /// the project registry, and an ambiguous local name proves nothing.
    fn call_returns(&self, callee: &str) -> Option<TypeAnn> {
        match self.returns.get(callee) {
            Some(local) => local.clone(),
            None => crate::ir::with_global_enums(|g| g.fn_returns.get(callee).cloned()),
        }
    }

    fn field_ty(&self, owner: &str, field: &str) -> Option<TypeAnn> {
        let find = |fields: &[(String, TypeAnn)]| {
            fields
                .iter()
                .find(|(name, _)| name == field)
                .map(|(_, ty)| ty.clone())
        };
        match self.structs.get(owner) {
            Some(local) => local.as_deref().and_then(find),
            None => crate::ir::with_global_enums(|g| {
                let (names, types) = g
                    .structs
                    .get(owner)
                    .or_else(|| g.qualified_structs.get(owner))?;
                names
                    .iter()
                    .position(|name| name == field)
                    .and_then(|i| types.get(i).cloned())
            }),
        }
    }

    fn knows_struct(&self, name: &str) -> bool {
        match self.structs.get(name) {
            Some(local) => local.is_some(),
            None => crate::ir::with_global_enums(|g| {
                g.structs.contains_key(name) || g.qualified_structs.contains_key(name)
            }),
        }
    }

    fn struct_name(&self, ty: &TypeAnn) -> Option<String> {
        match self.aliases.resolve(ty) {
            TypeAnn::Named(name) if !self.is_string(ty) && self.knows_struct(&name) => Some(name),
            _ => None,
        }
    }

    fn has_string_elements(&self, ty: &TypeAnn) -> bool {
        match self.aliases.resolve(ty) {
            TypeAnn::Generic { name, args } if name == "array" => {
                matches!(args.as_slice(), [element] if self.is_string(element))
            }
            TypeAnn::Array { element, .. } | TypeAnn::Slice { element, .. } => {
                self.is_string(&element)
            }
            _ => false,
        }
    }

    fn typed_fact(&self, ty: &TypeAnn) -> Fact {
        if self.is_string(ty) {
            return Fact::Str;
        }
        if self.has_string_elements(ty) {
            return Fact::StrArray;
        }
        self.struct_name(ty).map_or(Fact::Opaque, Fact::Record)
    }

    /// Whether `node` denotes a collection whose elements are strings.
    pub(super) fn is_string_array(&self, node: &Node, env: &Env) -> bool {
        match node {
            Node::Paren(inner, _) => self.is_string_array(inner, env),
            Node::Lit(Literal::Ident(name), _) => env.get(name) == Some(&Fact::StrArray),
            Node::Call { callee, .. } => self
                .call_returns(callee)
                .is_some_and(|ty| self.has_string_elements(&ty)),
            Node::FieldAccess {
                receiver, field, ..
            } => self
                .record_of(receiver, env)
                .and_then(|owner| self.field_ty(&owner, field))
                .is_some_and(|ty| self.has_string_elements(&ty)),
            _ => false,
        }
    }

    fn is_string_const(&self, name: &str) -> bool {
        self.string_consts.get(name) == Some(&Some(true))
    }

    /// Bind `name` to `fact`. A binding without a fact is kept out of the map
    /// (so the map stays as small as the facts it holds) unless it must shadow
    /// a string constant of the same name.
    pub(super) fn bind(&self, env: &mut Env, name: &str, fact: Fact) {
        if fact == Fact::Opaque && !self.is_string_const(name) {
            if env.contains_key(name) {
                Rc::make_mut(env).remove(name);
            }
            return;
        }
        if env.get(name) != Some(&fact) {
            Rc::make_mut(env).insert(name.to_string(), fact);
        }
    }

    /// Bind a declaration's parameters over `env`. A reassigned parameter
    /// keeps its fact only when it is a declared string never given a numeric
    /// literal.
    pub(super) fn param_env(&self, mut env: Env, params: &[Param], body: &[Node]) -> Env {
        for param in params {
            let assignments = super::assignment_values(body, &param.name);
            let fact = match self.typed_fact(&param.ty) {
                fact if assignments.is_empty() => fact,
                Fact::Str if !assignments.iter().any(|v| is_numeric_literal(v)) => Fact::Str,
                _ => Fact::Opaque,
            };
            self.bind(&mut env, &param.name, fact);
        }
        env
    }

    pub(super) fn fn_env(&self, fd: &FnDefData) -> Env {
        self.param_env(Env::default(), &fd.params, &fd.body)
    }

    /// The fact for a reassigned `let`: a declared string keeps it unless a
    /// numeric literal is assigned; an unannotated binding keeps it only when
    /// every assigned value is a string too. A record fact is not kept.
    pub(super) fn reassigned_let_fact(
        &self,
        ann: Option<&TypeAnn>,
        value: &Node,
        assignments: &[&Node],
        env: &Env,
    ) -> Fact {
        if assignments.iter().any(|v| is_numeric_literal(v)) {
            return Fact::Opaque;
        }
        let declared = ann.is_some_and(|ty| self.is_string(ty));
        let all_strings = || {
            assignments
                .iter()
                .all(|v| self.classify(v, env) == Class::Str)
        };
        match self.let_fact(ann, value, env) {
            Fact::Str if declared || all_strings() => Fact::Str,
            _ => Fact::Opaque,
        }
    }

    /// The fact for an immutable `let`. A string annotation is trusted unless
    /// the value is a numeric literal, which no string can be.
    pub(super) fn let_fact(&self, ann: Option<&TypeAnn>, value: &Node, env: &Env) -> Fact {
        if let Some(ty) = ann {
            if self.is_string(ty) && is_numeric_literal(value) {
                return Fact::Opaque;
            }
            return self.typed_fact(ty);
        }
        if self.classify(value, env) == Class::Str {
            return Fact::Str;
        }
        if self.is_string_array(value, env) {
            return Fact::StrArray;
        }
        self.record_of(value, env)
            .map_or(Fact::Opaque, Fact::Record)
    }

    /// The struct a receiver expression denotes, when its type is declared.
    fn record_of(&self, node: &Node, env: &Env) -> Option<String> {
        match node {
            Node::Paren(inner, _) => self.record_of(inner, env),
            Node::Lit(Literal::Ident(name), _) => match env.get(name) {
                Some(Fact::Record(owner)) => Some(owner.clone()),
                _ => None,
            },
            Node::StructLit { name, .. } => self.knows_struct(name).then(|| name.clone()),
            Node::Call { callee, .. } => self.struct_name(&self.call_returns(callee)?),
            Node::FieldAccess {
                receiver, field, ..
            } => {
                let owner = self.record_of(receiver, env)?;
                self.struct_name(&self.field_ty(&owner, field)?)
            }
            _ => None,
        }
    }

    pub(super) fn classify(&self, mut node: &Node, env: &Env) -> Class {
        loop {
            return match node {
                Node::Lit(Literal::Str(_), _) => Class::Str,
                Node::Lit(Literal::Int(_), _) => Class::IntLiteral,
                Node::Neg { operand, .. } if matches!(**operand, Node::Lit(Literal::Int(_), _)) => {
                    Class::IntLiteral
                }
                Node::Lit(Literal::Ident(name), _) => match env.get(name) {
                    Some(Fact::Str) => Class::Str,
                    Some(_) => Class::Other,
                    None if self.is_string_const(name) => Class::Str,
                    None => Class::Other,
                },
                Node::Paren(inner, _) => {
                    node = inner;
                    continue;
                }
                Node::MethodCall {
                    receiver, method, ..
                } if string_result_method(method) => {
                    node = receiver;
                    continue;
                }
                Node::Call { callee, .. } => match self.call_returns(callee) {
                    Some(ty) if self.is_string(&ty) => Class::Str,
                    _ => Class::Other,
                },
                Node::IndexAccess { receiver, .. } if self.is_string_array(receiver, env) => {
                    Class::Str
                }
                Node::FieldAccess {
                    receiver, field, ..
                } => match self
                    .record_of(receiver, env)
                    .and_then(|owner| self.field_ty(&owner, field))
                {
                    Some(ty) if self.is_string(&ty) => Class::Str,
                    _ => Class::Other,
                },
                _ => Class::Other,
            };
        }
    }
}
