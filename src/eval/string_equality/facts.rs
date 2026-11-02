// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Declaration facts for typed string equality.
//!
//! A value is a string when its declared type says so: a `string`/`String`
//! literal, parameter, annotated binding, constant, call result, field or
//! element. A value whose type the pass cannot establish is [`Fact::Unknown`]:
//! it is never assumed not to be a string, because a string compared as an
//! integer compares handles and reports equal text as unequal. The
//! interpreter and every lowering read the same facts, so a comparison is
//! byte-wise on every backend or refused on every backend.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::ast::{EnumVariant, FnDefData, Literal, Module, Node, Param, TypeAnn};
use crate::eval::type_aliases::LocalTypeAliases;

/// What the pass knows about one value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Fact {
    /// A `string`/`String` record.
    Str,
    /// A record of the named struct.
    Record(String),
    /// An `array<T>`, `[T; N]`, `&[T]`, `set<T>` or tensor; the fact is what
    /// one element is.
    Elems(Box<Fact>),
    /// A `map<K, V>`; the fact is what one value is.
    Map(Box<Fact>),
    /// A tuple: one fact per position.
    Tuple(Vec<Fact>),
    /// A scalar (integer, float, bool, enum, pointer, unit) that is not a
    /// string and holds none.
    NotStr,
    /// An expression that produces no value (`return`, `break`, `continue`).
    Never,
    /// A value whose type is not known: it may be a string.
    Unknown,
}

/// The equality class of one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Class {
    Str,
    /// An integer literal: a handle compared with it is a null or sentinel test.
    IntLiteral,
    /// A value whose type is known and is not a string.
    NotStr,
    /// A value whose type is not known.
    Unknown,
}

/// The equality class of a value with fact `fact`.
pub(super) fn fact_class(fact: &Fact) -> Class {
    match fact {
        Fact::Str => Class::Str,
        Fact::Unknown => Class::Unknown,
        Fact::Record(_)
        | Fact::Elems(_)
        | Fact::Map(_)
        | Fact::Tuple(_)
        | Fact::NotStr
        | Fact::Never => Class::NotStr,
    }
}

/// The fact of a value that is either `a` or `b`. Two different facts prove
/// only that the value is a scalar when both are scalars.
pub(super) fn join(a: Fact, b: Fact) -> Fact {
    match (a, b) {
        (Fact::Never, other) | (other, Fact::Never) => other,
        (a, b) if a == b => a,
        _ => Fact::Unknown,
    }
}

/// Facts in scope, shared copy-on-write: a statement's sub-walks clone the
/// pointer, and only a binding that changes a fact copies the map.
pub(super) type Env = Rc<BTreeMap<String, Fact>>;

/// Scalar type names: no value of these types is a string.
const SCALAR_NAMES: &[&str] = &[
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64", "bool", "usize", "isize",
    "char", "()",
];

pub(super) fn is_numeric_literal(node: &Node) -> bool {
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

/// A callee's declared result.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Ret {
    Unit,
    Ty(TypeAnn),
    /// The result type names a generic parameter.
    Generic,
}

impl Ret {
    fn of(ret: Option<&TypeAnn>, type_params: &[String]) -> Self {
        match ret {
            None => Ret::Unit,
            Some(TypeAnn::Named(n)) if type_params.contains(n) => Ret::Generic,
            Some(ty) => Ret::Ty(ty.clone()),
        }
    }
}

/// Declarations visible to the comparisons of one module, collected once.
pub(super) struct Decls {
    aliases: LocalTypeAliases,
    /// Callee name → declared result; `None` when two declarations disagree.
    pub(super) returns: BTreeMap<String, Option<Ret>>,
    structs: BTreeMap<String, Option<Vec<(String, TypeAnn)>>>,
    /// Module-local enum name → its variants.
    enums: BTreeMap<String, Vec<EnumVariant>>,
    /// Module-level `const` and `let` names → the fact of their value.
    consts: BTreeMap<String, Option<Fact>>,
}

impl Decls {
    pub(super) fn new(module: &Module) -> Self {
        let mut decls = Self {
            aliases: LocalTypeAliases::new(&module.items),
            returns: BTreeMap::new(),
            structs: BTreeMap::new(),
            enums: BTreeMap::new(),
            consts: BTreeMap::new(),
        };
        let mut globals = Vec::new();
        let mut pending: Vec<&Node> = module.items.iter().collect();
        while let Some(item) = pending.pop() {
            match item {
                Node::FnDef(fd, _) => {
                    let ret = Ret::of(fd.ret_type.as_ref(), &fd.type_params);
                    merge(&mut decls.returns, &fd.name, Some(ret));
                }
                Node::ExternBlock { fns, .. } => {
                    for f in fns {
                        merge(
                            &mut decls.returns,
                            &f.name,
                            Some(Ret::of(f.ret_type.as_ref(), &[])),
                        );
                    }
                }
                Node::StructDef { name, fields, .. } => {
                    let fields = fields.iter().map(|f| (f.name.clone(), f.ty.clone()));
                    merge(&mut decls.structs, name, Some(fields.collect()));
                }
                Node::EnumDef { name, variants, .. } => {
                    decls.enums.insert(name.clone(), variants.clone());
                }
                Node::Const { .. } | Node::Let { .. } | Node::ExternConst { .. } => {
                    globals.push(item);
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
            decls
                .returns
                .entry(name)
                .or_insert(Some(Ret::of(ret.as_ref(), &[])));
        }
        // A module-level binding is visible in every function body; its fact
        // is computed once every type it may name is registered.
        let empty = Env::default();
        for item in globals {
            let (name, fact) = match item {
                Node::Const {
                    name, ty, value, ..
                } => (name, decls.let_fact(ty.as_ref(), value, &empty)),
                Node::Let {
                    name, ann, value, ..
                } => (name, decls.let_fact(ann.as_ref(), value, &empty)),
                Node::ExternConst { name, ty, .. } => (name, decls.typed_fact(ty)),
                _ => continue,
            };
            merge(&mut decls.consts, name, Some(fact));
        }
        decls
    }

    pub(super) fn resolve(&self, ty: &TypeAnn) -> TypeAnn {
        self.aliases.resolve_with_imports(ty)
    }

    pub(super) fn field_ty(&self, owner: &str, field: &str) -> Option<TypeAnn> {
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

    pub(super) fn knows_struct(&self, name: &str) -> bool {
        match self.structs.get(name) {
            Some(local) => local.is_some(),
            None => crate::ir::with_global_enums(|g| {
                g.structs.contains_key(name) || g.qualified_structs.contains_key(name)
            }),
        }
    }

    pub(super) fn knows_enum(&self, name: &str) -> bool {
        self.enums.contains_key(name)
            || crate::ir::with_global_enums(|g| g.names.iter().any(|n| n == name))
    }

    /// The fact of a value of declared type `ty`.
    pub(super) fn typed_fact(&self, ty: &TypeAnn) -> Fact {
        let elems = |element: &TypeAnn| Fact::Elems(Box::new(self.typed_fact(element)));
        match self.resolve(ty) {
            TypeAnn::Named(n) if n == "string" || n == "String" => Fact::Str,
            TypeAnn::Named(n) if SCALAR_NAMES.contains(&n.as_str()) => Fact::NotStr,
            TypeAnn::Named(n) if n == "bytes" || n.starts_with("bytes[") => {
                Fact::Elems(Box::new(Fact::NotStr))
            }
            TypeAnn::Named(n) if self.knows_enum(&n) => Fact::NotStr,
            TypeAnn::Named(n) if self.knows_struct(&n) => Fact::Record(n),
            TypeAnn::Named(_) => Fact::Unknown,
            TypeAnn::Generic { name, args } => match (name.as_str(), args.as_slice()) {
                ("array" | "set", [element]) => elems(element),
                ("map", [_, value]) => Fact::Map(Box::new(self.typed_fact(value))),
                ("Option" | "Result", _) => Fact::NotStr,
                _ => Fact::Unknown,
            },
            TypeAnn::Array { element, .. } | TypeAnn::Slice { element, .. } => elems(&element),
            TypeAnn::Ref { target, .. } => self.typed_fact(&target),
            TypeAnn::Tuple { elements } => {
                Fact::Tuple(elements.iter().map(|t| self.typed_fact(t)).collect())
            }
            TypeAnn::Tensor { .. } | TypeAnn::DiffTensor { .. } | TypeAnn::SparseTensor { .. } => {
                Fact::Elems(Box::new(Fact::NotStr))
            }
            TypeAnn::ScalarI32
            | TypeAnn::ScalarI64
            | TypeAnn::ScalarF32
            | TypeAnn::ScalarF64
            | TypeAnn::ScalarBool
            | TypeAnn::ScalarU32
            | TypeAnn::RawPtr { .. }
            | TypeAnn::FnPtr { .. } => Fact::NotStr,
        }
    }

    /// The fact of a callee's result, when the callee is declared.
    pub(super) fn ret_fact(&self, ret: &Ret) -> Fact {
        match ret {
            Ret::Unit => Fact::NotStr,
            Ret::Ty(ty) => self.typed_fact(ty),
            Ret::Generic => Fact::Unknown,
        }
    }

    /// A callee's declared result. A module-local declaration wins over the
    /// project registry; an ambiguous local name proves nothing.
    pub(super) fn call_ret(&self, callee: &str) -> Option<Ret> {
        match self.returns.get(callee) {
            Some(local) => local.clone(),
            None => {
                crate::ir::with_global_enums(|g| g.fn_returns.get(callee).cloned()).map(Ret::Ty)
            }
        }
    }

    /// The fact of a name with no binding in scope: a module-level constant, a
    /// function, an enum variant, or unknown.
    pub(super) fn global_fact(&self, name: &str) -> Fact {
        match self.consts.get(name) {
            Some(fact) => fact.clone().unwrap_or(Fact::Unknown),
            None if self.returns.contains_key(name) || self.is_variant(name) => Fact::NotStr,
            None => Fact::Unknown,
        }
    }

    /// Whether `name` names an enum variant (`E::V`, or a bare registered `V`).
    pub(super) fn is_variant(&self, name: &str) -> bool {
        // The built-in `Option` / `Result` prelude.
        if matches!(name, "Some" | "None" | "Ok" | "Err") {
            return true;
        }
        if let Some((head, _)) = name.rsplit_once("::") {
            let owner = head.rsplit('.').next().unwrap_or(head);
            return owner == "Option" || owner == "Result" || self.knows_enum(owner);
        }
        let suffix = format!("::{name}");
        self.enums.values().flatten().any(|v| v.name == name)
            || crate::ir::with_global_enums(|g| g.variant_tags.keys().any(|k| k.ends_with(&suffix)))
    }

    /// The declared payload types of the variant a pattern names, with the
    /// field names of a struct variant. `None` when the variant is unknown or
    /// a bare name is ambiguous.
    pub(super) fn variant_payload(&self, path: &str) -> Option<(Vec<TypeAnn>, Vec<String>)> {
        let of = |v: &EnumVariant| (v.payload.clone(), v.field_names.clone());
        let Some((head, variant)) = path.rsplit_once("::") else {
            let mut local = self.enums.values().flatten().filter(|v| v.name == path);
            return match (local.next(), local.next()) {
                (Some(v), None) => Some(of(v)),
                (Some(_), Some(_)) => None,
                (None, _) => {
                    self.global_payload(|key| key.rsplit_once("::").is_some_and(|(_, v)| v == path))
                }
            };
        };
        let owner = head.rsplit('.').next().unwrap_or(head);
        if let Some(variants) = self.enums.get(owner) {
            return variants.iter().find(|v| v.name == variant).map(of);
        }
        let key = format!("{owner}::{variant}");
        self.global_payload(|k| k == key)
    }

    /// The payload of the one registered variant whose key satisfies `wanted`.
    fn global_payload(&self, wanted: impl Fn(&str) -> bool) -> Option<(Vec<TypeAnn>, Vec<String>)> {
        crate::ir::with_global_enums(|g| {
            let mut keys = g.payload_types.keys().filter(|k| wanted(k.as_str()));
            let (Some(key), None) = (keys.next(), keys.next()) else {
                return None;
            };
            let names = g.struct_field_names.get(key).cloned().unwrap_or_default();
            Some((g.payload_types[key].clone(), names))
        })
    }

    /// Bind `name` to `fact`. Every binding is recorded, so a name with no
    /// entry is never a local of unknown type mistaken for a constant.
    pub(super) fn bind(&self, env: &mut Env, name: &str, fact: Fact) {
        if env.get(name) != Some(&fact) {
            Rc::make_mut(env).insert(name.to_string(), fact);
        }
    }

    /// The fact of binding `name`, whose initial value has `fact`, after the
    /// later `assignments`. The assigned values are read with `name` holding
    /// `fact`: when every one of them keeps it, no assignment can change it.
    ///
    /// A declared binding keeps its declared fact unless an assignment
    /// contradicts the declaration (a numeric literal into a string, a string
    /// into anything else); an unannotated binding keeps its fact only when
    /// every assigned value has the same fact.
    pub(super) fn reassigned_fact(
        &self,
        name: &str,
        fact: Fact,
        declared: bool,
        assignments: &[&Node],
        env: &Env,
    ) -> Fact {
        if assignments.is_empty() {
            return fact;
        }
        let mut env = env.clone();
        self.bind(&mut env, name, fact.clone());
        if !declared {
            return assignments
                .iter()
                .fold(fact, |fact, v| join(fact, self.value_fact(v, &env)));
        }
        let contradicted = if fact == Fact::Str {
            assignments.iter().any(|v| is_numeric_literal(v))
        } else {
            assignments
                .iter()
                .any(|v| self.classify(v, &env) == Class::Str)
        };
        if contradicted { Fact::Unknown } else { fact }
    }

    /// Bind a declaration's parameters over `env`.
    pub(super) fn param_env(&self, mut env: Env, params: &[Param], body: &[Node]) -> Env {
        for param in params {
            let assignments = super::assignment_values(body, &param.name);
            let typed = self.typed_fact(&param.ty);
            let fact = self.reassigned_fact(&param.name, typed, true, &assignments, &env);
            self.bind(&mut env, &param.name, fact);
        }
        env
    }

    pub(super) fn fn_env(&self, fd: &FnDefData) -> Env {
        self.param_env(Env::default(), &fd.params, &fd.body)
    }

    /// The fact for an immutable `let` (see [`Self::binding_fact`]).
    pub(super) fn let_fact(&self, ann: Option<&TypeAnn>, value: &Node, env: &Env) -> Fact {
        self.binding_fact(ann, value, env, 0)
    }
}
