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

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::ast::{EnumVariant, FnDefData, Module, Node, Param, TypeAnn};
use crate::eval::type_aliases::LocalTypeAliases;

use super::slots::{Contradicted, contradicts, holds_str, is_null};
use super::sums::Variant;

mod payloads;

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
    /// An `Option<T>` or `Result<T, E>`: the fact of the `Some` / `Ok`
    /// payload, then of the `Err` payload ([`Fact::Never`] where there is
    /// none).
    Sum(Box<Fact>, Box<Fact>),
    /// A scalar (integer, float, bool, enum, pointer, unit) that is not a
    /// string and holds none.
    NotStr,
    /// An expression that produces no value (`return`, `break`, `continue`).
    Never,
    /// A value whose type the pass did not establish: it may be a string,
    /// so it compares as one may, and storing it stores one that may be.
    Unknown,
    /// A value of a type this module cannot see: a generic type parameter, or
    /// the result of an imported function with no declared result type. It
    /// may be anything, a string included: it compares like
    /// [`Fact::Unknown`], and storing it stores a value that may be a string.
    Any,
    /// A value that holds a string on some path: one branch or assignment is
    /// a string, or a string is stored into its declared non-string slot. It
    /// compares like [`Fact::Unknown`], and storing it stores a string.
    MaybeStr,
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
        Fact::Unknown | Fact::MaybeStr | Fact::Any => Class::Unknown,
        Fact::Record(_)
        | Fact::Elems(_)
        | Fact::Map(_)
        | Fact::Tuple(_)
        | Fact::Sum(..)
        | Fact::NotStr
        | Fact::Never => Class::NotStr,
    }
}

/// The fact of a value that is either `a` or `b`, joined part by part. Two
/// facts of different shapes prove nothing, except that the value may hold
/// a string when either may hold one.
pub(super) fn join(a: Fact, b: Fact) -> Fact {
    let parts = |a: Box<Fact>, b: Box<Fact>| Box::new(join(*a, *b));
    match (a, b) {
        (Fact::Never, other) | (other, Fact::Never) => other,
        (a, b) if a == b => a,
        (Fact::Elems(a), Fact::Elems(b)) => Fact::Elems(parts(a, b)),
        (Fact::Map(a), Fact::Map(b)) => Fact::Map(parts(a, b)),
        (Fact::Sum(a, e), Fact::Sum(b, f)) => Fact::Sum(parts(a, b), parts(e, f)),
        (a, b) if holds_str(&a) || holds_str(&b) => Fact::MaybeStr,
        _ => Fact::Unknown,
    }
}

/// The facts of the `n` parts of a value with `fact` destructured as a tuple
/// (`let (a, b) = v`, a tuple pattern). A part of a value that may hold a
/// string may be one; a part of a value whose shape is not known is unknown.
pub(super) fn tuple_parts(fact: Fact, n: usize) -> Vec<Fact> {
    match fact {
        Fact::Tuple(facts) if facts.len() == n => facts,
        Fact::MaybeStr | Fact::Any => vec![fact; n],
        _ => vec![Fact::Unknown; n],
    }
}

/// One name in scope: the fact of its value and, for a local binding, where
/// it is bound.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Bound {
    pub(super) fact: Fact,
    /// The binding site ([`site`]) and whether its type is declared (an
    /// annotated `let` or a parameter); `None` for a name an assignment
    /// cannot store into here (a capture or a module-level binding).
    pub(super) site: Option<(usize, bool)>,
}

/// Facts in scope, shared copy-on-write: a statement's sub-walks clone the
/// pointer, and only a binding that changes a fact copies the map.
pub(super) type Env = Rc<BTreeMap<String, Bound>>;

/// The identity of the binding whose name is `name`: the address of the
/// name in the analysed AST (a `let`, a tuple `let` position, a parameter, a
/// loop variable or a pattern binding), stable while the module is analysed.
pub(super) fn site(name: &String) -> usize {
    name as *const String as usize
}

/// Scalar type names: no value of these types is a string.
pub(super) const SCALAR_NAMES: &[&str] = &[
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64", "bool", "usize", "isize",
    "char", "()",
];

/// The name a declaration is keyed by: its source name, without the owner
/// `mindc test` qualifies the items of its merged module with, so calls and
/// the project summary, which use source names, find it.
pub(super) fn fn_key(name: &str) -> &str {
    crate::eval::module_bindings::symbol_display_name(name)
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

/// The payload of one enum variant.
pub(super) struct Payload {
    /// `Enum::Variant`, the key of [`Contradicted::payloads`].
    pub(super) key: String,
    pub(super) types: Vec<TypeAnn>,
    /// The field names of a struct variant, by position.
    pub(super) names: Vec<String>,
}

/// A callee's declared result.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Ret {
    /// No declared result type and no value returned.
    Unit,
    Ty(TypeAnn),
    /// The result type names a generic parameter.
    Generic,
    /// No declared result type, but the body returns a value (`mindc check`
    /// accepts `return <value>` with no `->`).
    Undeclared,
    /// No declared result type and no visible body (an imported
    /// declaration): the result may be anything.
    Invisible,
    /// The body returns a string through a non-string result type, or with no
    /// result type.
    Stored,
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

/// Whether the last statement of a function body is an expression: the
/// function returns its value.
pub(super) fn tail_value(body: &[Node]) -> bool {
    !matches!(
        body.last(),
        None | Some(
            Node::Let { .. }
                | Node::LetTuple { .. }
                | Node::Assign { .. }
                | Node::IndexAssign { .. }
                | Node::FieldAssign { .. }
                | Node::For { .. }
                | Node::ForEach { .. }
                | Node::While { .. }
                | Node::Print { .. }
                | Node::Assert { .. }
                | Node::Return { .. }
                | Node::Break { .. }
                | Node::Continue { .. }
        )
    )
}

/// Whether `body` returns a value: a `return <value>` outside nested
/// declarations, or a tail expression.
fn returns_value(body: &[Node]) -> bool {
    if tail_value(body) {
        return true;
    }
    let mut pending: Vec<&Node> = body.iter().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Return { value: Some(_), .. } => return true,
            Node::FnDef(..) | Node::Closure(..) => {}
            _ => pending.extend(crate::eval::closures::node_children_ref(node)),
        }
    }
    false
}

/// Function name → declared parameter types (`None` when two declarations
/// disagree).
pub(super) type Fns = BTreeMap<String, Option<Vec<TypeAnn>>>;

/// Declarations visible to the comparisons of one module, collected once.
pub(super) struct Decls {
    aliases: LocalTypeAliases,
    /// Callee name → declared result; `None` when two declarations disagree.
    pub(super) returns: BTreeMap<String, Option<Ret>>,
    /// Local function → declared parameter types; `None` when two
    /// declarations disagree.
    pub(super) params: BTreeMap<String, Option<Vec<TypeAnn>>>,
    /// Extern, imported and bundled std function → declared parameter types:
    /// callees whose body this module does not analyse.
    pub(super) opaque_params: BTreeMap<String, Vec<TypeAnn>>,
    /// Slots whose declaration a stored value contradicts (empty until the
    /// first walk has recorded them, except for the slots the other modules
    /// of a project build store into).
    pub(super) contradicted: Contradicted,
    /// Every function the modules of the active project build declare (see
    /// [`super::project`]); empty outside a project build of several modules.
    pub(super) project_fns: std::sync::Arc<Fns>,
    /// The project functions whose own module refuses a comparison in them.
    pub(super) project_refusing: std::sync::Arc<BTreeSet<String>>,
    /// Every function the interpreter's merged module defines, by source
    /// name (empty for a build): a method on a string desugars to the
    /// `string_<method>` call only where one runs.
    pub(super) supplied: Rc<BTreeSet<String>>,
    structs: BTreeMap<String, Option<Vec<(String, TypeAnn)>>>,
    /// Module-local enum name → its variants.
    enums: BTreeMap<String, Vec<EnumVariant>>,
    /// Module-level `const` and `let` names → the fact of their value.
    consts: BTreeMap<String, Option<Fact>>,
}

impl Decls {
    /// The declarations of `items`, the items of `module` one analysis
    /// decides, starting from the stores the other modules of its project
    /// build make (see [`super::project`]).
    pub(super) fn new(module: &Module, items: &[&Node]) -> Self {
        #[cfg(feature = "cross-module-imports")]
        if let Some(project) = super::project::current() {
            let fns = project.fns.clone();
            let mut decls = Self::seeded_items(module, items, fns, project.slots.shared());
            decls.project_refusing = project.refusing.clone();
            return decls;
        }
        Self::seeded_items(module, items, Default::default(), Contradicted::default())
    }

    /// The declarations of `module`, with `contradicted` already recorded
    /// and the functions of the other modules of its project build.
    pub(super) fn seeded(
        module: &Module,
        project_fns: std::sync::Arc<Fns>,
        contradicted: Contradicted,
    ) -> Self {
        let items: Vec<&Node> = module.items.iter().collect();
        Self::seeded_items(module, &items, project_fns, contradicted)
    }

    /// As [`Self::seeded`], for the `items` of `module` only: its imports
    /// and type aliases still come from the whole module.
    pub(super) fn seeded_items(
        module: &Module,
        items: &[&Node],
        project_fns: std::sync::Arc<Fns>,
        contradicted: Contradicted,
    ) -> Self {
        let mut decls = Self {
            aliases: LocalTypeAliases::new(&module.items),
            returns: BTreeMap::new(),
            params: BTreeMap::new(),
            opaque_params: BTreeMap::new(),
            contradicted,
            project_fns,
            project_refusing: Default::default(),
            supplied: Default::default(),
            structs: BTreeMap::new(),
            enums: BTreeMap::new(),
            consts: BTreeMap::new(),
        };
        let mut globals = Vec::new();
        let mut pending: Vec<&Node> = items.to_vec();
        while let Some(item) = pending.pop() {
            match item {
                Node::FnDef(fd, _) => {
                    let ret = match &fd.ret_type {
                        None if returns_value(&fd.body) => Ret::Undeclared,
                        ret => Ret::of(ret.as_ref(), &fd.type_params),
                    };
                    let name = fn_key(&fd.name);
                    merge(&mut decls.returns, name, Some(ret));
                    let types = fd.params.iter().map(|p| p.ty.clone()).collect();
                    merge(&mut decls.params, name, Some(types));
                }
                Node::ExternBlock { fns, .. } => {
                    for f in fns {
                        merge(
                            &mut decls.returns,
                            &f.name,
                            Some(Ret::of(f.ret_type.as_ref(), &[])),
                        );
                        let types = f.params.iter().map(|p| p.ty.clone()).collect();
                        decls.opaque_params.insert(f.name.clone(), types);
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
        let imported = crate::type_checker::cm_all_imported_fn_signatures(module);
        #[cfg(not(feature = "cross-module-imports"))]
        let imported: Vec<_> = crate::type_checker::bundled_std_fn_signatures(module)
            .cloned()
            .collect();
        // The bundled std after them, whatever the project's module table
        // makes visible: a project build resolves `string_len` or `vec_push`
        // with no `use`, and `mindc check` of the same file reads the bundled
        // declarations, so both must see the same ones. An imported
        // declaration with no result type may return anything: its body is
        // not visible here.
        #[cfg(feature = "cross-module-imports")]
        let imported = imported
            .into_iter()
            .chain(crate::type_checker::bundled_std_fn_signatures(module).cloned());
        for (name, params, ret) in imported {
            let ret = ret.map_or(Ret::Invisible, |ty| Ret::of(Some(&ty), &[]));
            decls.opaque_params.entry(name.clone()).or_insert(params);
            decls.returns.entry(name).or_insert(Some(ret));
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
            merge(&mut decls.consts, fn_key(name), Some(fact));
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

    /// The fact of field `field` of struct `owner`, when declared.
    pub(super) fn declared_field_fact(&self, owner: &str, field: &str) -> Option<Fact> {
        let ty = self.field_ty(owner, field)?;
        let stored = self
            .contradicted
            .fields
            .get(super::slots::struct_key(owner));
        if stored.is_some_and(|fields| fields.contains(field)) {
            return Some(Fact::MaybeStr);
        }
        Some(self.typed_fact(&ty))
    }

    /// Every struct (local or registered) that declares a field `field`.
    pub(super) fn structs_with_field(&self, field: &str) -> Vec<String> {
        let mut owners: Vec<String> = self
            .structs
            .iter()
            .filter(|(_, fields)| {
                fields
                    .as_ref()
                    .is_some_and(|fields| fields.iter().any(|(name, _)| name == field))
            })
            .map(|(owner, _)| owner.clone())
            .collect();
        crate::ir::with_global_enums(|g| {
            for (owner, (names, _)) in g.structs.iter().chain(g.qualified_structs.iter()) {
                if names.iter().any(|name| name == field) && !owners.contains(owner) {
                    owners.push(owner.clone());
                }
            }
        });
        owners
    }

    pub(super) fn knows_struct(&self, name: &str) -> bool {
        match self.structs.get(name) {
            Some(local) => local.is_some(),
            None => crate::ir::with_global_enums(|g| {
                g.structs.contains_key(name) || g.qualified_structs.contains_key(name)
            }),
        }
    }

    /// Whether this module declares an enum `name`.
    pub(super) fn local_enum(&self, name: &str) -> bool {
        self.enums.contains_key(name)
    }

    /// Whether an enum of this module declares a variant `name`.
    pub(super) fn local_variant(&self, name: &str) -> bool {
        self.enums.values().flatten().any(|v| v.name == name)
    }

    pub(super) fn knows_enum(&self, name: &str) -> bool {
        self.enums.contains_key(name)
            || crate::ir::with_global_enums(|g| g.names.iter().any(|n| n == name))
    }

    /// The fact of a value of declared type `ty`.
    pub(super) fn typed_fact(&self, ty: &TypeAnn) -> Fact {
        let elems = |element: &TypeAnn| Fact::Elems(Box::new(self.typed_fact(element)));
        let never = || Box::new(Fact::Never);
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
                ("Option", [value]) => Fact::Sum(Box::new(self.typed_fact(value)), never()),
                ("Result", [value, error]) => Fact::Sum(
                    Box::new(self.typed_fact(value)),
                    Box::new(self.typed_fact(error)),
                ),
                ("Option" | "Result", _) => {
                    Fact::Sum(Box::new(Fact::Unknown), Box::new(Fact::Unknown))
                }
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
            Ret::Undeclared => Fact::Unknown,
            Ret::Generic | Ret::Invisible => Fact::Any,
            Ret::Stored => Fact::MaybeStr,
        }
    }

    /// A callee's declared result. A module-local declaration wins over the
    /// project registry; an ambiguous local name proves nothing.
    pub(super) fn call_ret(&self, callee: &str) -> Option<Ret> {
        // This also covers a function of another module of the project that
        // returns a string through its result: the project summary records
        // it under the name the project build's calls use (see
        // [`super::project`]).
        if self.contradicted.returns.contains(callee) {
            return Some(Ret::Stored);
        }
        match self.returns.get(callee) {
            Some(local) => local.clone(),
            None => {
                crate::ir::with_global_enums(|g| g.fn_returns.get(callee).cloned()).map(Ret::Ty)
            }
        }
    }

    /// The name of the project function another module declares that
    /// `callee` calls (`provider.eqh`, or a bare imported `eqh`), when it is
    /// not a function of this module.
    pub(super) fn project_fn<'c>(&self, callee: &'c str) -> Option<&'c str> {
        let name = callee.rsplit_once('.').map_or(callee, |(_, tail)| tail);
        let local = matches!(self.params.get(callee), Some(Some(_)));
        (!local && self.project_fns.contains_key(name)).then_some(name)
    }

    /// The project function of another module that `callee` calls, when
    /// that module refuses a comparison in it: calling it makes the project
    /// fail to build, so the call is refused where it is checked too.
    pub(super) fn refusing_project_fn<'c>(&self, callee: &'c str, env: &Env) -> Option<&'c str> {
        if env.contains_key(callee) {
            return None;
        }
        self.project_fn(callee)
            .filter(|name| self.project_refusing.contains(*name))
    }

    /// The fact of a name with no binding in scope: a module-level constant, a
    /// function, an enum variant, or unknown.
    pub(super) fn global_fact(&self, name: &str) -> Fact {
        match self.consts.get(name) {
            Some(_) if self.contradicted.globals.contains(name) => Fact::MaybeStr,
            Some(fact) => fact.clone().unwrap_or(Fact::Unknown),
            None if self.builtin_variant(name) == Some(Variant::None) => {
                Variant::None.build(Fact::Never)
            }
            None if self.returns.contains_key(name) || self.is_variant(name) => Fact::NotStr,
            None => Fact::Unknown,
        }
    }

    /// The fact a module-level binding's declaration gives it, when `name`
    /// is one and its declarations agree.
    pub(super) fn global_declared(&self, name: &str) -> Option<Fact> {
        self.consts.get(name).cloned().flatten()
    }

    /// Whether `name` is a registered owner-qualified variant tag.
    ///
    /// The project rebuild keeps those tags on the qualified registry, which
    /// exists only when cross-module imports are compiled in.
    fn owner_variant(name: &str) -> bool {
        crate::ir::with_global_enums(|g| {
            if g.variant_tags.contains_key(name) {
                return true;
            }
            #[cfg(feature = "cross-module-imports")]
            {
                g.qualified.variant_tags.contains_key(name)
            }
            #[cfg(not(feature = "cross-module-imports"))]
            {
                false
            }
        })
    }

    /// Whether `name` names an enum variant (`E::V`, or a bare registered `V`).
    pub(super) fn is_variant(&self, name: &str) -> bool {
        // The built-in `Option` / `Result` prelude.
        if matches!(name, "Some" | "None" | "Ok" | "Err") {
            return true;
        }
        // A colliding enum is stored under its owner (`crate.table::Table::Raw`).
        // That spelling is still an enum value, not a string.
        if Self::owner_variant(name) {
            return true;
        }
        if let Some((head, _)) = name.rsplit_once("::") {
            let owner = head.rsplit('.').next().unwrap_or(head);
            return owner == "Option" || owner == "Result" || self.knows_enum(owner);
        }
        let suffix = format!("::{name}");
        self.enums.values().flatten().any(|v| v.name == name)
            || crate::ir::with_global_enums(|g| {
                // A project build registers its variants in the qualified registry.
                #[cfg(feature = "cross-module-imports")]
                if g.qualified
                    .variant_tags
                    .keys()
                    .any(|k| k.ends_with(&suffix))
                {
                    return true;
                }
                g.variant_tags.keys().any(|k| k.ends_with(&suffix))
            })
    }

    /// Bind `name` as `bound`. Every binding is recorded, so a name with no
    /// entry is never a local of unknown type mistaken for a constant.
    fn bind_bound(&self, env: &mut Env, name: &str, bound: Bound) {
        if env.get(name) != Some(&bound) {
            Rc::make_mut(env).insert(name.to_string(), bound);
        }
    }

    /// Bind the local binding `name` (see [`site`]), whose initial value has
    /// `fact`, as every store into it leaves it (see [`Self::local_fact`]).
    pub(super) fn bind_local(&self, env: &mut Env, name: &String, fact: Fact, declared: bool) {
        let at = site(name);
        let fact = self.local_fact(at, fact, declared);
        self.bind_bound(
            env,
            name,
            Bound {
                fact,
                site: Some((at, declared)),
            },
        );
    }

    /// The fact of the local binding at `site` whose initial value has
    /// `fact`, after every assignment a walk recorded (see
    /// `Decls::record_local_assign`): a declared binding keeps its declared
    /// fact unless an assignment contradicts the declaration, and an
    /// unannotated one keeps its fact only while every assigned value has
    /// the same fact. Each assigned value is read in its own scope, with the
    /// binding holding the fact the previous walk gave it, so the module is
    /// walked again until no binding changes.
    pub(super) fn local_fact(&self, site: usize, fact: Fact, declared: bool) -> Fact {
        match self.contradicted.locals.get(&site) {
            None => fact,
            Some(_) if declared => Fact::MaybeStr,
            Some(assigned) => join(fact, assigned.clone()),
        }
    }

    /// Whether storing `value` where `declared` is declared contradicts the
    /// declaration (see [`contradicts`]). The null handle `0` is a valid
    /// value of a `string` slot.
    pub(super) fn stores_contradiction(&self, declared: &Fact, value: &Node, env: &Env) -> bool {
        if *declared == Fact::Str && is_null(value) {
            return false;
        }
        contradicts(declared, &self.value_fact(value, env))
    }

    /// Bind a declaration's parameters over `env`. `owner` names the function
    /// they belong to (`None` for a closure): a parameter some call in the
    /// module gives a value its declaration does not hold is not proven to
    /// hold what is declared. A parameter of a type parameter of the function
    /// (`type_params`) may hold anything.
    pub(super) fn param_env(
        &self,
        mut env: Env,
        params: &[Param],
        owner: Option<&str>,
        type_params: &[String],
    ) -> Env {
        let stored = owner.and_then(|name| self.contradicted.params.get(name));
        for (index, param) in params.iter().enumerate() {
            let typed = if stored.is_some_and(|at| at.contains(&index)) {
                Fact::MaybeStr
            } else {
                match &param.ty {
                    TypeAnn::Named(name) if type_params.contains(name) => Fact::Any,
                    ty => self.typed_fact(ty),
                }
            };
            self.bind_local(&mut env, &param.name, typed, true);
        }
        env
    }

    pub(super) fn fn_env(&self, fd: &FnDefData) -> Env {
        let owner = Some(fn_key(&fd.name));
        self.param_env(Env::default(), &fd.params, owner, &fd.type_params)
    }

    /// The fact for an immutable `let` (see [`Self::binding_fact`]).
    pub(super) fn let_fact(&self, ann: Option<&TypeAnn>, value: &Node, env: &Env) -> Fact {
        self.binding_fact(ann, value, env, 0)
    }
}
