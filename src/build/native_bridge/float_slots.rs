// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Source-level proof for every slot the program declares `f64`.
//!
//! The frozen compiler reads a slot declared `f64` (a `let` annotation, a
//! parameter, a return type, a struct field) as an IEEE-754 double and converts at
//! the edge of a slot declared otherwise. The MLIR backend refuses a mismatch at
//! type check where its class checker can see one, and otherwise keeps the value it
//! was handed. A `let` annotation never reaches the IR, so the IR operand proof
//! cannot tell `let x: f64 = __mind_load_i64(p); return x;` (native 0, MLIR 7) from
//! an i64 move. Measured the same way: an i64 struct field read into an `f64` let
//! (native 0, MLIR 7), and an i64 value or integer literal handed to an `f64` let,
//! parameter or return (native builds, MLIR refuses with E2015 / E2027 / E2010).
//!
//! So the bridge proves each such slot on the source, both ways:
//! * `type.f64_slot_unproven`: a slot declared f64 takes only a value PROVEN to be
//!   an f64: a float literal, an `as f64`, a name every binding of which is an f64
//!   (an annotated one, or an unannotated `let` of a proven value), float
//!   arithmetic over those, or a call to a function declared to return one.
//!   Anything else (an intrinsic result, an integer, a field read, a branch value)
//!   is refused, never guessed.
//! * `type.f64_value_in_non_f64_slot`: a value proven an f64 may not reach a slot
//!   declared anything else.
//! * `type.f64_in_aggregate`: an f64 inside an array, tuple, generic, pointer,
//!   alias or enum payload has no slot-by-slot proof, so it is refused.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{BinOp, Literal, Module, Node, Param, Pattern, TypeAnn};
use crate::eval::narrow_scan::type_mentions_float;
use crate::type_checker::nerve_walk;

pub(super) const UNPROVEN: &str = "type.f64_slot_unproven";
pub(super) const MISMATCH: &str = "type.f64_value_in_non_f64_slot";
pub(super) const AGGREGATE: &str = "type.f64_in_aggregate";

/// A refusal: the construct name and where it was found.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct FloatSlotRefusal {
    pub construct: &'static str,
    pub detail: String,
}

type Checked = Result<(), FloatSlotRefusal>;

fn refuse(construct: &'static str, detail: String) -> Checked {
    Err(FloatSlotRefusal { construct, detail })
}

/// A scalar float type. Only `f64` reaches here (narrower widths are refused by
/// the source profile first); the others are listed so the proof never depends on
/// that order.
fn is_float(ty: &TypeAnn) -> bool {
    match ty {
        TypeAnn::ScalarF64 | TypeAnn::ScalarF32 => true,
        TypeAnn::Named(n) => matches!(
            n.as_str(),
            "f64" | "F64" | "f32" | "F32" | "f16" | "F16" | "bf16" | "BF16"
        ),
        _ => false,
    }
}

/// How one binding binds its name.
#[derive(Clone, Copy)]
enum Binding<'a> {
    /// Declared a scalar float.
    Float,
    /// Declared any other type.
    NonFloat,
    /// Unannotated, holding this value.
    Value(&'a Node),
    /// Assigned this value after its declaration.
    Assigned(&'a Node),
    /// Bound by a loop, a pattern or a tuple, from this source expression:
    /// nothing is proven about it, but it may be an f64 if its source may be.
    Opaque(&'a Node),
}

type Bindings<'a> = BTreeMap<&'a str, Vec<Binding<'a>>>;

fn bind<'a>(b: &mut Bindings<'a>, name: &'a str, how: Binding<'a>) {
    b.entry(name).or_default().push(how);
}

fn declared<'a>(ty: &TypeAnn) -> Binding<'a> {
    if is_float(ty) {
        Binding::Float
    } else {
        Binding::NonFloat
    }
}

fn bind_pattern<'a>(b: &mut Bindings<'a>, pat: &'a Pattern, src: &'a Node) {
    match pat {
        Pattern::Ident(n) => bind(b, n, Binding::Opaque(src)),
        Pattern::EnumVariant { args, .. } | Pattern::Tuple(args) => {
            args.iter().for_each(|p| bind_pattern(b, p, src));
        }
        Pattern::EnumStruct { fields, .. } => {
            fields.iter().for_each(|(_, p)| bind_pattern(b, p, src));
        }
        Pattern::Literal(_) | Pattern::Wildcard => {}
    }
}

/// Record the bindings `node` itself introduces (not its children's).
fn bindings_of<'a>(node: &'a Node, b: &mut Bindings<'a>) {
    match node {
        Node::Let {
            name, ann, value, ..
        } => match ann {
            Some(ty) => bind(b, name, declared(ty)),
            None => bind(b, name, Binding::Value(value)),
        },
        Node::Assign { name, value, .. } => bind(b, name, Binding::Assigned(value)),
        Node::Const {
            name, ty, value, ..
        } => match ty {
            Some(ty) => bind(b, name, declared(ty)),
            None => bind(b, name, Binding::Value(value)),
        },
        Node::ExternConst { name, ty, .. } => bind(b, name, declared(ty)),
        Node::LetTuple { names, value, .. } => names
            .iter()
            .for_each(|n| bind(b, n, Binding::Opaque(value))),
        Node::For {
            var, start, end, ..
        } => {
            bind(b, var, Binding::Opaque(start));
            bind(b, var, Binding::Opaque(end));
        }
        Node::ForEach {
            var, collection, ..
        } => bind(b, var, Binding::Opaque(collection)),
        Node::Match {
            scrutinee, arms, ..
        } => arms
            .iter()
            .for_each(|a| bind_pattern(b, &a.pattern, scrutinee)),
        Node::FnDef(fd, _) => fd
            .params
            .iter()
            .for_each(|p| bind(b, &p.name, declared(&p.ty))),
        Node::Closure(c, _) => c
            .params
            .iter()
            .for_each(|p| bind(b, &p.name, declared(&p.ty))),
        _ => {}
    }
}

/// One function signature the program declares.
struct Sig<'a> {
    params: &'a [Param],
    ret: Option<&'a TypeAnn>,
}

/// What the whole program declares, collected once.
#[derive(Default)]
struct Program<'a> {
    fns: BTreeMap<&'a str, Vec<Sig<'a>>>,
    /// Struct name to its fields, every declaration of the name merged.
    structs: BTreeMap<&'a str, Vec<(&'a str, &'a TypeAnn)>>,
    /// Field name to (declared a float somewhere, declared otherwise somewhere).
    field_kinds: BTreeMap<&'a str, (bool, bool)>,
    /// Bindings made by top-level items (consts, module-level lets).
    globals: Bindings<'a>,
}

impl<'a> Program<'a> {
    fn sigs(&self, callee: &str) -> Vec<&Sig<'a>> {
        let short = callee.rsplit([':', '.']).next().unwrap_or(callee);
        let mut out: Vec<&Sig<'a>> = self.fns.get(callee).into_iter().flatten().collect();
        if short != callee {
            out.extend(self.fns.get(short).into_iter().flatten());
        }
        out
    }

    fn field_types(&self, strukt: &str, field: &str) -> Vec<&'a TypeAnn> {
        let short = strukt.rsplit([':', '.']).next().unwrap_or(strukt);
        [strukt, short]
            .iter()
            .filter_map(|s| self.structs.get(s))
            .flatten()
            .filter(|(f, _)| *f == field)
            .map(|(_, t)| *t)
            .collect()
    }

    /// Is `v` proven an f64, given the names `s` already proven?
    fn proves(&self, v: &Node, s: &BTreeSet<&str>) -> bool {
        match v {
            Node::Lit(Literal::Float(_), _) => true,
            Node::Lit(Literal::Ident(n), _) => s.contains(n.as_str()),
            Node::As { ty, .. } => is_float(ty),
            Node::Paren(inner, _) | Node::Neg { operand: inner, .. } => self.proves(inner, s),
            Node::Binary {
                op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
                left,
                right,
                ..
            } => self.proves(left, s) && self.proves(right, s),
            Node::Call { callee, .. } => {
                let sigs = self.sigs(callee);
                !sigs.is_empty() && sigs.iter().all(|g| g.ret.is_some_and(is_float))
            }
            _ => false,
        }
    }

    /// The names proven f64.
    ///
    /// A name declared f64 whose every other binding is an assignment is proven
    /// outright: each of those assignments is itself an f64 slot, checked by
    /// `Unit::node` under this same assumption, so every write keeps the invariant
    /// (`x = x + 1.5`). Any other name joins in a LEAST fixpoint, only once each of
    /// its bindings is grounded in a proof (two names feeding each other prove
    /// nothing).
    fn proven_names(&self, b: &Bindings<'a>) -> BTreeSet<&'a str> {
        let mut s: BTreeSet<&'a str> = b
            .iter()
            .filter(|(_, binds)| {
                binds.iter().any(|bd| matches!(bd, Binding::Float))
                    && binds
                        .iter()
                        .all(|bd| matches!(bd, Binding::Float | Binding::Assigned(_)))
            })
            .map(|(name, _)| *name)
            .collect();
        loop {
            let joined: Vec<&'a str> = b
                .iter()
                .filter(|(name, _)| !s.contains(*name))
                .filter(|(_, binds)| {
                    binds.iter().all(|bd| match bd {
                        Binding::Float => true,
                        Binding::NonFloat | Binding::Opaque(_) => false,
                        Binding::Value(v) | Binding::Assigned(v) => self.proves(v, &s),
                    })
                })
                .map(|(name, _)| *name)
                .collect();
            if joined.is_empty() {
                return s;
            }
            s.extend(joined);
        }
    }

    /// May `v` be an f64? An OVER-approximation, used to keep an f64 out of a slot
    /// declared otherwise: a float literal or `as f64`, a name in `m`, a call to a
    /// function some declaration of which returns a float, a read of a field some
    /// struct declares a float, or any expression with such a part (an `if` /
    /// `match` / block value, arithmetic). A comparison, `!`, `&&` / `||` and an
    /// `as` to a non-float type yield an integer whatever their operands.
    fn may_be_f64(&self, v: &Node, m: &BTreeSet<&str>) -> bool {
        let returns_float =
            |name: &str| self.sigs(name).iter().any(|g| g.ret.is_some_and(is_float));
        match v {
            Node::Lit(Literal::Float(_), _) => true,
            Node::Lit(Literal::Ident(n), _) => m.contains(n.as_str()),
            Node::As { ty, .. } => is_float(ty),
            Node::Call { callee, .. } => returns_float(callee),
            Node::MethodCall { method, .. } => returns_float(method),
            Node::FieldAccess { field, .. } => self
                .field_kinds
                .get(field.as_str())
                .is_some_and(|(f, _)| *f),
            Node::Binary {
                op: BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne,
                ..
            }
            | Node::Not { .. }
            | Node::Logical { .. } => false,
            _ => {
                let mut any = false;
                nerve_walk::for_each_child(v, &mut |c| any = any || self.may_be_f64(c, m));
                any
            }
        }
    }

    /// The names that may hold an f64: a LEAST fixpoint from the names declared a
    /// float, adding any name one binding of which may be an f64.
    fn may_names(&self, b: &Bindings<'a>) -> BTreeSet<&'a str> {
        let mut m: BTreeSet<&'a str> = BTreeSet::new();
        loop {
            let joined: Vec<&'a str> = b
                .iter()
                .filter(|(name, _)| !m.contains(*name))
                .filter(|(_, binds)| {
                    binds.iter().any(|bd| match bd {
                        Binding::Float => true,
                        Binding::NonFloat => false,
                        Binding::Value(v) | Binding::Assigned(v) | Binding::Opaque(v) => {
                            self.may_be_f64(v, &m)
                        }
                    })
                })
                .map(|(name, _)| *name)
                .collect();
            if joined.is_empty() {
                return m;
            }
            m.extend(joined);
        }
    }
}

/// One function (or closure, or the module's top level) being checked.
struct Unit<'p, 'a> {
    prog: &'p Program<'a>,
    name: &'p str,
    binds: Bindings<'a>,
    proven: BTreeSet<&'a str>,
    /// The names that may hold an f64 (`Program::may_names`).
    may: BTreeSet<&'a str>,
    /// The return slot: `Some(true)` f64, `Some(false)` other (including a
    /// function with no declared return type), `None` at the module top level.
    ret: Option<bool>,
}

impl<'a> Unit<'_, 'a> {
    /// Check `value` against a slot: `Some(true)` f64, `Some(false)` other.
    fn value(&self, slot: Option<bool>, value: &Node, what: impl Fn() -> String) -> Checked {
        match slot {
            Some(true) if !self.prog.proves(value, &self.proven) => refuse(
                UNPROVEN,
                format!(
                    "in `{}`, {} is declared f64 but is given a value not proven to be an f64",
                    self.name,
                    what()
                ),
            ),
            Some(false) if self.prog.may_be_f64(value, &self.may) => refuse(
                MISMATCH,
                format!(
                    "in `{}`, {} is not declared f64 but is given an f64 value",
                    self.name,
                    what()
                ),
            ),
            _ => Ok(()),
        }
    }

    /// Check `value` against every slot in `slots`. A name or field declared f64
    /// in one place and otherwise in another (shadowing, two structs) may be
    /// writing either, so the value must satisfy both, which no value does.
    fn each(&self, slots: [bool; 2], value: &Node, what: impl Fn() -> String) -> Checked {
        for (declared, slot) in slots.into_iter().zip([true, false]) {
            if declared {
                self.value(Some(slot), value, &what)?;
            }
        }
        Ok(())
    }

    /// The slots a plain assignment to `name` may write: (some binding declares it
    /// f64, some binding declares it otherwise).
    fn assigned_slots(&self, name: &str) -> [bool; 2] {
        let binds = self.binds.get(name).map_or(&[][..], Vec::as_slice);
        [
            binds.iter().any(|b| matches!(b, Binding::Float)),
            binds.iter().any(|b| matches!(b, Binding::NonFloat)),
        ]
    }

    /// Check each argument against the callee's parameter slots. A callee the
    /// program does not declare (a frozen intrinsic, a linked std seed function)
    /// has i64 parameters only (no seed signature mentions a float), so each of
    /// its arguments is checked against a slot declared otherwise.
    fn args<'n>(&self, callee: &str, args: impl Iterator<Item = &'n Node> + Clone) -> Checked {
        let sigs = self.prog.sigs(callee);
        if sigs.is_empty() {
            for (i, a) in args.enumerate() {
                self.value(Some(false), a, || {
                    format!("argument {} of the undeclared callee `{callee}`", i + 1)
                })?;
            }
            return Ok(());
        }
        for sig in sigs {
            for (i, (p, a)) in sig.params.iter().zip(args.clone()).enumerate() {
                self.value(Some(is_float(&p.ty)), a, || {
                    format!("argument {} (`{}`) of `{callee}`", i + 1, p.name)
                })?;
            }
        }
        Ok(())
    }

    fn node(&self, node: &Node) -> Checked {
        match node {
            // A nested function or closure is its own unit.
            Node::FnDef(..) | Node::Closure(..) => return Ok(()),
            Node::Let {
                name,
                ann: Some(t),
                value,
                ..
            } => self.value(Some(is_float(t)), value, || format!("`let {name}`"))?,
            Node::Const {
                name,
                ty: Some(t),
                value,
                ..
            } => self.value(Some(is_float(t)), value, || format!("`const {name}`"))?,
            Node::Assign { name, value, .. } => {
                self.each(self.assigned_slots(name), value, || format!("`{name} =`"))?
            }
            Node::Return { value: Some(v), .. } => {
                self.value(self.ret, v, || "the return value".to_string())?
            }
            Node::Call { callee, args, .. } => self.args(callee, args.iter())?,
            Node::MethodCall {
                receiver,
                method,
                args,
                ..
            } => self.args(method, std::iter::once(receiver.as_ref()).chain(args))?,
            Node::StructLit { name, fields, .. } => {
                for f in fields {
                    for ty in self.prog.field_types(name, &f.name) {
                        self.value(Some(is_float(ty)), &f.value, || {
                            format!("field `{name}.{}`", f.name)
                        })?;
                    }
                }
            }
            Node::FieldAssign { field, value, .. } => {
                let (f, other) = self
                    .prog
                    .field_kinds
                    .get(field.as_str())
                    .copied()
                    .unwrap_or_default();
                self.each([f, other], value, || format!("field `.{field}`"))?
            }
            _ => {}
        }
        let mut res = Ok(());
        nerve_walk::for_each_child(node, &mut |c| {
            if res.is_ok() {
                res = self.node(c);
            }
        });
        res
    }

    /// The implicit return: the trailing expression of a body, through `if`,
    /// block and `match` tails.
    fn tail(&self, body: &[Node]) -> Checked {
        let Some(last) = body.last() else {
            return Ok(());
        };
        match last {
            Node::If {
                then_branch,
                else_branch,
                ..
            } => {
                self.tail(then_branch)?;
                else_branch.as_deref().map_or(Ok(()), |e| self.tail(e))
            }
            Node::Block { stmts, .. } | Node::Region { body: stmts, .. } => self.tail(stmts),
            Node::Match { arms, .. } => arms.iter().try_for_each(|a| match &a.body {
                Node::Block { stmts, .. } => self.tail(stmts),
                expr => self.tail(std::slice::from_ref(expr)),
            }),
            Node::Let { .. }
            | Node::LetTuple { .. }
            | Node::Assign { .. }
            | Node::Return { .. }
            | Node::While { .. }
            | Node::For { .. }
            | Node::ForEach { .. }
            | Node::Break { .. }
            | Node::Continue { .. }
            | Node::Print { .. }
            | Node::Assert { .. }
            | Node::IndexAssign { .. }
            | Node::FieldAssign { .. }
            | Node::FnDef(..)
            | Node::StructDef { .. }
            | Node::EnumDef { .. }
            | Node::TraitDef { .. }
            | Node::ImplBlock { .. }
            | Node::Import { .. }
            | Node::Export { .. }
            | Node::TypeAlias { .. }
            | Node::Const { .. }
            | Node::ExternConst { .. }
            | Node::ExternBlock { .. } => Ok(()),
            expr => self.value(self.ret, expr, || "the trailing value".to_string()),
        }
    }
}

/// Refuse a float type nested where no slot proof is made.
fn aggregate(ty: &TypeAnn, what: impl Fn() -> String) -> Checked {
    if !is_float(ty) && type_mentions_float(ty) {
        return refuse(
            AGGREGATE,
            format!("{} holds an f64 inside an aggregate or alias", what()),
        );
    }
    Ok(())
}

/// Every type annotation in `node` itself that could hide an f64 slot.
fn annotations(node: &Node) -> Checked {
    let params = |ps: &[Param], ret: Option<&TypeAnn>, who: &str| -> Checked {
        for p in ps {
            aggregate(&p.ty, || format!("parameter `{}` of `{who}`", p.name))?;
        }
        ret.map_or(Ok(()), |t| {
            aggregate(t, || format!("the return type of `{who}`"))
        })
    };
    match node {
        Node::Let {
            name, ann: Some(t), ..
        } => aggregate(t, || format!("`let {name}`")),
        Node::Const {
            name, ty: Some(t), ..
        }
        | Node::ExternConst { name, ty: t, .. } => aggregate(t, || format!("`const {name}`")),
        Node::As { ty, .. } => aggregate(ty, || "an `as` target".to_string()),
        Node::FnDef(fd, _) => params(&fd.params, fd.ret_type.as_ref(), &fd.name),
        Node::Closure(c, _) => params(&c.params, c.ret_type.as_ref(), "a closure"),
        Node::ExternBlock { fns, .. } => fns
            .iter()
            .try_for_each(|f| params(&f.params, f.ret_type.as_ref(), &f.name)),
        Node::StructDef { name, fields, .. } => fields
            .iter()
            .try_for_each(|f| aggregate(&f.ty, || format!("field `{name}.{}`", f.name))),
        Node::EnumDef { name, variants, .. } => {
            match variants
                .iter()
                .find(|v| v.payload.iter().any(type_mentions_float))
            {
                Some(v) => refuse(
                    AGGREGATE,
                    format!("variant `{name}::{}` carries an f64 payload", v.name),
                ),
                None => Ok(()),
            }
        }
        Node::TypeAlias { name, target, .. } if type_mentions_float(target) => {
            refuse(AGGREGATE, format!("type alias `{name}` names an f64"))
        }
        _ => Ok(()),
    }
}

/// Collect the program's declarations; refuse an f64 in an aggregate on the way.
fn collect<'a>(module: &'a Module) -> Result<Program<'a>, FloatSlotRefusal> {
    let mut prog = Program::default();
    let mut res = Ok(());
    for item in &module.items {
        nerve_walk::walk(item, &mut |n| {
            if res.is_err() {
                return;
            }
            res = annotations(n);
            match n {
                Node::FnDef(fd, _) => prog.fns.entry(&fd.name).or_default().push(Sig {
                    params: &fd.params,
                    ret: fd.ret_type.as_ref(),
                }),
                Node::ExternBlock { fns, .. } => {
                    for f in fns {
                        prog.fns.entry(&f.name).or_default().push(Sig {
                            params: &f.params,
                            ret: f.ret_type.as_ref(),
                        });
                    }
                }
                Node::StructDef { name, fields, .. } => {
                    for f in fields {
                        prog.structs.entry(name).or_default().push((&f.name, &f.ty));
                        let kind = prog.field_kinds.entry(&f.name).or_default();
                        if is_float(&f.ty) {
                            kind.0 = true;
                        } else {
                            kind.1 = true;
                        }
                    }
                }
                _ => {}
            }
        });
    }
    res?;
    // Top-level bindings, through transparent `module` blocks.
    fn top<'a>(items: &'a [Node], b: &mut Bindings<'a>) {
        for item in items {
            match item {
                Node::Block { stmts, .. } => top(stmts, b),
                Node::FnDef(..) => {}
                other => bindings_of(other, b),
            }
        }
    }
    top(&module.items, &mut prog.globals);
    Ok(prog)
}

/// Admit `module` only if every slot it declares f64 is fed a proven f64, no
/// proven f64 reaches a slot declared otherwise, and no f64 hides in an aggregate.
pub(super) fn admit_f64_slots(module: &Module) -> Checked {
    let prog = collect(module)?;
    let check = |name: &str, params: &[Param], ret: Option<&TypeAnn>, body: &[Node]| {
        // A function's names: the program's top-level ones plus every binding in
        // its body (nested ones included, which only ever withholds a proof).
        let mut binds = prog.globals.clone();
        for p in params {
            bind(&mut binds, &p.name, declared(&p.ty));
        }
        for stmt in body {
            nerve_walk::walk(stmt, &mut |n| bindings_of(n, &mut binds));
        }
        let unit = Unit {
            prog: &prog,
            name,
            proven: prog.proven_names(&binds),
            may: prog.may_names(&binds),
            binds,
            // No declared return type is an i64 return slot (the MLIR backend
            // types it i64), never an unchecked one.
            ret: Some(ret.is_some_and(is_float)),
        };
        body.iter().try_for_each(|s| unit.node(s))?;
        unit.tail(body)
    };
    // The top level: consts and module-level lets, against the global names only.
    let top = Unit {
        prog: &prog,
        name: "<module>",
        proven: prog.proven_names(&prog.globals),
        may: prog.may_names(&prog.globals),
        binds: prog.globals.clone(),
        ret: None,
    };
    module.items.iter().try_for_each(|i| top.node(i))?;
    // Every function and closure, wherever it is declared.
    let mut res = Ok(());
    for item in &module.items {
        nerve_walk::walk(item, &mut |n| {
            if res.is_err() {
                return;
            }
            res = match n {
                Node::FnDef(fd, _) => check(&fd.name, &fd.params, fd.ret_type.as_ref(), &fd.body),
                Node::Closure(c, _) => check("<closure>", &c.params, c.ret_type.as_ref(), &c.body),
                _ => Ok(()),
            };
        });
    }
    res
}

#[cfg(test)]
mod tests;
