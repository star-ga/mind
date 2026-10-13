// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Bool parameters are the one integer the two backends read at different widths.
//!
//! A `bool` parameter travels in an i64 ABI slot. The frozen compiler uses the
//! whole word (nonzero is true, `!b` is `b == 0`, arithmetic is i64). The MLIR
//! backend keeps the parameter as `ScalarBool` and, for an operator whose result
//! is i1, truncates each bool operand to its low bit first. A direct `if` / `while`
//! and a returned parameter stay nonzero / raw on both, and so does a comparison
//! or arithmetic against a non-literal i64 (that widens). Measured on
//! `fn f(b: bool)` fed `2`, `256`, `-1` and `3`:
//!
//! * `if b`, `return b`, `b == <non-literal>`, `b & 0`, `b & 1` agree for every i64.
//! * `!b` and `b == 0` / `b == 1` agree only when the reaching value is 0 or 1.
//! * `b == 2`, `b + 10`, `b + b` disagree even for 0 and 1.
//!
//! A bool *let* is not this slot: it stays an i64 on both backends. Only a
//! parameter, and an unannotated name that is a pure copy of one, is checked.
//! An uncalled function is not checked: no value reaches it, so the program
//! result matches.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{BinOp, BitOp, Literal, Module, Node, Param, TypeAnn};
use crate::type_checker::nerve_walk;

pub(super) const I1: &str = "type.bool_param_i1";
pub(super) const ARITH: &str = "type.bool_param_arith";

/// A refusal: the construct name and where it was found.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct BoolParamRefusal {
    pub construct: &'static str,
    pub detail: String,
}

/// How a use of a bool parameter disagrees with the MLIR backend.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Agrees when the reaching value is 0 or 1, and not otherwise.
    WhenCanonical,
    /// An i1 operator that disagrees even for 0 and 1.
    AlwaysI1,
    /// Arithmetic that disagrees even for 0 and 1.
    AlwaysArith,
}

struct Func<'a> {
    name: &'a str,
    params: &'a [Param],
    body: &'a [Node],
    /// Parameter index of each `bool` parameter.
    bool_params: Vec<usize>,
    /// Name → parameter index. Parameters, plus an unannotated `let` that is a
    /// pure copy of one (a bool annotation is a fresh i64, not this slot).
    aliases: BTreeMap<&'a str, usize>,
    /// The single `let` of a name, when it has exactly one.
    lets: BTreeMap<&'a str, &'a Node>,
    /// Sole unannotated `let`s, which may be pure copies of a bool parameter.
    alias_ok: BTreeSet<&'a str>,
}

struct Use {
    func: usize,
    param: usize,
    kind: Kind,
}

struct Feed<'a> {
    caller: usize,
    callee: usize,
    param: usize,
    arg: &'a Node,
}

fn is_bool_ty(ty: &TypeAnn) -> bool {
    match ty {
        TypeAnn::ScalarBool => true,
        TypeAnn::Named(n) => n == "bool",
        _ => false,
    }
}

fn strip(node: &Node) -> &Node {
    let mut cur = node;
    while let Node::Paren(inner, _) = cur {
        cur = inner;
    }
    cur
}

fn int_lit(node: &Node) -> Option<i64> {
    match strip(node) {
        Node::Lit(Literal::Int(v), _) => Some(*v),
        Node::Neg { operand, .. } => int_lit(operand).map(|v| v.wrapping_neg()),
        _ => None,
    }
}

fn alias_of(func: &Func<'_>, node: &Node) -> Option<usize> {
    match strip(node) {
        Node::Lit(Literal::Ident(name), _) => func.aliases.get(name.as_str()).copied(),
        _ => None,
    }
}

fn contains_alias(func: &Func<'_>, node: &Node) -> bool {
    if alias_of(func, node).is_some() {
        return true;
    }
    let mut found = false;
    nerve_walk::for_each_child(node, &mut |child| {
        found = found || contains_alias(func, child);
    });
    found
}

/// The value a body produces when control falls off the end, if it is an
/// expression rather than a statement.
fn tail_expr(body: &[Node]) -> Option<&Node> {
    let last = body.last()?;
    match last {
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
        | Node::ExternBlock { .. } => None,
        Node::Block { stmts, .. } | Node::Region { body: stmts, .. } => tail_expr(stmts),
        other => Some(other),
    }
}

fn walk_returns<'a>(node: &'a Node, out: &mut Vec<&'a Node>) {
    if let Node::Return { value: Some(v), .. } = node {
        out.push(v);
    }
    if matches!(node, Node::FnDef(..) | Node::Closure(..)) {
        return;
    }
    nerve_walk::for_each_child(node, &mut |child| walk_returns(child, out));
}

fn return_exprs(body: &[Node]) -> Vec<&Node> {
    let mut out = Vec::new();
    for stmt in body {
        walk_returns(stmt, &mut out);
    }
    if let Some(tail) = tail_expr(body) {
        out.push(tail);
    }
    out
}

fn collect_funcs(module: &Module) -> Vec<Func<'_>> {
    let mut funcs = Vec::new();
    for item in &module.items {
        nerve_walk::walk(item, &mut |node| {
            let (name, params, body) = match node {
                Node::FnDef(fd, _) => (fd.name.as_str(), fd.params.as_slice(), fd.body.as_slice()),
                Node::Closure(c, _) => ("<closure>", c.params.as_slice(), c.body.as_slice()),
                _ => return,
            };
            let bool_params: Vec<usize> = params
                .iter()
                .enumerate()
                .filter(|(_, p)| is_bool_ty(&p.ty))
                .map(|(i, _)| i)
                .collect();
            let mut aliases = BTreeMap::new();
            for &i in &bool_params {
                aliases.insert(params[i].name.as_str(), i);
            }
            funcs.push(Func {
                name,
                params,
                body,
                bool_params,
                aliases,
                lets: BTreeMap::new(),
                alias_ok: BTreeSet::new(),
            });
        });
    }
    for func in &mut funcs {
        bind_lets(func);
    }
    funcs
}

/// Record sole `let`s and unannotated copies of a bool parameter.
fn bind_lets(func: &mut Func<'_>) {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut values: BTreeMap<&str, (&Node, bool)> = BTreeMap::new();
    fn walk<'a>(
        node: &'a Node,
        counts: &mut BTreeMap<&'a str, usize>,
        values: &mut BTreeMap<&'a str, (&'a Node, bool)>,
    ) {
        if let Node::Let {
            name, ann, value, ..
        } = node
        {
            *counts.entry(name.as_str()).or_default() += 1;
            values.insert(name.as_str(), (value, ann.is_none()));
        }
        if matches!(node, Node::FnDef(..) | Node::Closure(..)) {
            return;
        }
        nerve_walk::for_each_child(node, &mut |child| walk(child, counts, values));
    }
    for stmt in func.body {
        walk(stmt, &mut counts, &mut values);
    }
    for (name, n) in &counts {
        if *n != 1 {
            continue;
        }
        let Some((value, alias_ok)) = values.get(name) else {
            continue;
        };
        func.lets.insert(name, value);
        if *alias_ok {
            func.alias_ok.insert(name);
            if let Node::Lit(Literal::Ident(src), _) = strip(value) {
                if let Some(&pi) = func.aliases.get(src.as_str()) {
                    func.aliases.insert(name, pi);
                }
            }
        }
    }
    // `let d = c` where `c` was itself a copy. A copy chain is acyclic in source
    // order; the fixpoint covers either order. Only an unannotated sole `let`
    // is a copy — a bool annotation is a fresh i64 slot.
    loop {
        let mut added = false;
        let extra: Vec<(&str, usize)> = func
            .alias_ok
            .iter()
            .filter(|name| !func.aliases.contains_key(*name))
            .filter_map(|name| {
                let value = func.lets.get(name)?;
                let Node::Lit(Literal::Ident(src), _) = strip(value) else {
                    return None;
                };
                func.aliases.get(src.as_str()).map(|pi| (*name, *pi))
            })
            .collect();
        for (name, pi) in extra {
            func.aliases.insert(name, pi);
            added = true;
        }
        if !added {
            break;
        }
    }
}

fn record(uses: &mut Vec<Use>, func: usize, param: usize, kind: Kind) {
    uses.push(Use { func, param, kind });
}

fn note_operand(
    uses: &mut Vec<Use>,
    func_i: usize,
    param: Option<usize>,
    other: &Node,
    func: &Func<'_>,
    kind_of: impl Fn(Option<i64>, bool) -> Option<Kind>,
) {
    let Some(param) = param else {
        return;
    };
    let other_param = alias_of(func, other).is_some();
    if let Some(kind) = kind_of(int_lit(other), other_param) {
        record(uses, func_i, param, kind);
    }
}

fn classify_binary(uses: &mut Vec<Use>, func_i: usize, func: &Func<'_>, node: &Node) {
    let Node::Binary {
        op, left, right, ..
    } = node
    else {
        return;
    };
    let kind_of = |lit: Option<i64>, other_param: bool| -> Option<Kind> {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                if lit.is_some() || other_param {
                    Some(Kind::AlwaysArith)
                } else {
                    None
                }
            }
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => match lit {
                Some(0) | Some(1) => Some(Kind::WhenCanonical),
                Some(_) => Some(Kind::AlwaysI1),
                None if other_param => Some(Kind::WhenCanonical),
                None => None,
            },
        }
    };
    note_operand(uses, func_i, alias_of(func, left), right, func, kind_of);
    note_operand(uses, func_i, alias_of(func, right), left, func, kind_of);
}

fn classify_bit(uses: &mut Vec<Use>, func_i: usize, func: &Func<'_>, node: &Node) {
    let Node::Bitwise {
        op, left, right, ..
    } = node
    else {
        return;
    };
    let kind_of = |lit: Option<i64>, other_param: bool| -> Option<Kind> {
        match op {
            BitOp::And => match lit {
                Some(0) | Some(1) => None,
                Some(_) => Some(Kind::WhenCanonical),
                None if other_param => Some(Kind::WhenCanonical),
                None => None,
            },
            BitOp::Or | BitOp::Xor => match lit {
                Some(0) | Some(1) => Some(Kind::WhenCanonical),
                Some(_) => Some(Kind::AlwaysI1),
                None if other_param => Some(Kind::WhenCanonical),
                None => None,
            },
            BitOp::Shl | BitOp::Shr => match lit {
                Some(0) => Some(Kind::WhenCanonical),
                Some(_) => Some(Kind::AlwaysI1),
                None if other_param => Some(Kind::AlwaysI1),
                None => None,
            },
        }
    };
    note_operand(uses, func_i, alias_of(func, left), right, func, kind_of);
    note_operand(uses, func_i, alias_of(func, right), left, func, kind_of);
}

fn classify_node(uses: &mut Vec<Use>, func_i: usize, func: &Func<'_>, node: &Node) {
    match node {
        Node::Not { operand, .. } => {
            if let Some(param) = alias_of(func, operand) {
                record(uses, func_i, param, Kind::WhenCanonical);
            }
        }
        Node::Binary { .. } => classify_binary(uses, func_i, func, node),
        Node::Bitwise { .. } => classify_bit(uses, func_i, func, node),
        _ => {}
    }
}

fn walk_uses(uses: &mut Vec<Use>, func_i: usize, func: &Func<'_>, node: &Node) {
    classify_node(uses, func_i, func, node);
    if matches!(node, Node::FnDef(..) | Node::Closure(..)) {
        return;
    }
    nerve_walk::for_each_child(node, &mut |child| walk_uses(uses, func_i, func, child));
}

fn resolve<'a>(funcs: &'a [Func<'a>], name: &str) -> Vec<usize> {
    let short = name.rsplit([':', '.']).next().unwrap_or(name);
    funcs
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name == name || f.name == short)
        .map(|(i, _)| i)
        .collect()
}

fn walk_calls<'a>(
    feeds: &mut Vec<Feed<'a>>,
    edges: &mut Vec<(usize, usize)>,
    funcs: &'a [Func<'a>],
    caller: usize,
    node: &'a Node,
) {
    match node {
        Node::Call { callee, args, .. } => {
            push_calls(feeds, edges, funcs, caller, callee, args.iter());
        }
        Node::MethodCall {
            receiver,
            method,
            args,
            ..
        } => {
            push_calls(
                feeds,
                edges,
                funcs,
                caller,
                method,
                std::iter::once(receiver.as_ref()).chain(args.iter()),
            );
        }
        _ => {}
    }
    if matches!(node, Node::FnDef(..) | Node::Closure(..)) {
        return;
    }
    nerve_walk::for_each_child(node, &mut |child| {
        walk_calls(feeds, edges, funcs, caller, child);
    });
}

fn push_calls<'a>(
    feeds: &mut Vec<Feed<'a>>,
    edges: &mut Vec<(usize, usize)>,
    funcs: &'a [Func<'a>],
    caller: usize,
    callee: &str,
    args: impl Iterator<Item = &'a Node> + Clone,
) {
    for ci in resolve(funcs, callee) {
        edges.push((caller, ci));
        for &pi in &funcs[ci].bool_params {
            if let Some(arg) = args.clone().nth(pi) {
                feeds.push(Feed {
                    caller,
                    callee: ci,
                    param: pi,
                    arg,
                });
            }
        }
    }
}

fn reachable(funcs: &[Func<'_>], edges: &[(usize, usize)]) -> BTreeSet<usize> {
    let mains: Vec<usize> = funcs
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name == "main")
        .map(|(i, _)| i)
        .collect();
    let roots: Vec<usize> = if mains.is_empty() {
        (0..funcs.len()).collect()
    } else {
        mains
    };
    let mut seen = BTreeSet::new();
    let mut stack = roots;
    while let Some(i) = stack.pop() {
        if !seen.insert(i) {
            continue;
        }
        for &(caller, callee) in edges {
            if caller == i {
                stack.push(callee);
            }
        }
    }
    seen
}

fn is_canon(
    funcs: &[Func<'_>],
    func_i: usize,
    node: &Node,
    safe: &BTreeSet<(usize, usize)>,
    stack: &mut Vec<(usize, String)>,
    depth: u32,
) -> bool {
    if depth > 64 {
        return false;
    }
    let func = &funcs[func_i];
    match strip(node) {
        Node::Lit(Literal::Int(0 | 1), _) => true,
        Node::Neg { .. } => false,
        Node::Not { operand, .. } => {
            if let Some(pi) = alias_of(func, operand) {
                safe.contains(&(func_i, pi))
            } else {
                !contains_alias(func, operand)
            }
        }
        Node::Binary {
            op: BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge,
            left,
            right,
            ..
        } => cmp_canon(funcs, func_i, left, right, safe, stack, depth),
        Node::Binary { .. } | Node::Bitwise { .. } | Node::Logical { .. } => false,
        Node::Lit(Literal::Ident(name), _) => ident_canon(funcs, func_i, name, safe, stack, depth),
        Node::Call { callee, .. } => {
            let targets = resolve(funcs, callee);
            !targets.is_empty()
                && targets
                    .iter()
                    .all(|&t| returns_canon(funcs, t, safe, stack, depth))
        }
        Node::If {
            then_branch,
            else_branch: Some(else_branch),
            ..
        } => {
            branch_canon(funcs, func_i, then_branch, safe, stack, depth)
                && branch_canon(funcs, func_i, else_branch, safe, stack, depth)
        }
        Node::Block { stmts, .. } | Node::Region { body: stmts, .. } => {
            tail_expr(stmts).is_some_and(|e| is_canon(funcs, func_i, e, safe, stack, depth + 1))
        }
        Node::Match { arms, .. } => arms.iter().all(|arm| match &arm.body {
            Node::Block { stmts, .. } => {
                tail_expr(stmts).is_some_and(|e| is_canon(funcs, func_i, e, safe, stack, depth + 1))
            }
            other => is_canon(funcs, func_i, other, safe, stack, depth + 1),
        }),
        Node::Paren(inner, _) => is_canon(funcs, func_i, inner, safe, stack, depth + 1),
        _ => false,
    }
}

fn branch_canon(
    funcs: &[Func<'_>],
    func_i: usize,
    body: &[Node],
    safe: &BTreeSet<(usize, usize)>,
    stack: &mut Vec<(usize, String)>,
    depth: u32,
) -> bool {
    // An explicit `return` inside the branch, otherwise the branch tail.
    let mut rets = Vec::new();
    for stmt in body {
        walk_returns(stmt, &mut rets);
    }
    if rets.is_empty() {
        tail_expr(body).is_some_and(|e| is_canon(funcs, func_i, e, safe, stack, depth + 1))
    } else {
        rets.iter()
            .all(|e| is_canon(funcs, func_i, e, safe, stack, depth + 1))
    }
}

fn cmp_canon(
    funcs: &[Func<'_>],
    func_i: usize,
    left: &Node,
    right: &Node,
    safe: &BTreeSet<(usize, usize)>,
    _stack: &mut Vec<(usize, String)>,
    _depth: u32,
) -> bool {
    let func = &funcs[func_i];
    match (alias_of(func, left), alias_of(func, right)) {
        (None, None) => true,
        (Some(p), None) => one_side_canon(func_i, p, right, safe),
        (None, Some(p)) => one_side_canon(func_i, p, left, safe),
        (Some(p), Some(q)) => safe.contains(&(func_i, p)) && safe.contains(&(func_i, q)),
    }
}

/// A comparison of a bool parameter with `other`. A non-literal widens to i64
/// and agrees for every word; a literal agrees for a canonical parameter only
/// when the literal is 0 or 1.
fn one_side_canon(
    func_i: usize,
    param: usize,
    other: &Node,
    safe: &BTreeSet<(usize, usize)>,
) -> bool {
    match int_lit(other) {
        None => true,
        Some(0) | Some(1) => safe.contains(&(func_i, param)),
        Some(_) => false,
    }
}

fn ident_canon(
    funcs: &[Func<'_>],
    func_i: usize,
    name: &str,
    safe: &BTreeSet<(usize, usize)>,
    stack: &mut Vec<(usize, String)>,
    depth: u32,
) -> bool {
    let func = &funcs[func_i];
    if let Some(&pi) = func.aliases.get(name) {
        return safe.contains(&(func_i, pi));
    }
    let Some(value) = func.lets.get(name) else {
        return false;
    };
    if stack.iter().any(|(f, n)| *f == func_i && n == name) {
        return false;
    }
    stack.push((func_i, name.to_string()));
    let ok = is_canon(funcs, func_i, value, safe, stack, depth + 1);
    stack.pop();
    ok
}

fn returns_canon(
    funcs: &[Func<'_>],
    func_i: usize,
    safe: &BTreeSet<(usize, usize)>,
    stack: &mut Vec<(usize, String)>,
    depth: u32,
) -> bool {
    if stack.iter().any(|(f, n)| *f == func_i && n.is_empty()) {
        return false;
    }
    stack.push((func_i, String::new()));
    let rets = return_exprs(funcs[func_i].body);
    let ok = !rets.is_empty()
        && rets
            .iter()
            .all(|e| is_canon(funcs, func_i, e, safe, stack, depth + 1));
    stack.pop();
    ok
}

/// Parameters that stay canonical: every reachable call passes a value that is
/// 0 or 1 on both backends. A parameter with no call (a root, or an uncalled
/// function) is not in the set.
fn safe_params(
    funcs: &[Func<'_>],
    feeds: &[Feed<'_>],
    reach: &BTreeSet<usize>,
) -> BTreeSet<(usize, usize)> {
    let live: Vec<&Feed<'_>> = feeds.iter().filter(|f| reach.contains(&f.caller)).collect();
    let mut safe: BTreeSet<(usize, usize)> = live.iter().map(|f| (f.callee, f.param)).collect();
    let mut stack = Vec::new();
    loop {
        let dropped: Vec<(usize, usize)> = safe
            .iter()
            .copied()
            .filter(|&key| {
                live.iter()
                    .filter(|f| (f.callee, f.param) == key)
                    .any(|f| !is_canon(funcs, f.caller, f.arg, &safe, &mut stack, 0))
            })
            .collect();
        if dropped.is_empty() {
            return safe;
        }
        for key in dropped {
            safe.remove(&key);
        }
    }
}

fn refuse(construct: &'static str, detail: String) -> Result<(), BoolParamRefusal> {
    Err(BoolParamRefusal { construct, detail })
}

/// Admit `module` unless a bool parameter is used where the frozen compiler and
/// the MLIR backend disagree.
pub(super) fn admit_bool_params(module: &Module) -> Result<(), BoolParamRefusal> {
    let funcs = collect_funcs(module);
    let mut uses = Vec::new();
    let mut feeds = Vec::new();
    let mut edges = Vec::new();
    for (i, func) in funcs.iter().enumerate() {
        for stmt in func.body {
            walk_uses(&mut uses, i, func, stmt);
            walk_calls(&mut feeds, &mut edges, &funcs, i, stmt);
        }
    }
    let reach = reachable(&funcs, &edges);
    let safe = safe_params(&funcs, &feeds, &reach);
    for site in &uses {
        if !reach.contains(&site.func) {
            continue;
        }
        let bad = match site.kind {
            Kind::AlwaysArith => true,
            Kind::AlwaysI1 => true,
            Kind::WhenCanonical => !safe.contains(&(site.func, site.param)),
        };
        if !bad {
            continue;
        }
        let func = &funcs[site.func];
        let pname = func
            .params
            .get(site.param)
            .map(|p| p.name.as_str())
            .unwrap_or("?");
        let (construct, why) = match site.kind {
            Kind::AlwaysArith => (
                ARITH,
                "arithmetic on a bool parameter is i64 natively and i1 under MLIR",
            ),
            Kind::AlwaysI1 => (
                I1,
                "this i1 use of a bool parameter disagrees with MLIR even for 0 and 1",
            ),
            Kind::WhenCanonical => (
                I1,
                "this i1 use of a bool parameter agrees with MLIR only when the value is 0 or 1",
            ),
        };
        return refuse(
            construct,
            format!("in `{}`, parameter `{pname}`: {why}", func.name),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admit(src: &str) -> Result<(), &'static str> {
        let ast = crate::parser::parse(src).expect("parse");
        admit_bool_params(&ast).map_err(|r| r.construct)
    }

    #[test]
    fn a_noncanonical_i64_fed_to_an_i1_use_is_refused() {
        for src in [
            "fn f(b: bool) -> i64 { if !b { return 1; } return 2; } \
             fn main() -> i64 { return f(2); }",
            "fn f(b: bool) -> i64 { if !b { return 1; } return 2; } \
             fn main() -> i64 { let a: i64 = 2; return f(a); }",
            "fn g(a: i64) -> bool { return a; } \
             fn f(b: bool) -> i64 { if !b { return 1; } return 2; } \
             fn main() -> i64 { return f(g(2)); }",
            "fn f(b: bool) -> i64 { if b == true { return 1; } return 2; } \
             fn main() -> i64 { return f(3); }",
            "fn f(b: bool) -> i64 { if b == 1 { return 1; } return 2; } \
             fn main() -> i64 { return f(0 - 1); }",
            "fn f(b: bool) -> i64 { if b != false { return 1; } return 2; } \
             fn main() -> i64 { return f(2); }",
            "fn f(b: bool) -> i64 { if b == 2 { return 1; } return 2; } \
             fn main() -> i64 { return f(1); }",
        ] {
            assert_eq!(admit(src), Err(I1), "{src}");
        }
    }

    #[test]
    fn arithmetic_on_a_bool_parameter_is_refused_even_for_a_canonical_argument() {
        for src in [
            "fn f(b: bool) -> i64 { return b + 10; } fn main() -> i64 { return f(true); }",
            "fn f(b: bool) -> i64 { return b + 10; } fn main() -> i64 { return f(1); }",
            "fn f(a: bool, b: bool) -> i64 { return a + b; } \
             fn main() -> i64 { return f(true, false); }",
        ] {
            assert_eq!(admit(src), Err(ARITH), "{src}");
        }
    }

    #[test]
    fn uses_that_agree_with_mlir_stay_admitted() {
        for src in [
            // Nonzero `if` agrees for every i64, including a non-canonical one.
            "fn f(b: bool) -> i64 { if b { return 1; } return 2; } \
             fn main() -> i64 { return f(2); }",
            "fn f(b: bool) -> i64 { if b { return 1; } return 2; } \
             fn main() -> i64 { return f(256); }",
            // `!` of a real 0 or 1 agrees.
            "fn f(b: bool) -> i64 { if !b { return 1; } return 2; } \
             fn main() -> i64 { return f(true); }",
            "fn f(b: bool) -> i64 { if !b { return 1; } return 2; } \
             fn main() -> i64 { return f(0); }",
            // A returned parameter is the raw word on both.
            "fn f(b: bool) -> i64 { return b; } \
             fn main() -> i64 { let v = f(2); if v { return 1; } return 2; }",
            // A non-literal compare widens.
            "fn f(b: bool) -> i64 { let t: bool = true; if b == t { return 1; } return 2; } \
             fn main() -> i64 { return f(2); }",
            // `b & 1` is the low bit on both.
            "fn f(b: bool) -> i64 { if b & true { return 1; } return 2; } \
             fn main() -> i64 { return f(2); }",
            // A comparison that yields a real bool is a canonical argument.
            "fn g(a: i64) -> bool { return a != 0; } \
             fn f(b: bool) -> i64 { if !b { return 1; } return 2; } \
             fn main() -> i64 { return f(g(2)); }",
            // No value reaches an uncalled function.
            "fn f(b: bool) -> i64 { if !b { return 1; } return 2; } \
             fn main() -> i64 { return 0; }",
            // An unannotated copy is the same parameter; a canonical feed still admits.
            "fn f(b: bool) -> i64 { let c = b; if !c { return 1; } return 2; } \
             fn main() -> i64 { return f(1); }",
        ] {
            assert_eq!(admit(src), Ok(()), "{src}");
        }
    }
}
