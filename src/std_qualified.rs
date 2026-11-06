// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Qualified calls into the world-reading std modules.
//!
//! `dns_resolve`, `tcp_connect_host`, `tcp_connect_host_timeout` and
//! `entropy_fill` read the world only when the call's qualifier resolves to
//! `std.dns` or `std.entropy`. A user function of the same bare name is
//! deterministic. The lowered IR keeps only the bare name (the qualifier lives
//! on the call span, in `SourceSpelling`), so this module reads the source.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::ast::{FnDefData, Module, Node, Span};
use crate::ir::{IRModule, Instr};

/// `(module path, public function)` rows. A call matches only both.
const ROWS: &[(&str, &str)] = &[
    ("std.dns", "dns_resolve"),
    ("std.dns", "tcp_connect_host"),
    ("std.dns", "tcp_connect_host_timeout"),
    ("std.entropy", "entropy_fill"),
];

/// Whether `name` is a world-reading public function of `module`.
pub fn is_row(module: &str, name: &str) -> bool {
    ROWS.iter().any(|(m, n)| *m == module && *n == name)
}

/// Qualifier spellings of one parsed module, and the qualifier recorded for
/// each desugared call span.
struct QualMap {
    /// Qualifier as written (`dns`, an alias, or the dotted path) to the
    /// module paths it can name. More than one path is fail-closed: a call
    /// matches when any of them is a world row.
    by_qual: BTreeMap<String, Vec<String>>,
    calls: BTreeMap<Span, String>,
}

impl QualMap {
    fn from_module(module: &Module) -> Self {
        let mut by_qual: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut imports = Vec::new();
        for item in &module.items {
            collect_imports(item, &mut imports);
        }
        for node in imports {
            let Node::Import { path, alias, .. } = node else {
                continue;
            };
            let dotted = path.join(".");
            let last = path.last().map(String::as_str).unwrap_or("");
            if let Some(alias) = alias {
                add_qual(&mut by_qual, alias, &dotted);
                add_qual(&mut by_qual, last, &dotted);
            } else {
                add_qual(&mut by_qual, last, &dotted);
            }
            add_qual(&mut by_qual, &dotted, &dotted);
        }
        let mut calls = BTreeMap::new();
        for (span, qual) in &module.spelling.call_qualifiers {
            calls.insert(*span, qual.clone());
        }
        Self { by_qual, calls }
    }

    fn call_is_world(&self, span: Span, name: &str) -> bool {
        let Some(qual) = self.calls.get(&span) else {
            return false;
        };
        match self.by_qual.get(qual.as_str()) {
            Some(paths) => paths.iter().any(|path| is_row(path, name)),
            None => is_row(qual, name),
        }
    }
}

fn add_qual(map: &mut BTreeMap<String, Vec<String>>, qual: &str, path: &str) {
    if qual.is_empty() {
        return;
    }
    let entry = map.entry(qual.to_string()).or_default();
    if !entry.iter().any(|have| have == path) {
        entry.push(path.to_string());
    }
}

fn collect_imports<'a>(node: &'a Node, out: &mut Vec<&'a Node>) {
    if matches!(node, Node::Import { .. }) {
        out.push(node);
    }
    walk(node, true, &mut |child| collect_imports(child, out));
}

fn collect_fns<'a>(items: &'a [Node], out: &mut Vec<&'a FnDefData>) {
    for item in items {
        match item {
            Node::FnDef(fd, _) => out.push(fd),
            Node::Block { stmts, .. } => collect_fns(stmts, out),
            Node::ImplBlock { methods, .. } => collect_fns(methods, out),
            _ => {}
        }
    }
}

/// The first qualified std-world call in `body`. Nested functions are their
/// own scan: a caller is not blamed for a callee's body. The project walker
/// reaches that callee on its own.
fn note_call(node: &Node, map: &QualMap, found: &mut Option<String>) {
    if found.is_some() {
        return;
    }
    if let Node::Call { callee, span, .. } = node {
        if map.call_is_world(*span, callee) {
            *found = Some(callee.clone());
        }
    }
}

fn body_offender(body: &[Node], map: &QualMap) -> Option<String> {
    let mut found = None;
    for stmt in body {
        // `walk` visits children, not the node it is given.
        note_call(stmt, map, &mut found);
        walk(stmt, false, &mut |node| note_call(node, map, &mut found));
    }
    found
}

/// The first qualified std-world call in any function of `source`.
/// `None` when the source does not parse or holds no such call. Whole-module,
/// matching the entry IR scan: an uncalled function still refuses the module.
pub fn module_offender(source: &str) -> Option<String> {
    let module = crate::parser::parse(source).ok()?;
    let map = QualMap::from_module(&module);
    let mut fns = Vec::new();
    collect_fns(&module.items, &mut fns);
    fns.iter().find_map(|fd| body_offender(&fd.body, &map))
}

/// The qualified std-world call made directly by `func` in `source`.
/// `None` when `func` is absent, the source does not parse, or that function's
/// own body makes no such call. Other functions of the module do not count.
pub fn reached_offender(source: &str, func: &str) -> Option<String> {
    let module = crate::parser::parse(source).ok()?;
    let map = QualMap::from_module(&module);
    let mut fns = Vec::new();
    collect_fns(&module.items, &mut fns);
    let fd = fns.iter().find(|fd| fd.name == func)?;
    body_offender(&fd.body, &map)
}

thread_local! {
    static INSTALLED: RefCell<Option<QualMap>> = const { RefCell::new(None) };
}

/// Installs `module`'s qualifier map for [`call_is_world`] on this thread.
/// Dropping the guard clears it.
pub struct Guard;

impl Guard {
    pub fn install(module: &Module) -> Self {
        let map = QualMap::from_module(module);
        INSTALLED.with(|slot| *slot.borrow_mut() = Some(map));
        Guard
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        INSTALLED.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Whether the call at `span` named `name` is a qualified std-world call in
/// the map [`Guard`] installed. False when no guard is installed, so a path
/// that never parsed a module cannot invent a refusal.
pub fn call_is_world(span: Span, name: &str) -> bool {
    INSTALLED.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|map| map.call_is_world(span, name))
    })
}

/// A copy of `module` whose first function body starts with a zero-argument
/// call the existing classifier already treats as nondeterministic.
///
/// Used only for the evidence artifact of a program whose nondeterminism sits
/// in a callee the entry body does not hold. The verifier re-derives from the
/// hashed body, so a label rewritten to `deterministic` then disagrees with
/// the body. Object code is never stamped.
pub fn stamp_unheld_world(module: &IRModule) -> IRModule {
    let mut cloned = module.clone();
    let dst = cloned.fresh();
    let call = Instr::legacy_call(dst, "__mind_now_ns", Vec::new());
    if let Some(body) = cloned.instrs.iter_mut().find_map(|instr| match instr {
        Instr::FnDef { body, .. } => Some(body),
        _ => None,
    }) {
        body.insert(0, call);
    } else {
        cloned.instrs.push(call);
    }
    cloned
}

/// `enter_fns` collects imports inside functions. A body scan passes false so
/// a nested function is classified on its own.
fn walk<'a>(node: &'a Node, enter_fns: bool, f: &mut impl FnMut(&'a Node)) {
    let descend_fn = enter_fns;
    match node {
        Node::FnDef(fd, _) => {
            if descend_fn {
                for stmt in &fd.body {
                    f(stmt);
                    walk(stmt, enter_fns, f);
                }
            }
            return;
        }
        Node::ImplBlock { methods, .. } => {
            if descend_fn {
                for method in methods {
                    f(method);
                    walk(method, enter_fns, f);
                }
            }
            return;
        }
        Node::Closure(data, _) => {
            for stmt in &data.body {
                f(stmt);
                walk(stmt, enter_fns, f);
            }
        }
        Node::Call { args, .. } | Node::Print { args, .. } => {
            for arg in args {
                f(arg);
                walk(arg, enter_fns, f);
            }
        }
        Node::MethodCall { receiver, args, .. } => {
            f(receiver);
            walk(receiver, enter_fns, f);
            for arg in args {
                f(arg);
                walk(arg, enter_fns, f);
            }
        }
        Node::Binary { left, right, .. } | Node::Logical { left, right, .. } => {
            f(left);
            walk(left, enter_fns, f);
            f(right);
            walk(right, enter_fns, f);
        }
        #[cfg(feature = "std-surface")]
        Node::Bitwise { left, right, .. } => {
            f(left);
            walk(left, enter_fns, f);
            f(right);
            walk(right, enter_fns, f);
        }
        Node::Paren(inner, _)
        | Node::Return {
            value: Some(inner), ..
        }
        | Node::As { expr: inner, .. }
        | Node::Assert { cond: inner, .. }
        | Node::Try { inner, .. }
        | Node::Ref { inner, .. }
        | Node::Neg { operand: inner, .. }
        | Node::Not { operand: inner, .. }
        | Node::BitNot { operand: inner, .. } => {
            f(inner);
            walk(inner, enter_fns, f);
        }
        Node::Tuple { elements, .. } => {
            for elem in elements {
                f(elem);
                walk(elem, enter_fns, f);
            }
        }
        Node::Let { value, .. } | Node::LetTuple { value, .. } | Node::Assign { value, .. } => {
            f(value);
            walk(value, enter_fns, f);
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            f(cond);
            walk(cond, enter_fns, f);
            for stmt in then_branch {
                f(stmt);
                walk(stmt, enter_fns, f);
            }
            if let Some(else_branch) = else_branch {
                for stmt in else_branch {
                    f(stmt);
                    walk(stmt, enter_fns, f);
                }
            }
        }
        Node::Block { stmts, .. } => {
            for stmt in stmts {
                f(stmt);
                walk(stmt, enter_fns, f);
            }
        }
        Node::For {
            start, end, body, ..
        } => {
            f(start);
            walk(start, enter_fns, f);
            f(end);
            walk(end, enter_fns, f);
            for stmt in body {
                f(stmt);
                walk(stmt, enter_fns, f);
            }
        }
        Node::ForEach {
            collection, body, ..
        } => {
            f(collection);
            walk(collection, enter_fns, f);
            for stmt in body {
                f(stmt);
                walk(stmt, enter_fns, f);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            f(cond);
            walk(cond, enter_fns, f);
            for stmt in body {
                f(stmt);
                walk(stmt, enter_fns, f);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => {
            for stmt in body {
                f(stmt);
                walk(stmt, enter_fns, f);
            }
        }
        Node::Match {
            scrutinee, arms, ..
        } => {
            f(scrutinee);
            walk(scrutinee, enter_fns, f);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    f(guard);
                    walk(guard, enter_fns, f);
                }
                f(&arm.body);
                walk(&arm.body, enter_fns, f);
            }
        }
        Node::IndexAccess {
            receiver, index, ..
        } => {
            f(receiver);
            walk(receiver, enter_fns, f);
            f(index);
            walk(index, enter_fns, f);
        }
        Node::SliceRange {
            receiver,
            start,
            end,
            ..
        } => {
            f(receiver);
            walk(receiver, enter_fns, f);
            f(start);
            walk(start, enter_fns, f);
            f(end);
            walk(end, enter_fns, f);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            f(receiver);
            walk(receiver, enter_fns, f);
            f(index);
            walk(index, enter_fns, f);
            f(value);
            walk(value, enter_fns, f);
        }
        Node::FieldAssign {
            receiver, value, ..
        } => {
            f(receiver);
            walk(receiver, enter_fns, f);
            f(value);
            walk(value, enter_fns, f);
        }
        Node::StructLit { fields, .. } => {
            for field in fields {
                f(&field.value);
                walk(&field.value, enter_fns, f);
            }
        }
        _ => {}
    }
}
