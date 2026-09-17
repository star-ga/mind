// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0

//! Which `&mut` parameters are CELL parameters in the deref-assign subset.
//!
//! A `&mut T` parameter has two possible run-time representations:
//!
//! * a POINTER — the address of the referent record itself. This is what main has
//!   always passed (`&mut h.pt` lowers to the value of `h.pt`, i.e. the record's
//!   address), and it is exactly right for everything main supports: `p.x`,
//!   `p.x = v`, forwarding, `let r = &mut h.pt`, `&mut a` on a whole local.
//! * a CELL — the address of the SLOT that holds the record's address. Only a cell
//!   can support identity place replacement, `*p = v`, which rewrites the slot.
//!
//! D3/D4 first made EVERY `&mut <struct>` a cell and refused the pointer uses
//! (measured 2026-09-16: eight programs main compiles and runs correctly were
//! refused). The two representations are now chosen PER PARAMETER: a parameter is a
//! cell iff its function dereferences it (`*p` / `*p = v`) or forwards it to a cell
//! parameter of another function (a fixpoint). Every other `&mut` parameter keeps
//! main's pointer semantics, and all of the deref-assign admission rules and the
//! cell lowering of `&mut r.f` apply to cell parameters only.
//!
//! Why this cannot change any program main accepted: main does not parse `*p`
//! (E1001), so no main-accepted program has a cell parameter. Why the per-module
//! inference is sound: a non-`pub` function is not callable from another module
//! (E2003), and a `pub` function with a cell parameter is refused by the type
//! checker, so a caller always sees the same module's inference as its callee.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{Literal, Node, TypeAnn};

/// `function name -> indices of its CELL parameters`.
pub(crate) type CellParams = BTreeMap<String, BTreeSet<usize>>;

/// Infer the cell parameters of every function defined in `items` (at any depth).
pub(crate) fn cell_ref_params(items: &[Node]) -> CellParams {
    let mut fns: Vec<&crate::ast::FnDefData> = Vec::new();
    collect_fns(items, &mut fns);

    let mut cells = CellParams::new();
    // (caller, caller param index, callee, callee arg index)
    let mut forwards: Vec<(String, usize, String, usize)> = Vec::new();
    for fd in &fns {
        let mut_params: BTreeMap<&str, usize> = fd
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p.ty, TypeAnn::Ref { mutable: true, .. }))
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();
        if mut_params.is_empty() {
            continue;
        }
        let mut seeds = BTreeSet::new();
        for stmt in &fd.body {
            scan(stmt, &mut_params, &fd.name, &mut seeds, &mut forwards);
        }
        if !seeds.is_empty() {
            cells.insert(fd.name.clone(), seeds);
        }
    }
    // Fixpoint: forwarding a parameter into a cell formal makes it a cell.
    loop {
        let mut changed = false;
        for (caller, param_idx, callee, arg_idx) in &forwards {
            let callee_cell = cells.get(callee).is_some_and(|s| s.contains(arg_idx));
            if callee_cell && cells.entry(caller.clone()).or_default().insert(*param_idx) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    cells.retain(|_, s| !s.is_empty());
    cells
}

/// Is argument `index` of a call to `callee` a cell parameter?
pub(crate) fn is_cell(cells: &CellParams, callee: &str, index: usize) -> bool {
    cells.get(callee).is_some_and(|s| s.contains(&index))
}

fn collect_fns<'a>(items: &'a [Node], out: &mut Vec<&'a crate::ast::FnDefData>) {
    for item in items {
        if let Node::FnDef(fd, _) = item {
            out.push(fd);
            // `node_children_ref` does not expose a FnDef's body, so walk it here.
            collect_fns(&fd.body, out);
            continue;
        }
        for child in crate::eval::closures::node_children_ref(item) {
            collect_fns(std::slice::from_ref(child), out);
        }
    }
}

fn param_ident<'a>(node: &Node, params: &BTreeMap<&'a str, usize>) -> Option<usize> {
    let mut n = node;
    while let Node::Paren(inner, _) = n {
        n = inner;
    }
    match n {
        Node::Lit(Literal::Ident(name), _) => params.get(name.as_str()).copied(),
        _ => None,
    }
}

fn scan(
    node: &Node,
    params: &BTreeMap<&str, usize>,
    owner: &str,
    seeds: &mut BTreeSet<usize>,
    forwards: &mut Vec<(String, usize, String, usize)>,
) {
    match node {
        // A nested function has its own parameters and its own inference.
        Node::FnDef(..) => return,
        Node::Deref { operand, .. } => {
            if let Some(i) = param_ident(operand, params) {
                seeds.insert(i);
            }
        }
        Node::DerefAssign { target, .. } => {
            if let Some(i) = param_ident(target, params) {
                seeds.insert(i);
            }
        }
        Node::Call { callee, args, .. } => {
            for (arg_idx, arg) in args.iter().enumerate() {
                if let Some(param_idx) = param_ident(arg, params) {
                    forwards.push((owner.to_string(), param_idx, callee.clone(), arg_idx));
                }
            }
        }
        _ => {}
    }
    for child in crate::eval::closures::node_children_ref(node) {
        scan(child, params, owner, seeds, forwards);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(src: &str) -> CellParams {
        cell_ref_params(&crate::parser::parse(src).expect("parse").items)
    }

    #[test]
    fn pointer_uses_are_not_cells() {
        let c = cells(
            "struct Pair {\n    x: i64,\n    y: i64\n}\nfn set(p: &mut Pair) {\n    p.x = 5\n}\nfn mid(p: &mut Pair) {\n    set(p)\n}\n",
        );
        assert!(
            c.is_empty(),
            "field access and forwarding to a pointer formal are not cells: {c:?}"
        );
    }

    #[test]
    fn deref_and_forwarding_to_a_cell_make_cells() {
        let c = cells(
            "struct Pair {\n    x: i64,\n    y: i64\n}\nfn replace(p: &mut Pair, new: Pair) {\n    *p = new\n}\nfn mid(q: &mut Pair, n: Pair) {\n    replace(q, n)\n}\nfn top(r: &mut Pair, n: Pair) {\n    mid((r), n)\n}\nfn reader(p: &mut Pair) -> Pair {\n    return *p\n}\n",
        );
        assert!(is_cell(&c, "replace", 0));
        assert!(!is_cell(&c, "replace", 1));
        assert!(
            is_cell(&c, "mid", 0),
            "forwarding into a cell formal: {c:?}"
        );
        assert!(
            is_cell(&c, "top", 0),
            "transitive + parenthesised forwarding: {c:?}"
        );
        assert!(is_cell(&c, "reader", 0), "a deref READ is a cell use too");
    }

    #[test]
    fn nested_fn_does_not_leak_into_its_parent() {
        let c = cells(
            "struct Pair {\n    x: i64,\n    y: i64\n}\nfn outer(p: &mut Pair) {\n    fn inner(p: &mut Pair, n: Pair) {\n        *p = n\n    }\n    p.x = 1\n}\n",
        );
        assert!(!is_cell(&c, "outer", 0), "{c:?}");
        assert!(is_cell(&c, "inner", 0), "{c:?}");
    }
}
