// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Typed string equality normalization before recursive expression lowering.
//!
//! `==` and `!=` on two strings compare bytes. An operand is a string when its
//! declared type says so (see [`facts`] and [`classify`]): a literal, a
//! `string`/`String` parameter or annotated binding, a string constant, call
//! result (including a method call that lowers to a `string_<method>` or
//! `<type>_<method>` function), struct field, collection element, an `if` /
//! `match` whose every branch is a string, or a payload of an `Option` /
//! `Result` declared (or constructed) to carry one. Such a comparison is
//! rewritten to the compiler-owned `__mind_string_eq` before the interpreter
//! or any lowering sees it, so every backend computes the same byte-wise
//! answer. The null handle `0` is a `string` value equal only to itself; no
//! record is read through it.
//!
//! Every other comparison that could compare string handles, which is false
//! for equal text, is refused at check time with `E2038`, never lowered
//! silently:
//!
//! * a string compared with an operand that is not a string (an integer
//!   literal is exempt: a handle compared with it is a null or sentinel test);
//! * two operands of which neither has a known type, so both may be strings.
//!
//! An operand whose type is known and is not a string makes the comparison an
//! integer comparison on every backend. A declaration proves what its slot
//! holds only while the module stores nothing else there (see [`slots`] and
//! [`flows`]): a parameter, result, field, payload, module binding or
//! collection element given a value that may be a string where none is
//! declared, or an integer other than the null handle where a string is
//! declared, is read as maybe a string. So is the integer result of a callee
//! whose body is not analysed here (a std collection, raw memory) after a
//! string was handed to such a callee. A value of a type parameter, or the
//! result of an imported function with no declared result type, may be
//! anything. A function with no declared result type that still returns a
//! value has an unknown result.
//!
//! A reassigned name keeps its fact only when every assignment keeps it (see
//! `Decls::reassigned_fact`); facts are never joined across branches of
//! statements, which avoids branch, loop, and shadowing ambiguity.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{BinOp, Literal, Module, Node, Span};

mod classify;
mod facts;
mod flows;
#[cfg(test)]
mod slot_tests;
mod slots;
mod sums;
#[cfg(test)]
mod tests;
mod walk;

use facts::Class;

type Targets = BTreeSet<usize>;
type Assigned = BTreeSet<String>;
type Ordinals = BTreeMap<usize, usize>;

/// Why a comparison is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefusedKind {
    /// A string compared with an operand not known to be a string.
    Mixed,
    /// Two operands, neither of a known type.
    Untyped,
}

/// A comparison that would compare string handles instead of bytes.
pub(crate) struct Refused {
    pub(crate) op: BinOp,
    pub(crate) kind: RefusedKind,
    pub(crate) span: Span,
}

/// The `E2038` message for `refused`, naming the byte-wise std function.
pub(crate) fn refusal_message(refused: &Refused) -> String {
    let symbol = if refused.op == BinOp::Eq { "==" } else { "!=" };
    let operands = match refused.kind {
        RefusedKind::Mixed => "a string with an operand not known to be a string",
        RefusedKind::Untyped => {
            "two operands whose types are not known at compile time, so both may be strings"
        }
    };
    format!(
        "`{symbol}` compares {operands}; it would compare handles, not bytes, and \
         report equal text as unequal — use `string_eq(a, b)` for byte-wise equality"
    )
}

/// The `E2038` help text for `refused`.
pub(crate) fn refusal_help(refused: &Refused) -> &'static str {
    match refused.kind {
        RefusedKind::Mixed => {
            "give the other operand a `string` type (parameter, annotated binding, \
             field or function return) and store only strings (or the null handle \
             `0`) where `string` is declared, or compare bytes explicitly with \
             `string_eq(a, b)` from std.string"
        }
        RefusedKind::Untyped => {
            "give an operand a declared type (an annotated binding such as \
             `let n: i64 = ...`, a parameter, a field or a function return `-> T`) \
             and store no string where another type is declared, or compare \
             strings explicitly with `string_eq(a, b)` from std.string"
        }
    }
}

/// Decide one equality from its operand classes: `Ok(true)` rewrites it to the
/// byte-wise comparison, `Ok(false)` keeps it an integer comparison, and
/// `Err` refuses it.
fn decide(left: Class, right: Class) -> Result<bool, RefusedKind> {
    match (left, right) {
        (Class::Str, Class::Str) => Ok(true),
        (Class::Str, Class::IntLiteral) | (Class::IntLiteral, Class::Str) => Ok(false),
        (Class::Str, _) | (_, Class::Str) => Err(RefusedKind::Mixed),
        (Class::Unknown, Class::Unknown) => Err(RefusedKind::Untyped),
        _ => Ok(false),
    }
}

fn children_ref(node: &Node) -> Vec<&Node> {
    match node {
        Node::FnDef(fd, _) => fd.body.iter().collect(),
        Node::Closure(data, _) => data.body.iter().collect(),
        _ => super::closures::node_children_ref(node),
    }
}

fn children_mut(node: &mut Node) -> Vec<&mut Node> {
    match node {
        Node::FnDef(fd, _) => fd.body.iter_mut().collect(),
        Node::Closure(data, _) => data.body.iter_mut().collect(),
        _ => super::closures::node_children_mut(node),
    }
}

fn assigned_in(stmts: &[Node]) -> Assigned {
    let mut assigned = Assigned::new();
    let mut pending: Vec<&Node> = stmts.iter().rev().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Assign { name, value, .. } => {
                assigned.insert(name.clone());
                pending.push(value);
            }
            // Each declaration owns its own assignment analysis.
            Node::FnDef(..) | Node::Closure(..) => {}
            _ => {
                let children = super::closures::node_children_ref(node);
                pending.extend(children.into_iter().rev());
            }
        }
    }
    assigned
}

/// Every value assigned to `target` in `stmts`, outside nested declarations.
fn assignment_values<'a>(stmts: &'a [Node], target: &str) -> Vec<&'a Node> {
    let mut values = Vec::new();
    let mut pending: Vec<&Node> = stmts.iter().rev().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Assign { name, value, .. } => {
                if name == target {
                    values.push(&**value);
                }
                pending.push(value);
            }
            Node::FnDef(..) | Node::Closure(..) => {}
            _ => {
                let children = super::closures::node_children_ref(node);
                pending.extend(children.into_iter().rev());
            }
        }
    }
    values
}

/// Give every original equality node a deterministic preorder number. Pointer
/// identity is used only inside this immutable AST snapshot; the numbers, not
/// pointers or source-local spans, are carried into the rewritten clone.
fn equality_ordinals(module: &Module) -> Ordinals {
    let mut ordinals = Ordinals::new();
    let mut next = 0;
    let mut pending: Vec<&Node> = module.items.iter().rev().collect();
    while let Some(node) = pending.pop() {
        if matches!(
            node,
            Node::Binary {
                op: BinOp::Eq | BinOp::Ne,
                ..
            }
        ) {
            ordinals.insert(node as *const Node as usize, next);
            next += 1;
        }
        let children = children_ref(node);
        pending.extend(children.into_iter().rev());
    }
    ordinals
}

/// Rewrite in deterministic preorder on an explicit heap worklist. A target is
/// replaced before its original operands are queued, so the synthesized `!= 0`
/// comparison is not counted as another source equality.
fn rewrite(module: &mut Module, targets: &mut Targets) {
    let mut next_equality = 0;
    let mut pending: Vec<&mut Node> = module.items.iter_mut().rev().collect();
    while let Some(node) = pending.pop() {
        let target = if matches!(
            node,
            Node::Binary {
                op: BinOp::Eq | BinOp::Ne,
                ..
            }
        ) {
            let ordinal = next_equality;
            next_equality += 1;
            targets.remove(&ordinal)
        } else {
            false
        };
        if target {
            let old = std::mem::replace(node, Node::Lit(Literal::Int(0), node.span()));
            let Node::Binary {
                op,
                left,
                right,
                span,
            } = old
            else {
                unreachable!("target ordinal did not name an equality node")
            };
            let eq = Node::Call {
                callee: "__mind_string_eq".to_string(),
                args: vec![*left, *right],
                span,
            };
            *node = if op == BinOp::Eq {
                eq
            } else {
                Node::Binary {
                    op: BinOp::Eq,
                    left: Box::new(eq),
                    right: Box::new(Node::Lit(Literal::Int(0), span)),
                    span,
                }
            };
        }
        let children = children_mut(node);
        pending.extend(children.into_iter().rev());
    }
}

/// The analysis of one module: the rewritten clone (only when a string
/// comparison exists) and every refused comparison in source order.
pub(crate) struct Analysis {
    pub(crate) rewritten: Option<Module>,
    pub(crate) refused: Vec<Refused>,
}

/// Analyze `module`. The walk is iterative: modules with no target keep
/// lower_expr's stack shape, emitted bytes, and the Windows debug-stack margin
/// unchanged.
pub(crate) fn analyze(module: &Module) -> Analysis {
    let ordinals = equality_ordinals(module);
    if ordinals.is_empty() {
        return Analysis {
            rewritten: None,
            refused: Vec::new(),
        };
    }
    let (mut targets, refused) = walk::collect_targets(module, &ordinals);
    if targets.is_empty() {
        return Analysis {
            rewritten: None,
            refused,
        };
    }
    let mut rewritten = module.clone();
    rewrite(&mut rewritten, &mut targets);
    assert!(
        targets.is_empty(),
        "string equality normalization lost an analyzed AST target"
    );
    Analysis {
        rewritten: Some(rewritten),
        refused,
    }
}

/// Lowering entry: report each refused comparison through the check-time
/// refusal sink (which `mindc check` and `mindc build` consult before
/// lowering), then return the rewritten clone.
pub(crate) fn normalize(module: &Module) -> Option<Module> {
    let analysis = analyze(module);
    for refused in &analysis.refused {
        crate::type_checker::lowering_refusals::string_equality_operand(refused);
    }
    analysis.rewritten
}
