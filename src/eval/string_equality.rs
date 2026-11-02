// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Typed string equality normalization before recursive expression lowering.
//!
//! `==` and `!=` on two strings compare bytes. An operand is a string when its
//! declared type says so (see [`facts`]): a literal, a `string`/`String`
//! parameter or annotated binding, a string constant, call result or struct
//! field. Such a comparison is rewritten to the compiler-owned
//! `__mind_string_eq` before the interpreter or any lowering sees it, so every
//! backend computes the same byte-wise answer. A string compared with an
//! operand whose type is not a string (other than an integer literal, which is
//! a null or sentinel test) would compare handles, which is false for equal
//! text; it is refused at check time with `E2038`, never lowered silently.
//!
//! A reassigned name keeps a string fact only when every assignment keeps it a
//! string (see `Decls::reassigned_let_fact`); facts are never joined across
//! branches, which avoids branch, loop, and shadowing ambiguity.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::ast::{BinOp, Literal, Module, Node, Pattern, Span};

mod facts;
#[cfg(test)]
mod tests;

use facts::{Class, Decls, Env, Fact};

type Targets = BTreeSet<usize>;
type Assigned = BTreeSet<String>;
type Ordinals = BTreeMap<usize, usize>;

/// A comparison of a string with an operand that is not known to be a string.
pub(crate) struct Refused {
    pub(crate) op: BinOp,
    pub(crate) span: Span,
}

/// The `E2038` message for `refused`, naming the byte-wise std function.
pub(crate) fn refusal_message(op: BinOp) -> String {
    let symbol = if op == BinOp::Eq { "==" } else { "!=" };
    format!(
        "`{symbol}` compares a string with an operand whose type is not a string; \
         it would compare handles, not bytes, and report equal text as unequal \
         — use `string_eq(a, b)` for byte-wise equality"
    )
}

/// The `E2038` help text.
pub(crate) const REFUSAL_HELP: &str = "give the other operand a `string` type (parameter, \
     annotated binding, field or function return), or compare bytes explicitly with \
     `string_eq(a, b)` from std.string";

fn remove_pattern_bindings(pattern: &Pattern, env: &mut Env, decls: &Decls) {
    let mut pending = vec![pattern];
    while let Some(pattern) = pending.pop() {
        match pattern {
            Pattern::Ident(name) => {
                decls.bind(env, name, Fact::Opaque);
            }
            Pattern::EnumVariant { args, .. } | Pattern::Tuple(args) => pending.extend(args),
            Pattern::EnumStruct { fields, .. } => {
                pending.extend(fields.iter().map(|(_, pattern)| pattern));
            }
            Pattern::Literal(_) | Pattern::Wildcard => {}
        }
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

enum Task<'a> {
    Stmts {
        stmts: &'a [Node],
        index: usize,
        env: Env,
        assigned: Rc<Assigned>,
    },
    Node {
        node: &'a Node,
        env: Env,
        assigned: Rc<Assigned>,
    },
}

fn push_stmts<'a>(pending: &mut Vec<Task<'a>>, stmts: &'a [Node], env: Env) {
    pending.push(Task::Stmts {
        stmts,
        index: 0,
        env,
        assigned: Rc::new(assigned_in(stmts)),
    });
}

/// Record equality nodes with two string operands, and the comparisons of a
/// string with an operand that is not one. This visitor owns an explicit heap
/// worklist so deeply nested std modules do not consume the compiler thread
/// stack. Since assigned names are excluded for their whole declaration,
/// branch-local facts never need an optimistic control-flow join.
fn collect_targets(module: &Module, ordinals: &Ordinals) -> (Targets, Vec<Refused>) {
    let decls = Decls::new(module);
    let mut targets = Targets::new();
    let mut refused = Vec::new();
    let mut pending = Vec::new();
    push_stmts(&mut pending, &module.items, Env::default());
    while let Some(task) = pending.pop() {
        match task {
            Task::Stmts {
                stmts,
                index,
                mut env,
                assigned,
            } => {
                let Some(stmt) = stmts.get(index) else {
                    continue;
                };
                match stmt {
                    Node::Let {
                        name, ann, value, ..
                    } => {
                        let value_env = env.clone();
                        let fact = if assigned.contains(name) {
                            let later = assignment_values(&stmts[index + 1..], name);
                            decls.reassigned_let_fact(ann.as_ref(), value, &later, &env)
                        } else {
                            decls.let_fact(ann.as_ref(), value, &env)
                        };
                        decls.bind(&mut env, name, fact);
                        pending.push(Task::Stmts {
                            stmts,
                            index: index + 1,
                            env,
                            assigned: assigned.clone(),
                        });
                        pending.push(Task::Node {
                            node: value,
                            env: value_env,
                            assigned,
                        });
                    }
                    // The binding's fact already accounts for every later
                    // assignment (`reassigned_let_fact`, `param_env`).
                    Node::Assign { value, .. } => {
                        let value_env = env.clone();
                        pending.push(Task::Stmts {
                            stmts,
                            index: index + 1,
                            env,
                            assigned: assigned.clone(),
                        });
                        pending.push(Task::Node {
                            node: value,
                            env: value_env,
                            assigned,
                        });
                    }
                    Node::FnDef(fd, _) => {
                        pending.push(Task::Stmts {
                            stmts,
                            index: index + 1,
                            env,
                            assigned,
                        });
                        push_stmts(&mut pending, &fd.body, decls.fn_env(fd));
                    }
                    _ => {
                        pending.push(Task::Stmts {
                            stmts,
                            index: index + 1,
                            env: env.clone(),
                            assigned: assigned.clone(),
                        });
                        pending.push(Task::Node {
                            node: stmt,
                            env,
                            assigned,
                        });
                    }
                }
            }
            Task::Node {
                node,
                env,
                assigned,
            } => match node {
                Node::Block { stmts, .. } | Node::Region { body: stmts, .. } => {
                    pending.push(Task::Stmts {
                        stmts,
                        index: 0,
                        env,
                        assigned,
                    });
                }
                Node::If {
                    cond,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    if let Some(branch) = else_branch {
                        pending.push(Task::Stmts {
                            stmts: branch,
                            index: 0,
                            env: env.clone(),
                            assigned: assigned.clone(),
                        });
                    }
                    pending.push(Task::Stmts {
                        stmts: then_branch,
                        index: 0,
                        env: env.clone(),
                        assigned: assigned.clone(),
                    });
                    pending.push(Task::Node {
                        node: cond,
                        env,
                        assigned,
                    });
                }
                Node::For {
                    var,
                    start,
                    end,
                    body,
                    ..
                } => {
                    let mut body_env = env.clone();
                    decls.bind(&mut body_env, var, Fact::Opaque);
                    pending.push(Task::Stmts {
                        stmts: body,
                        index: 0,
                        env: body_env,
                        assigned: assigned.clone(),
                    });
                    pending.push(Task::Node {
                        node: end,
                        env: env.clone(),
                        assigned: assigned.clone(),
                    });
                    pending.push(Task::Node {
                        node: start,
                        env,
                        assigned,
                    });
                }
                Node::ForEach {
                    var,
                    collection,
                    body,
                    ..
                } => {
                    let mut body_env = env.clone();
                    let element = if decls.is_string_array(collection, &env)
                        && assignment_values(body, var).is_empty()
                    {
                        Fact::Str
                    } else {
                        Fact::Opaque
                    };
                    decls.bind(&mut body_env, var, element);
                    pending.push(Task::Stmts {
                        stmts: body,
                        index: 0,
                        env: body_env,
                        assigned: assigned.clone(),
                    });
                    pending.push(Task::Node {
                        node: collection,
                        env,
                        assigned,
                    });
                }
                Node::While { cond, body, .. } => {
                    pending.push(Task::Stmts {
                        stmts: body,
                        index: 0,
                        env: env.clone(),
                        assigned: assigned.clone(),
                    });
                    pending.push(Task::Node {
                        node: cond,
                        env,
                        assigned,
                    });
                }
                Node::Match {
                    scrutinee, arms, ..
                } => {
                    for arm in arms.iter().rev() {
                        let mut arm_env = env.clone();
                        remove_pattern_bindings(&arm.pattern, &mut arm_env, &decls);
                        pending.push(Task::Node {
                            node: &arm.body,
                            env: arm_env.clone(),
                            assigned: assigned.clone(),
                        });
                        if let Some(guard) = &arm.guard {
                            pending.push(Task::Node {
                                node: guard,
                                env: arm_env,
                                assigned: assigned.clone(),
                            });
                        }
                    }
                    pending.push(Task::Node {
                        node: scrutinee,
                        env,
                        assigned,
                    });
                }
                Node::Binary {
                    op,
                    left,
                    right,
                    span,
                } => {
                    if matches!(op, BinOp::Eq | BinOp::Ne) {
                        match (decls.classify(left, &env), decls.classify(right, &env)) {
                            (Class::Str, Class::Str) => {
                                targets.insert(ordinals[&(node as *const Node as usize)]);
                            }
                            (Class::Str, Class::Other) | (Class::Other, Class::Str) => {
                                refused.push(Refused {
                                    op: *op,
                                    span: *span,
                                });
                            }
                            _ => {}
                        }
                    }
                    let children = super::closures::node_children_ref(node);
                    for child in children.into_iter().rev() {
                        pending.push(Task::Node {
                            node: child,
                            env: env.clone(),
                            assigned: assigned.clone(),
                        });
                    }
                }
                Node::Closure(data, _) => {
                    let closure_assigned = Rc::new(assigned_in(&data.body));
                    let captured: Env = Rc::new(
                        data.captures
                            .iter()
                            .filter(|name| !closure_assigned.contains(*name))
                            .filter_map(|name| env.get(name).map(|f| (name.clone(), f.clone())))
                            .collect(),
                    );
                    let closure_env = decls.param_env(captured, &data.params, &data.body);
                    pending.push(Task::Stmts {
                        stmts: &data.body,
                        index: 0,
                        env: closure_env,
                        assigned: closure_assigned,
                    });
                }
                Node::FnDef(fd, _) => push_stmts(&mut pending, &fd.body, decls.fn_env(fd)),
                _ => {
                    let children = super::closures::node_children_ref(node);
                    for child in children.into_iter().rev() {
                        pending.push(Task::Node {
                            node: child,
                            env: env.clone(),
                            assigned: assigned.clone(),
                        });
                    }
                }
            },
        }
    }
    (targets, refused)
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
    let (mut targets, refused) = collect_targets(module, &ordinals);
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
