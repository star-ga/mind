// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The scoped walk that decides every `==` / `!=` of one module.
//!
//! The walk owns an explicit heap worklist so deeply nested std modules do not
//! consume the compiler thread stack. Each statement list carries the facts in
//! scope; a binding is applied when the walk reaches the next statement, after
//! the bound value's own sub-walk has released its copy of the scope, so a
//! binding does not copy the scope map.

use std::rc::Rc;

use crate::ast::{BinOp, Module, Node};

use super::facts::{Decls, Env, Fact};
use super::{Assigned, Ordinals, Refused, Targets, assigned_in, assignment_values, decide};

enum Task<'a> {
    Stmts {
        stmts: &'a [Node],
        index: usize,
        env: Env,
        assigned: Rc<Assigned>,
        /// A binding made by the previous statement, applied on entry.
        bind: Vec<(&'a str, Fact)>,
    },
    Node {
        node: &'a Node,
        env: Env,
        assigned: Rc<Assigned>,
    },
}

struct Walk<'a> {
    decls: Decls,
    ordinals: &'a Ordinals,
    targets: Targets,
    refused: Vec<Refused>,
    pending: Vec<Task<'a>>,
}

/// Record equality nodes with two string operands, and every comparison that
/// would compare string handles. Since assigned names are excluded for their
/// whole declaration, branch-local facts never need an optimistic join.
pub(super) fn collect_targets(module: &Module, ordinals: &Ordinals) -> (Targets, Vec<Refused>) {
    let mut walk = Walk {
        decls: Decls::new(module),
        ordinals,
        targets: Targets::new(),
        refused: Vec::new(),
        pending: Vec::new(),
    };
    walk.push_stmts(&module.items, Env::default());
    while let Some(task) = walk.pending.pop() {
        match task {
            Task::Stmts {
                stmts,
                index,
                mut env,
                assigned,
                bind,
            } => {
                for (name, fact) in bind {
                    walk.decls.bind(&mut env, name, fact);
                }
                if let Some(stmt) = stmts.get(index) {
                    walk.stmt(stmts, index, stmt, env, assigned);
                }
            }
            Task::Node {
                node,
                env,
                assigned,
            } => walk.node(node, env, assigned),
        }
    }
    (walk.targets, walk.refused)
}

impl<'a> Walk<'a> {
    fn push_stmts(&mut self, stmts: &'a [Node], env: Env) {
        self.pending.push(Task::Stmts {
            stmts,
            index: 0,
            env,
            assigned: Rc::new(assigned_in(stmts)),
            bind: Vec::new(),
        });
    }

    fn push_block(&mut self, stmts: &'a [Node], env: Env, assigned: &Rc<Assigned>) {
        self.pending.push(Task::Stmts {
            stmts,
            index: 0,
            env,
            assigned: assigned.clone(),
            bind: Vec::new(),
        });
    }

    fn push_node(&mut self, node: &'a Node, env: Env, assigned: &Rc<Assigned>) {
        self.pending.push(Task::Node {
            node,
            env,
            assigned: assigned.clone(),
        });
    }

    /// Continue with the statement after `index`, applying `bind` first.
    fn push_next(
        &mut self,
        stmts: &'a [Node],
        index: usize,
        env: Env,
        assigned: &Rc<Assigned>,
        bind: Vec<(&'a str, Fact)>,
    ) {
        self.pending.push(Task::Stmts {
            stmts,
            index: index + 1,
            env,
            assigned: assigned.clone(),
            bind,
        });
    }

    fn stmt(
        &mut self,
        stmts: &'a [Node],
        index: usize,
        stmt: &'a Node,
        env: Env,
        assigned: Rc<Assigned>,
    ) {
        match stmt {
            Node::Let {
                name, ann, value, ..
            } => {
                let fact = if assigned.contains(name) {
                    let later = assignment_values(&stmts[index + 1..], name);
                    let fact = self.decls.let_fact(ann.as_ref(), value, &env);
                    self.decls
                        .reassigned_fact(name, fact, ann.is_some(), &later, &env)
                } else {
                    self.decls.let_fact(ann.as_ref(), value, &env)
                };
                self.push_next(
                    stmts,
                    index,
                    env.clone(),
                    &assigned,
                    vec![(name.as_str(), fact)],
                );
                self.push_node(value, env, &assigned);
            }
            Node::LetTuple { names, value, .. } => {
                let facts = match self.decls.value_fact(value, &env) {
                    Fact::Tuple(facts) if facts.len() == names.len() => facts,
                    _ => vec![Fact::Unknown; names.len()],
                };
                let bind = names
                    .iter()
                    .zip(facts)
                    .map(|(name, fact)| {
                        let fact = if assigned.contains(name) {
                            Fact::Unknown
                        } else {
                            fact
                        };
                        (name.as_str(), fact)
                    })
                    .collect();
                self.push_next(stmts, index, env.clone(), &assigned, bind);
                self.push_node(value, env, &assigned);
            }
            // The binding's fact already accounts for every later assignment
            // (`reassigned_fact`, `param_env`).
            Node::Assign { value, .. } => {
                self.push_next(stmts, index, env.clone(), &assigned, Vec::new());
                self.push_node(value, env, &assigned);
            }
            Node::FnDef(fd, _) => {
                self.push_next(stmts, index, env, &assigned, Vec::new());
                let fn_env = self.decls.fn_env(fd);
                self.push_stmts(&fd.body, fn_env);
            }
            _ => {
                self.push_next(stmts, index, env.clone(), &assigned, Vec::new());
                self.push_node(stmt, env, &assigned);
            }
        }
    }

    fn node(&mut self, node: &'a Node, env: Env, assigned: Rc<Assigned>) {
        match node {
            Node::Block { stmts, .. } => self.push_block(stmts, env, &assigned),
            #[cfg(feature = "std-surface")]
            Node::Region { body, .. } => self.push_block(body, env, &assigned),
            Node::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                if let Some(branch) = else_branch {
                    self.push_block(branch, env.clone(), &assigned);
                }
                self.push_block(then_branch, env.clone(), &assigned);
                self.push_node(cond, env, &assigned);
            }
            Node::For {
                var,
                start,
                end,
                body,
                ..
            } => {
                let fact = if assignment_values(body, var).is_empty() {
                    Fact::NotStr
                } else {
                    Fact::Unknown
                };
                self.loop_body(body, var, fact, &env, &assigned);
                self.push_node(end, env.clone(), &assigned);
                self.push_node(start, env, &assigned);
            }
            Node::ForEach {
                var,
                collection,
                body,
                ..
            } => {
                let fact = if assignment_values(body, var).is_empty() {
                    self.decls.element_fact(collection, &env)
                } else {
                    Fact::Unknown
                };
                self.loop_body(body, var, fact, &env, &assigned);
                self.push_node(collection, env, &assigned);
            }
            Node::While { cond, body, .. } => {
                self.push_block(body, env.clone(), &assigned);
                self.push_node(cond, env, &assigned);
            }
            Node::Match {
                scrutinee, arms, ..
            } => {
                let scrutinee_fact = self.decls.value_fact(scrutinee, &env);
                for arm in arms.iter().rev() {
                    let mut arm_env = env.clone();
                    self.decls
                        .bind_pattern(&arm.pattern, scrutinee_fact.clone(), &mut arm_env);
                    self.push_node(&arm.body, arm_env.clone(), &assigned);
                    if let Some(guard) = &arm.guard {
                        self.push_node(guard, arm_env, &assigned);
                    }
                }
                self.push_node(scrutinee, env, &assigned);
            }
            Node::Binary {
                op,
                left,
                right,
                span,
            } => {
                if matches!(op, BinOp::Eq | BinOp::Ne) {
                    let classes = (
                        self.decls.classify(left, &env),
                        self.decls.classify(right, &env),
                    );
                    match decide(classes.0, classes.1) {
                        Ok(true) => {
                            self.targets
                                .insert(self.ordinals[&(node as *const Node as usize)]);
                        }
                        Ok(false) => {}
                        Err(kind) => self.refused.push(Refused {
                            op: *op,
                            kind,
                            span: *span,
                        }),
                    }
                }
                self.push_children(node, &env, &assigned);
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
                let closure_env = self.decls.param_env(captured, &data.params, &data.body);
                self.pending.push(Task::Stmts {
                    stmts: &data.body,
                    index: 0,
                    env: closure_env,
                    assigned: closure_assigned,
                    bind: Vec::new(),
                });
            }
            Node::FnDef(fd, _) => {
                let fn_env = self.decls.fn_env(fd);
                self.push_stmts(&fd.body, fn_env);
            }
            _ => self.push_children(node, &env, &assigned),
        }
    }

    fn loop_body(
        &mut self,
        body: &'a [Node],
        var: &str,
        fact: Fact,
        env: &Env,
        assigned: &Rc<Assigned>,
    ) {
        let mut body_env = env.clone();
        self.decls.bind(&mut body_env, var, fact);
        self.push_block(body, body_env, assigned);
    }

    fn push_children(&mut self, node: &'a Node, env: &Env, assigned: &Rc<Assigned>) {
        let children = super::super::closures::node_children_ref(node);
        for child in children.into_iter().rev() {
            self.push_node(child, env.clone(), assigned);
        }
    }
}
