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
//!
//! Each walk also records every string stored into a declared non-string
//! slot (see [`super::flows`]). While a walk records a slot the previous ones
//! had not, the module is walked again with every recorded slot read as
//! maybe a string, so every comparison is decided against every store in the
//! module, including a store of a value that is a string only through an
//! earlier store. Slots are only ever added and are finite, so this ends.

use std::rc::Rc;

use crate::ast::{BinOp, FnDefData, Module, Node};

use super::facts::{Contradicted, Decls, Env, Fact};
use super::{Assigned, Ordinals, Refused, Targets, assigned_in, assignment_values, decide};

/// What a statement list belongs to: the names assigned in the enclosing
/// declaration, and the function whose `return` it holds (`None` at module
/// level and in a closure).
struct Scope<'a> {
    assigned: Assigned,
    owner: Option<&'a str>,
}

enum Task<'a> {
    Stmts {
        stmts: &'a [Node],
        index: usize,
        env: Env,
        scope: Rc<Scope<'a>>,
        /// A binding made by the previous statement, applied on entry.
        bind: Vec<(&'a str, Fact)>,
    },
    Node {
        node: &'a Node,
        env: Env,
        scope: Rc<Scope<'a>>,
    },
}

struct Walk<'a> {
    decls: Decls,
    ordinals: &'a Ordinals,
    targets: Targets,
    refused: Vec<Refused>,
    /// Stores of strings into declared non-string slots seen by this walk.
    flows: Contradicted,
    pending: Vec<Task<'a>>,
}

/// Record equality nodes with two string operands, and every comparison that
/// would compare string handles. Since assigned names are excluded for their
/// whole declaration, branch-local facts never need an optimistic join.
pub(super) fn collect_targets(module: &Module, ordinals: &Ordinals) -> (Targets, Vec<Refused>) {
    let mut decls = Decls::new(module);
    // shortcut: one whole-module walk per round of newly recorded slots; a
    // chain of stores through N slots costs N walks. Upgrade to a worklist
    // over the recorded slots if a module ever needs many rounds.
    loop {
        let walk = Walk::run(decls, ordinals, module);
        let Walk {
            decls: mut walked,
            targets,
            refused,
            flows,
            ..
        } = walk;
        if !walked.contradicted.absorb(flows) {
            return (targets, refused);
        }
        decls = walked;
    }
}

impl<'a> Walk<'a> {
    fn run(decls: Decls, ordinals: &'a Ordinals, module: &'a Module) -> Self {
        let mut walk = Walk {
            decls,
            ordinals,
            targets: Targets::new(),
            refused: Vec::new(),
            flows: Contradicted::default(),
            pending: Vec::new(),
        };
        walk.push_stmts(&module.items, Env::default(), None);
        while let Some(task) = walk.pending.pop() {
            match task {
                Task::Stmts {
                    stmts,
                    index,
                    mut env,
                    scope,
                    bind,
                } => {
                    for (name, fact) in bind {
                        walk.decls.bind(&mut env, name, fact);
                    }
                    if let Some(stmt) = stmts.get(index) {
                        walk.stmt(stmts, index, stmt, env, scope);
                    }
                }
                Task::Node { node, env, scope } => walk.node(node, env, scope),
            }
        }
        walk
    }

    fn push_stmts(&mut self, stmts: &'a [Node], env: Env, owner: Option<&'a str>) {
        let scope = Scope {
            assigned: assigned_in(stmts),
            owner,
        };
        self.pending.push(Task::Stmts {
            stmts,
            index: 0,
            env,
            scope: Rc::new(scope),
            bind: Vec::new(),
        });
    }

    fn push_block(&mut self, stmts: &'a [Node], env: Env, scope: &Rc<Scope<'a>>) {
        self.pending.push(Task::Stmts {
            stmts,
            index: 0,
            env,
            scope: scope.clone(),
            bind: Vec::new(),
        });
    }

    fn push_node(&mut self, node: &'a Node, env: Env, scope: &Rc<Scope<'a>>) {
        self.pending.push(Task::Node {
            node,
            env,
            scope: scope.clone(),
        });
    }

    /// Continue with the statement after `index`, applying `bind` first.
    fn push_next(
        &mut self,
        stmts: &'a [Node],
        index: usize,
        env: Env,
        scope: &Rc<Scope<'a>>,
        bind: Vec<(&'a str, Fact)>,
    ) {
        self.pending.push(Task::Stmts {
            stmts,
            index: index + 1,
            env,
            scope: scope.clone(),
            bind,
        });
    }

    fn stmt(
        &mut self,
        stmts: &'a [Node],
        index: usize,
        stmt: &'a Node,
        env: Env,
        scope: Rc<Scope<'a>>,
    ) {
        match stmt {
            Node::Let {
                name, ann, value, ..
            } => {
                let fact = if scope.assigned.contains(name) {
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
                    &scope,
                    vec![(name.as_str(), fact)],
                );
                self.push_node(value, env, &scope);
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
                        let fact = if scope.assigned.contains(name) {
                            Fact::Unknown
                        } else {
                            fact
                        };
                        (name.as_str(), fact)
                    })
                    .collect();
                self.push_next(stmts, index, env.clone(), &scope, bind);
                self.push_node(value, env, &scope);
            }
            // The binding's fact already accounts for every later assignment
            // (`reassigned_fact`, `param_env`).
            Node::Assign { value, .. } => {
                self.push_next(stmts, index, env.clone(), &scope, Vec::new());
                self.push_node(value, env, &scope);
            }
            Node::FnDef(fd, _) => {
                self.push_next(stmts, index, env, &scope, Vec::new());
                self.enter_fn(fd);
            }
            _ => {
                self.push_next(stmts, index, env.clone(), &scope, Vec::new());
                self.push_node(stmt, env, &scope);
            }
        }
    }

    fn node(&mut self, node: &'a Node, env: Env, scope: Rc<Scope<'a>>) {
        self.decls
            .record_flows(node, &env, scope.owner, &mut self.flows);
        match node {
            Node::Block { stmts, .. } => self.push_block(stmts, env, &scope),
            #[cfg(feature = "std-surface")]
            Node::Region { body, .. } => self.push_block(body, env, &scope),
            Node::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                if let Some(branch) = else_branch {
                    self.push_block(branch, env.clone(), &scope);
                }
                self.push_block(then_branch, env.clone(), &scope);
                self.push_node(cond, env, &scope);
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
                self.loop_body(body, var, fact, &env, &scope);
                self.push_node(end, env.clone(), &scope);
                self.push_node(start, env, &scope);
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
                self.loop_body(body, var, fact, &env, &scope);
                self.push_node(collection, env, &scope);
            }
            Node::While { cond, body, .. } => {
                self.push_block(body, env.clone(), &scope);
                self.push_node(cond, env, &scope);
            }
            Node::Match {
                scrutinee, arms, ..
            } => {
                let scrutinee_fact = self.decls.value_fact(scrutinee, &env);
                for arm in arms.iter().rev() {
                    let mut arm_env = env.clone();
                    self.decls
                        .bind_pattern(&arm.pattern, scrutinee_fact.clone(), &mut arm_env);
                    self.push_node(&arm.body, arm_env.clone(), &scope);
                    if let Some(guard) = &arm.guard {
                        self.push_node(guard, arm_env, &scope);
                    }
                }
                self.push_node(scrutinee, env, &scope);
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
                self.push_children(node, &env, &scope);
            }
            Node::Closure(data, _) => {
                let closure_scope = Scope {
                    assigned: assigned_in(&data.body),
                    owner: None,
                };
                let captured: Env = Rc::new(
                    data.captures
                        .iter()
                        .filter(|name| !closure_scope.assigned.contains(*name))
                        .filter_map(|name| env.get(name).map(|f| (name.clone(), f.clone())))
                        .collect(),
                );
                let closure_env = self
                    .decls
                    .param_env(captured, &data.params, &data.body, None);
                self.pending.push(Task::Stmts {
                    stmts: &data.body,
                    index: 0,
                    env: closure_env,
                    scope: Rc::new(closure_scope),
                    bind: Vec::new(),
                });
            }
            Node::FnDef(fd, _) => self.enter_fn(fd),
            _ => self.push_children(node, &env, &scope),
        }
    }

    /// Walk a function body with its parameters bound.
    fn enter_fn(&mut self, fd: &'a FnDefData) {
        let fn_env = self.decls.fn_env(fd);
        self.decls.record_tail_flow(fd, &fn_env, &mut self.flows);
        self.push_stmts(&fd.body, fn_env, Some(fd.name.as_str()));
    }

    fn loop_body(
        &mut self,
        body: &'a [Node],
        var: &str,
        fact: Fact,
        env: &Env,
        scope: &Rc<Scope<'a>>,
    ) {
        let mut body_env = env.clone();
        self.decls.bind(&mut body_env, var, fact);
        self.push_block(body, body_env, scope);
    }

    fn push_children(&mut self, node: &'a Node, env: &Env, scope: &Rc<Scope<'a>>) {
        let children = super::super::closures::node_children_ref(node);
        for child in children.into_iter().rev() {
            self.push_node(child, env.clone(), scope);
        }
    }
}
