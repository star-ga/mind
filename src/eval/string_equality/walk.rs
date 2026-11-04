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

use crate::ast::{BinOp, FnDefData, Literal, Module, Node, Pattern};

use super::facts::{Bound, Class, Decls, Env, Fact, tuple_parts};
use super::slots::Contradicted;
use super::{MethodTargets, Ordinals, Refused, RefusedKind, Targets, assigned_in, decide};

/// What a statement list belongs to: the function whose `return` it holds
/// (`None` at module level and in a closure).
struct Scope<'a> {
    owner: Option<&'a str>,
}

/// A local binding made by a statement: its name, the fact of its initial
/// value, and whether its type is declared.
type Binding<'a> = (&'a String, Fact, bool);

enum Task<'a> {
    Stmts {
        stmts: &'a [Node],
        index: usize,
        env: Env,
        scope: Rc<Scope<'a>>,
        /// The bindings made by the previous statement, applied on entry.
        bind: Vec<Binding<'a>>,
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
    /// Method calls on a string (see `super::desugar_string_method`).
    methods: MethodTargets,
    refused: Vec<Refused>,
    /// Stores of strings into declared non-string slots seen by this walk.
    flows: Contradicted,
    pending: Vec<Task<'a>>,
}

/// Record equality nodes with two string operands, and every comparison that
/// would compare string handles. Since assigned names are excluded for their
/// whole declaration, branch-local facts never need an optimistic join.
pub(super) fn collect_targets(
    module: &Module,
    ordinals: &Ordinals,
    project: bool,
) -> (Targets, MethodTargets, Vec<Refused>) {
    let decls = Decls::new(module, project);
    let (targets, methods, refused, _) = fixpoint(decls, ordinals, module);
    (targets, methods, refused)
}

/// Walk `module` until a walk records no slot the previous ones had not,
/// starting from `decls`; the last walk's decisions and the final slots.
pub(super) fn fixpoint<'a>(
    mut decls: Decls,
    ordinals: &'a Ordinals,
    module: &'a Module,
) -> (Targets, MethodTargets, Vec<Refused>, Decls) {
    // shortcut: one whole-module walk per round of newly recorded slots; a
    // chain of stores through N slots costs N walks. Upgrade to a worklist
    // over the recorded slots if a module ever needs many rounds.
    loop {
        let walk = Walk::run(decls, ordinals, module);
        let Walk {
            decls: mut walked,
            targets,
            methods,
            refused,
            flows,
            ..
        } = walk;
        if !walked.contradicted.absorb(flows) {
            return (targets, methods, refused, walked);
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
            methods: MethodTargets::new(),
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
                    for (name, fact, declared) in bind {
                        walk.decls.bind_local(&mut env, name, fact, declared);
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
        let scope = Scope { owner };
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
        bind: Vec<Binding<'a>>,
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
                let fact = self.decls.let_fact(ann.as_ref(), value, &env);
                let bind = vec![(name, fact, ann.is_some())];
                self.push_next(stmts, index, env.clone(), &scope, bind);
                self.push_node(value, env, &scope);
            }
            Node::LetTuple { names, value, .. } => {
                let facts = tuple_parts(self.decls.value_fact(value, &env), names.len());
                let bind = names
                    .iter()
                    .zip(facts)
                    .map(|(name, fact)| (name, fact, false))
                    .collect();
                self.push_next(stmts, index, env.clone(), &scope, bind);
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
            // A store into a binding: a local one in scope (`local_fact`), or
            // a module-level one.
            Node::Assign { name, value, .. } => {
                self.decls
                    .record_local_assign(name, value, &env, &mut self.flows);
                self.decls
                    .record_global_assign(name, value, &env, &mut self.flows);
                self.push_node(value, env, &scope);
            }
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
                self.loop_body(body, var, Fact::NotStr, &env, &scope);
                self.push_node(end, env.clone(), &scope);
                self.push_node(start, env, &scope);
            }
            Node::ForEach {
                var,
                collection,
                body,
                ..
            } => {
                let fact = self.decls.element_fact(collection, &env);
                self.loop_body(body, var, fact, &env, &scope);
                self.push_node(collection, env, &scope);
            }
            Node::While { cond, body, .. } => {
                self.push_block(body, env.clone(), &scope);
                self.push_node(cond, env, &scope);
            }
            Node::Match {
                scrutinee,
                arms,
                span,
            } => {
                // A string literal arm compares bytes on every backend
                // (`__mind_string_eq`), so the scrutinee must be a string.
                let literal_arm = arms
                    .iter()
                    .any(|arm| matches!(arm.pattern, Pattern::Literal(Literal::Str(_))));
                if literal_arm && self.decls.classify(scrutinee, &env) != Class::Str {
                    self.refused.push(Refused {
                        op: BinOp::Eq,
                        kind: RefusedKind::Pattern,
                        span: *span,
                        function: scope.owner.map(str::to_string),
                    });
                }
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
                            let at = node as *const Node as usize;
                            self.targets.insert(self.ordinals.equalities[&at]);
                        }
                        Ok(false) => {}
                        Err(kind) => self.refused.push(Refused {
                            op: *op,
                            kind,
                            span: *span,
                            function: scope.owner.map(str::to_string),
                        }),
                    }
                }
                self.push_children(node, &env, &scope);
            }
            Node::Closure(data, _) => {
                let closure_scope = Scope { owner: None };
                // A capture keeps its fact unless the closure assigns it; it
                // is the closure's own copy, so no store reaches its binding.
                let assigned = assigned_in(&data.body);
                let captured: Env = Rc::new(
                    data.captures
                        .iter()
                        .filter(|name| !assigned.contains(*name))
                        .filter_map(|name| {
                            let fact = env.get(name)?.fact.clone();
                            Some((name.clone(), Bound { fact, site: None }))
                        })
                        .collect(),
                );
                let closure_env = self.decls.param_env(captured, &data.params, None, &[]);
                self.pending.push(Task::Stmts {
                    stmts: &data.body,
                    index: 0,
                    env: closure_env,
                    scope: Rc::new(closure_scope),
                    bind: Vec::new(),
                });
            }
            Node::FnDef(fd, _) => self.enter_fn(fd),
            Node::FieldAccess {
                receiver,
                field,
                span,
            } if field == "length" && self.decls.classify(receiver, &env) == Class::Str => {
                self.refused.push(Refused {
                    op: BinOp::Eq,
                    kind: RefusedKind::Member,
                    span: *span,
                    function: scope.owner.map(str::to_string),
                });
                self.push_children(node, &env, &scope);
            }
            Node::Call { callee, span, .. } => {
                if let Some(name) = self.decls.refusing_project_fn(callee, &env) {
                    self.refused.push(Refused {
                        op: BinOp::Eq,
                        kind: RefusedKind::Callee,
                        span: *span,
                        function: Some(name.to_string()),
                    });
                }
                self.push_children(node, &env, &scope);
            }
            Node::MethodCall {
                receiver, method, ..
            } => {
                if let Some(passes) = self.decls.string_method(receiver, method, &env) {
                    let at = node as *const Node as usize;
                    self.methods.insert(self.ordinals.methods[&at], passes);
                }
                self.push_children(node, &env, &scope);
            }
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
        var: &String,
        fact: Fact,
        env: &Env,
        scope: &Rc<Scope<'a>>,
    ) {
        let mut body_env = env.clone();
        self.decls.bind_local(&mut body_env, var, fact, false);
        self.push_block(body, body_env, scope);
    }

    fn push_children(&mut self, node: &'a Node, env: &Env, scope: &Rc<Scope<'a>>) {
        let children = super::super::closures::node_children_ref(node);
        for child in children.into_iter().rev() {
            self.push_node(child, env.clone(), scope);
        }
    }
}
