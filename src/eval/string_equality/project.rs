// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Slots of one project build that another module of the project stores
//! into.
//!
//! A project build analyses one module at a time, but a module can store a
//! string into a slot another module declares: an `i64` parameter of an
//! imported function (`provider.eqh("abc", "abc")`), an `i64` field of an
//! imported struct, a std collection. And it can read a slot another module
//! stores a string into: the `i64` result of an imported function that
//! returns one. The declaring module alone would read such a slot as its
//! declaration says, so `==` on two of them would compare string handles.
//!
//! So the whole-project registry carries the project's modules (all but the
//! bundled std, whose stores are tracked by the collections they keep), and
//! the first analysis of a project module walks every one of them, each with
//! the slots the others store into, until no module records a new one. Every
//! module's own analysis then starts from that set: a slot any module of the
//! project stores a contradicting value into is maybe a string in all of
//! them. Functions and structs are keyed by name, so two modules declaring
//! the same name share the set, which can only refuse more.

use std::cell::Cell;
use std::collections::BTreeSet;

use std::sync::{Arc, OnceLock};

use crate::ast::{Module, Node, TypeAnn};

use super::facts::{Decls, Fns};
use super::slots::Contradicted;

/// The modules of one project build other than the bundled std, recorded
/// with the whole-project registry (`crate::ir::GlobalEnums`), which every
/// project check, build and test installs and removes.
#[derive(Debug)]
pub(crate) struct ProjectModules {
    modules: Vec<(String, Module)>,
    summary: OnceLock<Arc<Summary>>,
}

impl ProjectModules {
    /// The project's modules, or `None` when it has fewer than two besides
    /// the bundled std (`std.*`): one module's own analysis then sees every
    /// store into its slots.
    pub(crate) fn of(parsed: &[(String, Module)]) -> Option<Arc<Self>> {
        let modules: Vec<(String, Module)> = parsed
            .iter()
            .filter(|(path, _)| !path.starts_with("std."))
            .cloned()
            .collect();
        (modules.len() > 1).then(|| {
            Arc::new(Self {
                modules,
                summary: OnceLock::new(),
            })
        })
    }

    /// `enums` carrying the modules of `parsed` (see [`Self::of`]).
    pub(crate) fn attach(
        parsed: &[(String, Module)],
        mut enums: crate::ir::GlobalEnums,
    ) -> crate::ir::GlobalEnums {
        enums.string_project = Self::of(parsed);
        enums
    }
}

/// What every module of a project stores into the slots of the others.
#[derive(Debug, Default)]
pub(super) struct Summary {
    /// Every function a project module declares.
    pub(super) fns: Arc<Fns>,
    /// Every slot a module stores a contradicting value into (module-local
    /// bindings excluded).
    pub(super) slots: Contradicted,
    /// Every function in which its own module refuses a comparison, once
    /// every store is known.
    pub(super) refusing: Arc<BTreeSet<String>>,
}

thread_local! {
    /// Set while [`compute`] walks the project, whose analyses take the
    /// partial summary explicitly.
    static COMPUTING: Cell<bool> = const { Cell::new(false) };
}

struct Computing;

impl Computing {
    fn enter() -> Self {
        COMPUTING.with(|c| c.set(true));
        Computing
    }
}

impl Drop for Computing {
    fn drop(&mut self) {
        COMPUTING.with(|c| c.set(false));
    }
}

/// The summary for the module being analysed, when it is a module of the
/// active project build; computed on the first such analysis.
pub(super) fn current() -> Option<Arc<Summary>> {
    if COMPUTING.with(Cell::get) {
        return None;
    }
    let project = crate::ir::with_global_enums(|g| g.string_project.clone())?;
    let path = crate::qualified_enums::current_module_path()?;
    if !project.modules.iter().any(|(p, _)| *p == path) {
        return None;
    }
    Some(
        project
            .summary
            .get_or_init(|| Arc::new(compute(&project.modules)))
            .clone(),
    )
}

/// Walk every module, each with the slots the others store into, until no
/// module records a new one. Slots are only ever added and are finite, so
/// this ends.
fn compute(modules: &[(String, Module)]) -> Summary {
    let _computing = Computing::enter();
    let mut summary = Summary {
        fns: Arc::new(declared_fns(modules)),
        slots: Contradicted::default(),
        refusing: Arc::default(),
    };
    // shortcut: every round walks every module again; a chain of stores
    // through N modules costs N rounds. Upgrade to a per-module worklist if a
    // project ever needs many.
    loop {
        let mut grew = false;
        let mut refusing = BTreeSet::new();
        for (path, module) in modules {
            let _module = crate::qualified_enums::ModuleGuard::install(path.clone());
            let ordinals = super::equality_ordinals(module);
            let decls = Decls::seeded(module, summary.fns.clone(), summary.slots.shared());
            let (_, _, refused, decls) = super::walk::fixpoint(decls, &ordinals, module);
            refusing.extend(refused.into_iter().filter_map(|r| r.function));
            grew |= summary.slots.absorb(decls.contradicted.shared());
        }
        if !grew {
            // This round walked every module with every store known.
            summary.refusing = Arc::new(refusing);
            return summary;
        }
    }
}

/// Every function the modules declare, with its parameter types.
fn declared_fns(modules: &[(String, Module)]) -> Fns {
    let mut fns = Fns::new();
    for (_, module) in modules {
        let mut pending: Vec<&Node> = module.items.iter().collect();
        while let Some(item) = pending.pop() {
            match item {
                Node::FnDef(fd, _) => {
                    let types: Vec<TypeAnn> = fd.params.iter().map(|p| p.ty.clone()).collect();
                    match fns.get_mut(&fd.name) {
                        Some(slot) if slot.as_ref() != Some(&types) => *slot = None,
                        Some(_) => {}
                        None => {
                            fns.insert(fd.name.clone(), Some(types));
                        }
                    }
                }
                Node::Block { stmts, .. } => pending.extend(stmts),
                _ => {}
            }
        }
    }
    fns
}
