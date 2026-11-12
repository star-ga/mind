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
//! them. Functions are keyed by their source name, structs by
//! `slots::struct_key` and variants by their registry key, so two modules
//! declaring the same function name share the set, which can only refuse
//! more.
//!
//! Element facts are the exception: they are keyed by the element fact, not
//! by a slot, so a store into one module's own `[u8; N]` would name every
//! `[u8; N]` of the project. They stay with the module that stores them and
//! reach only the modules it can exchange a collection with: two modules one
//! module of the project imports, directly or not (itself included). A
//! collection crosses modules only through a call, a result, a field or a
//! payload, and a module names another's functions and types only through
//! an import, so a module no importer shares with another cannot hand it a
//! collection.
//!
//! A `const` a module reads but another declares is read as its declaring
//! module reads it (see `const_facts`): a const keeps its value, so handing
//! one to an `i64` parameter stores no value of unknown type there.
//!
//! `mindc test` runs this analysis on a worker thread that has the project
//! registry but not the module table; it installs the table the project's
//! own check reads (see `install_scope`), so both decide one analysis.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use std::sync::{Arc, OnceLock};

use crate::ast::{Module, Node, TypeAnn};
use crate::project::active_module_table::{Guard, ResolvedImportsGuard};
use crate::project::module_table::ModuleTable;

use super::facts::{Decls, Fact, Fns};
use super::slots::Contradicted;

/// Importing module and written import path → the module it selects.
type ResolvedImports = BTreeMap<(String, Vec<String>), String>;

/// The modules of one project build other than the bundled std, recorded
/// with the whole-project registry (`crate::ir::GlobalEnums`), which every
/// project check, build and test installs and removes. For a check or a
/// test of one file, the modules of the program that file links (see
/// [`Self::of_program`]).
#[derive(Debug)]
pub(crate) struct ProjectModules {
    modules: Vec<(String, Module)>,
    /// The names of the `const`s the project's std modules declare.
    std_consts: BTreeSet<String>,
    summary: OnceLock<Arc<Summary>>,
    /// The module table and import resolution the project's own check
    /// installs (see [`install_scope`]).
    scope: OnceLock<(ModuleTable, ResolvedImports)>,
}

impl ProjectModules {
    /// The project's modules, or `None` when it has fewer than two besides
    /// the bundled std (`std.*`): one module's own analysis then sees every
    /// store into its slots.
    pub(crate) fn of(parsed: &[(String, Module)]) -> Option<Arc<Self>> {
        Self::of_where(parsed, |_| true).map(Arc::new)
    }

    /// The modules of the program `linked` names (a file and every project
    /// module it imports, directly or not), with the module table and import
    /// resolution its check installs (see [`install_scope`]). A value moves
    /// between modules only inside one program, so a module the program does
    /// not link stores nothing it reads.
    pub(crate) fn of_program(
        parsed: &[(String, Module)],
        linked: &BTreeSet<String>,
        table: ModuleTable,
        resolved: ResolvedImports,
    ) -> Option<Arc<Self>> {
        let program = Self::of_where(parsed, |path| linked.contains(path))?;
        let _ = program.scope.set((table, resolved));
        Some(Arc::new(program))
    }

    /// The project modules of `parsed` whose path `keep` admits, with the
    /// names of the consts of every std module of `parsed`.
    fn of_where(parsed: &[(String, Module)], keep: impl Fn(&str) -> bool) -> Option<Self> {
        let modules: Vec<(String, Module)> = parsed
            .iter()
            .filter(|(path, _)| !path.starts_with("std.") && keep(path))
            .cloned()
            .collect();
        let std_consts = parsed
            .iter()
            .filter(|(path, _)| path.starts_with("std."))
            .flat_map(|(_, module)| const_names(module))
            .map(str::to_string)
            .collect();
        (modules.len() > 1).then(|| Self {
            modules,
            std_consts,
            summary: OnceLock::new(),
            scope: OnceLock::new(),
        })
    }

    /// The AST of project module `path`.
    pub(super) fn module(&self, path: &str) -> Option<&Module> {
        self.modules
            .iter()
            .find_map(|(p, module)| (p == path).then_some(module))
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
    /// bindings and element facts excluded).
    pub(super) slots: Contradicted,
    /// Module path → the element facts that module stores a contradicting
    /// value into.
    elements: BTreeMap<String, Vec<Fact>>,
    /// Module path → the modules it can exchange a collection with.
    related: BTreeMap<String, BTreeSet<String>>,
    /// Every function in which its own module refuses a comparison, once
    /// every store is known.
    pub(super) refusing: Arc<BTreeSet<String>>,
    /// The fact of each `const` the project's modules declare (see
    /// [`const_facts`]).
    pub(super) consts: Arc<BTreeMap<String, Fact>>,
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

/// The modules of the active project build, when it has several.
pub(super) fn active_modules() -> Option<Arc<ProjectModules>> {
    crate::ir::with_global_enums(|g| g.string_project.clone())
}

/// Holds the module table and import resolution [`install_scope`]
/// installed, and restores the previous ones on drop.
pub(super) struct ScopeGuard {
    _scope: Option<(Guard, ResolvedImportsGuard)>,
}

/// Install the module table and import resolution the active project's own
/// check reads, while none is installed: `mindc test` analyses its merged
/// module on a worker thread that has the project registry and the entry's
/// module path but no module table, so without them an imported function's
/// declaration (`cm_all_imported_fn_signatures`) and an import's target
/// (`related_modules`) would be missing and the test would decide another
/// analysis than the build of the same modules.
pub(super) fn install_scope() -> ScopeGuard {
    let installed = crate::project::active_module_table::with(|table| table.is_some());
    let scope = (!installed)
        .then(active_modules)
        .flatten()
        .and_then(|project| project.scope.get().cloned())
        .map(|(table, resolved)| {
            (
                Guard::install(table),
                ResolvedImportsGuard::install(resolved),
            )
        });
    ScopeGuard { _scope: scope }
}

/// The summary for the module being analysed, when it is a module of the
/// active project build; computed on the first such analysis.
pub(super) fn current() -> Option<Arc<Summary>> {
    if COMPUTING.with(Cell::get) {
        return None;
    }
    let project = active_modules()?;
    let path = crate::qualified_enums::current_module_path()?;
    if !project.modules.iter().any(|(p, _)| *p == path) {
        return None;
    }
    Some(
        project
            .summary
            .get_or_init(|| Arc::new(compute(&project.modules, &project.std_consts)))
            .clone(),
    )
}

impl Summary {
    /// The slots module `path` starts its analysis from: every shared slot,
    /// and the element facts of the modules it can exchange a collection
    /// with.
    ///
    /// A module the summary does not know gets every module's element facts
    /// (fail closed).
    pub(super) fn seed(&self, path: &str) -> Contradicted {
        let mut seed = self.slots.shared();
        let elements: Vec<&Fact> = match self.related.get(path) {
            Some(related) => related
                .iter()
                .filter_map(|other| self.elements.get(other))
                .flatten()
                .collect(),
            None => self.elements.values().flatten().collect(),
        };
        for element in elements {
            seed.add_element(element.clone());
        }
        seed
    }

    /// The seed of the module being analysed (see [`Self::seed`]); every
    /// module's element facts when no module is being analysed.
    pub(super) fn current_seed(&self) -> Contradicted {
        let path = crate::qualified_enums::current_module_path().unwrap_or_default();
        self.seed(&path)
    }
}

/// Walk every module, each with the slots the others store into, until no
/// module records a new one. Slots are only ever added and are finite, so
/// this ends.
fn compute(modules: &[(String, Module)], std_consts: &BTreeSet<String>) -> Summary {
    let _computing = Computing::enter();
    let unseeded = vec![Contradicted::default(); modules.len()];
    let mut summary = Summary {
        fns: Arc::new(declared_fns(modules)),
        slots: Contradicted::default(),
        elements: BTreeMap::new(),
        related: related_modules(modules),
        refusing: Arc::default(),
        consts: Arc::new(const_facts(modules, std_consts, &unseeded, &Arc::default())),
    };
    // shortcut: every round walks every module again; a chain of stores
    // through N modules costs N rounds. Upgrade to a per-module worklist if a
    // project ever needs many.
    loop {
        let mut grew = false;
        let mut refusing = BTreeSet::new();
        let mut walked = Vec::with_capacity(modules.len());
        for (path, module) in modules {
            let _module = crate::qualified_enums::ModuleGuard::install(path.clone());
            let ordinals = super::equality_ordinals(module);
            let mut decls = Decls::seeded(module, summary.fns.clone(), summary.seed(path));
            decls.project_consts = summary.consts.clone();
            let runs = [module.items.as_slice()];
            let (_, _, refused, decls, own) =
                super::walk::fixpoint_with_elements(decls, &ordinals, &runs);
            refusing.extend(refused.into_iter().filter_map(|r| r.function));
            grew |= summary.slots.absorb(decls.contradicted.shared());
            let stored = summary.elements.entry(path.clone()).or_default();
            for element in own {
                if !stored.contains(&element) {
                    stored.push(element);
                    grew = true;
                }
            }
            walked.push(decls.contradicted);
        }
        // Each const read again with every store its module now knows: a
        // fact only weakens (joins) or is dropped, so this ends too.
        let read = const_facts(modules, std_consts, &walked, &summary.consts);
        let consts: BTreeMap<String, Fact> = summary
            .consts
            .iter()
            .filter_map(|(name, fact)| {
                let now = read.get(name)?;
                Some((name.clone(), super::facts::join(fact.clone(), now.clone())))
            })
            .collect();
        if consts != *summary.consts {
            summary.consts = Arc::new(consts);
            grew = true;
        }
        if !grew {
            // This round walked every module with every store known.
            summary.refusing = Arc::new(refusing);
            return summary;
        }
    }
}

/// Module path → the modules it can exchange a collection with: every
/// module that a module of the project imports, directly or not, together
/// with it (itself included). An import names a module by the trailing
/// segments of its path; one naming several modules relates them all,
/// which can only refuse more.
// shortcut: a module-level binding is not treated as a channel between two
// modules that only share an import. A compiled build keeps no value a
// function assigns to one (a const keeps its value, a `let` refuses with
// E5005). Upgrade when that changes: relate every module that reaches a
// module whose binding a function assigns.
fn related_modules(modules: &[(String, Module)]) -> BTreeMap<String, BTreeSet<String>> {
    let imports: Vec<BTreeSet<usize>> = modules
        .iter()
        .map(|(path, module)| imported(path, module, modules))
        .collect();
    let mut related: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for start in 0..modules.len() {
        // The modules `start` imports, directly or not, itself included.
        let mut reached = BTreeSet::from([start]);
        let mut pending = vec![start];
        while let Some(at) = pending.pop() {
            for &next in &imports[at] {
                if reached.insert(next) {
                    pending.push(next);
                }
            }
        }
        for &a in &reached {
            let at = related.entry(modules[a].0.clone()).or_default();
            at.extend(reached.iter().map(|&b| modules[b].0.clone()));
        }
    }
    related
}

/// The indices of the project modules `module` (at `owner`) imports: the
/// module the build's import resolver selects, and every module whose path
/// ends with the import's segments.
fn imported(owner: &str, module: &Module, modules: &[(String, Module)]) -> BTreeSet<usize> {
    let mut found = BTreeSet::new();
    let mut pending: Vec<&Node> = module.items.iter().collect();
    while let Some(item) = pending.pop() {
        match item {
            Node::Import { path, .. } => {
                let name = path.join(".");
                let suffix = format!(".{name}");
                // `import lib;` of `src/x/lib.mind` resolves to `crate.x`, which
                // no suffix match finds.
                let resolved =
                    crate::project::active_module_table::resolve_import(Some(owner), path);
                found.extend(
                    modules
                        .iter()
                        .enumerate()
                        .filter(|(_, (p, _))| {
                            *p == name || p.ends_with(&suffix) || resolved.as_ref() == Some(p)
                        })
                        .map(|(index, _)| index),
                );
            }
            Node::Block { stmts, .. } => pending.extend(stmts),
            _ => {}
        }
    }
    found
}

/// The names of the module-level `const`s of `module`.
fn const_names(module: &Module) -> Vec<&str> {
    let mut names = Vec::new();
    let mut pending: Vec<&Node> = module.items.iter().collect();
    while let Some(item) = pending.pop() {
        match item {
            Node::Const { name, .. } => names.push(name.as_str()),
            Node::Block { stmts, .. } => pending.extend(stmts),
            _ => {}
        }
    }
    names
}

/// Name → the fact of the `const` of that name the project's modules
/// declare, as its own module reads it, for a module that reads the name
/// without declaring it. A const keeps its value, so its declaration (and,
/// where the value contradicts it, its value) says what every read of it
/// holds. Lowering inlines a sibling's const by its bare name (whichever
/// module declares it), so a name that two modules declare with different
/// facts, or that a std module declares too, is left out: it stays unknown.
/// A record its module names by its bare name is named by its owner-keyed
/// registry entry (`crate.a.Foo`), which every module reads the same; a
/// fact with a record no such entry pins is left out too.
///
/// Each module's consts are read with `seeds` (that module's stores, by
/// position) and the project consts `read` already known.
fn const_facts(
    modules: &[(String, Module)],
    std_consts: &BTreeSet<String>,
    seeds: &[Contradicted],
    read: &Arc<BTreeMap<String, Fact>>,
) -> BTreeMap<String, Fact> {
    let mut facts: BTreeMap<String, Option<Fact>> = BTreeMap::new();
    for ((path, module), seed) in modules.iter().zip(seeds) {
        let names = const_names(module);
        if names.is_empty() {
            continue;
        }
        let _module = crate::qualified_enums::ModuleGuard::install(path.clone());
        let mut decls = Decls::seeded(module, Arc::default(), seed.clone());
        decls.project_consts = read.clone();
        for name in names {
            // A const its own module assigns holds that value too.
            let fact = if seed.globals.contains(name) {
                Some(Fact::MaybeStr)
            } else {
                decls.global_declared(super::facts::fn_key(name))
            };
            let fact = fact.and_then(|fact| pin_records(fact, path));
            match facts.get_mut(name) {
                Some(slot) if *slot != fact => *slot = None,
                Some(_) => {}
                None => {
                    facts.insert(name.to_string(), fact);
                }
            }
        }
    }
    facts
        .into_iter()
        .filter(|(name, _)| !std_consts.contains(name))
        .filter_map(|(name, fact)| Some((name, fact?)))
        .collect()
}

/// `fact` as any module reads it: each record module `owner` names by its
/// bare name renamed to its owner-keyed registry entry; `None` when one has
/// no such entry.
fn pin_records(fact: Fact, owner: &str) -> Option<Fact> {
    let pin = |part: Box<Fact>| pin_records(*part, owner).map(Box::new);
    Some(match fact {
        Fact::Record(name) if name.contains('.') => Fact::Record(name),
        Fact::Record(name) => {
            let key = format!("{owner}.{name}");
            #[cfg(feature = "std-surface")]
            let pinned = crate::ir::with_global_enums(|g| g.qualified_structs.contains_key(&key));
            #[cfg(not(feature = "std-surface"))]
            let pinned = false;
            return pinned.then_some(Fact::Record(key));
        }
        Fact::Elems(part) => Fact::Elems(pin(part)?),
        Fact::Map(part) => Fact::Map(pin(part)?),
        Fact::Sum(ok, err) => Fact::Sum(pin(ok)?, pin(err)?),
        Fact::Tuple(parts) => Fact::Tuple(
            parts
                .into_iter()
                .map(|part| pin_records(part, owner))
                .collect::<Option<_>>()?,
        ),
        other => other,
    })
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
