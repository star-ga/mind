// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! One analysis per module of the merged module `mindc test` interprets.
//!
//! `mindc test` evaluates one module: the bodies of the std functions the
//! test reaches, the declarations of the project modules it imports, and the
//! test file. Each merged item is named for its owner
//! (`crate::eval::eval_owned_symbol`), so the items split back into the
//! modules the build compiles one at a time:
//!
//! * a std module's items are decided with the facts of that module alone,
//!   walked in full under the scope the build compiles it in (std only), so
//!   a string the test hands to `vec_push` is never a fact of std.ecdsa_p256;
//! * a project module's items are decided with that module's declarations
//!   and the project summary, as its own build is;
//! * everything else (the test file, the bare std constants, the test call)
//!   is the entry module, decided as `mindc build` of that file decides it.
//!
//! A forwarder the merge builds to bind a bare name to its owner's definition
//! belongs to no module: in the build that name is the callee's opaque
//! signature, not a local function whose parameters the caller's strings
//! would reach.
//!
//! Outside `mindc test` no item is owner-named, so the whole module is the
//! entry module.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::ast::{Module, Node};
use crate::eval::module_bindings::{forwarder_target, symbol_owner};

use super::facts::Decls;
use super::slots::Contradicted;
use super::walk::fixpoint;
use super::{MethodTargets, Ordinals, Refused, Targets};

/// The module an item of the merged module belongs to, in the order the
/// analyses report refusals: the entry first, so a refusal cites the test.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Origin {
    Entry,
    Project(String),
    Std(String),
}

/// The items of one module, as runs of consecutive merged items.
pub(super) struct Segment<'a> {
    pub(super) origin: Origin,
    pub(super) runs: Vec<&'a [Node]>,
}

/// The owner a merged item is named for (through `module { .. }` blocks).
fn item_owner(item: &Node) -> Option<&str> {
    match item {
        Node::FnDef(fd, _) => symbol_owner(&fd.name),
        Node::Const { name, .. } | Node::Let { name, .. } => symbol_owner(name),
        Node::Block { stmts, .. } => stmts.iter().find_map(item_owner),
        _ => None,
    }
}

/// The owner the merge names the entry module's own items for, if any.
fn entry_owner() -> Option<String> {
    #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
    return crate::qualified_enums::current_module_path();
    #[cfg(not(any(feature = "cross-module-imports", feature = "std-surface")))]
    None
}

/// Split `module` into the modules its items belong to; forwarders are in
/// none.
pub(super) fn partition(module: &Module) -> Vec<Segment<'_>> {
    let entry = entry_owner();
    let place = |item: &Node| -> Option<Origin> {
        if let Node::FnDef(fd, _) = item
            && forwarder_target(fd).is_some()
        {
            return None;
        }
        Some(match item_owner(item) {
            Some(owner) if owner.starts_with("std.") => Origin::Std(owner.to_string()),
            Some(owner) if Some(owner) != entry.as_deref() => Origin::Project(owner.to_string()),
            _ => Origin::Entry,
        })
    };
    let mut segments: BTreeMap<Origin, Vec<&[Node]>> = BTreeMap::new();
    segments.insert(Origin::Entry, Vec::new());
    let items = module.items.as_slice();
    let mut start = 0;
    while start < items.len() {
        let origin = place(&items[start]);
        let end = start
            + items[start..]
                .iter()
                .take_while(|item| place(item) == origin)
                .count();
        if let Some(origin) = origin {
            segments.entry(origin).or_default().push(&items[start..end]);
        }
        start = end;
    }
    segments
        .into_iter()
        .map(|(origin, runs)| Segment { origin, runs })
        .collect()
}

/// The decisions of every module of the merged `module`, each from its own
/// facts; refusals in [`Origin`] order.
pub(super) fn collect_targets(
    module: &Module,
    ordinals: &Ordinals,
) -> (Targets, MethodTargets, Vec<Refused>) {
    let ctx = Ctx {
        module,
        ordinals,
        supplied: Rc::new(supplied_fns(&module.items)),
    };
    let (mut targets, mut methods, mut refused) =
        (Targets::new(), MethodTargets::new(), Vec::new());
    for segment in partition(module) {
        let (t, m, r) = ctx.analyze(&segment);
        targets.extend(t);
        methods.extend(m);
        refused.extend(r);
    }
    (targets, methods, refused)
}

/// The source name of every function the merged module defines, forwarders
/// included: a `string_<method>` call a method desugars to runs if any
/// module of the merge supplies it.
fn supplied_fns(items: &[Node]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut pending: Vec<&Node> = items.iter().collect();
    while let Some(item) = pending.pop() {
        match item {
            Node::FnDef(fd, _) => {
                names.insert(super::facts::fn_key(&fd.name).to_string());
            }
            Node::Block { stmts, .. } => pending.extend(stmts),
            _ => {}
        }
    }
    names
}

/// What every analysis of one merged module shares.
struct Ctx<'m> {
    module: &'m Module,
    ordinals: &'m Ordinals,
    supplied: Rc<BTreeSet<String>>,
}

impl Ctx<'_> {
    fn analyze(&self, segment: &Segment<'_>) -> (Targets, MethodTargets, Vec<Refused>) {
        let runs = segment.runs.as_slice();
        #[cfg(feature = "cross-module-imports")]
        let _module = match &segment.origin {
            Origin::Project(path) => Some(crate::qualified_enums::ModuleGuard::install(path)),
            _ => None,
        };
        let (decls, _scope) = match &segment.origin {
            Origin::Entry => (self.own_decls(runs), None),
            Origin::Project(path) => (self.project_decls(path, runs), None),
            Origin::Std(name) => match std_decls(name) {
                Some((decls, scope)) => (decls, Some(scope)),
                None => (self.own_decls(runs), None),
            },
        };
        let mut decls = decls;
        decls.supplied = self.supplied.clone();
        let (targets, methods, refused, _) = fixpoint(decls, self.ordinals, runs);
        (targets, methods, refused)
    }

    /// The declarations of `runs` read as items of the merged module itself.
    fn own_decls(&self, runs: &[&[Node]]) -> Decls {
        let items: Vec<&Node> = runs.iter().flat_map(|run| run.iter()).collect();
        Decls::new(self.module, &items)
    }

    /// A project module's declarations and the project summary, as its own
    /// build reads them (its module path is current).
    #[cfg(feature = "cross-module-imports")]
    fn project_decls(&self, path: &str, runs: &[&[Node]]) -> Decls {
        let project = super::project::active_modules();
        match project.as_deref().and_then(|p| p.module(path)) {
            Some(own) => {
                let items: Vec<&Node> = own.items.iter().collect();
                Decls::new(own, &items)
            }
            // shortcut: without the project registry the module's items are
            // read as the entry's; every project test installs it.
            None => self.own_decls(runs),
        }
    }

    #[cfg(not(feature = "cross-module-imports"))]
    fn project_decls(&self, _path: &str, runs: &[&[Node]]) -> Decls {
        self.own_decls(runs)
    }
}

/// The declarations a merged std module's items are decided with: its own
/// module's, with the slots its full body contradicts, under the scope its
/// build compiles it in (held by the returned guard).
fn std_decls(name: &str) -> Option<(Decls, StdScope)> {
    let ast = crate::project::stdlib::bundled_module_ast(name)?;
    let facts = std_facts(name)?;
    let scope = StdScope::install(name);
    let decls = Decls::seeded(ast, Default::default(), facts);
    Some((decls, scope))
}

thread_local! {
    /// [`std_facts`] per bundled std module.
    static FACTS: RefCell<BTreeMap<String, Option<Contradicted>>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// The slots bundled std module `name` contradicts, from a walk of its whole
/// body (not only the items one test merges), cached per thread. Local
/// bindings are keyed by AST address, so they are left out.
pub(super) fn std_facts(name: &str) -> Option<Contradicted> {
    if let Some(facts) = FACTS.with(|f| f.borrow().get(name).cloned()) {
        return facts;
    }
    let computed = crate::project::stdlib::bundled_module_ast(name).map(|ast| {
        let _scope = StdScope::install(name);
        let ordinals = super::equality_ordinals(ast);
        let decls = Decls::seeded(ast, Default::default(), Contradicted::default());
        let (_, _, _, decls) = fixpoint(decls, &ordinals, &[ast.items.as_slice()]);
        let mut facts = decls.contradicted;
        facts.locals.clear();
        facts
    });
    FACTS.with(|f| f.borrow_mut().insert(name.to_string(), computed.clone()));
    computed
}

/// The scope the build compiles bundled std module `name` in: a module table
/// and enum registry of std alone, and `name` as the current module (see
/// `project::substrate_link::std_only_scope`).
#[cfg(feature = "cross-module-imports")]
pub(super) struct StdScope {
    _table: crate::project::single_file_scope::ProjectTableGuard,
    _module: crate::qualified_enums::ModuleGuard,
}

#[cfg(feature = "cross-module-imports")]
type StdTables = (
    crate::project::module_table::ModuleTable,
    crate::ir::GlobalEnums,
);

#[cfg(feature = "cross-module-imports")]
thread_local! {
    /// The std-only table and registry per std module, built once per thread.
    static SCOPES: RefCell<BTreeMap<String, StdTables>> = const { RefCell::new(BTreeMap::new()) };
}

#[cfg(feature = "cross-module-imports")]
impl StdScope {
    pub(super) fn install(name: &str) -> Self {
        let (table, enums) = SCOPES.with(|scopes| {
            scopes
                .borrow_mut()
                .entry(name.to_string())
                .or_insert_with(|| {
                    let parsed = crate::project::stdlib::parsed_std_modules([name.to_string()]);
                    let refs: Vec<(String, &Module)> =
                        parsed.iter().map(|(p, m)| (p.clone(), m)).collect();
                    (
                        crate::project::module_table::build_module_table(&refs),
                        crate::project::build_global_enums(&parsed),
                    )
                })
                .clone()
        });
        Self {
            _table: crate::project::single_file_scope::ProjectTableGuard::install_with_enums(
                table, enums,
            ),
            _module: crate::qualified_enums::ModuleGuard::install(name),
        }
    }
}

/// Without cross-module imports there is no module table or registry to
/// scope: std code resolves against the bundled signatures alone.
#[cfg(not(feature = "cross-module-imports"))]
pub(super) struct StdScope;

#[cfg(not(feature = "cross-module-imports"))]
impl StdScope {
    pub(super) fn install(_name: &str) -> Self {
        StdScope
    }
}
