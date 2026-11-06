// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Determinism of the project code a single-file entry calls.
//!
//! `mindc <entry> --emit-evidence/--emit-obj/--emit-shared` lowers the entry
//! alone. A call into a sibling project module reaches the entry's IR as a bare
//! callee name that no registry knows, so the determinism veto and the evidence
//! label used to admit it: a sibling calling `entropy.entropy_fill`,
//! `dns.dns_resolve` or `time.now_ns` was attested `deterministic`.
//!
//! [`sibling_world_reach`] follows those calls. It starts from every callee the
//! entry module calls, resolves each through the imports the scope captured for
//! that module, lowers each project module it reaches with the same pipeline
//! `mindc build` uses for siblings, and walks from function to function, across
//! modules and transitively, with the evidence classifier applied to each body
//! (`ir::reach::fn_reach`). The entry's own bodies are not revisited: the
//! whole-module scan already classifies every one of them.
//!
//! Precision: a function is classified by its own body, so a sibling that holds
//! one world-reading function does not taint a caller of its pure functions.
//! Conservative where it cannot see: a callee that matches the exports of two
//! imported modules is followed into both. A qualified call into `std.dns` or
//! `std.entropy` is read from the sibling source, because the lowered body
//! holds only the bare name. A reached module that does not lower, or an
//! export that has no body, is refused and labelled nondeterministic: the
//! program is not shipped on an unclassified sibling. An unclassified
//! `extern "C"` callee stays a different path and is still not a hard veto.
//!
//! shortcut: a module the project supplies under a `std.` path is not followed
//! (the capture records local imports only); upgrade when such a module can be
//! linked by the single-file path.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::ir::IRModule;
use crate::ir::reach::{FnReach, called_names, fn_reach};
use crate::runtime::types::BackendTarget;

use super::ProjectScope;

/// A nondeterministic callee reached through a project module the entry calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiblingReach {
    /// The callee the classifier names (`entropy_fill`, `now_ns`), or the
    /// project function itself when its body could not be classified.
    pub offender: String,
    /// The project function whose body makes the call, as `module::fn`.
    pub via: String,
}

/// What the project modules an entry calls contribute to its determinism.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectReach {
    /// The first HARD nondeterministic callee: the determinism veto refuses it.
    pub hard: Option<SiblingReach>,
    /// The first callee the attestation counts: the evidence is labelled
    /// `nondeterministic`.
    pub taint: Option<SiblingReach>,
}

/// Follow the calls of `entry_ir` (the lowered entry of `scope`) into the
/// project modules it reaches, transitively; see the module docs.
pub fn sibling_world_reach(
    scope: &ProjectScope,
    entry_ir: &IRModule,
    target: BackendTarget,
) -> ProjectReach {
    let mut out = ProjectReach::default();
    let Some(entry) = scope.sources.first().map(|s| s.module_path.clone()) else {
        return out;
    };
    let mut queue: VecDeque<(String, String)> = called_names(&entry_ir.instrs)
        .iter()
        .flat_map(|callee| imported_targets(scope, &entry, callee))
        .collect();
    if queue.is_empty() {
        return out;
    }
    let _scope = scope.install();
    let mut lowered: BTreeMap<String, Option<BTreeMap<String, FnReach>>> = BTreeMap::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    while let Some((owner, name)) = queue.pop_front() {
        if owner == entry || !seen.insert((owner.clone(), name.clone())) {
            continue;
        }
        let fns = lowered
            .entry(owner.clone())
            .or_insert_with(|| lower(scope, &owner, target));
        let via = format!("{owner}::{name}");
        let ast_offender = scope
            .sources
            .iter()
            .find(|source| source.module_path == owner)
            .and_then(|source| crate::std_qualified::reached_offender(&source.source, &name));
        let Some(reach) = fns.as_ref().and_then(|fns| fns.get(&name)) else {
            // No lowered body. Refuse and label, even when the source shows no
            // qualified std call: an unclassifiable sibling is not shipped.
            // The two assignments are separate so each one is load-bearing.
            let offender = ast_offender.unwrap_or_else(|| name.clone());
            if out.hard.is_none() {
                out.hard = Some(SiblingReach {
                    offender: offender.clone(),
                    via: via.clone(),
                });
            }
            if out.taint.is_none() {
                out.taint = Some(SiblingReach { offender, via });
            }
            continue;
        };
        if let Some(offender) = ast_offender {
            if out.hard.is_none() {
                out.hard = Some(SiblingReach {
                    offender: offender.clone(),
                    via: via.clone(),
                });
            }
            if out.taint.is_none() {
                out.taint = Some(SiblingReach {
                    offender,
                    via: via.clone(),
                });
            }
        }
        let found = |offender: &String| SiblingReach {
            offender: offender.clone(),
            via: via.clone(),
        };
        if out.hard.is_none() {
            out.hard = reach.hard.as_ref().map(found);
        }
        if out.taint.is_none() {
            out.taint = reach.taint.as_ref().map(found);
        }
        let local = fns.as_ref().map(|fns| {
            reach
                .calls
                .iter()
                .filter(|c| fns.contains_key(*c))
                .cloned()
                .collect::<Vec<_>>()
        });
        for callee in local.unwrap_or_default() {
            queue.push_back((owner.clone(), callee));
        }
        for callee in &reach.calls {
            queue.extend(imported_targets(scope, &owner, callee));
        }
    }
    out
}

/// `(module, fn)` for every project module `owner` imports that exports a
/// function named by `callee`'s last segment. A callee no import exports (a
/// local, std, intrinsic or builtin call) has none.
fn imported_targets(scope: &ProjectScope, owner: &str, callee: &str) -> Vec<(String, String)> {
    let name = callee.rsplit(['.', ':']).next().unwrap_or(callee);
    scope
        .resolved_imports
        .iter()
        .filter(|((importer, _), _)| importer == owner)
        .map(|(_, target)| target)
        .filter(|target| {
            scope
                .table
                .get(target)
                .is_some_and(|module| module.exported_fns.iter().any(|f| f.name == name))
        })
        .map(|target| (target.clone(), name.to_string()))
        .collect()
}

/// Lower the captured source of module `owner` as `mindc build` lowers a
/// sibling, under its own module identity; `None` when it does not lower.
fn lower(
    scope: &ProjectScope,
    owner: &str,
    target: BackendTarget,
) -> Option<BTreeMap<String, FnReach>> {
    let source = scope.sources.iter().find(|s| s.module_path == owner)?;
    let _module = crate::qualified_enums::ModuleGuard::install(owner);
    let opts = crate::pipeline::CompileOptions {
        target,
        ..Default::default()
    };
    let name = source.path.to_string_lossy().into_owned();
    crate::pipeline::compile_source_with_name(&source.source, Some(&name), &opts)
        .ok()
        .map(|products| fn_reach(&products.ir))
}
