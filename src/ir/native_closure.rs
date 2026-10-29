// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Entry-rooted admission closure for the `--backend native` multi-module bridge.
//!
//! # Why this file exists
//!
//! The frozen-profile fence ([`crate::ir::frozen_profile::profile_frozen_admits`])
//! scans EVERY function in a lowered module. That is correct for a single
//! translation unit, where every top-level item is part of the program the user
//! wrote. It is wrong once a program imports siblings: an out-of-profile body in
//! an imported module refuses an entry that never calls it. Measured 2026-09-07 on
//! the shipped compiler: a file whose `main` returns `7` is refused `binop.div`
//! because an UNCALLED `unused_div` elsewhere in the file contains `/`.
//!
//! This module computes the set of functions reachable from the declared roots.
//!
//! **That view is DIAGNOSTIC ONLY.** The fence shipped today is applied to the
//! WHOLE IMAGE: `admit_merged_program` runs this admission for its ownership,
//! ambiguity and unresolved-edge refusals and then DISCARDS the restricted
//! module, handing the full lowering to `profile_frozen_admits`. An uncalled
//! out-of-profile body therefore still refuses the build. The reachability set
//! exists to explain WHICH definitions a refusal concerns, not to narrow what is
//! admitted, and the older description of it as narrowing the fence was wrong.
//!
//! # What this is NOT
//!
//! It is **not** dead-code elimination. The wire image still carries the full
//! source text of every linked module, and the frozen compiler still emits
//! unreachable bodies into the artifact (measured: a module with one uncalled
//! function is 541 bytes against 397 for the same module without it). Restricting
//! the module here changes ADMISSION only. Retaining the text is deliberate:
//! pruning it would change the emitted bytes of single-file programs that contain
//! an uncalled in-profile helper, which the byte-preservation obligation forbids.
//!
//! # Fail-closed obligations
//!
//! Every operation here refuses rather than guesses:
//!
//! * A call whose callee has no definition in the merged module is an
//!   `unresolved_edge` refusal, never a silently dropped edge. Under-approximating
//!   reachability would hide an out-of-profile body from the fence — fail-OPEN, the
//!   one failure this module must not have.
//! * Two definitions of one bare name are a `duplicate_definition` refusal. The
//!   frozen compiler resolves the flat image last-definition-wins, silently:
//!   measured, modules `a,b,main` return 20 and `b,a,main` return 10 for the same
//!   three sources. Until a symbol projection exists, ambiguity is refused, never
//!   resolved by position.
//! * A name colliding with the std seed blob is a `std_symbol_collision` refusal,
//!   for the same reason — std prefixing is convention, not a guarantee
//!   (`std/toml.mind` ships a bare `bytes_eq`, `std/io.mind` a bare `stdin`).
//!   The `reserved` set is the caller's: the historical bridge removes from it
//!   only names whose user definition is an exact private copy of the seed's
//!   single private definition, where either copy denotes the same function.
//!
//! Traversal goes through [`crate::ir::instr_bodies`], the single enumeration of
//! body-carrying variants, so a newly added nested-body variant cannot silently
//! become invisible to reachability while remaining visible to the fence.
//!
//! The versioned candidate lane has a separate entry point below. Its caller
//! must first validate the complete source and IR under that lane's typed
//! profile. It then checks EVERY retained function against exact compiler ABI
//! bindings, with reserved-definition and arity guards. The historical APIs
//! keep their original strict behavior. Prefix omission, when eligible, is
//! performed by the bridge's namespace module; no user body is removed here.

#![cfg(feature = "std-surface")]

use crate::ir::{IRModule, Instr, instr_bodies};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Why a closure could not be formed. `kind` is a stable label for tests and
/// diagnostics; `detail` names the offending symbol so a refusal is actionable
/// (the fence's own rejection carries only a construct class, never a callee).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureRejection {
    pub kind: &'static str,
    pub detail: String,
}

fn reject<T>(kind: &'static str, detail: impl Into<String>) -> Result<T, ClosureRejection> {
    Err(ClosureRejection {
        kind,
        detail: detail.into(),
    })
}

/// Every function definition in `instrs`, at any nesting depth, with the number of
/// times each name is defined.
///
/// Crosses `FnDef` boundaries deliberately: a nested definition still contributes a
/// name to the flat image the frozen compiler parses, so it can still collide.
fn definition_counts(instrs: &[Instr], out: &mut BTreeMap<String, usize>) {
    for instr in instrs {
        if let Instr::FnDef { name, .. } = instr {
            *out.entry(name.clone()).or_insert(0) += 1;
        }
        for body in instr_bodies(instr) {
            definition_counts(body, out);
        }
    }
}

/// Names of every function defined anywhere in `module`.
pub fn defined_fn_names(module: &IRModule) -> BTreeSet<String> {
    let mut counts = BTreeMap::new();
    definition_counts(&module.instrs, &mut counts);
    counts.into_keys().collect()
}

// The IR-level duplicate detector that used to live here has been REMOVED, not
// left unwired. It crossed `FnDef` boundaries, which the AST-level check does
// not, so the question was whether a duplicate hidden in a nested or
// block-wrapped definition could evade detection. Measured on the frozen
// compiler:
//
//   `module inner { pub fn v() ... }`   -> "unsupported construct", refused
//   `fn outer() { fn inner() ... }`     -> "unsupported construct", refused
//   two `fn v()` on ONE line            -> refused, closure.duplicate_definition
//
// Neither nesting shape compiles natively at all, so an IR-level counter that
// descends into them can refuse nothing the frozen compiler does not already
// refuse; and the one-line case the line-oriented scan used to miss is caught by
// the AST counter. Keeping a second detector that cannot fire would be a
// duplicate implementation with nothing asserting the two agree -- the defect
// class this file already documents elsewhere.
//
// `definition_counts` itself is retained and IS live: `reachable_from` uses it
// through `defined_fn_names`, where crossing `FnDef` boundaries is correct
// because a nested definition is still a resolvable call target.

/// Direct call edges out of one instruction stream, not descending into nested
/// `FnDef` bodies (those are separate nodes in the call graph).
fn direct_calls(instrs: &[Instr], out: &mut Vec<(String, usize)>) {
    for instr in instrs {
        if let Instr::Call { name, args, .. } = instr {
            out.push((name.clone(), args.len()));
        }
        if matches!(instr, Instr::FnDef { .. }) {
            continue;
        }
        for body in instr_bodies(instr) {
            direct_calls(body, out);
        }
    }
}

/// The body of one named function, searched at any nesting depth.
fn find_body<'a>(instrs: &'a [Instr], want: &str) -> Option<&'a [Instr]> {
    for instr in instrs {
        if let Instr::FnDef { name, body, .. } = instr {
            if name == want {
                return Some(body.as_slice());
            }
        }
        for nested in instr_bodies(instr) {
            if let Some(found) = find_body(nested, want) {
                return Some(found);
            }
        }
    }
    None
}

/// Roots for an executable build: the configured entry, plus every exported symbol.
///
/// A function named by an `export { ... }` block can execute WITHOUT being called
/// from the entry, so omitting it would make the closure smaller than the set that
/// can actually run — hiding an out-of-profile body from the fence. That is the
/// fail-OPEN direction, so exports are roots.
///
/// Scope, precisely: this is `IRModule.exports`, which lowering populates ONLY
/// from `export { ... }` blocks (`src/eval/lower.rs:1560`, `:10081`). A bare
/// `pub fn` does NOT reach it — verified, not assumed. So "externally callable"
/// here means the declared export set, not every public function.
///
/// `IRModule.exports` is a `HashSet`, whose iteration order is not stable. The
/// result is collected through a `BTreeSet` so the root list — and therefore which
/// symbol a refusal names first — is deterministic across runs.
pub fn executable_roots(module: &IRModule, entry: &str) -> Vec<String> {
    let mut roots: BTreeSet<String> = BTreeSet::new();
    roots.insert(entry.to_string());
    roots.extend(module.exports.iter().cloned());
    roots.into_iter().collect()
}

/// Functions reachable from `roots` by direct call edges.
///
/// Refuses on a missing root and on any call to a name with no definition. The
/// latter covers builtins, intrinsics and externs, which the fence rejects anyway
/// (`admit_call` admits only callees defined in the same module) — refusing here
/// keeps the two in agreement and names the callee, which the fence cannot.
pub fn reachable_from(
    module: &IRModule,
    roots: &[String],
) -> Result<BTreeSet<String>, ClosureRejection> {
    reachable_with_bindings(module, roots, &BTreeMap::new())
}

fn reachable_with_bindings(
    module: &IRModule,
    roots: &[String],
    bindings: &BTreeMap<String, usize>,
) -> Result<BTreeSet<String>, ClosureRejection> {
    let defined = defined_fn_names(module);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    for root in roots {
        if !defined.contains(root) {
            return reject(
                "closure.missing_entry_root",
                format!("declared root `{root}` is not defined in the linked closure"),
            );
        }
        if seen.insert(root.clone()) {
            queue.push_back(root.clone());
        }
    }
    while let Some(current) = queue.pop_front() {
        let body = match find_body(&module.instrs, &current) {
            Some(b) => b,
            None => {
                return reject(
                    "closure.missing_body",
                    format!("`{current}` is named but has no body"),
                );
            }
        };
        let mut callees = Vec::new();
        direct_calls(body, &mut callees);
        for (callee, arity) in callees {
            if let Some(expected) = bindings.get(&callee) {
                if arity != *expected {
                    return reject(
                        "closure.intrinsic_arity",
                        format!(
                            "`{current}` calls `{callee}` with {arity} arguments; expected {expected}"
                        ),
                    );
                }
                continue;
            }
            if !defined.contains(&callee) {
                return reject(
                    "closure.unresolved_edge",
                    format!("`{current}` calls `{callee}`, which has no definition in the closure"),
                );
            }
            if seen.insert(callee.clone()) {
                queue.push_back(callee);
            }
        }
    }
    Ok(seen)
}

/// A view of `module` keeping only top-level functions in `keep`.
///
/// Non-function top-level instructions are retained unchanged: they are module
/// state, not call-graph nodes, and dropping them would change what the fence sees
/// about the module's constants and float taint seed.
///
/// This filters ONE lowered module. It never splices instructions from separately
/// lowered modules: `IRModule::default` starts `next_id` at 0, so concatenating two
/// independently lowered modules aliases `ValueId`s and corrupts `value_types`. The
/// merged program must come from lowering one merged AST.
pub fn restrict_to_reachable(module: &IRModule, keep: &BTreeSet<String>) -> IRModule {
    let mut out = module.clone();
    out.instrs.retain(|instr| match instr {
        // A top-level function is kept when it is itself reachable OR when it
        // encloses a reachable nested definition. `definition_counts` crosses
        // `FnDef` boundaries, so a nested `inner` is a resolvable, walkable node;
        // dropping its enclosing `outer` would leave the closure naming a
        // definition the restricted module no longer contains. The fence would
        // catch that as an undefined call, but the two halves of this module must
        // not disagree about what a definition is.
        Instr::FnDef { name, body, .. } => {
            keep.contains(name) || encloses_kept_definition(body, keep)
        }
        _ => true,
    });
    out
}

/// Does `instrs` contain a function definition, at any depth, that is kept?
fn encloses_kept_definition(instrs: &[Instr], keep: &BTreeSet<String>) -> bool {
    for instr in instrs {
        if let Instr::FnDef { name, .. } = instr {
            if keep.contains(name) {
                return true;
            }
        }
        for body in instr_bodies(instr) {
            if encloses_kept_definition(body, keep) {
                return true;
            }
        }
    }
    false
}

/// Function definitions counted in the PARSED AST, not the lowered IR.
///
/// This is the representation the frozen compiler's own parser sees, and the two
/// disagree in ways that matter for collision detection: lowering drops generic
/// templates (`src/eval/lower.rs:5845-5849`) and any user `fn __mind_*`
/// (`:5835-5839`) before an `Instr::FnDef` is ever produced. A `fn v<T>` beside a
/// `fn v()` is ONE definition in IR and TWO `fn v` in the text the frozen parser
/// resolves last-definition-wins — so an IR-level check would admit exactly the
/// collision this module exists to refuse.
pub fn ast_definition_counts(module: &crate::ast::Module) -> BTreeMap<String, usize> {
    // Counts definitions THROUGH transparent blocks, not just direct items.
    //
    // `module NAME { ... }` is parsed into a transparent `Node::Block`, so its
    // functions are not direct `module.items`. Counting only direct items left
    // two same-named functions in two module blocks uncounted, and this guard
    // silent. Measured before this traversal existed: such a program type-checks
    // (the only `check` diagnostic was formatting drift) and was refused
    // natively ONLY by the frozen compiler as an unsupported construct.
    //
    // That refusal is a downstream behaviour dependency, not this guard doing
    // its job. If the frozen profile ever admits module blocks, the flattened
    // image would carry two definitions of one bare name and the frozen
    // compiler resolves those last-definition-wins, silently. The guard has to
    // own the case itself, so the traversal is shared rather than assumed.
    fn walk(items: &[crate::ast::Node], counts: &mut BTreeMap<String, usize>) {
        for item in items {
            match item {
                crate::ast::Node::FnDef(fd, _) => {
                    *counts.entry(fd.name.clone()).or_insert(0) += 1;
                }
                crate::ast::Node::Block { stmts, .. } => walk(stmts, counts),
                _ => {}
            }
        }
    }
    let mut counts = BTreeMap::new();
    walk(&module.items, &mut counts);
    counts
}

/// Reserved function names scanned TEXTUALLY out of the std seed blob.
///
/// Deliberately not derived by parsing: the native bridge must keep working on a
/// build whose feature set the Rust parser would choke on, and a parse-derived
/// reserved set would make the backend go dark exactly there. A textual scan
/// over-approximates — it will pick up a `fn` inside a comment or string — and
/// over-approximation is the safe direction for a set whose only use is to
/// REFUSE.
pub fn reserved_names_from_text(blob: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in blob.lines() {
        let t = line.trim_start();
        let t = t.strip_prefix("pub ").unwrap_or(t);
        let Some(rest) = t.strip_prefix("fn ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.insert(name);
        }
    }
    out
}

/// Refuse a duplicate or reserved-colliding definition, judged on the AST.
pub fn reject_ambiguous_ast(
    module: &crate::ast::Module,
    reserved: &BTreeSet<String>,
) -> Result<(), ClosureRejection> {
    for (name, n) in ast_definition_counts(module) {
        if n > 1 {
            return reject(
                "closure.duplicate_definition",
                format!("`{name}` is defined {n} times in the linked closure"),
            );
        }
        if reserved.contains(&name) {
            return reject(
                "closure.std_symbol_collision",
                format!("`{name}` collides with a std seed-blob symbol"),
            );
        }
    }
    Ok(())
}

/// THE entry point. Admission for one merged program, in the only correct order.
///
/// The pieces are public for testing, but callers must use this: `reachable_from`
/// alone will happily walk the FIRST definition of a duplicated name while the
/// frozen compiler executes the LAST, so ambiguity has to be refused before
/// reachability is computed. Encoding the order here means a caller cannot get it
/// wrong.
///
/// Roots are the entry, every export (externally callable without a call site),
/// and any function called from module top level (those calls execute, and are
/// outside every `FnDef` body).
pub fn admit_native_closure(
    ast: &crate::ast::Module,
    ir: &IRModule,
    entry: &str,
    reserved: &BTreeSet<String>,
) -> Result<IRModule, ClosureRejection> {
    reject_ambiguous_ast(ast, reserved)?;
    let mut root_set: BTreeSet<String> = executable_roots(ir, entry).into_iter().collect();
    let mut top_level = Vec::new();
    direct_calls(&ir.instrs, &mut top_level);
    root_set.extend(top_level.into_iter().map(|(name, _)| name));
    let roots: Vec<String> = root_set.into_iter().collect();
    let reachable = reachable_from(ir, &roots)?;
    Ok(restrict_to_reachable(ir, &reachable))
}

/// Resolve the complete user image against checked compiler/extern bindings.
///
/// This is a binding check, NOT a feature or type admission. The caller MUST
/// first validate the complete source and IR under the versioned native lane
/// profile; `bindings` contains only that profile's exact ABI names/arities.
/// Unlike historical entry-rooted diagnostics, every retained user body is a
/// root here. Therefore omitting an unused std prefix cannot leave an uncalled
/// function silently bound to it. Intrinsics are terminal edges, never fake
/// user definitions. The historical API above remains strict and unchanged.
pub fn admit_native_lane_closure(
    ast: &crate::ast::Module,
    ir: &IRModule,
    entry: &str,
    reserved: &BTreeSet<String>,
    bindings: &BTreeMap<String, usize>,
) -> Result<(), ClosureRejection> {
    reject_ambiguous_ast(ast, reserved)?;
    let mut override_name = None;
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| match node {
            crate::ast::Node::FnDef(f, _)
                if f.name.starts_with("__mind_") || bindings.contains_key(&f.name) =>
            {
                override_name.get_or_insert_with(|| f.name.clone());
            }
            crate::ast::Node::ExternBlock { fns, .. } => {
                for f in fns {
                    if f.name.starts_with("__mind_") {
                        override_name.get_or_insert_with(|| f.name.clone());
                    }
                }
            }
            _ => {}
        });
    }
    if let Some(name) = override_name {
        return reject(
            "closure.reserved_intrinsic_definition",
            format!("`{name}` cannot override a compiler or external binding"),
        );
    }
    let mut roots = defined_fn_names(ir);
    roots.extend(executable_roots(ir, entry));
    let mut top_level = Vec::new();
    direct_calls(&ir.instrs, &mut top_level);
    for (name, arity) in top_level {
        if let Some(expected) = bindings.get(&name) {
            if arity != *expected {
                return reject(
                    "closure.intrinsic_arity",
                    format!("module calls `{name}` with {arity} arguments; expected {expected}"),
                );
            }
        } else {
            roots.insert(name);
        }
    }
    reachable_with_bindings(ir, &roots.into_iter().collect::<Vec<_>>(), bindings)?;
    Ok(())
}
