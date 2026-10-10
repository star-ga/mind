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
//!
//! Traversal goes through [`crate::ir::instr_bodies`], the single enumeration of
//! body-carrying variants, so a newly added nested-body variant cannot silently
//! become invisible to reachability while remaining visible to the fence.

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
fn direct_calls(instrs: &[Instr], out: &mut Vec<String>) {
    for instr in instrs {
        if let Instr::Call { name, .. } = instr {
            out.push(name.clone());
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
/// latter covers builtins, externs and every intrinsic the frozen emitter does not
/// know, which the fence rejects anyway — refusing here keeps the two in agreement
/// and names the callee, which the fence cannot. A call to one of the frozen
/// emitter's own intrinsics (`frozen_intrinsics::FROZEN_NATIVE_INTRINSICS`) is a
/// resolved LEAF: the stage1 emitter expands it inline, and the fence checks its
/// arity and operands.
pub fn reachable_from(
    module: &IRModule,
    roots: &[String],
) -> Result<BTreeSet<String>, ClosureRejection> {
    reachable_from_linked(module, roots, &BTreeSet::new())
}

/// [`reachable_from`] where a call may also resolve to a `linked` name defined
/// outside `module` (a std seed function). Linked callees are leaves here: their
/// bodies live in the std seed and are walked by [`admit_native_closure_linked`].
fn reachable_from_linked(
    module: &IRModule,
    roots: &[String],
    linked: &BTreeSet<String>,
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
        for callee in callees {
            if !defined.contains(&callee) {
                if is_frozen_intrinsic(&callee) || linked.contains(&callee) {
                    continue;
                }
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

/// Struct names scanned TEXTUALLY out of the std seed blob, by the same rule as
/// [`reserved_names_from_text`] (`struct NAME` or `pub struct NAME` at the start of a
/// line). Over-approximation is again the safe direction: the set only refuses.
pub fn reserved_struct_names_from_text(blob: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in blob.lines() {
        let t = line.trim_start();
        let t = t.strip_prefix("pub ").unwrap_or(t);
        let Some(rest) = t.strip_prefix("struct ") else {
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

/// Refuse a program that declares a struct with the name of a std seed struct.
///
/// The frozen compiler compiles `std blob ++ user region` as one program, so the
/// image then declares the struct twice. It does not refuse that: it dies with
/// SIGSEGV (measured: `struct Value { h: i64 }` beside `std/json.mind`'s `Value`,
/// for every seed struct name, used or not, inside a module block or not, and with
/// no call into std). Judged on every `StructDef` anywhere in the AST.
fn reject_std_struct_collision(
    ast: &crate::ast::Module,
    seed_structs: &BTreeSet<String>,
) -> Result<(), ClosureRejection> {
    let mut found: Option<String> = None;
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            if let crate::ast::Node::StructDef { name, .. } = node {
                if found.is_none() && seed_structs.contains(name) {
                    found = Some(name.clone());
                }
            }
        });
    }
    match found {
        Some(name) => reject(
            "closure.std_struct_collision",
            format!(
                "struct `{name}` is also declared by the std seed blob the native backend \
                 compiles with every program; rename the program's `{name}` or build with \
                 `--backend mlir`"
            ),
        ),
        None => Ok(()),
    }
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
    root_set.extend(top_level);
    let roots: Vec<String> = root_set.into_iter().collect();
    let reachable = reachable_from(ir, &roots)?;
    Ok(restrict_to_reachable(ir, &reachable))
}

fn is_frozen_intrinsic(name: &str) -> bool {
    crate::ir::frozen_intrinsics::frozen_native_intrinsic(name).is_some()
}

/// What the std seed blob contributes to closure admission.
///
/// The frozen compiler compiles `std blob ++ user region` as ONE flat program: it
/// prunes to the functions reachable from EVERY user function (all user functions
/// are its roots), and it resolves every call — including a call written inside a
/// std function — to the LAST definition of that name. A user definition therefore
/// shadows a std function of the same name everywhere, which is exactly what makes
/// shadowing safe when no retained std code calls the shadowed name, and exactly
/// what makes it a silent wrong binding when some does.
pub struct StdSeed {
    /// Seed modules in manifest order, each with the function names its text
    /// defines (the textual scan of [`reserved_names_from_text`]).
    pub modules: Vec<(String, BTreeSet<String>)>,
    /// Every struct name any seed module's text declares
    /// ([`reserved_struct_names_from_text`]). A program may not declare one.
    pub structs: BTreeSet<String>,
    /// The seed blob parsed and lowered ONCE, as one program. `None` when this
    /// build could not: then no call resolves into std and every seed name stays
    /// reserved — the behaviour before std linkage existed, which only refuses.
    pub lowered: Option<(crate::ast::Module, IRModule)>,
}

impl StdSeed {
    /// Every function name any seed module's text defines.
    pub fn all_names(&self) -> BTreeSet<String> {
        self.modules
            .iter()
            .flat_map(|(_, names)| names.iter().cloned())
            .collect()
    }

    /// The parsed definition of seed function `name`, as a one-item module, for the
    /// source-level half of the fence. `None` when the seed is not lowered or the
    /// name is not a top-level seed function.
    pub fn fn_source(&self, name: &str) -> Option<crate::ast::Module> {
        let (ast, _) = self.lowered.as_ref()?;
        let item = ast
            .items
            .iter()
            .find(|item| matches!(item, crate::ast::Node::FnDef(f, _) if f.name == name))?;
        Some(crate::ast::Module {
            items: vec![item.clone()],
            spelling: Default::default(),
        })
    }

    /// The lowered body of seed function `name` alone, for the IR half of the fence.
    pub fn fn_ir(&self, name: &str) -> Option<IRModule> {
        let (_, ir) = self.lowered.as_ref()?;
        let keep: BTreeSet<String> = std::iter::once(name.to_string()).collect();
        find_body(&ir.instrs, name)?;
        let mut only = restrict_to_reachable(ir, &keep);
        only.instrs.retain(|i| matches!(i, Instr::FnDef { .. }));
        Some(only)
    }
}

/// A closure admitted with std linkage.
pub struct LinkedClosure {
    /// The program restricted to what its roots reach (diagnostic view, as for
    /// [`admit_native_closure`]).
    pub restricted: IRModule,
    /// The std seed functions the frozen compiler will retain for this program —
    /// every seed function reachable from ANY user function. Each one's body must
    /// pass the frozen-profile fence; calls to them are admitted only on that basis.
    pub std_linked: BTreeSet<String>,
}

/// The `std.<module>` names `ast` imports, at any depth.
fn imported_std_modules(ast: &crate::ast::Module) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            if let crate::ast::Node::Import { path, .. } = node {
                if path.first().map(String::as_str) == Some("std") {
                    if let Some(module) = path.get(1) {
                        out.insert(module.clone());
                    }
                }
            }
        });
    }
    out
}

/// Every call name anywhere in `instrs`, including inside function bodies.
fn all_calls(instrs: &[Instr], out: &mut Vec<String>) {
    for instr in instrs {
        if let Instr::Call { name, .. } = instr {
            out.push(name.clone());
        }
        for body in instr_bodies(instr) {
            all_calls(body, out);
        }
    }
}

/// Admission for one merged program WITH std linkage, in the only correct order:
///
/// 1. a duplicate user definition is refused (as in [`admit_native_closure`]);
/// 2. a user definition of a frozen intrinsic name is refused — the frozen
///    emitter expands the intrinsic at every call site, lowering drops the user
///    body, so the definition could never be the one that runs;
/// 3. a struct declared with the name of a std seed struct is refused
///    (`closure.std_struct_collision`): the frozen compiler crashes on it;
/// 4. a user definition of a name that an IMPORTED std module defines is refused:
///    the program asked for `std.<m>` and redefined one of its members, so which
///    one a call means is ambiguous (`closure.std_symbol_collision`);
/// 5. entry-rooted reachability, where a call may resolve to a user function, a
///    frozen intrinsic, or a std seed function;
/// 6. std linkage: the seed functions reachable from every user function. One
///    that calls a name the program redefines is refused
///    (`closure.std_callee_shadowed`) — the frozen compiler would bind that call
///    to the program's definition. One that calls a name with no definition is
///    refused (`closure.unresolved_edge`).
///
/// A user definition that shadows a seed name the program does not import, and
/// that no retained seed function calls, is admitted: the frozen compiler binds
/// every call to it and retains no std code that could mean the other one.
///
/// Without a lowered seed (`StdSeed::lowered == None`) this is exactly
/// [`admit_native_closure`] with every seed name reserved.
pub fn admit_native_closure_linked(
    ast: &crate::ast::Module,
    ir: &IRModule,
    entry: &str,
    seed: &StdSeed,
) -> Result<LinkedClosure, ClosureRejection> {
    let counts = ast_definition_counts(ast);
    reject_ambiguous_ast(ast, &BTreeSet::new())?;
    // Before the seed branch: the frozen emitter expands an intrinsic whether or
    // not std linkage is available, so the refusal must not depend on it.
    if let Some(name) = counts.keys().find(|name| is_frozen_intrinsic(name)) {
        return reject(
            "closure.intrinsic_redefinition",
            format!(
                "`{name}` is a compiler intrinsic the native backend expands inline at \
                 every call site; a program definition of it would never run"
            ),
        );
    }
    // Also before the seed branch: the struct is in the image whether or not the
    // seed lowered.
    reject_std_struct_collision(ast, &seed.structs)?;
    let Some((_, std_ir)) = &seed.lowered else {
        let restricted = admit_native_closure(ast, ir, entry, &seed.all_names())?;
        return Ok(LinkedClosure {
            restricted,
            std_linked: BTreeSet::new(),
        });
    };
    let user_names: BTreeSet<String> = counts.into_keys().collect();
    let imported = imported_std_modules(ast);
    for (module, names) in &seed.modules {
        if !imported.contains(module) {
            continue;
        }
        if let Some(name) = names.intersection(&user_names).next() {
            return reject(
                "closure.std_symbol_collision",
                format!(
                    "`{name}` is defined by this program and by `std.{module}`, which it \
                     imports; rename the program's `{name}` or drop the import"
                ),
            );
        }
    }

    let user_ir_defined = defined_fn_names(ir);
    let std_defined = defined_fn_names(std_ir);
    // A user call binds to std only when no user definition (in IR or source)
    // claims the name and it is not an intrinsic: last-definition-wins.
    let std_callable: BTreeSet<String> = std_defined
        .iter()
        .filter(|name| !user_names.contains(*name) && !is_frozen_intrinsic(name))
        .cloned()
        .collect();

    let mut root_set: BTreeSet<String> = executable_roots(ir, entry).into_iter().collect();
    let mut top_level = Vec::new();
    direct_calls(&ir.instrs, &mut top_level);
    root_set.extend(top_level);
    let roots: Vec<String> = root_set.into_iter().collect();
    let reachable = reachable_from_linked(ir, &roots, &std_callable)?;

    // Std linkage is seeded from EVERY user function, not only the reachable ones:
    // the frozen compiler roots its prune at every user function, so a std body an
    // uncalled user function names is still emitted.
    let mut calls = Vec::new();
    all_calls(&ir.instrs, &mut calls);
    let mut std_linked: BTreeSet<String> = BTreeSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    for callee in calls {
        if !user_ir_defined.contains(&callee)
            && std_callable.contains(&callee)
            && std_linked.insert(callee.clone())
        {
            queue.push_back(callee);
        }
    }
    while let Some(current) = queue.pop_front() {
        let Some(body) = find_body(&std_ir.instrs, &current) else {
            return reject(
                "closure.missing_body",
                format!("std seed function `{current}` is named but has no body"),
            );
        };
        let mut callees = Vec::new();
        direct_calls(body, &mut callees);
        for callee in callees {
            if user_names.contains(&callee) {
                return reject(
                    "closure.std_callee_shadowed",
                    format!(
                        "std seed function `{current}`, which this program reaches, calls \
                         `{callee}`, and this program also defines `{callee}`. The native \
                         backend binds every call to the LAST definition of a name, so \
                         `{current}` would call the program's `{callee}` instead of std's. \
                         Rename the program's `{callee}`"
                    ),
                );
            }
            if is_frozen_intrinsic(&callee) {
                continue;
            }
            if !std_defined.contains(&callee) {
                return reject(
                    "closure.unresolved_edge",
                    format!(
                        "std seed function `{current}` calls `{callee}`, which has no \
                         definition in the closure"
                    ),
                );
            }
            if std_linked.insert(callee.clone()) {
                queue.push_back(callee);
            }
        }
    }
    Ok(LinkedClosure {
        restricted: restrict_to_reachable(ir, &reachable),
        std_linked,
    })
}
