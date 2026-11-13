// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Calls from a lane-checked user image into the std seed (the seed tier).
//!
//! The native image is `[std seed blob][user region]` and the native compiler
//! compiles the whole blob, so a seed function needs admission, not a new
//! lowering. A user call binds to a PUBLIC function of a seed module the image
//! imports (`import std.<m>`: the MLIR build links only imported modules); a
//! user function of the same name makes it `lane.call.ambiguous`. Every seed
//! function that body calls is then reached too, whatever its visibility (the
//! image is one flat namespace). Only reached functions are typed and walked, under the
//! rules for user functions: source first, then their own lowering. Unreached
//! seed bodies are neither admitted nor walked, and an image that calls no
//! seed function never parses the seed. A refusal inside a reached function
//! names `std/<m>.mind` and the function.
//!
//! Seed `extern "C"` declarations (one rule; `examples/mindc_mind/nbx_bind.mind`
//! is its compiler twin). A reached seed body binds a seed declaration when it
//! calls that name and the selected compiler carries the `nbsext01` mark. Each
//! seed declaration of a called name must pass the `admit_extern` table. A
//! declaration no reached body calls is not read. A user body never binds one.

use super::admit::{Admitted, extern_gate, signature, source_facts, walk_function};
use super::admit_extern::admit_extern_fn;
use super::admit_ir::check_ir;
use super::types::{
    Callees, Check, Functions, LaneProfileRejection, PRINT_HELPERS, intrinsic_arity,
};
use crate::ast::{self, CallConv, ExternFn, FnDefData, Node, Span};
use crate::ir::IRModule;
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};

/// The std seed modules (manifest `seed` rows, in blob order) and their bytes,
/// parsed at most once and only when the image calls into them.
pub struct LaneSeed<'a> {
    modules: &'a [&'a str],
    sources: &'a [Vec<u8>],
    parsed: OnceCell<Option<Vec<ast::Module>>>,
    /// The selected compiler carries the `nbsext01` mark.
    binds_externs: bool,
    /// The selected compiler carries the `nbprint1` mark (`with_print`).
    print: bool,
}

impl<'a> LaneSeed<'a> {
    pub fn new(modules: &'a [&'a str], sources: &'a [Vec<u8>]) -> Self {
        LaneSeed {
            modules,
            sources,
            parsed: OnceCell::new(),
            binds_externs: false,
            print: false,
        }
    }

    /// `marked` is the selected compiler's `nbsext01` mark: it binds the seed
    /// `extern "C"` declarations that reached seed bodies call. Without it such
    /// a call stays `lane.call.unknown` in its seed body.
    pub fn binding_seed_externs(mut self, marked: bool) -> Self {
        self.binds_externs = marked;
        self
    }

    /// Admit the builtin `print` statement: only for a selected compiler that
    /// carries the `nbprint1` mark (src/build/native_bridge/seed.rs).
    pub fn with_print(mut self, print: bool) -> Self {
        self.print = print;
        self
    }

    pub fn modules(&self) -> &'a [&'a str] {
        self.modules
    }

    pub fn sources(&self) -> &'a [Vec<u8>] {
        self.sources
    }

    /// The text of a refusal's `seed_module` (`std/<m>.mind`).
    pub fn text(&self, seed_module: &str) -> Option<&'a str> {
        let name = seed_module.strip_prefix("std/")?.strip_suffix(".mind")?;
        let i = self.modules.iter().position(|m| *m == name)?;
        std::str::from_utf8(self.sources.get(i)?).ok()
    }

    fn parsed(&self) -> Check<&[ast::Module]> {
        let parse = || {
            (self.modules.len() == self.sources.len()).then_some(())?;
            self.sources
                .iter()
                .map(|bytes| crate::parser::parse(std::str::from_utf8(bytes).ok()?).ok())
                .collect()
        };
        self.parsed
            .get_or_init(parse)
            .as_deref()
            .ok_or(LaneProfileRejection::new("lane.seed.unparsed"))
    }
}

/// One seed definition: its module, whether that module makes it public, the
/// definition and its span in the module's own text.
#[derive(Clone, Copy)]
struct SeedFn<'s> {
    module: usize,
    public: bool,
    def: &'s FnDefData,
    span: Span,
}

/// Every seed definition by name.
fn index(modules: &[ast::Module]) -> BTreeMap<&str, Vec<SeedFn<'_>>> {
    let mut out: BTreeMap<&str, Vec<SeedFn<'_>>> = BTreeMap::new();
    for (module, parsed) in modules.iter().enumerate() {
        let mut exported = BTreeSet::new();
        for item in &parsed.items {
            if let Node::Export { names, .. } = item {
                exported.extend(names.iter().map(String::as_str));
            }
        }
        for item in &parsed.items {
            if let Node::FnDef(def, span) = item {
                let public = def.is_pub || exported.contains(def.name.as_str());
                let (def, span) = (&**def, *span);
                out.entry(def.name.as_str()).or_default().push(SeedFn {
                    module,
                    public,
                    def,
                    span,
                });
            }
        }
    }
    out
}

/// One seed `extern "C"` declaration: its module, callconv and block offset.
#[derive(Clone, Copy)]
struct SeedExtern<'s> {
    module: usize,
    callconv: CallConv,
    decl: &'s ExternFn,
    at: usize,
}

/// Every seed `extern "C"` declaration by name, in blob order.
fn extern_index(modules: &[ast::Module]) -> BTreeMap<&str, Vec<SeedExtern<'_>>> {
    let mut out: BTreeMap<&str, Vec<SeedExtern<'_>>> = BTreeMap::new();
    for (module, parsed) in modules.iter().enumerate() {
        for item in &parsed.items {
            // Route A binds base declarations only: a `bind` block is no seed declaration.
            if let Node::ExternBlock {
                callconv,
                binding: ast::ExternBinding::Base,
                fns,
                span,
            } = item
            {
                for decl in fns {
                    out.entry(decl.name.as_str()).or_default().push(SeedExtern {
                        module,
                        callconv: *callconv,
                        decl,
                        at: span.start(),
                    });
                }
            }
        }
    }
    out
}

fn calls(nodes: &[Node], out: &mut BTreeSet<String>) {
    for node in nodes {
        crate::type_checker::nerve_walk::walk(node, &mut |n| {
            if let Node::Call { callee, .. } = n {
                out.insert(callee.clone());
            }
        });
    }
}

/// The reached seed functions and their signatures (the seed tier), and the
/// part of it a user body may call.
struct Reached<'s> {
    functions: BTreeMap<String, SeedFn<'s>>,
    tier: Functions,
    visible: Functions,
    /// The seed `extern "C"` declarations the reached bodies call.
    externs: Functions,
}

fn reach<'s>(source: &ast::Module, seed: &'s LaneSeed) -> Check<Reached<'s>> {
    let mut reached = Reached {
        functions: BTreeMap::new(),
        tier: Functions::new(),
        visible: Functions::new(),
        externs: Functions::new(),
    };
    let mut user = BTreeMap::new();
    for item in &source.items {
        if let Node::FnDef(def, _) = item {
            user.insert(def.name.as_str(), &**def);
        }
    }
    let mut imported = BTreeSet::new();
    for item in &source.items {
        crate::type_checker::nerve_walk::walk(item, &mut |n| {
            if let Node::Import { path, .. } = n
                && let [std, module, ..] = path.as_slice()
                && std == "std"
            {
                imported.insert(module.as_str());
            }
        });
    }
    let mut wanted = BTreeSet::new();
    calls(&source.items, &mut wanted);
    wanted.retain(|n| intrinsic_arity(n).is_none());
    if wanted.is_empty() || imported.is_empty() {
        return Ok(reached);
    }
    let index = index(seed.parsed()?);
    // A user call binds only to a public function of an imported seed module;
    // any other call stays unresolved and refuses `lane.call.unknown` at its
    // own call site.
    let bindable = |f: &SeedFn| f.public && imported.contains(seed.modules[f.module]);
    let bound = |n: &String| {
        index
            .get(n.as_str())
            .is_some_and(|d| d.iter().any(bindable))
    };
    let mut queue = Vec::new();
    let mut called = BTreeSet::new();
    for name in wanted.into_iter().filter(bound) {
        match user.get(name.as_str()) {
            // The user image and a bindable seed function both define the
            // name. The seed body is neither typed nor reached: its entry
            // carries the user signature only so the chain sees two tiers and
            // refuses `lane.call.ambiguous` at the call site, whatever the seed
            // signature. A user signature outside the lane refuses first.
            Some(def) => {
                if let Ok(sig) = signature(def) {
                    reached.visible.insert(name, sig);
                }
            }
            None => queue.push(name),
        }
    }
    while let Some(name) = queue.pop() {
        if reached.functions.contains_key(&name) {
            continue;
        }
        let defs = &index[name.as_str()];
        let found = defs[0];
        let refuse = |r: LaneProfileRejection, f: SeedFn| {
            r.located(f.span.start())
                .in_seed(seed.modules[f.module], &name)
        };
        if let Some(&second) = defs.get(1) {
            let duplicate = LaneProfileRejection::new("lane.call.ambiguous");
            return Err(refuse(duplicate, second));
        }
        let sig = signature(found.def).map_err(|r| refuse(r, found))?;
        // A seed body's call reaches the seed definition even when the user
        // image defines the same name; the body walk then refuses it as
        // `lane.call.ambiguous` rather than letting the user copy rebind it.
        let mut next = BTreeSet::new();
        calls(&found.def.body, &mut next);
        called.extend(next.iter().cloned());
        queue.extend(next.into_iter().filter(|n| index.contains_key(n.as_str())));
        if bindable(&found) {
            reached.visible.insert(name.clone(), sig.clone());
        }
        reached.tier.insert(name.clone(), sig);
        reached.functions.insert(name, found);
    }
    if !seed.binds_externs {
        return Ok(reached);
    }
    // Every seed declaration of a called name must match its table row, so
    // they all carry one signature: the declarations are one symbol.
    let externs = extern_index(seed.parsed()?);
    for name in called {
        for d in externs.get(name.as_str()).into_iter().flatten() {
            let (_, sig) = admit_extern_fn(d.decl, d.callconv)
                .map_err(|r| r.located(d.at).in_seed(seed.modules[d.module], &name))?;
            reached.externs.insert(name.clone(), sig);
        }
    }
    Ok(reached)
}

/// The part of the user image a seed body's calls collide with.
struct Shadow {
    user: Functions,
    externs: Functions,
}

impl Shadow {
    /// The chain a reached seed body resolves its calls against.
    fn callees<'a>(&'a self, tier: &'a Functions) -> Callees<'a> {
        Callees {
            user: &self.user,
            externs: &self.externs,
            seed: tier,
            print: false,
        }
    }
}

impl Reached<'_> {
    /// The user functions and admitted externs a seed body's calls collide
    /// with, and the seed declarations it binds. A seed body sees only these
    /// of the user image, so a user function or extern named like a reached
    /// seed function, or a function named like a bound seed declaration,
    /// refuses as `lane.call.ambiguous`; any other user name stays unknown to
    /// it. A user declaration of a bound seed declaration's name passed the
    /// same table row: it is the same symbol.
    fn shadow(&self, facts: &Admitted) -> Shadow {
        let named = |table: &Functions| -> Functions {
            table
                .iter()
                .filter(|(name, _)| {
                    self.tier.contains_key(*name) || self.externs.contains_key(*name)
                })
                .map(|(name, sig)| (name.clone(), sig.clone()))
                .collect()
        };
        let mut externs = named(&facts.externs);
        externs.extend(self.externs.clone());
        Shadow {
            user: named(&facts.user),
            externs,
        }
    }

    /// The source leg for every reached seed body: its parameters only.
    // shortcut: seed module constants (json/toml `MAX_DEPTH`) are not bound, so
    // a reached body reading one refuses `lane.source.unbound`; bind them when
    // a seed helper the lane must admit needs one.
    ///
    /// `from_loops` names seed functions the user image reached from a loop.
    /// Those bodies, and the seed helpers they reach, are walked as loop
    /// bodies so a repeated free refuses. Every other seed body is typed with
    /// the size proof off: the blob already ships those dynamic sizes.
    fn walk(&self, callees: Callees, seed: &LaneSeed, from_loops: &BTreeSet<String>) -> Check<()> {
        let refuse = |r: LaneProfileRejection, f: &SeedFn<'_>, name: &str| {
            r.located(f.span.start())
                .in_seed(seed.modules[f.module], name)
        };
        let mut scratch = BTreeSet::new();
        for (name, f) in &self.functions {
            if from_loops.contains(name) {
                continue;
            }
            walk_function(
                f.def,
                &self.tier[name],
                &BTreeMap::new(),
                callees,
                &mut scratch,
                0,
                false,
            )
            .map_err(|r| refuse(r, f, name))?;
        }
        let mut pending: BTreeSet<String> = from_loops
            .iter()
            .filter(|name| self.functions.contains_key(*name))
            .cloned()
            .collect();
        let mut checked = BTreeSet::new();
        while let Some(name) = pending.iter().find(|n| !checked.contains(*n)).cloned() {
            checked.insert(name.clone());
            let f = &self.functions[&name];
            let mut found = BTreeSet::new();
            walk_function(
                f.def,
                &self.tier[&name],
                &BTreeMap::new(),
                callees,
                &mut found,
                1,
                false,
            )
            .map_err(|r| refuse(r, f, &name))?;
            for next in found {
                if self.functions.contains_key(&next) {
                    pending.insert(next);
                }
            }
        }
        Ok(())
    }

    /// The IR leg: each module's reached functions lowered together.
    fn lowered(&self, callees: Callees, seed: &LaneSeed) -> Check<()> {
        // No reached seed function: nothing to lower, and the seed stays unparsed.
        if self.tier.is_empty() {
            return Ok(());
        }
        let parsed = seed.parsed()?;
        for (index, module) in seed.modules.iter().enumerate() {
            let own: Functions = self
                .tier
                .iter()
                .filter(|(name, _)| self.functions[*name].module == index)
                .map(|(name, sig)| (name.clone(), sig.clone()))
                .collect();
            let Some(first) = own.keys().next() else {
                continue;
            };
            let in_seed = |r: LaneProfileRejection| {
                let name = r.function.clone().unwrap_or_else(|| first.clone());
                r.in_seed(module, &name)
            };
            let items = own
                .keys()
                .map(|name| {
                    let f = self.functions[name];
                    Node::FnDef(Box::new(f.def.clone()), f.span)
                })
                .collect();
            let lowering = ast::Module {
                items,
                spelling: parsed[index].spelling.clone(),
            };
            let ir = crate::eval::lower_to_ir(&lowering)
                .map_err(|_| in_seed(LaneProfileRejection::new("lane.seed.lowering")))?;
            check_ir(&ir, &own, callees).map_err(in_seed)?;
        }
        Ok(())
    }
}

/// The terminal edges of one admitted image, as name -> arity: its admitted
/// `extern "C"` tier, every reached seed function and every seed declaration
/// a reached seed body binds (`seed_externs`: the image is then dynamic). The
/// closure check binds all three; seed omission sees `externs` only.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LaneEdges {
    pub externs: BTreeMap<String, usize>,
    pub seed: BTreeMap<String, usize>,
    pub seed_externs: BTreeMap<String, usize>,
    /// The runtime helpers an admitted `print` lowers to that no tier defines
    /// (`print_helper_type`); the closure binds them like intrinsics.
    pub print: BTreeMap<String, usize>,
}

/// The user facts and the reached seed tier for one image.
fn admitted<'s>(
    source: &ast::Module,
    seed: &'s LaneSeed,
    extern_fixed: bool,
) -> Check<(Reached<'s>, Admitted)> {
    extern_gate(source, extern_fixed)?;
    let reached = reach(source, seed)?;
    let facts = source_facts(source, &reached.visible, seed.print)?;
    Ok((reached, facts))
}

/// Source-first admission with the std seed tier (see the module docs).
/// `extern_fixed` is the selected compiler's `nbextc01` mark.
pub fn profile_lane_admits_source_seeded(
    source: &ast::Module,
    seed: &LaneSeed,
    extern_fixed: bool,
) -> Check<()> {
    let (reached, facts) = admitted(source, seed, extern_fixed)?;
    let shadow = reached.shadow(&facts);
    reached.walk(shadow.callees(&reached.tier), seed, &facts.loop_calls)
}

/// Source and IR admission with the std seed tier. Returns the image's
/// terminal edges: the closure check binds them like intrinsics.
pub fn profile_lane_admits_seeded(
    source: &ast::Module,
    ir: &IRModule,
    seed: &LaneSeed,
    extern_fixed: bool,
) -> Check<LaneEdges> {
    let (reached, facts) = admitted(source, seed, extern_fixed)?;
    let shadow = reached.shadow(&facts);
    let seed_callees = shadow.callees(&reached.tier);
    reached.walk(seed_callees, seed, &facts.loop_calls)?;
    let callees = facts.callees(&reached.visible);
    check_ir(ir, &facts.user, callees)?;
    reached.lowered(seed_callees, seed)?;
    let print = PRINT_HELPERS
        .iter()
        .filter(|(n, _)| callees.print && callees.lookup(n).is_empty())
        .map(|&(n, arity)| (n.to_string(), arity))
        .collect();
    let arity = |table: &Functions| -> BTreeMap<String, usize> {
        table
            .iter()
            .map(|(n, sig)| (n.clone(), sig.arity()))
            .collect()
    };
    Ok(LaneEdges {
        externs: arity(&facts.externs),
        seed: arity(&reached.tier),
        seed_externs: arity(&reached.externs),
        print,
    })
}

#[cfg(test)]
#[path = "seed_tests.rs"]
mod tests;
