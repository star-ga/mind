// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Calls into the world-reading std modules.
//!
//! `dns_resolve`, `tcp_connect_host`, `tcp_connect_host_timeout`,
//! `entropy_fill` and the `https_*` entry points read the world only when the
//! call resolves to `std.dns`, `std.entropy`, `std.https`, `std.https_probe`
//! or `std.https_io`. A user function of the same bare name is deterministic.
//! The lowered IR keeps only the bare name (the qualifier lives on the call
//! span, in `SourceSpelling`), so this module reads the source.
//!
//! A call resolves in one of two ways, the way the build resolves it. A
//! qualified call (`dns.dns_resolve(..)`) resolves through its qualifier. A
//! bare call (`dns_resolve(..)`, no qualifier recorded) resolves by name: it is
//! the std function when the module imports the std module that owns that row
//! and defines no function of that name. A bare name that the module defines
//! is the user's function and stays deterministic.
//!
//! Every spelling of the qualifier counts: the module name, an alias, the
//! dotted path, and the `::` forms of each. The scan uses the exhaustive AST
//! walker of the numerics lint (`type_checker::nerve_walk`), so a call in any
//! expression position (an array, map or set literal, a field receiver, a
//! top-level `const`, a closure) is found, and a function-value read of a row
//! counts as a use.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::ast::{Literal, Module, Node, Span};
use crate::ir::{IRModule, Instr};
use crate::type_checker::nerve_walk;

/// `(module path, public function)` rows. A call matches only both.
///
/// deferred: would also list the World-reaching functions of std.net, std.fs,
/// std.process and std.iouring, not done because the veto would then refuse
/// their existing callers without the flag (a policy change, not a fix) —
/// upgrade path: add each module to `NAMED_STD_MODULES` in the intrinsics
/// tests, which derives the rows from the module source.
const ROWS: &[(&str, &str)] = &[
    ("std.dns", "dns_resolve"),
    ("std.dns", "tcp_connect_host"),
    ("std.dns", "tcp_connect_host_timeout"),
    ("std.entropy", "entropy_fill"),
    // std.https reaches DNS, the network, entropy, the clock and the CA file.
    ("std.https", "https_get"),
    ("std.https", "https_get_ca"),
    ("std.https", "https_post"),
    ("std.https", "https_post_ca"),
    ("std.https", "https_get_with_headers"),
    ("std.https", "https_post_with_headers"),
    ("std.https_probe", "https_probe"),
    // std.https_io reads the clocks (hio_arm and hio_io_failure through the
    // deadline). Its CA bundle file read is a declared input, not a row.
    ("std.https_io", "hio_now_unix"),
    ("std.https_io", "hio_now_mono_ms"),
    ("std.https_io", "hio_remaining_ms"),
    ("std.https_io", "hio_arm"),
    ("std.https_io", "hio_io_failure"),
];

/// Whether `name` is a world-reading public function of `module`.
pub fn is_row(module: &str, name: &str) -> bool {
    ROWS.iter().any(|(m, n)| *m == module && *n == name)
}

/// Qualifier spellings of one parsed module, and the qualifier recorded for
/// each desugared call or value-read span.
struct QualMap {
    /// Qualifier (`dns`, an alias, or the dotted path) to the module paths it
    /// can name. More than one path is fail-closed: a call matches when any of
    /// them is a world row.
    by_qual: BTreeMap<String, Vec<String>>,
    /// Keyed by span, with `::` written as `.`. Only looked up, never iterated.
    refs: HashMap<Span, String>,
    /// Dotted path of every module the source imports, under any alias.
    imported: BTreeSet<String>,
    /// Names of the functions the module itself defines, at the top level or
    /// in a `module NAME { .. }` block. Impl methods, nested functions and
    /// `extern` declarations are not here: a bare call does not resolve to
    /// them, so they cannot hide a std call.
    defined: BTreeSet<String>,
}

impl QualMap {
    fn from_module(module: &Module) -> Self {
        let mut by_qual: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut imported = BTreeSet::new();
        for item in &module.items {
            nerve_walk::walk(item, &mut |node| {
                let Node::Import { path, alias, .. } = node else {
                    return;
                };
                let dotted = path.join(".");
                imported.insert(dotted.clone());
                let last = path.last().map(String::as_str).unwrap_or("");
                if let Some(alias) = alias {
                    add_qual(&mut by_qual, alias, &dotted);
                }
                add_qual(&mut by_qual, last, &dotted);
                add_qual(&mut by_qual, &dotted, &dotted);
            });
        }
        let refs = module
            .spelling
            .call_qualifiers
            .iter()
            .map(|(span, qual)| (*span, qual.replace("::", ".")))
            .collect();
        let mut defined = BTreeSet::new();
        collect_defined(&module.items, &mut defined);
        Self {
            by_qual,
            refs,
            imported,
            defined,
        }
    }

    fn is_world(&self, span: Span, name: &str) -> bool {
        let Some(qual) = self.refs.get(&span) else {
            // A bare name resolves to the std row it names when the module
            // imports that std module and defines no function of that name.
            return !self.defined.contains(name)
                && self.imported.iter().any(|path| is_row(path, name));
        };
        match self.by_qual.get(qual.as_str()) {
            Some(paths) => paths.iter().any(|path| is_row(path, name)),
            None => is_row(qual, name),
        }
    }

    /// The first std-world call or function-value read in `node`'s
    /// subtree, `node` included.
    fn offender_in(&self, node: &Node) -> Option<String> {
        let mut found = None;
        nerve_walk::walk(node, &mut |n| {
            let (name, span) = match n {
                Node::Call { callee, span, .. } => (callee, *span),
                Node::Lit(Literal::Ident(name), span) => (name, *span),
                _ => return,
            };
            if found.is_none() && self.is_world(span, name) {
                found = Some(name.clone());
            }
        });
        found
    }
}

/// Names of the free functions in `items`, descending into transparent
/// `module NAME { .. }` blocks but not into function bodies or impls.
fn collect_defined(items: &[Node], out: &mut BTreeSet<String>) {
    for item in items {
        match item {
            Node::FnDef(fd, _) => {
                out.insert(fd.name.clone());
            }
            Node::Block { stmts, .. } => collect_defined(stmts, out),
            _ => {}
        }
    }
}

fn add_qual(map: &mut BTreeMap<String, Vec<String>>, qual: &str, path: &str) {
    if qual.is_empty() {
        return;
    }
    let entry = map.entry(qual.to_string()).or_default();
    if !entry.iter().any(|have| have == path) {
        entry.push(path.to_string());
    }
}

/// What [`module_offender`] names for a source it cannot parse.
pub const UNPARSED: &str = "<source that does not parse>";

/// The first std-world use anywhere in `source`: every function at
/// any depth, impl methods, module blocks and top-level items. Whole-module,
/// matching the entry IR scan: an uncalled function still refuses the module.
/// A source that does not parse is [`UNPARSED`], never clean (the CLI calls
/// this only after the same parser accepted the entry).
pub fn module_offender(source: &str) -> Option<String> {
    let Ok(module) = crate::parser::parse(source) else {
        return Some(UNPARSED.to_string());
    };
    let map = QualMap::from_module(&module);
    module.items.iter().find_map(|item| map.offender_in(item))
}

/// The std-world use in the body of `func` in `source`, nested
/// closures and functions included. Every function of that name counts (a
/// free function and a method alike), so a same-named sibling cannot hide one.
/// Other functions of the module do not count. `None` when `func` is absent,
/// the source does not parse (the caller's lowering of that source then fails,
/// which refuses it), or no such body makes such a use.
pub fn reached_offender(source: &str, func: &str) -> Option<String> {
    let module = crate::parser::parse(source).ok()?;
    let map = QualMap::from_module(&module);
    let mut found = None;
    for item in &module.items {
        nerve_walk::walk(item, &mut |node| {
            if found.is_none()
                && let Node::FnDef(fd, _) = node
                && fd.name == func
            {
                found = map.offender_in(node);
            }
        });
    }
    found
}

thread_local! {
    static INSTALLED: RefCell<Option<QualMap>> = const { RefCell::new(None) };
}

/// Installs `module`'s qualifier map for [`call_is_world`] on this thread.
/// Dropping the guard clears it.
pub struct Guard;

impl Guard {
    pub fn install(module: &Module) -> Self {
        let map = QualMap::from_module(module);
        INSTALLED.with(|slot| *slot.borrow_mut() = Some(map));
        Guard
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        INSTALLED.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Whether the call at `span` named `name` is a std-world call (qualified, or
/// bare and resolving by name) in
/// the map [`Guard`] installed. False when no guard is installed, so a path
/// that never parsed a module cannot invent a refusal.
pub fn call_is_world(span: Span, name: &str) -> bool {
    INSTALLED.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|map| map.is_world(span, name))
    })
}

/// A copy of `module` with a zero-argument call the existing classifier
/// already treats as nondeterministic, first in the top-level stream.
///
/// Used only for the evidence artifact of a program whose nondeterminism sits
/// in a callee the entry body does not hold. The verifier re-derives from the
/// hashed body, so a label rewritten to `deterministic` then disagrees with
/// the body. Object code is never stamped. The top level is its own SSA
/// namespace (each function body numbers its values from zero), so the
/// witness takes an id past every top-level one and cannot collide.
pub fn stamp_unheld_world(module: &IRModule) -> IRModule {
    let mut cloned = module.clone();
    let bound = cloned
        .instrs
        .iter()
        .map(crate::opt::ir_canonical::instruction_id_bound)
        .max()
        .unwrap_or(0);
    cloned.next_id = cloned.next_id.max(bound);
    let dst = cloned.fresh();
    cloned
        .instrs
        .insert(0, Instr::legacy_call(dst, "__mind_now_ns", Vec::new()));
    cloned
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `module_offender` of a fixture that must parse, so a `None` is never a
    /// parse failure in disguise.
    fn offender(src: &str) -> Option<String> {
        assert!(
            crate::parser::parse(src).is_ok(),
            "fixture does not parse:\n{src}"
        );
        module_offender(src)
    }

    /// `(import, call, the row it names)`. `std::entropy::` is the `::` form
    /// a qualifier compared as written did not resolve.
    const SPELLINGS: &[(&str, &str, &str)] = &[
        (
            "import std.entropy;",
            "entropy.entropy_fill(p, 8)",
            "entropy_fill",
        ),
        (
            "import std.entropy;",
            "std.entropy.entropy_fill(p, 8)",
            "entropy_fill",
        ),
        (
            "import std.entropy;",
            "entropy::entropy_fill(p, 8)",
            "entropy_fill",
        ),
        (
            "import std.entropy;",
            "std::entropy::entropy_fill(p, 8)",
            "entropy_fill",
        ),
        (
            "import std.entropy as e;",
            "e.entropy_fill(p, 8)",
            "entropy_fill",
        ),
        (
            "import std.entropy as e;",
            "e::entropy_fill(p, 8)",
            "entropy_fill",
        ),
        (
            "import std.dns;",
            "std::dns::dns_resolve(p, 9)",
            "dns_resolve",
        ),
        (
            "import std.dns;",
            "dns.tcp_connect_host(p, 9, 80)",
            "tcp_connect_host",
        ),
        (
            "import std.dns;",
            "dns::tcp_connect_host_timeout(p, 9, 80, 100)",
            "tcp_connect_host_timeout",
        ),
        // A bare call of an imported std function: the qualifier is absent,
        // the name resolves to the std row.
        ("import std.entropy;", "entropy_fill(p, 8)", "entropy_fill"),
        ("import std.dns;", "dns_resolve(p, 9)", "dns_resolve"),
        (
            "import std.dns;",
            "tcp_connect_host(p, 9, 80)",
            "tcp_connect_host",
        ),
        (
            "import std.dns as d;",
            "tcp_connect_host_timeout(p, 9, 80, 100)",
            "tcp_connect_host_timeout",
        ),
    ];

    #[test]
    fn every_qualifier_spelling_is_a_world_use() {
        for (import, call, row) in SPELLINGS {
            let src = format!("{import}\n\npub fn f(p: i64) -> i64 {{\n    {call}\n}}\n");
            assert_eq!(offender(&src).as_deref(), Some(*row), "{call}");
        }
    }

    /// A same-named user function is not a world use, bare or qualified by a
    /// project module. The std calls are the positive control, the pure
    /// `dns_status` the negative one.
    #[test]
    fn a_user_function_named_like_a_row_is_not_a_world_use() {
        let bare = "fn entropy_fill(p: i64, n: i64) -> i64 {\n    p + n\n}\n\n\
                    pub fn f(p: i64) -> i64 {\n    entropy_fill(p, 8)\n}\n";
        assert_eq!(offender(bare), None);
        let project = "import w;\n\npub fn f(p: i64) -> i64 {\n    \
                       w.dns_resolve(p, 9) + w.tcp_connect_host(p, 9, 80)\n}\n";
        assert_eq!(offender(project), None);
        let qualified =
            "import std.dns;\n\npub fn f(p: i64) -> i64 {\n    dns.dns_resolve(p, 9)\n}\n";
        assert_eq!(offender(qualified).as_deref(), Some("dns_resolve"));
        let pure = "import std.dns;\n\npub fn f(p: i64) -> i64 {\n    dns.dns_status(p)\n}\n";
        assert_eq!(offender(pure), None);
    }

    /// A bare name is a world call only when it resolves to the std row: the
    /// module imports that std module and defines no function of that name.
    /// Each fixture below fails one of the two.
    #[test]
    fn a_bare_call_that_does_not_resolve_to_the_std_row_is_not_a_world_use() {
        // Defined in the module (top level, and inside a module block).
        let shadow = "import std.entropy;\n\nfn entropy_fill(p: i64, n: i64) -> i64 {\n    \
                      p + n\n}\n\npub fn f(p: i64) -> i64 {\n    entropy_fill(p, 8)\n}\n";
        assert_eq!(offender(shadow), None);
        let in_block = "import std.dns;\n\nmodule m {\n    pub fn dns_resolve(p: i64, n: i64) -> \
                        i64 {\n        p + n\n    }\n}\n\npub fn f(p: i64) -> i64 {\n    \
                        dns_resolve(p, 9)\n}\n";
        assert_eq!(offender(in_block), None);
        // The module imports a different std module than the row's.
        let other = "import std.dns;\n\npub fn f(p: i64) -> i64 {\n    entropy_fill(p, 8)\n}\n";
        assert_eq!(offender(other), None);
        // No import at all.
        let none = "pub fn f(p: i64) -> i64 {\n    dns_resolve(p, 9)\n}\n";
        assert_eq!(offender(none), None);
        // A pure function of the imported module.
        let pure = "import std.dns;\n\npub fn f(p: i64) -> i64 {\n    dns_status(p)\n}\n";
        assert_eq!(offender(pure), None);
        // Positive control: the shadow fixture without the definition.
        let world = "import std.entropy;\n\npub fn f(p: i64) -> i64 {\n    entropy_fill(p, 8)\n}\n";
        assert_eq!(offender(world).as_deref(), Some("entropy_fill"));
        // The sibling reach reads the same resolution.
        assert_eq!(
            reached_offender(world, "f").as_deref(),
            Some("entropy_fill")
        );
        assert_eq!(reached_offender(shadow, "f"), None);
    }

    /// `Q` is the qualifier. Each position holds a `std.entropy` use (a world
    /// use) and a project-module use of the same name (not one). The array,
    /// map, set, field-receiver, top-level `const` and value-read positions
    /// were outside the hand-written walker this scan replaced.
    const POSITIONS: &[&str] = &[
        "pub fn f(p: i64) -> i64 {\n    let a = [Q.entropy_fill(p, 8), 1];\n    p\n}\n",
        "pub fn f(p: i64) -> i64 {\n    let m = { 1: Q.entropy_fill(p, 8) };\n    p\n}\n",
        "pub fn f(p: i64) -> i64 {\n    let s = { Q.entropy_fill(p, 8), 1 };\n    p\n}\n",
        "pub fn f(p: i64) -> i64 {\n    Q.entropy_fill(p, 8).lo\n}\n",
        "const K: i64 = Q.entropy_fill(0, 8)\n\npub fn f(p: i64) -> i64 {\n    p\n}\n",
        "pub fn f(p: i64) -> i64 {\n    let h = Q.entropy_fill;\n    p\n}\n",
        "pub fn f(p: i64) -> i64 {\n    let h = |; x: i64| -> i64 { Q.entropy_fill(x, 8) };\n    p\n}\n",
        "module m {\n    pub fn f(p: i64) -> i64 {\n        Q.entropy_fill(p, 8)\n    }\n}\n",
    ];

    #[test]
    fn a_world_use_in_any_position_is_found() {
        for position in POSITIONS {
            let body = format!("import std.entropy;\nimport w;\n\n{position}");
            let world = body.replace("Q.", "entropy.");
            assert_eq!(offender(&world).as_deref(), Some("entropy_fill"), "{world}");
            let user = body.replace("Q.", "w.");
            assert_eq!(offender(&user), None, "{user}");
        }
    }

    /// A sibling function is judged by its own body only, nested uses
    /// included.
    #[test]
    fn reached_offender_reads_only_the_named_function() {
        let src = "import std.entropy;\n\npub fn g(p: i64) -> i64 {\n    p + 1\n}\n\n\
                   pub fn h(p: i64) -> i64 {\n    let a = [entropy.entropy_fill(p, 8)];\n    p\n}\n";
        assert_eq!(reached_offender(src, "g"), None);
        assert_eq!(reached_offender(src, "h").as_deref(), Some("entropy_fill"));
        assert_eq!(reached_offender(src, "absent"), None);
    }

    /// A source the parser rejects is never reported clean.
    #[test]
    fn an_unparsed_source_is_not_clean() {
        assert!(crate::parser::parse("pub fn f(p: i64) -> i64 {\n").is_err());
        assert_eq!(
            module_offender("pub fn f(p: i64) -> i64 {\n").as_deref(),
            Some(UNPARSED)
        );
    }

    /// The witness turns a deterministic body nondeterministic for the
    /// classifier `mindc verify` re-derives with, and leaves the input alone.
    #[test]
    fn the_stamp_is_a_witness_the_verifier_rederives() {
        let opts = crate::pipeline::CompileOptions::default();
        let products =
            crate::pipeline::compile_source("pub fn f(p: i64) -> i64 {\n    p + 1\n}\n", &opts)
                .expect("compile");
        assert!(crate::ir::ir_declares_deterministic(&products.ir));
        let stamped = stamp_unheld_world(&products.ir);
        assert!(!crate::ir::ir_declares_deterministic(&stamped));
        assert!(crate::ir::ir_declares_deterministic(&products.ir));
        // `mindc verify` runs both checks before it reads the label.
        assert_eq!(crate::ir::check_ssa_well_formed(&stamped), Ok(()));
        assert!(crate::ir::verify_module(&stamped).is_ok());
    }
}
