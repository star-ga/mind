// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unit tests for the intrinsic registry and the determinism classifier
//! (`crate::intrinsics`), in their own file so the registry stays under the
//! 800-line ceiling `tests/module_size_ratchet.rs` pins.

use super::*;

/// Every `__mind_*` symbol `runtime-support/mind_intrinsics.c` EXPORTS, in
/// the order the C file declares them.
fn exported_runtime_support_intrinsics() -> Vec<String> {
    const C_SRC: &str = include_str!("../../runtime-support/mind_intrinsics.c");
    let mut names = Vec::new();
    for line in C_SRC.lines() {
        let line = line.trim_start();
        // `#  define MIND_EXPORT ...` starts with `#`, so macro definitions
        // are skipped and only definition sites are scanned.
        if !line.starts_with("MIND_EXPORT") {
            continue;
        }
        let Some(at) = line.find("__mind_") else {
            continue;
        };
        let rest = &line[at..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        // Only a call-shaped `__mind_x(` is a definition site.
        if rest[end..].starts_with('(') {
            names.push(rest[..end].to_string());
        }
    }
    names.sort();
    names.dedup();
    names
}

/// THE structural gate this module exists for: a new `__mind_*` intrinsic in
/// the C runtime fails this test until someone classifies it, instead of
/// being silently admitted as deterministic by a blanket `__mind_*` rule.
#[test]
fn classification_covers_every_runtime_support_intrinsic() {
    let exported = exported_runtime_support_intrinsics();
    // A scan that finds nothing must not pass vacuously.
    assert!(
        exported.len() >= 25,
        "runtime-support scan found only {} intrinsics — the scan is broken, \
         not the runtime: {exported:?}",
        exported.len()
    );
    let unclassified: Vec<&String> = exported
        .iter()
        .filter(|n| intrinsic_determinism(n).is_none())
        .collect();
    assert!(
        unclassified.is_empty(),
        "these `__mind_*` intrinsics are exported by runtime-support but not \
         classified deterministic or non-deterministic: {unclassified:?} — add \
         each to STD_SURFACE_INTRINSICS (with its Det) or to \
         UNREGISTERED_{{WORLD,PURE}}_INTRINSICS. An unclassified intrinsic that \
         reads the clock/entropy would be attested `deterministic`."
    );
}

/// The classifier must actually READ the registry row, not a parallel array.
#[test]
fn classifier_agrees_with_every_registry_row() {
    for spec in STD_SURFACE_INTRINSICS {
        assert_eq!(
            callee_is_nondeterministic(spec.physical_name),
            spec.det == Det::World,
            "`{}` is classified {:?} in the registry but the classifier disagrees",
            spec.physical_name,
            spec.det
        );
    }
}

/// The F5 regression: the registered mind-nerve host surface (clock, entropy,
/// stdin, getenv) and the wall clock must all report non-deterministic. Each
/// of these was attested `deterministic` — and passed `mindc verify
/// --require-deterministic` — before the registry carried the verdict.
#[test]
fn world_reading_intrinsics_are_nondeterministic() {
    for name in [
        "__mind_nerve_rt_monotonic_ns",
        "__mind_nerve_rt_os_entropy",
        "__mind_nerve_rt_read_stdin",
        "__mind_nerve_rt_getenv",
        "__mind_now_ns",
        "__mind_blas_get_use_avx2",
        "now_ns",
    ] {
        assert!(
            callee_is_nondeterministic(name),
            "`{name}` reads the world but is classified deterministic"
        );
    }
}

/// The pure side of the same invariant: the allocator / raw-memory / BLAS
/// surface must NOT be tainted, or every std-surface program is attested
/// `nondeterministic` (the over-taint failure mode).
#[test]
fn pure_intrinsics_are_not_tainted() {
    for name in [
        "__mind_alloc",
        "__mind_load_i64",
        "__mind_store_i64",
        "__mind_read",
        "__mind_write",
        "__mind_blas_dot_q16",
        "__mind_nerve_rt_write_stdout",
        "__mind_nerve_blas_dot_q16_i64",
    ] {
        assert!(
            !callee_is_nondeterministic(name),
            "`{name}` is pure w.r.t. declared inputs but is tainted"
        );
    }
}

#[test]
fn registry_and_side_lists_do_not_overlap_or_repeat() {
    let mut seen = std::collections::BTreeSet::new();
    for spec in STD_SURFACE_INTRINSICS {
        assert!(
            seen.insert(spec.physical_name),
            "duplicate registry entry `{}`",
            spec.physical_name
        );
    }
    for name in UNREGISTERED_WORLD_INTRINSICS
        .iter()
        .chain(UNREGISTERED_PURE_INTRINSICS)
    {
        assert!(
            seen.insert(*name),
            "`{name}` is in a side list AND the registry — the registry row wins, \
             so the side entry is dead and can only mislead"
        );
    }
}

/// A qualified path must resolve to the same verdict as the bare name, or a
/// module-qualified clock read walks straight past the classifier.
#[test]
fn qualified_paths_resolve_to_the_same_verdict() {
    assert!(callee_is_nondeterministic("std.rand.random"));
    assert!(callee_is_nondeterministic("rng::rand_uniform"));
    assert!(callee_is_nondeterministic("time.now_ns"));
    assert!(callee_is_nondeterministic(
        "nerve::__mind_nerve_rt_os_entropy"
    ));
    assert!(!callee_is_nondeterministic("std.blas.dot_q16"));
}

/// The std modules whose World-reaching public functions are rows of
/// `std_qualified`, with their embedded sources.
#[cfg(feature = "std-surface")]
const NAMED_STD_MODULES: &[(&str, &str)] = &[
    ("std.dns", include_str!("../../std/dns.mind")),
    ("std.entropy", include_str!("../../std/entropy.mind")),
];

/// Every callee named anywhere in `instrs`, through every nested body.
#[cfg(feature = "std-surface")]
fn callees(instrs: &[crate::ir::Instr], out: &mut std::collections::BTreeSet<String>) {
    for instr in instrs {
        if let crate::ir::Instr::Call { name, .. } = instr {
            out.insert(name.clone());
        }
        for body in crate::ir::instr_bodies(instr) {
            callees(body, out);
        }
    }
}

/// `(reaches the world, does not)`: the `pub fn`s of `src`, split by whether
/// their body reaches a World callee directly or through other functions of the
/// module. Derived from the compiled IR, never from the rows under test: a
/// callee of the module itself counts only through its own body.
#[cfg(feature = "std-surface")]
fn public_world_reach(module: &str, src: &str) -> (Vec<String>, Vec<String>) {
    use std::collections::{BTreeMap, BTreeSet};
    let ast = crate::parser::parse(src).unwrap_or_else(|e| panic!("{module}: parse: {e:?}"));
    let public: Vec<String> = ast
        .items
        .iter()
        .filter_map(|n| match n {
            crate::ast::Node::FnDef(f, _) if f.is_pub => Some(f.name.clone()),
            _ => None,
        })
        .collect();
    let ir = crate::pipeline::compile_source(src, &crate::pipeline::CompileOptions::default())
        .unwrap_or_else(|e| panic!("{module}: compile: {e:?}"))
        .ir;
    let mut calls: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for instr in &ir.instrs {
        if let crate::ir::Instr::FnDef { name, body, .. } = instr {
            let mut out = BTreeSet::new();
            callees(body, &mut out);
            calls.insert(name.clone(), out);
        }
    }
    let mut world: BTreeSet<String> = BTreeSet::new();
    loop {
        let before = world.len();
        for (f, cs) in &calls {
            let reaches = cs.iter().any(|c| {
                if calls.contains_key(c) {
                    world.contains(c)
                } else {
                    callee_is_nondeterministic(c)
                }
            });
            if reaches {
                world.insert(f.clone());
            }
        }
        if world.len() == before {
            break;
        }
    }
    for f in &public {
        assert!(
            calls.contains_key(f),
            "{module}: `pub fn {f}` has no IR body"
        );
    }
    public.into_iter().partition(|f| world.contains(f))
}

/// A program reaches these modules only by a qualified call, so each public
/// function whose body reads the world is a row of `std_qualified` and is NOT
/// a bare nondeterministic name (a user function of that spelling is
/// deterministic). No other public function is a row. Two-sided, so a row
/// dropped and a row too many both fail, and a scan that found nothing on
/// either side fails rather than passing vacuously.
#[cfg(feature = "std-surface")]
#[test]
fn public_world_reach_of_std_entropy_and_dns_is_named() {
    let mut named = 0usize;
    for (module, src) in NAMED_STD_MODULES {
        let (reach, pure) = public_world_reach(module, src);
        assert!(
            !reach.is_empty() && !pure.is_empty(),
            "{module}: vacuous scan, reach={reach:?} pure={pure:?}"
        );
        for f in &reach {
            assert!(
                crate::std_qualified::is_row(module, f),
                "{module}: `{f}` reaches a World extern but is not a qualified row, \
                 so a caller of it is attested deterministic"
            );
            assert!(
                !callee_is_nondeterministic(f),
                "{module}: `{f}` is classified by its bare name, so a user function \
                 of that spelling is refused"
            );
            named += 1;
        }
        for f in &pure {
            assert!(
                !crate::std_qualified::is_row(module, f) && !callee_is_nondeterministic(f),
                "{module}: `{f}` reaches no World callee but is classified \
                 nondeterministic"
            );
        }
    }
    assert_eq!(
        named, 4,
        "dns_resolve, tcp_connect_host(_timeout), entropy_fill"
    );
}
