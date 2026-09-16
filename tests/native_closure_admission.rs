// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Entry-rooted admission controls for the native multi-module bridge.
//!
//! These lock the behaviour measured on the shipped compiler on 2026-09-07 and
//! the decisions ruled on it:
//!
//! * an UNCALLED out-of-profile body must not refuse an entry that never reaches
//!   it (the `unused_div` class), while the same body made reachable MUST refuse;
//! * a duplicate bare name is refused, never resolved by position — the frozen
//!   compiler resolves the flat image last-definition-wins and silently returned
//!   20 for `a,b,main` and 10 for `b,a,main`;
//! * a call with no definition in the closure is refused and NAMES the callee.
//!
//! The mutation control at the bottom is the one that makes the rest mean
//! something: it removes the reachability edge and asserts the verdict flips.

#![cfg(feature = "std-surface")]

use libmind::ir::IRModule;
use libmind::ir::frozen_profile::profile_frozen_admits;
use libmind::ir::native_closure::{
    admit_native_closure, defined_fn_names, executable_roots, reachable_from, restrict_to_reachable,
};
use std::collections::BTreeSet;

/// Lower one merged source text to one IRModule, exactly as the bridge must:
/// a single parse of the concatenated program, never spliced IR.
fn lower(src: &str) -> IRModule {
    let module = libmind::parser::parse(src).expect("fixture must parse");
    // `lower_to_ir` is fallible: a fixture that cannot lower is a broken
    // fixture, not a result this file is testing.
    libmind::eval::lower_to_ir(&module).expect("fixture must lower")
}

fn roots(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

/// Whole-module admission REFUSES an uncalled out-of-profile body.
///
/// The out-of-profile construct in these fixtures is `>>` (`binop.shr`). It used to be
/// `/` (`binop.div`), but signed i64 Div/Mod joined the frozen profile on 2026-09-16
/// (RH native slice), so `/` no longer demonstrates a refusal. The rulings under test
/// are about whole-module admission and reachability, not about which operator is out
/// of profile; `>>` is refused unconditionally, which keeps every fixture meaningful.
///
/// This test previously asserted the opposite -- that entry-rooted admission
/// should ACCEPT such a program -- under a ruling that has since been reversed.
/// Root preserves whole-user-program admission: the image is not pruned, so an
/// unreachable out-of-profile body still becomes bytes in the artifact, and
/// admitting it by reachability would let unproven bytes into a byte-identity
/// claim. Reachability supports diagnostics; it is not an admission optimisation.
#[test]
fn whole_module_admission_refuses_an_uncalled_out_of_profile_body() {
    let ir = lower(
        "fn unused_div(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
         fn main() -> i64 {\n    return 7;\n}\n",
    );
    let verdict = profile_frozen_admits(&ir);
    assert!(
        verdict.is_err(),
        "an uncalled out-of-profile body must still refuse under whole-module admission"
    );
    assert_eq!(verdict.unwrap_err().construct, "binop.shr");
}

/// The same body, now reachable, must still refuse. This is the half of the
/// ruling that keeps the fence honest.
#[test]
fn reachable_out_of_profile_body_still_refuses() {
    let ir = lower(
        "fn halve(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
         fn main() -> i64 {\n    return halve(14);\n}\n",
    );
    let reach = reachable_from(&ir, &roots(&["main"])).expect("main must resolve");
    assert!(reach.contains("halve"), "halve IS called from main");
    let restricted = restrict_to_reachable(&ir, &reach);
    let verdict = profile_frozen_admits(&restricted);
    assert!(verdict.is_err(), "a reachable `>>` must refuse");
    assert_eq!(verdict.unwrap_err().construct, "binop.shr");
}

/// Transitive reachability: main -> mid -> leaf. An out-of-profile construct two
/// hops away must still be seen.
#[test]
fn transitive_reachability_reaches_two_hops() {
    let ir = lower(
        "fn leaf(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
         fn mid(x: i64) -> i64 {\n    return leaf(x);\n}\n\n\
         fn main() -> i64 {\n    return mid(8);\n}\n",
    );
    let reach = reachable_from(&ir, &roots(&["main"])).expect("main must resolve");
    assert!(reach.contains("leaf"), "leaf is reachable through mid");
    let verdict = profile_frozen_admits(&restrict_to_reachable(&ir, &reach));
    assert_eq!(verdict.unwrap_err().construct, "binop.shr");
}

/// Reachability must follow a call made inside a nested control-flow body, not
/// only a call in the function's top-level instruction stream. Missing such an
/// edge would under-approximate the closure — the fail-OPEN direction.
#[test]
fn reachability_follows_calls_nested_in_control_flow() {
    let ir = lower(
        "fn hidden(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
         fn main() -> i64 {\n    let mut t: i64 = 0;\n    \
         if 1 < 2 {\n        t = hidden(4);\n    }\n    return t;\n}\n",
    );
    let reach = reachable_from(&ir, &roots(&["main"])).expect("main must resolve");
    assert!(
        reach.contains("hidden"),
        "a call inside an if-body is a real reachability edge"
    );
    assert!(profile_frozen_admits(&restrict_to_reachable(&ir, &reach)).is_err());
}

/// A call with no definition in the closure refuses AND names the callee — the
/// diagnostic the fence itself cannot produce.
#[test]
fn unresolved_edge_is_refused_and_names_the_callee() {
    let ir = lower("fn main() -> i64 {\n    return absent_helper(1);\n}\n");
    let err = reachable_from(&ir, &roots(&["main"])).expect_err("an undefined callee must refuse");
    assert_eq!(err.kind, "closure.unresolved_edge");
    assert!(
        err.detail.contains("absent_helper"),
        "refusal must name the unresolved callee, got: {}",
        err.detail
    );
}

/// A declared root that does not exist refuses rather than yielding an empty
/// closure — an empty closure would vacuously admit anything.
#[test]
fn missing_entry_root_is_refused() {
    let ir = lower("fn helper() -> i64 {\n    return 1;\n}\n");
    let err = reachable_from(&ir, &roots(&["main"]))
        .expect_err("a missing root must refuse, never admit vacuously");
    assert_eq!(err.kind, "closure.missing_entry_root");
}

/// Every definition is still visible to the resolver even when unreachable:
/// restriction is an ADMISSION view, and the wire image keeps the full text.
#[test]
fn restriction_does_not_pretend_the_body_is_gone() {
    let ir = lower(
        "fn unused_div(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
         fn main() -> i64 {\n    return 7;\n}\n",
    );
    let all = defined_fn_names(&ir);
    assert!(all.contains("unused_div") && all.contains("main"));
    let reach = reachable_from(&ir, &roots(&["main"])).unwrap();

    // Exercise restriction itself, not just two set sizes: the body must be gone
    // from the ADMISSION view while still present in the module the wire carries.
    let restricted = restrict_to_reachable(&ir, &reach);
    let kept = defined_fn_names(&restricted);
    assert!(kept.contains("main"), "entry survives restriction");
    assert!(
        !kept.contains("unused_div"),
        "the unreachable body is absent from the admission view"
    );
    assert!(
        defined_fn_names(&ir).contains("unused_div"),
        "and is still present in the module the wire image is built from"
    );
}

/// MUTATION CONTROL. Removing the reachability edge must flip the verdict.
///
/// Without this, every acceptance test above could pass against a closure that
/// silently returned nothing. Here one call site is the only difference between
/// the two programs, and it must decide admission.
#[test]
fn removing_the_reachability_edge_flips_admission() {
    const REACHING: &str = "fn halve(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
                            fn main() -> i64 {\n    return halve(14);\n}\n";
    const NOT_REACHING: &str = "fn halve(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
                                fn main() -> i64 {\n    return 7;\n}\n";

    let reaching = lower(REACHING);
    let reach = reachable_from(&reaching, &roots(&["main"])).unwrap();
    assert!(profile_frozen_admits(&restrict_to_reachable(&reaching, &reach)).is_err());

    let not_reaching = lower(NOT_REACHING);
    let reach2 = reachable_from(&not_reaching, &roots(&["main"])).unwrap();
    assert!(profile_frozen_admits(&restrict_to_reachable(&not_reaching, &reach2)).is_ok());
}

/// An EXPORTED function is externally callable, so it is a root even when the
/// entry never calls it. Missing this would make the closure smaller than the set
/// that can actually execute — the fail-OPEN direction.
#[test]
fn exported_functions_are_roots() {
    const SRC: &str = "export { exposed }\n\n\
                       fn exposed(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
                       fn main() -> i64 {\n    return 7;\n}\n";
    let ir = lower(SRC);
    assert!(
        ir.exports.contains("exposed"),
        "fixture must actually export, or this test proves nothing"
    );

    // Entry-only roots would MISS it and wrongly admit.
    let entry_only = reachable_from(&ir, &roots(&["main"])).unwrap();
    assert!(!entry_only.contains("exposed"));
    assert!(profile_frozen_admits(&restrict_to_reachable(&ir, &entry_only)).is_ok());

    // executable_roots includes the export, and admission then refuses.
    let all = executable_roots(&ir, "main");
    assert!(all.contains(&"exposed".to_string()));
    let reach = reachable_from(&ir, &all).unwrap();
    assert!(reach.contains("exposed"));
    let verdict = profile_frozen_admits(&restrict_to_reachable(&ir, &reach));
    assert!(
        verdict.is_err(),
        "an exported out-of-profile body must refuse"
    );
    assert_eq!(verdict.unwrap_err().construct, "binop.shr");
}

/// Control for the above: with no export, the same body is unreachable and the
/// program is admitted. So the refusal is caused by the export, not the body.
#[test]
fn without_the_export_the_same_body_is_admitted() {
    const SRC: &str = "fn exposed(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
                       fn main() -> i64 {\n    return 7;\n}\n";
    let ir = lower(SRC);
    assert!(ir.exports.is_empty());
    let reach = reachable_from(&ir, &executable_roots(&ir, "main")).unwrap();
    assert!(!reach.contains("exposed"));
    assert!(profile_frozen_admits(&restrict_to_reachable(&ir, &reach)).is_ok());
}

/// Root ordering is deterministic even though `IRModule.exports` is a HashSet.
#[test]
fn executable_roots_are_deterministically_ordered() {
    let ir = lower(
        "export { a_fn, b_fn, c_fn }\n\n\
         fn a_fn() -> i64 {\n    return 1;\n}\n\n\
         fn b_fn() -> i64 {\n    return 2;\n}\n\n\
         fn c_fn() -> i64 {\n    return 3;\n}\n\n\
         fn main() -> i64 {\n    return 0;\n}\n",
    );
    let first = executable_roots(&ir, "main");
    for _ in 0..16 {
        assert_eq!(
            executable_roots(&ir, "main"),
            first,
            "root order must be stable"
        );
    }
    assert_eq!(first, vec!["a_fn", "b_fn", "c_fn", "main"]);
}

// ---------------------------------------------------------------------------
// Controls added after architectural review.
// ---------------------------------------------------------------------------

/// THE BACKSTOP. If reachability ever under-approximates, the result must be a
/// loud refusal and never a silent admission.
///
/// Simulates a missed edge by restricting to the entry alone while `main` really
/// does call `helper`. The fence recomputes its `defined` set over the restricted
/// module, so the surviving call now names an undefined symbol. This is the
/// property that makes an under-approximating closure safe, and nothing pinned it.
#[test]
fn an_under_approximated_closure_refuses_rather_than_admits() {
    let ir = lower(
        "fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n\n\
         fn main() -> i64 {\n    return helper(6);\n}\n",
    );
    // The honest closure admits.
    let honest = reachable_from(&ir, &roots(&["main"])).unwrap();
    assert!(honest.contains("helper"));
    assert!(profile_frozen_admits(&restrict_to_reachable(&ir, &honest)).is_ok());

    // A closure that lost the edge must NOT admit.
    let mut starved = BTreeSet::new();
    starved.insert("main".to_string());
    let verdict = profile_frozen_admits(&restrict_to_reachable(&ir, &starved));
    assert!(
        verdict.is_err(),
        "an under-approximated closure must refuse, never admit silently"
    );
    assert_eq!(verdict.unwrap_err().construct, "call.undefined_or_builtin");
}

/// A call in a `while` CONDITION is a reachability edge, not only a call in its body.
#[test]
fn reachability_follows_a_call_in_a_while_condition() {
    let ir = lower(
        "fn limit() -> i64 {\n    return 3 >> 1;\n}\n\n\
         fn main() -> i64 {\n    let mut i: i64 = 0;\n    \
         while i < limit() {\n        i = i + 1;\n    }\n    return i;\n}\n",
    );
    let reach = reachable_from(&ir, &roots(&["main"])).expect("main resolves");
    assert!(
        reach.contains("limit"),
        "a call in a while-condition is an edge"
    );
    assert!(profile_frozen_admits(&restrict_to_reachable(&ir, &reach)).is_err());
}

/// A function called from MODULE TOP LEVEL executes even though no root body
/// names it. `admit_native_closure` seeds those calls as roots.
///
/// Both halves are asserted unconditionally: entry-only roots MUST miss it (or the
/// fixture stopped exercising the gap), and the entry point MUST catch it.
#[test]
fn top_level_calls_are_roots() {
    const SRC: &str = "fn side(x: i64) -> i64 {\n    return x >> 2;\n}\n\n\
                       fn main() -> i64 {\n    return 7;\n}\n\n\
                       side(8);\n";
    let ast = libmind::parser::parse(SRC).expect("fixture must parse");
    let ir = lower(SRC);

    let entry_only = reachable_from(&ir, &roots(&["main"])).unwrap();
    assert!(
        !entry_only.contains("side"),
        "fixture must exercise the gap: entry-only roots have to MISS a top-level call"
    );

    let restricted = admit_native_closure(&ast, &ir, "main", &BTreeSet::new())
        .expect("no ambiguity in this fixture");
    let verdict = profile_frozen_admits(&restricted);
    assert!(
        verdict.is_err(),
        "a top-level call reaches `side`, whose body is out of profile"
    );
    assert_eq!(verdict.unwrap_err().construct, "binop.shr");
}

/// The one entry point refuses ambiguity BEFORE walking, so a duplicated name can
/// never have its first definition walked while the compiler runs its last.
#[test]
fn the_entry_point_refuses_ambiguity_before_walking() {
    const SRC: &str = "fn v() -> i64 {\n    return 10;\n}\n\n\
                       fn v() -> i64 {\n    return 20 >> 2;\n}\n\n\
                       fn main() -> i64 {\n    return v();\n}\n";
    let ast = libmind::parser::parse(SRC).expect("parses");
    let ir = lower(SRC);
    let err = admit_native_closure(&ast, &ir, "main", &BTreeSet::new())
        .expect_err("duplicate must refuse at the entry point");
    assert_eq!(err.kind, "closure.duplicate_definition");
}

/// A reachable function nested inside an UNREACHABLE parent keeps its parent in
/// the restricted module, so the closure and the restriction agree about what a
/// definition is.
#[test]
fn a_reachable_nested_definition_keeps_its_enclosing_parent() {
    let ir = lower(
        "fn outer() -> i64 {\n    fn inner() -> i64 {\n        return 5;\n    }\n    \
         return inner();\n}\n\n\
         fn main() -> i64 {\n    return 7;\n}\n",
    );
    let mut keep = BTreeSet::new();
    keep.insert("main".to_string());
    keep.insert("inner".to_string());
    let restricted = restrict_to_reachable(&ir, &keep);
    let kept = defined_fn_names(&restricted);
    assert!(
        kept.contains("inner"),
        "a kept nested definition must survive restriction, got {kept:?}"
    );
}
