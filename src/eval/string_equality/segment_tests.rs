// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unit controls for the per-module analyses of the merged `mindc test`
//! module (see `segments`) and for variant payloads stored across the
//! modules of a project build.

use super::*;
use crate::eval::eval_owned_symbol;

const UNTYPED: RefusedKind = RefusedKind::Untyped;

fn parse(src: &str) -> Module {
    crate::parser::parse(src).unwrap()
}

/// `item` (a `FnDef`) named for `owner`, as the test merge names it.
fn owned(owner: &str, mut item: Node) -> Node {
    if let Node::FnDef(fd, _) = &mut item {
        fd.name = eval_owned_symbol(owner, &fd.name);
    }
    item
}

/// The forwarder the merge builds for a bare `name`: `src` is its
/// signature with a body calling `name` itself, retargeted to `owner`'s
/// definition.
fn forwarder(owner: &str, name: &str, src: &str) -> Node {
    let mut module = parse(src);
    let mut item = module.items.remove(0);
    let Node::FnDef(fd, _) = &mut item else {
        panic!("forwarder source is a function")
    };
    let call = match fd.body.as_mut_slice() {
        [Node::Return { value: Some(v), .. }] => v.as_mut(),
        _ => panic!("forwarder body is one return"),
    };
    let Node::Call { callee, .. } = call else {
        panic!("forwarder returns a call")
    };
    *callee = eval_owned_symbol(owner, name);
    item
}

fn interpreted_refusals(module: &Module) -> Vec<RefusedKind> {
    analyze_interpreted(module)
        .refused
        .iter()
        .map(|r| r.kind)
        .collect()
}

/// A string the test stores to raw memory is a fact of the test only: the
/// merged std.ecdsa_p256 body compares two raw loads as integers, as the
/// build of that module alone does. Mutant M3 (std code decided with the
/// test's facts, or one walk of the whole merged module) refuses it.
#[test]
fn merged_std_code_keeps_its_own_facts() {
    let std = crate::project::stdlib::bundled_module_ast("std.ecdsa_p256").unwrap();
    let eq17 = std
        .items
        .iter()
        .find(|item| matches!(item, Node::FnDef(fd, _) if fd.name == "p256_bn_eq17"))
        .cloned()
        .unwrap();
    let user = "fn probe(p: i64) -> i64 {
        __mind_store_i64(p, \"abc\")
        return 1
    }";
    let mut module = parse(user);
    module
        .items
        .insert(0, owned("std.ecdsa_p256", eq17.clone()));
    assert_eq!(interpreted_refusals(&module), vec![]);
    // Positive control: the same body as the test's own code is refused.
    let mut module = parse(user);
    module.items.insert(0, eq17);
    assert_eq!(interpreted_refusals(&module), vec![UNTYPED]);
}

/// A bare `vec_push` the merge forwards to std.vec is the opaque std
/// function the build calls: the test's string escapes into a `Vec`, which
/// `string_len` does not take, so its results stay integers. Mutant M4
/// (forwarders decided as the test's own functions) makes every opaque
/// result maybe a string and refuses the comparison.
#[test]
fn forwarders_are_the_opaque_callee() {
    let mut module = parse(
        "use std.vec;
        use std.string;
        fn keep(v: Vec, s: string) -> Vec { return vec_push(v, s) }
        fn probe() -> i64 {
            let v = keep(vec_new(), \"abc\")
            if string_len(\"ab\") == string_len(\"ab\") { return 1 }
            return 0
        }",
    );
    let bridge = forwarder(
        "std.vec",
        "vec_push",
        "fn vec_push(v: Vec, value: i64) -> Vec { return vec_push(v, value) }",
    );
    module.items.insert(0, bridge);
    assert_eq!(interpreted_refusals(&module), vec![]);
    // Positive control: a raw store reaches every opaque integer result.
    let raw = parse(
        "use std.string;
        fn probe(p: i64) -> i64 {
            __mind_store_i64(p, \"abc\")
            if string_len(\"ab\") == string_len(\"ab\") { return 1 }
            return 0
        }",
    );
    assert_eq!(interpreted_refusals(&raw), vec![UNTYPED]);
}

/// The facts of a std module come from a walk of its whole body. No shipped
/// std module tells a walk of only the merged functions apart (mutant M6),
/// so this pins the source of the facts.
#[test]
fn std_facts_are_the_full_module_fixpoint() {
    let none = format!("{:?}", slots::Contradicted::default());
    let mut recorded = Vec::new();
    for name in ["std.json", "std.vec", "std.string", "std.fs"] {
        let ast = crate::project::stdlib::bundled_module_ast(name).unwrap();
        let fresh = {
            let _scope = segments::StdScope::install(name);
            let ordinals = equality_ordinals(ast);
            let decls = facts::Decls::seeded(ast, Default::default(), Default::default());
            let (_, _, _, decls) = walk::fixpoint(decls, &ordinals, &[ast.items.as_slice()]);
            let mut fresh = decls.contradicted;
            fresh.locals.clear();
            format!("{fresh:?}")
        };
        let cached = format!("{:?}", segments::std_facts(name).unwrap());
        assert_eq!(cached, fresh, "{name}");
        if fresh != none {
            recorded.push(name);
        }
    }
    // Positive control: some module records a slot, so a walk of nothing
    // would differ.
    assert!(!recorded.is_empty(), "no std module recorded a slot");
}

/// The merge names a project test's own functions for its module; a string
/// passed to its own `eqh(a: i64, b: i64)` is still a store into that
/// function's parameters. Mutant M5 (functions keyed by the merged name)
/// misses it.
#[cfg(feature = "cross-module-imports")]
#[test]
fn project_test_own_functions_keep_bare_keys() {
    let _module = crate::qualified_enums::ModuleGuard::install("crate.main");
    let mut module = parse(
        "fn eqh(a: i64, b: i64) -> i64 {
            if a == b { return 1 }
            return 0
        }
        fn probe() -> i64 { return eqh(\"abc\", \"abc\") }",
    );
    module.items = module
        .items
        .into_iter()
        .map(|item| owned("crate.main", item))
        .collect();
    assert_eq!(interpreted_refusals(&module), vec![UNTYPED]);
}

/// Each module of a project build analysed with the whole-project registry
/// installed and its module path current: the refusal kinds of `analysed`.
#[cfg(feature = "cross-module-imports")]
fn project_refusals_reparsed(modules: &[(&str, &str)], analysed: &str) -> Vec<RefusedKind> {
    let parsed: Vec<(String, Module)> = modules
        .iter()
        .map(|(path, src)| (path.to_string(), parse(src)))
        .collect();
    let enums = crate::project::build_global_enums(&parsed);
    let _enums = crate::qualified_enums::GlobalGuard::install(enums);
    let _module = crate::qualified_enums::ModuleGuard::install(analysed);
    // The project build parses each module with the whole-project registry
    // installed. In particular, this rewrites an imported constructor to its
    // registry key and makes the colliding-owner case exercise that lookup.
    let (_, source) = modules.iter().find(|(path, _)| *path == analysed).unwrap();
    let module = parse(source);
    analyze(&module).refused.iter().map(|r| r.kind).collect()
}

#[cfg(feature = "cross-module-imports")]
const EVENTS: &str = "pub enum XmlEvent { Start, Text(i64), End }
pub fn same_text(a: XmlEvent, b: XmlEvent) -> i64 {
    let x = match a {
        XmlEvent::Text(n) => n,
        _ => 0,
    }
    let y = match b {
        XmlEvent::Text(n) => n,
        _ => 1,
    }
    if x == y { return 1 }
    return 0
}
";

#[cfg(feature = "cross-module-imports")]
const SHADOW: &str = "export { shadow_code }
enum XmlEvent { Open, Close }
fn shadow_code() -> i64 { return 7 }
";

#[cfg(feature = "cross-module-imports")]
fn probe_main(payload: &str, shadow: bool) -> String {
    let imports = if shadow {
        "import events\nimport shadow\n"
    } else {
        "import events\n"
    };
    format!(
        "{imports}pub fn probe() -> i64 {{
            let a = XmlEvent::Text({payload})
            let b = XmlEvent::Text({payload})
            return same_text(a, b)
        }}"
    )
}

/// A string another module stores into the `i64` payload of an imported
/// variant is read by the declaring module, which refuses its comparison;
/// the storing module refuses the call. Mutant M1 (payload types read from
/// the legacy table only, which the project rebuild clears) misses the
/// store; mutant M2 (a local enum keyed by its bare name) misses it when a
/// third module declares a private enum of the same name.
#[cfg(feature = "cross-module-imports")]
#[test]
fn imported_payload_store_is_read_by_the_declaring_module() {
    for shadow in [false, true] {
        let main = probe_main("\"abc\"", shadow);
        let mut modules = vec![("crate.events", EVENTS), ("crate.main", main.as_str())];
        if shadow {
            modules.push(("crate.shadow", SHADOW));
        }
        assert_eq!(
            project_refusals_reparsed(&modules, "crate.events"),
            vec![UNTYPED],
            "shadow={shadow}"
        );
        assert_eq!(
            project_refusals_reparsed(&modules, "crate.main"),
            vec![RefusedKind::Callee],
            "shadow={shadow}"
        );
        // Positive control: integers in the payload.
        let ints = probe_main("7", shadow);
        modules[1].1 = ints.as_str();
        assert_eq!(project_refusals_reparsed(&modules, "crate.events"), vec![]);
        assert_eq!(project_refusals_reparsed(&modules, "crate.main"), vec![]);
    }
}

/// A struct variant's `i64` field given a string by another module.
#[cfg(feature = "cross-module-imports")]
#[test]
fn imported_struct_variant_store_is_read_by_the_declaring_module() {
    let events = "pub enum XmlEvent { Start, Text { s: i64 }, End }
    pub fn same_text(a: XmlEvent, b: XmlEvent) -> i64 {
        let x = match a {
            XmlEvent.Text { s } => s,
            _ => 0,
        }
        let y = match b {
            XmlEvent.Text { s } => s,
            _ => 1,
        }
        if x == y { return 1 }
        return 0
    }";
    let main = "import events
    pub fn probe() -> i64 {
        let a = XmlEvent.Text { s: \"abc\" }
        let b = XmlEvent.Text { s: \"abc\" }
        return same_text(a, b)
    }";
    let modules = [("crate.events", events), ("crate.main", main)];
    assert_eq!(
        project_refusals_reparsed(&modules, "crate.events"),
        vec![UNTYPED]
    );
}

/// An imported variant declared to carry a `string` compares bytes in the
/// importing module: its payload type is read from the project registry.
#[cfg(feature = "cross-module-imports")]
#[test]
fn an_imported_string_payload_compares_bytes() {
    let modules = [
        ("crate.names", "pub enum E { Name(string), Other }\n"),
        (
            "crate.main",
            "import names
            pub fn probe(e: E) -> i64 {
                return match e {
                    E::Name(s) => if s == \"x\" { 1 } else { 0 },
                    _ => 2,
                }
            }",
        ),
    ];
    assert_eq!(project_refusals_reparsed(&modules, "crate.main"), vec![]);
}
