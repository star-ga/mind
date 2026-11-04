// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unit controls for strings that reach an integer slot through a value whose
//! type the pass did not establish, through raw reads of collection storage,
//! through another module of a project, or through a string literal `match`
//! pattern; and for the interpreter's string method calls.

use super::*;

/// The refusal kinds `analyze` reports for `src`, in source order.
fn refusals(src: &str) -> Vec<RefusedKind> {
    let module = crate::parser::parse(src).unwrap();
    analyze(&module).refused.iter().map(|r| r.kind).collect()
}

const UNTYPED: RefusedKind = RefusedKind::Untyped;

const EQH: &str = "fn eqh(a: i64, b: i64) -> i64 {
    if a == b { return 1 }
    return 0
}
";

/// A value block whose result is a local it reassigns (`{ let mut t = "x";
/// t = "abc"; t }`): its type was not established, so storing it in an `i64`
/// parameter, result or field was not recorded.
const BLOCK: &str =
    "if true { let mut t = \"x\"; t = \"abc\"; t } else { let mut u = \"y\"; u = \"abc\"; u }";

#[test]
fn values_of_unestablished_type_stored_in_integer_slots_are_not_trusted() {
    let tuple = format!(
        "{EQH}fn pair() -> (i64, i64) {{ return (\"abc\", \"abc\") }}
        fn p() -> i64 {{
            let (a, b) = pair()
            return eqh(a, b)
        }}"
    );
    assert_eq!(refusals(&tuple), vec![UNTYPED]);
    let param = format!(
        "{EQH}fn p() -> i64 {{
            let a = {BLOCK}
            let b = {BLOCK}
            return eqh(a, b)
        }}"
    );
    assert_eq!(refusals(&param), vec![UNTYPED]);
    let result = format!(
        "fn mk() -> i64 {{
            let a = {BLOCK}
            return a
        }}
        fn p() -> i64 {{
            if mk() == mk() {{ return 1 }}
            return 0
        }}"
    );
    assert_eq!(refusals(&result), vec![UNTYPED]);
    let field = format!(
        "struct Box {{ v: i64 }}
        fn p() -> i64 {{
            let x = Box {{ v: {BLOCK} }}
            let y = Box {{ v: {BLOCK} }}
            if x.v == y.v {{ return 1 }}
            return 0
        }}"
    );
    assert_eq!(refusals(&field), vec![UNTYPED]);
    // Positive controls: the same shapes carrying integers.
    let ints = format!(
        "{EQH}fn pair() -> (i64, i64) {{ return (7, 7) }}
        fn p() -> i64 {{
            let (a, b) = pair()
            let c = if true {{ let mut t = 1; t = 7; t }} else {{ 7 }}
            return eqh(a, b) + eqh(c, a)
        }}"
    );
    assert_eq!(refusals(&ints), vec![]);
}

/// A local binding is a slot like any other: a value assigned in an inner
/// scope (a loop body's own binding) is read in that scope.
#[test]
fn local_bindings_record_every_assigned_value() {
    let string = format!(
        "{EQH}fn p() -> i64 {{
            let mut a = 0
            let mut b: i64 = 0
            let mut i = 0
            while i < 1 {{
                let s = \"abc\"
                a = s
                b = s
                i = i + 1
            }}
            return eqh(a, b)
        }}"
    );
    assert_eq!(refusals(&string), vec![UNTYPED]);
    // Positive control: an integer bound in the inner scope.
    let ints = format!(
        "{EQH}fn p() -> i64 {{
            let mut a = 0
            let mut b: i64 = 0
            let mut i = 0
            while i < 3 {{
                let v: i64 = i * 2
                a = v
                b = v
                i = i + 1
            }}
            return eqh(a, b)
        }}"
    );
    assert_eq!(refusals(&ints), vec![]);
}

/// A math builtin (`sqrt`, `exp`) returns a number: giving its result to an
/// `f64` parameter stores no string there.
#[test]
fn math_builtin_results_are_not_strings() {
    let src = "fn is_nan(x: f64) -> i64 {
        if x != x { return 1 }
        return 0
    }
    fn p() -> i64 { return is_nan(sqrt(0.0 - 4.0)) + is_nan(exp(1.0)) }";
    assert_eq!(refusals(src), vec![]);
}

/// A string kept by a std collection: a value of unestablished type pushed,
/// and raw 64-bit reads of the collection's storage (`vec_addr`,
/// `map_vals_addr`); a raw store read back through the collection.
#[test]
fn collection_storage_read_raw_may_hold_strings() {
    let unknown = format!(
        "fn p() -> i64 {{
            let v = vec_push(vec_push(vec_new(), {BLOCK}), {BLOCK})
            if vec_get(v, 0) == vec_get(v, 1) {{ return 1 }}
            return 0
        }}"
    );
    assert_eq!(refusals(&unknown), vec![UNTYPED]);
    let vec_raw = "fn p() -> i64 {
        let v = vec_push(vec_push(vec_new(), \"abc\"), \"abc\")
        let a = vec_addr(v)
        if __mind_load_i64(a) == __mind_load_i64(a + 8) { return 1 }
        return 0
    }";
    assert_eq!(refusals(vec_raw), vec![UNTYPED]);
    let map_raw = "fn p() -> i64 {
        let m = map_insert(map_insert(map_new(), 1, \"abc\"), 2, \"abc\")
        let a = map_vals_addr(m)
        if __mind_load_i64(a) == __mind_load_i64(a + 8) { return 1 }
        return 0
    }";
    assert_eq!(refusals(map_raw), vec![UNTYPED]);
    let raw_then_get = "fn p() -> i64 {
        let v = vec_push(vec_push(vec_new(), 0), 0)
        __mind_store_i64(vec_addr(v), \"abc\")
        __mind_store_i64(vec_addr(v) + 8, \"abc\")
        if vec_get(v, 0) == vec_get(v, 1) { return 1 }
        return 0
    }";
    assert_eq!(refusals(raw_then_get), vec![UNTYPED]);
    // Positive control: integers kept and read raw.
    let ints = "fn p() -> i64 {
        let v = vec_push(vec_push(vec_new(), 7), 7)
        let a = vec_addr(v)
        if __mind_load_i64(a) == __mind_load_i64(a + 8) { return 1 }
        if vec_get(v, 0) == vec_get(v, 1) { return 2 }
        return 0
    }";
    assert_eq!(refusals(ints), vec![]);
}

/// A string literal `match` arm compares bytes, so its scrutinee must be a
/// string; a match with no `==` anywhere is still analysed.
#[test]
fn string_literal_patterns_need_a_string_scrutinee() {
    let integer = "fn is_abc(s: i64) -> i64 {
        match s {
            \"abc\" => 1,
            _ => 0,
        }
    }";
    assert_eq!(refusals(integer), vec![RefusedKind::Pattern]);
    let string = "fn is_abc(s: string) -> i64 {
        match s {
            \"abc\" => 1,
            _ => 0,
        }
    }";
    assert_eq!(refusals(string), vec![]);
    let numbers = "fn one(n: i64) -> i64 {
        match n {
            1 => 1,
            _ => 0,
        }
    }";
    assert_eq!(refusals(numbers), vec![]);
}

/// `.length` on a string reads a member no string record has; on an array
/// it is the element count.
#[test]
fn the_length_member_of_a_string_is_refused() {
    let string = "fn slen(s: string) -> i64 { return s.length }";
    assert_eq!(refusals(string), vec![RefusedKind::Member]);
    let array = "fn alen(xs: [i64; 2]) -> i64 { return xs.length }";
    assert_eq!(refusals(array), vec![]);
}

/// The `string_*` calls a method on a string becomes for the interpreter,
/// sorted: `(callee, argument count)`, calls written as such left out.
fn interpreted_calls(src: &str) -> Vec<(String, usize)> {
    fn collect(node: &Node, out: &mut Vec<(String, usize)>) {
        if let Node::Call { callee, args, .. } = node {
            if callee.starts_with("string_") {
                out.push((callee.clone(), args.len()));
            }
        }
        children_ref(node).into_iter().for_each(|c| collect(c, out));
    }
    let module = crate::parser::parse(src).unwrap();
    assert!(
        analyze(&module).rewritten.is_none(),
        "lowering keeps the method call: {src}"
    );
    let Some(rewritten) = analyze_interpreted(&module).rewritten else {
        return Vec::new();
    };
    let mut written = Vec::new();
    module
        .items
        .iter()
        .for_each(|item| collect(item, &mut written));
    let mut out = Vec::new();
    rewritten
        .items
        .iter()
        .for_each(|item| collect(item, &mut out));
    for call in written {
        let at = out.iter().position(|c| *c == call).expect("a written call");
        out.remove(at);
    }
    out.sort();
    out
}

/// `mindc test` runs a method on a string as the `string_<method>` call the
/// MLIR lowering makes for it; the lowering's own input is unchanged.
#[test]
fn string_methods_become_calls_for_the_interpreter_only() {
    let calls = interpreted_calls(
        "fn p(s: string) -> i64 {
            return s.len() + string_len(s.slice(1, 3)) + string_len(s.trim())
        }",
    );
    let expected = [("string_len", 1), ("string_slice", 3), ("string_trim", 1)];
    // Sorted, as `interpreted_calls` returns them.
    let expected: Vec<(String, usize)> =
        expected.iter().map(|(c, n)| (c.to_string(), *n)).collect();
    assert_eq!(calls, expected);
    // Positive control: a method on a record is not a string method.
    let calls = interpreted_calls(
        "struct Box { v: i64 }
        fn p(b: Box) -> i64 { return b.v() }",
    );
    assert_eq!(calls, vec![]);
}

/// A string stored into a slot another module of the project declares (an
/// `i64` parameter, result or field) is a store that module reads.
#[cfg(feature = "cross-module-imports")]
#[test]
fn stores_from_another_project_module_are_not_trusted() {
    fn project_refusals(modules: &[(&str, &str)], analysed: &str) -> Vec<RefusedKind> {
        let parsed: Vec<(String, Module)> = modules
            .iter()
            .map(|(path, src)| (path.to_string(), crate::parser::parse(src).unwrap()))
            .collect();
        let enums = crate::project::build_global_enums(&parsed);
        assert!(enums.string_project.is_some(), "two project modules");
        let _enums = crate::qualified_enums::GlobalGuard::install(enums);
        let _module = crate::qualified_enums::ModuleGuard::install(analysed);
        let (_, module) = parsed.iter().find(|(path, _)| path == analysed).unwrap();
        analyze(module).refused.iter().map(|r| r.kind).collect()
    }
    let provider = format!("{EQH}fn mk() -> i64 {{ return \"abc\" }}");
    let param = [
        ("crate.provider", provider.as_str()),
        (
            "crate.main",
            "import provider;\nfn probe() -> i64 { return provider.eqh(\"abc\", \"abc\") }",
        ),
    ];
    assert_eq!(project_refusals(&param, "crate.provider"), vec![UNTYPED]);
    // The importing module, whose own `mindc check` sees no comparison,
    // refuses the call of the refused function.
    assert_eq!(
        project_refusals(&param, "crate.main"),
        vec![RefusedKind::Callee]
    );
    let result = [
        ("crate.provider", provider.as_str()),
        (
            "crate.main",
            "import provider;
            fn probe() -> i64 {
                if provider.mk() == provider.mk() { return 1 }
                return 0
            }",
        ),
    ];
    assert_eq!(project_refusals(&result, "crate.main"), vec![UNTYPED]);
    let field = [
        (
            "crate.provider",
            "struct Box { v: i64 }
            fn same_box(x: Box, y: Box) -> i64 {
                if x.v == y.v { return 1 }
                return 0
            }",
        ),
        (
            "crate.main",
            "import provider;
            fn probe(b: Box) -> i64 {
                b.v = \"abc\"
                return provider.same_box(b, b)
            }",
        ),
    ];
    assert_eq!(project_refusals(&field, "crate.provider"), vec![UNTYPED]);
    // Positive control: the other module passes integers.
    let ints = [
        ("crate.provider", EQH),
        (
            "crate.main",
            "import provider;\nfn probe() -> i64 { return provider.eqh(7, 7) }",
        ),
    ];
    assert_eq!(project_refusals(&ints, "crate.provider"), vec![]);
}
