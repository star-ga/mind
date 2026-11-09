// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unit controls for strings stored where a declaration says otherwise, and
//! for the null string handle.

use super::*;

/// The refusal kinds `analyze` reports for `src`, in source order.
fn refusals(src: &str) -> Vec<RefusedKind> {
    let module = crate::parser::parse(src).unwrap();
    analyze(&module).refused.iter().map(|r| r.kind).collect()
}

/// The byte-wise comparisons `analyze` rewrites `src` to.
fn byte_compares(src: &str) -> usize {
    let module = crate::parser::parse(src).unwrap();
    let analysis = analyze(&module);
    assert!(analysis.refused.is_empty(), "{src}");
    let Some(rewritten) = analysis.rewritten else {
        return 0;
    };
    fn count(node: &Node) -> usize {
        let here = matches!(node, Node::Call { callee, .. } if callee == "__mind_string_eq");
        usize::from(here) + children_ref(node).into_iter().map(count).sum::<usize>()
    }
    rewritten.items.iter().map(count).sum()
}

const UNTYPED: RefusedKind = RefusedKind::Untyped;

/// A string stored as an element of a collection whose elements are declared
/// integers: by index assignment (also into a field's array and a nested
/// array), by `push` / `set`, and through a parameter alias. `==` on two
/// elements compiled to a handle compare.
#[test]
fn elements_given_strings_are_not_trusted() {
    let index = "fn p() -> i64 {
        let mut xs: [i64; 2] = [0, 0]
        xs[0] = \"abc\"
        if xs[0] == xs[1] { return 1 }
        return 0
    }";
    assert_eq!(refusals(index), vec![UNTYPED]);
    let push = "fn p() -> i64 {
        let mut xs: array<i64> = array<i64>.new()
        xs.push(\"abc\")
        if xs[0] == xs[1] { return 1 }
        return 0
    }";
    assert_eq!(refusals(push), vec![UNTYPED]);
    let set = "fn p() -> i64 {
        let mut xs: array<i64> = array<i64>.new()
        xs.set(0, \"abc\")
        if xs[0] == xs[1] { return 1 }
        return 0
    }";
    assert_eq!(refusals(set), vec![UNTYPED]);
    let field = "struct S { xs: [i64; 2] }
    fn p(s: S) -> i64 {
        s.xs[0] = \"abc\"
        if s.xs[0] == s.xs[1] { return 1 }
        return 0
    }";
    assert_eq!(refusals(field), vec![UNTYPED]);
    let alias = "fn fill(a: [i64; 2]) { a[0] = \"abc\" }
    fn p(xs: [i64; 2]) -> i64 {
        fill(xs)
        for x in xs { if x == xs[1] { return 1 } }
        return 0
    }";
    assert_eq!(refusals(alias), vec![UNTYPED]);
    let unknown_receiver = "fn fill<T>(a: T) { a[0] = \"abc\" }
    fn p(xs: [i64; 2]) -> i64 {
        if xs[0] == xs[1] { return 1 }
        return 0
    }";
    assert_eq!(refusals(unknown_receiver), vec![UNTYPED]);
    // Positive control: the same collections given integers.
    let ints = "fn p() -> i64 {
        let mut xs: [i64; 2] = [0, 0]
        xs[0] = 7
        xs.push(7)
        xs.set(1, 7)
        if xs[0] == xs[1] { return 1 }
        return 0
    }";
    assert_eq!(refusals(ints), vec![]);
}

/// A string handed to a callee whose body is not analysed here (a std
/// collection, raw memory) through an integer parameter and read back
/// through an integer result.
#[test]
fn strings_handed_to_opaque_callees_are_not_trusted() {
    let vec = "fn p() -> i64 {
        let v = vec_push(vec_new(), \"abc\")
        if vec_get(v, 0) == vec_get(v, 1) { return 1 }
        return 0
    }";
    assert_eq!(refusals(vec), vec![UNTYPED]);
    let raw = "fn p(a: i64) -> i64 {
        __mind_store_i64(a, \"abc\")
        if __mind_load_i64(a) == __mind_load_i64(a + 8) { return 1 }
        return 0
    }";
    assert_eq!(refusals(raw), vec![UNTYPED]);
    let wrapped = "fn put(a: i64, v: i64) { __mind_store_i64(a, v) }
    fn get(a: i64) -> i64 { return __mind_load_i64(a) }
    fn p(a: i64) -> i64 {
        put(a, \"abc\")
        if get(a) == get(a + 8) { return 1 }
        return 0
    }";
    assert_eq!(refusals(wrapped), vec![UNTYPED]);
    // Positive controls: integers handed over; a byte store keeps no
    // handle; a record the string was not handed over with.
    let ints = "fn p(a: i64, s: string) -> i64 {
        let v = vec_push(vec_new(), 7)
        __mind_store_i8(a, s)
        if vec_get(v, 0) == vec_get(v, 1) { return 1 }
        if __mind_load_i64(a) == __mind_load_i64(a + 8) { return 2 }
        return 0
    }";
    assert_eq!(refusals(ints), vec![]);
    let other_record = "fn p(a: string, b: string) -> i64 {
        let v = vec_push(vec_new(), \"abc\")
        if string_len(a) == string_len(b) { return 1 }
        return 0
    }";
    assert_eq!(refusals(other_record), vec![]);
}

/// The payload of an `Option` / `Result` has the fact its declaration (or
/// its constructor's argument) gives it.
#[test]
fn option_and_result_payloads_are_typed() {
    let into_param = "fn eqh(a: i64, b: i64) -> i64 {
        if a == b { return 1 }
        return 0
    }
    fn get() -> Option<string> { return Some(\"abc\") }
    fn p() -> i64 {
        match get() {
            Some(x) => { return eqh(x, x) },
            None => { return 2 },
        }
    }";
    assert_eq!(refusals(into_param), vec![UNTYPED]);
    let built = "fn eqh(a: i64, b: i64) -> i64 {
        if a == b { return 1 }
        return 0
    }
    fn p() -> i64 {
        let r: Result<i64, i64> = Ok(\"abc\")
        match r {
            Ok(x) => { return eqh(x, x) },
            Err(e) => { return 2 },
        }
    }";
    assert_eq!(refusals(built), vec![UNTYPED]);
    let string_payloads = "fn get() -> Option<string> { return Some(\"abc\") }
    fn p() -> i64 {
        match get() {
            Some(x) => { if x == \"abc\" { return 1 } return 0 },
            None => { return 2 },
        }
    }";
    assert_eq!(byte_compares(string_payloads), 1);
    // Positive control: integer payloads stay integers.
    let ints = "fn eqh(a: i64, b: i64) -> i64 {
        if a == b { return 1 }
        return 0
    }
    fn get() -> Option<i64> { return Some(7) }
    fn p() -> i64 {
        match get() {
            Some(x) => { return eqh(x, x) },
            None => { return 2 },
        }
    }";
    assert_eq!(refusals(ints), vec![]);
}

/// A `Some(x)` payload of an `Option<i64>` constructed with a string: the
/// result type no longer proves the payload an integer.
#[test]
fn a_string_payload_returned_as_an_integer_option_is_not_trusted() {
    let src = "fn get() -> Option<i64> { return Some(\"abc\") }
    fn p() -> i64 {
        match get() {
            Some(x) => { match get() { Some(y) => { if x == y { return 1 } return 0 }, None => { return 3 } } },
            None => { return 2 },
        }
    }";
    assert_eq!(refusals(src), vec![UNTYPED]);
}

/// A value of a type parameter, or the result of a generic function, may be
/// a string: storing it in an integer slot does not keep the slot an integer.
#[test]
fn type_parameter_values_may_be_strings() {
    let result = "fn id<T>(x: T) -> T { return x }
    fn eqh(a: i64, b: i64) -> i64 {
        if a == b { return 1 }
        return 0
    }
    fn p() -> i64 { return eqh(id(\"abc\"), id(\"abc\")) }";
    assert_eq!(refusals(result), vec![UNTYPED]);
    let param = "fn keep<T>(x: T) -> i64 { return x }
    fn p() -> i64 {
        if keep(\"abc\") == keep(\"abc\") { return 1 }
        return 0
    }";
    assert_eq!(refusals(param), vec![UNTYPED]);
}

/// An integer other than the null handle `0` in a `string` slot is no string
/// record: comparing it with text is refused. The null handle is a defined
/// `string` value, so `same(0, "abc")` still compares bytes (the null handle
/// equals only itself).
#[test]
fn integers_in_string_slots_are_refused_and_the_null_handle_is_not() {
    let integer = "fn is_abc(s: string) -> i64 {
        if s == \"abc\" { return 1 }
        return 0
    }
    fn h(n: i64) -> i64 { return is_abc(n) }";
    assert_eq!(refusals(integer), vec![RefusedKind::Mixed]);
    let binding = "fn p(n: i64) -> i64 {
        let s: string = n
        if s == \"abc\" { return 1 }
        return 0
    }";
    assert_eq!(refusals(binding), vec![RefusedKind::Mixed]);
    let null = "struct D { name: string }
    fn same(a: string, b: string) -> i64 {
        if a == b { return 1 }
        return 0
    }
    fn find(c: i64) -> string {
        if c == 1 { return \"abc\" }
        return 0
    }
    fn p() -> i64 {
        let d = D { name: 0 }
        if d.name == \"abc\" { return 1 }
        if find(0) == \"abc\" { return 2 }
        return same(0, \"abc\")
    }";
    assert_eq!(byte_compares(null), 3);
}

/// A module-level binding assigned a string in a function, and a field
/// assigned a string through a receiver whose struct is not known.
#[test]
fn global_and_unknown_receiver_stores_are_not_trusted() {
    let global = "let mut g = 0
    let mut h = 0
    fn p() -> i64 {
        g = \"abc\"
        h = \"abc\"
        if g == h { return 1 }
        return 0
    }";
    assert_eq!(refusals(global), vec![UNTYPED]);
    let field = "struct Box { v: i64 }
    fn set_v<T>(b: T) { b.v = \"abc\" }
    fn p(a: Box, b: Box) -> i64 {
        if a.v == b.v { return 1 }
        return 0
    }";
    assert_eq!(refusals(field), vec![UNTYPED]);
}

/// A literal of the std `String` record is a string value: returning one
/// through a `-> String` result does not store a non-string there.
#[test]
fn a_string_record_literal_is_a_string() {
    let src = "fn mk() -> String { return String { addr: 0, len: 0, cap: 0 } }
    fn p(a: string) -> i64 {
        if mk() == a { return 1 }
        return 0
    }";
    assert_eq!(byte_compares(src), 1);
}
