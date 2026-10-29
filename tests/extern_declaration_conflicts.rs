// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

#![cfg(feature = "std-surface")]

use libmind::{parser, type_checker};

const CONFLICT: &str = "safety::extern_conflicting_declaration";

fn diagnostics(source: &str) -> Vec<libmind::diagnostics::Diagnostic> {
    let module = parser::parse(source).expect("extern fixture must parse");
    type_checker::check_module_types(&module, source, &Default::default())
}

#[test]
fn contradictory_extern_abis_are_normal_source_diagnostics() {
    let cases = [
        (
            "return",
            "extern \"C\" { fn same() -> i64; fn same() -> i32; }",
        ),
        (
            "return-reversed",
            "extern \"C\" { fn same() -> i32; fn same() -> i64; }",
        ),
        (
            "parameter",
            "extern \"C\" { fn same(x: i64) -> i32; fn same(x: f64) -> i32; }",
        ),
        (
            "arity",
            "extern \"C\" { fn same() -> i32; fn same(x: i64) -> i32; }",
        ),
        (
            "varargs",
            "extern \"C\" { fn same(x: i64) -> i32; fn same(x: i64, ...) -> i32; }",
        ),
        (
            "callconv",
            "extern \"C\" callconv(.sysv) { fn same() -> i32; } extern \"C\" callconv(.win64) { fn same() -> i32; }",
        ),
        (
            "blocks",
            "extern \"C\" { fn same() -> i64; } extern \"C\" { fn same() -> i32; }",
        ),
        (
            "transparent-module",
            "extern \"C\" { fn same() -> i64; } module inner { extern \"C\" { fn same() -> i32; } }",
        ),
        (
            "third-declaration",
            "extern \"C\" { fn same() -> i64; fn same() -> i32; fn same() -> i64; }",
        ),
    ];
    for (name, source) in cases {
        let all = diagnostics(source);
        let conflicts: Vec<_> = all.iter().filter(|d| d.code == CONFLICT).collect();
        assert_eq!(conflicts.len(), 1, "{name}: {all:?}");
        assert!(conflicts[0].message.contains("same"), "{name}: {all:?}");
        assert!(conflicts[0].span.is_some(), "{name}: {all:?}");
    }
}

#[test]
fn identical_redeclarations_and_independent_owners_remain_valid() {
    for source in [
        "extern \"C\" { fn same(x: i64) -> i32; fn same(renamed: i64) -> i32; }",
        "extern \"C\" { fn same() -> i32; } extern \"C\" { fn same() -> i32; }",
        "extern \"C\" { fn same(x: i64, ...) -> i32; fn same(x: i64, ...) -> i32; }",
        "extern \"C\" { fn same() -> i32; } module inner { extern \"C\" { fn same() -> i32; } }",
        "extern \"C\" { fn same() -> i64; }",
        "extern \"C\" { fn same() -> i32; }",
    ] {
        assert!(diagnostics(source).is_empty(), "{source}");
    }
}

#[test]
fn default_and_explicit_host_c_conventions_are_consistent() {
    let host = if cfg!(target_os = "windows") {
        "win64"
    } else {
        "sysv"
    };
    let source = format!(
        "extern \"C\" {{ fn same() -> i32; }} extern \"C\" callconv(.{host}) {{ fn same() -> i32; }}"
    );
    assert!(diagnostics(&source).is_empty(), "{source}");
}

#[test]
fn nested_extern_conflicts_share_the_source_modules_abi_namespace() {
    let cases = [
        (
            "within-function",
            "fn main() -> i64 { extern \"C\" { fn same() -> i64; fn same() -> i32; } return 0; }",
        ),
        (
            "module-and-function",
            "extern \"C\" { fn same() -> i64; } fn main() -> i64 { extern \"C\" { fn same() -> i32; } return 0; }",
        ),
        (
            "separate-functions",
            "fn first() -> i64 { extern \"C\" { fn same() -> i64; } return 0; } fn second() -> i64 { extern \"C\" { fn same() -> i32; } return 0; }",
        ),
        (
            "if-branches",
            "fn main() -> i64 { if true { extern \"C\" { fn same() -> i64; } } else { extern \"C\" { fn same() -> i32; } } return 0; }",
        ),
        (
            "while-body",
            "extern \"C\" { fn same() -> i64; } fn main() -> i64 { while false { extern \"C\" { fn same() -> i32; } } return 0; }",
        ),
        (
            "for-body",
            "extern \"C\" { fn same() -> i64; } fn main() -> i64 { for i in 0..1 { extern \"C\" { fn same() -> i32; } } return 0; }",
        ),
        (
            "region-body",
            "extern \"C\" { fn same() -> i64; } fn main() -> i64 { region { extern \"C\" { fn same() -> i32; } } return 0; }",
        ),
    ];
    for (name, source) in cases {
        let all = diagnostics(source);
        let conflicts: Vec<_> = all.iter().filter(|d| d.code == CONFLICT).collect();
        assert_eq!(conflicts.len(), 1, "{name}: {all:?}");
        assert!(conflicts[0].span.is_some(), "{name}: {all:?}");
    }
}

#[test]
fn identical_nested_extern_declarations_remain_valid() {
    for source in [
        "fn main() -> i64 { extern \"C\" { fn same(x: i64) -> i32; fn same(renamed: i64) -> i32; } return 0; }",
        "extern \"C\" { fn same() -> i32; } fn first() -> i64 { extern \"C\" { fn same() -> i32; } return 0; } fn second() -> i64 { if true { extern \"C\" { fn same() -> i32; } } return 0; }",
    ] {
        assert!(diagnostics(source).is_empty(), "{source}");
    }
    let host = if cfg!(target_os = "windows") {
        "win64"
    } else {
        "sysv"
    };
    let source = format!(
        "extern \"C\" {{ fn same() -> i32; }} fn main() -> i64 {{ while false {{ extern \"C\" callconv(.{host}) {{ fn same() -> i32; }} }} return 0; }}"
    );
    assert!(diagnostics(&source).is_empty(), "{source}");
}

#[test]
fn nested_externs_refuse_unresolved_aliases_and_aggregate_layouts() {
    for source in [
        "type Alias = i32; fn main() -> i64 { extern \"C\" { fn same() -> Alias; } return 0; }",
        "fn main() -> i64 { type Alias = i32; extern \"C\" { fn same() -> Alias; } return 0; }",
        "#[repr(C)] struct Pair { x: i64, y: i64 } fn main() -> i64 { extern \"C\" { fn same(x: Pair) -> i32; } return 0; }",
        "fn main() -> i64 { extern \"C\" { fn same(x: String) -> i32; } return 0; }",
    ] {
        let all = diagnostics(source);
        assert!(
            all.iter().any(|d| d.code == "safety::extern_non_repr_c"),
            "nested extern type must not silently become i64: {source}: {all:?}"
        );
    }
}
