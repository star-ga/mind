// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

#![cfg(feature = "std-surface")]

use libmind::target::{CompileTargetGuard, Target};
use libmind::{parser, type_checker};

const CONFLICT: &str = "safety::extern_conflicting_declaration";

/// The explicit tag of each target's native C convention, then the tags of a
/// different ABI (RFC 0010 section 4.1, docs/errors.md E2039): `.sysv` on
/// x86-64 Linux and macOS, `.win64` on x86-64 Windows, `.aapcs` on aarch64.
const NATIVE_C: [(&str, &str, [&str; 2]); 5] = [
    ("x86_64-unknown-linux-gnu", "sysv", ["win64", "aapcs"]),
    ("x86_64-apple-darwin", "sysv", ["win64", "aapcs"]),
    ("x86_64-w64-windows-gnu", "win64", ["sysv", "aapcs"]),
    ("aarch64-unknown-linux-gnu", "aapcs", ["sysv", "win64"]),
    ("aarch64-apple-darwin", "aapcs", ["sysv", "win64"]),
];

/// The native C tag of the host that runs this test (the default target).
fn host_native_c() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "aapcs"
    } else if cfg!(target_os = "windows") {
        "win64"
    } else {
        "sysv"
    }
}

fn top_level_pair(tag: &str) -> String {
    format!(
        "extern \"C\" {{ fn same() -> i32; }} extern \"C\" callconv(.{tag}) {{ fn same() -> i32; }}"
    )
}

fn nested_pair(tag: &str) -> String {
    format!(
        "extern \"C\" {{ fn same() -> i32; }} fn main() -> i64 {{ while false {{ extern \"C\" callconv(.{tag}) {{ fn same() -> i32; }} }} return 0; }}"
    )
}

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
    let source = top_level_pair(host_native_c());
    assert!(diagnostics(&source).is_empty(), "{source}");
}

/// Each compile target, from any build host: `extern "C"` and the explicit tag
/// of the target's native C convention are one ABI; a tag of another ABI is a
/// conflict. aarch64 records the SysV byte for `extern "C"` and lowers it as
/// AAPCS64, so the conflict check must compare the lowered convention.
#[test]
fn default_c_matches_the_native_tag_of_each_compile_target() {
    let mut failures = Vec::new();
    for (triple, native, foreign) in NATIVE_C {
        let _target = CompileTargetGuard::install(Target::from_triple(triple).expect(triple));
        let rows = [(native, true)]
            .into_iter()
            .chain(foreign.map(|tag| (tag, false)));
        for (tag, same_abi) in rows {
            for source in [top_level_pair(tag), nested_pair(tag)] {
                let all = diagnostics(&source);
                let conflicts = all.iter().filter(|d| d.code == CONFLICT).count();
                let ok = if same_abi {
                    all.is_empty()
                } else {
                    conflicts == 1
                };
                if !ok {
                    failures.push(format!(
                        "{triple} .{tag} same_abi={same_abi}: {source}: {all:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
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
    let source = nested_pair(host_native_c());
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
