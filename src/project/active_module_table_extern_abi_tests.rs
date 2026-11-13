// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! C ABI owner-selection controls for `visible_externs` and the imported
//! extern declarations it feeds.

use super::*;
use crate::ast::CallConv;
use crate::project::module_table::build_module_table;

fn selected(
    provider: &str,
    sibling: Option<&str>,
    captured: bool,
    owner: &str,
) -> Vec<super::super::module_table::ExportedExtern> {
    let provider = crate::parser::parse(provider).expect("provider fixture");
    let sibling = sibling.map(|s| crate::parser::parse(s).expect("sibling fixture"));
    let mut modules = vec![("crate.ffi".to_string(), &provider)];
    if let Some(sibling) = &sibling {
        modules.push(("crate.other".to_string(), sibling));
    }
    let _table = Guard::install(build_module_table(&modules));
    let _imports = if captured {
        ResolvedImportsGuard::install(BTreeMap::from([(
            (owner.to_string(), vec!["ffi".to_string()]),
            "crate.ffi".to_string(),
        )]))
    } else {
        ResolvedImportsGuard {
            previous: ACTIVE_RESOLVED_IMPORTS.with(|cell| cell.replace(None)),
        }
    };
    visible_externs(Some(owner))
}

#[test]
fn one_owner_conflicts_remain_ambiguous_in_every_scope() {
    let conflicting = "extern \"C\" { fn same() -> i64; fn same() -> i32; }";
    for owner in ["crate.user", "crate.ffi"] {
        for captured in [false, true] {
            assert!(
                selected(conflicting, None, captured, owner).is_empty(),
                "owner={owner}, captured={captured}"
            );
        }
    }
}

#[test]
fn identical_owner_declarations_and_unimported_conflicts_are_stable() {
    let same = "extern \"C\" { fn same(x: i64) -> i32; fn same(other: i64) -> i32; }";
    let unrelated = "extern \"C\" { fn same(x: i64) -> i64; }";
    let expected = selected(same, None, true, "crate.user");
    assert_eq!(expected.len(), 1);
    assert_eq!(
        selected(same, Some(unrelated), true, "crate.user"),
        expected
    );
    assert_eq!(selected(same, Some(same), false, "crate.user"), expected);
    assert!(selected(same, Some(unrelated), false, "crate.user").is_empty());
}

#[test]
fn exported_extern_default_convention_matches_explicit_host_convention() {
    let host = if cfg!(target_os = "windows") {
        "win64"
    } else {
        "sysv"
    };
    let source = format!(
        "extern \"C\" {{ fn same() -> i32; }} extern \"C\" callconv(.{host}) {{ fn same() -> i32; }}"
    );
    let found = selected(&source, None, true, "crate.user");
    assert_eq!(found.len(), 1);
    let expected = if cfg!(target_os = "windows") {
        CallConv::Win64
    } else {
        CallConv::SysV
    };
    assert_eq!(found[0].callconv, expected);
}

fn extern_decl(module: &crate::ast::Module) -> (Vec<String>, Option<String>, CallConv) {
    let ir = crate::eval::lower::lower_to_ir(module).expect("declaration lowering");
    let found: Vec<_> = ir
        .instrs
        .into_iter()
        .filter_map(|instr| match instr {
            crate::ir::Instr::ExternFnDecl {
                name,
                param_types,
                ret_type,
                callconv,
                ..
            } if name == "same" => Some((param_types, ret_type, callconv)),
            _ => None,
        })
        .collect();
    assert_eq!(found.len(), 1, "one resolved declaration required");
    found.into_iter().next().unwrap()
}

#[test]
fn imported_declarations_preserve_local_target_and_explicit_conventions() {
    for (annotation, expected) in [
        (
            "",
            if cfg!(target_os = "windows") {
                CallConv::Win64
            } else {
                CallConv::SysV
            },
        ),
        ("callconv(.sysv)", CallConv::SysV),
        ("callconv(.win64)", CallConv::Win64),
        ("callconv(.aapcs)", CallConv::Aapcs),
    ] {
        let provider = crate::parser::parse(&format!(
            "extern \"C\" {annotation} {{ fn same(x: i64) -> i32; }}"
        ))
        .unwrap();
        let local = extern_decl(&provider);
        assert_eq!(local.2, expected, "local {annotation}");
        let _table = Guard::install(build_module_table(&[("crate.ffi".into(), &provider)]));
        let _owner = crate::qualified_enums::ModuleGuard::install("crate.user");
        let _imports = ResolvedImportsGuard::install(BTreeMap::from([(
            ("crate.user".to_string(), vec!["ffi".to_string()]),
            "crate.ffi".to_string(),
        )]));
        let consumer =
            crate::parser::parse("import ffi; pub fn use_same() -> i64 { return same(1); }")
                .unwrap();
        assert_eq!(extern_decl(&consumer), local, "imported {annotation}");
    }
}

#[test]
fn uncalled_import_adds_no_extern_decl() {
    let provider = crate::parser::parse("extern \"C\" { fn same(x: i64) -> i32; }").unwrap();
    let _table = Guard::install(build_module_table(&[("crate.ffi".into(), &provider)]));
    let _owner = crate::qualified_enums::ModuleGuard::install("crate.user");
    let _imports = ResolvedImportsGuard::install(BTreeMap::from([(
        ("crate.user".to_string(), vec!["ffi".to_string()]),
        "crate.ffi".to_string(),
    )]));
    let consumer = crate::parser::parse("import ffi;").unwrap();
    let ir = crate::eval::lower::lower_to_ir(&consumer).expect("lower import");
    let decls = ir.instrs.iter().filter(
        |instr| matches!(instr, crate::ir::Instr::ExternFnDecl { name, .. } if name == "same"),
    );
    assert_eq!(
        decls.count(),
        0,
        "an uncalled import must not declare the extern"
    );
}

#[test]
fn owner_extern_is_not_declared_twice() {
    let provider = crate::parser::parse(
        "extern \"C\" { fn same(x: i64) -> i32; } pub fn use_same() -> i64 { return same(1); }",
    )
    .unwrap();
    let _table = Guard::install(build_module_table(&[("crate.ffi".into(), &provider)]));
    let _owner = crate::qualified_enums::ModuleGuard::install("crate.ffi");
    let _imports = ResolvedImportsGuard::install(BTreeMap::new());
    let found = extern_decl(&provider);
    assert_eq!(found.0, vec!["i64".to_string()]);
}

#[test]
fn user_fn_close_stays_visible_beside_std_fs_extern() {
    let fs = crate::parser::parse(
        "extern \"C\" { fn close(fd: i32) -> i32; fn open(p: *const u8, flags: i32, mode: i32) -> i32; }",
    )
    .unwrap();
    let user =
        crate::parser::parse("import std.fs; pub fn close(fd: i64) -> i64 { return fd; }").unwrap();
    let _table = Guard::install(build_module_table(&[
        ("std.fs".into(), &fs),
        ("crate.user".into(), &user),
    ]));
    let _imports = ResolvedImportsGuard::install(BTreeMap::from([
        (
            ("crate.user".to_string(), vec!["fs".to_string()]),
            "std.fs".to_string(),
        ),
        (
            ("crate.consumer".to_string(), vec!["user".to_string()]),
            "crate.user".to_string(),
        ),
        (
            ("crate.consumer".to_string(), vec!["fs".to_string()]),
            "std.fs".to_string(),
        ),
    ]));
    let symbols = visible_symbols(Some("crate.consumer"));
    assert!(
        symbols.contains("close"),
        "pub fn close must not become ambiguous beside an imported extern"
    );
    assert!(
        symbols.contains("open"),
        "a unique extern must stay visible"
    );
    let resolved = lookup_visible_fn(Some("crate.consumer"), "close").expect("user fn");
    assert_eq!(resolved.ret_type, Some(crate::ast::TypeAnn::ScalarI64));
    let local = lookup_visible_fn(Some("crate.user"), "close").expect("owner fn");
    assert_eq!(local.ret_type, Some(crate::ast::TypeAnn::ScalarI64));
}
