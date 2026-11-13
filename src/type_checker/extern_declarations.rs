// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

use super::{check_extern_type_with_repr_c, diag_from_span, nerve_walk};
use crate::ast::{CallConv, ExternBinding, Node, TypeAnn};
use crate::diagnostics::Diagnostic;
use crate::eval::type_aliases::LocalTypeAliases;
use std::collections::{BTreeMap, BTreeSet};

type Signature = (CallConv, bool, Vec<TypeAnn>, Option<TypeAnn>);

/// A source module owns one extern namespace: lowering hoists declarations
/// from function and control-flow bodies into the same foreign-symbol table.
/// Repeated declarations may rename parameters, but must agree on the complete
/// ABI. A separate public checker call creates a separate source-owner table.
pub(super) fn check(items: &[Node], src: &str, file: Option<&str>, errs: &mut Vec<Diagnostic>) {
    let aliases = LocalTypeAliases::new(items);
    let mut seen = BTreeMap::new();
    fn visit<'a>(
        item: &'a Node,
        scope: (&LocalTypeAliases, bool),
        seen: &mut BTreeMap<&'a str, Signature>,
        src: &str,
        file: Option<&str>,
        errs: &mut Vec<Diagnostic>,
    ) {
        let (aliases, nested) = scope;
        if let Node::ExternBlock { callconv, fns, .. } = item {
            for function in fns {
                refuse_unrealized_callconv(*callconv, function, (src, file), errs);
                if nested {
                    // Nested lowering consumes raw annotations and has no
                    // reviewed alias/aggregate-layout ownership contract.
                    // Reuse the existing C-type admission rule; never let an
                    // unknown Named/aggregate annotation fall back to i64.
                    let no_layouts = BTreeSet::new();
                    for param in &function.params {
                        if let Err(reason) = check_extern_type_with_repr_c(&param.ty, &no_layouts) {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "nested extern `{}` parameter `{}`: {reason}; move layout-dependent declarations to module scope",
                                    function.name, param.name
                                ),
                                param.span,
                                "safety::extern_non_repr_c",
                            ));
                        }
                    }
                    if let Some(ret) = &function.ret_type {
                        if let Err(reason) = check_extern_type_with_repr_c(ret, &no_layouts) {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "nested extern `{}` return type: {reason}; move layout-dependent declarations to module scope",
                                    function.name
                                ),
                                function.span,
                                "safety::extern_non_repr_c",
                            ));
                        }
                    }
                }
                let canonical_type = |ty: &TypeAnn| {
                    if nested {
                        ty.clone()
                    } else {
                        aliases.resolve(ty)
                    }
                };
                let signature = (
                    crate::target::compile_target().lowered_callconv(*callconv),
                    function.is_varargs,
                    function
                        .params
                        .iter()
                        .map(|param| canonical_type(&param.ty))
                        .collect(),
                    function.ret_type.as_ref().map(canonical_type),
                );
                if let Some(previous) = seen.get(function.name.as_str()) {
                    if previous != &signature {
                        errs.push(diag_from_span(
                                    src,
                                    file,
                                    format!(
                                        "conflicting extern declaration `{}` in the same source module; parameter types, return type, variadic status and calling convention must agree",
                                        function.name
                                    ),
                                    function.span,
                                    "safety::extern_conflicting_declaration",
                                ));
                    }
                } else {
                    seen.insert(function.name.as_str(), signature);
                }
            }
        }
        // Only a root-level transparent module block preserves declaration
        // scope. Functions and executable expressions share symbol ownership
        // but must not inherit unimplemented nested alias/layout semantics.
        let child_nested = nested || !matches!(item, Node::Block { .. });
        nerve_walk::for_each_child(item, &mut |child| {
            visit(child, (aliases, child_nested), seen, src, file, errs)
        });
    }
    for item in items {
        visit(item, (&aliases, false), &mut seen, src, file, errs);
    }
    refuse_by_value_structs(items, &aliases, (src, file), errs);
    check_bindings(items, (src, file), errs);
    #[cfg(feature = "cross-module-imports")]
    refuse_non_primitive_imports(src, file, errs, &seen);
}

/// A declared convention with no lowering row on the compile target.
pub(crate) const EXTERN_CALLCONV_ROW_CODE: &str = "E2039";

/// Refuse a convention the compile target cannot lower (AAPCS on x86-64,
/// Win64 varargs, SysV on Windows, SysV/Win64 on aarch64). Lowering it with
/// another convention would read the wrong registers.
fn refuse_unrealized_callconv(
    callconv: CallConv,
    function: &crate::ast::ExternFn,
    (src, file): (&str, Option<&str>),
    errs: &mut Vec<Diagnostic>,
) {
    let target = crate::target::compile_target();
    if target.realizes_callconv(callconv, function.is_varargs) {
        return;
    }
    let spelling = match callconv {
        CallConv::C => "C",
        CallConv::SysV => "callconv(.sysv)",
        CallConv::Win64 => "callconv(.win64)",
        CallConv::Aapcs => "callconv(.aapcs)",
    };
    let varargs = if function.is_varargs {
        " with varargs"
    } else {
        ""
    };
    errs.push(diag_from_span(
        src,
        file,
        format!(
            "extern fn `{}`: {spelling}{varargs} has no ABI row on target `{}`; \
             it is refused, never lowered with another convention",
            function.name,
            target.llvm_triple()
        ),
        function.span,
        EXTERN_CALLCONV_ROW_CODE,
    ));
}

/// A by-value `#[repr(C)]` struct parameter or return in an extern signature.
pub(crate) const EXTERN_BY_VALUE_STRUCT_CODE: &str = "E2040";

/// Refuse each by-value `#[repr(C)]` struct in a module-scope extern
/// signature, callback types included. The x86-64 classifiers give wrong
/// values for some layouts (`{f32, f32}` passes one f32), so 0.10.3 refuses
/// the whole set; a pointer to the struct stays accepted. Nested blocks are
/// refused by the layout rule above.
fn refuse_by_value_structs(
    items: &[Node],
    aliases: &LocalTypeAliases,
    (src, file): (&str, Option<&str>),
    errs: &mut Vec<Diagnostic>,
) {
    let mut repr_c = BTreeSet::new();
    for item in items {
        nerve_walk::walk(item, &mut |node| {
            if let Node::StructDef { name, attrs, .. } = node
                && attrs
                    .iter()
                    .any(|a| a.name == "repr" && a.args.iter().any(|arg| arg == "C"))
            {
                repr_c.insert(name.clone());
            }
        });
    }
    fn by_value(ty: &TypeAnn, repr_c: &BTreeSet<String>) -> Option<String> {
        match ty {
            TypeAnn::Named(name) if repr_c.contains(name) => Some(name.clone()),
            TypeAnn::FnPtr { params, ret } => params
                .iter()
                .chain(ret.as_deref())
                .find_map(|t| by_value(t, repr_c)),
            _ => None,
        }
    }
    fn scan(
        item: &Node,
        ctx: (&LocalTypeAliases, &BTreeSet<String>),
        at: (&str, Option<&str>),
        errs: &mut Vec<Diagnostic>,
    ) {
        let (aliases, repr_c) = ctx;
        match item {
            Node::Block { stmts, .. } => stmts.iter().for_each(|s| scan(s, ctx, at, errs)),
            Node::ExternBlock { fns, .. } => {
                for f in fns {
                    let slots = f
                        .params
                        .iter()
                        .map(|p| (format!("parameter `{}`", p.name), &p.ty));
                    let ret = f.ret_type.iter().map(|t| ("return type".to_string(), t));
                    for (slot, ty) in slots.chain(ret) {
                        let Some(name) = by_value(&aliases.resolve(ty), repr_c) else {
                            continue;
                        };
                        errs.push(diag_from_span(
                            at.0,
                            at.1,
                            format!(
                                "extern fn `{}` {slot}: by-value #[repr(C)] struct `{name}` is \
                                 refused in 0.10.3; pass a pointer `*const {name}` or `*mut {name}`",
                                f.name
                            ),
                            f.span,
                            EXTERN_BY_VALUE_STRUCT_CODE,
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    if repr_c.is_empty() {
        return;
    }
    for item in items {
        scan(item, (aliases, &repr_c), (src, file), errs);
    }
}

use crate::project::link::EXTERN_BINDING_CODE;

/// Why `bind("<name>")` is refused, or `None` when the name is a valid loader
/// name that is not in the base set (rules B2 and B4).
pub(crate) fn library_name_refusal(name: &str) -> Option<String> {
    let bytes = name.as_bytes();
    let charset = |b: &u8| b.is_ascii_alphanumeric() || b".+_-".contains(b);
    if bytes.is_empty() || bytes.len() > 255 || !bytes.iter().all(charset) {
        return Some(format!(
            "`bind(\"{name}\")`: a library name is 1 to 255 bytes from `A-Z a-z 0-9 . _ + -`"
        ));
    }
    if !bytes[0].is_ascii_alphanumeric() {
        return Some(format!(
            "`bind(\"{name}\")`: a library name starts with a letter or a digit"
        ));
    }
    if crate::project::link::BASE_LIBRARIES
        .iter()
        .any(|(base, _)| *base == name)
    {
        return Some(format!(
            "`bind(\"{name}\")`: `{name}` is in the base set; write the block with no `bind` clause"
        ));
    }
    None
}

/// Check every `bind` clause of the module and every call through an address
/// binding: a bad or base-set library name, varargs in a `bind(.address)` block,
/// one name with two bindings (E2045), and an address call whose operand count
/// is not the declared parameter count plus one (E2005).
fn check_bindings(items: &[Node], (src, file): (&str, Option<&str>), errs: &mut Vec<Diagnostic>) {
    let mut bindings: BTreeMap<&str, &ExternBinding> = BTreeMap::new();
    let mut address_arity: BTreeMap<&str, usize> = BTreeMap::new();
    let mut refuse =
        |msg: String, span, code| errs.push(diag_from_span(src, file, msg, span, code));
    for item in items {
        nerve_walk::walk(item, &mut |node| {
            let Node::ExternBlock {
                binding, fns, span, ..
            } = node
            else {
                return;
            };
            if let ExternBinding::Library(name) = binding
                && let Some(msg) = library_name_refusal(name)
            {
                refuse(msg, *span, EXTERN_BINDING_CODE);
            }
            for f in fns {
                if *binding == ExternBinding::Address && f.is_varargs {
                    let msg = format!(
                        "extern fn `{}`: varargs in a `bind(.address)` block",
                        f.name
                    );
                    refuse(msg, f.span, EXTERN_BINDING_CODE);
                }
                if *bindings.entry(f.name.as_str()).or_insert(binding) != binding {
                    let msg = format!("extern fn `{}` has two different bindings", f.name);
                    refuse(msg, f.span, EXTERN_BINDING_CODE);
                }
                if *binding == ExternBinding::Address {
                    address_arity.insert(f.name.as_str(), f.params.len());
                }
            }
        });
    }
    for item in items {
        nerve_walk::walk(item, &mut |node| {
            if let Node::Call { callee, args, span } = node
                && let Some(n) = address_arity.get(callee.as_str())
                && args.len() != n + 1
            {
                let msg = format!(
                    "`{callee}` is bound by address: a call passes the code address, then {n} \
                     argument(s); got {} operand(s)",
                    args.len()
                );
                refuse(msg, *span, super::CALL_ARITY_CODE);
            }
        });
    }
    #[cfg(feature = "cross-module-imports")]
    {
        let owner = crate::qualified_enums::current_module_path();
        for ext in crate::project::active_module_table::visible_externs(owner.as_deref()) {
            if bindings
                .get(ext.name.as_str())
                .is_some_and(|b| **b != ext.binding)
            {
                let msg = format!("extern fn `{}` has two different bindings", ext.name);
                refuse(msg, crate::ast::Span::new(0, 0), EXTERN_BINDING_CODE);
            }
        }
    }
}

/// An imported extern may not name a struct or other non-primitive type.
/// The importer does not own that layout, so classifying it would be a guess.
#[cfg(feature = "cross-module-imports")]
pub(crate) fn imported_type_is_primitive(ty: &TypeAnn) -> bool {
    check_extern_type_with_repr_c(ty, &BTreeSet::new()).is_ok()
}

#[cfg(feature = "cross-module-imports")]
fn refuse_non_primitive_imports(
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
    local_names: &BTreeMap<&str, Signature>,
) {
    let owner = crate::qualified_enums::current_module_path();
    for ext in crate::project::active_module_table::visible_externs(owner.as_deref()) {
        // A whole-project check exposes every extern to every module; only a
        // module that names the extern can be importing it.
        if local_names.contains_key(ext.name.as_str()) || !src.contains(ext.name.as_str()) {
            continue;
        }
        let mut bad = Vec::new();
        for ty in ext.param_types.iter().chain(ext.ret_type.as_ref()) {
            if !imported_type_is_primitive(ty) {
                bad.push(match ty {
                    TypeAnn::Named(name) => name.clone(),
                    other => format!("{other:?}"),
                });
            }
        }
        if bad.is_empty() {
            continue;
        }
        errs.push(diag_from_span(
            src,
            file,
            format!(
                "imported extern `{}` names a non-primitive type ({}); the imported C signature is refused",
                ext.name,
                bad.join(", ")
            ),
            crate::ast::Span::new(0, 0),
            "safety::imported_extern_non_primitive",
        ));
    }
}

#[cfg(all(test, feature = "cross-module-imports"))]
mod imported_primitive_tests {
    use super::check;
    use crate::project::active_module_table::{Guard, ResolvedImportsGuard};
    use crate::project::module_table::build_module_table;
    use std::collections::BTreeMap;

    #[test]
    fn imported_extern_refuses_a_non_primitive_type() {
        let provider = crate::parser::parse(
            "#[repr(C)] struct Pair { x: i64, y: i64 } extern \"C\" { fn take(p: Pair) -> i64; fn plain(n: i64) -> i64; }",
        )
        .unwrap();
        let importer_src = "import ffi;\npub fn go() -> i64 { return take(plain(1)); }\n";
        let importer = crate::parser::parse(importer_src).unwrap();
        let _table = Guard::install(build_module_table(&[
            ("crate.ffi".into(), &provider),
            ("crate.user".into(), &importer),
        ]));
        let _owner = crate::qualified_enums::ModuleGuard::install("crate.user");
        let _imports = ResolvedImportsGuard::install(BTreeMap::from([(
            ("crate.user".to_string(), vec!["ffi".to_string()]),
            "crate.ffi".to_string(),
        )]));
        let mut diags = Vec::new();
        check(&importer.items, importer_src, Some("user.mind"), &mut diags);
        let hits: Vec<_> = diags
            .iter()
            .filter(|d| d.code == "safety::imported_extern_non_primitive")
            .collect();
        assert_eq!(hits.len(), 1, "{diags:?}");
        assert!(hits[0].message.contains("take"), "{diags:?}");
        assert!(hits[0].message.contains("Pair"), "{diags:?}");
        let ir = crate::eval::lower::lower_to_ir(&importer).expect("lower");
        assert!(
            ir.instrs.iter().any(|instr| {
                matches!(instr, crate::ir::Instr::ExternFnDecl { name, .. } if name == "plain")
            }),
            "a called primitive import must still be classified"
        );
        assert!(
            ir.instrs.iter().all(|instr| {
                !matches!(instr, crate::ir::Instr::ExternFnDecl { name, .. } if name == "take")
            }),
            "a refused imported extern must not be classified"
        );
    }

    #[test]
    fn a_module_that_never_names_a_non_primitive_extern_is_not_refused() {
        let provider = crate::parser::parse(
            "#[repr(C)] struct Pair { x: i64, y: i64 } extern \"C\" { fn take(p: Pair) -> i64; }",
        )
        .unwrap();
        let other_src = "pub fn go() -> i64 { return 1; }\n";
        let other = crate::parser::parse(other_src).unwrap();
        let _table = Guard::install(build_module_table(&[
            ("crate.ffi".into(), &provider),
            ("crate.user".into(), &other),
        ]));
        let _owner = crate::qualified_enums::ModuleGuard::install("crate.user");
        let mut diags = Vec::new();
        check(&other.items, other_src, Some("user.mind"), &mut diags);
        assert!(
            diags
                .iter()
                .all(|d| d.code != "safety::imported_extern_non_primitive"),
            "{diags:?}"
        );
    }
}

#[cfg(test)]
mod binding_tests {
    use crate::ast::{ExternBinding, Node};

    fn codes(src: &str) -> Vec<String> {
        let module = crate::parser::parse(src).expect("parse");
        let env = crate::type_checker::TypeEnv::default();
        crate::type_checker::check_module_types_in_file(&module, src, None, &env)
            .into_iter()
            .map(|d| format!("{} {}", d.code, d.message))
            .collect()
    }

    fn binding_of(src: &str) -> ExternBinding {
        let module = crate::parser::parse(src).expect("parse");
        match &module.items[0] {
            Node::ExternBlock { binding, .. } => binding.clone(),
            other => panic!("not an extern block: {other:?}"),
        }
    }

    #[test]
    fn the_parser_reads_both_clauses_after_callconv() {
        let lib = "extern \"C\" bind(\"libz.so.1\") { fn crc32(c: u64) -> u64; }";
        assert_eq!(binding_of(lib), ExternBinding::Library("libz.so.1".into()));
        let addr = "extern \"C\" callconv(.sysv) bind(.address) { fn f(x: i32) -> i32; }";
        assert_eq!(binding_of(addr), ExternBinding::Address);
        assert_eq!(
            binding_of("extern \"C\" { fn f(x: i32) -> i32; }"),
            ExternBinding::Base
        );
        let reversed = "extern \"C\" bind(.address) callconv(.sysv) { fn f(x: i32) -> i32; }";
        assert!(
            crate::parser::parse(reversed).is_err(),
            "callconv comes first"
        );
        assert!(crate::parser::parse("extern \"C\" bind(.other) { }").is_err());
    }

    #[test]
    fn each_refused_binding_reports_e2043() {
        let refused = [
            (
                "extern \"C\" bind(\"lib/z.so\") { fn f(x: i64) -> i64; }",
                "1 to 255 bytes",
            ),
            (
                "extern \"C\" bind(\"\") { fn f(x: i64) -> i64; }",
                "1 to 255 bytes",
            ),
            (
                "extern \"C\" bind(\"-lz\") { fn f(x: i64) -> i64; }",
                "starts with a letter",
            ),
            (
                "extern \"C\" bind(\"libc.so.6\") { fn f(x: i64) -> i64; }",
                "base set",
            ),
            (
                "extern \"C\" bind(\"libm.so.6\") { fn f(x: i64) -> i64; }",
                "base set",
            ),
            (
                "extern \"C\" bind(.address) { fn f(x: i64, ...) -> i64; }",
                "varargs",
            ),
            (
                "extern \"C\" { fn f(x: i64) -> i64; }\nextern \"C\" bind(\"libz.so.1\") { fn f(x: i64) -> i64; }",
                "two different bindings",
            ),
        ];
        for (src, why) in refused {
            let got = codes(src);
            assert!(
                got.iter()
                    .any(|d| d.starts_with("E2045") && d.contains(why)),
                "{src}: want E2045 `{why}`, got {got:?}"
            );
        }
        let long = format!(
            "extern \"C\" bind(\"{}\") {{ fn f(x: i64) -> i64; }}",
            "a".repeat(256)
        );
        assert!(codes(&long).iter().any(|d| d.starts_with("E2045")));
    }

    #[test]
    fn valid_bindings_report_nothing() {
        let ok = "extern \"C\" bind(\"libz.so.1\") { fn crc32(c: u64, p: *const u8, n: u32) -> u64; }\n\
                  extern \"C\" bind(.address) { fn add2(a: i64, b: i64) -> i64; }\n\
                  fn main() -> i64 { let p: i64 = 0; return add2(p, 1, 2); }";
        let got = codes(ok);
        assert!(
            !got.iter()
                .any(|d| d.starts_with("E2045") || d.starts_with("E2005")),
            "{got:?}"
        );
        let name255 = format!(
            "extern \"C\" bind(\"{}\") {{ fn f(x: i64) -> i64; }}",
            "a".repeat(255)
        );
        assert!(!codes(&name255).iter().any(|d| d.starts_with("E2045")));
    }

    #[test]
    fn an_address_call_takes_the_address_then_the_declared_arguments() {
        let src = "extern \"C\" bind(.address) { fn add2(a: i64, b: i64) -> i64; }\n\
                   fn main() -> i64 { return add2(1, 2); }";
        let got = codes(src);
        assert!(
            got.iter()
                .any(|d| d.starts_with("E2005") && d.contains("bound by address")),
            "{got:?}"
        );
    }
}
