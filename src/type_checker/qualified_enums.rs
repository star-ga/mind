// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Module-qualified type validation and enum payload resolution.

#[cfg(feature = "cross-module-imports")]
use std::collections::BTreeSet;

use crate::ast::{Module, Node, Pattern, TypeAnn};
use crate::diagnostics::Diagnostic;

/// Resolve payload types locally first, then by the canonical project enum key.
pub(super) fn variant_payload_of(enum_name: &str, variant: &str) -> Option<Vec<TypeAnn>> {
    let key = format!("{enum_name}::{variant}");
    let payload =
        super::ENUM_PAYLOADS.with(|cell| cell.borrow().as_ref().and_then(|t| t.get(&key).cloned()));
    #[cfg(feature = "cross-module-imports")]
    let payload = payload
        .or_else(|| crate::ir::with_global_enums(|g| g.qualified.payload_types.get(&key).cloned()));
    payload
}

pub(super) fn validate(module: &Module, src: &str, file: Option<&str>) -> Vec<Diagnostic> {
    let mut errors = Vec::new();
    #[cfg(feature = "cross-module-imports")]
    let imports = crate::qualified_enums::module_imports(module);
    #[cfg(feature = "cross-module-imports")]
    let local_types = local_type_names(module);
    let mut pending: Vec<&Node> = module.items.iter().rev().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Let {
                ann: Some(ann),
                span,
                ..
            } => validate_type(
                ann,
                *span,
                src,
                file,
                #[cfg(feature = "cross-module-imports")]
                &imports,
                &mut errors,
            ),
            Node::FnDef(fd, span) => {
                if let Some(ret) = &fd.ret_type {
                    validate_type(
                        ret,
                        *span,
                        src,
                        file,
                        #[cfg(feature = "cross-module-imports")]
                        &imports,
                        &mut errors,
                    );
                }
                for param in &fd.params {
                    validate_type(
                        &param.ty,
                        param.span,
                        src,
                        file,
                        #[cfg(feature = "cross-module-imports")]
                        &imports,
                        &mut errors,
                    );
                }
            }
            Node::Const {
                ty: Some(ann),
                span,
                ..
            }
            | Node::ExternConst { ty: ann, span, .. }
            | Node::TypeAlias {
                target: ann, span, ..
            }
            | Node::As { ty: ann, span, .. } => validate_type(
                ann,
                *span,
                src,
                file,
                #[cfg(feature = "cross-module-imports")]
                &imports,
                &mut errors,
            ),
            Node::StructDef { fields, .. } => {
                for field in fields {
                    validate_type(
                        &field.ty,
                        field.span,
                        src,
                        file,
                        #[cfg(feature = "cross-module-imports")]
                        &imports,
                        &mut errors,
                    );
                }
            }
            Node::EnumDef { variants, span, .. } => {
                for ann in variants.iter().flat_map(|variant| &variant.payload) {
                    validate_type(
                        ann,
                        *span,
                        src,
                        file,
                        #[cfg(feature = "cross-module-imports")]
                        &imports,
                        &mut errors,
                    );
                }
            }
            Node::ExternBlock { fns, .. } => {
                for function in fns {
                    if let Some(ret) = &function.ret_type {
                        validate_type(
                            ret,
                            function.span,
                            src,
                            file,
                            #[cfg(feature = "cross-module-imports")]
                            &imports,
                            &mut errors,
                        );
                    }
                    for param in &function.params {
                        validate_type(
                            &param.ty,
                            param.span,
                            src,
                            file,
                            #[cfg(feature = "cross-module-imports")]
                            &imports,
                            &mut errors,
                        );
                    }
                }
            }
            Node::Closure(cd, span) => {
                if let Some(ret) = &cd.ret_type {
                    validate_type(
                        ret,
                        *span,
                        src,
                        file,
                        #[cfg(feature = "cross-module-imports")]
                        &imports,
                        &mut errors,
                    );
                }
                for param in &cd.params {
                    validate_type(
                        &param.ty,
                        param.span,
                        src,
                        file,
                        #[cfg(feature = "cross-module-imports")]
                        &imports,
                        &mut errors,
                    );
                }
            }
            Node::TraitDef { methods, .. } => {
                for method in methods {
                    if let Some(ret) = &method.ret_type {
                        validate_type(
                            ret,
                            method.span,
                            src,
                            file,
                            #[cfg(feature = "cross-module-imports")]
                            &imports,
                            &mut errors,
                        );
                    }
                    for param in &method.params {
                        validate_type(
                            &param.ty,
                            param.span,
                            src,
                            file,
                            #[cfg(feature = "cross-module-imports")]
                            &imports,
                            &mut errors,
                        );
                    }
                }
            }
            Node::Match { arms, .. } => {
                for arm in arms {
                    validate_pattern(
                        &arm.pattern,
                        arm.span,
                        src,
                        file,
                        #[cfg(feature = "cross-module-imports")]
                        &imports,
                        #[cfg(feature = "cross-module-imports")]
                        &local_types,
                        &mut errors,
                    );
                }
            }
            #[cfg(feature = "cross-module-imports")]
            Node::Lit(crate::ast::Literal::Ident(path), span) => {
                check_bare_variant_path(path, *span, src, file, &imports, &local_types, &mut errors)
            }
            _ => {}
        }
        super::nerve_walk::for_each_child(node, &mut |child| pending.push(child));
    }
    errors
}

/// The width of an `iN` / `uN` integer type name wider than 64 bits (`i128`). No backend
/// represents one: the annotation was accepted and the value lowered as `i64`, so
/// `(a * 4 / 2) / a` with `a = i64::MAX` returned 0.
fn wider_than_64_bits(name: &str) -> Option<u32> {
    let digits = name.strip_prefix('i').or_else(|| name.strip_prefix('u'))?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u32>().ok().filter(|bits| *bits > 64)
}

fn validate_type(
    ann: &TypeAnn,
    span: crate::ast::Span,
    src: &str,
    file: Option<&str>,
    #[cfg(feature = "cross-module-imports")] imports: &[String],
    errors: &mut Vec<Diagnostic>,
) {
    // The root is visited without a heap worklist: a scalar or tensor
    // annotation (the common case) has no nested types to queue.
    let mut pending: Vec<&TypeAnn> = Vec::new();
    let mut next = Some(ann);
    while let Some(ann) = next.take().or_else(|| pending.pop()) {
        let name = match ann {
            TypeAnn::Named(name) => {
                if let Some(bits) = wider_than_64_bits(name) {
                    errors.push(super::diag_from_span(
                        src,
                        file,
                        format!(
                            "`{name}` is not supported: integer types are at most 64 bits wide, \
                             and a {bits}-bit value would be computed in 64 bits"
                        ),
                        span,
                        super::resolve::UNKNOWN_IDENT_CODE,
                    ));
                    continue;
                }
                Some(name.as_str())
            }
            TypeAnn::Tensor { dtype, .. } | TypeAnn::DiffTensor { dtype, .. } => {
                Some(dtype.as_str())
            }
            TypeAnn::Generic { name, args } => {
                pending.extend(args.iter().rev());
                Some(name.as_str())
            }
            TypeAnn::Slice { element, .. }
            | TypeAnn::Array { element, .. }
            | TypeAnn::SparseTensor { element, .. } => {
                pending.push(element);
                None
            }
            TypeAnn::Ref { target, .. } => {
                pending.push(target);
                None
            }
            TypeAnn::RawPtr { pointee, .. } => {
                pending.push(pointee);
                None
            }
            TypeAnn::Tuple { elements } => {
                pending.extend(elements.iter().rev());
                None
            }
            TypeAnn::FnPtr { params, ret } => {
                if let Some(ret) = ret {
                    pending.push(ret);
                }
                pending.extend(params.iter().rev());
                None
            }
            TypeAnn::ScalarI32
            | TypeAnn::ScalarI64
            | TypeAnn::ScalarF32
            | TypeAnn::ScalarF64
            | TypeAnn::ScalarBool
            | TypeAnn::ScalarU32 => None,
        };
        let Some(name) = name else {
            continue;
        };
        let known = crate::ir::with_global_enums(|_g| {
            #[cfg(feature = "cross-module-imports")]
            {
                _g.qualified.is_canonical_type(name)
                    || _g.qualified.source_type_is_visible(
                        name,
                        imports,
                        crate::qualified_enums::current_module_path().as_deref(),
                    )
            }
            #[cfg(not(feature = "cross-module-imports"))]
            {
                false
            }
        });
        #[cfg(feature = "cross-module-imports")]
        if !name.contains('.')
            && crate::ir::with_global_enums(|g| {
                g.qualified.bare_type_is_ambiguous(
                    name,
                    imports,
                    crate::qualified_enums::current_module_path().as_deref(),
                )
            })
        {
            errors.push(super::diag_from_span(
                src,
                file,
                format!("ambiguous bare type `{name}`; qualify it with its module"),
                span,
                super::resolve::UNKNOWN_IDENT_CODE,
            ));
            continue;
        }
        if name.contains('.') && !known {
            errors.push(super::diag_from_span(
                src,
                file,
                format!("unknown module-qualified type `{name}`"),
                span,
                super::resolve::UNKNOWN_IDENT_CODE,
            ));
        }
    }
}

fn validate_pattern(
    pattern: &Pattern,
    span: crate::ast::Span,
    src: &str,
    file: Option<&str>,
    #[cfg(feature = "cross-module-imports")] imports: &[String],
    #[cfg(feature = "cross-module-imports")] local_types: &BTreeSet<String>,
    errors: &mut Vec<Diagnostic>,
) {
    let mut pending = vec![pattern];
    while let Some(pattern) = pending.pop() {
        let path = match pattern {
            Pattern::EnumVariant { path, args } => {
                pending.extend(args);
                Some(path)
            }
            Pattern::EnumStruct { path, fields } => {
                pending.extend(fields.iter().map(|(_, pattern)| pattern));
                Some(path)
            }
            Pattern::Tuple(items) => {
                pending.extend(items);
                None
            }
            Pattern::Ident(path) if path.matches('.').count() >= 2 => Some(path),
            _ => None,
        };
        let Some(path) = path else {
            continue;
        };
        if !path.contains('.') {
            #[cfg(feature = "cross-module-imports")]
            {
                check_bare_variant_path(path, span, src, file, imports, local_types, errors);
                let (visible, exists) = crate::ir::with_global_enums(|g| {
                    (
                        g.qualified.bare_variant_is_visible(
                            path,
                            imports,
                            crate::qualified_enums::current_module_path().as_deref(),
                        ),
                        g.qualified.has_bare_variant(path),
                    )
                });
                if exists && !visible {
                    errors.push(super::diag_from_span(
                        src,
                        file,
                        format!("enum variant `{path}` is not visible in this module"),
                        span,
                        super::resolve::UNKNOWN_IDENT_CODE,
                    ));
                }
            }
            continue;
        }
        let known = crate::ir::with_global_enums(|_g| {
            #[cfg(feature = "cross-module-imports")]
            {
                _g.qualified.is_canonical_variant(path)
            }
            #[cfg(not(feature = "cross-module-imports"))]
            {
                false
            }
        });
        if !known {
            errors.push(super::diag_from_span(
                src,
                file,
                format!("unknown module-qualified enum variant `{path}`"),
                span,
                super::resolve::UNKNOWN_IDENT_CODE,
            ));
        }
    }
}

/// Refuse a bare-headed `Enum::Variant` value or pattern that names another
/// module's enum but does not resolve from this module: the enum is not
/// imported, several imports export the name, or the variant does not exist.
/// Lowering has no tag for such a path and would otherwise stop at its
/// fail-closed undefined-identifier / dangling-variant panic. Reported as
/// E2002 (an unresolved reference): the project builder refuses to embed a
/// module with that code as a runtime fallback, so `mindc build` fails closed
/// instead of lowering the module anyway.
#[cfg(feature = "cross-module-imports")]
fn check_bare_variant_path(
    path: &str,
    span: crate::ast::Span,
    src: &str,
    file: Option<&str>,
    imports: &[String],
    local_types: &BTreeSet<String>,
    errors: &mut Vec<Diagnostic>,
) {
    use crate::qualified_enums::BareVariantProblem;
    let Some((head, variant)) = path.split_once("::") else {
        return;
    };
    let declared_locally = local_types.contains(head);
    let problem = crate::ir::with_global_enums(|g| {
        g.qualified
            .bare_variant_problem(path, imports, declared_locally)
    });
    let message = match problem {
        None => return,
        Some(BareVariantProblem::NotVisible) => format!(
            "enum `{head}` in `{path}` is not visible in this module; \
             import the module that declares it"
        ),
        Some(BareVariantProblem::Ambiguous) => format!(
            "ambiguous enum `{head}` in `{path}`: several imported modules \
             export it; qualify it with its module"
        ),
        Some(BareVariantProblem::UnknownVariant) => {
            format!("unknown variant `{variant}` of imported enum `{head}`")
        }
    };
    errors.push(super::diag_from_span(
        src,
        file,
        message,
        span,
        super::resolve::UNKNOWN_IDENT_CODE,
    ));
}

/// Type names (enum, struct, alias) the module declares itself, including
/// inside `module { … }` blocks. They take lexical precedence over imported
/// types of the same name.
#[cfg(feature = "cross-module-imports")]
fn local_type_names(module: &Module) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut pending: Vec<&Node> = module.items.iter().collect();
    while let Some(item) = pending.pop() {
        match item {
            Node::EnumDef { name, .. }
            | Node::StructDef { name, .. }
            | Node::TypeAlias { name, .. } => {
                names.insert(name.clone());
            }
            Node::Block { stmts, .. } => pending.extend(stmts),
            _ => {}
        }
    }
    names
}
