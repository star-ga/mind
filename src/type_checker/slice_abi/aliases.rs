// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Alias-transparent types and constant classes for slice-call provenance.
//!
//! A type alias names its target type, so `array<Q>` with `type Q = i32` is an
//! `array<i32>`. The provenance pass compares element types structurally;
//! comparing source spellings instead refused a valid `array<Q>` argument to a
//! `&[Q]` parameter. Every type this pass reads (parameters, `let`
//! annotations, call signatures, struct fields, results) is therefore
//! canonicalized first: through the checked file's aliases and the project
//! registry for a type this file writes, and through the owner module's scope
//! for a type read from an imported signature or field. The pass descends into
//! each module item in isolation, so the file's alias and constant tables are
//! installed once by the file-level entry point.
//!
//! Lowering reads an annotation's own type constructor. A slice or collection
//! type named through an alias (`type V = array<i32>`) is not lowered as a
//! Vec-layout handle, so such an annotation keeps its alias spelling here too:
//! the slice boundary refuses a parameter, result or binding spelled this way
//! instead of admitting a call lowering rejects.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use super::super::ScalarClass;
use super::{
    SLICE_ARG_ABI_CODE, declared_kind, intra_lookup_fn, provenance::is_collection_owner_type,
    report, scalar_class_of_ann, type_contains_slice,
};
use crate::ast::{BinOp, FnDefData, Literal, Node, Span, TypeAnn};
use crate::diagnostics::Diagnostic;
use crate::eval::type_aliases::LocalTypeAliases;

/// Nesting bound for constants defined through other constants.
const CONST_DEPTH_LIMIT: usize = 32;

struct FileScope {
    aliases: LocalTypeAliases,
    /// Module-level constants of the checked file (module blocks included):
    /// declared type, value.
    consts: BTreeMap<String, (Option<TypeAnn>, Node)>,
}

thread_local! {
    static FILE_SCOPE: RefCell<Option<Rc<FileScope>>> = const { RefCell::new(None) };
}

/// Keeps one checked file's aliases and constants visible to the slice pass
/// and restores the enclosing scope on drop.
pub(crate) struct AliasScope(Option<Rc<FileScope>>);

impl AliasScope {
    pub(crate) fn install(items: &[Node]) -> Self {
        let mut consts = BTreeMap::new();
        collect_consts(items, &mut consts);
        let scope = Rc::new(FileScope {
            aliases: LocalTypeAliases::new(items),
            consts,
        });
        Self(FILE_SCOPE.with(|cell| cell.borrow_mut().replace(scope)))
    }
}

impl Drop for AliasScope {
    fn drop(&mut self) {
        FILE_SCOPE.with(|cell| *cell.borrow_mut() = self.0.take());
    }
}

fn collect_consts(items: &[Node], consts: &mut BTreeMap<String, (Option<TypeAnn>, Node)>) {
    for item in items {
        match item {
            Node::Const {
                name, ty, value, ..
            } => {
                consts.insert(name.clone(), (ty.clone(), (**value).clone()));
            }
            Node::Block { stmts, .. } => collect_consts(stmts, consts),
            _ => {}
        }
    }
}

/// `ty`, written in the checked file, with every visible alias replaced by
/// its target type, except a slice or collection named through an alias.
pub(super) fn canonical(ty: &TypeAnn) -> TypeAnn {
    let resolved = resolve_fully(ty);
    if names_handle_through_alias(ty, &resolved) {
        ty.clone()
    } else {
        resolved
    }
}

/// `ty`, written in the checked file, with every visible alias replaced by
/// its target type.
pub(super) fn resolve_fully(ty: &TypeAnn) -> TypeAnn {
    FILE_SCOPE.with(|cell| match cell.borrow().as_ref() {
        Some(scope) => scope.aliases.resolve_with_imports(ty),
        None => ty.clone(),
    })
}

/// Whether a value of type `ty` crosses the collection or slice boundary,
/// aliases included, so the flow pass must check its function.
pub(super) fn type_needs_collection_flow(ty: &TypeAnn) -> bool {
    let ty = &resolve_fully(ty);
    super::type_contains_collection_owner(ty)
        || declared_kind(ty).is_some()
        || type_contains_slice(ty)
}

/// Whether the annotation `ty` names a slice type, or an aggregate holding
/// one, through an alias. The slice pass refuses this spelling with the
/// diagnostic its spelled-out form gets, where it does not lower it.
pub(super) fn slice_through_alias(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::Named(_)) && type_contains_slice(&resolve_fully(ty))
}

/// Bit-width of an integer type annotation, or `None` for a non-integer or
/// unknown annotation. A literal element proven to be wider than a slice's
/// element type silently truncates in the native build, so the literal lane
/// compares element widths with it.
pub(super) fn int_width_of_ann(ty: &TypeAnn) -> Option<u32> {
    match ty {
        TypeAnn::ScalarI32 | TypeAnn::ScalarU32 => Some(32),
        TypeAnn::ScalarI64 => Some(64),
        TypeAnn::Named(name) => match name.as_str() {
            "i8" | "u8" => Some(8),
            "i16" | "u16" => Some(16),
            "i32" | "u32" => Some(32),
            "i64" | "u64" | "isize" | "usize" => Some(64),
            _ => None,
        },
        _ => None,
    }
}

/// Parameter and result types of `callee`, in the checked file's terms.
pub(super) fn call_signature(callee: &str) -> Option<(Vec<TypeAnn>, Option<TypeAnn>)> {
    if let Some(sig) = intra_lookup_fn(callee) {
        return Some((
            sig.param_types.iter().map(canonical).collect(),
            sig.ret_type.as_ref().map(canonical),
        ));
    }
    #[cfg(feature = "cross-module-imports")]
    if let Some((owner, sig)) = crate::project::active_module_table::lookup_visible_fn_owned(
        crate::qualified_enums::current_module_path().as_deref(),
        callee,
    ) {
        // An imported signature is written in its owner's scope.
        let read = |ty: &TypeAnn| match &owner {
            Some(owner) => canonical_in_owner(ty, owner),
            None => canonical(ty),
        };
        return Some((
            sig.param_types.iter().map(read).collect(),
            sig.ret_type.as_ref().map(read),
        ));
    }
    None
}

/// `ty`, written in module `owner` (an imported signature or a field of an
/// imported struct), in the checked file's terms.
#[cfg(feature = "cross-module-imports")]
pub(super) fn canonical_in_owner(ty: &TypeAnn, owner: &str) -> TypeAnn {
    let resolved = crate::qualified_enums::canonical_type_from(ty, owner);
    if !names_handle_through_alias(ty, &resolved) {
        return resolved;
    }
    // Keep the alias, qualified by its owner, so it cannot meet a same-named
    // alias of the checked file.
    match ty {
        TypeAnn::Named(name) if !name.contains('.') => TypeAnn::Named(format!("{owner}.{name}")),
        _ => ty.clone(),
    }
}

/// Refuse a parameter or result of `fd` that names a slice or collection type
/// through an alias: lowering reads only a spelled `&[T]`/`array<T>` there, so
/// a signature spelled through `type SL = &[i32]` or `type V = array<i32>`
/// passes the check and dies in lowering.
pub(super) fn refuse_slice_aliases_in_signature(
    fd: &FnDefData,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    for param in &fd.params {
        let message = if slice_through_alias(&param.ty) {
            Some(format!(
                "parameter `{}` names a slice type through a type alias, which the slice call ABI does not resolve; spell the parameter `&[T]` or `&mut [T]`",
                param.name
            ))
        } else if names_handle_through_alias(&param.ty, &resolve_fully(&param.ty)) {
            Some(format!(
                "parameter `{}` names a collection type through a type alias, which the slice call ABI does not resolve; spell the parameter `array<T>`, `map<K, V>` or `set<T>`",
                param.name
            ))
        } else {
            None
        };
        if let Some(message) = message {
            report(errs, src, file, message, param.span, SLICE_ARG_ABI_CODE);
        }
    }
    if let Some(ret) = fd.ret_type.as_ref() {
        let message = if slice_through_alias(ret) {
            Some(
                "function return type names a slice type through a type alias, which the slice call ABI does not resolve; spell it `&[T]`"
                    .to_string(),
            )
        } else if names_handle_through_alias(ret, &resolve_fully(ret)) {
            Some(
                "function return type names a collection type through a type alias, which the slice call ABI does not resolve; spell it `array<T>`"
                    .to_string(),
            )
        } else {
            None
        };
        if let Some(message) = message {
            let span = fd.body.last().map(Node::span);
            report(
                errs,
                src,
                file,
                message,
                span.or_else(|| fd.params.first().map(|p| p.span))
                    .unwrap_or_else(|| Span::new(0, 0)),
                SLICE_ARG_ABI_CODE,
            );
        }
    }
}

/// Refuse a `let` whose annotation names a slice or collection type through an
/// alias, like a spelled slice binding.
pub(super) fn refuse_slice_alias_binding(
    name: &str,
    ann: Option<&TypeAnn>,
    value: &Node,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    let Some(ann) = ann else {
        return;
    };
    let message = if slice_through_alias(ann) {
        Some(format!(
            "binding `{name}` names a slice type through a type alias, which the current Option-C lowering does not resolve; keep the value as `array<T>`"
        ))
    } else if names_handle_through_alias(ann, &resolve_fully(ann)) {
        Some(format!(
            "binding `{name}` names a collection type through a type alias, which the current Option-C lowering does not resolve; spell it `array<T>`, `map<K, V>` or `set<T>`"
        ))
    } else {
        None
    };
    if let Some(message) = message {
        report(errs, src, file, message, value.span(), SLICE_ARG_ABI_CODE);
    }
}

/// Whether the annotation `spelled` is an alias whose target is a slice or a
/// collection owner.
fn names_handle_through_alias(spelled: &TypeAnn, resolved: &TypeAnn) -> bool {
    matches!(spelled, TypeAnn::Named(_))
        && (declared_kind(resolved).is_some() || is_collection_owner_type(resolved))
}

/// Scalar class of the module constant `name`, as lowering inlines it: the
/// checked file's constant first, then the project's cross-module constant.
/// A declared type must agree with the value's class.
pub(super) fn const_class(name: &str) -> Option<ScalarClass> {
    const_class_at(name, 0)
}

/// The declaration of the constant `name`: every declared type, read in its
/// owner's scope, and the value lowering inlines. The nesting bound keeps a
/// cyclic constant (`const A = B`, `const B = A`) from recursing without end.
fn const_record(name: &str, depth: usize) -> Option<(Vec<TypeAnn>, Node)> {
    if depth > CONST_DEPTH_LIMIT {
        return None;
    }
    let local = FILE_SCOPE.with(|cell| cell.borrow().as_ref()?.consts.get(name).cloned());
    match local {
        Some((ty, value)) => Some((ty.map(|ty| canonical(&ty)).into_iter().collect(), value)),
        None => imported_const(name),
    }
}

fn const_class_at(name: &str, depth: usize) -> Option<ScalarClass> {
    let (declared, value) = const_record(name, depth)?;
    let value_class = const_value_class(&value, depth + 1)?;
    declared
        .iter()
        .all(|ty: &TypeAnn| scalar_class_of_ann(ty) == Some(value_class))
        .then_some(value_class)
}

/// Proven integer bit-width of the module constant `name`, as lowering
/// inlines it. Every declared type must agree on one integer width; an
/// undeclared constant takes the width of its value, and a value of no proven
/// width (a plain literal) has none. `None` never refuses.
pub(super) fn const_int_width(name: &str) -> Option<u32> {
    const_int_width_at(name, 0)
}

fn const_int_width_at(name: &str, depth: usize) -> Option<u32> {
    let (declared, value) = const_record(name, depth)?;
    if !declared.is_empty() {
        let widths: Vec<u32> = declared.iter().filter_map(int_width_of_ann).collect();
        let [first, rest @ ..] = widths.as_slice() else {
            return None;
        };
        return rest.iter().all(|width| width == first).then_some(*first);
    }
    const_value_width(&value, depth + 1)
}

fn const_value_width(value: &Node, depth: usize) -> Option<u32> {
    match value {
        Node::As { ty, .. } => int_width_of_ann(&canonical(ty)),
        Node::Lit(Literal::Ident(name), _) => const_int_width_at(name, depth),
        Node::Paren(inner, _) | Node::Neg { operand: inner, .. } => const_value_width(inner, depth),
        Node::Binary {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
            left,
            right,
            ..
        } => match (
            const_value_width(left, depth),
            const_value_width(right, depth),
        ) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(one), None) | (None, Some(one)) => Some(one),
            (None, None) => None,
        },
        _ => None,
    }
}

/// The value lowering inlines for the cross-module constant `name`, with the
/// declared type of every module-level constant of that name, read in its
/// owner's scope.
#[cfg(feature = "cross-module-imports")]
fn imported_const(name: &str) -> Option<(Vec<TypeAnn>, Node)> {
    let value = crate::ir::project_consts::project_const_value(name)?;
    let declared = crate::qualified_enums::imported_const_declared_types(name)
        .into_iter()
        .map(|(owner, ty)| canonical_in_owner(&ty, &owner))
        .collect();
    Some((declared, value))
}

#[cfg(not(feature = "cross-module-imports"))]
fn imported_const(_name: &str) -> Option<(Vec<TypeAnn>, Node)> {
    None
}

fn const_value_class(value: &Node, depth: usize) -> Option<ScalarClass> {
    match value {
        Node::Lit(Literal::Int(_), _) => Some(ScalarClass::Int),
        Node::Lit(Literal::Float(_), _) => Some(ScalarClass::Float),
        Node::Lit(Literal::Ident(name), _) => const_class_at(name, depth),
        Node::As { ty, .. } => scalar_class_of_ann(&canonical(ty)),
        Node::Paren(inner, _) | Node::Neg { operand: inner, .. } => const_value_class(inner, depth),
        Node::Binary {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
            left,
            right,
            ..
        } => {
            let class = const_value_class(left, depth)?;
            (const_value_class(right, depth) == Some(class)).then_some(class)
        }
        _ => None,
    }
}
