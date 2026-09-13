// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Checked source-function bindings for the opt-in canonical IR frontend.
//!
//! This module deliberately owns no ambient state.  A caller supplies the
//! captured project scope and logical owner, and this builder returns a plan
//! which the lowering context consumes at the original AST call sites.

use std::collections::{BTreeMap, HashMap};

use crate::ast::{Module, Node, Span, TypeAnn};
use crate::parser::{EvalImportRefKind, parse_for_eval};
use crate::type_checker::canonical_facts::CheckedFactTable;
use crate::types::{
    CanonicalModuleTypes, FunctionDeclaration, FunctionIdentity, FunctionKind, FunctionSignature,
    ScalarType, SchemaRegistryBuilder, SemanticType,
};

use super::call_bindings::{FunctionReference, FunctionResolution};
use super::single_file_scope::ProjectScope;

/// An error at the checked source-to-canonical boundary.  The legacy frontend
/// does not see this type; unsupported source forms never become a partial IR.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CanonicalBindingError {
    #[error("canonical source scope does not contain logical owner `{0}`")]
    UnknownOwner(String),
    #[error("canonical source scope snapshot does not match owner `{0}`")]
    SnapshotMismatch(String),
    #[error("canonical source binding at {span:?}: {message}")]
    Binding { span: Span, message: String },
    #[error("canonical lowering refused: {0}")]
    Lowering(String),
    #[error("canonical metadata verification failed: {0}")]
    Verification(String),
}

/// The immutable source authority consumed by lowering.  Identity is logical
/// owner/name; spans are only occurrence keys for attaching metadata to the
/// already parsed call/definition nodes.
#[derive(Debug, Clone)]
pub struct CanonicalBindingPlan {
    // A source span is the lexical occurrence key for one captured source
    // owner. Duplicate spans are rejected during collection; function-local
    // producer ledgers are kept separately during lowering, so a colliding
    // span can never fall through to a callee-name lookup.
    calls: HashMap<Span, FunctionIdentity>,
    functions: HashMap<Span, FunctionDeclaration>,
    bundle: CanonicalModuleTypes,
    checked_facts: CheckedFactTable,
}

impl CanonicalBindingPlan {
    pub fn build(
        module: &Module,
        source: &str,
        scope: &ProjectScope,
        owner: &str,
    ) -> Result<Self, CanonicalBindingError> {
        Self::build_with_checked_facts(module, source, scope, owner, CheckedFactTable::default())
    }

    pub(crate) fn build_with_checked_facts(
        module: &Module,
        source: &str,
        scope: &ProjectScope,
        owner: &str,
        checked_facts: CheckedFactTable,
    ) -> Result<Self, CanonicalBindingError> {
        let captured = scope
            .sources()
            .iter()
            .find(|source| source.module_path() == owner)
            .ok_or_else(|| CanonicalBindingError::UnknownOwner(owner.to_string()))?;
        if captured.source() != source || captured.module() != module {
            return Err(CanonicalBindingError::SnapshotMismatch(owner.to_string()));
        }

        let mut declarations = BTreeMap::new();
        let mut functions = HashMap::new();
        collect_functions(
            &module.items,
            owner,
            scope,
            &mut functions,
            &mut declarations,
            true,
        )?;

        let eval_refs = parse_for_eval(source)
            .map_err(|errors| CanonicalBindingError::Binding {
                span: Span::new(0, source.len()),
                message: errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            })?
            .import_refs;
        let mut calls = HashMap::new();
        collect_calls(
            &module.items,
            owner,
            scope,
            &eval_refs,
            &mut calls,
            &mut declarations,
        )?;

        let registry = SchemaRegistryBuilder::default().finish().map_err(|error| {
            CanonicalBindingError::Binding {
                span: Span::new(0, source.len()),
                message: error.to_string(),
            }
        })?;
        let mut bundle = CanonicalModuleTypes::new(registry);
        for declaration in declarations.into_values() {
            bundle.add_declaration(declaration).map_err(|error| {
                CanonicalBindingError::Binding {
                    span: Span::new(0, source.len()),
                    message: error.to_string(),
                }
            })?;
        }
        Ok(Self {
            calls,
            functions,
            bundle,
            checked_facts,
        })
    }

    pub(crate) fn take_call(&mut self, span: Span) -> Option<FunctionIdentity> {
        self.calls.remove(&span)
    }

    pub(crate) fn function(&self, span: Span) -> Option<&FunctionDeclaration> {
        self.functions.get(&span)
    }

    pub(crate) fn take_function(&mut self, span: Span) -> Option<FunctionDeclaration> {
        self.functions.remove(&span)
    }

    pub(crate) fn first_unconsumed_call(&self) -> Option<(Span, &FunctionIdentity)> {
        self.calls
            .iter()
            .min_by_key(|(span, _)| (span.start(), span.end()))
            .map(|(span, identity)| (*span, identity))
    }

    pub(crate) fn first_unconsumed_function(&self) -> Option<(Span, &FunctionDeclaration)> {
        self.functions
            .iter()
            .min_by_key(|(span, _)| (span.start(), span.end()))
            .map(|(span, declaration)| (*span, declaration))
    }

    pub(crate) fn declarations(&self) -> impl Iterator<Item = &FunctionDeclaration> {
        self.bundle.functions().values()
    }

    pub(crate) fn checked_fact(
        &self,
        function: Option<Span>,
        expression: Span,
    ) -> Option<&crate::type_checker::canonical_facts::CheckedFact> {
        self.checked_facts.get(function, expression)
    }

    pub(crate) fn into_bundle(self) -> CanonicalModuleTypes {
        self.bundle
    }
}

// The parser represents explicit `-> ()` as an empty tuple because TypeAnn has
// no separate Unit variant. Keep this source-level distinction here, before
// semantic conversion rejects non-scalar annotations.
fn is_explicit_unit_return(return_type: Option<&TypeAnn>) -> bool {
    matches!(
        return_type,
        Some(TypeAnn::Tuple { elements }) if elements.is_empty()
    )
}

fn collect_functions(
    items: &[Node],
    owner: &str,
    scope: &ProjectScope,
    functions: &mut HashMap<Span, FunctionDeclaration>,
    declarations: &mut BTreeMap<FunctionIdentity, FunctionDeclaration>,
    top_level: bool,
) -> Result<(), CanonicalBindingError> {
    for item in items {
        match item {
            Node::FnDef(function, span) => {
                if !top_level {
                    return Err(binding_error(
                        *span,
                        "nested function definitions are outside the canonical source slice",
                    ));
                }
                let resolution =
                    scope.resolve_function(owner, FunctionReference::Bare(&function.name));
                let declaration = unique_declaration(*span, owner, resolution)?;
                if declaration.signature().return_type().is_none() {
                    return Err(binding_error(
                        *span,
                        "unit/implicit-return functions are outside the canonical source slice",
                    ));
                }
                let identity = declaration.identity().clone();
                if declaration.kind() != FunctionKind::Local || identity.owner() != owner {
                    return Err(binding_error(
                        *span,
                        "local function resolved to a non-local owner",
                    ));
                }
                if functions.insert(*span, declaration.clone()).is_some() {
                    return Err(binding_error(*span, "duplicate function occurrence span"));
                }
                if let Some(previous) = declarations.insert(identity.clone(), declaration.clone()) {
                    if previous != declaration {
                        return Err(binding_error(*span, "conflicting function declaration"));
                    }
                }
                collect_functions(&function.body, owner, scope, functions, declarations, false)?;
            }
            Node::Block { stmts, .. } => {
                collect_functions(stmts, owner, scope, functions, declarations, top_level)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn collect_calls(
    items: &[Node],
    owner: &str,
    scope: &ProjectScope,
    eval_refs: &[crate::parser::EvalImportRef],
    calls: &mut HashMap<Span, FunctionIdentity>,
    declarations: &mut BTreeMap<FunctionIdentity, FunctionDeclaration>,
) -> Result<(), CanonicalBindingError> {
    for item in items {
        collect_node_calls(item, owner, scope, eval_refs, calls, declarations)?;
    }
    Ok(())
}

fn collect_node_calls(
    node: &Node,
    owner: &str,
    scope: &ProjectScope,
    eval_refs: &[crate::parser::EvalImportRef],
    calls: &mut HashMap<Span, FunctionIdentity>,
    declarations: &mut BTreeMap<FunctionIdentity, FunctionDeclaration>,
) -> Result<(), CanonicalBindingError> {
    match node {
        Node::Call { callee, args, span } => {
            let reference = eval_refs
                .iter()
                .find(|reference| {
                    reference.kind == EvalImportRefKind::Call
                        && reference.span == *span
                        && reference.symbol == *callee
                })
                .map(|reference| FunctionReference::Qualified {
                    import_path: reference.qualifier.as_slice(),
                    name: callee,
                })
                .unwrap_or(FunctionReference::Bare(callee));
            let declaration =
                unique_declaration(*span, owner, scope.resolve_function(owner, reference))?;
            if declaration.signature().return_type().is_none() {
                return Err(binding_error(
                    *span,
                    "unit/implicit-return calls are outside the canonical source slice",
                ));
            }
            let identity = declaration.identity().clone();
            declarations.entry(identity.clone()).or_insert(declaration);
            if calls.insert(*span, identity).is_some() {
                return Err(binding_error(*span, "duplicate call occurrence span"));
            }
            for arg in args {
                collect_node_calls(arg, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::FnDef(function, _) => {
            for body in &function.body {
                collect_node_calls(body, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::Block { stmts, .. } => {
            for child in stmts {
                collect_node_calls(child, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch: None,
            ..
        } => {
            collect_node_calls(cond, owner, scope, eval_refs, calls, declarations)?;
            for child in then_branch {
                collect_node_calls(child, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch: Some(else_branch),
            ..
        } => {
            collect_node_calls(cond, owner, scope, eval_refs, calls, declarations)?;
            for child in then_branch.iter().chain(else_branch) {
                collect_node_calls(child, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::Binary { left, right, .. } | Node::Logical { left, right, .. } => {
            collect_node_calls(left, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(right, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::Paren(inner, ..)
        | Node::Neg { operand: inner, .. }
        | Node::Not { operand: inner, .. }
        | Node::BitNot { operand: inner, .. }
        | Node::Ref { inner, .. }
        | Node::Try { inner, .. }
        | Node::As { expr: inner, .. }
        | Node::CallTensorRelu { x: inner, .. }
        | Node::CallReshape { x: inner, .. }
        | Node::CallExpandDims { x: inner, .. }
        | Node::CallSqueeze { x: inner, .. }
        | Node::CallTranspose { x: inner, .. }
        | Node::CallIndex { x: inner, .. }
        | Node::Deref { operand: inner, .. } => {
            collect_node_calls(inner, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::DerefAssign { target, value, .. } => {
            collect_node_calls(target, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(value, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::Tuple { elements, .. }
        | Node::ArrayLit { elements, .. }
        | Node::SetLit { elements, .. } => {
            for child in elements {
                collect_node_calls(child, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::Let { value, .. }
        | Node::Assign { value, .. }
        | Node::Return {
            value: Some(value), ..
        }
        | Node::Assert { cond: value, .. } => {
            collect_node_calls(value, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::LetTuple { value, .. } => {
            collect_node_calls(value, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::Print { args, .. } => {
            for arg in args {
                collect_node_calls(arg, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::MethodCall { span, .. } => {
            return Err(binding_error(
                *span,
                "method calls are generated authority outside the scalar source-function slice",
            ));
        }
        Node::StructLit { fields, .. } => {
            for field in fields {
                collect_node_calls(&field.value, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::MapLit { entries, .. } => {
            for (key, value) in entries {
                collect_node_calls(key, owner, scope, eval_refs, calls, declarations)?;
                collect_node_calls(value, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::FieldAccess { receiver, .. } => {
            collect_node_calls(receiver, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::IndexAccess {
            receiver, index, ..
        } => {
            collect_node_calls(receiver, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(index, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::SliceRange {
            receiver,
            start,
            end,
            ..
        } => {
            for child in [receiver, start, end] {
                collect_node_calls(child, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            collect_node_calls(receiver, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(index, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(value, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::FieldAssign {
            receiver, value, ..
        } => {
            collect_node_calls(receiver, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(value, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::Match {
            scrutinee, arms, ..
        } => {
            collect_node_calls(scrutinee, owner, scope, eval_refs, calls, declarations)?;
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_node_calls(guard, owner, scope, eval_refs, calls, declarations)?;
                }
                collect_node_calls(&arm.body, owner, scope, eval_refs, calls, declarations)?;
            }
        }
        #[cfg(feature = "std-surface")]
        Node::Bitwise { left, right, .. } => {
            collect_node_calls(left, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(right, owner, scope, eval_refs, calls, declarations)?;
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            collect_node_calls(cond, owner, scope, eval_refs, calls, declarations)?;
            collect_calls(body, owner, scope, eval_refs, calls, declarations)?;
        }
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => {
            collect_calls(body, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::For {
            start, end, body, ..
        } => {
            collect_node_calls(start, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(end, owner, scope, eval_refs, calls, declarations)?;
            collect_calls(body, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::ForEach {
            collection, body, ..
        } => {
            collect_node_calls(collection, owner, scope, eval_refs, calls, declarations)?;
            collect_calls(body, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::CallGather { x, idx, .. } => {
            collect_node_calls(x, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(idx, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::CallDot { a, b, .. }
        | Node::CallMatMul { a, b, .. }
        | Node::TensorMatmul { lhs: a, rhs: b, .. }
        | Node::TensorElemwise { lhs: a, rhs: b, .. } => {
            collect_node_calls(a, owner, scope, eval_refs, calls, declarations)?;
            collect_node_calls(b, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::CallTensorSum { x, .. }
        | Node::CallTensorMean { x, .. }
        | Node::CallTensorConv2d { x, .. } => {
            collect_node_calls(x, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::CallGrad { span, .. } => {
            return Err(binding_error(
                *span,
                "generated gradient calls are outside the canonical source slice",
            ));
        }
        Node::CallSlice { x, .. } | Node::CallSliceStride { x, .. } => {
            collect_node_calls(x, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::Const { value, .. } => {
            collect_node_calls(value, owner, scope, eval_refs, calls, declarations)?;
        }
        Node::ExternConst { span, .. } => {
            return Err(binding_error(
                *span,
                "external constants are outside the canonical source slice",
            ));
        }
        Node::Closure(_, span) => {
            return Err(binding_error(
                *span,
                "generated closures are outside the canonical source slice",
            ));
        }
        Node::ExternBlock { span, .. } => {
            return Err(binding_error(
                *span,
                "external function blocks are outside the canonical source slice",
            ));
        }
        Node::TraitDef { span, .. } | Node::ImplBlock { span, .. } => {
            return Err(binding_error(
                *span,
                "trait-generated functions are outside the canonical source slice",
            ));
        }
        Node::Return { value: None, span } => {
            return Err(binding_error(
                *span,
                "return type mismatch: canonical scalar return has no value",
            ));
        }
        Node::Lit(_, _)
        | Node::Import { .. }
        | Node::Break { .. }
        | Node::Continue { .. }
        | Node::CallTensorRand { .. }
        | Node::TypeAlias { .. }
        | Node::Export { .. }
        | Node::StructDef { .. }
        | Node::EnumDef { .. } => {}
    }
    Ok(())
}

fn unique_declaration(
    span: Span,
    source_owner: &str,
    resolution: FunctionResolution,
) -> Result<FunctionDeclaration, CanonicalBindingError> {
    let candidate = match resolution {
        FunctionResolution::Unique(candidate) => candidate,
        FunctionResolution::Ambiguous(declarations) => {
            return Err(binding_error(
                span,
                format!("ambiguous function: {} candidate(s)", declarations.len()),
            ));
        }
        FunctionResolution::Unsupported(function) => {
            return Err(binding_error(
                span,
                format!(
                    "unsupported function authority for {}",
                    function.declaration().name()
                ),
            ));
        }
        FunctionResolution::Missing => return Err(binding_error(span, "missing source function")),
    };
    if is_explicit_unit_return(candidate.signature().return_type()) {
        return Err(binding_error(
            span,
            "unit/implicit-return functions are outside the canonical source slice",
        ));
    }
    let params = candidate
        .signature()
        .param_types()
        .iter()
        .map(semantic_scalar)
        .collect::<Result<Vec<_>, _>>()?;
    let return_type = candidate
        .signature()
        .return_type()
        .map(semantic_scalar)
        .transpose()?;
    let kind = if candidate.identity().owner() == source_owner {
        FunctionKind::Local
    } else {
        FunctionKind::External
    };
    Ok(FunctionDeclaration::new(
        candidate.identity().clone(),
        kind,
        FunctionSignature::new(params, return_type),
    ))
}

fn semantic_scalar(ty: &TypeAnn) -> Result<SemanticType, CanonicalBindingError> {
    let scalar = match ty {
        TypeAnn::ScalarI32 => ScalarType::I32,
        TypeAnn::ScalarI64 => ScalarType::I64,
        TypeAnn::ScalarF32 => ScalarType::F32,
        TypeAnn::ScalarF64 => ScalarType::F64,
        TypeAnn::ScalarBool => ScalarType::Bool,
        TypeAnn::ScalarU32 => ScalarType::U32,
        other => {
            return Err(binding_error(
                Span::new(0, 0),
                format!("non-scalar function annotation {other:?}"),
            ));
        }
    };
    Ok(SemanticType::Scalar(scalar))
}

fn binding_error(span: Span, message: impl Into<String>) -> CanonicalBindingError {
    CanonicalBindingError::Binding {
        span,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_occurrences_are_consumed_exactly() {
        let span = Span::new(4, 8);
        let identity = FunctionIdentity::new("crate", "f");
        let registry = SchemaRegistryBuilder::default()
            .finish()
            .expect("empty schema registry");
        let mut plan = CanonicalBindingPlan {
            calls: HashMap::from([(span, identity.clone())]),
            functions: HashMap::new(),
            bundle: CanonicalModuleTypes::new(registry),
            checked_facts: CheckedFactTable::default(),
        };
        assert_eq!(plan.first_unconsumed_call().map(|(at, _)| at), Some(span));
        assert_eq!(plan.take_call(span), Some(identity));
        assert!(plan.first_unconsumed_call().is_none());
        assert!(plan.take_call(span).is_none());
    }
}
