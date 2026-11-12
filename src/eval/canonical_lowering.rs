// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Transient state for the opt-in source-to-canonical lowering entry point.
//! This stays outside `IRModule` construction until the wire boundary exists.

use std::collections::{BTreeMap, HashMap};

use super::{LoweringContext, lower_entry::lower_module};
use crate::eval::materialization::MaterializationLimits;
use crate::ir::{IRModule, Instr, ValueId};
use crate::project::canonical_bridge::{CanonicalBindingError, CanonicalBindingPlan};
use crate::type_checker::canonical_facts::CheckedFact;
use crate::types::{FunctionDeclaration, FunctionSemanticTypes, SemanticType};

fn canonical_types_compatible(actual: &SemanticType, expected: &SemanticType) -> bool {
    actual == expected
        || matches!(
            (actual, expected),
            (
                SemanticType::Scalar(crate::types::ScalarType::I32),
                SemanticType::Scalar(crate::types::ScalarType::I64)
            )
        )
}

pub(crate) struct CanonicalLoweringState {
    pub(super) plan: CanonicalBindingPlan,
    pub(super) function_spans: Vec<crate::ast::Span>,
    pub(super) module_values: BTreeMap<ValueId, SemanticType>,
    /// Source-expression producers recorded by the canonical lowering path.
    /// This is deliberately separate from the final carrier: a declaration
    /// may validate a call, but it may not manufacture a type for an
    /// otherwise untyped SSA value.
    pub(super) module_producers: HashMap<crate::ast::Span, SemanticType>,
    pub(super) functions: Vec<FunctionSemanticTypes>,
    pub(super) function_producers: Vec<HashMap<crate::ast::Span, SemanticType>>,
    pub(super) refusal: Option<String>,
}

impl CanonicalLoweringState {
    pub(super) fn new(plan: CanonicalBindingPlan) -> Self {
        Self {
            function_spans: Vec::new(),
            plan,
            module_values: BTreeMap::new(),
            module_producers: HashMap::new(),
            functions: Vec::new(),
            function_producers: Vec::new(),
            refusal: None,
        }
    }

    fn current_mut(&mut self) -> Option<&mut FunctionSemanticTypes> {
        self.functions.last_mut()
    }

    fn record(&mut self, value: ValueId, ty: SemanticType) -> Result<(), String> {
        if let Some(function) = self.current_mut() {
            if let Some(existing) = function.values().get(&value) {
                if existing == &ty {
                    return Ok(());
                }
                return Err(format!("conflicting semantic types for {value}"));
            }
            function
                .set_value_type(value, ty)
                .map_err(|error| error.to_string())
        } else if let Some(existing) = self.module_values.get(&value) {
            if existing == &ty {
                Ok(())
            } else {
                Err(format!("conflicting semantic types for {value}"))
            }
        } else {
            self.module_values.insert(value, ty);
            Ok(())
        }
    }

    fn value_type(&self, value: ValueId) -> Option<SemanticType> {
        if let Some(function) = self.functions.last() {
            return function.values().get(&value).cloned();
        }
        self.module_values.get(&value).cloned()
    }

    fn record_producer(
        &mut self,
        span: crate::ast::Span,
        value: ValueId,
        ty: SemanticType,
    ) -> Result<(), String> {
        let producers = if self.functions.is_empty() {
            &mut self.module_producers
        } else {
            self.function_producers
                .last_mut()
                .ok_or_else(|| "canonical function producer scope missing".to_string())?
        };
        if let Some(existing) = producers.get(&span) {
            if existing != &ty {
                return Err(format!("conflicting producer types at {span:?}"));
            }
        } else {
            producers.insert(span, ty.clone());
        }
        self.record(value, ty)
    }
}

impl LoweringContext {
    pub(super) fn canonical_active(&self) -> bool {
        self.canonical.is_some()
    }

    pub(super) fn canonical_failed(&self) -> bool {
        self.canonical
            .as_ref()
            .is_some_and(|state| state.refusal.is_some())
    }

    pub(super) fn canonical_refuse(&mut self, message: impl Into<String>) {
        if let Some(state) = self.canonical.as_mut() {
            if state.refusal.is_none() {
                state.refusal = Some(message.into());
            }
        }
    }

    pub(super) fn canonical_call_identity(
        &mut self,
        span: crate::ast::Span,
        callee: &str,
    ) -> Option<crate::types::FunctionIdentity> {
        let state = self.canonical.as_mut()?;
        let identity = state.plan.take_call(span);
        if identity.is_none() {
            self.canonical_refuse(format!("source call `{callee}` has no canonical binding"));
        }
        identity
    }

    pub(super) fn canonical_declaration(
        &self,
        span: crate::ast::Span,
    ) -> Option<FunctionDeclaration> {
        self.canonical
            .as_ref()
            .and_then(|state| state.plan.function(span).cloned())
    }

    pub(super) fn canonical_begin_function_for_span(
        &mut self,
        span: crate::ast::Span,
        name: &str,
        params: &[(String, ValueId)],
    ) {
        if self.canonical.is_none() {
            return;
        }
        if let Some(declaration) = self.canonical_declaration(span) {
            if let Some(state) = self.canonical.as_mut() {
                state.plan.take_function(span);
            }
            self.begin_canonical_function(span, &declaration, params);
        } else {
            self.canonical_refuse(format!("function `{name}` has no canonical declaration"));
        }
    }

    pub(super) fn canonical_callee_declaration(
        &self,
        identity: &crate::types::FunctionIdentity,
    ) -> Option<FunctionDeclaration> {
        self.canonical.as_ref().and_then(|state| {
            state
                .plan
                .declarations()
                .find(|declaration| declaration.identity() == identity)
                .cloned()
        })
    }

    pub(super) fn canonical_record_producer(
        &mut self,
        span: crate::ast::Span,
        value: ValueId,
        ty: SemanticType,
    ) {
        let error = self
            .canonical
            .as_mut()
            .and_then(|state| state.record_producer(span, value, ty).err());
        if let Some(error) = error {
            self.canonical_refuse(error);
        }
    }

    pub(super) fn canonical_checked_fact(&self, span: crate::ast::Span) -> Option<CheckedFact> {
        self.canonical.as_ref().and_then(|state| {
            let function = state.function_spans.last().copied();
            state.plan.checked_fact(function, span).cloned()
        })
    }

    pub(super) fn canonical_value_type(&self, value: ValueId) -> Option<SemanticType> {
        self.canonical
            .as_ref()
            .and_then(|state| state.value_type(value))
    }

    pub(super) fn canonical_record_call(
        &mut self,
        span: crate::ast::Span,
        identity: Option<&crate::types::FunctionIdentity>,
        args: &[ValueId],
        dst: ValueId,
    ) {
        let Some(identity) = identity else { return };
        let Some(declaration) = self.canonical_callee_declaration(identity) else {
            return;
        };
        for (value, ty) in args.iter().zip(declaration.signature().params()) {
            let Some(actual) = self.canonical_value_type(*value) else {
                self.canonical_refuse(format!(
                    "canonical call argument {value} has no checked producer type"
                ));
                continue;
            };
            if !canonical_types_compatible(&actual, ty) {
                self.canonical_refuse(format!(
                    "canonical call argument {value} has producer type {actual:?}, expected {ty:?}"
                ));
            }
        }
        let actual = match self.canonical_checked_fact(span) {
            Some(CheckedFact::Exact(ty)) => {
                let Some(ty) = super::canonical_producers::semantic_type(ty) else {
                    self.canonical_refuse(format!(
                        "canonical call `{}` has an unsupported checked result type",
                        identity.name()
                    ));
                    return;
                };
                ty
            }
            Some(CheckedFact::Unknown) | None => {
                self.canonical_refuse(format!(
                    "canonical call `{}` has no exact checked producer type",
                    identity.name()
                ));
                return;
            }
        };
        let Some(expected) = declaration.signature().return_type() else {
            self.canonical_refuse(format!(
                "canonical call `{}` has no supported return type",
                identity.name()
            ));
            return;
        };
        if !canonical_types_compatible(&actual, expected) {
            self.canonical_refuse(format!(
                "canonical call `{}` producer type {actual:?} conflicts with declared return {expected:?}",
                identity.name()
            ));
            return;
        }
        self.canonical_record_producer(span, dst, actual);
    }

    /// Validate an explicit return at the point where its lowered value is
    /// available.  The final function result is insufficient for a function
    /// with branch-local returns: every reachable return must carry an exact
    /// checked producer fact.
    pub(super) fn canonical_validate_return(&mut self, span: crate::ast::Span, value: ValueId) {
        let Some((identity, actual)) = self.canonical.as_ref().and_then(|state| {
            state.functions.last().map(|function| {
                (
                    function.identity().clone(),
                    function.values().get(&value).cloned(),
                )
            })
        }) else {
            if self.canonical.is_some() {
                self.canonical_refuse("canonical return is outside a function");
            }
            return;
        };
        let Some(expected) = self.canonical.as_ref().and_then(|state| {
            state
                .plan
                .declarations()
                .find(|declaration| declaration.identity() == &identity)
                .and_then(|declaration| declaration.signature().return_type())
                .cloned()
        }) else {
            self.canonical_refuse("canonical return has no supported declared type");
            return;
        };
        let Some(actual) = actual else {
            self.canonical_refuse(format!(
                "return value {value} at {span:?} has no exact checked producer type"
            ));
            return;
        };
        if !canonical_types_compatible(&actual, &expected) {
            self.canonical_refuse(format!(
                "return value {value} at {span:?} has type {actual:?}, expected {expected:?}"
            ));
        }
    }

    pub(super) fn begin_canonical_function(
        &mut self,
        span: crate::ast::Span,
        declaration: &FunctionDeclaration,
        params: &[(String, ValueId)],
    ) {
        let mut semantic = FunctionSemanticTypes::new(declaration.identity().clone());
        for ((_, value), ty) in params.iter().zip(declaration.signature().params()) {
            if let Err(error) = semantic.set_value_type(*value, ty.clone()) {
                self.canonical_refuse(error.to_string());
                break;
            }
        }
        if let Some(state) = self.canonical.as_mut() {
            state.functions.push(semantic);
            state.function_spans.push(span);
            state.function_producers.push(HashMap::new());
        }
    }

    pub(super) fn end_canonical_function(
        &mut self,
        ret_id: Option<ValueId>,
    ) -> Option<FunctionSemanticTypes> {
        let (semantic, return_type) = {
            let state = self.canonical.as_mut()?;
            let semantic = state.functions.pop()?;
            state.function_spans.pop();
            state.function_producers.pop();
            let return_type = state
                .plan
                .declarations()
                .find(|declaration| declaration.identity() == semantic.identity())
                .and_then(|declaration| declaration.signature().return_type())
                .cloned();
            (semantic, return_type)
        };
        let mut error = None;
        if let (Some(value), Some(ty)) = (ret_id, return_type.as_ref()) {
            match semantic.values().get(&value) {
                Some(existing) if canonical_types_compatible(existing, ty) => {}
                Some(_) => {
                    error = Some(format!(
                        "return value {value} has conflicting semantic type"
                    ));
                }
                None => {
                    error = Some(format!("return value {value} has no checked producer type"));
                }
            }
        } else if ret_id.is_some() && return_type.is_none() {
            error = Some("unit function produced an untyped return value".to_string());
        } else if ret_id.is_none() && return_type.is_some() {
            error = Some("typed function produced no return value".to_string());
        }
        if let Some(error) = error {
            self.canonical_refuse(error);
        }
        Some(semantic)
    }

    pub(super) fn canonical_finish_function(
        &mut self,
        ret_id: Option<ValueId>,
    ) -> Option<FunctionSemanticTypes> {
        self.canonical
            .is_some()
            .then(|| self.end_canonical_function(ret_id))
            .flatten()
    }

    pub(super) fn emit_canonical_call(
        &self,
        ir: &mut IRModule,
        dst: ValueId,
        name: String,
        args: Vec<ValueId>,
        identity: Option<crate::types::FunctionIdentity>,
    ) -> bool {
        if self.canonical.is_none() {
            return false;
        }
        ir.instrs.push(Instr::Call {
            dst,
            name,
            args,
            resolved_callee: identity.map(Box::new),
        });
        true
    }
}

/// Lower through the opt-in source-to-canonical bridge. The plan is consumed
/// at AST call/definition sites; it is never reconstructed from emitted names.
pub fn lower_to_ir_with_canonical_plan(
    module: &crate::ast::Module,
    limits: MaterializationLimits,
    plan: CanonicalBindingPlan,
) -> Result<IRModule, CanonicalBindingError> {
    let mut context = LoweringContext::new(limits);
    context.canonical = Some(CanonicalLoweringState::new(plan));
    let mut ir = lower_module(module, &mut context);
    if let Some(reason) = context.refusal.as_ref() {
        return Err(CanonicalBindingError::Lowering(reason.to_string()));
    }
    let Some(mut state) = context.canonical.take() else {
        return Err(CanonicalBindingError::Lowering(
            "canonical lowering context was not initialized".to_string(),
        ));
    };
    if let Some(reason) = state.refusal.take() {
        return Err(CanonicalBindingError::Lowering(reason));
    }
    if let Some((span, _)) = state.plan.first_unconsumed_call() {
        return Err(CanonicalBindingError::Lowering(format!(
            "canonical call occurrence at {span:?} was not consumed"
        )));
    }
    if let Some((span, _)) = state.plan.first_unconsumed_function() {
        return Err(CanonicalBindingError::Lowering(format!(
            "canonical function occurrence at {span:?} was not consumed"
        )));
    }
    let mut bundle = state.plan.into_bundle();
    for (value, ty) in state.module_values {
        bundle
            .set_module_value_type(value, ty)
            .map_err(|error| CanonicalBindingError::Verification(error.to_string()))?;
    }
    if !state.functions.is_empty() {
        return Err(CanonicalBindingError::Lowering(
            "canonical function scope remained open after lowering".to_string(),
        ));
    }
    ir.canonical_types = Some(Box::new(bundle));
    crate::ir::verify_canonical_metadata(&ir)
        .map_err(|error| CanonicalBindingError::Verification(error.to_string()))?;
    Ok(ir)
}
