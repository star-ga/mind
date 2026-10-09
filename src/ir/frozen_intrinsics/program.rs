// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! The whole program a native build admits: the user module plus every linked
//! std seed function, as one set of function scopes.
//!
//! The frozen compiler executes the user region and the retained std seed as ONE
//! program, so a fact one function relies on about another — "my parameter is an
//! i64", "this callee returns an i64", "this callee allocates" — can only be
//! proven by looking at both ends of the call. Each lowered function is its own
//! SSA namespace (`ValueId`s restart per function), so a value is identified by
//! its scope index AND its id.

use std::collections::BTreeMap;

use crate::ir::{IRModule, Instr, ValueId};

/// A value in the program: the scope that defines it and its id in that scope.
pub(super) type Key = (usize, ValueId);

/// The function a scope belongs to.
#[derive(Clone, Copy)]
pub(super) struct FnInfo<'m> {
    pub(super) name: &'m str,
    pub(super) n_params: usize,
    pub(super) ret_id: Option<ValueId>,
}

/// One SSA scope: a module's top-level instructions (`func: None`) or one
/// function body, with the module that lowered it (its declared signatures and
/// struct names). `part` 0 is the user program; every later part is a linked
/// std seed function.
#[derive(Clone, Copy)]
pub(super) struct Scope<'m> {
    pub(super) part: usize,
    pub(super) module: &'m IRModule,
    pub(super) body: &'m [Instr],
    pub(super) func: Option<FnInfo<'m>>,
}

pub(crate) struct Program<'m> {
    pub(super) scopes: Vec<Scope<'m>>,
    /// Function name -> its one defining scope, or `None` when more than one
    /// scope defines the name (nothing about an ambiguous name is trusted).
    by_name: BTreeMap<&'m str, Option<usize>>,
}

impl<'m> Program<'m> {
    pub(crate) fn new(parts: &[&'m IRModule]) -> Self {
        let mut scopes = Vec::new();
        for (part, module) in parts.iter().enumerate() {
            scopes.push(Scope {
                part,
                module,
                body: &module.instrs,
                func: None,
            });
            collect_fn_scopes(part, module, &module.instrs, &mut scopes);
        }
        let mut by_name: BTreeMap<&'m str, Option<usize>> = BTreeMap::new();
        for (i, scope) in scopes.iter().enumerate() {
            if let Some(f) = scope.func {
                by_name
                    .entry(f.name)
                    .and_modify(|slot| *slot = None)
                    .or_insert(Some(i));
            }
        }
        Program { scopes, by_name }
    }

    /// The one scope defining `name`, if exactly one does.
    pub(super) fn callee(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied().flatten()
    }

    /// Does more than one scope define `name`?
    pub(super) fn is_ambiguous(&self, name: &str) -> bool {
        matches!(self.by_name.get(name), Some(None))
    }

    /// Is parameter `index` of the function in scope `s` declared an i64 slot,
    /// for a call passing `arity` arguments? Trusted only when the call passes
    /// exactly the function's parameters and its declared signature lists exactly
    /// that many (a signature recorded without an implicit receiver would shift
    /// every index onto its neighbour's type).
    pub(super) fn param_is_slot(&self, s: usize, index: usize, arity: usize) -> bool {
        let scope = &self.scopes[s];
        let Some(f) = scope.func else { return false };
        f.n_params == arity
            && super::operand_types::declared_param(scope.module, f.name, index, f.n_params)
                .is_some_and(|t| super::operand_types::type_is_i64_slot(scope.module, t))
    }

    /// Is the function in scope `s` declared to return an i64 slot?
    pub(super) fn ret_is_slot(&self, s: usize) -> bool {
        let scope = &self.scopes[s];
        scope.func.is_some_and(|f| {
            super::operand_types::declared_ret(scope.module, f.name)
                .is_some_and(|t| super::operand_types::type_is_i64_slot(scope.module, t))
        })
    }
}

/// Every `FnDef` at any depth below `instrs`, each as its own scope.
fn collect_fn_scopes<'m>(
    part: usize,
    module: &'m IRModule,
    instrs: &'m [Instr],
    out: &mut Vec<Scope<'m>>,
) {
    for instr in instrs {
        if let Instr::FnDef {
            name,
            params,
            body,
            ret_id,
            ..
        } = instr
        {
            out.push(Scope {
                part,
                module,
                body,
                func: Some(FnInfo {
                    name,
                    n_params: params.len(),
                    ret_id: *ret_id,
                }),
            });
            collect_fn_scopes(part, module, body, out);
            continue;
        }
        #[cfg(feature = "std-surface")]
        for nested in crate::ir::instr_bodies(instr) {
            collect_fn_scopes(part, module, nested, out);
        }
    }
}
