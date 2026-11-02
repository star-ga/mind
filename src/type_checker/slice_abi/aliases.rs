// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Alias-transparent types for slice-call provenance.
//!
//! A type alias names its target type, so `array<Q>` with `type Q = i32` is an
//! `array<i32>`. The provenance pass compares element types structurally;
//! comparing source spellings instead refused a valid `array<Q>` argument to a
//! `&[Q]` parameter. Every type this pass reads (parameters, `let`
//! annotations, call signatures, struct fields, results) is therefore
//! canonicalized through the checked file's aliases and the project's exported
//! aliases first. The pass descends into each module item in isolation, so the
//! file's alias table is installed once by the file-level entry point.

use std::cell::RefCell;
use std::rc::Rc;

use crate::ast::{Node, TypeAnn};
use crate::eval::type_aliases::LocalTypeAliases;

thread_local! {
    static FILE_ALIASES: RefCell<Option<Rc<LocalTypeAliases>>> = const { RefCell::new(None) };
}

/// Keeps one checked file's aliases visible to the slice pass and restores
/// the enclosing scope on drop.
pub(crate) struct AliasScope(Option<Rc<LocalTypeAliases>>);

impl AliasScope {
    pub(crate) fn install(items: &[Node]) -> Self {
        let aliases = Rc::new(LocalTypeAliases::new(items));
        Self(FILE_ALIASES.with(|cell| cell.borrow_mut().replace(aliases)))
    }
}

impl Drop for AliasScope {
    fn drop(&mut self) {
        FILE_ALIASES.with(|cell| *cell.borrow_mut() = self.0.take());
    }
}

/// `ty` with every visible alias replaced by its target type.
pub(super) fn canonical(ty: &TypeAnn) -> TypeAnn {
    FILE_ALIASES.with(|cell| match cell.borrow().as_ref() {
        Some(aliases) => aliases.resolve_with_imports(ty),
        None => ty.clone(),
    })
}

/// The project registry key (`owner.Name`) of a qualified struct spelling.
#[cfg(feature = "cross-module-imports")]
pub(super) fn qualified_type_source(name: &str) -> Option<String> {
    FILE_ALIASES.with(|cell| cell.borrow().as_ref()?.imported_type_source(name))
}
