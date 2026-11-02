// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The std-surface names a module's resolver sees.
//!
//! Every module sees the names of the implicit std modules
//! (`IMPLICIT_STD_MODULES`). A module that imports another bundled std module
//! also sees that module's names, and those of the std modules it imports in
//! turn. The name of a module nothing imported stays unresolved, exactly as
//! before that module was bundled, and resolving it costs no parse.

use std::sync::OnceLock;

use super::{FxSet, fn_values::only_fns};
use crate::ast::Module;

/// The name-level facts one std module, or a union of several, declares.
#[derive(Default)]
pub(super) struct StdNames {
    /// Every top-level declaration name: the resolvable set.
    pub(super) names: FxSet,
    /// Function and extern-function names.
    fns: FxSet,
    /// Non-function declaration names plus enum variant names.
    pub(super) values: FxSet,
}

impl StdNames {
    #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
    fn of(module: &Module) -> Self {
        let mut out = Self::default();
        super::collect_decl_names_into(module, &mut out.names);
        super::collect_non_fn_decl_names(module, &mut out.values, &mut out.fns);
        let mut enums = super::FxMap::default();
        super::collect_enum_variants(module, &mut enums);
        out.values.extend(enums.into_values().flatten());
        out
    }

    #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
    fn absorb(&mut self, other: &Self) {
        self.names.extend(other.names.iter().cloned());
        self.fns.extend(other.fns.iter().cloned());
        self.values.extend(other.values.iter().cloned());
    }

    /// Names this set declares as functions and as nothing else, with `hidden`
    /// (values declared elsewhere) also excluded.
    pub(super) fn fn_only(&self, hidden: &FxSet) -> FxSet {
        only_fns(&self.fns, &self.values)
            .into_iter()
            .filter(|name| !hidden.contains(name))
            .collect()
    }
}

/// The implicit modules' names and their function-only subset, built once.
pub(super) struct Implicit {
    pub(super) names: StdNames,
    pub(super) fn_only: FxSet,
}

pub(super) fn implicit() -> &'static Implicit {
    static CACHE: OnceLock<Implicit> = OnceLock::new();
    CACHE.get_or_init(|| {
        let names = implicit_union();
        Implicit {
            fn_only: names.fn_only(&FxSet::default()),
            names,
        }
    })
}

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
fn implicit_union() -> StdNames {
    let mut all = StdNames::default();
    for module in crate::project::stdlib::IMPLICIT_STD_MODULES {
        if let Some(names) = module_names(module) {
            all.absorb(names);
        }
    }
    all
}

#[cfg(not(any(feature = "cross-module-imports", feature = "std-surface")))]
fn implicit_union() -> StdNames {
    StdNames::default()
}

/// The names `module`'s imports of non-implicit std modules bring in, or
/// `None` when it imports none (nothing is parsed then).
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
pub(super) fn imported(module: &Module) -> Option<StdNames> {
    let extras = crate::project::stdlib::imported_std_extras(module);
    if extras.is_empty() {
        return None;
    }
    let mut all = StdNames::default();
    for name in extras {
        if let Some(names) = module_names(name) {
            all.absorb(names);
        }
    }
    Some(all)
}

#[cfg(not(any(feature = "cross-module-imports", feature = "std-surface")))]
pub(super) fn imported(module: &Module) -> Option<StdNames> {
    let _ = module;
    None
}

/// One bundled module's names, built once per process from its cached AST.
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
fn module_names(module: &str) -> Option<&'static StdNames> {
    use crate::project::stdlib::{STDLIB_MIND_SOURCES, bundled_module_ast};
    const COUNT: usize = STDLIB_MIND_SOURCES.len();
    static CACHE: [OnceLock<Option<StdNames>>; COUNT] = [const { OnceLock::new() }; COUNT];
    let index = STDLIB_MIND_SOURCES
        .iter()
        .position(|(name, _)| *name == module)?;
    CACHE[index]
        .get_or_init(|| bundled_module_ast(module).map(StdNames::of))
        .as_ref()
}
