// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Lexical module identity for the tree evaluator's linked test modules.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::ast::Span;

use super::{EvalError, UserFn, fn_table_lookup};

const OWNER_PREFIX: &str = "\0mind-eval-owner\0";

pub(crate) fn owned_symbol(owner: &str, name: &str) -> String {
    format!("{OWNER_PREFIX}{owner}\0{name}")
}

pub(super) fn symbol_owner(name: &str) -> Option<&str> {
    name.strip_prefix(OWNER_PREFIX)
        .and_then(|rest| rest.split_once('\0'))
        .map(|(owner, _)| owner)
}

pub(super) fn symbol_display_name(name: &str) -> &str {
    name.strip_prefix(OWNER_PREFIX)
        .and_then(|rest| rest.split_once('\0'))
        .map_or(name, |(_, symbol)| symbol)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BindingKind {
    Call,
    Value,
}

#[derive(Clone, Default)]
pub(crate) struct Bindings {
    calls: HashMap<(String, Span), String>,
    values: HashMap<(String, Span), String>,
    /// Bare (unqualified) calls a module may make into the exported functions
    /// of the sibling modules it imports, keyed by `(owner, name)`. `None`
    /// records an ambiguity: two imported siblings export the same name.
    imported_calls: HashMap<(String, String), Option<String>>,
    /// Bare (unqualified) constants a module may reference from the exported
    /// constants of the sibling modules it imports, keyed by `(owner, name)`.
    /// `None` records an ambiguity: two imported siblings export the same name.
    imported_values: HashMap<(String, String), Option<String>>,
}

impl Bindings {
    #[cfg(any(feature = "cross-module-imports", test))]
    pub(crate) fn insert(
        &mut self,
        owner: &str,
        span: Span,
        kind: BindingKind,
        target_owner: &str,
        symbol: &str,
    ) {
        let key = (owner.to_string(), span);
        let target = owned_symbol(target_owner, symbol);
        match kind {
            BindingKind::Call => self.calls.insert(key, target),
            BindingKind::Value => self.values.insert(key, target),
        };
    }

    /// Record that `owner` imports `target_owner`, which exports `symbol`, so
    /// a bare `symbol(..)` call in `owner` reaches it. A second, different
    /// exporter of the same name makes the bare call ambiguous.
    #[cfg(any(feature = "cross-module-imports", test))]
    pub(crate) fn insert_imported_call(&mut self, owner: &str, target_owner: &str, symbol: &str) {
        let target = owned_symbol(target_owner, symbol);
        self.imported_calls
            .entry((owner.to_string(), symbol.to_string()))
            .and_modify(|slot| {
                if slot.as_deref() != Some(target.as_str()) {
                    *slot = None;
                }
            })
            .or_insert(Some(target));
    }

    /// Record that `owner` imports `target_owner`, which exports the constant
    /// `symbol`, so a bare `symbol` reference in `owner` reaches it. A second,
    /// different exporter of the same name makes the bare reference ambiguous.
    #[cfg(any(feature = "cross-module-imports", test))]
    pub(crate) fn insert_imported_value(&mut self, owner: &str, target_owner: &str, symbol: &str) {
        let target = owned_symbol(target_owner, symbol);
        self.imported_values
            .entry((owner.to_string(), symbol.to_string()))
            .and_modify(|slot| {
                if slot.as_deref() != Some(target.as_str()) {
                    *slot = None;
                }
            })
            .or_insert(Some(target));
    }

    /// Lower-priority variant of [`Self::insert_imported_call`] used for
    /// bundled `std` exports: a name already bound (the owner's own
    /// declaration or a sibling's export) keeps its binding, so a std export
    /// can never steal or ambiguify a source-resolved callee.
    #[cfg(any(feature = "cross-module-imports", test))]
    pub(crate) fn insert_std_imported_call(
        &mut self,
        owner: &str,
        target_owner: &str,
        symbol: &str,
    ) {
        let key = (owner.to_string(), symbol.to_string());
        if self.imported_calls.contains_key(&key) {
            return;
        }
        self.imported_calls
            .insert(key, Some(owned_symbol(target_owner, symbol)));
    }
}

/// Resolution of a bare call through the current owner's imports.
pub(super) enum ImportedCall {
    /// No imported sibling exports this name.
    None,
    /// Exactly one imported sibling exports it.
    Unique(String),
    /// More than one imported sibling exports it.
    Ambiguous,
}

pub(super) fn imported_call(name: &str) -> ImportedCall {
    let Some(owner) = current_owner() else {
        return ImportedCall::None;
    };
    BINDINGS.with(|bindings| {
        match bindings
            .borrow()
            .imported_calls
            .get(&(owner, name.to_string()))
        {
            None => ImportedCall::None,
            Some(Some(target)) => ImportedCall::Unique(target.clone()),
            Some(None) => ImportedCall::Ambiguous,
        }
    })
}

/// Resolution of a bare constant through the current owner's imports.
pub(super) fn imported_value(name: &str) -> ImportedCall {
    let Some(owner) = current_owner() else {
        return ImportedCall::None;
    };
    BINDINGS.with(|bindings| {
        match bindings
            .borrow()
            .imported_values
            .get(&(owner, name.to_string()))
        {
            None => ImportedCall::None,
            Some(Some(target)) => ImportedCall::Unique(target.clone()),
            Some(None) => ImportedCall::Ambiguous,
        }
    })
}

/// Candidate `CONST_TABLE` keys for a bare constant, in the order the type
/// checker resolves it: the owner's own qualified key first, then the unique
/// export of one of the owner's imports, then the bare key (the single-
/// translation-unit table). A name exported by two imports is an ambiguity and
/// is refused, never resolved to either exporter.
pub(super) fn const_lookup_keys(name: &str) -> Result<Vec<String>, EvalError> {
    let mut keys = Vec::with_capacity(3);
    if let Some(owner) = current_owner() {
        keys.push(owned_symbol(&owner, name));
    }
    match imported_value(name) {
        ImportedCall::None => {}
        ImportedCall::Unique(target) => keys.push(target),
        ImportedCall::Ambiguous => {
            return Err(EvalError::UnsupportedMsg(format!(
                "imported constant `{name}` is ambiguous: more than one imported module exports it"
            )));
        }
    }
    keys.push(name.to_string());
    Ok(keys)
}

/// Resolve a bare `callee` in the current owner's scope, in the order the type
/// checker resolves it: the owner's own definition first, then the unique
/// export of one of the owner's imports. A name exported by two imports is an
/// ambiguity and is refused, never resolved to either exporter.
pub(super) fn owner_scoped_fn(callee: &str) -> Result<Option<UserFn>, EvalError> {
    let Some(owner) = current_owner() else {
        return Ok(None);
    };
    if let Some(local) = fn_table_lookup(&owned_symbol(&owner, callee)) {
        return Ok(Some(local));
    }
    match imported_call(callee) {
        ImportedCall::Unique(target) => Ok(fn_table_lookup(&target)),
        ImportedCall::Ambiguous => Err(EvalError::UnsupportedMsg(format!(
            "call to `{callee}` is ambiguous: more than one imported module exports it"
        ))),
        ImportedCall::None => Ok(None),
    }
}

thread_local! {
    static BINDINGS: RefCell<Bindings> = RefCell::new(Bindings::default());
    static OWNER: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub(super) fn current_owner() -> Option<String> {
    OWNER.with(|owner| owner.borrow().clone())
}

pub(super) fn bound_symbol(span: Span, kind: BindingKind) -> Option<String> {
    let owner = current_owner()?;
    BINDINGS.with(|bindings| {
        let bindings = bindings.borrow();
        match kind {
            BindingKind::Call => bindings.calls.get(&(owner, span)).cloned(),
            BindingKind::Value => bindings.values.get(&(owner, span)).cloned(),
        }
    })
}

pub(crate) struct BindingsGuard {
    previous_bindings: Bindings,
    previous_owner: Option<String>,
}

impl BindingsGuard {
    pub(crate) fn install(bindings: Bindings, owner: Option<String>) -> Self {
        let previous_bindings =
            BINDINGS.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), bindings));
        let previous_owner = OWNER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), owner));
        Self {
            previous_bindings,
            previous_owner,
        }
    }
}

impl Drop for BindingsGuard {
    fn drop(&mut self) {
        BINDINGS.with(|slot| *slot.borrow_mut() = std::mem::take(&mut self.previous_bindings));
        OWNER.with(|slot| *slot.borrow_mut() = self.previous_owner.take());
    }
}

pub(super) struct OwnerGuard(Option<String>);

impl OwnerGuard {
    pub(super) fn enter(owner: Option<&str>) -> Self {
        let previous = OWNER
            .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), owner.map(str::to_string)));
        Self(previous)
    }
}

impl Drop for OwnerGuard {
    fn drop(&mut self) {
        OWNER.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_key_includes_owner_and_owner_guards_restore_on_unwind() {
        let span = Span::new(7, 11);
        let mut bindings = Bindings::default();
        bindings.insert("crate.a", span, BindingKind::Call, "crate.left", "value");
        bindings.insert("crate.b", span, BindingKind::Call, "crate.right", "value");
        let _bindings = BindingsGuard::install(bindings, Some("crate.a".into()));
        assert_eq!(
            bound_symbol(span, BindingKind::Call),
            Some(owned_symbol("crate.left", "value"))
        );
        let _ = std::panic::catch_unwind(|| {
            let _owner = OwnerGuard::enter(Some("crate.b"));
            assert_eq!(
                bound_symbol(span, BindingKind::Call),
                Some(owned_symbol("crate.right", "value"))
            );
            panic!("exercise unwind restoration");
        });
        assert_eq!(current_owner().as_deref(), Some("crate.a"));
    }

    #[test]
    fn bare_imported_values_are_owner_scoped_and_ambiguity_is_kept() {
        let mut bindings = Bindings::default();
        bindings.insert_imported_value("crate.a", "crate.left", "LIMIT");
        bindings.insert_imported_value("crate.a", "crate.left", "LIMIT");
        bindings.insert_imported_value("crate.b", "crate.left", "LIMIT");
        bindings.insert_imported_value("crate.b", "crate.right", "LIMIT");
        let _bindings = BindingsGuard::install(bindings, Some("crate.a".into()));
        assert!(matches!(
            imported_value("LIMIT"),
            ImportedCall::Unique(target) if target == owned_symbol("crate.left", "LIMIT")
        ));
        assert!(matches!(imported_value("other"), ImportedCall::None));
        let keys = const_lookup_keys("LIMIT").expect("unambiguous owner");
        assert_eq!(keys[0], owned_symbol("crate.a", "LIMIT"));
        assert_eq!(keys[1], owned_symbol("crate.left", "LIMIT"));
        assert_eq!(keys[2], "LIMIT");
        let _ = std::panic::catch_unwind(|| {
            let _owner = OwnerGuard::enter(Some("crate.b"));
            assert!(matches!(imported_value("LIMIT"), ImportedCall::Ambiguous));
            assert!(const_lookup_keys("LIMIT").is_err());
        });
    }

    #[test]
    fn bare_imported_calls_are_owner_scoped_and_ambiguity_is_kept() {
        let mut bindings = Bindings::default();
        bindings.insert_imported_call("crate.a", "crate.left", "value");
        bindings.insert_imported_call("crate.a", "crate.left", "value");
        bindings.insert_imported_call("crate.b", "crate.left", "value");
        bindings.insert_imported_call("crate.b", "crate.right", "value");
        let _bindings = BindingsGuard::install(bindings, Some("crate.a".into()));
        assert!(matches!(
            imported_call("value"),
            ImportedCall::Unique(target) if target == owned_symbol("crate.left", "value")
        ));
        assert!(matches!(imported_call("other"), ImportedCall::None));
        let _owner = OwnerGuard::enter(Some("crate.b"));
        assert!(matches!(imported_call("value"), ImportedCall::Ambiguous));
        let _none = OwnerGuard::enter(None);
        assert!(matches!(imported_call("value"), ImportedCall::None));
    }
}
