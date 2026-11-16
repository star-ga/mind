// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Closure of an exported type alias over owner-qualified alias targets.

use std::collections::{BTreeMap, BTreeSet};

use super::{canonical_module, qualifier_matches};
use crate::ast::TypeAnn;

pub(super) fn resolve_qualified_alias_target(
    target: &TypeAnn,
    owner: &str,
    raw: &BTreeMap<String, TypeAnn>,
    imports: &BTreeMap<String, Vec<String>>,
    exported: &BTreeSet<String>,
    resolving: &mut BTreeSet<String>,
) -> Option<TypeAnn> {
    let TypeAnn::Named(name) = target else {
        return Some(target.clone());
    };
    let Some((qualifier, alias_name)) = name.rsplit_once('.') else {
        return Some(target.clone());
    };
    let mut owners = BTreeSet::new();
    if qualifier_matches(qualifier, owner) {
        owners.insert(owner.to_string());
    }
    for import in imports.get(owner).into_iter().flatten() {
        let imported_owner = canonical_module(import);
        if qualifier_matches(qualifier, &imported_owner) {
            owners.insert(imported_owner);
        }
    }
    let resolved_owner = (owners.len() == 1)
        .then(|| owners.into_iter().next())
        .flatten()?;
    let source_key = format!("{resolved_owner}.{alias_name}");
    if resolved_owner != owner && !exported.contains(&source_key) {
        return Some(target.clone());
    }
    let nested = raw.get(&source_key)?;
    if !resolving.insert(source_key.clone()) {
        return None;
    }
    let result =
        resolve_qualified_alias_target(nested, &resolved_owner, raw, imports, exported, resolving);
    resolving.remove(&source_key);
    result
}

impl super::Registry {
    /// Distinct types the bare `name` denotes through `imports`. Each import
    /// counts through its direct key or, failing that, the terminal-segment
    /// fallback. An alias that re-exports a type denotes that type, so two
    /// imports that name one type are one name, not an ambiguity.
    pub(super) fn bare_denotations(&self, name: &str, imports: &[String]) -> BTreeSet<String> {
        imports
            .iter()
            .filter_map(|import| {
                let direct = format!("{}.{name}", canonical_module(import));
                if self.exported_types.contains(&direct) {
                    Some(direct)
                } else {
                    self.imported_exported_source(import, name, std::slice::from_ref(import))
                }
            })
            .map(|source| self.denotation(&source))
            .collect()
    }

    /// What the declared `owner.Name` denotes: an enum its collision-safe key,
    /// an alias the declared type its target names in the alias owner's scope,
    /// any other type itself. An alias whose target names no declared type
    /// (a scalar, a tensor, an unresolved or cyclic name) denotes itself, so it
    /// still counts as its own owner, as before.
    fn denotation(&self, source: &str) -> String {
        let mut seen = BTreeSet::new();
        let mut current = source.to_string();
        loop {
            if let Some(key) = self.type_keys.get(&current) {
                return key.clone();
            }
            let Some(target) = self.alias_written.get(&current) else {
                return current;
            };
            if !seen.insert(current.clone()) {
                return current;
            }
            let TypeAnn::Named(name) = target else {
                return current;
            };
            let owner = current.rsplit_once('.').map_or("", |(owner, _)| owner);
            let local = format!("{owner}.{name}");
            let next = if local != current && self.declared_types.contains(&local) {
                Some(local)
            } else if self.is_canonical_type(name) {
                // The parser already rewrote a qualified enum target to its key.
                return name.clone();
            } else if name.contains('.') {
                let imports = self
                    .module_imports
                    .get(owner)
                    .map_or(&[][..], Vec::as_slice);
                self.type_source_in(name, imports, Some(owner))
            } else {
                None
            };
            match next {
                Some(next) => current = next,
                None => return current,
            }
        }
    }
}

impl super::Registry {
    /// Enum names a bare key could not name alone: one module declares the
    /// enum, and another declared type of that name (a struct, or an alias of
    /// a different type) exists. Such an enum takes its owner-qualified key,
    /// so a module-qualified spelling never resolves to a bare key that a bare
    /// spelling would read as ambiguous. Reads `self` with the bare keys
    /// installed provisionally; `rebuild` installs the final keys afterwards.
    pub(super) fn shadowed_enum_names<T>(
        &mut self,
        by_name: &BTreeMap<String, Vec<(String, T)>>,
    ) -> BTreeSet<String> {
        for (name, owners) in by_name {
            if let [(owner, _)] = owners.as_slice() {
                self.type_keys
                    .insert(format!("{owner}.{name}"), name.clone());
            }
        }
        let shadowed = self
            .declared_types
            .iter()
            .filter_map(|source| {
                let (_, name) = source.rsplit_once('.')?;
                let bare_enum = by_name.get(name).is_some_and(|owners| owners.len() == 1);
                (bare_enum
                    && !self.type_keys.contains_key(source)
                    && self.denotation(source) != name)
                    .then(|| name.to_string())
            })
            .collect();
        self.type_keys.clear();
        shadowed
    }
}
