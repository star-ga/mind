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
