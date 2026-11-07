// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The declared payload of an enum variant, keyed as the project registry
//! keys it.
//!
//! A project build analyses one module at a time. A module that stores a
//! string into the payload of another module's variant records it under the
//! variant's key, and the declaring module reads that record. Both use the
//! registry key: `XmlEvent::Text`, or `crate.events::XmlEvent::Text` when
//! another module declares an enum of the same name. The payload types of an
//! imported variant come from the qualified registry, where the project
//! rebuild keeps them (it clears the legacy table, which other builds fill).

use std::collections::BTreeMap;

use crate::ast::EnumVariant;

use super::{Decls, Fact, Payload};

/// The registry key of the enum the module being analysed declares as
/// `name`: in a project build a colliding enum is keyed by its owner.
fn local_enum_key(name: &str) -> String {
    #[cfg(feature = "cross-module-imports")]
    return crate::qualified_enums::current_enum_key(name);
    #[cfg(not(feature = "cross-module-imports"))]
    {
        name.to_string()
    }
}

/// The registry key an `Enum::Variant` spelling of an imported enum names in
/// the module being analysed, resolved through its imports as the parser
/// resolves it. The project summary walks ASTs parsed before the registry
/// existed, which keep the source spelling of an owner-keyed variant.
fn imported_variant_key(path: &str) -> Option<String> {
    #[cfg(feature = "cross-module-imports")]
    {
        let current = crate::qualified_enums::current_module_path()?;
        crate::ir::with_global_enums(|g| {
            let imports = g.qualified.module_imports.get(&current)?;
            g.qualified
                .resolve_variant_in(path, imports, Some(&current))
        })
    }
    #[cfg(not(feature = "cross-module-imports"))]
    {
        let _ = path;
        None
    }
}

impl Decls {
    /// The variant a constructor or pattern names (`Enum::Variant`), with its
    /// declared payload types and the field names of a struct variant. `None`
    /// when the variant is unknown or a bare name is ambiguous.
    pub(in super::super) fn variant_payload(&self, path: &str) -> Option<Payload> {
        // The parser rewrites an imported spelling to its registry key.
        if let Some(payload) = self.global_payload(|key| key == path) {
            return Some(payload);
        }
        let of = |owner: &str, v: &EnumVariant| Payload {
            key: format!("{}::{}", local_enum_key(owner), v.name),
            types: v.payload.clone(),
            names: v.field_names.clone(),
        };
        let Some((head, variant)) = path.rsplit_once("::") else {
            let mut local = self.enums.iter().flat_map(|(owner, variants)| {
                variants
                    .iter()
                    .filter(|v| v.name == path)
                    .map(move |v| (owner, v))
            });
            return match (local.next(), local.next()) {
                (Some((owner, v)), None) => Some(of(owner, v)),
                (Some(_), Some(_)) => None,
                (None, _) => {
                    self.global_payload(|key| key.rsplit_once("::").is_some_and(|(_, v)| v == path))
                }
            };
        };
        let owner = head.rsplit('.').next().unwrap_or(head);
        if let Some(variants) = self.enums.get(owner) {
            return variants
                .iter()
                .find(|v| v.name == variant)
                .map(|v| of(owner, v));
        }
        let key = imported_variant_key(path).unwrap_or_else(|| format!("{owner}::{variant}"));
        self.global_payload(|k| k == key)
    }

    /// The payload of the one registered variant whose key satisfies `wanted`,
    /// in the qualified registry or the legacy table.
    fn global_payload(&self, wanted: impl Fn(&str) -> bool) -> Option<Payload> {
        crate::ir::with_global_enums(|g| {
            #[cfg(feature = "cross-module-imports")]
            let qualified = [(&g.qualified.payload_types, &g.qualified.struct_field_names)];
            #[cfg(not(feature = "cross-module-imports"))]
            let qualified = [];
            let legacy = [(&g.payload_types, &g.struct_field_names)];
            let mut hits = BTreeMap::new();
            for (types, names) in qualified.into_iter().chain(legacy) {
                for (key, types) in types.iter().filter(|(k, _)| wanted(k.as_str())) {
                    hits.entry(key).or_insert((types, names.get(key)));
                }
            }
            let mut hits = hits.into_iter();
            let (Some((key, (types, names))), None) = (hits.next(), hits.next()) else {
                return None;
            };
            Some(Payload {
                key: key.clone(),
                types: types.clone(),
                names: names.cloned().unwrap_or_default(),
            })
        })
    }

    /// The fact of position `index` of `payload`: its declared type, unless a
    /// constructor stores a string there.
    pub(in super::super) fn payload_fact(&self, payload: &Payload, index: usize) -> Fact {
        let stored = self.contradicted.payloads.get(&payload.key);
        match payload.types.get(index) {
            Some(_) if stored.is_some_and(|at| at.contains(&index)) => Fact::MaybeStr,
            Some(ty) => self.typed_fact(ty),
            None => Fact::Unknown,
        }
    }
}
