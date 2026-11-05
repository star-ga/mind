// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Owner-aware canonical spelling of project types.
//!
//! Module `defs` names its struct `Bank`; a module that imports `defs` names
//! the same struct `defs.Bank`. A type read from an imported signature or
//! field is spelled in its owner's scope, so comparing spellings refused a
//! value of the right type. This pass rewrites every project alias to its
//! target and every project struct, enum or alias name to its source key
//! (`crate.defs.Bank`), so both spellings meet. A type that the reading
//! module declares keeps its bare spelling, because the checker's local
//! tables key it bare. A name the registry does not know keeps its spelling.

use std::collections::BTreeSet;

use super::{Registry, canonical_module, current_module_path};
use crate::ast::TypeAnn;

/// `ty` as the current module (importing `imports`) writes it: project
/// aliases resolved, other project types named by source key.
pub(crate) fn canonical_type_here(ty: &TypeAnn, imports: &[String]) -> TypeAnn {
    let current = current_module_path();
    crate::ir::with_global_enums(|global| {
        global
            .qualified
            .canonical_type_in(ty, current.as_deref(), imports, current.as_deref())
    })
}

/// `ty` as module `owner` writes it (an imported signature or field type),
/// spelled for the current module.
pub(crate) fn canonical_type_from(ty: &TypeAnn, owner: &str) -> TypeAnn {
    let current = current_module_path();
    crate::ir::with_global_enums(|global| {
        let registry = &global.qualified;
        registry.canonical_type_in(
            ty,
            Some(owner),
            registry.imports_of(owner),
            current.as_deref(),
        )
    })
}

/// Every `(owner, declared type)` of a module-level constant named `name`.
/// Lowering inlines a cross-module constant by its bare name, so each module
/// that declares one is a candidate.
pub(crate) fn imported_const_declared_types(name: &str) -> Vec<(String, TypeAnn)> {
    crate::ir::with_global_enums(|global| {
        global
            .qualified
            .const_types
            .iter()
            .filter_map(|(key, ty)| {
                let (owner, declared) = key.rsplit_once('.')?;
                (declared == name).then(|| (owner.to_string(), ty.clone()))
            })
            .collect()
    })
}

impl Registry {
    /// `ty` as written in module `scope` (which imports `imports`), spelled
    /// for the module `reader`. A cyclic alias keeps the original type.
    pub(super) fn canonical_type_in(
        &self,
        ty: &TypeAnn,
        scope: Option<&str>,
        imports: &[String],
        reader: Option<&str>,
    ) -> TypeAnn {
        self.canonical_rec(ty, scope, imports, reader, &mut BTreeSet::new())
            .unwrap_or_else(|| ty.clone())
    }

    /// Imports recorded for `owner` when the registry was built.
    pub(super) fn imports_of(&self, owner: &str) -> &[String] {
        self.module_imports
            .get(owner)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn canonical_rec(
        &self,
        ty: &TypeAnn,
        scope: Option<&str>,
        imports: &[String],
        reader: Option<&str>,
        resolving: &mut BTreeSet<String>,
    ) -> Option<TypeAnn> {
        let mut rec =
            |inner: &TypeAnn| self.canonical_rec(inner, scope, imports, reader, resolving);
        Some(match ty {
            TypeAnn::Named(name) => {
                return self.canonical_name(name, scope, imports, reader, resolving);
            }
            TypeAnn::Slice { mutable, element } => TypeAnn::Slice {
                mutable: *mutable,
                element: Box::new(rec(element)?),
            },
            TypeAnn::Array { element, length } => TypeAnn::Array {
                element: Box::new(rec(element)?),
                length: *length,
            },
            TypeAnn::Ref { mutable, target } => TypeAnn::Ref {
                mutable: *mutable,
                target: Box::new(rec(target)?),
            },
            TypeAnn::Generic { name, args } => TypeAnn::Generic {
                name: name.clone(),
                args: args.iter().map(&mut rec).collect::<Option<Vec<_>>>()?,
            },
            TypeAnn::Tuple { elements } => TypeAnn::Tuple {
                elements: elements.iter().map(&mut rec).collect::<Option<Vec<_>>>()?,
            },
            TypeAnn::SparseTensor {
                layout,
                element,
                shape,
            } => TypeAnn::SparseTensor {
                layout: *layout,
                element: Box::new(rec(element)?),
                shape: shape.clone(),
            },
            TypeAnn::RawPtr { mutable, pointee } => TypeAnn::RawPtr {
                mutable: *mutable,
                pointee: Box::new(rec(pointee)?),
            },
            TypeAnn::FnPtr { params, ret } => TypeAnn::FnPtr {
                params: params.iter().map(&mut rec).collect::<Option<Vec<_>>>()?,
                ret: match ret {
                    Some(ret) => Some(Box::new(rec(ret)?)),
                    None => None,
                },
            },
            _ => ty.clone(),
        })
    }

    fn canonical_name(
        &self,
        name: &str,
        scope: Option<&str>,
        imports: &[String],
        reader: Option<&str>,
        resolving: &mut BTreeSet<String>,
    ) -> Option<TypeAnn> {
        let Some(source) = self.declared_source(name, scope, imports) else {
            return Some(TypeAnn::Named(name.to_string()));
        };
        let (owner, bare) = source.rsplit_once('.')?;
        if let Some(target) = self.alias_targets.get(&source) {
            if !resolving.insert(source.clone()) {
                return None;
            }
            let resolved = self.canonical_rec(
                target,
                Some(owner),
                self.imports_of(owner),
                reader,
                resolving,
            );
            resolving.remove(&source);
            return resolved;
        }
        Some(TypeAnn::Named(if reader == Some(owner) {
            bare.to_string()
        } else {
            source.clone()
        }))
    }

    /// The source key of the project type `name` names inside `scope`: an
    /// existing source key, a qualified spelling, a type `scope` declares, or
    /// a type exactly one of `imports` exports.
    fn declared_source(
        &self,
        name: &str,
        scope: Option<&str>,
        imports: &[String],
    ) -> Option<String> {
        if self.declared_types.contains(name) {
            return Some(name.to_string());
        }
        if name.contains('.') {
            return self
                .type_source_in(name, imports, scope)
                .filter(|source| self.declared_types.contains(source));
        }
        if let Some(module) = scope {
            let local = format!("{module}.{name}");
            if self.declared_types.contains(&local) {
                return Some(local);
            }
        }
        let mut hits = BTreeSet::new();
        for import in imports {
            let direct = format!("{}.{name}", canonical_module(import));
            if self.exported_types.contains(&direct) {
                hits.insert(direct);
            } else if let Some(source) =
                self.imported_exported_source(import, name, std::slice::from_ref(import))
            {
                hits.insert(source);
            }
        }
        (hits.len() == 1).then(|| hits.into_iter().next()).flatten()
    }
}
