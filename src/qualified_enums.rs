// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Owner-preserving enum metadata for manifest projects.
//!
//! Canonical metadata includes private declarations because their defining
//! module still has to lower functions that return them. Source lookup is a
//! separate operation: it receives the current module and its imports, and
//! exposes another module's enum only when that enum is exported.

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
use std::collections::BTreeMap;
#[cfg(feature = "cross-module-imports")]
use std::collections::BTreeSet;

#[cfg(feature = "cross-module-imports")]
use crate::ast::TypeAnn;

#[cfg(feature = "cross-module-imports")]
#[derive(Clone, Debug, Default)]
pub(crate) struct Registry {
    /// `crate.path.Enum` -> collision-safe internal enum key.
    type_keys: BTreeMap<String, String>,
    /// Canonical source keys for every declared enum, struct, and alias.
    declared_types: BTreeSet<String>,
    /// Canonical source type keys that may be named outside their owner.
    exported_types: BTreeSet<String>,
    /// Canonical source alias key -> target resolved within that source owner.
    alias_targets: BTreeMap<String, TypeAnn>,
    pub(crate) variant_tags: BTreeMap<String, i64>,
    pub(crate) payload_types: BTreeMap<String, Vec<TypeAnn>>,
    pub(crate) slots: BTreeMap<String, usize>,
    pub(crate) boxed: BTreeSet<String>,
    pub(crate) struct_field_names: BTreeMap<String, Vec<String>>,
}

#[cfg(feature = "cross-module-imports")]
impl Registry {
    fn owner_for(
        &self,
        qualifier: &str,
        imports: &[String],
        current: Option<&str>,
    ) -> Option<String> {
        let mut owners = BTreeSet::new();
        if let Some(module) = current {
            if qualifier_matches(qualifier, module) {
                owners.insert(module.to_string());
            }
        }
        for import in imports {
            let canonical = canonical_module(import);
            if qualifier_matches(qualifier, &canonical) {
                owners.insert(canonical);
            }
        }
        (owners.len() == 1)
            .then(|| owners.into_iter().next())
            .flatten()
    }

    fn resolve_type_in(
        &self,
        spelling: &str,
        imports: &[String],
        current: Option<&str>,
    ) -> Option<String> {
        let (qualifier, enum_name) = spelling.rsplit_once('.')?;
        let owner = self.owner_for(qualifier, imports, current)?;
        let source_key = format!("{owner}.{enum_name}");
        if current != Some(owner.as_str()) && !self.exported_types.contains(&source_key) {
            return None;
        }
        self.type_keys.get(&source_key).cloned()
    }

    fn resolve_variant_in(
        &self,
        spelling: &str,
        imports: &[String],
        current: Option<&str>,
    ) -> Option<String> {
        let (type_spelling, variant) = spelling
            .rsplit_once("::")
            .or_else(|| spelling.rsplit_once('.'))?;
        let enum_key = if is_bare_name(type_spelling) {
            // A bare head (`Table::RawResponses`) written outside the enum's
            // module names an IMPORTED enum whose registry key is owner-qualified
            // because another module declares the same enum name. Left bare, the
            // spelling misses the registry and lowering finds no tag for it. A
            // type the current module declares keeps lexical precedence and its
            // historical spelling (lowering resolves that one against the current
            // module); a non-colliding key equals the bare head and is unchanged.
            let current = current?;
            if self
                .declared_types
                .contains(&format!("{current}.{type_spelling}"))
            {
                return None;
            }
            match self.imported_enum_key(type_spelling, imports) {
                Ok(Some(key)) if key != type_spelling => key,
                _ => return None,
            }
        } else {
            self.resolve_type_in(type_spelling, imports, current)?
        };
        let key = format!("{enum_key}::{variant}");
        self.variant_tags.contains_key(&key).then_some(key)
    }

    /// Resolve a bare type name through the importing module's imports alone:
    /// `Ok(Some(key))` when exactly one import exports it and it is an enum,
    /// `Ok(None)` when exactly one exports it as a non-enum type, and
    /// `Err(exporters)` with the number of exporters (0, or 2 and more)
    /// otherwise.
    fn imported_enum_key(&self, name: &str, imports: &[String]) -> Result<Option<String>, usize> {
        let owners: BTreeSet<String> = imports
            .iter()
            .map(|import| format!("{}.{name}", canonical_module(import)))
            .filter(|source| self.exported_types.contains(source))
            .collect();
        match owners.first() {
            Some(source) if owners.len() == 1 => Ok(self.type_keys.get(source).cloned()),
            _ => Err(owners.len()),
        }
    }

    /// Why a bare-headed `Enum::Variant` path that names ANOTHER module's enum
    /// cannot be lowered, or `None` when it resolves or names no project enum.
    /// `declared_locally` is true when the writing module declares a type of
    /// that name (lexical precedence; checked by the resolver instead). Lets
    /// the checker refuse with a located diagnostic where lowering would
    /// otherwise stop at its fail-closed panic.
    pub(crate) fn bare_variant_problem(
        &self,
        path: &str,
        imports: &[String],
        declared_locally: bool,
    ) -> Option<BareVariantProblem> {
        let (head, variant) = path.split_once("::")?;
        if declared_locally
            || !is_bare_name(head)
            || variant.contains("::")
            || matches!(head, "Option" | "Result")
        {
            return None;
        }
        let project_enum = || {
            self.type_keys
                .keys()
                .any(|source| source.rsplit_once('.').is_some_and(|(_, n)| n == head))
        };
        match self.imported_enum_key(head, imports) {
            Ok(Some(key)) => (!self.variant_tags.contains_key(&format!("{key}::{variant}")))
                .then_some(BareVariantProblem::UnknownVariant),
            Ok(None) => None,
            Err(0) => project_enum().then_some(BareVariantProblem::NotVisible),
            Err(_) => project_enum().then_some(BareVariantProblem::Ambiguous),
        }
    }

    fn bare_visible(&self, name: &str, imports: &[String], current: Option<&str>) -> bool {
        let mut owners = BTreeSet::new();
        if let Some(module) = current {
            if self.type_keys.contains_key(&format!("{module}.{name}")) {
                // A declaration in the current module has lexical precedence
                // over imported owners with the same bare spelling.
                return true;
            }
        }
        for import in imports {
            let owner = canonical_module(import);
            let key = format!("{owner}.{name}");
            if self.exported_types.contains(&key) {
                owners.insert(owner);
            }
        }
        owners.len() == 1
    }

    pub(crate) fn bare_type_is_ambiguous(
        &self,
        name: &str,
        imports: &[String],
        current: Option<&str>,
    ) -> bool {
        if current.is_some_and(|module| self.type_keys.contains_key(&format!("{module}.{name}"))) {
            return false;
        }
        let mut owners = BTreeSet::new();
        for import in imports {
            let owner = canonical_module(import);
            let key = format!("{owner}.{name}");
            if self.exported_types.contains(&key) {
                owners.insert(owner);
            }
        }
        owners.len() > 1
    }

    pub(crate) fn is_canonical_type(&self, name: &str) -> bool {
        self.type_keys.values().any(|key| key == name)
    }

    pub(crate) fn source_type_is_visible(
        &self,
        spelling: &str,
        imports: &[String],
        current: Option<&str>,
    ) -> bool {
        let Some((qualifier, type_name)) = spelling.rsplit_once('.') else {
            return false;
        };
        let Some(owner) = self.owner_for(qualifier, imports, current) else {
            return false;
        };
        let source_key = format!("{owner}.{type_name}");
        self.declared_types.contains(&source_key)
            && (current == Some(owner.as_str()) || self.exported_types.contains(&source_key))
    }

    fn resolve_alias_in(
        &self,
        spelling: &str,
        imports: &[String],
        current: Option<&str>,
    ) -> Option<TypeAnn> {
        let (qualifier, alias_name) = spelling.rsplit_once('.')?;
        let owner = self.owner_for(qualifier, imports, current)?;
        let source_key = format!("{owner}.{alias_name}");
        if current != Some(owner.as_str()) && !self.exported_types.contains(&source_key) {
            return None;
        }
        self.alias_targets.get(&source_key).cloned()
    }

    pub(crate) fn is_canonical_variant(&self, name: &str) -> bool {
        self.variant_tags.contains_key(name)
    }

    pub(crate) fn tag(&self, key: &str) -> Option<i64> {
        self.variant_tags.get(key).copied()
    }

    pub(crate) fn bare_variant_is_visible(
        &self,
        variant: &str,
        imports: &[String],
        current: Option<&str>,
    ) -> bool {
        self.visible_enum_keys(imports, current)
            .iter()
            .any(|key| self.variant_tags.contains_key(&format!("{key}::{variant}")))
    }

    pub(crate) fn has_bare_variant(&self, variant: &str) -> bool {
        self.variant_tags.keys().any(|key| {
            key.rsplit_once("::")
                .is_some_and(|(_, name)| name == variant)
        })
    }

    fn visible_enum_keys(&self, imports: &[String], current: Option<&str>) -> BTreeSet<String> {
        self.type_keys
            .iter()
            .filter(|(source, _)| {
                let (owner, _) = source.rsplit_once('.').unwrap();
                current == Some(owner)
                    || (self.exported_types.contains(*source)
                        && imports
                            .iter()
                            .map(|path| canonical_module(path))
                            .any(|import| import == owner))
            })
            .map(|(_, key)| key.clone())
            .collect()
    }

    #[cfg(feature = "std-surface")]
    pub(crate) fn install_ir_metadata(
        &self,
        ir: &mut crate::ir::IRModule,
        module: &crate::ast::Module,
    ) {
        let imports = module_imports(module);
        let visible = self.visible_enum_keys(&imports, current_module_path().as_deref());
        for (key, tag) in &self.variant_tags {
            if key
                .rsplit_once("::")
                .is_some_and(|(owner, _)| visible.contains(owner))
            {
                ir.enum_variant_tags.insert(key.clone(), *tag);
            }
        }
        for (key, payload) in &self.payload_types {
            if key
                .rsplit_once("::")
                .is_some_and(|(owner, _)| visible.contains(owner))
            {
                ir.enum_payload_types.insert(key.clone(), payload.clone());
            }
        }
        for key in &visible {
            if let Some(slots) = self.slots.get(key) {
                ir.enum_payload_slots.insert(key.clone(), *slots);
            }
            if self.boxed.contains(key) {
                ir.boxed_enums.insert(key.clone());
            }
        }
        for (key, fields) in &self.struct_field_names {
            if key
                .rsplit_once("::")
                .is_some_and(|(owner, _)| visible.contains(owner))
            {
                ir.enum_struct_field_names
                    .insert(key.clone(), fields.clone());
            }
        }
    }
}

#[cfg(feature = "cross-module-imports")]
pub(crate) fn module_imports(module: &crate::ast::Module) -> Vec<String> {
    let owner = current_module_path();
    module
        .items
        .iter()
        .filter_map(|item| match item {
            crate::ast::Node::Import { path, .. } => {
                crate::project::active_module_table::resolve_import(owner.as_deref(), path)
                    .or_else(|| Some(path.join(".")))
            }
            _ => None,
        })
        .collect()
}

#[cfg(feature = "cross-module-imports")]
pub(crate) fn resolve_alias_target(name: &str, imports: &[String]) -> Option<TypeAnn> {
    let current = current_module_path();
    crate::ir::with_global_enums(|global| {
        global
            .qualified
            .resolve_alias_in(name, imports, current.as_deref())
    })
}

/// A bare-headed `Enum::Variant` path that names a project enum but cannot be
/// resolved from the module that writes it.
#[cfg(feature = "cross-module-imports")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BareVariantProblem {
    NotVisible,
    Ambiguous,
    UnknownVariant,
}

/// A single identifier segment: no module qualifier and no `::` path.
#[cfg(feature = "cross-module-imports")]
fn is_bare_name(name: &str) -> bool {
    !name.is_empty() && !name.contains('.') && !name.contains("::")
}

#[cfg(feature = "cross-module-imports")]
fn canonical_module(path: &str) -> String {
    if path == "crate" || path.starts_with("crate.") {
        path.to_string()
    } else {
        format!("crate.{path}")
    }
}

#[cfg(feature = "cross-module-imports")]
fn qualifier_matches(qualifier: &str, canonical: &str) -> bool {
    qualifier == canonical
        || canonical
            .strip_prefix("crate.")
            .is_some_and(|short| qualifier == short)
        || canonical
            .rsplit_once('.')
            .is_some_and(|(_, terminal)| qualifier == terminal)
}

#[cfg(feature = "cross-module-imports")]
fn resolve_qualified_alias_target(
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

#[derive(Clone, Default)]
pub(crate) struct ParseScope {
    names: Vec<String>,
    #[cfg(feature = "cross-module-imports")]
    registry: Registry,
    #[cfg(feature = "cross-module-imports")]
    current: Option<String>,
}

impl ParseScope {
    pub(crate) fn capture() -> Self {
        crate::ir::with_global_enums(|g| Self {
            names: g.names.clone(),
            #[cfg(feature = "cross-module-imports")]
            registry: g.qualified.clone(),
            #[cfg(feature = "cross-module-imports")]
            current: current_module_path(),
        })
    }

    pub(crate) fn is_enum_name(&self, name: &str, imports: &[String]) -> bool {
        if !self.names.iter().any(|known| known == name) {
            return false;
        }
        #[cfg(feature = "cross-module-imports")]
        return self.registry.type_keys.is_empty()
            || self
                .registry
                .bare_visible(name, imports, self.current.as_deref());
        #[cfg(not(feature = "cross-module-imports"))]
        {
            let _ = imports;
            true
        }
    }

    pub(crate) fn resolve_type(&self, name: &str, imports: &[String]) -> Option<String> {
        #[cfg(feature = "cross-module-imports")]
        return self
            .registry
            .resolve_type_in(name, imports, self.current.as_deref());
        #[cfg(not(feature = "cross-module-imports"))]
        {
            let _ = (name, imports);
            None
        }
    }

    pub(crate) fn resolve_variant(&self, name: &str, imports: &[String]) -> Option<String> {
        #[cfg(feature = "cross-module-imports")]
        return self
            .registry
            .resolve_variant_in(name, imports, self.current.as_deref());
        #[cfg(not(feature = "cross-module-imports"))]
        {
            let _ = (name, imports);
            None
        }
    }
}

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
thread_local! {
    static CURRENT_MODULE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "cross-module-imports")]
pub(crate) struct ModuleGuard(Option<String>);

#[cfg(feature = "cross-module-imports")]
impl ModuleGuard {
    pub(crate) fn install(path: impl Into<String>) -> Self {
        Self(CURRENT_MODULE.with(|cell| cell.borrow_mut().replace(path.into())))
    }
}

#[cfg(feature = "cross-module-imports")]
impl Drop for ModuleGuard {
    fn drop(&mut self) {
        CURRENT_MODULE.with(|cell| *cell.borrow_mut() = self.0.take());
    }
}

#[cfg(feature = "cross-module-imports")]
pub(crate) struct GlobalGuard(crate::ir::GlobalEnums);

#[cfg(feature = "cross-module-imports")]
impl GlobalGuard {
    pub(crate) fn install(enums: crate::ir::GlobalEnums) -> Self {
        let previous = crate::ir::with_global_enums(Clone::clone);
        crate::ir::set_global_enums(enums);
        Self(previous)
    }
}

#[cfg(feature = "cross-module-imports")]
impl Drop for GlobalGuard {
    fn drop(&mut self) {
        crate::ir::set_global_enums(std::mem::take(&mut self.0));
    }
}

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
pub(crate) fn current_module_path() -> Option<String> {
    CURRENT_MODULE.with(|cell| cell.borrow().clone())
}

#[cfg(all(feature = "cross-module-imports", feature = "std-surface"))]
pub(crate) fn current_enum_key(name: &str) -> String {
    let Some(module) = current_module_path() else {
        return name.to_string();
    };
    crate::ir::with_global_enums(|g| {
        if let Some(key) = g.qualified.type_keys.get(&format!("{module}.{name}")) {
            return key.clone();
        }
        name.to_string()
    })
}

#[cfg(feature = "std-surface")]
pub(crate) fn local_variant_key(name: &str, tags: &BTreeMap<String, i64>) -> Option<String> {
    if tags.contains_key(name) {
        return Some(name.to_string());
    }
    let key = format!("{}::{name}", current_module_path()?);
    tags.contains_key(&key).then_some(key)
}

#[cfg(feature = "cross-module-imports")]
pub(crate) fn rebuild(parsed: &[(String, crate::ast::Module)], enums: &mut crate::ir::GlobalEnums) {
    use crate::project::module_table::{collect_module_enums, collect_module_exports};

    type Decl = (String, crate::project::module_table::ExportedEnum);
    let mut by_name: BTreeMap<String, Vec<Decl>> = BTreeMap::new();
    let mut exported = BTreeSet::new();
    let mut declared = BTreeSet::new();
    let mut raw_alias_targets = BTreeMap::new();
    let mut alias_imports = BTreeMap::new();
    for (path, module) in parsed {
        let _module_guard = ModuleGuard::install(path.clone());
        let exports = collect_module_exports(path, module);
        let aliases = crate::eval::type_aliases::LocalTypeAliases::new(&module.items);
        alias_imports.insert(path.clone(), module_imports(module));
        let mut explicit_exports = false;
        let mut pending: Vec<&crate::ast::Node> = module.items.iter().rev().collect();
        while let Some(item) = pending.pop() {
            match item {
                crate::ast::Node::Block { stmts, .. } => pending.extend(stmts.iter().rev()),
                crate::ast::Node::Export { .. } => explicit_exports = true,
                _ => {}
            }
        }
        let mut pending: Vec<&crate::ast::Node> = module.items.iter().rev().collect();
        while let Some(item) = pending.pop() {
            match item {
                crate::ast::Node::Block { stmts, .. } => pending.extend(stmts.iter().rev()),
                crate::ast::Node::TypeAlias { name, target, .. } => {
                    let source_key = format!("{path}.{name}");
                    raw_alias_targets.insert(source_key.clone(), aliases.resolve(target));
                    declared.insert(source_key.clone());
                    if !explicit_exports || exports.exported.iter().any(|item| item == name) {
                        exported.insert(source_key);
                    }
                }
                crate::ast::Node::EnumDef { name, .. }
                | crate::ast::Node::StructDef { name, .. } => {
                    let source_key = format!("{path}.{name}");
                    declared.insert(source_key.clone());
                    if !explicit_exports || exports.exported.iter().any(|item| item == name) {
                        exported.insert(source_key);
                    }
                }
                _ => {}
            }
        }
        for decl in collect_module_enums(module) {
            by_name
                .entry(decl.name.clone())
                .or_default()
                .push((path.clone(), decl));
        }
    }
    for owners in by_name.values_mut() {
        owners.sort_by(|(a, _), (b, _)| a.cmp(b));
    }
    let alias_targets = raw_alias_targets
        .iter()
        .filter_map(|(source_key, target)| {
            let owner = source_key.rsplit_once('.')?.0;
            let resolved = resolve_qualified_alias_target(
                target,
                owner,
                &raw_alias_targets,
                &alias_imports,
                &exported,
                &mut BTreeSet::new(),
            )?;
            Some((source_key.clone(), resolved))
        })
        .collect();

    enums.names.clear();
    enums.variant_tags.clear();
    enums.payload_types.clear();
    enums.slots.clear();
    enums.boxed.clear();
    enums.struct_field_names.clear();
    enums.fn_returns.clear();
    enums.qualified = Registry::default();
    enums.qualified.declared_types = declared;
    enums.qualified.exported_types = exported;
    enums.qualified.alias_targets = alias_targets;

    for (enum_name, owners) in by_name {
        enums.names.push(enum_name.clone());
        let collision = owners.len() > 1;
        for (owner_index, (module_path, decl)) in owners.into_iter().enumerate() {
            let enum_key = if collision {
                format!("{module_path}::{enum_name}")
            } else {
                enum_name.clone()
            };
            enums
                .qualified
                .type_keys
                .insert(format!("{module_path}.{enum_name}"), enum_key.clone());
            let tag_base = if collision {
                ((owner_index as i64) + 1) << 32
            } else {
                0
            };
            for (ordinal, variant) in decl.variants.iter().enumerate() {
                let key = format!("{enum_key}::{}", variant.name);
                enums
                    .qualified
                    .variant_tags
                    .insert(key.clone(), tag_base + ordinal as i64);
                if !variant.payload.is_empty() {
                    enums
                        .qualified
                        .payload_types
                        .insert(key.clone(), variant.payload.clone());
                }
                if !variant.field_names.is_empty() {
                    enums
                        .qualified
                        .struct_field_names
                        .insert(key.clone(), variant.field_names.clone());
                }
            }
            let max_arity = decl
                .variants
                .iter()
                .map(|variant| variant.payload.len())
                .max()
                .unwrap_or(0);
            if max_arity > 0 {
                let slots = 1 + max_arity;
                enums.qualified.slots.insert(enum_key.clone(), slots);
                enums.qualified.boxed.insert(enum_key.clone());
            }
        }
    }
    enums.names.sort();
    enums.names.dedup();

    for (module_path, module) in parsed {
        let mut stack: Vec<&crate::ast::Node> = module.items.iter().collect();
        while let Some(item) = stack.pop() {
            match item {
                crate::ast::Node::FnDef(fd, _) => {
                    if let Some(ret) = &fd.ret_type {
                        let ret = match ret {
                            TypeAnn::Named(name) => enums
                                .qualified
                                .type_keys
                                .get(&format!("{module_path}.{name}"))
                                .cloned()
                                .map(TypeAnn::Named)
                                .unwrap_or_else(|| ret.clone()),
                            _ => ret.clone(),
                        };
                        enums.fn_returns.insert(fd.name.clone(), ret);
                    }
                }
                crate::ast::Node::Block { stmts, .. } => stack.extend(stmts),
                _ => {}
            }
        }
    }
}
