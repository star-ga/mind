// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Scoped ownership of the active cross-module export table.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};

use super::module_table::ModuleTable;

type ResolvedImportMap = BTreeMap<(String, Vec<String>), String>;
type QualifiedImportMap = BTreeMap<(String, usize, usize, String, bool), String>;

thread_local! {
    static ACTIVE: RefCell<Option<ModuleTable>> = const { RefCell::new(None) };
    /// Legacy whole-project checks intentionally expose the supplied files as
    /// one translation unit. Per-call table checks must remain import-scoped.
    static LEGACY_VISIBLE_SCOPE: Cell<bool> = const { Cell::new(false) };
    /// Import spellings resolved during manifest-bounded source capture.
    /// Keys include the importing owner so a unique basename cannot leak
    /// across modules or override an exact qualified path.
    static ACTIVE_RESOLVED_IMPORTS: RefCell<Option<ResolvedImportMap>> =
        const { RefCell::new(None) };
    /// Evaluator parsing keeps the owner qualifier for namespace references,
    /// while the ordinary AST intentionally folds `mod.name` to `name`.
    /// Keep those references span-scoped during evaluator setup so qualified
    /// access can be admitted without making an ambiguous bare name visible.
    static ACTIVE_QUALIFIED_IMPORTS: RefCell<Option<QualifiedImportMap>> =
        const { RefCell::new(None) };
}

/// Replace the active table. Retained for project-build compatibility.
pub(crate) fn set(table: Option<ModuleTable>) {
    ACTIVE.with(|cell| *cell.borrow_mut() = table);
}

/// Borrow the active table for one non-reentrant lookup.
pub(crate) fn with<R>(f: impl FnOnce(Option<&ModuleTable>) -> R) -> R {
    ACTIVE.with(|cell| f(cell.borrow().as_ref()))
}

pub(crate) fn legacy_visible_scope() -> bool {
    LEGACY_VISIBLE_SCOPE.with(Cell::get)
}

/// Resolve an import through captured project bindings, falling back to the
/// ordinary exact module-table lookup when no captured binding exists.
pub(crate) fn resolve_import(owner: Option<&str>, path: &[String]) -> Option<String> {
    ACTIVE_RESOLVED_IMPORTS.with(|resolved| {
        if let (Some(owner), Some(map)) = (owner, resolved.borrow().as_ref()) {
            if let Some(target) = map.get(&(owner.to_string(), path.to_vec())) {
                return Some(target.clone());
            }
        }
        ACTIVE.with(|cell| {
            cell.borrow()
                .as_ref()
                .and_then(|table| table.get_import(path))
                .map(|module| module.module_path.clone())
        })
    })
}

/// Return the exact project modules selected by imports captured for `owner`.
/// `Some(empty)` means the owner was captured but has no local imports; `None`
/// means this is a legacy table installation without captured project scope.
fn captured_import_targets_from_map(
    map: Option<&ResolvedImportMap>,
    owner: Option<&str>,
) -> Option<BTreeSet<String>> {
    let owner = owner?;
    Some(
        map?.iter()
            .filter(|((importer, _), _)| importer == owner)
            .map(|(_, target)| target.clone())
            .collect(),
    )
}

/// `std.posix` publishes libc names (`open`, `close`, `umask`, …). Those
/// names are visible only to a file that imports `std.posix` (or to
/// `std.posix` itself). Every other file keeps the pre-posix resolution:
/// its own `extern "C"` declaration is the callee.
const IMPORT_SCOPED_MODULE: &str = "std.posix";

thread_local! {
    /// Modules whose exports the file currently being checked imported by
    /// path. Independent of the captured import map: std imports are dropped
    /// from that map because the seed blob satisfies them.
    static ADMITTED_IMPORT_SCOPED: RefCell<BTreeSet<String>> = const { RefCell::new(BTreeSet::new()) };
    /// `extern "C"` names declared in the file currently being checked.
    /// They hide a same-named import-scoped export for the whole check, so
    /// the call stays on the extern lowering path.
    static LOCAL_EXTERN_NAMES: RefCell<BTreeSet<String>> = const { RefCell::new(BTreeSet::new()) };
    static QUALIFIED_POSIX_CALLS: RefCell<BTreeSet<(usize, usize)>> = const { RefCell::new(BTreeSet::new()) };
}

pub(crate) fn is_import_scoped_module(module: &str) -> bool {
    module == IMPORT_SCOPED_MODULE
}

fn admitted_import_scoped(module: &str) -> bool {
    ADMITTED_IMPORT_SCOPED.with(|cell| cell.borrow().contains(module))
}

fn local_extern_hides(name: &str) -> bool {
    LOCAL_EXTERN_NAMES.with(|cell| cell.borrow().contains(name))
}

/// Whether `module`'s bare exports may satisfy a lookup in `owner`.
/// Import-scoped modules are excluded from the legacy whole-table set
/// unless this file imported them, the captured map names them, or the
/// file being checked is that module.
fn bare_module_visible(
    module: &str,
    owner: Option<&str>,
    captured: &Option<BTreeSet<String>>,
) -> bool {
    if is_import_scoped_module(module) {
        return owner == Some(module)
            || captured
                .as_ref()
                .is_some_and(|targets| targets.contains(module))
            || admitted_import_scoped(module);
    }
    captured
        .as_ref()
        .is_none_or(|targets| targets.contains(module))
}

fn collect_file_scope(
    items: &[crate::ast::Node],
    admitted: &mut BTreeSet<String>,
    externs: &mut BTreeSet<String>,
    qualifiers: &mut BTreeMap<String, BTreeSet<String>>,
) {
    for item in items {
        match item {
            crate::ast::Node::Import { path, alias, .. } => {
                let joined = path.join(".");
                qualifiers
                    .entry(joined.clone())
                    .or_default()
                    .insert(joined.clone());
                for qualifier in path.last().into_iter().chain(alias.iter()) {
                    qualifiers
                        .entry(qualifier.clone())
                        .or_default()
                        .insert(joined.clone());
                }
                if is_import_scoped_module(&joined) {
                    admitted.insert(joined);
                }
            }
            crate::ast::Node::ExternBlock { fns, .. } => {
                for function in fns {
                    externs.insert(function.name.clone());
                }
            }
            crate::ast::Node::Block { stmts, .. } => {
                collect_file_scope(stmts, admitted, externs, qualifiers);
            }
            _ => {}
        }
    }
}

/// Installs the file-local import admission and extern-name set for one
/// check or lower, and restores the previous sets on drop.
pub(crate) struct FileScopeGuard {
    previous_admitted: BTreeSet<String>,
    previous_externs: BTreeSet<String>,
    previous_qualified: BTreeSet<(usize, usize)>,
}

impl FileScopeGuard {
    pub(crate) fn install(module: &crate::ast::Module) -> Self {
        let mut admitted = BTreeSet::new();
        let mut externs = BTreeSet::new();
        let mut qualifiers = BTreeMap::new();
        collect_file_scope(&module.items, &mut admitted, &mut externs, &mut qualifiers);
        // The parser folds a qualified call to its bare member. Retain its
        // recorded qualifier so a same-named local extern hides only bare calls.
        let qualified = module
            .spelling
            .call_qualifiers
            .iter()
            .filter_map(|(span, qualifier)| {
                let targets = qualifiers.get(&qualifier.replace("::", "."))?;
                (targets.len() == 1 && targets.contains(IMPORT_SCOPED_MODULE))
                    .then_some((span.start(), span.end()))
            })
            .collect();
        let previous_qualified = QUALIFIED_POSIX_CALLS.with(|cell| cell.replace(qualified));
        let previous_admitted = ADMITTED_IMPORT_SCOPED.with(|cell| cell.replace(admitted));
        let previous_externs = LOCAL_EXTERN_NAMES.with(|cell| cell.replace(externs));
        Self {
            previous_admitted,
            previous_externs,
            previous_qualified,
        }
    }
}

impl Drop for FileScopeGuard {
    fn drop(&mut self) {
        QUALIFIED_POSIX_CALLS.with(|cell| {
            *cell.borrow_mut() = std::mem::take(&mut self.previous_qualified);
        });
        ADMITTED_IMPORT_SCOPED.with(|cell| {
            *cell.borrow_mut() = std::mem::take(&mut self.previous_admitted);
        });
        LOCAL_EXTERN_NAMES.with(|cell| {
            *cell.borrow_mut() = std::mem::take(&mut self.previous_externs);
        });
    }
}

/// The defining module and signature of a bare name visible to `owner`.
/// A local `extern "C"` hides the name. The current module's own export
/// wins over an import.
fn resolve_visible(
    owner: Option<&str>,
    name: &str,
) -> Option<(String, super::module_table::ExportedFn)> {
    if local_extern_hides(name) {
        return None;
    }
    ACTIVE_RESOLVED_IMPORTS.with(|resolved| {
        let captured = captured_import_targets_from_map(resolved.borrow().as_ref(), owner);
        ACTIVE.with(|active| {
            let table = active.borrow();
            let table = table.as_ref()?;
            let candidates = table.exported_fn_candidate_refs(name);
            if let Some(owner) = owner {
                if let Some((module, function)) =
                    candidates.iter().find(|(module, _)| *module == owner)
                {
                    return Some(((*module).to_string(), (*function).clone()));
                }
            }
            let mut selected = candidates
                .iter()
                .filter(|(module, _)| bare_module_visible(module, owner, &captured));
            let first = selected.next()?;
            if selected.next().is_some() {
                // Explicit multi-file checking historically exposed one
                // translation-unit namespace. Preserve that scope only when
                // every duplicate carries the exact same ABI; manifest
                // captures and conflicting signatures remain fail-closed.
                if captured.is_some() || candidates.iter().any(|(_, function)| *function != first.1)
                {
                    return None;
                }
            }
            Some((first.0.to_string(), first.1.clone()))
        })
    })
}

/// Linker symbol for a call or definition that resolved to `std.posix`.
/// The source name stays `close`; the object symbol is `mind_posix_close`
/// so it cannot interpose libc when the same program also has
/// `extern "C" { fn close... }`.
pub(crate) fn link_name(owner: Option<&str>, name: &str) -> String {
    match resolve_visible(owner, name) {
        Some((module, _)) if is_import_scoped_module(&module) => {
            format!("mind_posix_{name}")
        }
        _ => name.to_string(),
    }
}

/// Resolve a bare imported function using the current module's captured
/// imports. A single imported owner wins over unimported siblings; multiple
/// owners or a missing imported symbol return `None` so the checker refuses
/// the unresolved/ambiguous spelling before native lowering.
/// A name declared `extern "C"` in the current file resolves to that
/// declaration, never to an import-scoped std export of the same name.
pub(crate) fn lookup_visible_fn(
    owner: Option<&str>,
    name: &str,
) -> Option<super::module_table::ExportedFn> {
    resolve_visible(owner, name).map(|(_, function)| function)
}

/// Return all typed function signatures visible to the current module. This
/// applies the same owner selection as [`lookup_visible_fn`], omitting
/// ambiguous names so lowering cannot register an arbitrary ABI.
pub(crate) fn visible_fns(owner: Option<&str>) -> Vec<super::module_table::ExportedFn> {
    ACTIVE_RESOLVED_IMPORTS.with(|resolved| {
        let captured = captured_import_targets_from_map(resolved.borrow().as_ref(), owner);
        ACTIVE.with(|active| {
            let table = active.borrow();
            let Some(table) = table.as_ref() else {
                return Vec::new();
            };
            let mut by_name: BTreeMap<String, Vec<(&str, &super::module_table::ExportedFn)>> =
                BTreeMap::new();
            for (module, function) in table.all_exported_fn_refs() {
                if local_extern_hides(&function.name) {
                    continue;
                }
                by_name
                    .entry(function.name.clone())
                    .or_default()
                    .push((module, function));
            }
            by_name
                .into_values()
                .filter_map(|candidates| {
                    if let Some(owner) = owner {
                        if let Some((_, function)) =
                            candidates.iter().find(|(module, _)| *module == owner)
                        {
                            return Some((*function).clone());
                        }
                    }
                    let mut selected = candidates
                        .iter()
                        .filter(|(module, _)| bare_module_visible(module, owner, &captured));
                    let first = selected.next()?;
                    if selected.next().is_some() {
                        (captured.is_none()
                            && candidates.iter().all(|(_, function)| *function == first.1))
                        .then(|| (*first.1).clone())
                    } else {
                        Some(first.1.clone())
                    }
                })
                .collect()
        })
    })
}

/// Whether a bare exported symbol is uniquely visible to the current module.
/// Imported owners are authoritative when present; otherwise the legacy
/// whole-project metadata surface remains available only for one owner.
pub(crate) fn symbol_visible(owner: Option<&str>, name: &str) -> bool {
    if local_extern_hides(name) {
        return false;
    }
    ACTIVE_RESOLVED_IMPORTS.with(|resolved| {
        let captured = captured_import_targets_from_map(resolved.borrow().as_ref(), owner);
        ACTIVE.with(|active| {
            let table = active.borrow();
            let Some(table) = table.as_ref() else {
                return false;
            };
            let mut selected = table
                .exported_symbol_candidate_refs(name)
                .into_iter()
                .filter(|module| bare_module_visible(module, owner, &captured));
            selected.next().is_some() && selected.next().is_none()
        })
    })
}

/// Resolve a bare symbol for the current owner; empty without project scope.
pub(crate) fn current_symbol_exported(name: &str) -> bool {
    symbol_visible(
        crate::qualified_enums::current_module_path().as_deref(),
        name,
    )
}

pub(crate) fn is_qualified_posix_call(span: crate::ast::Span) -> bool {
    QUALIFIED_POSIX_CALLS.with(|cell| cell.borrow().contains(&(span.start(), span.end())))
}

fn qualified_import_target(span: crate::ast::Span, name: &str, is_call: bool) -> Option<String> {
    if is_call && is_qualified_posix_call(span) {
        return Some(IMPORT_SCOPED_MODULE.to_string());
    }
    let owner = crate::qualified_enums::current_module_path()?;
    ACTIVE_QUALIFIED_IMPORTS.with(|refs| {
        refs.borrow()
            .as_ref()?
            .get(&(owner, span.start(), span.end(), name.to_string(), is_call))
            .cloned()
    })
}

/// Whether this span is a validated evaluator namespace value reference.
pub(crate) fn qualified_import_value(span: crate::ast::Span, name: &str) -> bool {
    qualified_import_target(span, name, false).is_some()
}

/// Resolve a validated evaluator namespace call to its defining signature.
/// Explicitly exported functions carry signatures in the same table as
/// auto-exported functions; a missing signature remains a loose i64 call.
pub(crate) fn qualified_import_fn(
    span: crate::ast::Span,
    name: &str,
) -> Option<super::module_table::ExportedFn> {
    let target = qualified_import_target(span, name, true)?;
    ACTIVE.with(|active| {
        active
            .borrow()
            .as_ref()?
            .get(&target)?
            .exported_fns
            .iter()
            .find(|f| f.name == name)
            .cloned()
    })
}

/// Return all bare exported symbols that have exactly one visible owner.
/// Building this set once lets import injection avoid an all-module scan for
/// every exported name while retaining the legacy whole-table behavior when
/// no captured project scope is installed.
pub(crate) fn visible_symbols(owner: Option<&str>) -> BTreeSet<String> {
    ACTIVE_RESOLVED_IMPORTS.with(|resolved| {
        let captured = captured_import_targets_from_map(resolved.borrow().as_ref(), owner);
        ACTIVE.with(|active| {
            let table = active.borrow();
            let Some(table) = table.as_ref() else {
                return BTreeSet::new();
            };
            let mut counts = BTreeMap::<String, u8>::new();
            for (module, name) in table.all_exported_symbol_refs() {
                if local_extern_hides(name) || !bare_module_visible(module, owner, &captured) {
                    continue;
                }
                let count = counts.entry(name.to_owned()).or_default();
                *count = count.saturating_add(1);
            }
            counts
                .into_iter()
                .filter_map(|(name, count)| (count == 1).then_some(name))
                .collect()
        })
    })
}

/// Whether an imported typed function would collide with a local definition
/// in the native link unit. Until linker symbols carry their defining owner,
/// the runnable-artifact preflight refuses this shape before lowering can let
/// clang report a duplicate symbol.
pub(crate) fn imported_fn_conflicts_with_local(owner: Option<&str>, name: &str) -> bool {
    ACTIVE_RESOLVED_IMPORTS.with(|resolved| {
        let captured = captured_import_targets_from_map(resolved.borrow().as_ref(), owner);
        let Some(targets) = captured else {
            return false;
        };
        ACTIVE.with(|active| {
            let table = active.borrow();
            let Some(table) = table.as_ref() else {
                return false;
            };
            table
                .exported_fn_candidate_refs(name)
                .into_iter()
                .any(|(module, _)| targets.contains(module))
        })
    })
}

/// Installs a table and restores the previous value on every exit path.
pub(crate) struct Guard {
    previous: Option<ModuleTable>,
    previous_legacy_scope: bool,
}

impl Guard {
    pub(crate) fn install(table: ModuleTable) -> Self {
        let previous = ACTIVE.with(|cell| cell.borrow_mut().replace(table));
        let previous_legacy_scope = LEGACY_VISIBLE_SCOPE.with(|cell| cell.replace(false));
        Self {
            previous,
            previous_legacy_scope,
        }
    }

    pub(crate) fn install_legacy(table: ModuleTable) -> Self {
        let previous = ACTIVE.with(|cell| cell.borrow_mut().replace(table));
        let previous_legacy_scope = LEGACY_VISIBLE_SCOPE.with(|cell| cell.replace(true));
        Self {
            previous,
            previous_legacy_scope,
        }
    }
}

/// Installs captured owner-aware import bindings and restores the previous
/// map on every exit path. Kept separate from the export-table guard so the
/// pre-existing table API remains unchanged for non-project callers.
pub(crate) struct ResolvedImportsGuard {
    previous: Option<ResolvedImportMap>,
}

/// Installs the validated namespace references for one evaluator module and
/// restores the previous set on every exit path.
pub(crate) struct QualifiedImportsGuard {
    previous: Option<QualifiedImportMap>,
}

impl QualifiedImportsGuard {
    pub(crate) fn install(refs: QualifiedImportMap) -> Self {
        let previous = ACTIVE_QUALIFIED_IMPORTS.with(|cell| cell.borrow_mut().replace(refs));
        Self { previous }
    }
}

impl Drop for QualifiedImportsGuard {
    fn drop(&mut self) {
        ACTIVE_QUALIFIED_IMPORTS.with(|cell| *cell.borrow_mut() = self.previous.take());
    }
}

impl ResolvedImportsGuard {
    pub(crate) fn install(imports: ResolvedImportMap) -> Self {
        let previous = ACTIVE_RESOLVED_IMPORTS.with(|cell| cell.borrow_mut().replace(imports));
        Self { previous }
    }
}

impl Drop for ResolvedImportsGuard {
    fn drop(&mut self) {
        ACTIVE_RESOLVED_IMPORTS.with(|cell| *cell.borrow_mut() = self.previous.take());
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        ACTIVE.with(|cell| *cell.borrow_mut() = self.previous.take());
        LEGACY_VISIBLE_SCOPE.with(|cell| cell.set(self.previous_legacy_scope));
    }
}
