// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Manifest-bounded module scope for single-file compiler entry points.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::ast::Module;

use super::sources::{canonical_project_root, resolve_project_entry};
use super::{
    DEFAULT_TARGET_BLOCK, find_project_root_for_file_checked, load_manifest, resolve_sources,
};

pub enum Discovery {
    SingleTranslationUnit,
    MissingProject(Vec<String>),
    Project(ProjectScope),
}

/// One immutable source snapshot, parsed under its canonical module path.
#[derive(Clone)]
pub struct CapturedSource {
    path: PathBuf,
    module_path: String,
    source: String,
    module: Module,
}

impl CapturedSource {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn module_path(&self) -> &str {
        &self.module_path
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn module(&self) -> &Module {
        &self.module
    }
}

pub struct ProjectScope {
    entry: PathBuf,
    /// Entry plus every parseable declared source, in canonical module order.
    /// These modules supply the established implicit whole-project type/enum
    /// metadata surface even when no body import names them.
    sources: Vec<CapturedSource>,
    /// Canonical module paths whose native bodies the entry imports
    /// transitively. The entry itself is always present.
    linked: BTreeSet<String>,
    /// Exact project-discovery result for each importing module and written
    /// import path. This retains the unique-stem resolution performed during
    /// discovery so later consumers never have to reconstruct it by name.
    resolved_imports: BTreeMap<(String, Vec<String>), String>,
    table: super::module_table::ModuleTable,
    enums: Box<crate::ir::GlobalEnums>,
    /// Every module-level `const` of the project, by name, for inlining a
    /// sibling's const at lowering (`ir::set_project_consts`), as the
    /// executable build does.
    consts: std::collections::BTreeMap<String, crate::ast::Node>,
}

impl ProjectScope {
    pub fn entry(&self) -> &Path {
        &self.entry
    }
    pub fn sources(&self) -> &[CapturedSource] {
        &self.sources
    }
    pub fn linked_sources(&self) -> impl Iterator<Item = &CapturedSource> {
        self.sources
            .iter()
            .filter(|source| self.linked.contains(source.module_path()))
    }
    pub fn has_linked_siblings(&self) -> bool {
        self.linked.len() > 1
    }
    /// Resolve a source-level import through the same canonical module table
    /// used by project checking and native compilation.
    pub(crate) fn resolve_import_path(&self, owner: &str, path: &[String]) -> Option<&str> {
        super::call_bindings::resolve_import_target(
            &self.table,
            &self.resolved_imports,
            owner,
            path,
        )
    }
    /// Check a qualified symbol against the exact module selected during
    /// captured project discovery, including that module's export boundary.
    pub(crate) fn resolves_imported_symbol(
        &self,
        owner: &str,
        path: &[String],
        symbol: &str,
    ) -> bool {
        self.resolve_import_path(owner, path)
            .and_then(|target| self.table.get(target))
            .is_some_and(|module| module.exported.iter().any(|name| name == symbol))
    }
    /// Resolve one source function without losing its declaration owner or
    /// conflating ambiguity with absence. This metadata API is not wired into
    /// the legacy checker or lowering paths.
    pub fn resolve_function(
        &self,
        source_owner: &str,
        reference: super::call_bindings::FunctionReference<'_>,
    ) -> super::call_bindings::FunctionResolution {
        let source_module = self
            .sources
            .iter()
            .find(|source| source.module_path() == source_owner)
            .map(CapturedSource::module);
        super::call_bindings::resolve_function(
            &self.table,
            &self.resolved_imports,
            source_owner,
            source_module,
            reference,
        )
    }
    pub fn install(&self) -> ProjectTableGuard {
        let entry_module = self
            .sources
            .first()
            .map(|source| source.module_path().to_string())
            .unwrap_or_else(|| "crate".to_string());
        let mut guard = ProjectTableGuard::install_with_scope(
            self.table.clone(),
            (*self.enums).clone(),
            self.resolved_imports.clone(),
            entry_module,
        );
        guard._consts = Some(ConstsGuard::install(&self.consts));
        guard
    }
}

thread_local! {
    /// How many [`ConstsGuard`]s are live on this thread.
    static CONSTS_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Seeds the project's cross-module consts for lowering while a scope is
/// installed. A sibling `a.K` reached the flat `--emit-*` and cdylib lowering
/// with no value and panicked ("undefined identifier `K` reached lowering"):
/// only the executable build seeded them. Nested installs of a scope (its
/// siblings' compiles) keep the outermost table; the last guard clears it.
struct ConstsGuard;

impl ConstsGuard {
    fn install(consts: &std::collections::BTreeMap<String, crate::ast::Node>) -> Self {
        if CONSTS_DEPTH.with(|d| d.replace(d.get() + 1)) == 0 {
            crate::ir::set_project_consts(consts.clone());
        }
        ConstsGuard
    }
}

impl Drop for ConstsGuard {
    fn drop(&mut self) {
        if CONSTS_DEPTH.with(|d| d.replace(d.get() - 1)) == 1 {
            crate::ir::clear_project_consts();
        }
    }
}

/// Install the correct table for source bytes already captured by `check`.
pub fn install_for_check(files: &[(PathBuf, String)]) -> Result<ProjectTableGuard> {
    if files.len() == 1 {
        match discover_with_source(&files[0].0, DEFAULT_TARGET_BLOCK, &files[0].1)? {
            Discovery::Project(scope) => return Ok(scope.install()),
            Discovery::MissingProject(imports) => {
                anyhow::bail!(
                    "error[type-check][E2003]: local import(s) {} require an enclosing Mind.toml \
                     project that declares the source set",
                    imports.join(", ")
                );
            }
            Discovery::SingleTranslationUnit => {}
        }
    }
    let paths = files
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let source_root = common_ancestor_dir(&paths);
    let parsed = files
        .iter()
        .filter_map(|(path, source)| {
            let module = crate::parser::parse(source).ok()?;
            Some((
                super::module_table::module_path_of(path, &source_root),
                module,
            ))
        })
        .collect();
    Ok(install_parsed(parsed))
}

fn install_parsed(project_modules: Vec<(String, Module)>) -> ProjectTableGuard {
    let mut parsed = crate::project::stdlib::parsed_stdlib_modules();
    parsed.extend(project_modules);
    let refs: Vec<(String, &Module)> = parsed
        .iter()
        .map(|(path, module)| (path.clone(), module))
        .collect();
    ProjectTableGuard::install_with_enums(
        super::module_table::build_module_table(&refs),
        super::build_global_enums(&parsed),
    )
}

pub struct ProjectTableGuard {
    _guard: super::active_module_table::Guard,
    _enums: Option<crate::qualified_enums::GlobalGuard>,
    _resolved_imports: Option<super::active_module_table::ResolvedImportsGuard>,
    _module_path: Option<crate::qualified_enums::ModuleGuard>,
    _consts: Option<ConstsGuard>,
}

impl ProjectTableGuard {
    pub(crate) fn install(table: super::module_table::ModuleTable) -> Self {
        Self {
            _guard: super::active_module_table::Guard::install_legacy(table),
            _enums: None,
            _resolved_imports: None,
            _module_path: None,
            _consts: None,
        }
    }

    pub(crate) fn install_with_enums(
        table: super::module_table::ModuleTable,
        enums: crate::ir::GlobalEnums,
    ) -> Self {
        Self {
            _guard: super::active_module_table::Guard::install_legacy(table),
            _enums: Some(crate::qualified_enums::GlobalGuard::install(enums)),
            _resolved_imports: None,
            _module_path: None,
            _consts: None,
        }
    }

    fn install_with_scope(
        table: super::module_table::ModuleTable,
        enums: crate::ir::GlobalEnums,
        resolved_imports: BTreeMap<(String, Vec<String>), String>,
        module_path: String,
    ) -> Self {
        Self {
            _guard: super::active_module_table::Guard::install(table),
            _enums: Some(crate::qualified_enums::GlobalGuard::install(enums)),
            _resolved_imports: Some(super::active_module_table::ResolvedImportsGuard::install(
                resolved_imports,
            )),
            _module_path: Some(crate::qualified_enums::ModuleGuard::install(module_path)),
            _consts: None,
        }
    }
}

/// Discover project context without rereading an entry the caller captured.
pub fn discover_with_source(
    entry: &Path,
    target_block: &str,
    entry_source: &str,
) -> Result<Discovery> {
    let (entry_module, entry_imports) = match crate::parser::parse_with_imports(entry_source) {
        Ok(pair) => pair,
        Err(_) => return Ok(Discovery::SingleTranslationUnit),
    };
    let _ = &entry_module;
    let direct = local_import_segments(&entry_imports);
    if direct.is_empty() {
        return Ok(Discovery::SingleTranslationUnit);
    }
    let entry_dir = entry.parent().unwrap_or_else(|| Path::new("."));
    let Some(project_root) = find_project_root_for_file_checked(entry_dir)? else {
        return Ok(Discovery::MissingProject(
            direct.iter().map(|path| path.join(".")).collect(),
        ));
    };
    let project_root = canonical_project_root(&project_root)?;
    let manifest = load_manifest(&project_root)?;
    let selected = manifest
        .targets
        .get(target_block)
        .or_else(|| manifest.targets.get(DEFAULT_TARGET_BLOCK));
    let (candidates, explicit_sources) = resolve_sources(
        &project_root,
        &manifest.build.entry,
        selected.and_then(|target| target.sources.as_deref()),
        false,
    )?;
    let resolved_entry = resolve_project_entry(&project_root, &manifest.build.entry)?;
    let source_root = if explicit_sources {
        project_root.clone()
    } else {
        resolved_entry
            .parent()
            .unwrap_or(&project_root)
            .to_path_buf()
    };
    let entry = entry
        .canonicalize()
        .with_context(|| format!("cannot resolve single-file entry {}", entry.display()))?;
    Ok(Discovery::Project(capture_scope_from_paths(
        &entry,
        entry_source,
        entry_module,
        entry_imports,
        &candidates,
        &source_root,
    )?))
}

pub fn discover(entry: &Path, target_block: &str) -> Result<Discovery> {
    let entry_source = fs::read_to_string(entry)
        .with_context(|| format!("cannot read single-file entry {}", entry.display()))?;
    discover_with_source(entry, target_block, &entry_source)
}

/// Capture a known project source set once for type-check and sibling linking.
pub fn capture_project_scope(
    entry: &Path,
    entry_source: &str,
    candidates: &[PathBuf],
    source_root: &Path,
) -> Result<ProjectScope> {
    let (entry_module, entry_imports) = crate::parser::parse_with_imports(entry_source)
        .map_err(|_| anyhow::anyhow!("entry {} does not parse", entry.display()))?;
    capture_scope_from_paths(
        entry,
        entry_source,
        entry_module,
        entry_imports,
        candidates,
        source_root,
    )
}

fn capture_scope_from_paths(
    entry: &Path,
    entry_source: &str,
    entry_module: Module,
    entry_imports: Vec<String>,
    candidates: &[PathBuf],
    source_root: &Path,
) -> Result<ProjectScope> {
    let entry_canonical = entry.canonicalize().unwrap_or_else(|_| entry.to_path_buf());
    let mut captured = Vec::with_capacity(candidates.len());
    for path in candidates {
        if path.canonicalize().unwrap_or_else(|_| path.clone()) == entry_canonical {
            continue;
        }
        let source = fs::read_to_string(path)
            .with_context(|| format!("cannot read declared project source {}", path.display()))?;
        captured.push((path.as_path(), source));
    }
    capture_scope(
        entry,
        entry_source,
        entry_module,
        entry_imports,
        captured
            .iter()
            .map(|(path, source)| (*path, source.as_str())),
        source_root,
    )
}

/// Build project scope from source text already captured by the build driver.
#[cfg(feature = "mlir-build")]
pub(crate) fn capture_project_scope_from_texts<'a>(
    entry: &Path,
    entry_source: &str,
    candidates: impl IntoIterator<Item = (&'a Path, &'a str)>,
    source_root: &Path,
) -> Result<ProjectScope> {
    let (entry_module, entry_imports) = crate::parser::parse_with_imports(entry_source)
        .map_err(|_| anyhow::anyhow!("entry {} does not parse", entry.display()))?;
    capture_scope(
        entry,
        entry_source,
        entry_module,
        entry_imports,
        candidates,
        source_root,
    )
}

fn capture_scope<'a>(
    entry: &Path,
    entry_source: &str,
    entry_module: Module,
    entry_imports: Vec<String>,
    candidates: impl IntoIterator<Item = (&'a Path, &'a str)>,
    source_root: &Path,
) -> Result<ProjectScope> {
    struct Candidate {
        path: PathBuf,
        module_path: String,
        source: String,
        module: Option<Module>,
        /// The parser's own record for this candidate, captured with the same
        /// parse that produced `module`. Empty when the source did not parse.
        imports: Vec<String>,
    }
    let entry_canonical = entry.canonicalize().unwrap_or_else(|_| entry.to_path_buf());
    let entry_module_path = super::module_table::module_path_of(entry, source_root);
    let mut by_path = BTreeMap::<String, Candidate>::new();
    let mut stems = BTreeMap::<String, Vec<String>>::new();
    for (path, source) in candidates {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if canonical == entry_canonical {
            continue;
        }
        let module_path = super::module_table::module_path_of(path, source_root);
        if module_path == entry_module_path || by_path.contains_key(&module_path) {
            anyhow::bail!("multiple project sources resolve to module path {module_path}");
        }
        let source = source.to_string();
        let (module, imports) = match crate::parser::parse_with_imports(&source) {
            Ok((m, i)) => (Some(m), i),
            Err(_) => (None, Vec::new()),
        };
        if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
            stems
                .entry(stem.to_string())
                .or_default()
                .push(module_path.clone());
        }
        by_path.insert(
            module_path.clone(),
            Candidate {
                path: path.to_path_buf(),
                module_path,
                source,
                module,
                imports,
            },
        );
    }

    fn resolve_import(
        path: &[String],
        candidates: &BTreeMap<String, Candidate>,
        stems: &BTreeMap<String, Vec<String>>,
    ) -> Option<String> {
        if matches!(path.first().map(String::as_str), Some("std")) {
            return None;
        }
        let key = path.join(".");
        if candidates.contains_key(&key) {
            return Some(key);
        }
        if !key.starts_with("crate.") {
            let crate_key = format!("crate.{key}");
            if candidates.contains_key(&crate_key) {
                return Some(crate_key);
            }
        }
        if path.len() == 1 {
            if let Some(matches) = stems.get(&path[0]) {
                if matches.len() == 1 {
                    return matches.first().cloned();
                }
            }
        }
        None
    }

    let mut pending = BTreeSet::new();
    for path in local_import_segments(&entry_imports) {
        if let Some(key) = resolve_import(&path, &by_path, &stems) {
            pending.insert(key);
        }
    }
    let mut linked = BTreeSet::from([entry_module_path.clone()]);
    while let Some(key) = pending.pop_first() {
        if linked.contains(&key) {
            continue;
        }
        let Some(candidate) = by_path.get(&key) else {
            continue;
        };
        let module = candidate.module.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "imported sibling {} does not parse",
                candidate.path.display()
            )
        })?;
        let _ = module;
        for path in local_import_segments(&candidate.imports) {
            if let Some(dep) = resolve_import(&path, &by_path, &stems) {
                if !linked.contains(&dep) {
                    pending.insert(dep);
                }
            }
        }
        linked.insert(key);
    }

    let mut resolved_imports = BTreeMap::new();
    for path in local_import_segments(&entry_imports) {
        if let Some(target) = resolve_import(&path, &by_path, &stems) {
            resolved_imports.insert((entry_module_path.clone(), path), target);
        }
    }
    for (owner, candidate) in &by_path {
        if candidate.module.is_some() {
            for path in local_import_segments(&candidate.imports) {
                if let Some(target) = resolve_import(&path, &by_path, &stems) {
                    resolved_imports.insert((owner.clone(), path), target);
                }
            }
        }
    }

    let mut sources = vec![CapturedSource {
        path: entry.to_path_buf(),
        module_path: entry_module_path,
        source: entry_source.to_string(),
        module: entry_module,
    }];
    sources.extend(by_path.into_values().filter_map(|candidate| {
        candidate.module.map(|module| CapturedSource {
            path: candidate.path,
            module_path: candidate.module_path,
            source: candidate.source,
            module,
        })
    }));
    let mut parsed = crate::project::stdlib::parsed_stdlib_modules();
    parsed.extend(
        sources
            .iter()
            .map(|source| (source.module_path.clone(), source.module.clone())),
    );
    let refs = parsed
        .iter()
        .map(|(path, module)| (path.clone(), module))
        .collect::<Vec<_>>();
    let table = super::module_table::build_module_table(&refs);
    // Cross-module const values for lowering. A name two modules define with
    // different values is left out of the table (`build_project_consts`), so
    // lowering never inlines the wrong owner's value; owner-qualified
    // references (`left.VALUE`) still check and evaluate.
    let (consts, _ambiguous) = super::build_project_consts(&parsed);
    // Build qualified type visibility with the same owner bindings used by
    // native linking; raw basenames are insufficient for a root-level entry.
    let _resolved_guard =
        super::active_module_table::ResolvedImportsGuard::install(resolved_imports.clone());
    let enums = super::build_global_enums(&parsed);
    Ok(ProjectScope {
        entry: entry.to_path_buf(),
        sources,
        linked,
        resolved_imports,
        table,
        enums: Box::new(enums),
        consts,
    })
}

/// Local (non-`std`) import paths, from the PARSER'S OWN record.
///
/// Replaces a hand-rolled walk that recursed into `Node::Block` only. That walk
/// could not see an import inside a function body, a branch, a loop, a match arm
/// or a closure, and it could not see one under an expression wrapper such as a
/// closure bound by `let` — a wildcard arm skips the wrapper before any
/// statement-bearing node is reached. Enumerating containers cannot fix that
/// class; it only moves the boundary.
///
/// `P::import_paths` is pushed unconditionally by both `parse_import` and
/// `parse_use`, and `parse_stmt` accepts those keywords wherever a statement is
/// accepted, so the record is container-independent by construction and is
/// written at parse time — before any question of use, reachability or support.
///
/// Policy preserved exactly: `std` is dropped (satisfied by the seed blob), FULL
/// dotted paths are split into ordered segments, and the `BTreeSet` return keeps
/// the previous sorted, deduplicated set semantics these call sites rely on.
fn local_import_segments(raw: &[String]) -> BTreeSet<Vec<String>> {
    raw.iter()
        .map(|dotted| {
            dotted
                .split('.')
                .map(str::to_string)
                .collect::<Vec<String>>()
        })
        .filter(|segs| !matches!(segs.first().map(String::as_str), Some("std")))
        .collect()
}

fn common_ancestor_dir(files: &[PathBuf]) -> PathBuf {
    let mut dirs = files.iter().filter_map(|file| file.parent());
    let Some(first) = dirs.next() else {
        return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    };
    let mut common = first.to_path_buf();
    for dir in dirs {
        while !dir.starts_with(&common) {
            if !common.pop() {
                return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            }
        }
    }
    common
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(module_path: &str, function: &str) -> super::super::module_table::ModuleTable {
        let source = format!("pub fn {function}() -> i64 {{ return 1; }}\n");
        let module = crate::parser::parse(&source).unwrap();
        super::super::module_table::build_module_table(&[(module_path.to_string(), &module)])
    }

    #[test]
    fn discovery_links_a_sibling_imported_only_from_inside_a_function() {
        // The control root asked for: it must NOT be able to pass merely because
        // the frozen compiler refuses nested imports. Nothing here runs the
        // frozen compiler, or any backend at all. It asserts what DISCOVERY
        // resolved, so the only way it passes is if the nested import was seen.
        //
        // MUTATION: restore the Block-only walk and this goes red — the entry's
        // sole import lives inside a function body, so a Block-only walk sees an
        // import-free entry and returns SingleTranslationUnit instead of a
        // two-module project.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Mind.toml"),
            "[package]\nname = \"nested\"\nversion = \"0.1.0\"\n\n\
             [build]\ntarget = \"cpu\"\nentry = \"src/main.mind\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/helper.mind"),
            "export { helper }\n\npub fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n",
        )
        .unwrap();
        // The ONLY import is nested inside the function body.
        let main_src = "fn main() -> i64 {\n    import helper;\n    return helper(6);\n}\n";
        std::fs::write(root.join("src/main.mind"), main_src).unwrap();
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(root)
            .status();

        let entry = root.join("src/main.mind");
        match discover_with_source(&entry, "cpu", main_src) {
            Ok(Discovery::Project(scope)) => {
                let linked: Vec<String> = scope
                    .linked_sources()
                    .map(|c| c.module_path().to_string())
                    .collect();
                assert!(
                    linked.iter().any(|m| m.ends_with("helper")),
                    "discovery must link the sibling named by a NESTED import; linked {linked:?}"
                );
            }
            Ok(Discovery::SingleTranslationUnit) => panic!(
                "discovery missed the nested import and treated the entry as a single \
                 translation unit — this is the Block-only walk's failure mode"
            ),
            Ok(Discovery::MissingProject(imports)) => {
                // Also acceptable evidence the import was SEEN, as long as it is named.
                assert!(
                    imports.iter().any(|i| i.contains("helper")),
                    "the nested import must be named; got {imports:?}"
                );
            }
            Err(e) => panic!("discovery failed: {e:#}"),
        }
    }

    #[test]
    fn local_import_segments_are_sorted_and_exclude_std() {
        // Now driven by the parser's own record, and covering a NESTED import
        // the previous Block-only walk could not see.
        let (_module, raw) = crate::parser::parse_with_imports(
            "import zed;\nimport std.vec;\nimport crate.alpha;\nimport zed;\n\
             fn main() -> i64 {\n    import nested.dep;\n    return 1;\n}\n",
        )
        .unwrap();
        assert_eq!(
            local_import_segments(&raw).into_iter().collect::<Vec<_>>(),
            [
                vec![String::from("crate"), String::from("alpha")],
                vec![String::from("nested"), String::from("dep")],
                vec![String::from("zed")],
            ],
            "std excluded, duplicates absorbed, sorted, and the NESTED import present"
        );
    }

    #[test]
    fn nested_scope_restores_the_previous_table_after_error() {
        crate::type_checker::cm_set_project_table(Some(table("crate.outer", "outer_fn")));
        let nested = || -> Result<()> {
            let _guard = ProjectTableGuard::install(table("crate.inner", "inner_fn"));
            assert_eq!(
                crate::type_checker::cm_imported_export_names(&["crate".into(), "inner".into()]),
                ["inner_fn"]
            );
            anyhow::bail!("test error")
        };
        assert!(nested().is_err());
        assert_eq!(
            crate::type_checker::cm_imported_export_names(&["crate".into(), "outer".into()]),
            ["outer_fn"]
        );
        crate::type_checker::cm_set_project_table(None);
    }

    #[test]
    fn metadata_scope_is_broader_than_the_linked_body_closure() {
        let root = tempfile::tempdir().unwrap();
        let entry = root.path().join("main.mind");
        let types = root.path().join("types.mind");
        let malformed = root.path().join("stray.mind");
        let entry_source = "pub fn field(p: &Point) -> i64 { return p.y; }\n";
        fs::write(&entry, entry_source).unwrap();
        fs::write(&types, "struct Point { x: i64, y: i64 }\n").unwrap();
        fs::write(&malformed, "fn broken( -> i64 {\n").unwrap();

        let scope = capture_project_scope(
            &entry,
            entry_source,
            &[entry.clone(), types, malformed],
            root.path(),
        )
        .unwrap();

        assert_eq!(
            scope
                .sources()
                .iter()
                .map(CapturedSource::module_path)
                .collect::<Vec<_>>(),
            ["crate", "crate.types"]
        );
        assert_eq!(
            scope
                .linked_sources()
                .map(CapturedSource::module_path)
                .collect::<Vec<_>>(),
            ["crate"]
        );
    }
}
