// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Cross-module import resolution — module table (Phase 10.6 item 9 /
//! Phase 15 self-hosting prerequisite).
//!
//! `use crate::foo::bar` parses today into `Node::Import { path }` and is
//! then discarded (eval returns `Int(0)`, the type-checker returns a
//! placeholder) — identical to the state `Node::Export` was in before
//! RFC 0002 D1. A multi-file MIND program (a compiler is many files)
//! cannot resolve a symbol declared in another file.
//!
//! This module is **deliverable 1 of the cross-module work**: the
//! `ModuleTable` data structure plus its population from parsed project
//! sources. It is intentionally *not yet wired* into the type-checker —
//! exactly how RFC 0002 landed `IRModule.exports` as an inert field
//! before the codegen pass consumed it. Landing the structure first,
//! gated and behavior-neutral, keeps the change reviewable and the
//! compile-speed moat provably untouched.
//!
//! A module's public surface is its `export { ... }` block names
//! (parser treats bare `pub` as a no-op; module visibility is the
//! export block — see `src/parser/mod.rs`), consistent with RFC 0002.
//!
//! Gated entirely behind `feature = "cross-module-imports"`. The
//! default build never compiles this file; default-build MLIR / IR /
//! frontend timings are byte-identical and the headline criterion
//! benches are unaffected (module-level gate, no per-statement cfg).

use std::collections::HashMap;
use std::path::Path;

use crate::ast::{Module, Node, TypeAnn};

/// Per-fn signature carried alongside the exported name list — RFC
/// 0005 Phase B.  Phase A let `use std.vec` make `vec_push` callable
/// at the type-checker level as a `ScalarI64`-returning intrinsic-
/// shape; Phase B carries the imported fn's full param + return
/// types so the import site can validate arity and per-arg types
/// against the declaration in the source module.
///
/// `Node::FnDef` and `extern "C"` function declarations populate this;
/// struct / const / enum exports are name-only. When the imported module has
/// an explicit `export { ... }` block (RFC 0002 surface), signatures are
/// captured only for names that the block exports and that resolve to
/// module-local functions or extern declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedFn {
    pub name: String,
    /// Source generic parameters are declaration authority, even though the
    /// current project table does not monomorphize them. Consumers must not
    /// mistake a non-empty list for a concrete, owner-qualified function.
    pub type_params: Vec<String>,
    pub param_types: Vec<TypeAnn>,
    pub ret_type: Option<TypeAnn>,
}

/// Type-only metadata for an enum exported by a module. Enum definitions do
/// not occupy the runtime value namespace, but their variants and payloads are
/// part of the module's type surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedEnum {
    pub name: String,
    pub variants: Vec<ExportedEnumVariant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedEnumVariant {
    pub name: String,
    pub payload: Vec<TypeAnn>,
    pub field_names: Vec<String>,
}

/// The public symbols a single module exports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModuleExports {
    /// Dotted module path, e.g. `crate.lexer.token`.
    pub module_path: String,
    /// Names declared in this module's `export { ... }` block(s),
    /// sorted + deduplicated for deterministic resolution.
    pub exported: Vec<String>,
    /// RFC 0005 Phase B — full signatures for every auto-exported
    /// `pub fn`.  Indexed by fn name; populated only on the
    /// auto-export path (the `export { ... }` block path stays
    /// empty by construction).
    pub exported_fns: Vec<ExportedFn>,
    /// `extern "C"` declarations. Kept off `exported_fns` so a cross-module
    /// call is not typed as an ordinary MIND `func.call`. The lowerer emits
    /// `ExternFnDecl` from this list and uses the C return map.
    pub exported_externs: Vec<ExportedExtern>,
    /// Type-only enum metadata. Kept separate from `exported` so enum types
    /// never become callable/runtime values merely because they are public.
    pub exported_enums: Vec<ExportedEnum>,
}

/// An exported `extern "C"` declaration, including the calling convention
/// the ordinary function signature does not carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedExtern {
    pub name: String,
    pub callconv: crate::ast::CallConv,
    pub is_varargs: bool,
    pub param_types: Vec<TypeAnn>,
    pub ret_type: Option<TypeAnn>,
}

/// Maps dotted module path -> its exported surface.
#[derive(Debug, Clone, Default)]
pub struct ModuleTable {
    modules: HashMap<String, ModuleExports>,
}

impl ModuleTable {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn get(&self, module_path: &str) -> Option<&ModuleExports> {
        self.modules.get(module_path)
    }

    /// Resolve an import path against canonical `crate.*` module keys.
    pub fn get_import(&self, import_path: &[String]) -> Option<&ModuleExports> {
        let key = import_path.join(".");
        self.get(&key).or_else(|| {
            (!key.starts_with("crate."))
                .then(|| format!("crate.{key}"))
                .and_then(|key| self.get(&key))
        })
    }

    /// Insert a module's exports. A re-inserted path overwrites
    /// (last-write-wins); callers feed sources in a deterministic order.
    pub fn insert(&mut self, exports: ModuleExports) {
        self.modules.insert(exports.module_path.clone(), exports);
    }

    /// Remove `names` from every module's exported value surface. Used to drop
    /// POISONED cross-module const names — a const defined with CONFLICTING
    /// values in more than one module (see `build_project_consts`) — so a
    /// cross-module reference to such a name is reported unresolved (E2002) and
    /// fails closed with a clean diagnostic, instead of resolving here and then
    /// reaching the value-less fail-closed panic in lowering. A module's OWN use
    /// of the const is unaffected: it resolves against the module's local scope /
    /// MODULE_CONSTS, not this cross-module surface. No-op when `names` is empty
    /// (the common case), so a project with no conflicting consts is untouched.
    pub fn prune_exported(&mut self, names: &std::collections::BTreeSet<String>) {
        if names.is_empty() {
            return;
        }
        for m in self.modules.values_mut() {
            m.exported.retain(|e| !names.contains(e));
        }
    }

    /// True iff `symbol` is exported by the module at `import_path`.
    /// `import_path` is the dotted path from a `use a.b.c` (the
    /// `Node::Import.path` segments). Resolution is exact — no globs,
    /// no re-export chains (deliverable 2+).
    pub fn resolves(&self, import_path: &[String], symbol: &str) -> bool {
        self.get_import(import_path)
            .is_some_and(|m| m.exported.iter().any(|e| e == symbol))
    }

    /// True iff ANY module in the table exports `symbol` (any kind — fn,
    /// const, type, struct). Answers the bare cross-module resolvability
    /// question for the type-checker's undefined-reference pass; module
    /// functions/consts share one global link unit, so an exported name
    /// referenced from a sibling module is never genuinely-undefined.
    pub fn exports_symbol(&self, symbol: &str) -> bool {
        self.modules
            .values()
            .any(|m| m.exported.iter().any(|e| e == symbol))
    }

    /// Extern declarations with one signature across the table.
    /// Conflicting declarations omit the name. These are not ordinary MIND functions.
    /// Lowering uses the import-scoped active resolver instead of this global view.
    pub fn unique_exported_externs(&self) -> Vec<ExportedExtern> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        let mut grouped: std::collections::BTreeMap<&str, Vec<&ExportedExtern>> =
            std::collections::BTreeMap::new();
        for key in keys {
            if let Some(module) = self.modules.get(key) {
                for ext in &module.exported_externs {
                    grouped.entry(ext.name.as_str()).or_default().push(ext);
                }
            }
        }
        grouped
            .into_iter()
            .filter_map(|(_, group)| {
                let first = *group.first()?;
                group.iter().all(|ext| *ext == first).then(|| first.clone())
            })
            .collect()
    }

    /// RFC 0005 Phase B — look up an imported fn's signature by name,
    /// searching every module in the table. A name with more than one owner
    /// is ambiguous and returns `None`; callers must not select a hash-map
    /// iteration winner for a cross-module ABI. Returns `None` if the name
    /// resolves only to a struct / const / enum or to an `export { ... }`-
    /// block name without a captured signature.
    pub fn lookup_imported_fn(&self, name: &str) -> Option<&ExportedFn> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        let mut found = None;
        for key in keys {
            if let Some(m) = self.modules.get(key) {
                if let Some(f) = m.exported_fns.iter().find(|f| f.name == name) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some(f);
                }
            }
        }
        found
    }

    /// RFC 0012 §5.1 — every imported `pub fn` signature across the whole
    /// project, in deterministic (sorted-path, then declaration) order. The
    /// AST→IR lowering registers these into `IRModule::fn_signatures` so a
    /// cross-module `func.call` is typed against the callee's DECLARED param /
    /// return ABI (i64/i32/f32/f64/bool) instead of the legacy all-i64 default
    /// — the default was a silent-miscompile-or-loud-mlir-error for any
    /// non-i64-signatured imported callee (e.g. an f64 `scale`). Empty on the
    /// single-file / default-feature path (no project table), so that path is
    /// byte-identical.
    pub fn all_exported_fns(&self) -> Vec<&ExportedFn> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            if let Some(m) = self.modules.get(key) {
                out.extend(m.exported_fns.iter());
            }
        }
        out
    }

    /// Borrow C declarations with their defining owner for import-scoped ABI selection.
    pub(crate) fn all_exported_extern_refs(&self) -> Vec<(&str, &ExportedExtern)> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            if let Some(module) = self.modules.get(key) {
                out.extend(
                    module
                        .exported_externs
                        .iter()
                        .map(|ext| (key.as_str(), ext)),
                );
            }
        }
        out
    }

    /// Borrow every typed function candidate with its defining owner.  The
    /// active project resolver uses this form while the table is installed;
    /// cloning a whole `ModuleTable` or each candidate would make every
    /// visible-signature query scale with the complete project payload.
    pub(crate) fn exported_fn_candidate_refs(&self, name: &str) -> Vec<(&str, &ExportedFn)> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        keys.into_iter()
            .filter_map(|key| {
                self.modules.get(key).and_then(|module| {
                    module
                        .exported_fns
                        .iter()
                        .find(|function| function.name == name)
                        .map(|function| (key.as_str(), function))
                })
            })
            .collect()
    }

    /// Borrow all typed functions in deterministic owner/declaration order.
    pub(crate) fn all_exported_fn_refs(&self) -> Vec<(&str, &ExportedFn)> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            if let Some(module) = self.modules.get(key) {
                out.extend(
                    module
                        .exported_fns
                        .iter()
                        .map(|function| (key.as_str(), function)),
                );
            }
        }
        out
    }

    /// Borrow every owner that exports a value/type name.  This is the
    /// allocation-light counterpart used by the active resolver.
    pub(crate) fn exported_symbol_candidate_refs(&self, name: &str) -> Vec<&str> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        keys.into_iter()
            .filter(|key| {
                self.modules
                    .get(*key)
                    .is_some_and(|module| module.exported.iter().any(|item| item == name))
            })
            .map(String::as_str)
            .collect()
    }

    /// Borrow every exported value/type name with its defining owner in one
    /// deterministic pass. The active resolver uses this to build the whole
    /// visible symbol set without rescanning every module for each name.
    pub(crate) fn all_exported_symbol_refs(&self) -> Vec<(&str, &str)> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        let mut out = Vec::new();
        for key in keys {
            if let Some(module) = self.modules.get(key) {
                out.extend(
                    module
                        .exported
                        .iter()
                        .map(|name| (key.as_str(), name.as_str())),
                );
            }
        }
        out
    }
}

/// Module-file stems that name the CRATE ROOT rather than a submodule.
/// A file `main.mind` / `lib.mind` / `mod.mind` collapses to its parent path
/// (`src/main.mind` -> `crate`), so it can NOT be imported under its own stem
/// (`use lib;` will not resolve to a `lib` module — the file is `crate`, not
/// `crate.lib`). This is the single source of truth for that reserved set.
pub const RESERVED_ROOT_STEMS: [&str; 3] = ["main", "lib", "mod"];

/// `true` when `stem` is a reserved crate-root module name (see
/// [`RESERVED_ROOT_STEMS`]). Callers use this to diagnose a file whose name
/// collapses to the crate root instead of resolving as a named module.
pub fn is_reserved_root_stem(stem: &str) -> bool {
    RESERVED_ROOT_STEMS.contains(&stem)
}

/// Derive a dotted module path from a source file path relative to the
/// project source root. `src/lexer/token.mind` -> `crate.lexer.token`;
/// `src/main.mind` -> `crate`.
pub fn module_path_of(file: &Path, src_root: &Path) -> String {
    let rel = file.strip_prefix(src_root).unwrap_or(file);
    let mut parts: Vec<String> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .map(|s| s.trim_end_matches(".mind").to_string())
        .filter(|s| !s.is_empty())
        .collect();
    // `main` / `lib` / `mod` are the crate root, not a named submodule.
    if matches!(parts.last().map(String::as_str), Some(s) if is_reserved_root_stem(s)) {
        parts.pop();
    }
    if parts.is_empty() {
        "crate".to_string()
    } else {
        format!("crate.{}", parts.join("."))
    }
}

/// Collect a module's exported names from its parsed AST.
///
/// Resolution order (RFC 0005 Phase 2 ergonomics):
///   1. If the module declares one or more `export { ... }` blocks,
///      those names are the exported surface — this is the explicit,
///      pre-existing contract (RFC 0002).
///   2. Otherwise, every top-level `Node::FnDef` and `Node::Struct`
///      name is auto-exported.  This is what lets `std/*.mind` files
///      compose into the `use std.vec` surface without a per-file
///      `export` block — the parser strips `pub` to a no-op already
///      (`src/parser/mod.rs:736`), so a `pub fn`-only file would
///      otherwise have an empty exported surface.
///
/// Deterministic (sorted, deduped) in both branches.
///
/// The `module NAME { ... }` block form (used by whole-file `module` modules)
/// parses to a transparent top-level `Node::Block` whose `stmts` are the real
/// module items (`export` / `FnDef` / `StructDef` …). Flatten those blocks
/// first — mirroring `resolve::collect_decl_names` — so a block-wrapped
/// module's exports are collected rather than silently dropped (which left its
/// public surface EMPTY and every cross-module call to it unresolved → E2003).
pub fn collect_module_exports(module_path: &str, ast: &Module) -> ModuleExports {
    // Flatten transparent `module { ... }` blocks into one item list.
    let mut items: Vec<&Node> = Vec::new();
    fn flatten<'a>(src: &'a [Node], out: &mut Vec<&'a Node>) {
        for item in src {
            match item {
                Node::Block { stmts, .. } => flatten(stmts, out),
                other => out.push(other),
            }
        }
    }
    flatten(&ast.items, &mut items);
    // Exported signatures retain the defining module's structural types.
    // Resolve aliases here, while their owner scope is available, so a
    // consumer cannot mistake `Items` for an unknown scalar annotation.
    let aliases = crate::eval::type_aliases::LocalTypeAliases::new(&ast.items);

    fn remember_extern_fn(
        exported_externs: &mut Vec<ExportedExtern>,
        aliases: &crate::eval::type_aliases::LocalTypeAliases,
        callconv: crate::ast::CallConv,
        function: &crate::ast::ExternFn,
    ) {
        let candidate = ExportedExtern {
            name: function.name.clone(),
            callconv: crate::eval::lower::resolve_callconv(callconv),
            is_varargs: function.is_varargs,
            param_types: function
                .params
                .iter()
                .map(|param| aliases.resolve(&param.ty))
                .collect(),
            ret_type: function.ret_type.as_ref().map(|ty| aliases.resolve(ty)),
        };
        // Coalesce only identical declarations. Preserve conflicting records
        // so callers cannot recover a first-wins ABI from an invalid module;
        // the source checker reports the conflict before any lowering.
        if !exported_externs.contains(&candidate) {
            exported_externs.push(candidate);
        }
    }

    let mut exported: Vec<String> = Vec::new();
    let mut exported_externs: Vec<ExportedExtern> = Vec::new();
    let mut exported_fns: Vec<ExportedFn> = Vec::new();
    let mut has_explicit_export = false;
    for item in &items {
        if let Node::Export { names, .. } = item {
            has_explicit_export = true;
            exported.extend(names.iter().cloned());
        }
    }
    if has_explicit_export {
        // An `export { ... }` block declares NAMES only. Capture the full
        // signature for every exported name that resolves to a module-local
        // `fn` so `cm_lookup_fn` (call-site arity/type check + resolver
        // `call_resolvable`) sees explicitly-exported fns, exactly as the
        // auto-export path below already does for `pub fn`. Without this, a
        // `module NAME { export fn f  fn f() {} }` file exported the NAME `f`
        // but no signature, so a sibling's `f(...)` call stayed unresolved.
        let exported_set: std::collections::BTreeSet<&String> = exported.iter().collect();
        for item in &items {
            match item {
                Node::FnDef(fd, _) => {
                    let (name, params, ret_type) = (&fd.name, &fd.params, &fd.ret_type);
                    if exported_set.contains(name) {
                        exported_fns.push(ExportedFn {
                            name: name.clone(),
                            type_params: fd.type_params.clone(),
                            param_types: params.iter().map(|p| aliases.resolve(&p.ty)).collect(),
                            ret_type: ret_type.as_ref().map(|ty| aliases.resolve(ty)),
                        });
                    }
                }
                Node::ExternBlock { callconv, fns, .. } => {
                    for function in fns {
                        if exported_set.contains(&function.name) {
                            remember_extern_fn(
                                &mut exported_externs,
                                &aliases,
                                *callconv,
                                function,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    } else {
        for item in &items {
            match item {
                // Phase B: capture the full signature alongside the
                // name.  `params` is `Vec<Param>` and each `Param`
                // already carries its declared `ty: TypeAnn` (or a
                // synthesized one for untyped params).
                Node::FnDef(fd, _) => {
                    let (name, params, ret_type) = (&fd.name, &fd.params, &fd.ret_type);
                    exported.push(name.clone());
                    exported_fns.push(ExportedFn {
                        name: name.clone(),
                        type_params: fd.type_params.clone(),
                        param_types: params.iter().map(|p| aliases.resolve(&p.ty)).collect(),
                        ret_type: ret_type.as_ref().map(|ty| aliases.resolve(ty)),
                    });
                }
                Node::StructDef { name, .. } => exported.push(name.clone()),
                // A cross-module `pub const` is referenced BARE at a sibling use
                // site — e.g. mind-nerve's `MAX_REQUEST_TOKENS` (defined in
                // `src.lib`, used bare in `src.tokenizer`) — so it must be in the
                // exported surface, or `cm_symbol_exported` reports it as a
                // genuinely-undefined identifier (E2002) and lowering hits the
                // fail-closed panic. ONLY `Node::Const` is exported for
                // value-position resolution: its VALUE is carried by the project
                // const table (`build_project_consts`). ExternConst / TypeAlias /
                // EnumDef are deliberately NOT exported here — a bare value-use of
                // them has NO value backing in that table, so accepting the name
                // would merely move the fail-closed panic onto the native compile
                // path with no diagnostic. Enum variants
                // already resolve via the global-enums registry; a type-position
                // use of a sibling type/alias goes through type resolution, not
                // this value-ident surface.
                Node::Const { name, .. } => exported.push(name.clone()),
                Node::ExternBlock { callconv, fns, .. } => {
                    for function in fns {
                        if !exported.iter().any(|name| name == &function.name) {
                            exported.push(function.name.clone());
                        }
                        remember_extern_fn(&mut exported_externs, &aliases, *callconv, function);
                    }
                }
                _ => {}
            }
        }
    }
    exported.sort();
    exported.dedup();
    // Sort exported_fns by name so resolution is deterministic across
    // file orderings — same discipline as `exported`.  Names are
    // unique within a single module (the parser rejects duplicates),
    // so this is also a stable order.
    exported_fns.sort_by(|a, b| a.name.cmp(&b.name));
    // A MIND function and an extern of the same name are not one ABI.
    // Drop the extern record rather than letting the first declaration win.
    exported_externs.retain(|ext| exported_fns.iter().all(|fun| fun.name != ext.name));
    exported_externs.sort_by(|a, b| a.name.cmp(&b.name));
    let mut exported_enums = collect_module_enums(ast);
    if has_explicit_export {
        exported_enums.retain(|decl| exported.iter().any(|name| name == &decl.name));
    }
    ModuleExports {
        module_path: module_path.to_string(),
        exported,
        exported_fns,
        exported_externs,
        exported_enums,
    }
}

/// Collect all module-level enum declarations as type-only export metadata.
/// Transparent `module { ... }` blocks are flattened using the same traversal
/// as value exports, keeping this the single AST→module-enum metadata source.
pub fn collect_module_enums(ast: &Module) -> Vec<ExportedEnum> {
    let mut enums = Vec::new();
    let mut pending: Vec<&Node> = ast.items.iter().rev().collect();
    while let Some(item) = pending.pop() {
        match item {
            Node::Block { stmts, .. } => pending.extend(stmts.iter().rev()),
            Node::EnumDef { name, variants, .. } => enums.push(ExportedEnum {
                name: name.clone(),
                variants: variants
                    .iter()
                    .map(|v| ExportedEnumVariant {
                        name: v.name.clone(),
                        payload: v.payload.clone(),
                        field_names: v.field_names.clone(),
                    })
                    .collect(),
            }),
            _ => {}
        }
    }
    enums.sort_by(|a, b| a.name.cmp(&b.name));
    enums
}

/// Build a `ModuleTable` from `(module_path, parsed AST)` pairs. Pure;
/// no I/O. The project loader supplies the pairs in a deterministic
/// (sorted by path) order.
pub fn build_module_table(parsed: &[(String, &Module)]) -> ModuleTable {
    let mut table = ModuleTable::new();
    for (path, ast) in parsed {
        table.insert(collect_module_exports(path, ast));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use std::path::PathBuf;

    #[test]
    fn empty_program_yields_no_exports() {
        let m = parse("1 + 2").expect("parse");
        let ex = collect_module_exports("crate.x", &m);
        assert_eq!(ex.module_path, "crate.x");
        assert!(ex.exported.is_empty());
    }

    #[test]
    fn export_block_names_are_collected_sorted() {
        let m = parse("export { zeta, alpha, mid }").expect("parse");
        let ex = collect_module_exports("crate.lex", &m);
        assert_eq!(ex.exported, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn table_resolves_exact_path_and_symbol() {
        let a = parse("export { tokenize }").expect("parse");
        let b = parse("export { parse_expr }").expect("parse");
        let table = build_module_table(&[
            ("crate.lexer".to_string(), &a),
            ("crate.parser".to_string(), &b),
        ]);
        assert_eq!(table.len(), 2);
        assert!(table.resolves(&["crate".into(), "lexer".into()], "tokenize"));
        assert!(table.resolves(&["crate".into(), "parser".into()], "parse_expr"));
        // wrong module
        assert!(!table.resolves(&["crate".into(), "lexer".into()], "parse_expr"));
        // unknown module
        assert!(!table.resolves(&["crate".into(), "nope".into()], "tokenize"));
    }

    #[test]
    fn module_path_derivation() {
        let root = PathBuf::from("/p/src");
        assert_eq!(
            module_path_of(&PathBuf::from("/p/src/lexer/token.mind"), &root),
            "crate.lexer.token"
        );
        assert_eq!(
            module_path_of(&PathBuf::from("/p/src/main.mind"), &root),
            "crate"
        );
        assert_eq!(
            module_path_of(&PathBuf::from("/p/src/lib.mind"), &root),
            "crate"
        );
        assert_eq!(
            module_path_of(&PathBuf::from("/p/src/util.mind"), &root),
            "crate.util"
        );
    }

    // Phase 17.8: `lib` (and `main`/`mod`) is a RESERVED crate-root stem, not an
    // importable module name. `src/lib.mind` collapses to `crate` — so `use lib;`
    // can never resolve to a `crate.lib` module; the byte-identical file under any
    // other name (`mylib.mind`) becomes `crate.mylib` and resolves. This
    // documents the collision by name so the driver can diagnose it.
    #[test]
    fn lib_is_a_reserved_root_stem_not_an_importable_module() {
        let root = PathBuf::from("/p/src");
        // Reserved: collapses to the crate root.
        assert_eq!(
            module_path_of(&PathBuf::from("/p/src/lib.mind"), &root),
            "crate"
        );
        assert!(is_reserved_root_stem("lib"));
        assert!(is_reserved_root_stem("main"));
        assert!(is_reserved_root_stem("mod"));
        // NOT reserved: resolves as a named module.
        assert_eq!(
            module_path_of(&PathBuf::from("/p/src/mylib.mind"), &root),
            "crate.mylib"
        );
        assert!(!is_reserved_root_stem("mylib"));
    }

    #[test]
    fn auto_export_path_captures_fn_signatures() {
        // RFC 0005 Phase B — when no `export { ... }` block is present,
        // every top-level `pub fn` is auto-exported AND its full
        // signature lands in `exported_fns`.
        let src = "pub fn vec_get(v: i64, i: i64) -> i64 { v }\n\
                   pub fn vec_new() -> i64 { 0 }\n";
        let ast = parse(src).expect("parse");
        let ex = collect_module_exports("std.vec", &ast);
        assert_eq!(ex.exported, vec!["vec_get", "vec_new"]);
        assert_eq!(ex.exported_fns.len(), 2);
        // Sorted by name.
        assert_eq!(ex.exported_fns[0].name, "vec_get");
        assert_eq!(ex.exported_fns[0].param_types.len(), 2);
        assert_eq!(ex.exported_fns[1].name, "vec_new");
        assert!(ex.exported_fns[1].param_types.is_empty());
        assert!(ex.exported_fns[1].ret_type.is_some());
    }

    #[test]
    fn exported_fn_signatures_resolve_owner_aliases() {
        let src = "struct Item { value: i64 }\n\
                   type Items = [Item; 2]\n\
                   pub fn make() -> Items { [Item { value: 7 }, Item { value: 9 }] }\n";
        let ast = parse(src).expect("parse");
        let ex = collect_module_exports("crate.items", &ast);
        let make = ex
            .exported_fns
            .iter()
            .find(|f| f.name == "make")
            .expect("make signature");
        assert_eq!(
            make.ret_type,
            Some(TypeAnn::Array {
                element: Box::new(TypeAnn::Named("Item".to_string())),
                length: 2,
            })
        );
    }

    #[test]
    fn explicit_export_block_captures_local_fn_signatures() {
        // An `export { ... }` block declares names; when an exported name
        // resolves to a module-LOCAL `fn`, its signature is captured in
        // `exported_fns` too — so a sibling module's call to that fn resolves
        // (with arity/type info) instead of staying undefined (E2003). An
        // exported name with no local `fn` (a re-exported symbol / const)
        // contributes a NAME only, no signature.
        let src = "export { foo, bar, aconst }\nfn foo() {}\nfn bar(x: i64) {}\n";
        let ast = parse(src).expect("parse");
        let ex = collect_module_exports("crate.x", &ast);
        assert_eq!(ex.exported, vec!["aconst", "bar", "foo"]);
        // Both local fns get signatures; `aconst` (no local fn) does not.
        assert_eq!(ex.exported_fns.len(), 2);
        assert_eq!(ex.exported_fns[0].name, "bar");
        assert_eq!(ex.exported_fns[0].param_types.len(), 1);
        assert_eq!(ex.exported_fns[1].name, "foo");
        assert!(ex.exported_fns[1].param_types.is_empty());
    }

    #[test]
    fn extern_decl_return_abi_is_part_of_the_export_surface() {
        let src = "extern \"C\" {\n    fn reduce_sum(x: i64) -> i32\n}\n";
        let ast = parse(src).expect("parse");
        let ex = collect_module_exports("crate.numerics", &ast);
        assert_eq!(ex.exported, vec!["reduce_sum"]);
        assert!(ex.exported_fns.is_empty());
        assert_eq!(ex.exported_externs.len(), 1);
        assert_eq!(ex.exported_externs[0].name, "reduce_sum");
        assert_eq!(ex.exported_externs[0].param_types, vec![TypeAnn::ScalarI64]);
        assert_eq!(ex.exported_externs[0].ret_type, Some(TypeAnn::ScalarI32));
        assert!(!ex.exported_externs[0].is_varargs);
    }

    #[test]
    fn explicit_export_keeps_listed_extern_return() {
        let src = "export { reduce_sum }\nextern \"C\" {\n    fn reduce_sum(x: i64) -> i32\n    fn hidden(x: i64) -> i32\n}\n";
        let ast = parse(src).expect("parse");
        let ex = collect_module_exports("crate.numerics", &ast);
        assert_eq!(ex.exported, vec!["reduce_sum"]);
        assert!(ex.exported_fns.is_empty());
        assert_eq!(ex.exported_externs.len(), 1);
        assert_eq!(ex.exported_externs[0].ret_type, Some(TypeAnn::ScalarI32));
    }

    #[test]
    fn imported_extern_lookup_keeps_declared_return() {
        let src = "extern \"C\" {\n    fn reduce_sum(x: i64) -> i32\n}\n";
        let ast = parse(src).expect("parse");
        let table = build_module_table(&[("crate.numerics".to_string(), &ast)]);
        assert!(table.lookup_imported_fn("reduce_sum").is_none());
        let found = table
            .unique_exported_externs()
            .into_iter()
            .find(|ext| ext.name == "reduce_sum")
            .expect("extern record");
        assert_eq!(found.ret_type, Some(TypeAnn::ScalarI32));
        assert_eq!(found.param_types, vec![TypeAnn::ScalarI64]);
    }

    #[test]
    fn explicit_exports_do_not_publish_unlisted_enum_types() {
        let ast = parse(
            "export { make }\nenum Color { Red(i64), Blue }\nfn make() -> Color { Color.Red(1) }\n",
        )
        .expect("parse");
        let ex = collect_module_exports("crate.defs", &ast);
        assert!(ex.exported_enums.is_empty());

        let public = parse("enum Color { Red(i64), Blue }\n").expect("parse");
        let ex = collect_module_exports("crate.defs", &public);
        assert_eq!(ex.exported_enums.len(), 1);
        assert_eq!(ex.exported_enums[0].name, "Color");
    }

    #[test]
    fn module_block_form_exports_are_collected() {
        // A `module NAME { export fn f  fn f() {} }` block parses to a
        // transparent top-level `Node::Block`; its nested exports must be
        // collected (previously dropped, leaving the surface empty → every
        // cross-module call to `f` unresolved).
        let src = "module m {\n  export fn f\n  fn f(x: i64) -> i64 { x }\n}\n";
        let ast = parse(src).expect("parse");
        let ex = collect_module_exports("crate.m", &ast);
        assert_eq!(ex.exported, vec!["f"]);
        assert_eq!(ex.exported_fns.len(), 1);
        assert_eq!(ex.exported_fns[0].name, "f");
    }

    #[test]
    fn lookup_imported_fn_searches_every_module() {
        // The cross-module Phase-B resolver needs to find `vec_new` by
        // name from any consumer file that issued `use std.vec`.  The
        // lookup walks every module in deterministic order.
        let vec_ast = parse("pub fn vec_new() -> i64 { 0 }").expect("parse");
        let io_ast = parse("pub fn stdout() -> i64 { 1 }").expect("parse");
        let table = build_module_table(&[
            ("std.vec".to_string(), &vec_ast),
            ("std.io".to_string(), &io_ast),
        ]);
        assert!(table.lookup_imported_fn("vec_new").is_some());
        assert!(table.lookup_imported_fn("stdout").is_some());
        assert!(table.lookup_imported_fn("not_a_fn").is_none());
    }

    #[test]
    fn lookup_imported_fn_refuses_same_name_from_multiple_owners() {
        let a = parse("fn f(x: i64) -> i64 { x }").expect("parse");
        let b = parse("fn f(x: f64) -> f64 { x }").expect("parse");
        let table = build_module_table(&[("crate.a".into(), &a), ("crate.b".into(), &b)]);
        assert!(table.lookup_imported_fn("f").is_none());
    }

    #[test]
    fn reinsert_is_last_write_wins() {
        let mut t = ModuleTable::new();
        t.insert(ModuleExports {
            module_path: "crate.a".into(),
            exported: vec!["old".into()],
            exported_fns: vec![],
            exported_externs: vec![],
            exported_enums: vec![],
        });
        t.insert(ModuleExports {
            module_path: "crate.a".into(),
            exported: vec!["new".into()],
            exported_fns: vec![],
            exported_externs: vec![],
            exported_enums: vec![],
        });
        assert!(t.resolves(&["crate".into(), "a".into()], "new"));
        assert!(!t.resolves(&["crate".into(), "a".into()], "old"));
    }
}
