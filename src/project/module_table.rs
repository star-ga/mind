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
/// Only `Node::FnDef` populates this; struct / const / enum exports
/// are name-only.  When the imported module has an explicit
/// `export { ... }` block (RFC 0002 surface), `exported_fns` stays
/// empty because the block declares names, not signatures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedFn {
    pub name: String,
    pub param_types: Vec<TypeAnn>,
    pub ret_type: Option<TypeAnn>,
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

    /// Insert a module's exports. A re-inserted path overwrites
    /// (last-write-wins); callers feed sources in a deterministic order.
    pub fn insert(&mut self, exports: ModuleExports) {
        self.modules.insert(exports.module_path.clone(), exports);
    }

    /// True iff `symbol` is exported by the module at `import_path`.
    /// `import_path` is the dotted path from a `use a.b.c` (the
    /// `Node::Import.path` segments). Resolution is exact — no globs,
    /// no re-export chains (deliverable 2+).
    pub fn resolves(&self, import_path: &[String], symbol: &str) -> bool {
        let key = import_path.join(".");
        self.modules
            .get(&key)
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

    /// RFC 0005 Phase B — look up an imported fn's signature by name,
    /// searching every module in the table.  Returns the first match
    /// in deterministic (sorted-path) iteration order; in practice
    /// `pub fn` names are unique within a project (parser-enforced
    /// per module; the project-loader convention prevents
    /// cross-module shadowing).  Returns `None` if the name resolves
    /// only to a struct / const / enum or to an `export { ... }`-
    /// block name without a captured signature — the caller falls
    /// back to Phase-A loose typing in that case.
    pub fn lookup_imported_fn(&self, name: &str) -> Option<&ExportedFn> {
        let mut keys: Vec<&String> = self.modules.keys().collect();
        keys.sort();
        for key in keys {
            if let Some(m) = self.modules.get(key) {
                if let Some(f) = m.exported_fns.iter().find(|f| f.name == name) {
                    return Some(f);
                }
            }
        }
        None
    }
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
    if matches!(
        parts.last().map(String::as_str),
        Some("main" | "lib" | "mod")
    ) {
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

    let mut exported: Vec<String> = Vec::new();
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
            if let Node::FnDef {
                name,
                params,
                ret_type,
                ..
            } = item
            {
                if exported_set.contains(name) {
                    exported_fns.push(ExportedFn {
                        name: name.clone(),
                        param_types: params.iter().map(|p| p.ty.clone()).collect(),
                        ret_type: ret_type.clone(),
                    });
                }
            }
        }
    } else {
        for item in &items {
            match item {
                // Phase B: capture the full signature alongside the
                // name.  `params` is `Vec<Param>` and each `Param`
                // already carries its declared `ty: TypeAnn` (or a
                // synthesized one for untyped params).
                Node::FnDef {
                    name,
                    params,
                    ret_type,
                    ..
                } => {
                    exported.push(name.clone());
                    exported_fns.push(ExportedFn {
                        name: name.clone(),
                        param_types: params.iter().map(|p| p.ty.clone()).collect(),
                        ret_type: ret_type.clone(),
                    });
                }
                Node::StructDef { name, .. } => exported.push(name.clone()),
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
    ModuleExports {
        module_path: module_path.to_string(),
        exported,
        exported_fns,
    }
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
    fn reinsert_is_last_write_wins() {
        let mut t = ModuleTable::new();
        t.insert(ModuleExports {
            module_path: "crate.a".into(),
            exported: vec!["old".into()],
            exported_fns: vec![],
        });
        t.insert(ModuleExports {
            module_path: "crate.a".into(),
            exported: vec!["new".into()],
            exported_fns: vec![],
        });
        assert!(t.resolves(&["crate".into(), "a".into()], "new"));
        assert!(!t.resolves(&["crate".into(), "a".into()], "old"));
    }
}
