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

use crate::ast::{Module, Node};

/// The public symbols a single module exports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModuleExports {
    /// Dotted module path, e.g. `crate.lexer.token`.
    pub module_path: String,
    /// Names declared in this module's `export { ... }` block(s),
    /// sorted + deduplicated for deterministic resolution.
    pub exported: Vec<String>,
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

/// Collect a module's exported names from its parsed AST: every name in
/// every `export { ... }` block. Deterministic (sorted, deduped).
pub fn collect_module_exports(module_path: &str, ast: &Module) -> ModuleExports {
    let mut exported: Vec<String> = Vec::new();
    for item in &ast.items {
        if let Node::Export { names, .. } = item {
            exported.extend(names.iter().cloned());
        }
    }
    exported.sort();
    exported.dedup();
    ModuleExports {
        module_path: module_path.to_string(),
        exported,
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
    fn reinsert_is_last_write_wins() {
        let mut t = ModuleTable::new();
        t.insert(ModuleExports {
            module_path: "crate.a".into(),
            exported: vec!["old".into()],
        });
        t.insert(ModuleExports {
            module_path: "crate.a".into(),
            exported: vec!["new".into()],
        });
        assert!(t.resolves(&["crate".into(), "a".into()], "new"));
        assert!(!t.resolves(&["crate".into(), "a".into()], "old"));
    }
}
