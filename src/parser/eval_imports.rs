// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Evaluator-only provenance for namespace references.
//!
//! The compiler AST intentionally normalises `module.symbol` to a bare call or
//! identifier. The test evaluator requests this companion data so it can keep
//! lexical module ownership without changing the ordinary parser result.

use crate::ast::{Module, Node, Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvalImportRefKind {
    Call,
    Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvalImportRef {
    pub(crate) span: Span,
    pub(crate) qualifier: Vec<String>,
    pub(crate) symbol: String,
    pub(crate) kind: EvalImportRefKind,
}

pub(crate) struct EvalParsedModule {
    pub(crate) module: Module,
    #[allow(dead_code)]
    pub(crate) import_refs: Vec<EvalImportRef>,
    /// Every import path the parser recorded, in declaration order, deduped,
    /// as full dotted paths.
    ///
    /// This is `P::import_paths` MOVED out at a successful parse -- not a second
    /// walk and not a second parse. Both `parse_import` and `parse_use` push
    /// here unconditionally, and `parse_stmt` dispatches those keywords wherever
    /// a statement is accepted, so the record is container-independent by
    /// construction: no AST shape, nesting depth, or future `Node` variant can
    /// hide an import from it. It is recorded at parse time, so it also holds
    /// imports that are unused, unreachable, unresolvable, or rejected by a
    /// later compiler stage.
    ///
    /// Full paths, ordered. A last-segment view must never become the authority
    /// -- `a.helper` and `b.helper` are different modules.
    pub(crate) import_paths: Vec<String>,
}

impl<'a> crate::parser::P<'a> {
    pub(crate) fn capture_path_call(&mut self, node: &Node) {
        let Node::Call { callee, span, .. } = node else {
            return;
        };
        let Some((qualifier, symbol)) = callee.rsplit_once("::") else {
            return;
        };
        self.capture_path_ref(*span, qualifier, symbol, EvalImportRefKind::Call);
    }

    pub(crate) fn capture_path_value(&mut self, ident: &str, span: Span) {
        let Some((qualifier, symbol)) = ident.rsplit_once("::") else {
            return;
        };
        self.capture_path_ref(span, qualifier, symbol, EvalImportRefKind::Value);
    }

    fn capture_path_ref(
        &mut self,
        span: Span,
        qualifier: &str,
        symbol: &str,
        kind: EvalImportRefKind,
    ) {
        if self.eval_import_refs.is_none() {
            return;
        }
        let qualifier = if self.imports.iter().any(|item| item == qualifier) {
            vec![qualifier.to_string()]
        } else {
            let dotted = qualifier.replace("::", ".");
            if !self.import_paths.iter().any(|item| item == &dotted) {
                return;
            }
            dotted.split('.').map(str::to_string).collect()
        };
        self.record_eval_import_ref(span, qualifier, symbol, kind);
    }
}

impl EvalParsedModule {
    /// Take both provenance records off the parser at a successful parse.
    ///
    /// `p` is a local in `parse_internal` and dies at its return, so this is the
    /// one point where the parser's own records can be moved out. Both are
    /// MOVED, never cloned and never re-derived by a second walk.
    ///
    /// It is a constructor rather than a struct literal at the call site because
    /// `parser/mod.rs` is over the size ceiling and pinned: this keeps the new
    /// logic in a focused module and leaves that file smaller than before.
    pub(crate) fn take_from(module: Module, p: &mut crate::parser::P<'_>) -> Self {
        Self {
            module,
            import_refs: p.eval_import_refs.take().unwrap_or_default(),
            import_paths: std::mem::take(&mut p.import_paths),
        }
    }
}

/// Parse, and also return the parser's own record of every declared import.
///
/// Crate-private and additive: [`crate::parser::parse`] keeps its exact
/// signature and returns an UNCHANGED `Module`, so no existing caller changes
/// and no AST behaviour moves. "Unchanged" here means equal as an AST — that is
/// what the control compares — and is deliberately NOT a claim of serialized
/// byte identity, which this function neither measures nor promises. The second element is
/// `P::import_paths` moved out of that same parse — there is no second walk and
/// no second parse pass.
///
/// Paths are full and dotted, in declaration order, deduped. `std` paths are
/// included; filtering them is the caller's policy decision, unchanged by this.
///
/// It lives here, beside [`EvalParsedModule`] whose field it reads, rather than
/// in `parser/mod.rs`: that file is already over the size ceiling and pinned, so
/// carrying new logic into a focused module is what the ratchet asks for.
pub(crate) fn parse_with_imports(
    input: &str,
) -> Result<(Module, Vec<String>), Vec<crate::parser::ParseError>> {
    crate::parser::parse_internal(input, false).map(|parsed| (parsed.module, parsed.import_paths))
}

#[cfg(test)]
#[path = "import_list_tests.rs"]
mod import_list_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluator_capture_does_not_change_the_compiler_ast() {
        let source = "use crate.src.dep; fn f() -> i64 { return dep.call(dep.VALUE); }";
        let ordinary = crate::parser::parse(source).expect("ordinary parse");
        let captured = crate::parser::parse_for_eval(source).expect("evaluator parse");
        assert_eq!(captured.module, ordinary);
        assert_eq!(captured.import_refs.len(), 2);
        assert_eq!(captured.import_refs[0].qualifier, ["dep"]);
        assert_eq!(captured.import_refs[0].symbol, "VALUE");
        assert_eq!(captured.import_refs[0].kind, EvalImportRefKind::Value);
        assert_eq!(captured.import_refs[1].qualifier, ["dep"]);
        assert_eq!(captured.import_refs[1].symbol, "call");
        assert_eq!(captured.import_refs[1].kind, EvalImportRefKind::Call);
    }

    /// Regression: the PATH-syntax forms `dep::call(...)` and `dep::VALUE` must
    /// record the same evaluator import refs the dot forms do. Before this a
    /// path-spelled imported call/const carried NO binding and failed the tree
    /// evaluator with a bare "unsupported operation" (call) / "unknown variable"
    /// (value). The AST is
    /// still unchanged by the capture (refs are side data, not AST).
    #[test]
    fn evaluator_capture_records_path_syntax_refs() {
        let source = "use crate.src.dep; fn f() -> i64 { return dep::call(dep::VALUE); }";
        let ordinary = crate::parser::parse(source).expect("ordinary parse");
        let captured = crate::parser::parse_for_eval(source).expect("evaluator parse");
        assert_eq!(captured.module, ordinary);
        assert_eq!(captured.import_refs.len(), 2);
        assert_eq!(captured.import_refs[0].qualifier, ["dep"]);
        assert_eq!(captured.import_refs[0].symbol, "VALUE");
        assert_eq!(captured.import_refs[0].kind, EvalImportRefKind::Value);
        assert_eq!(captured.import_refs[1].qualifier, ["dep"]);
        assert_eq!(captured.import_refs[1].symbol, "call");
        assert_eq!(captured.import_refs[1].kind, EvalImportRefKind::Call);
    }

    /// An enum unit-variant path (`Color::Red`) whose leading segment is a TYPE,
    /// not a declared import, must NOT be captured as an evaluator import ref —
    /// the import-qualifier guard is what keeps the `::` value arm from
    /// shadowing enum-variant resolution.
    #[test]
    fn evaluator_capture_skips_enum_variant_paths() {
        let source = "fn f() -> i64 { let c = Color::Red; return 0; }";
        let captured = crate::parser::parse_for_eval(source).expect("evaluator parse");
        assert!(
            captured.import_refs.is_empty(),
            "enum-variant path must not be an import ref: {:?}",
            captured.import_refs
        );
    }
}
