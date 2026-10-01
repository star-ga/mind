// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Import qualifiers and the surface-spelling side table.
//!
//! The parser desugars `q.f(x)` to `f(x)`, `q.CONST` to `CONST`, `module m { … }` to a
//! transparent block and `true` to `1`. The compiler wants exactly that, but the formatter
//! prints this same AST back out, so whatever the desugar discards is recorded here in
//! [`crate::ast::SourceSpelling`], keyed by span. Nothing outside the formatter reads it.

use super::{P, ParseError};
use crate::ast::{Attribute, Literal, Node, Span};

impl P<'_> {
    /// Consume an optional `as NAME` rename and return the name.
    ///
    /// Shared by `parse_import` and `parse_use`, and it must be: the formatter canonicalises
    /// `use` to `import` while re-emitting the alias, so `import X as Y;` is a form the printer
    /// PRODUCES. When only `parse_use` understood `as`, formatting a file twice turned
    /// `import X as Y;` into three statements (`import X;` / `as` / `Y`) and wrote it to disk.
    /// The identifier-loss guard cannot see that, because nothing is deleted.
    pub(super) fn parse_optional_as_alias(&mut self) -> Result<Option<String>, ParseError> {
        self.skip_ws();
        if !self.at_keyword(b"as") {
            return Ok(None);
        }
        self.pos += 2; // "as"
        self.skip_ws();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected a name after `as`".into()))?
            .to_string();
        self.skip_ws();
        Ok(Some(name))
    }

    /// Register the qualifier a later `q.fn(args)` call desugars through: the alias when one is
    /// written, the last path segment otherwise.
    pub(super) fn register_import_qualifier(&mut self, path: &[String], alias: Option<&str>) {
        let qualifier = alias
            .map(str::to_string)
            .unwrap_or_else(|| path.last().cloned().unwrap_or_default());
        if !qualifier.is_empty() && !self.imports.iter().any(|s| s == &qualifier) {
            self.imports.push(qualifier);
        }
        // The unaliased last segment stays registered TOO when an alias renames it. Dropping it
        // broke any file that wrote both spellings: the original name fell through to a
        // MethodCall whose receiver has no struct type and tripped the #306 fail-closed guard.
        if alias.is_some() {
            if let Some(last) = path.last() {
                if !last.is_empty() && !self.imports.iter().any(|s| s == last) {
                    self.imports.push(last.clone());
                }
            }
        }
        // Every import's full dotted path, single-segment ones included:
        // `parse_with_imports` hands this list out as the authoritative record
        // of what the file imports (see `EvalParsedModule::import_paths`), and a
        // call spelled through the whole path (`bridge.mcp.call(x)`) desugars the
        // same way as the qualifier form (`mcp.call(x)`).
        let dotted = path.join(".");
        if !dotted.is_empty() && !self.import_paths.iter().any(|s| s == &dotted) {
            self.import_paths.push(dotted);
        }
    }

    /// Whether `node` is a bare identifier naming an imported module. Such a receiver is a
    /// namespace, never a tensor: without this check `use bytes; … bytes.sum()` was silently
    /// rewritten to `tensor.sum(bytes, axes=[])`.
    pub(super) fn is_import_qualifier(&self, node: &Node) -> bool {
        matches!(node, Node::Lit(Literal::Ident(n), _) if self.imports.iter().any(|s| s == n))
    }

    /// Record the module-qualified spelling of a desugared `q.f(x)` call or `q.CONST` read.
    /// Both desugar to the node an unqualified reference produces, so the span is the only
    /// record that `q` was written.
    pub(super) fn record_qualifier(&mut self, span: Span, qualifier: &str) {
        self.spelling
            .call_qualifiers
            .push((span, qualifier.to_string()));
    }

    /// Record a `module NAME { … }` header and its attributes against the transparent block
    /// the module parses to.
    pub(super) fn record_module_header(&mut self, span: Span, path: String, attrs: Vec<Attribute>) {
        self.spelling.module_headers.push((span, path));
        if !attrs.is_empty() {
            self.spelling.module_attrs.push((span, attrs));
        }
    }
}
