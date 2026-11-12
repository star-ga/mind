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

//! Import declarations, their qualifiers, and the surface-spelling side table.
//!
//! The parser desugars `q.f(x)` to `f(x)`, `q.CONST` to `CONST`, `module m { … }` to a
//! transparent block and `true` to `1`. The compiler wants exactly that, but the formatter
//! prints this same AST back out, so whatever the desugar discards is recorded here in
//! [`crate::ast::SourceSpelling`], keyed by span. Qualified POSIX call resolution also reads the retained qualifier.

use super::{P, ParseError};
use crate::ast::{Attribute, Literal, Node, Span};

impl P<'_> {
    /// `import a.b [as c];`, after an optional `pub` (`public`) the caller consumed.
    pub(super) fn parse_import(&mut self, public: bool) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "import"
        self.skip_ws();
        let mut path = Vec::new();
        let first = self
            .word()
            .ok_or_else(|| self.err("expected module name".into()))?;
        path.push(first.to_string());
        while self.eat(b'.') {
            let part = self
                .word()
                .ok_or_else(|| self.err("expected module name after '.'".into()))?;
            path.push(part.to_string());
        }
        self.finish_import(start, path, public)
    }

    /// `use a.b [as c];` or `use a::b`, after an optional `pub` (`public`).
    pub(super) fn parse_use(&mut self, public: bool) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 3; // "use"
        self.skip_ws();
        let mut path = Vec::new();
        let first = self
            .word()
            .ok_or_else(|| self.err("expected module name".into()))?;
        path.push(first.to_string());
        // #263 surface 1: accept BOTH the shipped dotted form (`use a.b.c`) and
        // the path form (`use std::fixed_point::Q16_16`). A `::` separator is
        // the Rust-style module path; the final segment is bound as the imported
        // name, exactly as the dotted form does. Segments may not mix within one
        // `use`, but this loop handles either separator per step, which subsumes
        // both spellings without a separate parse path.
        loop {
            if self.eat(b'.') {
                let part = self
                    .word()
                    .ok_or_else(|| self.err("expected module name after '.'".into()))?;
                path.push(part.to_string());
            } else if self.pos + 1 < self.b.len()
                && self.b[self.pos] == b':'
                && self.b[self.pos + 1] == b':'
            {
                self.pos += 2; // consume `::`
                let part = self
                    .word()
                    .ok_or_else(|| self.err("expected path segment after '::'".into()))?;
                path.push(part.to_string());
            } else {
                break;
            }
        }
        self.finish_import(start, path, public)
    }

    /// The tail `import` and `use` share: the `as` rename, the optional `;`, the qualifier.
    fn finish_import(
        &mut self,
        start: usize,
        path: Vec<String>,
        public: bool,
    ) -> Result<Node, ParseError> {
        let alias = self.parse_optional_as_alias()?;
        self.eat(b';'); // optional semicolon
        self.register_import_qualifier(&path, alias.as_deref());
        let span = Span::new(start, self.pos);
        Ok(Node::Import {
            path,
            alias,
            public,
            span,
        })
    }

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
