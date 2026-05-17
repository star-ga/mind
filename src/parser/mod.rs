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

// Part of the MIND project (Machine Intelligence Native Design).

//! # Example
//! ```
//! use libmind::{parser, eval};
//! let module = parser::parse("1 + 2 * 3").unwrap();
//! assert_eq!(eval::eval_first_expr(&module).unwrap(), 7);
//! ```

use crate::ast::{BinOp, Literal, Module, Node, Param, Span, TypeAnn};
use crate::diagnostics::{Diagnostic as PrettyDiagnostic, Span as DiagnosticSpan};
use crate::types::ConvPadding;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at offset {}: {}", self.offset, self.message)
    }
}

struct P<'a> {
    b: &'a [u8],
    pos: usize,
}

/// Operator-token kind used by the Pratt expression parser
/// (`P::parse_pratt`). Held separate from the AST `BinOp`/`BitOp`/`LogicalOp`
/// enums so the dispatch table stays a single, pointer-free value.
#[derive(Debug, Clone, Copy)]
enum PrattOp {
    LogicalOr,
    LogicalAnd,
    Cmp(BinOp),
    Arith(BinOp),
    Bit(crate::ast::BitOp),
    /// `expr as type` postfix cast (right operand is a `TypeAnn`,
    /// not an expression).
    AsCast,
}

impl<'a> P<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            b: src.as_bytes(),
            pos: 0,
        }
    }

    #[inline(always)]
    fn at_end(&self) -> bool {
        self.pos >= self.b.len()
    }

    #[inline(always)]
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    #[inline(always)]
    fn at(&self, ch: u8) -> bool {
        self.peek() == Some(ch)
    }

    #[inline(always)]
    fn advance(&mut self) {
        self.pos += 1;
    }

    fn skip_ws(&mut self) {
        while self.pos < self.b.len() {
            match self.b[self.pos] {
                b' ' | b'\t' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn skip_ws_and_newlines(&mut self) {
        loop {
            while self.pos < self.b.len() {
                match self.b[self.pos] {
                    b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                    _ => break,
                }
            }
            // Skip line comments
            if self.pos + 1 < self.b.len()
                && self.b[self.pos] == b'/'
                && self.b[self.pos + 1] == b'/'
            {
                while self.pos < self.b.len() && self.b[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
    }

    fn eat(&mut self, ch: u8) -> bool {
        if self.at(ch) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), ParseError> {
        if self.eat(ch) {
            Ok(())
        } else {
            Err(self.err(format!(
                "expected '{}', found {:?}",
                ch as char,
                self.peek().map(|c| c as char),
            )))
        }
    }

    fn err(&self, message: String) -> ParseError {
        ParseError {
            offset: self.pos,
            message,
        }
    }

    fn starts_with(&self, s: &[u8]) -> bool {
        self.b[self.pos..].starts_with(s)
    }

    fn is_ident_start(ch: u8) -> bool {
        ch.is_ascii_alphabetic() || ch == b'_'
    }

    fn is_ident_cont(ch: u8) -> bool {
        ch.is_ascii_alphanumeric() || ch == b'_'
    }

    /// Read an identifier word (no dots). Returns None if not at an ident.
    fn word(&mut self) -> Option<&'a str> {
        let start = self.pos;
        if self.pos >= self.b.len() || !Self::is_ident_start(self.b[self.pos]) {
            return None;
        }
        while self.pos < self.b.len() && Self::is_ident_cont(self.b[self.pos]) {
            self.pos += 1;
        }
        Some(std::str::from_utf8(&self.b[start..self.pos]).unwrap())
    }

    /// Read a dotted identifier like `tensor.matmul` or `foo.bar.baz`.
    fn dotted_ident(&mut self) -> Option<String> {
        let first = self.word()?;
        let mut name = first.to_string();
        while self.pos < self.b.len() && self.b[self.pos] == b'.' {
            let saved = self.pos;
            self.pos += 1; // skip '.'
            match self.word() {
                Some(part) => {
                    name.push('.');
                    name.push_str(part);
                }
                None => {
                    self.pos = saved;
                    break;
                }
            }
        }
        Some(name)
    }

    /// Read digits as a string.
    fn digits(&mut self) -> Option<String> {
        let start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return None;
        }
        Some(
            std::str::from_utf8(&self.b[start..self.pos])
                .unwrap()
                .to_string(),
        )
    }

    /// Check if keyword `kw` is at current position followed by non-ident char.
    fn at_keyword(&self, kw: &[u8]) -> bool {
        if !self.starts_with(kw) {
            return false;
        }
        let after = self.pos + kw.len();
        after >= self.b.len() || !Self::is_ident_cont(self.b[after])
    }

    /// Consume keyword if present, returning true.
    fn eat_keyword(&mut self, kw: &str) -> bool {
        if self.at_keyword(kw.as_bytes()) {
            self.pos += kw.len();
            true
        } else {
            false
        }
    }

    fn signed_int(&mut self) -> Result<i32, ParseError> {
        self.skip_ws();
        let neg = self.eat(b'-');
        self.skip_ws();
        let d = self
            .digits()
            .ok_or_else(|| self.err("expected integer".into()))?;
        let mut v: i32 = d.parse().map_err(|_| self.err("integer overflow".into()))?;
        if neg {
            v = -v;
        }
        Ok(v)
    }

    fn axes_list(&mut self) -> Result<Vec<i32>, ParseError> {
        self.skip_ws();
        self.expect(b'[')?;
        let mut axes = Vec::new();
        self.skip_ws();
        if !self.at(b']') {
            axes.push(self.signed_int()?);
            loop {
                self.skip_ws();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws();
                if self.at(b']') {
                    break;
                } // trailing comma
                axes.push(self.signed_int()?);
            }
        }
        self.skip_ws();
        self.expect(b']')?;
        Ok(axes)
    }

    fn dim_list_parens(&mut self) -> Result<Vec<String>, ParseError> {
        self.skip_ws();
        self.expect(b'(')?;
        let mut dims = Vec::new();
        self.skip_ws();
        if !self.at(b')') {
            dims.push(self.dim_value()?);
            loop {
                self.skip_ws();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws();
                if self.at(b')') {
                    break;
                }
                dims.push(self.dim_value()?);
            }
        }
        self.skip_ws();
        self.expect(b')')?;
        Ok(dims)
    }

    fn dim_value(&mut self) -> Result<String, ParseError> {
        self.skip_ws();
        if let Some(d) = self.digits() {
            return Ok(d);
        }
        if let Some(w) = self.word() {
            return Ok(w.to_string());
        }
        Err(self.err("expected dimension (integer or symbol)".into()))
    }

    fn dim_list_brackets(&mut self) -> Result<Vec<String>, ParseError> {
        self.skip_ws();
        self.expect(b'[')?;
        let mut dims = Vec::new();
        self.skip_ws();
        if !self.at(b']') {
            dims.push(self.dim_value()?);
            loop {
                self.skip_ws();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws();
                if self.at(b']') {
                    break;
                }
                dims.push(self.dim_value()?);
            }
        }
        self.skip_ws();
        self.expect(b']')?;
        Ok(dims)
    }

    fn dtype(&mut self) -> Result<String, ParseError> {
        self.skip_ws();
        for (kw, name) in [
            (&b"bf16"[..], "bf16"),
            (b"BF16", "bf16"),
            (b"f16", "f16"),
            (b"F16", "f16"),
            (b"f32", "f32"),
            (b"F32", "f32"),
            (b"f64", "f64"),
            (b"F64", "f64"),
            (b"i32", "i32"),
            (b"I32", "i32"),
        ] {
            if self.at_keyword(kw) {
                self.pos += kw.len();
                return Ok(name.into());
            }
        }
        Err(self.err("expected dtype".into()))
    }

    fn type_ann(&mut self) -> Result<TypeAnn, ParseError> {
        self.skip_ws();
        // Tensor[f32,(dims)] — legacy syntax
        if self.at_keyword(b"Tensor") {
            self.pos += 6; // "Tensor"
            self.skip_ws();
            if self.at(b'<') {
                // Tensor<F32, [dims]> syntax (angle bracket)
                self.pos += 1;
                self.skip_ws();
                let dt = self.dtype()?;
                self.skip_ws();
                let dims = if self.eat(b',') {
                    self.skip_ws();
                    let d = self.dim_list_brackets()?;
                    self.skip_ws();
                    d
                } else {
                    Vec::new()
                };
                self.expect(b'>')?;
                return Ok(TypeAnn::Tensor { dtype: dt, dims });
            }
            self.expect(b'[')?;
            let dt = self.dtype()?;
            self.skip_ws();
            self.expect(b',')?;
            let dims = self.dim_list_parens()?;
            self.skip_ws();
            self.expect(b']')?;
            return Ok(TypeAnn::Tensor { dtype: dt, dims });
        }
        // diff tensor<f32[dims]> or diff tensor<f32>
        if self.at_keyword(b"diff") {
            self.pos += 4;
            self.skip_ws_and_newlines();
            if !self.eat_keyword("tensor") {
                return Err(self.err("expected 'tensor' after 'diff'".into()));
            }
            self.skip_ws();
            self.expect(b'<')?;
            let dt = self.dtype()?;
            self.skip_ws();
            let dims = if self.at(b'[') {
                self.dim_list_brackets()?
            } else {
                Vec::new()
            };
            self.skip_ws();
            self.expect(b'>')?;
            return Ok(TypeAnn::DiffTensor { dtype: dt, dims });
        }
        // tensor<f32[dims]> or tensor<f32> (no dims = unspecified shape)
        if self.at_keyword(b"tensor") {
            self.pos += 6;
            self.skip_ws();
            self.expect(b'<')?;
            let dt = self.dtype()?;
            self.skip_ws();
            let dims = if self.at(b'[') {
                self.dim_list_brackets()?
            } else {
                Vec::new()
            };
            self.skip_ws();
            self.expect(b'>')?;
            return Ok(TypeAnn::Tensor { dtype: dt, dims });
        }
        // Scalar types
        if self.at_keyword(b"i32") {
            self.pos += 3;
            return Ok(TypeAnn::ScalarI32);
        }
        if self.at_keyword(b"i64") {
            self.pos += 3;
            return Ok(TypeAnn::ScalarI64);
        }
        if self.at_keyword(b"f32") {
            self.pos += 3;
            return Ok(TypeAnn::ScalarF32);
        }
        if self.at_keyword(b"f64") {
            self.pos += 3;
            return Ok(TypeAnn::ScalarF64);
        }
        if self.at_keyword(b"bool") {
            self.pos += 4;
            return Ok(TypeAnn::ScalarBool);
        }
        if self.at_keyword(b"u32") {
            self.pos += 3;
            return Ok(TypeAnn::ScalarU32);
        }
        // Phase 10.5 Tier-1: user-defined type names (aliases, structs, enums).
        // Falls through to here only if no built-in scalar/tensor matched.
        // Recognized as a bare identifier — or a module-qualified path
        // `module.Name` (Phase 10.6, RFC 0003) — at type position. The
        // dotted form is collected into a single `Named` string with the
        // separator preserved; the type checker resolves the path against
        // the `use` scope.
        let pre_pos = self.pos;
        if let Some(first) = self.word() {
            // Reject keywords reused at type position to avoid odd matches.
            if matches!(
                first,
                "fn" | "let"
                    | "if"
                    | "else"
                    | "for"
                    | "in"
                    | "return"
                    | "module"
                    | "const"
                    | "type"
                    | "struct"
                    | "enum"
                    | "use"
                    | "export"
                    | "import"
                    | "print"
                    | "diff"
            ) {
                self.pos = pre_pos;
                return Err(self.err(format!(
                    "expected type annotation, found keyword `{}`",
                    first
                )));
            }
            // Fast path: bare identifier with no qualifier. Keeps the
            // no-dot case bit-identical to the pre-Phase-10.6 hot loop.
            if self.pos >= self.b.len() || self.b[self.pos] != b'.' {
                return Ok(TypeAnn::Named(first.to_string()));
            }
            // Slow path: module-qualified path `a.b.c`. Accumulate dotted
            // segments without crossing whitespace or newlines — `a . b`
            // is rejected; `a.b.c` becomes a single `Named("a.b.c")`. The
            // type checker resolves the path against the `use` scope.
            let mut name = String::with_capacity(first.len() * 2);
            name.push_str(first);
            while self.pos < self.b.len() && self.b[self.pos] == b'.' {
                let dot_pos = self.pos;
                self.pos += 1;
                match self.word() {
                    Some(seg) => {
                        name.push('.');
                        name.push_str(seg);
                    }
                    None => {
                        self.pos = dot_pos;
                        break;
                    }
                }
            }
            return Ok(TypeAnn::Named(name));
        }
        Err(self.err("expected type annotation".into()))
    }

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let mut items = Vec::new();
        self.skip_ws_and_newlines();
        while !self.at_end() {
            items.push(self.parse_stmt()?);
            // Skip statement separators
            self.skip_ws();
            while self.pos < self.b.len() && (self.b[self.pos] == b';' || self.b[self.pos] == b'\n')
            {
                self.pos += 1;
                self.skip_ws();
            }
        }
        Ok(Module { items })
    }

    fn parse_fn_body_stmts(&mut self) -> Result<Vec<Node>, ParseError> {
        let mut stmts = Vec::new();
        self.skip_ws_and_newlines();
        while !self.at_end() && !self.at(b'}') {
            stmts.push(self.parse_stmt()?);
            self.skip_ws();
            while self.pos < self.b.len() && (self.b[self.pos] == b';' || self.b[self.pos] == b'\n')
            {
                self.pos += 1;
                self.skip_ws();
            }
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Node, ParseError> {
        self.skip_ws_and_newlines();
        // Skip line comments
        while self.pos + 1 < self.b.len()
            && self.b[self.pos] == b'/'
            && self.b[self.pos + 1] == b'/'
        {
            while self.pos < self.b.len() && self.b[self.pos] != b'\n' {
                self.pos += 1;
            }
            self.skip_ws_and_newlines();
        }
        // Phase 10.5 Tier-1: collect [attribute] lines before the dispatched item.
        // Attributes are parsed and (per architect review) recorded but not
        // interpreted by public mindc.
        let attrs = self.parse_attribute_list()?;
        if !attrs.is_empty() {
            return self.parse_attributed_item(attrs);
        }
        // Phase 10.6: optional `pub` visibility marker before struct/enum/fn/
        // type/const. mindc treats it as a no-op — module-level visibility
        // is controlled by the `export` block. Accepting `pub` lets
        // rfn-mind/src/bitlinear.mind and src/ternary.mind parse without
        // forcing a source rewrite of every existing pub-prefixed item.
        if self.at_keyword(b"pub") {
            self.pos += 3;
            self.skip_ws_and_newlines();
        }
        if self.at_keyword(b"import") {
            return self.parse_import();
        }
        if self.at_keyword(b"use") {
            return self.parse_use();
        }
        if self.at_keyword(b"const") {
            return self.parse_const(Vec::new());
        }
        if self.at_keyword(b"type") {
            return self.parse_type_alias(Vec::new());
        }
        if self.at_keyword(b"module") {
            return self.parse_module_block(Vec::new());
        }
        if self.at_keyword(b"export") {
            return self.parse_export_block();
        }
        if self.at_keyword(b"struct") {
            return self.parse_struct(Vec::new());
        }
        if self.at_keyword(b"enum") {
            return self.parse_enum(Vec::new());
        }
        if self.at_keyword(b"fn") {
            return self.parse_fn_def();
        }
        if self.at_keyword(b"assert") {
            return self.parse_assert();
        }
        if self.at_keyword(b"return") {
            return self.parse_return();
        }
        if self.at_keyword(b"for") {
            return self.parse_for();
        }
        if self.at_keyword(b"print") {
            return self.parse_print();
        }
        if self.at_keyword(b"let") {
            return self.parse_let();
        }
        if self.at_keyword(b"if") {
            return self.parse_if_expr();
        }
        // Expression or assignment
        let start = self.pos;
        let expr = self.parse_expr()?;
        self.skip_ws();
        // Check for assignment: bare ident followed by '='
        if let Node::Lit(Literal::Ident(ref name), _) = expr {
            if self.at(b'=') && !self.starts_with(b"==") {
                let name = name.clone();
                self.advance(); // consume '='
                self.skip_ws_and_newlines();
                let value = self.parse_expr()?;
                let span = Span::new(start, self.pos);
                return Ok(Node::Assign {
                    name,
                    value: Box::new(value),
                    span,
                });
            }
        }
        Ok(expr)
    }

    fn parse_import(&mut self) -> Result<Node, ParseError> {
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
        self.skip_ws();
        self.eat(b';'); // optional semicolon
        let span = Span::new(start, self.pos);
        Ok(Node::Import { path, span })
    }

    fn parse_use(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 3; // "use"
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
        self.skip_ws();
        self.eat(b';'); // optional semicolon
        let span = Span::new(start, self.pos);
        Ok(Node::Import { path, span })
    }

    // ── Phase 10.5 Tier-1 / Tier-2 declarations ──────────────────────

    /// Collect zero or more `[attribute]` lines preceding an item.
    /// Each attribute is a single bracketed identifier optionally followed by
    /// a parenthesized argument list of identifiers/strings.
    fn parse_attribute_list(&mut self) -> Result<Vec<crate::ast::Attribute>, ParseError> {
        let mut attrs = Vec::new();
        loop {
            self.skip_ws_and_newlines();
            if !self.at(b'[') {
                break;
            }
            // Lookahead: the bracketed content must be a single ident
            // optionally followed by `(...)` and a closing `]`. If the
            // bracketed expression looks like an array literal (numbers,
            // commas, etc.), back out and let the expression path consume.
            let save = self.pos;
            self.pos += 1; // consume '['
            self.skip_ws();
            let name_pos = self.pos;
            let name = match self.word() {
                Some(n) => n.to_string(),
                None => {
                    self.pos = save;
                    break;
                }
            };
            // Reject if the next non-ident char isn't ']' or '(' — that
            // indicates an array literal like [1, 2, 3] or [foo, bar].
            self.skip_ws();
            let mut args = Vec::new();
            if self.at(b'(') {
                self.pos += 1;
                self.skip_ws();
                while !self.at(b')') && !self.at_end() {
                    if let Some(w) = self.word() {
                        args.push(w.to_string());
                    } else {
                        // Not an attribute — back out
                        self.pos = save;
                        return Ok(attrs);
                    }
                    self.skip_ws();
                    if self.at(b',') {
                        self.pos += 1;
                        self.skip_ws();
                    } else {
                        break;
                    }
                }
                if !self.eat(b')') {
                    self.pos = save;
                    return Ok(attrs);
                }
                self.skip_ws();
            }
            if !self.eat(b']') {
                self.pos = save;
                return Ok(attrs);
            }
            // Successful attribute parse
            let span = Span::new(name_pos, self.pos);
            attrs.push(crate::ast::Attribute { name, args, span });
        }
        Ok(attrs)
    }

    /// Dispatch on the attributed item that follows a `[...]` attribute list.
    fn parse_attributed_item(
        &mut self,
        attrs: Vec<crate::ast::Attribute>,
    ) -> Result<Node, ParseError> {
        self.skip_ws_and_newlines();
        if self.at_keyword(b"module") {
            return self.parse_module_block(attrs);
        }
        if self.at_keyword(b"const") {
            return self.parse_const(attrs);
        }
        if self.at_keyword(b"type") {
            return self.parse_type_alias(attrs);
        }
        if self.at_keyword(b"struct") {
            return self.parse_struct(attrs);
        }
        if self.at_keyword(b"enum") {
            return self.parse_enum(attrs);
        }
        if self.at_keyword(b"fn") {
            // Attributes on `fn` are recorded but not yet stored in FnDef
            // node (FnDef has no attrs field today). Drop silently — public
            // mindc semantics ignore attributes anyway.
            let _ = attrs;
            return self.parse_fn_def();
        }
        Err(self.err(
            "expected `module`, `const`, `type`, `struct`, `enum`, or `fn` after attributes".into(),
        ))
    }

    /// Parse `const NAME[: type] = expr[;]`.
    fn parse_const(&mut self, attrs: Vec<crate::ast::Attribute>) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 5; // "const"
        self.skip_ws();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected const name".into()))?
            .to_string();
        self.skip_ws();
        let ty = if self.eat(b':') {
            self.skip_ws();
            Some(self.type_ann()?)
        } else {
            None
        };
        self.skip_ws();
        if !self.eat(b'=') {
            return Err(self.err("expected `=` in const declaration".into()));
        }
        self.skip_ws_and_newlines();
        let value = self.parse_expr()?;
        self.skip_ws();
        self.eat(b';'); // optional trailing semicolon
        let span = Span::new(start, self.pos);
        Ok(Node::Const {
            name,
            ty,
            value: Box::new(value),
            attrs,
            span,
        })
    }

    /// Parse `type X = Y[;]`.
    fn parse_type_alias(&mut self, attrs: Vec<crate::ast::Attribute>) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 4; // "type"
        self.skip_ws();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected type alias name".into()))?
            .to_string();
        self.skip_ws();
        if !self.eat(b'=') {
            return Err(self.err("expected `=` in type alias".into()));
        }
        self.skip_ws();
        let target = self.type_ann()?;
        self.skip_ws();
        self.eat(b';');
        let span = Span::new(start, self.pos);
        Ok(Node::TypeAlias {
            name,
            target,
            attrs,
            span,
        })
    }

    /// Parse `module NAME { items }` and unwrap the inner items into a
    /// surrounding `Block` so the module walker sees them at flat depth.
    /// Per architect review: no AST module-decl node; pure unwrap.
    fn parse_module_block(
        &mut self,
        _attrs: Vec<crate::ast::Attribute>,
    ) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "module"
        self.skip_ws();
        let _name = self
            .word()
            .ok_or_else(|| self.err("expected module name".into()))?
            .to_string();
        self.skip_ws_and_newlines();
        if !self.eat(b'{') {
            return Err(self.err("expected `{` after module name".into()));
        }
        let mut stmts = Vec::new();
        self.skip_ws_and_newlines();
        while !self.at_end() && !self.at(b'}') {
            stmts.push(self.parse_stmt()?);
            self.skip_ws();
            while self.pos < self.b.len() && (self.b[self.pos] == b';' || self.b[self.pos] == b'\n')
            {
                self.pos += 1;
                self.skip_ws();
            }
        }
        if !self.eat(b'}') {
            return Err(self.err("expected `}` to close module".into()));
        }
        let span = Span::new(start, self.pos);
        // Wrap the items in a Block node; downstream module walker treats
        // a top-level Block as transparent (a do-nothing item list).
        Ok(Node::Block { stmts, span })
    }

    /// Parse export forms:
    ///   - `export { name1, name2 }` (block list)
    ///   - `export name1, name2` (bare list)
    ///   - `export const NAME1, NAME2, ...` (category + names)
    ///   - `export fn f1, f2`, `export type T1`, `export struct S`,
    ///     `export enum E` (category + names)
    fn parse_export_block(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "export"
        self.skip_ws();
        let mut names = Vec::new();
        if self.eat(b'{') {
            self.skip_ws_and_newlines();
            while !self.at(b'}') && !self.at_end() {
                if let Some(n) = self.word() {
                    names.push(n.to_string());
                } else {
                    return Err(self.err("expected export name".into()));
                }
                self.skip_ws();
                if self.eat(b',') {
                    self.skip_ws_and_newlines();
                } else {
                    break;
                }
            }
            self.skip_ws_and_newlines();
            if !self.eat(b'}') {
                return Err(self.err("expected `}` to close export list".into()));
            }
        } else {
            // Optional category keyword: const | type | fn | struct | enum
            // The category is recorded as a synthetic prefix (currently dropped).
            for kw in ["const", "type", "fn", "struct", "enum"] {
                if self.at_keyword(kw.as_bytes()) {
                    self.pos += kw.len();
                    self.skip_ws();
                    break;
                }
            }
            // Comma-separated bare list
            while let Some(n) = self.word() {
                names.push(n.to_string());
                self.skip_ws();
                if self.eat(b',') {
                    self.skip_ws_and_newlines();
                } else {
                    break;
                }
            }
        }
        self.skip_ws();
        self.eat(b';');
        let span = Span::new(start, self.pos);
        Ok(Node::Export { names, span })
    }

    /// Parse `struct NAME { field: T, field: T }`.
    fn parse_struct(&mut self, attrs: Vec<crate::ast::Attribute>) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "struct"
        self.skip_ws();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected struct name".into()))?
            .to_string();
        self.skip_ws_and_newlines();
        if !self.eat(b'{') {
            return Err(self.err("expected `{` after struct name".into()));
        }
        let mut fields = Vec::new();
        self.skip_ws_and_newlines();
        while !self.at(b'}') && !self.at_end() {
            let f_start = self.pos;
            // Phase 10.6: optional `pub` visibility marker on the field.
            // mindc-level visibility is controlled by the surrounding
            // module's `export` block; `pub` on a field is accepted as a
            // source-code annotation and ignored semantically.
            if self.at_keyword(b"pub") {
                self.pos += 3;
                self.skip_ws();
            }
            let f_name = self
                .word()
                .ok_or_else(|| self.err("expected field name".into()))?
                .to_string();
            self.skip_ws();
            if !self.eat(b':') {
                return Err(self.err("expected `:` after field name".into()));
            }
            self.skip_ws();
            let ty = self.type_ann()?;
            let f_span = Span::new(f_start, self.pos);
            fields.push(crate::ast::Field {
                name: f_name,
                ty,
                span: f_span,
            });
            self.skip_ws();
            if self.eat(b',') {
                self.skip_ws_and_newlines();
            } else {
                break;
            }
        }
        self.skip_ws_and_newlines();
        if !self.eat(b'}') {
            return Err(self.err("expected `}` to close struct".into()));
        }
        let span = Span::new(start, self.pos);
        Ok(Node::StructDef {
            name,
            fields,
            attrs,
            span,
        })
    }

    /// Parse `enum NAME { Variant, Variant(T), ... }`.
    fn parse_enum(&mut self, attrs: Vec<crate::ast::Attribute>) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 4; // "enum"
        self.skip_ws();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected enum name".into()))?
            .to_string();
        self.skip_ws_and_newlines();
        if !self.eat(b'{') {
            return Err(self.err("expected `{` after enum name".into()));
        }
        let mut variants = Vec::new();
        self.skip_ws_and_newlines();
        while !self.at(b'}') && !self.at_end() {
            let v_start = self.pos;
            let v_name = self
                .word()
                .ok_or_else(|| self.err("expected variant name".into()))?
                .to_string();
            self.skip_ws();
            let mut payload = Vec::new();
            if self.eat(b'(') {
                self.skip_ws();
                while !self.at(b')') && !self.at_end() {
                    payload.push(self.type_ann()?);
                    self.skip_ws();
                    if self.eat(b',') {
                        self.skip_ws();
                    } else {
                        break;
                    }
                }
                if !self.eat(b')') {
                    return Err(self.err("expected `)` to close variant payload".into()));
                }
                self.skip_ws();
            }
            // Optional `= discriminant` — ignore the value but consume it
            if self.eat(b'=') {
                self.skip_ws();
                let _ = self.parse_expr();
                self.skip_ws();
            }
            let v_span = Span::new(v_start, self.pos);
            variants.push(crate::ast::EnumVariant {
                name: v_name,
                payload,
                span: v_span,
            });
            if self.eat(b',') {
                self.skip_ws_and_newlines();
            } else {
                break;
            }
        }
        self.skip_ws_and_newlines();
        if !self.eat(b'}') {
            return Err(self.err("expected `}` to close enum".into()));
        }
        let span = Span::new(start, self.pos);
        Ok(Node::EnumDef {
            name,
            variants,
            attrs,
            span,
        })
    }

    fn parse_fn_def(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 2; // "fn"
        self.skip_ws_and_newlines();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected function name".into()))?
            .to_string();
        self.skip_ws_and_newlines();
        // Parameter list
        self.expect(b'(')?;
        let mut params = Vec::new();
        self.skip_ws_and_newlines();
        if !self.at(b')') {
            params.push(self.parse_param()?);
            loop {
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
                if self.at(b')') {
                    break;
                }
                params.push(self.parse_param()?);
            }
        }
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        self.skip_ws_and_newlines();
        // Optional return type
        let ret_type = if self.starts_with(b"->") {
            self.pos += 2;
            self.skip_ws_and_newlines();
            Some(self.type_ann()?)
        } else {
            None
        };
        self.skip_ws_and_newlines();
        // Body
        self.expect(b'{')?;
        let body = self.parse_fn_body_stmts()?;
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        let span = Span::new(start, self.pos);
        Ok(Node::FnDef {
            name,
            params,
            ret_type,
            body,
            span,
        })
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        self.skip_ws_and_newlines();
        let start = self.pos;
        let name = self
            .word()
            .ok_or_else(|| self.err("expected parameter name".into()))?
            .to_string();
        self.skip_ws();
        self.expect(b':')?;
        self.skip_ws_and_newlines();
        let ty = self.type_ann()?;
        let span = Span::new(start, self.pos);
        Ok(Param { name, ty, span })
    }

    fn parse_return(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "return"
        self.skip_ws();
        // Check if there's a value expression
        let value = if !self.at_end() && !self.at(b'}') && !self.at(b';') && !self.at(b'\n') {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        let span = Span::new(start, self.pos);
        Ok(Node::Return { value, span })
    }

    /// Parse `assert <expr>[, "message"]`.
    /// Phase 10.5 stretch.
    fn parse_assert(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "assert"
        self.skip_ws();
        let cond = self.parse_expr()?;
        self.skip_ws();
        let msg = if self.eat(b',') {
            self.skip_ws_and_newlines();
            // Expect a string literal
            if self.at(b'"') {
                self.pos += 1;
                let m_start = self.pos;
                while self.pos < self.b.len() && self.b[self.pos] != b'"' {
                    if self.b[self.pos] == b'\\' && self.pos + 1 < self.b.len() {
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
                let s = std::str::from_utf8(&self.b[m_start..self.pos])
                    .unwrap_or("")
                    .to_string();
                if !self.eat(b'"') {
                    return Err(self.err("unterminated assert message string".into()));
                }
                Some(s)
            } else {
                None
            }
        } else {
            None
        };
        self.skip_ws();
        self.eat(b';');
        let span = Span::new(start, self.pos);
        Ok(Node::Assert {
            cond: Box::new(cond),
            msg,
            span,
        })
    }

    fn parse_let(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 3; // "let"
        self.skip_ws_and_newlines();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected variable name".into()))?
            .to_string();
        self.skip_ws_and_newlines();
        // Optional type annotation
        let ann = if self.eat(b':') {
            self.skip_ws_and_newlines();
            Some(self.type_ann()?)
        } else {
            None
        };
        self.skip_ws_and_newlines();
        self.expect(b'=')?;
        self.skip_ws_and_newlines();
        let value = self.parse_expr()?;
        let span = Span::new(start, self.pos);
        Ok(Node::Let {
            name,
            ann,
            value: Box::new(value),
            span,
        })
    }

    fn parse_for(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 3; // "for"
        self.skip_ws_and_newlines();
        let var = self
            .word()
            .ok_or_else(|| self.err("expected loop variable name".into()))?
            .to_string();
        self.skip_ws_and_newlines();
        if !self.eat_keyword("in") {
            return Err(self.err("expected 'in' after loop variable".into()));
        }
        self.skip_ws_and_newlines();
        let start_expr = self.parse_atom()?;
        self.skip_ws();
        // Expect '..'
        if !(self.pos + 1 < self.b.len()
            && self.b[self.pos] == b'.'
            && self.b[self.pos + 1] == b'.')
        {
            return Err(self.err("expected '..' in range".into()));
        }
        self.pos += 2;
        self.skip_ws_and_newlines();
        let end_expr = self.parse_atom()?;
        self.skip_ws_and_newlines();
        self.expect(b'{')?;
        let body = self.parse_fn_body_stmts()?;
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        let span = Span::new(start, self.pos);
        Ok(Node::For {
            var,
            start: Box::new(start_expr),
            end: Box::new(end_expr),
            body,
            span,
        })
    }

    fn parse_print(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 5; // "print"
        self.skip_ws();
        self.expect(b'(')?;
        let mut args = Vec::new();
        self.skip_ws_and_newlines();
        if !self.at(b')') {
            args.push(self.parse_expr()?);
            loop {
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
                if self.at(b')') {
                    break;
                }
                args.push(self.parse_expr()?);
            }
        }
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::Print { args, span })
    }

    fn parse_if_expr(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 2; // "if"
        self.skip_ws_and_newlines();
        let cond = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b'{')?;
        let then_branch = self.parse_fn_body_stmts()?;
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        self.skip_ws_and_newlines();
        let else_branch = if self.eat_keyword("else") {
            self.skip_ws_and_newlines();
            if self.at_keyword(b"if") {
                let nested = self.parse_if_expr()?;
                Some(vec![nested])
            } else {
                self.expect(b'{')?;
                let stmts = self.parse_fn_body_stmts()?;
                self.skip_ws_and_newlines();
                self.expect(b'}')?;
                Some(stmts)
            }
        } else {
            None
        };
        let span = Span::new(start, self.pos);
        Ok(Node::If {
            cond: Box::new(cond),
            then_branch,
            else_branch,
            span,
        })
    }

    /// Pratt operator-precedence parser for expressions (mindc 0.2.5).
    ///
    /// Replaces the recursive-descent chain
    /// (`parse_logical_or` → `parse_logical_and` → `parse_comparison`
    /// → `parse_additive` → `parse_bitwise` → `parse_multiplicative`)
    /// with a single dispatch loop driven by a binding-power table.
    ///
    /// Precedence (higher binds tighter; left-associative throughout):
    ///
    /// | level | operators                | (lbp, rbp) |
    /// |-------|--------------------------|------------|
    /// | 1     | `\|\|`                   | (1, 2)     |
    /// | 2     | `&&`                     | (3, 4)     |
    /// | 3     | `==` `!=` `<` `<=` `>` `>=` | (5, 6) |
    /// | 4     | `as` (postfix cast)      | (7, _)     |
    /// | 5     | `+` `-`                  | (9, 10)    |
    /// | 6     | `\|` `&` `^` `<<` `>>`   | (11, 12)   |
    /// | 7     | `*` `/`                  | (13, 14)   |
    /// | leaf  | `parse_atom`             | —          |
    ///
    /// The chain is preserved exactly: ordering matches the prior recursive
    /// descent, so AST output is byte-for-byte identical for all valid
    /// programs.
    #[inline]
    fn parse_expr(&mut self) -> Result<Node, ParseError> {
        self.parse_pratt(0)
    }

    fn parse_pratt(&mut self, min_bp: u8) -> Result<Node, ParseError> {
        let mut left = self.parse_atom()?;
        loop {
            self.skip_ws();
            let (op, lbp, rbp, advance) = match self.peek_binop() {
                Some(o) => o,
                None => break,
            };
            if lbp < min_bp {
                break;
            }
            self.pos += advance;
            // Postfix `as` cast — RHS is a TypeAnn, not an expression.
            if matches!(op, PrattOp::AsCast) {
                self.skip_ws();
                let ty = self.type_ann()?;
                let span = Span::new(left.span_start(), self.pos);
                left = Node::As {
                    expr: Box::new(left),
                    ty,
                    span,
                };
                continue;
            }
            self.skip_ws_and_newlines();
            let right = self.parse_pratt(rbp)?;
            let span = Span::new(left.span_start(), right.span_end());
            left = match op {
                PrattOp::LogicalOr => Node::Logical {
                    op: crate::ast::LogicalOp::Or,
                    left: Box::new(left),
                    right: Box::new(right),
                    span,
                },
                PrattOp::LogicalAnd => Node::Logical {
                    op: crate::ast::LogicalOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                    span,
                },
                PrattOp::Cmp(b) => Node::Binary {
                    op: b,
                    left: Box::new(left),
                    right: Box::new(right),
                    span,
                },
                PrattOp::Arith(b) => Node::Binary {
                    op: b,
                    left: Box::new(left),
                    right: Box::new(right),
                    span,
                },
                PrattOp::Bit(b) => Node::Bitwise {
                    op: b,
                    left: Box::new(left),
                    right: Box::new(right),
                    span,
                },
                PrattOp::AsCast => unreachable!(),
            };
        }
        Ok(left)
    }

    /// Peek the next binary operator at `self.pos` without advancing.
    /// Caller MUST have already invoked `skip_ws()`. Returns
    /// `(op, left_bp, right_bp, byte_advance)` or `None` if the next token
    /// is not a recognised binary operator.
    #[inline]
    fn peek_binop(&self) -> Option<(PrattOp, u8, u8, usize)> {
        let p = self.pos;
        if p >= self.b.len() {
            return None;
        }
        let b0 = self.b[p];
        let b1 = self.b.get(p + 1).copied().unwrap_or(0);

        // Two-char operators take priority over their one-char prefixes.
        match (b0, b1) {
            (b'|', b'|') => return Some((PrattOp::LogicalOr, 1, 2, 2)),
            (b'&', b'&') => return Some((PrattOp::LogicalAnd, 3, 4, 2)),
            (b'=', b'=') => return Some((PrattOp::Cmp(BinOp::Eq), 5, 6, 2)),
            (b'!', b'=') => return Some((PrattOp::Cmp(BinOp::Ne), 5, 6, 2)),
            (b'<', b'=') => return Some((PrattOp::Cmp(BinOp::Le), 5, 6, 2)),
            (b'>', b'=') => return Some((PrattOp::Cmp(BinOp::Ge), 5, 6, 2)),
            (b'<', b'<') => return Some((PrattOp::Bit(crate::ast::BitOp::Shl), 11, 12, 2)),
            (b'>', b'>') => return Some((PrattOp::Bit(crate::ast::BitOp::Shr), 11, 12, 2)),
            _ => {}
        }

        // `as` keyword (postfix cast). Requires a word-boundary on the right
        // so identifiers like `assert` and `ascii` are not misparsed.
        if b0 == b'a' && b1 == b's' {
            let nb = self.b.get(p + 2).copied().unwrap_or(0);
            if !Self::is_ident_cont(nb) {
                return Some((PrattOp::AsCast, 7, 0, 2));
            }
        }

        // One-char operators.
        match b0 {
            b'<' => Some((PrattOp::Cmp(BinOp::Lt), 5, 6, 1)),
            b'>' => Some((PrattOp::Cmp(BinOp::Gt), 5, 6, 1)),
            b'+' => Some((PrattOp::Arith(BinOp::Add), 9, 10, 1)),
            b'-' => Some((PrattOp::Arith(BinOp::Sub), 9, 10, 1)),
            b'|' => Some((PrattOp::Bit(crate::ast::BitOp::Or), 11, 12, 1)),
            b'&' => Some((PrattOp::Bit(crate::ast::BitOp::And), 11, 12, 1)),
            b'^' => Some((PrattOp::Bit(crate::ast::BitOp::Xor), 11, 12, 1)),
            b'*' => Some((PrattOp::Arith(BinOp::Mul), 13, 14, 1)),
            b'/' => Some((PrattOp::Arith(BinOp::Div), 13, 14, 1)),
            _ => None,
        }
    }

    fn parse_string_lit(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.expect(b'"')?;
        let str_start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos] != b'"' {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.b[str_start..self.pos])
            .unwrap()
            .to_string();
        self.expect(b'"')?;
        let span = Span::new(start, self.pos);
        Ok(Node::Lit(Literal::Str(s), span))
    }

    fn parse_array_lit(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.expect(b'[')?;
        let mut elements = Vec::new();
        self.skip_ws_and_newlines();
        if !self.at(b']') {
            elements.push(self.parse_expr()?);
            loop {
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
                if self.at(b']') {
                    break;
                }
                elements.push(self.parse_expr()?);
            }
        }
        self.skip_ws_and_newlines();
        self.expect(b']')?;
        let span = Span::new(start, self.pos);
        Ok(Node::ArrayLit { elements, span })
    }

    fn parse_atom(&mut self) -> Result<Node, ParseError> {
        let mut node = self.parse_primary()?;
        loop {
            self.skip_ws();
            if self.at(b'.') {
                let dot_pos = self.pos;
                if dot_pos + 1 < self.b.len() && Self::is_ident_start(self.b[dot_pos + 1]) {
                    self.pos = dot_pos + 1;
                    let method = self.word().unwrap().to_string();
                    self.skip_ws();
                    if self.at(b'(') {
                        self.expect(b'(')?;
                        let mut args = Vec::new();
                        self.skip_ws_and_newlines();
                        if !self.at(b')') {
                            args.push(self.parse_call_arg()?);
                            loop {
                                self.skip_ws_and_newlines();
                                if !self.eat(b',') {
                                    break;
                                }
                                self.skip_ws_and_newlines();
                                if self.at(b')') {
                                    break;
                                }
                                args.push(self.parse_call_arg()?);
                            }
                        }
                        self.skip_ws_and_newlines();
                        self.expect(b')')?;
                        let span = Span::new(node.span_start(), self.pos);
                        node = Node::MethodCall {
                            receiver: Box::new(node),
                            method,
                            args,
                            span,
                        };
                        continue;
                    } else {
                        let span = Span::new(node.span_start(), self.pos);
                        node = Node::FieldAccess {
                            receiver: Box::new(node),
                            field: method,
                            span,
                        };
                        continue;
                    }
                }
            }
            break;
        }
        Ok(node)
    }

    fn parse_primary(&mut self) -> Result<Node, ParseError> {
        self.skip_ws_and_newlines();
        if self.at_end() {
            return Err(self.err("unexpected end of input".into()));
        }
        if self.at(b'(') {
            return self.parse_tuple_or_paren();
        }
        if self.at(b'[') {
            return self.parse_array_lit();
        }
        if self.at(b'"') {
            return self.parse_string_lit();
        }
        if self.at(b'-') {
            let start = self.pos;
            self.advance();
            self.skip_ws();
            let operand = self.parse_atom()?;
            let span = Span::new(start, self.pos);
            return Ok(Node::Neg {
                operand: Box::new(operand),
                span,
            });
        }
        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            return self.parse_number_lit();
        }
        // If expression in expression position
        if self.at_keyword(b"if") {
            return self.parse_if_expr();
        }
        let start = self.pos;
        let ident = self
            .dotted_ident()
            .ok_or_else(|| self.err("expected expression".into()))?;
        self.skip_ws();
        match ident.as_str() {
            "grad" if self.at(b'(') => self.parse_grad_call(start),
            "tensor.sum" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_sum(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.sum".into(), start)
                    }
                }
            }
            "tensor.mean" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_mean(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.mean".into(), start)
                    }
                }
            }
            "tensor.reshape" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_reshape(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.reshape".into(), start)
                    }
                }
            }
            "tensor.expand_dims" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_expand_dims(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.expand_dims".into(), start)
                    }
                }
            }
            "tensor.squeeze" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_squeeze(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.squeeze".into(), start)
                    }
                }
            }
            "tensor.transpose" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_transpose(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.transpose".into(), start)
                    }
                }
            }
            "tensor.index" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_index(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.index".into(), start)
                    }
                }
            }
            "tensor.slice_stride" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_slice_stride(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.slice_stride".into(), start)
                    }
                }
            }
            "tensor.slice" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_slice(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.slice".into(), start)
                    }
                }
            }
            "tensor.gather" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_gather(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.gather".into(), start)
                    }
                }
            }
            "tensor.dot" if self.at(b'(') => self.parse_tensor_dot(start),
            "tensor.matmul" if self.at(b'(') => self.parse_tensor_matmul(start),
            "tensor.relu" if self.at(b'(') => self.parse_tensor_relu(start),
            "tensor.rand" if self.at(b'(') => self.parse_tensor_rand(start),
            "tensor.conv2d" if self.at(b'(') => {
                let saved = self.pos;
                match self.parse_tensor_conv2d(start) {
                    Ok(n) => Ok(n),
                    Err(_) => {
                        self.pos = saved;
                        self.parse_generic_call("tensor.conv2d".into(), start)
                    }
                }
            }
            _ => {
                // If the ident contains a dot and the first segment is not a known
                // namespace like "tensor", backtrack to the first segment so
                // parse_atom's dot-loop can handle method calls.
                if let Some(dot_idx) = ident.find('.') {
                    let first = &ident[..dot_idx];
                    if first != "tensor" {
                        self.pos = start + first.len();
                        self.skip_ws();
                        if self.at(b'(') {
                            return self.parse_generic_call(first.to_string(), start);
                        } else {
                            let span = Span::new(start, start + first.len());
                            return Ok(Node::Lit(Literal::Ident(first.to_string()), span));
                        }
                    }
                }
                if self.at(b'(') {
                    self.parse_generic_call(ident, start)
                } else {
                    let span = Span::new(start, self.pos);
                    Ok(Node::Lit(Literal::Ident(ident), span))
                }
            }
        }
    }

    fn parse_number_lit(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        let d = self
            .digits()
            .ok_or_else(|| self.err("expected number".into()))?;
        // Check for decimal point → float literal
        if self.pos < self.b.len() && self.b[self.pos] == b'.' {
            // Disambiguate: `1.0` is float, `1..10` is range
            if self.pos + 1 < self.b.len() && self.b[self.pos + 1] == b'.' {
                // Range syntax `N..M` — return integer
                let val: i64 = d.parse().map_err(|_| self.err("integer overflow".into()))?;
                let span = Span::new(start, self.pos);
                return Ok(Node::Lit(Literal::Int(val), span));
            }
            self.pos += 1; // skip '.'
            let frac = self.digits().unwrap_or_default();
            let mut num_str = d;
            num_str.push('.');
            num_str.push_str(&frac);
            // Optional exponent: 1.0e-5
            if self.pos < self.b.len() && (self.b[self.pos] == b'e' || self.b[self.pos] == b'E') {
                num_str.push('e');
                self.pos += 1;
                if self.pos < self.b.len() && (self.b[self.pos] == b'-' || self.b[self.pos] == b'+')
                {
                    num_str.push(self.b[self.pos] as char);
                    self.pos += 1;
                }
                let exp = self
                    .digits()
                    .ok_or_else(|| self.err("expected exponent digits".into()))?;
                num_str.push_str(&exp);
            }
            let val: f64 = num_str
                .parse()
                .map_err(|_| self.err("invalid float".into()))?;
            let span = Span::new(start, self.pos);
            return Ok(Node::Lit(Literal::Float(val), span));
        }
        // Optional exponent without decimal: 1e5
        if self.pos < self.b.len() && (self.b[self.pos] == b'e' || self.b[self.pos] == b'E') {
            let mut num_str = d;
            num_str.push('e');
            self.pos += 1;
            if self.pos < self.b.len() && (self.b[self.pos] == b'-' || self.b[self.pos] == b'+') {
                num_str.push(self.b[self.pos] as char);
                self.pos += 1;
            }
            let exp = self
                .digits()
                .ok_or_else(|| self.err("expected exponent digits".into()))?;
            num_str.push_str(&exp);
            let val: f64 = num_str
                .parse()
                .map_err(|_| self.err("invalid float".into()))?;
            let span = Span::new(start, self.pos);
            return Ok(Node::Lit(Literal::Float(val), span));
        }
        let val: i64 = d.parse().map_err(|_| self.err("integer overflow".into()))?;
        let span = Span::new(start, self.pos);
        Ok(Node::Lit(Literal::Int(val), span))
    }

    fn parse_tuple_or_paren(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.expect(b'(')?;
        let mut items = Vec::new();
        self.skip_ws_and_newlines();
        if !self.at(b')') {
            items.push(self.parse_expr()?);
            loop {
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
                if self.at(b')') {
                    break;
                }
                items.push(self.parse_expr()?);
            }
        }
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        if items.len() == 1 {
            Ok(Node::Paren(
                Box::new(items.into_iter().next().unwrap()),
                span,
            ))
        } else {
            Ok(Node::Tuple {
                elements: items,
                span,
            })
        }
    }

    fn parse_generic_call(&mut self, callee: String, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        let mut args = Vec::new();
        self.skip_ws_and_newlines();
        if !self.at(b')') {
            args.push(self.parse_call_arg()?);
            loop {
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
                if self.at(b')') {
                    break;
                }
                args.push(self.parse_call_arg()?);
            }
        }
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::Call { callee, args, span })
    }

    /// Parse a call argument, handling `name=expr` keyword syntax by skipping the name.
    fn parse_call_arg(&mut self) -> Result<Node, ParseError> {
        // Try to detect keyword arg: ident followed by '=' (but not '==')
        let saved = self.pos;
        if let Some(name) = self.try_ident() {
            self.skip_ws();
            if self.at(b'=') && !(self.pos + 1 < self.b.len() && self.b[self.pos + 1] == b'=') {
                // keyword arg — skip name=, parse the value
                self.pos += 1; // skip '='
                self.skip_ws();
                let _ = name; // drop the keyword name
                return self.parse_expr();
            }
            // Not a keyword arg — restore position
            self.pos = saved;
        }
        self.parse_expr()
    }

    fn try_ident(&mut self) -> Option<String> {
        let start = self.pos;
        if self.pos >= self.b.len()
            || !(self.b[self.pos].is_ascii_alphabetic() || self.b[self.pos] == b'_')
        {
            return None;
        }
        while self.pos < self.b.len()
            && (self.b[self.pos].is_ascii_alphanumeric() || self.b[self.pos] == b'_')
        {
            self.pos += 1;
        }
        Some(String::from_utf8_lossy(&self.b[start..self.pos]).to_string())
    }

    fn parse_grad_call(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let loss = self.parse_expr()?;
        self.skip_ws_and_newlines();
        // Optional wrt=[...]
        let wrt = if self.eat(b',') {
            self.skip_ws_and_newlines();
            if self.eat_keyword("wrt") {
                self.skip_ws();
                self.expect(b'=')?;
                self.skip_ws();
                self.expect(b'[')?;
                let mut vars = Vec::new();
                self.skip_ws();
                if !self.at(b']') {
                    let w = self
                        .word()
                        .ok_or_else(|| self.err("expected variable name".into()))?;
                    vars.push(w.to_string());
                    loop {
                        self.skip_ws();
                        if !self.eat(b',') {
                            break;
                        }
                        self.skip_ws();
                        if self.at(b']') {
                            break;
                        }
                        let w = self
                            .word()
                            .ok_or_else(|| self.err("expected variable name".into()))?;
                        vars.push(w.to_string());
                    }
                }
                self.skip_ws();
                self.expect(b']')?;
                vars
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallGrad {
            loss: Box::new(loss),
            wrt,
            span,
        })
    }

    fn parse_reduce_args(&mut self) -> Result<(Vec<i32>, bool), ParseError> {
        let mut axes = Vec::new();
        let mut keepdims = false;
        while self.eat(b',') {
            self.skip_ws_and_newlines();
            if self.at_keyword(b"axes") {
                self.pos += 4;
                self.skip_ws();
                self.expect(b'=')?;
                self.skip_ws();
                axes = self.axes_list()?;
            } else if self.at_keyword(b"keepdims") {
                self.pos += 8;
                self.skip_ws();
                self.expect(b'=')?;
                self.skip_ws();
                if self.eat_keyword("true") {
                    keepdims = true;
                } else if self.eat_keyword("false") {
                    keepdims = false;
                } else {
                    return Err(self.err("expected 'true' or 'false'".into()));
                }
            } else {
                break;
            }
            self.skip_ws_and_newlines();
        }
        Ok((axes, keepdims))
    }

    fn parse_tensor_sum(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        let (axes, keepdims) = self.parse_reduce_args()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallTensorSum {
            x: Box::new(x),
            axes,
            keepdims,
            span,
        })
    }

    fn parse_tensor_mean(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        let (axes, keepdims) = self.parse_reduce_args()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallTensorMean {
            x: Box::new(x),
            axes,
            keepdims,
            span,
        })
    }

    fn parse_tensor_reshape(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        let dims = self.dim_list_parens()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallReshape {
            x: Box::new(x),
            dims,
            span,
        })
    }

    fn parse_tensor_expand_dims(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("axis") {
            return Err(self.err("expected 'axis='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let axis = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallExpandDims {
            x: Box::new(x),
            axis,
            span,
        })
    }

    fn parse_tensor_squeeze(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        let axes = if self.eat(b',') {
            self.skip_ws_and_newlines();
            if self.eat_keyword("axes") {
                self.skip_ws();
                self.expect(b'=')?;
                self.skip_ws();
                self.axes_list()?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallSqueeze {
            x: Box::new(x),
            axes,
            span,
        })
    }

    fn parse_tensor_transpose(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        let axes = if self.eat(b',') {
            self.skip_ws_and_newlines();
            if self.eat_keyword("axes") {
                self.skip_ws();
                self.expect(b'=')?;
                self.skip_ws();
                Some(self.axes_list()?)
            } else {
                None
            }
        } else {
            None
        };
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallTranspose {
            x: Box::new(x),
            axes,
            span,
        })
    }

    fn parse_tensor_index(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("axis") {
            return Err(self.err("expected 'axis='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let axis = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("i") {
            return Err(self.err("expected 'i='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let i = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallIndex {
            x: Box::new(x),
            axis,
            i,
            span,
        })
    }

    fn parse_tensor_slice(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("axis") {
            return Err(self.err("expected 'axis='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let axis = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("start") {
            return Err(self.err("expected 'start='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let s = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("end") {
            return Err(self.err("expected 'end='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let e = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallSlice {
            x: Box::new(x),
            axis,
            start: s,
            end: e,
            span,
        })
    }

    fn parse_tensor_slice_stride(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("axis") {
            return Err(self.err("expected 'axis='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let axis = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("start") {
            return Err(self.err("expected 'start='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let s = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("end") {
            return Err(self.err("expected 'end='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let e = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("step") {
            return Err(self.err("expected 'step='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let step = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallSliceStride {
            x: Box::new(x),
            axis,
            start: s,
            end: e,
            step,
            span,
        })
    }

    fn parse_tensor_gather(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        if !self.eat_keyword("axis") {
            return Err(self.err("expected 'axis='".into()));
        }
        self.skip_ws();
        self.expect(b'=')?;
        let axis = self.signed_int()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        // Optional "idx=" prefix — only consume if '=' follows
        if self.at_keyword(b"idx") {
            let saved = self.pos;
            self.pos += 3;
            self.skip_ws();
            if self.eat(b'=') {
                self.skip_ws_and_newlines();
            } else {
                self.pos = saved; // not a keyword arg, restore
            }
        }
        let idx = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallGather {
            x: Box::new(x),
            axis,
            idx: Box::new(idx),
            span,
        })
    }

    fn parse_tensor_dot(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let a = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        let b = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallDot {
            a: Box::new(a),
            b: Box::new(b),
            span,
        })
    }

    fn parse_tensor_rand(&mut self, start: usize) -> Result<Node, ParseError> {
        // tensor.rand(d0, d1, ...) → random-filled f32 tensor
        self.expect(b'(')?;
        let mut dims = Vec::new();
        self.skip_ws_and_newlines();
        let d = self
            .digits()
            .ok_or_else(|| self.err("expected dimension".into()))?;
        dims.push(
            d.parse::<usize>()
                .map_err(|_| self.err("invalid dimension".into()))?,
        );
        loop {
            self.skip_ws_and_newlines();
            if !self.at(b',') {
                break;
            }
            self.advance(); // consume the comma
            self.skip_ws_and_newlines();
            let d = self
                .digits()
                .ok_or_else(|| self.err("expected dimension".into()))?;
            dims.push(
                d.parse::<usize>()
                    .map_err(|_| self.err("invalid dimension".into()))?,
            );
        }
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallTensorRand { shape: dims, span })
    }

    fn parse_tensor_matmul(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let a = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        let b = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallMatMul {
            a: Box::new(a),
            b: Box::new(b),
            span,
        })
    }

    fn parse_tensor_relu(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallTensorRelu {
            x: Box::new(x),
            span,
        })
    }

    fn parse_tensor_conv2d(&mut self, start: usize) -> Result<Node, ParseError> {
        self.expect(b'(')?;
        self.skip_ws_and_newlines();
        let x = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b',')?;
        self.skip_ws_and_newlines();
        let w = self.parse_expr()?;
        self.skip_ws_and_newlines();
        // Optional keyword args
        let mut stride_h = 1usize;
        let mut stride_w = 1usize;
        let mut padding = ConvPadding::Valid;
        while self.eat(b',') {
            self.skip_ws_and_newlines();
            if self.eat_keyword("stride_h") {
                self.skip_ws();
                self.expect(b'=')?;
                self.skip_ws();
                let v = self.signed_int()?;
                if v <= 0 {
                    return Err(self.err("stride must be positive".into()));
                }
                stride_h = v as usize;
            } else if self.eat_keyword("stride_w") {
                self.skip_ws();
                self.expect(b'=')?;
                self.skip_ws();
                let v = self.signed_int()?;
                if v <= 0 {
                    return Err(self.err("stride must be positive".into()));
                }
                stride_w = v as usize;
            } else if self.eat_keyword("padding") {
                self.skip_ws();
                self.expect(b'=')?;
                self.skip_ws();
                self.expect(b'"')?;
                let pstart = self.pos;
                while self.pos < self.b.len() && self.b[self.pos] != b'"' {
                    self.pos += 1;
                }
                let pval = std::str::from_utf8(&self.b[pstart..self.pos]).unwrap();
                padding = ConvPadding::parse(pval)
                    .ok_or_else(|| self.err("padding must be \"valid\" or \"same\"".into()))?;
                self.expect(b'"')?;
            } else {
                break;
            }
            self.skip_ws_and_newlines();
        }
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        let span = Span::new(start, self.pos);
        Ok(Node::CallTensorConv2d {
            x: Box::new(x),
            w: Box::new(w),
            stride_h,
            stride_w,
            padding,
            span,
        })
    }
}

/// Strip single-line comments (`// ...`) from source code.
/// Preserves line structure for accurate error reporting.
/// Correctly handles `//` inside string literals.
fn strip_comments(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            let bytes = line.as_bytes();
            let mut in_string = false;
            let mut i = 0;
            while i < bytes.len() {
                if in_string {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 2; // skip escaped character
                        continue;
                    }
                    if bytes[i] == b'"' {
                        in_string = false;
                    }
                } else if bytes[i] == b'"' {
                    in_string = true;
                } else if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    return &line[..i];
                }
                i += 1;
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn parse(input: &str) -> Result<Module, Vec<ParseError>> {
    let stripped = strip_comments(input);
    let mut p = P::new(&stripped);
    match p.parse_module() {
        Ok(m) => Ok(m),
        Err(e) => Err(vec![e]),
    }
}

/// Parse with pretty diagnostics instead of raw parse errors.
pub fn parse_with_diagnostics(input: &str) -> Result<Module, Vec<PrettyDiagnostic>> {
    parse_with_diagnostics_in_file(input, None)
}

pub fn parse_with_diagnostics_in_file(
    input: &str,
    file: Option<&str>,
) -> Result<Module, Vec<PrettyDiagnostic>> {
    let stripped = strip_comments(input);
    let mut p = P::new(&stripped);
    match p.parse_module() {
        Ok(m) => Ok(m),
        Err(e) => {
            let diag = PrettyDiagnostic {
                phase: "parse",
                code: "E1001",
                severity: crate::diagnostics::Severity::Error,
                message: e.message,
                span: Some(DiagnosticSpan::from_offsets(
                    &stripped,
                    e.offset,
                    e.offset + 1,
                    file,
                )),
                notes: Vec::new(),
                help: None,
            };
            Err(vec![diag])
        }
    }
}
