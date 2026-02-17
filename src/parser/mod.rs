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
        while self.pos < self.b.len() {
            match self.b[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
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
        if self.at_keyword(b"f32") {
            self.pos += 3;
            return Ok("f32".into());
        }
        if self.at_keyword(b"i32") {
            self.pos += 3;
            return Ok("i32".into());
        }
        Err(self.err("expected dtype (f32 or i32)".into()))
    }

    fn type_ann(&mut self) -> Result<TypeAnn, ParseError> {
        self.skip_ws();
        // Tensor[f32,(dims)] — legacy syntax
        if self.at_keyword(b"Tensor") {
            self.pos += 6; // "Tensor"
            self.skip_ws();
            self.expect(b'[')?;
            let dt = self.dtype()?;
            self.skip_ws();
            self.expect(b',')?;
            let dims = self.dim_list_parens()?;
            self.skip_ws();
            self.expect(b']')?;
            return Ok(TypeAnn::Tensor { dtype: dt, dims });
        }
        // diff tensor<f32[dims]>
        if self.at_keyword(b"diff") {
            self.pos += 4;
            self.skip_ws_and_newlines();
            if !self.eat_keyword("tensor") {
                return Err(self.err("expected 'tensor' after 'diff'".into()));
            }
            self.skip_ws();
            self.expect(b'<')?;
            let dt = self.dtype()?;
            let dims = self.dim_list_brackets()?;
            self.skip_ws();
            self.expect(b'>')?;
            return Ok(TypeAnn::DiffTensor { dtype: dt, dims });
        }
        // tensor<f32[dims]>
        if self.at_keyword(b"tensor") {
            self.pos += 6;
            self.skip_ws();
            self.expect(b'<')?;
            let dt = self.dtype()?;
            let dims = self.dim_list_brackets()?;
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
        if self.at_keyword(b"import") {
            return self.parse_import();
        }
        if self.at_keyword(b"fn") {
            return self.parse_fn_def();
        }
        if self.at_keyword(b"return") {
            return self.parse_return();
        }
        if self.at_keyword(b"let") {
            return self.parse_let();
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

    fn parse_expr(&mut self) -> Result<Node, ParseError> {
        self.parse_additive()
    }

    fn parse_additive(&mut self) -> Result<Node, ParseError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            self.skip_ws();
            let op = if self.at(b'+') {
                self.advance();
                BinOp::Add
            } else if self.at(b'-') {
                // Distinguish subtraction from negative number by context:
                // After a complete expression, '-' is always subtraction.
                self.advance();
                BinOp::Sub
            } else {
                break;
            };
            self.skip_ws_and_newlines();
            let right = self.parse_multiplicative()?;
            let span = Span::new(left.span_start(), right.span_end());
            left = Node::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Node, ParseError> {
        let mut left = self.parse_atom()?;
        loop {
            self.skip_ws();
            let op = if self.at(b'*') {
                self.advance();
                BinOp::Mul
            } else if self.at(b'/') {
                self.advance();
                BinOp::Div
            } else {
                break;
            };
            self.skip_ws_and_newlines();
            let right = self.parse_atom()?;
            let span = Span::new(left.span_start(), right.span_end());
            left = Node::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }
        Ok(left)
    }

    fn parse_atom(&mut self) -> Result<Node, ParseError> {
        self.skip_ws_and_newlines();
        if self.at_end() {
            return Err(self.err("unexpected end of input".into()));
        }
        // Parenthesized expression or tuple
        if self.at(b'(') {
            return self.parse_tuple_or_paren();
        }
        // Integer literal
        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            return self.parse_int_lit();
        }
        // Must be identifier-based
        let start = self.pos;
        let ident = self
            .dotted_ident()
            .ok_or_else(|| self.err("expected expression".into()))?;
        self.skip_ws();
        match ident.as_str() {
            "grad" if self.at(b'(') => self.parse_grad_call(start),
            "tensor.sum" if self.at(b'(') => self.parse_tensor_sum(start),
            "tensor.mean" if self.at(b'(') => self.parse_tensor_mean(start),
            "tensor.reshape" if self.at(b'(') => self.parse_tensor_reshape(start),
            "tensor.expand_dims" if self.at(b'(') => self.parse_tensor_expand_dims(start),
            "tensor.squeeze" if self.at(b'(') => self.parse_tensor_squeeze(start),
            "tensor.transpose" if self.at(b'(') => self.parse_tensor_transpose(start),
            "tensor.index" if self.at(b'(') => self.parse_tensor_index(start),
            "tensor.slice_stride" if self.at(b'(') => self.parse_tensor_slice_stride(start),
            "tensor.slice" if self.at(b'(') => self.parse_tensor_slice(start),
            "tensor.gather" if self.at(b'(') => self.parse_tensor_gather(start),
            "tensor.dot" if self.at(b'(') => self.parse_tensor_dot(start),
            "tensor.matmul" if self.at(b'(') => self.parse_tensor_matmul(start),
            "tensor.relu" if self.at(b'(') => self.parse_tensor_relu(start),
            "tensor.conv2d" if self.at(b'(') => self.parse_tensor_conv2d(start),
            _ => {
                if self.at(b'(') {
                    self.parse_generic_call(ident, start)
                } else {
                    let span = Span::new(start, self.pos);
                    Ok(Node::Lit(Literal::Ident(ident), span))
                }
            }
        }
    }

    fn parse_int_lit(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        let d = self
            .digits()
            .ok_or_else(|| self.err("expected integer".into()))?;
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
        Ok(Node::Call { callee, args, span })
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
                } else {
                    if bytes[i] == b'"' {
                        in_string = true;
                    } else if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                        return &line[..i];
                    }
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
