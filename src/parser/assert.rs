// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `assert cond[, "msg"]` and its parenthesised spelling `assert(cond, "msg")`.

use super::{P, ParseError};
use crate::ast::{Literal, Node, Span};

impl P<'_> {
    pub(super) fn parse_assert(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "assert"
        self.skip_ws();
        let cond = self.parse_expr()?;
        let (cond, paren_msg) = self.split_paren_assert(cond)?;
        let paren_assert = paren_msg.is_some();
        self.skip_ws();
        let msg = if paren_msg.is_some() {
            paren_msg
        } else if self.eat(b',') {
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
        if paren_assert {
            self.spelling.paren_asserts.push(span);
        }
        Ok(Node::Assert {
            cond: Box::new(cond),
            msg,
            span,
        })
    }

    /// Split the parenthesised `assert(cond, "msg")` form into the condition and the
    /// message `assert cond, "msg"` spells, returning `None` for the message when `cond` is
    /// not such a pair.
    ///
    /// The parentheses parse as one tuple, and a tuple condition is always truthy, so the
    /// assert could never fire: a native build passed every one silently. A message is kept
    /// as written between the quotes, like the bare form keeps it. Any other tuple condition,
    /// a non-string second operand (`assert(cond, 9)`) included, is refused for the same
    /// reason.
    pub(super) fn split_paren_assert(
        &mut self,
        cond: Node,
    ) -> Result<(Node, Option<String>), ParseError> {
        let Node::Tuple { elements, .. } = cond else {
            return Ok((cond, None));
        };
        match <[Node; 2]>::try_from(elements) {
            Ok([cond, Node::Lit(Literal::Str(_), lit)]) => {
                let raw = &self.b[lit.start() + 1..lit.end() - 1];
                Ok((cond, Some(String::from_utf8_lossy(raw).into_owned())))
            }
            Ok(_) => Err(self.err(
                "an assert's second operand must be a string message: `assert(cond, \"message\")`; a failed assert always aborts".into(),
            )),
            Err(_) => Err(self.err(
                "an assert condition must be one expression, not a tuple".into(),
            )),
        }
    }
}
