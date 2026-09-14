// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Prefix and assignment parsing for D4 dereference nodes.

use crate::ast::{Node, Span};

use super::{P, ParseError};

pub(super) fn parse_assignment(
    parser: &mut P<'_>,
    operand: Box<Node>,
    start: usize,
) -> Result<Node, ParseError> {
    parser.advance();
    parser.skip_ws_and_newlines();
    let value = parser.parse_expr()?;
    let span = Span::new(start, parser.pos);
    Ok(Node::DerefAssign {
        target: operand,
        value: Box::new(value),
        span,
    })
}

pub(super) fn parse_prefix(parser: &mut P<'_>) -> Result<Option<Node>, ParseError> {
    if !parser.at(b'*') {
        return Ok(None);
    }
    let start = parser.pos;
    parser.pos += 1;
    parser.skip_ws();
    let operand = parser.parse_atom()?;
    let span = Span::new(start, parser.pos);
    Ok(Some(Node::Deref {
        operand: Box::new(operand),
        span,
    }))
}
