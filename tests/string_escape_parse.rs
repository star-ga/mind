// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! String-literal escape-aware scanning gate.
//!
//! A `\"` (escaped quote) or `\\` (escaped backslash) inside a string literal
//! must not terminate the literal early. Before this was fixed, a string like
//! `"\""` mis-lexed: the scanner stopped at the escaped quote, consumed it as
//! the closing delimiter, and left the trailing `"` to open a phantom string —
//! corrupting every construct that followed (the lexer in `_lexer.mind` failed
//! to parse far downstream as a result).

use libmind::parser;

/// A bare escaped-quote string must parse on its own.
#[test]
fn escaped_quote_string_parses() {
    parser::parse(r#"fn f() -> string { return "\"" }"#).expect("escaped-quote string");
}

/// A bare escaped-backslash string must parse on its own.
#[test]
fn escaped_backslash_string_parses() {
    parser::parse(r#"fn f() -> string { return "\\" }"#).expect("escaped-backslash string");
}

/// The reduced `_lexer.mind` shape: an `if`/`else if` chain whose branches
/// concatenate escaped strings. Previously the `"\""` branch corrupted the
/// parser so the following `"\\"` branch reported `expected expression`.
#[test]
fn escaped_string_if_chain_parses() {
    let src = r#"
fn f(x: u8) -> string {
    let buf = ""
    if x == 1 { buf = buf + "\"" }
    else if x == 2 { buf = buf + "\\" }
    else { buf = "x" }
    return buf
}
"#;
    parser::parse(src).expect("escaped-string if/else-if chain");
}

/// Constructs after an escaped-quote string must still parse — i.e. the scan
/// position is left exactly past the real closing quote.
#[test]
fn construct_after_escaped_quote_is_intact() {
    let src = r#"
fn f() -> string {
    let a = "\""
    let b = 1 + 2
    return a
}
"#;
    parser::parse(src).expect("construct after escaped-quote string");
}
