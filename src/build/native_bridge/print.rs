// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Rewrites supported numeric `print` statements into one private helper.
//!
//! The native lane refuses `Node::Print` because the frozen compiler has no
//! `printI64`. This adapter accepts only a statement `print` of one i64-shaped
//! argument, keeps that argument's source bytes, and appends `__native_print_i64`.
//! Anything else is left unchanged so the existing source refusal still fires.
//! A source with no `print` is not rewritten.

use std::collections::BTreeSet;

use crate::ast::{BinOp, Literal, Node, Span};
use crate::parser;

/// Private helper appended to the captured user region.
///
/// The native lane refuses any user function whose name starts with
/// `__mind_`, so this helper stays outside that prefix.
pub const HELPER_NAME: &str = "__native_print_i64";

/// Decimal stdout helper. The accumulator stays non-positive so `i64::MIN`
/// never negates. Digit is `0 - (v % 10)` under toward-zero remainder.
/// `__mind_write` must advance; a negative or zero write frees and fails.
const HELPER_SOURCE: &str = "\
fn __native_print_i64(n: i64) -> i64 {
    let buf: i64 = __mind_alloc(24);
    if buf == 0 {
        return 0 - 1;
    }
    let mut v: i64 = n;
    if v > 0 {
        v = 0 - v;
    }
    let mut pos: i64 = 21;
    if v == 0 {
        pos = 20;
        __mind_store_i8(buf + 20, 48);
    }
    while v < 0 {
        pos = pos - 1;
        let digit: i64 = 0 - (v % 10);
        __mind_store_i8(buf + pos, 48 + digit);
        v = v / 10;
    }
    if n < 0 {
        pos = pos - 1;
        __mind_store_i8(buf + pos, 45);
    }
    __mind_store_i8(buf + 21, 10);
    let mut off: i64 = 0;
    let count: i64 = 22 - pos;
    while off < count {
        let w: i64 = __mind_write(1, buf + pos + off, count - off, 0 - 1);
        if w < 0 {
            let _ = __mind_free(buf);
            return 0 - 1;
        }
        if w == 0 {
            let _ = __mind_free(buf);
            return 0 - 1;
        }
        if w > count - off {
            let _ = __mind_free(buf);
            return 0 - 1;
        }
        off = off + w;
    }
    let _ = __mind_free(buf);
    0
}
";

/// Original sources, or the same list with supported prints rewritten.
#[derive(Debug)]
pub enum PrintSources {
    Same,
    Expanded(Vec<Vec<u8>>),
}

/// Rewrite supported numeric statement prints. `reserved` is the seed name set.
pub fn expand_numeric_statement_prints(
    sources: &[Vec<u8>],
    reserved: &BTreeSet<String>,
) -> Result<PrintSources, String> {
    if sources.iter().any(|src| contains_ident(src, HELPER_NAME)) || reserved.contains(HELPER_NAME)
    {
        return Err(format!(
            "numeric print helper `{HELPER_NAME}` collides with a user, import, or seed name. \
             Refusing rather than shadowing it. (native.print.collision)"
        ));
    }
    let mut modules = Vec::with_capacity(sources.len());
    for src in sources {
        let text = match std::str::from_utf8(src) {
            Ok(text) => text,
            Err(_) => return Ok(PrintSources::Same),
        };
        match parser::parse(text) {
            Ok(module) => modules.push(module),
            Err(_) => return Ok(PrintSources::Same),
        }
    }
    let mut supported: Vec<Vec<Span>> = Vec::with_capacity(modules.len());
    let mut all_prints = 0usize;
    for module in &modules {
        let mut stmts = Vec::new();
        walk_stmt_prints(&module.items, &mut stmts);
        let mut found = Vec::new();
        for item in &module.items {
            walk_all(item, &mut found);
        }
        all_prints += found.len();
        if found.len() != stmts.len() || found.iter().any(|span| !stmts.contains(span)) {
            return Ok(PrintSources::Same);
        }
        supported.push(stmts);
    }
    if all_prints == 0 {
        return Ok(PrintSources::Same);
    }
    let mut expanded = Vec::with_capacity(sources.len());
    for (src, spans) in sources.iter().zip(supported.iter()) {
        expanded.push(rewrite_source(src, spans)?);
    }
    append_helper(expanded.last_mut().expect("print sources are non-empty"));
    for src in &expanded {
        let text = std::str::from_utf8(src).map_err(|_| {
            "numeric print rewrite produced non-utf8 source. (native.print.rewrite)".to_string()
        })?;
        let module = parser::parse(text).map_err(|_| {
            "numeric print rewrite did not parse. (native.print.rewrite)".to_string()
        })?;
        let mut left = Vec::new();
        for item in &module.items {
            walk_all(item, &mut left);
        }
        if !left.is_empty() {
            return Err(
                "numeric print rewrite left a print statement. (native.print.rewrite)".to_string(),
            );
        }
    }
    Ok(PrintSources::Expanded(expanded))
}

fn rewrite_source(src: &[u8], spans: &[Span]) -> Result<Vec<u8>, String> {
    let mut spans = spans.to_vec();
    spans.sort_by_key(|span| span.start());
    let mut out = Vec::with_capacity(src.len() + spans.len() * HELPER_NAME.len());
    let mut cursor = 0usize;
    for span in spans {
        let start = span.start();
        let end = span.end();
        if start < cursor
            || end > src.len()
            || start + 5 > end
            || &src[start..start + 5] != b"print"
        {
            return Err(
                "numeric print span does not cover the print keyword. (native.print.rewrite)"
                    .to_string(),
            );
        }
        out.extend_from_slice(&src[cursor..start]);
        out.extend_from_slice(HELPER_NAME.as_bytes());
        out.extend_from_slice(&src[start + 5..end]);
        cursor = end;
    }
    out.extend_from_slice(&src[cursor..]);
    Ok(out)
}

fn append_helper(src: &mut Vec<u8>) {
    if src.last().copied() != Some(b'\n') {
        src.push(b'\n');
    }
    src.extend_from_slice(HELPER_SOURCE.as_bytes());
}

fn contains_ident(src: &[u8], name: &str) -> bool {
    let name = name.as_bytes();
    let mut i = 0usize;
    while i + name.len() <= src.len() {
        if &src[i..i + name.len()] == name
            && (i == 0 || !is_ident_byte(src[i - 1]))
            && (i + name.len() == src.len() || !is_ident_byte(src[i + name.len()]))
        {
            return true;
        }
        i += 1;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_i64_expr(node: &Node) -> bool {
    match node {
        Node::Lit(Literal::Int(_), _) | Node::Lit(Literal::Ident(_), _) => true,
        Node::Paren(inner, _) => is_i64_expr(inner),
        Node::Neg { operand, .. } => is_i64_expr(operand),
        Node::Binary {
            op, left, right, ..
        } => {
            matches!(
                op,
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
            ) && is_i64_expr(left)
                && is_i64_expr(right)
        }
        _ => false,
    }
}

fn walk_stmt_prints(stmts: &[Node], spans: &mut Vec<Span>) {
    for stmt in stmts {
        if let Node::Print { args, span } = stmt {
            if args.len() == 1 && is_i64_expr(&args[0]) {
                spans.push(*span);
            }
        }
        match stmt {
            Node::FnDef(fd, _) => walk_stmt_prints(&fd.body, spans),
            Node::Block { stmts, .. } => walk_stmt_prints(stmts, spans),
            Node::If {
                then_branch,
                else_branch,
                ..
            } => {
                walk_stmt_prints(then_branch, spans);
                if let Some(else_branch) = else_branch {
                    walk_stmt_prints(else_branch, spans);
                }
            }
            Node::While { body, .. } => walk_stmt_prints(body, spans),
            Node::For { body, .. } => walk_stmt_prints(body, spans),
            Node::ForEach { body, .. } => walk_stmt_prints(body, spans),
            Node::Closure(data, _) => walk_stmt_prints(&data.body, spans),
            _ => {}
        }
    }
}

fn walk_all<'a>(node: &'a Node, out: &mut Vec<Span>) {
    if let Node::Print { span, .. } = node {
        out.push(*span);
    }
    match node {
        Node::FnDef(fd, _) => {
            for stmt in &fd.body {
                walk_all(stmt, out);
            }
        }
        Node::Closure(data, _) => {
            for stmt in &data.body {
                walk_all(stmt, out);
            }
        }
        other => {
            for child in crate::eval::closures::node_children_ref(other) {
                walk_all(child, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(sources: &[&str]) -> Result<PrintSources, String> {
        let owned: Vec<Vec<u8>> = sources.iter().map(|s| s.as_bytes().to_vec()).collect();
        expand_numeric_statement_prints(&owned, &BTreeSet::new())
    }

    fn text(sources: PrintSources) -> String {
        match sources {
            PrintSources::Same => "SAME".to_string(),
            PrintSources::Expanded(parts) => {
                String::from_utf8(parts.into_iter().flatten().collect()).unwrap()
            }
        }
    }

    #[test]
    fn no_print_stays_same() {
        let src = "fn main() -> i64 {\n    0\n}\n";
        assert!(matches!(expand(&[src]).unwrap(), PrintSources::Same));
    }

    #[test]
    fn numeric_statements_rewrite_and_keep_arguments() {
        let src = "fn f(id: i64, which: i64) -> i64 {\n    print(id);\n    print(9000 + which);\n    print(0);\n    0\n}\n";
        let out = text(expand(&[src]).unwrap());
        assert!(!out.contains("print("));
        assert!(out.contains("__native_print_i64(id);"));
        assert!(out.contains("__native_print_i64(9000 + which);"));
        assert!(out.contains("__native_print_i64(0);"));
        assert_eq!(out.matches("fn __native_print_i64").count(), 1);
        let again = text(expand(&[src]).unwrap());
        assert_eq!(out, again);
    }

    #[test]
    fn string_and_multi_arg_stay_refused() {
        assert!(matches!(
            expand(&["fn main() -> i64 {\n    print(\"hi\");\n    0\n}\n"]).unwrap(),
            PrintSources::Same
        ));
        assert!(matches!(
            expand(&["fn main() -> i64 {\n    print(1, 2);\n    0\n}\n"]).unwrap(),
            PrintSources::Same
        ));
    }

    #[test]
    fn helper_name_collision_refuses() {
        let src = "fn __native_print_i64(n: i64) -> i64 {\n    print(n);\n    0\n}\n";
        let err = expand(&[src]).unwrap_err();
        assert!(err.contains("native.print.collision"));
    }

    #[test]
    fn expanded_numeric_print_passes_native_lane() {
        let src = "fn main() -> i64 {\n    print(1);\n    print(0 - 2);\n    0\n}\n";
        let PrintSources::Expanded(parts) = expand(&[src]).unwrap() else {
            panic!("expected rewrite");
        };
        let expanded = String::from_utf8(parts.into_iter().flatten().collect()).unwrap();
        let ast = crate::parser::parse(&expanded).expect("expanded parses");
        crate::ir::native_lane_profile::profile_lane_admits_source(&ast).unwrap_or_else(|err| {
            panic!(
                "source {} in {:?} at {:?}",
                err.construct, err.function, err.at
            )
        });
        let ir = crate::eval::lower_to_ir(&ast).expect("expanded lowers");
        crate::ir::native_lane_profile::profile_lane_admits(&ast, &ir).unwrap_or_else(|err| {
            panic!("ir {} in {:?} at {:?}", err.construct, err.function, err.at)
        });
    }

    #[test]
    fn seed_name_collision_refuses() {
        let src = "fn main() -> i64 {\n    print(1);\n    0\n}\n";
        let owned = vec![src.as_bytes().to_vec()];
        let mut reserved = BTreeSet::new();
        reserved.insert(HELPER_NAME.to_string());
        let err = expand_numeric_statement_prints(&owned, &reserved).unwrap_err();
        assert!(err.contains("native.print.collision"));
    }
}
