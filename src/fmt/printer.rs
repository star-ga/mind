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

//! AST walker that emits canonical MIND source.
//!
//! Phase 2A: no soft line-wrap.  Emits from the AST; comments are re-attached
//! from the [`TriviaStream`].

use crate::ast::{
    Attribute, BinOp, BitOp, CallConv, EnumVariant, ExternFn, Field, Literal, LogicalOp, MatchArm,
    Module, Node, Param, Pattern, Span, SparseLayout, StructLitField, TypeAnn,
};
use crate::parser::{TriviaKind, TriviaStream};
use crate::project::MindcraftFormatConfig;

// ---------------------------------------------------------------------------
// Line-number index
// ---------------------------------------------------------------------------

/// Byte-offset-to-line-number index built from a source string.
///
/// `line_of(offset)` returns the zero-based line number for `offset`.
struct LineIndex {
    /// Byte offsets of each line's first character.  `starts[0] == 0` always.
    starts: Vec<usize>,
}

impl LineIndex {
    fn build(src: &str) -> Self {
        let mut starts = vec![0usize];
        for (i, &b) in src.as_bytes().iter().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        Self { starts }
    }

    /// Zero-based line number for `byte_offset`.
    fn line_of(&self, byte_offset: usize) -> usize {
        match self.starts.binary_search(&byte_offset) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
    }
}

// ---------------------------------------------------------------------------
// Trivia attachment
// ---------------------------------------------------------------------------

/// Trivia groups indexed by line number in the original source.
///
/// We attach a group of trivia lines to the first AST item whose start line
/// (in the stripped source, which has the same line count) is ≥ the trivia
/// group's line.
#[derive(Default)]
struct TriviaMap {
    /// `groups[line]` = trivia texts (comment strings) that appear on line
    /// `line` in the original source, in order.  Only comments are stored;
    /// blank-line records just influence spacing logic.
    comments_at: Vec<Vec<String>>,
    /// Set of line numbers that carry a blank line in the original source.
    blank_lines: std::collections::HashSet<usize>,
}

impl TriviaMap {
    fn build(trivia: &TriviaStream, orig_src: &str) -> Self {
        let idx = LineIndex::build(orig_src);
        let max_line = idx.starts.len() + 1;
        let mut comments_at: Vec<Vec<String>> = vec![Vec::new(); max_line + 1];
        let mut blank_lines = std::collections::HashSet::new();

        for t in &trivia.0 {
            let line = idx.line_of(t.byte_offset);
            match t.kind {
                TriviaKind::BlankLine => {
                    blank_lines.insert(line);
                }
                TriviaKind::LineComment | TriviaKind::DocComment => {
                    if line < comments_at.len() {
                        comments_at[line].push(t.text.clone());
                    }
                }
            }
        }
        Self {
            comments_at,
            blank_lines,
        }
    }
}

// ---------------------------------------------------------------------------
// Printer state
// ---------------------------------------------------------------------------

struct Printer<'a> {
    out: String,
    cfg: &'a MindcraftFormatConfig,
    indent: usize,
    trivia: TriviaMap,
    /// Line index into the stripped source (same line count as original).
    stripped_idx: LineIndex,
    /// Last trivia line we've already consumed, to avoid double-emit.
    last_trivia_line: usize,
}

impl<'a> Printer<'a> {
    fn new(cfg: &'a MindcraftFormatConfig, trivia_map: TriviaMap, stripped_src: &str) -> Self {
        Self {
            out: String::new(),
            cfg,
            indent: 0,
            trivia: trivia_map,
            stripped_idx: LineIndex::build(stripped_src),
            last_trivia_line: 0,
        }
    }

    fn indent_str(&self) -> String {
        " ".repeat(self.indent * self.cfg.indent_width as usize)
    }

    fn push(&mut self, s: &str) {
        self.out.push_str(s);
    }

    /// Emit any leading comments/blank lines that appear before line `item_line`
    /// in the original source, preserving blank lines between comment groups.
    /// Updates `last_trivia_line` to avoid re-emitting.
    fn emit_leading_trivia(&mut self, item_line: usize) {
        if item_line <= self.last_trivia_line {
            return;
        }
        // Walk line by line from last_trivia_line to item_line, emitting
        // comments and blank lines in source order.
        let from = self.last_trivia_line;
        let to = item_line.min(self.trivia.comments_at.len());
        let indent = self.indent_str();
        for line in from..to {
            // Blank line on this source line → emit only if output is non-empty.
            if self.trivia.blank_lines.contains(&line) && !self.out.is_empty() {
                self.out.push('\n');
            }
            // Comments on this line.
            let comments: Vec<String> = if line < self.trivia.comments_at.len() {
                self.trivia.comments_at[line].clone()
            } else {
                Vec::new()
            };
            for c in &comments {
                self.out.push_str(&indent);
                self.out.push_str(c);
                self.out.push('\n');
            }
        }
        self.last_trivia_line = item_line;
    }
}

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

/// Emit a canonical MIND source string for `module`.
///
/// `orig_src` is the **original** (pre-comment-strip) source text; it is used
/// only for trivia line-number computation and is never written to the output.
pub fn print_module(
    module: &Module,
    trivia: &TriviaStream,
    cfg: &MindcraftFormatConfig,
    orig_src: &str,
) -> String {
    // Build a stripped copy for span→line mapping.
    let stripped = strip_for_lines(orig_src);
    let total_lines = orig_src.lines().count();

    let trivia_map = TriviaMap::build(trivia, orig_src);
    let mut p = Printer::new(cfg, trivia_map, &stripped);

    // Emit the file-top copyright/license header (trivia before line 0 or any
    // comment group that precedes the first item).
    let first_item_line = module
        .items
        .first()
        .map(|n| p.stripped_idx.line_of(n.span_start()))
        .unwrap_or(total_lines);

    // Emit leading comments (copyright header, etc.) before first item.
    p.emit_leading_trivia(first_item_line);

    let item_count = module.items.len();
    for (idx, item) in module.items.iter().enumerate() {
        let item_line = p.stripped_idx.line_of(item.span_start());

        // Emit any leading trivia (comments + blank lines) that precede this
        // item in the original source.  The walk covers the region from
        // last_trivia_line up to (but not including) item_line, naturally
        // emitting blank separators between items and between comment groups.
        p.emit_leading_trivia(item_line);

        // Emit the item itself.
        emit_node(&mut p, item, 0);
        p.push("\n");

        // After the last item, flush any trailing trivia (comments that
        // appear after the last AST node — e.g. file-end doc comments,
        // or block-interior comments the trivia walk places past the
        // last item).  Use total_lines (exclusive upper bound) so the
        // loop in emit_leading_trivia covers every line including the
        // last one.
        if idx + 1 == item_count {
            p.emit_leading_trivia(total_lines);
        }
    }

    // Ensure exactly one trailing newline.
    let out = p.out.trim_end_matches('\n').to_string();
    if out.is_empty() {
        String::new()
    } else {
        out + "\n"
    }
}

/// Build a line-structure-preserving stripped copy (comments removed but line
/// count preserved) using a simple fast pass.  Only used for span→line mapping.
fn strip_for_lines(src: &str) -> String {
    // Mirrors `strip_comments_fast` in the trivia module: blank each comment
    // region with ASCII spaces **in place** rather than removing the bytes, so
    // the result is byte-length identical to `src`. This keeps the span→line
    // index built here aligned with the AST spans produced by
    // `parse_with_trivia` (which is also length-preserving) — a comment's bytes
    // occupy the same positions in both, so `line_of(span_start)` is exact.
    let bytes = src.as_bytes();
    let mut out = bytes.to_vec();
    let mut line_start = 0usize;
    while line_start < bytes.len() {
        let newline_pos = bytes[line_start..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|rel| line_start + rel)
            .unwrap_or(bytes.len());
        let line = &bytes[line_start..newline_pos];
        let mut in_string = false;
        let mut i = 0;
        while i < line.len() {
            if in_string {
                if line[i] == b'\\' && i + 1 < line.len() {
                    i += 2;
                    continue;
                }
                if line[i] == b'"' {
                    in_string = false;
                }
            } else if line[i] == b'"' {
                in_string = true;
            } else if line[i] == b'/' && i + 1 < line.len() && line[i + 1] == b'/' {
                // Blank `[line_start + i, newline_pos)` with ASCII spaces.
                for b in &mut out[line_start + i..newline_pos] {
                    *b = b' ';
                }
                break;
            }
            i += 1;
        }
        line_start = newline_pos + 1;
    }
    String::from_utf8(out).expect("blanking comment bytes with ASCII spaces preserves UTF-8")
}

// ---------------------------------------------------------------------------
// Node emission
// ---------------------------------------------------------------------------

fn emit_node(p: &mut Printer, node: &Node, _extra_indent: usize) {
    match node {
        Node::FnDef {
            is_pub,
            name,
            params,
            ret_type,
            body,
            attrs,
            span,
            ..
        } => {
            emit_fn_def(p, *is_pub, name, params, ret_type, body, attrs, *span);
        }
        Node::StructDef {
            is_pub,
            name,
            fields,
            attrs,
            ..
        } => {
            emit_struct_def(p, *is_pub, name, fields, attrs);
        }
        Node::EnumDef {
            is_pub,
            name,
            variants,
            attrs,
            ..
        } => {
            emit_enum_def(p, *is_pub, name, variants, attrs);
        }
        Node::Const {
            name,
            ty,
            value,
            attrs,
            ..
        } => {
            emit_const(p, name, ty, value, attrs);
        }
        Node::TypeAlias {
            name,
            target,
            attrs,
            ..
        } => {
            emit_type_alias(p, name, target, attrs);
        }
        Node::Import { path, .. } => {
            emit_import(p, path);
        }
        Node::Export { names, .. } => {
            emit_export(p, names);
        }
        Node::Let {
            name,
            mutable,
            ann,
            value,
            ..
        } => {
            emit_let(p, name, *mutable, ann, value);
        }
        Node::Assign { name, value, .. } => {
            emit_assign(p, name, value);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            emit_index_assign(p, receiver, index, value);
        }
        Node::FieldAssign {
            receiver,
            field,
            value,
            ..
        } => {
            emit_field_assign(p, receiver, field, value);
        }
        Node::Return { value, .. } => {
            emit_return(p, value.as_deref());
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            span,
        } => {
            emit_if(p, cond, then_branch, else_branch.as_deref(), *span);
        }
        Node::For {
            var,
            start,
            end,
            body,
            span,
        } => {
            emit_for(p, var, start, end, body, *span);
        }
        Node::ForEach {
            var,
            collection,
            body,
            span,
        } => {
            emit_foreach(p, var, collection, body, *span);
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, span } => {
            emit_while(p, cond, body, *span);
        }
        #[cfg(feature = "std-surface")]
        Node::Region { body, span } => {
            emit_region(p, body, *span);
        }
        Node::Assert { cond, msg, .. } => {
            emit_assert(p, cond, msg.as_deref());
        }
        Node::Print { args, .. } => {
            emit_print(p, args);
        }
        Node::Block { stmts, span } => {
            let close_line = p.stripped_idx.line_of(span.end());
            emit_block_stmts(p, stmts, close_line);
        }
        Node::Match {
            scrutinee, arms, ..
        } => {
            emit_match(p, scrutinee, arms);
        }
        // Expression-statements (top-level or in fn body)
        _ => {
            let ind = p.indent_str();
            p.push(&ind);
            emit_expr(p, node);
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level item emitters
// ---------------------------------------------------------------------------

/// Emit an attribute list in canonical Rust-style `#[name]` / `#[name(args)]`
/// form (RFC 0012 §5), one per line at the current indent. Shared by every
/// item emitter so the canonical form lives in exactly one place.
fn emit_attrs(p: &mut Printer, attrs: &[Attribute]) {
    for attr in attrs {
        let ind = p.indent_str();
        p.push(&ind);
        p.push("#[");
        p.push(&attr.name);
        if !attr.args.is_empty() {
            p.push("(");
            p.push(&attr.args.join(", "));
            p.push(")");
        }
        p.push("]\n");
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_fn_def(
    p: &mut Printer,
    is_pub: bool,
    name: &str,
    params: &[Param],
    ret_type: &Option<TypeAnn>,
    body: &[Node],
    attrs: &[Attribute],
    span: Span,
) {
    // Previously fn attributes were parsed then silently dropped by the
    // formatter; emit them so `[test]` / `[deterministic]` / `[reap_threshold]`
    // round-trip through `mindc fmt` instead of being erased.
    emit_attrs(p, attrs);
    let ind = p.indent_str();
    p.push(&ind);
    if is_pub {
        p.push("pub ");
    }
    p.push("fn ");
    p.push(name);
    p.push("(");
    for (i, param) in params.iter().enumerate() {
        if i > 0 {
            p.push(", ");
        }
        p.push(&param.name);
        p.push(": ");
        emit_type_ann(p, &param.ty);
    }
    p.push(")");
    if let Some(rt) = ret_type {
        p.push(" -> ");
        emit_type_ann(p, rt);
    }
    p.push(" {\n");
    p.indent += 1;
    // The close_line is the line of the `}` that ends this function body.
    // Passing it lets emit_body_stmts flush any trailing body comments
    // (after the last statement, before `}`) at the correct indent level
    // rather than letting them float to after the `}`.
    let close_line = p.stripped_idx.line_of(span.end());
    emit_body_stmts(p, body, close_line);
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
}

fn emit_struct_def(
    p: &mut Printer,
    is_pub: bool,
    name: &str,
    fields: &[Field],
    attrs: &[Attribute],
) {
    emit_attrs(p, attrs);
    let ind = p.indent_str();
    p.push(&ind);
    if is_pub {
        p.push("pub ");
    }
    p.push("struct ");
    p.push(name);
    p.push(" {\n");
    p.indent += 1;
    for field in fields {
        let ind = p.indent_str();
        p.push(&ind);
        if field.is_pub {
            p.push("pub ");
        }
        p.push(&field.name);
        p.push(": ");
        emit_type_ann(p, &field.ty);
        p.push(",\n");
    }
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
}

fn emit_enum_def(
    p: &mut Printer,
    is_pub: bool,
    name: &str,
    variants: &[EnumVariant],
    attrs: &[Attribute],
) {
    emit_attrs(p, attrs);
    let ind = p.indent_str();
    p.push(&ind);
    if is_pub {
        p.push("pub ");
    }
    p.push("enum ");
    p.push(name);
    p.push(" {\n");
    p.indent += 1;
    for v in variants {
        let ind = p.indent_str();
        p.push(&ind);
        p.push(&v.name);
        if !v.field_names.is_empty() {
            // Struct variant `V { f: T, g: U }` — names parallel to payload.
            p.push(" { ");
            for (i, fname) in v.field_names.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                p.push(fname);
                p.push(": ");
                emit_type_ann(p, &v.payload[i]);
            }
            p.push(" }");
        } else if !v.payload.is_empty() {
            p.push("(");
            for (i, ty) in v.payload.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_type_ann(p, ty);
            }
            p.push(")");
        }
        p.push(",\n");
    }
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
}

fn emit_const(
    p: &mut Printer,
    name: &str,
    ty: &Option<TypeAnn>,
    value: &Node,
    attrs: &[Attribute],
) {
    emit_attrs(p, attrs);
    let ind = p.indent_str();
    p.push(&ind);
    p.push("const ");
    p.push(name);
    if let Some(t) = ty {
        p.push(": ");
        emit_type_ann(p, t);
    }
    p.push(" = ");
    emit_expr(p, value);
}

fn emit_type_alias(p: &mut Printer, name: &str, target: &TypeAnn, attrs: &[Attribute]) {
    emit_attrs(p, attrs);
    let ind = p.indent_str();
    p.push(&ind);
    p.push("type ");
    p.push(name);
    p.push(" = ");
    emit_type_ann(p, target);
}

fn emit_import(p: &mut Printer, path: &[String]) {
    let ind = p.indent_str();
    p.push(&ind);
    p.push("import ");
    p.push(&path.join("."));
    p.push(";");
}

fn emit_export(p: &mut Printer, names: &[String]) {
    let ind = p.indent_str();
    p.push(&ind);
    p.push("export { ");
    p.push(&names.join(", "));
    p.push(" }");
}

// ---------------------------------------------------------------------------
// Statement emitters (used inside fn bodies)
// ---------------------------------------------------------------------------

/// Emit a list of statements that form a fn body, with collapse of excess
/// blank lines (max 1) and stripping leading/trailing blank lines.
///
/// `close_line` is the line number (in the stripped source) of the `}` that
/// closes this block.  It is used to flush any comments that appear after the
/// last statement but before the closing brace, so they are emitted at the
/// correct indent level rather than floating past the `}`.
///
/// Semicolon policy (canonical MIND style — matches std/*.mind):
/// - `let` bindings, `return`, `assign`, `assert`, `print` → always `;`
/// - Block-structured nodes (`if`, `for`, `while`, `match`, inner blocks) → no `;`
/// - Bare expression-statements that are the LAST statement → no `;`
///   (implicit return expression; adding `;` is valid but not canonical)
/// - Bare expression-statements that are NOT last → `;`
fn emit_body_stmts(p: &mut Printer, stmts: &[Node], close_line: usize) {
    let len = stmts.len();
    for (i, stmt) in stmts.iter().enumerate() {
        let is_last = i + 1 == len;
        // Flush any comments (and blank lines) that appear in the source
        // before this statement.  This covers leading body comments, comments
        // between statements, and trailing same-line comments from the
        // preceding statement (which live on the preceding line and are
        // consumed here as "leading trivia" of the current statement).
        let stmt_line = p.stripped_idx.line_of(stmt.span_start());
        p.emit_leading_trivia(stmt_line);
        match stmt {
            Node::If {
                cond,
                then_branch,
                else_branch,
                span,
            } => {
                let ind = p.indent_str();
                p.push(&ind);
                emit_if_inline(p, cond, then_branch, else_branch.as_deref(), *span);
                p.push("\n");
            }
            Node::For {
                var,
                start,
                end,
                body,
                span,
            } => {
                let ind = p.indent_str();
                p.push(&ind);
                emit_for_inline(p, var, start, end, body, *span);
                p.push("\n");
            }
            Node::ForEach {
                var,
                collection,
                body,
                span,
            } => {
                let ind = p.indent_str();
                p.push(&ind);
                emit_foreach_inline(p, var, collection, body, *span);
                p.push("\n");
            }
            #[cfg(feature = "std-surface")]
            Node::While { cond, body, span } => {
                let ind = p.indent_str();
                p.push(&ind);
                emit_while_inline(p, cond, body, *span);
                p.push("\n");
            }
            #[cfg(feature = "std-surface")]
            Node::Region { body, span } => {
                let ind = p.indent_str();
                p.push(&ind);
                emit_region_inline(p, body, *span);
                p.push("\n");
            }
            Node::Match {
                scrutinee, arms, ..
            } => {
                let ind = p.indent_str();
                p.push(&ind);
                emit_match_inline(p, scrutinee, arms);
                p.push("\n");
            }
            Node::Block { stmts: inner, span } => {
                let ind = p.indent_str();
                p.push(&ind);
                p.push("{\n");
                p.indent += 1;
                let inner_close = p.stripped_idx.line_of(span.end());
                emit_body_stmts(p, inner, inner_close);
                p.indent -= 1;
                let ind = p.indent_str();
                p.push(&ind);
                p.push("}\n");
            }
            // FnDef nested inside a body (uncommon but valid in MIND)
            Node::FnDef {
                is_pub,
                name,
                params,
                ret_type,
                body,
                attrs,
                span,
                ..
            } => {
                emit_fn_def(p, *is_pub, name, params, ret_type, body, attrs, *span);
                p.push("\n");
            }
            // Statements with mandatory semicolons regardless of position.
            Node::Let { .. }
            | Node::Return { .. }
            | Node::Assign { .. }
            | Node::IndexAssign { .. }
            | Node::FieldAssign { .. }
            | Node::Assert { .. }
            | Node::Print { .. } => {
                emit_stmt(p, stmt);
                p.push(";\n");
            }
            // Bare expression-statements: add `;` only when not the last
            // statement (non-last = definitely a side-effecting call, not
            // an implicit return).  The last bare expression is the
            // canonical implicit-return form used throughout std/*.mind.
            _ => {
                emit_stmt(p, stmt);
                if is_last {
                    p.push("\n");
                } else {
                    p.push(";\n");
                }
            }
        }
    }
    // Flush any comments that appear after the last statement but before the
    // closing `}`.  Without this flush they would escape into the enclosing
    // scope and appear after the `}` instead of inside the block.
    p.emit_leading_trivia(close_line);
}

fn emit_block_stmts(p: &mut Printer, stmts: &[Node], close_line: usize) {
    emit_body_stmts(p, stmts, close_line);
}

fn emit_stmt(p: &mut Printer, node: &Node) {
    let ind = p.indent_str();
    p.push(&ind);
    match node {
        Node::Let {
            name,
            mutable,
            ann,
            value,
            ..
        } => {
            emit_let_inline(p, name, *mutable, ann, value);
        }
        Node::LetTuple {
            names,
            mutable,
            value,
            ..
        } => {
            p.push("let ");
            if *mutable {
                p.push("mut ");
            }
            p.push("(");
            for (i, nm) in names.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                p.push(nm);
            }
            p.push(") = ");
            emit_expr(p, value);
        }
        Node::Assign { name, value, .. } => {
            p.push(name);
            p.push(" = ");
            emit_expr(p, value);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            emit_expr(p, receiver);
            p.push("[");
            emit_expr(p, index);
            p.push("] = ");
            emit_expr(p, value);
        }
        Node::FieldAssign {
            receiver,
            field,
            value,
            ..
        } => {
            emit_expr(p, receiver);
            p.push(".");
            p.push(field);
            p.push(" = ");
            emit_expr(p, value);
        }
        Node::Return { value, .. } => {
            p.push("return");
            if let Some(v) = value {
                p.push(" ");
                emit_expr(p, v);
            }
        }
        Node::Assert { cond, msg, .. } => {
            p.push("assert ");
            emit_expr(p, cond);
            if let Some(m) = msg {
                p.push(", \"");
                p.push(m);
                p.push("\"");
            }
        }
        Node::Print { args, .. } => {
            p.push("print(");
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_expr(p, a);
            }
            p.push(")");
        }
        // Expression statements
        _ => {
            emit_expr(p, node);
        }
    }
}

fn emit_let(p: &mut Printer, name: &str, mutable: bool, ann: &Option<TypeAnn>, value: &Node) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_let_inline(p, name, mutable, ann, value);
}

fn emit_let_inline(
    p: &mut Printer,
    name: &str,
    mutable: bool,
    ann: &Option<TypeAnn>,
    value: &Node,
) {
    p.push("let ");
    if mutable {
        p.push("mut ");
    }
    p.push(name);
    if let Some(t) = ann {
        p.push(": ");
        emit_type_ann(p, t);
    }
    p.push(" = ");
    emit_expr(p, value);
}

fn emit_assign(p: &mut Printer, name: &str, value: &Node) {
    let ind = p.indent_str();
    p.push(&ind);
    p.push(name);
    p.push(" = ");
    emit_expr(p, value);
}

fn emit_index_assign(p: &mut Printer, receiver: &Node, index: &Node, value: &Node) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_expr(p, receiver);
    p.push("[");
    emit_expr(p, index);
    p.push("] = ");
    emit_expr(p, value);
}

fn emit_field_assign(p: &mut Printer, receiver: &Node, field: &str, value: &Node) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_expr(p, receiver);
    p.push(".");
    p.push(field);
    p.push(" = ");
    emit_expr(p, value);
}

fn emit_return(p: &mut Printer, value: Option<&Node>) {
    let ind = p.indent_str();
    p.push(&ind);
    p.push("return");
    if let Some(v) = value {
        p.push(" ");
        emit_expr(p, v);
    }
}

fn emit_assert(p: &mut Printer, cond: &Node, msg: Option<&str>) {
    let ind = p.indent_str();
    p.push(&ind);
    p.push("assert ");
    emit_expr(p, cond);
    if let Some(m) = msg {
        p.push(", \"");
        p.push(m);
        p.push("\"");
    }
}

fn emit_print(p: &mut Printer, args: &[Node]) {
    let ind = p.indent_str();
    p.push(&ind);
    p.push("print(");
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            p.push(", ");
        }
        emit_expr(p, a);
    }
    p.push(")");
}

fn emit_if(
    p: &mut Printer,
    cond: &Node,
    then_branch: &[Node],
    else_branch: Option<&[Node]>,
    span: Span,
) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_if_inline(p, cond, then_branch, else_branch, span);
}

fn emit_if_inline(
    p: &mut Printer,
    cond: &Node,
    then_branch: &[Node],
    else_branch: Option<&[Node]>,
    span: Span,
) {
    p.push("if ");
    emit_expr(p, cond);
    p.push(" {\n");
    p.indent += 1;
    // The If parser calls `skip_ws_and_newlines()` after consuming the then-`}`
    // before setting `span.end()`, so `span.end()` may point to the first token
    // of the NEXT statement (a different line).  To avoid consuming trivia that
    // belongs outside this block we derive `then_close` from the last statement
    // inside the branch rather than from `span.end()`.
    //
    // For If-with-else the parser does NOT skip after the else-`}`, so
    // `span.end()` correctly points just past the else-`}` and is safe to use
    // as `else_close`.
    let then_close = then_branch
        .last()
        .map(|s| p.stripped_idx.line_of(s.span_end()))
        .unwrap_or_else(|| p.stripped_idx.line_of(span.start()));
    emit_body_stmts(p, then_branch, then_close);
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
    if let Some(eb) = else_branch {
        p.push(" else {\n");
        p.indent += 1;
        // For the else branch, span.end() is just after the else-`}` — correct.
        let else_close = p.stripped_idx.line_of(span.end());
        emit_body_stmts(p, eb, else_close);
        p.indent -= 1;
        let ind = p.indent_str();
        p.push(&ind);
        p.push("}");
    }
}

fn emit_for(p: &mut Printer, var: &str, start: &Node, end: &Node, body: &[Node], span: Span) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_for_inline(p, var, start, end, body, span);
}

fn emit_for_inline(
    p: &mut Printer,
    var: &str,
    start: &Node,
    end: &Node,
    body: &[Node],
    span: Span,
) {
    p.push("for ");
    p.push(var);
    p.push(" in ");
    emit_expr(p, start);
    p.push("..");
    emit_expr(p, end);
    p.push(" {\n");
    p.indent += 1;
    let close_line = p.stripped_idx.line_of(span.end());
    emit_body_stmts(p, body, close_line);
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
}

fn emit_foreach(p: &mut Printer, var: &str, collection: &Node, body: &[Node], span: Span) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_foreach_inline(p, var, collection, body, span);
}

fn emit_foreach_inline(p: &mut Printer, var: &str, collection: &Node, body: &[Node], span: Span) {
    p.push("for ");
    p.push(var);
    p.push(" in ");
    emit_expr(p, collection);
    p.push(" {\n");
    p.indent += 1;
    let close_line = p.stripped_idx.line_of(span.end());
    emit_body_stmts(p, body, close_line);
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
}

#[cfg(feature = "std-surface")]
fn emit_while(p: &mut Printer, cond: &Node, body: &[Node], span: Span) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_while_inline(p, cond, body, span);
}

#[cfg(feature = "std-surface")]
fn emit_while_inline(p: &mut Printer, cond: &Node, body: &[Node], span: Span) {
    p.push("while ");
    emit_expr(p, cond);
    p.push(" {\n");
    p.indent += 1;
    let close_line = p.stripped_idx.line_of(span.end());
    emit_body_stmts(p, body, close_line);
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
}

#[cfg(feature = "std-surface")]
fn emit_region(p: &mut Printer, body: &[Node], span: Span) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_region_inline(p, body, span);
}

#[cfg(feature = "std-surface")]
fn emit_region_inline(p: &mut Printer, body: &[Node], span: Span) {
    p.push("region {\n");
    p.indent += 1;
    let close_line = p.stripped_idx.line_of(span.end());
    emit_body_stmts(p, body, close_line);
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
}

fn emit_match(p: &mut Printer, scrutinee: &Node, arms: &[MatchArm]) {
    let ind = p.indent_str();
    p.push(&ind);
    emit_match_inline(p, scrutinee, arms);
}

fn emit_match_inline(p: &mut Printer, scrutinee: &Node, arms: &[MatchArm]) {
    p.push("match ");
    emit_expr(p, scrutinee);
    p.push(" {\n");
    p.indent += 1;
    for arm in arms {
        let ind = p.indent_str();
        p.push(&ind);
        emit_pattern(p, &arm.pattern);
        p.push(" => ");
        // arm body: if it's a block, inline it; otherwise expression
        match &arm.body {
            Node::Block { stmts, span } => {
                p.push("{\n");
                p.indent += 1;
                let close_line = p.stripped_idx.line_of(span.end());
                emit_body_stmts(p, stmts, close_line);
                p.indent -= 1;
                let ind = p.indent_str();
                p.push(&ind);
                p.push("},\n");
            }
            _ => {
                emit_expr(p, &arm.body);
                p.push(",\n");
            }
        }
    }
    p.indent -= 1;
    let ind = p.indent_str();
    p.push(&ind);
    p.push("}");
}

// ---------------------------------------------------------------------------
// Expression emitters
// ---------------------------------------------------------------------------

fn emit_expr(p: &mut Printer, node: &Node) {
    match node {
        Node::Lit(lit, _) => emit_literal(p, lit),
        #[cfg(feature = "std-surface")]
        Node::Break { .. } => p.push("break"),
        #[cfg(feature = "std-surface")]
        Node::Continue { .. } => p.push("continue"),
        Node::Binary {
            op, left, right, ..
        } => emit_binary(p, op, left, right),
        Node::Logical {
            op, left, right, ..
        } => emit_logical(p, op, left, right),
        Node::Bitwise {
            op, left, right, ..
        } => emit_bitwise(p, op, left, right),
        Node::Paren(inner, _) => {
            p.push("(");
            emit_expr(p, inner);
            p.push(")");
        }
        Node::Tuple { elements, .. } => {
            p.push("(");
            for (i, e) in elements.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_expr(p, e);
            }
            p.push(")");
        }
        Node::LetTuple {
            names,
            mutable,
            value,
            ..
        } => {
            // A destructuring `let` is a statement; if it ever reaches expression
            // position it still round-trips faithfully.
            p.push("let ");
            if *mutable {
                p.push("mut ");
            }
            p.push("(");
            for (i, nm) in names.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                p.push(nm);
            }
            p.push(") = ");
            emit_expr(p, value);
        }
        Node::Neg { operand, .. } => {
            p.push("-");
            emit_expr(p, operand);
        }
        Node::Not { operand, .. } => {
            p.push("!");
            emit_expr(p, operand);
        }
        Node::Call { callee, args, .. } => emit_call(p, callee, args),
        Node::MethodCall {
            receiver,
            method,
            args,
            ..
        } => {
            emit_expr(p, receiver);
            p.push(".");
            p.push(method);
            p.push("(");
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_expr(p, a);
            }
            p.push(")");
        }
        Node::FieldAccess {
            receiver, field, ..
        } => {
            emit_expr(p, receiver);
            p.push(".");
            p.push(field);
        }
        Node::IndexAccess {
            receiver, index, ..
        } => {
            emit_expr(p, receiver);
            p.push("[");
            emit_expr(p, index);
            p.push("]");
        }
        Node::StructLit { name, fields, .. } => emit_struct_lit(p, name, fields),
        Node::ArrayLit { elements, .. } => {
            p.push("[");
            for (i, e) in elements.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_expr(p, e);
            }
            p.push("]");
        }
        Node::MapLit { entries, .. } => {
            p.push("{");
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_expr(p, k);
                p.push(": ");
                emit_expr(p, v);
            }
            p.push("}");
        }
        Node::As { expr, ty, .. } => {
            emit_expr(p, expr);
            p.push(" as ");
            emit_type_ann(p, ty);
        }
        Node::Ref { mutable, inner, .. } => {
            if *mutable {
                p.push("&mut ");
            } else {
                p.push("&");
            }
            emit_expr(p, inner);
        }
        // These are builtins that appear as expressions
        Node::CallGrad { loss, wrt, .. } => {
            p.push("grad(");
            emit_expr(p, loss);
            p.push("; ");
            p.push(&wrt.join(", "));
            p.push(")");
        }
        Node::CallTensorSum {
            x, axes, keepdims, ..
        } => {
            p.push("tensor.sum(");
            emit_expr(p, x);
            p.push(", axes=[");
            for (i, ax) in axes.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                p.push(&ax.to_string());
            }
            p.push("]");
            if *keepdims {
                p.push(", keepdims=true");
            }
            p.push(")");
        }
        Node::CallTensorMean {
            x, axes, keepdims, ..
        } => {
            p.push("tensor.mean(");
            emit_expr(p, x);
            p.push(", axes=[");
            for (i, ax) in axes.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                p.push(&ax.to_string());
            }
            p.push("]");
            if *keepdims {
                p.push(", keepdims=true");
            }
            p.push(")");
        }
        Node::CallReshape { x, dims, .. } => {
            p.push("tensor.reshape(");
            emit_expr(p, x);
            p.push(", [");
            p.push(&dims.join(", "));
            p.push("])");
        }
        Node::CallExpandDims { x, axis, .. } => {
            p.push("tensor.expand_dims(");
            emit_expr(p, x);
            p.push(", ");
            p.push(&axis.to_string());
            p.push(")");
        }
        Node::CallSqueeze { x, axes, .. } => {
            p.push("tensor.squeeze(");
            emit_expr(p, x);
            p.push(", [");
            let parts: Vec<String> = axes.iter().map(|a| a.to_string()).collect();
            p.push(&parts.join(", "));
            p.push("])");
        }
        Node::CallTranspose { x, axes, .. } => {
            p.push("tensor.transpose(");
            emit_expr(p, x);
            if let Some(axes) = axes {
                p.push(", [");
                let parts: Vec<String> = axes.iter().map(|a| a.to_string()).collect();
                p.push(&parts.join(", "));
                p.push("]");
            }
            p.push(")");
        }
        Node::CallIndex { x, axis, i, .. } => {
            p.push("tensor.index(");
            emit_expr(p, x);
            p.push(", ");
            p.push(&axis.to_string());
            p.push(", ");
            p.push(&i.to_string());
            p.push(")");
        }
        Node::CallSlice {
            x,
            axis,
            start,
            end,
            ..
        } => {
            p.push("tensor.slice(");
            emit_expr(p, x);
            p.push(", ");
            p.push(&axis.to_string());
            p.push(", ");
            p.push(&start.to_string());
            p.push(", ");
            p.push(&end.to_string());
            p.push(")");
        }
        Node::CallSliceStride {
            x,
            axis,
            start,
            end,
            step,
            ..
        } => {
            p.push("tensor.slice_stride(");
            emit_expr(p, x);
            p.push(", ");
            p.push(&axis.to_string());
            p.push(", ");
            p.push(&start.to_string());
            p.push(", ");
            p.push(&end.to_string());
            p.push(", ");
            p.push(&step.to_string());
            p.push(")");
        }
        Node::CallGather { x, axis, idx, .. } => {
            p.push("tensor.gather(");
            emit_expr(p, x);
            p.push(", ");
            p.push(&axis.to_string());
            p.push(", ");
            emit_expr(p, idx);
            p.push(")");
        }
        Node::CallDot { a, b, .. } => {
            p.push("tensor.dot(");
            emit_expr(p, a);
            p.push(", ");
            emit_expr(p, b);
            p.push(")");
        }
        Node::CallMatMul { a, b, .. } => {
            p.push("tensor.matmul(");
            emit_expr(p, a);
            p.push(", ");
            emit_expr(p, b);
            p.push(")");
        }
        // RFC 0012 Phase B — tensor operator pretty-printing.
        Node::TensorMatmul { lhs, rhs, .. } => {
            emit_expr(p, lhs);
            p.push(" @ ");
            emit_expr(p, rhs);
        }
        Node::TensorElemwise { op, lhs, rhs, .. } => {
            use crate::ast::TensorElemOp;
            emit_expr(p, lhs);
            let op_str = match op {
                TensorElemOp::Add => " .+ ",
                TensorElemOp::Sub => " .- ",
                TensorElemOp::Mul => " .* ",
                TensorElemOp::Div => " ./ ",
            };
            p.push(op_str);
            emit_expr(p, rhs);
        }
        Node::CallTensorRelu { x, .. } => {
            p.push("tensor.relu(");
            emit_expr(p, x);
            p.push(")");
        }
        Node::CallTensorRand { shape, .. } => {
            p.push("tensor.rand([");
            let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            p.push(&parts.join(", "));
            p.push("])");
        }
        Node::CallTensorConv2d {
            x,
            w,
            stride_h,
            stride_w,
            padding,
            ..
        } => {
            p.push("tensor.conv2d(");
            emit_expr(p, x);
            p.push(", ");
            emit_expr(p, w);
            p.push(", stride_h=");
            p.push(&stride_h.to_string());
            p.push(", stride_w=");
            p.push(&stride_w.to_string());
            p.push(", padding=");
            let pad_str = match padding {
                crate::types::ConvPadding::Valid => "valid",
                crate::types::ConvPadding::Same => "same",
            };
            p.push(pad_str);
            p.push(")");
        }
        // Statements that can appear as expressions in certain contexts
        Node::If {
            cond,
            then_branch,
            else_branch,
            span,
        } => {
            emit_if_inline(p, cond, then_branch, else_branch.as_deref(), *span);
        }
        Node::Block { stmts, span } => {
            p.push("{\n");
            p.indent += 1;
            let close_line = p.stripped_idx.line_of(span.end());
            emit_body_stmts(p, stmts, close_line);
            p.indent -= 1;
            let ind = p.indent_str();
            p.push(&ind);
            p.push("}");
        }
        Node::Let {
            name,
            mutable,
            ann,
            value,
            ..
        } => {
            emit_let_inline(p, name, *mutable, ann, value);
        }
        Node::Match {
            scrutinee, arms, ..
        } => {
            emit_match_inline(p, scrutinee, arms);
        }
        Node::Print { args, .. } => {
            p.push("print(");
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_expr(p, a);
            }
            p.push(")");
        }
        Node::Return { value, .. } => {
            p.push("return");
            if let Some(v) = value {
                p.push(" ");
                emit_expr(p, v);
            }
        }
        Node::Assert { cond, msg, .. } => {
            p.push("assert ");
            emit_expr(p, cond);
            if let Some(m) = msg {
                p.push(", \"");
                p.push(m);
                p.push("\"");
            }
        }
        // Items that shouldn't appear as expressions; emit placeholder
        Node::FnDef { name, .. } => p.push(name),
        Node::StructDef { name, .. } => p.push(name),
        Node::EnumDef { name, .. } => p.push(name),
        Node::Const { name, .. } => p.push(name),
        Node::TypeAlias { name, .. } => p.push(name),
        Node::Import { path, .. } => p.push(&path.join(".")),
        Node::Export { names, .. } => p.push(&names.join(", ")),
        Node::Assign { name, value, .. } => {
            p.push(name);
            p.push(" = ");
            emit_expr(p, value);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            emit_expr(p, receiver);
            p.push("[");
            emit_expr(p, index);
            p.push("] = ");
            emit_expr(p, value);
        }
        Node::FieldAssign {
            receiver,
            field,
            value,
            ..
        } => {
            emit_expr(p, receiver);
            p.push(".");
            p.push(field);
            p.push(" = ");
            emit_expr(p, value);
        }
        Node::For {
            var,
            start,
            end,
            body,
            span,
        } => {
            emit_for_inline(p, var, start, end, body, *span);
        }
        Node::ForEach {
            var,
            collection,
            body,
            span,
        } => {
            emit_foreach_inline(p, var, collection, body, *span);
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, span } => {
            emit_while_inline(p, cond, body, *span);
        }
        #[cfg(feature = "std-surface")]
        Node::Region { body, span } => {
            emit_region_inline(p, body, *span);
        }
        // RFC 0010 Phase A: emit the extern "C" block in canonical form.
        Node::ExternBlock { callconv, fns, .. } => {
            emit_extern_block(p, callconv, fns);
        }
    }
}

fn emit_literal(p: &mut Printer, lit: &Literal) {
    match lit {
        Literal::Int(n) => p.push(&n.to_string()),
        Literal::Float(f) => {
            // Emit with enough precision to round-trip.
            let s = format!("{f}");
            // Ensure there's always a decimal point for float literals.
            if s.contains('.') || s.contains('e') || s.contains('E') {
                p.push(&s);
            } else {
                p.push(&s);
                p.push(".0");
            }
        }
        Literal::Str(s) => {
            p.push("\"");
            p.push(s);
            p.push("\"");
        }
        Literal::Ident(name) => p.push(name),
    }
}

fn emit_binary(p: &mut Printer, op: &BinOp, left: &Node, right: &Node) {
    emit_expr(p, left);
    p.push(" ");
    p.push(binop_str(op));
    p.push(" ");
    emit_expr(p, right);
}

fn binop_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
    }
}

fn emit_logical(p: &mut Printer, op: &LogicalOp, left: &Node, right: &Node) {
    emit_expr(p, left);
    p.push(" ");
    p.push(match op {
        LogicalOp::And => "&&",
        LogicalOp::Or => "||",
    });
    p.push(" ");
    emit_expr(p, right);
}

fn emit_bitwise(p: &mut Printer, op: &BitOp, left: &Node, right: &Node) {
    emit_expr(p, left);
    p.push(" ");
    p.push(match op {
        BitOp::Or => "|",
        BitOp::And => "&",
        BitOp::Xor => "^",
        BitOp::Shl => "<<",
        BitOp::Shr => ">>",
    });
    p.push(" ");
    emit_expr(p, right);
}

fn emit_call(p: &mut Printer, callee: &str, args: &[Node]) {
    p.push(callee);
    p.push("(");
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            p.push(", ");
        }
        emit_expr(p, a);
    }
    p.push(")");
}

fn emit_struct_lit(p: &mut Printer, name: &str, fields: &[StructLitField]) {
    p.push(name);
    p.push(" { ");
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            p.push(", ");
        }
        p.push(&f.name);
        p.push(": ");
        emit_expr(p, &f.value);
    }
    p.push(" }");
}

// ---------------------------------------------------------------------------
// Type annotation emitter
// ---------------------------------------------------------------------------

fn emit_type_ann(p: &mut Printer, ty: &TypeAnn) {
    match ty {
        TypeAnn::ScalarI32 => p.push("i32"),
        TypeAnn::ScalarI64 => p.push("i64"),
        TypeAnn::ScalarF32 => p.push("f32"),
        TypeAnn::ScalarF64 => p.push("f64"),
        TypeAnn::ScalarBool => p.push("bool"),
        TypeAnn::ScalarU32 => p.push("u32"),
        TypeAnn::Named(name) => p.push(name),
        TypeAnn::Tensor { dtype, dims } => {
            p.push("tensor<");
            p.push(dtype);
            p.push("[");
            p.push(&dims.join(", "));
            p.push("]>");
        }
        TypeAnn::DiffTensor { dtype, dims } => {
            p.push("diff tensor<");
            p.push(dtype);
            p.push("[");
            p.push(&dims.join(", "));
            p.push("]>");
        }
        TypeAnn::Slice { mutable, element } => {
            if *mutable {
                p.push("&mut [");
            } else {
                p.push("&[");
            }
            emit_type_ann(p, element);
            p.push("]");
        }
        TypeAnn::Array { element, length } => {
            p.push("[");
            emit_type_ann(p, element);
            p.push("; ");
            p.push(&length.to_string());
            p.push("]");
        }
        TypeAnn::Ref { mutable, target } => {
            if *mutable {
                p.push("&mut ");
            } else {
                p.push("&");
            }
            emit_type_ann(p, target);
        }
        TypeAnn::Generic { name, args } => {
            p.push(name);
            p.push("<");
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_type_ann(p, a);
            }
            p.push(">");
        }
        TypeAnn::Tuple { elements } => {
            p.push("(");
            for (i, e) in elements.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_type_ann(p, e);
            }
            p.push(")");
        }
        TypeAnn::SparseTensor {
            layout,
            element,
            shape,
        } => {
            let layout_str = match layout {
                SparseLayout::Csr => "csr",
                SparseLayout::Csc => "csc",
                SparseLayout::Coo => "coo",
                SparseLayout::Bsr => "bsr",
            };
            p.push("tensor<sparse[");
            p.push(layout_str);
            p.push("], ");
            emit_type_ann(p, element);
            if !shape.is_empty() {
                p.push(", [");
                let parts: Vec<String> = shape
                    .iter()
                    .map(|d| match d {
                        crate::types::ShapeDim::Known(n) => n.to_string(),
                        crate::types::ShapeDim::Sym(s) => s.to_string(),
                    })
                    .collect();
                p.push(&parts.join(", "));
                p.push("]");
            }
            p.push(">");
        }
        // RFC 0010 Phase A: raw pointer type `*const T` / `*mut T`.
        TypeAnn::RawPtr { mutable, pointee } => {
            if *mutable {
                p.push("*mut ");
            } else {
                p.push("*const ");
            }
            emit_type_ann(p, pointee);
        }
        // RFC 0010 Phase B: callback function pointer `extern "C" fn(T) -> R`.
        TypeAnn::FnPtr { params, ret } => {
            p.push("extern \"C\" fn(");
            for (i, param) in params.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_type_ann(p, param);
            }
            p.push(")");
            if let Some(r) = ret {
                p.push(" -> ");
                emit_type_ann(p, r);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RFC 0010 Phase A: extern "C" block emitter
// ---------------------------------------------------------------------------

fn emit_extern_block(p: &mut Printer, callconv: &CallConv, fns: &[ExternFn]) {
    let ind = p.indent_str();
    p.push(&ind);
    p.push("extern \"C\"");
    match callconv {
        CallConv::C => {}
        CallConv::SysV => p.push(" callconv(.sysv)"),
        CallConv::Win64 => p.push(" callconv(.win64)"),
        CallConv::Aapcs => p.push(" callconv(.aapcs)"),
    }
    p.push(" {\n");
    p.indent += 1;
    for efn in fns {
        let inner_ind = p.indent_str();
        p.push(&inner_ind);
        p.push(if efn.is_unsafe {
            "unsafe fn "
        } else {
            "safe fn "
        });
        p.push(&efn.name);
        p.push("(");
        for (i, param) in efn.params.iter().enumerate() {
            if i > 0 {
                p.push(", ");
            }
            p.push(&param.name);
            p.push(": ");
            emit_type_ann(p, &param.ty);
        }
        if efn.is_varargs {
            if !efn.params.is_empty() {
                p.push(", ");
            }
            p.push("...");
        }
        p.push(")");
        if let Some(ret) = &efn.ret_type {
            p.push(" -> ");
            emit_type_ann(p, ret);
        }
        p.push(";\n");
    }
    p.indent -= 1;
    let close_ind = p.indent_str();
    p.push(&close_ind);
    p.push("}");
}

// ---------------------------------------------------------------------------
// Pattern emitter (for match arms)
// ---------------------------------------------------------------------------

fn emit_pattern(p: &mut Printer, pat: &Pattern) {
    match pat {
        Pattern::Wildcard => p.push("_"),
        Pattern::Ident(name) => p.push(name),
        Pattern::Literal(lit) => emit_literal(p, lit),
        Pattern::EnumVariant { path, args } => {
            p.push(path);
            if !args.is_empty() {
                p.push("(");
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        p.push(", ");
                    }
                    emit_pattern(p, a);
                }
                p.push(")");
            }
        }
        Pattern::Tuple(elems) => {
            p.push("(");
            for (i, e) in elems.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                emit_pattern(p, e);
            }
            p.push(")");
        }
        Pattern::EnumStruct { path, fields } => {
            p.push(path);
            p.push(" { ");
            for (i, (fname, sub)) in fields.iter().enumerate() {
                if i > 0 {
                    p.push(", ");
                }
                p.push(fname);
                // Emit the shorthand `{ f }` when the sub-pattern just binds the
                // field's own name; otherwise `{ f: <pat> }`. Keeps the formatted
                // text stable (round-trips to the same AST).
                match sub {
                    Pattern::Ident(n) if n == fname => {}
                    _ => {
                        p.push(": ");
                        emit_pattern(p, sub);
                    }
                }
            }
            p.push(" }");
        }
    }
}
