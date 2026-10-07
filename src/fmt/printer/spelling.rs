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

//! Re-emitting the surface spelling the parser's desugars drop.
//!
//! The parser records what it discards in [`crate::ast::SourceSpelling`]; this module reads it
//! back so `mindc fmt` prints `true`, `q.f(x)`, `module m { … }`, `import X as Y` and
//! `export fn a` instead of deleting the words the AST no longer carries.

use std::collections::{HashMap, HashSet};

use super::{Printer, emit_attrs, emit_block_stmts, emit_expr};
use crate::ast::{Module, Node, Span};

/// The span-keyed spelling tables, indexed for lookup while printing.
///
/// Keyed on `(start, end)` rather than `start` alone because a desugared call and its callee
/// can share a start offset.
#[derive(Default)]
pub(super) struct SpellingTables {
    bool_literals: HashMap<(usize, usize), bool>,
    qualifiers: HashMap<(usize, usize), String>,
    paren_asserts: HashSet<(usize, usize)>,
    path_qualifiers: HashSet<(usize, usize)>,
}

impl SpellingTables {
    pub(super) fn from_module(module: &Module) -> Self {
        let key = |sp: &Span| (sp.start(), sp.end());
        let s = &module.spelling;
        Self {
            bool_literals: s
                .bool_literals
                .iter()
                .map(|(sp, b)| (key(sp), *b))
                .collect(),
            qualifiers: s
                .call_qualifiers
                .iter()
                .map(|(sp, q)| (key(sp), q.clone()))
                .collect(),
            paren_asserts: s.paren_asserts.iter().map(key).collect(),
            path_qualifiers: s.path_qualifiers.iter().map(key).collect(),
        }
    }
}

/// Emit `assert cond, "msg"`, or `assert (cond, "msg")` for an assert the source wrote with
/// its condition and message in one pair of parentheses (the spelling the formatter has
/// always printed for that form).
pub(super) fn emit_assert(p: &mut Printer, cond: &Node, msg: Option<&str>, span: &Span) {
    let paren = p
        .spelling
        .paren_asserts
        .contains(&(span.start(), span.end()));
    p.push(if paren { "assert (" } else { "assert " });
    emit_expr(p, cond);
    if let Some(m) = msg {
        p.push(", \"");
        p.push(m);
        p.push("\"");
    }
    if paren {
        p.push(")");
    }
}

/// Emit `true` / `false` for a literal the source wrote as a bool. Returns `false` when the
/// literal at `span` was not a bool, so the caller prints it as usual.
pub(super) fn emit_bool_spelling(p: &mut Printer, span: &Span) -> bool {
    match p.spelling.bool_literals.get(&(span.start(), span.end())) {
        Some(spelled) => {
            p.push(if *spelled { "true" } else { "false" });
            true
        }
        None => false,
    }
}

/// Emit `q.` (or `q::`, as written) before a node whose module qualifier the parser
/// desugared away. The desugared node is indistinguishable from an unqualified one, so the
/// span lookup is the only record.
pub(super) fn emit_qualifier(p: &mut Printer, span: &Span) {
    let key = (span.start(), span.end());
    if let Some(q) = p.spelling.qualifiers.get(&key).cloned() {
        p.push(&q);
        p.push(if p.spelling.path_qualifiers.contains(&key) {
            "::"
        } else {
            "."
        });
    }
}

/// Emit a top-level `module NAME { … }` wrapper around the transparent block it parsed to,
/// with its attributes. Returns `false` when `item` is not a recorded module block.
///
/// Without this the formatter deleted the wrapper, the identifier-loss guard refused the
/// write, and no module-wrapped file could ever clear `fmt::drift`.
pub(super) fn emit_module_wrapper(p: &mut Printer, module: &Module, item: &Node) -> bool {
    let Node::Block { stmts, span } = item else {
        return false;
    };
    let spelling = &module.spelling;
    let Some((_, path)) = spelling
        .module_headers
        .iter()
        .find(|(sp, _)| sp.start() == span.start())
    else {
        return false;
    };
    // Attributes come before the header, at the same indentation.
    if let Some((_, attrs)) = spelling
        .module_attrs
        .iter()
        .find(|(sp, _)| sp.start() == span.start())
    {
        emit_attrs(p, attrs);
    }
    p.push(&format!("module {path} {{\n"));
    p.indent += 1;
    let close_line = p.stripped_idx.line_of(span.end());
    emit_block_stmts(p, stmts, close_line);
    p.indent -= 1;
    // The body emitter may or may not leave a trailing newline depending on whether the block
    // was empty; normalise so `}` always starts its own line and an empty `module X { }` does
    // not collapse into `module X {}`.
    if !p.out.ends_with('\n') {
        p.push("\n");
    }
    p.push("}");
    true
}

/// Emit an `import` or `export` item, indented.
pub(super) fn emit_import_export(p: &mut Printer, node: &Node) {
    let ind = p.indent_str();
    p.push(&ind);
    match node {
        Node::Import { path, alias, .. } => {
            p.push("import ");
            push_path_alias(p, path, alias.as_deref());
            p.push(";");
        }
        Node::Export {
            names, category, ..
        } => {
            p.push("export ");
            // `export fn a, b` is the CATEGORY form and has no braces; `export { a, b }` is the
            // block form. Printing the block form for a categorised export would delete the
            // category word.
            match category {
                Some(kw) => {
                    p.push(kw);
                    p.push(" ");
                    p.push(&names.join(", "));
                }
                None => {
                    p.push("{ ");
                    p.push(&names.join(", "));
                    p.push(" }");
                }
            }
        }
        _ => {}
    }
}

/// An `import` or `export` in expression position: the path or name list without the keyword,
/// still carrying the alias or category word.
pub(super) fn push_import_export_expr(p: &mut Printer, node: &Node) {
    match node {
        Node::Import { path, alias, .. } => push_path_alias(p, path, alias.as_deref()),
        Node::Export {
            names, category, ..
        } => {
            if let Some(kw) = category {
                p.push(kw);
                p.push(" ");
            }
            p.push(&names.join(", "));
        }
        _ => {}
    }
}

/// `a.b` or `a.b as c`. The alias is re-emitted, not canonicalised away: it is the name every
/// call site in the file uses, so dropping it changes what the source means rather than how it
/// looks. (The `use` -> `import` keyword swap IS a deliberate canonicalisation, and is the one
/// rewrite `CANONICAL_KEYWORD_REWRITES` permits the loss guard to accept.)
fn push_path_alias(p: &mut Printer, path: &[String], alias: Option<&str>) {
    p.push(&path.join("."));
    if let Some(name) = alias {
        p.push(" as ");
        p.push(name);
    }
}
