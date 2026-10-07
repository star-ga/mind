// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Surface spelling the parser's desugars would otherwise throw away.

use super::{Attribute, Span};

/// Surface spelling the parser's desugars would otherwise throw away.
///
/// The parser performs semantic desugars and discards how the source was written; the
/// FORMATTER round-trips through this same AST, so every discarded spelling became text
/// `mindc fmt` deleted:
///
///   1. `true` / `false` parsed to `Literal::Int(1)` / `Int(0)` — `bool_literals` below.
///   2. `module NAME { … }` dropped its name and attributes — `module_headers` and
///      `module_attrs` below.
///   3. `use X as Y` never consumed `as Y` — fixed with `Node::Import::alias`.
///   4. `q.fn(args)` / `q.CONST` on an imported module desugared to `fn(args)` / `CONST`,
///      dropping `q` — `call_qualifiers` below.
///
/// The identifier-loss guard in `fmt` catches each of these and refuses to write rather than
/// corrupt the file, so the visible symptom was always the same: a source that can never clear
/// `fmt::drift`, and therefore can never pass `mindc check` at all.
///
/// These tables are keyed by SPAN and carry spelling ONLY. Nothing in the compiler reads them,
/// so the AST the type checker and lowering see is byte-for-byte what it was.
///
/// THE RULE for the next spelling that needs preserving:
///
/// * A new FIELD on an existing node is safe — it is additive, and every existing match arm
///   keeps compiling and keeps meaning what it meant. `Import::alias` and `Export::category`
///   are that shape.
/// * A new VARIANT in an enum that consumers match is NOT safe. Exhaustive matches fail loudly
///   (fine), but every `_ =>` and every `if let` that tested only the old variant fails
///   SILENTLY. A `Literal::Bool` variant was tried first for (1): `mindc check` returned 0
///   while `build` panicked ("no IR lowering for `Lit` in value position"), and constant
///   folding, comptime evaluation, SCEV and autodiff stopped treating a bool as a constant,
///   which changes the mic@3 digest of any program containing `true`. Spelling that would
///   need a new variant belongs HERE instead.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceSpelling {
    /// `(span of the desugared literal, the spelling)` per `true` / `false` source literal.
    ///
    /// The parser emits `Literal::Int(1)` / `Int(0)` for these, so every value consumer sees
    /// exactly the integer it always saw, and the printer recovers the spelling from here.
    ///
    /// KNOWN GAP: a bool in a MATCH PATTERN (`match b { true => … }`) is still desugared to
    /// `Pattern::Literal(Literal::Int(1))` and still prints as `1`, because `Pattern::Literal`
    /// carries no span to key on. deferred: no known source writes a bool pattern today.
    /// Upgrade path: add a `Span` to `Pattern::Literal` (or key on the enclosing
    /// `MatchArm::span` for the single-pattern case), record it alongside this table, and add
    /// the round-trip case to `fmt_bool_literal_roundtrip.rs`.
    pub bool_literals: Vec<(Span, bool)>,
    /// `(span of the transparent Block, dotted module path)` per `module NAME { … }` header.
    pub module_headers: Vec<(Span, String)>,
    /// `(span of the transparent Block, attributes)` per ATTRIBUTED `module NAME { … }` header,
    /// so `#[protection] module m { … }` keeps its attribute through the formatter.
    pub module_attrs: Vec<(Span, Vec<Attribute>)>,
    /// `(span of the desugared node, receiver as written)` per module-qualified reference —
    /// both `q.fn(args)` calls and `q.CONST` value reads, which desugar to a bare `Call` and a
    /// bare `Ident` respectively and lose `q` either way.
    pub call_qualifiers: Vec<(Span, String)>,
    /// Spans of the `assert` statements written `assert(cond, "msg")`: the condition and the
    /// message inside one pair of parentheses, which the parser splits into both fields.
    pub paren_asserts: Vec<Span>,
    /// Spans in `call_qualifiers` whose qualifier was written with `::` (`m::f(x)`) rather
    /// than `.`.
    pub path_qualifiers: Vec<Span>,
}
