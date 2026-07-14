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

use crate::ast::{
    BinOp, CallConv, ExternFn, Literal, MatchArm, Module, Node, Param, Pattern, Span, TensorElemOp,
    TypeAnn,
};
use crate::diagnostics::{Diagnostic as PrettyDiagnostic, Span as DiagnosticSpan};
use crate::types::ConvPadding;

mod expand_bimap;
mod trivia;
pub use trivia::{Trivia, TriviaKind, TriviaStream};
use trivia::{TriviaCollector, strip_comments_with_trivia};

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
    /// Module names brought into scope by `import X` / `use X` (the last path
    /// segment). A method call whose receiver is one of these is a MODULE-
    /// QUALIFIED call `mod.fn(args)` — desugared to the bare cross-module call
    /// `fn(args)` (all module functions share one global link unit), not a UFCS
    /// method on a value. Imports precede fn bodies, so the set is complete by
    /// the time a body parses. A `Vec` (not a `HashSet`) avoids per-parse
    /// `RandomState` hasher init — import lists are tiny, so linear `contains`
    /// is faster and keeps `compile_small` at its nanosecond floor.
    imports: Vec<String>,
    /// Invariant block names declared in this module (`invariant NAME { … }`).
    /// A dotted call `NAME.pred(args)` whose receiver is one of these is a
    /// namespace access onto the predicate free function `NAME_pred`, not a
    /// UFCS method on a value — mirrors the `imports` rewrite. Invariant blocks
    /// precede the fn bodies that call into them, so the set is complete in
    /// time. A small `Vec` keeps linear `contains` at its nanosecond floor.
    invariants: Vec<String>,
    /// Enum type names declared in this module. A dotted reference whose first
    /// segment is one of these is a VARIANT access `Enum.Variant`, normalised to
    /// the canonical `Enum::Variant` — both as a value (`Color.Red`) and in a
    /// pattern (`Color.Red => …`). A receiver that is NOT a declared enum stays a
    /// struct field access. Enum declarations precede the fn bodies that use
    /// them, so the set is complete in time.
    enum_names: Vec<String>,
    /// Whole-project enum type names, captured ONCE at construction. This is an
    /// EXPLICIT snapshot of the cross-module registry, NOT a live read: once a
    /// `P` exists, enum-name resolution is a pure function of `(enum_names,
    /// global_enums)` and is independent of any later mutation of the ambient
    /// thread-local. `P::new` captures the current registry (the back-compat
    /// default the project builder relies on via its set-before / clear-after
    /// discipline); `P::new_with_enums` lets a caller supply the registry
    /// explicitly, making parsing a pure function of `(source, registry)`.
    /// Empty on the single-file / no-project path, so those parses pay zero
    /// per-dotted-ident registry cost and stay byte-identical.
    global_enums: Vec<String>,
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
    /// RFC 0012 Phase B: `A @ B` tensor matmul. Precedence: between
    /// elementwise-mul/div and elementwise-add/sub (tighter than add,
    /// same group as ordinary multiplication).
    TensorMatmul,
    /// RFC 0012 Phase B: `A .+ B`, `A .- B`, `A .* B`, `A ./ B`
    /// elementwise tensor operators.
    TensorElem(TensorElemOp),
}

/// A compound-assignment operator (`+= -= *= /= %= &= |= ^= <<= >>=`). These
/// are NOT infix binary operators: `peek_binop` refuses to bind an `OP=` shape
/// so the Pratt parse stops at the LHS, and `parse_stmt` desugars
/// `lhs OP= rhs` -> `lhs = lhs OP rhs` (zero new IR — mirrors how match and the
/// tensor operators desugar at parse time). Arithmetic ops build a
/// `Node::Binary`, bitwise ops a `Node::Bitwise`.
#[derive(Debug, Clone, Copy)]
enum CompoundOp {
    Arith(BinOp),
    Bit(crate::ast::BitOp),
}

/// The closed set of statement-leading keywords `parse_stmt` dispatches on.
///
/// The set is fixed at compile time and every member is spelled here exactly
/// once — the recogniser below is the single place a statement keyword is
/// matched, replacing the old sequential `at_keyword(b"…")` ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StmtKw {
    Pub,
    Import,
    Use,
    Const,
    Type,
    Module,
    Export,
    Invariant,
    Struct,
    Enum,
    Fn,
    Extern,
    Assert,
    Return,
    For,
    Print,
    Let,
    If,
    While,
    Loop,
    Region,
    Break,
    Continue,
}

/// Recognise a statement-leading keyword from the identifier run at the cursor.
///
/// This is a **compile-time perfect-hash keyword recogniser** in the classic
/// (gperf) shape: the discriminator `(len, word[0])` maps the 23-keyword set to
/// at most TWO candidates — `(6, b'e')` = {export, extern} and `(6, b'r')` =
/// {region, return} are the only pairs; every other `(len, byte0)` cell holds a
/// single candidate. The candidate is then confirmed with ONE full-slice
/// equality. Worst case: one integer switch + two byte compares + one `memcmp`;
/// the old ladder averaged ~12 failed byte-prefix probes before a hit and paid
/// the FULL ~23 probes for the common case (a plain expression / assignment
/// statement, which matches no keyword at all) — that case now costs a single
/// switch miss.
///
/// **Determinism.** The discriminator is a fixed structural function of the key
/// — `(slice length, first byte, third byte)` — with NO seed, NO search, NO
/// randomness, and no dependence on host `HashMap` iteration order, pointer
/// values, time, or `-march`. It is textually pinned in this source, so it is
/// byte-for-byte the same decision on x86 and ARM. It also emits no IR: it only
/// picks which existing `parse_*` routine runs, exactly as the ladder did.
///
/// The trailing full-slice compare is **load-bearing, not belt-and-braces**: the
/// discriminator alone is not injective over arbitrary identifiers (e.g. the
/// ident `expand` lands in the `(6, b'e')` cell). The word is the whole
/// identifier run, so its end is precisely the `at_keyword` word boundary and a
/// prefix such as `letx` can never be mistaken for `let`.
#[inline]
fn stmt_keyword(w: &[u8]) -> Option<StmtKw> {
    // `(len, first byte)` — one dense integer switch. `w` is never empty when a
    // candidate exists, so `w[0]` is in bounds inside every arm.
    let (cand, kw): (&[u8], StmtKw) = match (w.len(), *w.first()?) {
        (2, b'f') => (b"fn", StmtKw::Fn),
        (2, b'i') => (b"if", StmtKw::If),
        (3, b'f') => (b"for", StmtKw::For),
        (3, b'l') => (b"let", StmtKw::Let),
        (3, b'p') => (b"pub", StmtKw::Pub),
        (3, b'u') => (b"use", StmtKw::Use),
        (4, b'e') => (b"enum", StmtKw::Enum),
        (4, b'l') => (b"loop", StmtKw::Loop),
        (4, b't') => (b"type", StmtKw::Type),
        (5, b'b') => (b"break", StmtKw::Break),
        (5, b'c') => (b"const", StmtKw::Const),
        (5, b'p') => (b"print", StmtKw::Print),
        (5, b'w') => (b"while", StmtKw::While),
        (6, b'a') => (b"assert", StmtKw::Assert),
        // The only two ambiguous cells; `w[2]` separates them (`export`/`extern`
        // agree on bytes 0..2, differ at 2 — `p` vs `t`).
        (6, b'e') => {
            if w[2] == b'p' {
                (b"export", StmtKw::Export)
            } else {
                (b"extern", StmtKw::Extern)
            }
        }
        (6, b'i') => (b"import", StmtKw::Import),
        (6, b'm') => (b"module", StmtKw::Module),
        (6, b'r') => {
            if w[2] == b'g' {
                (b"region", StmtKw::Region)
            } else {
                (b"return", StmtKw::Return)
            }
        }
        (6, b's') => (b"struct", StmtKw::Struct),
        (8, b'c') => (b"continue", StmtKw::Continue),
        (9, b'i') => (b"invariant", StmtKw::Invariant),
        _ => return None,
    };
    if w == cand { Some(kw) } else { None }
}

impl<'a> P<'a> {
    /// Construct a parser, capturing the active cross-module enum registry as an
    /// explicit snapshot. The snapshot is taken ONCE here (not re-read per
    /// dotted-ident), so parsing is order-independent w.r.t. any later registry
    /// mutation. The project builder's set-before / clear-after discipline means
    /// the snapshot reflects exactly the project's enums during a build, and is
    /// empty for single-file parses.
    fn new(src: &'a str) -> Self {
        let global_enums = crate::ir::with_global_enums(|g| g.names.clone());
        Self::new_with_enums(src, global_enums)
    }

    /// Construct a parser with an EXPLICIT enum registry, making the parse a pure
    /// function of `(source, global_enums)` with no ambient thread-local read.
    /// Pass `Vec::new()` for a single-file / no-cross-module parse.
    fn new_with_enums(src: &'a str, global_enums: Vec<String>) -> Self {
        Self {
            b: src.as_bytes(),
            pos: 0,
            imports: Vec::new(),
            invariants: Vec::new(),
            enum_names: Vec::new(),
            global_enums,
        }
    }

    /// Is `name` a declared enum type — either in THIS module (`enum_names`) or
    /// anywhere in the active project (the global registry the project builder
    /// populates for cross-module enums)? A dotted `name.Variant` reference is
    /// then normalised to `name::Variant`. Outside a project the global registry
    /// is empty, so this reduces to the same-module check and single-file parses
    /// stay byte-identical.
    #[inline]
    fn is_enum_name(&self, name: &str) -> bool {
        if self.enum_names.iter().any(|n| n == name) {
            return true;
        }
        // Pure check against the registry snapshot captured at construction — no
        // live thread-local borrow, so resolution can't drift if the ambient
        // registry is mutated after this parser was built. Empty on the
        // single-file path, so that case short-circuits to `false` immediately.
        self.global_enums.iter().any(|n| n == name)
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

    /// Render the token starting at the cursor as a short human string, for
    /// diagnostics. An identifier reads to its word boundary; any other token
    /// renders as its leading UTF-8 scalar (`@`, `%`, …), falling back to a
    /// `\xNN` byte escape for invalid UTF-8. Never advances the cursor.
    fn render_cur_token(&self) -> String {
        if self.pos >= self.b.len() {
            return "<eof>".to_string();
        }
        let start = self.pos;
        let c = self.b[start];
        if Self::is_ident_start(c) {
            let mut end = start;
            while end < self.b.len() && Self::is_ident_cont(self.b[end]) {
                end += 1;
            }
            return String::from_utf8_lossy(&self.b[start..end]).into_owned();
        }
        let take = (self.b.len() - start).min(4);
        for len in 1..=take {
            if let Ok(s) = std::str::from_utf8(&self.b[start..start + len]) {
                if let Some(ch) = s.chars().next() {
                    return ch.to_string();
                }
            }
        }
        format!("\\x{c:02x}")
    }

    /// Error for a token that cannot begin an expression, reported AT the
    /// offending token's own position (fixes #200: the old recursive-descent
    /// chain unwound and re-reported at an earlier checkpoint; the Pratt parser
    /// keeps `self.pos` on the real site, and this names the token too).
    fn unexpected_prefix_err(&self) -> ParseError {
        let tok = self.render_cur_token();
        self.err(format!("unexpected `{tok}`, expected an expression"))
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

    /// Issue #205: at the current position (immediately after an integer
    /// literal's digits), try to consume a trailing integer-type suffix
    /// (`u8`/`u16`/`u32`/`u64`/`i8`/`i16`/`i32`/`i64`) so `2u32`, `-1i32`,
    /// `0xFFu8` and friends parse in any expression position. The suffix must
    /// end at a word boundary (no trailing ident-cont byte) so `2u32x` is not
    /// silently split. On a match the position is advanced past the suffix and
    /// the corresponding `TypeAnn` is returned; otherwise the position is left
    /// untouched and `None` is returned. The literal is desugared by the caller
    /// into `Node::As`, reusing the existing `expr as type` typecheck/codegen
    /// path exactly — no new IR is introduced, so suffix-free sources (e.g. the
    /// keystone) are byte-identical.
    fn int_type_suffix(&mut self) -> Option<TypeAnn> {
        // Fast path: an integer type suffix can only begin with `u` or `i`. For
        // the overwhelmingly common UNSUFFIXED literal the next byte is
        // whitespace, an operator, `)`, `,`, `;`, … — bail before the 8-way
        // compare so unsuffixed literals (incl. the entire keystone) pay ~one
        // byte check, keeping compile_small at the nanosecond floor.
        if self.pos >= self.b.len() || !matches!(self.b[self.pos], b'u' | b'i') {
            return None;
        }
        // Each candidate suffix and the `TypeAnn` it maps to. `u32`/`i32`/`i64`
        // have dedicated scalar variants; the remaining widths ride through the
        // `Named` path (same as writing `as u64`).
        const SUFFIXES: &[&str] = &["u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64"];
        for lit in SUFFIXES {
            let bytes = lit.as_bytes();
            let end = self.pos + bytes.len();
            if end <= self.b.len()
                && &self.b[self.pos..end] == bytes
                && (end >= self.b.len() || !Self::is_ident_cont(self.b[end]))
            {
                self.pos = end;
                return Some(match *lit {
                    "u32" => TypeAnn::ScalarU32,
                    "i32" => TypeAnn::ScalarI32,
                    "i64" => TypeAnn::ScalarI64,
                    other => TypeAnn::Named(other.to_string()),
                });
            }
        }
        None
    }

    /// Read a dotted identifier like `tensor.matmul` or `foo.bar.baz`.
    fn dotted_ident(&mut self) -> Option<String> {
        let first = self.word()?;
        let mut name = first.to_string();
        loop {
            if self.pos < self.b.len() && self.b[self.pos] == b'.' {
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
                continue;
            }
            // Phase 10.6: accept `Type::Variant` path segments
            // (`config.AddressingMode::Content`, `Side::Left`, etc.).
            // We accumulate the `::Variant` into the identifier so the
            // expression node carries the full path string; the type
            // checker resolves it later.
            if self.pos + 1 < self.b.len()
                && self.b[self.pos] == b':'
                && self.b[self.pos + 1] == b':'
            {
                let saved = self.pos;
                self.pos += 2;
                match self.word() {
                    Some(part) => {
                        name.push_str("::");
                        name.push_str(part);
                    }
                    None => {
                        self.pos = saved;
                        break;
                    }
                }
                continue;
            }
            break;
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

    /// Parse a decimal digit string as an i64 literal.
    ///
    /// Values in `[0, i64::MAX]` are returned as-is.
    /// Values in `(i64::MAX, u64::MAX]` are reinterpreted as signed i64 via
    /// the standard Rust `u64 as i64` bit-pattern cast (two's complement), so
    /// `14695981039346656037` (FNV-1a 64-bit offset basis) becomes
    /// `-3750762994362895579i64` with the same byte representation.
    /// Values exceeding `u64::MAX` produce an `integer overflow` error.
    fn parse_i64_literal(&self, d: &str) -> Result<i64, ParseError> {
        if let Ok(v) = d.parse::<i64>() {
            return Ok(v);
        }
        d.parse::<u64>()
            .map(|u| u as i64)
            .map_err(|_| self.err("integer overflow".into()))
    }

    /// Same as `parse_i64_literal` but for pattern-match positions; the
    /// error message reads "integer overflow in pattern" so diagnostics remain
    /// consistent with the pre-existing wording at those sites.
    fn parse_i64_pattern(&self, d: &str) -> Result<i64, ParseError> {
        if let Ok(v) = d.parse::<i64>() {
            return Ok(v);
        }
        d.parse::<u64>()
            .map(|u| u as i64)
            .map_err(|_| self.err("integer overflow in pattern".into()))
    }

    /// Read a numeric literal token for use inside attribute argument lists.
    /// Accepts optional leading `-`, digits, optional `.` + digits.
    /// Returns the raw text as a string, or `None` if nothing numeric is here.
    fn read_attr_literal(&mut self) -> Option<String> {
        let start = self.pos;
        // Optional leading minus
        let neg = self.at(b'-');
        if neg {
            self.pos += 1;
        }
        // Must have at least one digit
        let digit_start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == digit_start {
            self.pos = start;
            return None;
        }
        // Optional fractional part
        if self.pos < self.b.len() && self.b[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
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

    /// The identifier run at the cursor, WITHOUT advancing. Empty when the cursor
    /// is not on an identifier start (an operator, a literal, EOF). The word ends
    /// at the first non-`is_ident_cont` byte, which is exactly the boundary
    /// `at_keyword` tests — so `stmt_keyword(self.cur_word())` is semantically
    /// identical to a chain of `at_keyword(b"…")` probes, but scans the run once.
    #[inline]
    fn cur_word(&self) -> &'a [u8] {
        let start = self.pos;
        if start >= self.b.len() || !Self::is_ident_start(self.b[start]) {
            return &[];
        }
        let mut end = start;
        while end < self.b.len() && Self::is_ident_cont(self.b[end]) {
            end += 1;
        }
        &self.b[start..end]
    }

    /// Non-consuming lookahead: is `kw` the next keyword after `lead` (+ inter-
    /// vening whitespace)? Used to disambiguate `extern const …` (an extern
    /// constant) from an `extern "C" { … }` block without committing the cursor.
    fn peek_keyword_after(&self, lead: &[u8], kw: &[u8]) -> bool {
        if !self.at_keyword(lead) {
            return false;
        }
        let mut i = self.pos + lead.len();
        while i < self.b.len() && (self.b[i] == b' ' || self.b[i] == b'\t') {
            i += 1;
        }
        if !self.b[i..].starts_with(kw) {
            return false;
        }
        let after = i + kw.len();
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
            // RFC 0012 §3.2 — additional dtypes
            (b"i64", "i64"),
            (b"I64", "i64"),
            (b"q16", "q16"),
            (b"Q16", "q16"),
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
        // RFC 0010 Phase B: callback function pointer type `extern "C" fn(T, U) -> R`.
        // Accepted in `extern "C"` block parameter/return positions.
        // Lowers to opaque `!llvm.ptr` in MLIR; the parameter and return types
        // are stored for type-checking only.
        if self.at_keyword(b"extern") {
            let save = self.pos;
            self.pos += 6; // "extern"
            self.skip_ws();
            // Must be followed by `"C"` and then `fn`.
            if self.at(b'"')
                && self.pos + 2 < self.b.len()
                && self.b[self.pos + 1] == b'C'
                && self.b[self.pos + 2] == b'"'
            {
                self.pos += 3; // skip `"C"`
                self.skip_ws();
                if self.at_keyword(b"fn") {
                    self.pos += 2; // "fn"
                    self.skip_ws();
                    self.expect(b'(')?;
                    let mut params: Vec<TypeAnn> = Vec::new();
                    self.skip_ws_and_newlines();
                    if !self.at(b')') {
                        loop {
                            self.skip_ws_and_newlines();
                            if self.starts_with(b"...") {
                                self.pos += 3;
                                self.skip_ws();
                                break;
                            }
                            params.push(self.type_ann()?);
                            self.skip_ws_and_newlines();
                            if !self.eat(b',') {
                                break;
                            }
                            self.skip_ws_and_newlines();
                            if self.at(b')') {
                                break;
                            }
                        }
                    }
                    self.skip_ws_and_newlines();
                    self.expect(b')')?;
                    self.skip_ws();
                    let ret = if self.starts_with(b"->") {
                        self.pos += 2;
                        self.skip_ws();
                        Some(Box::new(self.type_ann()?))
                    } else {
                        None
                    };
                    return Ok(TypeAnn::FnPtr { params, ret });
                }
            }
            // Not a function pointer — back up and fall through.
            self.pos = save;
        }
        // RFC 0010 Phase A: raw pointer types `*const T` / `*mut T`.
        // These are valid in `extern "C"` signatures. The pointee type is
        // recorded for documentation; Phase A lowers all raw pointers to
        // opaque `!llvm.ptr` regardless of pointee.
        if self.at(b'*') {
            self.pos += 1;
            self.skip_ws();
            let mutable = if self.at_keyword(b"mut") {
                self.pos += 3;
                true
            } else if self.at_keyword(b"const") {
                self.pos += 5;
                false
            } else {
                return Err(
                    self.err("expected `const` or `mut` after `*` in raw pointer type".into())
                );
            };
            self.skip_ws();
            let pointee = self.type_ann()?;
            return Ok(TypeAnn::RawPtr {
                mutable,
                pointee: Box::new(pointee),
            });
        }
        // Phase 10.6: borrowed reference types.
        //   `&[T]`     -> Slice (sized buffer, e.g. reduce/conv inputs)
        //   `&mut [T]` -> Slice mutable (e.g. in-place normalize / SGD updates)
        //   `&T`       -> Ref (struct passed by reference)
        //   `&mut T`   -> Ref mutable
        if self.at(b'&') {
            self.pos += 1;
            self.skip_ws();
            let mutable = if self.at_keyword(b"mut") {
                self.pos += 3;
                self.skip_ws();
                true
            } else {
                false
            };
            if self.eat(b'[') {
                self.skip_ws();
                let element = self.type_ann()?;
                self.skip_ws();
                if !self.eat(b']') {
                    return Err(self.err("expected `]` to close slice type".into()));
                }
                return Ok(TypeAnn::Slice {
                    mutable,
                    element: Box::new(element),
                });
            }
            // Single-value reference: `&T` / `&mut T`. Element type recurses
            // through type_ann so qualified names (`&module.Type`) work.
            let target = self.type_ann()?;
            return Ok(TypeAnn::Ref {
                mutable,
                target: Box::new(target),
            });
        }
        // Phase 10.6: tuple type `(T, U, ...)`. Used for fns that
        // return multiple values, e.g. `fn defaults() -> (i32, u32)`.
        if self.at(b'(') {
            self.pos += 1;
            self.skip_ws_and_newlines();
            let mut elements: Vec<TypeAnn> = Vec::new();
            if !self.at(b')') {
                elements.push(self.type_ann()?);
                loop {
                    self.skip_ws_and_newlines();
                    if !self.eat(b',') {
                        break;
                    }
                    self.skip_ws_and_newlines();
                    if self.at(b')') {
                        break;
                    }
                    elements.push(self.type_ann()?);
                }
            }
            self.skip_ws_and_newlines();
            if !self.eat(b')') {
                return Err(self.err("expected `)` to close tuple type".into()));
            }
            return Ok(TypeAnn::Tuple { elements });
        }
        // Phase 10.6: fixed-size array `[T; N]`. The N is a compile-time
        // u32. Used for LUT tables and other static buffers.
        //
        // Bare `[T]` (no `; N`) is a dynamic slice: a sized run of `T` whose
        // length is not part of the type. MIND's slice type carries no
        // borrow/owned distinction — the type checker treats `[T]` and `&[T]`
        // identically as a contiguous run of `T` (see `TypeAnn::Slice`). So
        // `[T]` is accepted as sugar for `&[T]` and shares the `Slice`
        // representation; `mindc fmt` surface-canonicalises `[T]` to `&[T]`
        // (no semantic loss — the source spelling is normalised — and the
        // rewrite is idempotent). `[T; N]` (with a compile-time length) stays
        // a distinct fixed-size `Array`.
        if self.at(b'[') {
            self.pos += 1;
            self.skip_ws();
            let element = self.type_ann()?;
            self.skip_ws();
            if self.eat(b']') {
                return Ok(TypeAnn::Slice {
                    mutable: false,
                    element: Box::new(element),
                });
            }
            if !self.eat(b';') {
                return Err(self.err(
                    "expected `;` before the length in `[T; N]`, or `]` to close a dynamic slice `[T]`"
                        .into(),
                ));
            }
            self.skip_ws();
            let length_str = self
                .digits()
                .ok_or_else(|| self.err("expected array length (positive integer)".into()))?;
            let length: u32 = length_str
                .parse()
                .map_err(|_| self.err("array length out of u32 range".into()))?;
            self.skip_ws();
            if !self.eat(b']') {
                return Err(self.err("expected `]` to close array type".into()));
            }
            return Ok(TypeAnn::Array {
                element: Box::new(element),
                length,
            });
        }
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
        // tensor<sparse[layout], element_type[shape]>   — sparse tensor
        // tensor<f32[dims]>                             — dense tensor
        // tensor<f32>                                   — dense, no shape
        if self.at_keyword(b"tensor") {
            self.pos += 6;
            self.skip_ws();
            self.expect(b'<')?;
            self.skip_ws();
            // Lookahead: is the first type argument `sparse[...]`?
            if self.at_keyword(b"sparse") {
                return self.parse_sparse_tensor_type();
            }
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
            // Fast path: bare identifier, no qualifier, no generic args.
            // Keeps the common case bit-identical to the pre-Phase-10.6
            // hot loop.
            if self.pos >= self.b.len() || (self.b[self.pos] != b'.' && self.b[self.pos] != b'<') {
                // `Name[N]` — a fixed-size buffer type (e.g. `bytes[32]`,
                // `bytes[8]` for hashes), the size suffixing the name. Consume the
                // `[N]` and keep the opaque Named handle (i64 ABI); N is only
                // material at the `Name[N].zero()` value constructor (which
                // recovers it from the IndexAccess). Distinct from `[T; N]` (a
                // const array, parsed from a LEADING `[`).
                if self.pos < self.b.len() && self.b[self.pos] == b'[' {
                    let save = self.pos;
                    self.pos += 1;
                    self.skip_ws();
                    if let Some(n) = self.digits() {
                        self.skip_ws();
                        if self.eat(b']') {
                            return Ok(TypeAnn::Named(format!("{first}[{n}]")));
                        }
                    }
                    self.pos = save; // not `[N]` — leave the `[` for the caller
                }
                return Ok(TypeAnn::Named(first.to_string()));
            }
            // Path accumulation: `a.b.c` becomes a single Named("a.b.c").
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
            // Phase 10.6 generic application: `Name<A, B, ...>`. The
            // bare-Name fast-path above only hands us `<` here in type
            // position, never expression position, so there's no
            // ambiguity with the comparison operator.
            if self.pos < self.b.len() && self.b[self.pos] == b'<' {
                self.pos += 1;
                self.skip_ws_and_newlines();
                let mut args: Vec<TypeAnn> = Vec::new();
                if self.pos < self.b.len() && self.b[self.pos] != b'>' {
                    args.push(self.type_ann()?);
                    loop {
                        self.skip_ws_and_newlines();
                        if !self.eat(b',') {
                            break;
                        }
                        self.skip_ws_and_newlines();
                        if self.pos < self.b.len() && self.b[self.pos] == b'>' {
                            break;
                        }
                        args.push(self.type_ann()?);
                    }
                }
                self.skip_ws_and_newlines();
                if !self.eat(b'>') {
                    return Err(self.err("expected `>` to close generic type arguments".into()));
                }
                return Ok(TypeAnn::Generic { name, args });
            }
            // `Name[N]` — a fixed-size buffer type (e.g. `bytes[32]`, `bytes[8]`
            // for hashes), the size suffixing the name. Treated as the opaque
            // Named handle (i64 ABI); N is only material at the `Name[N].zero()`
            // value constructor, which recovers it from the IndexAccess. Distinct
            // from `[T; N]` (a const array, parsed from a LEADING `[`).
            if self.pos < self.b.len() && self.b[self.pos] == b'[' {
                let save = self.pos;
                self.pos += 1;
                self.skip_ws();
                if let Some(n) = self.digits() {
                    self.skip_ws();
                    if self.eat(b']') {
                        return Ok(TypeAnn::Named(format!("{name}[{n}]")));
                    }
                }
                self.pos = save; // not `[N]` — leave the `[` for the caller
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
        // type/const. Captured as `is_pub` and propagated into the AST node
        // so the formatter can round-trip the keyword faithfully.
        //
        // Statement dispatch: ONE `stmt_keyword` recognition of the identifier run
        // at the cursor replaces the old sequential `at_keyword(b"…")` ladder over
        // the 23 statement-leading keywords. Semantics are unchanged — the keywords
        // are pairwise distinct under the word-boundary rule, so at most one ladder
        // arm could ever fire and the ladder's ORDER was never load-bearing. The
        // recogniser is a pure, seedless structural function of the word (see
        // `stmt_keyword`): it selects which existing `parse_*` routine runs and
        // emits no IR, so every artifact stays byte-identical.
        let mut kw = stmt_keyword(self.cur_word());
        let is_pub = if kw == Some(StmtKw::Pub) {
            self.pos += 3;
            self.skip_ws_and_newlines();
            kw = stmt_keyword(self.cur_word());
            true
        } else {
            false
        };
        match kw {
            Some(StmtKw::Import) => return self.parse_import(),
            Some(StmtKw::Use) => return self.parse_use(),
            Some(StmtKw::Const) => return self.parse_const(Vec::new()),
            Some(StmtKw::Type) => return self.parse_type_alias(Vec::new()),
            Some(StmtKw::Module) => return self.parse_module_block(Vec::new()),
            Some(StmtKw::Export) => return self.parse_export_block(),
            // `invariant NAME { ... }` — a governance/DIFC contract declaration
            // (512-mind invariant system). It produces NO executable code: the
            // native compiler accepts it as a transparent marker and the governance
            // tooling (arch-mind / the 512-mind runtime) consumes the body.
            // std-surface-gated; the keystone source has no invariants, so its emit
            // stays byte-identical.
            // deferred: the `check(...)` body is currently skipped, not lowered —
            // upgrade path is to register invariant checks with the governance pass
            // so they can be verified at runtime, rather than dropped here.
            #[cfg(feature = "std-surface")]
            Some(StmtKw::Invariant) => return self.parse_invariant_block(),
            Some(StmtKw::Struct) => return self.parse_struct(Vec::new(), is_pub),
            Some(StmtKw::Enum) => return self.parse_enum(Vec::new(), is_pub),
            Some(StmtKw::Fn) => return self.parse_fn_def(is_pub),
            // RFC 0010 Phase A: `extern "C" [callconv(.x)] { ... }` block.
            // `extern const NAME: [T; N]` is a distinct form — an externally-provided
            // constant (e.g. a Q16.16 LUT table supplied by the build system), NOT an
            // `extern "C"` block. Disambiguate by peeking the word after `extern`.
            Some(StmtKw::Extern) => {
                if self.peek_keyword_after(b"extern", b"const") {
                    return self.parse_extern_const(Vec::new());
                }
                return self.parse_extern_block();
            }
            Some(StmtKw::Assert) => return self.parse_assert(),
            Some(StmtKw::Return) => return self.parse_return(),
            Some(StmtKw::For) => return self.parse_for(),
            Some(StmtKw::Print) => return self.parse_print(),
            Some(StmtKw::Let) => return self.parse_let(),
            Some(StmtKw::If) => return self.parse_if_expr(),
            // RFC 0005 Gap 1: `while` statement — gated to std-surface so the
            // default-build hot path stays byte-identical.
            #[cfg(feature = "std-surface")]
            Some(StmtKw::While) => return self.parse_while(),
            // `loop { body }` — unconditional loop, desugared to `while 1 { body }`
            // so it reuses the while machinery (break/continue, region-scoped exit
            // SSA) verbatim. Gated to std-surface like `while`.
            #[cfg(feature = "std-surface")]
            Some(StmtKw::Loop) => return self.parse_loop(),
            // RFC 0010 Phase J-A: `region { }` block — gated to std-surface so
            // the default-build hot path stays byte-identical.
            #[cfg(feature = "std-surface")]
            Some(StmtKw::Region) => return self.parse_region(),
            // Loop control: `break` / `continue`. Intercepted here, before the
            // expr/assign fallthrough, so they do not parse as `Node::Lit(Ident)`.
            #[cfg(feature = "std-surface")]
            Some(StmtKw::Break) => {
                let start = self.pos;
                self.pos += 5; // "break"
                return Ok(Node::Break {
                    span: Span::new(start, self.pos),
                });
            }
            #[cfg(feature = "std-surface")]
            Some(StmtKw::Continue) => {
                let start = self.pos;
                self.pos += 8; // "continue"
                return Ok(Node::Continue {
                    span: Span::new(start, self.pos),
                });
            }
            // `pub` has no arm here by design: the modifier was consumed above and
            // any real item keyword after it was already dispatched. A degenerate
            // second `pub` (`pub pub …`) does re-recognise as `Some(StmtKw::Pub)` and
            // reaches this arm — it then falls through to the expression/assignment
            // path exactly as the old ladder did (which had no post-modifier `pub`
            // probe). Likewise any keyword whose feature gate is off in this build,
            // and every non-keyword word, fall through here — the same behaviour the
            // ladder had when no `at_keyword` probe matched.
            _ => {}
        }
        // Expression or assignment
        let start = self.pos;
        let expr = self.parse_expr()?;
        self.skip_ws();
        // Compound assignment: `lhs OP= rhs` desugars to `lhs = lhs OP rhs`.
        // The Pratt parse stopped at the LHS (peek_binop refused the `OP=`
        // shape), so the cursor is on the operator. Same three LHS shapes as
        // plain `=`; the LHS expression is cloned to become the binop's left
        // operand. Zero new IR — the desugared node lowers like any assignment.
        if let Some((cop, width)) = self.compound_assign_op(self.pos) {
            self.pos += width;
            self.skip_ws_and_newlines();
            let rhs = self.parse_expr()?;
            let span = Span::new(start, self.pos);
            let combine = |cop: CompoundOp, left: Node, right: Node| -> Node {
                match cop {
                    CompoundOp::Arith(op) => Node::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                        span,
                    },
                    CompoundOp::Bit(op) => Node::Bitwise {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                        span,
                    },
                }
            };
            match expr {
                Node::Lit(Literal::Ident(name), lspan) => {
                    let left = Node::Lit(Literal::Ident(name.clone()), lspan);
                    return Ok(Node::Assign {
                        name,
                        value: Box::new(combine(cop, left, rhs)),
                        span,
                    });
                }
                Node::IndexAccess {
                    receiver,
                    index,
                    span: lspan,
                } => {
                    let left = Node::IndexAccess {
                        receiver: receiver.clone(),
                        index: index.clone(),
                        span: lspan,
                    };
                    return Ok(Node::IndexAssign {
                        receiver,
                        index,
                        value: Box::new(combine(cop, left, rhs)),
                        span,
                    });
                }
                Node::FieldAccess {
                    receiver,
                    field,
                    span: lspan,
                } => {
                    let left = Node::FieldAccess {
                        receiver: receiver.clone(),
                        field: field.clone(),
                        span: lspan,
                    };
                    return Ok(Node::FieldAssign {
                        receiver,
                        field,
                        value: Box::new(combine(cop, left, rhs)),
                        span,
                    });
                }
                _ => {
                    return Err(self.err(
                        "invalid compound-assignment target (expected a variable, index, or field)"
                            .to_string(),
                    ));
                }
            }
        }
        // Check for assignment. Three LHS shapes accepted:
        //   1) bare ident:  `x = expr`           -> Node::Assign
        //   2) indexed:     `xs[i] = expr`       -> Node::IndexAssign
        //   3) field:       `obj.field = expr`   -> Node::FieldAssign
        if self.at(b'=') && !self.starts_with(b"==") {
            match expr {
                Node::Lit(Literal::Ident(name), _) => {
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
                Node::IndexAccess {
                    receiver, index, ..
                } => {
                    self.advance(); // consume '='
                    self.skip_ws_and_newlines();
                    let value = self.parse_expr()?;
                    let span = Span::new(start, self.pos);
                    return Ok(Node::IndexAssign {
                        receiver,
                        index,
                        value: Box::new(value),
                        span,
                    });
                }
                Node::FieldAccess {
                    receiver, field, ..
                } => {
                    self.advance(); // consume '='
                    self.skip_ws_and_newlines();
                    let value = self.parse_expr()?;
                    let span = Span::new(start, self.pos);
                    return Ok(Node::FieldAssign {
                        receiver,
                        field,
                        value: Box::new(value),
                        span,
                    });
                }
                other => return Ok(other),
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
        // Record the qualifier (last path segment) so a later `mod.fn(args)`
        // call desugars to the bare cross-module `fn(args)`.
        if let Some(last) = path.last() {
            if !self.imports.iter().any(|s| s == last) {
                self.imports.push(last.clone());
            }
        }
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
        if let Some(last) = path.last() {
            if !self.imports.iter().any(|s| s == last) {
                self.imports.push(last.clone());
            }
        }
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
            // RFC 0012 §5: attributes are written Rust-style `#[name]`. The `#`
            // is required — it disambiguates an attribute from the `@` matmul
            // operator and from a bare `[` array literal, so MIND has exactly
            // one attribute form. A leading token that is not `#[` ends the
            // attribute list (a lone `#`, or a bare `[`, is not an attribute).
            if !(self.at(b'#') && self.b.get(self.pos + 1).copied() == Some(b'[')) {
                break;
            }
            // `save` is the `#`; the array-literal backout below restores the
            // whole `#[` so the (erroring) expression path sees it intact.
            let save = self.pos;
            self.pos += 2; // consume '#['
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
                    } else if let Some(tok) = self.read_attr_literal() {
                        // Accept numeric literals (e.g. `0.5` in
                        // `[reap_threshold(0.5)]`) as raw string tokens.
                        args.push(tok);
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
        // Optional `pub` after an attribute list: `[attr] pub fn foo() { }`.
        let is_pub = if self.at_keyword(b"pub") {
            self.pos += 3;
            self.skip_ws_and_newlines();
            true
        } else {
            false
        };
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
            return self.parse_struct(attrs, is_pub);
        }
        if self.at_keyword(b"enum") {
            return self.parse_enum(attrs, is_pub);
        }
        if self.at_keyword(b"fn") {
            // Extract `[reap_threshold(t)]` from the attribute list.
            let reap = extract_reap_threshold(&attrs);
            // Extract `[test]` (RFC 0008 Phase B).
            let is_test = extract_is_test(&attrs);
            // RFC 0012 Phase C.0: also record the raw attribute list on the
            // FnDef so later phases can interpret `[deterministic]`,
            // `[target(...)]`, and `[q16]`. `reap`/`is_test` stay as the typed
            // fast-path; `attrs` is the full record.
            return self.parse_fn_def_with_attrs(reap, is_test, is_pub, attrs);
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

    /// Parse `extern const NAME: TYPE[;]` — an externally-provided constant
    /// (typically a fixed-size Q16.16 LUT table, `extern const T: [E; N]`). The
    /// type annotation is REQUIRED (there is no `= value` to infer from); the
    /// value is supplied out-of-band by the build system. Produces a typed,
    /// resolvable name so consumers that index the table type-check.
    fn parse_extern_const(
        &mut self,
        attrs: Vec<crate::ast::Attribute>,
    ) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "extern"
        self.skip_ws();
        self.pos += 5; // "const" (guaranteed by peek_keyword_after)
        self.skip_ws();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected name after `extern const`".into()))?
            .to_string();
        self.skip_ws();
        if !self.eat(b':') {
            return Err(self.err("expected `:` and a type in `extern const` declaration".into()));
        }
        self.skip_ws();
        let ty = self.type_ann()?;
        self.skip_ws();
        self.eat(b';'); // optional trailing semicolon
        let span = Span::new(start, self.pos);
        Ok(Node::ExternConst {
            name,
            ty,
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
        self.skip_ws_and_newlines();
        // `type X { field: ty, … }` — a record type defined with the `type`
        // keyword (not `type X = Y`). Parse it as a struct definition. (Keystone
        // uses `struct`, not `type {…}`, so this branch never fires there → the
        // bootstrap fixed point is byte-identical.)
        if self.at(b'{') {
            return self.parse_struct_body(name, attrs, true, start);
        }
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
        // Consume a DOTTED module path (`module backends.tool`) — the `.segment`
        // continuation that `word()` stops at. Without this the trailing
        // `.tool` is left for the next item parse → "expected expression". The
        // path is a transparent marker (the file IS the module); we only need to
        // consume it. (Keystone has no dotted module decls → byte-identical.)
        while self.at(b'.') {
            self.pos += 1; // '.'
            self.word()
                .ok_or_else(|| self.err("expected module path segment after `.`".into()))?;
        }
        self.skip_ws_and_newlines();
        if !self.eat(b'{') {
            // File-level `module NAME` header (no block). The build path accepts
            // this — the file IS the module, the name implicit from the filename —
            // so the single-file parser (`mindc check`, `mindc test` discovery)
            // must too, or it falsely rejects files that `mindc build` compiles
            // fine. Treat it as a no-op transparent marker. (The keystone uses no
            // file-level module decls, so this branch never fires there → the
            // bootstrap fixed point is byte-identical.)
            let span = Span::new(start, self.pos);
            return Ok(Node::Block {
                stmts: Vec::new(),
                span,
            });
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

    /// Parse `invariant NAME { ... }` — a governance/DIFC contract declaration.
    ///
    /// The invariant is a 512-mind governance construct (a `description` plus one
    /// or more `check(...)` predicates). It is NOT part of the executable lowering:
    /// the native compiler accepts the whole `{ ... }` body as opaque and emits a
    /// transparent empty block, so a module that declares invariants compiles to
    /// native code while the governance tooling consumes the contract separately.
    ///
    /// The body is skipped with brace-balancing that is aware of string literals
    /// and `//` / `/* */` comments so a `}` inside a string or comment does not
    /// close the block prematurely.
    #[cfg(feature = "std-surface")]
    fn parse_invariant_block(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += "invariant".len();
        self.skip_ws();
        // Block name (`invariant evidence_per_edge { ... }`). Required so each
        // predicate becomes the callable free fn `NAME_pred`.
        let name = self
            .word()
            .ok_or_else(|| self.err("expected invariant name".into()))?
            .to_string();
        self.invariants.push(name.clone());
        self.skip_ws_and_newlines();
        if !self.eat(b'{') {
            return Err(self.err("expected `{` to open invariant block".into()));
        }
        // Each predicate `pred(params): ret { body }` lowers to a `pub fn`
        // named `NAME_pred`, enabling the `NAME.pred(args)` call site (rewritten
        // to a bare `NAME_pred(args)` call by the method-call desugar). Other
        // fields (`description: "…"`) are metadata and produce no code. Brace /
        // string / comment handling stays so unrecognised content cannot close
        // the block prematurely.
        let mut fndefs: Vec<Node> = Vec::new();
        loop {
            self.skip_invariant_trivia();
            if self.at_end() {
                return Err(self.err("unterminated invariant block".into()));
            }
            if self.eat(b'}') {
                break;
            }
            let field_start = self.pos;
            let ident = self
                .word()
                .ok_or_else(|| self.err("expected invariant field or predicate".into()))?
                .to_string();
            self.skip_ws();
            if self.at(b'(') {
                // Predicate definition `ident(params): ret { body }`.
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
                // Return type: the predicate form uses `: T`; also accept `-> T`.
                let ret_type = if self.eat(b':') {
                    self.skip_ws_and_newlines();
                    Some(self.type_ann()?)
                } else if self.starts_with(b"->") {
                    self.pos += 2;
                    self.skip_ws_and_newlines();
                    Some(self.type_ann()?)
                } else {
                    None
                };
                self.skip_ws_and_newlines();
                self.expect(b'{')?;
                let body = self.parse_fn_body_stmts()?;
                self.skip_ws_and_newlines();
                self.expect(b'}')?;
                let span = Span::new(field_start, self.pos);
                fndefs.push(Node::FnDef {
                    is_pub: true,
                    is_test: false,
                    name: format!("{name}_{ident}"),
                    type_params: Vec::new(),
                    params,
                    ret_type,
                    body,
                    reap_threshold: None,
                    attrs: Vec::new(),
                    span,
                });
            } else if self.eat(b':') {
                // Metadata field (`description: "…"`): consume one value and
                // drop it — invariant metadata produces no executable code.
                self.skip_ws_and_newlines();
                self.skip_invariant_field_value();
            } else {
                return Err(self.err(format!(
                    "unexpected token in invariant block after `{ident}`"
                )));
            }
        }
        let span = Span::new(start, self.pos);
        Ok(Node::Block {
            stmts: fndefs,
            span,
        })
    }

    /// Skip whitespace, newlines, and `//` / `/* */` comments between invariant
    /// fields.
    #[cfg(feature = "std-surface")]
    fn skip_invariant_trivia(&mut self) {
        loop {
            self.skip_ws_and_newlines();
            if self.b.get(self.pos) == Some(&b'/') && self.b.get(self.pos + 1) == Some(&b'/') {
                while !self.at_end() && self.b[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else if self.b.get(self.pos) == Some(&b'/') && self.b.get(self.pos + 1) == Some(&b'*')
            {
                self.pos += 2;
                while !(self.at_end()
                    || self.b[self.pos] == b'*' && self.b.get(self.pos + 1) == Some(&b'/'))
                {
                    self.pos += 1;
                }
                if !self.at_end() {
                    self.pos += 2;
                }
            } else {
                break;
            }
        }
    }

    /// Skip a single invariant metadata field value (string, comment-/string-
    /// aware braced group, or a bare token), producing no code.
    #[cfg(feature = "std-surface")]
    fn skip_invariant_field_value(&mut self) {
        if self.at_end() {
            return;
        }
        match self.b[self.pos] {
            b'"' => {
                self.pos += 1;
                while !self.at_end() && self.b[self.pos] != b'"' {
                    if self.b[self.pos] == b'\\' && self.pos + 1 < self.b.len() {
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
                if !self.at_end() {
                    self.pos += 1; // closing quote
                }
            }
            b'{' => {
                let mut depth: u32 = 0;
                loop {
                    if self.at_end() {
                        return;
                    }
                    match self.b[self.pos] {
                        b'"' => {
                            self.pos += 1;
                            while !self.at_end() && self.b[self.pos] != b'"' {
                                if self.b[self.pos] == b'\\' && self.pos + 1 < self.b.len() {
                                    self.pos += 2;
                                } else {
                                    self.pos += 1;
                                }
                            }
                            if !self.at_end() {
                                self.pos += 1;
                            }
                        }
                        b'/' if self.b.get(self.pos + 1) == Some(&b'/') => {
                            while !self.at_end() && self.b[self.pos] != b'\n' {
                                self.pos += 1;
                            }
                        }
                        b'/' if self.b.get(self.pos + 1) == Some(&b'*') => {
                            self.pos += 2;
                            while !(self.at_end()
                                || self.b[self.pos] == b'*'
                                    && self.b.get(self.pos + 1) == Some(&b'/'))
                            {
                                self.pos += 1;
                            }
                            if !self.at_end() {
                                self.pos += 2;
                            }
                        }
                        b'{' => {
                            depth += 1;
                            self.pos += 1;
                        }
                        b'}' => {
                            depth -= 1;
                            self.pos += 1;
                            if depth == 0 {
                                return;
                            }
                        }
                        _ => self.pos += 1,
                    }
                }
            }
            _ => {
                // Bare token to end of line (or block close).
                while !self.at_end() && self.b[self.pos] != b'\n' && self.b[self.pos] != b'}' {
                    self.pos += 1;
                }
            }
        }
    }

    /// Parse `struct NAME { field: T, field: T }`.
    fn parse_struct(
        &mut self,
        attrs: Vec<crate::ast::Attribute>,
        is_pub: bool,
    ) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "struct"
        self.skip_ws();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected struct name".into()))?
            .to_string();
        self.skip_ws_and_newlines();
        self.parse_struct_body(name, attrs, is_pub, start)
    }

    /// Parse a struct BODY `{ field: ty, … }` after the name has been read, into
    /// a `Node::StructDef`. Shared by `struct X { … }` and the `type X { … }`
    /// record-definition form (a struct defined with the `type` keyword).
    fn parse_struct_body(
        &mut self,
        name: String,
        attrs: Vec<crate::ast::Attribute>,
        is_pub: bool,
        start: usize,
    ) -> Result<Node, ParseError> {
        if !self.eat(b'{') {
            return Err(self.err("expected `{` after struct name".into()));
        }
        let mut fields = Vec::new();
        self.skip_ws_and_newlines();
        while !self.at(b'}') && !self.at_end() {
            let f_start = self.pos;
            // Optional `pub` visibility marker on the field — captured so
            // the formatter can round-trip it faithfully.
            let field_is_pub = if self.at_keyword(b"pub") {
                self.pos += 3;
                self.skip_ws();
                true
            } else {
                false
            };
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
                is_pub: field_is_pub,
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
            is_pub,
            name,
            fields,
            attrs,
            span,
        })
    }

    /// Parse `enum NAME { Variant, Variant(T), ... }`.
    fn parse_enum(
        &mut self,
        attrs: Vec<crate::ast::Attribute>,
        is_pub: bool,
    ) -> Result<Node, ParseError> {
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
            let mut field_names = Vec::new();
            if self.eat(b'(') {
                // Tuple variant `V(T, U)` — positional payload, no field names.
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
            } else if self.eat(b'{') {
                // Struct variant `V { f: T, g: U }` — named payload. The fields
                // lower to the SAME positional boxed-record slots as a tuple
                // variant (declaration order), with `field_names` recording the
                // names so construction/match can resolve each by name.
                self.skip_ws_and_newlines();
                while !self.at(b'}') && !self.at_end() {
                    let fname = self
                        .word()
                        .ok_or_else(|| self.err("expected field name in struct variant".into()))?
                        .to_string();
                    self.skip_ws();
                    if !self.eat(b':') {
                        return Err(self.err("expected `:` after struct-variant field name".into()));
                    }
                    self.skip_ws();
                    payload.push(self.type_ann()?);
                    field_names.push(fname);
                    self.skip_ws_and_newlines();
                    if self.eat(b',') {
                        self.skip_ws_and_newlines();
                    } else {
                        break;
                    }
                }
                self.skip_ws_and_newlines();
                if !self.eat(b'}') {
                    return Err(self.err("expected `}` to close struct variant".into()));
                }
                self.skip_ws();
            }
            // Optional `= discriminant`. Historically parsed-and-discarded. Now
            // additively CAPTURED: a string literal becomes `paired` (the
            // `#[bimap]` derive's single-source pair value); any other
            // expression sets `paired_is_nonstring` (the E2020 trigger under
            // `#[bimap]`). Behaviour for ordinary enums is unchanged — both
            // fields are simply ignored when the enum carries no `bimap` attr.
            let mut paired: Option<String> = None;
            let mut paired_is_nonstring = false;
            if self.eat(b'=') {
                self.skip_ws();
                if let Ok(discr) = self.parse_expr() {
                    match discr {
                        crate::ast::Node::Lit(crate::ast::Literal::Str(s), _) => {
                            paired = Some(s);
                        }
                        _ => paired_is_nonstring = true,
                    }
                }
                self.skip_ws();
            }
            let v_span = Span::new(v_start, self.pos);
            variants.push(crate::ast::EnumVariant {
                name: v_name,
                payload,
                field_names,
                paired,
                paired_is_nonstring,
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
        // Reject duplicate variant names. The resolver collects variants into a
        // `BTreeSet`, which would SILENTLY absorb a duplicate — leaving the enum
        // with fewer distinct tags than written and deferring the error to a
        // runtime non-dense-keys failure (a fail-open). Fail loud at parse.
        for (i, v) in variants.iter().enumerate() {
            if variants[..i].iter().any(|prev| prev.name == v.name) {
                return Err(self.err(format!("duplicate enum variant `{}`", v.name)));
            }
        }
        // Record the enum name so a later `Name.Variant` reference normalises to
        // `Name::Variant` instead of parsing as a struct field access.
        if !self.enum_names.iter().any(|n| n == &name) {
            self.enum_names.push(name.clone());
        }
        let span = Span::new(start, self.pos);
        Ok(Node::EnumDef {
            is_pub,
            name,
            variants,
            attrs,
            span,
        })
    }

    /// Parse `sparse[layout], element_type[shape]>` after the cursor sits just
    /// past the `tensor<` prefix.  The `<` has already been consumed.
    ///
    /// Grammar (inside the outer `< >`):
    ///   `sparse` `[` layout_name `]` `,` type_ann ( dim_list_brackets )? `>`
    ///
    /// `layout_name` is one of: `csr`, `csc`, `coo`, `bsr`.
    fn parse_sparse_tensor_type(&mut self) -> Result<TypeAnn, ParseError> {
        // "sparse" keyword already confirmed by lookahead; consume it.
        self.pos += 6; // "sparse"
        self.skip_ws();
        self.expect(b'[')?;
        self.skip_ws();
        let layout_name = self
            .word()
            .ok_or_else(|| self.err("expected sparse layout name (csr, csc, coo, bsr)".into()))?;
        let layout = match layout_name {
            "csr" => crate::ast::SparseLayout::Csr,
            "csc" => crate::ast::SparseLayout::Csc,
            "coo" => crate::ast::SparseLayout::Coo,
            "bsr" => crate::ast::SparseLayout::Bsr,
            other => {
                return Err(self.err(format!(
                    "unknown sparse layout `{other}` — expected one of: csr, csc, coo, bsr"
                )));
            }
        };
        self.skip_ws();
        self.expect(b']')?;
        self.skip_ws();
        self.expect(b',')?;
        self.skip_ws();
        // Element type — can be any valid type annotation.
        let element = self.type_ann()?;
        self.skip_ws();
        // Optional shape `[d0, d1, ...]` expressed as ShapeDim strings and
        // then parsed into the ShapeDim vector via the existing dim_list logic.
        let shape = if self.at(b'[') {
            self.parse_sparse_shape()?
        } else {
            Vec::new()
        };
        self.skip_ws();
        self.expect(b'>')?;
        Ok(TypeAnn::SparseTensor {
            layout,
            element: Box::new(element),
            shape,
        })
    }

    /// Parse a shape list `[d0, d1, ...]` for a sparse tensor type, returning
    /// `Vec<ShapeDim>`.  Dimension tokens are either decimal integers (Known)
    /// or identifiers (Sym via a leaked `&'static str` — same convention as
    /// the rest of mindc's type system).
    fn parse_sparse_shape(&mut self) -> Result<Vec<crate::types::ShapeDim>, ParseError> {
        self.expect(b'[')?;
        let mut dims = Vec::new();
        self.skip_ws();
        if self.at(b']') {
            self.pos += 1;
            return Ok(dims);
        }
        loop {
            self.skip_ws();
            let dim = self.parse_one_shape_dim()?;
            dims.push(dim);
            self.skip_ws();
            if !self.eat(b',') {
                break;
            }
        }
        self.skip_ws();
        self.expect(b']')?;
        Ok(dims)
    }

    /// Parse one shape dimension: a decimal integer → `Known(n)`, or an
    /// identifier → `Sym` (leaked to `'static`; matches existing convention).
    fn parse_one_shape_dim(&mut self) -> Result<crate::types::ShapeDim, ParseError> {
        if let Some(d) = self.digits() {
            let n: usize = d
                .parse()
                .map_err(|_| self.err("shape dim overflow".into()))?;
            return Ok(crate::types::ShapeDim::Known(n));
        }
        if let Some(name) = self.word() {
            // Leak to 'static — same pattern as the rest of mindc's ShapeDim::Sym usage.
            let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
            return Ok(crate::types::ShapeDim::Sym(leaked));
        }
        Err(self.err("expected shape dimension (integer or identifier)".into()))
    }

    fn parse_fn_def(&mut self, is_pub: bool) -> Result<Node, ParseError> {
        self.parse_fn_def_with_attrs(None, false, is_pub, Vec::new())
    }

    /// Core `fn` parser.
    ///
    /// `reap_threshold` is `Some(t)` when a preceding `[reap_threshold(t)]`
    /// attribute was found by `parse_attributed_item`.
    /// `is_test` is `true` when a `[test]` attribute was present (RFC 0008 Phase B).
    /// `is_pub` is `true` when a `pub` keyword preceded `fn`.
    ///
    /// For `[test]` functions: the function must have zero parameters. Return
    /// type must be absent (defaults to `()`) or `bool`. A non-zero arity is a
    /// parse-time error so that bad test signatures surface before execution.
    fn parse_fn_def_with_attrs(
        &mut self,
        reap_threshold: Option<f64>,
        is_test: bool,
        is_pub: bool,
        attrs: Vec<crate::ast::Attribute>,
    ) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 2; // "fn"
        self.skip_ws_and_newlines();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected function name".into()))?
            .to_string();
        self.skip_ws_and_newlines();
        // Optional generic type-parameter list: `<T, U, ...>`.
        //
        // Empty for non-generic functions, so their AST and all downstream
        // lowering/codegen are byte-identical to before. Each name is a plain
        // identifier; the interpreter binds it to the concrete argument value
        // at the call site (no monomorphization at the interpreter level).
        let mut type_params: Vec<String> = Vec::new();
        if self.at(b'<') {
            self.pos += 1; // '<'
            self.skip_ws_and_newlines();
            if !self.at(b'>') {
                loop {
                    let tp = self
                        .word()
                        .ok_or_else(|| self.err("expected type parameter name".into()))?
                        .to_string();
                    type_params.push(tp);
                    self.skip_ws_and_newlines();
                    if !self.eat(b',') {
                        break;
                    }
                    self.skip_ws_and_newlines();
                    if self.at(b'>') {
                        break;
                    }
                }
            }
            self.skip_ws_and_newlines();
            self.expect(b'>')?;
            self.skip_ws_and_newlines();
        }
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

        // RFC 0008 Phase B: a `[test]` fn must have zero parameters.
        if is_test && !params.is_empty() {
            return Err(self.err(format!(
                "test function '{}' must have zero parameters (found {})",
                name,
                params.len()
            )));
        }

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
            is_pub,
            is_test,
            name,
            type_params,
            params,
            ret_type,
            body,
            reap_threshold,
            attrs,
            span,
        })
    }

    /// Kept for back-compat with call sites that do not pass `is_test`.
    #[allow(dead_code)]
    fn parse_fn_def_with_reap(
        &mut self,
        reap_threshold: Option<f64>,
        is_pub: bool,
    ) -> Result<Node, ParseError> {
        self.parse_fn_def_with_attrs(reap_threshold, false, is_pub, Vec::new())
    }

    /// Parse `extern "C" [callconv(.x)] { fn_decls... }` (RFC 0010 Phase A).
    ///
    /// Grammar:
    ///   `extern` `"C"` [`callconv` `(` `.` tag `)`] `{`
    ///       ( [`safe` | `unsafe`] `fn` name `(` params `)` \[`->` ret\] `;` )*
    ///   `}`
    ///
    /// Phase A accepts all four `callconv` tags syntactically; lowering
    /// selects the platform default (`.sysv` on Linux x86_64) regardless
    /// of which tag is stored — Phase C/D handle Win64/AAPCS quirks.
    fn parse_extern_block(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // "extern"
        self.skip_ws();
        // Require `"C"` string (the only ABI MIND supports in Phase A).
        if !self.at(b'"') {
            return Err(self.err("expected `\"C\"` after `extern`".into()));
        }
        self.pos += 1;
        if !self.starts_with(b"C\"") {
            return Err(self.err("only `extern \"C\"` is supported in RFC 0010 Phase A".into()));
        }
        self.pos += 2; // C"
        self.skip_ws();
        // Optional `callconv(.tag)` annotation.
        let callconv = if self.at_keyword(b"callconv") {
            self.pos += 8; // "callconv"
            self.skip_ws();
            self.expect(b'(')?;
            self.skip_ws();
            if !self.at(b'.') {
                return Err(self.err("expected `.tag` inside `callconv(...)`".into()));
            }
            self.pos += 1; // '.'
            let tag = self
                .word()
                .ok_or_else(|| self.err("expected callconv tag name after `.`".into()))?;
            let cc = match tag {
                "c" => CallConv::C,
                "sysv" => CallConv::SysV,
                "win64" => CallConv::Win64,
                "aapcs" => CallConv::Aapcs,
                other => return Err(self.err(format!(
                    "unknown callconv tag `{other}` — expected one of: .c, .sysv, .win64, .aapcs"
                ))),
            };
            self.skip_ws();
            self.expect(b')')?;
            self.skip_ws();
            cc
        } else {
            CallConv::C
        };
        self.expect(b'{')?;
        let mut fns = Vec::new();
        loop {
            self.skip_ws_and_newlines();
            if self.at(b'}') || self.at_end() {
                break;
            }
            // Skip line comments inside the block.
            if self.pos + 1 < self.b.len()
                && self.b[self.pos] == b'/'
                && self.b[self.pos + 1] == b'/'
            {
                while self.pos < self.b.len() && self.b[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            fns.push(self.parse_extern_fn()?);
            self.skip_ws();
            self.eat(b';'); // optional trailing semicolon after each fn decl
        }
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        let span = Span::new(start, self.pos);
        Ok(Node::ExternBlock {
            callconv,
            fns,
            span,
        })
    }

    /// Parse one `[safe | unsafe] fn name(params) [-> ret] [;]` inside an
    /// `extern "C"` block (RFC 0010 Phase A).
    fn parse_extern_fn(&mut self) -> Result<ExternFn, ParseError> {
        let start = self.pos;
        // Optional `safe` / `unsafe` keyword — default is `unsafe` for
        // compatibility with the conservative Phase A baseline; the RFC
        // says every symbol must carry an explicit tag, but we default to
        // unsafe if absent so existing test patterns work.
        let is_unsafe = if self.at_keyword(b"safe") {
            self.pos += 4;
            self.skip_ws();
            false
        } else if self.at_keyword(b"unsafe") {
            self.pos += 6;
            self.skip_ws();
            true
        } else {
            // Neither keyword — default to unsafe for a bare `fn`.
            true
        };
        if !self.at_keyword(b"fn") {
            return Err(self.err("expected `fn` inside `extern \"C\"` block".into()));
        }
        self.pos += 2; // "fn"
        self.skip_ws();
        let name = self
            .word()
            .ok_or_else(|| self.err("expected function name in extern declaration".into()))?
            .to_string();
        self.skip_ws();
        self.expect(b'(')?;
        let mut params = Vec::new();
        let mut is_varargs = false;
        self.skip_ws_and_newlines();
        if !self.at(b')') {
            loop {
                self.skip_ws_and_newlines();
                // Varargs sentinel `...`
                if self.starts_with(b"...") {
                    self.pos += 3;
                    is_varargs = true;
                    self.skip_ws();
                    // `...` must be the last parameter.
                    break;
                }
                params.push(self.parse_param()?);
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
                if self.at(b')') {
                    break;
                }
            }
        }
        self.skip_ws_and_newlines();
        self.expect(b')')?;
        self.skip_ws();
        let ret_type = if self.starts_with(b"->") {
            self.pos += 2;
            self.skip_ws();
            Some(self.type_ann()?)
        } else {
            None
        };
        let span = Span::new(start, self.pos);
        Ok(ExternFn {
            is_unsafe,
            name,
            params,
            ret_type,
            is_varargs,
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
        // Phase 10.6: accept `let mut name = ...`. mindc semantics treat
        // mut as informational — the eval/codegen path mutates the binding
        // in the local env regardless of the marker; the parser records
        // it so the formatter can round-trip the keyword (it is otherwise
        // not load-bearing for type-checking or codegen).
        let mutable = if self.at_keyword(b"mut") {
            self.pos += 3;
            self.skip_ws_and_newlines();
            true
        } else {
            false
        };
        // Tuple-destructuring binding: `let (a, b, ...) = expr`. Each name binds
        // the corresponding tuple element. A single-name `(a)` collapses to a
        // plain `Let` on `a` (grouping parens), mirroring the 1-tuple collapse in
        // expression and pattern position.
        if self.at(b'(') {
            self.pos += 1;
            let mut names = Vec::new();
            self.skip_ws_and_newlines();
            while !self.at(b')') && !self.at_end() {
                let nm = self
                    .word()
                    .ok_or_else(|| self.err("expected variable name in tuple binding".into()))?
                    .to_string();
                names.push(nm);
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
            }
            self.expect(b')')?;
            self.skip_ws_and_newlines();
            self.expect(b'=')?;
            self.skip_ws_and_newlines();
            let value = self.parse_expr()?;
            let span = Span::new(start, self.pos);
            if names.len() == 1 {
                return Ok(Node::Let {
                    name: names.into_iter().next().unwrap(),
                    mutable,
                    ann: None,
                    value: Box::new(value),
                    span,
                });
            }
            return Ok(Node::LetTuple {
                names,
                mutable,
                value: Box::new(value),
                span,
            });
        }
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
            mutable,
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
        // No `..` → this is a FOR-EACH over a collection: `for x in coll { … }`.
        // `start_expr` is the collection expression. (A range `for i in a..b`
        // takes the branch below.)
        if !(self.pos + 1 < self.b.len()
            && self.b[self.pos] == b'.'
            && self.b[self.pos + 1] == b'.')
        {
            self.skip_ws_and_newlines();
            self.expect(b'{')?;
            let body = self.parse_fn_body_stmts()?;
            self.skip_ws_and_newlines();
            self.expect(b'}')?;
            let span = Span::new(start, self.pos);
            return Ok(Node::ForEach {
                var,
                collection: Box::new(start_expr),
                body,
                span,
            });
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

    /// Parse `while cond { body }` (RFC 0005 Gap 1).
    ///
    /// The condition is a full expression terminated by `{`; the body is a
    /// block of statements terminated by `}`. No `break` / `continue` in
    /// this phase (noted as follow-on in the design doc).
    #[cfg(feature = "std-surface")]
    fn parse_while(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 5; // consume "while"
        self.skip_ws_and_newlines();
        let cond = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b'{')?;
        let body = self.parse_fn_body_stmts()?;
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        let span = Span::new(start, self.pos);
        Ok(Node::While {
            cond: Box::new(cond),
            body,
            span,
        })
    }

    /// Parse `loop { <stmts> }` — an unconditional loop. Desugared to
    /// `while 1 { <stmts> }`, so the body must `break`/`return` to terminate;
    /// the `while` lowering owns iteration semantics (and the compile-time-eval
    /// iteration cap that makes a genuinely-infinite loop fail loudly).
    #[cfg(feature = "std-surface")]
    fn parse_loop(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 4; // consume "loop"
        self.skip_ws_and_newlines();
        self.expect(b'{')?;
        let body = self.parse_fn_body_stmts()?;
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        let span = Span::new(start, self.pos);
        Ok(Node::While {
            cond: Box::new(Node::Lit(Literal::Int(1), span)),
            body,
            span,
        })
    }

    /// Parse `region { <stmts> }` (RFC 0010 Phase J-A).
    ///
    /// A region block is an expression that evaluates to the last statement
    /// in its body. Allocations made via `__mind_alloc` inside the block are
    /// tracked and freed automatically when the block exits.
    #[cfg(feature = "std-surface")]
    fn parse_region(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 6; // consume "region"
        self.skip_ws_and_newlines();
        self.expect(b'{')?;
        let body = self.parse_fn_body_stmts()?;
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        let span = Span::new(start, self.pos);
        Ok(Node::Region { body, span })
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
            // Fast path: operator continues on the same line. skip_ws is
            // a tight inner loop (whitespace bytes only) and matches the
            // pre-Phase-10.6 behaviour exactly, so the bench-baseline
            // numbers carry over here.
            self.skip_ws();
            let (op, lbp, rbp, advance) = match self.peek_binop() {
                Some(o) => o,
                None => {
                    // Slow path: maybe `\n... +` continuation. Only pay
                    // the saved_pos / cross-newline scan when an actual
                    // newline is at `self.pos` — otherwise the loop must
                    // exit (atom is done) and any further work is wasted.
                    // peek_binop only returns Some() for actual operator
                    // bytes, and statement-level keywords (`let`, `for`,
                    // `return`, etc.) do not start a binop, so the
                    // widened skip cannot spill into the next statement.
                    // skip_ws already consumed `\r`, so only `\n` matters
                    // as the continuation trigger.
                    if !self.at(b'\n') {
                        break;
                    }
                    let saved_pos = self.pos;
                    self.skip_ws_and_newlines();
                    match self.peek_binop() {
                        Some(o) => o,
                        None => {
                            // Restore position so the outer loop's
                            // separator handling sees the newlines.
                            self.pos = saved_pos;
                            break;
                        }
                    }
                }
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
                // RFC 0012 Phase B: tensor operators desugar in parse_pratt
                // to their dedicated AST nodes. lower_expr handles both at
                // a single well-defined desugar point.
                PrattOp::TensorMatmul => Node::TensorMatmul {
                    lhs: Box::new(left),
                    rhs: Box::new(right),
                    span,
                },
                PrattOp::TensorElem(elem_op) => Node::TensorElemwise {
                    op: elem_op,
                    lhs: Box::new(left),
                    rhs: Box::new(right),
                    span,
                },
            };
        }
        Ok(left)
    }

    /// Peek the next binary operator at `self.pos` without advancing.
    /// Caller MUST have already invoked `skip_ws()`. Returns
    /// `(op, left_bp, right_bp, byte_advance)` or `None` if the next token
    /// is not a recognised binary operator.
    #[inline]
    /// Detect a compound-assignment operator at byte offset `p`. Returns the
    /// underlying binary op and the operator's byte width. Excludes the
    /// comparison/logical/shift shapes (`== != <= >= && || << >>`): only a
    /// single-char arith/bit op (or `<<`/`>>`) immediately followed by `=`
    /// qualifies. Shared by `peek_binop` (returns None so the Pratt parse stops
    /// at the LHS) and `parse_stmt` (desugars `lhs OP= rhs`).
    fn compound_assign_op(&self, p: usize) -> Option<(CompoundOp, usize)> {
        let b0 = *self.b.get(p)?;
        let b1 = self.b.get(p + 1).copied().unwrap_or(0);
        let b2 = self.b.get(p + 2).copied().unwrap_or(0);
        // Three-char shift-assign (checked before the `<<`/`>>` infix shapes).
        match (b0, b1, b2) {
            (b'<', b'<', b'=') => return Some((CompoundOp::Bit(crate::ast::BitOp::Shl), 3)),
            (b'>', b'>', b'=') => return Some((CompoundOp::Bit(crate::ast::BitOp::Shr), 3)),
            _ => {}
        }
        if b1 != b'=' {
            return None;
        }
        // `<=`/`>=` are comparisons (b0 not in this set), so they never match.
        let op = match b0 {
            b'+' => CompoundOp::Arith(BinOp::Add),
            b'-' => CompoundOp::Arith(BinOp::Sub),
            b'*' => CompoundOp::Arith(BinOp::Mul),
            b'/' => CompoundOp::Arith(BinOp::Div),
            b'%' => CompoundOp::Arith(BinOp::Mod),
            b'&' => CompoundOp::Bit(crate::ast::BitOp::And),
            b'|' => CompoundOp::Bit(crate::ast::BitOp::Or),
            b'^' => CompoundOp::Bit(crate::ast::BitOp::Xor),
            _ => return None,
        };
        Some((op, 2))
    }

    fn peek_binop(&self) -> Option<(PrattOp, u8, u8, usize)> {
        let p = self.pos;
        if p >= self.b.len() {
            return None;
        }
        // A compound-assignment operator (`+= <<= ...`) is not an infix binary
        // operator — stop the Pratt parse at the LHS so `parse_stmt` can desugar
        // it. Without this `i += 1` parses `i +` then errors on `=`.
        if self.compound_assign_op(p).is_some() {
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
            // RFC 0012 Phase B: dot-prefix elementwise operators `.+ .- .* ./`.
            // Precedence mirrors the RFC 0012 §4.7 table:
            //   .* and ./ bind tighter than .+ and .- (levels 13/14 and 9/10).
            // These MUST be checked before the one-char `.` fall-through in
            // `parse_atom` (which looks for `is_ident_start` after the dot and
            // won't match `+/-/*//`).
            (b'.', b'+') => return Some((PrattOp::TensorElem(TensorElemOp::Add), 9, 10, 2)),
            (b'.', b'-') => return Some((PrattOp::TensorElem(TensorElemOp::Sub), 9, 10, 2)),
            (b'.', b'*') => return Some((PrattOp::TensorElem(TensorElemOp::Mul), 13, 14, 2)),
            (b'.', b'/') => return Some((PrattOp::TensorElem(TensorElemOp::Div), 13, 14, 2)),
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
            b'%' => Some((PrattOp::Arith(BinOp::Mod), 13, 14, 1)),
            // RFC 0012 Phase B: `@` matmul. Precedence between elementwise
            // mul/div (13) and add/sub (9): use 11 (left) / 12 (right) —
            // same slot as bitwise, tighter than elementwise-add, looser
            // than elementwise-mul.
            b'@' => Some((PrattOp::TensorMatmul, 11, 12, 1)),
            _ => None,
        }
    }

    /// Parse a character literal `'c'` / `'\n'` into an integer constant equal
    /// to the character's Unicode scalar value (for ASCII this is the byte). The
    /// usual escapes are recognised (`\n \t \r \0 \\ \' \"`); any other `\x` is
    /// the literal `x`. A char literal lowers exactly like the equivalent integer
    /// literal — no new IR — so it is fully deterministic and RUNS.
    fn parse_char_lit(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 1; // consume opening `'`
        let val: i64 = if self.at(b'\\') {
            self.pos += 1;
            let c = self
                .peek()
                .ok_or_else(|| self.err("unterminated character escape".into()))?;
            self.pos += 1;
            match c {
                b'n' => 10,
                b't' => 9,
                b'r' => 13,
                b'0' => 0,
                b'\\' => 92,
                b'\'' => 39,
                b'"' => 34,
                other => other as i64,
            }
        } else {
            // Decode one UTF-8 scalar from the source so a non-ASCII char literal
            // carries its full codepoint (ASCII is the single-byte fast path).
            let rest = std::str::from_utf8(&self.b[self.pos..])
                .map_err(|_| self.err("invalid UTF-8 in character literal".into()))?;
            let ch = rest
                .chars()
                .next()
                .ok_or_else(|| self.err("empty character literal".into()))?;
            self.pos += ch.len_utf8();
            ch as i64
        };
        if !self.eat(b'\'') {
            return Err(self.err("expected `'` to close character literal".into()));
        }
        let span = Span::new(start, self.pos);
        Ok(Node::Lit(Literal::Int(val), span))
    }

    /// Decode a string body — the opening `"` already consumed, `self.pos` at
    /// the first body byte — up to but NOT including the closing `"` (the caller
    /// consumes it). Applies the ONE escape table (`\n`→0x0A, `\t`, `\r`, `\0`,
    /// `\\`, `\'`, `\"`; unknown escape keeps the literal byte) so string
    /// LITERALS and string PATTERNS decode identically. A `\` still escapes the
    /// next byte during the scan so `\"` / `\\` never prematurely terminate.
    /// Shared because the pattern arm previously copied the RAW slice — a
    /// `"\n"` pattern stored the 2 bytes `\` `n` and so never matched a decoded
    /// scrutinee (a latent silent-miscompile if string patterns become runnable;
    /// today the runnable lowering rejects string patterns fail-loud).
    fn decode_string_body(&mut self) -> String {
        let mut decoded: Vec<u8> = Vec::new();
        while self.pos < self.b.len() && self.b[self.pos] != b'"' {
            if self.b[self.pos] == b'\\' && self.pos + 1 < self.b.len() {
                let esc = self.b[self.pos + 1];
                self.pos += 2;
                decoded.push(match esc {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    b'0' => b'\0',
                    b'\\' => b'\\',
                    b'\'' => b'\'',
                    b'"' => b'"',
                    other => other,
                });
            } else {
                decoded.push(self.b[self.pos]);
                self.pos += 1;
            }
        }
        // Decoded bytes are UTF-8-valid (source was UTF-8; each escape maps to a
        // single ASCII byte; non-escaped multi-byte scalars are copied verbatim);
        // fall back to a lossy decode rather than panic if that ever breaks.
        String::from_utf8(decoded)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }

    fn parse_string_lit(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.expect(b'"')?;
        // Escape-aware scan that DECODES escapes into their byte values, so the
        // stored `Literal::Str` carries the same bytes the program observes at
        // runtime (string_len / string_get_byte read these bytes verbatim via
        // the `__mind_alloc` + `__mind_store_i8` lowering in eval/lower.rs).
        //
        // The escape table is identical to `parse_char_lit` so `'\n'` and the
        // `"\n"` inside a string agree on the byte (10) — the prior verbatim
        // retention produced the 2-byte sequence `\` `n`, a SILENT MISCOMPILE
        // (`"\n".len()` returned 2, `string_get_byte("\n",0)` returned 92).
        // A `\` still escapes the following byte during the scan so `\"`
        // (escaped quote) and `\\` (escaped backslash) do not prematurely
        // terminate the literal. Unknown escapes keep the literal following
        // byte (`other => other`), matching `parse_char_lit`.
        //
        // Keystone byte-identity: the bootstrap source uses no string-literal
        // escapes (every `\` in main.mind is in a `//` comment), so the decoder
        // never rewrites anything for it and its emit stays byte-identical.
        let s = self.decode_string_body();
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

    /// Parse a brace literal: a MAP `{ key: value, … }` (colon-separated) or a
    /// SET `{ a, b, c }` (comma-separated, no colons). Disambiguated by the
    /// token after the first element: `:` → map, `,`/`}` → set. The empty `{}`
    /// is a MapLit (both lower to `map_new`). Trailing comma allowed.
    fn parse_map_lit(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.expect(b'{')?;
        self.skip_ws_and_newlines();
        if self.at(b'}') {
            self.pos += 1;
            return Ok(Node::MapLit {
                entries: Vec::new(),
                span: Span::new(start, self.pos),
            });
        }
        // Parse the first element, then decide map vs set by the next token.
        let first = self.parse_expr()?;
        self.skip_ws();
        if self.at(b':') {
            // MAP: `{ k: v, … }`.
            self.pos += 1; // ':'
            self.skip_ws_and_newlines();
            let first_val = self.parse_expr()?;
            let mut entries: Vec<(Node, Node)> = vec![(first, first_val)];
            self.skip_ws_and_newlines();
            while self.eat(b',') {
                self.skip_ws_and_newlines();
                if self.at(b'}') {
                    break;
                }
                let key = self.parse_expr()?;
                self.skip_ws();
                self.expect(b':')?;
                self.skip_ws_and_newlines();
                let value = self.parse_expr()?;
                entries.push((key, value));
                self.skip_ws_and_newlines();
            }
            self.skip_ws_and_newlines();
            self.expect(b'}')?;
            return Ok(Node::MapLit {
                entries,
                span: Span::new(start, self.pos),
            });
        }
        // SET: `{ a, b, c }`.
        let mut elements: Vec<Node> = vec![first];
        self.skip_ws_and_newlines();
        while self.eat(b',') {
            self.skip_ws_and_newlines();
            if self.at(b'}') {
                break;
            }
            elements.push(self.parse_expr()?);
            self.skip_ws_and_newlines();
        }
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        Ok(Node::SetLit {
            elements,
            span: Span::new(start, self.pos),
        })
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
                        // RFC 0012 Phase B.2: `a.sum()` / `a.mean()` (no args)
                        // are tensor reductions over all axes, desugared to the
                        // existing CallTensorSum / CallTensorMean nodes (which
                        // type-check via reduce_shape and lower to the reduction
                        // IR). `sum`/`mean` are reserved as zero-arg reduction
                        // methods; axis-specified and `.max` are future work.
                        if args.is_empty() && (method == "sum" || method == "mean") {
                            let x = Box::new(node);
                            node = if method == "sum" {
                                Node::CallTensorSum {
                                    x,
                                    axes: Vec::new(),
                                    keepdims: false,
                                    span,
                                }
                            } else {
                                Node::CallTensorMean {
                                    x,
                                    axes: Vec::new(),
                                    keepdims: false,
                                    span,
                                }
                            };
                            continue;
                        }
                        // Module-qualified call `mod.fn(args)` → bare cross-module
                        // call `fn(args)`: when the receiver is a bare identifier
                        // naming an imported MODULE, the `.fn(...)` is a namespace
                        // access, not a UFCS method on a value. All module
                        // functions share one global link unit, so the bare call
                        // resolves (the import injects `fn`) and lowers directly.
                        // A receiver that is NOT an imported module (a local
                        // value, a field access like `p.lexer`) keeps the
                        // MethodCall path untouched.
                        if let Node::Lit(Literal::Ident(recv_name), _) = &node {
                            if self.imports.iter().any(|s| s == recv_name) {
                                node = Node::Call {
                                    callee: method,
                                    args,
                                    span,
                                };
                                continue;
                            }
                            // Invariant predicate call: `inv.check(args)` where
                            // `inv` names an invariant block resolves to the free
                            // function `<inv>_check(args)` that parse_invariant_block
                            // emits for each `check(...)` predicate.
                            if self.invariants.iter().any(|s| s == recv_name) {
                                node = Node::Call {
                                    callee: format!("{recv_name}_{method}"),
                                    args,
                                    span,
                                };
                                continue;
                            }
                        }
                        node = Node::MethodCall {
                            receiver: Box::new(node),
                            method,
                            args,
                            span,
                        };
                        continue;
                    } else if method == "T" {
                        // RFC 0012 Phase B.2: `a.T` is the transpose operator,
                        // desugared to the existing CallTranspose node (which
                        // already type-checks and lowers to Instr::Transpose).
                        // `T` is reserved as a postfix op, not a field name.
                        let span = Span::new(node.span_start(), self.pos);
                        node = Node::CallTranspose {
                            x: Box::new(node),
                            axes: None,
                            span,
                        };
                        continue;
                    } else {
                        // Module-qualified value `mod.CONST` → bare cross-module
                        // reference `CONST`: the namespace-access twin of the
                        // `mod.fn(args)` → `fn(args)` normalisation above. When the
                        // receiver is a bare identifier naming an imported MODULE,
                        // `.CONST` selects a module-level export (a `const`, not a
                        // value's field), and all module symbols share one global
                        // link unit — so the bare reference resolves (the import
                        // surface carries the name) and lowers directly, exactly as
                        // the qualified-call form already does. A receiver that is
                        // NOT an imported module keeps the normal FieldAccess path.
                        if let Node::Lit(Literal::Ident(recv_name), _) = &node {
                            if self.imports.iter().any(|s| s == recv_name) {
                                let span = Span::new(node.span_start(), self.pos);
                                node = Node::Lit(Literal::Ident(method), span);
                                continue;
                            }
                        }
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
            // Phase 10.6: postfix index `receiver[index]` for slice /
            // array / Vec element access. The cast subtlety: `[...]`
            // in expression position without a receiver is still an
            // array literal via parse_primary; this path triggers only
            // when something else has already been parsed and `[`
            // follows directly (no whitespace consumed crossing
            // newlines).
            if self.at(b'[') {
                self.pos += 1;
                self.skip_ws();
                let index = self.parse_expr()?;
                self.skip_ws();
                if !self.eat(b']') {
                    return Err(self.err("expected `]` to close index expression".into()));
                }
                let span = Span::new(node.span_start(), self.pos);
                node = Node::IndexAccess {
                    receiver: Box::new(node),
                    index: Box::new(index),
                    span,
                };
                continue;
            }
            break;
        }
        Ok(node)
    }

    /// Parse a generic-type STATIC CONSTRUCTOR `Type<args>.method(args)` after the
    /// `Type` name (`map`/`set`/`array`) has been read and the cursor is at `<`.
    /// Returns `Some(Call)` mapping to the runtime constructor (`map_new` for
    /// map/set, `vec_new` for array) — or `None` (cursor fully restored) when it
    /// is NOT actually a `<…>.method(…)` form, so a real `map < x` comparison or
    /// any other use parses normally.
    fn try_parse_collection_ctor(
        &mut self,
        type_name: &str,
        start: usize,
    ) -> Result<Option<Node>, ParseError> {
        let saved = self.pos;
        // Consume a balanced `<…>` (handles nested generics like
        // `map<string, array<string>>`).
        debug_assert!(self.at(b'<'));
        let mut depth: i32 = 0;
        loop {
            if self.at_end() {
                self.pos = saved;
                return Ok(None);
            }
            let c = self.b[self.pos];
            self.pos += 1;
            if c == b'<' {
                depth += 1;
            } else if c == b'>' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
        self.skip_ws();
        if !self.eat(b'.') {
            self.pos = saved;
            return Ok(None);
        }
        let method = match self.word() {
            Some(w) => w.to_string(),
            None => {
                self.pos = saved;
                return Ok(None);
            }
        };
        self.skip_ws();
        if !self.eat(b'(') {
            self.pos = saved;
            return Ok(None);
        }
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
        // map/set are both the std.map runtime; array is std.vec.
        let prefix = if type_name == "array" {
            "vec"
        } else {
            type_name
        };
        let callee = if method == "new" && type_name == "set" {
            "map_new".to_string()
        } else {
            format!("{prefix}_{method}")
        };
        Ok(Some(Node::Call { callee, args, span }))
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
        // A bare `{` in expression position is a MAP literal (`{}` or
        // `{ k: v, … }`). A named struct literal `Name { … }` is parsed in the
        // identifier path (the name comes first), so a leading `{` here is never
        // a struct literal; statement blocks are parsed by the statement layer,
        // not parse_primary.
        if self.at(b'{') {
            return self.parse_map_lit();
        }
        // Byte-string literal `b"..."` (#5): the `b` prefix marks a raw byte-slice
        // literal. The bytes are identical to a plain string literal, so consume the
        // `b` and reuse parse_string_lit (which yields `Literal::Str`, already a byte
        // sequence). `b"` is unambiguous — no valid expression has an ident `b`
        // immediately followed by `"`.
        if self.at(b'b') && self.pos + 1 < self.b.len() && self.b[self.pos + 1] == b'"' {
            self.pos += 1;
            return self.parse_string_lit();
        }
        if self.at(b'"') {
            return self.parse_string_lit();
        }
        if self.at(b'\'') {
            return self.parse_char_lit();
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
        // Unary logical NOT `!expr`. Disambiguated from the `!=` binary operator,
        // which is only recognised in infix position by `peek_binop` after a left
        // operand is already parsed; here in prefix position a `!` not followed by
        // `=` is unambiguously logical-not (enum_match #9, `if !parser_at(..)`).
        if self.at(b'!') && self.b.get(self.pos + 1) != Some(&b'=') {
            let start = self.pos;
            self.advance();
            self.skip_ws();
            let operand = self.parse_atom()?;
            let span = Span::new(start, self.pos);
            return Ok(Node::Not {
                operand: Box::new(operand),
                span,
            });
        }
        // Phase 10.7: `&expr` and `&mut expr` reference-taking prefix.
        //
        // Disambiguation: `&` is also the infix bitwise-AND operator.
        // `peek_binop` only fires when `&` appears in *infix* position
        // (after the Pratt loop has already parsed a left operand).
        // Here we are in *prefix* position (inside `parse_primary`, which
        // is called by `parse_atom` before the Pratt loop), so `&` is
        // unambiguously a ref-take. No saved_pos / backtrack needed.
        if self.at(b'&') {
            let start = self.pos;
            self.pos += 1; // consume `&`
            self.skip_ws();
            let mutable = if self.at_keyword(b"mut") {
                self.pos += 3;
                self.skip_ws();
                true
            } else {
                false
            };
            let inner = self.parse_atom()?;
            let span = Span::new(start, self.pos);
            return Ok(Node::Ref {
                mutable,
                inner: Box::new(inner),
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
        // Phase 10.7: `match` expression. Fast-path: every primary parse pays
        // for the keyword check, so short-circuit on the leading byte first
        // (the full `at_keyword` is a 5-byte compare + boundary check).
        if self.at(b'm') && self.at_keyword(b"match") {
            return self.parse_match_expr();
        }
        // RFC 0010 Phase J-A: `region { }` as a primary expression.
        // Gated to std-surface; fires when a region appears on the RHS of
        // a `let` binding or as a bare expression statement.
        #[cfg(feature = "std-surface")]
        if self.at(b'r') && self.at_keyword(b"region") {
            return self.parse_region();
        }
        let start = self.pos;
        let ident = self
            .dotted_ident()
            .ok_or_else(|| self.unexpected_prefix_err())?;
        self.skip_ws();
        // `map<K,V>.new(args)` / `set<T>.new()` / `array<T>.new()` — a generic-type
        // STATIC CONSTRUCTOR in expression position. Gated on the exact collection
        // type-name followed by `<` (so a real `map < x` comparison is untouched),
        // with full position restore if it isn't actually a `<…>.method(…)` form.
        if matches!(ident.as_str(), "map" | "set" | "array") && self.at(b'<') {
            if let Some(node) = self.try_parse_collection_ctor(&ident, start)? {
                return Ok(node);
            }
        }
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
                // Bare boolean literals `true` / `false` in value position lower
                // to the i64 truthiness encoding (1 / 0) — the same mapping the
                // pattern side uses and the representation MIND's bool ABI
                // expects (there is no dedicated `Literal::Bool`). Without this
                // they fell through to `Literal::Ident` and the type-checker
                // reported them as unknown identifiers (e.g. `return Err(false)`,
                // `T { flag: false }`). main.mind never uses a bare bool value,
                // so this path never fires for the keystone self-compile.
                if ident == "true" || ident == "false" {
                    let span = Span::new(start, self.pos);
                    let v = if ident == "true" { 1 } else { 0 };
                    return Ok(Node::Lit(Literal::Int(v), span));
                }
                // Phase 10.6: identifiers that contain `::` segment
                // separators (enum variant access — e.g.
                // `config.AddressingMode::Content`) keep the full path
                // as an identifier; the catch-all backtrack below
                // would slice off everything after the first `.` which
                // would corrupt the path.
                if ident.contains("::") {
                    if self.at(b'(') {
                        return self.parse_generic_call(ident, start);
                    } else if self.at(b'{') && self.struct_lit_body_ahead() {
                        return self.parse_struct_literal(ident, start);
                    } else {
                        let span = Span::new(start, self.pos);
                        return Ok(Node::Lit(Literal::Ident(ident), span));
                    }
                }
                // If the ident contains a dot and the first segment is not a known
                // namespace like "tensor", backtrack to the first segment so
                // parse_atom's dot-loop can handle method calls.
                if let Some(dot_idx) = ident.find('.') {
                    let first = &ident[..dot_idx];
                    // `Enum.Variant` (dot) → canonical `Enum::Variant` when the
                    // first segment is a declared enum: a unit value
                    // (`Color.Red`), a payload constructor (`Expr.IntLit(x)`), or
                    // a struct variant (`Expr.StringLit { … }`). A non-enum
                    // receiver falls through to the struct-field / module-call
                    // handling below. A cross-module enum (defined in a sibling
                    // source file) resolves via the project-wide registry too.
                    if self.is_enum_name(first) {
                        let qualified = format!("{}::{}", first, &ident[dot_idx + 1..]);
                        self.skip_ws();
                        if self.at(b'(') {
                            return self.parse_generic_call(qualified, start);
                        } else if self.at(b'{') && self.struct_lit_body_ahead() {
                            return self.parse_struct_literal(qualified, start);
                        } else {
                            let span = Span::new(start, self.pos);
                            return Ok(Node::Lit(Literal::Ident(qualified), span));
                        }
                    }
                    if first != "tensor" {
                        // A dot-qualified name immediately followed by a
                        // `{ field: value }` body is a QUALIFIED struct/enum-variant
                        // literal — e.g. `Expr.StringLit { value: .. }` (a struct
                        // variant of enum `Expr`) or `mod.Point { x: 1 }`. Keep the
                        // FULL dotted path as the type name rather than slicing off
                        // everything after the first dot. The `struct_lit_body_ahead`
                        // guard (requires `IDENT :`) keeps this from stealing a
                        // control-flow `{ }` body that merely follows a field access
                        // like `if cfg.on { … }`.
                        if self.at(b'{') && self.struct_lit_body_ahead() {
                            return self.parse_struct_literal(ident, start);
                        }
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
                } else if self.at(b'{') && self.struct_lit_body_ahead() {
                    // Phase 10.6: struct literal `Name { field: value, ... }`.
                    // The lookahead guard rejects any `{ ... }` that doesn't
                    // open with `IDENT :` — protects callers that pass a
                    // bare identifier into a control-flow `{...}` body.
                    self.parse_struct_literal(ident, start)
                } else {
                    let span = Span::new(start, self.pos);
                    Ok(Node::Lit(Literal::Ident(ident), span))
                }
            }
        }
    }

    fn parse_number_lit(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        // Radix-prefixed integer literals: `0x..` (hex), `0o..` (octal),
        // `0b..` (binary). Always integers — never floats — and accept the same
        // `u8`..`i64` suffix as decimal literals, desugared through the existing
        // `as`-cast path. `_` digit separators are permitted and ignored. The
        // keystone source uses no radix literals, so this branch never fires for
        // it and its emit stays byte-identical.
        if self.b.get(self.pos) == Some(&b'0')
            && matches!(
                self.b.get(self.pos + 1),
                Some(b'x' | b'X' | b'o' | b'O' | b'b' | b'B')
            )
        {
            let radix: u32 = match self.b[self.pos + 1].to_ascii_lowercase() {
                b'x' => 16,
                b'o' => 8,
                _ => 2,
            };
            self.pos += 2; // consume the `0x` / `0o` / `0b` prefix
            let digit_start = self.pos;
            while self.pos < self.b.len()
                && (self.b[self.pos] == b'_' || (self.b[self.pos] as char).is_digit(radix))
            {
                self.pos += 1;
            }
            if self.pos == digit_start {
                return Err(self.err("expected digits after integer radix prefix".into()));
            }
            let raw: String = std::str::from_utf8(&self.b[digit_start..self.pos])
                .unwrap()
                .chars()
                .filter(|c| *c != '_')
                .collect();
            // Accept the full unsigned range, reinterpreting as two's-complement
            // i64 exactly like `parse_i64_literal` does for large decimals.
            let val = i64::from_str_radix(&raw, radix)
                .or_else(|_| u64::from_str_radix(&raw, radix).map(|u| u as i64))
                .map_err(|_| self.err("integer overflow".into()))?;
            let pre_suffix = self.pos;
            if let Some(ty) = self.int_type_suffix() {
                let span = Span::new(start, self.pos);
                let lit_span = Span::new(start, pre_suffix);
                return Ok(Node::As {
                    expr: Box::new(Node::Lit(Literal::Int(val), lit_span)),
                    ty,
                    span,
                });
            }
            let span = Span::new(start, self.pos);
            return Ok(Node::Lit(Literal::Int(val), span));
        }
        let mut d = self
            .digits()
            .ok_or_else(|| self.err("expected number".into()))?;
        // Digit separators in DECIMAL literals: `120_000`, `1_000_000`. An `_` is
        // part of the literal ONLY between two digits (`d _ d`); a trailing `_` or
        // an `_` before a non-digit is left for the surrounding parse. The `_`s are
        // dropped, so the value is `120000`. The radix path (above) already does
        // this; the decimal path was missing it, breaking `const X: u64 = 120_000`.
        // The keystone source uses no separators, so its emit stays byte-identical.
        while self.pos + 1 < self.b.len()
            && self.b[self.pos] == b'_'
            && self.b[self.pos + 1].is_ascii_digit()
        {
            self.pos += 1; // consume the `_`
            if let Some(more) = self.digits() {
                d.push_str(&more);
            }
        }
        // Check for decimal point → float literal. A `.` is a decimal point ONLY
        // when a DIGIT follows it (`1.0`). `1..10` (range) and `1.method()` /
        // `1.field` (a non-digit after `.`, e.g. a method call on an integer
        // literal) are NOT floats — they fall through to the integer literal so
        // the range/postfix machinery handles the `.`/`..`. Without the
        // digit-after-`.` guard, `48.byte()` mis-parsed as the float `48.` + a
        // dangling `byte`.
        if self.pos + 1 < self.b.len()
            && self.b[self.pos] == b'.'
            && self.b[self.pos + 1].is_ascii_digit()
        {
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
            // Cross-substrate byte-identity (THE WEDGE): `str::parse::<f64>()`
            // is std's `FromStr for f64`, which is spec'd to round to the
            // NEAREST representable value (ties-to-even) and is implemented as a
            // pure software routine in `core` (Eisel-Lemire fast path + a
            // big-integer slow-path fallback, since Rust 1.55). It does NOT call
            // libc `strtod` and does NOT consult locale/platform state, so the
            // same decimal literal yields the identical IEEE-754 bit pattern on
            // x86, ARM, and GPU hosts. The parsed `f64` therefore does not
            // poison constant folding / lowering. Do NOT swap this for a custom
            // or libc routine — that would risk changing valid float bits.
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
            // Deterministic across substrates for the same reason as the
            // decimal-point case above: std's correctly-rounded, locale-free
            // `f64::from_str`. See the note at the `1.0` site.
            let val: f64 = num_str
                .parse()
                .map_err(|_| self.err("invalid float".into()))?;
            let span = Span::new(start, self.pos);
            return Ok(Node::Lit(Literal::Float(val), span));
        }
        let val = self.parse_i64_literal(&d)?;
        // Issue #205: integer-type suffix (`2u32`, `-1i32`, ...). Desugar to an
        // `as`-cast so the typed literal flows through the existing cast path.
        let pre_suffix = self.pos;
        if let Some(ty) = self.int_type_suffix() {
            let span = Span::new(start, self.pos);
            let lit_span = Span::new(start, pre_suffix);
            return Ok(Node::As {
                expr: Box::new(Node::Lit(Literal::Int(val), lit_span)),
                ty,
                span,
            });
        }
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

    /// Phase 10.6 lookahead: from a position right at `{`, decide whether
    /// the brace opens a struct-literal body (`{ ident : ...` or `{}`) or
    /// some other construct that happens to follow an identifier. Pure
    /// lookahead — does not consume input. Returns false on shapes that
    /// look like blocks of statements rather than `field: value` pairs.
    fn struct_lit_body_ahead(&self) -> bool {
        let mut i = self.pos;
        if i >= self.b.len() || self.b[i] != b'{' {
            return false;
        }
        i += 1;
        // Skip whitespace and newlines.
        while i < self.b.len()
            && (self.b[i] == b' ' || self.b[i] == b'\t' || self.b[i] == b'\n' || self.b[i] == b'\r')
        {
            i += 1;
        }
        // Empty body `{ }` is a valid (zero-field) struct literal.
        if i < self.b.len() && self.b[i] == b'}' {
            return true;
        }
        // First non-whitespace token must be an identifier.
        let id_start = i;
        while i < self.b.len() && (self.b[i].is_ascii_alphanumeric() || self.b[i] == b'_') {
            i += 1;
        }
        if i == id_start {
            return false;
        }
        // After the identifier, skip whitespace and look for `:` that is NOT
        // followed by another `:` (which would be `::`, a path separator, not
        // a struct-field colon).
        while i < self.b.len() && (self.b[i] == b' ' || self.b[i] == b'\t') {
            i += 1;
        }
        i < self.b.len() && self.b[i] == b':' && self.b.get(i + 1).copied() != Some(b':')
    }

    /// Phase 10.6: parse `Name { field: value, field: value, ... }` after
    /// the name identifier has already been consumed by the caller.
    fn parse_struct_literal(&mut self, name: String, start: usize) -> Result<Node, ParseError> {
        self.expect(b'{')?;
        let mut fields: Vec<crate::ast::StructLitField> = Vec::new();
        self.skip_ws_and_newlines();
        while !self.at(b'}') && !self.at_end() {
            let f_start = self.pos;
            let f_name = self
                .word()
                .ok_or_else(|| self.err("expected struct field name".into()))?
                .to_string();
            self.skip_ws();
            if !self.eat(b':') {
                return Err(self.err("expected `:` after struct field name".into()));
            }
            self.skip_ws_and_newlines();
            let value = self.parse_expr()?;
            let f_span = Span::new(f_start, self.pos);
            fields.push(crate::ast::StructLitField {
                name: f_name,
                value,
                span: f_span,
            });
            self.skip_ws_and_newlines();
            if self.eat(b',') {
                self.skip_ws_and_newlines();
            } else {
                break;
            }
        }
        self.skip_ws_and_newlines();
        if !self.eat(b'}') {
            return Err(self.err("expected `}` to close struct literal".into()));
        }
        let span = Span::new(start, self.pos);
        Ok(Node::StructLit { name, fields, span })
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
        // `u32(x)` / `i64(x)` / … — a scalar CAST written in call form. Desugar
        // to `x as <type>`, reusing the existing cast typecheck/codegen path (no
        // new IR). mind-flow writes `u32(i + 1)`. Only the reserved scalar type
        // names with exactly one argument convert; everything else is a call.
        if args.len() == 1 {
            let cast_ty = match callee.as_str() {
                "u32" => Some(TypeAnn::ScalarU32),
                "i32" => Some(TypeAnn::ScalarI32),
                "i64" => Some(TypeAnn::ScalarI64),
                "f64" => Some(TypeAnn::ScalarF64),
                "u8" | "u16" | "u64" | "i8" | "i16" | "usize" | "f32" => {
                    Some(TypeAnn::Named(callee.clone()))
                }
                _ => None,
            };
            if let Some(ty) = cast_ty {
                let expr = args.pop().unwrap();
                return Ok(Node::As {
                    expr: Box::new(expr),
                    ty,
                    span,
                });
            }
        }
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

    // ── Phase 10.7: match expressions ─────────────────────────────────

    /// Parse `match <scrutinee> { arm, ... }`.
    ///
    /// Arms are separated by `,` (trailing comma optional). The body of
    /// each arm is either a bare expression or a `{ ... }` block.
    fn parse_match_expr(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.pos += 5; // "match"
        self.skip_ws_and_newlines();
        let scrutinee = self.parse_expr()?;
        self.skip_ws_and_newlines();
        self.expect(b'{')?;
        let mut arms = Vec::new();
        self.skip_ws_and_newlines();
        while !self.at(b'}') && !self.at_end() {
            let arm_start = self.pos;
            let pattern = self.parse_pattern()?;
            self.skip_ws_and_newlines();
            if !self.starts_with(b"=>") {
                return Err(self.err("expected `=>` after match pattern".into()));
            }
            self.pos += 2;
            self.skip_ws_and_newlines();
            let body = self.parse_match_arm_body(arm_start)?;
            let arm_span = Span::new(arm_start, self.pos);
            arms.push(MatchArm {
                pattern,
                body,
                span: arm_span,
            });
            self.skip_ws_and_newlines();
            if self.eat(b',') {
                self.skip_ws_and_newlines();
            }
        }
        self.skip_ws_and_newlines();
        self.expect(b'}')?;
        let span = Span::new(start, self.pos);
        Ok(Node::Match {
            scrutinee: Box::new(scrutinee),
            arms,
            span,
        })
    }

    /// Parse a match-arm body: a `{ … }` block, a statement (`return` /
    /// `continue` / `break` / `lhs = rhs` assignment), or an expression. The
    /// statement forms matter for Rust-ish match arms that mutate (`Ok(p) =>
    /// set = f(set, p)`) or skip (`Err(_) => continue`) — `parse_expr` handles
    /// none of these, so without this the loop mis-reads the remainder as a new
    /// arm pattern.
    fn parse_match_arm_body(&mut self, arm_start: usize) -> Result<Node, ParseError> {
        if self.at(b'{') {
            let blk_start = self.pos;
            self.pos += 1;
            let stmts = self.parse_fn_body_stmts()?;
            self.skip_ws_and_newlines();
            self.expect(b'}')?;
            return Ok(Node::Block {
                stmts,
                span: Span::new(blk_start, self.pos),
            });
        }
        if self.at_keyword(b"return") {
            return self.parse_return();
        }
        #[cfg(feature = "std-surface")]
        if self.at_keyword(b"break") {
            let s = self.pos;
            self.pos += 5;
            return Ok(Node::Break {
                span: Span::new(s, self.pos),
            });
        }
        #[cfg(feature = "std-surface")]
        if self.at_keyword(b"continue") {
            let s = self.pos;
            self.pos += 8;
            return Ok(Node::Continue {
                span: Span::new(s, self.pos),
            });
        }
        let expr = self.parse_expr()?;
        self.skip_ws();
        // `=> lhs = rhs` — a bare assignment as the arm body. parse_expr stops
        // at `=`, so detect an `ident =` (not `==`) and parse the assignment.
        if self.at(b'=') && !self.starts_with(b"==") {
            if let Node::Lit(Literal::Ident(name), _) = &expr {
                let name = name.clone();
                self.pos += 1; // consume '='
                self.skip_ws_and_newlines();
                let value = self.parse_expr()?;
                return Ok(Node::Assign {
                    name,
                    value: Box::new(value),
                    span: Span::new(arm_start, self.pos),
                });
            }
        }
        Ok(expr)
    }

    /// Parse a single match-arm pattern.
    ///
    /// Patterns recognised (in order):
    /// - `_`            — wildcard
    /// - string literal — `"hello"`
    /// - `-N`           — negative integer
    /// - `N[.N]`        — positive numeric literal
    /// - `true`/`false` — boolean literals (mapped to Int(1/0))
    /// - `Path::Variant[(sub_patterns)]` — enum variant
    /// - bare `ident`   — binding (matches anything)
    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        self.skip_ws_and_newlines();
        if self.at_keyword(b"_") {
            self.pos += 1;
            return Ok(Pattern::Wildcard);
        }
        if self.at(b'"') {
            self.pos += 1;
            // DECODE escapes (shared with `parse_string_lit`) so a `"\n"` pattern
            // stores the single byte 0x0A — matching how the scrutinee's string
            // literal was decoded. The prior raw-slice copy stored `\` `n`
            // (2 bytes) and so never matched a decoded newline.
            let s = self.decode_string_body();
            if !self.eat(b'"') {
                return Err(self.err("unterminated string pattern".into()));
            }
            return Ok(Pattern::Literal(Literal::Str(s)));
        }
        if self.at(b'-') {
            self.pos += 1;
            self.skip_ws();
            let d = self
                .digits()
                .ok_or_else(|| self.err("expected digits after `-` in pattern".into()))?;
            let v = self.parse_i64_pattern(&d)?;
            return Ok(Pattern::Literal(Literal::Int(-v)));
        }
        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            // Radix-prefixed integer literal pattern: `0x..`/`0o..`/`0b..`.
            // A pattern matches on the VALUE, so the optional integer type
            // suffix (`u8`/`i8`/…) is consumed and discarded (width is purely
            // annotation here). Mirrors the expression-position radix path so
            // `match c { 0x00u8 => …, 0b10i8 => … }` parse. Additive: the
            // keystone uses no radix/suffixed literal patterns → byte-identical.
            if self.b[self.pos] == b'0'
                && self
                    .b
                    .get(self.pos + 1)
                    .is_some_and(|c| matches!(c.to_ascii_lowercase(), b'x' | b'o' | b'b'))
            {
                let radix: u32 = match self.b[self.pos + 1].to_ascii_lowercase() {
                    b'x' => 16,
                    b'o' => 8,
                    _ => 2,
                };
                self.pos += 2; // consume `0x` / `0o` / `0b`
                let dstart = self.pos;
                while self.pos < self.b.len()
                    && (self.b[self.pos] == b'_' || (self.b[self.pos] as char).is_digit(radix))
                {
                    self.pos += 1;
                }
                let raw: String = self.b[dstart..self.pos]
                    .iter()
                    .filter(|&&c| c != b'_')
                    .map(|&c| c as char)
                    .collect();
                if raw.is_empty() {
                    return Err(
                        self.err("expected digits after integer radix prefix in pattern".into())
                    );
                }
                let v = i64::from_str_radix(&raw, radix)
                    .or_else(|_| u64::from_str_radix(&raw, radix).map(|u| u as i64))
                    .map_err(|_| self.err("integer overflow in pattern".into()))?;
                let _ = self.int_type_suffix(); // discard a trailing width suffix
                return Ok(Pattern::Literal(Literal::Int(v)));
            }
            let d = self.digits().unwrap();
            if self.at(b'.') && self.b.get(self.pos + 1).is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
                let frac = self.digits().unwrap_or_default();
                let s = format!("{d}.{frac}");
                let f: f64 = s
                    .parse()
                    .map_err(|_| self.err("float parse error in pattern".into()))?;
                return Ok(Pattern::Literal(Literal::Float(f)));
            }
            let v = self.parse_i64_pattern(&d)?;
            let _ = self.int_type_suffix(); // discard a trailing width suffix (e.g. `0u8`)
            return Ok(Pattern::Literal(Literal::Int(v)));
        }
        // A leading `(` with no preceding name is a tuple pattern:
        // `(a, b)`, or nested inside a variant payload `Ok((p1, decorators))`
        // (enum_match #9). A single `(p)` collapses to its inner pattern
        // (grouping parens), matching how a 1-element tuple is just the value.
        if self.at(b'(') {
            self.pos += 1;
            let mut elems = Vec::new();
            self.skip_ws_and_newlines();
            while !self.at(b')') && !self.at_end() {
                elems.push(self.parse_pattern()?);
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
            }
            self.expect(b')')?;
            if elems.len() == 1 {
                return Ok(elems.into_iter().next().unwrap());
            }
            return Ok(Pattern::Tuple(elems));
        }
        let mut name = self
            .dotted_ident()
            .ok_or_else(|| self.err("expected pattern".into()))?;
        match name.as_str() {
            "true" => return Ok(Pattern::Literal(Literal::Int(1))),
            "false" => return Ok(Pattern::Literal(Literal::Int(0))),
            _ => {}
        }
        // `Enum.Variant` (dot) → canonical `Enum::Variant` when the first segment
        // is a declared enum, so the variant-pattern handling below treats it as
        // a variant rather than a dotted binding name (mirrors the value side).
        if let Some(dot_idx) = name.find('.') {
            if self.is_enum_name(&name[..dot_idx]) {
                name = format!("{}::{}", &name[..dot_idx], &name[dot_idx + 1..]);
            }
        }
        // A payload list `(...)` makes this a variant pattern regardless of
        // qualification: `Ok(x)`, `Err(e)`, `Some(v)` (unqualified) AND
        // `Result::Ok(x)`, `Mode::On(p)` (qualified). The sub-patterns bind the
        // variant payload (enum_match #9). Previously only `::`-qualified names
        // got this, so unqualified `Err(e)` left the `(e)` unconsumed and the
        // arm parser then saw `(` instead of `=>`.
        self.skip_ws();
        if self.at(b'(') {
            self.pos += 1;
            let mut sub = Vec::new();
            self.skip_ws_and_newlines();
            while !self.at(b')') && !self.at_end() {
                sub.push(self.parse_pattern()?);
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
            }
            self.expect(b')')?;
            return Ok(Pattern::EnumVariant {
                path: name,
                args: sub,
            });
        }
        // A `{ field, field: pat }` body makes this a STRUCT-variant pattern
        // `E.V { w, h }` / `E.V { w: pat }` — the named fields bind the variant's
        // declared slots (resolved at lower time via the enum's field_names).
        // Shorthand `{ f }` binds `f`. (enum_match #9 struct variants.)
        if self.at(b'{') {
            self.pos += 1;
            let mut fields = Vec::new();
            self.skip_ws_and_newlines();
            while !self.at(b'}') && !self.at_end() {
                // `..` rest pattern — bind the listed fields, ignore the rest
                // (`E.V { span, .. }`). Unmentioned fields already lower to a
                // `Wildcard` slot, so this only needs to be accepted + skipped.
                if self.at(b'.') && self.b.get(self.pos + 1) == Some(&b'.') {
                    self.pos += 2;
                    self.skip_ws_and_newlines();
                    // `..` must be the LAST element; otherwise the `expect('}')`
                    // below would fail with a confusing "found ','" message.
                    if !self.at(b'}') {
                        return Err(self.err(
                            "`..` rest pattern must be last in a struct-variant pattern".into(),
                        ));
                    }
                    break;
                }
                let fname = self
                    .word()
                    .ok_or_else(|| {
                        self.err("expected field name in struct-variant pattern".into())
                    })?
                    .to_string();
                self.skip_ws();
                let sub = if self.eat(b':') {
                    self.skip_ws_and_newlines();
                    self.parse_pattern()?
                } else {
                    // Shorthand `{ f }` binds a variable named `f`.
                    Pattern::Ident(fname.clone())
                };
                fields.push((fname, sub));
                self.skip_ws_and_newlines();
                if !self.eat(b',') {
                    break;
                }
                self.skip_ws_and_newlines();
            }
            self.expect(b'}')?;
            return Ok(Pattern::EnumStruct { path: name, fields });
        }
        // No payload: a `::`-qualified name is a unit variant (`Mode::On`); a
        // bare lowercase-or-any name binds the whole scrutinee (`x`).
        if name.contains("::") {
            return Ok(Pattern::EnumVariant {
                path: name,
                args: Vec::new(),
            });
        }
        Ok(Pattern::Ident(name))
    }
}

/// Scan an attribute list for `[reap_threshold(t)]` and return `Some(t)`.
///
/// Threshold must be a float literal in `[0.0, 1.0)`. Values outside that
/// range or unparseable values are ignored. When multiple `reap_threshold`
/// attributes are present, the last one wins.
fn extract_reap_threshold(attrs: &[crate::ast::Attribute]) -> Option<f64> {
    let mut result: Option<f64> = None;
    for attr in attrs {
        if attr.name == "reap_threshold" {
            if let Some(raw) = attr.args.first() {
                if let Ok(v) = raw.parse::<f64>() {
                    if (0.0..1.0).contains(&v) {
                        result = Some(v);
                    }
                }
            }
        }
    }
    result
}

/// Return `true` when the attribute list contains a `[test]` attribute.
///
/// RFC 0008 Phase B. Only `[test]` is recognized on `fn` declarations in this
/// phase; `[bench]`, `[ignore]`, and `[should_panic]` are future work.
fn extract_is_test(attrs: &[crate::ast::Attribute]) -> bool {
    attrs.iter().any(|a| a.name == "test")
}

pub fn parse(input: &str) -> Result<Module, Vec<ParseError>> {
    let stripped_owned: String;
    let src: &str = if input.contains("//") {
        stripped_owned = strip_comments_with_trivia(input, &mut None).0;
        &stripped_owned
    } else {
        input
    };
    let mut p = P::new(src);
    match p.parse_module() {
        // Single expansion chokepoint (raw-error adapter). Every
        // module-CONSUMING front-end that takes raw `ParseError`s — the project
        // module loaders, the stdlib loader, the `mindc test` runner, and the
        // `mindc check` type-check + cross-module-table paths — parses here, so
        // `#[bimap]` derives land uniformly. The pretty-diagnostic adapter
        // `parse_with_diagnostics_in_file` shares the SAME `expand_bimap`
        // implementation; `parse_with_trivia` deliberately opts OUT so
        // `fmt`/`doc`/`lint` see the raw (un-expanded) attribute, never
        // synthesised fns re-emitted into user source (the excluded
        // source-mutation hazard). Rejected alternative — wiring each front-end
        // (build/run_eval_once/REPL/check/lower_to_ir) individually — is the
        // two-sources-of-truth smell one level up.
        Ok(mut m) => {
            let diags = expand_bimap::expand_bimap(&mut m, src, None);
            if diags.is_empty() {
                Ok(m)
            } else {
                Err(diags
                    .into_iter()
                    .map(|d| ParseError {
                        offset: 0,
                        message: format!("{}: {}", d.code, d.message),
                    })
                    .collect())
            }
        }
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
    let stripped_owned: String;
    let src: &str = if input.contains("//") {
        stripped_owned = strip_comments_with_trivia(input, &mut None).0;
        &stripped_owned
    } else {
        input
    };
    let mut p = P::new(src);
    match p.parse_module() {
        // Single expansion chokepoint (pretty-diagnostic adapter). Every
        // module-CONSUMING front-end that takes `PrettyDiagnostic`s — the build
        // pipeline, `run_eval_once`, the REPL, and the project single-module
        // load path — parses here. Shares the SAME `expand_bimap`
        // implementation as the raw-error adapter `parse`; `parse_with_trivia`
        // opts OUT (fmt/doc/lint). The E2017-E2020 bijectivity diagnostics carry
        // their real E-code + span here (the raw adapter folds the code into the
        // message string, `ParseError` having no code field).
        Ok(mut m) => {
            let diags = expand_bimap::expand_bimap(&mut m, src, file);
            if diags.is_empty() {
                Ok(m)
            } else {
                Err(diags)
            }
        }
        Err(e) => {
            let diag = PrettyDiagnostic {
                phase: "parse",
                code: "E1001",
                severity: crate::diagnostics::Severity::Error,
                message: e.message,
                span: Some(DiagnosticSpan::from_offsets(
                    src,
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

/// Parse `input`, collecting trivia (comments and blank lines) in addition to
/// the AST.
///
/// Returns `(module, trivia_stream)` where `module` is identical to what
/// [`parse`] returns for the same input, and `trivia_stream` contains one
/// [`Trivia`] record per `//` comment, `///` doc-comment, and blank line in the
/// original source, in source order.
///
/// The `module` field is unaffected by trivia collection — its `Node` tree is
/// byte-identical to the result of `parse(input)`.
pub fn parse_with_trivia(input: &str) -> Result<(Module, TriviaStream), Vec<ParseError>> {
    // All trivia is collected during the comment-stripping pass, before the
    // parser sees the source.  This is zero-overhead for regular `parse` calls
    // (which pass `None` and take the fast path).
    let mut collector = Some(TriviaCollector::new());
    let (stripped, _offset_map) = strip_comments_with_trivia(input, &mut collector);

    let stream = collector
        .expect("collector must be Some after strip_comments_with_trivia")
        .into_stream();

    let mut p = P::new(&stripped);
    match p.parse_module() {
        Ok(m) => Ok((m, stream)),
        Err(e) => Err(vec![e]),
    }
}
