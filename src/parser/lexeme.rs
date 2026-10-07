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

//! Lexeme readers: identifiers, dotted paths, digit runs and integer-type
//! suffixes. Each returns text borrowed from the source where it can — the
//! source is already a `&str`, and every reader starts and stops at an ASCII
//! byte, so a slice is on char boundaries without re-validating UTF-8.

use super::P;
use crate::ast::TypeAnn;

impl<'a> P<'a> {
    /// The source between two byte offsets that sit at ASCII bytes (or the end).
    pub(super) fn text(&self, start: usize, end: usize) -> &'a str {
        &self.src[start..end]
    }

    pub(super) fn is_ident_start(ch: u8) -> bool {
        ch.is_ascii_alphabetic() || ch == b'_'
    }

    pub(super) fn is_ident_cont(ch: u8) -> bool {
        ch.is_ascii_alphanumeric() || ch == b'_'
    }

    /// Read an identifier word (no dots). Returns None if not at an ident.
    pub(super) fn word(&mut self) -> Option<&'a str> {
        let start = self.pos;
        if self.pos >= self.b.len() || !Self::is_ident_start(self.b[self.pos]) {
            return None;
        }
        while self.pos < self.b.len() && Self::is_ident_cont(self.b[self.pos]) {
            self.pos += 1;
        }
        Some(self.text(start, self.pos))
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
    pub(super) fn int_type_suffix(&mut self) -> Option<TypeAnn> {
        // Fast path: an integer type suffix can only begin with `u` or `i`. For
        // the overwhelmingly common UNSUFFIXED literal the next byte is
        // whitespace, an operator, `)`, `,`, `;`, … — bail before the 8-way
        // compare so unsuffixed literals (incl. the entire keystone) pay ~one
        // byte check, keeping compile_small at the nanosecond floor.
        if self.pos >= self.b.len() || !matches!(self.b[self.pos], b'u' | b'i') {
            return None;
        }
        // Each candidate suffix and the `TypeAnn` it maps to. `u32`/`i32`/`i64`
        // have dedicated scalar variants; the remaining widths (incl. the
        // pointer-sized `usize`/`isize`, #263 surface 2 — `0usize` in a match
        // arm) ride through the `Named` path (same as writing `as u64`). The
        // type checker already recognises `usize`/`isize` as integer-class
        // named scalars, so the desugared `as`-cast type-checks unchanged. Order
        // is irrelevant: the word-boundary check rejects a shorter prefix like
        // `u8` against `usize` (the `s` after `u` is an ident-cont byte).
        const SUFFIXES: &[&str] = &[
            "usize", "isize", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64",
        ];
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
    ///
    /// Phase 10.6: `Type::Variant` path segments (`config.AddressingMode::Content`,
    /// `Side::Left`, etc.) are accepted too, so the expression node carries the
    /// full path string and the type checker resolves it later. The segments are
    /// contiguous in the source, so the name is that source slice, taken in one
    /// exact-size allocation once the scan ends.
    pub(super) fn dotted_ident(&mut self) -> Option<String> {
        let start = self.pos;
        self.word()?;
        loop {
            let saved = self.pos;
            if self.at(b'.') {
                self.pos += 1;
            } else if self.starts_with(b"::") {
                self.pos += 2;
            } else {
                break;
            }
            if self.word().is_none() {
                self.pos = saved;
                break;
            }
        }
        Some(self.text(start, self.pos).to_string())
    }

    /// Read a run of ASCII digits, borrowed from the source.
    pub(super) fn digits(&mut self) -> Option<&'a str> {
        let start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return None;
        }
        Some(self.text(start, self.pos))
    }
}
