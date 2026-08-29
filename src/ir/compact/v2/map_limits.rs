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

//! mic@2.1 MAP limits (§3.2 key grammar + §3.5 size bounds) — **one definition,
//! two enforcement sites**.
//!
//! The `Map` data model has two serializations, text (`v2/parse.rs`) and binary
//! (`v2/binary.rs`), and they must accept exactly the same set of `Map` values.
//! They did not: every rule below lived only in the text parser, so a MIC-B blob
//! could carry a MAP the text parser would refuse. That is not a cosmetic
//! asymmetry — it breaks the project's own round trip, because `emit_mic2`
//! writes MAP keys **raw and unescaped** (`v2/emit.rs`, `emit_map_block`):
//! a binary-accepted key such as `a = 1\ninjected.key` re-emits as *two* valid
//! text entries where the blob carried one, so `binary → Map → text → Map` is
//! not the identity. `parse_micb` is a public re-export documented as accepting
//! untrusted input, so the caller cannot be assumed to pre-validate.
//!
//! Every check here is therefore called from **both** parsers. Adding a limit
//! means editing this file; it must never be re-stated at a call site.
//!
//! ## Where each check is enforced
//!
//! | Rule | Text site | Binary site |
//! |------|-----------|-------------|
//! | [`check_map_key`] | `parse_map_entries` | `decode_map_entries` |
//! | [`check_map_entry_count`] | `parse_map_entries` | `decode_map_entries` |
//! | [`check_map_nesting_depth`] | `parse_map_entries` (descend) | `decode_map_value` (descend) |
//! | [`check_map_string_len`] | `parse_map_string_value` | `decode_map_value` tag 0 |
//! | [`check_map_bytes_len`] | `parse_map_bytes_value` | `decode_map_value` tag 2 |
//!
//! ## Units
//!
//! Both size limits are stated over the **decoded value**, not its wire
//! encoding, because that is the only quantity the two serializations share.
//! Measuring the text side's *escaped* bytes instead would make it strictly
//! stricter than the binary side for escape-heavy strings (the emitter expands
//! a control character to a 6-byte `\uXXXX`), reintroducing exactly the
//! text-refuses / binary-accepts gap this module exists to close.

/// §3.5: a MAP key is at most 256 bytes.
pub(super) const MAX_MAP_KEY_BYTES: usize = 256;

/// §3.5: a MAP key has at most 8 dot-separated segments.
pub(super) const MAX_MAP_KEY_SEGMENTS: usize = 8;

/// §3.5: a MAP subtree holds at most 4096 entries (counted recursively).
pub(super) const MAX_MAP_ENTRIES: usize = 4096;

/// §3.5: maps nest at most 4 levels below the top-level `map { … }` block.
///
/// The top-level map is level 0, so a map may exist at levels 0..=4 and a
/// descent *from* level 4 is refused.
pub(super) const MAX_MAP_NESTING: usize = 4;

/// §3.5: a MAP string value is at most 64 KiB of decoded UTF-8.
pub(super) const MAX_MAP_STRING_BYTES: usize = 64 * 1024;

/// §3.5: a MAP bytes value is at most 1 MiB of decoded bytes.
pub(super) const MAX_MAP_BYTES: usize = 1024 * 1024;

/// Validate a MAP key against the §3.2 grammar (`ident ("." ident)*`) and the
/// §3.5 key bounds, in that order.
///
/// The syntax half is the load-bearing one for the text round trip: `emit_mic2`
/// emits keys verbatim, so any key containing `\n`, `=`, `"`, `#`, a space, or
/// a leading/trailing/doubled `.` produces text that does not parse back to the
/// same `Map` — or, worse, parses back to a *different* valid `Map`.
///
/// Returns the complete diagnostic; callers only wrap it in their own error
/// type so that the two parsers report identical text.
pub(super) fn check_map_key(key: &str) -> Result<(), String> {
    if let Err(reason) = validate_map_key_syntax(key) {
        return Err(format!("invalid MAP key '{key}': {reason}"));
    }
    if key.len() > MAX_MAP_KEY_BYTES {
        return Err(format!(
            "MAP key too long ({} bytes, max {}): {}",
            key.len(),
            MAX_MAP_KEY_BYTES,
            key
        ));
    }
    if key.split('.').count() > MAX_MAP_KEY_SEGMENTS {
        return Err(format!(
            "MAP key depth exceeds {MAX_MAP_KEY_SEGMENTS} segments: {key}"
        ));
    }
    Ok(())
}

/// §3.5 total entry count. `count` is the number of entries the subtree would
/// hold once the entry being parsed is added, counted recursively.
pub(super) fn check_map_entry_count(count: usize) -> Result<(), String> {
    if count > MAX_MAP_ENTRIES {
        return Err(format!("MAP entry count exceeds limit {MAX_MAP_ENTRIES}"));
    }
    Ok(())
}

/// §3.5 nesting limit, checked at the point of **descending** into a nested map.
/// `depth` is the level of the map currently being parsed (top level = 0).
///
/// Both parsers are mutually recursive through the nested-map arm, so this is
/// also the stack bound on untrusted input: the binary decoder previously had
/// no depth check at all, and `MAX_MICB_INPUT` bounds total *size* while saying
/// nothing about *depth*, so a small deeply-nested blob drove the recursion
/// until the stack aborted the process.
pub(super) fn check_map_nesting_depth(depth: usize) -> Result<(), String> {
    if depth >= MAX_MAP_NESTING {
        return Err(format!("MAP nesting depth exceeds limit {MAX_MAP_NESTING}"));
    }
    Ok(())
}

/// §3.5 string-value size limit, over the decoded UTF-8 byte length.
pub(super) fn check_map_string_len(len: usize) -> Result<(), String> {
    if len > MAX_MAP_STRING_BYTES {
        return Err(format!("MAP string value exceeds 64 KiB: {len} bytes"));
    }
    Ok(())
}

/// §3.5 bytes-value size limit, over the decoded byte length.
pub(super) fn check_map_bytes_len(len: usize) -> Result<(), String> {
    if len > MAX_MAP_BYTES {
        return Err(format!("bytes() value exceeds 1 MiB: {len} bytes"));
    }
    Ok(())
}

/// §3.2 `map_key` grammar: `ident ("." ident)*`, `ident = [A-Za-z_][A-Za-z0-9_]*`.
fn validate_map_key_syntax(key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err("key is empty".into());
    }
    for segment in key.split('.') {
        if segment.is_empty() {
            return Err("key segment is empty (double dot or leading/trailing dot)".into());
        }
        let mut chars = segment.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            Some(c) => {
                return Err(format!(
                    "segment '{segment}' starts with invalid char '{c}'"
                ));
            }
            None => return Err("empty segment".into()),
        }
        for c in chars {
            if !c.is_ascii_alphanumeric() && c != '_' {
                return Err(format!("segment '{segment}' contains invalid char '{c}'"));
            }
        }
    }
    Ok(())
}
