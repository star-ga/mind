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

//! Thread-safe string interning for symbolic dimensions.
//!
//! This module provides a global string interner that converts dynamic strings
//! into `&'static str` references without memory leaks. Interned strings are
//! never freed, but each unique string is stored only once.
//!
//! # Example
//!
//! ```
//! use libmind::types::intern::intern_str;
//!
//! let s1 = intern_str("batch_size");
//! let s2 = intern_str("batch_size");
//! assert!(std::ptr::eq(s1, s2)); // Same pointer
//! ```

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Global string interner.
static INTERNER: OnceLock<Mutex<StringInterner>> = OnceLock::new();

/// Maximum interned strings before refusing new entries (DoS prevention).
const MAX_INTERNED_STRINGS: usize = 100_000;

/// Thread-safe string interner.
struct StringInterner {
    strings: HashSet<&'static str>,
}

impl StringInterner {
    fn new() -> Self {
        Self {
            strings: HashSet::new(),
        }
    }

    /// Intern a string, returning a `&'static str`.
    ///
    /// If the string is already interned, returns the existing reference.
    /// If the interner is at capacity (DoS protection), returns None for fail-fast.
    fn intern(&mut self, s: &str) -> Option<&'static str> {
        // Check if already interned
        if let Some(&existing) = self.strings.get(s) {
            return Some(existing);
        }

        // Security: Enforce maximum interned strings to prevent memory DoS
        if self.strings.len() >= MAX_INTERNED_STRINGS {
            eprintln!(
                "[ERROR] String interner at capacity ({}), refusing to intern '{}' - FAIL FAST",
                MAX_INTERNED_STRINGS, s
            );
            return None; // Fail-fast: caller must handle this
        }

        // Allocate new static string
        let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
        self.strings.insert(leaked);
        Some(leaked)
    }
}

/// Intern a string, returning a `&'static str`.
///
/// This function is thread-safe. If the string was previously interned,
/// the existing reference is returned. Otherwise, a new static allocation
/// is made and stored.
///
/// # Returns
///
/// Returns "?" if the interner capacity is exceeded. For explicit error
/// handling, use `try_intern_str` which returns `None` on capacity exhaustion.
///
/// # Example
///
/// ```
/// use libmind::types::intern::intern_str;
///
/// let s = intern_str("N");
/// assert_eq!(s, "N");
/// ```
pub fn intern_str(s: &str) -> &'static str {
    // For backwards compatibility, return "?" on capacity exceeded
    // Callers requiring strict behavior should use try_intern_str
    try_intern_str(s).unwrap_or("?")
}

/// Try to intern a string, returning `None` if capacity is exceeded.
///
/// This is the fallible version of `intern_str` for callers that want
/// to handle capacity exhaustion gracefully.
///
/// # Example
///
/// ```
/// use libmind::types::intern::try_intern_str;
///
/// let s = try_intern_str("N");
/// assert_eq!(s, Some("N"));
/// ```
pub fn try_intern_str(s: &str) -> Option<&'static str> {
    // Check for common compile-time known symbols first (fast path)
    match s {
        "?" => return Some("?"),
        "N" => return Some("N"),
        "B" => return Some("B"),
        "C" => return Some("C"),
        "H" => return Some("H"),
        "W" => return Some("W"),
        "T" => return Some("T"),
        "batch" => return Some("batch"),
        "seq" => return Some("seq"),
        "hidden" => return Some("hidden"),
        _ => {}
    }

    let interner = INTERNER.get_or_init(|| Mutex::new(StringInterner::new()));

    // Use lock_or_recover pattern for poison safety
    let mut guard = match interner.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    guard.intern(s)
}

/// Get statistics about the string interner.
#[allow(dead_code)]
pub fn interner_stats() -> (usize, usize) {
    let interner = INTERNER.get_or_init(|| Mutex::new(StringInterner::new()));

    let guard = match interner.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    let count = guard.strings.len();
    let bytes: usize = guard.strings.iter().map(|s| s.len()).sum();
    (count, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_same_string() {
        let s1 = intern_str("test_symbol");
        let s2 = intern_str("test_symbol");
        assert!(std::ptr::eq(s1, s2));
    }

    #[test]
    fn test_intern_different_strings() {
        let s1 = intern_str("alpha");
        let s2 = intern_str("beta");
        assert!(!std::ptr::eq(s1, s2));
        assert_eq!(s1, "alpha");
        assert_eq!(s2, "beta");
    }

    #[test]
    fn test_intern_fast_path() {
        let s1 = intern_str("?");
        let s2 = intern_str("N");
        assert_eq!(s1, "?");
        assert_eq!(s2, "N");
    }

    #[test]
    fn test_interner_stats() {
        intern_str("unique_test_1");
        intern_str("unique_test_2");
        let (count, _bytes) = interner_stats();
        assert!(count >= 2);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_interner_capacity_documented() {
        // Verify that the interner has a documented capacity limit
        // This is a documentation test to ensure the limit exists
        assert!(super::MAX_INTERNED_STRINGS > 0);
        assert!(
            super::MAX_INTERNED_STRINGS <= 1_000_000,
            "Limit should be reasonable"
        );
    }
}
