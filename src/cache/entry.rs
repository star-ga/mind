// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Cache key + cached value types.

use super::fingerprint::ProfileTag;

/// Cache lookup key. Two keys are equal iff every field matches — that means
/// a compiler bump, profile change, or source/import edit all force a miss.
///
/// `compiler_version` must carry the compiler's **binary identity**, not just
/// its semver: a dev rebuild of `mindc` at the same `CARGO_PKG_VERSION` emits
/// potentially different IR, so callers must store the combined
/// `"<version>+<binary-identity>"` string (see
/// [`crate::build::cache::compiler_identity_string`]) — a bare
/// `CARGO_PKG_VERSION` would silently serve a stale entry (issue #96).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Combined compiler identity: `"<CARGO_PKG_VERSION>+<binary-identity>"`
    /// (see [`crate::build::cache::compiler_identity_string`]). A bare semver
    /// here reintroduces the issue #96 staleness bug.
    pub compiler_version: String,
    pub profile: ProfileTag,
    pub source_hash: String,
    pub imports_hash: String,
}

impl CacheKey {
    pub fn new(
        compiler_version: impl Into<String>,
        profile: ProfileTag,
        source_hash: impl Into<String>,
        imports_hash: impl Into<String>,
    ) -> Self {
        Self {
            compiler_version: compiler_version.into(),
            profile,
            source_hash: source_hash.into(),
            imports_hash: imports_hash.into(),
        }
    }

    /// Stable string representation used by the disk store.
    pub fn render(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.compiler_version, self.profile, self.source_hash, self.imports_hash
        )
    }
}

/// Value stored under a cache key — bincode-encoded IR plus build metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    pub ir_bytes: Vec<u8>,
    pub built_at_unix_seconds: u64,
}

impl CacheEntry {
    pub fn new(ir_bytes: Vec<u8>, built_at_unix_seconds: u64) -> Self {
        Self {
            ir_bytes,
            built_at_unix_seconds,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.ir_bytes.len()
    }
}

/// Snapshot of cache health.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub live_entries: u64,
    pub evictions: u64,
    pub total_bytes: u64,
}

impl CacheStats {
    pub fn hit_ratio(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f32) / (total as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_equality_requires_every_field() {
        let a = CacheKey::new("0.2.6", ProfileTag::Default, "h1", "i1");
        let b = CacheKey::new("0.2.6", ProfileTag::Default, "h1", "i1");
        let c = CacheKey::new("0.2.6", ProfileTag::Systems, "h1", "i1");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn key_render_is_stable() {
        let k = CacheKey::new("0.2.6", ProfileTag::Default, "AAA", "BBB");
        assert_eq!(k.render(), "0.2.6:default:AAA:BBB");
    }

    #[test]
    fn entry_reports_size() {
        let e = CacheEntry::new(vec![0u8; 128], 1714000000);
        assert_eq!(e.size_bytes(), 128);
    }

    #[test]
    fn hit_ratio_handles_zero_traffic() {
        let s = CacheStats::default();
        assert_eq!(s.hit_ratio(), 0.0);
    }

    #[test]
    fn hit_ratio_correct() {
        let s = CacheStats {
            hits: 9,
            misses: 1,
            live_entries: 1,
            evictions: 0,
            total_bytes: 10,
        };
        assert!((s.hit_ratio() - 0.9).abs() < 1e-6);
    }
}
