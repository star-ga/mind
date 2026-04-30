// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Incremental compilation + content-addressed result cache.
//!
//! The cache sits between the source loader and the parser/typechecker. A
//! compilation request is keyed by:
//!   * compiler version
//!   * profile (default | systems | embedded)
//!   * SHA-256 of source bytes
//!   * SHA-256 of every transitive import
//!
//! If the same key has been seen before the cached IR is returned and the
//! parse / typecheck / shape-check / IR-build pipeline is skipped. This is
//! the path that closes the gap between the cold-start frontend latency
//! (1.8–15.5 µs) and the warm-start target (< 1 µs).
//!
//! Storage layout under the cache root (default `$XDG_CACHE_HOME/mindc/`):
//!
//! ```text
//! cache_root/
//! ├── meta.json                    # cache header + compiler fingerprint
//! └── objects/
//!     ├── <hash[..2]>/
//!     │   └── <hash[2..]>          # bincode-encoded CacheEntry
//! ```
//!
//! Eviction is LRU by access time recorded in `meta.json`. A single mindc
//! process holds an in-memory layer in front of the on-disk objects so a
//! repeated build of the same module hits memory, not the filesystem.

pub mod entry;
pub mod fingerprint;
pub mod store;

pub use entry::{CacheEntry, CacheKey, CacheStats};
pub use fingerprint::{Fingerprint, ProfileTag};
pub use store::{CacheStore, MemoryStore};

/// Public façade — the type the rest of mindc holds.
pub struct CompilationCache {
    inner: Box<dyn CacheStore>,
}

impl CompilationCache {
    pub fn new(store: impl CacheStore + 'static) -> Self {
        Self {
            inner: Box::new(store),
        }
    }

    pub fn in_memory() -> Self {
        Self::new(MemoryStore::default())
    }

    pub fn lookup(&self, key: &CacheKey) -> Option<CacheEntry> {
        self.inner.get(key)
    }

    pub fn insert(&mut self, key: CacheKey, entry: CacheEntry) {
        self.inner.put(key, entry);
    }

    pub fn stats(&self) -> CacheStats {
        self.inner.stats()
    }

    pub fn clear(&mut self) {
        self.inner.clear()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_in_memory_cache() {
        let mut c = CompilationCache::in_memory();
        let key = CacheKey::new(
            "0.2.6",
            ProfileTag::Default,
            "deadbeef",
            "cafebabe",
        );
        let entry = CacheEntry::new(b"<ir bytes>".to_vec(), 1234);
        c.insert(key.clone(), entry.clone());
        let got = c.lookup(&key).expect("must hit cache");
        assert_eq!(got.ir_bytes, entry.ir_bytes);
    }

    #[test]
    fn cache_miss_returns_none() {
        let c = CompilationCache::in_memory();
        let key = CacheKey::new("0.2.6", ProfileTag::Default, "x", "y");
        assert!(c.lookup(&key).is_none());
    }

    #[test]
    fn stats_increment() {
        let mut c = CompilationCache::in_memory();
        let key = CacheKey::new("0.2.6", ProfileTag::Default, "x", "y");
        c.insert(key.clone(), CacheEntry::new(vec![1, 2, 3], 0));
        let _ = c.lookup(&key);
        let _ = c.lookup(&CacheKey::new("0.2.6", ProfileTag::Default, "z", "y"));
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!(s.live_entries >= 1);
    }

    #[test]
    fn clear_drops_entries() {
        let mut c = CompilationCache::in_memory();
        c.insert(
            CacheKey::new("0.2.6", ProfileTag::Default, "x", "y"),
            CacheEntry::new(vec![1], 0),
        );
        assert!(!c.is_empty());
        c.clear();
        assert!(c.is_empty());
    }
}
