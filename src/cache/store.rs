// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Cache backends. The in-memory store is always available; a disk-backed
//! store is implemented behind the `cache-disk` feature so the core
//! compiler doesn't take a hard dep on the filesystem code path.

use super::entry::{CacheEntry, CacheKey, CacheStats};
use std::collections::HashMap;

pub trait CacheStore: Send + Sync {
    fn get(&self, key: &CacheKey) -> Option<CacheEntry>;
    fn put(&mut self, key: CacheKey, entry: CacheEntry);
    fn clear(&mut self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn stats(&self) -> CacheStats;
}

/// Process-local in-memory store, mostly useful for tests and for the
/// hot front-line cache that sits in front of any persistent backend.
pub struct MemoryStore {
    inner: std::sync::Mutex<MemoryInner>,
}

struct MemoryInner {
    entries: HashMap<CacheKey, CacheEntry>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self {
            inner: std::sync::Mutex::new(MemoryInner {
                entries: HashMap::new(),
                hits: 0,
                misses: 0,
                evictions: 0,
            }),
        }
    }
}

impl CacheStore for MemoryStore {
    fn get(&self, key: &CacheKey) -> Option<CacheEntry> {
        let mut inner = self.inner.lock().expect("memory cache poisoned");
        if let Some(entry) = inner.entries.get(key).cloned() {
            inner.hits += 1;
            Some(entry)
        } else {
            inner.misses += 1;
            None
        }
    }

    fn put(&mut self, key: CacheKey, entry: CacheEntry) {
        let mut inner = self.inner.lock().expect("memory cache poisoned");
        if inner.entries.insert(key, entry).is_some() {
            inner.evictions += 1;
        }
    }

    fn clear(&mut self) {
        let mut inner = self.inner.lock().expect("memory cache poisoned");
        let n = inner.entries.len();
        inner.entries.clear();
        inner.evictions = inner.evictions.saturating_add(n as u64);
    }

    fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("memory cache poisoned")
            .entries
            .len()
    }

    fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().expect("memory cache poisoned");
        let total_bytes: u64 = inner.entries.values().map(|e| e.size_bytes() as u64).sum();
        CacheStats {
            hits: inner.hits,
            misses: inner.misses,
            live_entries: inner.entries.len() as u64,
            evictions: inner.evictions,
            total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::fingerprint::ProfileTag;

    fn key(s: &str) -> CacheKey {
        CacheKey::new("0.2.6", ProfileTag::Default, s, "imports")
    }

    #[test]
    fn put_then_get_returns_entry() {
        let mut s = MemoryStore::default();
        s.put(key("a"), CacheEntry::new(vec![1, 2, 3], 0));
        let g = s.get(&key("a")).expect("hit");
        assert_eq!(g.ir_bytes, vec![1, 2, 3]);
    }

    #[test]
    fn get_missing_increments_miss() {
        let s = MemoryStore::default();
        assert!(s.get(&key("none")).is_none());
        let st = s.stats();
        assert_eq!(st.hits, 0);
        assert_eq!(st.misses, 1);
    }

    #[test]
    fn put_overwrite_counts_eviction() {
        let mut s = MemoryStore::default();
        s.put(key("a"), CacheEntry::new(vec![1], 0));
        s.put(key("a"), CacheEntry::new(vec![2], 0));
        assert_eq!(s.stats().evictions, 1);
        assert_eq!(s.get(&key("a")).unwrap().ir_bytes, vec![2]);
    }

    #[test]
    fn clear_drops_everything_and_records_evictions() {
        let mut s = MemoryStore::default();
        s.put(key("a"), CacheEntry::new(vec![1], 0));
        s.put(key("b"), CacheEntry::new(vec![1], 0));
        s.clear();
        assert!(s.is_empty());
        let st = s.stats();
        assert!(st.evictions >= 2);
    }

    #[test]
    fn total_bytes_matches_entries() {
        let mut s = MemoryStore::default();
        s.put(key("a"), CacheEntry::new(vec![0u8; 16], 0));
        s.put(key("b"), CacheEntry::new(vec![0u8; 32], 0));
        assert_eq!(s.stats().total_bytes, 48);
    }
}
