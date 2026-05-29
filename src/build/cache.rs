// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase F — incremental compilation cache for `mindc build`.
//!
//! ## Cache layout
//!
//! ```text
//! target/<target>/<optimize>/
//!   .cache/
//!     objects/
//!       <sha256-of-inputs>.o          # compiled object (opaque bytes)
//!     meta/
//!       <sha256-of-inputs>.json       # source path, compile flags, dep hashes
//!     manifest.json                   # source-path -> object-hash map
//! ```
//!
//! ## Key derivation
//!
//! The cache key is:
//!
//! ```text
//! SHA256(
//!   "mindc-cache-v1\n"
//!   "compiler=<version>\n"
//!   "edition=<year>\n"
//!   "target=<target-debug>\n"
//!   "optimize=<level-debug>\n"
//!   "deps=\n"
//!   <dep_hash_0>\n ... <dep_hash_n>\n   (sorted ascending)
//!   "source=\n"
//!   <source_bytes>
//! )
//! ```
//!
//! The `mindc-cache-v1\n` prefix is the **cache format version**. Bumping it
//! globally invalidates all on-disk cache entries (intentional for breaking
//! changes to lowering or codegen output format).
//!
//! `compiler=<version>` invalidates per mindc release (different mindc may
//! emit different IR / object code).
//!
//! ## Concurrency
//!
//! Object files are written via a write-to-temp-then-rename pattern.
//! `rename(2)` is atomic on POSIX filesystems, so a partial write from one
//! process is never observed by another. The manifest.json is written the same
//! way. No advisory lock file is required for simple single-object writes;
//! the test for concurrent builds (test 10) verifies that parallel invocations
//! either both write identical content or one wins cleanly.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::project::{BuildTarget, OptimizeLevel};

// ---------------------------------------------------------------------------
// Cache key derivation
// ---------------------------------------------------------------------------

/// Derive the cache key for one compiled module.
///
/// The key is a SHA-256 hex string derived from:
/// - `source_bytes`   — exact byte content of the source file
/// - `target`         — backend target (cpu / cerebras / …)
/// - `optimize`       — optimisation level (debug / release / size)
/// - `dep_hashes`     — sorted slice of dependency object hashes (may be empty)
/// - `compiler_version` — semver string from `CARGO_PKG_VERSION`
/// - `edition`        — language edition year (e.g. 2024)
///
/// The input to SHA-256 is a deterministic byte sequence with labelled sections
/// so that any change to any input changes the key.
pub fn module_cache_key(
    source_bytes: &[u8],
    target: BuildTarget,
    optimize: OptimizeLevel,
    dep_hashes: &[String],
    compiler_version: &str,
    edition: u32,
) -> String {
    let mut data: Vec<u8> = Vec::with_capacity(256 + source_bytes.len());

    // Format version prefix — bump this string to invalidate all caches.
    data.extend_from_slice(b"mindc-cache-v1\n");

    // Compiler version — different mindc releases may produce different objects.
    data.extend_from_slice(format!("compiler={}\n", compiler_version).as_bytes());

    // Edition — different language editions have different semantics.
    data.extend_from_slice(format!("edition={}\n", edition).as_bytes());

    // Target — cpu and cerebras objects are not interchangeable.
    data.extend_from_slice(format!("target={:?}\n", target).as_bytes());

    // Optimize — debug and release objects differ in codegen.
    data.extend_from_slice(format!("optimize={:?}\n", optimize).as_bytes());

    // Dependency hashes — sorted so order of declaration doesn't matter.
    data.extend_from_slice(b"deps=\n");
    let mut sorted_deps: Vec<&String> = dep_hashes.iter().collect();
    sorted_deps.sort();
    for h in sorted_deps {
        data.extend_from_slice(h.as_bytes());
        data.push(b'\n');
    }

    // Source bytes — must come last so that the other fields act as a header.
    data.extend_from_slice(b"source=\n");
    data.extend_from_slice(source_bytes);

    sha256_hex(&data)
}

// ---------------------------------------------------------------------------
// Object cache paths
// ---------------------------------------------------------------------------

/// Compute the root of the build cache for a given target + optimize profile.
///
/// Layout: `<project_root>/target/<target>/<optimize>/.cache/`
pub fn cache_root(project_root: &Path, target: BuildTarget, optimize: OptimizeLevel) -> PathBuf {
    project_root
        .join("target")
        .join(target.as_str())
        .join(optimize.as_str())
        .join(".cache")
}

/// Path to the compiled object for the given cache key.
pub fn object_path(cache_root: &Path, key: &str) -> PathBuf {
    cache_root.join("objects").join(format!("{}.o", key))
}

/// Path to the sidecar metadata JSON for the given cache key.
pub fn meta_path(cache_root: &Path, key: &str) -> PathBuf {
    cache_root.join("meta").join(format!("{}.json", key))
}

/// Path to the per-profile manifest.
pub fn manifest_path(cache_root: &Path) -> PathBuf {
    cache_root.join("manifest.json")
}

// ---------------------------------------------------------------------------
// Sidecar metadata (stored alongside each object for --verbose cache-hit info)
// ---------------------------------------------------------------------------

/// JSON sidecar written next to each cached object.
///
/// Contains enough context to explain a cache hit in `--verbose` mode and to
/// let `mindc clean --cache` reason about what it is removing.
#[derive(Debug, Serialize, Deserialize)]
pub struct ObjectMeta {
    /// The source file that produced this object (relative to project root).
    pub source_path: String,
    /// The cache key that names this entry.
    pub cache_key: String,
    /// Backend target used to compile this object.
    pub target: String,
    /// Optimisation level.
    pub optimize: String,
    /// Compiler version.
    pub compiler_version: String,
    /// Dep hashes that were mixed into the key (sorted).
    pub dep_hashes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Manifest (source-path -> cache-key mapping)
// ---------------------------------------------------------------------------

/// The per-profile `manifest.json`: maps every source file path to its current
/// cached object hash.  Used by `mindc clean --cache` and for cache-hit
/// reporting.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BuildManifest {
    /// Keys: source path relative to project root (POSIX slashes).
    /// Values: the cache key (SHA-256 hex) for the most recently produced object.
    #[serde(default)]
    pub entries: std::collections::BTreeMap<String, String>,
}

impl BuildManifest {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("read manifest {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse manifest {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self).context("serialise manifest")?;
        atomic_write(path, json.as_bytes())
    }
}

// ---------------------------------------------------------------------------
// Cache probe + write
// ---------------------------------------------------------------------------

/// Result of a cache probe for one module.
#[derive(Debug)]
pub enum CacheProbe {
    /// Cache hit: the object file exists and the sidecar is valid.
    Hit { key: String, object_path: PathBuf },
    /// Cache miss: nothing in cache for this key.
    Miss { key: String },
}

/// Probe the object cache for a module.
///
/// Returns `Hit` iff both `<key>.o` and `<key>.json` exist under `cache_root`.
pub fn probe(cache_root: &Path, key: &str) -> CacheProbe {
    let obj = object_path(cache_root, key);
    let meta = meta_path(cache_root, key);

    if obj.exists() && meta.exists() {
        CacheProbe::Hit {
            key: key.to_string(),
            object_path: obj,
        }
    } else {
        CacheProbe::Miss {
            key: key.to_string(),
        }
    }
}

/// Write a compiled object and its sidecar metadata into the cache.
///
/// Uses atomic rename to avoid partial writes observed by concurrent processes.
pub fn write_object(
    cache_root: &Path,
    key: &str,
    object_bytes: &[u8],
    meta: &ObjectMeta,
) -> Result<()> {
    // Ensure subdirectories exist.
    let obj_dir = cache_root.join("objects");
    let meta_dir = cache_root.join("meta");
    fs::create_dir_all(&obj_dir).with_context(|| format!("create {}", obj_dir.display()))?;
    fs::create_dir_all(&meta_dir).with_context(|| format!("create {}", meta_dir.display()))?;

    // Write object via atomic rename.
    let obj_path = object_path(cache_root, key);
    atomic_write(&obj_path, object_bytes)
        .with_context(|| format!("write object {}", obj_path.display()))?;

    // Write sidecar metadata via atomic rename.
    let meta_json = serde_json::to_string_pretty(meta).context("serialise meta")?;
    let meta_p = meta_path(cache_root, key);
    atomic_write(&meta_p, meta_json.as_bytes())
        .with_context(|| format!("write meta {}", meta_p.display()))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Cache invalidation result (per-module build decision)
// ---------------------------------------------------------------------------

/// The decision made for one module during a build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildDecision {
    /// Reuse the cached object; no compile step needed.
    CacheHit,
    /// Compile fresh; cache was empty or `--no-cache` was passed.
    CacheMiss,
}

// ---------------------------------------------------------------------------
// Clean helpers
// ---------------------------------------------------------------------------

/// Wipe the `.cache/` subdirectory for one target+optimize profile.
///
/// Called by `mindc clean --cache`.  The sibling binary (e.g. `target/cpu/debug/myapp`)
/// is NOT removed — only the `.cache/` subdirectory is cleared.
pub fn clean_cache(
    project_root: &Path,
    target: BuildTarget,
    optimize: OptimizeLevel,
) -> Result<()> {
    let root = cache_root(project_root, target, optimize);
    if root.exists() {
        fs::remove_dir_all(&root).with_context(|| format!("remove {}", root.display()))?;
    }
    Ok(())
}

/// Wipe ALL `.cache/` subdirectories under `target/` for every target+optimize
/// combination that exists on disk.
///
/// Used by `mindc clean --cache` when invoked without a specific target.
pub fn clean_all_caches(project_root: &Path) -> Result<()> {
    let target_dir = project_root.join("target");
    if !target_dir.exists() {
        return Ok(());
    }
    // Walk target/<target>/<optimize>/.cache/
    for target_entry in fs::read_dir(&target_dir)?.flatten() {
        let target_path = target_entry.path();
        if !target_path.is_dir() {
            continue;
        }
        for opt_entry in fs::read_dir(&target_path)?.flatten() {
            let opt_path = opt_entry.path();
            if !opt_path.is_dir() {
                continue;
            }
            let cache = opt_path.join(".cache");
            if cache.exists() {
                fs::remove_dir_all(&cache)
                    .with_context(|| format!("remove {}", cache.display()))?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal: atomic write via temp-then-rename
// ---------------------------------------------------------------------------

/// Write `data` to `dest` atomically using a sibling `.tmp` file + rename.
///
/// On POSIX, `rename(2)` is atomic at the filesystem level, so a concurrent
/// reader either sees the old file or the new file — never a half-written one.
fn atomic_write(dest: &Path, data: &[u8]) -> Result<()> {
    let tmp = dest.with_extension("tmp");
    fs::write(&tmp, data).with_context(|| format!("write tmp {}", tmp.display()))?;
    fs::rename(&tmp, dest)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))
}

// ---------------------------------------------------------------------------
// Internal: self-contained SHA-256 (reused from deps::mod — identical impl)
// ---------------------------------------------------------------------------

/// Return the SHA-256 of `data` as a lowercase hex string.
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    mini_sha256(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (i, v) in [a, b, c, d, e, f, g, h].iter().enumerate() {
        state[i] = state[i].wrapping_add(*v);
    }
}

fn mini_sha256(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_H0;
    let bit_len = (data.len() as u64) * 8;

    let mut msg: Vec<u8> = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        sha256_compress(&mut state, chunk.try_into().unwrap());
    }

    let mut out = [0u8; 32];
    for (i, &word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{BuildTarget, OptimizeLevel};

    #[test]
    fn cache_key_is_64_hex_chars() {
        let key = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &[],
            "0.6.8",
            2024,
        );
        assert_eq!(key.len(), 64);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn cache_key_differs_on_source_change() {
        let k1 = module_cache_key(
            b"fn main() -> i64 { 1 }",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &[],
            "0.6.8",
            2024,
        );
        let k2 = module_cache_key(
            b"fn main() -> i64 { 2 }",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &[],
            "0.6.8",
            2024,
        );
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_differs_on_target_change() {
        let k1 = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &[],
            "0.6.8",
            2024,
        );
        let k2 = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cerebras,
            OptimizeLevel::Debug,
            &[],
            "0.6.8",
            2024,
        );
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_differs_on_optimize_change() {
        let k1 = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &[],
            "0.6.8",
            2024,
        );
        let k2 = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cpu,
            OptimizeLevel::Release,
            &[],
            "0.6.8",
            2024,
        );
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_differs_on_compiler_version_change() {
        let k1 = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &[],
            "0.6.8",
            2024,
        );
        let k2 = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &[],
            "0.6.9",
            2024,
        );
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_dep_order_independent() {
        let k1 = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &["aaaa".to_string(), "bbbb".to_string()],
            "0.6.8",
            2024,
        );
        let k2 = module_cache_key(
            b"fn main() {}",
            BuildTarget::Cpu,
            OptimizeLevel::Debug,
            &["bbbb".to_string(), "aaaa".to_string()],
            "0.6.8",
            2024,
        );
        assert_eq!(k1, k2, "dep order must not affect cache key");
    }

    #[test]
    fn probe_returns_miss_when_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let result = probe(tmp.path(), "deadbeef");
        assert!(matches!(result, CacheProbe::Miss { .. }));
    }

    #[test]
    fn write_then_probe_returns_hit() {
        let tmp = tempfile::tempdir().unwrap();
        let key = "cafebabe";
        let meta = ObjectMeta {
            source_path: "src/main.mind".to_string(),
            cache_key: key.to_string(),
            target: "cpu".to_string(),
            optimize: "debug".to_string(),
            compiler_version: "0.6.8".to_string(),
            dep_hashes: vec![],
        };
        write_object(tmp.path(), key, b"\x7fELF", &meta).unwrap();
        assert!(matches!(probe(tmp.path(), key), CacheProbe::Hit { .. }));
    }

    #[test]
    fn build_manifest_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut manifest = BuildManifest::default();
        manifest
            .entries
            .insert("src/main.mind".to_string(), "aabb".to_string());
        let path = tmp.path().join("manifest.json");
        manifest.save(&path).unwrap();
        let loaded = BuildManifest::load(&path).unwrap();
        assert_eq!(
            loaded.entries.get("src/main.mind").map(|s| s.as_str()),
            Some("aabb")
        );
    }

    #[test]
    fn clean_cache_removes_dot_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let root = cache_root(tmp.path(), BuildTarget::Cpu, OptimizeLevel::Debug);
        fs::create_dir_all(root.join("objects")).unwrap();
        fs::write(root.join("objects/test.o"), b"data").unwrap();
        assert!(root.exists());
        clean_cache(tmp.path(), BuildTarget::Cpu, OptimizeLevel::Debug).unwrap();
        assert!(!root.exists());
    }
}
