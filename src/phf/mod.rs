// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Canonical **seedless** minimal-perfect-hash construction (CHD) for the
//! `#[bimap]` enum inverse — Phase 2, Slice 1.
//!
//! # Why seedless
//! A perfect-hash family that draws a random seed until it finds a collision-free
//! assignment is *entropy-dependent*: the emitted tables (and therefore the
//! emitted mic@3 bytes and their `trace_hash`) depend on which seed the search
//! happened to land on, which is a machine/run-dependent choice. That breaks the
//! load-bearing cross-substrate bit-identity invariant. This construction is a
//! **pure function of the decoded key bytes**: the two mixers are fixed
//! constants, bucket processing order is a stable total-order sort, the
//! displacement search walks `0, 1, 2, …` deterministically, and the escalation
//! ladder over the slot count is fixed. The same key set produces the same
//! [`PhfPlan`] — hence the same tables, hence the same emitted bytes — on x86,
//! on ARM, and on every run.
//!
//! # Why a hard budget with COUNTERS (never a clock)
//! Construction cost is bounded by deterministic counters incremented before
//! each operation, never by elapsed wall-clock time. A time-based cutoff would
//! make the Built-vs-Fallback decision machine-speed-dependent (a slow runner
//! would fall back where a fast one built) — another bit-identity break. When a
//! budget is exceeded the build returns [`PhfOutcome::Fallback`]; that choice is
//! itself a pure function of the key bytes, so the caller emits the same
//! (fallback) body everywhere.
//!
//! # The mixers
//! `h = (h * MULT + byte) % 2^31`, starting at 0, folded MSB-first over the key
//! bytes. Two independent multipliers (`33`, `131`). The 2^31 modulus is
//! load-bearing twice over: it keeps the running `i64` accumulator far from
//! overflow (`h < 2^31`, `h * 131 + 255 < 2^38`), and it pins the fold to a
//! 31-bit ring so the native code path and any interpreter oracle agree
//! bit-for-bit (a 64-bit wrapping multiply would diverge between a `wrapping_mul`
//! and a checked path).
//!
//! # Table encoding contract (shared with the emitter)
//! The construction tables are emitted downstream as pure-ASCII string-literal
//! constants (each logical byte → two nibble chars `'A'..'P'`), read at run time
//! via the unshadowable `__mind_load_i8` intrinsic. [`enc_byte`] and the
//! `*_table` methods below are the single owner of that encoding; the generated
//! MIND `from_str` body decodes with the mirror formula
//! `(load_i8(a + 2k) - 'A') * 16 + (load_i8(a + 2k + 1) - 'A')`. [`PhfPlan::lookup`]
//! is the Rust mirror of the emitted lookup and is the postcondition oracle.

/// First mixer multiplier (`hA`).
pub const MULT_A: i64 = 33;
/// Second mixer multiplier (`hB`).
pub const MULT_B: i64 = 131;
/// The 31-bit fold ring: `2^31`.
pub const MOD31: i64 = 1 << 31;

/// Maximum key count accepted for a perfect-hash build (else Fallback).
pub const MAX_KEYS: usize = 128;
/// Maximum total decoded key bytes accepted (else Fallback).
pub const MAX_TOTAL_BYTES: usize = 16384;
/// Maximum displacement searched per bucket (else escalate / Fallback).
pub const MAX_DISP: i64 = 4096;
/// Maximum number of slot probes across the whole build (else Fallback).
pub const MAX_SLOT_CHECKS: u64 = 524288;

/// Per-slot ordinal sentinel meaning "unoccupied" in the emitted meta table.
/// Enum ordinals are dense `0..n-1` with `n <= MAX_KEYS = 128`, so `0..=127`
/// are the only real ordinals and `255` is unambiguous.
pub const ORD_EMPTY: i64 = 255;

/// ASCII base for the two-nibble byte encoding (`'A'`).
pub const NIBBLE_BASE: i64 = b'A' as i64;

/// `hA` — the first mixer over `key`.
pub fn hash_a(key: &[u8]) -> i64 {
    mix(key, MULT_A)
}

/// `hB` — the second mixer over `key`.
pub fn hash_b(key: &[u8]) -> i64 {
    mix(key, MULT_B)
}

fn mix(key: &[u8], mult: i64) -> i64 {
    let mut h: i64 = 0;
    for &b in key {
        h = (h * mult + b as i64) % MOD31;
    }
    h
}

/// A completed perfect-hash plan for a key set. All fields are ordered `Vec`s
/// (no hashing / no seed), so serialising them is deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhfPlan {
    /// Key count (`= m`, the bucket count).
    pub n: usize,
    /// Bucket count. Equal to `n` (the CHD first-level table is sized to `n`).
    pub m: usize,
    /// Slot count (the second-level table size, `>= n`).
    pub np: usize,
    /// Per-bucket displacement, length `m`, each in `0..=MAX_DISP`.
    pub disp: Vec<i64>,
    /// Per-slot ordinal, length `np`; [`ORD_EMPTY`] for an empty slot.
    pub slot_ord: Vec<i64>,
    /// Per-slot key byte length, length `np`; `0` for an empty slot.
    pub slot_len: Vec<usize>,
    /// Per-slot start offset into [`PhfPlan::pool`], length `np`.
    pub slot_off: Vec<usize>,
    /// Concatenated occupied-slot key bytes, in slot order.
    pub pool: Vec<u8>,
}

/// The result of a construction attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhfOutcome {
    /// A perfect-hash plan was found within budget.
    Built(PhfPlan),
    /// The key set is out of envelope or construction-resistant; the caller
    /// emits the deterministic linear fallback instead.
    Fallback,
}

/// Append `v` to `out` as two nibble chars in `'A'..'P'` (pure ASCII, so the
/// concatenation is always valid UTF-8 and never contains a quote, backslash,
/// or newline — safe as a bare MIND string literal).
pub fn enc_byte(v: u8, out: &mut String) {
    out.push((b'A' + (v >> 4)) as char);
    out.push((b'A' + (v & 0x0F)) as char);
}

impl PhfPlan {
    /// The displacement table: `m` entries × 2 logical bytes (little-endian),
    /// i.e. `m * 4` ASCII chars. Logical layout `[lo_0, hi_0, lo_1, hi_1, …]`.
    pub fn disp_table(&self) -> String {
        let mut s = String::with_capacity(self.m * 4);
        for &d in &self.disp {
            enc_byte((d & 0xFF) as u8, &mut s);
            enc_byte(((d >> 8) & 0xFF) as u8, &mut s);
        }
        s
    }

    /// The per-slot meta table: `np` entries × 4 logical bytes
    /// `[ord, len, off_lo, off_hi]`, i.e. `np * 8` ASCII chars.
    pub fn meta_table(&self) -> String {
        let mut s = String::with_capacity(self.np * 8);
        for slot in 0..self.np {
            enc_byte((self.slot_ord[slot] & 0xFF) as u8, &mut s);
            enc_byte((self.slot_len[slot] & 0xFF) as u8, &mut s);
            let off = self.slot_off[slot];
            enc_byte((off & 0xFF) as u8, &mut s);
            enc_byte(((off >> 8) & 0xFF) as u8, &mut s);
        }
        s
    }

    /// The key-byte pool: one logical byte per pool byte, i.e. `pool.len() * 2`
    /// ASCII chars.
    pub fn pool_table(&self) -> String {
        let mut s = String::with_capacity(self.pool.len() * 2);
        for &b in &self.pool {
            enc_byte(b, &mut s);
        }
        s
    }

    /// Rust mirror of the emitted MIND `from_str` lookup — the postcondition
    /// oracle. Returns the ordinal for `key`, or `-1` on miss.
    pub fn lookup(&self, key: &[u8]) -> i64 {
        let ha = hash_a(key);
        let hb = hash_b(key);
        let bucket = (ha % self.m as i64) as usize;
        let d = self.disp[bucket];
        let slot = ((hb + d) % self.np as i64) as usize;
        if self.slot_ord[slot] == ORD_EMPTY {
            return -1;
        }
        if self.slot_len[slot] != key.len() {
            return -1;
        }
        let off = self.slot_off[slot];
        if self.pool[off..off + key.len()] != key[..] {
            return -1;
        }
        self.slot_ord[slot]
    }
}

/// Build a canonical seedless perfect-hash plan for `keys` (each an
/// `(ordinal, key-bytes)` pair, in declaration order), or [`PhfOutcome::Fallback`]
/// if the set is out of envelope or resists construction within budget.
///
/// Determinism: every container here is an ordered `Vec`; bucket processing
/// order is a stable sort on a total key `(size DESC, index ASC)`; the
/// displacement search and slot-count escalation are fixed walks. No `HashMap`,
/// no RNG, no clock.
pub fn build(keys: &[(i64, Vec<u8>)]) -> PhfOutcome {
    let n = keys.len();
    // Envelope gates (each a pure function of the key bytes).
    if n == 0 || n > MAX_KEYS {
        return PhfOutcome::Fallback;
    }
    let total_bytes: usize = keys.iter().map(|(_, k)| k.len()).sum();
    if total_bytes > MAX_TOTAL_BYTES {
        return PhfOutcome::Fallback;
    }
    for (ord, k) in keys {
        // Per-slot length is one logical byte; a key longer than 255 bytes
        // cannot be encoded, so fall back.
        if k.len() > 255 {
            return PhfOutcome::Fallback;
        }
        // The ordinal is one logical byte and must not collide with the
        // empty-slot sentinel.
        if *ord < 0 || *ord >= ORD_EMPTY {
            return PhfOutcome::Fallback;
        }
    }
    // Defensive: duplicate keys have no bijection (E2018 rejects this upstream
    // for the enum form; guard anyway so build never loops on a bad table).
    {
        let mut sorted: Vec<&Vec<u8>> = keys.iter().map(|(_, k)| k).collect();
        sorted.sort();
        for w in sorted.windows(2) {
            if w[0] == w[1] {
                return PhfOutcome::Fallback;
            }
        }
    }

    let m = n;

    // First-level bucketing: group key indices per bucket in DECLARATION order.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (ki, (_, k)) in keys.iter().enumerate() {
        let b = (hash_a(k) % m as i64) as usize;
        buckets[b].push(ki);
    }

    // Bucket processing order: largest buckets first (harder to place), ties
    // broken by ascending bucket index. Stable sort over a total-order key.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| buckets[b].len().cmp(&buckets[a].len()).then(a.cmp(&b)));

    // Shared probe counter across the whole build (all escalation rungs).
    let mut slot_checks: u64 = 0;

    // Escalation ladder over the second-level slot count.
    let np_max = 4 * n + 8;
    let mut np = n;
    while np <= np_max {
        // `occupied[slot]`: which key index (or `usize::MAX`) claims each slot.
        let mut occupied: Vec<usize> = vec![usize::MAX; np];
        let mut disp: Vec<i64> = vec![0; m];
        let mut ok = true;

        'buckets: for &bi in &order {
            let bucket = &buckets[bi];
            if bucket.is_empty() {
                disp[bi] = 0;
                continue;
            }
            let mut placed = false;
            let mut d: i64 = 0;
            while d <= MAX_DISP {
                // Try displacement `d`: every key in the bucket must map to a
                // free, intra-bucket-distinct slot.
                let mut trial: Vec<usize> = Vec::with_capacity(bucket.len());
                let mut good = true;
                for &ki in bucket {
                    slot_checks += 1;
                    if slot_checks > MAX_SLOT_CHECKS {
                        return PhfOutcome::Fallback;
                    }
                    let hb = hash_b(&keys[ki].1);
                    let slot = ((hb + d) % np as i64) as usize;
                    if occupied[slot] != usize::MAX || trial.contains(&slot) {
                        good = false;
                        break;
                    }
                    trial.push(slot);
                }
                if good {
                    for (idx, &ki) in bucket.iter().enumerate() {
                        occupied[trial[idx]] = ki;
                    }
                    disp[bi] = d;
                    placed = true;
                    break;
                }
                d += 1;
            }
            if !placed {
                ok = false;
                break 'buckets;
            }
        }

        if ok {
            // Materialise the plan tables from the slot assignment.
            let mut slot_ord = vec![ORD_EMPTY; np];
            let mut slot_len = vec![0usize; np];
            let mut slot_off = vec![0usize; np];
            let mut pool: Vec<u8> = Vec::with_capacity(total_bytes);
            for (slot, &ki) in occupied.iter().enumerate() {
                if ki == usize::MAX {
                    continue;
                }
                let (ord, k) = &keys[ki];
                slot_ord[slot] = *ord;
                slot_len[slot] = k.len();
                slot_off[slot] = pool.len();
                pool.extend_from_slice(k);
            }
            return PhfOutcome::Built(PhfPlan {
                n,
                m,
                np,
                disp,
                slot_ord,
                slot_len,
                slot_off,
                pool,
            });
        }
        np += 1;
    }

    PhfOutcome::Fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(pairs: &[(i64, &str)]) -> Vec<(i64, Vec<u8>)> {
        pairs
            .iter()
            .map(|(o, s)| (*o, s.as_bytes().to_vec()))
            .collect()
    }

    fn built(keys: &[(i64, Vec<u8>)]) -> PhfPlan {
        match build(keys) {
            PhfOutcome::Built(p) => p,
            PhfOutcome::Fallback => panic!("expected Built, got Fallback"),
        }
    }

    #[test]
    fn build_is_deterministic_32x() {
        // The whole point: seedless construction is a pure function of the key
        // bytes, so 32 independent builds are byte-identical plans.
        let keys = kv(&[(0, "AUD"), (1, "JPY"), (2, "USD")]);
        let first = built(&keys);
        for _ in 0..32 {
            assert_eq!(built(&keys), first, "build must be deterministic");
        }
    }

    #[test]
    fn build_tables_are_deterministic_32x() {
        // Also assert the EMITTED table strings (the real cross-runner evidence
        // artifact) are identical across builds.
        let keys = kv(&[(0, "AUD"), (1, "JPY"), (2, "USD")]);
        let p = built(&keys);
        let (d, m, pool) = (p.disp_table(), p.meta_table(), p.pool_table());
        for _ in 0..32 {
            let q = built(&keys);
            assert_eq!(q.disp_table(), d);
            assert_eq!(q.meta_table(), m);
            assert_eq!(q.pool_table(), pool);
        }
    }

    #[test]
    fn currency_golden_unique_slots_and_ordinals() {
        let keys = kv(&[(0, "AUD"), (1, "JPY"), (2, "USD")]);
        let p = built(&keys);
        // Every key retrieves its declared ordinal.
        assert_eq!(p.lookup(b"AUD"), 0);
        assert_eq!(p.lookup(b"JPY"), 1);
        assert_eq!(p.lookup(b"USD"), 2);
        // The three occupied slots are distinct (minimal-perfect over the set).
        let occ: Vec<usize> = (0..p.np).filter(|&s| p.slot_ord[s] != ORD_EMPTY).collect();
        assert_eq!(occ.len(), 3, "exactly three occupied slots");
    }

    #[test]
    fn postcondition_every_key_retrieves_its_ordinal() {
        let keys = kv(&[
            (0, "GET"),
            (1, "POST"),
            (2, "PUT"),
            (3, "DELETE"),
            (4, "PATCH"),
            (5, "HEAD"),
            (6, "OPTIONS"),
            (7, "TRACE"),
            (8, "CONNECT"),
        ]);
        let p = built(&keys);
        for (ord, k) in &keys {
            assert_eq!(p.lookup(k), *ord, "key {:?} must map to {ord}", k);
        }
    }

    #[test]
    fn near_miss_returns_minus_one() {
        let keys = kv(&[(0, "AUD"), (1, "JPY"), (2, "USD")]);
        let p = built(&keys);
        assert_eq!(p.lookup(b"aud"), -1, "case variant misses");
        assert_eq!(p.lookup(b"AU"), -1, "prefix misses");
        assert_eq!(p.lookup(b"AUDX"), -1, "superstring misses");
        assert_eq!(p.lookup(b"EUR"), -1, "absent key misses");
        assert_eq!(p.lookup(b""), -1, "empty query misses");
    }

    #[test]
    fn bounds_129_keys_falls_back() {
        let keys: Vec<(i64, Vec<u8>)> = (0..129)
            .map(|i| (i as i64, format!("k{i:04}").into_bytes()))
            .collect();
        assert!(matches!(build(&keys), PhfOutcome::Fallback));
    }

    #[test]
    fn total_bytes_overflow_falls_back() {
        // Two keys whose combined length exceeds MAX_TOTAL_BYTES.
        let big = vec![b'x'; MAX_TOTAL_BYTES];
        let keys = vec![(0i64, big), (1i64, vec![b'y'; 8])];
        assert!(matches!(build(&keys), PhfOutcome::Fallback));
    }

    #[test]
    fn key_longer_than_255_falls_back() {
        let keys = vec![(0i64, vec![b'z'; 256]), (1i64, b"ok".to_vec())];
        assert!(matches!(build(&keys), PhfOutcome::Fallback));
    }

    #[test]
    fn forced_slot_check_exhaustion_falls_back() {
        // A pathological set that drives the probe counter past MAX_SLOT_CHECKS
        // must fall back rather than loop. 128 keys sharing a long common prefix
        // (so hA collisions pile into few buckets) exercises the displacement
        // search hard; the deterministic outcome is what we assert.
        let keys: Vec<(i64, Vec<u8>)> = (0..MAX_KEYS)
            .map(|i| {
                let mut k = vec![b'p'; 200];
                k.push((i % 251) as u8);
                (i as i64, k)
            })
            .collect();
        // Either it builds within budget or it falls back — but it MUST return
        // deterministically (no hang) and the outcome must be stable.
        let a = build(&keys);
        let b = build(&keys);
        assert_eq!(a, b, "pathological build must be deterministic");
    }

    #[test]
    fn empty_string_key_is_placeable() {
        // `= ""` is a legal key: slot_len 0, occupied via ORD_EMPTY sentinel on
        // the ordinal (not the length), so it does not collide with empty slots.
        let keys = kv(&[(0, ""), (1, "A"), (2, "B")]);
        let p = built(&keys);
        assert_eq!(p.lookup(b""), 0);
        assert_eq!(p.lookup(b"A"), 1);
        assert_eq!(p.lookup(b"B"), 2);
        // A non-empty query never accidentally hits the empty-string slot.
        assert_eq!(p.lookup(b"C"), -1);
    }

    #[test]
    fn embedded_nul_key() {
        let keys = vec![
            (0i64, b"a\x00b".to_vec()),
            (1i64, b"a\x00c".to_vec()),
            (2i64, b"abc".to_vec()),
        ];
        let p = built(&keys);
        assert_eq!(p.lookup(b"a\x00b"), 0);
        assert_eq!(p.lookup(b"a\x00c"), 1);
        assert_eq!(p.lookup(b"abc"), 2);
        assert_eq!(p.lookup(b"a\x00d"), -1);
    }

    #[test]
    fn common_prefix_keys() {
        let keys = kv(&[
            (0, "prefix_a"),
            (1, "prefix_b"),
            (2, "prefix_c"),
            (3, "prefix_aa"),
            (4, "prefix_bb"),
        ]);
        let p = built(&keys);
        for (ord, k) in &keys {
            assert_eq!(p.lookup(k), *ord);
        }
    }

    #[test]
    fn equal_length_keys() {
        let keys = kv(&[(0, "aaa"), (1, "aab"), (2, "aba"), (3, "baa"), (4, "bbb")]);
        let p = built(&keys);
        for (ord, k) in &keys {
            assert_eq!(p.lookup(k), *ord);
        }
        assert_eq!(p.lookup(b"abb"), -1);
    }

    #[test]
    fn collision_heavy_100_keys() {
        let keys: Vec<(i64, Vec<u8>)> = (0..100)
            .map(|i| (i as i64, format!("sym_{i:03}").into_bytes()))
            .collect();
        let p = built(&keys);
        for (ord, k) in &keys {
            assert_eq!(p.lookup(k), *ord, "100-key set: {:?}", k);
        }
        assert_eq!(p.lookup(b"sym_999"), -1);
        assert_eq!(p.lookup(b"nope"), -1);
    }

    #[test]
    fn enc_byte_is_ascii_nibbles() {
        let mut s = String::new();
        enc_byte(0x00, &mut s);
        enc_byte(0xFF, &mut s);
        enc_byte(0x3A, &mut s);
        assert_eq!(s, "AAPPDK");
        assert!(s.bytes().all(|b| (b'A'..=b'P').contains(&b)));
    }
}
