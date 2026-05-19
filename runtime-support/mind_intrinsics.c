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

// RFC 0005 Phase 6.5 Stage 1b — runtime-support stub.
//
// Provides C implementations of the seven RFC 0005 i64-ABI intrinsics and the
// pure-MIND std.vec, std.map, and std.string surface functions.  This object
// is statically linked into every --emit-shared cdylib so the resulting .so is
// self-contained and dlopen-able without an external libmind_std.
//
// All functions use the i64 opaque-address ABI (RFC 0005 P0a):
//   - pointers are passed and returned as int64_t
//   - no built-in MIND pointer type is introduced
//
// Heap-record layouts (RFC 0005 Option C):
//
// Vec / String (3×i64, 24 bytes):
//   offset  0: addr  (i64) — backing-store base address (0 == empty)
//   offset  8: len   (i64) — logical element count
//   offset 16: cap   (i64) — backing-store capacity in elements
//
// Map (4×i64, 32 bytes):
//   offset  0: keys_addr (i64) — backing-store for keys (0 == empty)
//   offset  8: vals_addr (i64) — backing-store for values (0 == empty)
//   offset 16: len       (i64) — logical entry count
//   offset 24: cap       (i64) — backing-store capacity in entries
//
// Growth policy is fixed at "doubling, min-cap 4" (deterministic for the
// evidence chain and bootstrap fixed point).

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Seven RFC 0005 intrinsics
// ---------------------------------------------------------------------------

int64_t __mind_alloc(int64_t bytes) {
    if (bytes <= 0) return 0;
    void *p = malloc((size_t)bytes);
    return (int64_t)(uintptr_t)p;
}

int64_t __mind_realloc(int64_t addr, int64_t new_bytes) {
    void *p = realloc((void *)(uintptr_t)addr, (size_t)new_bytes);
    return (int64_t)(uintptr_t)p;
}

int64_t __mind_free(int64_t addr) {
    free((void *)(uintptr_t)addr);
    return 0;
}

int64_t __mind_load_i64(int64_t addr) {
    int64_t val;
    memcpy(&val, (void *)(uintptr_t)addr, sizeof(int64_t));
    return val;
}

int64_t __mind_store_i64(int64_t addr, int64_t val) {
    memcpy((void *)(uintptr_t)addr, &val, sizeof(int64_t));
    return 0;
}

int64_t __mind_read(int64_t path_addr, int64_t path_len,
                    int64_t buf_addr,  int64_t buf_cap) {
    (void)path_addr; (void)path_len; (void)buf_addr; (void)buf_cap;
    return -1; // not needed for Phase 6.5 Stage 1
}

int64_t __mind_write(int64_t path_addr, int64_t path_len,
                     int64_t buf_addr,  int64_t buf_len) {
    (void)path_addr; (void)path_len; (void)buf_addr; (void)buf_len;
    return -1; // not needed for Phase 6.5 Stage 1
}

// ---------------------------------------------------------------------------
// std.vec surface — matches RFC 0005 Option C heap-record layout.
//
// Each Vec value is an i64 base address into a 3×i64 block:
//   [addr | len | cap]  at 8-byte stride.
// ---------------------------------------------------------------------------

// Allocate a new Vec heap record with addr=0, len=0, cap=0.
int64_t vec_new(void) {
    int64_t rec = __mind_alloc(24); // 3 × i64
    __mind_store_i64(rec,      0);  // addr
    __mind_store_i64(rec + 8,  0);  // len
    __mind_store_i64(rec + 16, 0);  // cap
    return rec;
}

int64_t vec_len(int64_t v) {
    return __mind_load_i64(v + 8);
}

int64_t vec_cap(int64_t v) {
    return __mind_load_i64(v + 16);
}

int64_t vec_addr(int64_t v) {
    return __mind_load_i64(v);
}

int64_t vec_get(int64_t v, int64_t i) {
    int64_t base = __mind_load_i64(v);
    return __mind_load_i64(base + i * 8);
}

int64_t vec_set(int64_t v, int64_t i, int64_t value) {
    int64_t base = __mind_load_i64(v);
    return __mind_store_i64(base + i * 8, value);
}

// vec_push — append value, growing the backing store with doubling policy.
//
// Returns the same Vec record address (mutates in place).
// Growth: cap 0 → 4, otherwise double when len == cap.
int64_t vec_push(int64_t v, int64_t value) {
    int64_t len  = __mind_load_i64(v + 8);
    int64_t cap  = __mind_load_i64(v + 16);
    int64_t base = __mind_load_i64(v);

    if (len >= cap) {
        int64_t new_cap = (cap == 0) ? 4 : cap * 2;
        int64_t new_base = __mind_alloc(new_cap * 8);
        if (cap > 0 && base != 0) {
            memcpy((void *)(uintptr_t)new_base,
                   (void *)(uintptr_t)base,
                   (size_t)(len * 8));
        }
        __mind_store_i64(v,      new_base);
        __mind_store_i64(v + 16, new_cap);
        base = new_base;
    }

    __mind_store_i64(base + len * 8, value);
    __mind_store_i64(v + 8, len + 1);
    return v;
}

// ---------------------------------------------------------------------------
// std.map surface — RFC 0005 Option C heap-record layout.
//
// Map value is an i64 base address into a 4×i64 block:
//   [keys_addr | vals_addr | len | cap]  at 8-byte stride.
//
// keys_addr and vals_addr each point to an i64[] backing store of `cap`
// elements.  Insertion order is preserved (linear append-only table).
// Lookup is linear (O(n)) — matches std/map.mind's documented semantics.
// Growth: cap 0 → 4, then doubles.
// ---------------------------------------------------------------------------

// map_new — empty Map heap record, all fields zero.
int64_t map_new(void) {
    int64_t rec = __mind_alloc(32); // 4 × i64
    __mind_store_i64(rec,      0);  // keys_addr
    __mind_store_i64(rec + 8,  0);  // vals_addr
    __mind_store_i64(rec + 16, 0);  // len
    __mind_store_i64(rec + 24, 0);  // cap
    return rec;
}

// map_len — current entry count.
int64_t map_len(int64_t m) {
    return __mind_load_i64(m + 16);
}

// map_cap — current backing-store capacity.
int64_t map_cap(int64_t m) {
    return __mind_load_i64(m + 24);
}

// map_keys_addr — opaque i64 base address of the keys array.
int64_t map_keys_addr(int64_t m) {
    return __mind_load_i64(m);
}

// map_vals_addr — opaque i64 base address of the values array.
int64_t map_vals_addr(int64_t m) {
    return __mind_load_i64(m + 8);
}

// map_key_at — key at logical index i (no bounds check).
int64_t map_key_at(int64_t m, int64_t i) {
    int64_t keys = __mind_load_i64(m);
    return __mind_load_i64(keys + i * 8);
}

// map_value_at — value at logical index i (no bounds check).
int64_t map_value_at(int64_t m, int64_t i) {
    int64_t vals = __mind_load_i64(m + 8);
    return __mind_load_i64(vals + i * 8);
}

// map_insert — append (key, value) at the tail.
//
// Non-mutating ABI: allocates a fresh 4-field Map heap record on every call
// (matches std/map.mind's non-mutating `map_insert` semantics — each call
// returns a new Map handle).  The backing stores are either reused (when
// len < cap) or reallocated (when len == cap, doubling policy).
// Growth: cap 0 → 4, then doubles.
int64_t map_insert(int64_t m, int64_t key, int64_t value) {
    int64_t keys_addr = __mind_load_i64(m);
    int64_t vals_addr = __mind_load_i64(m + 8);
    int64_t len       = __mind_load_i64(m + 16);
    int64_t cap       = __mind_load_i64(m + 24);

    int64_t new_cap;
    int64_t new_keys;
    int64_t new_vals;

    if (len < cap) {
        // Still room in the existing backing stores — reuse them.
        new_cap  = cap;
        new_keys = keys_addr;
        new_vals = vals_addr;
    } else {
        // Grow: allocate fresh backing stores and copy existing entries.
        new_cap  = (cap == 0) ? 4 : cap * 2;
        new_keys = __mind_alloc(new_cap * 8);
        new_vals = __mind_alloc(new_cap * 8);
        if (len > 0 && keys_addr != 0) {
            memcpy((void *)(uintptr_t)new_keys,
                   (void *)(uintptr_t)keys_addr,
                   (size_t)(len * 8));
            memcpy((void *)(uintptr_t)new_vals,
                   (void *)(uintptr_t)vals_addr,
                   (size_t)(len * 8));
        }
    }

    // Append the new entry.
    __mind_store_i64(new_keys + len * 8, key);
    __mind_store_i64(new_vals + len * 8, value);

    // Allocate a fresh Map header and populate it.
    int64_t rec = __mind_alloc(32);
    __mind_store_i64(rec,      new_keys);
    __mind_store_i64(rec + 8,  new_vals);
    __mind_store_i64(rec + 16, len + 1);
    __mind_store_i64(rec + 24, new_cap);
    return rec;
}

// ---------------------------------------------------------------------------
// std.string surface — RFC 0005 Option C heap-record layout.
//
// String value is an i64 base address into a 3×i64 block (identical layout
// to Vec):  [addr | len | cap]  at 8-byte stride.
//
// `addr` points to a byte[] backing store of `cap` bytes.
// Growth: cap 0 → 16 (small-string amortisation), then doubles.
// ---------------------------------------------------------------------------

// string_new — empty String heap record, all fields zero.
int64_t string_new(void) {
    int64_t rec = __mind_alloc(24); // 3 × i64
    __mind_store_i64(rec,      0);  // addr
    __mind_store_i64(rec + 8,  0);  // len
    __mind_store_i64(rec + 16, 0);  // cap
    return rec;
}

// string_len — current byte length.
int64_t string_len(int64_t s) {
    return __mind_load_i64(s + 8);
}

// string_cap — backing-store capacity in bytes.
int64_t string_cap(int64_t s) {
    return __mind_load_i64(s + 16);
}

// string_addr — opaque i64 base address of the byte content.
int64_t string_addr(int64_t s) {
    return __mind_load_i64(s);
}

// string_get_byte — single byte read (lower 8 bits, no bounds check).
int64_t string_get_byte(int64_t s, int64_t i) {
    int64_t base = __mind_load_i64(s);
    return __mind_load_i64(base + i) & 0xFF;
}

// string_push_byte — append a single byte, returning a new String handle.
//
// Non-mutating ABI: allocates a fresh 3-field String heap record on every
// call (matches std/string.mind's non-mutating semantics).  The backing
// store is reused when len < cap and reallocated (doubling) when len == cap.
// Growth: cap 0 → 16, then doubles.
int64_t string_push_byte(int64_t s, int64_t b) {
    int64_t base    = __mind_load_i64(s);
    int64_t len     = __mind_load_i64(s + 8);
    int64_t cap     = __mind_load_i64(s + 16);

    int64_t new_cap;
    int64_t new_base;

    if (len < cap) {
        new_cap  = cap;
        new_base = base;
    } else {
        new_cap  = (cap == 0) ? 16 : cap * 2;
        new_base = __mind_alloc(new_cap);
        if (len > 0 && base != 0) {
            memcpy((void *)(uintptr_t)new_base,
                   (void *)(uintptr_t)base,
                   (size_t)len);
        }
    }

    // Write the new byte at offset `len` (each byte is stored in a full i64
    // slot matching the MIND intrinsic convention — only low 8 bits matter).
    __mind_store_i64(new_base + len, b & 0xFF);

    // Allocate a fresh String header and populate it.
    int64_t rec = __mind_alloc(24);
    __mind_store_i64(rec,      new_base);
    __mind_store_i64(rec + 8,  len + 1);
    __mind_store_i64(rec + 16, new_cap);
    return rec;
}
