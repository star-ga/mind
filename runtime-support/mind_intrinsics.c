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
// pure-MIND std.vec surface functions.  This object is statically linked into
// every --emit-shared cdylib so the resulting .so is self-contained and
// dlopen-able without an external libmind_std.
//
// All functions use the i64 opaque-address ABI (RFC 0005 P0a):
//   - pointers are passed and returned as int64_t
//   - no built-in MIND pointer type is introduced
//
// The Vec heap-record layout matches RFC 0005 Option C (multi-LLM consensus):
//   offset  0: addr  (i64) — backing-store base address (0 == empty)
//   offset  8: len   (i64) — logical element count
//   offset 16: cap   (i64) — backing-store capacity in elements
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
