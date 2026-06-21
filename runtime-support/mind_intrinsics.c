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
// Provides C implementations of the RFC 0005 i64-ABI intrinsics and the
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

// --- #225: compiler/OS portability shim (MSVC + Windows) -----------------
// Function/data attributes that vary between GCC/Clang and MSVC, plus
// POSIX I/O replacements. Defined here so every code path below stays
// untouched and works under both clang/gcc (Linux/Mac) and cl.exe
// (Windows).  See tests/blas_smoke.rs for the Windows un-skip.
#if defined(_MSC_VER) && !defined(__clang__)
#  define MIND_TARGET_AVX2   /* cl.exe needs /arch:AVX2 globally; no per-fn attr */
#  define MIND_ALIGN32       __declspec(align(32))
#else
#  define MIND_TARGET_AVX2   __attribute__((target("avx2,fma")))
#  define MIND_ALIGN32       __attribute__((aligned(32)))
#endif

// DLL symbol export. MSVC + clang-cl need __declspec(dllexport) explicitly
// on every public symbol (Windows OpenSSH-built DLLs cannot rely on the
// ELF-style "all global symbols exported" default). On ELF / Mach-O the
// macro expands to nothing; default-visibility makes them externally
// linkable already, matching the historical clang -shared -fPIC behaviour.
#if defined(_WIN32) || defined(_WIN64)
#  define MIND_EXPORT __declspec(dllexport)
#else
#  define MIND_EXPORT
#endif

// POSIX I/O: Windows has _read / _write in <io.h> and lacks pread/pwrite.
// Emulate pread/pwrite via _lseeki64+_read/_write — non-atomic, matches
// the existing single-threaded runtime use.
#if defined(_WIN32)
#  include <stdio.h>      // SEEK_SET / SEEK_CUR (Unix gets these via <unistd.h>)
#  include <io.h>
#  include <intrin.h>     // __cpuid / __cpuidex (file-scope; not inside fn body)
#  include <BaseTsd.h>
typedef SSIZE_T ssize_t;
typedef long long mind_off_t;
static ssize_t mind_pread_emu(int fd, void *buf, size_t count, mind_off_t off) {
    long long cur = _lseeki64(fd, 0, SEEK_CUR);
    if (cur < 0) return -1;
    if (_lseeki64(fd, off, SEEK_SET) < 0) return -1;
    int r = _read(fd, buf, (unsigned)count);
    _lseeki64(fd, cur, SEEK_SET);
    return (ssize_t)r;
}
static ssize_t mind_pwrite_emu(int fd, const void *buf, size_t count, mind_off_t off) {
    long long cur = _lseeki64(fd, 0, SEEK_CUR);
    if (cur < 0) return -1;
    if (_lseeki64(fd, off, SEEK_SET) < 0) return -1;
    int r = _write(fd, buf, (unsigned)count);
    _lseeki64(fd, cur, SEEK_SET);
    return (ssize_t)r;
}
#  define MIND_READ(fd, b, c)      _read((fd), (b), (unsigned)(c))
#  define MIND_WRITE(fd, b, c)     _write((fd), (b), (unsigned)(c))
#  define MIND_PREAD(fd, b, c, o)  mind_pread_emu((fd), (b), (c), (mind_off_t)(o))
#  define MIND_PWRITE(fd, b, c, o) mind_pwrite_emu((fd), (b), (c), (mind_off_t)(o))
#else
#  include <unistd.h>
#  define MIND_READ(fd, b, c)      read((fd), (b), (c))
#  define MIND_WRITE(fd, b, c)     write((fd), (b), (c))
#  define MIND_PREAD(fd, b, c, o)  pread((fd), (b), (c), (off_t)(o))
#  define MIND_PWRITE(fd, b, c, o) pwrite((fd), (b), (c), (off_t)(o))
#endif

// ---------------------------------------------------------------------------
// Seven RFC 0005 intrinsics
// ---------------------------------------------------------------------------

// __mind_assert_fail(msg_len) — deterministic conditional-trap target for the
// `assert cond, "msg"` statement (#203). `mindc` lowers `assert` to
// `if cond { } else { __mind_assert_fail(<msg-len>); }`, so this is reached
// ONLY when the asserted condition is false. It aborts the process via the
// same `abort()` path the region / GenRef runtime already uses on its own
// unrecoverable invariants — a single deterministic SIGABRT, identical across
// CPU/ARM (no pointer bits, no float, no clock). The i64 argument is the
// message byte length, carried for diagnostics; control flow does not depend
// on it. Declared `(i64) -> i64` to match the auto-generated extern signature
// (`func.func private @__mind_assert_fail(i64) -> i64`); it never returns.
MIND_EXPORT int64_t __mind_assert_fail(int64_t msg_len) {
    (void)msg_len;
    abort();
    return 0; /* unreachable — abort() does not return */
}

MIND_EXPORT int64_t __mind_alloc(int64_t bytes) {
    if (bytes <= 0) return 0;
    void *p = malloc((size_t)bytes);
    return (int64_t)(uintptr_t)p;
}

MIND_EXPORT int64_t __mind_realloc(int64_t addr, int64_t new_bytes) {
    void *p = realloc((void *)(uintptr_t)addr, (size_t)new_bytes);
    return (int64_t)(uintptr_t)p;
}

MIND_EXPORT int64_t __mind_free(int64_t addr) {
    free((void *)(uintptr_t)addr);
    return 0;
}

MIND_EXPORT int64_t __mind_load_i64(int64_t addr) {
    int64_t val;
    memcpy(&val, (void *)(uintptr_t)addr, sizeof(int64_t));
    return val;
}

MIND_EXPORT int64_t __mind_store_i64(int64_t addr, int64_t val) {
    memcpy((void *)(uintptr_t)addr, &val, sizeof(int64_t));
    return 0;
}

// RFC 0005 Phase 1.6 (task #306) — single-byte load/store.
// The std.string / std.sha256 / std.toml / std.tui byte-buffer code currently
// uses `__mind_store_i64(base + i, b)` to write one byte at byte offset `i`,
// which clobbers 7 bytes per store; the 7-byte backing-store pad below absorbs
// the OOB at end-of-buffer but the high bytes within the buffer are stale
// (overwritten by the next byte store) and the garbage-past-len is a
// cross-substrate bit-identity landmine.  `__mind_load_i8` zero-extends to i64
// so existing `& 255` mask semantics are preserved during call-site migration.
MIND_EXPORT int64_t __mind_load_i8(int64_t addr) {
    uint8_t b;
    memcpy(&b, (void *)(uintptr_t)addr, 1);
    return (int64_t)b;
}

MIND_EXPORT int64_t __mind_store_i8(int64_t addr, int64_t val) {
    uint8_t b = (uint8_t)(val & 0xFF);
    memcpy((void *)(uintptr_t)addr, &b, 1);
    return 0;
}

// 4-byte load/store — a proper i32 ABI for the u32-field structs of kernel
// interfaces (notably the io_uring SQ/CQ ring head/tail and other ABI fields).
// `__mind_load_i32` zero-extends to i64 (unsigned 32-bit semantics);
// `__mind_store_i32` writes EXACTLY 4 bytes, so it never clobbers the adjacent
// u32 the way `__mind_store_i64` at a 4-byte-aligned offset would. memcpy keeps
// it alignment-safe and cross-substrate byte-identical.
MIND_EXPORT int64_t __mind_load_i32(int64_t addr) {
    uint32_t w;
    memcpy(&w, (void *)(uintptr_t)addr, 4);
    return (int64_t)w;
}

MIND_EXPORT int64_t __mind_store_i32(int64_t addr, int64_t val) {
    uint32_t w = (uint32_t)(val & 0xFFFFFFFF);
    memcpy((void *)(uintptr_t)addr, &w, 4);
    return 0;
}

// 2-byte load/store — `i16`/`u16` struct fields. Mirrors the i32 helpers: load
// zero-extends to i64 (unsigned 16-bit; a signed `i16` field applies a shl/ashr
// sign-extend in the IR), store writes EXACTLY 2 bytes so it never clobbers an
// adjacent field. memcpy keeps it alignment-safe and cross-substrate byte-identical.
MIND_EXPORT int64_t __mind_load_i16(int64_t addr) {
    uint16_t w;
    memcpy(&w, (void *)(uintptr_t)addr, 2);
    return (int64_t)w;
}

MIND_EXPORT int64_t __mind_store_i16(int64_t addr, int64_t val) {
    uint16_t w = (uint16_t)(val & 0xFFFF);
    memcpy((void *)(uintptr_t)addr, &w, 2);
    return 0;
}

// __mind_read(fd, buf_addr, count, offset) — POSIX read/pread.
// offset == -1 means "use current stream position" (plain read).
MIND_EXPORT int64_t __mind_read(int64_t fd, int64_t buf_addr, int64_t count, int64_t offset) {
    if (buf_addr == 0 || count <= 0) return 0;
    void *buf = (void *)(uintptr_t)buf_addr;
    if (offset < 0) {
        return (int64_t)MIND_READ((int)fd, buf, (size_t)count);
    }
    return (int64_t)MIND_PREAD((int)fd, buf, (size_t)count, offset);
}

// __mind_write(fd, buf_addr, count, offset) — POSIX write/pwrite.
// offset == -1 means "use current stream position" (plain write).
MIND_EXPORT int64_t __mind_write(int64_t fd, int64_t buf_addr, int64_t count, int64_t offset) {
    if (buf_addr == 0 || count <= 0) return 0;
    void *buf = (void *)(uintptr_t)buf_addr;
    if (offset < 0) {
        return (int64_t)MIND_WRITE((int)fd, buf, (size_t)count);
    }
    return (int64_t)MIND_PWRITE((int)fd, buf, (size_t)count, offset);
}

// print_bytes — convenience: write `count` bytes from `buf_addr` to stdout.
// Corresponds to std/io.mind `print_bytes(buf_addr, count)`.
// Compiled by mindc as an external call (the MIND stdlib function body is
// not inlined into the cdylib during --emit-shared).
MIND_EXPORT int64_t print_bytes(int64_t buf_addr, int64_t count) {
    return __mind_write(1, buf_addr, count, -1);
}

// printI64 / printNewline — MLIR executable-print helpers (#306).
// `mindc` lowers scalar `print(x)` outputs to `func.call @printI64(%x)`
// followed by `func.call @printNewline()` and emits both as `func.func
// private` (external) declarations — see
// src/eval/mlir_export.rs::emit_executable_prints / emit_executable_helpers.
// The definitions must live in this statically-linked runtime-support shim:
// without them the keystone self-host link fails with an undefined reference
// to `printI64`, and `mindc build --emit=cdylib` falls back to a launcher
// stub instead of a real ELF cdylib (the #306 keystone real-ELF blocker).
// Output goes through the same raw fd-1 write as `print_bytes` so the two
// print surfaces share one unbuffered, deterministic path (no stdio buffer
// interleave, no locale dependence — required for byte-identical bootstrap).
MIND_EXPORT void printI64(int64_t value) {
    // Decimal-format into a fixed buffer (max i64 = 20 digits + sign), then
    // emit in a single write. Two's-complement-safe negation for INT64_MIN.
    char digits[20];
    int n = 0;
    uint64_t mag = (value < 0) ? (~(uint64_t)value + 1ULL) : (uint64_t)value;
    do {
        digits[n++] = (char)('0' + (int)(mag % 10ULL));
        mag /= 10ULL;
    } while (mag != 0 && n < (int)sizeof(digits));
    char out[24];
    int len = 0;
    if (value < 0) out[len++] = '-';
    while (n > 0) out[len++] = digits[--n];
    MIND_WRITE(1, out, (size_t)len);
}

MIND_EXPORT void printNewline(void) {
    char nl = '\n';
    MIND_WRITE(1, &nl, 1);
}

// ---------------------------------------------------------------------------
// std.vec surface — matches RFC 0005 Option C heap-record layout.
//
// Each Vec value is an i64 base address into a 3×i64 block:
//   [addr | len | cap]  at 8-byte stride.
// ---------------------------------------------------------------------------

// Allocate a new Vec heap record with addr=0, len=0, cap=0.
MIND_EXPORT int64_t vec_new(void) {
    int64_t rec = __mind_alloc(24); // 3 × i64
    __mind_store_i64(rec,      0);  // addr
    __mind_store_i64(rec + 8,  0);  // len
    __mind_store_i64(rec + 16, 0);  // cap
    return rec;
}

MIND_EXPORT int64_t vec_len(int64_t v) {
    return __mind_load_i64(v + 8);
}

MIND_EXPORT int64_t vec_cap(int64_t v) {
    return __mind_load_i64(v + 16);
}

MIND_EXPORT int64_t vec_addr(int64_t v) {
    return __mind_load_i64(v);
}

MIND_EXPORT int64_t vec_get(int64_t v, int64_t i) {
    int64_t base = __mind_load_i64(v);
    return __mind_load_i64(base + i * 8);
}

MIND_EXPORT int64_t vec_set(int64_t v, int64_t i, int64_t value) {
    int64_t base = __mind_load_i64(v);
    return __mind_store_i64(base + i * 8, value);
}

// vec_push — append value, growing the backing store with doubling policy.
//
// Returns the same Vec record address (mutates in place).
// Growth: cap 0 → 4, otherwise double when len == cap.
MIND_EXPORT int64_t vec_push(int64_t v, int64_t value) {
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

// vec_zeroed — allocate a backing store of `n` zeroed i64 elements and
// return its i64 base address (issue #204). Mirrors std/vec.mind's pure-MIND
// definition: `__mind_alloc` hands back uninitialised memory, so this clears
// it with a deterministic forward fill. A non-positive `n` allocates nothing
// and returns 0 (the __mind_alloc contract). Read back with
// __mind_load_i64(base + i * 8).
MIND_EXPORT int64_t __mind_vec_zeroed(int64_t n) {
    int64_t base = __mind_alloc(n * 8);
    for (int64_t i = 0; i < n; i++) {
        __mind_store_i64(base + i * 8, 0);
    }
    return base;
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
MIND_EXPORT int64_t map_new(void) {
    int64_t rec = __mind_alloc(32); // 4 × i64
    __mind_store_i64(rec,      0);  // keys_addr
    __mind_store_i64(rec + 8,  0);  // vals_addr
    __mind_store_i64(rec + 16, 0);  // len
    __mind_store_i64(rec + 24, 0);  // cap
    return rec;
}

// map_len — current entry count.
MIND_EXPORT int64_t map_len(int64_t m) {
    return __mind_load_i64(m + 16);
}

// map_cap — current backing-store capacity.
MIND_EXPORT int64_t map_cap(int64_t m) {
    return __mind_load_i64(m + 24);
}

// map_keys_addr — opaque i64 base address of the keys array.
MIND_EXPORT int64_t map_keys_addr(int64_t m) {
    return __mind_load_i64(m);
}

// map_vals_addr — opaque i64 base address of the values array.
MIND_EXPORT int64_t map_vals_addr(int64_t m) {
    return __mind_load_i64(m + 8);
}

// map_key_at — key at logical index i (no bounds check).
MIND_EXPORT int64_t map_key_at(int64_t m, int64_t i) {
    int64_t keys = __mind_load_i64(m);
    return __mind_load_i64(keys + i * 8);
}

// map_value_at — value at logical index i (no bounds check).
MIND_EXPORT int64_t map_value_at(int64_t m, int64_t i) {
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
MIND_EXPORT int64_t map_insert(int64_t m, int64_t key, int64_t value) {
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

// map_get — first value whose key == `key` (i64 identity), or 0 if absent.
// Linear scan over insertion order (deterministic). For STRING keys use
// map_get_str: an i64 == on two String handles compares pointers, not contents.
MIND_EXPORT int64_t map_get(int64_t m, int64_t key) {
    int64_t keys = __mind_load_i64(m);
    int64_t vals = __mind_load_i64(m + 8);
    int64_t len  = __mind_load_i64(m + 16);
    for (int64_t i = 0; i < len; i++) {
        if (__mind_load_i64(keys + i * 8) == key) {
            return __mind_load_i64(vals + i * 8);
        }
    }
    return 0;
}

// map_contains_key — 1 if some key == `key` (i64 identity), else 0.
MIND_EXPORT int64_t map_contains_key(int64_t m, int64_t key) {
    int64_t keys = __mind_load_i64(m);
    int64_t len  = __mind_load_i64(m + 16);
    for (int64_t i = 0; i < len; i++) {
        if (__mind_load_i64(keys + i * 8) == key) {
            return 1;
        }
    }
    return 0;
}

// map_str_key_eq — byte-content equality of two String handles. A String is a
// heap record {addr@0, len@8, cap@16} passed by handle, so the handle points at
// that record. Reads len/addr directly and compares bytes — the correct
// equality for string keys (distinct allocations of equal content must match).
static int64_t map_str_key_eq(int64_t a, int64_t b) {
    int64_t alen = __mind_load_i64(a + 8);
    int64_t blen = __mind_load_i64(b + 8);
    if (alen != blen) {
        return 0;
    }
    int64_t aaddr = __mind_load_i64(a);
    int64_t baddr = __mind_load_i64(b);
    return memcmp((const void *)(uintptr_t)aaddr,
                  (const void *)(uintptr_t)baddr,
                  (size_t)alen) == 0
               ? 1
               : 0;
}

// map_get_str — value for the first key whose String CONTENTS equal `key`'s
// (content equality, not handle identity), or 0 if absent. For map<string,_>.
MIND_EXPORT int64_t map_get_str(int64_t m, int64_t key) {
    int64_t keys = __mind_load_i64(m);
    int64_t vals = __mind_load_i64(m + 8);
    int64_t len  = __mind_load_i64(m + 16);
    for (int64_t i = 0; i < len; i++) {
        if (map_str_key_eq(__mind_load_i64(keys + i * 8), key)) {
            return __mind_load_i64(vals + i * 8);
        }
    }
    return 0;
}

// map_contains_key_str — 1 if some key's String CONTENTS equal `key`'s, else 0.
MIND_EXPORT int64_t map_contains_key_str(int64_t m, int64_t key) {
    int64_t keys = __mind_load_i64(m);
    int64_t len  = __mind_load_i64(m + 16);
    for (int64_t i = 0; i < len; i++) {
        if (map_str_key_eq(__mind_load_i64(keys + i * 8), key)) {
            return 1;
        }
    }
    return 0;
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
MIND_EXPORT int64_t string_new(void) {
    int64_t rec = __mind_alloc(24); // 3 × i64
    __mind_store_i64(rec,      0);  // addr
    __mind_store_i64(rec + 8,  0);  // len
    __mind_store_i64(rec + 16, 0);  // cap
    return rec;
}

// string_len — current byte length.
MIND_EXPORT int64_t string_len(int64_t s) {
    return __mind_load_i64(s + 8);
}

// string_cap — backing-store capacity in bytes.
MIND_EXPORT int64_t string_cap(int64_t s) {
    return __mind_load_i64(s + 16);
}

// string_addr — opaque i64 base address of the byte content.
MIND_EXPORT int64_t string_addr(int64_t s) {
    return __mind_load_i64(s);
}

// string_get_byte — single byte read (lower 8 bits, no bounds check).
MIND_EXPORT int64_t string_get_byte(int64_t s, int64_t i) {
    int64_t base = __mind_load_i64(s);
    return __mind_load_i64(base + i) & 0xFF;
}

// string_push_byte — append a single byte, returning a new String handle.
//
// Non-mutating ABI: allocates a fresh 3-field String heap record on every
// call (matches std/string.mind's non-mutating semantics).  The backing
// store is reused when len < cap and reallocated (doubling) when len == cap.
// Growth: cap 0 → 16, then doubles.
//
// Allocation size: new_cap + 7 bytes.  The byte at logical position i is
// written via __mind_store_i64(base + i, …) which copies 8 bytes starting
// at byte offset i.  At the last valid position (i = new_cap − 1) that
// write needs bytes [new_cap−1 .. new_cap+6], so the backing store must be
// at least new_cap + 7 bytes.  The extra 7 bytes are never part of the
// logical string content (cap field controls the logical boundary); they
// exist purely to make the 8-byte i64 store safe at every position.
MIND_EXPORT int64_t string_push_byte(int64_t s, int64_t b) {
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
        // Allocate new_cap + 7 bytes: the 7-byte pad ensures that a
        // __mind_store_i64 at any byte offset in [0, new_cap) writes
        // entirely within the allocated region (see comment above).
        new_base = __mind_alloc(new_cap + 7);
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

// ---------------------------------------------------------------------------
// mind-blas Track A — runtime-support SIMD bridge (RFC 0006).
//
// Six i64-ABI intrinsics (Option C convention) callable from MIND via
// `use std.blas`.  All pointers are passed as opaque int64_t addresses, all
// f32 / Q16.16 results are packed into the low 32 bits of an i64 result.
//
// Each intrinsic has two implementations:
//   * `*_scalar`  — portable C reference, also the cross-arch oracle.
//   * `*_avx2`    — AVX2/FMA hot path, compiled via __attribute__((target))
//                   so this translation unit stays buildable on any host.
//
// Runtime dispatch is decided once at .so load time by an `__attribute__
// ((constructor))` that probes `__builtin_cpu_supports("avx2")` and
// `__builtin_cpu_supports("fma")`.  The dispatcher is a single branch on a
// cached flag — zero per-call overhead on the hot path.
//
// Numerical contract:
//   * f32 path: AVX2 reduction reorders summation; tolerance ~1e-6 vs scalar
//     on 1M-element vectors is documented and gated by `tests/blas_smoke.rs`.
//     Byte-identical on lengths < 8 (no SIMD lanes used).
//   * Q16.16 path: integer accumulation with explicit i64 widening per lane
//     is associative; AVX2 result is byte-identical to scalar at every
//     length — required by the cross-arch bit-identity gate (task #57).
//
// The intrinsics never allocate, never read past `len`, and tolerate
// `len <= 0` by returning 0.
// ---------------------------------------------------------------------------

#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#  define MIND_BLAS_X86_64 1
#else
#  define MIND_BLAS_X86_64 0
#endif

// One-time CPU-feature probe, populated by the .so-load constructor below.
// 0 = scalar fallback path, 1 = AVX2+FMA hot path.  Single byte read on each
// intrinsic entry — the dispatch overhead is below the branch-predictor floor.
static int mind_blas_use_avx2 = 0;

// Test-only dispatcher override.
//
// Exposed for the cross-arch bit-identity tests in `tests/blas_smoke.rs` so
// the harness can force the scalar path on an AVX2 host and compare the two
// outputs.  Production callers never touch this — the constructor below sets
// the dispatcher correctly on .so load.
//
// Returns the previous value so the harness can save/restore.  Argument:
//   0 -> force scalar path
//   1 -> force AVX2 path (no-ops to scalar if AVX2 wasn't detected at load)
MIND_EXPORT int __mind_blas_set_use_avx2(int v) {
    int prev = mind_blas_use_avx2;
    mind_blas_use_avx2 = (v != 0) ? 1 : 0;
    return prev;
}

// Read the dispatcher flag — the harness uses this to sanity-check that AVX2
// was detected at .so load on hosts that have it.
MIND_EXPORT int __mind_blas_get_use_avx2(void) {
    return mind_blas_use_avx2;
}

// CPU-feature probe — AVX2 (256-bit lanes) + FMA (fused multiply-add).
// CPUs that have AVX2 typically also have FMA, but we check the pair so a
// Haswell-without-FMA outlier falls back to scalar instead of SIGILL-ing
// inside _mm256_fmadd_ps.
//
// On Windows we use the __cpuid / __cpuidex intrinsics (`<intrin.h>`) for
// BOTH cl.exe AND clang-cl/clang -- clang's `__builtin_cpu_supports`
// emits references to compiler-rt symbols (`__cpu_indicator_init`,
// `__cpu_model`) that don't auto-link with `clang -shared` + lld-link on
// PE/COFF. The CPUID intrinsics are documented + portable across both
// Windows toolchains.
static int mind_blas_cpu_has_avx2_fma(void) {
#if MIND_BLAS_X86_64
#  if defined(_WIN32)
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) return 0;
    __cpuidex(regs, 7, 0);
    int has_avx2 = (regs[1] >> 5) & 1;    // CPUID 7,0 EBX bit 5
    __cpuid(regs, 1);
    int has_fma  = (regs[2] >> 12) & 1;   // CPUID 1   ECX bit 12
    return has_avx2 && has_fma;
#  else
    // Linux/Mac GCC/Clang: builtins — clang's __builtin_cpu_init is a
    // no-op, gcc needs it. Both link the compiler-rt symbols fine on ELF.
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#  endif
#else
    return 0;
#endif
}

#if defined(_MSC_VER) && !defined(__clang__)
// MSVC: register a CRT-initializer in the .CRT$XCU section so the
// dispatcher runs once at DLL load, mirroring GCC's __attribute__((constructor))
// semantics. This is the documented Microsoft pattern (see MS Learn /
// "CRT Initialization") — the dispatcher function pointer goes into the
// CRT's startup walk between $XCA and $XCZ.
#  pragma section(".CRT$XCU", read)
static void mind_blas_init_dispatch(void);
__declspec(allocate(".CRT$XCU")) static void (*mind_blas_ctor_p)(void) = mind_blas_init_dispatch;
static void mind_blas_init_dispatch(void) {
    if (mind_blas_cpu_has_avx2_fma()) {
        mind_blas_use_avx2 = 1;
    }
}
#else
__attribute__((constructor))
static void mind_blas_init_dispatch(void) {
    if (mind_blas_cpu_has_avx2_fma()) {
        mind_blas_use_avx2 = 1;
    }
}
#endif

// Pack an f32 into the low 32 bits of an i64 (sign-extended zero of high 32).
// The caller is expected to reinterpret the low 32 bits as IEEE-754 f32 via
// memcpy / std.blas's `f32_from_bits` helper.  Using a union-style memcpy
// keeps strict-aliasing rules happy on every supported compiler.
static inline int64_t mind_blas_pack_f32(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (int64_t)(uint64_t)bits;
}

// ── f32: dot product (sum of element-wise products) ─────────────────────────

static float mind_blas_dot_f32_scalar(const float *a, const float *b, int64_t len) {
    float acc = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

#if MIND_BLAS_X86_64
MIND_TARGET_AVX2
static float mind_blas_dot_f32_avx2(const float *a, const float *b, int64_t len) {
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // Horizontal reduce: two 128-bit halves -> 4-wide -> 2-wide -> scalar.
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float tail = _mm_cvtss_f32(sum);
    for (; i < len; ++i) {
        tail += a[i] * b[i];
    }
    return tail;
}
#endif

MIND_EXPORT int64_t __mind_blas_dot_f32(int64_t a_addr, int64_t b_addr, int64_t len) {
    if (len <= 0 || a_addr == 0 || b_addr == 0) return mind_blas_pack_f32(0.0f);
    const float *a = (const float *)(uintptr_t)a_addr;
    const float *b = (const float *)(uintptr_t)b_addr;
    float r;
#if MIND_BLAS_X86_64
    if (mind_blas_use_avx2) {
        r = mind_blas_dot_f32_avx2(a, b, len);
    } else {
        r = mind_blas_dot_f32_scalar(a, b, len);
    }
#else
    r = mind_blas_dot_f32_scalar(a, b, len);
#endif
    return mind_blas_pack_f32(r);
}

// ── f32: L1 (Manhattan) — sum of |a[i] - b[i]| ──────────────────────────────

static float mind_blas_dot_l1_f32_scalar(const float *a, const float *b, int64_t len) {
    float acc = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
        float d = a[i] - b[i];
        acc += d < 0.0f ? -d : d;
    }
    return acc;
}

#if MIND_BLAS_X86_64
MIND_TARGET_AVX2
static float mind_blas_dot_l1_f32_avx2(const float *a, const float *b, int64_t len) {
    // Sign mask: all bits set except the IEEE-754 sign bit -> bitwise AND
    // clears the sign and produces |x| in a single instruction.
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        d = _mm256_and_ps(d, abs_mask);
        acc = _mm256_add_ps(acc, d);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float tail = _mm_cvtss_f32(sum);
    for (; i < len; ++i) {
        float d = a[i] - b[i];
        tail += d < 0.0f ? -d : d;
    }
    return tail;
}
#endif

MIND_EXPORT int64_t __mind_blas_dot_l1_f32(int64_t a_addr, int64_t b_addr, int64_t len) {
    if (len <= 0 || a_addr == 0 || b_addr == 0) return mind_blas_pack_f32(0.0f);
    const float *a = (const float *)(uintptr_t)a_addr;
    const float *b = (const float *)(uintptr_t)b_addr;
    float r;
#if MIND_BLAS_X86_64
    if (mind_blas_use_avx2) {
        r = mind_blas_dot_l1_f32_avx2(a, b, len);
    } else {
        r = mind_blas_dot_l1_f32_scalar(a, b, len);
    }
#else
    r = mind_blas_dot_l1_f32_scalar(a, b, len);
#endif
    return mind_blas_pack_f32(r);
}

// ── f32: L∞ (Chebyshev) — max of |a[i] - b[i]| ──────────────────────────────

static float mind_blas_dot_linf_f32_scalar(const float *a, const float *b, int64_t len) {
    float m = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
        float d = a[i] - b[i];
        if (d < 0.0f) d = -d;
        if (d > m) m = d;
    }
    return m;
}

#if MIND_BLAS_X86_64
MIND_TARGET_AVX2
static float mind_blas_dot_linf_f32_avx2(const float *a, const float *b, int64_t len) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        d = _mm256_and_ps(d, abs_mask);
        acc = _mm256_max_ps(acc, d);
    }
    // Horizontal max across 8 lanes.
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 m = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(m, m);
    m = _mm_max_ps(m, shuf);
    shuf = _mm_shuffle_ps(m, m, 0x55);
    m = _mm_max_ss(m, shuf);
    float tail = _mm_cvtss_f32(m);
    for (; i < len; ++i) {
        float d = a[i] - b[i];
        if (d < 0.0f) d = -d;
        if (d > tail) tail = d;
    }
    return tail;
}
#endif

MIND_EXPORT int64_t __mind_blas_dot_linf_f32(int64_t a_addr, int64_t b_addr, int64_t len) {
    if (len <= 0 || a_addr == 0 || b_addr == 0) return mind_blas_pack_f32(0.0f);
    const float *a = (const float *)(uintptr_t)a_addr;
    const float *b = (const float *)(uintptr_t)b_addr;
    float r;
#if MIND_BLAS_X86_64
    if (mind_blas_use_avx2) {
        r = mind_blas_dot_linf_f32_avx2(a, b, len);
    } else {
        r = mind_blas_dot_linf_f32_scalar(a, b, len);
    }
#else
    r = mind_blas_dot_linf_f32_scalar(a, b, len);
#endif
    return mind_blas_pack_f32(r);
}

// ── f32: row-major matmul y = W · x where W is rows×cols, x is cols-vector ──
//
// y must point at `rows` writable f32 slots.  Returns 0 on success, -1 if any
// pointer is null.  Inner loop reuses the dot_f32 dispatcher so any AVX2
// improvement applies row-by-row without duplication.
MIND_EXPORT int64_t __mind_blas_matmul_rmajor_f32(
    int64_t w_addr, int64_t x_addr, int64_t y_addr,
    int64_t rows, int64_t cols
) {
    if (w_addr == 0 || x_addr == 0 || y_addr == 0) return -1;
    if (rows <= 0 || cols <= 0) return 0;
    const float *W = (const float *)(uintptr_t)w_addr;
    const float *x = (const float *)(uintptr_t)x_addr;
    float       *y = (float       *)(uintptr_t)y_addr;
    for (int64_t r = 0; r < rows; ++r) {
        const float *row = W + (size_t)r * (size_t)cols;
        float v;
#if MIND_BLAS_X86_64
        if (mind_blas_use_avx2) {
            v = mind_blas_dot_f32_avx2(row, x, cols);
        } else {
            v = mind_blas_dot_f32_scalar(row, x, cols);
        }
#else
        v = mind_blas_dot_f32_scalar(row, x, cols);
#endif
        y[r] = v;
    }
    return 0;
}

// ── Q16.16: dot product, byte-identical scalar-vs-AVX2 (task #57 gate) ──────
//
// Q16.16 multiplication of two 32-bit fixed-point operands produces a 64-bit
// intermediate that is then right-shifted by 16 to land back in Q16.16 form.
// SIMD splits the 8-lane 32×32 -> 32 mullo into two i32×i32 -> i64 widening
// products via `_mm256_mul_epi32` (which consumes the even-indexed 32-bit
// lanes of each register and returns four i64 results), then shifts each i64
// right by 16 and accumulates.  We deliberately do NOT use `_mm256_mullo_epi32`
// because its 32-bit truncating result loses the high half — Q16.16 needs the
// full 64-bit intermediate for bit-identity with the scalar oracle.
//
// Bit-identity contract: for every (a, b, len), this function returns exactly
// the same i64 the scalar fallback returns.  This is required by task #57
// (cross-arch determinism) — see RFC 0006 §3.

static int64_t mind_blas_dot_q16_scalar(const int32_t *a, const int32_t *b, int64_t len) {
    // Accumulate in i64 to avoid overflow on long vectors; the final Q16.16
    // value is the low 32 bits of acc, sign-extended into i64.
    int64_t acc = 0;
    for (int64_t i = 0; i < len; ++i) {
        int64_t prod = (int64_t)a[i] * (int64_t)b[i];
        // Arithmetic right shift on a signed i64 is implementation-defined
        // in C but every supported compiler implements it as sign-preserving
        // (the LLVM-bundled clang we target documents `ashr` semantics).
        acc += prod >> 16;
    }
    // Sign-extend the Q16.16 result (low 32 bits) into i64 for the ABI.
    return (int64_t)(int32_t)acc;
}

#if MIND_BLAS_X86_64
// Arithmetic right shift of four i64 lanes by 16 bits.
//
// AVX2 only ships logical `_mm256_srli_epi64`; arithmetic right shift for
// i64 lanes is an AVX-512 instruction (`_mm256_srai_epi64`, requires
// AVX512VL).  We emulate it by `or`-ing the logical-shifted value with a
// sign-fill mask derived from `_mm256_cmpgt_epi64(0, x)` (which produces
// all-1s in lanes where x < 0, all-0s otherwise), shifted into the top 16
// bits.  This matches the bit pattern the C-level `x >> 16` expression
// produces under the LLVM `ashr` semantics our toolchain documents.
MIND_TARGET_AVX2
static inline __m256i mind_blas_srai_epi64_q16(__m256i x) {
    __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), x);
    __m256i logical = _mm256_srli_epi64(x, 16);
    __m256i fill = _mm256_slli_epi64(sign, 64 - 16);
    return _mm256_or_si256(logical, fill);
}

MIND_TARGET_AVX2
static int64_t mind_blas_dot_q16_avx2(const int32_t *a, const int32_t *b, int64_t len) {
    // The widening multiply `_mm256_mul_epi32` takes the even-indexed 32-bit
    // lanes of each input, sign-extends them to 64 bits, multiplies, and
    // produces four i64 results.  To cover all eight 32-bit lanes per
    // iteration we issue two widening multiplies — one on the even lanes
    // (original positions 0,2,4,6) and one on the odd lanes shifted down by
    // 4 bytes (positions 1,3,5,7).
    __m256i acc = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));

        // Even-lane widening multiply: 4 × i64.
        __m256i prod_even = _mm256_mul_epi32(va, vb);

        // Odd-lane: shift each 64-bit slot right by 32 bits so the odd
        // 32-bit lane lands in the low half, which _mm256_mul_epi32 reads.
        // _mm256_srli_epi64 zero-fills the high bits — but that's fine for
        // the input to _mm256_mul_epi32 because the intrinsic only consumes
        // the low 32 bits of each i64 lane and sign-extends them itself.
        __m256i va_odd = _mm256_srli_epi64(va, 32);
        __m256i vb_odd = _mm256_srli_epi64(vb, 32);
        __m256i prod_odd = _mm256_mul_epi32(va_odd, vb_odd);

        // Arithmetic right-shift each i64 lane by 16 to land in Q16.16 form.
        prod_even = mind_blas_srai_epi64_q16(prod_even);
        prod_odd  = mind_blas_srai_epi64_q16(prod_odd);

        acc = _mm256_add_epi64(acc, prod_even);
        acc = _mm256_add_epi64(acc, prod_odd);
    }
    // Horizontal sum of four i64 lanes.
    int64_t buf[4] MIND_ALIGN32;
    _mm256_store_si256((__m256i *)buf, acc);
    int64_t sum = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < len; ++i) {
        int64_t prod = (int64_t)a[i] * (int64_t)b[i];
        sum += prod >> 16;
    }
    return (int64_t)(int32_t)sum;
}
#endif

MIND_EXPORT int64_t __mind_blas_dot_q16(int64_t a_addr, int64_t b_addr, int64_t len) {
    if (len <= 0 || a_addr == 0 || b_addr == 0) return 0;
    const int32_t *a = (const int32_t *)(uintptr_t)a_addr;
    const int32_t *b = (const int32_t *)(uintptr_t)b_addr;
#if MIND_BLAS_X86_64
    if (mind_blas_use_avx2) {
        return mind_blas_dot_q16_avx2(a, b, len);
    }
#endif
    return mind_blas_dot_q16_scalar(a, b, len);
}

// ── Q16.16: L1 (Manhattan) — sum of |a[i] - b[i]|, byte-identical to scalar ─

static int64_t mind_blas_dot_l1_q16_scalar(const int32_t *a, const int32_t *b, int64_t len) {
    // Accumulate in i64; the absolute difference of two i32 fits in i33,
    // and a long-vector sum easily overflows i32.  Result is the low 32 bits
    // (Q16.16) sign-extended.
    int64_t acc = 0;
    for (int64_t i = 0; i < len; ++i) {
        int64_t d = (int64_t)a[i] - (int64_t)b[i];
        if (d < 0) d = -d;
        acc += d;
    }
    return (int64_t)(int32_t)acc;
}

#if MIND_BLAS_X86_64
MIND_TARGET_AVX2
static int64_t mind_blas_dot_l1_q16_avx2(const int32_t *a, const int32_t *b, int64_t len) {
    // Lane-wise: |a - b| as i32, then widen + sum.  _mm256_abs_epi32 is a
    // single AVX2 instruction; for the widening sum we go via two
    // sign-extending unpacks into i64 so the accumulator never overflows.
    __m256i acc = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i d  = _mm256_sub_epi32(va, vb);
        d = _mm256_abs_epi32(d);

        // Widen the eight i32 lanes into two i64 vectors of four lanes each.
        __m128i lo128 = _mm256_castsi256_si128(d);
        __m128i hi128 = _mm256_extracti128_si256(d, 1);
        __m256i lo64 = _mm256_cvtepi32_epi64(lo128);
        __m256i hi64 = _mm256_cvtepi32_epi64(hi128);
        acc = _mm256_add_epi64(acc, lo64);
        acc = _mm256_add_epi64(acc, hi64);
    }
    int64_t buf[4] MIND_ALIGN32;
    _mm256_store_si256((__m256i *)buf, acc);
    int64_t sum = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < len; ++i) {
        int64_t d = (int64_t)a[i] - (int64_t)b[i];
        if (d < 0) d = -d;
        sum += d;
    }
    return (int64_t)(int32_t)sum;
}
#endif

MIND_EXPORT int64_t __mind_blas_dot_l1_q16(int64_t a_addr, int64_t b_addr, int64_t len) {
    if (len <= 0 || a_addr == 0 || b_addr == 0) return 0;
    const int32_t *a = (const int32_t *)(uintptr_t)a_addr;
    const int32_t *b = (const int32_t *)(uintptr_t)b_addr;
#if MIND_BLAS_X86_64
    if (mind_blas_use_avx2) {
        return mind_blas_dot_l1_q16_avx2(a, b, len);
    }
#endif
    return mind_blas_dot_l1_q16_scalar(a, b, len);
}

// ---------------------------------------------------------------------------
// RFC 0010 Phase J-A — region-interior allocation tracking
//
// Three helper functions implement the enter/track/exit lifecycle for
// `region { }` blocks (Tier 2 — region-interior heap, §3.2).
//
// Data structure: a fixed-depth stack of frames.  Each frame holds a
// dynamic array of i64 (opaque heap addresses) allocated inside one region.
// `__mind_region_enter` pushes a fresh frame; `__mind_region_track` appends
// to the top frame; `__mind_region_exit` frees every recorded address then
// pops the frame.
//
// Design choices:
//   - MIND_REGION_STACK_DEPTH 64: 64 nested `region { }` levels is well
//     beyond any plausible MIND program; a stack overflow panics via abort().
//   - Initial frame capacity 8: doubles on overflow (doubling policy).
//   - All allocations use malloc/realloc/free directly (no __mind_alloc
//     indirection) so that region-interior ptrs are NOT themselves tracked
//     as region allocations — the frame storage is outside the region model.
//
// Thread safety: not required.  MIND programs are single-threaded until
// RFC 0011 (async).  The stack is a plain static array.
// ---------------------------------------------------------------------------

#define MIND_REGION_STACK_DEPTH 64
#define MIND_REGION_FRAME_INIT_CAP 8

typedef struct {
    int64_t *ptrs;   /* heap addresses tracked in this frame */
    int64_t  len;    /* number of live entries             */
    int64_t  cap;    /* allocated capacity of ptrs[]       */
} MindRegionFrame;

static MindRegionFrame mind_region_stack[MIND_REGION_STACK_DEPTH];
static int             mind_region_depth = 0;

// __mind_region_enter() — push a new allocation-tracking frame.
//
// Returns 0 (unit i64). Called at the entry of every `region { }` block.
MIND_EXPORT int64_t __mind_region_enter(void) {
    if (mind_region_depth >= MIND_REGION_STACK_DEPTH) {
        /* region nesting overflow — abort deterministically */
        abort();
    }
    MindRegionFrame *f = &mind_region_stack[mind_region_depth];
    f->ptrs = (int64_t *)malloc((size_t)MIND_REGION_FRAME_INIT_CAP * sizeof(int64_t));
    f->len  = 0;
    f->cap  = MIND_REGION_FRAME_INIT_CAP;
    mind_region_depth++;
    return 0;
}

// __mind_region_track(ptr) — record an allocation made inside a region.
//
// `ptr` is the i64 opaque address returned by `__mind_alloc`. Appended to
// the top frame's list. Returns `ptr` unchanged (so the call can be
// composed with the alloc result in a single SSA chain).
//
// No-ops when called outside a region (mind_region_depth == 0) so that
// user code that calls `__mind_alloc` directly from outside any region does
// not fault.
MIND_EXPORT int64_t __mind_region_track(int64_t ptr) {
    if (mind_region_depth == 0 || ptr == 0) return ptr;
    MindRegionFrame *f = &mind_region_stack[mind_region_depth - 1];
    if (f->len >= f->cap) {
        int64_t new_cap = f->cap * 2;
        f->ptrs = (int64_t *)realloc(f->ptrs, (size_t)new_cap * sizeof(int64_t));
        f->cap  = new_cap;
    }
    f->ptrs[f->len++] = ptr;
    return ptr;
}

// __mind_region_exit() — free every allocation in the top frame, then pop.
//
// Returns 0 (unit i64). Called at the exit of every `region { }` block.
// The order of freeing is LIFO (last allocated, first freed) — deterministic
// and consistent with the evidence-chain contract.
MIND_EXPORT int64_t __mind_region_exit(void) {
    if (mind_region_depth == 0) return 0;
    mind_region_depth--;
    MindRegionFrame *f = &mind_region_stack[mind_region_depth];
    /* Free in reverse insertion order — LIFO */
    for (int64_t i = f->len - 1; i >= 0; --i) {
        if (f->ptrs[i] != 0) {
            free((void *)(uintptr_t)f->ptrs[i]);
        }
    }
    free(f->ptrs);
    f->ptrs = NULL;
    f->len  = 0;
    f->cap  = 0;
    return 0;
}

// ---------------------------------------------------------------------------
// RFC 0010 Phase J-B — GenRef generation-checked references (Tier 3).
//
// A GenRef<T> is a 64-bit opaque handle that packs a slot index and a
// generation counter, giving safe access to long-lived heap allocations
// that can outlive the scope that created them.
//
// Handle encoding (little-endian):
//   bits [63:32] — slot index  (uint32_t, zero-based into genref_table)
//   bits [31: 0] — generation  (uint32_t, increments on every gen_free)
//
// This packing fits comfortably in the i64 ABI (RFC 0005 Option C) and
// keeps both fields accessible with a single i64 extraction.
//
// Generation table layout:
//   A growable array of GenRefSlot structs.  Initial capacity 16; doubles
//   on overflow.  The table itself uses direct malloc/realloc — it is NOT
//   tracked by __mind_region_track so the frame storage sits outside the
//   region model.
//
// Wrap-around (RFC 0010 §7):
//   The generation counter is 32 bits.  After 2^32 frees of a single slot
//   the counter wraps to 0.  On wrap, the runtime aborts deterministically
//   (abort()) rather than silently passing a stale-gen check.  In practice,
//   exhausting a 32-bit counter at a single slot would require sustained
//   allocation+free at >1 GHz for over 4 seconds — unreachable in any real
//   workload.  Choosing 32 bits (not 64) keeps the packed handle within a
//   single i64 with no side table, which matches the i64 ABI throughout.
//
// Thread safety: none required.  MIND programs are single-threaded until
// RFC 0011 (async).  The table is a plain static pointer.
// ---------------------------------------------------------------------------

#define MIND_GENREF_INIT_CAP 16

typedef struct {
    void    *ptr;        /* live allocation, or NULL when the slot is free */
    uint32_t generation; /* bumped on every gen_free of this slot          */
    int      in_use;     /* 1 if the slot holds a live allocation          */
} GenRefSlot;

static GenRefSlot *genref_table   = NULL;
static uint32_t    genref_cap     = 0;
static uint32_t    genref_len     = 0;

// Lazy initialise the table on first use.
//
// Generation counters are initialised to 1, not 0.  This ensures that the
// packed handle for slot 0 at its first allocation is genref_pack(0, 1) = 1,
// which is non-zero and distinguishable from the null handle (0).
// The invariant is: a valid live handle is always non-zero.
static void genref_ensure_init(void) {
    if (genref_table != NULL) return;
    genref_table = (GenRefSlot *)malloc(
        (size_t)MIND_GENREF_INIT_CAP * sizeof(GenRefSlot));
    if (!genref_table) abort();
    for (uint32_t i = 0; i < MIND_GENREF_INIT_CAP; ++i) {
        genref_table[i].ptr        = NULL;
        genref_table[i].generation = 1; /* start at 1 so pack(slot,gen)!=0 */
        genref_table[i].in_use     = 0;
    }
    genref_cap = MIND_GENREF_INIT_CAP;
    genref_len = 0;
}

// Pack (slot, gen) into a single i64 handle.
// bits [63:32] = slot index, bits [31:0] = generation.
static inline int64_t genref_pack(uint32_t slot, uint32_t gen) {
    return (int64_t)(((uint64_t)slot << 32) | (uint64_t)gen);
}

// Unpack slot index from a handle.
static inline uint32_t genref_slot(int64_t handle) {
    return (uint32_t)((uint64_t)(unsigned long long)handle >> 32);
}

// Unpack generation from a handle.
static inline uint32_t genref_gen(int64_t handle) {
    return (uint32_t)((uint64_t)(unsigned long long)handle & 0xFFFFFFFFu);
}

// __mind_gen_alloc(bytes) — allocate `bytes` bytes on the heap and return a
// packed (slot, generation) handle.
//
// On success: returns a non-zero handle that encodes the slot and its current
// generation counter.  The slot is reused from the free-list when available;
// a new slot is appended when the table is full.
//
// On allocation failure: aborts deterministically — MIND does not model OOM
// as a recoverable condition at the GenRef tier (matching region OOM policy).
//
// Returns 0 on invalid input (bytes <= 0) — callers should treat handle 0
// as a null/invalid GenRef.
MIND_EXPORT int64_t __mind_gen_alloc(int64_t bytes) {
    if (bytes <= 0) return 0;

    genref_ensure_init();

    void *p = malloc((size_t)bytes);
    if (!p) abort(); /* deterministic OOM panic */

    /* Scan for a free slot first (reuse freed slots). */
    uint32_t slot;
    int found_free = 0;
    for (uint32_t i = 0; i < genref_len; ++i) {
        if (!genref_table[i].in_use) {
            slot = i;
            found_free = 1;
            break;
        }
    }

    if (!found_free) {
        /* No free slot — grow the table or append a new slot. */
        if (genref_len >= genref_cap) {
            uint32_t new_cap = genref_cap * 2;
            GenRefSlot *t = (GenRefSlot *)realloc(
                genref_table, (size_t)new_cap * sizeof(GenRefSlot));
            if (!t) abort();
            /* Initialise new slots with generation = 1 (not 0) so that
             * genref_pack(slot, 1) is always non-zero for any slot index. */
            for (uint32_t k = genref_cap; k < new_cap; ++k) {
                t[k].ptr        = NULL;
                t[k].generation = 1;
                t[k].in_use     = 0;
            }
            genref_table = t;
            genref_cap   = new_cap;
        }
        slot = genref_len++;
    }

    /* Populate the slot.  The generation counter is NOT reset on reuse —
     * it retains the value it had when the slot was last freed, so stale
     * handles (that hold the old generation) continue to return 0 on deref.
     */
    genref_table[slot].ptr    = p;
    genref_table[slot].in_use = 1;

    return genref_pack(slot, genref_table[slot].generation);
}

// __mind_gen_deref(handle) — check the handle's generation against the slot's
// current generation.
//
// Returns the live pointer as i64 when the generations match (the allocation
// is still live), or 0 when they differ (the allocation was freed, or the
// handle is stale from a slot reuse).
//
// Returning 0 (not panicking) matches the Phase J-B design: the caller's
// match/if guard branches on zero and handles the dangling case safely.
// The type-checker emits `safety::genref_unchecked_deref` when the return
// value of gen_deref is used without such a guard.
MIND_EXPORT int64_t __mind_gen_deref(int64_t handle) {
    if (handle == 0) return 0;

    uint32_t slot = genref_slot(handle);
    uint32_t gen  = genref_gen(handle);

    if (genref_table == NULL || slot >= genref_len) return 0;
    if (genref_table[slot].generation != gen) return 0;
    if (!genref_table[slot].in_use)           return 0;

    return (int64_t)(uintptr_t)genref_table[slot].ptr;
}

// __mind_gen_free(handle) — free the allocation at `slot` and increment the
// generation counter so that any surviving handles with the old generation
// return 0 from __mind_gen_deref.
//
// Calling gen_free on an already-freed handle (generation mismatch) is a
// no-op — double-free is silently ignored rather than aborting, because the
// MIND programmer may not be able to prevent a duplicate free in all code
// paths (the type-checker's genref_unchecked_deref diagnostic is the
// compile-time signal; the runtime makes the double-free harmless).
//
// Generation wrap-around (RFC 0010 §7): after 2^32 frees of a single slot
// the 32-bit counter wraps to 0.  The runtime aborts on wrap rather than
// silently creating a window where a 4-billion-calls-stale handle passes the
// check.  This condition is unreachable in practice.
MIND_EXPORT int64_t __mind_gen_free(int64_t handle) {
    if (handle == 0) return 0;

    uint32_t slot = genref_slot(handle);
    uint32_t gen  = genref_gen(handle);

    if (genref_table == NULL || slot >= genref_len) return 0;

    /* Only free if the handle's generation matches the slot's current gen
     * AND the slot is in use.  Stale-handle free is a no-op. */
    if (genref_table[slot].generation != gen) return 0;
    if (!genref_table[slot].in_use)           return 0;

    free(genref_table[slot].ptr);
    genref_table[slot].ptr    = NULL;
    genref_table[slot].in_use = 0;

    /* Increment generation.  Wrap-around guard: abort on overflow rather
     * than silently cycling back to a previously-valid generation. */
    if (genref_table[slot].generation == 0xFFFFFFFFu) {
        abort(); /* generation counter wrap — RFC 0010 §7 */
    }
    genref_table[slot].generation++;

    return 0;
}
