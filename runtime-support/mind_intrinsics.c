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
int __mind_blas_set_use_avx2(int v) {
    int prev = mind_blas_use_avx2;
    mind_blas_use_avx2 = (v != 0) ? 1 : 0;
    return prev;
}

// Read the dispatcher flag — the harness uses this to sanity-check that AVX2
// was detected at .so load on hosts that have it.
int __mind_blas_get_use_avx2(void) {
    return mind_blas_use_avx2;
}

// CPU-feature probe — AVX2 (256-bit lanes) + FMA (fused multiply-add).
// CPUs that have AVX2 typically also have FMA, but we check the pair so a
// Haswell-without-FMA outlier falls back to scalar instead of SIGILL-ing
// inside _mm256_fmadd_ps.
static int mind_blas_cpu_has_avx2_fma(void) {
#if MIND_BLAS_X86_64
#  if defined(_MSC_VER) && !defined(__clang__)
    // MSVC: documented __cpuid / __cpuidex intrinsics from <intrin.h>.
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) return 0;
    __cpuidex(regs, 7, 0);
    int has_avx2 = (regs[1] >> 5) & 1;    // CPUID 7,0 EBX bit 5
    __cpuid(regs, 1);
    int has_fma  = (regs[2] >> 12) & 1;   // CPUID 1   ECX bit 12
    return has_avx2 && has_fma;
#  else
    // GCC/Clang: builtins — clang's __builtin_cpu_init is a no-op, gcc needs it.
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
