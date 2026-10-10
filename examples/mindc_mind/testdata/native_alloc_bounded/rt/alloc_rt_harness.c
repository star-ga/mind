// Standalone stress harness for the bounded-profile allocator blob (rt.bin).
// Maps a small arena + overflow window at fixed addresses, patches the blob's
// placeholders, then runs random alloc/free/realloc traces against a model
// and walks the heap after every operation to check its invariants.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static uint64_t A, AB, ST, OB, CAP;
typedef int64_t (*fn2)(int64_t, int64_t);
static fn2 f_alloc, f_free, f_realloc;

static void die(const char *m, long a, long b) {
    fprintf(stderr, "FAIL %s %ld %ld\n", m, a, b);
    exit(3);
}

static void setup(const char *bin, uint64_t arena_bytes) {
    A = 0x10000000ULL;
    AB = arena_bytes;
    ST = A + AB - 4096;
    OB = 0x200000000ULL;
    CAP = AB - 8 - 4096;
    void *ar = mmap((void *)A, AB, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (ar != (void *)A) die("arena map", 0, 0);
    FILE *f = fopen(bin, "rb");
    uint8_t buf[8192];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    uint64_t vals[7] = {0, A, A + 8, ST, OB, A + 16, OB + 8};
    for (size_t i = 0; i + 8 <= n; i++) {
        if (buf[i] >= 1 && buf[i] <= 6) {
            int ok = 1;
            for (int j = 1; j < 8; j++) if (buf[i + j] != 0x11) ok = 0;
            if (ok) { memcpy(buf + i, &vals[buf[i]], 8); i += 7; }
        }
    }
    buf[n] = 0xC3;  // ret after rt_end
    uint8_t *code = mmap(NULL, 16384, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memcpy(code, buf, n + 1);
    ((void (*)(void))code)();
    f_alloc = (fn2)(*(uint64_t *)(ST + 536));
    f_free = (fn2)(*(uint64_t *)(ST + 544));
    f_realloc = (fn2)(*(uint64_t *)(ST + 552));
}

static int64_t top(void) { return (int64_t)(A + 8 + *(uint64_t *)A); }
static uint64_t W(uint64_t a) { return *(uint64_t *)a; }

// Walk one region [lo, end) of blocks; check headers, P bits, footers, free
// blocks in bins. Returns number of free blocks seen.
static long walk(uint64_t lo, uint64_t end, int is_top_region) {
    uint64_t h = lo, prev_free = 0;
    long nfree = 0;
    while (h < end) {
        uint64_t hd = W(h), s = hd & ~7ULL;
        if (((hd & 2) != 0) != (prev_free != 0)) die("pbit", (long)h, (long)hd);
        if (h + 8 + s > end) die("overrun", (long)h, (long)s);
        if (hd & 1) {
            if (prev_free) die("adjacent free", (long)h, 0);
            if (s < 24) die("small free", (long)h, (long)s);
            if (W(h + s) != s) die("footer", (long)h, (long)s);
            int k = 63 - __builtin_clzll(s);
            uint64_t d = W(ST + 8 * k), pv = 0, found = 0;
            while (d) {
                if (W(d + 8) != pv) die("prev link", (long)d, (long)pv);
                if (d == h + 8) found = 1;
                pv = d; d = W(d);
            }
            if (!found) die("free not in bin", (long)h, k);
            if (!(W(ST + 520) >> k & 1)) die("bitmap", k, 0);
            nfree++;
            if (is_top_region && h + 8 + s == end) die("free below top", (long)h, 0);
        }
        prev_free = hd & 1;
        h += 8 + s;
    }
    if (h != end) die("walk end", (long)h, (long)end);
    return nfree;
}

static void check(void) {
    uint64_t t = (uint64_t)top();
    long nf = 0;
    if (t <= ST) {
        nf += walk(A + 8, t, 1);
    } else {
        nf += walk(A + 8, ST, 0);
        nf += walk(OB, t, 1);
    }
    long nb = 0;
    for (int k = 0; k < 64; k++) {
        uint64_t d = W(ST + 8 * k);
        if (((W(ST + 520) >> k) & 1) != (d != 0)) die("bitmap/bin", k, 0);
        while (d) {
            uint64_t s = W(d - 8) & ~7ULL;
            if (!(W(d - 8) & 1)) die("binned not free", (long)d, k);
            if (63 - __builtin_clzll(s) != k) die("wrong bin", (long)d, k);
            nb++;
            d = W(d);
        }
    }
    if (nb != nf) die("bin count", nb, nf);
}

#define MAXL 4096
static int64_t lp[MAXL], ln[MAXL], lw[MAXL];
static uint8_t lseed[MAXL];
static int nl;
static uint64_t rng = 88172645463325252ULL;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static void fill(int64_t p, int64_t n, uint8_t seed) {
    for (int64_t i = 0; i < n; i++) ((uint8_t *)p)[i] = (uint8_t)(seed + i * 7);
}
static void verify(int64_t p, int64_t n, uint8_t seed, const char *m) {
    for (int64_t i = 0; i < n; i++)
        if (((uint8_t *)p)[i] != (uint8_t)(seed + i * 7)) die(m, p, i);
}
static void zero_ok(int64_t p, int64_t from, int64_t n, const char *m) {
    for (int64_t i = from; i < n; i++)
        if (((uint8_t *)p)[i] != 0) { fprintf(stderr, "from=%ld n=%ld ", (long)from, (long)n); die(m, p, i); }
}
static void overlap(int64_t p, int64_t n, int skip) {
    for (int i = 0; i < nl; i++) {
        if (i == skip) continue;
        if (p < lp[i] + ln[i] && lp[i] < p + n) die("overlap", p, lp[i]);
    }
}

static int64_t gen_size(int mode) {
    uint64_t r = rnd() % 100;
    if (mode == 1) return (int64_t)(rnd() % 3000000) + 1;
    if (r < 50) return (int64_t)(rnd() % 64) + 1;
    if (r < 80) return (int64_t)(rnd() % 1024) + 1;
    if (r < 97) return (int64_t)(rnd() % 65536) + 1;
    return (int64_t)(rnd() % 4000000) + 1;
}

static void trace(long steps, int mode) {
    for (long st = 0; st < steps; st++) {
        uint64_t op = rnd() % 10;
        if ((op < 4 && nl < MAXL) || nl == 0) {
            int64_t n = gen_size(mode);
            int64_t p = f_alloc(0, n);
            if (!p || (p & 7)) die("alloc ret", p, n);
            zero_ok(p, 0, n, "alloc not zero");
            overlap(p, n, -1);
            lw[nl] = n; lp[nl] = p; ln[nl] = n; lseed[nl] = (uint8_t)rnd(); fill(p, n, lseed[nl]); nl++;
        } else if (op < 7) {
            int i = (int)(rnd() % nl);
            verify(lp[i], ln[i], lseed[i], "before free");
            if (f_free(lp[i], 0) != 0) die("free ret", 0, 0);
            lp[i] = lp[nl - 1]; ln[i] = ln[nl - 1]; lw[i] = lw[nl - 1]; lseed[i] = lseed[nl - 1]; nl--;
        } else {
            int i = (int)(rnd() % nl);
            int64_t n = (rnd() % 3) ? ln[i] + gen_size(mode) / 4 + 1 : (int64_t)(rnd() % (ln[i] + 1)) + 1;
            verify(lp[i], ln[i], lseed[i], "before realloc");
            int64_t q = f_realloc(lp[i], n);
            if (!q || (q & 7)) die("realloc ret", q, n);
            int64_t keep = n < ln[i] ? n : ln[i];
            verify(q, keep, lseed[i], "realloc kept");
            if (n > lw[i]) zero_ok(q, lw[i], n, q == lp[i] ? "inplace not zero" : "moved not zero");
            overlap(q, n, i);
            lp[i] = q; ln[i] = n; if (n > lw[i]) lw[i] = n; fill(q, n, lseed[i]);
        }
        check();
    }
    while (nl) {
        int i = nl - 1;
        verify(lp[i], ln[i], lseed[i], "drain");
        f_free(lp[i], 0);
        nl--;
        check();
    }
    if (top() <= (int64_t)ST) { if (*(uint64_t *)A != 0) die("cursor not back at 0", (long)*(uint64_t *)A, 0); }
    else {
        if ((uint64_t)top() != OB) die("O not empty", top(), 0);
        if ((W(A + 8) & ~7ULL) != ST - A - 16 || !(W(A + 8) & 1)) die("P not one free block", (long)W(A + 8), 0);
        printf("drained into O: P is one free block of %lu\n", (unsigned long)(W(A + 8) & ~7ULL));
    }
}

// A fitting block deep in the request's own bin must be found: bin 9 holds
// 1000 (head) then 1016; a 1016-byte request must reuse the second block
// instead of bumping the cursor.
static void bin_search_test(void) {
    int64_t a = f_alloc(0, 1016), s1 = f_alloc(0, 8), b = f_alloc(0, 1000), s2 = f_alloc(0, 8);
    f_free(a, 0);
    f_free(b, 0);
    uint64_t c0 = W(A);
    int64_t q = f_alloc(0, 1016);
    if (q != a) die("bin search missed a fitting block", q, a);
    if (W(A) != c0) die("bin search bumped", (long)W(A), (long)c0);
    f_free(q, 0); f_free(s1, 0); f_free(s2, 0);
    check();
    if (W(A) != 0) die("bin test not drained", (long)W(A), 0);
}

// A large block freed at the top gives its pages back (madvise) and the next,
// larger block still reads zero everywhere.
static void big_top_test(void) {
    int64_t n = 100 << 20;
    if ((uint64_t)n * 2 > AB) return;
    int64_t p = f_alloc(0, n);
    memset((void *)p, 0x5a, (size_t)n);
    f_free(p, 0);
    check();
    if (W(ST + 560) > A + 8 + 4096) die("pages above the top not released", (long)W(ST + 560), 0);
    int64_t q = f_alloc(0, n + 4096);
    if (q != p) die("big block not reused at the top", q, p);
    zero_ok(q, 0, n + 4096, "big reuse not zero");
    f_free(q, 0);
    check();
}

int main(int argc, char **argv) {
    uint64_t ab = argc > 2 ? strtoull(argv[2], 0, 0) : (16u << 20);
    long steps = argc > 3 ? atol(argv[3]) : 20000;
    int mode = argc > 4 ? atoi(argv[4]) : 0;
    rng ^= argc > 5 ? strtoull(argv[5], 0, 0) : 0;
    setup(argv[1], ab);
    // edge cases (C runtime parity)
    if (f_alloc(0, 0) != 0 || f_alloc(0, -1) != 0 || f_alloc(0, INT64_MIN) != 0) die("alloc nonpos", 0, 0);
    int64_t z = f_realloc(0, 0);
    if (z == 0) die("realloc(0,0) null", 0, 0);
    if (f_realloc(0, -5) != 0) die("realloc(0,-5)", 0, 0);
    if (f_realloc(z, -5) != 0) die("realloc(p,-5)", 0, 0);
    if (f_realloc(z, 0) != 0) die("realloc(p,0)", 0, 0);
    if (f_free(0, 0) != 0 || f_free(12345, 0) != 0) die("free foreign", 0, 0);
    check();
    if (*(uint64_t *)A != 0) die("realloc(p,0) did not free", (long)*(uint64_t *)A, 0);
    bin_search_test();
    big_top_test();
    trace(steps, mode);
    // grow loops: in-place at top
    int64_t b = f_alloc(0, 8);
    for (int64_t i = 1; i < 30000; i++) {
        int64_t q = f_realloc(b, i * 8 + 8);
        if (q != b) die("grow8 moved", i, 0);
        *(int64_t *)(q + i * 8) = i;
    }
    f_free(b, 0);
    check();
    for (int64_t i = 1; i < 30000; i++) {
        int64_t p = f_alloc(0, i * 8);
        *(int64_t *)(p + i * 8 - 8) = i;
        f_free(p, 0);
    }
    check();
    if (top() != (int64_t)(A + 8) && top() != (int64_t)OB) die("afl_inc cursor", (long)*(uint64_t *)A, 0);
    // interleaved two-buffer growth
    int64_t x = 0, y = 0;
    for (int64_t i = 1; i < 20000; i++) {
        x = f_realloc(x, i * 8);
        *(int64_t *)(x + i * 8 - 8) = i;
        y = f_realloc(y, i * 16);
        *(int64_t *)(y + i * 16 - 8) = -i;
        if (i % 1000 == 0) {
            for (int64_t j = 1; j <= i; j++)
                if (*(int64_t *)(x + j * 8 - 8) != j) die("x lost", i, j);
            for (int64_t j = 1; j <= i; j++)
                if (*(int64_t *)(y + j * 16 - 8) != -j) die("y lost", i, j);
            check();
        }
    }
    printf("interleaved peak cursor=%lu top=%lx\n", (unsigned long)*(uint64_t *)A, (unsigned long)top());
    f_free(x, 0); f_free(y, 0);
    check();
    printf("OK cursor=%lu top=%lx ME=%lx\n", (unsigned long)*(uint64_t *)A, (unsigned long)top(), (unsigned long)W(ST + 512));
    return 0;
}
