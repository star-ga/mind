// Cross-check harness: load the MIND-compiled fft256 from a .so and assert its
// Q16.16 FFT output is BYTE-IDENTICAL to the C reference on the same input and
// twiddle table. Prints both FNV-1a hashes. Exit 0 iff identical.
//
// Usage: ./fft_verify <path-to-mind.so> [seed]

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

extern int64_t fft256_c(int64_t re, int64_t im, int64_t tw, int64_t logn, int64_t n);

#define N 256
#define LOGN 8

typedef int64_t (*fft_fn)(int64_t, int64_t, int64_t, int64_t, int64_t);

typedef struct { uint64_t s; } Lcg;
static uint32_t lcg_next_u32(Lcg *g) {
    g->s = g->s * 1664525ull + 1013904223ull;
    return (uint32_t)(g->s >> 16);
}
static int64_t lcg_next_q16(Lcg *g) {
    return (int64_t)((int32_t)lcg_next_u32(g) >> 13);
}
static void build_twiddles(int64_t *tw) {
    for (int k = 0; k < N / 2; k++) {
        double ang = -2.0 * M_PI * (double)k / (double)N;
        tw[2 * k + 0] = (int64_t)llround(cos(ang) * 65536.0);
        tw[2 * k + 1] = (int64_t)llround(sin(ang) * 65536.0);
    }
}
static void make_input(int64_t *re, int64_t *im, uint64_t seed) {
    Lcg g; g.s = seed;
    for (int i = 0; i < N; i++) re[i] = lcg_next_q16(&g);
    for (int i = 0; i < N; i++) im[i] = lcg_next_q16(&g);
}
static uint64_t hash_buffers(const int64_t *re, const int64_t *im) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char *p;
    for (int i = 0; i < N; i++) { p = (const unsigned char *)&re[i];
        for (int b = 0; b < 8; b++) { h ^= p[b]; h *= 1099511628211ull; } }
    for (int i = 0; i < N; i++) { p = (const unsigned char *)&im[i];
        for (int b = 0; b < 8; b++) { h ^= p[b]; h *= 1099511628211ull; } }
    return h;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <mind.so> [seed]\n", argv[0]); return 2; }
    uint64_t seed = (argc > 2) ? strtoull(argv[2], NULL, 0) : 0x12345678ull;

    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 2; }
    fft_fn mind_fft = (fft_fn)dlsym(h, "fft256");
    if (!mind_fft) { fprintf(stderr, "dlsym fft256 failed: %s\n", dlerror()); return 2; }

    int64_t tw[N];
    build_twiddles(tw);

    // C reference
    int64_t re_c[N], im_c[N];
    make_input(re_c, im_c, seed);
    fft256_c((int64_t)(uintptr_t)re_c, (int64_t)(uintptr_t)im_c,
             (int64_t)(uintptr_t)tw, LOGN, N);
    uint64_t hc = hash_buffers(re_c, im_c);

    // MIND .so
    int64_t re_m[N], im_m[N];
    make_input(re_m, im_m, seed);
    mind_fft((int64_t)(uintptr_t)re_m, (int64_t)(uintptr_t)im_m,
             (int64_t)(uintptr_t)tw, LOGN, N);
    uint64_t hm = hash_buffers(re_m, im_m);

    int identical = (hc == hm) && (memcmp(re_c, re_m, sizeof(re_c)) == 0)
                                && (memcmp(im_c, im_m, sizeof(im_c)) == 0);
    printf("C_REF   fnv1a=0x%016llx\n", (unsigned long long)hc);
    printf("MIND_SO fnv1a=0x%016llx\n", (unsigned long long)hm);
    printf("BYTE_IDENTICAL=%s\n", identical ? "YES" : "NO");

    if (!identical) {
        int shown = 0;
        for (int i = 0; i < N && shown < 8; i++) {
            if (re_c[i] != re_m[i] || im_c[i] != im_m[i]) {
                printf("  diff[%d] C=(%lld,%lld) MIND=(%lld,%lld)\n", i,
                       (long long)re_c[i], (long long)im_c[i],
                       (long long)re_m[i], (long long)im_m[i]);
                shown++;
            }
        }
    }
    dlclose(h);
    return identical ? 0 : 1;
}
