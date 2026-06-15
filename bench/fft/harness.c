// Self-contained benchmark harness for the deterministic Q16.16 N=256 FFT.
//
// Compares the MIND-compiled .so (dlopen'd) against the byte-identical C
// reference kernel (fft_ref.c) that is COMPILED INTO THIS HARNESS. Build this
// file with gcc / clang / nvcc to obtain that compiler's baseline; the MIND
// number is invariant of which compiler builds the harness (it comes from the
// pre-built .so).
//
// Reports, per kernel:
//   - byte-identity vs the C reference (must be YES, hash a5b24cb31a7f2c7f)
//   - p50 / p95 ns per FFT over ITERS calls (default 200000)
//   - GFLOP/s at p50 = (5*N*log2(N)) / ns_p50 = 10240 / ns_p50
//
// Timing: each call is a single in-place FFT on a fixed buffer. The FFT control
// flow is fully data-independent (same butterfly count every call), so repeated
// in-place calls are a fair, standard microbench. We clock_gettime around the
// bare kernel call only — no memcpy / input regen inside the timed region.
//
// Usage: ./harness [so_path] [iters]   (defaults: /tmp/fft_test.so 200000)
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N 256
#define LOGN 8

typedef int64_t (*fft_fn)(int64_t, int64_t, int64_t, int64_t, int64_t);
// C reference (byte-identical algorithm), compiled into this harness.
extern int64_t fft256_c(int64_t, int64_t, int64_t, int64_t, int64_t);

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static uint64_t fnv(const int64_t *a, int n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)a;
    for (int i = 0; i < n * 8; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
// LCG-free fixed input: matches the original harness so the hash stays pinned.
static uint64_t lcg(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}

int main(int argc, char **argv) {
    const char *so_path = (argc > 1) ? argv[1] : "/tmp/fft_test.so";
    int ITERS = (argc > 2) ? atoi(argv[2]) : 200000;
    if (ITERS < 1) ITERS = 200000;

    void *h = dlopen(so_path, RTLD_NOW);
    if (!h) { printf("dlopen %s: %s\n", so_path, dlerror()); return 1; }
    fft_fn mind = (fft_fn)dlsym(h, "fft256");
    if (!mind) { printf("dlsym fft256 failed\n"); return 1; }

    // Real twiddle table Q16.16: W_N^k = (cos, sin) for DIT, k=0..N/2-1.
    int64_t tw[N];
    for (int k = 0; k < N / 2; k++) {
        double a = -2.0 * M_PI * (double)k / (double)N;
        tw[2 * k]     = (int64_t)llround(cos(a) * 65536.0);
        tw[2 * k + 1] = (int64_t)llround(sin(a) * 65536.0);
    }
    // Deterministic random Q16.16 signal (fixed seed -> pinned hash).
    int64_t re0[N], im0[N];
    uint64_t s = 0x1234567;
    for (int i = 0; i < N; i++) {
        re0[i] = (int64_t)(lcg(&s) % 131072) - 65536;
        im0[i] = (int64_t)(lcg(&s) % 131072) - 65536;
    }

    // --- correctness gate: run both on a fresh copy, compare bytes ---
    int64_t Are[N], Aim[N], Bre[N], Bim[N];
    memcpy(Are, re0, sizeof re0); memcpy(Aim, im0, sizeof im0);
    memcpy(Bre, re0, sizeof re0); memcpy(Bim, im0, sizeof im0);
    mind((int64_t)(uintptr_t)Are, (int64_t)(uintptr_t)Aim, (int64_t)(uintptr_t)tw, LOGN, N);
    fft256_c((int64_t)(uintptr_t)Bre, (int64_t)(uintptr_t)Bim, (int64_t)(uintptr_t)tw, LOGN, N);
    int ident = (memcmp(Are, Bre, sizeof Are) == 0 && memcmp(Aim, Bim, sizeof Aim) == 0);
    uint64_t mind_hash = fnv(Are, N) ^ fnv(Aim, N);
    uint64_t ref_hash  = fnv(Bre, N) ^ fnv(Bim, N);
    printf("byte_identical=%s mind_hash=%016llx ref_hash=%016llx\n",
           ident ? "YES" : "NO",
           (unsigned long long)mind_hash, (unsigned long long)ref_hash);
    if (!ident) { printf("ABORT: outputs differ, benchmark invalid\n"); return 2; }

    double *sm = (double *)malloc(sizeof(double) * ITERS);
    double *sc = (double *)malloc(sizeof(double) * ITERS);
    if (!sm || !sc) { printf("malloc\n"); return 1; }

    // Warm-up both kernels (in place on a fixed working buffer).
    memcpy(Are, re0, sizeof re0); memcpy(Aim, im0, sizeof im0);
    memcpy(Bre, re0, sizeof re0); memcpy(Bim, im0, sizeof im0);
    for (int w = 0; w < 2000; w++) {
        mind((int64_t)(uintptr_t)Are, (int64_t)(uintptr_t)Aim, (int64_t)(uintptr_t)tw, LOGN, N);
        fft256_c((int64_t)(uintptr_t)Bre, (int64_t)(uintptr_t)Bim, (int64_t)(uintptr_t)tw, LOGN, N);
    }

    // MIND timed: per-call ns sample.
    for (int it = 0; it < ITERS; it++) {
        double t0 = now_sec();
        mind((int64_t)(uintptr_t)Are, (int64_t)(uintptr_t)Aim, (int64_t)(uintptr_t)tw, LOGN, N);
        double t1 = now_sec();
        sm[it] = (t1 - t0) * 1e9;
    }
    // C ref timed: per-call ns sample.
    for (int it = 0; it < ITERS; it++) {
        double t0 = now_sec();
        fft256_c((int64_t)(uintptr_t)Bre, (int64_t)(uintptr_t)Bim, (int64_t)(uintptr_t)tw, LOGN, N);
        double t1 = now_sec();
        sc[it] = (t1 - t0) * 1e9;
    }
    // Guard against dead-code elimination of the FFTs.
    volatile int64_t sink = Are[1] ^ Aim[2] ^ Bre[3] ^ Bim[4];
    (void)sink;

    qsort(sm, ITERS, sizeof(double), cmp_d);
    qsort(sc, ITERS, sizeof(double), cmp_d);
    double m_p50 = sm[ITERS / 2], m_p95 = sm[(int)(ITERS * 0.95)];
    double c_p50 = sc[ITERS / 2], c_p95 = sc[(int)(ITERS * 0.95)];

    printf("MIND   ns_p50=%.1f ns_p95=%.1f gflops_p50=%.3f\n",
           m_p50, m_p95, 10240.0 / m_p50);
    printf("CREF   ns_p50=%.1f ns_p95=%.1f gflops_p50=%.3f\n",
           c_p50, c_p95, 10240.0 / c_p50);
    printf("ratio_cref_over_mind_p50=%.3f (>1 => MIND faster) iters=%d\n",
           c_p50 / m_p50, ITERS);
    free(sm); free(sc);
    dlclose(h);
    return 0;
}
