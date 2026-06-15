// GPU benchmark harness for the deterministic Q16.16 N=256 FFT.
//
// Races the IDENTICAL integer kernel compiled by nvcc at -O3 vs -O0 (a codegen
// race, the GPU analogue of the CPU bench's gcc/clang/nvcc race), and validates:
//   1. byte-identity vs the CPU reference (fft_ref.c, compiled into this host),
//   2. all FFTs in the batch identical to each other,
//   3. identical across repeated kernel launches (determinism / the wedge).
// Timing: cudaEvent, kernel-only AND with host<->device transfer, p50/p95.
//
// Usage: ./harness_gpu [batch] [iters]   (defaults: 4096 200)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N 256
#define LOGN 8
#define HALF 128

// The GPU kernel (defined in fft_q16_gpu.cu, compiled into this TU).
extern "C" __global__ void fft256_batch(
    int64_t*, int64_t*, const int64_t*, const int32_t*, int);

// ---- CPU reference (byte-identical algorithm), compiled into this harness ----
static inline int64_t qmul_ref(int64_t a, int64_t b) { return (a * b) >> 16; }
static void fft256_ref(int64_t* re, int64_t* im, const int64_t* tw,
                       int64_t logn, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int64_t x = i, j = 0, b = 0;
        while (b < logn) { j = (j << 1) | (x & 1); x >>= 1; b++; }
        if (j > i) {
            int64_t t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    int64_t length = 2;
    while (length <= n) {
        int64_t half = length >> 1, step = n / length;
        for (int64_t start = 0; start < n; start += length) {
            int64_t k = 0;
            for (int64_t jj = 0; jj < half; jj++) {
                int64_t wr = tw[k * 2], wi = tw[k * 2 + 1];
                int64_t a = start + jj, bidx = start + jj + half;
                int64_t xbr = re[bidx], xbi = im[bidx];
                int64_t trr = qmul_ref(wr, xbr) - qmul_ref(wi, xbi);
                int64_t tii = qmul_ref(wr, xbi) + qmul_ref(wi, xbr);
                int64_t uur = re[a], uui = im[a];
                re[a] = uur + trr;  im[a] = uui + tii;
                re[bidx] = uur - trr;  im[bidx] = uui - tii;
                k += step;
            }
        }
        length <<= 1;
    }
}

static uint64_t fnv(const int64_t* a, int n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = (const unsigned char*)a;
    for (int i = 0; i < n * 8; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static uint64_t lcg(uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}
static int cmp_d(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
    exit(3);} } while(0)

int main(int argc, char** argv) {
    int batch = (argc > 1) ? atoi(argv[1]) : 4096;
    int iters = (argc > 2) ? atoi(argv[2]) : 200;
    if (batch < 1) batch = 4096;
    if (iters < 1) iters = 200;

    // --- twiddle table Q16.16 (identical to the CPU bench) ---
    int64_t tw[N];
    for (int k = 0; k < N / 2; k++) {
        double a = -2.0 * M_PI * (double)k / (double)N;
        tw[2 * k]     = (int64_t)llround(cos(a) * 65536.0);
        tw[2 * k + 1] = (int64_t)llround(sin(a) * 65536.0);
    }
    // --- bit-reversal permutation table ---
    int32_t brev[N];
    for (int i = 0; i < N; i++) {
        int x = i, j = 0;
        for (int b = 0; b < LOGN; b++) { j = (j << 1) | (x & 1); x >>= 1; }
        brev[i] = j;
    }

    // --- deterministic input signal (same seed as the CPU bench -> same hash) ---
    int64_t re0[N], im0[N];
    uint64_t s = 0x1234567;
    for (int i = 0; i < N; i++) {
        re0[i] = (int64_t)(lcg(&s) % 131072) - 65536;
        im0[i] = (int64_t)(lcg(&s) % 131072) - 65536;
    }

    // --- CPU reference output for the byte-identity gate ---
    int64_t Rre[N], Rim[N];
    memcpy(Rre, re0, sizeof re0); memcpy(Rim, im0, sizeof im0);
    fft256_ref(Rre, Rim, tw, LOGN, N);
    uint64_t ref_hash = fnv(Rre, N) ^ fnv(Rim, N);

    // --- device buffers ---
    size_t bytes = (size_t)batch * N * sizeof(int64_t);
    int64_t *d_re, *d_im, *d_tw; int32_t* d_brev;
    CK(cudaMalloc(&d_re, bytes));
    CK(cudaMalloc(&d_im, bytes));
    CK(cudaMalloc(&d_tw, N * sizeof(int64_t)));
    CK(cudaMalloc(&d_brev, N * sizeof(int32_t)));
    CK(cudaMemcpy(d_tw, tw, N * sizeof(int64_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_brev, brev, N * sizeof(int32_t), cudaMemcpyHostToDevice));

    // host batch buffers: every FFT gets the SAME input signal
    std::vector<int64_t> hre((size_t)batch * N), him((size_t)batch * N);
    for (int f = 0; f < batch; f++) {
        memcpy(&hre[(size_t)f * N], re0, sizeof re0);
        memcpy(&him[(size_t)f * N], im0, sizeof im0);
    }

    auto upload = [&]() {
        CK(cudaMemcpy(d_re, hre.data(), bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_im, him.data(), bytes, cudaMemcpyHostToDevice));
    };

    dim3 grid(batch), block(HALF);

    // ===== correctness gate =====
    upload();
    fft256_batch<<<grid, block>>>(d_re, d_im, d_tw, d_brev, batch);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<int64_t> ore((size_t)batch * N), oim((size_t)batch * N);
    CK(cudaMemcpy(ore.data(), d_re, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(oim.data(), d_im, bytes, cudaMemcpyDeviceToHost));

    // (a) FFT #0 byte-identical to CPU reference?
    int ident_ref = (memcmp(&ore[0], Rre, sizeof Rre) == 0 &&
                     memcmp(&oim[0], Rim, sizeof Rim) == 0);
    uint64_t gpu_hash = fnv(&ore[0], N) ^ fnv(&oim[0], N);

    // (b) all FFTs in the batch identical to FFT #0?
    int ident_batch = 1;
    for (int f = 1; f < batch && ident_batch; f++) {
        if (memcmp(&ore[(size_t)f * N], &ore[0], sizeof Rre) != 0 ||
            memcmp(&oim[(size_t)f * N], &oim[0], sizeof Rim) != 0)
            ident_batch = 0;
    }

    // (c) determinism across repeated launches (fresh upload each time)
    int ident_runs = 1;
    uint64_t first_run_hash = gpu_hash;
    for (int r = 0; r < 5; r++) {
        upload();
        fft256_batch<<<grid, block>>>(d_re, d_im, d_tw, d_brev, batch);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ore.data(), d_re, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(oim.data(), d_im, bytes, cudaMemcpyDeviceToHost));
        uint64_t h = fnv(&ore[0], N) ^ fnv(&oim[0], N);
        if (h != first_run_hash) ident_runs = 0;
    }

    printf("byte_identical_vs_cpu=%s  all_batch_identical=%s  deterministic_across_runs=%s\n",
           ident_ref ? "YES" : "NO",
           ident_batch ? "YES" : "NO",
           ident_runs ? "YES" : "NO");
    printf("gpu_hash=%016llx  cpu_ref_hash=%016llx\n",
           (unsigned long long)gpu_hash, (unsigned long long)ref_hash);
    if (!ident_ref || !ident_batch || !ident_runs) {
        printf("ABORT: byte-identity gate FAILED, benchmark invalid\n");
        return 2;
    }

    // ===== timing =====
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // warm-up
    upload();
    for (int w = 0; w < 20; w++) {
        fft256_batch<<<grid, block>>>(d_re, d_im, d_tw, d_brev, batch);
    }
    CK(cudaDeviceSynchronize());

    std::vector<double> kern_ms(iters), full_ms(iters);

    // kernel-only timing (data already on device; re-uploaded each iter so the
    // input is fixed, then timed region is the launch only)
    for (int it = 0; it < iters; it++) {
        upload();
        CK(cudaEventRecord(e0));
        fft256_batch<<<grid, block>>>(d_re, d_im, d_tw, d_brev, batch);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        kern_ms[it] = ms;
    }

    // with-transfer timing (H2D + kernel + D2H, the realistic end-to-end cost)
    for (int it = 0; it < iters; it++) {
        CK(cudaEventRecord(e0));
        CK(cudaMemcpy(d_re, hre.data(), bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_im, him.data(), bytes, cudaMemcpyHostToDevice));
        fft256_batch<<<grid, block>>>(d_re, d_im, d_tw, d_brev, batch);
        CK(cudaMemcpy(ore.data(), d_re, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(oim.data(), d_im, bytes, cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        full_ms[it] = ms;
    }

    qsort(kern_ms.data(), iters, sizeof(double), cmp_d);
    qsort(full_ms.data(), iters, sizeof(double), cmp_d);
    double k_p50 = kern_ms[iters / 2], k_p95 = kern_ms[(int)(iters * 0.95)];
    double f_p50 = full_ms[iters / 2], f_p95 = full_ms[(int)(iters * 0.95)];

    // per-FFT ns and aggregate throughput
    double k_ns_per_fft = k_p50 * 1e6 / batch;   // ms -> ns, / batch
    double ffts_per_sec = (double)batch / (k_p50 * 1e-3);
    double gflops = (10240.0 * batch) / (k_p50 * 1e6); // 10240 flop/FFT / ns_total

    printf("KERNEL_ONLY  batch=%d  ms_p50=%.4f  ms_p95=%.4f  ns_per_fft=%.2f  ffts_per_sec=%.3e  gflops=%.2f\n",
           batch, k_p50, k_p95, k_ns_per_fft, ffts_per_sec, gflops);
    printf("WITH_XFER    batch=%d  ms_p50=%.4f  ms_p95=%.4f  ns_per_fft=%.2f\n",
           batch, f_p50, f_p95, f_p50 * 1e6 / batch);

    cudaFree(d_re); cudaFree(d_im); cudaFree(d_tw); cudaFree(d_brev);
    return 0;
}
