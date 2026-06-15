// Context-only comparison: cuFFT FP32 batched N=256 C2C forward FFT.
//
// This is NOT an apples-to-apples codegen race (cuFFT is a closed FP32 library,
// our kernel is a deterministic Q16.16 integer kernel). It exists purely to give
// a sense of scale vs the vendor library — and, crucially, to demonstrate that
// cuFFT FP32 is NOT bit-reproducible the way the integer kernel is: rerun it and
// the hash of the FP32 output is shown; the WEDGE is that an FP32 FFT cannot
// promise byte-identity across order/width/thread-count (IEEE-754 add is
// non-associative). Here on a fixed plan it happens to repeat, but it carries no
// such guarantee across configs/substrates, which the integer kernel does.
//
// Usage: ./cufft_context [batch] [iters]   (defaults: 65536 300)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>

#define N 256

static uint64_t lcg(uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}
static int cmp_d(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static uint64_t fnv32(const float* a, int n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = (const unsigned char*)a;
    for (int i = 0; i < n * 4; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %d\n",cudaGetErrorString(e),__LINE__); exit(3);} } while(0)
#define CF(call) do { cufftResult e=(call); if(e!=CUFFT_SUCCESS){ \
    printf("cuFFT error %d at %d\n",(int)e,__LINE__); exit(3);} } while(0)

int main(int argc, char** argv) {
    int batch = (argc > 1) ? atoi(argv[1]) : 65536;
    int iters = (argc > 2) ? atoi(argv[2]) : 300;
    if (batch < 1) batch = 65536;
    if (iters < 1) iters = 300;

    // same deterministic input signal, as FP32 complex (cufftComplex)
    std::vector<cufftComplex> h((size_t)batch * N);
    {
        cufftComplex sig[N];
        uint64_t s = 0x1234567;
        for (int i = 0; i < N; i++) {
            // mirror the Q16.16 input but as float (value = q16 / 65536)
            int64_t r = (int64_t)(lcg(&s) % 131072) - 65536;
            int64_t im = (int64_t)(lcg(&s) % 131072) - 65536;
            sig[i].x = (float)r / 65536.0f;
            sig[i].y = (float)im / 65536.0f;
        }
        for (int f = 0; f < batch; f++)
            memcpy(&h[(size_t)f * N], sig, sizeof sig);
    }

    cufftComplex* d;
    size_t bytes = (size_t)batch * N * sizeof(cufftComplex);
    CK(cudaMalloc(&d, bytes));

    cufftHandle plan;
    CF(cufftPlan1d(&plan, N, CUFFT_C2C, batch));

    auto upload = [&]() { CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice)); };

    // determinism probe (fixed plan): rerun, hash FFT #0
    upload();
    CF(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
    CK(cudaDeviceSynchronize());
    std::vector<cufftComplex> out((size_t)batch * N);
    CK(cudaMemcpy(out.data(), d, bytes, cudaMemcpyDeviceToHost));
    uint64_t h1 = fnv32((const float*)&out[0], N * 2);
    int repeats_same = 1;
    for (int r = 0; r < 3; r++) {
        upload();
        CF(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), d, bytes, cudaMemcpyDeviceToHost));
        if (fnv32((const float*)&out[0], N * 2) != h1) repeats_same = 0;
    }
    printf("cufft_fp32_hash=%016llx  repeats_same_on_fixed_plan=%s  (NO cross-config guarantee)\n",
           (unsigned long long)h1, repeats_same ? "yes" : "no");

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    upload();
    for (int w = 0; w < 20; w++) CF(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
    CK(cudaDeviceSynchronize());

    std::vector<double> ms(iters);
    for (int it = 0; it < iters; it++) {
        upload();
        CK(cudaEventRecord(e0));
        CF(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float t = 0; CK(cudaEventElapsedTime(&t, e0, e1));
        ms[it] = t;
    }
    qsort(ms.data(), iters, sizeof(double), cmp_d);
    double p50 = ms[iters / 2], p95 = ms[(int)(iters * 0.95)];
    printf("CUFFT_FP32   batch=%d  ms_p50=%.4f  ms_p95=%.4f  ns_per_fft=%.2f  ffts_per_sec=%.3e\n",
           batch, p50, p95, p50 * 1e6 / batch, (double)batch / (p50 * 1e-3));

    cufftDestroy(plan);
    cudaFree(d);
    return 0;
}
