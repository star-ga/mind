// Deterministic Q16.16 radix-2 DIT FFT, N=256, on the GPU (CUDA).
//
// This kernel implements the EXACT SAME integer arithmetic, butterfly order,
// bit-reversal, and twiddle indexing as the CPU reference (bench/fft/fft_ref.c)
// and the MIND .so kernel (examples/fft_q16.mind). Same integer ops => same
// bits. The complex multiply is fixed-point qmul(a,b) = (a*b)>>16 (arithmetic
// shift) on int64, then integer add/sub — associative and exact, so the result
// is byte-identical regardless of how the work is parallelized across threads.
//
// Layout: a BATCH of many independent N=256 FFTs. One FFT per thread-block;
// HALF (=128) threads per block, one thread per butterfly within a stage. The
// 256-point (re,im) data is staged in shared memory; __syncthreads() between
// the 8 DIT stages enforces the same data dependency the sequential CPU loop
// has. Because each butterfly reads/writes a disjoint pair within a stage, the
// per-stage parallelization is order-independent and the output is identical to
// the sequential reference.
//
// IMPORTANT HONESTY NOTE: This kernel is HAND-WRITTEN CUDA, NOT MIND-emitted.
// MIND has no GPU/PTX codegen path today (src/eval/mlir_gpu.rs is a stub that
// returns "not available; falling back"; --emit-shared/--emit-obj are CPU-only;
// the GPU backends require the proprietary runtime). This bench therefore races
// nvcc's own codegen of the identical integer kernel (-O3 vs -O0) and reports
// the deterministic byte-identity property, which is the actual wedge.

#include <stdint.h>

#define N 256
#define LOGN 8
#define HALF 128  // N/2 butterflies per stage

// Arithmetic-shift fixed-point multiply, identical to the CPU reference.
__device__ __forceinline__ int64_t qmul(int64_t a, int64_t b) {
    return (a * b) >> 16;
}

// Precomputed bit-reversal of i over LOGN bits (host fills this; kept identical
// to the reference's inline bit-reverse loop).
//
// One block == one FFT. blockIdx.x selects which FFT in the batch.
// gre/gim: batch*N int64 each (Q16.16 re/im, in place). tw: N int64 twiddles.
// brev: N int32 bit-reversal permutation (brev[i] = reverse(i, LOGN)).
extern "C" __global__ void fft256_batch(
    int64_t* __restrict__ gre,
    int64_t* __restrict__ gim,
    const int64_t* __restrict__ tw,
    const int32_t* __restrict__ brev,
    int batch)
{
    int fft = blockIdx.x;
    if (fft >= batch) return;

    __shared__ int64_t sre[N];
    __shared__ int64_t sim[N];

    int64_t* re = gre + (int64_t)fft * N;
    int64_t* im = gim + (int64_t)fft * N;

    int t = threadIdx.x;  // 0..HALF-1

    // --- Load with bit-reversal permutation into shared memory ---
    // Each thread moves two elements (2*HALF = N). Writing to position brev[i]
    // reproduces the reference's in-place swap result exactly (a permutation is
    // a permutation regardless of swap vs scatter).
    {
        int i0 = t;
        int i1 = t + HALF;
        sre[brev[i0]] = re[i0];  sim[brev[i0]] = im[i0];
        sre[brev[i1]] = re[i1];  sim[brev[i1]] = im[i1];
    }
    __syncthreads();

    // --- 8 DIT stages ---
    // length = 2,4,...,256 ; half = length/2 ; step = N/length.
    // Reference butterfly index mapping for a flat butterfly id `bf` in [0,HALF):
    //   group  = bf / half          (which length-block)
    //   inner  = bf % half          (jj within the block)
    //   start  = group * length
    //   a      = start + inner
    //   bidx   = a + half
    //   k      = inner * step       (twiddle index)
    int length = 2;
    int step   = N / 2;   // = HALF initially; step = N/length
    #pragma unroll
    for (int s = 0; s < LOGN; s++) {
        int half = length >> 1;
        int group = t / half;
        int inner = t % half;
        int start = group * length;
        int a = start + inner;
        int bidx = a + half;
        int k = inner * step;

        int64_t wr = tw[k * 2];
        int64_t wi = tw[k * 2 + 1];
        int64_t xbr = sre[bidx], xbi = sim[bidx];
        int64_t trr = qmul(wr, xbr) - qmul(wi, xbi);
        int64_t tii = qmul(wr, xbi) + qmul(wi, xbr);
        int64_t uur = sre[a], uui = sim[a];

        __syncthreads();  // all reads of this stage done before any write
        sre[a]    = uur + trr;  sim[a]    = uui + tii;
        sre[bidx] = uur - trr;  sim[bidx] = uui - tii;
        __syncthreads();  // all writes done before next stage reads

        length <<= 1;
        step   >>= 1;
    }

    // --- Store back ---
    {
        int i0 = t;
        int i1 = t + HALF;
        re[i0] = sre[i0];  im[i0] = sim[i0];
        re[i1] = sre[i1];  im[i1] = sim[i1];
    }
}
