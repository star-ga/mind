/* A test-only stand-in for libcuda.so.1 (ruling 16b): the thirteen driver
 * rows with deterministic, pointer-value-free behaviour, so the same native
 * and MLIR programs exit alike against it on a host with no GPU. Compiled by
 * the lane's test harness with cc, for this fixture only, and published to
 * the programs under the filename libcuda.so.1. */
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

/* MIND_FAKE_CUDA_THREAD set: cuInit starts a helper thread that never
 * returns, as the real driver does, so only exit_group ends the process. */
static void *fake_helper(void *arg) {
    for (;;) pause();
    return arg;
}

int cuInit(unsigned int flags) {
    pthread_t helper;
    if (getenv("MIND_FAKE_CUDA_THREAD") != 0) pthread_create(&helper, 0, fake_helper, 0);
    return (int)(flags & 1u);
}

int cuDeviceGet(int *device, int ordinal) {
    *device = 7;
    return ordinal * 3;
}

int cuCtxCreate_v2(void **pctx, unsigned int flags, int dev) {
    *pctx = (void *)0x11;
    return (int)(flags + (unsigned int)dev);
}

int cuCtxDestroy_v2(void *ctx) { return ctx == (void *)0x11 ? 3 : 9; }

int cuCtxSynchronize(void) { return 4; }

int cuModuleLoadData(void **module, const void *image) {
    *module = (void *)0x22;
    return (int)(*(const unsigned char *)image & 0x40);
}

int cuModuleGetFunction(void **hfunc, void *hmod, const char *name) {
    *hfunc = (void *)0x33;
    return (int)(name[0] + 1);
}

int cuModuleUnload(void *hmod) { return 2; }

int cuMemAlloc_v2(unsigned long long *dptr, unsigned long long bytesize) {
    *dptr = 0x1234ull;
    return (int)(bytesize / 32ull);
}

int cuMemcpyHtoD_v2(unsigned long long dst, const void *src,
                    unsigned long long count) {
    return (int)(count % 3ull);
}

int cuMemcpyDtoH_v2(void *dst, unsigned long long src,
                    unsigned long long count) {
    *(unsigned long long *)dst = 11ull;
    return (int)(count % 5ull);
}

int cuMemFree_v2(unsigned long long dptr) { return (int)(dptr & 0xfull); }

/* The 11-argument call, in the driver's order: stream is the ninth
 * parameter (CUstream), then the void** kernel-parameter array, then
 * void** extra. Every register argument and every stack argument
 * contributes. params[0], extra[0] and the stream word are dereferenced,
 * and f is the handle cuModuleGetFunction wrote. The expected value of
 * the lane program is 1920. */
int cuLaunchKernel(void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                   unsigned int bx, unsigned int by, unsigned int bz,
                   unsigned int shm, void *stream, void **params,
                   void **extra) {
    long long acc = 1000;
    acc += (long long)gx + 2 * (long long)gy + 3 * (long long)gz;
    acc += 4 * (long long)bx + 5 * (long long)by + 6 * (long long)bz;
    acc += 7 * (long long)shm;
    acc += 8 * *(long long *)params[0];
    acc += 16 * *(long long *)extra[0];
    acc += 64 * *(long long *)stream;
    acc += (((unsigned long)f & 0xful) == 3ul) ? 100 : 0;
    return (int)acc;
}

/* ---- MC-6 rows: the cuda.h 12.6 macro targets (fake_rows2.mind) ----
 * Every parameter folds into the result or is read or written through, so a
 * wrong order or width changes the exit. */
#include <stddef.h>
typedef long long w64;
typedef unsigned long long u64;

/* CUDA_MEMCPY2D as cuda.h 12.6 declares it (CUDA_MEMCPY2D_v2): the offsets
 * the loader writes are asserted here, so a layout drift fails the build. */
typedef struct {
    size_t srcXInBytes, srcY;
    int srcMemoryType;
    const void *srcHost;
    u64 srcDevice;
    void *srcArray;
    size_t srcPitch, dstXInBytes, dstY;
    int dstMemoryType;
    void *dstHost;
    u64 dstDevice;
    void *dstArray;
    size_t dstPitch, WidthInBytes, Height;
} memcpy2d;
_Static_assert(offsetof(memcpy2d, srcMemoryType) == 16, "srcMemoryType");
_Static_assert(offsetof(memcpy2d, srcDevice) == 32, "srcDevice");
_Static_assert(offsetof(memcpy2d, srcPitch) == 48, "srcPitch");
_Static_assert(offsetof(memcpy2d, dstMemoryType) == 72, "dstMemoryType");
_Static_assert(offsetof(memcpy2d, dstHost) == 80, "dstHost");
_Static_assert(offsetof(memcpy2d, WidthInBytes) == 112, "WidthInBytes");
_Static_assert(offsetof(memcpy2d, Height) == 120, "Height");
_Static_assert(sizeof(memcpy2d) == 128, "CUDA_MEMCPY2D size");

static w64 word(const void *p) { return *(const w64 *)p; }

int cuStreamBeginCapture_v2(void *s, int mode) { return (int)(word(s) * 2 + mode); }

int cuGraphInstantiateWithFlags(void **exec, void *g, u64 flags) {
    *exec = (void *)0x44;
    return (int)(word(g) + (w64)flags * 3);
}

int cuMemcpyHtoDAsync_v2(u64 dst, const void *src, u64 n, void *s) {
    return (int)(dst % 7 + *(const unsigned char *)src + n * 2 + (u64)word(s));
}

int cuMemcpyDtoHAsync_v2(void *dst, u64 src, u64 n, void *s) {
    *(w64 *)dst = 13;
    return (int)(src % 11 + n + (u64)word(s) * 3);
}

int cuEventDestroy_v2(void *e) { return (int)(word(e) + 1); }

int cuStreamDestroy_v2(void *s) { return (int)(word(s) + 2); }

int cuMemHostRegister_v2(void *p, u64 n, unsigned int flags) {
    *(w64 *)p = 17;
    return (int)(n % 13 + flags * 5);
}

int cuMemAllocHost_v2(void **pp, u64 n) {
    *pp = (void *)0x55;
    return (int)(n / 8);
}

int cuGraphAddKernelNode_v2(void **node, void *g, void *const *deps, u64 ndeps,
                            const void *params) {
    *node = (void *)0x66;
    return (int)(word(g) + word(deps[0]) + (w64)ndeps * 4 + *(const unsigned char *)params);
}

int cuGraphExecUpdate_v2(void *exec, void *g, void *info) {
    *(w64 *)info = 23;
    return (int)(word(exec) * 2 + word(g));
}

/* `unsigned char uc`: the row passes it as u32 under a 0..255 fence. */
int cuMemsetD8_v2(u64 d, unsigned char uc, u64 n) { return (int)(d % 3 + uc + n % 5); }

int cuStreamGetCaptureInfo_v2(void *s, int *status, u64 *id, void **graph, void **deps,
                              u64 *ndeps) {
    *status = 1;
    *id = 77;
    *graph = (void *)0x77;
    *deps = (void *)0x88;
    *ndeps = 2;
    return (int)(word(s) + 3);
}

int cuDeviceGetAttribute(int *pi, int attrib, int dev) {
    *pi = attrib * 10 + dev;
    return dev + 1;
}

int cuStreamCreate(void **ps, unsigned int flags) {
    *ps = (void *)0x99;
    return (int)flags + 4;
}

int cuStreamSynchronize(void *s) { return (int)(word(s) % 4); }
int cuStreamQuery(void *s) { return (int)(word(s) + 6); }

int cuMemFreeHost(void *p) { return (int)(word(p) + 1); }

int cuEventCreate(void **pe, unsigned int flags) {
    *pe = (void *)0xaa;
    return (int)flags * 2;
}

int cuEventRecord(void *e, void *s) { return (int)(word(e) + word(s) * 10); }

int cuEventSynchronize(void *e) { return (int)(word(e) - 20); }
int cuEventQuery(void *e) { return (int)(word(e) - 15); }

int cuEventElapsedTime(float *ms, void *start, void *end) {
    *ms = 1.5f;
    return (int)(word(end) - word(start));
}

int cuFuncGetAttribute(int *pi, int attrib, void *f) {
    *pi = 1024;
    return (int)(attrib + word(f));
}

int cuOccupancyMaxActiveBlocksPerMultiprocessor(int *nb, void *f, int bs, u64 smem) {
    *nb = bs / 32;
    return (int)(word(f) + (w64)(smem % 9));
}

/* The 10-argument row: parameters 6..9 travel on the SysV stack, all of
 * them contribute. The expected value of the lane program is 1241. */
int cuLaunchCooperativeKernel(void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                              unsigned int bx, unsigned int by, unsigned int bz,
                              unsigned int shm, void *stream, void **params) {
    w64 acc = 500;
    acc += (w64)gx + 2 * (w64)gy + 3 * (w64)gz;
    acc += 4 * (w64)bx + 5 * (w64)by + 6 * (w64)bz + 7 * (w64)shm;
    acc += 8 * word(params[0]) + 64 * word(stream) + word(f) % 10;
    return (int)acc;
}

int cuGraphDestroy(void *g) { return (int)(word(g) + 10); }

int cuGraphExecDestroy(void *e) { return (int)(word(e) + 20); }

int cuGraphLaunch(void *e, void *s) { return (int)(word(e) * 3 + word(s)); }

static int copy2d(const memcpy2d *c) {
    *(w64 *)c->dstHost = 31;
    return (int)(c->WidthInBytes + c->Height * 2 + (size_t)c->srcMemoryType * 3 +
                 (size_t)c->dstMemoryType * 5 + c->srcPitch % 7 + c->srcDevice % 11);
}

int cuMemcpy2D_v2(const void *p) { return copy2d((const memcpy2d *)p); }

int cuMemcpy2DAsync_v2(const void *p, void *s) {
    return copy2d((const memcpy2d *)p) + (int)word(s) * 10;
}

int cuMemHostAlloc(void **pp, u64 n, unsigned int flags) {
    *pp = (void *)0xbb;
    return (int)(n / 16 + flags);
}

int cuMemHostGetDevicePointer_v2(u64 *pd, void *p, unsigned int flags) {
    *pd = 0x2000ull + flags;
    return (int)(word(p) % 5);
}

int cuStreamEndCapture(void *s, void **pg) {
    *pg = (void *)0xcc;
    return (int)(word(s) + 1);
}
