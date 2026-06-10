/*
 * xnode_driver.c — cross-MACHINE byte-identity driver for the MIND Track-B
 * deterministic kernels (RFC 0020 §10, RFC 0015 §3.1).
 *
 * Purpose: the in-tree `cross_substrate_identity` Rust gate proves byte-identity
 * across-build / across-time on ONE box. This standalone driver dlopens the SAME
 * workload .so the gate builds (mind_xsi_dot_q16.so), regenerates each workload's
 * seeded LCG input byte-for-byte identically to the Rust harness, runs every
 * kernel, and prints the SHA-256 of the canonical output buffer per canary. Run
 * it on box A and box B (both Linux x86_64) with the SAME .so to prove
 * MACHINE-to-MACHINE byte-identity (this box == S1).
 *
 * Zero external deps: libc + libdl + a vendored public-domain SHA-256. Build:
 *   cc -O2 -o xnode_driver xnode_driver.c -ldl
 * Run:
 *   ./xnode_driver /path/to/mind_xsi_dot_q16.so
 *
 * Encoding contract (mirrors tests/cross_substrate_identity.rs):
 *   seed        = 0xDEADBEEF for every workload
 *   LCG         = s = s*1664525 + 1013904223; out_u32 = (s >> 16)
 *   next_q16    = (int32)out_u32 >> 12
 *   next_i8     = (int8)(out_u32 >> 16)
 *   next_i16    = (int16)(out_u32 >> 16)
 *   scalar dot result is hashed as 8 LE bytes (i64);
 *   vector/matrix results are hashed as their i32 LE bytes in order.
 *
 * STARGA Inc.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------------- *
 * Vendored SHA-256 (public domain, derived from the FIPS-180-4 reference). *
 * ----------------------------------------------------------------------- */
typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  data[64];
    uint32_t datalen;
} sha256_ctx;

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define ROTR(a,b) (((a) >> (b)) | ((a) << (32-(b))))

static void sha256_transform(sha256_ctx *c, const uint8_t *d) {
    uint32_t a,b,e,f,g,h,i,t1,t2,m[64],ss;
    for (i=0,ss=0; i<16; ++i, ss+=4)
        m[i] = (d[ss]<<24)|(d[ss+1]<<16)|(d[ss+2]<<8)|(d[ss+3]);
    for (; i<64; ++i)
        m[i] = (ROTR(m[i-2],17)^ROTR(m[i-2],19)^(m[i-2]>>10)) + m[i-7]
             + (ROTR(m[i-15],7)^ROTR(m[i-15],18)^(m[i-15]>>3)) + m[i-16];
    a=c->state[0]; b=c->state[1]; uint32_t cc=c->state[2]; uint32_t dd=c->state[3];
    e=c->state[4]; f=c->state[5]; g=c->state[6]; h=c->state[7];
    for (i=0; i<64; ++i) {
        t1 = h + (ROTR(e,6)^ROTR(e,11)^ROTR(e,25)) + ((e&f)^((~e)&g)) + K256[i] + m[i];
        t2 = (ROTR(a,2)^ROTR(a,13)^ROTR(a,22)) + ((a&b)^(a&cc)^(b&cc));
        h=g; g=f; f=e; e=dd+t1; dd=cc; cc=b; b=a; a=t1+t2;
    }
    c->state[0]+=a; c->state[1]+=b; c->state[2]+=cc; c->state[3]+=dd;
    c->state[4]+=e; c->state[5]+=f; c->state[6]+=g; c->state[7]+=h;
}

static void sha256_init(sha256_ctx *c) {
    c->datalen=0; c->bitlen=0;
    c->state[0]=0x6a09e667; c->state[1]=0xbb67ae85; c->state[2]=0x3c6ef372; c->state[3]=0xa54ff53a;
    c->state[4]=0x510e527f; c->state[5]=0x9b05688c; c->state[6]=0x1f83d9ab; c->state[7]=0x5be0cd19;
}

static void sha256_update(sha256_ctx *c, const uint8_t *d, size_t len) {
    for (size_t i=0; i<len; ++i) {
        c->data[c->datalen++] = d[i];
        if (c->datalen == 64) { sha256_transform(c, c->data); c->bitlen += 512; c->datalen = 0; }
    }
}

static void sha256_final(sha256_ctx *c, uint8_t *hash) {
    uint32_t i = c->datalen;
    c->data[i++] = 0x80;
    if (c->datalen < 56) { while (i < 56) c->data[i++] = 0; }
    else { while (i < 64) c->data[i++] = 0; sha256_transform(c, c->data); memset(c->data, 0, 56); }
    c->bitlen += (uint64_t)c->datalen * 8;
    for (int j=7; j>=0; --j) c->data[56 + (7-j)] = (uint8_t)(c->bitlen >> (j*8));
    sha256_transform(c, c->data);
    for (i=0; i<4; ++i)
        for (int j=0; j<8; ++j)
            hash[i + j*4] = (uint8_t)(c->state[j] >> (24 - i*8));
}

static void hex_of(const uint8_t *h, char *out) {
    static const char *hx = "0123456789abcdef";
    for (int i=0; i<32; ++i) { out[i*2] = hx[h[i]>>4]; out[i*2+1] = hx[h[i]&0xf]; }
    out[64] = 0;
}

/* ----------------------------------------------------------------------- *
 * Deterministic LCG — byte-identical to tests/cross_substrate_identity.rs  *
 * ----------------------------------------------------------------------- */
static uint64_t g_state;
static void     lcg_seed(uint64_t s) { g_state = s; }
static uint32_t lcg_u32(void) {
    g_state = g_state * 1664525ULL + 1013904223ULL;
    return (uint32_t)(g_state >> 16);
}
static int32_t next_q16(void) { return (int32_t)lcg_u32() >> 12; }
static int8_t  next_i8(void)  { return (int8_t)(lcg_u32() >> 16); }
static int16_t next_i16(void) { return (int16_t)(lcg_u32() >> 16); }

/* Kernel ABIs (pointers passed as i64, matching the MIND intrinsic surface). */
typedef int64_t (*dot_fn)(int64_t, int64_t, int64_t);
typedef int64_t (*mat_fn)(int64_t, int64_t, int64_t, int64_t, int64_t);
typedef int64_t (*gemm_fn)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

static int failures = 0;

static void emit(const char *id, const char *hash, const char *expect) {
    int ok = (expect == NULL) || (strcmp(hash, expect) == 0);
    printf("  %-22s %s", id, hash);
    if (expect) printf("  %s", ok ? "MATCH" : "*** MISMATCH ***");
    printf("\n");
    if (!ok) failures++;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <workload.so> [node-label]\n", argv[0]);
        return 2;
    }
    const char *so_path = argv[1];
    const char *label   = (argc > 2) ? argv[2] : "this-node";

    void *lib = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 3; }

    dot_fn  dotq   = (dot_fn) dlsym(lib, "dotq");
    dot_fn  dotl1q = (dot_fn) dlsym(lib, "dotl1q");
    mat_fn  mmq    = (mat_fn) dlsym(lib, "mmq");
    mat_fn  mmi16  = (mat_fn) dlsym(lib, "mmi16");
    gemm_fn gemmq  = (gemm_fn)dlsym(lib, "gemmq");
    gemm_fn gemmi8 = (gemm_fn)dlsym(lib, "gemmi8");
    if (!dotq||!dotl1q||!mmq||!mmi16||!gemmq||!gemmi8) {
        fprintf(stderr, "missing symbol(s) in %s\n", so_path); return 4;
    }

    /* Canonical committed references (RFC 0015 §3.1: avx2 == neon == x86). */
    const char *REF_GEMM_Q16 = "92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f";
    const char *REF_GEMM_I8  = "917d353b18fd7f5ea4dab7dd02b786f5ccc4a2d954f695084ca0a88214d699c7";
    const char *REF_GEMV_I16 = "3238e8c7e1e9ee9937503700f63eda350fcd10e7db28d470c3dbc26592d0a936";

    sha256_ctx ctx; uint8_t dg[32]; char hex[65];

    printf("node=%s  so=%s\n", label, so_path);
    printf("workload                hash                                                              vs-canary\n");

    /* dot-l2-q16: scalar i64 result, 65536-length. */
    {
        const int N = 65536;
        int32_t *a = malloc(N*4), *b = malloc(N*4);
        lcg_seed(0xDEADBEEF);
        for (int i=0;i<N;++i) a[i]=next_q16();
        for (int i=0;i<N;++i) b[i]=next_q16();
        int64_t r = dotq((int64_t)(intptr_t)a, (int64_t)(intptr_t)b, N);
        sha256_init(&ctx); sha256_update(&ctx, (uint8_t*)&r, 8); sha256_final(&ctx, dg); hex_of(dg, hex);
        emit("dot-l2-q16", hex, NULL);
        free(a); free(b);
    }
    /* dot-l1-q16: scalar i64 result. */
    {
        const int N = 65536;
        int32_t *a = malloc(N*4), *b = malloc(N*4);
        lcg_seed(0xDEADBEEF);
        for (int i=0;i<N;++i) a[i]=next_q16();
        for (int i=0;i<N;++i) b[i]=next_q16();
        int64_t r = dotl1q((int64_t)(intptr_t)a, (int64_t)(intptr_t)b, N);
        sha256_init(&ctx); sha256_update(&ctx, (uint8_t*)&r, 8); sha256_final(&ctx, dg); hex_of(dg, hex);
        emit("dot-l1-q16", hex, NULL);
        free(a); free(b);
    }
    /* gemv-q16-256x256: i32[256] output. */
    {
        const int R=256,C=256;
        int32_t *w = malloc((size_t)R*C*4), *x = malloc(C*4), *y = calloc(R,4);
        lcg_seed(0xDEADBEEF);
        for (int i=0;i<R*C;++i) w[i]=next_q16();
        for (int i=0;i<C;++i) x[i]=next_q16();
        mmq((int64_t)(intptr_t)w,(int64_t)(intptr_t)x,(int64_t)(intptr_t)y,R,C);
        sha256_init(&ctx); sha256_update(&ctx,(uint8_t*)y,(size_t)R*4); sha256_final(&ctx,dg); hex_of(dg,hex);
        emit("gemv-q16-256x256", hex, NULL);
        free(w); free(x); free(y);
    }
    /* gemv-i16-256x256: i32[256] output — CANARY. */
    {
        const int R=256,C=256;
        int16_t *w = malloc((size_t)R*C*2), *x = malloc(C*2);
        int32_t *y = calloc(R,4);
        lcg_seed(0xDEADBEEF);
        for (int i=0;i<R*C;++i) w[i]=next_i16();
        for (int i=0;i<C;++i) x[i]=next_i16();
        mmi16((int64_t)(intptr_t)w,(int64_t)(intptr_t)x,(int64_t)(intptr_t)y,R,C);
        sha256_init(&ctx); sha256_update(&ctx,(uint8_t*)y,(size_t)R*4); sha256_final(&ctx,dg); hex_of(dg,hex);
        emit("gemv-i16-256x256", hex, REF_GEMV_I16);
        free(w); free(x); free(y);
    }
    /* gemm-q16-64x64x64: i32[64*64] output, kernel consumes Bt (N*K). CANARY. */
    {
        const int M=64,K=64,N=64;
        int32_t *a = malloc((size_t)M*K*4), *b = malloc((size_t)K*N*4);
        int32_t *bt = malloc((size_t)N*K*4), *c = calloc((size_t)M*N,4);
        lcg_seed(0xDEADBEEF);
        for (int i=0;i<M*K;++i) a[i]=next_q16();
        for (int i=0;i<K*N;++i) b[i]=next_q16();
        for (int kk=0;kk<K;++kk) for (int j=0;j<N;++j) bt[j*K+kk]=b[kk*N+j];
        gemmq((int64_t)(intptr_t)a,(int64_t)(intptr_t)bt,(int64_t)(intptr_t)c,M,K,N);
        sha256_init(&ctx); sha256_update(&ctx,(uint8_t*)c,(size_t)M*N*4); sha256_final(&ctx,dg); hex_of(dg,hex);
        emit("gemm-q16-64x64x64", hex, REF_GEMM_Q16);
        free(a); free(b); free(bt); free(c);
    }
    /* gemm-i8-64x64x64: i32[64*64] output, kernel consumes B (K*N, untransposed). CANARY. */
    {
        const int M=64,K=64,N=64;
        int8_t *a = malloc((size_t)M*K), *b = malloc((size_t)K*N);
        int32_t *c = calloc((size_t)M*N,4);
        lcg_seed(0xDEADBEEF);
        for (int i=0;i<M*K;++i) a[i]=next_i8();
        for (int i=0;i<K*N;++i) b[i]=next_i8();
        gemmi8((int64_t)(intptr_t)a,(int64_t)(intptr_t)b,(int64_t)(intptr_t)c,M,K,N);
        sha256_init(&ctx); sha256_update(&ctx,(uint8_t*)c,(size_t)M*N*4); sha256_final(&ctx,dg); hex_of(dg,hex);
        emit("gemm-i8-64x64x64", hex, REF_GEMM_I8);
        free(a); free(b); free(c);
    }

    dlclose(lib);
    if (failures) { printf("\nRESULT: %d canary MISMATCH(es) on node %s — DETERMINISM BUG\n", failures, label); return 1; }
    printf("\nRESULT: all canaries MATCH on node %s\n", label);
    return 0;
}
