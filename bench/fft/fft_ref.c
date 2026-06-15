// Q16.16 deterministic radix-2 DIT FFT, N=256 — BYTE-IDENTICAL algorithm to
// examples/fft_q16.mind. This is the C codegen baseline (compiled -O3 by the
// system C compiler). Same integer arithmetic => same bits as the MIND .so.
#include <stdint.h>
static inline int64_t qmul(int64_t a,int64_t b){ return (a*b)>>16; }
void fft256_ref(int64_t* re,int64_t* im,const int64_t* tw,int64_t logn,int64_t n){
    // bit-reversal
    for(int64_t i=0;i<n;i++){
        int64_t x=i,j=0,b=0;
        while(b<logn){ j=(j<<1)|(x&1); x>>=1; b++; }
        if(j>i){
            int64_t t=re[i]; re[i]=re[j]; re[j]=t;
            t=im[i]; im[i]=im[j]; im[j]=t;
        }
    }
    int64_t length=2;
    while(length<=n){
        int64_t half=length>>1, step=n/length;
        for(int64_t start=0;start<n;start+=length){
            int64_t k=0;
            for(int64_t jj=0;jj<half;jj++){
                int64_t wr=tw[k*2], wi=tw[k*2+1];
                int64_t a=start+jj, bidx=start+jj+half;
                int64_t xbr=re[bidx], xbi=im[bidx];
                int64_t trr=qmul(wr,xbr)-qmul(wi,xbi);
                int64_t tii=qmul(wr,xbi)+qmul(wi,xbr);
                int64_t uur=re[a], uui=im[a];
                re[a]=uur+trr; im[a]=uui+tii;
                re[bidx]=uur-trr; im[bidx]=uui-tii;
                k+=step;
            }
        }
        length<<=1;
    }
}

// ABI shim matching MIND's `fft256(re,im,tw,logn,n)` exactly: addresses are
// passed as int64 (the same `__mind_load_i64`/`__mind_store_i64` convention),
// and the function returns i64 0. This calls the *same* integer butterflies
// above, so the C `-O3` output is byte-identical to the MIND `.so` output.
int64_t fft256_c(int64_t re, int64_t im, int64_t tw, int64_t logn, int64_t n){
    fft256_ref((int64_t*)(uintptr_t)re, (int64_t*)(uintptr_t)im,
               (const int64_t*)(uintptr_t)tw, logn, n);
    return 0;
}
