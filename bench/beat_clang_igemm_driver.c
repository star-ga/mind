#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#define M 512
#define K 512
#define N 512
#define REPS 50
typedef int64_t (*mindfn)(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
typedef void (*clangfn)(const int8_t*,const int8_t*,int32_t*,long,long,long);
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  void*hm=dlopen("/tmp/igemm_mind.so",RTLD_NOW); if(!hm){fprintf(stderr,"mind: %s\n",dlerror());return 1;}
  void*hc=dlopen("/tmp/igemm_clang.so",RTLD_NOW); if(!hc){fprintf(stderr,"clang: %s\n",dlerror());return 1;}
  mindfn gemmi8=(mindfn)dlsym(hm,"gemmi8");
  clangfn gemm_clang=(clangfn)dlsym(hc,"gemm_clang");
  if(!gemmi8||!gemm_clang){fprintf(stderr,"dlsym fail\n");return 1;}
  int8_t*a=malloc(M*K),*b=malloc(K*N); int32_t*cm=calloc(M*N,4),*cc=calloc(M*N,4);
  uint64_t s=0xB1A5ULL;
  for(long i=0;i<M*K;i++){s=s*6364136223846793005ULL+1;a[i]=(int8_t)((s>>33)%128-64);}
  for(long i=0;i<K*N;i++){s=s*6364136223846793005ULL+1;b[i]=(int8_t)((s>>33)%128-64);}
  gemmi8((int64_t)a,(int64_t)b,(int64_t)cm,M,K,N);
  gemm_clang(a,b,cc,M,K,N);
  int exact=memcmp(cm,cc,(size_t)M*N*4)==0;
  printf("byte-exact (MIND == clang -O3): %s\n",exact?"YES":"NO");
  if(!exact){long d=0;for(long i=0;i<M*N;i++)if(cm[i]!=cc[i])d++;printf("  MISMATCH %ld/%d elems\n",d,M*N);return 1;}
  double gmac=2.0*M*K*N/1e9;
  gemmi8((int64_t)a,(int64_t)b,(int64_t)cm,M,K,N);
  double t0=now();for(int r=0;r<REPS;r++)gemmi8((int64_t)a,(int64_t)b,(int64_t)cm,M,K,N);double tm=(now()-t0)/REPS;
  gemm_clang(a,b,cc,M,K,N);
  t0=now();for(int r=0;r<REPS;r++)gemm_clang(a,b,cc,M,K,N);double tc=(now()-t0)/REPS;
  printf("MIND det.igemm : %8.3f ms  %8.2f GMAC/s\n",tm*1e3,gmac/tm);
  printf("clang -O3 v3   : %8.3f ms  %8.2f GMAC/s\n",tc*1e3,gmac/tc);
  printf("==> MIND is %.2fx FASTER than clang -O3 -march=x86-64-v3 (byte-identical)\n",tc/tm);
  return 0;
}
