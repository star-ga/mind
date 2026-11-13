/* Copyright 2026 STARGA Inc.
 * Independent host diagnostic of MIND-emitted machine code. Not product code,
 * compiler fallback, mathematical authority or a production libc entry path.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef int64_t (*Fn)(int64_t,int64_t);
static uint64_t A=0x1000000, ST, OB, OE;
static Fn init, alloc_f, free_f, realloc_f;
extern int probe_init(Fn);
static void fail(int line) { fprintf(stderr,"check failed at %d\n",line); exit(3); }
#define C(x) do { if (!(x)) fail(__LINE__); } while (0)
#define W(p) (*(uint64_t *)(uintptr_t)(p))
static uint64_t allocn(int64_t n) { return alloc_f(0,n); }
static void fill(uint64_t p,uint64_t n,uint8_t v) { memset((void*)p,v,n); }
static void bytes(uint64_t p,uint64_t n,uint8_t v) { for(uint64_t i=0;i<n;i++) C(*(uint8_t*)(p+i)==v); }
static void map_at(uint64_t p,uint64_t n) {
 C(mmap((void*)p,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0)==(void*)p);
}
int main(int argc,char **argv) {
 C(argc==5); const char *name=argv[2]; uint64_t ab=strtoull(argv[3],0,10),ov=strtoull(argv[4],0,10);
 ST=A+ab-4096;OB=(A+ab+4096+1048575)&~1048575ULL;OE=OB+ov; map_at(A,ab);
 FILE *f=fopen(argv[1],"rb");C(f); uint64_t m[8];C(fread(m,1,64,f)==64);C(m[0]==1 && m[5]>0 && m[5]<65536 && m[6]==0 && m[7]==0);
 size_t cap=(m[5]+4095)&~4095ULL; unsigned char *code=mmap(0,cap,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);C(code!=MAP_FAILED);
 C(fread(code,1,m[5],f)==m[5]);C(fgetc(f)==EOF);fclose(f);C(mprotect(code,cap,PROT_READ|PROT_EXEC)==0);
 for(int i=1;i<=4;i++)C(m[i]<m[5]);
 init=(Fn)(code+m[1]);alloc_f=(Fn)(code+m[2]);free_f=(Fn)(code+m[3]);realloc_f=(Fn)(code+m[4]);
 if(!strcmp(name,"dirty_cursor")){W(A)=8;C(init(0,0)==2 && W(A)==8 && W(ST+528)==0);return 0;}
 if(!strcmp(name,"dirty_marker")){W(ST+528)=77;C(init(0,0)==2 && W(ST+528)==77 && W(A)==0);return 0;}
 if(!strcmp(name,"dirty_state")){W(ST+1024)=77;C(init(0,0)==2 && W(ST+1024)==77 && W(A)==0);return 0;}
 for(int i=1;i<8;i++)W(A+i*8)=0x1020304050607000ULL+i;
 if(!strcmp(name,"init_registers"))C(probe_init(init)==0);else C(init(0,0)==0);
 C(W(A)==56 && W(ST+512)==OB && W(ST+536)==(uint64_t)alloc_f && W(ST+544)==(uint64_t)free_f && W(ST+552)==(uint64_t)realloc_f);
 for(int i=1;i<8;i++)C(W(A+i*8)==0x1020304050607000ULL+i);
 if(!strcmp(name,"init")||!strcmp(name,"init_registers"))return 0;
 if(!strncmp(name,"repeat_",7)) {
  uint64_t p=allocn(64);fill(p,64,0xa5);uint64_t cursor=W(A);
  if(!strcmp(name,"repeat_live"))C(init(0,0)==1);
  else {if(!strcmp(name,"repeat_alloc_slot"))W(ST+536)++;
   else if(!strcmp(name,"repeat_free_slot"))W(ST+544)++;
   else if(!strcmp(name,"repeat_realloc_slot"))W(ST+552)++;
   else if(!strcmp(name,"repeat_mapped_low"))W(ST+512)=OB-1048576;
   else if(!strcmp(name,"repeat_mapped_high"))W(ST+512)=OE+1048576;
   else if(!strcmp(name,"repeat_mapped_unaligned"))W(ST+512)=OB+1;else C(0);
   uint64_t before[512];memcpy(before,(void*)ST,4096);C(init(0,0)==2);C(memcmp(before,(void*)ST,4096)==0);
  } C(W(A)==cursor);bytes(p,64,0xa5);return 0;
 }
 if(!strcmp(name,"zero_negative")){C(allocn(0)==0 && allocn(-1)==0 && W(A)==56);return 0;}
 uint64_t primary=ST-A-72;
 if(!strcmp(name,"primary_exact")){uint64_t p=allocn(primary);C(p==A+72 && A+8+W(A)==ST && W(ST+512)==OB);bytes(p,primary,0);return 0;}
 if(!strcmp(name,"primary_cross")||!strcmp(name,"no_overflow")){uint64_t p=allocn(primary+1);C(p==OB+8);bytes(p,primary+1,0);return 0;}
 if(!strncmp(name,"overflow_",9)&&strcmp(name,"overflow_regrow")){
  uint64_t n=ov-8;
  if(!strcmp(name,"overflow_one_over")||!strcmp(name,"overflow_two_over"))n++;
  uint64_t p=allocn(n);C(p==OB+8 && W(ST+512)==OE);bytes(p,n,0);return 0;
 }
 if(!strcmp(name,"realloc_top_exact")||!strcmp(name,"realloc_top_over")){
  uint64_t p=allocn(primary+1);fill(p,primary+1,0x5a);uint64_t n=ov-8+(!strcmp(name,"realloc_top_over"));
  uint64_t q=realloc_f(p,n);C(q==p && W(ST+512)==OE);bytes(q,primary+1,0x5a);bytes(q+primary+1,n-primary-1,0);return 0;
 }
 if(!strcmp(name,"realloc_move_over")){uint64_t p=allocn(primary+1);C(allocn(primary+1)>p);C(realloc_f(p,ov-8)!=0);return 0;}
 if(!strcmp(name,"realloc_move")){uint64_t p=allocn(32),q=allocn(32);fill(p,32,0x51);fill(q,32,0x62);uint64_t r=realloc_f(p,256);C(r!=p);bytes(r,32,0x51);bytes(r+32,224,0);bytes(q,32,0x62);return 0;}
 if(!strcmp(name,"reuse_split")){uint64_t p=allocn(256),q=allocn(32);fill(p,256,0x51);fill(q,32,0x62);C(free_f(p,0)==0);C(allocn(128)==p);bytes(p,128,0);bytes(q,32,0x62);return 0;}
 if(!strcmp(name,"coalesce")){uint64_t p=allocn(128),q=allocn(256),r=allocn(64);fill(r,64,0x62);free_f(p,0);free_f(q,0);C(allocn(300)==p);bytes(p,300,0);bytes(r,64,0x62);return 0;}
 if(!strcmp(name,"shrink_split")){uint64_t p=allocn(256),q=allocn(32);fill(p,256,0x51);fill(q,32,0x62);C((uint64_t)realloc_f(p,64)==p);C(allocn(128)==p+72);bytes(p,64,0x51);bytes(p+72,128,0);bytes(q,32,0x62);return 0;}
 if(!strcmp(name,"realloc_null_zero")){uint64_t p=realloc_f(0,0);C(p==A+72);bytes(p,24,0);C(realloc_f(p,0)==0 && W(A)==56);return 0;}
 if(!strcmp(name,"realloc_negative")){uint64_t p=allocn(64);fill(p,64,0x62);C(realloc_f(p,-1)==0);bytes(p,64,0x62);C(realloc_f(0,-1)==0);return 0;}
 if(!strcmp(name,"foreign_intmax")||!strcmp(name,"foreign_over_limit")||!strcmp(name,"reserved_intmax")||!strcmp(name,"foreign_zero")||!strcmp(name,"foreign_negative")){
  uint64_t p=!strcmp(name,"reserved_intmax")?A+16:0x1000;
  int64_t n=INT64_MAX;
  if(!strcmp(name,"foreign_over_limit"))n=(INT64_C(1)<<46)+1;
  if(!strcmp(name,"foreign_zero"))n=0;
  if(!strcmp(name,"foreign_negative"))n=-1;
  C(realloc_f(p,n)==0 && W(A)==56);
  for(int i=1;i<8;i++)C(W(A+i*8)==0x1020304050607000ULL+i);
  return 0;
 }
 if(!strcmp(name,"owned_intmax")||!strcmp(name,"owned_over_limit")||!strcmp(name,"null_intmax")||!strcmp(name,"null_over_limit")){
  uint64_t p=!strncmp(name,"owned_",6)?allocn(64):0;
  int64_t n=strstr(name,"over_limit")?(INT64_C(1)<<46)+1:INT64_MAX;
  C(realloc_f(p,n)!=0);return 0;
 }
 if(!strcmp(name,"owned_zero")){uint64_t p=allocn(64);fill(p,64,0x62);C(realloc_f(p,0)==0 && W(A)==56);C(allocn(64)==p);bytes(p,64,0);return 0;}
 if(!strcmp(name,"owned_negative")){uint64_t p=allocn(64);fill(p,64,0x62);uint64_t cursor=W(A);C(realloc_f(p,-1)==0 && W(A)==cursor);bytes(p,64,0x62);return 0;}
 if(!strcmp(name,"null_zero_again")){uint64_t p=realloc_f(0,0);C(p==A+72);bytes(p,24,0);return 0;}
 if(!strcmp(name,"foreign")){C(free_f(0x1000,0)==0 && realloc_f(0x1000,32)==0);C(free_f(A+16,0)==0 && realloc_f(A+16,32)==0);C(W(A)==56);return 0;}
 if(!strcmp(name,"reserved_cells")){uint64_t p=allocn(100),q=allocn(primary+1);C(p>=A+72 && q==OB+8);free_f(p,0);free_f(q,0);for(int i=1;i<8;i++)C(W(A+i*8)==0x1020304050607000ULL+i);return 0;}
 if(!strcmp(name,"zero_reuse")){uint64_t p=allocn(64);fill(p,64,0x62);free_f(p,0);C(allocn(64)==p);bytes(p,64,0);return 0;}
 if(!strncmp(name,"mapped_",7)){if(!strcmp(name,"mapped_low"))W(ST+512)=OB-1048576;else if(!strcmp(name,"mapped_high"))W(ST+512)=OE+1048576;else W(ST+512)=OB+1;C(allocn(24)!=0);return 0;}
 if(!strcmp(name,"size_huge")){C(allocn(INT64_MAX)!=0);return 0;}
 if(!strcmp(name,"collision_first")){map_at(OB,1048576);fill(OB,4096,0x62);C(allocn(primary+1)!=0);bytes(OB,4096,0x62);return 0;}
 if(!strcmp(name,"collision_next")){C(allocn(1048576-8)==OB+8);map_at(OB+1048576,1048576);C(allocn(primary+1)!=0);return 0;}
 if(!strcmp(name,"overflow_regrow")){uint64_t p=allocn(ov-8);fill(p,ov-8,0x62);free_f(p,0);C(allocn(ov-8)==p);bytes(p,ov-8,0);return 0;}
 C(0);
}
