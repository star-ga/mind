/* Copyright 2026 STARGA Inc. External diagnostic only; not product runtime. */
#define _GNU_SOURCE
#include <link.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int modify(struct dl_phdr_info *info,size_t size,void *unused) {
 (void)size;(void)unused;
 if(info->dlpi_addr!=0)return 0;
 const char *mode=getenv("NS_FIXTURE_MODE");
 if(!mode)return 0;
 if(!strcmp(mode,"dirty_allocator")){*(volatile uint64_t*)0x1000000=1;return 1;}
 for(int i=0;i<info->dlpi_phnum;i++) {
  const ElfW(Phdr) *p=&info->dlpi_phdr[i];
  if(p->p_type==PT_LOAD && p->p_flags==(PF_R|PF_W) && p->p_filesz==0 && p->p_memsz==4096 && p->p_vaddr<0x800000){
   if(!strcmp(mode,"dirty_context"))*(volatile uint64_t*)p->p_vaddr=1;
   return 1;
  }
 }
 _exit(78);
}
__attribute__((constructor)) static void setup(void) {
 const char *mode=getenv("NS_FIXTURE_MODE");
 if(mode && !strcmp(mode,"env_constructor")) {
  if(setenv("NS_NEW_ADAPTER_ENTRY","1",1) || setenv("NS_STARTUP_ENV","changed",1))_exit(78);
 }
 dl_iterate_phdr(modify,0);
}
__attribute__((destructor)) static void finish(void) { if(write(1,"D",1)!=1)_exit(78); }
