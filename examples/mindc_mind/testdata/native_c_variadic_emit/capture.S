# Copyright 2026 STARGA Inc. Independent SysV observation fixture.
.text
.globl ve_capture_entry
ve_capture_entry:
mov ve_capture(%rip), %r10
mov %rdi, 64(%r10)
mov %rsi, 72(%r10)
mov %rdx, 80(%r10)
mov %rcx, 88(%r10)
mov %r8, 96(%r10)
mov %r9, 104(%r10)
movq %xmm0, 112(%r10)
movq %xmm1, 120(%r10)
movq %xmm2, 128(%r10)
movq %xmm3, 136(%r10)
movq %xmm4, 144(%r10)
movq %xmm5, 152(%r10)
movq %xmm6, 160(%r10)
movq %xmm7, 168(%r10)
movzbl %al,%r11d
mov %r11,16(%r10)
mov %rsp,%r11
and $15,%r11
mov %r11,24(%r10)
lea 8(%rsp),%rsi
lea 176(%r10),%rdi
mov ve_reserve_words(%rip),%rcx
cld
rep movsq
mov 64(%r10),%rdi
mov 72(%r10),%rsi
mov 80(%r10),%rdx
mov 88(%r10),%rcx
mov 96(%r10),%r8
mov 104(%r10),%r9
movq 112(%r10),%xmm0
movq 120(%r10),%xmm1
movq 128(%r10),%xmm2
movq 136(%r10),%xmm3
movq 144(%r10),%xmm4
movq 152(%r10),%xmm5
movq 160(%r10),%xmm6
movq 168(%r10),%xmm7
mov 16(%r10),%eax
jmp *ve_target(%rip)
.globl ve_case_C25
ve_case_C25:
mov ve_capture(%rip),%r10
mov %rdi,2224(%r10)
movabs $0x6767676767676767,%rdi
movabs $0x6767676767676767,%rsi
movabs $0x6767676767676767,%rdx
movabs $0x6767676767676767,%rcx
movabs $0x6767676767676767,%r8
movabs $0x6767676767676767,%r9
movabs $0x6767676767676767,%r10
movabs $0x6767676767676767,%r11
pxor %xmm0,%xmm0
pxor %xmm1,%xmm1
pxor %xmm2,%xmm2
pxor %xmm3,%xmm3
pxor %xmm4,%xmm4
pxor %xmm5,%xmm5
pxor %xmm6,%xmm6
pxor %xmm7,%xmm7
pxor %xmm8,%xmm8
pxor %xmm9,%xmm9
pxor %xmm10,%xmm10
pxor %xmm11,%xmm11
pxor %xmm12,%xmm12
pxor %xmm13,%xmm13
pxor %xmm14,%xmm14
pxor %xmm15,%xmm15
movabs $8526495043095935616,%rax
ret
.globl ve_case_C26
ve_case_C26:
mov ve_capture(%rip),%r10
mov %rdi,2224(%r10)
movabs $0x6767676767676767,%rdi
movabs $0x6767676767676767,%rsi
movabs $0x6767676767676767,%rdx
movabs $0x6767676767676767,%rcx
movabs $0x6767676767676767,%r8
movabs $0x6767676767676767,%r9
movabs $0x6767676767676767,%r10
movabs $0x6767676767676767,%r11
pxor %xmm0,%xmm0
pxor %xmm1,%xmm1
pxor %xmm2,%xmm2
pxor %xmm3,%xmm3
pxor %xmm4,%xmm4
pxor %xmm5,%xmm5
pxor %xmm6,%xmm6
pxor %xmm7,%xmm7
pxor %xmm8,%xmm8
pxor %xmm9,%xmm9
pxor %xmm10,%xmm10
pxor %xmm11,%xmm11
pxor %xmm12,%xmm12
pxor %xmm13,%xmm13
pxor %xmm14,%xmm14
pxor %xmm15,%xmm15
movabs $8526495043095920641,%rax
ret
.globl ve_case_C27
ve_case_C27:
mov ve_capture(%rip),%r10
mov %rdi,2224(%r10)
movabs $0x6767676767676767,%rdi
movabs $0x6767676767676767,%rsi
movabs $0x6767676767676767,%rdx
movabs $0x6767676767676767,%rcx
movabs $0x6767676767676767,%r8
movabs $0x6767676767676767,%r9
movabs $0x6767676767676767,%r10
movabs $0x6767676767676767,%r11
pxor %xmm0,%xmm0
pxor %xmm1,%xmm1
pxor %xmm2,%xmm2
pxor %xmm3,%xmm3
pxor %xmm4,%xmm4
pxor %xmm5,%xmm5
pxor %xmm6,%xmm6
pxor %xmm7,%xmm7
pxor %xmm8,%xmm8
pxor %xmm9,%xmm9
pxor %xmm10,%xmm10
pxor %xmm11,%xmm11
pxor %xmm12,%xmm12
pxor %xmm13,%xmm13
pxor %xmm14,%xmm14
pxor %xmm15,%xmm15
movabs $8526495040967540737,%rax
ret
.globl ve_case_C28
ve_case_C28:
mov ve_capture(%rip),%r10
mov %rdi,2224(%r10)
movabs $0x6767676767676767,%rdi
movabs $0x6767676767676767,%rsi
movabs $0x6767676767676767,%rdx
movabs $0x6767676767676767,%rcx
movabs $0x6767676767676767,%r8
movabs $0x6767676767676767,%r9
movabs $0x6767676767676767,%r10
movabs $0x6767676767676767,%r11
pxor %xmm0,%xmm0
pxor %xmm1,%xmm1
pxor %xmm2,%xmm2
pxor %xmm3,%xmm3
pxor %xmm4,%xmm4
pxor %xmm5,%xmm5
pxor %xmm6,%xmm6
pxor %xmm7,%xmm7
pxor %xmm8,%xmm8
pxor %xmm9,%xmm9
pxor %xmm10,%xmm10
pxor %xmm11,%xmm11
pxor %xmm12,%xmm12
pxor %xmm13,%xmm13
pxor %xmm14,%xmm14
pxor %xmm15,%xmm15
movabs $1234605616436508552,%rax
ret
.globl ve_case_C29
ve_case_C29:
mov ve_capture(%rip),%r10
mov %rdi,2224(%r10)
movabs $0x6767676767676767,%rdi
movabs $0x6767676767676767,%rsi
movabs $0x6767676767676767,%rdx
movabs $0x6767676767676767,%rcx
movabs $0x6767676767676767,%r8
movabs $0x6767676767676767,%r9
movabs $0x6767676767676767,%r10
movabs $0x6767676767676767,%r11
pxor %xmm0,%xmm0
pxor %xmm1,%xmm1
pxor %xmm2,%xmm2
pxor %xmm3,%xmm3
pxor %xmm4,%xmm4
pxor %xmm5,%xmm5
pxor %xmm6,%xmm6
pxor %xmm7,%xmm7
pxor %xmm8,%xmm8
pxor %xmm9,%xmm9
pxor %xmm10,%xmm10
pxor %xmm11,%xmm11
pxor %xmm12,%xmm12
pxor %xmm13,%xmm13
pxor %xmm14,%xmm14
pxor %xmm15,%xmm15
movabs $13835621005235585024,%rax
movq %rax,%xmm0
movabs $0x6767676767676767,%rax
ret
.section .note.GNU-stack,"",@progbits
