# Copyright 2026 STARGA Inc.
# Independent development reference for the MIND startup byte emitter.
# Linux x86-64 LP64 SysV. No libc implementation source is included.
.intel_syntax noprefix
.text
# Per-image constants:01 context VA,02 arena VA,03 exclusive stack top.
# Typed relative calls:0x22222221 allocator init,0x22222222 source main.
# Context words: marker,phase,OS RSP,rtld_fini,argc,argv,original envp,
# saved C RSP,current libc envp,source result. All other words untouched.
# Phase1 startup complete;2 source main active;3 returned;4 adapter refused.
.globl startup_entry, main_adapter, libc_fixup, init_disp, main_disp, startup_end
startup_entry:
    xor ebp, ebp
    movabs r11, 0x1111111111111101
    cmp qword ptr [r11], 0
    jne startup_fail
    cmp qword ptr [r11 + 8], 0
    jne startup_fail
    mov [r11 + 16], rsp
    mov [r11 + 24], rdx
    mov rsi, [rsp]
    cmp rsi, 0x7fffffff
    ja startup_fail
    lea rdx, [rsp + 8]
    mov [r11 + 32], rsi
    mov [r11 + 40], rdx
    lea rax, [rdx + rsi*8 + 8]
    mov [r11 + 48], rax
    movabs rax, 0x3154524154534d
    mov [r11], rax
    mov qword ptr [r11 + 8], 1
    mov r9, [r11 + 24]
    lea rdi, [rip + .Lmain_adapter]
    xor ecx, ecx
    xor r8d, r8d
    and rsp, -16
    xor eax, eax
    push rax
    push rsp
    .byte 0xff,0x15
libc_fixup:
    .long 0
    ud2                         # __libc_start_main is non-returning
startup_fail:
    mov edi, 70
    mov eax, 231
    syscall
    ud2

main_adapter:
.Lmain_adapter:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8                  # align before ordinary callable init
    movabs r12, 0x1111111111111101
    movabs rax, 0x3154524154534d
    cmp [r12], rax
    jne adapter_fail
    cmp qword ptr [r12 + 8], 1
    jne adapter_fail
    movsxd rax, edi
    cmp [r12 + 32], rax
    jne adapter_fail
    cmp [r12 + 40], rsi
    jne adapter_fail
    mov [r12 + 64], rdx         # retain libc's current environ separately
    .byte 0xe8
init_disp:
    .long 0x22222221
    test rax, rax
    jne adapter_fail            # fresh initialization only; no heap reset
    movabs rax, 0x1111111111111102
    mov rcx, [r12 + 32]
    mov [rax + 8], rcx
    mov rcx, [r12 + 40]
    mov [rax + 16], rcx
    mov rcx, [r12 + 64]
    mov [rax + 24], rcx
    mov [r12 + 56], rsp
    mov qword ptr [r12 + 8], 2
    movabs rsp, 0x1111111111111103
    mov rdi, [r12 + 32]
    mov rsi, [r12 + 40]
    .byte 0xe8
main_disp:
    .long 0x22222222
    # Internal source functions need not preserve SysV callee-saved registers.
    # Recover the C stack from a fresh context address, not a live GP register.
    movabs rcx, 0x1111111111111101
    mov [rcx + 72], rax
    mov rsp, [rcx + 56]
    mov qword ptr [rcx + 8], 3
    mov eax, eax                # documented i64 -> C int low32 result
    jmp adapter_out
adapter_fail:
    mov qword ptr [r12 + 8], 4
    mov eax, 70
adapter_out:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
startup_end:
.section .note.GNU-stack,"",@progbits
