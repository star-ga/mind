# Copyright 2026 STARGA Inc.
# Test-only ordinary source-main body. Placeholder constants are typed inputs to
# the fixture composer. Test syscalls transport data; product I/O uses libc.
.intel_syntax noprefix
.text
.global fixture_main
fixture_main:
    push rbp
    mov rbp,rsp
    push r12
    push r13
    push r14
    sub rsp,72
    movabs r12,0x3333333333333301
    lea r14,[r12+128]
    mov r13,144
read_header:
    xor edi,edi
    mov rsi,r14
    mov rdx,r13
    xor eax,eax
    syscall
    test rax,rax
    jle bad_input
    add r14,rax
    sub r13,rax
    jne read_header
    .byte 0xff,0x15
    .long 0x44444405
    mov edx,[r12+168]
    mov [rax],edx
    mov rdi,[r12+136]
    lea rsi,[r12+208]
    cmp qword ptr [r12+160],1
    jne not_null
    xor esi,esi
not_null:
    cmp qword ptr [r12+160],2
    jne valid_buffer
    mov esi,1
valid_buffer:
    mov rdx,[r12+144]
    mov rcx,[r12+152]
    cmp qword ptr [r12+128],0
    jne write_case
    movabs rax,0x3333333333333302
    call rax
    jmp capture
write_case:
    cmp qword ptr [r12+128],1
    jne bad_operation
    movabs rax,0x3333333333333303
    call rax
capture:
    mov [r12+512],rax
    .byte 0xff,0x15
    .long 0x44444405
    movsxd rax,dword ptr [rax]
    mov [r12+520],rax
    mov edi,10
    xor esi,esi
    mov edx,1
    .byte 0xff,0x15
    .long 0x44444406
    mov [r12+528],rax
    lea rsi,[r12+208]
    lea rdi,[r12+536]
    mov ecx,8
    rep movsq
    lea r14,[r12+512]
    mov r13,88
write_output:
    mov edi,1
    mov rsi,r14
    mov rdx,r13
    mov eax,1
    syscall
    test rax,rax
    jle bad_output
    add r14,rax
    sub r13,rax
    jne write_output
    xor eax,eax
    add rsp,72
    pop r14
    pop r13
    pop r12
    leave
    ret
bad_input:
    mov eax,71
    add rsp,72
    pop r14
    pop r13
    pop r12
    leave
    ret
bad_operation:
    mov eax,72
    add rsp,72
    pop r14
    pop r13
    pop r12
    leave
    ret
bad_output:
    mov eax,74
    add rsp,72
    pop r14
    pop r13
    pop r12
    leave
    ret
.section .note.GNU-stack,"",@progbits
