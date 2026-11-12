# Copyright 2026 STARGA Inc.
# Independent x86-64 SysV specification for the two fixed libc thunks.
.intel_syntax noprefix
.text
.global io_read_entry, io_write_entry
.global io_read_positioned, io_read_sequential
.global io_write_positioned, io_write_sequential
io_read_entry:
    test rsi,rsi
    je read_zero
    test rdx,rdx
    jle read_zero
    mov edi,edi
    test rcx,rcx
    js read_sequential
    sub rsp,8
io_read_positioned:
    .byte 0xff,0x15
    .long 0
    add rsp,8
    ret
read_sequential:
    sub rsp,8
io_read_sequential:
    .byte 0xff,0x15
    .long 0
    add rsp,8
    ret
read_zero:
    xor eax,eax
    ret
io_write_entry:
    test rsi,rsi
    je write_zero
    test rdx,rdx
    jle write_zero
    mov edi,edi
    test rcx,rcx
    js write_sequential
    sub rsp,8
io_write_positioned:
    .byte 0xff,0x15
    .long 0
    add rsp,8
    ret
write_sequential:
    sub rsp,8
io_write_sequential:
    .byte 0xff,0x15
    .long 0
    add rsp,8
    ret
write_zero:
    xor eax,eax
    ret
.section .note.GNU-stack,"",@progbits
