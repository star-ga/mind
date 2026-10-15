.intel_syntax noprefix
.text
# Dynamic-layout allocator reference; separate from the static runtime.
# The checked-in MIND emitter contains equivalent bytes with explicit constants.
# Placeholders (patched per image):
#   0x1111111111111101 A    arena base
#   0x1111111111111102 A8   A + 8 (header address of cursor offset 0)
#   0x1111111111111103 ST   state page (last 4096 bytes of the arena)
#   0x1111111111111104 OB   checked overflow base (below dedicated stack)
#   0x1111111111111105 FIRST first payload A + 72
#   0x1111111111111106 OB8  OB + 8
#   0x1111111111111107 OE   exclusive overflow end
#   0x1111111111111108 HEADER first block header A + 64
# State page words: [ST+8k] bin k head (k<64); [ST+512] overflow mapped end;
# [ST+520] non-empty-bin bitmap; [ST+528] initialized version marker;
# [ST+536/544/552] alloc/free/realloc entry;
# [ST+560] P-region high water; [ST+568] O-region high water;
# [ST+1024+8k] bin k size bound: no block in bin k is larger (insert raises
# it, a fruitless scan of the bin lowers it to the largest size it saw).
# Block: header word h = size | 1 (free) | 2 (previous block free); data at h+8.
# Free block (size >= 24): [d] next, [d+8] prev, [d+size-8] size (footer).
# Callable dynamic initializer. No process-entry/stack-switch semantics.
# Layout constants:01 A,02 A+8,03 ST,04 OB,05 first payload A+72,
# 06 OB+8,07 OE exclusive,08 first header A+64.
# State528 is the immutable-version marker; argc/argv prefix is untouched.
prefix:
    push rcx
    push rdx
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    movabs r9, 0x1111111111111103
    movabs r10, 0x31545241444e494d
    mov rax, [r9 + 528]
    cmp rax, r10
    je init_repeat
    test rax, rax
    jnz init_bad
    xor ecx, ecx
init_scan:
    cmp qword ptr [r9 + rcx*8], 0
    jne init_bad
    inc ecx
    cmp ecx, 512
    jb init_scan
    movabs r10, 0x1111111111111101
    cmp qword ptr [r10], 0
    jne init_bad
    mov qword ptr [r10], 56
    lea rax, [rip + rt_alloc]
    mov [r9 + 536], rax
    lea rax, [rip + rt_free]
    mov [r9 + 544], rax
    lea rax, [rip + rt_realloc]
    mov [r9 + 552], rax
    movabs rax, 0x1111111111111108
    mov [r9 + 560], rax
    movabs rax, 0x1111111111111104
    mov [r9 + 512], rax
    mov [r9 + 568], rax
    movabs rax, 0x31545241444e494d
    mov [r9 + 528], rax
    xor eax, eax
    jmp pub_out
init_repeat:
    lea rax, [rip + rt_alloc]
    cmp [r9 + 536], rax
    jne init_bad
    lea rax, [rip + rt_free]
    cmp [r9 + 544], rax
    jne init_bad
    lea rax, [rip + rt_realloc]
    cmp [r9 + 552], rax
    jne init_bad
    call mapped_ok
    test eax, eax
    jz init_bad
    mov eax, 1
    jmp pub_out
init_bad:
    mov eax, 2
    jmp pub_out

# ---- public entries: everything but rax (result) is preserved ----
rt_alloc:                       # rsi = n
    push rcx
    push rdx
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    movabs r9, 0x1111111111111103
    call check_map
    xor eax, eax
    test rsi, rsi
    jle pub_out
    call round_n
    call do_alloc
    jmp pub_out

rt_free:                        # rdi = p
    push rcx
    push rdx
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    movabs r9, 0x1111111111111103
    call check_map
    call valid
    test rax, rax
    jz free_out
    call do_free
free_out:
    xor eax, eax
    jmp pub_out

rt_realloc:                     # rdi = p, rsi = n
    push rcx
    push rdx
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    movabs r9, 0x1111111111111103
    call check_map
    test rdi, rdi
    jnz ra_have_p
    xor eax, eax
    test rsi, rsi
    js pub_out                  # realloc(NULL, n<0) = NULL
    jz ra_min                   # realloc(NULL, 0) = malloc(0): a minimum block
    call round_n
    call do_alloc
    jmp pub_out
ra_min:
    mov esi, 24
    call do_alloc
    jmp pub_out
ra_have_p:
    test rsi, rsi
    jg ra_resize
    jl ra_ret0                  # n<0: fails, p stays live
    call valid                  # n==0: free(p), NULL
    test rax, rax
    jz ra_ret0
    call do_free
ra_ret0:
    xor eax, eax
    jmp pub_out
ra_resize:
    call valid                  # rax = p (rdx = s, r8 = lim) or 0
    test rax, rax
    jz ra_foreign
    push rax                    # reject foreign ownership before size limits
    call round_n                # rsi = r; clobbers rax, preserves rdx/r8
    pop rax
    cmp rsi, rdx
    ja ra_grow
    mov rcx, rdx                # shrink / same size
    sub rcx, rsi
    cmp rcx, 32
    jb pub_out
    push rax
    mov r10, [rax - 8]
    and r10, 2
    or r10, rsi
    mov [rax - 8], r10          # header = r | P
    lea rdi, [rax + rsi]        # remainder header
    lea rdx, [rcx - 8]
    mov [rdi], rdx              # in-use remainder of size leftover-8
    lea rax, [rdi + 8]
    call do_free                # merge forward / return to the bump pointer
    pop rax
    jmp pub_out
ra_grow:
    lea rcx, [rax + rdx]        # next header
    movabs r10, 0x1111111111111101
    mov r11, [r10]
    movabs r10, 0x1111111111111102
    add r11, r10                # TOP
    cmp rcx, r11
    jne ra_nbr
    mov rcx, rax                # checked new TOP = d + r
    add rcx, rsi
    jc oom
    lea rdi, [rax - 8]
    cmp rdi, r9
    jae ra_top_o
    cmp rcx, r9
    ja ra_move                  # does not fit the primary region
    jmp ra_top_commit
ra_top_o:
    movabs r10, 0x1111111111111107
    cmp rcx, r10
    ja oom
    mov rdi, [r9 + 512]
    cmp rcx, rdi
    jbe ra_top_commit
    push rdx
    mov rdx, rcx
    call map_to
    mov rcx, rdx
    pop rdx
ra_top_commit:
    movabs r10, 0x1111111111111102
    mov r11, rcx
    sub r11, r10
    movabs r10, 0x1111111111111101
    mov [r10], r11              # cursor
    mov r10, [rax - 8]
    and r10, 2
    or r10, rsi
    mov [rax - 8], r10          # header = r | P
    lea rdi, [rax + rdx]        # zero [d+s, d+r)
    call zero_mark
    jmp pub_out
ra_nbr:
    cmp rcx, r8
    jae ra_move
    mov r10, [rcx]
    test r10b, 1
    jz ra_move
    and r10, -8
    lea r11, [rdx + r10 + 8]    # combined size
    cmp r11, rsi
    jb ra_move
    push rax
    push rdx
    push r11
    lea rax, [rcx + 8]
    call unlink
    pop r11
    pop rdx
    pop rax
    mov rcx, r11
    sub rcx, rsi
    cmp rcx, 32
    jb ra_absorb_all
    mov r10, [rax - 8]
    and r10, 2
    or r10, rsi
    mov [rax - 8], r10          # header = r | P
    lea rdi, [rax + rsi]        # remainder header (free; its successor keeps P)
    sub rcx, 8
    lea r10, [rcx + 1]
    mov [rdi], r10
    mov [rdi + rcx], rcx
    push rax
    push rdx
    lea rax, [rdi + 8]
    mov rdx, rcx
    call insert
    pop rdx
    pop rax
    lea rcx, [rax + rsi]
    jmp ra_nzero
ra_absorb_all:
    mov r10, [rax - 8]
    and r10, 2
    or r10, r11
    mov [rax - 8], r10          # header = combined | P
    lea rcx, [rax + r11]
    cmp rcx, r8
    jae ra_az
    and qword ptr [rcx], -3     # successor: previous block now in use
ra_az:
    lea rcx, [rax + r11]
ra_nzero:
    lea rdi, [rax + rdx]
    call zero_mark
    jmp pub_out
ra_move:
    push rax
    push rdx
    call do_alloc               # rax = q (zeroed)
    pop rcx                     # s
    pop rsi                     # p
    push rax
    mov rdi, rax
    push rsi
    push rcx
    shr rcx, 3
    rep movsq
    pop rdx
    pop rax
    call do_free
    pop rax
    jmp pub_out
ra_foreign:                     # foreign memory ownership stays with caller
    xor eax, eax
    jmp pub_out
pub_out:
    pop r11
    pop r10
    pop r9
    pop r8
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    ret

# ---- internal routines (r9 = ST) ----
round_n:                        # rsi = max(round8(n), 24); n > 2^46 -> oom
    movabs rax, 0x400000000000
    cmp rsi, rax
    ja oom
    add rsi, 7
    and rsi, -8
    cmp rsi, 24
    jae rn_ok
    mov esi, 24
rn_ok:
    ret

valid:                          # rdi = p -> rax = p, rdx = size, r8 = lim | rax = 0
    xor eax, eax
    test rdi, rdi
    jz v_no
    test dil, 7
    jnz v_no
    movabs rcx, 0x1111111111111105
    cmp rdi, rcx
    jb v_no
    cmp rdi, r9
    jb v_rng
    movabs rcx, 0x1111111111111106
    cmp rdi, rcx
    jb v_no
    cmp rdi, [r9 + 512]
    jae v_no
v_rng:
    mov rdx, [rdi - 8]
    test dl, 1
    jnz v_no
    and rdx, -8
    mov rax, rdi
    call lim
    lea rcx, [rax + rdx]
    cmp rcx, r8
    jbe v_ok
    xor eax, eax
v_ok:
v_no:
    ret

lim:                            # rax = d -> r8 = end of the block's live region; clobbers r10
    movabs r10, 0x1111111111111101
    mov r8, [r10]
    movabs r10, 0x1111111111111102
    add r8, r10                 # TOP
    lea r10, [rax - 8]
    cmp r10, r9
    jae lim_o
    cmp r8, r9
    jbe lim_r
    mov r8, r9                  # primary block, cursor in overflow: region end
lim_r:
    ret
lim_o:
    movabs r10, 0x1111111111111104
    cmp r8, r10
    jae lim_r
    mov r8, r10                 # overflow block, cursor in primary: empty
    ret

unlink:                         # rax = d; clobbers rcx r10 r11
    mov r10, [rax]
    mov r11, [rax + 8]
    test r11, r11
    jz ul_head
    mov [r11], r10
    jmp ul_nx
ul_head:
    mov rcx, [rax - 8]
    and rcx, -8
    bsr rcx, rcx
    mov [r9 + rcx*8], r10
    test r10, r10
    jnz ul_nx
    btr qword ptr [r9 + 520], rcx
ul_nx:
    test r10, r10
    jz ul_out
    mov [r10 + 8], r11
ul_out:
    ret

insert:                         # rax = d, rdx = size; clobbers rcx r10
    bsr rcx, rdx
    mov r10, [r9 + rcx*8]
    mov [rax], r10
    mov qword ptr [rax + 8], 0
    test r10, r10
    jz in_h
    mov [r10 + 8], rax
in_h:
    mov [r9 + rcx*8], rax
    bts qword ptr [r9 + 520], rcx
    cmp rdx, [r9 + rcx*8 + 1024]
    jbe in_out
    mov [r9 + rcx*8 + 1024], rdx
in_out:
    ret

zero_mark:                      # zero [rdi, rcx) below the region high water,
    lea r10, [r9 + 560]         # then raise it to rcx; keeps rax rdx rsi r8
    cmp rdi, r9
    jb zm_r
    lea r10, [r9 + 568]
zm_r:
    mov r11, [r10]
    cmp rcx, r11
    jbe zm_c
    mov [r10], rcx
    mov rcx, r11
zm_c:
    cmp rdi, rcx
    jae zm_out
    sub rcx, rdi
    shr rcx, 3
    push rax
    xor eax, eax
    rep stosq
    pop rax
zm_out:
    ret

# Mutable mapped-end is never authority for the immutable capacity.
mapped_ok:                      # eax boolean, clobbers r10/r11 only
    mov r10, [r9 + 512]
    movabs r11, 0x1111111111111104
    cmp r10, r11
    jb mapped_bad
    movabs r11, 0x1111111111111107
    cmp r10, r11
    ja mapped_bad
    test r10, 0xfffff
    jnz mapped_bad
    mov eax, 1
    ret
mapped_bad:
    xor eax, eax
    ret
check_map:                      # public callers already saved volatile registers
    movabs r10, 0x31545241444e494d
    cmp [r9 + 528], r10
    jne oom
    call mapped_ok
    test eax, eax
    jz oom
    ret

map_to:                         # rdi=mapped end; rdx=requested exclusive end
    movabs r10, 0x1111111111111104
    cmp rdi, r10
    jb oom
    cmp rdx, rdi
    jbe oom
    movabs r10, 0x1111111111111107
    cmp rdi, r10
    ja oom
    cmp rdx, r10
    ja oom
    mov r8, rdx
    add r8, 0xfffff
    jc oom
    and r8, -0x100000
    cmp r8, r10                 # EXCLUSIVE_END_GUARD
    ja oom
    push rax
    push rdx
    push rsi
    push r9
    push r8
    mov rsi, r8
    sub rsi, rdi
    mov edx, 3
    mov r10d, 0x100022
    mov r8, -1
    xor r9d, r9d
    mov eax, 9
    syscall
    pop r8
    pop r9
    pop rsi
    pop rdx
    cmp rax, rdi
    pop rax
    jne oom
    mov [r9 + 512], r8
    ret

do_alloc:                       # rsi = r (rounded) -> rax = zeroed block; keeps rsi
    bsr rcx, rsi
    cmp rsi, [r9 + rcx*8 + 1024]
    ja da_higher                # nothing in bin rcx is that large: skip the scan
    mov rax, [r9 + rcx*8]
    xor r8d, r8d                # largest size the scan saw
da_scan:
    test rax, rax
    jz da_miss
    mov rdx, [rax - 8]
    and rdx, -8
    cmp rdx, rsi
    jae da_take
    cmp rdx, r8
    jbe da_next
    mov r8, rdx
da_next:
    mov rax, [rax]
    jmp da_scan
da_miss:
    mov [r9 + rcx*8 + 1024], r8 # whole bin scanned: its exact largest size
da_higher:
    mov rdx, [r9 + 520]
    inc ecx
    shr rdx, cl
    test rdx, rdx
    jz da_bump
    bsf rdx, rdx
    add rcx, rdx
    mov rax, [r9 + rcx*8]
    mov rdx, [rax - 8]
    and rdx, -8
da_take:
    call unlink
    mov r8, rdx
    sub r8, rsi
    cmp r8, 32
    jb da_all
    mov r10, [rax - 8]
    and r10, 2
    or r10, rsi
    mov [rax - 8], r10
    lea rdi, [rax + rsi]
    sub r8, 8
    lea r10, [r8 + 1]
    mov [rdi], r10
    mov [rdi + r8], r8
    push rax
    lea rax, [rdi + 8]
    mov rdx, r8
    call insert
    pop rax
    mov rdx, rsi
    jmp da_zero
da_all:
    and qword ptr [rax - 8], -2
    lea rdi, [rax + rdx]
    call lim
    cmp rdi, r8
    jae da_zero
    and qword ptr [rdi], -3
da_zero:
    mov rdi, rax
    lea rcx, [rax + rdx]
    jmp zero_mark
da_bump:
    movabs r10, 0x1111111111111101
    mov rcx, [r10]
    movabs r10, 0x1111111111111102
    lea rax, [r10 + rcx]        # TOP
    mov rdx, rax
    add rdx, rsi
    jc oom
    add rdx, 8
    jc oom
    cmp rax, r9
    ja da_in_o
    cmp rdx, r9
    jbe da_commit
    mov rdi, r9                 # leaving the primary region: tile its tail
    sub rdi, rax
    jz da_to_o
    mov [r9 + 560], r9          # the tail's metadata dirties it: high water = ST
    sub rdi, 8
    cmp rdi, 24
    jb da_dummy
    lea r10, [rdi + 1]
    mov [rax], r10
    mov [rax + rdi], rdi
    add rax, 8
    mov rdx, rdi
    call insert
    jmp da_to_o
da_dummy:
    mov [rax], rdi
da_to_o:
    movabs rax, 0x1111111111111104
    cmp qword ptr [r9 + 512], 0
    jne da_o1
    mov [r9 + 512], rax
da_o1:
    mov rdx, rax
    add rdx, rsi
    jc oom
    add rdx, 8
    jc oom
da_in_o:
    movabs r10, 0x1111111111111107
    cmp rdx, r10
    ja oom
    mov rdi, [r9 + 512]
    cmp rdx, rdi
    jbe da_commit
    call map_to
da_commit:
    movabs r10, 0x1111111111111102
    mov rcx, rdx
    sub rcx, r10
    movabs r10, 0x1111111111111101
    mov [r10], rcx
    mov [rax], rsi
    add rax, 8
    mov rdx, rsi
    jmp da_zero

do_free:                        # rax = d, rdx = size of a validated live block
    lea rdi, [rax - 8]          # h
    call lim
    mov rsi, [rdi]
    and rsi, 2
    lea rax, [rdi + rdx + 8]    # next header
    cmp rax, r8
    jae df_bwd
    mov rcx, [rax]
    test cl, 1
    jz df_bwd
    and rcx, -8
    lea rdx, [rdx + rcx + 8]
    add rax, 8
    call unlink
df_bwd:
    test rsi, rsi
    jz df_place
    mov rcx, [rdi - 8]          # previous block's footer
    lea rax, [rdi - 8]
    sub rax, rcx
    mov rsi, [rax]
    and rsi, 2
    lea rdx, [rdx + rcx + 8]
    mov rdi, rax
    add rax, 8
    call unlink
    jmp df_bwd
df_place:
    lea rax, [rdi + rdx + 8]    # next header
    movabs r10, 0x1111111111111101
    mov rcx, [r10]
    movabs r11, 0x1111111111111102
    add rcx, r11
    cmp rax, rcx
    je df_retreat
    lea rcx, [rdx + 1]
    mov [rdi], rcx
    mov [rdi + rdx], rdx
    cmp rax, r8
    jae df_ins
    or qword ptr [rax], 2
df_ins:
    lea rax, [rdi + 8]
    jmp insert
df_retreat:
    mov rcx, rdi
    sub rcx, r11
    mov [r10], rcx              # cursor = h
    lea r10, [r9 + 560]
    cmp rdi, r9
    jb df_r1
    lea r10, [r9 + 568]
df_r1:
    mov r11, [r10]
    lea rax, [rdi + 4095]
    and rax, -4096
    mov rcx, r11
    sub rcx, rax
    cmp rcx, 0x100000
    jl df_out
    push rdi
    push rsi
    push rdx
    push r10
    push rax
    mov rdi, rax
    mov rsi, rcx
    mov edx, 4
    mov eax, 28
    syscall
    mov rcx, rax
    pop rax
    pop r10
    pop rdx
    pop rsi
    pop rdi
    test rcx, rcx
    jnz df_out
    mov [r10], rax              # pages above the new top read zero again
df_out:
    ret

oom:
    movabs rax, 11426315137909
    push rax
    movabs rax, 7018992394987204197
    push rax
    movabs rax, 8241904413215126625
    push rax
    movabs rax, 7935406806597268069
    push rax
    mov eax, 1
    mov edi, 2
    mov rsi, rsp
    mov edx, 30
    syscall
    mov eax, 231
    mov edi, 70
    syscall
    ud2
rt_end:
