// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Experimental **MIND-native backend**: `IRModule → x86-64 → static ELF64`, with
//! **zero LLVM / MLIR / clang / assembler / external linker**.
//!
//! This is the production-shaped home of the native codegen. The path was
//! de-risked incrementally — a single-function stack machine, then
//! multi-function + System-V calls + a minimal linker, then consuming the real
//! IR, then if-expression control flow — and consolidated here. It is **opt-in**
//! (`--features native-backend`) and is **not** wired into the default compile
//! pipeline — the shipping path is still `mic@3 → MLIR → ELF`. It exists so the
//! native-backend track has a real, linted, tested module to grow inside of.
//!
//! ## Why a native backend — determinism-by-CONSTRUCTION
//!
//! The cross-substrate byte-identity that is MIND's wedge is, on the MLIR path, a
//! *post-hoc* property: mic@3 is deterministic, but the bytes are handed to
//! `clang -O3` and a pinned hash is checked. A native backend with a **fixed
//! instruction encoder and a fixed `ValueId → frame-slot` mapping** makes it
//! *structural*: the emitted image is a pure function of the IR, so it
//! **cannot** differ across hosts/toolchain versions. "Better than LLVM" here is
//! not `-O3`; it is *determinism guaranteed by the codegen's shape*. The emitted
//! ELF embeds the IR's `trace_hash` as a PT_NOTE (`readelf -n` shows the `MIND`
//! note) — the wedge's "signed evidence chain in the artifact", native-side.
//!
//! ## Scope (honest) — the scalar-i64 vertical slice
//!
//! Supports `FnDef` / `Param` / `ConstI64` / `Call` / `Return`, arithmetic
//! `BinOp{Add,Sub,Mul,Div,Mod}` (Div/Mod via `cqo`+signed `idiv`, with MIND's
//! defined edge cases `x/0=0`/`x%0=0`/`INT_MIN/-1` handled branchlessly), bitwise/shift
//! `BinOp{BitAnd,BitOr,BitXor,Shl,Shr}` (Shr = arithmetic `sar`; shift count in
//! `cl`), signed comparison `BinOp{Lt,Le,Gt,Ge,Eq,Ne}`
//! (`cmp`+`setcc`+`movzx` → 0/1), `If` (if-expressions: `cmp`/`je`/`jmp` with
//! intra-function forward-jump patching and mem-to-mem phi resolution via the
//! merge list), and `While` loops with `break`/`continue` (header re-test +
//! back-edge `jmp`; loop-carried vars alias to one slot so no phi forwarding is
//! needed; break/continue jump via a loop-frame stack), integer args (≤6 in
//! System-V registers, 7th+ stack-passed with 16-byte-aligned call), a
//! regalloc-free mem-to-mem frame model, and intra-module calls (computed
//! PC-relative displacement). The entry point is the `FnDef` named `main`; its
//! `Return` becomes an `exit` syscall.
//!
//! deferred: register allocation
//!   (currently
//!   every SSA value is a frame slot), true ELF relocations + sections + symbols
//!   (for separate compilation / libc linking), float + tensor + SIMD kernels.
//!   upgrade path: extend `emit_seq`'s match arm-by-arm; add a deterministic
//!   linear-scan allocator as a pre-pass over the slot map; emit a `.o` with a
//!   symbol table + `R_X86_64_PC32` records for the external-link milestone.

use std::collections::HashMap;

use crate::ir::{BinOp, IRModule, Instr, ValueId};

/// Why a module could not be lowered by the (intentionally small) native backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeError {
    /// No `FnDef` named `main` to use as the program entry point.
    NoEntry,
    /// An instruction or operator outside the supported scalar-i64 slice.
    Unsupported(String),
    /// A call named a function that is not defined in this module.
    UndefinedCallee(String),
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeError::NoEntry => write!(f, "native: no `main` function to use as entry"),
            NativeError::Unsupported(what) => write!(f, "native: unsupported construct: {what}"),
            NativeError::UndefinedCallee(name) => write!(f, "native: call to undefined `{name}`"),
        }
    }
}

impl std::error::Error for NativeError {}

/// ============================ Option-C i64-handle ABI ============================
/// The front-end (`src/eval/lower.rs`) lowers a struct literal to three intrinsic
/// `Instr::Call`s — `__mind_alloc(bytes)->addr`, `__mind_store_i64(addr,val)`, and
/// (for field reads) `__mind_load_i64(addr)->val` — where the struct value is the
/// i64 heap-record base address. The native backend inlines these directly rather
/// than emitting a `call` (there is no runtime / libc to link). The crux is the
/// allocator: the returned address MUST be a pure function of allocation ORDER
/// (a fixed bump cursor), NEVER malloc / ASLR / a host-varying value — otherwise
/// the cross-substrate byte-identity wedge breaks at runtime.
///
/// The arena lives in a dedicated RW PT_LOAD at a FIXED vaddr, zero-initialized as
/// BSS (`p_filesz = 0`, `p_memsz = ARENA_BYTES`). Its first 8 bytes are the bump
/// cursor (a byte OFFSET into the data region); the data region begins at
/// `ARENA_ADDR + ARENA_CURSOR_RESERVE`. BSS zeroing means the cursor starts at 0
/// with no runtime initialization, so the first allocation deterministically lands
/// at `ARENA_DATA` (offset 0) and the cursor advances by `round_up(bytes, 8)`.
/// On exhaustion the allocator TRAPs (`ud2`) — it never wraps or aliases.
///
/// Fixed vaddr above the code segment (16 MiB), page-aligned. Distinct enough from
/// `LOAD_ADDR` (4 MiB) that the two PT_LOADs never overlap for any plausible code
/// size.
const ARENA_ADDR: u64 = 0x100_0000;
/// Bytes reserved at the arena base for the bump cursor (8) — the data region
/// starts here. Kept 8-aligned so every allocation stays 8-aligned.
const ARENA_CURSOR_RESERVE: u64 = 8;
/// Total mapped arena size, including the cursor reserve. 256 MiB of zero-filled
/// BSS — bump past it traps. Costs nothing on disk (BSS `p_filesz = 0`; only
/// `p_memsz` grows). Sized for the ULTIMATE self-host: the native compiler running
/// `selftest_mic3_module_nfn` on all of main.mind (~262 KB mic@3 out) allocates a
/// large transient heap (lexer tokens, AST, struct registry, the strtab + per-fn
/// emit buffers) — the 16 MiB the scalar-struct slice used overflowed (SIGSEGV on a
/// store past the BSS edge). Bump this (and document it) when programs need larger
/// heaps; the no-free bump arena never reclaims, so peak == sum of all allocations.
const ARENA_BYTES: u64 = 1024 * 1024 * 1024;
/// First address handed out by the allocator (offset 0 of the data region).
const ARENA_DATA: u64 = ARENA_ADDR + ARENA_CURSOR_RESERVE;
/// Usable data-region size (the bytes available to bump-allocate).
const ARENA_DATA_BYTES: u64 = ARENA_BYTES - ARENA_CURSOR_RESERVE;

/// Dedicated call-stack region (a fourth BSS PT_LOAD). The `main` entry switches
/// `rsp` to the TOP of this region in its first instructions, so the program runs
/// on a stack WE size — not the OS-provided one (which `ulimit -s` caps, typically
/// at 8 MiB). The self-host emitter recurses deeply (recursive-descent parse + AST
/// walks over ~17.9k lines), overflowing the default stack with a SIGSEGV on the
/// guard page. Switching to a fixed, large, zero-filled BSS stack makes the
/// artifact self-contained: it runs identically regardless of the caller's
/// `ulimit`, with no host-varying bytes (all phdr constants). Placed well ABOVE the
/// arena so the two BSS mappings never overlap for any `ARENA_BYTES`.
const STACK_BYTES: u64 = 4u64 * 1024 * 1024 * 1024;
/// Stack region base vaddr — page-aligned, above the arena's top
/// (`ARENA_ADDR + ARENA_BYTES`) with a one-page gap. The stack grows DOWN from the
/// region's top, so the initial `rsp` is `STACK_ADDR + STACK_BYTES` (kept
/// 16-aligned: both operands are page-multiples).
const STACK_ADDR: u64 = ARENA_ADDR + ARENA_BYTES + 0x1000;
/// Initial `rsp` — the top of the stack region (stack grows downward).
const STACK_TOP: u64 = STACK_ADDR + STACK_BYTES;

/// System-V integer argument registers, in order. `(modrm_reg_field, needs_rex)`.
const ARG_REGS: [(u8, bool); 6] = [
    (7, false), // rdi
    (6, false), // rsi
    (2, false), // rdx
    (1, false), // rcx
    (0, true),  // r8  (REX.R extends the reg field)
    (1, true),  // r9
];

/// One lowered function plus its still-unresolved call sites.
struct Func {
    name: String,
    code: Vec<u8>,
    /// `(offset_of_rel32_within_code, callee_name)`.
    calls: Vec<(usize, String)>,
}

/// Frame-slot displacement from `rbp`: slots grow downward, `rbp - 8*(i+1)`.
fn slot_disp(i: usize) -> i32 {
    -((8 * (i + 1)) as i32)
}

fn store_rax(code: &mut Vec<u8>, disp: i32) {
    code.extend_from_slice(&[0x48, 0x89, 0x85]); // mov [rbp+disp32], rax
    code.extend_from_slice(&disp.to_le_bytes());
}

fn load_rax(code: &mut Vec<u8>, disp: i32) {
    code.extend_from_slice(&[0x48, 0x8B, 0x85]); // mov rax, [rbp+disp32]
    code.extend_from_slice(&disp.to_le_bytes());
}

fn arith_rax_mem(code: &mut Vec<u8>, op: &BinOp, disp: i32) -> Result<(), NativeError> {
    let opbytes: &[u8] = match op {
        BinOp::Add => &[0x48, 0x03, 0x85],       // add  rax, [rbp+disp32]
        BinOp::Sub => &[0x48, 0x2B, 0x85],       // sub  rax, [rbp+disp32]
        BinOp::Mul => &[0x48, 0x0F, 0xAF, 0x85], // imul rax, [rbp+disp32]
        #[cfg(feature = "std-surface")]
        BinOp::BitAnd => &[0x48, 0x23, 0x85], // and rax, [rbp+disp32]
        #[cfg(feature = "std-surface")]
        BinOp::BitOr => &[0x48, 0x0B, 0x85], // or  rax, [rbp+disp32]
        #[cfg(feature = "std-surface")]
        BinOp::BitXor => &[0x48, 0x33, 0x85], // xor rax, [rbp+disp32]
        other => return Err(NativeError::Unsupported(format!("BinOp {other:?}"))),
    };
    code.extend_from_slice(opbytes);
    code.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

/// The `setcc` second opcode byte for a comparison `BinOp`, or `None` for a
/// non-comparison op. Signed forms (setl/setle/setg/setge) — MIND `i64` is
/// signed; `sete`/`setne` are sign-agnostic.
fn setcc_opcode(op: &BinOp) -> Option<u8> {
    Some(match op {
        BinOp::Lt => 0x9C, // setl
        BinOp::Le => 0x9E, // setle
        BinOp::Gt => 0x9F, // setg
        BinOp::Ge => 0x9D, // setge
        BinOp::Eq => 0x94, // sete
        BinOp::Ne => 0x95, // setne
        _ => return None,
    })
}

/// Emit a comparison `BinOp` whose `lhs` is already in rax: `cmp rax, [rhs] ;
/// setcc al ; movzx eax, al` — leaves the boolean result (0/1) in rax.
fn cmp_rax_mem(code: &mut Vec<u8>, setcc: u8, rhs_disp: i32) {
    code.extend_from_slice(&[0x48, 0x3B, 0x85]); // cmp rax, [rbp+disp32]
    code.extend_from_slice(&rhs_disp.to_le_bytes());
    code.extend_from_slice(&[0x0F, setcc, 0xC0]); // setcc al
    code.extend_from_slice(&[0x0F, 0xB6, 0xC0]); // movzx eax, al (zero-extends rax)
}

/// Signed Div/Mod with MIND's DEFINED edge cases, branchlessly (cmov, no jumps —
/// stays deterministic). `rax` holds the dividend on entry. Mirrors the MLIR
/// path: substitute divisor 1 when `rhs==0` OR (`lhs==INT_MIN && rhs==-1`) so
/// `idiv` never traps, then force the result to 0 when `rhs==0`. Net:
/// `x/0=0`, `x%0=0`, `INT_MIN/-1=INT_MIN`, `INT_MIN%-1=0`, else the usual idiv.
/// Clobbers rcx/rdx/r8/r9/r10/r11 — all dead between instrs in the slot model.
fn emit_div_mod_guarded(code: &mut Vec<u8>, is_mod: bool, rhs_disp: i32) {
    code.extend_from_slice(&[0x48, 0x8B, 0x8D]); // mov rcx, [rbp+rhs] (divisor)
    code.extend_from_slice(&rhs_disp.to_le_bytes());
    // r8 = (rhs == 0)
    code.extend_from_slice(&[0x4D, 0x31, 0xC0]); // xor r8, r8
    code.extend_from_slice(&[0x48, 0x85, 0xC9]); // test rcx, rcx
    code.extend_from_slice(&[0x41, 0x0F, 0x94, 0xC0]); // sete r8b
    // r9 = (rax == INT_MIN)
    code.extend_from_slice(&[0x49, 0xBA]); // movabs r10, INT_MIN
    code.extend_from_slice(&i64::MIN.to_le_bytes());
    code.extend_from_slice(&[0x4D, 0x31, 0xC9]); // xor r9, r9
    code.extend_from_slice(&[0x4C, 0x39, 0xD0]); // cmp rax, r10
    code.extend_from_slice(&[0x41, 0x0F, 0x94, 0xC1]); // sete r9b
    // r11 = (rcx == -1); r9 = overflow = (rax==INT_MIN) & (rcx==-1)
    code.extend_from_slice(&[0x4D, 0x31, 0xDB]); // xor r11, r11
    code.extend_from_slice(&[0x48, 0x83, 0xF9, 0xFF]); // cmp rcx, -1
    code.extend_from_slice(&[0x41, 0x0F, 0x94, 0xC3]); // sete r11b
    code.extend_from_slice(&[0x4D, 0x21, 0xD9]); // and r9, r11
    code.extend_from_slice(&[0x4D, 0x09, 0xC1]); // or  r9, r8   (substitute = ovf | rhs0)
    // dsf: rcx = substitute ? 1 : rcx
    code.extend_from_slice(&[0x49, 0xC7, 0xC2, 0x01, 0x00, 0x00, 0x00]); // mov r10, 1
    code.extend_from_slice(&[0x4D, 0x85, 0xC9]); // test r9, r9
    code.extend_from_slice(&[0x49, 0x0F, 0x45, 0xCA]); // cmovnz rcx, r10
    code.extend_from_slice(&[0x48, 0x99]); // cqo
    code.extend_from_slice(&[0x48, 0xF7, 0xF9]); // idiv rcx
    if is_mod {
        code.extend_from_slice(&[0x48, 0x89, 0xD0]); // mov rax, rdx (remainder)
    }
    // result = (rhs == 0) ? 0 : result
    code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10
    code.extend_from_slice(&[0x4D, 0x85, 0xC0]); // test r8, r8
    code.extend_from_slice(&[0x49, 0x0F, 0x45, 0xC2]); // cmovnz rax, r10
}

/// Core bump allocator: the rounded-up DATA byte-size is already in `rsi` on
/// entry. Reserves an 8-byte SIZE HEADER immediately before the returned data
/// pointer (`alloc bumps by 8 + round_up(n,8)`; writes the data size at the
/// header; returns `header_base + 8` = the data ptr in `rax`). The header lets
/// `__mind_realloc` recover how many bytes to copy without any per-allocation
/// side table. Inter-allocation offsets shift by 8 per block, but intra-block
/// field offsets (`data + 8*k`) are unchanged, so struct field access is intact.
///
/// Pure function of allocation ORDER — the address depends only on the bump
/// cursor, never on malloc / ASLR / time / any host-varying value. Sequence
/// (`rsi = round_up(bytes,8)` on entry):
///   rdi = rsi + 8                      ; header(8) + data, the total bump
///   rcx = [cursor]                     ; current free offset (BSS-zeroed → 0)
///   rax = ARENA_DATA + rcx + 8         ; the returned DATA address (past header)
///   [ARENA_DATA + rcx] = rsi           ; write the data byte-size into the header
///   rdx = rcx + rdi                    ; new cursor
///   if rdx > ARENA_DATA_BYTES { ud2 }  ; exhaustion → trap (never wrap/alias)
///   [cursor] = rdx
///
/// Clobbers rdi/rcx/rdx/r8/r9 (all dead between instrs in the slot model);
/// leaves the data address in rax and preserves nothing else.
fn emit_alloc_core(code: &mut Vec<u8>) {
    // rdi = rsi + 8   (total bump = 8-byte header + rounded data size)
    code.extend_from_slice(&[0x48, 0x89, 0xF7]); // mov rdi, rsi
    code.extend_from_slice(&[0x48, 0x83, 0xC7, 0x08]); // add rdi, 8
    // rcx = [cursor]   (absolute load from the fixed cursor vaddr)
    code.extend_from_slice(&[0x48, 0xB8]); // movabs rax, ARENA_ADDR (cursor vaddr)
    code.extend_from_slice(&ARENA_ADDR.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x8B, 0x08]); // mov rcx, [rax]
    // rax = ARENA_DATA + rcx   (header base for this allocation)
    code.extend_from_slice(&[0x48, 0xBA]); // movabs rdx, ARENA_DATA
    code.extend_from_slice(&ARENA_DATA.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x89, 0xC8]); // mov rax, rcx
    code.extend_from_slice(&[0x48, 0x01, 0xD0]); // add rax, rdx  (rax = ARENA_DATA + off = header base)
    // [rax] = rsi   (write the data byte-size into the 8-byte size header)
    code.extend_from_slice(&[0x48, 0x89, 0x30]); // mov [rax], rsi
    // rax = rax + 8   (advance past the header → the DATA pointer we return)
    code.extend_from_slice(&[0x48, 0x83, 0xC0, 0x08]); // add rax, 8
    // rdx = rcx + rdi   (new cursor offset = old + header + data)
    code.extend_from_slice(&[0x48, 0x89, 0xCA]); // mov rdx, rcx
    code.extend_from_slice(&[0x48, 0x01, 0xFA]); // add rdx, rdi
    // exhaustion check: if rdx > ARENA_DATA_BYTES { ud2 }
    code.extend_from_slice(&[0x49, 0xB8]); // movabs r8, ARENA_DATA_BYTES
    code.extend_from_slice(&ARENA_DATA_BYTES.to_le_bytes());
    code.extend_from_slice(&[0x49, 0x39, 0xD0]); // cmp r8, rdx   (r8 - rdx)
    code.extend_from_slice(&[0x0F, 0x83, 0x02, 0x00, 0x00, 0x00]); // jae +2 (skip ud2 when ARENA_DATA_BYTES >= rdx)
    code.extend_from_slice(&[0x0F, 0x0B]); // ud2 (trap on exhaustion — fail loud)
    // [cursor] = rdx   (advance the bump cursor; rax still holds the data address)
    code.extend_from_slice(&[0x49, 0xB9]); // movabs r9, ARENA_ADDR (cursor vaddr)
    code.extend_from_slice(&ARENA_ADDR.to_le_bytes());
    code.extend_from_slice(&[0x49, 0x89, 0x11]); // mov [r9], rdx
}

/// Inline `__mind_alloc(bytes) -> addr`: a deterministic bump allocator over the
/// fixed-vaddr BSS arena, with an 8-byte size header per allocation (see
/// `emit_alloc_core`). Leaves the allocated DATA address in `rax`. `bytes_disp`
/// is the frame slot holding the requested size.
fn emit_alloc(code: &mut Vec<u8>, bytes_disp: i32) {
    // rsi = [bytes]
    code.extend_from_slice(&[0x48, 0x8B, 0xB5]); // mov rsi, [rbp+disp32]
    code.extend_from_slice(&bytes_disp.to_le_bytes());
    // rsi = (rsi + 7) & ~7   — round up to the 8-byte stride
    code.extend_from_slice(&[0x48, 0x83, 0xC6, 0x07]); // add rsi, 7
    code.extend_from_slice(&[0x48, 0x83, 0xE6, 0xF8]); // and rsi, -8
    emit_alloc_core(code); // header + data; data ptr -> rax
}

/// Inline `__mind_realloc(addr, new_bytes) -> new_addr`: grow-and-preserve over
/// the no-free bump arena. The front-end (std/vec.mind:85, std/string.mind:97)
/// calls this to grow a Vec/String backing store while keeping the existing
/// elements. Contract (std/vec.mind:65):
///   - `addr == 0` (NULL) ⇒ a fresh allocation, exactly `__mind_alloc(new_bytes)`.
///   - `addr != 0` ⇒ allocate `new_bytes`, COPY the old block forward, return the
///     new data ptr. No free — the arena is no-free; the old block is abandoned.
///
/// The byte-count to copy is `min(old_size, round_up(new_bytes,8))`, where
/// `old_size` is read from the 8-byte size header at `[addr - 8]` (written by
/// `emit_alloc_core`). The copy is a deterministic 8-byte-stride forward loop
/// (every emitted byte is a pure function of the IR — fixed encodings, fixed
/// loop, absolute movabs only — so the cross-substrate byte-identity wedge
/// holds). `addr_disp` / `bytes_disp` are the two arg slots. Leaves the new data
/// address in `rax`.
///
/// The old size is always an exact multiple of 8 (the header stores the rounded
/// data size), and the new size is rounded up to 8 here, so the min is a multiple
/// of 8 and the 8-byte-stride copy never over- or under-runs either block.
fn emit_realloc(code: &mut Vec<u8>, addr_disp: i32, bytes_disp: i32) {
    // Re-read the OLD data ptr and the new size from their frame slots AFTER the
    // allocation (rather than caching them in registers across emit_alloc_core).
    // The slots are stable and re-reading keeps this a leaf sequence that touches
    // no callee-saved registers — so non-entry callees (vec_push compiled as a
    // function) never violate the System-V callee-save contract. No `call` is
    // emitted here, so rsp need not stay 16-aligned within the sequence.
    //
    // rsi = round_up([new_bytes], 8)   (the new DATA size, 8-aligned)
    code.extend_from_slice(&[0x48, 0x8B, 0xB5]); // mov rsi, [rbp+new_bytes]
    code.extend_from_slice(&bytes_disp.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x83, 0xC6, 0x07]); // add rsi, 7
    code.extend_from_slice(&[0x48, 0x83, 0xE6, 0xF8]); // and rsi, -8
    // Allocate the new block (header + data). New DATA ptr -> rax. Consumes rsi and
    // clobbers rdi/rcx/rdx/r8/r9 — all dead between instrs in the slot model.
    emit_alloc_core(code);
    // r8 = [addr]   (the OLD data ptr; 0 ⇒ fresh). Re-read post-alloc; r8 is caller-
    // saved (not callee-saved) and dead between instrs, so no ABI obligation.
    code.extend_from_slice(&[0x4C, 0x8B, 0x85]); // mov r8, [rbp+addr]
    code.extend_from_slice(&addr_disp.to_le_bytes());
    // If the old addr was NULL, we are done — return the fresh allocation in rax.
    code.extend_from_slice(&[0x4D, 0x85, 0xC0]); // test r8, r8
    let je_done = {
        code.extend_from_slice(&[0x0F, 0x84]); // je rel32 (skip the copy when addr==0)
        let site = code.len();
        code.extend_from_slice(&[0, 0, 0, 0]);
        site
    };
    // --- grow path: copy min(old_size, new_size) bytes from r8 to rax ---
    // r10 = old_size = [r8 - 8]   (the OLD block's size header, always 8-multiple)
    code.extend_from_slice(&[0x4D, 0x8B, 0x50, 0xF8]); // mov r10, [r8-8]
    // r11 = new_size = round_up([new_bytes],8); n = min(old r10, new r11) -> r10
    code.extend_from_slice(&[0x4C, 0x8B, 0x9D]); // mov r11, [rbp+new_bytes]
    code.extend_from_slice(&bytes_disp.to_le_bytes());
    code.extend_from_slice(&[0x49, 0x83, 0xC3, 0x07]); // add r11, 7
    code.extend_from_slice(&[0x49, 0x83, 0xE3, 0xF8]); // and r11, -8
    code.extend_from_slice(&[0x4D, 0x39, 0xDA]); // cmp r10, r11   (r10 - r11)
    code.extend_from_slice(&[0x4D, 0x0F, 0x4F, 0xD3]); // cmovg r10, r11  (if old > new, n = new)
    // 8-byte-stride forward copy loop: rcx = 0; while rcx < r10 { [rax+rcx] = [r8+rcx]; rcx += 8 }
    code.extend_from_slice(&[0x48, 0x31, 0xC9]); // xor rcx, rcx
    let loop_top = code.len();
    code.extend_from_slice(&[0x4C, 0x39, 0xD1]); // cmp rcx, r10   (rcx - r10)
    code.extend_from_slice(&[0x0F, 0x8D]); // jge rel32 (exit when rcx >= n)
    let jge_exit = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    code.extend_from_slice(&[0x49, 0x8B, 0x14, 0x08]); // mov rdx, [r8+rcx]  (old qword)
    code.extend_from_slice(&[0x48, 0x89, 0x14, 0x08]); // mov [rax+rcx], rdx  (new qword)
    code.extend_from_slice(&[0x48, 0x83, 0xC1, 0x08]); // add rcx, 8
    {
        code.push(0xE9); // jmp rel32 back to loop_top
        let site = code.len();
        code.extend_from_slice(&[0, 0, 0, 0]);
        patch_rel32(code, site, loop_top);
    }
    let after_loop = code.len();
    patch_rel32(code, jge_exit, after_loop);
    let done_at = code.len();
    patch_rel32(code, je_done, done_at);
    // rax holds the new DATA ptr (the contract's return value) in both paths.
}

/// Inline `__mind_store_i64(addr, val)`: `mov [addr], val` (8-byte store through
/// the i64 handle). `addr_disp` / `val_disp` are the two arg slots. The intrinsic
/// has no meaningful return value; the caller's dst slot is left untouched.
fn emit_store_i64(code: &mut Vec<u8>, addr_disp: i32, val_disp: i32) {
    code.extend_from_slice(&[0x48, 0x8B, 0x85]); // mov rax, [rbp+addr]
    code.extend_from_slice(&addr_disp.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x8B, 0x8D]); // mov rcx, [rbp+val]
    code.extend_from_slice(&val_disp.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x89, 0x08]); // mov [rax], rcx
}

/// Inline `__mind_load_i64(addr) -> val`: `mov rax, [addr]` (8-byte load through
/// the i64 handle). Leaves the loaded value in `rax`. `addr_disp` is the arg slot.
fn emit_load_i64(code: &mut Vec<u8>, addr_disp: i32) {
    code.extend_from_slice(&[0x48, 0x8B, 0x85]); // mov rax, [rbp+addr]
    code.extend_from_slice(&addr_disp.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x8B, 0x00]); // mov rax, [rax]
}

/// Inline `__mind_store_i8(addr, val)`: write the LOW BYTE of `val` only — a
/// 1-byte store through the i64 handle (`mov [addr], cl`). This is the byte-buffer
/// / string-literal lowering path: the front-end emits one `__mind_store_i8` per
/// UTF-8 byte (eval/lower.rs). The store touches exactly one byte and must NOT
/// clobber the adjacent 7 — so `mov byte [rax], cl` (no REX.W), not a qword store.
/// Mirrors the MLIR path's `llvm.trunc` to i8 + 1-byte store. No meaningful result.
fn emit_store_i8(code: &mut Vec<u8>, addr_disp: i32, val_disp: i32) {
    code.extend_from_slice(&[0x48, 0x8B, 0x85]); // mov rax, [rbp+addr]
    code.extend_from_slice(&addr_disp.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x8B, 0x8D]); // mov rcx, [rbp+val]
    code.extend_from_slice(&val_disp.to_le_bytes());
    code.extend_from_slice(&[0x88, 0x08]); // mov byte [rax], cl (1 byte, low byte of val)
}

/// Inline `__mind_load_i8(addr) -> val`: load ONE byte and ZERO-extend it to i64
/// (`movzx rax, byte [addr]`). Zero-extend (movzx), NOT sign-extend (movsx): the
/// byte is an unsigned `u8` (matches the MLIR path's `llvm.zext` to i64 — a byte
/// `0xFF` reads back as 255, never -1). Leaves the value in `rax`. This is the
/// `string_get_byte` / index read path. `addr_disp` is the arg slot.
fn emit_load_i8(code: &mut Vec<u8>, addr_disp: i32) {
    code.extend_from_slice(&[0x48, 0x8B, 0x85]); // mov rax, [rbp+addr]
    code.extend_from_slice(&addr_disp.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x0F, 0xB6, 0x00]); // movzx rax, byte [rax] (zero-extend)
}

/// Inline `__mind_write(fd, buf_addr, count, offset) -> i64`: the raw `write(2)`
/// syscall — NO libc. The std/io.mind contract (and `print_bytes` at io.mind:69)
/// passes `offset == -1` ("use the current stream position", the `pwrite(-1)`
/// convention), which is exactly `write(2)`'s behaviour. So the print path lowers
/// to plain `write(2)` (x86-64 syscall #1: rdi=fd, rsi=buf, rdx=count, number in
/// rax), and the IR's `offset` arg is intentionally unused — its only print-path
/// value is the `-1` sentinel that `write` already implements.
///
/// `write`'s return (bytes written, or a negative `-errno`) lands in rax and is
/// stored to the caller's dst slot — matching the i64 contract in io.mind:15.
///
/// Pure function of the IR: only frame-slot `mov`s, fixed syscall-number bytes,
/// and `syscall` — no host-varying bytes, so byte-identity holds. The syscall
/// clobbers rax (result), rcx and r11 (per the Linux/System-V syscall ABI); all
/// are dead between instructions in the slot model, so no save/restore is needed.
///
/// deferred: a non-`-1` (absolute) `offset` would need `pwrite64` (syscall #18,
///   with the offset in r10) instead of `write`. The print surface only ever
///   passes `-1`, so `write(2)` is correct here; upgrade path: branch on the
///   offset slot at runtime (offset == -1 → write, else → pwrite64), or have the
///   front-end emit a distinct `__mind_pwrite` intrinsic for the seekable path.
fn emit_write(code: &mut Vec<u8>, fd_disp: i32, buf_disp: i32, count_disp: i32) {
    // rdi = [fd]
    code.extend_from_slice(&[0x48, 0x8B, 0xBD]); // mov rdi, [rbp+fd]
    code.extend_from_slice(&fd_disp.to_le_bytes());
    // rsi = [buf_addr]
    code.extend_from_slice(&[0x48, 0x8B, 0xB5]); // mov rsi, [rbp+buf]
    code.extend_from_slice(&buf_disp.to_le_bytes());
    // rdx = [count]
    code.extend_from_slice(&[0x48, 0x8B, 0x95]); // mov rdx, [rbp+count]
    code.extend_from_slice(&count_disp.to_le_bytes());
    // rax = 1 (SYS_write)
    code.extend_from_slice(&[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00]); // mov rax, 1
    code.extend_from_slice(&[0x0F, 0x05]); // syscall — result (bytes written / -errno) in rax
}

/// Inline `__mind_read(fd, buf_addr, count, offset) -> i64`: the raw `read(2)`
/// syscall — NO libc. The exact mirror of `emit_write` but for the read path
/// (std/io.mind `file_read` / `read_stdin_bytes` at io.mind:53/81). Contract:
/// `__mind_read(fd, buf_addr, count, offset) -> i64`. The streaming-input surface
/// passes `offset == -1` ("current stream position"), which is exactly `read(2)`'s
/// behaviour, so the IR's `offset` arg is intentionally unused (same `-1`-sentinel
/// reasoning as `emit_write`).
///
/// x86-64 syscall #0: `rdi=fd, rsi=buf, rdx=count`, number in rax. `read`'s return
/// (byte count, 0 at EOF, or a negative `-errno`) lands in rax and is stored to the
/// caller's dst slot — matching the i64 contract in io.mind:13.
///
/// Pure function of the IR: only frame-slot `mov`s, the fixed syscall-number bytes,
/// and `syscall` — no host-varying bytes, so byte-identity holds. The syscall
/// clobbers rax (result), rcx and r11 (Linux/System-V syscall ABI); all are dead
/// between instructions in the slot model, so no save/restore is needed.
///
/// deferred: a non-`-1` (absolute) `offset` would need `pread64` (syscall #17, with
///   the offset in r10) instead of `read`. The streaming-input surface only ever
///   passes `-1`, so `read(2)` is correct here; upgrade path mirrors emit_write's
///   pwrite note (branch on the offset slot, or a distinct `__mind_pread`).
fn emit_read(code: &mut Vec<u8>, fd_disp: i32, buf_disp: i32, count_disp: i32) {
    // rdi = [fd]
    code.extend_from_slice(&[0x48, 0x8B, 0xBD]); // mov rdi, [rbp+fd]
    code.extend_from_slice(&fd_disp.to_le_bytes());
    // rsi = [buf_addr]
    code.extend_from_slice(&[0x48, 0x8B, 0xB5]); // mov rsi, [rbp+buf]
    code.extend_from_slice(&buf_disp.to_le_bytes());
    // rdx = [count]
    code.extend_from_slice(&[0x48, 0x8B, 0x95]); // mov rdx, [rbp+count]
    code.extend_from_slice(&count_disp.to_le_bytes());
    // rax = 0 (SYS_read)
    code.extend_from_slice(&[0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00]); // mov rax, 0
    code.extend_from_slice(&[0x0F, 0x05]); // syscall — result (bytes read / 0=EOF / -errno) in rax
}

/// `mov <argreg>, [rbp+disp32]` — marshal a slot value into a System-V arg reg.
fn load_argreg(code: &mut Vec<u8>, arg_index: usize, disp: i32) -> Result<(), NativeError> {
    let (reg, ext) = *ARG_REGS
        .get(arg_index)
        .ok_or_else(|| NativeError::Unsupported(format!("arg #{arg_index} (>6, stack-passed)")))?;
    code.push(0x48 | if ext { 0x04 } else { 0x00 });
    code.push(0x8B);
    code.push(0x80 | ((reg & 7) << 3) | 0x05);
    code.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

/// `mov [rbp+disp32], <argreg>` — spill an incoming param register to its slot.
fn store_argreg(code: &mut Vec<u8>, arg_index: usize, disp: i32) -> Result<(), NativeError> {
    let (reg, ext) = *ARG_REGS.get(arg_index).ok_or_else(|| {
        NativeError::Unsupported(format!("param #{arg_index} (>6, stack-passed)"))
    })?;
    code.push(0x48 | if ext { 0x04 } else { 0x00 });
    code.push(0x89);
    code.push(0x80 | ((reg & 7) << 3) | 0x05);
    code.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

/// Assign every value defined in `body` a frame slot, in first-appearance order,
/// recursing through `If` branches. A pure function of the IR.
///
/// A loop-carried variable occupies ONE dedicated working slot across iterations.
/// A `While`'s post-body (`live_vars`) and post-loop (`exit_ids`) SSA ids resolve
/// to that slot; its pre-loop value (`init_ids`) is COPIED into the slot before
/// the header (see the `While` arm in `emit_seq`) and the loop's cond+body reads
/// of the variable are remapped to the working slot. The mem-to-mem model then
/// needs no phi forwarding — the slot just holds the live value.
///
/// Why a dedicated slot + copy instead of aliasing `init_id`'s slot directly:
/// the front-end inlines `let mut tmp = nn` as a bare-identifier read, so `tmp`
/// reuses `nn`'s ValueId. If the loop simply aliased its working slot ONTO that
/// shared init id, every body write to `tmp` would also clobber `nn` — which is
/// still read after the loop (e.g. `string_push_i64`: the divisor loop mutates
/// `tmp` while the digit loop later reads `nn`). Two SEQUENTIAL loops carrying
/// the same variable hit the same hazard the other way: loop 1's `exit_id` IS
/// loop 2's `init_id`. Copying the init value in (rather than aliasing) and
/// remapping the in-loop reads keeps each loop's working slot private, so neither
/// hazard corrupts a live value.
///
/// Pure function of the IR: the slot numbering is the deterministic
/// first-appearance walk order; `union-find` coalesces post↔exit by the same
/// order, so the mapping is independent of union order.
fn assign_slots(body: &[Instr]) -> HashMap<ValueId, usize> {
    fn define(slot: &mut HashMap<ValueId, usize>, id: ValueId) {
        let n = slot.len();
        slot.entry(id).or_insert(n);
    }
    // Collect (a, b) slot-index pairs that must share one physical slot
    // (post-body id ↔ post-loop exit id — the variable's dedicated working slot).
    fn walk(
        body: &[Instr],
        slot: &mut HashMap<ValueId, usize>,
        unions: &mut Vec<(usize, usize)>,
    ) {
        for ins in body {
            match ins {
                Instr::Param { dst, .. }
                | Instr::ConstI64(dst, _)
                | Instr::BinOp { dst, .. }
                | Instr::Call { dst, .. } => define(slot, *dst),
                Instr::If {
                    cond_instrs,
                    then_instrs,
                    else_instrs,
                    dst,
                    merges,
                    ..
                } => {
                    walk(cond_instrs, slot, unions);
                    walk(then_instrs, slot, unions);
                    walk(else_instrs, slot, unions);
                    define(slot, *dst);
                    for (m, _, _) in merges {
                        define(slot, *m);
                    }
                }
                #[cfg(feature = "std-surface")]
                Instr::While {
                    cond_instrs,
                    body: loop_body,
                    live_vars,
                    init_ids,
                    exit_ids,
                    ..
                } => {
                    walk(cond_instrs, slot, unions);
                    walk(loop_body, slot, unions);
                    for (i, (_, post_id)) in live_vars.iter().enumerate() {
                        // The post-body id is the variable's dedicated working
                        // slot; the post-loop exit id reads the same slot. The
                        // init id is NOT coalesced here — it keeps its own slot and
                        // is copied in at runtime, so a body write never clobbers a
                        // value that an init id aliases elsewhere.
                        define(slot, *post_id);
                        let ps = slot[post_id];
                        if let Some(init_id) = init_ids.get(i) {
                            if init_id.0 != usize::MAX {
                                define(slot, *init_id);
                            }
                        }
                        if let Some(exit_id) = exit_ids.get(i) {
                            define(slot, *exit_id);
                            unions.push((slot[exit_id], ps));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut slot = HashMap::new();
    let mut unions: Vec<(usize, usize)> = Vec::new();
    walk(body, &mut slot, &mut unions);

    // Union-find over the raw slot indices: coalesce every post↔exit pair so a
    // variable's post-body value and its post-loop reads land on one slot.
    let n = slot.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (a, b) in unions {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            // Lower index wins as the representative — deterministic, independent
            // of union order.
            if ra < rb {
                parent[rb] = ra;
            } else {
                parent[ra] = rb;
            }
        }
    }
    for s in slot.values_mut() {
        *s = find(&mut parent, *s);
    }
    slot
}

/// `cmp qword [rbp+disp], 0 ; je rel32` — returns the offset of the `je`'s rel32
/// field, to be patched once the else-target is known.
fn emit_cmp_je_zero(code: &mut Vec<u8>, cond_disp: i32) -> usize {
    code.extend_from_slice(&[0x48, 0x83, 0xBD]); // cmp qword [rbp+disp32], imm8
    code.extend_from_slice(&cond_disp.to_le_bytes());
    code.push(0x00);
    code.extend_from_slice(&[0x0F, 0x84]); // je rel32
    let site = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    site
}

/// `jmp rel32` — returns the offset of the rel32 field, patched at the end-target.
fn emit_jmp(code: &mut Vec<u8>) -> usize {
    code.push(0xE9);
    let site = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    site
}

/// Patch a forward jump's rel32 (relative to the instruction *after* the field).
/// Intra-function and so position-independent: the value is invariant under the
/// linker relocating the whole function, since site and target move together.
/// Emit a function return, the value already in rax: an `exit` syscall for the
/// entry function, else the standard epilogue + `ret`.
fn emit_return(code: &mut Vec<u8>, is_entry: bool) {
    if is_entry {
        code.extend_from_slice(&[0x48, 0x89, 0xC7]); // mov rdi, rax
        code.extend_from_slice(&[0x48, 0xC7, 0xC0, 0x3C, 0x00, 0x00, 0x00]); // mov rax, 60
        code.extend_from_slice(&[0x0F, 0x05]); // syscall (exit)
    } else {
        code.extend_from_slice(&[0x48, 0x89, 0xEC, 0x5D, 0xC3]); // mov rsp,rbp; pop rbp; ret
    }
}

fn patch_rel32(code: &mut [u8], site: usize, target: usize) {
    let rel = target as i64 - (site as i64 + 4);
    code[site..site + 4].copy_from_slice(&(rel as i32).to_le_bytes());
}

/// Emit one instruction sequence into `code`, recursing through `If` branches.
/// `is_entry` makes a `Return` lower to an `exit` syscall instead of `ret`.
/// An enclosing `while` loop's jump targets: `header` (for `continue`, backward)
/// and the forward `break` jump sites to patch to the loop's exit.
struct LoopFrame {
    header: usize,
    breaks: Vec<usize>,
}

fn emit_seq(
    code: &mut Vec<u8>,
    calls: &mut Vec<(usize, String)>,
    body: &[Instr],
    slot: &HashMap<ValueId, usize>,
    is_entry: bool,
    loops: &mut Vec<LoopFrame>,
    remap: &HashMap<ValueId, ValueId>,
) -> Result<(), NativeError> {
    // `disp` resolves a ValueId to its frame displacement. `remap` redirects a
    // loop-carried variable's pre-loop `init_id` to the loop's dedicated working
    // `post_id` slot for the duration of the loop's cond+body — so the body reads
    // (which the front-end emits against the init id) and writes (post id) hit the
    // same slot across iterations, without the init id's own slot being aliased
    // (which would clobber any OTHER live value sharing that init id).
    let disp = |id: &ValueId| -> Result<i32, NativeError> {
        let id = remap.get(id).unwrap_or(id);
        slot.get(id)
            .map(|&i| slot_disp(i))
            .ok_or_else(|| NativeError::Unsupported(format!("use of undefined value {id}")))
    };
    for ins in body {
        match ins {
            Instr::Param { dst, index, .. } => {
                if *index < 6 {
                    store_argreg(code, *index, disp(dst)?)?;
                } else {
                    // 7th+ arg arrives on the stack at [rbp + 16 + 8*(index-6)]
                    // (above the saved rbp and return address).
                    let off = 16 + 8 * (*index as i32 - 6);
                    load_rax(code, off);
                    store_rax(code, disp(dst)?);
                }
            }
            Instr::ConstI64(dst, v) => {
                code.extend_from_slice(&[0x48, 0xB8]); // movabs rax, imm64
                code.extend_from_slice(&v.to_le_bytes());
                store_rax(code, disp(dst)?);
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                load_rax(code, disp(lhs)?);
                match op {
                    BinOp::Div | BinOp::Mod => {
                        // MIND-defined edge cases (x/0=0, INT_MIN/-1) — branchless.
                        emit_div_mod_guarded(code, matches!(op, BinOp::Mod), disp(rhs)?);
                    }
                    #[cfg(feature = "std-surface")]
                    BinOp::Shl | BinOp::Shr => {
                        // x86 variable shift takes the count in cl.
                        code.extend_from_slice(&[0x48, 0x8B, 0x8D]); // mov rcx, [rbp+disp32]
                        code.extend_from_slice(&disp(rhs)?.to_le_bytes());
                        // shl rax,cl (D3 /4 = 0xE0) ; Shr is arithmetic: sar rax,cl (/7 = 0xF8).
                        let modrm = if matches!(op, BinOp::Shl) { 0xE0 } else { 0xF8 };
                        code.extend_from_slice(&[0x48, 0xD3, modrm]);
                    }
                    _ if setcc_opcode(op).is_some() => {
                        cmp_rax_mem(code, setcc_opcode(op).unwrap(), disp(rhs)?); // -> 0/1
                    }
                    _ => arith_rax_mem(code, op, disp(rhs)?)?, // Add/Sub/Mul
                }
                store_rax(code, disp(dst)?);
            }
            Instr::Call {
                dst, name, args, ..
            } if name.starts_with("__mind_") => {
                // Option-C i64-handle ABI intrinsics: the front-end lowers struct
                // literals / field reads to these. There is no runtime to link —
                // inline the codegen directly (a fixed, deterministic sequence)
                // instead of routing to `link()`, which would (correctly) reject
                // them as undefined callees.
                match name.as_str() {
                    "__mind_alloc" => {
                        let n = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_alloc with no size arg".into())
                        })?;
                        emit_alloc(code, disp(n)?);
                        store_rax(code, disp(dst)?); // address -> dst slot
                    }
                    "__mind_store_i64" => {
                        let addr = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_store_i64 missing addr".into())
                        })?;
                        let val = args.get(1).ok_or_else(|| {
                            NativeError::Unsupported("__mind_store_i64 missing value".into())
                        })?;
                        emit_store_i64(code, disp(addr)?, disp(val)?);
                        // No meaningful result; the dst slot is intentionally left
                        // as-is (the front-end never reads a store's dst).
                    }
                    "__mind_load_i64" => {
                        let addr = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_load_i64 missing addr".into())
                        })?;
                        emit_load_i64(code, disp(addr)?);
                        store_rax(code, disp(dst)?); // loaded value -> dst slot
                    }
                    "__mind_store_i8" => {
                        // 1-byte store (string-literal / byte-buffer lowering): write
                        // only the low byte of `val`, never the adjacent 7.
                        let addr = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_store_i8 missing addr".into())
                        })?;
                        let val = args.get(1).ok_or_else(|| {
                            NativeError::Unsupported("__mind_store_i8 missing value".into())
                        })?;
                        emit_store_i8(code, disp(addr)?, disp(val)?);
                        // No meaningful result; the dst slot is intentionally left
                        // as-is (the front-end never reads a store's dst).
                    }
                    "__mind_load_i8" => {
                        // 1-byte load (string_get_byte / index read): zero-extend the
                        // unsigned byte to i64.
                        let addr = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_load_i8 missing addr".into())
                        })?;
                        emit_load_i8(code, disp(addr)?);
                        store_rax(code, disp(dst)?); // zero-extended byte -> dst slot
                    }
                    "__mind_realloc" => {
                        // Grow-and-preserve over the no-free bump arena. Contract
                        // (std/vec.mind:65): __mind_realloc(addr, new_bytes)->new_addr,
                        // addr==0 ⇒ fresh alloc, addr!=0 ⇒ alloc + copy old block
                        // forward + return new addr (old block abandoned). The 8-byte
                        // size header (emit_alloc_core) is what lets the copy know
                        // how many bytes to preserve.
                        let addr = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_realloc missing addr".into())
                        })?;
                        let new_bytes = args.get(1).ok_or_else(|| {
                            NativeError::Unsupported("__mind_realloc missing new_bytes".into())
                        })?;
                        emit_realloc(code, disp(addr)?, disp(new_bytes)?);
                        store_rax(code, disp(dst)?); // new data address -> dst slot
                    }
                    "__mind_free" => {
                        // No-free bump arena: `__mind_free(ptr)` reclaims nothing
                        // (the arena only ever bumps; see emit_alloc_core). The
                        // std/vec.mind `vec_free` (and the realloc/free contract at
                        // vec.mind:113) define the result as 0 and `__mind_free(0)`
                        // as a no-op — both fall out of "do nothing, return 0". The
                        // ptr arg is intentionally unread: there is no metadata to
                        // touch. Pure constant emit (no memory access, no host-
                        // varying bytes), so byte-identity holds.
                        //
                        // deferred: a real freelist/coalescing allocator would splice
                        //   the block back onto a free list keyed off the 8-byte size
                        //   header (emit_alloc_core). Upgrade path: emit that splice
                        //   here once the arena gains a freelist; the contract's
                        //   `-> 0` return is unchanged.
                        let _ptr = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_free missing ptr".into())
                        })?;
                        // rax = 0 — the intrinsic's contract value.
                        code.extend_from_slice(&[0x48, 0x31, 0xC0]); // xor rax, rax
                        store_rax(code, disp(dst)?); // 0 -> dst slot
                    }
                    "__mind_write" => {
                        // Raw write(2) syscall (NO libc) — the print path
                        // (std/io.mind print_bytes / file_write). Contract:
                        // __mind_write(fd, buf_addr, count, offset) -> i64.
                        // The print surface always passes offset == -1 ("current
                        // position"), which is exactly write(2); the offset arg is
                        // unused here (see emit_write's deferred-pwrite note).
                        let fd = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_write missing fd".into())
                        })?;
                        let buf = args.get(1).ok_or_else(|| {
                            NativeError::Unsupported("__mind_write missing buf_addr".into())
                        })?;
                        let count = args.get(2).ok_or_else(|| {
                            NativeError::Unsupported("__mind_write missing count".into())
                        })?;
                        // args[3] (offset) is the -1 "current position" sentinel on
                        // the print path; write(2) ignores it. It must still be
                        // present (the four-arg contract) — fail loud if absent.
                        let _offset = args.get(3).ok_or_else(|| {
                            NativeError::Unsupported("__mind_write missing offset".into())
                        })?;
                        emit_write(code, disp(fd)?, disp(buf)?, disp(count)?);
                        store_rax(code, disp(dst)?); // bytes written (or -errno) -> dst
                    }
                    "__mind_read" => {
                        // Raw read(2) syscall (NO libc) — the streaming-input path
                        // (std/io.mind file_read / read_stdin_bytes). Contract:
                        // __mind_read(fd, buf_addr, count, offset) -> i64. The input
                        // surface passes offset == -1 ("current position"), which is
                        // exactly read(2); the offset arg is unused (see emit_read's
                        // deferred-pread note). The exact dual of the __mind_write arm.
                        let fd = args.first().ok_or_else(|| {
                            NativeError::Unsupported("__mind_read missing fd".into())
                        })?;
                        let buf = args.get(1).ok_or_else(|| {
                            NativeError::Unsupported("__mind_read missing buf_addr".into())
                        })?;
                        let count = args.get(2).ok_or_else(|| {
                            NativeError::Unsupported("__mind_read missing count".into())
                        })?;
                        // args[3] (offset) is the -1 "current position" sentinel on
                        // the input path; read(2) ignores it. It must still be present
                        // (the four-arg contract) — fail loud if absent.
                        let _offset = args.get(3).ok_or_else(|| {
                            NativeError::Unsupported("__mind_read missing offset".into())
                        })?;
                        emit_read(code, disp(fd)?, disp(buf)?, disp(count)?);
                        store_rax(code, disp(dst)?); // bytes read (0=EOF / -errno) -> dst
                    }
                    other => {
                        // Wider sub-i64 helpers (__mind_store_i16/i32, …) are not yet
                        // inlined — fail loud rather than silently miscompile.
                        return Err(NativeError::Unsupported(format!(
                            "intrinsic {other} (only \
                             __mind_alloc/realloc/free/store_i64/load_i64/store_i8/load_i8/write/read inlined)"
                        )));
                    }
                }
            }
            Instr::Call {
                dst, name, args, ..
            } => {
                // Args 0..6 go in registers; 7th+ are pushed (right-to-left so
                // [rsp]=arg6 at the call). Keep rsp 16-aligned at the call: pad by
                // 8 when an odd number of stack args would misalign it.
                let n_stack = args.len().saturating_sub(6);
                let pad = (n_stack % 2) * 8;
                if pad > 0 {
                    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x08]); // sub rsp, 8
                }
                for a in args.iter().skip(6).rev() {
                    load_rax(code, disp(a)?); // mov rax, [slot]
                    code.push(0x50); // push rax
                }
                for (i, a) in args.iter().take(6).enumerate() {
                    load_argreg(code, i, disp(a)?)?;
                }
                code.push(0xE8); // call rel32 (placeholder; linker-patched)
                calls.push((code.len(), name.clone()));
                code.extend_from_slice(&[0, 0, 0, 0]);
                let cleanup = (8 * n_stack + pad) as i32;
                if cleanup > 0 {
                    code.extend_from_slice(&[0x48, 0x81, 0xC4]); // add rsp, imm32
                    code.extend_from_slice(&cleanup.to_le_bytes());
                }
                store_rax(code, disp(dst)?); // result rax -> slot
            }
            Instr::If {
                cond_id,
                cond_instrs,
                then_instrs,
                then_result,
                else_instrs,
                else_result,
                dst,
                merges,
                ..
            } => {
                // A branch that ends in a terminator (`return`/`break`/`continue`)
                // DIVERGES — control never falls through it to the merge. The
                // lowering encodes this by leaving that branch's merge-tuple edge
                // (and `then_result`/`else_result` placeholders) as the
                // `ValueId(usize::MAX)` sentinel, which has no frame slot. Mirror
                // the MLIR path (`then_ends_with_return` / `else_ends_with_return`
                // in mlir/lowering.rs): only emit the phi-forwarding + `dst` copy
                // (+ the join `jmp`) on the FALL-THROUGH side, never the diverging
                // one — otherwise `disp(usize::MAX)` is a use of an undefined
                // value. A diverging branch already emitted its own terminator
                // inside its `emit_seq`, so the copies after it are unreachable.
                let then_falls = !then_instrs.last().is_some_and(branch_diverges);
                let else_falls = !else_instrs.last().is_some_and(branch_diverges);

                // Evaluate the condition; branch to the else-block when it is 0.
                emit_seq(code, calls, cond_instrs, slot, is_entry, loops, remap)?;
                let je = emit_cmp_je_zero(code, disp(cond_id)?);

                // then-block: body, phi(then side), dst = then_result, jump to end.
                emit_seq(code, calls, then_instrs, slot, is_entry, loops, remap)?;
                let then_jmp = if then_falls {
                    for (m, then_val, _) in merges {
                        load_rax(code, disp(then_val)?);
                        store_rax(code, disp(m)?);
                    }
                    load_rax(code, disp(then_result)?);
                    store_rax(code, disp(dst)?);
                    Some(emit_jmp(code))
                } else {
                    None // diverged; its terminator was already emitted
                };

                // else-block: body, phi(else side), dst = else_result.
                let else_at = code.len();
                patch_rel32(code, je, else_at);
                emit_seq(code, calls, else_instrs, slot, is_entry, loops, remap)?;
                if else_falls {
                    for (m, _, else_val) in merges {
                        load_rax(code, disp(else_val)?);
                        store_rax(code, disp(m)?);
                    }
                    load_rax(code, disp(else_result)?);
                    store_rax(code, disp(dst)?);
                }

                let end_at = code.len();
                if let Some(jmp) = then_jmp {
                    patch_rel32(code, jmp, end_at);
                }
            }
            // `while cond { body }` — re-test the condition each iteration.
            //
            // Each loop-carried variable has a DEDICATED working slot (its
            // `post_id`'s slot; `exit_id` aliases it via `assign_slots`). Before
            // the header we COPY each variable's pre-loop value (`init_id`) into
            // that working slot, then emit the cond+body with a `remap` that
            // redirects reads of `init_id` to the working slot — so the body's
            // reads (front-end-emitted against the init id) and writes (post id)
            // share one slot across iterations, while the init id's OWN slot is
            // never aliased (it may still be live under another name after the
            // loop, e.g. `string_push_i64`'s `nn` vs `tmp`). A LoopFrame on `loops`
            // lets `break`/`continue` in the body target this loop's exit/header.
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_id,
                cond_instrs,
                body: loop_body,
                live_vars,
                init_ids,
                ..
            } => {
                // Build the per-loop remap (init_id -> post_id) and copy each
                // init value into the loop's working slot before the header.
                // Inherit the enclosing `remap` so nested loops compose.
                let mut loop_remap = remap.clone();
                for (i, (_, post_id)) in live_vars.iter().enumerate() {
                    if let Some(init_id) = init_ids.get(i) {
                        if init_id.0 != usize::MAX {
                            // Copy under the OUTER remap so a nested loop's init
                            // that names an outer loop var reads the outer slot.
                            let src = remap.get(init_id).copied().unwrap_or(*init_id);
                            // Only copy when the working slot differs from the
                            // source slot (a no-op copy is harmless but skipped).
                            if slot.get(&src) != slot.get(post_id) {
                                load_rax(code, disp(&src)?);
                                store_rax(code, disp(post_id)?);
                            }
                            loop_remap.insert(*init_id, *post_id);
                        }
                    }
                }

                let header = code.len();
                emit_seq(code, calls, cond_instrs, slot, is_entry, loops, &loop_remap)?;
                let je = emit_cmp_je_zero(code, {
                    let id = loop_remap.get(cond_id).copied().unwrap_or(*cond_id);
                    slot.get(&id)
                        .map(|&i| slot_disp(i))
                        .ok_or_else(|| {
                            NativeError::Unsupported(format!("use of undefined value {id}"))
                        })?
                }); // exit when cond == 0
                loops.push(LoopFrame {
                    header,
                    breaks: Vec::new(),
                });
                emit_seq(code, calls, loop_body, slot, is_entry, loops, &loop_remap)?;
                let frame = loops.pop().expect("loop frame pushed above");
                let back = emit_jmp(code);
                patch_rel32(code, back, header); // back-edge to the header
                let end = code.len();
                patch_rel32(code, je, end);
                for site in frame.breaks {
                    patch_rel32(code, site, end); // each `break` jumps past the loop
                }
            }
            // `continue` — jump back to the innermost loop's header (re-test).
            #[cfg(feature = "std-surface")]
            Instr::Continue { .. } => {
                let header = loops
                    .last()
                    .ok_or_else(|| NativeError::Unsupported("continue outside a loop".into()))?
                    .header;
                let jmp = emit_jmp(code);
                patch_rel32(code, jmp, header);
            }
            // `break` — jump past the innermost loop; patched once its end is known.
            #[cfg(feature = "std-surface")]
            Instr::Break { .. } => {
                let jmp = emit_jmp(code);
                loops
                    .last_mut()
                    .ok_or_else(|| NativeError::Unsupported("break outside a loop".into()))?
                    .breaks
                    .push(jmp);
            }
            Instr::Return { value } => {
                if let Some(v) = value {
                    load_rax(code, disp(v)?);
                }
                emit_return(code, is_entry);
            }
            other => return Err(NativeError::Unsupported(instr_kind(other).to_string())),
        }
    }
    Ok(())
}

/// Lower one function body to machine code. `is_entry` swaps the final `Return`
/// for an `exit` syscall (the program's `_start`).
fn lower_fn(
    name: &str,
    body: &[Instr],
    is_entry: bool,
    ret_id: Option<ValueId>,
) -> Result<Func, NativeError> {
    let slot = assign_slots(body);
    let frame = (8 * slot.len()).next_multiple_of(16) as i32; // keep rsp 16-aligned

    let mut code = Vec::new();

    // Entry-only stack switch: move rsp to the top of the dedicated BSS stack
    // region (STACK_TOP) before the standard prologue, so the whole program runs on
    // a stack WE size rather than the OS-provided one (`ulimit -s`-capped). The
    // self-host emitter's deep recursion overflows the default 8 MiB stack with a
    // SIGSEGV; STACK_TOP is 16-aligned (page-multiple), so the call/push ABI stays
    // correct. The OS-provided argc/argv/envp/auxv on the original stack are
    // abandoned — `main()` takes no args and never reads them, and the program's
    // terminal `exit` syscall needs no stack. Pure constant emit (a fixed movabs),
    // no host-varying bytes, so byte-identity holds.
    if is_entry {
        code.extend_from_slice(&[0x48, 0xBC]); // movabs rsp, STACK_TOP
        code.extend_from_slice(&STACK_TOP.to_le_bytes());
    }

    // Prologue: push rbp ; mov rbp, rsp ; sub rsp, frame
    code.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5, 0x48, 0x81, 0xEC]);
    code.extend_from_slice(&frame.to_le_bytes());

    let mut calls = Vec::new();
    let mut loops: Vec<LoopFrame> = Vec::new();
    let remap = HashMap::new();
    emit_seq(&mut code, &mut calls, body, &slot, is_entry, &mut loops, &remap)?;

    // If the body falls off the end without an explicit `Return` (a trailing
    // value-expression whose result is the FnDef's `ret_id` — e.g. a value-`if`
    // or block body), emit the implicit return so control never falls through.
    if !matches!(body.last(), Some(Instr::Return { .. })) {
        if let Some(rid) = ret_id {
            if let Some(&s) = slot.get(&rid) {
                load_rax(&mut code, slot_disp(s));
            }
        }
        emit_return(&mut code, is_entry);
    }

    Ok(Func {
        name: name.to_string(),
        code,
        calls,
    })
}

/// True if `i` terminates control flow — it never falls through to a following
/// instruction. Mirrors `mlir/lowering.rs::instr_is_block_terminator`: a `return`
/// (always), or — under std-surface — a `break`/`continue`. The `If` arm uses
/// this so a branch that diverges (and whose merge edges are therefore the
/// `ValueId(usize::MAX)` non-fall-through sentinel) skips its phi-forwarding /
/// `dst` copy / join `jmp`, never emitting `disp(usize::MAX)`.
fn branch_diverges(i: &Instr) -> bool {
    match i {
        Instr::Return { .. } => true,
        #[cfg(feature = "std-surface")]
        Instr::Break { .. } | Instr::Continue { .. } => true,
        _ => false,
    }
}

/// A short human label for an unsupported instruction (avoids `Debug` noise).
fn instr_kind(i: &Instr) -> &'static str {
    match i {
        Instr::MatMul { .. } => "MatMul",
        Instr::Dot { .. } => "Dot",
        Instr::ConstF64(..) => "ConstF64",
        Instr::ConstTensor(..) => "ConstTensor",
        Instr::Output(_) => "Output",
        Instr::FnDef { .. } => "nested FnDef",
        _ => "instruction outside the scalar-i64 slice",
    }
}

/// Minimal linker: fixed layout (entry first), resolve each call's PC-relative
/// `rel32` from the layout. Returns `(image, entry_offset)`.
fn link(funcs: &[Func]) -> Result<(Vec<u8>, u64), NativeError> {
    let mut starts = Vec::with_capacity(funcs.len());
    let mut cursor = 0usize;
    for f in funcs {
        starts.push(cursor);
        cursor += f.code.len();
    }
    let mut image = Vec::with_capacity(cursor);
    for (i, f) in funcs.iter().enumerate() {
        let mut code = f.code.clone();
        for (rel_off, callee) in &f.calls {
            let idx = funcs
                .iter()
                .position(|g| g.name == *callee)
                .ok_or_else(|| NativeError::UndefinedCallee(callee.clone()))?;
            let next = starts[i] + rel_off + 4;
            let d = starts[idx] as i64 - next as i64;
            code[*rel_off..*rel_off + 4].copy_from_slice(&(d as i32).to_le_bytes());
        }
        image.extend_from_slice(&code);
    }
    Ok((image, starts[0] as u64))
}

/// Wrap raw code in a minimal, deterministic static ELF64 executable.
/// The ELF note carrying the deterministic `trace_hash` — the artifact's embedded
/// provenance (MIND's evidence-chain wedge, native-side). Standard `Elf64_Nhdr`:
/// name "MIND\0" (padded to 4), desc = the 32-byte hash.
fn build_note(trace_hash: &[u8; 32]) -> Vec<u8> {
    let mut n = Vec::with_capacity(52);
    n.extend_from_slice(&5u32.to_le_bytes()); // n_namesz ("MIND\0")
    n.extend_from_slice(&32u32.to_le_bytes()); // n_descsz
    n.extend_from_slice(&1u32.to_le_bytes()); // n_type
    n.extend_from_slice(b"MIND\0\0\0\0"); // name, 4-byte-aligned
    n.extend_from_slice(trace_hash); // desc = trace hash
    n
}

/// Minimal deterministic static ELF64: a PT_LOAD (R+X over ehdr+phdrs+code), a
/// PT_NOTE carrying the trace-hash provenance, and a third PT_LOAD (RW, BSS-style:
/// `p_filesz = 0`, `p_memsz = ARENA_BYTES`) backing the deterministic bump
/// allocator at the fixed `ARENA_ADDR`. Every byte is a pure function of
/// `(code, entry_off, note)` — the arena phdr is all compile-time constants, so it
/// adds no host-varying bytes. The note sits just past the loaded code range —
/// present in the file (readable via `readelf -n`) but not mapped — and the arena
/// occupies zero file bytes (the kernel zero-fills it at exec), preserving the
/// existing on-disk determinism.
fn write_elf(code: &[u8], entry_off: u64, note: &[u8]) -> Vec<u8> {
    const LOAD_ADDR: u64 = 0x40_0000;
    const HDRS: u64 = 64 + 4 * 56; // ehdr + four phdrs (code, note, arena, stack)
    let entry = LOAD_ADDR + HDRS + entry_off;
    let load_sz = HDRS + code.len() as u64; // PT_LOAD covers headers + code
    let note_off = HDRS + code.len() as u64;

    let mut e = Vec::with_capacity((note_off + note.len() as u64) as usize);
    // --- Elf64_Ehdr ---
    e.extend_from_slice(&[0x7F, b'E', b'L', b'F', 2, 1, 1, 0]);
    e.extend_from_slice(&[0u8; 8]);
    e.extend_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    e.extend_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    e.extend_from_slice(&1u32.to_le_bytes()); // e_version
    e.extend_from_slice(&entry.to_le_bytes()); // e_entry
    e.extend_from_slice(&64u64.to_le_bytes()); // e_phoff
    e.extend_from_slice(&0u64.to_le_bytes()); // e_shoff
    e.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    e.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    e.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
    e.extend_from_slice(&4u16.to_le_bytes()); // e_phnum = 4 (code, note, arena, stack)
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
    // --- phdr 1: PT_LOAD (R+X) over headers + code ---
    e.extend_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    e.extend_from_slice(&5u32.to_le_bytes()); // R+X
    e.extend_from_slice(&0u64.to_le_bytes()); // p_offset
    e.extend_from_slice(&LOAD_ADDR.to_le_bytes()); // p_vaddr
    e.extend_from_slice(&LOAD_ADDR.to_le_bytes()); // p_paddr
    e.extend_from_slice(&load_sz.to_le_bytes()); // p_filesz
    e.extend_from_slice(&load_sz.to_le_bytes()); // p_memsz
    e.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align
    // --- phdr 2: PT_NOTE (trace-hash provenance) ---
    e.extend_from_slice(&4u32.to_le_bytes()); // PT_NOTE
    e.extend_from_slice(&4u32.to_le_bytes()); // R
    e.extend_from_slice(&note_off.to_le_bytes()); // p_offset
    e.extend_from_slice(&(LOAD_ADDR + note_off).to_le_bytes()); // p_vaddr
    e.extend_from_slice(&(LOAD_ADDR + note_off).to_le_bytes()); // p_paddr
    e.extend_from_slice(&(note.len() as u64).to_le_bytes()); // p_filesz
    e.extend_from_slice(&(note.len() as u64).to_le_bytes()); // p_memsz
    e.extend_from_slice(&4u64.to_le_bytes()); // p_align
    // --- phdr 3: PT_LOAD (RW) — the BSS bump-allocator arena ---
    // p_filesz = 0 (no file bytes; the kernel zero-fills the mapping at exec, which
    // is exactly the deterministic zero-initialized cursor + heap the allocator
    // assumes). p_memsz = ARENA_BYTES at the fixed ARENA_ADDR vaddr. All constants,
    // so this adds zero host-varying bytes to the image.
    e.extend_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    e.extend_from_slice(&6u32.to_le_bytes()); // RW (no execute)
    e.extend_from_slice(&0u64.to_le_bytes()); // p_offset (0 file bytes)
    e.extend_from_slice(&ARENA_ADDR.to_le_bytes()); // p_vaddr
    e.extend_from_slice(&ARENA_ADDR.to_le_bytes()); // p_paddr
    e.extend_from_slice(&0u64.to_le_bytes()); // p_filesz = 0 (BSS)
    e.extend_from_slice(&ARENA_BYTES.to_le_bytes()); // p_memsz = whole arena
    e.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align (page)
    // --- phdr 4: PT_LOAD (RW) — the dedicated call-stack region (BSS) ---
    // The entry switches rsp to STACK_TOP (the top of this region) so the program
    // runs on a stack WE size, not the `ulimit -s`-capped OS stack. p_filesz = 0
    // (kernel zero-fills at exec); p_memsz = STACK_BYTES at the fixed STACK_ADDR.
    // All constants — zero host-varying bytes, so on-disk determinism is preserved.
    e.extend_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    e.extend_from_slice(&6u32.to_le_bytes()); // RW (no execute)
    e.extend_from_slice(&0u64.to_le_bytes()); // p_offset (0 file bytes)
    e.extend_from_slice(&STACK_ADDR.to_le_bytes()); // p_vaddr
    e.extend_from_slice(&STACK_ADDR.to_le_bytes()); // p_paddr
    e.extend_from_slice(&0u64.to_le_bytes()); // p_filesz = 0 (BSS)
    e.extend_from_slice(&STACK_BYTES.to_le_bytes()); // p_memsz = whole stack
    e.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align (page)
    // --- code, then the (unmapped) note ---
    e.extend_from_slice(code);
    e.extend_from_slice(note);
    e
}

/// Synthesize the deterministic entry stub used when a module has no top-level
/// `main` (a library, e.g. the self-host compiler `examples/mindc_mind/main.mind`).
/// The stub is a bare `exit(0)` syscall — `xor edi,edi ; mov eax,60 ; syscall` —
/// with NO prologue/frame (it never returns), so the produced ELF is a valid,
/// runnable artifact whose execution is a pure constant. All real library
/// functions are still lowered alongside it, so any unsupported IR construct in a
/// library body surfaces as the next blocker rather than being masked.
///
/// deferred: this stub exits immediately instead of invoking any library export —
/// upgrade path: once the front-end designates an export as the program entry
/// (e.g. a `#[entry]` attribute or a conventional `mindc_compile`), `call` it
/// here and forward its i64 return into the exit status.
fn synth_lib_entry() -> Func {
    Func {
        name: "__mind_entry".to_string(),
        // xor edi,edi (status 0) ; mov eax,60 (SYS_exit) ; syscall
        code: vec![
            0x31, 0xFF, // xor edi, edi
            0xB8, 0x3C, 0x00, 0x00, 0x00, // mov eax, 60
            0x0F, 0x05, // syscall
        ],
        calls: Vec::new(),
    }
}

/// Lower an `IRModule` to a runnable, deterministic static ELF64 — zero LLVM.
///
/// Scans `ir.instrs` for `FnDef`s, lowers each (the `main` `FnDef` is the entry),
/// links them with computed call displacements, and wraps the result in an ELF.
/// Deterministic by construction: identical `ir` ⇒ byte-identical output.
pub fn compile_to_elf(ir: &IRModule) -> Result<Vec<u8>, NativeError> {
    // Collect (name, body) for every top-level FnDef, entry (`main`) first.
    type FnSig<'a> = (&'a str, &'a [Instr], Option<ValueId>);
    let mut entry: Option<FnSig> = None;
    let mut rest: Vec<FnSig> = Vec::new();
    for ins in &ir.instrs {
        if let Instr::FnDef {
            name, body, ret_id, ..
        } = ins
        {
            let sig: FnSig = (name.as_str(), body.as_slice(), *ret_id);
            if name == "main" {
                entry = Some(sig);
            } else {
                rest.push(sig);
            }
        }
    }

    // Shadowing: when the same function name is DEFINED more than once (the
    // native path bundles the whole stdlib ahead of the user file, so a user
    // `fn bytes_eq`/`fn load_byte`/… collides with a bundled std helper of the
    // same name — e.g. std/toml.mind's own `bytes_eq`/`load_byte`), the LAST
    // definition wins (user items are appended after the std modules — the
    // last-write-wins contract documented in `bin/mind-native.rs`). `link`
    // resolves a callee to a single `Func` by NAME, so leaving both bodies in
    // would let it bind a `call` to the WRONG (earlier, differently-typed) one:
    // the bundled `bytes_eq(aa, ba, n)` (3 params) vs the program's
    // `bytes_eq(buf, lo, hi, cmp, clo, chi)` (6 params) — a runtime miscompile
    // that lowers + exits 0 yet jumps into a function with the wrong ABI. Keep
    // only the LAST `FnDef` per name (deterministic — IR order is stable, and a
    // shadowed earlier body is dropped rather than emitted as dead code).
    {
        use std::collections::HashMap;
        let mut last_index: HashMap<&str, usize> = HashMap::new();
        for (i, (name, _, _)) in rest.iter().enumerate() {
            last_index.insert(name, i);
        }
        let kept: Vec<FnSig> = rest
            .iter()
            .enumerate()
            .filter(|(i, (name, _, _))| last_index.get(name) == Some(i))
            .map(|(_, sig)| *sig)
            .collect();
        rest = kept;
    }
    let mut funcs = Vec::with_capacity(rest.len() + 1);
    match entry {
        Some((entry_name, entry_body, entry_ret)) => {
            funcs.push(lower_fn(entry_name, entry_body, true, entry_ret)?);
        }
        None => {
            // Library form (e.g. `examples/mindc_mind/main.mind`, the self-host
            // compiler): a module of exported `fn`s with NO top-level `main`.
            // Rather than refuse with `NoEntry`, synthesize a deterministic entry
            // stub that exits cleanly (status 0) and still lower every function in
            // the module — so a valid runnable ELF is produced AND any unsupported
            // IR construct in a library body surfaces as the real next blocker
            // instead of being masked by the missing-entry check.
            funcs.push(synth_lib_entry());
        }
    }
    for (name, body, ret_id) in rest {
        funcs.push(lower_fn(name, body, false, ret_id)?);
    }

    let (code, entry_off) = link(&funcs)?;
    // Embed the deterministic trace hash as an ELF note — the artifact's own
    // provenance, the same hash the mic@3 evidence chain anchors on.
    let note = build_note(&crate::ir::ir_trace_hash(ir));
    Ok(write_elf(&code, entry_off, &note))
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    /// Build `fn f(a,b,c)=a+b+c ; fn main()=f(40,1,1)` as a real `IRModule`.
    fn sample_module() -> IRModule {
        let v = ValueId;
        let f = Instr::FnDef {
            name: "f".into(),
            params: vec![
                ("a".into(), v(10)),
                ("b".into(), v(11)),
                ("c".into(), v(12)),
            ],
            ret_id: Some(v(14)),
            reap_threshold: None,
            body: vec![
                Instr::Param {
                    dst: v(10),
                    name: "a".into(),
                    index: 0,
                },
                Instr::Param {
                    dst: v(11),
                    name: "b".into(),
                    index: 1,
                },
                Instr::Param {
                    dst: v(12),
                    name: "c".into(),
                    index: 2,
                },
                Instr::BinOp {
                    dst: v(13),
                    op: BinOp::Add,
                    lhs: v(10),
                    rhs: v(11),
                },
                Instr::BinOp {
                    dst: v(14),
                    op: BinOp::Add,
                    lhs: v(13),
                    rhs: v(12),
                },
                Instr::Return { value: Some(v(14)) },
            ],
        };
        let main = Instr::FnDef {
            name: "main".into(),
            params: vec![],
            ret_id: Some(v(23)),
            reap_threshold: None,
            body: vec![
                Instr::ConstI64(v(20), 40),
                Instr::ConstI64(v(21), 1),
                Instr::ConstI64(v(22), 1),
                Instr::Call {
                    dst: v(23),
                    name: "f".into(),
                    args: vec![v(20), v(21), v(22)],
                },
                Instr::Return { value: Some(v(23)) },
            ],
        };
        let mut m = IRModule::new();
        m.instrs = vec![main, f];
        m
    }

    #[test]
    fn compiles_real_irmodule_to_a_runnable_deterministic_elf() {
        let m = sample_module();

        // Determinism by construction: identical module -> byte-identical ELF.
        let a = compile_to_elf(&m).expect("lowers");
        let b = compile_to_elf(&m).expect("lowers");
        assert_eq!(a, b, "native lowering must be byte-identical (the wedge)");

        // Runnable: exec it, assert f(40,1,1) = 42.
        assert_eq!(
            run(&a, "mind_native_module_exe"),
            Some(42),
            "must compute f(40,1,1) and exit(42)"
        );
    }

    #[test]
    fn missing_main_compiles_as_a_library_with_synth_entry() {
        // A module of exported `fn`s with NO top-level `main` (the self-host
        // compiler shape) must NOT be refused: it lowers as a library, the
        // synthesized stub becomes the entry, and the artifact runs and exits 0.
        let mut m = IRModule::new();
        m.instrs = vec![Instr::FnDef {
            name: "f".into(),
            params: vec![],
            ret_id: None,
            reap_threshold: None,
            body: vec![Instr::Return { value: None }],
        }];
        let a = compile_to_elf(&m).expect("library module must lower, not NoEntry");
        // Deterministic: same IR -> byte-identical ELF.
        assert_eq!(
            a,
            compile_to_elf(&m).expect("lowers"),
            "library lowering must be byte-identical (the wedge)"
        );
        // Runnable: the synthesized entry exits cleanly with status 0.
        assert_eq!(
            run(&a, "mind_native_lib_exe"),
            Some(0),
            "synthesized library entry must exit(0)"
        );
    }

    #[test]
    fn elf_embeds_a_deterministic_trace_hash_note() {
        let m = sample_module();
        let a = compile_to_elf(&m).expect("lowers");
        // The MIND provenance note name + the 32-byte trace hash are both present.
        assert!(
            a.windows(4).any(|w| w == b"MIND"),
            "ELF must carry the MIND note"
        );
        let hash = crate::ir::ir_trace_hash(&m);
        assert!(
            a.windows(32).any(|w| w == hash),
            "ELF must embed the IR's trace_hash as provenance"
        );
        // Same IR -> same hash -> byte-identical ELF (determinism survives the note).
        assert_eq!(a, compile_to_elf(&m).expect("lowers"));
    }

    /// Write the ELF to a unique temp path, exec it, return its exit code.
    ///
    /// The path is unique per call (pid + atomic counter) so parallel tests never
    /// collide, and the write handle is closed (inner scope) before the exec. The
    /// exec is retried on ETXTBSY (errno 26): a just-written-and-chmod'd file can
    /// briefly read as "text file busy" until the kernel drops the write
    /// reference — a real race under parallel-test load, not a logic error.
    fn run(elf: &[u8], name: &str) -> Option<i32> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static CTR: AtomicU32 = AtomicU32::new(0);
        let uniq = CTR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{name}.{}.{uniq}", std::process::id()));
        {
            let mut fh = std::fs::File::create(&path).expect("create");
            fh.write_all(elf).expect("write");
            let mut perms = fh.metadata().unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&path, perms).expect("chmod");
        } // handle dropped here, before exec
        let mut code = None;
        for _ in 0..100 {
            match Command::new(&path).status() {
                Ok(s) => {
                    code = s.code();
                    break;
                }
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec failed: {e}"),
            }
        }
        let _ = std::fs::remove_file(&path);
        code
    }

    /// `fn main() { let x = cond; if x { 42 } else { 7 } }` as a real IRModule —
    /// exercises the native `If` arm (branch + forward-jump patching + phi).
    fn if_module(cond: i64) -> IRModule {
        let v = ValueId;
        let main = Instr::FnDef {
            name: "main".into(),
            params: vec![],
            ret_id: Some(v(3)),
            reap_threshold: None,
            body: vec![
                Instr::ConstI64(v(0), cond),
                Instr::If {
                    cond_id: v(0),
                    cond_instrs: vec![],
                    then_instrs: vec![Instr::ConstI64(v(1), 42)],
                    then_result: v(1),
                    else_instrs: vec![Instr::ConstI64(v(2), 7)],
                    else_result: v(2),
                    dst: v(3),
                    branch_bindings: vec![],
                    merges: vec![],
                },
                Instr::Return { value: Some(v(3)) },
            ],
        };
        let mut m = IRModule::new();
        m.instrs = vec![main];
        m
    }

    #[test]
    fn lowers_if_expression_both_branches_run_deterministically() {
        // Truthy condition takes the then-branch (42); zero takes the else (7) —
        // proving both the fall-through and the `je`-taken paths.
        for (cond, expected) in [(40i64, 42), (0, 7)] {
            let m = if_module(cond);
            let a = compile_to_elf(&m).expect("lowers");
            assert_eq!(a, compile_to_elf(&m).expect("lowers"), "deterministic");
            assert_eq!(
                run(&a, "mind_native_if_exe"),
                Some(expected),
                "if({cond}) must exit({expected})"
            );
        }
    }

    /// `fn main() { if (a <op> b) { 42 } else { 7 } }` — a comparison BinOp feeds
    /// the `If` condition (cmp+setcc producing the 0/1 the branch tests).
    fn cmp_if_module(op: BinOp, a: i64, b: i64) -> IRModule {
        let v = ValueId;
        let main = Instr::FnDef {
            name: "main".into(),
            params: vec![],
            ret_id: Some(v(5)),
            reap_threshold: None,
            body: vec![
                Instr::ConstI64(v(0), a),
                Instr::ConstI64(v(1), b),
                Instr::If {
                    cond_id: v(2),
                    cond_instrs: vec![Instr::BinOp {
                        dst: v(2),
                        op,
                        lhs: v(0),
                        rhs: v(1),
                    }],
                    then_instrs: vec![Instr::ConstI64(v(3), 42)],
                    then_result: v(3),
                    else_instrs: vec![Instr::ConstI64(v(4), 7)],
                    else_result: v(4),
                    dst: v(5),
                    branch_bindings: vec![],
                    merges: vec![],
                },
                Instr::Return { value: Some(v(5)) },
            ],
        };
        let mut m = IRModule::new();
        m.instrs = vec![main];
        m
    }

    #[test]
    fn lowers_comparison_conditions_with_correct_signed_semantics() {
        // (op, a, b) -> taken? 42 if the comparison holds, else 7.
        let cases = [
            (BinOp::Gt, 40, 2, 42),
            (BinOp::Gt, 2, 40, 7),
            (BinOp::Lt, -5, 1, 42), // signed: -5 < 1
            (BinOp::Lt, 1, -5, 7),
            (BinOp::Eq, 9, 9, 42),
            (BinOp::Ne, 9, 9, 7),
            (BinOp::Ge, 3, 3, 42),
            (BinOp::Le, 4, 3, 7),
        ];
        for (op, a, b, expected) in cases {
            let m = cmp_if_module(op, a, b);
            let elf = compile_to_elf(&m).expect("lowers");
            assert_eq!(elf, compile_to_elf(&m).expect("lowers"), "deterministic");
            assert_eq!(
                run(&elf, "mind_native_cmp_exe"),
                Some(expected),
                "{op:?}({a}, {b}) should take the {}-branch",
                if expected == 42 { "then" } else { "else" }
            );
        }
    }

    /// `fn main() -> i64 { a <op> b }` for a binary op over two constants.
    fn binop_module(op: BinOp, a: i64, b: i64) -> IRModule {
        let v = ValueId;
        let main = Instr::FnDef {
            name: "main".into(),
            params: vec![],
            ret_id: Some(v(2)),
            reap_threshold: None,
            body: vec![
                Instr::ConstI64(v(0), a),
                Instr::ConstI64(v(1), b),
                Instr::BinOp {
                    dst: v(2),
                    op,
                    lhs: v(0),
                    rhs: v(1),
                },
                Instr::Return { value: Some(v(2)) },
            ],
        };
        let mut m = IRModule::new();
        m.instrs = vec![main];
        m
    }

    #[test]
    fn lowers_signed_div_and_mod() {
        // Div = truncated-toward-zero signed quotient; Mod = signed remainder.
        let cases = [
            (BinOp::Div, 84, 2, 42),
            (BinOp::Div, 127, 3, 42), // 127/3 = 42 (trunc)
            (BinOp::Mod, 142, 50, 42),
            (BinOp::Mod, 85, 43, 42),
            (BinOp::Div, 7, 0, 0), // MIND-defined: x/0 = 0 (would SIGFPE without the guard)
            (BinOp::Mod, 7, 0, 0), // MIND-defined: x%0 = 0
        ];
        for (op, a, b, expected) in cases {
            let m = binop_module(op, a, b);
            let elf = compile_to_elf(&m).expect("lowers");
            assert_eq!(elf, compile_to_elf(&m).expect("lowers"), "deterministic");
            assert_eq!(
                run(&elf, "mind_native_divmod_exe"),
                Some(expected),
                "{op:?}({a},{b})"
            );
        }
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_a_real_while_loop() {
        // Lower a real loop through the front-end, then native-compile + run it —
        // proves the backend eats the actual While IR (cond sub-module, body,
        // loop-carried init/post/exit ids), not a hand-built toy.
        let src = "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; \
                   while i < 7 { s = s + 6; i = i + 1; } return s; }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers the while loop");
        assert_eq!(elf, compile_to_elf(&ir).expect("lowers"), "deterministic");
        assert_eq!(
            run(&elf, "mind_native_while_exe"),
            Some(42),
            "6 * 7 iterations = 42"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_more_than_six_args_stack_passed() {
        // 8 args: a..f in registers, g/h pushed on the stack (with alignment).
        let src = "fn s8(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64, h: i64) -> i64 \
                   { a + b + c + d + e + f + g + h } \
                   fn main() -> i64 { s8(1, 2, 3, 4, 5, 6, 7, 14) }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers >6 args");
        assert_eq!(elf, compile_to_elf(&ir).expect("lowers"), "deterministic");
        assert_eq!(
            run(&elf, "mind_native_args8_exe"),
            Some(42),
            "8-arg sum = 42"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_while_with_break_and_continue() {
        // i runs 1.., `continue` skips adding i==3, `break` stops once i>9:
        // s = 1+2+4+5+6+7+8+9 = 42.
        let src = "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; \
                   while i < 100 { i = i + 1; if i == 3 { continue; } \
                   if i > 9 { break; } s = s + i; } return s; }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers break/continue");
        assert_eq!(elf, compile_to_elf(&ir).expect("lowers"), "deterministic");
        assert_eq!(
            run(&elf, "mind_native_brk_exe"),
            Some(42),
            "skip 3 + break at 10 = 42"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_value_if_with_a_diverging_branch() {
        // A value-`if` whose ONE branch diverges (`return`) — control never falls
        // through that branch to the merge. The lowering leaves that branch's
        // merge edge (and result placeholder) as the `ValueId(usize::MAX)`
        // non-fall-through sentinel, which has NO frame slot. The native backend
        // must skip the diverging branch's phi-forwarding / `dst` copy / join
        // `jmp` (mirroring the MLIR `then_ends_with_return` path) — otherwise it
        // emits `disp(usize::MAX)` and fails with "use of undefined value
        // %18446744073709551615". This is the exact construct that blocked
        // main.mind. Front-end-lowered, not a hand-built IR toy.
        //
        // `classify(n) = if n < 0 { return 7 } else { n + 1 } + 100`:
        //   n >= 0 → falls through the else-edge → (n+1)+100
        //   n <  0 → the then-branch returns 7 (the merge is never reached)
        let src = "fn classify(n: i64) -> i64 { \
                   let r: i64 = if n < 0 { return 7; } else { n + 1 }; \
                   return r + 100; } \
                   fn main() -> i64 { return classify(41); }";
        let ir = crate::eval::lower::lower_to_ir(&crate::parser::parse(src).expect("parse"));
        let elf = compile_to_elf(&ir).expect("native-lowers a diverging-branch value-if");

        // Byte-identity: skipping the dead diverging-branch copies is a pure
        // function of the IR (no host-varying bytes), so two builds are identical.
        assert_eq!(
            elf,
            compile_to_elf(&ir).expect("lowers"),
            "diverging-branch if ELF must be byte-identical"
        );

        // Fall-through (else) path: classify(41) = (41+1)+100 = 142.
        assert_eq!(
            run(&elf, "mind_native_if_diverge_else_exe"),
            Some(142),
            "n>=0 takes the else-edge: (41+1)+100 = 142"
        );

        // Diverging (then) path: classify(-5) returns 7 directly — the merge,
        // whose then-edge is the usize::MAX sentinel, is never reached.
        let src_neg = "fn classify(n: i64) -> i64 { \
                       let r: i64 = if n < 0 { return 7; } else { n + 1 }; \
                       return r + 100; } \
                       fn main() -> i64 { return classify(-5); }";
        let ir_neg =
            crate::eval::lower::lower_to_ir(&crate::parser::parse(src_neg).expect("parse"));
        let elf_neg = compile_to_elf(&ir_neg).expect("lowers");
        assert_eq!(
            run(&elf_neg, "mind_native_if_diverge_then_exe"),
            Some(7),
            "n<0 diverges through the then-branch's `return 7`"
        );

        // The MIRROR case — the ELSE branch diverges, the THEN branch falls
        // through (its else-edge merge value is the usize::MAX sentinel).
        // `classify2(n) = if n < 0 { n + 50 } else { return 9 } + 100`.
        let src_else = "fn classify2(n: i64) -> i64 { \
                        let r: i64 = if n < 0 { n + 50 } else { return 9; }; \
                        return r + 100; } \
                        fn main() -> i64 { return classify2(-8); }";
        let ir_else =
            crate::eval::lower::lower_to_ir(&crate::parser::parse(src_else).expect("parse"));
        let elf_else = compile_to_elf(&ir_else).expect("lowers");
        assert_eq!(
            run(&elf_else, "mind_native_if_diverge_else_branch_exe"),
            Some(142),
            "n<0 takes the then-edge: (-8+50)+100 = 142"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_a_struct_program_via_the_option_c_intrinsics() {
        // The front-end lowers this struct literal + field read to the Option-C
        // i64-handle ABI intrinsics — `__mind_alloc(bytes)`, `__mind_store_i64`,
        // `__mind_load_i64` — which the native backend now inlines (deterministic
        // bump allocator over the fixed-vaddr BSS arena), with NO call to a runtime.
        let src = "struct P { x: i64, y: i64 } \
                   fn main() -> i64 { let p: P = P { x: 7, y: 9 }; return p.x; }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers the struct program");

        // Byte-identity (the wedge): the bump allocator returns order-determined
        // addresses, never a host-varying value — two builds must be identical.
        assert_eq!(
            elf,
            compile_to_elf(&ir).expect("lowers"),
            "struct ELF must be byte-identical (allocator leaks no host-varying address)"
        );

        // Runs and returns the field value: p.x == 7.
        assert_eq!(
            run(&elf, "mind_native_struct_exe"),
            Some(7),
            "P {{ x: 7, y: 9 }}.x must exit(7)"
        );

        // The second field reads back from offset 8 of the same record.
        let src_y = "struct P { x: i64, y: i64 } \
                     fn main() -> i64 { let p: P = P { x: 7, y: 9 }; return p.y; }";
        let ir_y = crate::eval::lower::lower_to_ir(&crate::parser::parse(src_y).expect("parse"));
        let elf_y = compile_to_elf(&ir_y).expect("lowers");
        assert_eq!(
            run(&elf_y, "mind_native_struct_y_exe"),
            Some(9),
            "P {{ x: 7, y: 9 }}.y must exit(9)"
        );

        // Two distinct allocations must NOT alias: the bump cursor advances, so
        // `a` and `b` get separate records — a.y + b.x = 20 + 5 = 25.
        let src2 = "struct P { x: i64, y: i64 } \
                    fn main() -> i64 { let a: P = P { x: 10, y: 20 }; \
                    let b: P = P { x: 5, y: 7 }; return a.y + b.x; }";
        let ir2 = crate::eval::lower::lower_to_ir(&crate::parser::parse(src2).expect("parse"));
        let elf2 = compile_to_elf(&ir2).expect("lowers");
        assert_eq!(
            run(&elf2, "mind_native_struct_two_exe"),
            Some(25),
            "two struct records must not alias (bump cursor advances): 20 + 5 = 25"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_a_byte_buffer_program_via_store_and_load_i8() {
        // The 1-byte Option-C intrinsics — `__mind_store_i8` (write the low byte
        // only) and `__mind_load_i8` (zero-extend the unsigned byte to i64) — are
        // the string-literal / byte-buffer memory path. The native backend now
        // inlines them directly (no runtime to link). Front-end-lowered, not a
        // hand-built IR toy: `__mind_alloc` a buffer, store two bytes, read them
        // back, and exit with their sum.
        let src = "fn main() -> i64 { \
                   let addr: i64 = __mind_alloc(8); \
                   let r0: i64 = __mind_store_i8(addr, 7); \
                   let r1: i64 = __mind_store_i8(addr + 1, 35); \
                   let b0: i64 = __mind_load_i8(addr); \
                   let b1: i64 = __mind_load_i8(addr + 1); \
                   return b0 + b1; }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers the byte-buffer program");

        // Byte-identity (the wedge): two builds of the same IR must be identical —
        // the store_i8 / load_i8 codegen is a pure function of the IR (absolute
        // movabs + frame-slot moves only, no host-varying bytes).
        assert_eq!(
            elf,
            compile_to_elf(&ir).expect("lowers"),
            "byte-buffer ELF must be byte-identical (store_i8/load_i8 emit no host-varying bytes)"
        );

        // Runs and reads both bytes back independently: 7 + 35 = 42. This also
        // proves the 1-byte store does NOT clobber the adjacent byte (if the store
        // wrote a full qword, byte 0's store would overwrite byte 1's slot).
        assert_eq!(
            run(&elf, "mind_native_byte_buf_exe"),
            Some(42),
            "store_i8(7) + store_i8(35) read back must exit(42)"
        );

        // Zero-extend, not sign-extend: a byte 0xC8 (200) must load as +200, never
        // -56. A signed `> 100` test takes the then-branch (42) only under movzx;
        // movsx would yield -56 → the else-branch (7). Distinguishes the two
        // unambiguously (truncated exit codes alone would alias 200 and -56).
        let src_z = "fn main() -> i64 { \
                     let addr: i64 = __mind_alloc(8); \
                     let r0: i64 = __mind_store_i8(addr, 200); \
                     let b0: i64 = __mind_load_i8(addr); \
                     if b0 > 100 { return 42; } else { return 7; } }";
        let ir_z = crate::eval::lower::lower_to_ir(&crate::parser::parse(src_z).expect("parse"));
        let elf_z = compile_to_elf(&ir_z).expect("lowers");
        assert_eq!(
            run(&elf_z, "mind_native_byte_zext_exe"),
            Some(42),
            "load_i8 must ZERO-extend (200 > 100 → 42); sign-extend would give -56 → 7"
        );

        // The 1-byte store overwrites ONLY the low byte of an existing qword: pre-
        // fill 0xFFFF with store_i64, then store_i8(0x2A) → 0xFF2A (65322). Reading
        // back the full i64 and subtracting 65280 (0xFF00) leaves 42 only if the
        // upper byte (0xFF) survived the 1-byte store.
        let src_c = "fn main() -> i64 { \
                     let addr: i64 = __mind_alloc(8); \
                     let big: i64 = __mind_store_i64(addr, 65535); \
                     let small: i64 = __mind_store_i8(addr, 42); \
                     let back: i64 = __mind_load_i64(addr); \
                     return back - 65280; }";
        let ir_c = crate::eval::lower::lower_to_ir(&crate::parser::parse(src_c).expect("parse"));
        let elf_c = compile_to_elf(&ir_c).expect("lowers");
        assert_eq!(
            run(&elf_c, "mind_native_byte_noclobber_exe"),
            Some(42),
            "store_i8 must touch ONLY the low byte (0xFFFF→store_i8 0x2A→0xFF2A): 65322-65280=42"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_mind_free_as_a_no_op_returning_zero() {
        // `__mind_free(ptr)` over the no-free bump arena is a pure no-op that
        // returns 0 (the std/vec.mind `vec_free` contract). The native backend
        // now inlines it (no runtime to link). This is the intrinsic that the
        // self-host compiler's `vec_free` path reaches once `vec_push` resolves.
        //
        // Two assertions in one program: (1) the freed pointer's data SURVIVES
        // the free (the arena reclaims nothing), and (2) `__mind_free` evaluates
        // to 0. Store 42 at `addr`, free `addr`, then read it back and ADD the
        // free's result (which must be 0) — exit(42) only if both hold.
        let src = "fn main() -> i64 { \
                   let addr: i64 = __mind_alloc(8); \
                   let w: i64 = __mind_store_i64(addr, 42); \
                   let f: i64 = __mind_free(addr); \
                   let back: i64 = __mind_load_i64(addr); \
                   return back + f; }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers the __mind_free program");

        // Byte-identity (the wedge): `__mind_free` emits a pure constant
        // (`xor rax,rax`) — no memory access, no host-varying bytes.
        assert_eq!(
            elf,
            compile_to_elf(&ir).expect("lowers"),
            "__mind_free ELF must be byte-identical (no-op emit has no host-varying bytes)"
        );

        // free is a no-op (data survives) AND returns 0: 42 + 0 = 42.
        assert_eq!(
            run(&elf, "mind_native_free_exe"),
            Some(42),
            "__mind_free must be a no-op returning 0: load-back(42) + free-result(0) = 42"
        );

        // A second program proves the RETURN value alone is 0 (not just additively
        // neutral): if `__mind_free` returned anything non-zero, `> 0` would take
        // the then-branch (7); a 0 return takes the else-branch (42).
        let src_z = "fn main() -> i64 { \
                     let addr: i64 = __mind_alloc(8); \
                     let f: i64 = __mind_free(addr); \
                     if f > 0 { return 7; } else { return 42; } }";
        let ir_z = crate::eval::lower::lower_to_ir(&crate::parser::parse(src_z).expect("parse"));
        let elf_z = compile_to_elf(&ir_z).expect("lowers");
        assert_eq!(
            run(&elf_z, "mind_native_free_zero_exe"),
            Some(42),
            "__mind_free's result must be exactly 0 (f > 0 is false → else → 42)"
        );
    }

    /// Write the ELF to a unique temp path, exec it, and capture both its stdout
    /// bytes and its exit code. Same ETXTBSY-retry + unique-path discipline as
    /// `run`, but returns the captured stdout alongside the code so a test can
    /// assert the EXACT bytes a `__mind_write`-emitting program prints.
    fn run_capture(elf: &[u8], name: &str) -> Option<(Vec<u8>, i32)> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static CTR: AtomicU32 = AtomicU32::new(0);
        let uniq = CTR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{name}.{}.{uniq}", std::process::id()));
        {
            let mut fh = std::fs::File::create(&path).expect("create");
            fh.write_all(elf).expect("write");
            let mut perms = fh.metadata().unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&path, perms).expect("chmod");
        } // handle dropped here, before exec
        let mut out = None;
        for _ in 0..100 {
            match Command::new(&path).output() {
                Ok(o) => {
                    out = Some((o.stdout, o.status.code().unwrap_or(-1)));
                    break;
                }
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec failed: {e}"),
            }
        }
        let _ = std::fs::remove_file(&path);
        out
    }

    /// Like `run_capture`, but FEED `stdin_bytes` to the program's stdin first.
    /// Lets a test exercise the read(2) path end-to-end (the program reads what we
    /// pipe in). Same ETXTBSY-retry + unique-path discipline as `run_capture`.
    fn run_capture_stdin(elf: &[u8], name: &str, stdin_bytes: &[u8]) -> Option<(Vec<u8>, i32)> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static CTR: AtomicU32 = AtomicU32::new(0);
        let uniq = CTR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{name}.{}.{uniq}", std::process::id()));
        {
            let mut fh = std::fs::File::create(&path).expect("create");
            fh.write_all(elf).expect("write");
            let mut perms = fh.metadata().unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&path, perms).expect("chmod");
        } // handle dropped here, before exec
        let mut out = None;
        for _ in 0..100 {
            let spawn = Command::new(&path)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn();
            match spawn {
                Ok(mut child) => {
                    child
                        .stdin
                        .take()
                        .expect("stdin pipe")
                        .write_all(stdin_bytes)
                        .expect("feed stdin");
                    let o = child.wait_with_output().expect("wait");
                    out = Some((o.stdout, o.status.code().unwrap_or(-1)));
                    break;
                }
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec failed: {e}"),
            }
        }
        let _ = std::fs::remove_file(&path);
        out
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_a_program_that_reads_stdin_then_echoes_via_mind_read_write() {
        // The raw read(2) syscall path (NO libc), the dual of the __mind_write test.
        // Front-end-lowered (not a hand-built IR toy): alloc a buffer, read up to 8
        // bytes from stdin (fd 0, offset -1 = "current position" — the exact shape
        // read_stdin_bytes at std/io.mind:81 emits), echo exactly `n` bytes back to
        // stdout via __mind_write, and exit with read's return (the byte count).
        let src = "fn main() -> i64 { \
                   let addr: i64 = __mind_alloc(8); \
                   let n: i64 = __mind_read(0, addr, 8, 0 - 1); \
                   let w: i64 = __mind_write(1, addr, n, 0 - 1); \
                   return n; }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers the __mind_read program");

        // Byte-identity (the wedge): the read codegen is a pure function of the IR
        // (frame-slot movs + fixed syscall-number bytes + `syscall`, no host-varying
        // bytes) — two builds of the same IR must be byte-for-byte identical.
        assert_eq!(
            elf,
            compile_to_elf(&ir).expect("lowers"),
            "__mind_read ELF must be byte-identical (no host-varying bytes)"
        );

        // RUN it: pipe "MIND!" to stdin. The program must read those 5 bytes and echo
        // them straight back to stdout, exiting with read's return value (5) — proving
        // the read syscall actually fired and landed the right bytes in the buffer.
        let (stdout, code) =
            run_capture_stdin(&elf, "mind_native_read_exe", b"MIND!").expect("the read ELF runs");
        assert_eq!(
            stdout, b"MIND!",
            "__mind_read(0, buf, 8, -1) must read stdin and the echo must print it back"
        );
        assert_eq!(
            code, 5,
            "read(2) returns the byte count (5 = len(\"MIND!\")); main returns it as the exit code"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_a_program_that_writes_bytes_to_stdout_via_mind_write() {
        // The raw write(2) syscall path (NO libc). Front-end-lowered (not a
        // hand-built IR toy): alloc a buffer, store the bytes "Hi!" (0x48 0x69
        // 0x21) one __mind_store_i8 at a time, then __mind_write(fd=1, buf, count=3,
        // offset=-1) — the exact shape print_bytes (std/io.mind:69) emits — and exit
        // with write's return (bytes written = 3).
        let src = "fn main() -> i64 { \
                   let addr: i64 = __mind_alloc(8); \
                   let r0: i64 = __mind_store_i8(addr, 72); \
                   let r1: i64 = __mind_store_i8(addr + 1, 105); \
                   let r2: i64 = __mind_store_i8(addr + 2, 33); \
                   let n: i64 = __mind_write(1, addr, 3, 0 - 1); \
                   return n; }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers the __mind_write program");

        // Byte-identity (the wedge): the write codegen is a pure function of the IR
        // (frame-slot movs + fixed syscall-number bytes + `syscall`, no host-varying
        // bytes) — two builds of the same IR must be byte-for-byte identical.
        assert_eq!(
            elf,
            compile_to_elf(&ir).expect("lowers"),
            "__mind_write ELF must be byte-identical (no host-varying bytes)"
        );

        // RUN it: capture stdout AND the exit code. The program must print exactly
        // the three bytes "Hi!" to fd 1, and exit with write's return value (3 bytes
        // written) — proving the syscall actually fired and wrote the right bytes.
        let (stdout, code) =
            run_capture(&elf, "mind_native_write_exe").expect("the __mind_write ELF must run");
        assert_eq!(
            stdout, b"Hi!",
            "__mind_write(1, buf, 3, -1) must print exactly the bytes 'Hi!' to stdout"
        );
        assert_eq!(
            code, 3,
            "write(2) returns the byte count (3); main returns it as the exit code"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_vec_growth_via_mind_realloc_preserving_old_contents() {
        // The grow-and-preserve path: __mind_realloc(addr, new_bytes). Mirrors the
        // std/vec.mind:85 vec_push doubling exactly — alloc a 4-element (cap 4)
        // backing store, fill it, then __mind_realloc it to 8 elements (the 5th
        // push's doubling). The realloc must COPY the original four i64s forward
        // into the new block (the old block is abandoned — the arena is no-free),
        // so reading element 0 back AFTER the grow proves the preservation. Writing
        // a 5th element into the grown block and summing with element 0 proves the
        // new region is usable and the copy did not clobber it.
        //
        // Front-end-lowered (not a hand-built IR toy): the realloc(addr, bytes) call
        // shape is exactly what std/vec.mind emits (Instr::Call __mind_realloc with
        // two args). Layout: cap4 = 32 bytes, cap8 = 64 bytes.
        let src = "fn main() -> i64 { \
                   let a0: i64 = __mind_alloc(32); \
                   let s0: i64 = __mind_store_i64(a0, 11); \
                   let s1: i64 = __mind_store_i64(a0 + 8, 22); \
                   let s2: i64 = __mind_store_i64(a0 + 16, 33); \
                   let s3: i64 = __mind_store_i64(a0 + 24, 4); \
                   let a1: i64 = __mind_realloc(a0, 64); \
                   let s4: i64 = __mind_store_i64(a1 + 32, 5); \
                   let e0: i64 = __mind_load_i64(a1); \
                   let e4: i64 = __mind_load_i64(a1 + 32); \
                   return e0 + e4; }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers the vec-growth program");

        // Byte-identity (the wedge): realloc codegen is a pure function of the IR
        // (absolute movabs + frame-slot moves + a fixed 8-byte-stride copy loop, no
        // host-varying bytes) — two builds of the same IR must be byte-for-byte
        // identical. The header change shifts base addresses but is itself constant.
        assert_eq!(
            elf,
            compile_to_elf(&ir).expect("lowers"),
            "vec-growth ELF must be byte-identical (realloc emits no host-varying bytes)"
        );

        // RUN it: element 0 (11) must survive the realloc copy into the grown block,
        // and the freshly-written element 4 (5) reads back from the new region.
        // 11 + 5 = 16 (well within a valid exit code). A realloc that lost the old
        // contents would read element 0 as 0 → exit(5); a copy that clobbered the
        // new region would not read element 4 back as 5.
        assert_eq!(
            run(&elf, "mind_native_vec_realloc_exe"),
            Some(16),
            "realloc must PRESERVE element 0 (11) + new element 4 (5) = 16"
        );

        // A NULL-addr realloc is a fresh allocation (the cap-0 → first-push path):
        // realloc(0, 24) === alloc(24). Store + read back proves the addr==0 branch
        // skips the copy and returns a usable fresh block.
        let src_null = "fn main() -> i64 { \
                        let a: i64 = __mind_realloc(0, 24); \
                        let s: i64 = __mind_store_i64(a + 8, 42); \
                        return __mind_load_i64(a + 8); }";
        let ir_null =
            crate::eval::lower::lower_to_ir(&crate::parser::parse(src_null).expect("parse"));
        let elf_null = compile_to_elf(&ir_null).expect("lowers");
        assert_eq!(
            elf_null,
            compile_to_elf(&ir_null).expect("lowers"),
            "NULL-realloc ELF must be byte-identical"
        );
        assert_eq!(
            run(&elf_null, "mind_native_realloc_null_exe"),
            Some(42),
            "realloc(0, n) must behave as a fresh alloc(n): store 42 reads back 42"
        );
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_recursive_fibonacci() {
        // Recursion exercises calls + value-`if` + arithmetic composing through the
        // native backend on a real front-end-lowered multi-function program.
        let src = "fn fib(n: i64) -> i64 { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } } \
                   fn main() -> i64 { fib(10) }";
        let module = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&module);
        let elf = compile_to_elf(&ir).expect("native-lowers recursive fib");
        assert_eq!(elf, compile_to_elf(&ir).expect("lowers"), "deterministic");
        assert_eq!(run(&elf, "mind_native_fib_exe"), Some(55), "fib(10) = 55");
    }

    #[test]
    #[cfg(feature = "std-surface")]
    fn lowers_bitwise_and_shift() {
        let cases = [
            (BinOp::BitAnd, 46, 26, 10), // 0b101110 & 0b011010 = 0b001010
            (BinOp::BitOr, 40, 2, 42),
            (BinOp::BitXor, 63, 21, 42), // 0b111111 ^ 0b010101 = 0b101010
            (BinOp::Shl, 21, 1, 42),
            (BinOp::Shr, 84, 1, 42), // arithmetic sar
        ];
        for (op, a, b, expected) in cases {
            let m = binop_module(op, a, b);
            let elf = compile_to_elf(&m).expect("lowers");
            assert_eq!(elf, compile_to_elf(&m).expect("lowers"), "deterministic");
            assert_eq!(
                run(&elf, "mind_native_bitshift_exe"),
                Some(expected),
                "{op:?}({a},{b})"
            );
        }
    }

    /// A loop-carried accumulator updated via a CHAINED expression
    /// (`acc = acc * 10 + 7`) — two BinOps reading the loop variable. Exercises the
    /// init-copy + read-remap on a value that flows through intermediates before
    /// landing back in its own slot. (Kept ≤ 255 so the exit code is exact:
    /// 3 iterations of `acc*10+7` from 0 give 777, whose low byte is 9 — so use a
    /// 2-iteration `div` that yields 77.)
    #[test]
    #[cfg(feature = "std-surface")]
    fn loop_carried_chained_accumulator() {
        let src = "fn r() -> i64 { let mut div: i64 = 10; let mut acc: i64 = 0; \
                     while div > 0 { acc = acc * 10 + 7; div = div / 10; } acc } \
                   fn main() -> i64 { r() }";
        let m = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&m);
        let elf = compile_to_elf(&ir).expect("lowers");
        assert_eq!(elf, compile_to_elf(&ir).expect("lowers"), "deterministic");
        // div = 10 → 2 iterations (10, 1): acc = 0→7→77.
        assert_eq!(run(&elf, "mind_native_accchain_exe"), Some(77), "acc*10+7 twice");
    }

    /// Two SEQUENTIAL `while` loops that carry the same variable: loop 1 computes a
    /// power-of-10 divisor, loop 2 walks it back down counting iterations. The
    /// loop-carried `div` is live across BOTH loops (loop 1's exit feeds loop 2's
    /// init), and `n` stays live as a value read inside loop 2's body even though
    /// loop 1 mutates a sibling derived from it. This is the `string_push_i64`
    /// digit-emit shape that silently miscompiled to 0 before the dedicated
    /// loop-slot + init-copy fix (the old direct init-alias clobbered the live
    /// value through a shared SSA id).
    #[test]
    #[cfg(feature = "std-surface")]
    fn sequential_loops_carry_shared_variable() {
        // f(n) = number of decimal digits in n (1 for n<10, 3 for 100..999, …).
        let src = "fn f(n: i64) -> i64 { \
                     let mut div: i64 = 1; let mut tmp: i64 = n; \
                     while tmp >= 10 { div = div * 10; tmp = tmp / 10; } \
                     let mut count: i64 = 0; \
                     while div > 0 { count = count + 1; div = div / 10; } \
                     count } \
                   fn main() -> i64 { f(SUBST) }";
        for (input, expect) in [(5i32, 1i32), (16, 2), (100, 3), (40, 2)] {
            let m = crate::parser::parse(&src.replace("SUBST", &input.to_string()))
                .expect("parse");
            let ir = crate::eval::lower::lower_to_ir(&m);
            let elf = compile_to_elf(&ir).expect("lowers");
            assert_eq!(elf, compile_to_elf(&ir).expect("lowers"), "deterministic");
            assert_eq!(
                run(&elf, "mind_native_seqloops_exe"),
                Some(expect),
                "digit-count f({input})"
            );
        }
    }

    /// A `while` body that READS one variable while MUTATING another that was
    /// initialised from it — the exact `string_push_i64` hazard: `nn` holds the
    /// number, the divisor loop mutates `tmp` (`let mut tmp = nn`, same SSA id as
    /// `nn`), then a second loop reads `nn` to extract each digit. The dedicated
    /// loop-slot + init-copy must keep `nn` intact across the first loop.
    #[test]
    #[cfg(feature = "std-surface")]
    fn loop_mutates_sibling_without_clobbering_live_value() {
        // Reconstruct the number from its decimal digits, MSB-first — the body of
        // string_push_i64's emit loop, returning the value if correct.
        let src = "fn rebuild(n: i64) -> i64 { \
                     let mut nn: i64 = n; \
                     let mut div: i64 = 1; let mut tmp: i64 = nn; \
                     while tmp >= 10 { div = div * 10; tmp = tmp / 10; } \
                     let mut acc: i64 = 0; \
                     while div > 0 { let digit: i64 = (nn / div) % 10; acc = acc * 10 + digit; div = div / 10; } \
                     acc } \
                   fn main() -> i64 { rebuild(SUBST) }";
        // `run` returns the process EXIT CODE — only the low 8 bits survive, so
        // keep every expected value ≤ 255 (the loop logic is width-independent).
        for input in [7i32, 16, 42, 90, 201] {
            let m = crate::parser::parse(&src.replace("SUBST", &input.to_string()))
                .expect("parse");
            let ir = crate::eval::lower::lower_to_ir(&m);
            let elf = compile_to_elf(&ir).expect("lowers");
            assert_eq!(
                run(&elf, "mind_native_rebuild_exe"),
                Some(input),
                "digit-rebuild rebuild({input}) == {input}"
            );
        }
    }

    /// Name-shadowing: when the same function name is DEFINED twice (the native
    /// path bundles the whole stdlib ahead of the user file, so a user
    /// `fn bytes_eq` collides with std/toml.mind's own `bytes_eq` — a DIFFERENT
    /// signature), a `call` must bind to the LAST (user) definition, never the
    /// earlier bundled one. Before the dedup fix, `link` resolved to the first
    /// match: a 6-arg `call` jumped into the 3-arg std body, which then did a
    /// 1-byte load through a garbage register and segfaulted — a miscompile that
    /// lowered and "exited 0" on the no-`main` library form yet crashed at run.
    #[test]
    #[cfg(feature = "std-surface")]
    fn user_function_shadows_a_bundled_std_name() {
        // Two definitions of `clash`: the FIRST (bundled-style, 3 params) is a
        // decoy with a divergent ABI; the SECOND (user, 1 param) must win.
        let src = "fn clash(a: i64, b: i64, c: i64) -> i64 { a + b + c } \
                   fn clash(x: i64) -> i64 { x + 100 } \
                   fn main() -> i64 { clash(7) }";
        let m = crate::parser::parse(src).expect("parse");
        let ir = crate::eval::lower::lower_to_ir(&m);
        let elf = compile_to_elf(&ir).expect("lowers");
        assert_eq!(elf, compile_to_elf(&ir).expect("lowers"), "deterministic");
        // The LAST `clash` (x + 100) wins: clash(7) = 107. The decoy would
        // mis-read args and yield garbage / crash.
        assert_eq!(
            run(&elf, "mind_native_shadow_exe"),
            Some(107),
            "user clash(x)=x+100 shadows the 3-arg decoy"
        );
    }
}
