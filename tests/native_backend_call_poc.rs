// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! MIND-native backend — **proof-of-concept #2: multi-function + internal call**.
//!
//! [`native_backend_poc`] proved `IR → x86-64 → runnable deterministic ELF` for a
//! single stack-machine function with no ABI. This retires the next named risk in
//! the native-backend design map: **a real System-V calling convention and a
//! minimal linker pass**. It compiles a two-function program —
//!
//! ```text
//! fn add(a: i64, b: i64) -> i64 { a + b }   // args in rdi/rsi, result in rax
//! fn main() -> i64 { add(40, 2) }            // exit(add(40, 2))
//! ```
//!
//! — straight to a runnable static ELF64, with **ZERO LLVM / MLIR / clang /
//! assembler / linker**. It proves the three things a toy stack machine cannot:
//!   1. **System-V ABI** — args passed in `rdi`/`rsi`, return value in `rax`.
//!   2. **`call`/`ret`** — a real function prologue-less leaf call.
//!   3. **A linker pass** — two functions are laid out, and the `call`'s
//!      PC-relative `rel32` displacement is *computed from the layout*, not
//!      hardcoded. (This is intra-module displacement resolution for a
//!      self-contained executable; true ELF relocations — `R_X86_64_PC32` in a
//!      `.o` for an external linker — are a separate, later milestone needed only
//!      for separate compilation / linking against libc. We are honest about that
//!      distinction: nothing here emits or needs a relocation record.)
//!
//! ## Determinism-by-construction (the wedge), extended to linking
//!
//! The layout order is fixed (`main` then `add`), the encoders are fixed, and the
//! displacement is a pure function of the two function lengths — so the linked
//! image is a pure function of the IR. The test asserts two independent
//! compile+link passes are byte-identical. Determinism survives the linker; it is
//! not a property of one isolated function but of the whole pipeline.
//!
//! Scope (honest): fixed-shape encoders for a five-instruction subset, two
//! functions, one call, no register allocation (args already in the right
//! registers), no stack frame, no spills. A go/no-go on multi-function scaling,
//! not a backend.
//!
//! Gate: `cargo test --test native_backend_call_poc` (Linux x86-64 only).

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

/// A function's machine code plus the call sites that still need their `rel32`
/// displacement patched once every function's address is known.
struct Func {
    code: Vec<u8>,
    /// `(offset_of_the_rel32_field_within_this_func, callee_index)`.
    calls: Vec<(usize, usize)>,
}

/// `fn add(a, b) -> a + b`: System-V leaf, args already in `rdi`/`rsi`, result in
/// `rax`. No prologue/epilogue — it touches no stack and no callee-saved regs.
fn func_add() -> Func {
    Func {
        code: vec![
            0x48, 0x89, 0xF8, // mov rax, rdi
            0x48, 0x01, 0xF0, // add rax, rsi
            0xC3, // ret
        ],
        calls: vec![],
    }
}

/// `fn main() -> exit(add(40, 2))`: load the two args, `call add`, move the result
/// into `rdi`, and `exit` via syscall 60. The `call`'s 4-byte displacement is left
/// zero here and filled in by `link()`.
fn func_main() -> Func {
    let mut code = vec![
        0x48, 0xC7, 0xC7, 0x28, 0x00, 0x00, 0x00, // mov rdi, 40
        0x48, 0xC7, 0xC6, 0x02, 0x00, 0x00, 0x00, // mov rsi, 2
        0xE8, 0x00, 0x00, 0x00, 0x00, // call add (rel32 placeholder @ +15)
        0x48, 0x89, 0xC7, // mov rdi, rax
        0x48, 0xC7, 0xC0, 0x3C, 0x00, 0x00, 0x00, // mov rax, 60 (exit)
        0x0F, 0x05, // syscall
    ];
    // Belt-and-suspenders: the placeholder is the 4 bytes after the 0xE8 at +14.
    debug_assert_eq!(code[14], 0xE8);
    for b in &mut code[15..19] {
        *b = 0;
    }
    Func {
        code,
        calls: vec![(15, 1)], // rel32 field at +15 targets function index 1 (add)
    }
}

/// The minimal **linker**: lay functions out head-to-tail in a fixed order, then
/// resolve every call's PC-relative `rel32 = callee_start - (call_site + 4)`.
/// Pure function of the inputs → deterministic. Returns `(image, entry_offset)`.
fn link(funcs: &[Func], entry_index: usize) -> (Vec<u8>, u64) {
    // Pass 1: section offset of each function (fixed layout order).
    let mut starts = Vec::with_capacity(funcs.len());
    let mut cursor = 0usize;
    for f in funcs {
        starts.push(cursor);
        cursor += f.code.len();
    }

    // Pass 2: emit, patching each call's rel32 from the now-known layout.
    let mut image = Vec::with_capacity(cursor);
    for (i, f) in funcs.iter().enumerate() {
        let mut code = f.code.clone();
        for &(rel_off, callee) in &f.calls {
            let site = starts[i] + rel_off; // absolute offset of the rel32 field
            let next_insn = site + 4; // x86 rel32 is relative to the *next* insn
            let disp = starts[callee] as i64 - next_insn as i64;
            code[rel_off..rel_off + 4].copy_from_slice(&(disp as i32).to_le_bytes());
        }
        image.extend_from_slice(&code);
    }
    (image, starts[entry_index] as u64)
}

/// Minimal, deterministic static ELF64 executable (identical structure to the
/// single-function PoC, with an explicit entry offset into the code).
fn write_elf(code: &[u8], entry_off: u64) -> Vec<u8> {
    const LOAD_ADDR: u64 = 0x40_0000;
    const HDRS: u64 = 64 + 56; // ehdr + one phdr
    let entry = LOAD_ADDR + HDRS + entry_off;
    let filesz = HDRS + code.len() as u64;

    let mut e = Vec::with_capacity(filesz as usize);
    // --- ELF header (Elf64_Ehdr) ---
    e.extend_from_slice(&[0x7F, b'E', b'L', b'F', 2, 1, 1, 0]); // magic, 64-bit, LE, v1, SysV
    e.extend_from_slice(&[0u8; 8]); // e_ident padding
    e.extend_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
    e.extend_from_slice(&62u16.to_le_bytes()); // e_machine = EM_X86_64
    e.extend_from_slice(&1u32.to_le_bytes()); // e_version
    e.extend_from_slice(&entry.to_le_bytes()); // e_entry
    e.extend_from_slice(&64u64.to_le_bytes()); // e_phoff
    e.extend_from_slice(&0u64.to_le_bytes()); // e_shoff
    e.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    e.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    e.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
    e.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
    // --- Program header (Elf64_Phdr): PT_LOAD, R+X ---
    e.extend_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
    e.extend_from_slice(&5u32.to_le_bytes()); // p_flags = R(4)+X(1)
    e.extend_from_slice(&0u64.to_le_bytes()); // p_offset
    e.extend_from_slice(&LOAD_ADDR.to_le_bytes()); // p_vaddr
    e.extend_from_slice(&LOAD_ADDR.to_le_bytes()); // p_paddr
    e.extend_from_slice(&filesz.to_le_bytes()); // p_filesz
    e.extend_from_slice(&filesz.to_le_bytes()); // p_memsz
    e.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align
    // --- Code ---
    e.extend_from_slice(code);
    e
}

/// Compile+link the two-function program to a finished ELF image.
fn build() -> Vec<u8> {
    // Layout order is fixed: main (index 0, the entry) then add (index 1).
    let funcs = [func_main(), func_add()];
    let (code, entry_off) = link(&funcs, 0);
    write_elf(&code, entry_off)
}

#[test]
fn native_two_function_call_links_and_runs_deterministically() {
    // 1) DETERMINISM BY CONSTRUCTION — survives the linker pass.
    let a = build();
    let b = build();
    assert_eq!(
        a, b,
        "compile+link must be byte-identical across runs (the wedge, through linking)"
    );

    // 2) The linker actually resolved the call: rel32 must be the computed 12,
    //    not the zero placeholder. (main is 31 bytes; call site rel32 @ +15, next
    //    insn @ +19; add starts @ +31; 31 - 19 = 12.)
    assert_eq!(
        &a[120 + 15..120 + 19], // 120 = ehdr(64) + phdr(56); +15 = rel32 field
        &12i32.to_le_bytes(),
        "the linker must patch `call add` with the layout-derived displacement"
    );

    // 3) RUNNABLE — exec it, assert it computed add(40, 2) = 42.
    let path = std::env::temp_dir().join("mind_native_call_poc_exe");
    {
        let mut f = std::fs::File::create(&path).expect("create exe");
        f.write_all(&a).expect("write exe");
        let mut perms = f.metadata().unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod +x");
    }
    let status = Command::new(&path).status().expect("exec native ELF");
    assert_eq!(
        status.code(),
        Some(42),
        "the linked two-function ELF must compute add(40, 2) and exit(42)"
    );

    let _ = std::fs::remove_file(&path);
}
