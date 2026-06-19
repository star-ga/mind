// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! MIND-native backend — **proof-of-concept** (architecture go/no-go).
//!
//! This is the smallest possible demonstration of the wedge's logical end-state:
//! lower a MIND-style IR straight to **x86-64 machine code → a runnable ELF
//! executable, with ZERO LLVM / MLIR / clang / assembler** in the loop. It does
//! NOT replace the production `mic@3 → MLIR → ELF` path (that stays); it proves
//! the *architecture* — that MIND can own the entire `IR → bytes` pipeline.
//!
//! ## Why this matters — determinism-by-CONSTRUCTION
//!
//! Today cross-substrate byte-identity is a **post-hoc empirical check**: mic@3
//! is deterministic, but the bytes are handed to `clang -O3`, whose register
//! allocator and scheduler are version/host sensitive, and the gate *verifies* a
//! pinned hash matched. A MIND-native backend with a **fixed instruction encoder
//! and a fixed value→slot mapping** makes determinism *structural*: the same IR
//! literally **cannot** produce different bytes, because there is no
//! nondeterministic stage left. This test asserts exactly that — two independent
//! emissions of the same IR are byte-identical.
//!
//! Scope (honest): a stack-machine codegen over a 3-op mini-IR (Const / Add /
//! Output→exit), a 120-byte static ELF64 writer, no register allocation, no
//! control flow, no kernels. It is a *go/no-go signal on the thesis*, not a
//! backend. The real native backend is multi-month (regalloc) to multi-year
//! (deterministic SIMD kernels) — see the native-backend design map.
//!
//! Gate: `cargo test --test native_backend_poc` (Linux x86-64 only).

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

/// A 3-op mini-IR — the smallest slice of the MIND IR that proves the path.
/// (The real surface is 43 mic@3 opcodes; instruction selection is O(1) per op.)
#[derive(Clone, Copy)]
enum Op {
    /// Push an i64 constant.
    Const(i32),
    /// Pop two, push their sum (two's-complement, exact mod 2^64).
    Add,
    /// Pop the top and `exit()` with it — the module result.
    Output,
}

/// Stack-machine instruction selection: each op emits a fixed x86-64 byte
/// sequence. No register allocation, no scheduling — so the output is a pure
/// function of the IR (determinism by construction). SysV: result→rdi, exit via
/// syscall 60.
fn codegen(ir: &[Op]) -> Vec<u8> {
    let mut c = Vec::new();
    for op in ir {
        match op {
            Op::Const(v) => {
                // mov rax, imm32 (sign-extended) ; push rax
                c.extend_from_slice(&[0x48, 0xC7, 0xC0]);
                c.extend_from_slice(&v.to_le_bytes());
                c.push(0x50);
            }
            Op::Add => {
                // pop rcx ; pop rax ; add rax, rcx ; push rax
                c.extend_from_slice(&[0x59, 0x58, 0x48, 0x01, 0xC8, 0x50]);
            }
            Op::Output => {
                // pop rdi ; mov rax, 60 ; syscall   (exit(rdi))
                c.extend_from_slice(&[0x5F, 0x48, 0xC7, 0xC0, 0x3C, 0x00, 0x00, 0x00, 0x0F, 0x05]);
            }
        }
    }
    c
}

/// Minimal, deterministic static ELF64 executable: 64-byte ELF header + one
/// 56-byte PT_LOAD program header + the code. No sections, no symbols, no
/// timestamps, no padding — every byte is a pure function of `code`.
fn write_elf(code: &[u8]) -> Vec<u8> {
    const LOAD_ADDR: u64 = 0x40_0000;
    const HDRS: u64 = 64 + 56; // ehdr + one phdr
    let entry = LOAD_ADDR + HDRS;
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

#[test]
fn native_codegen_emits_a_runnable_deterministic_elf_with_zero_llvm() {
    // mini-IR: 40 + 2, exit with the result.  (Const, Const, Add, Output)
    let ir = [Op::Const(40), Op::Const(2), Op::Add, Op::Output];

    // 1) DETERMINISM BY CONSTRUCTION: two independent emissions, byte-identical.
    let a = write_elf(&codegen(&ir));
    let b = write_elf(&codegen(&ir));
    assert_eq!(
        a, b,
        "native codegen must be byte-identical across emissions"
    );

    // 2) RUNNABLE: write the ELF, exec it, assert it computed 40+2 = 42.
    let path = std::env::temp_dir().join("mind_native_poc_exe");
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
        "the MIND-native ELF must compute 40+2 and exit(42)"
    );

    // 3) Sanity: the artifact is a real ELF, and tiny (no toolchain bloat).
    assert_eq!(&a[0..4], &[0x7F, b'E', b'L', b'F'], "valid ELF magic");
    assert!(a.len() < 256, "minimal static ELF, no toolchain padding");

    let _ = std::fs::remove_file(&path);
}
