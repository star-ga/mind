// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! MIND-native backend — **proof-of-concept #3: consume the REAL IR**.
//!
//! PoC #1 ([`native_backend_poc`]) proved `IR → x86-64 → ELF` for a toy stack
//! machine; PoC #2 ([`native_backend_call_poc`]) added System-V calls + a linker.
//! Both used *hand-rolled* `Op`/`Func` enums. This one closes that gap: it lowers
//! the **actual `libmind::ir::Instr`** the existing front-end emits — `FnDef`,
//! `Param`, `ConstI64`, `BinOp`, `Call`, `Return` — straight to a runnable static
//! ELF64 with **ZERO LLVM / MLIR / clang / assembler / linker**. The native
//! backend eats the *same* IR the MLIR path does.
//!
//! It also retires the next named risk after "multi-function + call": a real
//! **stack frame** and a **`ValueId → slot` mem-to-mem model**. Instead of
//! assuming args are already in registers, every SSA value gets a frame slot;
//! params are spilled from their System-V arg registers on entry, each `BinOp`
//! loads its operands from slots and stores its result, and a `Call` marshals its
//! arg slots into `rdi/rsi/rdx/...` before the call. This is the regalloc-free
//! lowering the real backend starts from — correct first, fast later (a
//! deterministic linear-scan allocator is the months-out follow-up).
//!
//! Program lowered (built directly as `Vec<Instr>`):
//! ```text
//! fn f(a: i64, b: i64, c: i64) -> i64 { a + b + c }
//! fn main() -> i64 { f(40, 1, 1) }          // exit(f(40,1,1)) == exit(42)
//! ```
//!
//! ## Determinism-by-construction (the wedge)
//!
//! Slot assignment is first-appearance order, encoders are fixed, layout order is
//! fixed — so the image is a pure function of the IR. Asserted: two independent
//! lowerings of the same `Vec<Instr>` are byte-identical, and it runs to
//! `exit(42)`.
//!
//! Honest scope: the scalar-i64 `Instr` subset (Add/Sub/Mul; Div/Mod would need
//! `cqo`/`idiv`), ≤6 args (System-V integer registers, no stack-passed args), no
//! spill-reload optimization, intra-module calls only (computed displacement, not
//! a true ELF relocation). A go/no-go on *real-IR consumption*, not a backend.
//!
//! Gate: `cargo test --test native_backend_ir_poc` (Linux x86-64 only).

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

use libmind::ir::{BinOp, Instr, ValueId};

/// System-V integer argument registers, in order. `(modrm_reg_field, needs_rex_b)`.
/// rdi, rsi, rdx, rcx are low regs; r8/r9 set REX.{R,B} when used as reg/rm.
const ARG_REGS: [(u8, bool); 6] = [
    (7, false), // rdi
    (6, false), // rsi
    (2, false), // rdx
    (1, false), // rcx
    (0, true),  // r8
    (1, true),  // r9
];

/// One lowered function plus its still-unresolved call sites.
struct Func {
    name: String,
    code: Vec<u8>,
    /// `(offset_of_rel32_within_code, callee_name)`.
    calls: Vec<(usize, String)>,
}

/// `mov [rbp+disp32], rax` — store rax into a frame slot.
fn store_rax(code: &mut Vec<u8>, disp: i32) {
    code.extend_from_slice(&[0x48, 0x89, 0x85]);
    code.extend_from_slice(&disp.to_le_bytes());
}

/// `mov rax, [rbp+disp32]` — load a frame slot into rax.
fn load_rax(code: &mut Vec<u8>, disp: i32) {
    code.extend_from_slice(&[0x48, 0x8B, 0x85]);
    code.extend_from_slice(&disp.to_le_bytes());
}

/// `<op> rax, [rbp+disp32]` for the supported arithmetic BinOps.
fn arith_rax_mem(code: &mut Vec<u8>, op: &BinOp, disp: i32) {
    let opbytes: &[u8] = match op {
        BinOp::Add => &[0x48, 0x03, 0x85],       // add rax, r/m64
        BinOp::Sub => &[0x48, 0x2B, 0x85],       // sub rax, r/m64
        BinOp::Mul => &[0x48, 0x0F, 0xAF, 0x85], // imul rax, r/m64
        other => panic!("native PoC supports Add/Sub/Mul only, got {other:?}"),
    };
    code.extend_from_slice(opbytes);
    code.extend_from_slice(&disp.to_le_bytes());
}

/// `mov <argreg>, [rbp+disp32]` — marshal a slot value into a System-V arg reg.
fn load_argreg(code: &mut Vec<u8>, arg_index: usize, disp: i32) {
    let (reg, ext) = ARG_REGS[arg_index];
    let rex = 0x48 | if ext { 0x04 } else { 0x00 }; // REX.W (+REX.R for r8/r9)
    let modrm = 0x80 | ((reg & 7) << 3) | 0x05; // mod=10, reg=arg, rm=rbp
    code.push(rex);
    code.push(0x8B);
    code.push(modrm);
    code.extend_from_slice(&disp.to_le_bytes());
}

/// `mov [rbp+disp32], <argreg>` — spill an incoming param register to its slot.
fn store_argreg(code: &mut Vec<u8>, arg_index: usize, disp: i32) {
    let (reg, ext) = ARG_REGS[arg_index];
    let rex = 0x48 | if ext { 0x04 } else { 0x00 };
    let modrm = 0x80 | ((reg & 7) << 3) | 0x05;
    code.push(rex);
    code.push(0x89);
    code.push(modrm);
    code.extend_from_slice(&disp.to_le_bytes());
}

/// Disp from rbp for slot `i`: slots grow downward, `rbp - 8*(i+1)`.
fn slot_disp(i: usize) -> i32 {
    -((8 * (i + 1)) as i32)
}

/// Lower one `FnDef` body to machine code. `is_entry` swaps the final `Return`
/// for an `exit` syscall (this is the program's `_start`).
fn lower_fn(name: &str, body: &[Instr], is_entry: bool) -> Func {
    // Pass 1: assign every defined SSA value a frame slot (first-appearance order).
    let mut slot: HashMap<ValueId, usize> = HashMap::new();
    let define = |id: ValueId, slot: &mut HashMap<ValueId, usize>| {
        let n = slot.len();
        slot.entry(id).or_insert(n);
    };
    for ins in body {
        match ins {
            Instr::Param { dst, .. } => define(*dst, &mut slot),
            Instr::ConstI64(dst, _) => define(*dst, &mut slot),
            Instr::BinOp { dst, .. } => define(*dst, &mut slot),
            Instr::Call { dst, .. } => define(*dst, &mut slot),
            _ => {}
        }
    }
    let frame = (8 * slot.len()).next_multiple_of(16) as i32; // keep rsp 16-aligned

    let disp = |id: &ValueId| slot_disp(slot[id]);

    // Prologue: push rbp ; mov rbp, rsp ; sub rsp, frame
    let mut code = vec![0x55, 0x48, 0x89, 0xE5];
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&frame.to_le_bytes());

    let mut calls = Vec::new();

    // Pass 2: emit.
    for ins in body {
        match ins {
            Instr::Param { dst, index, .. } => store_argreg(&mut code, *index, disp(dst)),
            Instr::ConstI64(dst, v) => {
                code.extend_from_slice(&[0x48, 0xB8]); // movabs rax, imm64
                code.extend_from_slice(&v.to_le_bytes());
                store_rax(&mut code, disp(dst));
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                load_rax(&mut code, disp(lhs));
                arith_rax_mem(&mut code, op, disp(rhs));
                store_rax(&mut code, disp(dst));
            }
            Instr::Call {
                dst, name, args, ..
            } => {
                for (i, a) in args.iter().enumerate() {
                    load_argreg(&mut code, i, disp(a));
                }
                code.push(0xE8); // call rel32 (placeholder, linker-patched)
                let rel_off = code.len();
                code.extend_from_slice(&[0, 0, 0, 0]);
                calls.push((rel_off, name.clone()));
                store_rax(&mut code, disp(dst)); // result in rax -> slot
            }
            Instr::Return { value } => {
                if let Some(v) = value {
                    load_rax(&mut code, disp(v));
                }
                if is_entry {
                    // exit(rax): mov rdi, rax ; mov rax, 60 ; syscall
                    code.extend_from_slice(&[0x48, 0x89, 0xC7]);
                    code.extend_from_slice(&[0x48, 0xC7, 0xC0, 0x3C, 0x00, 0x00, 0x00]);
                    code.extend_from_slice(&[0x0F, 0x05]);
                } else {
                    // epilogue: mov rsp, rbp ; pop rbp ; ret
                    code.extend_from_slice(&[0x48, 0x89, 0xEC, 0x5D, 0xC3]);
                }
            }
            other => panic!("native PoC: unsupported instr {other:?}"),
        }
    }

    Func {
        name: name.to_string(),
        code,
        calls,
    }
}

/// Minimal linker: fixed layout (entry first), resolve each call's PC-relative
/// `rel32` from the layout. Returns `(image, entry_offset)`.
fn link(funcs: &[Func]) -> (Vec<u8>, u64) {
    let mut starts = Vec::with_capacity(funcs.len());
    let mut cursor = 0usize;
    for f in funcs {
        starts.push(cursor);
        cursor += f.code.len();
    }
    let index_of = |name: &str| {
        funcs
            .iter()
            .position(|f| f.name == name)
            .expect("callee exists")
    };

    let mut image = Vec::with_capacity(cursor);
    for (i, f) in funcs.iter().enumerate() {
        let mut code = f.code.clone();
        for (rel_off, callee) in &f.calls {
            let site = starts[i] + rel_off;
            let next = site + 4;
            let disp = starts[index_of(callee)] as i64 - next as i64;
            code[*rel_off..*rel_off + 4].copy_from_slice(&(disp as i32).to_le_bytes());
        }
        image.extend_from_slice(&code);
    }
    (image, starts[0] as u64) // entry (main) is laid out first
}

/// Minimal deterministic static ELF64 (same structure as the other PoCs).
fn write_elf(code: &[u8], entry_off: u64) -> Vec<u8> {
    const LOAD_ADDR: u64 = 0x40_0000;
    const HDRS: u64 = 64 + 56;
    let entry = LOAD_ADDR + HDRS + entry_off;
    let filesz = HDRS + code.len() as u64;

    let mut e = Vec::with_capacity(filesz as usize);
    e.extend_from_slice(&[0x7F, b'E', b'L', b'F', 2, 1, 1, 0]);
    e.extend_from_slice(&[0u8; 8]);
    e.extend_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    e.extend_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    e.extend_from_slice(&1u32.to_le_bytes());
    e.extend_from_slice(&entry.to_le_bytes());
    e.extend_from_slice(&64u64.to_le_bytes()); // e_phoff
    e.extend_from_slice(&0u64.to_le_bytes());
    e.extend_from_slice(&0u32.to_le_bytes());
    e.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    e.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
    e.extend_from_slice(&1u16.to_le_bytes());
    e.extend_from_slice(&0u16.to_le_bytes());
    e.extend_from_slice(&0u16.to_le_bytes());
    e.extend_from_slice(&0u16.to_le_bytes());
    e.extend_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    e.extend_from_slice(&5u32.to_le_bytes()); // R+X
    e.extend_from_slice(&0u64.to_le_bytes());
    e.extend_from_slice(&LOAD_ADDR.to_le_bytes());
    e.extend_from_slice(&LOAD_ADDR.to_le_bytes());
    e.extend_from_slice(&filesz.to_le_bytes());
    e.extend_from_slice(&filesz.to_le_bytes());
    e.extend_from_slice(&0x1000u64.to_le_bytes());
    e.extend_from_slice(code);
    e
}

/// Build the real `Vec<Instr>` program, then lower → link → ELF.
fn compile() -> Vec<u8> {
    let v = ValueId; // terse constructor

    // fn f(a, b, c) -> i64 { a + b + c }
    let f_body = vec![
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
        }, // a+b
        Instr::BinOp {
            dst: v(14),
            op: BinOp::Add,
            lhs: v(13),
            rhs: v(12),
        }, // (a+b)+c
        Instr::Return { value: Some(v(14)) },
    ];

    // fn main() -> i64 { f(40, 1, 1) }
    let main_body = vec![
        Instr::ConstI64(v(20), 40),
        Instr::ConstI64(v(21), 1),
        Instr::ConstI64(v(22), 1),
        Instr::Call {
            dst: v(23),
            name: "f".into(),
            args: vec![v(20), v(21), v(22)],
        },
        Instr::Return { value: Some(v(23)) },
    ];

    // Layout order: entry (main) first, then f.
    let funcs = [
        lower_fn("main", &main_body, true),
        lower_fn("f", &f_body, false),
    ];
    let (code, entry_off) = link(&funcs);
    write_elf(&code, entry_off)
}

#[test]
fn native_backend_lowers_real_ir_to_a_runnable_deterministic_elf() {
    // 1) DETERMINISM BY CONSTRUCTION — same Vec<Instr> -> byte-identical image.
    let a = compile();
    let b = compile();
    assert_eq!(
        a, b,
        "lowering real IR must be byte-identical across runs (the wedge)"
    );

    // 2) RUNNABLE — exec it, assert f(40,1,1) = 42.
    let path = std::env::temp_dir().join("mind_native_ir_poc_exe");
    {
        let mut fh = std::fs::File::create(&path).expect("create exe");
        fh.write_all(&a).expect("write exe");
        let mut perms = fh.metadata().unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod +x");
    }
    let status = Command::new(&path).status().expect("exec native ELF");
    assert_eq!(
        status.code(),
        Some(42),
        "the native-lowered real-IR ELF must compute f(40,1,1) and exit(42)"
    );

    let _ = std::fs::remove_file(&path);
}
