// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Experimental **MIND-native backend**: `IRModule → x86-64 → static ELF64`, with
//! **zero LLVM / MLIR / clang / assembler / external linker**.
//!
//! This is the production-shaped home of the codegen proven by the three
//! `tests/native_backend_*_poc.rs` proofs-of-concept. It is **opt-in**
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
//! not `-O3`; it is *determinism guaranteed by the codegen's shape*.
//!
//! ## Scope (honest) — the scalar-i64 vertical slice
//!
//! Supports `FnDef` / `Param` / `ConstI64` / `BinOp{Add,Sub,Mul}` / `Call` /
//! `Return`, ≤6 integer args (System-V registers), a regalloc-free mem-to-mem
//! frame model, and intra-module calls (computed PC-relative displacement). The
//! entry point is the `FnDef` named `main`; its `Return` becomes an `exit`
//! syscall.
//!
//! deferred: control flow (If/While → cmp/jcc with intra-fn label patching),
//!   Div/Mod (cqo/idiv), >6 args (stack-passed), register allocation (currently
//!   every SSA value is a frame slot), true ELF relocations + sections + symbols
//!   (for separate compilation / libc linking), float + tensor + SIMD kernels.
//!   upgrade path: extend `lower_fn`'s match arm-by-arm; add a deterministic
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
        other => return Err(NativeError::Unsupported(format!("BinOp {other:?}"))),
    };
    code.extend_from_slice(opbytes);
    code.extend_from_slice(&disp.to_le_bytes());
    Ok(())
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

/// Assign every value defined in `body` a frame slot, in first-appearance order.
fn assign_slots(body: &[Instr]) -> HashMap<ValueId, usize> {
    let mut slot = HashMap::new();
    for ins in body {
        let dst = match ins {
            Instr::Param { dst, .. } => Some(*dst),
            Instr::ConstI64(dst, _) => Some(*dst),
            Instr::BinOp { dst, .. } => Some(*dst),
            Instr::Call { dst, .. } => Some(*dst),
            _ => None,
        };
        if let Some(d) = dst {
            let n = slot.len();
            slot.entry(d).or_insert(n);
        }
    }
    slot
}

/// Lower one function body to machine code. `is_entry` swaps the final `Return`
/// for an `exit` syscall (the program's `_start`).
fn lower_fn(name: &str, body: &[Instr], is_entry: bool) -> Result<Func, NativeError> {
    let slot = assign_slots(body);
    let disp = |id: &ValueId| -> Result<i32, NativeError> {
        slot.get(id)
            .map(|&i| slot_disp(i))
            .ok_or_else(|| NativeError::Unsupported(format!("use of undefined value {id}")))
    };
    let frame = (8 * slot.len()).next_multiple_of(16) as i32; // keep rsp 16-aligned

    // Prologue: push rbp ; mov rbp, rsp ; sub rsp, frame
    let mut code = vec![0x55, 0x48, 0x89, 0xE5, 0x48, 0x81, 0xEC];
    code.extend_from_slice(&frame.to_le_bytes());

    let mut calls = Vec::new();
    for ins in body {
        match ins {
            Instr::Param { dst, index, .. } => store_argreg(&mut code, *index, disp(dst)?)?,
            Instr::ConstI64(dst, v) => {
                code.extend_from_slice(&[0x48, 0xB8]); // movabs rax, imm64
                code.extend_from_slice(&v.to_le_bytes());
                store_rax(&mut code, disp(dst)?);
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                load_rax(&mut code, disp(lhs)?);
                arith_rax_mem(&mut code, op, disp(rhs)?)?;
                store_rax(&mut code, disp(dst)?);
            }
            Instr::Call {
                dst, name, args, ..
            } => {
                for (i, a) in args.iter().enumerate() {
                    load_argreg(&mut code, i, disp(a)?)?;
                }
                code.push(0xE8); // call rel32 (placeholder; linker-patched)
                calls.push((code.len(), name.clone()));
                code.extend_from_slice(&[0, 0, 0, 0]);
                store_rax(&mut code, disp(dst)?); // result rax -> slot
            }
            Instr::Return { value } => {
                if let Some(v) = value {
                    load_rax(&mut code, disp(v)?);
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
            other => return Err(NativeError::Unsupported(instr_kind(other).to_string())),
        }
    }

    Ok(Func {
        name: name.to_string(),
        code,
        calls,
    })
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
fn write_elf(code: &[u8], entry_off: u64) -> Vec<u8> {
    const LOAD_ADDR: u64 = 0x40_0000;
    const HDRS: u64 = 64 + 56; // ehdr + one phdr
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

/// Lower an `IRModule` to a runnable, deterministic static ELF64 — zero LLVM.
///
/// Scans `ir.instrs` for `FnDef`s, lowers each (the `main` `FnDef` is the entry),
/// links them with computed call displacements, and wraps the result in an ELF.
/// Deterministic by construction: identical `ir` ⇒ byte-identical output.
pub fn compile_to_elf(ir: &IRModule) -> Result<Vec<u8>, NativeError> {
    // Collect (name, body) for every top-level FnDef, entry (`main`) first.
    let mut entry: Option<(&str, &[Instr])> = None;
    let mut rest: Vec<(&str, &[Instr])> = Vec::new();
    for ins in &ir.instrs {
        if let Instr::FnDef { name, body, .. } = ins {
            if name == "main" {
                entry = Some((name, body));
            } else {
                rest.push((name, body));
            }
        }
    }
    let (entry_name, entry_body) = entry.ok_or(NativeError::NoEntry)?;

    let mut funcs = Vec::with_capacity(rest.len() + 1);
    funcs.push(lower_fn(entry_name, entry_body, true)?);
    for (name, body) in rest {
        funcs.push(lower_fn(name, body, false)?);
    }

    let (code, entry_off) = link(&funcs)?;
    Ok(write_elf(&code, entry_off))
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
        let path = std::env::temp_dir().join("mind_native_module_exe");
        let mut fh = std::fs::File::create(&path).expect("create");
        fh.write_all(&a).expect("write");
        let mut perms = fh.metadata().unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod");
        drop(fh);
        let status = Command::new(&path).status().expect("exec");
        assert_eq!(
            status.code(),
            Some(42),
            "must compute f(40,1,1) and exit(42)"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_main_is_a_clean_error_not_a_panic() {
        let mut m = IRModule::new();
        m.instrs = vec![Instr::FnDef {
            name: "f".into(),
            params: vec![],
            ret_id: None,
            reap_threshold: None,
            body: vec![Instr::Return { value: None }],
        }];
        assert_eq!(compile_to_elf(&m), Err(NativeError::NoEntry));
    }
}
