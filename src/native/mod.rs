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
//! not `-O3`; it is *determinism guaranteed by the codegen's shape*.
//!
//! ## Scope (honest) — the scalar-i64 vertical slice
//!
//! Supports `FnDef` / `Param` / `ConstI64` / `Call` / `Return`, arithmetic
//! `BinOp{Add,Sub,Mul,Div,Mod}` (Div/Mod via `cqo`+signed `idiv`), bitwise/shift
//! `BinOp{BitAnd,BitOr,BitXor,Shl,Shr}` (Shr = arithmetic `sar`; shift count in
//! `cl`), signed comparison `BinOp{Lt,Le,Gt,Ge,Eq,Ne}`
//! (`cmp`+`setcc`+`movzx` → 0/1), `If` (if-expressions: `cmp`/`je`/`jmp` with
//! intra-function forward-jump patching and mem-to-mem phi resolution via the
//! merge list), and `While` loops (header re-test + back-edge `jmp`; loop-carried
//! vars alias to one slot so no phi forwarding is needed), ≤6 integer args
//! (System-V registers), a regalloc-free mem-to-mem frame model, and intra-module
//! calls (computed PC-relative displacement). The entry point is the `FnDef`
//! named `main`; its `Return` becomes an `exit` syscall.
//!
//! deferred: `break`/`continue` (need a loop-context patch list), Div/Mod
//!   edge-case guards (raw idiv traps on /0 and INT_MIN/-1 — MIND's path defines
//!   them), >6 args (stack-passed), register allocation (currently
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
fn assign_slots(body: &[Instr]) -> HashMap<ValueId, usize> {
    fn define(slot: &mut HashMap<ValueId, usize>, id: ValueId) {
        let n = slot.len();
        slot.entry(id).or_insert(n);
    }
    fn walk(body: &[Instr], slot: &mut HashMap<ValueId, usize>) {
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
                    walk(cond_instrs, slot);
                    walk(then_instrs, slot);
                    walk(else_instrs, slot);
                    define(slot, *dst);
                    for (m, _, _) in merges {
                        define(slot, *m);
                    }
                }
                // A loop-carried variable is ONE memory slot across iterations:
                // its pre-loop (`init_ids`), post-body (`live_vars`), and post-loop
                // (`exit_ids`) SSA ids all alias to that slot, so the mem-to-mem
                // model needs no phi nodes — the slot just holds the live value.
                #[cfg(feature = "std-surface")]
                Instr::While {
                    cond_instrs,
                    body: loop_body,
                    live_vars,
                    init_ids,
                    exit_ids,
                    ..
                } => {
                    walk(cond_instrs, slot);
                    walk(loop_body, slot);
                    for (i, (_, post_id)) in live_vars.iter().enumerate() {
                        if let Some(&s) = slot.get(post_id) {
                            if let Some(init_id) = init_ids.get(i) {
                                if init_id.0 != usize::MAX {
                                    slot.insert(*init_id, s);
                                }
                            }
                            if let Some(exit_id) = exit_ids.get(i) {
                                slot.insert(*exit_id, s);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut slot = HashMap::new();
    walk(body, &mut slot);
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
fn emit_seq(
    code: &mut Vec<u8>,
    calls: &mut Vec<(usize, String)>,
    body: &[Instr],
    slot: &HashMap<ValueId, usize>,
    is_entry: bool,
) -> Result<(), NativeError> {
    let disp = |id: &ValueId| -> Result<i32, NativeError> {
        slot.get(id)
            .map(|&i| slot_disp(i))
            .ok_or_else(|| NativeError::Unsupported(format!("use of undefined value {id}")))
    };
    for ins in body {
        match ins {
            Instr::Param { dst, index, .. } => store_argreg(code, *index, disp(dst)?)?,
            Instr::ConstI64(dst, v) => {
                code.extend_from_slice(&[0x48, 0xB8]); // movabs rax, imm64
                code.extend_from_slice(&v.to_le_bytes());
                store_rax(code, disp(dst)?);
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                load_rax(code, disp(lhs)?);
                match op {
                    BinOp::Div | BinOp::Mod => {
                        // deferred: raw signed idiv — traps (#DE / SIGFPE) on
                        // divide-by-zero and on INT_MIN / -1. MIND's MLIR path
                        // gives these DEFINED results (div-by-zero decision +
                        // INT_MIN/-1 guard); upgrade path: emit the same guarded
                        // sequence (cmp rhs,0 + cmp pair) before the idiv.
                        code.extend_from_slice(&[0x48, 0x99]); // cqo (sign-extend rax -> rdx:rax)
                        code.extend_from_slice(&[0x48, 0xF7, 0xBD]); // idiv qword [rbp+disp32]
                        code.extend_from_slice(&disp(rhs)?.to_le_bytes());
                        if matches!(op, BinOp::Mod) {
                            code.extend_from_slice(&[0x48, 0x89, 0xD0]); // mov rax, rdx (remainder)
                        }
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
            } => {
                for (i, a) in args.iter().enumerate() {
                    load_argreg(code, i, disp(a)?)?;
                }
                code.push(0xE8); // call rel32 (placeholder; linker-patched)
                calls.push((code.len(), name.clone()));
                code.extend_from_slice(&[0, 0, 0, 0]);
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
                // Evaluate the condition; branch to the else-block when it is 0.
                emit_seq(code, calls, cond_instrs, slot, is_entry)?;
                let je = emit_cmp_je_zero(code, disp(cond_id)?);

                // then-block: body, phi(then side), dst = then_result, jump to end.
                emit_seq(code, calls, then_instrs, slot, is_entry)?;
                for (m, then_val, _) in merges {
                    load_rax(code, disp(then_val)?);
                    store_rax(code, disp(m)?);
                }
                load_rax(code, disp(then_result)?);
                store_rax(code, disp(dst)?);
                let jmp = emit_jmp(code);

                // else-block: body, phi(else side), dst = else_result.
                let else_at = code.len();
                patch_rel32(code, je, else_at);
                emit_seq(code, calls, else_instrs, slot, is_entry)?;
                for (m, _, else_val) in merges {
                    load_rax(code, disp(else_val)?);
                    store_rax(code, disp(m)?);
                }
                load_rax(code, disp(else_result)?);
                store_rax(code, disp(dst)?);

                let end_at = code.len();
                patch_rel32(code, jmp, end_at);
            }
            // `while cond { body }` — re-test the condition each iteration; the
            // loop-carried vars live in their (aliased) slots so no phi forwarding
            // is needed. (Bounded: `break`/`continue` fail-closed for now — they
            // need a loop-context patch list; a body Return still works.)
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_id,
                cond_instrs,
                body: loop_body,
                ..
            } => {
                let header = code.len();
                emit_seq(code, calls, cond_instrs, slot, is_entry)?;
                let je = emit_cmp_je_zero(code, disp(cond_id)?); // exit when cond == 0
                emit_seq(code, calls, loop_body, slot, is_entry)?;
                let back = emit_jmp(code);
                patch_rel32(code, back, header); // back-edge to the header
                let end = code.len();
                patch_rel32(code, je, end);
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

    // Prologue: push rbp ; mov rbp, rsp ; sub rsp, frame
    let mut code = vec![0x55, 0x48, 0x89, 0xE5, 0x48, 0x81, 0xEC];
    code.extend_from_slice(&frame.to_le_bytes());

    let mut calls = Vec::new();
    emit_seq(&mut code, &mut calls, body, &slot, is_entry)?;

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
    let (entry_name, entry_body, entry_ret) = entry.ok_or(NativeError::NoEntry)?;

    let mut funcs = Vec::with_capacity(rest.len() + 1);
    funcs.push(lower_fn(entry_name, entry_body, true, entry_ret)?);
    for (name, body, ret_id) in rest {
        funcs.push(lower_fn(name, body, false, ret_id)?);
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
        assert_eq!(
            run(&a, "mind_native_module_exe"),
            Some(42),
            "must compute f(40,1,1) and exit(42)"
        );
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
}
