// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Entry-point checks on the frozen compiler's emitted x86-64 executable: the
//! no-`main` library stub is refused, and `fn main(argc, argv)` is admitted only
//! when the entry binds both parameters to the process arguments (W4b).

/// The frozen compiler's synthesized no-`main` entry:
/// `xor edi,edi ; mov eax,60 ; syscall` (`main.mind::nb_synth_entry`).
pub(super) const NO_MAIN_ENTRY_STUB: [u8; 9] =
    [0x31, 0xFF, 0xB8, 0x3C, 0x00, 0x00, 0x00, 0x0F, 0x05];

/// Refuse an x86-64 executable whose entry point is the no-`main` stub, or whose
/// entry cannot be located in a loaded segment.
///
/// Scoped to `EM_X86_64` images because that is the only machine the frozen
/// compiler emits; the host-native transport fixture used on other CI hosts
/// writes a fixed `\x7fELF` + zeros anchor (machine 0) that carries no code.
pub(super) fn check_entry_is_not_no_main_stub(elf: &[u8]) -> Result<(), String> {
    match entry_code(elf, NO_MAIN_ENTRY_STUB.len())? {
        Some(code) if code == NO_MAIN_ENTRY_STUB => Err(
            "the frozen compiler emitted its no-`main` library entry stub \
                    (`exit(0)`), so it did not see the admitted `main`"
                .to_string(),
        ),
        _ => Ok(()),
    }
}

/// W4b: the entry prologue that binds `fn main(argc, argv)` to the process
/// arguments (`main.mind::nb_emit_main_params`): `mov rdi,[rsp] ; lea
/// rsi,[rsp+8]`, emitted before the entry switches to its own stack.
const MAIN_PARAMS_BINDING: [u8; 9] = [0x48, 0x8B, 0x3C, 0x24, 0x48, 0x8D, 0x74, 0x24, 0x08];

/// The 34-byte argc/argv capture (`main.mind::nb_emit_argv_capture`) that an
/// entry whose program calls `__mind_argc`/`__mind_argv` carries ahead of the
/// binding; it starts `mov rax,[rsp] ; movabs rcx,..`.
const ARGV_CAPTURE_PREFIX: [u8; 6] = [0x48, 0x8B, 0x04, 0x24, 0x48, 0xB9];
const ARGV_CAPTURE_LEN: usize = 34;

/// Refuse an artifact for `fn main(argc: i64, argv: i64)` whose entry does not
/// bind the parameters to the process arguments. A compiler seed that predates
/// W4b emits no binding, and its artifact would run `main` with argc = argv = 0;
/// this check turns that into a refusal, so the shape is admitted exactly when
/// the compiler in use binds it.
pub(super) fn check_entry_binds_main_params(elf: &[u8]) -> Result<(), String> {
    let want = ARGV_CAPTURE_LEN + MAIN_PARAMS_BINDING.len();
    let Some(code) = entry_code(elf, MAIN_PARAMS_BINDING.len())? else {
        return Ok(());
    };
    if code == MAIN_PARAMS_BINDING {
        return Ok(());
    }
    if code.starts_with(&ARGV_CAPTURE_PREFIX)
        && let Some(after) = entry_code(elf, want)?
        && after[ARGV_CAPTURE_LEN..] == MAIN_PARAMS_BINDING
    {
        return Ok(());
    }
    Err(
        "the native compiler in use does not bind `main`'s parameters to the process \
         arguments (its entry lacks the argc/argv binding), so the program would run with \
         argc = argv = 0"
            .to_string(),
    )
}

/// The first `len` code bytes at an x86-64 executable's entry point. `None` for
/// any other machine: the frozen compiler emits only `EM_X86_64`, and the
/// host-native transport fixture used on other CI hosts writes a fixed
/// `\x7fELF` + zeros anchor (machine 0) that carries no code.
fn entry_code(elf: &[u8], len: usize) -> Result<Option<&[u8]>, String> {
    const EM_X86_64: u16 = 62;
    const PT_LOAD: u32 = 1;
    let u16_at = |o: usize| elf.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |o: usize| {
        elf.get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4-byte slice")))
    };
    let u64_at = |o: usize| {
        elf.get(o..o + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8-byte slice")))
    };
    if u16_at(18) != Some(EM_X86_64) {
        return Ok(None);
    }
    let malformed = || "the emitted ELF's entry point lies in no loaded segment".to_string();
    let entry = u64_at(24).ok_or_else(malformed)?;
    let phoff = u64_at(32).ok_or_else(malformed)? as usize;
    let phentsize = u16_at(54).ok_or_else(malformed)? as usize;
    let phnum = u16_at(56).ok_or_else(malformed)? as usize;
    for i in 0..phnum {
        let ph = phoff.saturating_add(i.saturating_mul(phentsize));
        if u32_at(ph) != Some(PT_LOAD) {
            continue;
        }
        let (Some(off), Some(vaddr), Some(filesz)) =
            (u64_at(ph + 8), u64_at(ph + 16), u64_at(ph + 32))
        else {
            return Err(malformed());
        };
        if entry < vaddr || entry - vaddr >= filesz {
            continue;
        }
        let at = off.saturating_add(entry - vaddr) as usize;
        return elf.get(at..at + len).map(Some).ok_or_else(malformed);
    }
    Err(malformed())
}

#[cfg(test)]
mod entry_stub_tests {
    use super::{
        MAIN_PARAMS_BINDING, NO_MAIN_ENTRY_STUB, check_entry_binds_main_params,
        check_entry_is_not_no_main_stub,
    };

    /// A minimal ELF64 x86-64 image: one PT_LOAD at file 0 / vaddr 0x400000,
    /// entry 0x400078 (file offset 120, right after the header and one phdr).
    fn elf_with_entry_code(machine: u16, code: &[u8]) -> Vec<u8> {
        let mut b = vec![0u8; 120];
        b[0..4].copy_from_slice(b"\x7fELF");
        b[18..20].copy_from_slice(&machine.to_le_bytes());
        b[24..32].copy_from_slice(&0x400078u64.to_le_bytes());
        b[32..40].copy_from_slice(&64u64.to_le_bytes());
        b[54..56].copy_from_slice(&56u16.to_le_bytes());
        b[56..58].copy_from_slice(&1u16.to_le_bytes());
        b[64..68].copy_from_slice(&1u32.to_le_bytes());
        b[64 + 16..64 + 24].copy_from_slice(&0x400000u64.to_le_bytes());
        b.extend_from_slice(code);
        b.resize(400, 0);
        let len = b.len() as u64;
        b[64 + 32..64 + 40].copy_from_slice(&len.to_le_bytes());
        b
    }

    #[test]
    fn the_no_main_stub_at_the_entry_is_refused() {
        let elf = elf_with_entry_code(62, &NO_MAIN_ENTRY_STUB);
        assert!(check_entry_is_not_no_main_stub(&elf).is_err());
    }

    #[test]
    fn a_real_main_prologue_at_the_entry_is_admitted() {
        // movabs rsp, STACK_TOP -- the first bytes of every frozen `main`.
        let elf = elf_with_entry_code(62, &[0x48, 0xBC, 0xF8, 0x0F, 0x00, 0x41, 0x01, 0, 0]);
        assert_eq!(check_entry_is_not_no_main_stub(&elf), Ok(()));
    }

    #[test]
    fn an_entry_outside_every_segment_is_refused() {
        let mut elf = elf_with_entry_code(62, &NO_MAIN_ENTRY_STUB);
        elf[24..32].copy_from_slice(&0x900000u64.to_le_bytes());
        assert!(check_entry_is_not_no_main_stub(&elf).is_err());
    }

    #[test]
    fn a_non_x86_64_transport_anchor_is_not_inspected() {
        let mut anchor = b"\x7fELF".to_vec();
        anchor.resize(397, 0);
        assert_eq!(check_entry_is_not_no_main_stub(&anchor), Ok(()));
        assert_eq!(check_entry_binds_main_params(&anchor), Ok(()));
    }

    /// `movabs rsp, STACK_TOP-8` -- what follows the prefix of every entry.
    const STACK_SWITCH: [u8; 10] = [0x48, 0xBC, 0xF8, 0x0F, 0x00, 0x41, 0x01, 0, 0, 0];

    #[test]
    fn an_entry_that_binds_argc_argv_is_admitted() {
        let mut code = MAIN_PARAMS_BINDING.to_vec();
        code.extend_from_slice(&STACK_SWITCH);
        assert_eq!(
            check_entry_binds_main_params(&elf_with_entry_code(62, &code)),
            Ok(())
        );
    }

    #[test]
    fn an_entry_with_the_argv_capture_then_the_binding_is_admitted() {
        // nb_emit_argv_capture (34 bytes) ahead of the binding.
        let mut code = vec![0x48, 0x8B, 0x04, 0x24, 0x48, 0xB9];
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0x40, 0, 0, 0, 0]);
        code.extend_from_slice(&[0x48, 0x89, 0x41, 0x08, 0x48, 0x8D, 0x44, 0x24, 0x08]);
        code.extend_from_slice(&[0x48, 0x89, 0x41, 0x10, 0x48, 0xC7, 0x01, 0x10, 0, 0, 0]);
        assert_eq!(code.len(), 34);
        code.extend_from_slice(&MAIN_PARAMS_BINDING);
        code.extend_from_slice(&STACK_SWITCH);
        assert_eq!(
            check_entry_binds_main_params(&elf_with_entry_code(62, &code)),
            Ok(())
        );
    }

    #[test]
    fn an_entry_without_the_binding_is_refused() {
        // A pre-W4b seed: the stack switch comes first, nothing loads rdi/rsi.
        assert!(check_entry_binds_main_params(&elf_with_entry_code(62, &STACK_SWITCH)).is_err());
        // The argv capture alone does not bind the parameters either.
        let mut code = vec![0x48, 0x8B, 0x04, 0x24, 0x48, 0xB9];
        code.resize(34, 0);
        code.extend_from_slice(&STACK_SWITCH);
        assert!(check_entry_binds_main_params(&elf_with_entry_code(62, &code)).is_err());
    }
}
