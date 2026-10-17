// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Entry-point checks on the frozen compiler's emitted x86-64 executable: the
//! no-`main` library stub is refused, and `fn main(argc, argv)` is admitted only
//! when the entry binds both parameters to the process arguments (W4b).
//!
//! Both checks inspect the PROGRAM entry. In an explicit-layout image a
//! compiler may place its allocator runtime at the ELF entry; that prefix only
//! records the allocator entry points and jumps to the program entry, so the
//! checks follow exactly that recognised jump (see `program_entry_va`).

use crate::build::native_image::{ARENA_ADDR, NativeLayout};

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
pub(super) fn check_entry_is_not_no_main_stub(
    elf: &[u8],
    layout: Option<&NativeLayout>,
) -> Result<(), String> {
    match entry_code(elf, NO_MAIN_ENTRY_STUB.len(), layout)? {
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
pub(super) fn check_entry_binds_main_params(
    elf: &[u8],
    layout: Option<&NativeLayout>,
) -> Result<(), String> {
    let want = ARGV_CAPTURE_LEN + MAIN_PARAMS_BINDING.len();
    let Some(code) = entry_code(elf, MAIN_PARAMS_BINDING.len(), layout)? else {
        return Ok(());
    };
    if code == MAIN_PARAMS_BINDING {
        return Ok(());
    }
    if code.starts_with(&ARGV_CAPTURE_PREFIX)
        && let Some(after) = entry_code(elf, want, layout)?
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

/// The bounded-profile allocator runtime prefix that a compiler emits at code
/// offset 0 of an explicit-layout image (`main.mind::nb_rt_blob_p1`, assembly
/// `testdata/native_alloc_bounded/rt/alloc_rt.s`, label `prefix`):
///
/// ```text
/// movabs r9, ST ; lea rax,[rip+d1] ; mov [r9+536],rax ; lea rax,[rip+d2] ;
/// mov [r9+544],rax ; lea rax,[rip+d3] ; mov [r9+552],rax ; xor eax,eax ;
/// xor r9d,r9d ; jmp rt_end
/// ```
///
/// It touches no stack, rdi or rsi, and `rt_end` is the program entry. `None`
/// marks the bytes the recogniser checks against the layout's state page (ST,
/// 8 bytes) and the four rel32 displacements, which vary with the runtime size.
const RT_PREFIX: [Option<u8>; 62] = {
    const fn fixed(bytes: &[u8], at: usize, mut out: [Option<u8>; 62]) -> [Option<u8>; 62] {
        let mut i = 0;
        while i < bytes.len() {
            out[at + i] = Some(bytes[i]);
            i += 1;
        }
        out
    }
    let mut p = [None; 62];
    p = fixed(&[0x49, 0xB9], 0, p);
    p = fixed(&[0x48, 0x8D, 0x05], 10, p);
    p = fixed(&[0x49, 0x89, 0x81, 0x18, 0x02, 0x00, 0x00], 17, p);
    p = fixed(&[0x48, 0x8D, 0x05], 24, p);
    p = fixed(&[0x49, 0x89, 0x81, 0x20, 0x02, 0x00, 0x00], 31, p);
    p = fixed(&[0x48, 0x8D, 0x05], 38, p);
    p = fixed(&[0x49, 0x89, 0x81, 0x28, 0x02, 0x00, 0x00], 45, p);
    p = fixed(&[0x31, 0xC0, 0x45, 0x31, 0xC9, 0xE9], 52, p);
    p
};

/// The x86-64 executable's ELF entry address and its loaded segments.
struct Image<'a> {
    elf: &'a [u8],
    entry: u64,
    /// `(file offset, vaddr, filesz)` of every PT_LOAD, in header order.
    loads: Vec<(u64, u64, u64)>,
}

impl<'a> Image<'a> {
    /// `None` for any machine but x86-64 (see `entry_code`).
    fn parse(elf: &'a [u8]) -> Result<Option<Image<'a>>, String> {
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
        let entry = u64_at(24).ok_or_else(malformed)?;
        let phoff = u64_at(32).ok_or_else(malformed)? as usize;
        let phentsize = u16_at(54).ok_or_else(malformed)? as usize;
        let phnum = u16_at(56).ok_or_else(malformed)? as usize;
        let mut loads = Vec::new();
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
            loads.push((off, vaddr, filesz));
        }
        Ok(Some(Image { elf, entry, loads }))
    }

    /// `len` file-backed bytes at `va`, all inside the FIRST loaded segment
    /// that contains `va` (the historical lookup order).
    fn code_at(&self, va: u64, len: usize) -> Result<&'a [u8], String> {
        for &(off, vaddr, filesz) in &self.loads {
            if va < vaddr || va - vaddr >= filesz {
                continue;
            }
            let at = off.saturating_add(va - vaddr) as usize;
            return self.elf.get(at..at + len).ok_or_else(malformed);
        }
        Err(malformed())
    }

    /// The address the program's own entry code starts at. Without an explicit
    /// layout this is the ELF entry, exactly as before. With one, an ELF entry
    /// that is the allocator runtime prefix for THIS layout's state page is
    /// followed through its final `jmp`; any other entry is checked as is, so
    /// a seed that predates the runtime, or a prefix built for a different
    /// state page, is never skipped over.
    fn program_entry_va(&self, layout: Option<&NativeLayout>) -> Result<u64, String> {
        let Some(layout) = layout else {
            return Ok(self.entry);
        };
        let Ok(code) = self.code_at(self.entry, RT_PREFIX.len()) else {
            return Ok(self.entry);
        };
        let state_page = ARENA_ADDR + layout.arena_bytes - 4096;
        let shape = RT_PREFIX
            .iter()
            .zip(code)
            .all(|(want, got)| want.is_none_or(|w| w == *got));
        if !shape || code[2..10] != state_page.to_le_bytes() {
            return Ok(self.entry);
        }
        let rel = i32::from_le_bytes(code[58..62].try_into().expect("4-byte slice"));
        let next = self.entry + RT_PREFIX.len() as u64;
        // The runtime jumps FORWARD over itself; anything else is not the
        // recognised prefix and is refused rather than followed.
        if rel < 0 {
            return Err(
                "the explicit-layout runtime prefix does not jump forward to the \
                        program entry"
                    .to_string(),
            );
        }
        Ok(next + rel as u64)
    }
}

fn malformed() -> String {
    "the emitted ELF's entry point lies in no loaded segment".to_string()
}

/// The first `len` code bytes at an x86-64 executable's PROGRAM entry. `None`
/// for any other machine: the frozen compiler emits only `EM_X86_64`, and the
/// host-native transport fixture used on other CI hosts writes a fixed
/// `\x7fELF` + zeros anchor (machine 0) that carries no code.
fn entry_code<'a>(
    elf: &'a [u8],
    len: usize,
    layout: Option<&NativeLayout>,
) -> Result<Option<&'a [u8]>, String> {
    let Some(image) = Image::parse(elf)? else {
        return Ok(None);
    };
    let va = image.program_entry_va(layout)?;
    image.code_at(va, len).map(Some)
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
        assert!(check_entry_is_not_no_main_stub(&elf, None).is_err());
    }

    #[test]
    fn a_real_main_prologue_at_the_entry_is_admitted() {
        // movabs rsp, STACK_TOP -- the first bytes of every frozen `main`.
        let elf = elf_with_entry_code(62, &[0x48, 0xBC, 0xF8, 0x0F, 0x00, 0x41, 0x01, 0, 0]);
        assert_eq!(check_entry_is_not_no_main_stub(&elf, None), Ok(()));
    }

    #[test]
    fn an_entry_outside_every_segment_is_refused() {
        let mut elf = elf_with_entry_code(62, &NO_MAIN_ENTRY_STUB);
        elf[24..32].copy_from_slice(&0x900000u64.to_le_bytes());
        assert!(check_entry_is_not_no_main_stub(&elf, None).is_err());
    }

    #[test]
    fn a_non_x86_64_transport_anchor_is_not_inspected() {
        let mut anchor = b"\x7fELF".to_vec();
        anchor.resize(397, 0);
        assert_eq!(check_entry_is_not_no_main_stub(&anchor, None), Ok(()));
        assert_eq!(check_entry_binds_main_params(&anchor, None), Ok(()));
    }

    /// `movabs rsp, STACK_TOP-8` -- what follows the prefix of every entry.
    const STACK_SWITCH: [u8; 10] = [0x48, 0xBC, 0xF8, 0x0F, 0x00, 0x41, 0x01, 0, 0, 0];

    #[test]
    fn an_entry_that_binds_argc_argv_is_admitted() {
        let mut code = MAIN_PARAMS_BINDING.to_vec();
        code.extend_from_slice(&STACK_SWITCH);
        assert_eq!(
            check_entry_binds_main_params(&elf_with_entry_code(62, &code), None),
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
            check_entry_binds_main_params(&elf_with_entry_code(62, &code), None),
            Ok(())
        );
    }

    #[test]
    fn an_entry_without_the_binding_is_refused() {
        // A pre-W4b seed: the stack switch comes first, nothing loads rdi/rsi.
        assert!(
            check_entry_binds_main_params(&elf_with_entry_code(62, &STACK_SWITCH), None).is_err()
        );
        // The argv capture alone does not bind the parameters either.
        let mut code = vec![0x48, 0x8B, 0x04, 0x24, 0x48, 0xB9];
        code.resize(34, 0);
        code.extend_from_slice(&STACK_SWITCH);
        assert!(check_entry_binds_main_params(&elf_with_entry_code(62, &code), None).is_err());
    }

    /// The bounded layout whose state page the runtime prefix names.
    fn bounded() -> crate::build::native_image::NativeLayout {
        crate::build::native_image::NativeLayout::BOUNDED
    }

    /// The allocator runtime prefix exactly as `nb_rt_blob_p1` emits it for
    /// `state_page`, its `jmp` skipping `skip` bytes of runtime body.
    fn rt_prefix(state_page: u64, skip: i32) -> Vec<u8> {
        let mut p = vec![0x49, 0xB9];
        p.extend_from_slice(&state_page.to_le_bytes());
        for (lea, slot) in [(0x2Du32, 0x18u8), (0x4F, 0x20), (0x6D, 0x28)] {
            p.extend_from_slice(&[0x48, 0x8D, 0x05]);
            p.extend_from_slice(&lea.to_le_bytes());
            p.extend_from_slice(&[0x49, 0x89, 0x81, slot, 0x02, 0x00, 0x00]);
        }
        p.extend_from_slice(&[0x31, 0xC0, 0x45, 0x31, 0xC9, 0xE9]);
        p.extend_from_slice(&skip.to_le_bytes());
        assert_eq!(p.len(), 62);
        p
    }

    fn state_page() -> u64 {
        crate::build::native_image::ARENA_ADDR + bounded().arena_bytes - 4096
    }

    /// An explicit-layout image: runtime prefix at the ELF entry, `skip` bytes
    /// of (fake) runtime body, then `program` -- the program's own entry code.
    fn bounded_image(state_page: u64, program: &[u8]) -> Vec<u8> {
        let body = [0x90u8; 40];
        let mut code = rt_prefix(state_page, body.len() as i32);
        code.extend_from_slice(&body);
        code.extend_from_slice(program);
        elf_with_entry_code(62, &code)
    }

    #[test]
    fn a_bounded_runtime_prefix_is_followed_to_the_program_entry() {
        let layout = bounded();
        let mut bound = MAIN_PARAMS_BINDING.to_vec();
        bound.extend_from_slice(&STACK_SWITCH);
        let elf = bounded_image(state_page(), &bound);
        assert_eq!(check_entry_binds_main_params(&elf, Some(&layout)), Ok(()));
        assert_eq!(check_entry_is_not_no_main_stub(&elf, Some(&layout)), Ok(()));
        // Pre-change behaviour, kept for the default profile: the ELF entry
        // itself is the runtime prefix, which binds nothing.
        assert!(check_entry_binds_main_params(&elf, None).is_err());
    }

    #[test]
    fn a_bounded_program_entry_without_the_binding_is_refused() {
        let layout = bounded();
        let elf = bounded_image(state_page(), &STACK_SWITCH);
        assert!(check_entry_binds_main_params(&elf, Some(&layout)).is_err());
    }

    #[test]
    fn a_no_main_stub_behind_the_runtime_prefix_is_refused() {
        // Before following the prefix this backstop read the runtime's first
        // bytes and could never fire for an explicit-layout image.
        let layout = bounded();
        let elf = bounded_image(state_page(), &NO_MAIN_ENTRY_STUB);
        assert!(check_entry_is_not_no_main_stub(&elf, Some(&layout)).is_err());
    }

    #[test]
    fn a_prefix_for_another_state_page_is_not_followed() {
        let layout = bounded();
        let mut bound = MAIN_PARAMS_BINDING.to_vec();
        bound.extend_from_slice(&STACK_SWITCH);
        let elf = bounded_image(state_page() + 4096, &bound);
        assert!(check_entry_binds_main_params(&elf, Some(&layout)).is_err());
        // An altered instruction in the prefix is not the recognised prefix.
        let mut code = rt_prefix(state_page(), 0);
        code[18] = 0x8B; // mov [r9+536],rax -> a different instruction
        code.extend_from_slice(&MAIN_PARAMS_BINDING);
        let elf = elf_with_entry_code(62, &code);
        assert!(check_entry_binds_main_params(&elf, Some(&layout)).is_err());
    }

    #[test]
    fn a_backward_runtime_jump_is_refused() {
        let layout = bounded();
        let mut code = rt_prefix(state_page(), -62);
        code.extend_from_slice(&MAIN_PARAMS_BINDING);
        let elf = elf_with_entry_code(62, &code);
        assert!(check_entry_binds_main_params(&elf, Some(&layout)).is_err());
        assert!(check_entry_is_not_no_main_stub(&elf, Some(&layout)).is_err());
    }
}
