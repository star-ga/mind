// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Compilation-target abstraction — the single knob for MIND's multi-target
//! native codegen.
//!
//! One MIND source compiles once and produces a byte-identical *payload* on any
//! supported machine (the wedge: "the same bytes, on any machine"). A [`Target`]
//! names one `{arch} × {os}` and derives every target-dependent decision the
//! backend needs — the LLVM triple, the `-march`, the executable extension, and
//! the linker flavor. This replaces the scattered host-`cfg!(target_os/arch)`
//! tables and `string.contains("aarch64")` ladders with one authority, so cross
//! compilation is a data selection (`[targets.<name>].target = "<triple>"`), not
//! a compile-time host special-case. The container (ELF / PE / Mach-O) differs by
//! construction; only the emitted deterministic payload is held byte-identical.

use anyhow::{Result, bail};

/// Instruction-set architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
}

/// Target operating system (selects the object/link format and CRT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Linux,
    Windows,
    Darwin,
}

/// Target ABI environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Env {
    /// GNU userland (linux-gnu, w64-windows-gnu via MinGW).
    Gnu,
    /// No explicit env component (Apple Darwin).
    None,
}

/// Object/executable format flavor — selects the linker and its argument shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkFlavor {
    /// ELF + GNU ld / lld; supports `-z relro/now`, `-Bsymbolic-functions`, `-pie`.
    Elf,
    /// PE/COFF + lld-link semantics; DEP/ASLR are format defaults (no `-z`).
    Coff,
    /// Mach-O + ld64; two-level namespace gives closed-world binding.
    MachO,
}

/// A concrete compilation target: one architecture on one OS with one ABI env.
///
/// The canonical supported set is the six `{X86_64, Aarch64} × {Linux, Windows,
/// Darwin}` combinations. All target-dependent codegen/link decisions are derived
/// accessors on this type, never re-decided at the use site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub arch: Arch,
    pub os: Os,
    pub env: Env,
}

impl Target {
    /// The compile-host target. This is the ONLY place `cfg!(target_arch/os)` is
    /// consulted — every other decision flows from an explicit [`Target`], so a
    /// build never accidentally bakes in the host machine. Falls back to
    /// `x86_64-linux-gnu` for any host outside the supported set (mindc's own
    /// build hosts are always in-set).
    pub fn host() -> Target {
        let arch = match std::env::consts::ARCH {
            "aarch64" | "arm64" => Arch::Aarch64,
            _ => Arch::X86_64,
        };
        let (os, env) = match std::env::consts::OS {
            "windows" => (Os::Windows, Env::Gnu),
            "macos" => (Os::Darwin, Env::None),
            _ => (Os::Linux, Env::Gnu),
        };
        Target { arch, os, env }
    }

    /// Parse a canonical LLVM target triple string (the `[targets.<name>].target`
    /// / `--target` seam). Accepts the six supported `{arch}×{os}` combinations,
    /// tolerant of the vendor/env components (e.g. both `x86_64-apple-darwin` and
    /// `x86_64-apple-darwin23`). Rejects anything outside the set loudly — an
    /// unknown triple must never silently degrade to the host.
    pub fn from_triple(triple: &str) -> Result<Target> {
        let t = triple.trim().to_ascii_lowercase();
        let arch = if t.starts_with("x86_64") || t.starts_with("amd64") {
            Arch::X86_64
        } else if t.starts_with("aarch64") || t.starts_with("arm64") {
            Arch::Aarch64
        } else {
            bail!("unsupported target arch in triple `{triple}` (supported: x86_64, aarch64)");
        };
        let (os, env) = if t.contains("windows") {
            // Only the GNU (w64-mingw) Windows ABI is supported. Reject an MSVC
            // triple loudly rather than silently coercing it to MinGW semantics —
            // the two disagree on the C runtime, the `long` width is the same but
            // the CRT/link model is not, so a silent swap would miscompile.
            if t.contains("msvc") {
                bail!(
                    "unsupported Windows ABI `msvc` in triple `{triple}`; \
                     MIND targets the GNU ABI (e.g. `x86_64-w64-windows-gnu`)"
                );
            }
            (Os::Windows, Env::Gnu)
        } else if t.contains("darwin") || t.contains("macos") {
            // Apple *macOS* only — keyed on the `darwin` kernel or the `macos` OS
            // component, NOT a bare `apple` vendor (which also matches
            // `apple-ios` / `-tvos` / `-watchos`). Those non-macOS Apple targets
            // fall through to the loud OS bail below rather than mis-parsing as a
            // supported Darwin desktop target.
            (Os::Darwin, Env::None)
        } else if t.contains("linux") {
            (Os::Linux, Env::Gnu)
        } else {
            bail!("unsupported target OS in triple `{triple}` (supported: linux, windows, darwin)");
        };
        Ok(Target { arch, os, env })
    }

    /// The canonical LLVM triple this target lowers to (`clang --target=`).
    pub fn llvm_triple(&self) -> &'static str {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Linux) => "x86_64-unknown-linux-gnu",
            (Arch::Aarch64, Os::Linux) => "aarch64-unknown-linux-gnu",
            (Arch::X86_64, Os::Windows) => "x86_64-w64-windows-gnu",
            (Arch::Aarch64, Os::Windows) => "aarch64-w64-windows-gnu",
            (Arch::X86_64, Os::Darwin) => "x86_64-apple-darwin",
            (Arch::Aarch64, Os::Darwin) => "aarch64-apple-darwin",
        }
    }

    /// The `-march` for deterministic codegen. Pins the ISA level so operation
    /// selection (and thus the emitted payload) is fixed independent of the host
    /// — subsumes the `string.contains("aarch64")` march ladder in the backend.
    pub fn march(&self) -> &'static str {
        match self.arch {
            Arch::X86_64 => "x86-64-v3",
            Arch::Aarch64 => "armv8-a",
        }
    }

    /// Executable filename extension for this OS (`.exe` on Windows, else empty).
    pub fn exe_ext(&self) -> &'static str {
        match self.os {
            Os::Windows => ".exe",
            _ => "",
        }
    }

    /// The object/link format flavor — drives linker + argument selection.
    pub fn link_flavor(&self) -> LinkFlavor {
        match self.os {
            Os::Linux => LinkFlavor::Elf,
            Os::Windows => LinkFlavor::Coff,
            Os::Darwin => LinkFlavor::MachO,
        }
    }

    /// True when this target is the compile host (an in-process build, no cross
    /// toolchain/sysroot needed).
    pub fn is_host(&self) -> bool {
        *self == Target::host()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_triple_round_trips_all_six() {
        for t in [
            "x86_64-unknown-linux-gnu",
            "aarch64-unknown-linux-gnu",
            "x86_64-w64-windows-gnu",
            "aarch64-w64-windows-gnu",
            "x86_64-apple-darwin",
            "aarch64-apple-darwin",
        ] {
            let target = Target::from_triple(t).expect("canonical triple must parse");
            assert_eq!(target.llvm_triple(), t, "triple must round-trip");
        }
    }

    #[test]
    fn from_triple_tolerates_versioned_and_alias_forms() {
        assert_eq!(
            Target::from_triple("x86_64-apple-darwin23.6.0").unwrap(),
            Target {
                arch: Arch::X86_64,
                os: Os::Darwin,
                env: Env::None
            }
        );
        assert_eq!(
            Target::from_triple("arm64-apple-macos14").unwrap().arch,
            Arch::Aarch64
        );
    }

    #[test]
    fn unknown_triple_fails_loud_never_host_fallback() {
        assert!(Target::from_triple("riscv64-unknown-linux-gnu").is_err());
        assert!(Target::from_triple("x86_64-unknown-freebsd").is_err());
        assert!(Target::from_triple("wasm32-wasi").is_err());
    }

    #[test]
    fn unsupported_abi_or_apple_variant_fails_loud_not_coerced() {
        // MSVC must be rejected, not silently coerced to the MinGW/GNU ABI.
        assert!(Target::from_triple("x86_64-pc-windows-msvc").is_err());
        assert!(Target::from_triple("aarch64-pc-windows-msvc").is_err());
        // The GNU Windows ABI still parses.
        assert_eq!(
            Target::from_triple("x86_64-pc-windows-gnu").unwrap().os,
            Os::Windows
        );
        // A bare `apple` vendor that is NOT macOS must not parse as Darwin.
        assert!(Target::from_triple("arm64-apple-ios17.0").is_err());
        assert!(Target::from_triple("aarch64-apple-tvos").is_err());
        assert!(Target::from_triple("arm64-apple-watchos").is_err());
    }

    #[test]
    fn march_and_flavor_are_target_derived() {
        let win = Target::from_triple("x86_64-w64-windows-gnu").unwrap();
        assert_eq!(win.march(), "x86-64-v3");
        assert_eq!(win.exe_ext(), ".exe");
        assert_eq!(win.link_flavor(), LinkFlavor::Coff);
        let arm_lin = Target::from_triple("aarch64-unknown-linux-gnu").unwrap();
        assert_eq!(arm_lin.march(), "armv8-a");
        assert_eq!(arm_lin.exe_ext(), "");
        assert_eq!(arm_lin.link_flavor(), LinkFlavor::Elf);
    }

    #[test]
    fn host_is_in_the_supported_set() {
        let h = Target::host();
        // host() must produce a triple that round-trips (i.e. is canonical).
        assert_eq!(Target::from_triple(h.llvm_triple()).unwrap(), h);
    }
}
