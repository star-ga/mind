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
//! the linker flavor.
//!
//! This type is the single authority those decisions are *converging on*. Two
//! live target-dependent sites still exist outside it and are being threaded in
//! incrementally: the `-march` ladder in [`crate::eval::mlir_build`] (extracted
//! here as [`march_for`] and divergence-guarded against [`Target::march`]) and
//! `HOST_IS_X86` in [`crate::mlir::lowering`]. Until they are threaded, a unit
//! test pins them to this authority so the two sources of truth cannot silently
//! drift. Cross compilation is a data selection
//! (`[targets.<name>].target = "<triple>"`), not a compile-time host special-case.
//! The container (ELF / PE / Mach-O) differs by construction; only the emitted
//! deterministic payload is held byte-identical.

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

// mindc only builds on a supported host. These guards turn an out-of-set build
// host into a COMPILE error rather than letting `host()` fabricate a plausible
// but false `Target` (which would let a same-shaped cross request defeat the
// `requested != host` fail-loud seam). glibc is required on Linux so `host()`
// truthfully reports `Env::Gnu` — a musl-built mindc must not silently declare
// its host as GNU (the mirror of the from_triple libc coercion this module
// forbids).
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("mindc must be built for a supported host arch: x86_64 or aarch64");
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
compile_error!("mindc must be built for a supported host OS: linux, windows, or macos");
#[cfg(all(target_os = "linux", not(target_env = "gnu")))]
compile_error!(
    "mindc must be built against glibc (target_env = gnu) so host() reports the GNU env truthfully"
);

impl Target {
    /// The compile-host target. This is the sole `cfg!(target_arch/os)` site
    /// *within the target abstraction*; the out-of-set host arms are eliminated
    /// at build time by the module-level `compile_error!` guards above, so these
    /// selections are total and truthful — never a runtime fabrication that could
    /// defeat the fail-loud cross-compilation seam.
    pub fn host() -> Target {
        #[cfg(target_arch = "x86_64")]
        let arch = Arch::X86_64;
        #[cfg(target_arch = "aarch64")]
        let arch = Arch::Aarch64;

        #[cfg(target_os = "linux")]
        let (os, env) = (Os::Linux, Env::Gnu);
        #[cfg(target_os = "windows")]
        let (os, env) = (Os::Windows, Env::Gnu);
        #[cfg(target_os = "macos")]
        let (os, env) = (Os::Darwin, Env::None);

        Target { arch, os, env }
    }

    /// Parse a canonical LLVM target triple (the `[targets.<name>].target` /
    /// `--target` seam) with a real tokenized grammar, not substring sniffing.
    ///
    /// The triple is lowercased, trimmed, and split on `-`. `tokens[0]` is the
    /// arch by EXACT token match (so ABI-distinct variants like `aarch64_be`,
    /// `arm64e`, `arm64ec`, `x86_64_v2`, `x86_64h` are *different tokens* and are
    /// rejected, never coerced to the base arch). The OS token is located among
    /// the rest; the token before it is the vendor, the token after it the env;
    /// any leftover token is a hard error. Every component is validated against a
    /// positive allowlist and everything outside the six-target set fails loud —
    /// an unknown or ABI-mismatched triple must never silently degrade to the
    /// host (that is exactly the "quietly mis-target" defect this parser exists to
    /// prevent). By construction `from_triple` is injective onto the supported set
    /// and round-trips its own `llvm_triple()`.
    pub fn from_triple(triple: &str) -> Result<Target> {
        let t = triple.trim().to_ascii_lowercase();
        if t.is_empty() {
            bail!("empty target triple");
        }
        let tokens: Vec<&str> = t.split('-').collect();

        // Arch: exact token match. Endianness / pointer-auth / microarch variants
        // are distinct tokens and fall through to the loud bail.
        let arch = match tokens[0] {
            "x86_64" | "amd64" => Arch::X86_64,
            "aarch64" | "arm64" => Arch::Aarch64,
            other => bail!(
                "unsupported target arch `{other}` in triple `{triple}` \
                 (supported exact tokens: x86_64, amd64, aarch64, arm64)"
            ),
        };

        // Apple OS token: `darwin`/`macos`, optionally version-suffixed
        // (`darwin23`, `darwin23.6.0`, `macos14`). A version suffix is legal ONLY
        // on the Apple OS token.
        fn is_apple_os(tok: &str) -> bool {
            for base in ["darwin", "macos"] {
                if let Some(rest) = tok.strip_prefix(base) {
                    return rest.chars().all(|c| c.is_ascii_digit() || c == '.');
                }
            }
            false
        }

        // Locate the OS token among the non-arch tokens.
        let rest = &tokens[1..];
        let Some(os_idx) = rest
            .iter()
            .position(|&s| s == "linux" || s == "windows" || is_apple_os(s))
        else {
            bail!("unsupported target OS in triple `{triple}` (supported: linux, windows, darwin)");
        };
        let os_tok = rest[os_idx];
        let vendor = if os_idx >= 1 {
            Some(rest[os_idx - 1])
        } else {
            None
        };
        let env_tok = rest.get(os_idx + 1).copied();

        // Nothing may follow the env component.
        if rest.len() > os_idx + 2 {
            bail!("unexpected trailing components after the env in triple `{triple}`");
        }

        // Vendor allowlist (or absent).
        if let Some(v) = vendor {
            if !matches!(v, "unknown" | "pc" | "w64" | "apple") {
                bail!(
                    "unsupported target vendor `{v}` in triple `{triple}` \
                     (supported: unknown, pc, w64, apple)"
                );
            }
        }

        let (os, env) = if os_tok == "windows" {
            // Only the GNU (w64-mingw) Windows ABI is supported. Reject an MSVC
            // triple loudly rather than silently coercing it to MinGW semantics;
            // the two disagree on the C runtime and link model, so a silent swap
            // would miscompile.
            match env_tok {
                Some("gnu") => (Os::Windows, Env::Gnu),
                Some("msvc") => bail!(
                    "unsupported Windows ABI `msvc` in triple `{triple}`; \
                     MIND targets the GNU ABI (e.g. `x86_64-w64-windows-gnu`)"
                ),
                Some(other) => {
                    bail!("unsupported Windows env `{other}` in triple `{triple}` (supported: gnu)")
                }
                None => bail!(
                    "windows triple `{triple}` must name the GNU ABI env \
                     (e.g. `x86_64-w64-windows-gnu`)"
                ),
            }
        } else if is_apple_os(os_tok) {
            // Apple macOS only. The Darwin ABI is implicit — an explicit env
            // component is a malformed macOS triple (and the non-macOS Apple
            // targets ios/tvos/watchos never reach here: they are not the OS
            // token, so the OS-locate step above already bailed).
            if let Some(e) = env_tok {
                bail!(
                    "unsupported env `{e}` in Apple triple `{triple}` \
                     (darwin/macos takes no env component)"
                );
            }
            (Os::Darwin, Env::None)
        } else {
            // Linux: GNU (glibc) libc only. Reject musl / android / uclibc /
            // gnux32 / any non-gnu libc loudly — symmetric with the Windows arm.
            // Discarding the libc component here is exactly how a musl request
            // would parse equal to the glibc host and defeat the fail-loud seam.
            match env_tok {
                Some("gnu") => (Os::Linux, Env::Gnu),
                Some(other) => bail!(
                    "unsupported Linux libc/env `{other}` in triple `{triple}`; \
                     MIND targets the GNU (glibc) ABI (e.g. `x86_64-unknown-linux-gnu`)"
                ),
                None => bail!(
                    "linux triple `{triple}` must name the GNU libc env \
                     (e.g. `x86_64-unknown-linux-gnu`)"
                ),
            }
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
    /// selection (and thus the emitted payload) is fixed independent of the host.
    /// This MUST agree with the live `-march` ladder in
    /// [`crate::eval::mlir_build`] (extracted as [`march_for`]); the two are
    /// pinned together by `march_for_agrees_with_target_march` until the backend
    /// is threaded through [`Target`] and the ladder deleted.
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

    /// The C compiler / linker-driver command for a CROSS target's toolchain, or
    /// `None` when the caller should use the host driver (`$CC` / `cc`). A
    /// windows-gnu target selects the mingw-w64 cross gcc, whose driver knows the
    /// Win64 CRT, import libraries (winpthreads/bcrypt), and PE/COFF output — a
    /// host `cc` would emit ELF and mis-target. Linux/Darwin cross drivers are not
    /// yet wired (they return `None`; the fail-loud `[targets.*].target` seam plus
    /// the `link_coff`/`link_macho` "not wired" arms keep an un-wired cross request
    /// from silently linking with the host driver).
    pub fn cc_command(&self) -> Option<&'static str> {
        match (self.arch, self.os) {
            (Arch::X86_64, Os::Windows) => Some("x86_64-w64-mingw32-gcc"),
            (Arch::Aarch64, Os::Windows) => Some("aarch64-w64-mingw32-gcc"),
            _ => None,
        }
    }
}

/// The `-march` string for a build target — a verbatim, byte-identical extraction
/// of the historical `eval::mlir_build` ladder (deliberately the lenient
/// contains-based form the backend has always used, NOT the strict
/// [`Target::from_triple`] parser, so this refactor perturbs no emitted byte).
/// `None` means the compile host's arch. Kept in lock-step with
/// [`Target::march`] by `march_for_agrees_with_target_march`, so the two sources
/// of truth for the payload-`march` knob cannot silently diverge while the ladder
/// still exists.
pub(crate) fn march_for(triple: Option<&str>) -> Option<&'static str> {
    match triple {
        Some(t) if t.contains("aarch64") || t.contains("arm64") => Some("armv8-a"),
        Some(t) if t.contains("x86_64") => Some("x86-64-v3"),
        Some(_) => None,
        None if cfg!(target_arch = "x86_64") => Some("x86-64-v3"),
        None if cfg!(target_arch = "aarch64") => Some("armv8-a"),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANONICAL: [&str; 6] = [
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "x86_64-w64-windows-gnu",
        "aarch64-w64-windows-gnu",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
    ];

    #[test]
    fn from_triple_round_trips_all_six() {
        for t in CANONICAL {
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
        // Short (vendor-less) Linux form still parses.
        assert_eq!(
            Target::from_triple("aarch64-linux-gnu").unwrap(),
            Target {
                arch: Arch::Aarch64,
                os: Os::Linux,
                env: Env::Gnu
            }
        );
    }

    #[test]
    fn unknown_triple_fails_loud_never_host_fallback() {
        assert!(Target::from_triple("riscv64-unknown-linux-gnu").is_err());
        assert!(Target::from_triple("x86_64-unknown-freebsd").is_err());
        assert!(Target::from_triple("wasm32-wasi").is_err());
        assert!(Target::from_triple("").is_err());
        assert!(Target::from_triple("   ").is_err());
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
    fn non_gnu_linux_libc_fails_loud_not_coerced_to_host() {
        // The silent-mis-target class: a non-glibc Linux libc must NOT parse
        // equal to the glibc host and slip the `requested != host` guard.
        for t in [
            "x86_64-unknown-linux-musl",
            "aarch64-linux-android",
            "x86_64-unknown-linux-gnux32",
            "aarch64-unknown-linux-uclibc",
            "arm-unknown-linux-gnueabihf", // also non-exact arch
        ] {
            assert!(
                Target::from_triple(t).is_err(),
                "{t} must be rejected, not coerced to linux-gnu host"
            );
        }
    }

    #[test]
    fn abi_distinct_arch_tokens_fail_loud() {
        // Exact arch-token match: endianness / pointer-auth / microarch variants
        // are distinct tokens and must bail, never coerce to the base arch.
        for t in [
            "aarch64_be-unknown-linux-gnu",
            "arm64e-apple-darwin",
            "arm64ec-pc-windows-gnu",
            "x86_64_v2-unknown-linux-gnu",
            "x86_64h-apple-darwin",
        ] {
            assert!(
                Target::from_triple(t).is_err(),
                "{t} arch token must be rejected (exact-token match)"
            );
        }
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
    fn march_for_agrees_with_target_march() {
        // Divergence guard: the extracted backend ladder must never drift from
        // the Target authority for any canonical triple, nor for the host.
        for t in CANONICAL {
            let tgt = Target::from_triple(t).unwrap();
            assert_eq!(
                march_for(Some(t)),
                Some(tgt.march()),
                "march_for(Some({t})) must equal Target::march()"
            );
        }
        assert_eq!(
            march_for(None),
            Some(Target::host().march()),
            "march_for(None) must equal the host arch march"
        );
    }

    #[test]
    fn host_is_in_the_supported_set() {
        let h = Target::host();
        // host() must produce a triple that round-trips (i.e. is canonical).
        assert_eq!(Target::from_triple(h.llvm_triple()).unwrap(), h);
    }
}
