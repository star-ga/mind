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

//! Per-target executable link driver — the single place the final native binary
//! is produced, dispatched on the target's [`LinkFlavor`].
//!
//! MIND's deterministic *payload* (the emitted `.o` machine code) is held
//! byte-identical across every supported target; only the *container* — ELF vs
//! PE/COFF vs Mach-O — and the flavor-specific link flags differ. This driver is
//! where that container split lives: [`LinkDriver::link`] matches on
//! [`Target::link_flavor`] into one private linker invocation per flavor, so a
//! new OS is one new arm, never a scattered `cfg!(target_os)` ladder at the call
//! site.
//!
//! Today only the host ELF arm is wired; the PE/COFF and Mach-O arms fail loud
//! with an explicit "not yet wired" error (the same fail-closed discipline as the
//! `[targets.*].target` seam in [`crate::project`]). Each is turned on behind its
//! own byte-identity gate in a later slice.

use crate::target::{LinkFlavor, Target};
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Everything the final link needs, gathered by the caller so the driver itself
/// makes no build-orchestration decisions (it only assembles + runs the linker).
pub struct LinkCtx<'a> {
    /// C compiler / linker driver command (host builds: `$CC` or `cc`; a cross
    /// target selects its own toolchain in a later slice).
    pub cc: &'a str,
    /// Object files to link, in declared order.
    pub objects: &'a [PathBuf],
    /// Final executable path (the caller appends any OS extension).
    pub output: &'a Path,
    /// Directory holding the MIND runtime library (`-L`).
    pub lib_dir: &'a Path,
    /// Runtime library base name (`-l<name>`).
    pub runtime_link: &'a str,
    /// Release profile — adds `-O3 -flto`.
    pub release: bool,
    /// Print the assembled link command.
    pub verbose: bool,
    /// Pre-compiled runtime-support shim object (compiled by the caller through
    /// the MLIR toolchain, fixed basename + `-ffile-prefix-map`, no debug info),
    /// linked FIRST so the `MIND_EXPORT` intrinsics resolve statically. `None`
    /// when the shim is unavailable (e.g. a `--no-default-features` build without
    /// `mlir-build`) — byte-identical to the historical no-shim link.
    pub runtime_shim: Option<&'a Path>,
}

/// Produces the final native executable for one [`Target`].
pub struct LinkDriver {
    target: Target,
}

impl LinkDriver {
    /// Driver for an explicit target. `LinkDriver::new(Target::host())` is the
    /// host link path (ELF on Linux) — byte-identical to the pre-driver native
    /// link, proven by the keystone byte-identity gate.
    pub fn new(target: Target) -> Self {
        Self { target }
    }

    /// Link the final executable, dispatching on the target's object/link format.
    pub fn link(&self, ctx: &LinkCtx) -> Result<()> {
        match self.target.link_flavor() {
            LinkFlavor::Elf => self.link_elf(ctx),
            LinkFlavor::Coff => self.link_coff(ctx),
            LinkFlavor::MachO => self.link_macho(ctx),
        }
    }

    /// ELF (Linux) executable link.
    ///
    /// This is a faithful, order-preserving extraction of the historical native
    /// link path: shim → objects → `-o` → `-L`/`-l` → `-ldl -lpthread -lm` →
    /// the byte-neutral `-Wl,-z,relro/now/noexecstack` hardening → the Linux
    /// `-rpath` pair → the release `-O3 -flto`. The flag set and its ORDER are
    /// held identical so the emitted ELF stays byte-for-byte what it was before
    /// this driver existed (the keystone byte-identity gate is the proof).
    fn link_elf(&self, ctx: &LinkCtx) -> Result<()> {
        let mut cmd = Command::new(ctx.cc);

        // Runtime-support shim first, so its MIND_EXPORT symbols resolve.
        if let Some(shim) = ctx.runtime_shim {
            cmd.arg(shim);
        }

        // Object files.
        for obj in ctx.objects {
            cmd.arg(obj);
        }

        // Output path.
        cmd.arg("-o").arg(ctx.output);

        // MIND runtime library.
        cmd.arg(format!("-L{}", ctx.lib_dir.display()));
        cmd.arg(format!("-l{}", ctx.runtime_link));

        // Standard libraries.
        cmd.arg("-ldl"); // dlopen on Linux
        cmd.arg("-lpthread");
        cmd.arg("-lm");

        // Emitted-artifact hardening — byte-neutral linker directives (SAME class
        // as the cdylib path's `-Wl,-Bsymbolic-functions` / `-Wl,-z,*`). These
        // change only ELF segment permissions and DT_FLAGS bits, not one emitted
        // instruction byte, and are applied uniformly on every substrate, so the
        // native-ELF artifact stays byte-identical across builds and substrates
        // (proven via `objdump -d` .text identity before/after):
        //   -z relro       : GOT/relocated sections read-only after startup.
        //   -z now         : eager relocation (BIND_NOW / DF_1_NOW) — Full RELRO.
        //   -z noexecstack : non-executable stack (GNU_STACK without X).
        cmd.arg("-Wl,-z,relro");
        cmd.arg("-Wl,-z,now");
        cmd.arg("-Wl,-z,noexecstack");
        // deferred: the BYTE-CHANGING hardening pass — `-fstack-protector-strong`
        // (canary prologue/epilogue machine code) and `-pie` (PC-relative codegen
        // + ELF type DYN) — perturbs emitted instruction bytes and requires a
        // keystone (7/7) + cross-substrate re-bless. Upgrade path: land under a
        // fresh reference_hashes.toml blessing ceremony owned by
        // mind-cross-substrate (Constitution §I.3), NOT in this byte-neutral path.

        // Runtime-library rpath lookup (Linux).
        cmd.arg(format!("-Wl,-rpath,{}", ctx.lib_dir.display()));
        cmd.arg("-Wl,-rpath,$ORIGIN/../lib");

        // Optimization flags.
        if ctx.release {
            cmd.arg("-O3");
            cmd.arg("-flto");
        }

        self.run(cmd, ctx)
    }

    /// PE/COFF (Windows, GNU/w64-mingw ABI) executable link.
    ///
    /// The windows-gnu link differs materially from ELF and this arm encodes the
    /// differences explicitly rather than reusing the ELF flag set:
    ///   * NO `-ldl` — `dlopen` is absent on Windows (mingw has no `libdl`).
    ///   * `-lpthread` resolves against **winpthreads** (mingw's pthread port).
    ///   * `-lm` is present (mingw ships `libm`).
    ///   * `-lbcrypt` — the runtime RNG uses the Windows CNG `BCryptGenRandom`;
    ///     mingw always ships the `bcrypt` import lib, so this is safe to pass.
    ///   * NONE of the `-Wl,-z,relro/now/noexecstack` directives — they are
    ///     ELF-only; DEP/ASLR are PE format defaults set in the optional header.
    ///   * NO rpath — PE has no runtime search path; the runtime lib must be
    ///     static or sit beside the `.exe`.
    ///   * `ctx.output` is written verbatim — the caller has already appended
    ///     `Target::exe_ext()` (`.exe`) to the path.
    ///
    /// The emitted `.text` payload is byte-identical to every other substrate
    /// (that is the wedge); only this container/link differs. The mingw cross
    /// driver (`Target::cc_command`) and the per-target runtime-support shim
    /// (compiled with `int64_t`, since Win64 is LLP64 and `long` is 32-bit) are
    /// selected by the caller (`native_link`) and arrive through `ctx`.
    fn link_coff(&self, ctx: &LinkCtx) -> Result<()> {
        let mut cmd = Command::new(ctx.cc);

        // Runtime-support shim first, so its MIND_EXPORT symbols resolve.
        if let Some(shim) = ctx.runtime_shim {
            cmd.arg(shim);
        }

        // Object files.
        for obj in ctx.objects {
            cmd.arg(obj);
        }

        // Output path (caller has appended `.exe`).
        cmd.arg("-o").arg(ctx.output);

        // MIND runtime library.
        cmd.arg(format!("-L{}", ctx.lib_dir.display()));
        cmd.arg(format!("-l{}", ctx.runtime_link));

        // Standard libraries — Windows/mingw set (NO -ldl; winpthreads + libm +
        // the CNG bcrypt import lib). No ELF `-Wl,-z,*` and no rpath (see above).
        cmd.arg("-lpthread");
        cmd.arg("-lm");
        cmd.arg("-lbcrypt");

        // Optimization flags (same as ELF).
        if ctx.release {
            cmd.arg("-O3");
            cmd.arg("-flto");
        }

        self.run(cmd, ctx)
    }

    /// Mach-O (macOS) executable link.
    ///
    /// deferred: the darwin link uses ld64 (two-level namespace, no `-z` flags,
    /// `@executable_path`-relative rpath) and the Apple cross SDK. Upgrade path:
    /// the darwin slice, gated by the universal OS-keyed byte-identity gate.
    fn link_macho(&self, _ctx: &LinkCtx) -> Result<()> {
        Err(self.not_wired("Mach-O (apple-darwin)"))
    }

    /// Uniform fail-loud error for an un-wired flavor — never a silent no-op or a
    /// wrong-format link.
    fn not_wired(&self, flavor: &str) -> anyhow::Error {
        anyhow!(
            "native link for {flavor} target `{}` is not yet wired; only the host \
             ELF target links today (cross-target codegen lands per {{arch}}x{{os}} \
             incrementally)",
            self.target.llvm_triple()
        )
    }

    /// Run the assembled link command, surfacing a clear error on failure.
    fn run(&self, mut cmd: Command, ctx: &LinkCtx) -> Result<()> {
        if ctx.verbose {
            println!("  Link command: {cmd:?}");
        }
        let status = cmd
            .status()
            .map_err(|e| anyhow!("failed to run linker `{}`: {e}", ctx.cc))?;
        if !status.success() {
            return Err(anyhow!(
                "linking failed with exit code: {:?}",
                status.code()
            ));
        }
        Ok(())
    }
}
