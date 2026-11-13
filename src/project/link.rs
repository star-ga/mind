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
//! is produced, dispatched on the target's `LinkFlavor`.
//!
//! MIND's deterministic *payload* (the emitted `.o` machine code) is held
//! byte-identical across every supported target; only the *container* — ELF vs
//! PE/COFF vs Mach-O — and the flavor-specific link flags differ. This driver is
//! where that container split lives: `LinkDriver::link` matches on
//! `Target::link_flavor` into one private linker invocation per flavor, so a
//! new OS is one new arm, never a scattered `cfg!(target_os)` ladder at the call
//! site.
//!
//! Today only the host ELF arm is wired; the PE/COFF and Mach-O arms fail loud
//! with an explicit "not yet wired" error (the same fail-closed discipline as the
//! `[targets.*].target` seam in [`crate::project`]). Each is turned on behind its
//! own byte-identity gate in a later slice.

use super::runtime_link::RuntimeLink;
use super::{BuildOptions, EmitKind};
use crate::target::{LinkFlavor, Target};
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The base set of the ELF (x86_64-linux-gnu) target: the libraries an extern
/// block with no `bind` clause resolves against, each with the flag that links
/// it (`None`: the C driver links it by default). One table for the list and
/// the flags; `bind` refuses a name in it (spec "Extern bindings", rule B4).
pub(crate) const BASE_LIBRARIES: &[(&str, Option<&str>)] = &[
    ("libdl.so.2", Some("-ldl")),
    ("libpthread.so.0", Some("-lpthread")),
    ("libm.so.6", Some("-lm")),
    ("libc.so.6", None),
];

/// An extern binding the compiler refuses (spec "Extern bindings").
pub(crate) const EXTERN_BINDING_CODE: &str = "E2045";

/// The `bind` clauses of one linked unit (spec "Extern bindings"), gathered by
/// the MLIR lowering of each module the unit compiles: the declared library
/// names, and the called extern names that have the base binding (for E6012).
#[derive(Default)]
struct LinkedUnit {
    libraries: std::collections::BTreeSet<String>,
    base_calls: std::collections::BTreeSet<String>,
}

thread_local! {
    static UNIT: std::cell::RefCell<Option<LinkedUnit>> = const { std::cell::RefCell::new(None) };
}

/// Scope of one linked unit on this thread: it opens empty and the previous
/// unit comes back on drop. Outside a scope nothing is recorded, so state never
/// leaks from one build into the next.
pub struct LinkedUnitScope(Option<LinkedUnit>);

impl LinkedUnitScope {
    /// Open an empty linked unit on this thread.
    pub fn open() -> Self {
        Self(UNIT.with(|u| u.replace(Some(LinkedUnit::default()))))
    }
}

impl Drop for LinkedUnitScope {
    fn drop(&mut self) {
        UNIT.with(|u| *u.borrow_mut() = self.0.take());
    }
}

/// Record the binding of one extern declaration (`called`: one call to it).
#[cfg_attr(
    not(any(feature = "mlir-lowering", feature = "mlir-build")),
    allow(dead_code)
)]
pub(crate) fn record_extern(name: &str, binding: &crate::ast::ExternBinding, called: bool) {
    UNIT.with(|u| {
        let mut unit = u.borrow_mut();
        let Some(unit) = unit.as_mut() else { return };
        match binding {
            crate::ast::ExternBinding::Library(lib) => {
                unit.libraries.insert(lib.clone());
            }
            crate::ast::ExternBinding::Base if called => {
                unit.base_calls.insert(name.to_string());
            }
            _ => {}
        }
    });
}

/// The link arguments of the declared libraries, directly after the base
/// flags (rules B6, B7): libc first so it stays the first DT_NEEDED entry, then
/// each library once by its exact loader name, in ascending byte order. A unit
/// with no library binding gets no argument, so its link line is unchanged.
pub(crate) fn declared_library_args() -> Vec<String> {
    let libraries: Vec<String> = UNIT.with(|u| {
        u.borrow()
            .iter()
            .flat_map(|unit| unit.libraries.iter().cloned())
            .collect()
    });
    if libraries.is_empty() {
        return Vec::new();
    }
    let mut args = vec![
        "-Wl,--push-state,--no-as-needed".to_string(),
        "-lc".to_string(),
    ];
    args.extend(libraries.iter().map(|lib| format!("-l:{lib}")));
    args.push("-Wl,--pop-state".to_string());
    args
}

/// Refuse a declared library on a target with no link row for it (rule B10):
/// `mindc` never drops the clause.
pub(crate) fn refuse_libraries_off_elf(target: &str) -> Result<()> {
    let libraries = declared_library_args();
    if libraries.is_empty() {
        return Ok(());
    }
    Err(anyhow!(
        "error[{}]: target `{target}` has no link row for a `bind(\"<soname>\")` library; \
         the clause is refused, never dropped",
        EXTERN_BINDING_CODE
    ))
}

/// The library arguments of a shared-object link for `target_triple` (host:
/// `None`), or the B10 refusal text for a target with no ELF link row.
#[cfg_attr(not(feature = "mlir-build"), allow(dead_code))]
pub(crate) fn shared_link_args(
    target_triple: Option<&str>,
) -> std::result::Result<Vec<String>, String> {
    match target_triple {
        Some(t) if !t.contains("linux") => refuse_libraries_off_elf(t)
            .map(|()| Vec::new())
            .map_err(|e| e.to_string()),
        _ => Ok(declared_library_args()),
    }
}

/// The error text of a failed clang run: the E6012 text for a failed shared
/// link of a unit that calls base-bound extern names, else the tool's stderr.
#[cfg_attr(not(feature = "mlir-build"), allow(dead_code))]
pub(crate) fn failure_text(shared: bool, stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr).into_owned();
    shared.then(|| e6012(&text)).flatten().unwrap_or(text)
}

/// The E6012 text of a failed link, or `None` when the unit calls no extern
/// name with the base binding (the plain linker error is then the whole story).
pub(crate) fn e6012(linker_text: &str) -> Option<String> {
    let names: Vec<String> = UNIT.with(|u| {
        u.borrow()
            .iter()
            .flat_map(|unit| unit.base_calls.iter().cloned())
            .collect()
    });
    if names.is_empty() {
        return None;
    }
    Some(format!(
        "error[E6012]: the link failed. The unit calls these extern names with the base \
         binding: {}. A name from another library needs `extern \"C\" bind(\"<soname>\")` \
         on its block.\n{}",
        names.join(", "),
        linker_text.trim_end()
    ))
}

/// Everything the final link needs, gathered by the caller so the driver itself
/// makes no build-orchestration decisions (it only assembles + runs the linker).
pub struct LinkCtx<'a> {
    /// C compiler / linker driver command (host builds: `$CC` or `cc`; a cross
    /// target selects its own toolchain in a later slice).
    pub cc: &'a str,
    /// Object files to link, in declared order.
    pub objects: &'a [PathBuf],
    /// Output path: the executable, or the relocatable object for `--emit object`.
    pub output: &'a Path,
    /// Installed runtime for accelerator backends. Native CPU objects use the
    /// public runtime-support shim and carry no private runtime dependency.
    pub runtime: Option<&'a RuntimeLink>,
    /// Build options: `release` adds `-O3 -flto`, `verbose` prints the link
    /// command, and `emit` selects an executable or a relocatable object.
    pub opts: &'a BuildOptions,
    /// Pre-compiled runtime-support shim object (compiled by the caller through
    /// the MLIR toolchain, fixed basename + `-ffile-prefix-map`, no debug info),
    /// linked FIRST so the `MIND_EXPORT` intrinsics resolve statically. `None`
    /// when the shim is unavailable (e.g. a `--no-default-features` build without
    /// `mlir-build`) — byte-identical to the historical no-shim link.
    pub runtime_shim: Option<&'a Path>,
}

/// Produces the final native executable for one `Target`.
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

    /// Link the output: a relocatable object for `--emit object`, otherwise the
    /// final executable, dispatching on the target's object/link format.
    pub fn link(&self, ctx: &LinkCtx) -> Result<()> {
        // `--emit object` stops before the executable link: one relocatable
        // object with the compiled modules, the substrate archive members they
        // use and the runtime-support shim, for a host program to link. The
        // executable link used to run here too, so the `.o` was a linked PIE.
        if ctx.opts.emit == EmitKind::Object {
            return link_relocatable(ctx);
        }
        match self.target.link_flavor() {
            LinkFlavor::Elf => self.link_elf(ctx),
            LinkFlavor::Coff => self.link_coff(ctx),
            LinkFlavor::MachO => self.link_macho(ctx),
        }
    }

    /// ELF (Linux) executable link.
    ///
    /// Runtime-backed targets preserve the historical order: shim → objects →
    /// `-o` → `-L`/`-l` → system libraries → hardening → rpath → optimization.
    /// The public CPU path omits only the installed-runtime and rpath arguments;
    /// its MIND intrinsics resolve from the shim linked first.
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

        if let Some(runtime) = ctx.runtime {
            cmd.arg(format!("-L{}", runtime.dir.display()));
            cmd.arg(format!("-l{}", runtime.link_name));
        }

        // The base set (one table), then the declared libraries.
        cmd.args(BASE_LIBRARIES.iter().filter_map(|(_, flag)| *flag));
        cmd.args(declared_library_args());

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

        if let Some(runtime) = ctx.runtime {
            cmd.arg(format!("-Wl,-rpath,{}", runtime.dir.display()));
            cmd.arg("-Wl,-rpath,$ORIGIN/../lib");
        }

        // Optimization flags.
        if ctx.opts.release {
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
        refuse_libraries_off_elf(self.target.llvm_triple())?;
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

        if let Some(runtime) = ctx.runtime {
            cmd.arg(format!("-L{}", runtime.dir.display()));
            cmd.arg(format!("-l{}", runtime.link_name));
        }

        // Standard libraries — Windows/mingw set (NO -ldl; winpthreads + libm +
        // the CNG bcrypt import lib). No ELF `-Wl,-z,*` and no rpath (see above).
        cmd.arg("-lpthread");
        cmd.arg("-lm");
        cmd.arg("-lbcrypt");

        // Optimization flags (same as ELF).
        if ctx.opts.release {
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
        refuse_libraries_off_elf(self.target.llvm_triple())?;
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
        if ctx.opts.verbose {
            println!("  Link command: {cmd:?}");
        }
        let out = cmd
            .output()
            .map_err(|e| anyhow!("failed to run linker `{}`: {e}", ctx.cc))?;
        let text = String::from_utf8_lossy(&out.stderr);
        if !out.status.success() {
            let _ = std::fs::remove_file(ctx.output);
            return Err(match e6012(&text) {
                Some(msg) => anyhow!(msg),
                None => anyhow!(
                    "linking failed with exit code: {:?}\n{}",
                    out.status.code(),
                    text.trim_end()
                ),
            });
        }
        eprint!("{text}");
        Ok(())
    }
}

/// Pack `ctx.objects` and the runtime-support shim into one relocatable object
/// (`cc -r -nostdlib`). No startup files and no libc: the host program's own
/// link supplies those.
fn link_relocatable(ctx: &LinkCtx) -> Result<()> {
    let mut cmd = Command::new(ctx.cc);
    cmd.args(["-r", "-nostdlib", "-o"])
        .arg(ctx.output)
        .args(ctx.objects);
    if let Some(shim) = ctx.runtime_shim {
        cmd.arg(shim);
    }
    if ctx.opts.verbose {
        println!("  Relocatable link: {cmd:?}");
    }
    let out = cmd
        .output()
        .map_err(|e| anyhow!("spawn `{}` for the relocatable object link: {e}", ctx.cc))?;
    if !out.status.success() {
        return Err(anyhow!(
            "relocatable object link failed ({} -r): {}",
            ctx.cc,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}
