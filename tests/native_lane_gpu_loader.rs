// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! The native-lane GPU loader (ruling 16b): `DT_NEEDED libcuda.so.1` next to
//! `libc.so.6`, unversioned `_v2` driver symbols through eagerly bound GOT
//! slots, pointer-to-pointer and `*mut i64` out-params, a `void**` argument
//! array, and the 11-argument `cuLaunchKernel` row whose parameters 6..10
//! travel on the SysV stack.
//!
//! The ELF contract is asserted by parsing the built image here (the same
//! facts `readelf -d -r` reports: NEEDED entries, version tags, GLOB_DAT
//! relocations by symbol name); the call shapes run against the test-only
//! fake driver library (`fake_driver.c`, compiled with cc for this fixture
//! only), identically on the native and the MLIR backend; and each new guard
//! has a mutant minted from a mutated `main.mind` that must turn red.
//!
//! Libc-only images are compared to a stage1 minted from the nearest ancestor
//! whose `main.mind` does not name `libcuda.so.1`, not to another image from
//! this checkout. The 11-argument alignment probe (`align_driver.c`) returns
//! its frame nibble, so a missing pad or a cleanup that drops that pad fails.

#![cfg(all(feature = "std-surface", target_os = "linux", target_arch = "x86_64"))]

mod native_lane_extern_support;

use native_lane_extern_support::{
    Case, MINDC, build_align_libcuda, build_apart, build_fake_libcuda, builds_apart, call_cleanups,
    clang_shim, compiler_library, compiler_source, exit_code_or_signal, extern_corpus, gpu_fixture,
    mc6_cases, memset_program, mint_library, mint_stage1, repo, run_with_env, self_host_refusal,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// One built image parsed as ELF64: PT_INTERP, the dynamic entries, and the
/// GLOB_DAT relocations with their symbol names.
struct Elf<'a> {
    image: &'a [u8],
}

impl Elf<'_> {
    fn new(image: &[u8]) -> Elf<'_> {
        assert_eq!(&image[..4], b"\x7fELF", "not an ELF image");
        Elf { image }
    }

    fn word(&self, at: usize, n: usize) -> usize {
        self.image[at..at + n]
            .iter()
            .rev()
            .fold(0usize, |acc, b| (acc << 8) | usize::from(*b))
    }

    /// `(kind, offset, vaddr, filesz)` of every program header.
    fn phdrs(&self) -> Vec<(usize, usize, usize, usize)> {
        let (phoff, phentsize, phnum) = (self.word(32, 8), self.word(54, 2), self.word(56, 2));
        (0..phnum)
            .map(|k| {
                let ph = phoff + k * phentsize;
                (
                    self.word(ph, 4),
                    self.word(ph + 8, 8),
                    self.word(ph + 16, 8),
                    self.word(ph + 32, 8),
                )
            })
            .collect()
    }

    /// File offset of a virtual address, through its PT_LOAD mapping.
    fn off(&self, vaddr: usize) -> usize {
        for (_, off, va, filesz) in self.phdrs() {
            if vaddr >= va && vaddr < va + filesz {
                return vaddr - va + off;
            }
        }
        panic!("vaddr {vaddr:#x} is in no PT_LOAD");
    }

    /// The dynamic entries `(tag, value)` of PT_DYNAMIC, in order.
    fn dynamic(&self) -> Vec<(usize, usize)> {
        let (off, filesz) = self
            .phdrs()
            .into_iter()
            .find(|(kind, _, _, _)| *kind == 2)
            .map(|(_, off, _, filesz)| (off, filesz))
            .expect("PT_DYNAMIC");
        (0..filesz / 16)
            .map(|i| (self.word(off + i * 16, 8), self.word(off + i * 16 + 8, 8)))
            .take_while(|(tag, _)| *tag != 0)
            .collect()
    }

    fn dyn_value(&self, tag: usize) -> Option<usize> {
        self.dynamic()
            .iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, v)| *v)
    }

    /// The DT_NEEDED strings, in entry order.
    fn needed(&self) -> Vec<String> {
        let strtab = self.off(self.dyn_value(5).expect("DT_STRTAB"));
        self.dynamic()
            .iter()
            .filter(|(t, _)| *t == 1)
            .map(|(_, v)| {
                let at = strtab + v;
                let end = self.image[at..]
                    .iter()
                    .position(|b| *b == 0)
                    .expect("NUL-terminated NEEDED string");
                String::from_utf8(self.image[at..at + end].to_vec()).unwrap()
            })
            .collect()
    }

    /// The symbol names of every R_X86_64_GLOB_DAT (6) relocation.
    fn glob_dat_symbols(&self) -> Vec<String> {
        let strtab = self.off(self.dyn_value(5).expect("DT_STRTAB"));
        let symtab = self.off(self.dyn_value(6).expect("DT_SYMTAB"));
        let rela = self.off(self.dyn_value(7).expect("DT_RELA"));
        let relasz = self.dyn_value(8).expect("DT_RELASZ");
        (0..relasz / 24)
            .map(|i| {
                let at = rela + i * 24;
                let info = self.word(at + 8, 8);
                (self.word(at, 8), info >> 32, info & 0xffff_ffff)
            })
            .filter(|(_, _, kind)| *kind == 6)
            .filter(|(_, sym, _)| *sym != 0)
            .map(|(_, sym, _)| {
                let name_at = strtab + self.word(symtab + sym * 24, 4);
                let end = self.image[name_at..]
                    .iter()
                    .position(|b| *b == 0)
                    .expect("NUL-terminated symbol name");
                String::from_utf8(self.image[name_at..name_at + end].to_vec()).unwrap()
            })
            .collect()
    }

    /// filesz of PT_DYNAMIC: the dynamic segment the writer sized.
    fn dynamic_filesz(&self) -> usize {
        self.phdrs()
            .into_iter()
            .find(|(kind, _, _, _)| *kind == 2)
            .map(|(_, _, _, filesz)| filesz)
            .expect("PT_DYNAMIC")
    }

    fn interp(&self) -> Vec<u8> {
        let (off, filesz) = self
            .phdrs()
            .into_iter()
            .find(|(kind, _, _, _)| *kind == 3)
            .map(|(_, off, _, filesz)| (off, filesz))
            .expect("PT_INTERP");
        self.image[off..off + filesz].to_vec()
    }
}

const INTERP: &[u8] = b"/lib64/ld-linux-x86-64.so.2\0";

/// The mixed import (glibc getuid + every driver row): DT_NEEDED carries
/// libcuda.so.1 next to libc.so.6, the version tables stay (a glibc symbol
/// is imported), the dynamic segment grows past its old fixed 256 bytes, the
/// GOT holds every called driver symbol under R_X86_64_GLOB_DAT, and
/// PT_INTERP is the fixed loader path.
#[test]
fn the_mixed_driver_import_carries_both_needed_entries() {
    let case = Case::new(true);
    let bin = case.builds(&gpu_fixture("vec_add.mind"));
    // Keep the built image for the host-leg gates (readelf -d -r, the GPU
    // run) under the per-target scratch directory, overwriting any earlier
    // copy: the build is deterministic, so the bytes are the same.
    let keep = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gpu_lane_vec_add.elf");
    std::fs::copy(&bin, &keep).unwrap();
    let image = std::fs::read(&bin).unwrap();
    let elf = Elf::new(&image);
    assert_eq!(
        elf.needed(),
        vec!["libc.so.6".to_string(), "libcuda.so.1".to_string()],
        "DT_NEEDED must list libc.so.6 then libcuda.so.1"
    );
    assert_eq!(elf.interp(), INTERP);
    for tag in [1879048176, 1879048190, 1879048191] {
        assert!(elf.dyn_value(tag).is_some(), "version tag {tag} missing");
    }
    assert!(
        elf.dynamic_filesz() > 256,
        "the dynamic segment must grow past its fixed 256 bytes, got {}",
        elf.dynamic_filesz()
    );
    assert_eq!(elf.dynamic_filesz(), 272);
    let syms = elf.glob_dat_symbols();
    let got: Vec<&str> = syms.iter().map(|s| s.as_str()).collect();
    for name in [
        "getuid",
        "cuInit",
        "cuDeviceGet",
        "cuCtxCreate_v2",
        "cuCtxSynchronize",
        "cuModuleLoadData",
        "cuModuleGetFunction",
        "cuMemAlloc_v2",
        "cuMemcpyHtoD_v2",
        "cuLaunchKernel",
        "cuMemFree_v2",
        "cuModuleUnload",
        "cuCtxDestroy_v2",
    ] {
        assert!(got.contains(&name), "{name} has no GLOB_DAT relocation");
    }
}

/// The libcuda-only import: no imported symbol carries a glibc version, so
/// the writer emits neither the versym/verneed tables nor their three
/// dynamic tags, DT_NEEDED lists libcuda.so.1 only (no libc row is bound),
/// and the dynamic segment shrinks to 13 entries (208 bytes).
#[test]
fn the_libcuda_only_import_emits_no_version_tables() {
    let case = Case::new(true);
    let bin = case.builds(&gpu_fixture("gpu_only.mind"));
    let image = std::fs::read(&bin).unwrap();
    let elf = Elf::new(&image);
    assert_eq!(elf.needed(), vec!["libcuda.so.1".to_string()]);
    for tag in [1879048176, 1879048190, 1879048191] {
        assert!(
            elf.dyn_value(tag).is_none(),
            "version tag {tag} present on a libcuda-only import"
        );
    }
    assert_eq!(elf.dynamic_filesz(), 208);
    let got = elf.glob_dat_symbols();
    assert_eq!(
        got,
        vec!["cuInit".to_string(), "cuCtxSynchronize".to_string()]
    );
}

/// The fake driver library: the out-param shapes, the void** arrays and the
/// 11-argument call run identically on the native and the MLIR backend and
/// exit 42, every stack argument contributing to the checked result.
#[test]
fn the_fake_driver_library_runs_alike_on_both_backends() {
    runs_alike_on_both_backends("fake_rows.mind");
}

/// The MC-6 rows (each out-param, CUDA_MEMCPY2D, the byte argument, the 10-arg
/// launch): native == MLIR == 42 from a compiler carrying nbcuda02, the same
/// image from a second build directory, each row one GLOB_DAT import (long
/// names in full), BIND_NOW, no version tags, and the NEEDED list and dynamic
/// size of the other libcuda-only image (gpu_only.mind).
#[test]
fn the_mc6_driver_rows_run_alike_and_import_once_each() {
    let image = runs_alike_on_both_backends("fake_rows2.mind");
    let case = Case::new(true);
    let compiler = std::fs::read(case.compiler()).unwrap();
    let marked = compiler.windows(10).any(|w| w == b"\x48\xb8nbcuda02");
    assert!(marked, "the compiler carries no nbcuda02 mark");
    let again = std::fs::read(build_apart(&case, &gpu_fixture("fake_rows2.mind")).1).unwrap();
    assert!(
        again == image,
        "the fake_rows2.mind image depends on its build directory"
    );
    let only = std::fs::read(case.builds(&gpu_fixture("gpu_only.mind"))).unwrap();
    let (elf, only) = (Elf::new(&image), Elf::new(&only));
    assert_eq!(elf.needed(), only.needed());
    assert_eq!(elf.dynamic_filesz(), only.dynamic_filesz());
    for tag in [1879048176, 1879048190, 1879048191] {
        assert!(elf.dyn_value(tag).is_none(), "version tag {tag} present");
    }
    assert_eq!(elf.dyn_value(30), Some(8), "DT_FLAGS must be BIND_NOW");
    assert_eq!(elf.dyn_value(1879048187), Some(1), "DT_FLAGS_1 must be NOW");
    let (mut got, mut want) = (elf.glob_dat_symbols(), mc6_cases("row").concat());
    assert_eq!(want.len(), 33, "mc6_cases.tsv lists every MC-6 row");
    got.sort();
    want.sort();
    assert_eq!(got, want, "each MC-6 row must be one GLOB_DAT import");
}

/// Runs `fixture` natively and through MLIR on the fake driver: the native image.
fn runs_alike_on_both_backends(fixture: &str) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let lib = build_fake_libcuda(dir.path());
    let program = gpu_fixture(fixture);

    // Native: the image asks for libcuda.so.1 by soname, so the loader finds
    // the fixture under that filename through LD_LIBRARY_PATH.
    let case = Case::new(true);
    let native_bin = case.builds(&program);
    let ld = format!("{}", dir.path().display());
    let native = run_with_env(&native_bin, &[("LD_LIBRARY_PATH", ld.as_str())]);

    // MLIR: the build resolves CLANG for the IR compile and CC for the
    // executable link; the fixture-only shim appends the fake library to that
    // link, so the cu* declarations resolve and the linked binary records the
    // library path as its NEEDED entry.
    let shim = clang_shim(dir.path(), &lib);
    let mlir_dir = tempfile::tempdir().unwrap();
    let source_path = mlir_dir.path().join("main.mind");
    let artifact = mlir_dir.path().join("program");
    std::fs::write(&source_path, &program).unwrap();
    let built = Command::new(MINDC)
        .args(["build", "--backend", "mlir"])
        .arg(&source_path)
        .arg("--emit=binary")
        .arg(format!("--out={}", artifact.display()))
        .env("MINDC_STD_DIR", repo().join("std"))
        .env("CLANG", &shim)
        .env("CC", &shim)
        .env_remove("MINDC_NATIVE_ELF")
        .env_remove("MINDC_SO")
        .current_dir(repo())
        .output()
        .unwrap();
    assert_eq!(
        built.status.code(),
        Some(0),
        "the MLIR build failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let mlir = run_with_env(&artifact, &[]);

    assert_eq!(native, mlir, "native != MLIR on {fixture}");
    assert_eq!(native.0, 42, "a fake-driver step of {fixture} failed");
    std::fs::read(&native_bin).unwrap()
}

/// A stage1 minted from this checkout with every driver row deleted (the
/// thirteen of `nbx_row` and the MC-6 block of `nbcu_row`) still emits
/// byte-identical ELF for every program that imports none of them (the
/// libc-only corpus and the no-extern programs): the row data is inert for a
/// libc-only import, exactly the ruling's "identical_pairs stays all equal" force.
#[test]
fn removing_the_driver_rows_keeps_libc_only_bytes() {
    let rows_removed = Case::with_compiler(mint_library(&library_without_driver_rows()));
    let fresh = Case::new(true);
    for (name, src, _) in identity_programs() {
        let a = std::fs::read(fresh.builds(&src)).unwrap();
        let b = std::fs::read(rows_removed.builds(&src)).unwrap();
        assert!(
            a == b,
            "{name}: a program with no libcuda import changed its ELF bytes"
        );
    }
}

/// This checkout's compiler library with its 46 driver rows deleted: the
/// compiler before the driver rows, as a property of the source.
fn library_without_driver_rows() -> String {
    let source = compiler_library();
    let lines: Vec<&str> = source.lines().collect();
    let (mut kept, mut removed, mut k) = (Vec::new(), 0, 0);
    while k < lines.len() {
        // Only the `nbx_row` / `nbcu_row` rows, in the `mindc fmt` layout:
        // `if i == N {`, `return nbx_put(..);` (or `nbcu_long`), `}`. The other
        // `if i == N {` shapes (the SHA-256 constant table) are not rows and
        // must survive.
        let head = lines[k].trim_start();
        let call = lines.get(k + 1).map_or("", |l| l.trim_start());
        let row = head.starts_with("if i == ")
            && head.ends_with('{')
            && (call.contains("_put(out, ") || call.contains("nbcu_long(out, "))
            && lines.get(k + 2).is_some_and(|l| l.trim() == "}");
        let idx: i64 = head
            .strip_prefix("if i == ")
            .and_then(|t| t.split('{').next())
            .and_then(|t| t.trim().parse().ok())
            .unwrap_or(-1);
        if row && (41..=86).contains(&idx) {
            removed += 1;
            k += 3;
            continue;
        }
        kept.push(lines[k]);
        k += 1;
    }
    assert_eq!(removed, 46, "13 nbx_row driver rows + 33 MC-6 rows");
    kept.join("\n") + "\n"
}

/// How a mutant must fail: a build refusal (the MIND compiler rejects the
/// program), a wrong run against the fake driver library, or an image whose
/// parsed ELF contract no longer holds.
enum Red {
    Refused,
    RunNot42,
    NeededMissing,
    VersymPresent,
    DynszWrong(usize),
}

/// Writer and lowering mutants of the GPU loader guards. Each is minted into
/// its own stage1 from this checkout and must fail exactly the guard it
/// attacks; the fresh compiler builds and passes the same program.
#[test]
fn gpu_loader_mutants_turn_their_guards_red() {
    let source = compiler_source();
    let versym_block = "    if nv > 0 {\n        out = nbx_dyn(out, 1879048176, mv + \
__mind_load_i64(w + 32));\n        out = nbx_dyn(out, 1879048190, mv + __mind_load_i64(w + \
40));\n        out = nbx_dyn(out, 1879048191, 1);\n    }\n";
    assert_eq!(
        source.matches(versym_block).count(),
        1,
        "the tag block moved"
    );
    let needed_block =
        "    if has_cu == 1 {\n        out = nbx_dyn(out, 1, __mind_load_i64(w + 48));\n    }\n";
    assert_eq!(
        source.matches(needed_block).count(),
        1,
        "the DT_NEEDED libcuda emission moved"
    );
    let mutants: Vec<(&str, Red, String, &str)> = vec![
        (
            "name-truncation",
            Red::Refused,
            source.replacen(
                "if i == 47 {\n        return nbx_put(out, 19, 7308345369539999075, \
8386668381066257735, 7237481, 0, 0, 0, 67438849, 0, 3);\n    }",
                "if i == 47 {\n        return nbx_put(out, 19, 7308345369539999075, \
8386668381066257735, 0, 0, 0, 0, 67438849, 0, 3);\n    }",
                1,
            ),
            "cuModuleGetFunction must not bind through a truncated 16-byte name",
        ),
        (
            "arity-cap-back-to-six",
            Red::Refused,
            source.replacen(
                "pub fn nbx_row_sig(row: i64) -> i64 {\n    let n: i64 = __mind_load_i64(row + \
40);\n    if n > 11 {\n        return 0 - 1;\n    }",
                "pub fn nbx_row_sig(row: i64) -> i64 {\n    let n: i64 = __mind_load_i64(row + \
40);\n    if n > 6 {\n        return 0 - 1;\n    }",
                1,
            ),
            "the 11-argument row must pass the raised arity cap",
        ),
        (
            "overflow-push-order",
            Red::RunNot42,
            source
                .replacen(
                    "pub fn nbx_push_overflow(code: String, ids: i64, n: i64, j: i64) -> String \
{\n    if j < 6 {\n        return code;\n    }",
                    "pub fn nbx_push_overflow(code: String, ids: i64, n: i64, j: i64) -> String \
{\n    if j > n - 1 {\n        return code;\n    }",
                    1,
                )
                .replacen(
                    "nbx_push_overflow(nb_push(c1, 80), ids, n, j - 1)",
                    "nbx_push_overflow(nb_push(c1, 80), ids, n, j + 1)",
                    1,
                )
                .replacen(
                    "let p2: String = nbx_push_overflow(p1, ids, n, n - 1);",
                    "let p2: String = nbx_push_overflow(p1, ids, n, 6);",
                    1,
                ),
            "the stack arguments must arrive right-to-left",
        ),
        (
            "needed-entry-dropped",
            Red::NeededMissing,
            source.replacen(needed_block, "", 1),
            "DT_NEEDED libcuda.so.1 must be emitted next to libc.so.6",
        ),
        (
            "versym-tags-unconditional",
            Red::VersymPresent,
            source.replacen(
                versym_block,
                "    out = nbx_dyn(out, 1879048176, mv + __mind_load_i64(w + 32));\n    out = \
nbx_dyn(out, 1879048190, mv + __mind_load_i64(w + 40));\n    out = nbx_dyn(out, 1879048191, \
1);\n",
                1,
            ),
            "a libcuda-only import must emit no version tags",
        ),
        (
            "dynamic-size-frozen-at-256",
            Red::DynszWrong(208),
            source.replacen(
                "    let dynsz: i64 = 16 * (12 + nbe_needed_libc(nbx_vers_tags(nv), has_cu) + has_cu + 3 * nbx_vers_tags(nv));",
                "    let dynsz: i64 = 256;",
                1,
            ),
            "the dynamic segment must be sized, not fixed at 256 bytes",
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    let _lib = build_fake_libcuda(dir.path());
    let ld = format!("{}", dir.path().display());
    let program = gpu_fixture("fake_rows.mind");
    let fresh = Case::new(true);
    assert_eq!(
        run_with_env(&fresh.builds(&program), &[("LD_LIBRARY_PATH", ld.as_str())]).0,
        42,
        "the fresh compiler must build and run the fake-driver program"
    );
    let minted: Vec<PathBuf> = std::thread::scope(|scope| {
        let jobs: Vec<_> = mutants
            .iter()
            .map(|(_, _, text, _)| {
                let mutated = text.clone();
                scope.spawn(move || mint_stage1(&mutated))
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    for ((name, red, _, why), compiler) in mutants.iter().zip(minted) {
        let mutant = Case::with_compiler(compiler);
        let (build, binary) = mutant.build(&program);
        let stderr = String::from_utf8_lossy(&build.stderr);
        let killed = match red {
            Red::Refused => {
                build.status.code() == Some(3)
                    && stderr.contains("pure-MIND compiler rejected the program (exit 3)")
                    && !binary.exists()
            }
            Red::RunNot42 => {
                // A signal stays a signal. It fails this guard without becoming
                // an exit code that another crashed run could compare equal to.
                if build.status.code() != Some(0) {
                    false
                } else {
                    let outcome = exit_code_or_signal(&binary, &[("LD_LIBRARY_PATH", ld.as_str())]);
                    println!("gpu mutant {name} outcome={outcome:?}");
                    outcome.ok() != Some(42)
                }
            }
            Red::NeededMissing => {
                build.status.code() == Some(0)
                    && !Elf::new(&std::fs::read(&binary).unwrap())
                        .needed()
                        .iter()
                        .any(|n| n == "libcuda.so.1")
            }
            Red::VersymPresent => {
                build.status.code() == Some(0)
                    && Elf::new(&std::fs::read(&binary).unwrap())
                        .dyn_value(1879048176)
                        .is_some()
            }
            Red::DynszWrong(want) => {
                build.status.code() == Some(0)
                    && Elf::new(&std::fs::read(&binary).unwrap()).dynamic_filesz() != *want
            }
        };
        assert!(killed, "{name} survived: {why}\n{stderr}");
        println!("gpu mutant killed: {name}");
    }
}

/// The eight libc-only extern-corpus programs, plus a program with no extern
/// and one whose only call is a 9-argument local call.
fn identity_programs() -> Vec<(&'static str, String, bool)> {
    let mut out: Vec<(&'static str, String, bool)> = extern_corpus()
        .into_iter()
        .filter(|(name, _, _)| !name.starts_with("gpu_"))
        .map(|(name, src, _)| (name, src, true))
        .collect();
    assert_eq!(out.len(), 8, "the libc-only extern corpus changed size");
    out.push((
        "no_extern",
        "fn main() -> i64 {\n    return 1;\n}\n".to_string(),
        false,
    ));
    out.push((
        "no_extern_nine",
        "fn add9(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64, h: i64, i: i64) -> i64 {\n    \
         return a + b + c + d + e + f + g + h + i;\n}\nfn main() -> i64 {\n    \
         return add9(1, 2, 3, 4, 5, 6, 7, 8, 9);\n}\n"
            .to_string(),
        false,
    ));
    out
}

/// Head's libc-only and no-extern images match the compiler without the
/// driver rows. Always emitting the `libcuda.so.1` string (the guard's
/// mutant) moves every libc-only image and leaves the two no-extern images.
/// The base is this checkout with the rows deleted, not a revision: a
/// squashed history has no commit with the extern lane and without the rows.
#[test]
fn libc_only_bytes_match_the_compiler_before_the_driver_rows() {
    let base_lib = library_without_driver_rows();
    let head_src = compiler_source();
    assert!(head_src.contains("libcuda.so.1"));
    let from = "    if has_cu == 1 {\n        s = nbx_ztext(s, cu, 12);\n    }\n";
    assert_eq!(
        head_src.matches(from).count(),
        1,
        "the libcuda string gate moved"
    );
    let mutant_src = head_src.replacen(from, "    s = nbx_ztext(s, cu, 12);\n", 1);
    let (base_c, head_c, mut_c) = std::thread::scope(|scope| {
        let head_src = head_src.clone();
        let mutant_src = mutant_src.clone();
        let b = scope.spawn(move || mint_library(&base_lib));
        let h = scope.spawn(move || mint_stage1(&head_src));
        let m = scope.spawn(move || mint_stage1(&mutant_src));
        (b.join().unwrap(), h.join().unwrap(), m.join().unwrap())
    });
    let base_case = Case::with_compiler(base_c);
    let head_case = Case::with_compiler(head_c);
    let mut_case = Case::with_compiler(mut_c);
    let mut libc_moved = 0;
    for (name, src, libc_only) in identity_programs() {
        let base_img = std::fs::read(base_case.builds(&src)).unwrap();
        let head_img = std::fs::read(head_case.builds(&src)).unwrap();
        assert_eq!(
            base_img, head_img,
            "{name}: head moved an image the driver rows must not touch"
        );
        let mutant_img = std::fs::read(mut_case.builds(&src)).unwrap();
        if libc_only {
            assert_ne!(
                mutant_img, base_img,
                "{name}: unconditional libcuda string left the libc-only image unchanged"
            );
            libc_moved += 1;
        } else {
            assert_eq!(
                mutant_img, base_img,
                "{name}: a no-extern image changed under the string mutant"
            );
        }
        println!("libc-only identity {name} head=base mutant_moved={libc_only}");
    }
    assert_eq!(libc_moved, 8);
}

/// The odd-count pad and the cleanup that puts it back. Both mutants keep a
/// checksum-only fake driver at exit 42; the frame-nibble probe does not.
#[test]
fn stack_pad_and_cleanup_kill_their_mutants() {
    let source = compiler_source();
    let pad_from = "let pad: i64 = k - (k / 2) * 2;\n    let p1: String = nbx_stack_pad(code, k);";
    let clean_from = "nb_u32(c3, 8 * k + pad * 8)";
    assert_eq!(source.matches(pad_from).count(), 1, "the pad binding moved");
    assert_eq!(source.matches(clean_from).count(), 1, "the cleanup moved");
    let mutants = [
        (
            "no-alignment-pad",
            source.replacen(
                pad_from,
                "let pad: i64 = 0;\n    let p1: String = nbx_stack_pad(code, 0);",
                1,
            ),
        ),
        (
            "cleanup-drops-pad",
            source.replacen(clean_from, "nb_u32(c3, 8 * k)", 1),
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    let _lib = build_align_libcuda(dir.path());
    let ld = format!("{}", dir.path().display());
    let program = gpu_fixture("align_probe.mind");
    let fresh = Case::new(true);
    let head = run_with_env(&fresh.builds(&program), &[("LD_LIBRARY_PATH", ld.as_str())]);
    assert_eq!(head.0, 42, "head alignment probe exit {}", head.0);
    println!("alignment head exit={}", head.0);
    let minted: Vec<PathBuf> = std::thread::scope(|scope| {
        let jobs: Vec<_> = mutants
            .iter()
            .map(|(_, text)| {
                let mutated = text.clone();
                scope.spawn(move || mint_stage1(&mutated))
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    for ((name, _), compiler) in mutants.iter().zip(minted) {
        let mutant = Case::with_compiler(compiler);
        let (build, binary) = mutant.build(&program);
        let stderr = String::from_utf8_lossy(&build.stderr);
        assert_eq!(
            build.status.code(),
            Some(0),
            "{name} did not build:\n{stderr}"
        );
        let got = run_with_env(&binary, &[("LD_LIBRARY_PATH", ld.as_str())]);
        assert_ne!(got.0, 42, "{name} survived with exit {}", got.0);
        println!("alignment mutant killed: {name} exit={}", got.0);
    }
}

/// cuMemsetD8_v2's byte argument: a 0..=255 decimal literal or literal mask
/// builds, any other is refused by name (`extern.byte_arg_unproven`, the
/// argument's text); off-table names refuse by name (unknown).
#[test]
fn the_byte_argument_fence_and_off_table_names_refuse() {
    let case = Case::new(true);
    for arg in mc6_cases("admit") {
        let built = builds_apart(&case, &memset_program(&arg[0]));
        assert!(built, "{} refused", arg[0]);
    }
    for arg in mc6_cases("refuse") {
        let named = format!("error[extern]: extern.byte_arg_unproven: {}\n", arg[0]);
        let stderr = self_host_refusal(&case, &memset_program(&arg[0]));
        let fenced = stderr.is_some_and(|e| e.contains(&named));
        assert!(fenced, "{} not fenced", arg[0]);
    }
    for name in mc6_cases("unknown").concat() {
        let source =
            format!("extern \"C\" {{ fn {name}() -> i32; }}\nfn main() -> i64 {{ return 0; }}");
        let (build, binary) = build_apart(&case, &source);
        let stderr = String::from_utf8_lossy(&build.stderr);
        let refused = build.status.code() == Some(3) && !binary.exists();
        assert!(refused, "{name}: {stderr}");
        let named = stderr.contains("lane.extern.unknown_symbol") && stderr.contains(name.as_str());
        assert!(named, "{stderr}");
    }
}

/// MC-6 guards (`mc6_cases.tsv` mutants): long-name pointer (lookup, strtab),
/// 128-row scan, byte fence (each refused argument then builds), even-count
/// no-pad of the 10-argument row. The fresh compiler's driver calls drop
/// exactly their stack block (`tail` rows: add rsp right after the call).
#[test]
fn mc6_mutants_turn_their_guards_red() {
    let source = compiler_library();
    let mutants = mc6_cases("mutant");
    assert_eq!(mutants.len(), 5);
    let (fake, align) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let _lib = build_fake_libcuda(fake.path());
    let _alib = build_align_libcuda(align.path());
    let fake_ld = [("LD_LIBRARY_PATH", fake.path().to_str().unwrap())];
    let align_ld = [("LD_LIBRARY_PATH", align.path().to_str().unwrap())];
    let rows = gpu_fixture("fake_rows2.mind");
    let coop = gpu_fixture("align_probe_coop.mind");
    let fresh = Case::new(true);
    assert_eq!(run_with_env(&fresh.builds(&rows), &fake_ld).0, 42);
    let coop_bin = fresh.builds(&coop);
    assert_eq!(run_with_env(&coop_bin, &align_ld).0, 42);
    let coop_image = std::fs::read(&coop_bin).unwrap();
    for t in mc6_cases("tail") {
        let (imm, n): (u32, usize) = (t[0].parse().unwrap(), t[1].parse().unwrap());
        let got = call_cleanups(&coop_image, imm);
        assert_eq!(got, n, "calls dropping {imm} bytes");
    }
    let minted: Vec<PathBuf> = std::thread::scope(|scope| {
        let jobs: Vec<_> = mutants
            .iter()
            .map(|m| {
                assert_eq!(source.matches(&m[2]).count(), 1, "{} moved", m[2]);
                let text = source.replacen(&m[2], &m[3], 1);
                scope.spawn(move || mint_library(&text))
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    for (m, compiler) in mutants.iter().zip(minted) {
        let mutant = Case::with_compiler(compiler);
        // Built, then any outcome but exit 42: a loader error (exit 127 on
        // stderr, the mangled-name mutant) or a signal also fails the guard.
        let run_not_42 = |program: &str, ld: &[(&str, &str)]| {
            let (build, binary) = build_apart(&mutant, program);
            let mut run = Command::new(&binary);
            let ran = (build.status.code() == Some(0))
                .then(|| run.envs(ld.iter().copied()).stdin(Stdio::null()).output());
            ran.is_some_and(|r| r.unwrap().status.code() != Some(42))
        };
        let killed = match m[1].as_str() {
            "refused" => self_host_refusal(&mutant, &rows).is_some(),
            "run_rows" => run_not_42(&rows, &fake_ld),
            "run_coop" => run_not_42(&coop, &align_ld),
            "builds_refused" => mc6_cases("refuse")
                .iter()
                .all(|a| builds_apart(&mutant, &memset_program(&a[0]))),
            other => panic!("unknown red kind {other}"),
        };
        assert!(killed, "MC-6 mutant {} survived", m[0]);
        println!("MC-6 mutant killed: {}", m[0]);
    }
}
