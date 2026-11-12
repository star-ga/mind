// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Fixed-arity `extern "C"` admission and lowering for the native lane.
//!
//! End state this file pins: `mindc build --backend native` builds a
//! fixed-arity extern program against the closed `nbx_row` libc table, and
//! refuses BY NAME every declaration outside that table. The fresh compiler
//! (`MINDC_NATIVE_ELF`, else a stage1 minted from this checkout) must carry
//! the `nbextc01` mark; a bounded-profile request combined with a user extern
//! block is refused.

#![cfg(all(feature = "std-surface", target_os = "linux", target_arch = "x86_64"))]

mod native_lane_extern_support;

use native_lane_extern_support::{
    CLOSE_EXTERN, Case, FIXED_SIGNATURE_CORPUS, FSTAT_EXTERN, compiler_components, compiler_source,
    extern_corpus, mint_stage1, repo,
};
use std::os::unix::ffi::OsStrExt;
use std::process::Command;

/// The four table symbols with their declared widths build under the mark.
#[test]
fn table_externs_build_under_the_mark() {
    let case = Case::new(true);
    assert!(case.builds(CLOSE_EXTERN).exists());
    assert!(case.builds(FSTAT_EXTERN).exists());
}

/// A program with no user extern block keeps the static path and still builds.
#[test]
fn static_program_is_untouched() {
    let case = Case::new(true);
    let binary = case.builds("fn main() -> i64 { return 7; }\n");
    let run = Command::new(binary).output().unwrap();
    assert_eq!(run.status.code(), Some(7), "{run:?}");
}

/// Without the `nbextc01` mark the compiler cannot bind the table: the
/// admission fence refuses the block, named.
#[test]
fn unmarked_compiler_refuses_externs() {
    let case = Case::new(false);
    // No intrinsics: an unmarked compiler fails intrinsics first (closure), so
    // the extern-specific refusal is reached only by a table call alone.
    case.refuses_named(
        "extern \"C\" { fn getuid() -> i32; }\nfn main() -> i64 { return getuid() as i64; }\n",
        "unmarked",
        "lane.extern.compiler_unmarked",
    );
}

/// Everything outside the closed table refuses BY NAME. The refusal names the
/// symbol that cannot be bound. Variadic libc names stay outside the table,
/// so a fixed-looking declaration of one still refuses by that name.
#[test]
fn off_table_symbols_refuse_by_name() {
    let case = Case::new(true);
    for (decl, name) in [
        ("fn open(path: *const u8, flags: i32) -> i32;", "open"),
        ("fn read(fd: i32, buf: *mut u8, n: u32) -> i32;", "read"),
        ("fn write(fd: i32, buf: *const u8, n: u32) -> i32;", "write"),
        ("fn fcntl(fd: i32, cmd: i32) -> i32;", "fcntl"),
        (
            "fn openat(dirfd: i32, path: *const u8, flags: i32) -> i32;",
            "openat",
        ),
        ("fn prctl(option: i32) -> i32;", "prctl"),
    ] {
        let source = format!("extern \"C\" {{ {decl} }}\nfn main() -> i64 {{ return 0; }}\n");
        case.refuses_named(&source, name, name);
    }
}

/// Widths the classifier does not assign stay refused: a result on `_exit`,
/// a missing result on `close`, and every pointer-to-pointer shape other than
/// one `*const *const u8` parameter.
#[test]
fn out_of_range_widths_refuse() {
    let case = Case::new(true);
    for (decl, label, named) in [
        (
            "fn _exit(status: i32) -> i32;",
            "exit with a result",
            "_exit",
        ),
        ("fn close(fd: i32);", "close missing result", "close"),
        (
            "fn fexecve(fd: i32, argv: *mut *const u8, envp: *const *const u8) -> i32;",
            "mut const",
            "fexecve",
        ),
        (
            "fn fexecve(fd: i32, argv: *const *mut u8, envp: *const *const u8) -> i32;",
            "const mut",
            "fexecve",
        ),
        (
            "fn fexecve(fd: i32, argv: *mut *mut u8, envp: *const *const u8) -> i32;",
            "mut mut",
            "fexecve",
        ),
        (
            "fn fexecve(fd: i32, argv: *const *const i32, envp: *const *const u8) -> i32;",
            "const const i32",
            "fexecve",
        ),
        (
            "fn fexecve(fd: i32, argv: *const *const *const u8, envp: *const *const u8) -> i32;",
            "triple",
            "fexecve",
        ),
        (
            "fn fexecve(fd: i32, argv: *const *const u8, envp: *const *const u8) -> *const *const u8;",
            "ptrptr result",
            "fexecve",
        ),
        (
            "fn fexecve(fd: i32, argv: *const *const u8, envp: *const *const u8) -> *const u8;",
            "const u8 result",
            "fexecve",
        ),
    ] {
        let source = format!("extern \"C\" {{ {decl} }}\nfn main() -> i64 {{ return 0; }}\n");
        case.refuses_named(&source, label, named);
    }
}

/// A table name with a signature the row does not carry refuses as a mismatch.
#[test]
fn signature_mismatch_on_a_table_name_refuses() {
    let case = Case::new(true);
    for (decl, label, named) in [
        ("fn close(fd: i64) -> i32;", "close i64 arg", "close"),
        ("fn close() -> i32;", "close missing arg", "close"),
        (
            "fn close(fd: i32, extra: i32) -> i32;",
            "close extra arg",
            "close",
        ),
        ("fn close(fd: i32) -> i64;", "close i64 result", "close"),
        ("fn getuid() -> i64;", "getuid wrong result", "getuid"),
        (
            "fn fstat(fd: i32, buf: *const u8) -> i32;",
            "fstat const buf",
            "fstat",
        ),
        (
            "fn __errno_location() -> *const u8;",
            "errno const result",
            "__errno_location",
        ),
    ] {
        let source = format!("extern \"C\" {{ {decl} }}\nfn main() -> i64 {{ return 0; }}\n");
        case.refuses_named(&source, label, named);
    }
}

/// Varargs, a non-C/SysV callconv and a name outside the table each refuse
/// with their own construct.
#[test]
fn varargs_callconv_and_unknown_refuse() {
    let case = Case::new(true);
    case.refuses_named(
        "extern \"C\" { fn close(fd: i32, ...) -> i32; }\nfn main() -> i64 { return 0; }\n",
        "varargs",
        "varargs",
    );
    case.refuses_named(
        "extern \"C\" callconv(.win64) { fn close(fd: i32) -> i32; }\nfn main() -> i64 { return 0; }\n",
        "win64",
        "callconv",
    );
    case.refuses_named(
        "extern \"C\" { fn nosuchsymbol(x: i32) -> i32; }\nfn main() -> i64 { return 0; }\n",
        "unknown",
        "nosuchsymbol",
    );
}

/// The bounded profile's file-backed RW load cannot carry a dynamic image yet:
/// extern + bounded refuses at admission.
#[test]
fn bounded_profile_refuses_externs() {
    let case = Case::new(true);
    let (build, _) = case.build_in(case.dir.path(), CLOSE_EXTERN, "bounded");
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert_eq!(
        build.status.code(),
        Some(3),
        "expected bounded+extern refusal: {stderr}"
    );
    assert!(
        stderr.contains("bounded") || stderr.contains("scope.bounded_extern"),
        "refusal should name the bounded conflict:\n{stderr}"
    );
}

/// The fixed interpreter path every extern image must name in PT_INTERP.
const INTERP: &[u8] = b"/lib64/ld-linux-x86-64.so.2\0";

/// The PT_INTERP bytes of an ELF64 image, and whether one of its PT_NOTE
/// segments carries a GNU build-id note (name `GNU`, type 3). `Err` when a
/// program header or note runs past the image.
fn interp_and_build_id(image: &[u8]) -> Result<(Option<Vec<u8>>, bool), String> {
    let word = |at: usize, n: usize| -> Result<usize, String> {
        let bytes = image
            .get(at..at + n)
            .ok_or_else(|| format!("ELF read past the image at {at}"))?;
        Ok(bytes
            .iter()
            .rev()
            .fold(0usize, |acc, b| (acc << 8) | usize::from(*b)))
    };
    let (phoff, phentsize, phnum) = (word(32, 8)?, word(54, 2)?, word(56, 2)?);
    let mut interp = None;
    let mut build_id = false;
    for k in 0..phnum {
        let ph = phoff + k * phentsize;
        let (kind, off, size) = (word(ph, 4)?, word(ph + 8, 8)?, word(ph + 32, 8)?);
        let segment = image
            .get(off..off + size)
            .ok_or_else(|| format!("segment {k} runs past the image"))?;
        if kind == 3 {
            interp = Some(segment.to_vec());
        }
        let mut at = 0;
        while kind == 4 && at + 12 <= size {
            let (namesz, descsz, ntype) = (
                word(off + at, 4)?,
                word(off + at + 4, 4)?,
                word(off + at + 8, 4)?,
            );
            let name = segment.get(at + 12..at + 12 + namesz).unwrap_or_default();
            build_id |= name == b"GNU\0" && ntype == 3;
            at += 12 + namesz.next_multiple_of(4) + descsz.next_multiple_of(4);
        }
    }
    Ok((interp, build_id))
}

/// The dynamic-ELF contract one compiler must meet on one extern program: two
/// builds in two directories are byte-identical, PT_INTERP is the fixed path,
/// no build-id note, and no host or build path. `Err` names the arm that
/// failed.
fn dynamic_elf_contract(case: &Case, label: &str, program: &str) -> Result<(), String> {
    let a = case.dir.path().join(label).join("a");
    let b = case.dir.path().join(label).join("b");
    let mut images = Vec::new();
    for at in [&a, &b] {
        let (build, binary) = case.build_in(at, program, "default");
        if build.status.code() != Some(0) {
            return Err(format!(
                "build failed in {}: {}",
                at.display(),
                String::from_utf8_lossy(&build.stderr)
            ));
        }
        images.push(std::fs::read(&binary).unwrap());
    }
    let image = &images[0];
    let has = |needle: &[u8]| image.windows(needle.len()).any(|w| w == needle);
    if images[0] != images[1] {
        return Err("extern image is not build-dir independent".to_string());
    }
    let (interp, build_id) = interp_and_build_id(image)?;
    match interp {
        None => return Err("PT_INTERP missing".to_string()),
        Some(path) if path != INTERP => {
            return Err(format!(
                "PT_INTERP is not the fixed path: {:?}",
                String::from_utf8_lossy(&path)
            ));
        }
        Some(_) => {}
    }
    if build_id || has(b".note.gnu.build-id") {
        return Err("build-id present".to_string());
    }
    // Home directories (the builder's own and any `/home/` one), the checkout,
    // the compiler and both build dirs.
    let needles: [&[u8]; 5] = [
        b"/home/",
        repo().as_os_str().as_bytes(),
        case.compiler().as_os_str().as_bytes(),
        a.as_os_str().as_bytes(),
        b.as_os_str().as_bytes(),
    ];
    let home = std::env::var_os("HOME").unwrap_or_default();
    // A HOME of `/` (or none) names no directory of its own.
    let own_home = (home.len() > 1).then(|| home.as_bytes());
    for needle in needles.into_iter().chain(own_home) {
        if has(needle) {
            return Err(format!(
                "image leaked a path: {:?}",
                String::from_utf8_lossy(needle)
            ));
        }
    }
    Ok(())
}

/// Dynamic-ELF determinism under the fresh compiler, over every program of
/// the extern corpus: a refusal or an empty artifact fails, and the number of
/// identical image pairs must equal the corpus size.
#[test]
fn dynamic_elf_is_deterministic_across_directories() {
    let case = Case::new(true);
    let corpus = extern_corpus();
    assert_eq!(corpus.len(), 11, "the extern corpus changed size");
    let mut identical = 0;
    for (label, program, _) in &corpus {
        dynamic_elf_contract(&case, label, program).unwrap_or_else(|e| panic!("{label}: {e}"));
        identical += 1;
    }
    assert_eq!(identical, corpus.len());
    println!(
        "dynamic-ELF contract: corpus={} identical_pairs={identical}",
        corpus.len()
    );
}

/// Writer-side mutants of `nbx_write_elf` (W1 #3: break determinism, the gate
/// reds). Each appends bytes to the finished image, past every load segment,
/// so the program still loads and runs; each is minted into its own stage1 from
/// this checkout and must fail exactly the contract arm it attacks.
#[test]
fn writer_side_mutants_fail_the_dynamic_elf_contract() {
    const TAIL: &str = "    if nbx_link(out, hdrs, code_len, t, calls, ncalls, owner, starts, gv) == 0 \
                        {\n        return string_new();\n    }\n    out\n}\n";
    // The low 16 bytes a CLOCK_MONOTONIC read leaves: a timestamp in the image.
    const CLOCK: &str = "    let mut_ts: i64 = __mind_alloc(16);\n    \
                         let _mut_rc: i64 = __mind_sys_clock_gettime(1, mut_ts);\n    \
                         out = nbx_bytes(out, mut_ts, 16);\n";
    // The eight bytes `/home/mu`: a host path in the image.
    const HOST_PATH: &str = "    let mut_p: i64 = __mind_alloc(8);\n    \
                             __mind_store_i64(mut_p, 8461471387573250095);\n    \
                             out = nbx_bytes(out, mut_p, 8);\n";
    let source = compiler_source();
    assert_eq!(source.matches(TAIL).count(), 1, "nbx_write_elf tail moved");
    let mutants = [
        (CLOCK, "not build-dir independent"),
        (HOST_PATH, "image leaked a path: \"/home/\""),
    ];
    let minted: Vec<_> = std::thread::scope(|scope| {
        let jobs: Vec<_> = mutants
            .iter()
            .map(|(stamp, _)| {
                let mutated = source.replacen(
                    TAIL,
                    &TAIL.replacen("    out\n", &format!("{stamp}    out\n"), 1),
                    1,
                );
                scope.spawn(move || mint_stage1(&mutated))
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    for ((stamp, arm), compiler) in mutants.iter().zip(minted) {
        let verdict = dynamic_elf_contract(&Case::with_compiler(compiler), "close", CLOSE_EXTERN);
        assert!(
            verdict.as_ref().is_err_and(|e| e.contains(arm)),
            "writer mutant survived or failed another arm (want `{arm}`): {verdict:?}\n{stamp}"
        );
        println!("writer mutant killed: {}", verdict.unwrap_err());
    }
}

/// Mutation control (the lane's W1 #3 first half): a stage1 minted from this
/// checkout with the `nbx_call` arm of the call lowering removed still carries
/// `nbextc01`, so admission passes, but the compiler cannot lower a table call
/// and refuses the program: exit 3, no artifact. The fresh compiler builds the
/// same programs.
#[test]
fn missing_extern_arm_mutant_refuses_the_table_calls() {
    const ARM: &str = "                return nbx_call(sa, xt, xr, callee, ids, arg_dts, args_len, \
                       id, calls, buf, lets, lcell);\n";
    let source = compiler_source();
    assert_eq!(source.matches(ARM).count(), 1, "the nbx_call arm moved");
    let mutated = source.replacen(ARM, "                let _arm_gone: i64 = xr;\n", 1);
    let mutant = Case::with_compiler(mint_stage1(&mutated));
    let fresh = Case::new(true);
    for program in [CLOSE_EXTERN, FSTAT_EXTERN] {
        assert!(fresh.builds(program).exists());
        let (build, binary) = mutant.build(program);
        let stderr = String::from_utf8_lossy(&build.stderr);
        assert_eq!(
            build.status.code(),
            Some(3),
            "extern-arm mutant survived: {stderr}"
        );
        assert!(!binary.exists(), "extern-arm mutant wrote an artifact");
        assert!(
            stderr.contains("pure-MIND compiler rejected the program (exit 3)"),
            "extern-arm mutant failed another way: {stderr}"
        );
        println!("extern-arm mutant killed: {}", stderr.trim_end());
    }
}

/// Width classes the ABI loads. A u32 result is refused (`nbx_row_sig`), a
/// 6-parameter off-table name refuses by name, and the same shape under a
/// table name is a signature mismatch.
#[test]
fn wide_widths_are_admitted_by_class_and_u32_results_refuse() {
    let case = Case::new(true);
    // u32 is a parameter width, never a result width (nbx_row_sig).
    case.refuses_named(
        "extern \"C\" { fn close(fd: i32) -> u32; }\nfn main() -> i64 { return 0; }\n",
        "u32 result",
        "close",
    );
    // A 6-parameter off-table row refuses by name; the widths never get a say
    // until L2b adds the row. This is the rdx/rcx/r8/r9 shape (args 2..=5).
    case.refuses_named(
        "extern \"C\" { fn six(a: i32, b: u32, c: i64, d: *const u8, e: *mut u8, f: *mut i32) -> i32; }\nfn main() -> i64 { return 0; }\n",
        "six",
        "six",
    );
    // The same shape under a table name is a signature mismatch, which is
    // what an L2b row with the wrong packing will hit first.
    case.refuses_named(
        "extern \"C\" { fn close(a: i32, b: u32, c: i64, d: *const u8, e: *mut u8, f: *mut i32) -> i32; }\nfn main() -> i64 { return 0; }\n",
        "close six-arg",
        "close",
    );
}

/// The closed table's four rows, spelled exactly as `nbx_row` declares them.
/// A regression in the Rust table is caught here before any lowering runs.
#[test]
fn closed_table_rows_are_the_four_core_symbols() {
    let case = Case::new(true);
    for (decl, call) in [
        (
            "fn close(fd: i32) -> i32;",
            "let _r: i64 = close(0 as i32) as i64;",
        ),
        (
            "fn fstat(fd: i32, buf: *mut u8) -> i32;",
            "let p: i64 = __mind_alloc(144); let _r: i64 = fstat(0 as i32, p as *mut u8) as i64;",
        ),
        ("fn getuid() -> i32;", "let _u: i64 = getuid() as i64;"),
        (
            "fn __errno_location() -> *mut i32;",
            "let _e: i64 = __errno_location() as i64;",
        ),
    ] {
        let source =
            format!("extern \"C\" {{ {decl} }}\nfn main() -> i64 {{ {call} return 0; }}\n");
        let (build, _) = case.build(&source);
        assert_eq!(
            build.status.code(),
            Some(0),
            "{decl} should build: {}",
            String::from_utf8_lossy(&build.stderr)
        );
    }
}

/// Exercise the SysV argument-register paths rdx, rcx, r8 and r9, and the
/// i64/u32 width classes, through a six-parameter local call. The widest
/// table row (`setsockopt`) has five parameters, so this local call is what
/// executes r9. The local prologue uses the same `nb_argreg_field` /
/// `nb_load_argreg` helpers the extern arm stages into rdi..r9.
#[test]
fn six_argument_local_call_exercises_rdx_rcx_r8_r9() {
    let case = Case::new(true);
    let source = r#"fn add6(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64) -> i64 {
    return a + b * 2 + c * 3 + d * 4 + e * 5 + f * 6;
}
fn main() -> i64 {
    return add6(1, 2, 3, 4, 5, 6);
}
"#;
    // 1 + 4 + 9 + 16 + 25 + 36 = 91
    let binary = case.builds(source);
    let run = Command::new(binary).output().unwrap();
    assert_eq!(run.status.code(), Some(91), "six-arg local call: {run:?}");
}

/// The two-argument table row (`fstat(fd: i32, buf: *mut u8)`) stages rdi then
/// rsi. The image must carry the i32 movsxd into rdi and the 64-bit load into
/// rsi — the width-correct load the `nbx_call` arm emits per `AbiTy`.
#[test]
fn two_argument_extern_call_stages_width_matched_registers() {
    let case = Case::new(true);
    let bin = case.builds(FSTAT_EXTERN);
    let bytes = std::fs::read(&bin).unwrap();
    // movsxd rdi, [rbp+disp] is 48 63 7D / 48 63 BD; the extern arm uses
    // REX.W(+R) 63 /r. A 32-bit param into rdi (field 7) is 48 63 7d or 48 63 bd.
    let movsxd_rdi = [0x48u8, 0x63, 0x7d].as_slice();
    let movsxd_rdi2 = [0x48u8, 0x63, 0xbd].as_slice();
    assert!(
        bytes
            .windows(3)
            .any(|w| w == movsxd_rdi || w == movsxd_rdi2),
        "fstat image lacks a movsxd into rdi (i32 first argument)"
    );
    // 64-bit load into rsi (field 6): REX.W 8B /r with ModRM reg=6 → 48 8b 75
    // or 48 8b b5 (disp32). A pointer second argument is a plain mov.
    let load_rsi = [0x48u8, 0x8b, 0x75].as_slice();
    let load_rsi2 = [0x48u8, 0x8b, 0xb5].as_slice();
    assert!(
        bytes.windows(3).any(|w| w == load_rsi || w == load_rsi2),
        "fstat image lacks a 64-bit load into rsi (pointer second argument)"
    );
}

/// The L1 extern-refusal fixture flip: a correctly-typed `close` builds.
#[test]
fn l1_extern_close_fixture_flipped_builds() {
    let case = Case::new(true);
    let source = std::fs::read_to_string(
        repo().join("tests/fixtures/native_lane_extern/l1_extern_close_flip.mind"),
    )
    .unwrap();
    let binary = case.builds(&source);
    let run = Command::new(binary).output().unwrap();
    assert_eq!(
        run.status.code(),
        Some(0),
        "flipped L1 close fixture: {run:?}"
    );
}

const CORPUS_NAMES: &[&str] = &[
    "close",
    "fstat",
    "getuid",
    "__errno_location",
    "clock_gettime",
    "close_range",
    "closedir",
    "dup2",
    "fchdir",
    "fdopendir",
    "flock",
    "fork",
    "fstatat",
    "fsync",
    "getpgid",
    "getrlimit",
    "kill",
    "lseek",
    "memfd_create",
    "nanosleep",
    "pipe2",
    "readdir",
    "renameat",
    "setpgid",
    "setrlimit",
    "signal",
    "umask",
    "unlinkat",
    "waitid",
    "waitpid",
    "_exit",
    "fexecve",
];

#[test]
fn fixed_signature_corpus_of_thirty_two_symbols_builds() {
    assert_eq!(CORPUS_NAMES.len(), 32);
    for name in CORPUS_NAMES {
        assert!(
            FIXED_SIGNATURE_CORPUS.contains(&format!("fn {name}(")),
            "{name} missing from the corpus"
        );
    }
    for variadic in ["fn fcntl(", "fn open(", "fn openat(", "fn prctl("] {
        assert!(
            !FIXED_SIGNATURE_CORPUS.contains(variadic),
            "{variadic} is variadic and stays out"
        );
    }
    let case = Case::new(true);
    let binary = case.builds(FIXED_SIGNATURE_CORPUS);
    let bytes = std::fs::read(&binary).unwrap();
    for name in ["getuid", "_exit", "fexecve"] {
        assert!(
            bytes.windows(name.len()).any(|w| w == name.as_bytes()),
            "{name} was not imported"
        );
    }
    let run = Command::new(&binary).output().unwrap();
    assert_eq!(
        run.status.code(),
        Some(0),
        "corpus run took the _exit arm or failed: {run:?}"
    );
}

/// The text of `pub fn {name}(` in the compiler source, through its closing
/// brace at column 0.
fn compiler_fn(source: &str, name: &str) -> String {
    let start = source
        .find(&format!("\npub fn {name}("))
        .unwrap_or_else(|| panic!("{name} is not in main.mind"))
        + 1;
    let len = source[start..].find("\n}\n").expect("closing brace") + 3;
    source[start..start + len].to_string()
}

/// The `nb_argreg_field` / `nb_argreg_ext` SysV map: rdi=7, rsi=6, rdx=2,
/// rcx=1, r8 (ext, field 0), r9 (ext, field 1). The extern arm and the local
/// prologue share this table, so the test runs the two functions from this
/// checkout's main.mind through the tree evaluator for each argument index.
#[test]
fn argreg_field_map_is_the_sysv_six() {
    // rdi rsi rdx rcx r8 r9 → (ModRM reg field, REX extension bit)
    let want = [(7, 0), (6, 0), (2, 0), (1, 0), (0, 1), (1, 1)];
    let source = compiler_components()
        .iter()
        .fold(compiler_source(), |s, part| {
            s + "\n"
                + &std::fs::read_to_string(repo().join("examples/mindc_mind").join(part)).unwrap()
        });
    let table = format!(
        "{}{}",
        compiler_fn(&source, "nb_argreg_field"),
        compiler_fn(&source, "nb_argreg_ext")
    );
    let eval = |call: String| {
        let module = libmind::parser::parse(&format!("{table}{call}\n")).expect("table parses");
        libmind::eval::eval_module(&module).expect("table evaluates")
    };
    for (i, (field, ext)) in want.into_iter().enumerate() {
        assert_eq!(
            eval(format!("nb_argreg_field({i})")),
            field,
            "argreg field for arg {i}"
        );
        assert_eq!(
            eval(format!("nb_argreg_ext({i})")),
            ext,
            "argreg ext for arg {i}"
        );
    }
}

/// The socket rows carry std's spellings. Any other width, a pointer of the
/// other mutability, or a different arity is a mismatch named by the row.
#[test]
fn socket_rows_refuse_other_spellings_by_name() {
    let case = Case::new(true);
    for (decl, label, named) in [
        (
            "fn connect(fd: i32, addr: *const u8, len: u32) -> i32;",
            "connect u32 length",
            "connect",
        ),
        (
            "fn setsockopt(fd: i32, level: i32, name: i32, value: *const u8) -> i32;",
            "setsockopt without its length",
            "setsockopt",
        ),
        (
            "fn send(fd: i32, buf: *const u8, len: i32, flags: i32) -> i64;",
            "send i32 length",
            "send",
        ),
        (
            "fn recv(fd: i32, buf: *const u8, len: i64, flags: i32) -> i64;",
            "recv const buffer",
            "recv",
        ),
        (
            "fn memset(dst: *mut u8, value: i32, n: i64) -> *const u8;",
            "memset const result",
            "memset",
        ),
        (
            "fn getrandom(buf: *mut u8, len: i64, flags: u32) -> i64;",
            "getrandom u32 flags",
            "getrandom",
        ),
        (
            "fn getaddrinfo(node: *const u8, service: *const u8, hints: *const u8, res: *const *const u8) -> i32;",
            "getaddrinfo pointer-to-pointer slot",
            "getaddrinfo",
        ),
        (
            "fn inet_pton(af: i32, src: *const u8, dst: *mut u8) -> i64;",
            "inet_pton i64 result",
            "inet_pton",
        ),
        (
            "fn socket(domain: i32, kind: i32) -> i32;",
            "socket two arguments",
            "socket",
        ),
    ] {
        let source = format!("extern \"C\" {{ {decl} }}\nfn main() -> i64 {{ return 0; }}\n");
        case.refuses_named(&source, label, named);
    }
}

/// Socket and name-service functions the table does not carry refuse by name.
#[test]
fn socket_neighbours_outside_the_table_refuse_by_name() {
    let case = Case::new(true);
    for (decl, name) in [
        (
            "fn bind(fd: i32, addr: *const u8, len: i32) -> i32;",
            "bind",
        ),
        ("fn listen(fd: i32, backlog: i32) -> i32;", "listen"),
        (
            "fn accept(fd: i32, addr: *mut u8, len: *mut u8) -> i32;",
            "accept",
        ),
        (
            "fn getsockname(fd: i32, addr: *mut u8, len: *mut u8) -> i32;",
            "getsockname",
        ),
        ("fn freeaddrinfo(res: *mut u8);", "freeaddrinfo"),
    ] {
        let source = format!("extern \"C\" {{ {decl} }}\nfn main() -> i64 {{ return 0; }}\n");
        case.refuses_named(&source, name, name);
    }
}

/// Every `nbx_row` name, decoded from its `nbx_put` literals (three name
/// words: the driver rows spell up to 19 bytes) rather than its comment, is
/// declared by a program of the extern corpus, so a row cannot land without
/// a program that builds and runs it. The table has 54 rows.
#[test]
fn every_table_row_is_declared_by_a_corpus_program() {
    let source = compiler_source();
    let body = compiler_fn(&source, "nbx_row");
    let mut lines = body.lines().map(str::trim);
    let mut rows = Vec::new();
    while let Some(head) = lines.next() {
        // `mindc fmt` puts the writer call on the line after `if i == N {`.
        if head.starts_with("if i == ") && head.ends_with('{') {
            rows.push(format!("{head} {}", lines.next().unwrap_or("")));
        } else {
            rows.push(head.to_string());
        }
    }
    let names: Vec<String> = rows
        .iter()
        .filter_map(|line| {
            let args = line
                .strip_prefix("if i == ")?
                .split("nbx_put(out, ")
                .nth(1)?;
            let n: Vec<i64> = args
                .split(',')
                .take(4)
                .map(|w| w.trim().parse().unwrap())
                .collect();
            let mut bytes = n[1].to_le_bytes().to_vec();
            bytes.extend(n[2].to_le_bytes());
            bytes.extend(n[3].to_le_bytes());
            bytes.truncate(usize::try_from(n[0]).unwrap());
            Some(String::from_utf8(bytes).unwrap())
        })
        .collect();
    assert_eq!(names.len(), 54, "{names:?}");
    let corpus = extern_corpus();
    for name in &names {
        let decl = format!("fn {name}(");
        assert!(
            corpus.iter().any(|(_, program, _)| program.contains(&decl)),
            "{name} is declared by no corpus program"
        );
    }
}
