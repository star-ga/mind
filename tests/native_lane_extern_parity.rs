// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Native == MLIR execution parity on the fixed-arity `extern "C"` corpus.
//!
//! Every corpus program is built by `mindc build --backend native` under the
//! fresh compiler (see `native_lane_extern_support`) AND by the MLIR backend,
//! and both artifacts run with stdin on `/dev/null`. They must exit alike with
//! identical stdout, and the exit code is pinned. The two probes whose result
//! follows the host pin it to this host's `/dev/null` mode and owner. A signal
//! death never becomes an exit code, so two crashes cannot compare equal.

#![cfg(all(
    feature = "std-surface",
    feature = "mlir-build",
    target_os = "linux",
    target_arch = "x86_64"
))]

mod native_lane_extern_support;

use libmind::ir::native_lane_profile::{LaneSeed, profile_lane_admits_seeded};
use native_lane_extern_support::{
    CLOSE_EXTERN, Case, FIXED_SIGNATURE_CORPUS, FSTAT_EXTERN, MINDC, NET_ROWS, U32_CLOSE_RANGE,
    WIDE_ROWS, build_apart, compiler_library, mint_library, repo, run, self_host_refusal,
};
use std::collections::BTreeSet;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::process::Command;

unsafe extern "C" {
    fn getuid() -> u32;
}

/// `(st_mode & S_IFMT, st_uid)` of `/dev/null`, the stdin `run` gives every probe.
fn devnull_mode_and_uid() -> (u32, u32) {
    let meta = std::fs::metadata("/dev/null").expect("/dev/null");
    (meta.mode() & 0o170000, meta.uid())
}

/// `FSTAT_EXTERN`: 50 when `/dev/null` is not owned by this uid, else
/// `mode / 4096` (fstat succeeds, so the `r * 100` term is 0).
fn fstat_probe_exit() -> i32 {
    let (mode, st_uid) = devnull_mode_and_uid();
    if unsafe { getuid() } != st_uid {
        return 50;
    }
    (mode / 4096) as i32
}

/// `native_c_abi_fixed.mind`: 21 unless stdin is a character device, 22 unless
/// its owner is this uid, else 0.
fn fixed_probe_exit() -> i32 {
    let (mode, st_uid) = devnull_mode_and_uid();
    if mode != 8192 {
        return 21;
    }
    if unsafe { getuid() } != st_uid {
        return 22;
    }
    0
}

/// The MLIR build of `source` in its own temporary directory: the MLIR build
/// writes `target/` next to the source, never into the repository.
fn mlir_run(label: &str, source: &str) -> (i32, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("main.mind");
    let artifact = dir.path().join("program");
    std::fs::write(&source_path, source).unwrap();
    let built = Command::new(MINDC)
        .args(["build", "--backend", "mlir"])
        .arg(&source_path)
        .arg("--emit=binary")
        .arg(format!("--out={}", artifact.display()))
        .env("MINDC_STD_DIR", repo().join("std"))
        .env_remove("MINDC_NATIVE_ELF")
        .env_remove("MINDC_SO")
        .current_dir(repo())
        .output()
        .unwrap();
    assert_eq!(
        built.status.code(),
        Some(0),
        "{label}: the MLIR build failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    run(&artifact)
}

/// Native and MLIR agree, and both equal the pinned exit code.
fn parity(label: &str, source: &str, exit: i32) {
    let case = Case::new(true);
    let native = run(&case.builds(source));
    let mlir = mlir_run(label, source);
    assert_eq!(native.0, exit, "{label}: native exit");
    assert_eq!(mlir.0, exit, "{label}: MLIR exit");
    assert_eq!(native, mlir, "{label}: native != MLIR");
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(repo().join("tests/fixtures/native_lane_extern").join(name)).unwrap()
}

/// close(-1) fails with EBADF (9) through the errno TLS slot: (r + 1) * 100 + 9.
#[test]
fn close_and_errno_run_alike_on_both_backends() {
    parity("close", CLOSE_EXTERN, 9);
}

/// fstat of a character-device stdin and getuid. The exit is pinned to this
/// host's `/dev/null` mode and owner (`fstat_probe_exit`).
#[test]
fn fstat_and_getuid_run_alike_on_both_backends() {
    parity("fstat", FSTAT_EXTERN, fstat_probe_exit());
}

/// The lane's committed fixtures: the four core rows and the flipped L1 row.
#[test]
fn the_extern_fixtures_run_alike_on_both_backends() {
    parity(
        "native_c_abi_fixed",
        &fixture("native_c_abi_fixed.mind"),
        fixed_probe_exit(),
    );
    parity(
        "l1_extern_close_flip",
        &fixture("l1_extern_close_flip.mind"),
        0,
    );
}

/// All 32 table rows declared together; `getuid` runs and the `_exit` /
/// `fexecve` arm is never taken.
#[test]
fn the_thirty_two_symbol_corpus_runs_alike_on_both_backends() {
    parity("corpus", FIXED_SIGNATURE_CORPUS, 0);
}

/// `u32` parameters through `close_range`: exit 22 (EINVAL), or the failing
/// step (`U32_CLOSE_RANGE`).
#[test]
fn u32_parameters_run_alike_on_both_backends() {
    parity("close_range", U32_CLOSE_RANGE, 22);
}

/// The i64, pointer and four-argument rows: exit 42, or the failing step
/// (`WIDE_ROWS`).
#[test]
fn wide_fixed_rows_run_alike_on_both_backends() {
    parity("wide_rows", WIDE_ROWS, 42);
}

/// The socket, address, fill and entropy rows, `setsockopt` staging its fifth
/// argument in r8: exit 42, or the failing step (`NET_ROWS`).
#[test]
fn net_rows_run_alike_on_both_backends() {
    parity("net_rows", NET_ROWS, 42);
}

/// The text of the private `fn {name}(` in `std/{module}.mind`, through its
/// closing brace at column 0: a byte-identical vendored copy of a seed helper.
fn seed_helper(module: &str, name: &str) -> String {
    let text = std::fs::read_to_string(repo().join(format!("std/{module}.mind"))).unwrap();
    let start = text
        .find(&format!("\nfn {name}("))
        .unwrap_or_else(|| panic!("std/{module}.mind has no private {name}"))
        + 1;
    let len = text[start..].find("\n}\n").expect("closing brace") + 3;
    text[start..start + len].to_string()
}

/// An image that vendors an identical private copy of a seed helper (as the
/// consumer programs vendor the sha256 helpers) meets the seed-copy closure
/// guard before the lane fence. A declared extern is a terminal edge there:
/// the table call builds and runs like MLIR (EBADF 9 plus `sha_mask32` of 2^32 + 5),
/// and an off-table declaration beside it is refused by its own name rather
/// than as a missing definition of the table call.
#[test]
fn a_vendored_seed_helper_keeps_extern_calls_terminal() {
    let helper = seed_helper("sha256", "sha_mask32");
    let program = format!(
        "extern \"C\" {{\n    fn close(fd: i32) -> i32;\n    fn __errno_location() -> *mut i32;\n}}\n\
         {helper}fn main() -> i64 {{\n    let fd: i64 = 0 - 1;\n    \
         let r: i64 = close(fd as i32) as i64;\n    \
         let e: i64 = __mind_load_i32(__errno_location() as i64);\n    \
         return (r + 1) * 100 + e + sha_mask32(4294967301);\n}}\n"
    );
    parity("vendored_helper", &program, 14);
    let refused = program.replacen(
        "    fn close(fd: i32) -> i32;\n",
        "    fn close(fd: i32) -> i32;\n    fn open(path: *const u8, flags: i32, mode: i32) -> i32;\n",
        1,
    );
    let refused = refused.replacen(
        "    let fd: i64 = 0 - 1;\n",
        "    let fd: i64 = open(0 as *const u8, 0 as i32, 0 as i32) as i64;\n",
        1,
    );
    let case = Case::new(true);
    let (build, binary) = case.build(&refused);
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert_eq!(build.status.code(), Some(3), "{stderr}");
    assert!(!binary.exists(), "a refused build wrote an artifact");
    assert!(
        stderr.contains("lane.extern.unknown_symbol") && stderr.contains("open"),
        "the refusal must name the off-table declaration:\n{stderr}"
    );
}

/// Census R2b reproduction: an image that vendors a byte-identical private
/// seed helper AND calls an admitted extern builds; the same program with the
/// vendored name renamed (no seed-copy collision) also builds. Without extern
/// edges in the vendoring closure check the first program refuses as
/// `closure.unresolved_edge`; the renamed control proves the defect is in the
/// vendoring path, not in extern handling.
#[test]
fn the_r2b_vendored_helper_plus_extern_reproduction_builds() {
    let helper = seed_helper("sha256", "sha_mask32");
    let program = format!(
        "extern \"C\" {{\n    fn close(fd: i32) -> i32;\n    fn __errno_location() -> *mut i32;\n}}\n\
         {helper}fn main() -> i64 {{\n    let fd: i64 = 0 - 1;\n    \
         let r: i64 = close(fd as i32) as i64;\n    \
         let e: i64 = __mind_load_i32(__errno_location() as i64);\n    \
         return (r + 1) * 100 + e + sha_mask32(4294967301);\n}}\n"
    );
    // Part 1: vendored sha_mask32 + extern close builds and runs.
    parity("r2b_vendored", &program, 14);
    // Part 2 (control): rename the vendored helper so there is no seed-copy
    // collision; the same extern calls still build.
    let renamed = program.replace("sha_mask32", "user_mask32");
    parity("r2b_renamed", &renamed, 14);
}

/// Census R10 / N5: the `__mind_store_i16` FrozenNative contract row,
/// width-aware on both backends. A widened (8-byte) or sign-extended store
/// clobbers the neighbouring bytes; a correct 2-byte store does not. The
/// discarded result reads 0 in value and tail position.
#[test]
fn the_store_i16_contract_row_is_width_aware_on_both_backends() {
    // Width: prefill 8 bytes with -1, store 70000 (0x11170): the low 16 bits
    // are 0x1170, so bytes 0/1 = 112/17. Byte 2 must still read 255: a
    // widened (sign- or zero-extended) store writes 0 or 1 there.
    parity(
        "store_i16_width",
        "fn main() -> i64 {\n    let p: i64 = __mind_alloc(8);\n    __mind_store_i64(p, 0 - 1);\n    __mind_store_i16(p, 70000);\n    if __mind_load_i8(p) != 112 { return 1; }\n    if __mind_load_i8(p + 1) != 17 { return 2; }\n    if __mind_load_i8(p + 2) != 255 { return 3; }\n    return 42;\n}\n",
        42,
    );
    // Sign-extension width: 65535 as i16 is -1. A sign-extended widened store
    // fills every byte with 0xFF; a correct 2-byte store leaves byte 2 at 0.
    parity(
        "store_i16_sign_width",
        "fn main() -> i64 {\n    let p: i64 = __mind_alloc(8);\n    __mind_store_i64(p, 0);\n    __mind_store_i16(p, 65535);\n    if __mind_load_i8(p) != 255 { return 1; }\n    if __mind_load_i8(p + 1) != 255 { return 2; }\n    if __mind_load_i8(p + 2) != 0 { return 3; }\n    return 42;\n}\n",
        42,
    );
    // Value position: the discarded result reads 0, not the stored value.
    parity(
        "store_i16_discard_value",
        "fn main() -> i64 {\n    let p: i64 = __mind_alloc(8);\n    let r: i64 = __mind_store_i16(p, 5);\n    return r + 3;\n}\n",
        3,
    );
    // Tail position: same discard at the end of an i64 function.
    parity(
        "store_i16_discard_tail",
        "fn w(p: i64) -> i64 {\n    __mind_store_i16(p, 7)\n}\nfn main() -> i64 {\n    let p: i64 = __mind_alloc(8);\n    let r: i64 = w(p);\n    return r + 4;\n}\n",
        4,
    );
}

// ---- seed-region `extern "C"` binding (nbx_bind.mind, native_lane_profile/seed.rs) ----

const NO_SEED_CALL: &str = "fn main() -> i64 { return 7; }\n";
const SEED_CALL_ONLY: &str = "import std.net;\nfn main() -> i64 { return af_inet(); }\n";
/// std.net `tcp_close` calls the seed declaration `close`; fd -1 fails EBADF.
const SEED_CLOSE: &str = "import std.net;\nfn main() -> i64 {\n    let h: i64 = __mind_alloc(8);\n    \
    __mind_store_i64(h, 0 - 1);\n    tcp_close(h);\n    return af_inet();\n}\n";
/// The seed `close(-1)` sets errno 9, read through a user declaration: 2 * 10 + 9.
const SEED_CLOSE_ERRNO: &str = "import std.net;\nextern \"C\" { fn __errno_location() -> *mut i32; }\n\
    fn main() -> i64 {\n    let h: i64 = __mind_alloc(8);\n    __mind_store_i64(h, 0 - 1);\n    \
    tcp_close(h);\n    let e: i64 = __mind_load_i32(__errno_location() as i64);\n    \
    return af_inet() * 10 + e;\n}\n";
/// `close` declared by the user region and by the seed: one symbol. 2 + (-1) + 1.
const SAME_SYMBOL: &str = "import std.net;\nextern \"C\" { fn close(fd: i32) -> i32; }\n\
    fn main() -> i64 {\n    let h: i64 = __mind_alloc(8);\n    __mind_store_i64(h, 0 - 1);\n    \
    tcp_close(h);\n    let r: i64 = close((0 - 1) as i32) as i64;\n    return af_inet() + r + 1;\n}\n";
const TWO_SEED_CALLERS: &str = "import std.net;\nfn main() -> i64 {\n    let h: i64 = __mind_alloc(8);\n    \
    __mind_store_i64(h, 0 - 1);\n    tcp_close(h);\n    udp_close(h);\n    return af_inet();\n}\n";

/// (label, program, user declarations it calls, exit code).
const SEED_MATRIX: [(&str, &str, &[&str], i32); 7] = [
    ("no seed call", NO_SEED_CALL, &[], 7),
    ("seed call, no declaration reached", SEED_CALL_ONLY, &[], 2),
    ("one seed declaration", SEED_CLOSE, &[], 2),
    (
        "seed and user declarations",
        SEED_CLOSE_ERRNO,
        &["__errno_location"],
        29,
    ),
    ("one symbol in both regions", SAME_SYMBOL, &["close"], 2),
    (
        "two seed callers of one declaration",
        TWO_SEED_CALLERS,
        &[],
        2,
    ),
    (
        "user declarations only",
        CLOSE_EXTERN,
        &["__errno_location", "close"],
        9,
    ),
];

/// The std seed modules (manifest `seed` rows, in blob order) and their bytes.
fn std_seed() -> (Vec<String>, Vec<Vec<u8>>) {
    let manifest = repo().join("examples/mindc_mind/testdata/stdlib_manifest.txt");
    let manifest = std::fs::read_to_string(manifest).unwrap();
    let rows = manifest.lines().filter(|l| !l.starts_with('#'));
    let seed = rows.filter_map(|l| {
        let mut fields = l.split('\t');
        let module = fields.next()?;
        (fields.next() == Some("seed")).then(|| module.to_string())
    });
    let modules: Vec<String> = seed.collect();
    let read = |m: &String| std::fs::read(repo().join(format!("std/{m}.mind"))).unwrap();
    let sources = modules.iter().map(read).collect();
    (modules, sources)
}

/// The imported symbol names of a built image, one per relocation, in GOT
/// order. A static image (no PT_DYNAMIC) imports nothing.
fn imports(image: &[u8]) -> Vec<String> {
    let word = |o: usize| u64::from_le_bytes(image[o..o + 8].try_into().unwrap()) as usize;
    let half = |o: usize| u32::from_le_bytes(image[o..o + 4].try_into().unwrap()) as usize;
    let phnum = u16::from_le_bytes(image[56..58].try_into().unwrap()) as usize;
    // (type, file offset, address, file size) of each program header.
    let headers: Vec<(usize, usize, usize, usize)> = (0..phnum)
        .map(|i| word(32) + i * 56)
        .map(|p| (half(p), word(p + 8), word(p + 16), word(p + 32)))
        .collect();
    let Some(&(_, dynamic, _, size)) = headers.iter().find(|h| h.0 == 2) else {
        return Vec::new();
    };
    let file = |address: usize| {
        let load =
            |h: &&(usize, usize, usize, usize)| h.0 == 1 && address >= h.2 && address < h.2 + h.3;
        let h = headers
            .iter()
            .find(load)
            .expect("a dynamic table is file-backed");
        address - h.2 + h.1
    };
    let entry = |tag: usize| {
        let at = (0..size / 16).map(|i| dynamic + i * 16);
        let at = at.clone().find(|&o| word(o) == tag).expect("dynamic tag");
        word(at + 8)
    };
    let (strings, symbols) = (file(entry(5)), file(entry(6)));
    (1..=entry(8) / 24)
        .map(|i| {
            let name = &image[strings + half(symbols + i * 24)..];
            let end = name.iter().position(|b| *b == 0).unwrap();
            String::from_utf8(name[..end].to_vec()).unwrap()
        })
        .collect()
}

/// One rule in both compilers: over the matrix, the image stage1 writes imports
/// exactly the user declarations the program calls plus the seed declarations
/// the admission model binds, each once, and the program runs to its pinned
/// exit code. A program that reaches no seed declaration and has no user
/// declaration stays a static image.
#[test]
fn stage1_and_the_admission_model_bind_the_same_seed_declarations() {
    let case = Case::new(true);
    let (modules, sources) = std_seed();
    let names: Vec<&str> = modules.iter().map(String::as_str).collect();
    let mut bound = 0;
    for (label, source, user_called, exit) in SEED_MATRIX {
        let ast = libmind::parser::parse(source).expect(label);
        let ir = libmind::eval::lower_to_ir(&ast).expect(label);
        let seed = LaneSeed::new(&names, &sources).binding_seed_externs(true);
        let edges = profile_lane_admits_seeded(&ast, &ir, &seed, true)
            .unwrap_or_else(|r| panic!("{label}: the model refuses: {r:?}"));
        bound += edges.seed_externs.len();
        let mut want: BTreeSet<String> = edges.seed_externs.into_keys().collect();
        want.extend(user_called.iter().map(|name| name.to_string()));
        let (build, binary) = build_apart(&case, source);
        let stderr = String::from_utf8_lossy(&build.stderr);
        assert_eq!(build.status.code(), Some(0), "{label}: {stderr}");
        let listed = imports(&std::fs::read(&binary).unwrap());
        let got: BTreeSet<String> = listed.iter().cloned().collect();
        assert_eq!(got, want, "{label}");
        assert_eq!(listed.len(), got.len(), "{label}: {listed:?}");
        assert_eq!(run(&binary).0, exit, "{label}");
    }
    // Positive count: the matrix binds a seed declaration in four programs.
    assert_eq!(bound, 4);
}

/// The seed `close` runs alike on both backends, alone and beside a user declaration.
#[test]
fn seed_declarations_run_alike_on_both_backends() {
    parity("seed close", SEED_CLOSE, 2);
    parity("seed close, user errno", SEED_CLOSE_ERRNO, 29);
}

/// `tcp_close` binds the seed `close`; `main` calls `close` and declares nothing.
const USER_CALLS_SEED_DECLARATION: &str = "import std.net;\nfn main() -> i64 {\n    \
    let h: i64 = __mind_alloc(8);\n    __mind_store_i64(h, 0 - 1);\n    tcp_close(h);\n    \
    let fd: i64 = 0 - 1;\n    let r: i64 = close(fd as i32) as i64;\n    return af_inet() + r + 1;\n}\n";

/// The std seed blob stage1 reads ahead of the user region: each module, then `\n`.
fn std_blob() -> Vec<u8> {
    let (_, sources) = std_seed();
    let line = |s: &Vec<u8>| s.iter().copied().chain([b'\n']).collect::<Vec<u8>>();
    sources.iter().flat_map(line).collect()
}

/// The flat stdin protocol of stage1, `[user_lo][len][seed ++ program]`: no
/// Rust admission runs. The ELF it wrote, or None for a refusal (no bytes).
fn flat(compiler: &Path, seed: &[u8], program: &str) -> Option<Vec<u8>> {
    let dir = tempfile::tempdir().unwrap();
    let mut image = (seed.len() as i64).to_le_bytes().to_vec();
    image.extend_from_slice(&((seed.len() + program.len()) as i64).to_le_bytes());
    image.extend_from_slice(seed);
    image.extend_from_slice(program.as_bytes());
    let path = dir.path().join("image.bin");
    std::fs::write(&path, image).unwrap();
    let stdin = std::fs::File::open(&path).unwrap();
    let out = Command::new(compiler).stdin(stdin).output().unwrap();
    let built = out.status.success() && out.stdout.starts_with(b"\x7fELF");
    assert!(built || out.stdout.is_empty(), "a refusal wrote bytes");
    built.then_some(out.stdout)
}

/// The exit code of a flat-built image.
fn exit_of(image: &[u8]) -> i32 {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("program");
    std::fs::write(&path, image).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    run(&path).0
}

/// stage1 holds the user boundary itself: on the flat protocol (no admission)
/// a call in the user region never binds a seed declaration, so the image
/// refuses. Through `mindc build` the admission refuses first. Controls on the
/// same protocol: the seed call alone binds, and a user declaration of the
/// name gives the user call its own row and the image one import.
#[test]
fn a_user_call_never_binds_a_seed_declaration() {
    let case = Case::new(true);
    let seed = std_blob();
    case.refuses_named(
        USER_CALLS_SEED_DECLARATION,
        "user call of a seed declaration",
        "lane.call.unknown",
    );
    assert_eq!(
        flat(case.compiler(), &seed, USER_CALLS_SEED_DECLARATION),
        None
    );
    let alone = flat(case.compiler(), &seed, SEED_CLOSE).expect("the seed call binds");
    assert_eq!(imports(&alone), ["close"]);
    assert_eq!(exit_of(&alone), 2);
    let both = flat(case.compiler(), &seed, SAME_SYMBOL).expect("one symbol");
    assert_eq!(imports(&both), ["close"]);
    assert_eq!(exit_of(&both), 2);
}

/// R15 of `testdata/native_scoped_fixed` (expected admission: refuse).
fn r15() -> String {
    let path =
        "examples/mindc_mind/testdata/native_scoped_fixed/sources/R15-private-std-local-edge.mind";
    std::fs::read_to_string(repo().join(path)).unwrap()
}

/// A private std helper never crosses the user boundary: R15 (`proc_errno`,
/// private in std.process), alone and with the import, refuses through
/// `mindc build` (the admission) and on the flat protocol (stage1 itself).
/// Control: a program that calls public seed functions, which call private
/// helpers of their own module, builds on the flat protocol and runs.
#[test]
fn a_private_seed_helper_never_crosses_the_user_boundary() {
    let case = Case::new(true);
    let seed = std_blob();
    let imported = "import std.process;\nfn main() -> i64 { return proc_errno(); }\n";
    for (label, source) in [("R15", r15().as_str()), ("R15 with the import", imported)] {
        case.refuses_named(source, label, "lane.call.unknown");
        assert_eq!(flat(case.compiler(), &seed, source), None, "{label}");
    }
    let ring = repo().join("tests/fixtures/native_lane_std/ring_roundtrip.mind");
    let ring = std::fs::read_to_string(ring).unwrap();
    let image = flat(case.compiler(), &seed, &ring).expect("public seed calls build");
    assert_eq!(exit_of(&image), 131);
}

/// The seed-block rules stage1 applies on its own, on a synthetic seed region
/// (flat protocol): a block and `shut`, its public caller of `close`. The
/// called declaration binds and runs (`close(-1)` is -1, plus 3). A sibling
/// outside the table that nothing calls is not read. A called declaration that
/// does not match its row refuses. A declaration the registry cannot parse
/// drops every row of its block, before it or after it: fail-closed, the image
/// refuses.
#[test]
fn a_seed_block_binds_by_declaration_and_fails_closed_on_a_parse_error() {
    let case = Case::new(true);
    let user = "fn main() -> i64 {\n    return shut(0 - 1) + 3;\n}\n";
    let build = |declarations: &str| {
        let seed = format!(
            "extern \"C\" {{\n{declarations}}}\npub fn shut(fd: i64) -> i64 {{\n    \
             return close(fd as i32) as i64;\n}}\n"
        );
        flat(case.compiler(), seed.as_bytes(), user)
    };
    let close = "    fn close(fd: i32) -> i32;\n";
    let image = build(close).expect("the called declaration binds");
    assert_eq!(imports(&image), ["close"]);
    assert_eq!(exit_of(&image), 2);
    let sibling = format!("{close}    fn open(path: *const u8, flags: i32) -> i32;\n");
    let image = build(&sibling).expect("an unread sibling outside the table");
    assert_eq!(imports(&image), ["close"]);
    assert_eq!(build("    fn close(fd: i64) -> i32;\n"), None);
    let bad = "    fn odd(x i32) -> i32;\n";
    assert_eq!(build(&format!("{close}{bad}")), None);
    assert_eq!(build(&format!("{bad}{close}")), None);
}

/// The `nbsext01` mark (`SEED_EXTERN_MARK` of the bridge): the part spells it
/// as its `nbs_cap_seed_extern_v1` immediate, and the compiler the marked
/// cases run carries the bytes. The bridge binds a seed declaration only under
/// it, so the matrix test above is red when the two spellings differ.
#[test]
fn the_fresh_compiler_carries_the_seed_extern_mark() {
    const MARK: &[u8] = b"\x48\xb8nbsext01";
    let part = repo().join("examples/mindc_mind/nbx_bind.mind");
    let part = std::fs::read_to_string(part).unwrap();
    let imm = i64::from_le_bytes(MARK[2..].try_into().unwrap());
    let body = format!("pub fn nbs_cap_seed_extern_v1() -> i64 {{\n    {imm}\n}}");
    assert!(part.contains(&body), "{body}");
    let compiler = std::fs::read(Case::new(true).compiler()).unwrap();
    assert!(compiler.windows(MARK.len()).any(|w| w == MARK));
}

/// Mutants of the binding rule, each a compiler minted from this checkout's
/// library with one line changed. `bind none` skips the seed pass: the program
/// that calls a seed declaration refuses, and every program that reaches none
/// keeps its bytes (the pass adds nothing to them). `bind all` marks every
/// seed declaration: a program that reaches none loses its bytes. `no edge`
/// shows every row to every call site: the user call of a seed declaration
/// builds on the flat protocol. `no private` drops the private-helper
/// refusal: R15 builds on the flat protocol.
#[test]
fn seed_binding_mutants_are_caught() {
    let library = compiler_library();
    let edits = [
        (
            "    if nbs_off(user_lo, ut, fns, n_fns) == 1 {\n",
            "    if nbs_off(user_lo, ut, fns, n_fns) != 2 {\n",
        ),
        (
            "        __mind_store_i64(reach + i * 8, nbs_named_mark(all, reach, i));\n",
            "        __mind_store_i64(reach + i * 8, 1);\n",
        ),
        (
            "    let edge: i64 = __mind_load_i64(t + 40);\n",
            "    let edge: i64 = 0;\n",
        ),
        (
            "    if nbs_any(pr, __mind_load_i64(pf)) != 0 {\n",
            "    if nbs_any(pr, __mind_load_i64(pf)) < 0 {\n",
        ),
    ];
    let minted: Vec<Case> = std::thread::scope(|scope| {
        let mint = |(from, to): &(&str, &str)| {
            assert_eq!(library.matches(from).count(), 1, "{from} moved");
            let mutated = library.replacen(from, to, 1);
            scope.spawn(move || mint_library(&mutated))
        };
        let jobs: Vec<_> = edits.iter().map(mint).collect();
        let done = jobs.into_iter().map(|j| j.join().unwrap());
        done.map(Case::with_compiler).collect()
    });
    let [none, all, no_edge, no_private] = minted.as_slice() else {
        panic!("four mutants");
    };
    let bytes = |case: &Case, source: &str| {
        let (build, binary) = build_apart(case, source);
        (build.status.code() == Some(0)).then(|| std::fs::read(&binary).unwrap())
    };
    let fresh = Case::new(true);
    assert!(self_host_refusal(none, SEED_CLOSE).is_some());
    for source in [NO_SEED_CALL, SEED_CALL_ONLY, CLOSE_EXTERN] {
        let kept = bytes(&fresh, source);
        assert!(kept.is_some() && kept == bytes(none, source), "{source}");
    }
    assert_ne!(bytes(all, SEED_CALL_ONLY), bytes(&fresh, SEED_CALL_ONLY));
    assert!(bytes(&fresh, SEED_CLOSE).is_some());
    let seed = std_blob();
    assert!(flat(no_edge.compiler(), &seed, USER_CALLS_SEED_DECLARATION).is_some());
    assert_eq!(
        flat(fresh.compiler(), &seed, USER_CALLS_SEED_DECLARATION),
        None
    );
    assert!(flat(no_private.compiler(), &seed, &r15()).is_some());
    assert_eq!(flat(fresh.compiler(), &seed, &r15()), None);
}
