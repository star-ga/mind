// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Shared harness for the fixed-arity `extern "C"` lane tests.
//!
//! A `tests/<dir>/mod.rs` module, not a test target: it compiles into
//! `native_lane_extern_fixed.rs` and `native_lane_extern_parity.rs`, so neither
//! exceeds the ceiling `tests/module_size_ratchet.rs` pins.
//!
//! The extern lowering lives in this checkout's `main.mind`; the committed
//! testdata seed predates it and carries no `nbextc01`. The marked cases
//! therefore run a stage1 minted from this checkout (`mint_stage1`), or the
//! compiler `MINDC_NATIVE_ELF` names. The committed seed is never replaced.

// It writes and execs a POSIX shell stub, so it is unix-only at file scope
// (tests/harness_portability.rs).
#![cfg(unix)]
#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::sync::OnceLock;

pub const MINDC: &str = env!("CARGO_BIN_EXE_mindc");
pub const EXTERN_MARK: &[u8] = b"\x48\xb8nbextc01";
pub const LANE_MARK: &[u8] = b"\x48\xb8nblane01";
const SEED: &str = "examples/mindc_mind/testdata/selfhost_loop/stage1.elf";
const BOUNDED_SEED: &str = "examples/mindc_mind/testdata/selfhost_loop_bounded/stage1.elf";

pub fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// This checkout's compiler source (main.mind alone).
pub fn compiler_source() -> String {
    std::fs::read_to_string(repo().join("examples/mindc_mind/main.mind")).unwrap()
}

/// This checkout's compiler library: main.mind and its parts, the text a
/// part-row mutant edits (`mint_library`).
pub fn compiler_library() -> String {
    with_imported_parts(&compiler_source())
}

/// `main_source (\n part)*` in `COMPILER_COMPONENTS` order, as
/// `SourceBundle.library_bytes` joins them, when it imports a part; one
/// importing none (`base_compiler_source`) goes alone.
fn with_imported_parts(main_source: &str) -> String {
    let parts = compiler_components();
    let imported = |p: &String| main_source.contains(&format!("\nimport {};", &p[..p.len() - 5]));
    let mut library = main_source.to_string();
    for part in parts.iter().filter(|_| parts.iter().any(imported)) {
        library.push('\n');
        let text = std::fs::read_to_string(repo().join("examples/mindc_mind").join(part));
        library.push_str(&text.unwrap());
    }
    library
}

/// The compiler part file names in their one fixed order, read from the single
/// list (`COMPILER_COMPONENTS` in `_selfhost_sources.py`), never copied here.
pub fn compiler_components() -> Vec<String> {
    let py = std::fs::read_to_string(repo().join("examples/mindc_mind/_selfhost_sources.py"));
    let py = py.unwrap();
    let list = py
        .split("COMPILER_COMPONENTS")
        .nth(1)
        .and_then(|s| s.split("))").next());
    let names = list
        .expect("COMPILER_COMPONENTS")
        .split('"')
        .skip(1)
        .step_by(2);
    let parts: Vec<String> = names
        .filter(|s| s.ends_with(".mind"))
        .map(String::from)
        .collect();
    assert!(!parts.is_empty(), "COMPILER_COMPONENTS lists no part");
    parts
}

/// The stdin image the self-host loop's first hop feeds the seed:
/// `[8B user_lo][8B len][std seed blob ++ library \n selfhost_driver]`, the std
/// seed modules in `stdlib_manifest.txt` order, each followed by `\n`
/// (`_selfhost_sources.py`, `src/build/native_image.rs`).
fn stage1_image(library: &str) -> Vec<u8> {
    let manifest =
        std::fs::read_to_string(repo().join("examples/mindc_mind/testdata/stdlib_manifest.txt"))
            .unwrap();
    let mut source = Vec::new();
    for line in manifest
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let mut fields = line.split('\t');
        if let (Some(module), Some("seed")) = (fields.next(), fields.next()) {
            source.extend(std::fs::read(repo().join(format!("std/{module}.mind"))).unwrap());
            source.push(b'\n');
        }
    }
    let user_lo = source.len() as i64;
    source.extend_from_slice(library.as_bytes());
    source.push(b'\n');
    source.extend(std::fs::read(repo().join("examples/mindc_mind/selfhost_driver.mind")).unwrap());
    let mut image = user_lo.to_le_bytes().to_vec();
    image.extend_from_slice(&(source.len() as i64).to_le_bytes());
    image.extend_from_slice(&source);
    image
}

/// A stage1 compiled from `main_source` by the committed seed, carrying
/// `nbextc01`. The image is a pure function of the seed and its input, so it
/// is cached under cargo's per-target scratch directory by their digest; a
/// different source never reuses an older image.
pub fn mint_stage1(main_source: &str) -> PathBuf {
    mint_library(&with_imported_parts(main_source))
}

/// `mint_stage1` of a whole compiler library (`compiler_library`, edited).
pub fn mint_library(library: &str) -> PathBuf {
    let seed = std::fs::read(repo().join(SEED)).unwrap();
    let image = stage1_image(library);
    let key: String = Sha256::new()
        .chain_update(&seed)
        .chain_update(&image)
        .finalize()
        .iter()
        .take(16)
        .map(|b| format!("{b:02x}"))
        .collect();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("native_lane_stage1");
    let path = dir.join(format!("{key}.elf"));
    if path.is_file() {
        return path;
    }
    let mut child = Command::new(repo().join(SEED))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&image));
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap().unwrap();
    assert!(
        out.status.success() && out.stdout.starts_with(b"\x7fELF"),
        "the committed seed did not mint a stage1 from this checkout: exit {:?}: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout
            .windows(EXTERN_MARK.len())
            .any(|w| w == EXTERN_MARK),
        "the minted stage1 carries no nbextc01 mark"
    );
    std::fs::create_dir_all(&dir).unwrap();
    // Write, then rename: a concurrent test binary may be running the same key.
    let partial = dir.join(format!("{key}.{}.partial", std::process::id()));
    std::fs::write(&partial, &out.stdout).unwrap();
    std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::rename(&partial, &path).unwrap();
    path
}

/// The compiler the marked cases run: `MINDC_NATIVE_ELF` when it is set, else
/// a stage1 minted from this checkout. Never the committed testdata seed:
/// falling back to it turns every marked case into an unmarked refusal.
pub fn fresh_compiler() -> &'static Path {
    static FRESH: OnceLock<PathBuf> = OnceLock::new();
    FRESH.get_or_init(|| match std::env::var_os("MINDC_NATIVE_ELF") {
        Some(path) => PathBuf::from(path),
        None => mint_stage1(&compiler_source()),
    })
}

/// One build directory and the compiler its builds select.
pub struct Case {
    pub dir: tempfile::TempDir,
    compiler: PathBuf,
}

impl Case {
    /// `marked`: the fresh compiler. Unmarked: a shim that carries the lane
    /// mark (nblane01) but not nbextc01, so Rust-side lane admission runs and
    /// refuses the extern block by name before the shim runs. The mark rides
    /// in a shell comment so the shell ignores it and compiler_has_mark finds it.
    pub fn new(marked: bool) -> Self {
        if marked {
            return Self::with_compiler(fresh_compiler().to_path_buf());
        }
        let dir = tempfile::tempdir().unwrap();
        let compiler = dir.path().join("recording-compiler.sh");
        let mut bytes =
            b"#!/bin/sh\nset -eu\ncat > \"$MIND_NATIVE_TEST_CAPTURE\"\nexec \"$MIND_NATIVE_TEST_SEED\" < \"$MIND_NATIVE_TEST_CAPTURE\"\n"
                .to_vec();
        bytes.extend_from_slice(b"# ");
        bytes.extend_from_slice(LANE_MARK);
        bytes.extend_from_slice(b"\n");
        std::fs::write(&compiler, bytes).unwrap();
        std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self { dir, compiler }
    }

    pub fn with_compiler(compiler: PathBuf) -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
            compiler,
        }
    }

    pub fn compiler(&self) -> &Path {
        &self.compiler
    }

    pub fn build(&self, source: &str) -> (Output, PathBuf) {
        self.build_in(self.dir.path(), source, "default")
    }

    pub fn build_in(&self, at: &Path, source: &str, profile: &str) -> (Output, PathBuf) {
        let source_path = at.join("main.mind");
        let output_path = at.join("program");
        std::fs::create_dir_all(at).unwrap();
        std::fs::write(&source_path, source).unwrap();
        let output = Command::new(MINDC)
            .args(["build"])
            .arg(&source_path)
            .args(["--backend", "native", "--emit", "binary", "--out"])
            .arg(&output_path)
            .args(["--native-profile", profile])
            .env("MINDC_NATIVE_ELF", &self.compiler)
            .env("MINDC_STD_DIR", repo().join("std"))
            .env("MIND_NATIVE_TEST_CAPTURE", at.join("image.bin"))
            .env(
                "MIND_NATIVE_TEST_SEED",
                repo().join(if profile == "bounded" {
                    BOUNDED_SEED
                } else {
                    SEED
                }),
            )
            .env_remove("MINDC_SO")
            .current_dir(at)
            .output()
            .unwrap();
        (output, output_path)
    }

    pub fn builds(&self, source: &str) -> PathBuf {
        let (build, binary) = self.build(source);
        assert_eq!(
            build.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        binary
    }

    pub fn refuses_named(&self, source: &str, label: &str, text: &str) {
        let (build, _) = self.build(source);
        let stderr = String::from_utf8_lossy(&build.stderr);
        assert_eq!(
            build.status.code(),
            Some(3),
            "{label}: expected a refusal (exit 3): {stderr}"
        );
        assert!(
            stderr.contains(text),
            "{label}: expected the refusal to name `{text}`:\n{stderr}"
        );
    }
}

fn capture(binary: &Path, vars: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(binary);
    cmd.stdin(Stdio::null());
    for (k, v) in vars {
        cmd.env(k, v);
    }
    let ran = cmd.output().unwrap();
    assert!(ran.stderr.is_empty(), "{}: {ran:?}", binary.display());
    ran
}

/// The process exit code. A signal death has no exit code; mapping it to a
/// number would let two backends that died on the same signal compare equal,
/// so it fails the caller instead.
fn finished_code(binary: &Path, status: ExitStatus) -> i32 {
    status.code().unwrap_or_else(|| {
        panic!(
            "{} died on signal {} and that death is a failure",
            binary.display(),
            status.signal().unwrap_or(0)
        )
    })
}

/// Run a built program with stdin on `/dev/null` (a character device):
/// its exit code and stdout bytes.
pub fn run(binary: &Path) -> (i32, Vec<u8>) {
    let ran = capture(binary, &[]);
    (finished_code(binary, ran.status), ran.stdout)
}

/// `Ok(exit)` when the process exited, `Err(signal)` when it was killed.
/// A guard that must not exit 42 uses this so a crash is that failure. The
/// signal is not rewritten into an exit code.
pub fn exit_code_or_signal(binary: &Path, vars: &[(&str, &str)]) -> Result<i32, i32> {
    let ran = capture(binary, vars);
    match ran.status.code() {
        Some(code) => Ok(code),
        None => Err(ran.status.signal().unwrap_or(0)),
    }
}

pub const CLOSE_EXTERN: &str = r#"extern "C" {
    fn close(fd: i32) -> i32;
    fn __errno_location() -> *mut i32;
}
fn main() -> i64 {
    let fd: i64 = 0 - 1;
    let r: i64 = close(fd as i32) as i64;
    let e: i64 = __mind_load_i32(__errno_location() as i64);
    return (r + 1) * 100 + e;
}
"#;

pub const FSTAT_EXTERN: &str = r#"extern "C" {
    unsafe fn fstat(fd: i32, buf: *mut u8) -> i32;
    unsafe fn getuid() -> i32;
}
fn main() -> i64 {
    let p: i64 = __mind_alloc(144);
    let r: i64 = fstat(0 as i32, p as *mut u8) as i64;
    let mode: i64 = __mind_load_i32(p + 24) & 61440;
    let u: i64 = getuid() as i64;
    if u != (__mind_load_i64(p + 28) & 4294967295) { return 50; }
    return r * 100 + mode / 4096;
}
"#;

/// The 32 non-variadic table rows, declared together. `getuid` runs. `_exit`
/// and `fexecve` are compiled into the untaken arm: a real uid is never
/// negative, and a taken arm would exit 91.
pub const FIXED_SIGNATURE_CORPUS: &str = r#"extern "C" {
    fn close(fd: i32) -> i32;
    fn fstat(fd: i32, buf: *mut u8) -> i32;
    fn getuid() -> i32;
    fn __errno_location() -> *mut i32;
    fn clock_gettime(clk: i32, ts: *mut u8) -> i32;
    fn close_range(first: u32, last: u32, flags: i32) -> i32;
    fn closedir(dirp: *mut u8) -> i32;
    fn dup2(oldfd: i32, newfd: i32) -> i32;
    fn fchdir(fd: i32) -> i32;
    fn fdopendir(fd: i32) -> *mut u8;
    fn flock(fd: i32, op: i32) -> i32;
    fn fork() -> i32;
    fn fstatat(dirfd: i32, path: *const u8, buf: *mut u8, flags: i32) -> i32;
    fn fsync(fd: i32) -> i32;
    fn getpgid(pid: i32) -> i32;
    fn getrlimit(resource: i32, rlim: *mut u8) -> i32;
    fn kill(pid: i32, sig: i32) -> i32;
    fn lseek(fd: i32, off: i64, whence: i32) -> i64;
    fn memfd_create(name: *const u8, flags: i32) -> i32;
    fn nanosleep(req: *const u8, rem: *mut u8) -> i32;
    fn pipe2(pipefd: *mut i32, flags: i32) -> i32;
    fn readdir(dirp: *mut u8) -> *mut u8;
    fn renameat(olddir: i32, oldpath: *const u8, newdir: i32, newpath: *const u8) -> i32;
    fn setpgid(pid: i32, pgid: i32) -> i32;
    fn setrlimit(resource: i32, rlim: *const u8) -> i32;
    fn signal(sig: i32, handler: *const u8) -> *mut u8;
    fn umask(mask: i32) -> i32;
    fn unlinkat(dirfd: i32, path: *const u8, flags: i32) -> i32;
    fn waitid(idtype: i32, id: i32, infop: *mut u8, options: i32) -> i32;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn _exit(status: i32);
    fn fexecve(fd: i32, argv: *const *const u8, envp: *const *const u8) -> i32;
}
fn main() -> i64 {
    let uid: i64 = getuid() as i64;
    if uid < 0 {
        _exit(91 as i32);
        let code: i64 = fexecve(0 as i32, 0 as *const *const u8, 0 as *const *const u8) as i64;
        return code;
    }
    return 0;
}
"#;

/// `u32` parameters through `close_range`. `(0 - 1) as u32` keeps the low 32
/// bits on both backends; fd 50 (a dup of stdin) is closed by the range call,
/// so `close(50)` fails with EBADF; a reversed range fails with EINVAL (22),
/// the pinned exit. Any other exit names the failing step.
pub const U32_CLOSE_RANGE: &str = r#"extern "C" {
    fn close_range(first: u32, last: u32, flags: i32) -> i32;
    fn dup2(oldfd: i32, newfd: i32) -> i32;
    fn close(fd: i32) -> i32;
    fn __errno_location() -> *mut i32;
}
fn main() -> i64 {
    let minus_one: i64 = 0 - 1;
    if (minus_one as u32) as i64 != 4294967295 { return 1; }
    if dup2(0 as i32, 50 as i32) as i64 != 50 { return 2; }
    let hi: i64 = 4000000000;
    if close_range(50 as u32, hi as u32, 0 as i32) as i64 != 0 { return 3; }
    if close(50 as i32) as i64 != minus_one { return 4; }
    if __mind_load_i32(__errno_location() as i64) != 9 { return 5; }
    if close_range(51 as u32, 50 as u32, 0 as i32) as i64 != minus_one { return 6; }
    return __mind_load_i32(__errno_location() as i64);
}
"#;

/// The i64, `*const u8`, `*mut i32` and four-argument staging paths, each
/// checked against a host-independent result: exit 42, or the failing step.
pub const WIDE_ROWS: &str = r#"extern "C" {
    fn memfd_create(name: *const u8, flags: i32) -> i32;
    fn lseek(fd: i32, off: i64, whence: i32) -> i64;
    fn fstatat(dirfd: i32, path: *const u8, buf: *mut u8, flags: i32) -> i32;
    fn clock_gettime(clk: i32, ts: *mut u8) -> i32;
    fn getrlimit(resource: i32, rlim: *mut u8) -> i32;
    fn fsync(fd: i32) -> i32;
    fn dup2(oldfd: i32, newfd: i32) -> i32;
    fn pipe2(pipefd: *mut i32, flags: i32) -> i32;
    fn nanosleep(req: *const u8, rem: *mut u8) -> i32;
    fn umask(mask: i32) -> i32;
}
fn main() -> i64 {
    let name: i64 = __mind_alloc(8);
    __mind_store_i64(name, 120);
    let fd: i64 = memfd_create(name as *const u8, 0 as i32) as i64;
    if fd < 0 { return 1; }
    let big: i64 = 4294967301;
    if lseek(fd as i32, big, 0 as i32) != big { return 2; }
    let path: i64 = __mind_alloc(8);
    __mind_store_i64(path, 47);
    let st: i64 = __mind_alloc(144);
    let at: i64 = 0 - 100;
    if fstatat(at as i32, path as *const u8, st as *mut u8, 0 as i32) as i64 != 0 { return 3; }
    if (__mind_load_i32(st + 24) & 61440) != 16384 { return 4; }
    let ts: i64 = __mind_alloc(16);
    if clock_gettime(1 as i32, ts as *mut u8) as i64 != 0 { return 5; }
    let rl: i64 = __mind_alloc(16);
    if getrlimit(7 as i32, rl as *mut u8) as i64 != 0 { return 6; }
    if __mind_load_i64(rl) <= 0 { return 7; }
    if fsync(fd as i32) as i64 != 0 { return 8; }
    if dup2(fd as i32, 50 as i32) as i64 != 50 { return 9; }
    let pp: i64 = __mind_alloc(8);
    if pipe2(pp as *mut i32, 0 as i32) as i64 != 0 { return 10; }
    if __mind_load_i32(pp) < 0 { return 11; }
    let req: i64 = __mind_alloc(16);
    __mind_store_i64(req, 0);
    __mind_store_i64(req + 8, 1000);
    let rem: i64 = __mind_alloc(16);
    if nanosleep(req as *const u8, rem as *mut u8) as i64 != 0 { return 12; }
    let old: i64 = umask(18 as i32) as i64;
    if umask(old as i32) as i64 != 18 { return 13; }
    return 42;
}
"#;

/// The socket, address, fill and entropy rows (`close` included), each step
/// checked against a host-independent result: exit 42, or the failing step.
/// A UDP socket needs no peer: SO_REUSEADDR with a 4-byte and a 0-byte length
/// (the fifth argument, r8), a non-blocking receive (EAGAIN 11, the flags in
/// rcx), a send before connect (EDESTADDRREQ 89), a short and a full
/// `127.0.0.1:9` address for connect (EINVAL 22, then 0), a 5-byte send.
/// `memset` returns its destination and stops at its length; `getrandom`
/// fills 32 bytes and refuses an unknown flag (EINVAL); `getaddrinfo` parses
/// a numeric host and service and refuses a name under AI_NUMERICHOST
/// (EAI_NONAME, -2, sign-extended).
pub const NET_ROWS: &str = r#"extern "C" {
    fn socket(domain: i32, kind: i32, protocol: i32) -> i32;
    fn setsockopt(fd: i32, level: i32, name: i32, value: *const u8, len: i32) -> i32;
    fn recv(fd: i32, buf: *mut u8, len: i64, flags: i32) -> i64;
    fn send(fd: i32, buf: *const u8, len: i64, flags: i32) -> i64;
    fn inet_pton(af: i32, src: *const u8, dst: *mut u8) -> i32;
    fn connect(fd: i32, addr: *const u8, len: i32) -> i32;
    fn memset(dst: *mut u8, value: i32, n: i64) -> *mut u8;
    fn getrandom(buf: *mut u8, len: i64, flags: i32) -> i64;
    fn getaddrinfo(node: *const u8, service: *const u8, hints: *const u8, res: *mut u8) -> i32;
    fn close(fd: i32) -> i32;
    fn __errno_location() -> *mut i32;
}
fn main() -> i64 {
    let minus_one: i64 = 0 - 1;
    let fd: i64 = socket(2 as i32, 2 as i32, 0 as i32) as i64;
    if fd < 0 { return 1; }
    let one: i64 = __mind_alloc(8);
    __mind_store_i64(one, 1);
    if setsockopt(fd as i32, 1 as i32, 2 as i32, one as *const u8, 4 as i32) as i64 != 0 { return 2; }
    if setsockopt(fd as i32, 1 as i32, 2 as i32, one as *const u8, 0 as i32) as i64 != minus_one { return 3; }
    if __mind_load_i32(__errno_location() as i64) != 22 { return 4; }
    let buf: i64 = __mind_alloc(64);
    if recv(fd as i32, buf as *mut u8, 16, 64 as i32) != minus_one { return 5; }
    if __mind_load_i32(__errno_location() as i64) != 11 { return 6; }
    if send(fd as i32, buf as *const u8, 1, 0 as i32) != minus_one { return 7; }
    if __mind_load_i32(__errno_location() as i64) != 89 { return 8; }
    let text: i64 = __mind_alloc(16);
    __mind_store_i64(text, 3328210909095473713);
    __mind_store_i64(text + 8, 49);
    let sin: i64 = __mind_alloc(16);
    __mind_store_i64(sin, 0);
    __mind_store_i64(sin + 8, 0);
    let dst: i64 = sin + 4;
    if inet_pton(2 as i32, text as *const u8, dst as *mut u8) as i64 != 1 { return 9; }
    if __mind_load_i32(dst) != 16777343 { return 10; }
    __mind_store_i64(sin, 2 + 150994944 + __mind_load_i32(dst) * 4294967296);
    if __mind_load_i64(sin) != 72058139649769474 { return 11; }
    if connect(fd as i32, sin as *const u8, 8 as i32) as i64 != minus_one { return 12; }
    if __mind_load_i32(__errno_location() as i64) != 22 { return 13; }
    if connect(fd as i32, sin as *const u8, 16 as i32) as i64 != 0 { return 14; }
    if send(fd as i32, buf as *const u8, 5, 0 as i32) != 5 { return 15; }
    __mind_store_i64(buf + 16, 5);
    let filled: i64 = memset(buf as *mut u8, 17 as i32, 16) as i64;
    if __mind_load_i64(filled) != 1229782938247303441 { return 16; }
    if __mind_load_i64(buf + 8) != 1229782938247303441 { return 16; }
    if __mind_load_i64(buf + 16) != 5 { return 17; }
    let rnd: i64 = __mind_alloc(32);
    if getrandom(rnd as *mut u8, 32, 1 as i32) != 32 { return 18; }
    if (__mind_load_i64(rnd) | __mind_load_i64(rnd + 8) | __mind_load_i64(rnd + 16) | __mind_load_i64(rnd + 24)) == 0 { return 19; }
    if getrandom(rnd as *mut u8, 32, 128 as i32) != minus_one { return 20; }
    if __mind_load_i32(__errno_location() as i64) != 22 { return 21; }
    let hints: i64 = __mind_alloc(48);
    let _h: i64 = memset(hints as *mut u8, 0 as i32, 48) as i64;
    __mind_store_i64(hints, 8589935620);
    __mind_store_i64(hints + 8, 1);
    let svc: i64 = __mind_alloc(8);
    __mind_store_i64(svc, 12344);
    let res: i64 = __mind_alloc(8);
    __mind_store_i64(res, 0);
    if getaddrinfo(text as *const u8, svc as *const u8, hints as *const u8, res as *mut u8) as i64 != 0 { return 22; }
    let ai: i64 = __mind_load_i64(res);
    if __mind_load_i32(ai + 4) != 2 { return 23; }
    if __mind_load_i32(ai + 16) != 16 { return 24; }
    if __mind_load_i64(__mind_load_i64(ai + 24)) != 72058140840951810 { return 25; }
    let host: i64 = __mind_alloc(8);
    __mind_store_i64(host, 1701867374);
    if getaddrinfo(host as *const u8, svc as *const u8, hints as *const u8, res as *mut u8) as i64 != 0 - 2 { return 26; }
    if close(fd as i32) as i64 != 0 { return 27; }
    return 42;
}
"#;

/// One committed fixture of the GPU lane directory.
pub fn gpu_fixture(name: &str) -> String {
    std::fs::read_to_string(repo().join("tests/fixtures/native_lane_gpu").join(name)).unwrap()
}

/// Every program of the extern corpus, with the exit it pins (`None` where the
/// result depends on the host): the two probes, the declared-row programs, the
/// u32, wide and socket programs, both committed fixtures, and the three GPU
/// loader fixtures (the real-driver vector add, the fake-library parity
/// program and the libcuda-only import).
pub fn extern_corpus() -> Vec<(&'static str, String, Option<i32>)> {
    let fixture = |name: &str| {
        std::fs::read_to_string(repo().join("tests/fixtures/native_lane_extern").join(name))
            .unwrap()
    };
    vec![
        ("close", CLOSE_EXTERN.to_string(), Some(9)),
        ("fstat", FSTAT_EXTERN.to_string(), None),
        ("corpus", FIXED_SIGNATURE_CORPUS.to_string(), Some(0)),
        ("close_range", U32_CLOSE_RANGE.to_string(), Some(22)),
        ("wide_rows", WIDE_ROWS.to_string(), Some(42)),
        ("net_rows", NET_ROWS.to_string(), Some(42)),
        (
            "native_c_abi_fixed",
            fixture("native_c_abi_fixed.mind"),
            None,
        ),
        (
            "l1_extern_close_flip",
            fixture("l1_extern_close_flip.mind"),
            Some(0),
        ),
        ("gpu_vec_add", gpu_fixture("vec_add.mind"), None),
        ("gpu_fake_rows", gpu_fixture("fake_rows.mind"), None),
        ("gpu_libcuda_only", gpu_fixture("gpu_only.mind"), None),
    ]
}

/// Run a built program with extra environment variables on `/dev/null` stdin:
/// its exit code and stdout bytes. stderr must stay empty like `run`.
pub fn run_with_env(binary: &Path, vars: &[(&str, &str)]) -> (i32, Vec<u8>) {
    let ran = capture(binary, vars);
    (finished_code(binary, ran.status), ran.stdout)
}

/// `(revision, main.mind)` of the nearest ancestor of HEAD whose compiler
/// source does not name `libcuda.so.1`. The walk is the property "before the
/// driver rows"; it never returns HEAD, and it never reads the worktree copy
/// of the file. It follows the compiler's own line ([`compiler_parent`]).
pub fn base_compiler_source() -> (String, String) {
    fn git(args: &[&str]) -> Output {
        Command::new("git")
            .args(args)
            .current_dir(repo())
            .output()
            .expect("git")
    }
    let head = String::from_utf8(git(&["rev-parse", "HEAD"]).stdout).unwrap();
    let head = head.trim().to_string();
    assert_eq!(head.len(), 40, "HEAD is not a commit id: {head}");
    let mut rev = head.clone();
    // Ends at the pre-driver compiler, or fails closed at a commit with no parent.
    loop {
        let found = git(&[
            "grep",
            "-F",
            "-q",
            "-e",
            "libcuda.so.1",
            &rev,
            "--",
            "examples/mindc_mind/main.mind",
        ]);
        match found.status.code() {
            Some(1) => {
                assert_ne!(
                    rev, head,
                    "HEAD has no driver rows, so this walk would compare it to itself"
                );
                let spec = format!("{rev}:examples/mindc_mind/main.mind");
                let show = git(&["show", &spec]);
                assert!(
                    show.status.success(),
                    "git show {spec}: {}",
                    String::from_utf8_lossy(&show.stderr)
                );
                let text = String::from_utf8(show.stdout).unwrap();
                assert!(
                    !text.contains("libcuda.so.1"),
                    "{rev} was selected as the base but its main.mind names the driver"
                );
                return (rev, text);
            }
            Some(0) => {}
            other => panic!(
                "git grep {rev}: exit {other:?}: {}",
                String::from_utf8_lossy(&found.stderr)
            ),
        }
        rev = compiler_parent(&rev, git);
    }
}

/// Same main.mind blob, else the first parent. A pull_request first parent is the base.
fn compiler_parent(rev: &str, git: fn(&[&str]) -> Output) -> String {
    let blob = |r: &str| {
        let spec = format!("{r}:examples/mindc_mind/main.mind");
        let out = git(&["rev-parse", &spec]);
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        out.status.success().then_some(text)
    };
    let listed = git(&["rev-list", "--parents", "-n", "1", rev]);
    let err = String::from_utf8_lossy(&listed.stderr).into_owned();
    assert!(listed.status.success(), "{err}");
    let line = String::from_utf8(listed.stdout).unwrap();
    let parents: Vec<String> = line.split_whitespace().skip(1).map(String::from).collect();
    assert!(!parents.is_empty(), "no parent of {rev}: shallow or root");
    let own = blob(rev);
    for parent in &parents {
        if own.is_some() && blob(parent) == own {
            return parent.clone();
        }
    }
    parents[0].clone()
}

fn cc_shared(dir: &Path, source: &str, extra: &[&str], why: &str) -> PathBuf {
    let source = repo().join(source);
    let out = dir.join("libcuda.so.1");
    let built = Command::new("cc")
        .args(["-shared", "-fPIC"])
        .args(extra)
        .arg("-o")
        .arg(&out)
        .arg(&source)
        .output()
        .expect(why);
    assert!(
        built.status.success(),
        "{why}: {}",
        String::from_utf8_lossy(&built.stderr)
    );
    out
}

/// Alignment probe. `-O0 -fno-omit-frame-pointer` keeps the entry `rsp`.
pub fn build_align_libcuda(dir: &Path) -> PathBuf {
    cc_shared(
        dir,
        "tests/fixtures/native_lane_gpu/align_driver.c",
        &["-O0", "-fno-omit-frame-pointer"],
        "cc must run: the alignment probe is compiled with it",
    )
}

/// Fake driver, soname filename `libcuda.so.1`.
pub fn build_fake_libcuda(dir: &Path) -> PathBuf {
    cc_shared(
        dir,
        "tests/fixtures/native_lane_gpu/fake_driver.c",
        &["-O1"],
        "cc must run: the fake driver fixture is compiled with it",
    )
}

/// A fixture-only CLANG/CC shim: the MLIR build resolves its IR compile
/// through `CLANG=<shim>` and its executable link through `CC=<shim>`, and
/// the shim appends the fake library to the link invocation (the one without
/// `-c`), so the cu* declarations resolve and the linked binary records the
/// library path as its NEEDED entry. Compile invocations (the IR object and
/// the runtime-support C stub) and tool probes pass through untouched.
pub fn clang_shim(dir: &Path, lib: &Path) -> PathBuf {
    let shim = dir.join("clang-fixture-shim.sh");
    std::fs::write(
        &shim,
        format!(
            r#"#!/bin/sh
for a in "$@"; do
  case "$a" in
    -c|-S|-E|--version)
      exec clang "$@"
      ;;
  esac
done
exec clang "$@" -Wl,{}
"#,
            lib.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o700)).unwrap();
    shim
}

// MC-6 driver rows (tests/native_lane_gpu_loader.rs): case table, builds and probes.

/// The `kind` rows of `mc6_cases.tsv`, split on tabs (a literal `\n` is a newline).
pub fn mc6_cases(kind: &str) -> Vec<Vec<String>> {
    let cases: Vec<Vec<String>> = gpu_fixture("mc6_cases.tsv")
        .lines()
        .filter_map(|l| l.strip_prefix(kind)?.strip_prefix('\t'))
        .map(|l| l.split('\t').map(|f| f.replace("\\n", "\n")).collect())
        .collect();
    assert!(!cases.is_empty(), "no {kind} rows in mc6_cases.tsv");
    cases
}

/// `fake_rows2.mind` with both cuMemsetD8_v2 byte arguments set to `arg`.
pub fn memset_program(arg: &str) -> String {
    let mut program = gpu_fixture("fake_rows2.mind");
    for from in ["(4660, (v & 255) as u32, 12)", "(4660, 200 as u32, 12)"] {
        assert_eq!(program.matches(from).count(), 1, "{from} moved");
        program = program.replacen(from, &format!("(4660, {arg}, 12)"), 1);
    }
    program
}

/// Builds `source` in its own directory under `case`, so a refused build never
/// finds an earlier build's program: the output and the binary path.
pub fn build_apart(case: &Case, source: &str) -> (Output, PathBuf) {
    let at = tempfile::tempdir_in(case.dir.path()).unwrap().keep();
    case.build_in(&at, source, "default")
}

/// The stderr of a self-host refusal of `source` (exit 3, the bridge's
/// rejection line, no artifact), else None.
pub fn self_host_refusal(case: &Case, source: &str) -> Option<String> {
    let (build, binary) = build_apart(case, source);
    let stderr = String::from_utf8_lossy(&build.stderr).into_owned();
    let refused = build.status.code() == Some(3)
        && stderr.contains("pure-MIND compiler rejected the program (exit 3)")
        && !binary.exists();
    refused.then_some(stderr)
}

pub fn builds_apart(case: &Case, source: &str) -> bool {
    let (build, binary) = build_apart(case, source);
    build.status.code() == Some(0) && binary.exists()
}

/// `call [rip+disp32]` (ff 15) directly followed by `add rsp, imm32` (48 81 c4).
pub fn call_cleanups(image: &[u8], imm: u32) -> usize {
    let add = [&[0x48, 0x81, 0xc4][..], &imm.to_le_bytes()].concat();
    let call_then_add = |w: &&[u8]| w[..2] == [0xff, 0x15] && w[6..] == add[..];
    image.windows(13).filter(call_then_add).count()
}
