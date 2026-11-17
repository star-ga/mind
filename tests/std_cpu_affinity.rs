// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! 1024-bit CPU affinity mask (`std/cpu_affinity.mind`) and the
//! `io_reactor_*` forwards in `std/iouring.mind`.
//!
//! Runtime: pin cores 0, 63, 64, 71, read them back with
//! `io_reactor_affinity_has`, restore; cores -1 and 1024 return -22;
//! `io_reactor_pin_demo(70)` returns 1 and the 128-byte mask after restore
//! equals the mask before (libc `sched_getaffinity`, all 16 words).
//! A shim that fails the restore `sched_setaffinity` makes `pin_demo` return
//! -5 (EIO). A host with fewer than 72 CPUs, or a mask that lacks those cores, FAILS.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test std_cpu_affinity`

#![cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

const CPU_SRC: &str = include_str!("../std/cpu_affinity.mind");
const IOURING_SRC: &str = include_str!("../std/iouring.mind");

/// Thin forwards so ctypes calls the public `io_reactor_*` names. Concatenated
/// with `CPU_SRC` into one translation unit: a flat `mindc --emit-shared` of a
/// file that only `import`s `std.cpu_affinity` leaves those callees undefined.
const FORWARDS: &str = r#"
pub fn io_reactor_pin_cpu(core: i64) -> i64 {
    cpu_affinity_pin(core)
}

pub fn io_reactor_affinity_mask() -> i64 {
    cpu_affinity_mask()
}

pub fn io_reactor_affinity_has(core: i64) -> i64 {
    cpu_affinity_has(core)
}

pub fn io_reactor_pin_demo(core: i64) -> i64 {
    cpu_affinity_pin_demo(core)
}
"#;

fn fn_names(ir: &libmind::ir::IRModule) -> Vec<&str> {
    ir.instrs
        .iter()
        .filter_map(|i| match i {
            Instr::FnDef { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

fn fn_body<'a>(ir: &'a libmind::ir::IRModule, name: &str) -> &'a [Instr] {
    ir.instrs
        .iter()
        .find_map(|i| match i {
            Instr::FnDef { name: n, body, .. } if n == name => Some(body.as_slice()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing FnDef `{name}`"))
}

fn body_calls(body: &[Instr], want: &str) -> bool {
    body.iter()
        .any(|i| matches!(i, Instr::Call { name, .. } if name == want))
}

#[test]
fn cpu_affinity_parses_and_lowers() {
    let module = parser::parse(CPU_SRC).expect("std/cpu_affinity.mind must parse");
    let ir = lower_to_ir(&module).expect("lowering");
    let names = fn_names(&ir);
    for required in [
        "cpu_affinity_pin",
        "cpu_affinity_mask",
        "cpu_affinity_has",
        "cpu_affinity_pin_demo",
    ] {
        assert!(
            names.contains(&required),
            "std.cpu_affinity must define `{required}`; found {names:?}"
        );
    }
}

#[test]
fn iouring_forwards_io_reactor_affinity_names() {
    let module = parser::parse(IOURING_SRC).expect("std/iouring.mind must parse");
    let ir = lower_to_ir(&module).expect("lowering");
    let names = fn_names(&ir);
    for required in [
        "io_reactor_pin_cpu",
        "io_reactor_affinity_mask",
        "io_reactor_affinity_has",
        "io_reactor_pin_demo",
    ] {
        assert!(
            names.contains(&required),
            "std.iouring must define `{required}`; found {names:?}"
        );
    }
    let pairs = [
        ("io_reactor_pin_cpu", "cpu_affinity_pin"),
        ("io_reactor_affinity_mask", "cpu_affinity_mask"),
        ("io_reactor_affinity_has", "cpu_affinity_has"),
        ("io_reactor_pin_demo", "cpu_affinity_pin_demo"),
    ];
    for (fwd, dest) in pairs {
        assert!(
            body_calls(fn_body(&ir, fwd), dest),
            "`{fwd}` must call `{dest}`"
        );
    }
}

fn affinity_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        let mindc = common::mindc_bin();
        if !mindc.exists() {
            crate::common::gate::skipped("std_cpu_affinity", "affinity: mindc not found");
            return None;
        }
        let dir = common::scratch_dir("std_cpu_affinity");
        let src = dir.join("io_reactor_affinity.mind");
        let so = dir.join("io_reactor_affinity.so");
        let body = format!("{CPU_SRC}\n{FORWARDS}");
        std::fs::write(&src, body).expect("write affinity translation unit");
        let out = Command::new(&mindc)
            .args([
                src.to_str().unwrap(),
                "--allow-nondeterministic",
                "--emit-shared",
                so.to_str().unwrap(),
            ])
            .output()
            .expect("run mindc");
        if !crate::common::gate::compiled("std_cpu_affinity", &out) {
            return None;
        }
        Some(so)
    })
    .as_ref()
}

/// Independent 128-byte mask via libc, not via the MIND surface.
const PROBE: &str = r#"
import ctypes, errno, os, sys
assert errno.EINVAL == 22, errno.EINVAL
lib = ctypes.CDLL(SO)
libc = ctypes.CDLL('libc.so.6', use_errno=True)
libc.sched_getaffinity.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
libc.sched_getaffinity.restype = ctypes.c_int
libc.sched_setaffinity.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
libc.sched_setaffinity.restype = ctypes.c_int
for name in (
    'io_reactor_pin_cpu',
    'io_reactor_affinity_has',
    'io_reactor_pin_demo',
):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
lib.io_reactor_affinity_mask.restype = ctypes.c_int64
lib.io_reactor_affinity_mask.argtypes = []

def mask128():
    buf = ctypes.create_string_buffer(128)
    r = libc.sched_getaffinity(0, 128, buf)
    if r < 0:
        raise OSError(ctypes.get_errno(), 'sched_getaffinity')
    return bytes(buf.raw)

def set128(raw):
    buf = ctypes.create_string_buffer(raw)
    r = libc.sched_setaffinity(0, 128, buf)
    if r < 0:
        raise OSError(ctypes.get_errno(), 'sched_setaffinity')

def bit_set(raw, core):
    off = (core // 64) * 8
    w = int.from_bytes(raw[off:off + 8], 'little')
    return (w >> (core % 64)) & 1

ncpu = os.cpu_count()
assert ncpu is not None and ncpu >= 72, f'ncpu={ncpu} need >= 72'
saved = mask128()
need = (0, 63, 64, 71, 70)
missing = [c for c in need if bit_set(saved, c) != 1]
assert not missing, f'current mask lacks cores {missing}'
try:
    assert lib.io_reactor_pin_cpu(-1) == -22, 'pin(-1)'
    assert lib.io_reactor_pin_cpu(1024) == -22, 'pin(1024)'
    assert lib.io_reactor_affinity_has(-1) == -22, 'has(-1)'
    assert lib.io_reactor_affinity_has(1024) == -22, 'has(1024)'
    for core in (0, 63, 64, 71):
        rc = lib.io_reactor_pin_cpu(core)
        assert rc == 0, f'pin({core})={rc}'
        h = lib.io_reactor_affinity_has(core)
        assert h == 1, f'has({core})={h}'
        set128(saved)
    before = mask128()
    demo = lib.io_reactor_pin_demo(70)
    assert demo == 1, f'pin_demo(70)={demo}'
    after = mask128()
    assert after == before, 'pin_demo restore lost mask bits'
    print('ok')
finally:
    set128(saved)
"#;

#[test]
fn pin_has_restore_and_range() {
    let Some(so) = affinity_so() else {
        return;
    };
    let py = format!("SO = r'{}'\n{PROBE}", so.display());
    let out = Command::new("timeout")
        .args(["--kill-after=5", "30", "python3", "-c", &py])
        .output()
        .expect("run timeout(1) + python3");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let code = out.status.code();
    assert!(
        code != Some(124) && code != Some(137) && code.is_some(),
        "affinity probe did not finish within 30 s (exit {code:?}):\n\
         stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        out.status.success(),
        "affinity probe failed (exit {code:?}):\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("ok"),
        "affinity probe printed no ok:\nstdout: {stdout}\nstderr: {stderr}"
    );
}

/// Fails the restore `sched_setaffinity` (a mask with more than one bit) with
/// errno 5. `MIND_AFF_FAIL=both` also fails the single-bit pin with errno 1.
/// The real syscall stays available for the probe's own cleanup.
const RESTORE_SHIM: &str = r#"
#define _GNU_SOURCE
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

static int fail_pin = 0;
static int fail_restore = 0;

static int bits_of(const unsigned char *p, unsigned long n) {
    unsigned long i;
    int b = 0;
    if (n > 128) {
        n = 128;
    }
    for (i = 0; i < n; i++) {
        unsigned char c = p[i];
        while (c) {
            b += c & 1;
            c = (unsigned char)(c >> 1);
        }
    }
    return b;
}

int sched_setaffinity(int pid, unsigned long cpusetsize, const void *mask) {
    const char *mode = getenv("MIND_AFF_FAIL");
    int bits = 0;
    if (mask != 0) {
        bits = bits_of((const unsigned char *)mask, cpusetsize);
    }
    if (mode != 0 && strcmp(mode, "both") == 0 && bits == 1 && fail_pin == 0) {
        fail_pin = 1;
        errno = 1;
        return -1;
    }
    if (mode != 0 && bits > 1 && fail_restore == 0) {
        fail_restore = 1;
        errno = 5;
        return -1;
    }
    return (int)syscall(SYS_sched_setaffinity, pid, cpusetsize, mask);
}
"#;

const RESTORE_PROBE: &str = r#"
import ctypes, os
SYS_GET = 204
SYS_SET = 203
libc = ctypes.CDLL(None, use_errno=True)
syscall = libc.syscall
syscall.restype = ctypes.c_long

def mask128():
    buf = ctypes.create_string_buffer(128)
    r = syscall(SYS_GET, 0, 128, ctypes.cast(buf, ctypes.c_void_p))
    if r < 0:
        raise OSError(ctypes.get_errno(), 'sched_getaffinity')
    return buf

def set_raw(buf):
    r = syscall(SYS_SET, 0, 128, ctypes.cast(buf, ctypes.c_void_p))
    if r < 0:
        raise OSError(ctypes.get_errno(), 'sched_setaffinity')

def bit_set(raw, core):
    off = (core // 64) * 8
    w = int.from_bytes(raw[off:off + 8], 'little')
    return (w >> (core % 64)) & 1

ncpu = os.cpu_count()
assert ncpu is not None and ncpu >= 72, f'ncpu={ncpu} need >= 72'
saved = mask128()
raw = bytes(saved.raw)
assert bit_set(raw, 0) == 1 and bit_set(raw, 70) == 1, 'mask lacks 0 or 70'
lib = ctypes.CDLL(SO)
fn = lib.io_reactor_pin_demo
fn.restype = ctypes.c_int64
fn.argtypes = [ctypes.c_int64]
try:
    demo = int(fn(70))
    print(f'demo={demo}')
finally:
    set_raw(saved)
"#;

fn restore_shim(dir: &std::path::Path) -> PathBuf {
    let src = dir.join("restore_shim.c");
    let so = dir.join("restore_shim.so");
    std::fs::write(&src, RESTORE_SHIM).expect("write restore shim");
    let out = Command::new("gcc")
        .args(["-shared", "-fPIC", "-O2", "-o"])
        .arg(&so)
        .arg(&src)
        .output()
        .expect("gcc");
    assert!(
        out.status.success(),
        "gcc restore shim failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    so
}

fn run_restore_mode(so: &std::path::Path, shim: &std::path::Path, mode: &str) -> String {
    let py = format!("SO = r'{}'\n{RESTORE_PROBE}", so.display());
    let out = Command::new("timeout")
        .args(["--kill-after=5", "30", "python3", "-c", &py])
        .env("LD_PRELOAD", shim)
        .env("MIND_AFF_FAIL", mode)
        .output()
        .expect("run restore probe");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let code = out.status.code();
    assert!(
        code != Some(124) && code != Some(137) && code.is_some(),
        "restore probe mode={mode} did not finish (exit {code:?}):\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        out.status.success(),
        "restore probe mode={mode} failed (exit {code:?}):\nstdout: {stdout}\nstderr: {stderr}"
    );
    stdout
}

#[test]
fn pin_demo_reports_failed_restore() {
    let Some(so) = affinity_so() else {
        return;
    };
    let dir = common::scratch_dir("std_cpu_affinity_restore");
    let shim = restore_shim(&dir);
    for mode in ["restore", "both"] {
        let stdout = run_restore_mode(so, &shim, mode);
        assert!(
            stdout.contains("demo=-5"),
            "mode={mode} must return -5 (EIO) when restore fails:\n{stdout}"
        );
    }
}
