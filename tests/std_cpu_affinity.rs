// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! 1024-bit CPU affinity mask (`std/cpu_affinity.mind`) and the
//! `io_reactor_*` forwards in `std/iouring.mind`.
//!
//! Contract: the mask is proved on EVERY host, and a leg that cannot run says so.
//! No fixed core number is assumed to exist, and nothing passes silently.
//!
//! 1. Every host (`pin_records_one_bit_on_every_host`): a recording stand-in for
//!    `sched_setaffinity` / `sched_getaffinity` (an `LD_PRELOAD` library, the
//!    kernel is never called) captures the 128-byte mask the std code hands to
//!    the kernel. Pinning cores 0, 63, 64, 71 and 1023 records exactly one bit
//!    each, and `io_reactor_affinity_has` reads each bit back from a fed mask.
//!    The cores need not exist. This is the guard for "a core at or above 64
//!    overflowed".
//! 2. Real kernel (`pin_has_restore_and_range`), with cores taken from the
//!    CURRENT allowed mask: the lowest and the highest allowed core are pinned
//!    and read back; for every core 0..1023 `io_reactor_affinity_has` equals the
//!    bit libc `sched_getaffinity` reports; `io_reactor_pin_demo` on an allowed
//!    core returns 1 and the full 128-byte mask after it equals the mask before;
//!    cores -1 and 1024 return -22.
//! 3. High-core real-kernel leg: when the current mask has an allowed core at or
//!    above 64, that core is pinned for real and asserted. When it has none the
//!    leg cannot run, and the test reports it through the harness marker for an
//!    optional input (`SDLC-GATE std_cpu_affinity ran=0 class=optional`) with a
//!    reason that contains `no allowed CPU at or above 64`.
//! 4. `pin_demo_reports_failed_restore`: a shim that fails the restore
//!    `sched_setaffinity` makes `pin_demo` return -5 (EIO). The shim tells a
//!    restore from a pin by its bit count, so the leg needs two allowed CPUs;
//!    with one it reports the same optional-input marker.
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
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const CPU_SRC: &str = include_str!("../std/cpu_affinity.mind");
const IOURING_SRC: &str = include_str!("../std/iouring.mind");

/// Thin forwards so ctypes calls the public `io_reactor_*` names. Concatenated
/// with `CPU_SRC` into one translation unit, so this file tests the module
/// source itself and does not depend on the std link set. The import route is
/// covered by `tests/std_substrate_native_link.rs`.
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
        assert!(
            libmind::std_qualified::is_row("std.cpu_affinity", required),
            "`{required}` makes an affinity system call, so it is a World row of std_qualified"
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
        assert!(
            libmind::std_qualified::is_row("std.iouring", fwd),
            "`{fwd}` forwards to the World function `{dest}`, so it is a row of std_qualified"
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

/// Run `script` in `python3` under `timeout`. Panics when it does not finish in
/// 30 s or exits non-zero; returns its stdout. `env` is added to the child's
/// environment (the stand-in shims ride `LD_PRELOAD`).
fn run_py(label: &str, script: &str, env: &[(&str, &OsStr)]) -> String {
    let out = Command::new("timeout")
        .args(["--kill-after=5", "30", "python3", "-c", script])
        .envs(env.iter().copied())
        .output()
        .expect("run timeout(1) + python3");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let code = out.status.code();
    assert!(
        code != Some(124) && code != Some(137) && code.is_some(),
        "{label} did not finish within 30 s (exit {code:?}):\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        out.status.success(),
        "{label} failed (exit {code:?}):\nstdout: {stdout}\nstderr: {stderr}"
    );
    stdout
}

/// `run_py` for a probe that loads the compiled library: binds `SO` first.
fn run_probe(label: &str, so: &Path, script: &str, env: &[(&str, &OsStr)]) -> String {
    run_py(label, &format!("SO = r'{}'\n{script}", so.display()), env)
}

/// The cores this thread may run on, ascending. An oracle apart from the probes:
/// Python's own `os.sched_getaffinity`, run without any shim. A child inherits
/// the affinity of the thread that spawns it, so it sees what the probes see.
fn allowed_cores() -> Vec<u32> {
    let stdout = run_py(
        "allowed-core oracle",
        "import os\nprint(*sorted(os.sched_getaffinity(0)))",
        &[],
    );
    let cores: Vec<u32> = stdout
        .split_whitespace()
        .map(|t| {
            t.parse::<u32>()
                .unwrap_or_else(|e| panic!("allowed-core oracle printed `{t}`: {e}"))
        })
        .collect();
    assert!(
        !cores.is_empty(),
        "allowed-core oracle printed no core:\n{stdout}"
    );
    cores
}

/// Names and the 128-byte-mask helpers both probes share. The expected bit comes
/// from the kernel's byte layout (core / 8, core % 8), not from the word
/// arithmetic of the code under test.
const PY_PRELUDE: &str = r#"
import ctypes, errno
assert errno.EINVAL == 22, errno.EINVAL
lib = ctypes.CDLL(SO)
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

def bit_set(raw, core):
    return (raw[core // 8] >> (core % 8)) & 1

def one_bit(core):
    raw = bytearray(128)
    raw[core // 8] = 1 << (core % 8)
    return bytes(raw)

def bits(raw):
    return [c for c in range(1024) if bit_set(raw, c)]
"#;

/// Recording stand-in for the affinity system calls: `sched_setaffinity` keeps
/// the 128-byte mask it was handed, `sched_getaffinity` reports a mask the probe
/// feeds it, and neither calls the kernel, so no core has to exist.
const RECORD_SHIM: &str = r#"
#define _GNU_SOURCE
#include <errno.h>
#include <string.h>

static unsigned char rec_mask[128];
static unsigned char fed_mask[128];
static unsigned long rec_size = 0;
static long set_calls = 0;
static long get_calls = 0;

int sched_setaffinity(int pid, unsigned long cpusetsize, const void *mask) {
    (void)pid;
    if (mask == 0) {
        errno = EFAULT;
        return -1;
    }
    memset(rec_mask, 0, sizeof rec_mask);
    memcpy(rec_mask, mask, cpusetsize > 128 ? 128 : cpusetsize);
    rec_size = cpusetsize;
    set_calls++;
    return 0;
}

int sched_getaffinity(int pid, unsigned long cpusetsize, void *mask) {
    (void)pid;
    if (mask == 0) {
        errno = EFAULT;
        return -1;
    }
    memset(mask, 0, cpusetsize);
    memcpy(mask, fed_mask, cpusetsize > 128 ? 128 : cpusetsize);
    get_calls++;
    return 0;
}

long mind_aff_sets(void) { return set_calls; }
long mind_aff_gets(void) { return get_calls; }
unsigned long mind_aff_size(void) { return rec_size; }
void mind_aff_copy(unsigned char *out) { memcpy(out, rec_mask, sizeof rec_mask); }
void mind_aff_feed(const unsigned char *in) { memcpy(fed_mask, in, sizeof fed_mask); }
"#;

/// Leg 1: runs under `RECORD_SHIM`, so it needs no particular CPU.
const RECORD_PROBE: &str = r#"
shim = ctypes.CDLL(None)
shim.mind_aff_sets.restype = ctypes.c_long
shim.mind_aff_gets.restype = ctypes.c_long
shim.mind_aff_size.restype = ctypes.c_ulong
shim.mind_aff_copy.argtypes = [ctypes.c_void_p]
shim.mind_aff_feed.argtypes = [ctypes.c_void_p]

def recorded():
    buf = ctypes.create_string_buffer(128)
    shim.mind_aff_copy(buf)
    return bytes(buf.raw)

def feed(raw):
    shim.mind_aff_feed(ctypes.create_string_buffer(raw))

CORES = (0, 63, 64, 71, 1023)
for core in CORES:
    before = shim.mind_aff_sets()
    rc = lib.io_reactor_pin_cpu(core)
    assert rc == 0, f'pin({core})={rc}'
    assert shim.mind_aff_sets() == before + 1, f'pin({core}) made no single set call'
    assert shim.mind_aff_size() == 128, f'pin({core}) passed size {shim.mind_aff_size()}'
    got = recorded()
    assert got == one_bit(core), f'pin({core}) handed the kernel bits {bits(got)}'

nearby = sorted({c + d for c in CORES for d in (-1, 0, 1) if 0 <= c + d < 1024})
for core in CORES:
    feed(one_bit(core))
    for other in nearby:
        want = 1 if other == core else 0
        got = lib.io_reactor_affinity_has(other)
        assert got == want, f'has({other})={got} while only bit {core} is set'

sets, gets = shim.mind_aff_sets(), shim.mind_aff_gets()
for fn in (lib.io_reactor_pin_cpu, lib.io_reactor_affinity_has):
    for core in (-1, 1024):
        rc = fn(core)
        assert rc == -22, f'range {core} gave {rc}'
assert (shim.mind_aff_sets(), shim.mind_aff_gets()) == (sets, gets), 'a range error made a call'
print('ok')
"#;

/// Leg 2 and 3: the real kernel, cores taken from the current allowed mask.
const KERNEL_PROBE: &str = r#"
libc = ctypes.CDLL('libc.so.6', use_errno=True)
libc.sched_getaffinity.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
libc.sched_getaffinity.restype = ctypes.c_int
libc.sched_setaffinity.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
libc.sched_setaffinity.restype = ctypes.c_int

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

saved = mask128()
allowed = bits(saved)
assert allowed, 'sched_getaffinity reported an empty mask'

def pin_and_read_back(core):
    rc = lib.io_reactor_pin_cpu(core)
    assert rc == 0, f'pin({core})={rc}'
    h = lib.io_reactor_affinity_has(core)
    assert h == 1, f'has({core})={h}'
    got = mask128()
    assert got == one_bit(core), f'pin({core}) left the mask at {bits(got)}'
    set128(saved)

try:
    assert lib.io_reactor_pin_cpu(-1) == -22, 'pin(-1)'
    assert lib.io_reactor_pin_cpu(1024) == -22, 'pin(1024)'
    assert lib.io_reactor_affinity_has(-1) == -22, 'has(-1)'
    assert lib.io_reactor_affinity_has(1024) == -22, 'has(1024)'
    for core in range(1024):
        want = bit_set(saved, core)
        got = lib.io_reactor_affinity_has(core)
        assert got == want, f'has({core})={got}, libc says {want}'
    for core in sorted({allowed[0], allowed[-1]}):
        pin_and_read_back(core)
    before = mask128()
    demo = lib.io_reactor_pin_demo(allowed[0])
    assert demo == 1, f'pin_demo({allowed[0]})={demo}'
    after = mask128()
    assert after == before, f'pin_demo restore changed the mask: {bits(before)} -> {bits(after)}'
    top = [c for c in allowed if c >= 64]
    if top:
        for core in sorted({top[0], top[-1]}):
            pin_and_read_back(core)
        print(f'high_core=ran {top[0]} {top[-1]}')
    else:
        print('high_core=none')
    print('ok')
finally:
    set128(saved)
"#;

/// Compile `source` into `dir/<name>.so` with the host C compiler.
fn build_shim(dir: &Path, name: &str, source: &str) -> PathBuf {
    let src = dir.join(format!("{name}.c"));
    let so = dir.join(format!("{name}.so"));
    std::fs::write(&src, source).expect("write shim source");
    let out = Command::new("gcc")
        .args(["-shared", "-fPIC", "-O2", "-o"])
        .arg(&so)
        .arg(&src)
        .output()
        .expect("gcc");
    assert!(
        out.status.success(),
        "gcc {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    so
}

#[test]
fn pin_records_one_bit_on_every_host() {
    let Some(so) = affinity_so() else {
        return;
    };
    let dir = common::scratch_dir("std_cpu_affinity_record");
    let shim = build_shim(&dir, "record_shim", RECORD_SHIM);
    let script = format!("{PY_PRELUDE}\n{RECORD_PROBE}");
    let stdout = run_probe(
        "recording stand-in probe",
        so,
        &script,
        &[("LD_PRELOAD", shim.as_os_str())],
    );
    assert!(
        stdout.lines().any(|l| l == "ok"),
        "recording stand-in probe printed no ok:\nstdout: {stdout}"
    );
}

#[test]
fn pin_has_restore_and_range() {
    let Some(so) = affinity_so() else {
        return;
    };
    let cores = allowed_cores();
    let script = format!("{PY_PRELUDE}\n{KERNEL_PROBE}");
    let stdout = run_probe("affinity probe", so, &script, &[]);
    assert!(
        stdout.lines().any(|l| l == "ok"),
        "affinity probe printed no ok:\nstdout: {stdout}"
    );
    let high: Vec<u32> = cores.iter().copied().filter(|&c| c >= 64).collect();
    if let (Some(lowest), Some(highest)) = (high.first(), high.last()) {
        let want = format!("high_core=ran {lowest} {highest}");
        assert!(
            stdout.lines().any(|l| l == want),
            "allowed cores reach {highest}, so the high-core leg must run (`{want}`):\n{stdout}"
        );
    } else {
        assert!(
            stdout.lines().any(|l| l == "high_core=none"),
            "no allowed core is at or above 64, so the probe must say so:\n{stdout}"
        );
        let top = cores.last().copied().unwrap_or_default();
        // OPTIONAL INPUT: a core at or above 64 in this host's own affinity
        // mask. The host supplies it and a small runner has none; the
        // host-independent proof is `pin_records_one_bit_on_every_host`.
        crate::common::gate::skipped_optional(
            "std_cpu_affinity",
            &format!(
                "no allowed CPU at or above 64 (highest allowed core: {top}); the \
                 real-kernel high-core leg did not run"
            ),
        );
    }
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

/// Runs `pin_demo` on the lowest allowed core. With fewer than two allowed cores
/// the shim cannot tell a restore from a pin, so the probe says `restore=none`.
const RESTORE_PROBE: &str = r#"
import ctypes
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

saved = mask128()
raw = bytes(saved.raw)
allowed = [c for c in range(1024) if bit_set(raw, c)]
lib = ctypes.CDLL(SO)
fn = lib.io_reactor_pin_demo
fn.restype = ctypes.c_int64
fn.argtypes = [ctypes.c_int64]
if len(allowed) < 2:
    print('restore=none')
else:
    try:
        demo = int(fn(allowed[0]))
        print(f'demo={demo}')
    finally:
        set_raw(saved)
"#;

#[test]
fn pin_demo_reports_failed_restore() {
    let Some(so) = affinity_so() else {
        return;
    };
    let two_cores = allowed_cores().len() >= 2;
    let dir = common::scratch_dir("std_cpu_affinity_restore");
    let shim = build_shim(&dir, "restore_shim", RESTORE_SHIM);
    for mode in ["restore", "both"] {
        let stdout = run_probe(
            &format!("restore probe mode={mode}"),
            so,
            RESTORE_PROBE,
            &[
                ("LD_PRELOAD", shim.as_os_str()),
                ("MIND_AFF_FAIL", OsStr::new(mode)),
            ],
        );
        if two_cores {
            assert!(
                stdout.contains("demo=-5"),
                "mode={mode} must return -5 (EIO) when restore fails:\n{stdout}"
            );
        } else {
            assert!(
                stdout.contains("restore=none"),
                "mode={mode}: one allowed core, so the probe must say so:\n{stdout}"
            );
        }
    }
    if !two_cores {
        // OPTIONAL INPUT: a second allowed CPU in this host's own affinity
        // mask; the shim tells a restore from a pin by its bit count.
        crate::common::gate::skipped_optional(
            "std_cpu_affinity",
            "fewer than 2 allowed CPUs; the failed-restore leg did not run",
        );
    }
}
