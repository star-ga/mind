// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `__mind_open` failure contract: a failed open is NEGATIVE at every layer.
//!
//! `runtime-support/mind_intrinsics.c::__mind_open` must return a negative value
//! when `open(2)` fails; std.fs callers sign-test the result (`if fd < 0`) and
//! nothing finer. `tests/std_surface_net_fs_process.rs::fs_read_file_native`
//! cannot pin this by itself: an `__mind_open` that hands back a bogus
//! non-negative fd on failure still makes `read_file_native` return -1, because
//! the follow-up `read(2)` on an unopened descriptor fails with EBADF. A runtime
//! returning fd 999 for a missing path stayed green there — and with a live
//! descriptor at 999 the same call reads an unrelated file.
//!
//! This file asserts the contract three ways, so each layer is checked directly:
//!   1. the C intrinsic — `__mind_open(missing)` must be `< 0`, and
//!      `__mind_open(existing)` must be `>= 0` (positive control: the negative
//!      assertion is not vacuous because open never works);
//!   2. a MIND program sign-testing the result (`open_status`) must report
//!      failure (0) for the missing path and success (1) for a real file;
//!   3. `read_file_native` on a missing path must return -1 even when the
//!      descriptor numbers a bogus fd could alias are LIVE — every fd in a probe
//!      window is `dup2`'d onto a decoy file, so an implementation that ignores
//!      the open error reads the decoy instead of failing.
//!
//! Gate: `std-surface` + `mlir-build` + `cross-module-imports`, unix only — the
//! fixture is compiled with `mindc --emit-shared` and driven through ctypes.

#![cfg(feature = "std-surface")]

mod common;

#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
mod mlir_functional {
    use std::path::PathBuf;
    use std::process::Command;

    /// Test-only probe appended to `std/fs.mind`: 1 if `__mind_open` succeeded,
    /// 0 if it reported failure — exactly the sign test std.fs callers perform.
    const OPEN_STATUS_PROBE: &str = "
pub fn open_status(path_addr: i64) -> i64 {
    let fd: i64 = __mind_open(path_addr);
    if fd < 0 {
        return 0;
    }
    1
}
";

    /// Compile `std/fs.mind` + the probe to a shared library, or `None` once a
    /// missing `mindc` has been routed through the one fail-closed skip gate.
    ///
    /// `--allow-nondeterministic` is required for the same reason as every
    /// std.fs fixture in `std_surface_net_fs_process.rs`: the module genuinely
    /// reads the world, and `tests/determinism_veto_control.rs` is the positive
    /// control proving the veto still fires on `std/fs.mind` without the flag.
    fn compile_fs_with_probe(tag: &str) -> Option<PathBuf> {
        let mindc = crate::common::mindc_or_skip("std_surface_fs_mind_open")?;
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_fs_mind_open");
        std::fs::create_dir_all(&dir).expect("create output dir");
        let src_path = dir.join(format!("{tag}.mind"));
        let so_path = dir.join(format!("{tag}.so"));
        let src = format!("{}\n{OPEN_STATUS_PROBE}", include_str!("../std/fs.mind"));
        std::fs::write(&src_path, src).expect("write .mind source");
        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
                "--allow-nondeterministic",
            ])
            .status()
            .expect("spawn mindc");
        assert!(status.success(), "{tag}: mindc --emit-shared failed");
        Some(so_path)
    }

    /// Run a Python ctypes script; returns stdout trimmed.
    fn py(script: &str) -> String {
        let out = Command::new("python3")
            .args(["-c", script])
            .output()
            .expect("python3 not found on PATH");
        assert!(
            out.status.success(),
            "python3 script failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    #[test]
    fn mind_open_missing_path_fails_at_every_layer() {
        let Some(so) = compile_fs_with_probe("mind_open_missing_path") else {
            return;
        };
        let script = format!(
            r#"
import ctypes, os, tempfile
lib = ctypes.CDLL('{so}')
lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_open.restype = ctypes.c_int64; lib.__mind_open.argtypes = [ctypes.c_int64]
lib.open_status.restype = ctypes.c_int64; lib.open_status.argtypes = [ctypes.c_int64]
lib.read_file_native.restype = ctypes.c_int64
lib.read_file_native.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

def cstr(s):
    b = s.encode() + b'\x00'
    return (ctypes.c_uint8 * len(b))(*b)

d = tempfile.mkdtemp()
real = os.path.join(d, 'real.bin')
with open(real, 'wb') as f:
    f.write(b'real')
decoy = os.path.join(d, 'decoy.bin')
with open(decoy, 'wb') as f:
    f.write(b'DECOY_CONTENT')
missing = os.path.join(d, 'no_such_file.bin')
assert not os.path.exists(missing)

# 1. The intrinsic itself.
pm = cstr(missing)
fdm = lib.__mind_open(ctypes.addressof(pm))
assert fdm < 0, f'__mind_open(missing) must be negative, got {{fdm}}'
pr = cstr(real)
fdr = lib.__mind_open(ctypes.addressof(pr))
assert fdr >= 0, f'positive control: __mind_open(real) must succeed, got {{fdr}}'
os.close(fdr)

# 2. A MIND program sign-testing the result.
st_m = lib.open_status(ctypes.addressof(pm))
assert st_m == 0, f'MIND open_status(missing) must report failure (0), got {{st_m}}'
st_r = lib.open_status(ctypes.addressof(pr))
assert st_r == 1, f'MIND open_status(real) must report success (1), got {{st_r}}'

# 3. read_file_native with every probe-window descriptor live on a decoy.
dfd = os.open(decoy, os.O_RDONLY)
window = [fd for fd in range(900, 1100) if fd != dfd]
for fd in window:
    os.dup2(dfd, fd)
cap = 4096
buf = lib.__mind_alloc(cap)
assert buf != 0, '__mind_alloc returned NULL'
nm = lib.read_file_native(ctypes.addressof(pm), buf, cap)
for fd in window:
    os.close(fd)
os.close(dfd)
assert nm == -1, f'read_file_native(missing) must return -1, got {{nm}} (read a live wrong fd)'

os.unlink(real); os.unlink(decoy); os.rmdir(d)
print('ok', fdm, st_m, st_r, nm)
"#,
            so = so.to_string_lossy(),
        );
        let result = py(&script);
        assert!(
            result.starts_with("ok"),
            "mind_open_missing_path_fails_at_every_layer: {result}"
        );
    }

    /// The I/O intrinsics' FAILURE values are `-errno`, the raw-syscall contract
    /// the frozen native emitter has always had (`nb_emit_open` / `nb_emit_read` /
    /// `nb_emit_write` store the syscall's `rax`). The C runtime used to return -1
    /// (and 0 for a null buffer or a count <= 0 without any syscall), so the same
    /// program exited differently on the two backends. Each value is pinned exactly,
    /// with a positive control per call so a never-working intrinsic cannot pass.
    #[test]
    fn mind_io_failures_return_negative_errno() {
        let Some(so) = compile_fs_with_probe("mind_io_negative_errno") else {
            return;
        };
        let script = format!(
            r#"
import ctypes, errno, os
lib = ctypes.CDLL('{so}')
for f, n in (('__mind_open', 1), ('__mind_read', 4), ('__mind_write', 4), ('__mind_alloc', 1)):
    fn = getattr(lib, f); fn.restype = ctypes.c_int64; fn.argtypes = [ctypes.c_int64] * n

def cstr(s):
    b = s.encode() + b'\x00'
    return (ctypes.c_uint8 * len(b))(*b)

buf = lib.__mind_alloc(64)
assert buf != 0
got = {{}}
missing = cstr('/nonexistent-mind-dir/no/such/file')
got['open_missing'] = (lib.__mind_open(ctypes.addressof(missing)), -errno.ENOENT)
got['open_null'] = (lib.__mind_open(0), -errno.EFAULT)
got['read_badfd'] = (lib.__mind_read(99, buf, 1, -1), -errno.EBADF)
got['write_badfd'] = (lib.__mind_write(99, buf, 1, -1), -errno.EBADF)
got['write_badfd_empty'] = (lib.__mind_write(99, buf, 0, -1), -errno.EBADF)
got['pread_badfd'] = (lib.__mind_read(99, buf, 1, 0), -errno.EBADF)
zero = cstr('/dev/zero')
fd = lib.__mind_open(ctypes.addressof(zero))
assert fd >= 3, f'positive control: open(/dev/zero) must succeed, got {{fd}}'
got['read_null'] = (lib.__mind_read(fd, 0, 8, -1), -errno.EFAULT)
got['read_ok'] = (lib.__mind_read(fd, buf, 8, -1), 8)
got['read_empty'] = (lib.__mind_read(fd, buf, 0, -1), 0)
os.close(fd)
bad = {{k: v for k, v in got.items() if v[0] != v[1]}}
assert not bad, f'(got, want): {{bad}}'
print('ok', len(got))
"#,
            so = so.to_string_lossy(),
        );
        let result = py(&script);
        assert_eq!(
            result, "ok 9",
            "mind_io_failures_return_negative_errno: {result}"
        );
    }
}
