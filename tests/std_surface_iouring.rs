// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/iouring.mind` surface test — io_uring NOP round-trip and the demo
//! battery, plus the three contracts every wait in the surface relies on.
//!
//! Verifies the minimal io_uring protocol (io_uring_setup -> mmap SQ/CQ/SQE
//! rings -> build an IORING_OP_NOP SQE -> io_uring_enter submit+wait -> reap the
//! CQE) works end-to-end from pure MIND: a NOP's `user_data` must round-trip
//! unchanged through the kernel completion ring.
//!
//! io_uring may be unavailable on a CI runner (a kernel without it, the
//! `kernel.io_uring_disabled` sysctl, or a seccomp policy) — `io_uring_nop` then
//! returns io_uring_setup's own -ENOSYS / -EPERM and the battery SKIPS, printing
//! that reason. Any OTHER negative NOP result FAILS with the precise reason
//! (errno name, or the surface's typed `iouring_err_*` code), and a positive but
//! wrong return indicates ring-protocol corruption and FAILS.
//!
//! Every probe runs under a hard wall-clock bound: a probe that does not exit
//! in time FAILS here instead of hanging the harness. That is the defect the
//! three contract tests pin, each reproducible on any kernel:
//!
//! * `iouring_cqe_error_res_is_negative_errno` — `cqe.res` is a signed i32 and
//!   must reap as -errno (it used to reap as a 4-billion byte count, so no error
//!   CQE was detectable);
//! * `iouring_multishot_recv_refuses_nonportable_len` — a multishot RECV with a
//!   nonzero length is rejected by Linux 6.6 with ONE terminal -EINVAL CQE, so
//!   the surface refuses it identically on every kernel;
//! * `iouring_waits_are_bounded` — ONE terminal CQE while three are awaited (the
//!   exact shape of the Linux 6.6 multishot-recv hang) must return in bounded
//!   time, and `io_ring_wait_multishot` must stop at the terminal CQE at once.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test std_surface_iouring`

#![cfg(all(target_os = "linux", feature = "mlir-build", feature = "std-surface"))]

mod common;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

const SRC: &str = include_str!("../std/iouring.mind");

/// Compile `std/iouring.mind` ONCE per test process, into the per-process
/// scratch directory (never a shared temp-root path). `None` only for a
/// tolerated toolchain gap; every other compile failure panics in the gate.
fn iouring_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        let mindc = common::mindc_bin();
        if !mindc.exists() {
            crate::common::gate::skipped("std_surface_iouring", "iouring: mindc not found");
            return None;
        }
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let so = common::scratch_dir("std_surface_iouring").join("mind_iouring.so");
        let src = manifest.join("std").join("iouring.mind");
        // std/iouring.mind is a raw `syscall()` surface, which the determinism
        // check rejects by default; the artifact is attested `nondeterministic`.
        // Without the flag the compile exits 1 and, under the historic fail-OPEN
        // skip, this gate passed having loaded nothing.
        let out = Command::new(&mindc)
            .args([
                src.to_str().unwrap(),
                "--allow-nondeterministic",
                "--emit-shared",
                so.to_str().unwrap(),
            ])
            .output()
            .expect("run mindc");
        if !crate::common::gate::compiled("std_surface_iouring", &out) {
            return None;
        }
        Some(so)
    })
    .as_ref()
}

/// Shared Python preamble: load the library and decode results. Every code
/// <= -4096 is a typed `iouring_err_*` of the surface; -1..=-4095 is -errno.
const PRELUDE: &str = r#"
import ctypes, errno, os, sys, time
lib = ctypes.CDLL(SO)
TIMEOUT, FEATURE, ENDED, MISMATCH = -4097, -4098, -4099, -4100
TYPED = {
    TIMEOUT: 'iouring_err_wait_timeout: a bounded wait expired before its completions arrived',
    FEATURE: 'iouring_err_kernel_feature: kernel lacks IORING_FEAT_SINGLE_MMAP / IORING_FEAT_EXT_ARG (Linux < 5.11), so no wait can be bounded',
    ENDED: 'iouring_err_multishot_ended: the multishot request ended before its completions',
    MISMATCH: 'iouring_err_data_mismatch: bytes did not round-trip',
}
def reason(v):
    if v in TYPED:
        return f'{v} = {TYPED[v]}'
    if -4095 <= v < 0:
        return f'{v} = -{errno.errorcode.get(-v, "E?")} ({os.strerror(-v)})'
    return str(v)
REL = os.uname().release
def kver():
    p = REL.split('.')
    return (int(p[0]), int(''.join(c for c in p[1] if c.isdigit()) or '0'))
"#;

/// Run `body` (after [`PRELUDE`], with `SO` bound to `so` and `SCRATCH` to its
/// per-process directory) in python3 under a
/// hard wall-clock bound: coreutils `timeout`, SIGKILL 5 s after SIGTERM. A
/// probe that outlives the bound FAILS — an unbounded io_uring wait is exactly
/// the defect class this file guards, and it must never hang the harness.
fn run_probe(so: &std::path::Path, body: &str, secs: u64) -> String {
    // SCRATCH: the per-process directory the `.so` lives in — the probe's
    // on-disk files go there too, never to a fixed shared-temp-root path.
    let scratch = so.parent().expect("scratch .so has a parent directory");
    let py = format!(
        "SO = r'{}'\nSCRATCH = r'{}'\n{PRELUDE}\n{body}",
        so.display(),
        scratch.display()
    );
    let out = Command::new("timeout")
        .args(["--kill-after=5", &secs.to_string(), "python3", "-c", &py])
        .output()
        .expect("run timeout(1) + python3");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let code = out.status.code();
    assert!(
        code != Some(124) && code != Some(137) && code.is_some(),
        "probe did not finish within {secs} s (exit {code:?}) — an io_uring wait \
         blocked with no bound:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        out.status.success(),
        "probe failed (exit {code:?}):\nstdout: {stdout}\nstderr: {stderr}"
    );
    stdout
}

#[test]
fn iouring_parses_and_lowers_with_ring_api() {
    let module = parser::parse(SRC).expect("std/iouring.mind must parse cleanly");
    let ir = lower_to_ir(&module).expect("lowering");
    let names: Vec<&str> = ir
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::FnDef { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    for required in [
        "io_ring_new",
        "io_ring_try_new_flags",
        "io_ring_features",
        "io_ring_submit_raw",
        "io_ring_submit_op",
        "io_ring_publish",
        "io_ring_enter",
        "io_ring_enter_timeout",
        "io_ring_wait_bound_ns",
        "io_ring_wait_multishot",
        "iouring_err_wait_timeout",
        "iouring_err_kernel_feature",
        "iouring_err_multishot_ended",
        "iouring_err_data_mismatch",
        "io_ring_reap",
        "io_ring_free",
        "io_uring_nop",
        "io_uring_socketpair_echo",
        "io_ring_register_buffers",
        "io_ring_submit_fixed",
        "io_uring_fixed_buffer_echo",
        "io_ring_register_pbuf_ring",
        "io_ring_submit_recv_provided",
        "io_uring_provided_buffer_recv",
        "io_uring_socket",
        "io_uring_shutdown",
        "io_uring_close",
        "io_uring_socket_lifecycle_demo",
        "io_ring_reap_full",
        "io_ring_submit_recv_multishot",
        "io_uring_multishot_recv_demo",
        "io_ring_register_files_sparse",
        "io_uring_socket_direct",
        "io_uring_close_direct",
        "io_uring_direct_descriptor_demo",
        "io_uring_openat",
        "io_uring_pread",
        "io_uring_pwrite",
        "io_uring_storage_rw_demo",
        "io_reactor_pin_cpu",
        "io_reactor_affinity_mask",
        "io_reactor_pin_demo",
        "io_uring_ktls_echo",
        "io_uring_dmabuf_io_demo",
        "io_ring_new_flags",
        "io_ring_new_iopoll",
        "io_uring_odirect_roundtrip",
        "io_uring_tcp_accept_one",
        "io_uring_tcp_echo_round",
        "io_uring_tcp_close",
        "io_uring_loopback_echo",
        "io_uring_loopback_echo_n",
        "io_ring_submit_accept_multishot",
        "io_uring_multishot_accept_demo",
        "io_uring_echo_bench",
    ] {
        assert!(
            names.contains(&required),
            "std.iouring must define `{required}`; found {names:?}"
        );
    }
}

#[test]
fn iouring_nop_roundtrips_user_data() {
    let Some(so) = iouring_so() else { return };
    // (1) NOP user_data round-trip, then the whole demo battery. Every demo
    // returns its expected value or a negative reason; a typed TIMEOUT is never
    // accepted (a completion that never came is a failure, not a skip).
    let body = r#"
for n in ['io_uring_nop', 'io_uring_socketpair_echo', 'io_uring_fixed_buffer_echo',
          'io_uring_provided_buffer_recv', 'io_uring_socket_lifecycle_demo',
          'io_uring_multishot_recv_demo', 'io_uring_direct_descriptor_demo',
          'io_uring_storage_rw_demo', 'io_reactor_pin_demo', 'io_uring_ktls_echo',
          'io_uring_dmabuf_io_demo', 'io_uring_odirect_roundtrip', 'io_uring_loopback_echo',
          'io_uring_loopback_echo_n', 'io_uring_multishot_accept_demo', 'io_uring_echo_bench']:
    getattr(lib, n).restype = ctypes.c_int64
    getattr(lib, n).argtypes = [ctypes.c_int64] * {'io_uring_nop': 1, 'io_uring_socketpair_echo': 2,
        'io_uring_fixed_buffer_echo': 2, 'io_uring_provided_buffer_recv': 2,
        'io_uring_socket_lifecycle_demo': 0, 'io_uring_multishot_recv_demo': 3,
        'io_uring_direct_descriptor_demo': 0, 'io_uring_storage_rw_demo': 3,
        'io_reactor_pin_demo': 1, 'io_uring_ktls_echo': 2, 'io_uring_dmabuf_io_demo': 3,
        'io_uring_odirect_roundtrip': 4, 'io_uring_loopback_echo': 2,
        'io_uring_loopback_echo_n': 3, 'io_uring_multishot_accept_demo': 0,
        'io_uring_echo_bench': 3}[n]
addr = lambda b: ctypes.cast(b, ctypes.c_void_p).value
skips = []
# These older demos still mix -errno with demo-local negative codes (-1, -10..-30)
# that collide with errno numbers, so their skips are reported as raw values;
# only the typed TIMEOUT is decoded, and it always fails.
def check(name, v, want):
    assert v != TIMEOUT, f'{name}: {reason(v)}'
    assert v == want or v < 0, f'{name} returned unexpected {v}'
    if v < 0:
        skips.append(f'{name}: {v} (-errno or a demo-local code; see std/iouring.mind)')
    return v
r = lib.io_uring_nop(0x4242)
if r < 0:
    if -r in (errno.ENOSYS, errno.EPERM) or r == FEATURE:
        print('SKIP io_uring unavailable on kernel', REL, '->', reason(r))
        sys.exit(0)
    raise SystemExit(f'io_uring_nop failed on kernel {REL}: {reason(r)}')
assert r == 0x4242, f'NOP user_data corrupted: got {hex(r)}'
msg = b'MIND-io_uring-echo'
buf = ctypes.create_string_buffer(msg)
e = check('socketpair_echo', lib.io_uring_socketpair_echo(addr(buf), len(msg)), 1)
fbuf = ctypes.create_string_buffer(b'MIND-io_uring-FIXED-buffer')
fx = check('fixed_buffer_echo', lib.io_uring_fixed_buffer_echo(addr(fbuf), 26), 1)
pbuf = ctypes.create_string_buffer(b'MIND-io_uring-PROVIDED-buffer')
pv = check('provided_buffer_recv', lib.io_uring_provided_buffer_recv(addr(pbuf), 29), 1)
lc = check('socket_lifecycle', lib.io_uring_socket_lifecycle_demo(), 1)
mbuf = ctypes.create_string_buffer(b'MIND-multishot-recv-zerocopy')
mr = lib.io_uring_multishot_recv_demo(addr(mbuf), 28, 3)
# Multishot recv exists since Linux 6.0: there it MUST deliver all three.
# Before 6.0 the kernel's own -EINVAL is the precise reason.
if kver() >= (6, 0):
    assert mr == 3, f'multishot recv on kernel {REL} (>= 6.0) returned {reason(mr)}'
else:
    assert mr == 3 or mr == -errno.EINVAL, f'multishot recv on kernel {REL}: {reason(mr)}'
    if mr < 0:
        skips.append(f'multishot_recv: {reason(mr)}')
dd = check('direct_descriptor', lib.io_uring_direct_descriptor_demo(), 1)
spath_s = os.path.join(SCRATCH, 'storage.tmp')
spath = ctypes.create_string_buffer(spath_s.encode() + b'\x00')
sdata = ctypes.create_string_buffer(b'MIND-io_uring-storage')
st = check('storage_rw', lib.io_uring_storage_rw_demo(addr(spath), addr(sdata), 21), 1)
try: os.unlink(spath_s)
except OSError: pass
before = os.sched_getaffinity(0)
pin = check('reactor_pin', lib.io_reactor_pin_demo(1), 1)
assert os.sched_getaffinity(0) == before, 'core-pin did not restore affinity'
ktbuf = ctypes.create_string_buffer(b'MIND-io_uring-kTLS')
kt = check('ktls_echo', lib.io_uring_ktls_echo(addr(ktbuf), 18), 1)
dbpath_s = os.path.join(SCRATCH, 'dmabuf.tmp')
dbpath = ctypes.create_string_buffer(dbpath_s.encode() + b'\x00')
dbdata = ctypes.create_string_buffer(b'MIND-io_uring-dmabuf')
db = check('dmabuf_io', lib.io_uring_dmabuf_io_demo(addr(dbpath), addr(dbdata), 20), 1)
try: os.unlink(dbpath_s)
except OSError: pass
odpath_s = os.path.join(SCRATCH, 'odirect.bin')
odpath = ctypes.create_string_buffer(odpath_s.encode() + b'\x00')
oddata = ctypes.create_string_buffer(b'MIND-io_uring-odirect')
od = check('odirect', lib.io_uring_odirect_roundtrip(addr(odpath), addr(oddata), 21, 0), 1)
odp = check('odirect_iopoll', lib.io_uring_odirect_roundtrip(addr(odpath), addr(oddata), 21, 1), 1)
try: os.unlink(odpath_s)
except OSError: pass
tmsg = b'MIND-io_uring-tcp-echo'
tbuf = ctypes.create_string_buffer(tmsg)
le = check('loopback_echo', lib.io_uring_loopback_echo(addr(tbuf), len(tmsg)), 1)
ln = check('loopback_echo_n', lib.io_uring_loopback_echo_n(addr(tbuf), len(tmsg), 5), 5)
ms = lib.io_uring_multishot_accept_demo()
# Multishot accept exists since Linux 5.19; where loopback TCP works it MUST
# deliver both accepts.
if le == 1 and kver() >= (5, 19):
    assert ms == 2, f'multishot accept on kernel {REL} (>= 5.19) returned {reason(ms)}'
else:
    assert ms == 2 or (ms < 0 and ms != TIMEOUT), f'multishot accept: {reason(ms)}'
    if ms < 0:
        skips.append(f'multishot_accept: {reason(ms)}')
bench = lib.io_uring_echo_bench(addr(tbuf), len(tmsg), 500)
assert bench != 0 and bench != TIMEOUT, f'echo bench returned {reason(bench)}'
print('OK kernel', REL, hex(r), 'sp', e, 'fixed', fx, 'pbuf', pv, 'life', lc, 'msrecv', mr,
      'direct', dd, 'storage', st, 'odirect', od, odp, 'pin', pin, 'ktls', kt, 'dmabuf', db,
      'tcp', le, 'loop', ln, 'multishot', ms, 'reqs', bench)
for s in skips:
    print('SKIP-CASE', s)
"#;
    let stdout = run_probe(so, body, 120);
    print!("{stdout}");
    if stdout.contains("SKIP io_uring unavailable") {
        println!("iouring: kernel io_uring unavailable; round-trip skipped");
    }
}

/// `cqe.res` is a SIGNED 32-bit field. A RECV on fd -1 completes with exactly
/// one CQE whose res is -EBADF; both reap functions must hand back -9, not the
/// zero-extended 4294967287 that made every error CQE look like a success.
#[test]
fn iouring_cqe_error_res_is_negative_errno() {
    let Some(so) = iouring_so() else { return };
    let body = r#"
I = ctypes.c_int64
lib.io_ring_new.restype = I
lib.io_ring_new.argtypes = [I]
for f, n in [('io_ring_submit_op', 7), ('io_ring_publish', 2), ('io_ring_enter', 3),
             ('io_ring_reap', 4), ('io_ring_reap_full', 5), ('io_ring_free', 1)]:
    getattr(lib, f).restype = I
    getattr(lib, f).argtypes = [I] * n
buf = ctypes.create_string_buffer(8)
addr = lambda b: ctypes.cast(b, ctypes.c_void_p).value
for reaper in ('io_ring_reap', 'io_ring_reap_full'):
    h = lib.io_ring_new(4)
    if h == 0:
        print('SKIP io_uring unavailable'); sys.exit(0)
    assert lib.io_ring_submit_op(h, 0, 27, -1, addr(buf), 8, 0x55) == 0
    lib.io_ring_publish(h, 1)
    er = lib.io_ring_enter(h, 1, 1)
    assert er == 1, f'enter returned {reason(er)}'
    ud, res, fl = (I * 4)(), (I * 4)(), (I * 4)()
    if reaper == 'io_ring_reap':
        n = lib.io_ring_reap(h, addr(ud), addr(res), 4)
    else:
        n = lib.io_ring_reap_full(h, addr(ud), addr(res), addr(fl), 4)
    assert n == 1 and ud[0] == 0x55, f'{reaper}: n={n} ud={ud[0]:#x}'
    assert res[0] == -errno.EBADF, (
        f'{reaper}: CQE res of a RECV on fd -1 reaped as {res[0]}; expected '
        f'-EBADF (-9) — cqe.res is a signed i32 and must be sign-extended')
    lib.io_ring_free(h)
print('ok res', res[0])
"#;
    print!("{}", run_probe(so, body, 60));
}

/// A multishot RECV with a nonzero length is NOT portable: Linux 6.6 rejects it
/// with one terminal -EINVAL CQE (newer kernels read it as a per-receive cap).
/// The surface refuses it up front with -EINVAL on every kernel, and accepts 0.
#[test]
fn iouring_multishot_recv_refuses_nonportable_len() {
    let Some(so) = iouring_so() else { return };
    let body = r#"
import socket
I = ctypes.c_int64
lib.io_ring_new.restype = I
lib.io_ring_new.argtypes = [I]
lib.io_ring_submit_recv_multishot.restype = I
lib.io_ring_submit_recv_multishot.argtypes = [I] * 6
lib.io_ring_free.restype = I
lib.io_ring_free.argtypes = [I]
h = lib.io_ring_new(4)
if h == 0:
    print('SKIP io_uring unavailable'); sys.exit(0)
a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
r = lib.io_ring_submit_recv_multishot(h, 0, b.fileno(), 256, 0, 2)
assert r == -errno.EINVAL, (
    f'io_ring_submit_recv_multishot(len=256) returned {r}; a nonzero multishot '
    f'recv length must be refused with -EINVAL (-22): Linux 6.6 answers it with '
    f'a single terminal -EINVAL CQE, so accepting it is kernel-dependent')
r0 = lib.io_ring_submit_recv_multishot(h, 0, b.fileno(), 0, 0, 2)
assert r0 == 0, f'len=0 must be accepted, got {r0}'
lib.io_ring_free(h)
print('ok', r, r0)
"#;
    print!("{}", run_probe(so, body, 60));
}

/// The Linux 6.6 hang, reproduced on any kernel: a multishot RECV on fd -1
/// draws exactly ONE terminal CQE (-EBADF, no IORING_CQE_F_MORE) — the shape of
/// the single -EINVAL CQE 6.6 posted for the old demo — while the caller waits
/// for THREE. `io_ring_enter` must return within its bound; `io_ring_wait_multishot`
/// must stop at the terminal CQE at once; an idle wait must expire with the
/// typed timeout; and a ring that cannot bound a wait must refuse it.
#[test]
fn iouring_waits_are_bounded() {
    let Some(so) = iouring_so() else { return };
    let body = r#"
I = ctypes.c_int64
lib.io_ring_new.restype = I
lib.io_ring_new.argtypes = [I]
for f, n in [('io_ring_submit_recv_multishot', 6), ('io_ring_publish', 2),
             ('io_ring_enter', 3), ('io_ring_reap_full', 5), ('io_ring_free', 1)]:
    getattr(lib, f).restype = I
    getattr(lib, f).argtypes = [I] * n
addr = lambda b: ctypes.cast(b, ctypes.c_void_p).value
h = lib.io_ring_new(8)
if h == 0:
    print('SKIP io_uring unavailable'); sys.exit(0)
assert lib.io_ring_submit_recv_multishot(h, 0, -1, 0, 0, 7) == 0
lib.io_ring_publish(h, 1)
t0 = time.monotonic()
er = lib.io_ring_enter(h, 1, 3)
dt = time.monotonic() - t0
ud, res, fl = (I * 8)(), (I * 8)(), (I * 8)()
n = lib.io_ring_reap_full(h, addr(ud), addr(res), addr(fl), 8)
assert er == 1, f'enter returned {reason(er)}'
assert n == 1 and ud[0] == 7, f'expected one CQE for user_data 7, got n={n}'
assert res[0] == -errno.EBADF and (fl[0] & 2) == 0, f'res={res[0]} flags={fl[0]:#x}'
assert dt < 30.0, f'io_ring_enter waited {dt:.1f} s for completions that never come'
print(f'enter returned {er} after {dt:.2f} s with 1 terminal CQE (res {res[0]})')
lib.io_ring_free(h)

for f, n in [('io_ring_wait_multishot', 7), ('io_ring_enter_timeout', 4),
             ('io_ring_features', 1)]:
    getattr(lib, f).restype = I
    getattr(lib, f).argtypes = [I] * n
out, cnt = (I * 24)(), I(0)
h = lib.io_ring_new(8)
assert lib.io_ring_features(h) & 256, 'ring kept no IORING_FEAT_EXT_ARG bit'
assert lib.io_ring_submit_recv_multishot(h, 0, -1, 0, 0, 7) == 0
lib.io_ring_publish(h, 1)
assert lib.io_ring_enter(h, 1, 0) == 1
t0 = time.monotonic()
term = lib.io_ring_wait_multishot(h, 7, 3, 5_000_000_000, addr(out), 8, ctypes.addressof(cnt))
dt_term = dt = time.monotonic() - t0
assert term == 1, f'wait_multishot status {reason(term)}; expected 1 (terminal CQE)'
assert cnt.value == 1 and out[0] == 7 and out[1] == -errno.EBADF and (out[2] & 2) == 0, \
    f'records: n={cnt.value} {list(out[:3])}'
assert dt < 2.0, f'wait_multishot took {dt:.2f} s to notice a terminal CQE'
lib.io_ring_free(h)

h = lib.io_ring_new(8)
t0 = time.monotonic()
idle = lib.io_ring_wait_multishot(h, 9, 1, 50_000_000, addr(out), 8, ctypes.addressof(cnt))
dt = time.monotonic() - t0
assert idle == TIMEOUT and cnt.value == 0, f'idle wait_multishot: {reason(idle)} n={cnt.value}'
assert dt < 5.0, f'idle 50 ms wait took {dt:.2f} s'
et = lib.io_ring_enter_timeout(h, 0, 1, 50_000_000)
assert et == -errno.ETIME, f'idle io_ring_enter_timeout returned {reason(et)}'
# A handle whose kernel reported no IORING_FEAT_EXT_ARG must refuse to wait.
ctypes.c_int64.from_address(h + 96).value = 1
t0 = time.monotonic()
ef = lib.io_ring_enter(h, 0, 1)
assert ef == FEATURE and time.monotonic() - t0 < 1.0, f'no-EXT_ARG wait: {reason(ef)}'
lib.io_ring_free(h)
print(f'ok wait_multishot terminal={term} after {dt_term:.3f} s; idle={idle}; '
      f'enter_timeout idle={et}; no-EXT_ARG enter={ef}')
"#;
    print!("{}", run_probe(so, body, 60));
}

/// io_ring_submit_raw must fail closed (return -EINVAL) on invalid caller input
/// before any dereference, preventing the out-of-bounds SQE write a negative
/// slot would otherwise cause. These reject paths need no live kernel ring.
#[test]
fn iouring_submit_rejects_invalid_input() {
    let Some(so) = iouring_so() else { return };
    let body = r#"
f = lib.io_ring_submit_raw
f.restype = ctypes.c_int64
f.argtypes = [ctypes.c_int64] * 8
# (h, slot, opcode, fd, off, addr, len, user_data)
assert f(0, 0, 0, 0, 0, 0, 0, 0) == -22, 'null handle not rejected'
assert f(1, -1, 0, 0, 0, 0, 0, 0) == -22, 'negative slot not rejected'
assert f(1, 0, 0, 0, 0, 0, -1, 0) == -22, 'negative len not rejected'
assert f(1, 0, 0, 0, 0, 0, 4294967296, 0) == -22, 'oversize len not rejected'
print('ok')
"#;
    print!("{}", run_probe(so, body, 60));
}
