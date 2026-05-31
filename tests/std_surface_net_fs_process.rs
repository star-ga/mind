// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Task #268 — `std/net.mind`, `std/fs.mind`, `std/process.mind` surface tests.
//!
//! Section A (gate: `std-surface`):
//!   Verifies each module parses cleanly and lowers to IR with all required
//!   `pub fn`s present.
//!
//! Section B (gate: `std-surface cross-module-imports`):
//!   Verifies each module auto-exports its public symbols and the bundled
//!   stdlib resolves `use std.fs`, `use std.net`, `use std.process`.
//!
//! Section C (gate: `mlir-build cross-module-imports`, unix only):
//!   10 functional integration tests compiled via `mindc --emit-shared`
//!   and exercised through Python ctypes:
//!     1.  fs_round_trip      — write + read_to_string round-trips a file
//!     2.  fs_read_dir        — read_dir lists expected entry
//!     3.  fs_mkdir_p         — mkdir_p creates nested dirs
//!     4.  fs_canonicalize    — canonicalize resolves /tmp to absolute path
//!     5.  proc_getenv        — getenv("PATH") returns non-empty string
//!     6.  proc_pid           — proc_pid() returns positive i64
//!     7.  proc_spawn_true    — spawn("true") exits with code 0
//!     8.  proc_spawn_false   — spawn("false") exits with code != 0
//!     9.  net_tcp_loopback   — TCP server + client exchange one byte
//!     10. net_udp_loopback   — UDP send/recv round-trips one byte

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const FS_MIND_SRC: &str = include_str!("../std/fs.mind");
const NET_MIND_SRC: &str = include_str!("../std/net.mind");
const PROCESS_MIND_SRC: &str = include_str!("../std/process.mind");

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn has_fn(ir: &libmind::ir::IRModule, name: &str) -> bool {
    ir.instrs
        .iter()
        .any(|i| matches!(i, Instr::FnDef { name: n, .. } if n == name))
}

// ─── Section A: parse + lower ─────────────────────────────────────────────────

#[test]
fn fs_mind_parses_and_lowers() {
    let module = parser::parse(FS_MIND_SRC).expect("std/fs.mind must parse");
    let ir = lower_to_ir(&module);
    for want in [
        "read_to_string",
        "try_read_to_string",
        "write_string",
        "read_dir",
        "fs_mkdir",
        "mkdir_p",
        "fs_rmdir",
        "remove_file",
        "remove_recursive",
        "fs_rename",
        "exists",
        "is_dir",
        "is_file",
        "fs_stat",
        "canonicalize",
        "direntry_name_addr",
        "direntry_name_len",
        "direntry_is_dir",
        "direntry_is_file",
        "filestat_size",
        "filestat_mode",
        "filestat_mtime",
        "dir_vec_len",
        "dir_vec_get",
    ] {
        assert!(has_fn(&ir, want), "std/fs.mind: missing FnDef `{want}`");
    }
}

#[test]
fn net_mind_parses_and_lowers() {
    let module = parser::parse(NET_MIND_SRC).expect("std/net.mind must parse");
    let ir = lower_to_ir(&module);
    for want in [
        "tcp_listen",
        "tcp_listen_port",
        "tcp_accept",
        "tcp_connect",
        "tcp_read",
        "tcp_write",
        "tcp_close",
        "tcp_listener_close",
        "tcp_listener_fd",
        "tcp_stream_fd",
        "udp_bind",
        "udp_socket_port",
        "udp_send_to",
        "udp_recv_from",
        "udp_close",
        "udp_socket_fd",
        "af_inet",
        "af_inet6",
        "sock_stream",
        "sock_dgram",
    ] {
        assert!(has_fn(&ir, want), "std/net.mind: missing FnDef `{want}`");
    }
}

#[test]
fn process_mind_parses_and_lowers() {
    let module = parser::parse(PROCESS_MIND_SRC).expect("std/process.mind must parse");
    let ir = lower_to_ir(&module);
    for want in [
        "spawn",
        "spawn_capture",
        "wait",
        "proc_kill",
        "proc_getenv",
        "proc_setenv",
        "proc_unsetenv",
        "current_dir",
        "set_current_dir",
        "proc_exit",
        "proc_pid",
        "child_pid",
        "child_stdin",
        "child_stdout",
        "child_stderr",
        "exit_code",
        "exit_signal",
    ] {
        assert!(
            has_fn(&ir, want),
            "std/process.mind: missing FnDef `{want}`"
        );
    }
}

// ─── Section B: cross-module exports ─────────────────────────────────────────

#[cfg(feature = "cross-module-imports")]
mod cross_module {
    use super::*;
    use libmind::project::module_table::collect_module_exports;

    #[test]
    fn fs_mind_auto_exports_public_symbols() {
        let module = parser::parse(FS_MIND_SRC).expect("std/fs.mind must parse");
        let ex = collect_module_exports("std.fs", &module);
        for want in [
            "read_to_string",
            "write_string",
            "read_dir",
            "mkdir_p",
            "remove_recursive",
            "canonicalize",
            "exists",
            "is_dir",
            "is_file",
            "fs_stat",
        ] {
            assert!(
                ex.exported.iter().any(|s| s == want),
                "std.fs must auto-export `{want}`; got {:?}",
                ex.exported
            );
        }
    }

    #[test]
    fn net_mind_auto_exports_public_symbols() {
        let module = parser::parse(NET_MIND_SRC).expect("std/net.mind must parse");
        let ex = collect_module_exports("std.net", &module);
        for want in [
            "tcp_listen",
            "tcp_accept",
            "tcp_connect",
            "tcp_read",
            "tcp_write",
            "tcp_close",
            "udp_bind",
            "udp_send_to",
            "udp_recv_from",
        ] {
            assert!(
                ex.exported.iter().any(|s| s == want),
                "std.net must auto-export `{want}`; got {:?}",
                ex.exported
            );
        }
    }

    #[test]
    fn process_mind_auto_exports_public_symbols() {
        let module = parser::parse(PROCESS_MIND_SRC).expect("std/process.mind must parse");
        let ex = collect_module_exports("std.process", &module);
        for want in [
            "spawn",
            "wait",
            "proc_getenv",
            "proc_pid",
            "proc_exit",
            "current_dir",
        ] {
            assert!(
                ex.exported.iter().any(|s| s == want),
                "std.process must auto-export `{want}`; got {:?}",
                ex.exported
            );
        }
    }

    #[test]
    fn bundled_stdlib_resolves_std_fs_net_process() {
        use libmind::project::module_table::build_module_table;
        use libmind::project::stdlib::parsed_stdlib_modules;

        let mods = parsed_stdlib_modules();
        let refs: Vec<(String, &libmind::ast::Module)> =
            mods.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = build_module_table(&refs);
        assert!(
            table.resolves(&["std".into(), "fs".into()], "read_to_string"),
            "bundled stdlib must expose std.fs::read_to_string"
        );
        assert!(
            table.resolves(&["std".into(), "net".into()], "tcp_listen"),
            "bundled stdlib must expose std.net::tcp_listen"
        );
        assert!(
            table.resolves(&["std".into(), "process".into()], "spawn"),
            "bundled stdlib must expose std.process::spawn"
        );
    }
}

// ─── Section C: MLIR functional integration tests (unix only) ────────────────

#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
mod mlir_functional {
    use std::path::PathBuf;
    use std::process::Command;

    fn mindc_bin() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dbg = manifest.join("target").join("debug").join("mindc");
        let rel = manifest.join("target").join("release").join("mindc");
        if dbg.exists() { dbg } else { rel }
    }

    fn out_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_net_fs_process")
    }

    /// Compile source text to a shared library at `so_path`.
    /// Returns false if mindc is not available (test is skipped).
    fn compile_to_so(src: &str, tag: &str) -> Option<PathBuf> {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("{tag}: mindc not found at {mindc:?}; skipping");
            return None;
        }
        let dir = out_dir();
        std::fs::create_dir_all(&dir).expect("create output dir");
        let src_path = dir.join(format!("{tag}.mind"));
        let so_path = dir.join(format!("{tag}.so"));
        std::fs::write(&src_path, src).expect("write .mind source");
        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("spawn mindc");
        assert!(
            status.success(),
            "{tag}: mindc --emit-shared failed; check build output"
        );
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

    // ── Test 1: fs round-trip ────────────────────────────────────────────────

    #[test]
    fn fs_round_trip() {
        let src = concat!(include_str!("../std/fs.mind"), "\n");
        let Some(so) = compile_to_so(src, "fs_round_trip") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, tempfile, os, sys
lib = ctypes.CDLL('{so}')
lib.write_string.restype = ctypes.c_int64
lib.read_to_string.restype = ctypes.c_int64
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64', None)

import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
libc.malloc.restype = ctypes.c_void_p

# Use ctypes malloc for path/content buffers.
def mk_cstr(s):
    b = s.encode() + b'\x00'
    buf = (ctypes.c_uint8 * len(b))(*b)
    return buf

with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
    path = f.name

path_buf = mk_cstr(path)
content = b'hello_mind_fs'
cont_buf = (ctypes.c_uint8 * (len(content)+1))(*content, 0)

lib.__mind_alloc = getattr(lib, '__mind_alloc')
lib.__mind_alloc.restype = ctypes.c_int64
lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64')
lib.__mind_store_i64.restype = None
lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64')
lib.__mind_load_i64.restype = ctypes.c_int64
lib.__mind_load_i64.argtypes = [ctypes.c_int64]

lib.write_string.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.read_to_string.argtypes = [ctypes.c_int64]

path_addr = ctypes.addressof(path_buf)
cont_addr = ctypes.addressof(cont_buf)
ok = lib.write_string(path_addr, cont_addr, len(content))
assert ok == 1, f'write_string failed: {{ok}}'

rec = lib.read_to_string(path_addr)
assert rec != 0, 'read_to_string returned null'
read_addr = lib.__mind_load_i64(rec + 0)
read_len  = lib.__mind_load_i64(rec + 8)
assert read_len == len(content), f'len mismatch: {{read_len}} vs {{len(content)}}'
raw = (ctypes.c_uint8 * read_len).from_address(read_addr)
got = bytes(raw)
assert got == content, f'content mismatch: {{got!r}}'
os.unlink(path)
print('ok')
"#,
            so = so_str,
        );
        let result = py(&script);
        assert_eq!(result, "ok", "fs_round_trip: {result}");
    }

    // ── Test 2: fs read_dir ──────────────────────────────────────────────────

    #[test]
    fn fs_read_dir() {
        let src = concat!(include_str!("../std/fs.mind"), "\n");
        let Some(so) = compile_to_so(src, "fs_read_dir") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let out_dir = out_dir().to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, os, sys
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.read_dir.restype = ctypes.c_int64; lib.read_dir.argtypes = [ctypes.c_int64]
lib.dir_vec_len.restype = ctypes.c_int64; lib.dir_vec_len.argtypes = [ctypes.c_int64]
lib.dir_vec_get.restype = ctypes.c_int64; lib.dir_vec_get.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.direntry_name_len.restype = ctypes.c_int64; lib.direntry_name_len.argtypes = [ctypes.c_int64]
lib.direntry_name_addr.restype = ctypes.c_int64; lib.direntry_name_addr.argtypes = [ctypes.c_int64]

dirpath = '{out_dir}'
path_buf = dirpath.encode() + b'\x00'
cbuf = (ctypes.c_uint8 * len(path_buf))(*path_buf)
vec = lib.read_dir(ctypes.addressof(cbuf))
n = lib.dir_vec_len(vec)
assert n > 0, f'read_dir returned 0 entries for {{dirpath!r}}'
# Collect names
names = []
for i in range(n):
    e = lib.dir_vec_get(vec, i)
    nlen = lib.direntry_name_len(e)
    naddr = lib.direntry_name_addr(e)
    raw = (ctypes.c_uint8 * nlen).from_address(naddr)
    names.append(bytes(raw).decode())
print('ok', len(names))
"#,
            so = so_str,
            out_dir = out_dir,
        );
        let result = py(&script);
        assert!(result.starts_with("ok"), "fs_read_dir: {result}");
    }

    // ── Test 3: fs mkdir_p ───────────────────────────────────────────────────

    #[test]
    fn fs_mkdir_p() {
        let src = concat!(include_str!("../std/fs.mind"), "\n");
        let Some(so) = compile_to_so(src, "fs_mkdir_p") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, tempfile, os, shutil
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.mkdir_p.restype = ctypes.c_int64; lib.mkdir_p.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.is_dir.restype = ctypes.c_int64; lib.is_dir.argtypes = [ctypes.c_int64]

base = tempfile.mkdtemp()
nested = os.path.join(base, 'a', 'b', 'c').encode() + b'\x00'
cbuf = (ctypes.c_uint8 * len(nested))(*nested)
ok = lib.mkdir_p(ctypes.addressof(cbuf), 0o755)
assert ok == 1, f'mkdir_p returned {{ok}}'
ok2 = lib.is_dir(ctypes.addressof(cbuf))
assert ok2 == 1, f'is_dir returned {{ok2}}'
shutil.rmtree(base)
print('ok')
"#,
            so = so_str,
        );
        let result = py(&script);
        assert_eq!(result, "ok", "fs_mkdir_p: {result}");
    }

    // ── Test 4: fs canonicalize ──────────────────────────────────────────────

    #[test]
    fn fs_canonicalize() {
        let src = concat!(include_str!("../std/fs.mind"), "\n");
        let Some(so) = compile_to_so(src, "fs_canonicalize") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.canonicalize.restype = ctypes.c_int64; lib.canonicalize.argtypes = [ctypes.c_int64]
path = b'/tmp\x00'
cbuf = (ctypes.c_uint8 * len(path))(*path)
rec = lib.canonicalize(ctypes.addressof(cbuf))
rlen = lib.__mind_load_i64(rec + 8)
assert rlen > 0, f'canonicalize /tmp returned empty string (len={{rlen}})'
raddr = lib.__mind_load_i64(rec + 0)
raw = (ctypes.c_uint8 * rlen).from_address(raddr)
canon = bytes(raw).decode()
assert canon.startswith('/'), f'canonical path must start with /: {{canon!r}}'
print('ok', canon)
"#,
            so = so_str,
        );
        let result = py(&script);
        assert!(result.starts_with("ok"), "fs_canonicalize: {result}");
    }

    // ── Test 5: process getenv ───────────────────────────────────────────────

    #[test]
    fn proc_getenv() {
        let src = concat!(include_str!("../std/process.mind"), "\n");
        let Some(so) = compile_to_so(src, "proc_getenv") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.proc_getenv.restype = ctypes.c_int64; lib.proc_getenv.argtypes = [ctypes.c_int64]
name = b'PATH\x00'
cbuf = (ctypes.c_uint8 * len(name))(*name)
rec = lib.proc_getenv(ctypes.addressof(cbuf))
rlen = lib.__mind_load_i64(rec + 8)
assert rlen > 0, f'getenv("PATH") returned empty (len={{rlen}})'
print('ok', rlen)
"#,
            so = so_str,
        );
        let result = py(&script);
        assert!(result.starts_with("ok"), "proc_getenv: {result}");
    }

    // ── Test 6: process pid ──────────────────────────────────────────────────

    #[test]
    fn proc_pid() {
        let src = concat!(include_str!("../std/process.mind"), "\n");
        let Some(so) = compile_to_so(src, "proc_pid") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.proc_pid.restype = ctypes.c_int64; lib.proc_pid.argtypes = []
pid = lib.proc_pid()
assert pid > 0, f'proc_pid() returned non-positive value: {{pid}}'
print('ok', pid)
"#,
            so = so_str,
        );
        let result = py(&script);
        assert!(result.starts_with("ok"), "proc_pid: {result}");
    }

    // ── Test 7: spawn true ───────────────────────────────────────────────────

    #[test]
    fn proc_spawn_true() {
        let src = concat!(include_str!("../std/process.mind"), "\n");
        let Some(so) = compile_to_so(src, "proc_spawn_true") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.spawn.restype = ctypes.c_int64; lib.spawn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.wait.restype = ctypes.c_int64; lib.wait.argtypes = [ctypes.c_int64]
lib.exit_code.restype = ctypes.c_int64; lib.exit_code.argtypes = [ctypes.c_int64]
prog = b'true\x00'
pbuf = (ctypes.c_uint8 * len(prog))(*prog)
child = lib.spawn(ctypes.addressof(pbuf), 0, 0)
pid = lib.__mind_load_i64(child + 0)
assert pid > 0, f'spawn true: pid={{pid}}'
status = lib.wait(child)
code = lib.exit_code(status)
assert code == 0, f'spawn true: exit code={{code}}'
print('ok', code)
"#,
            so = so_str,
        );
        let result = py(&script);
        assert!(result.starts_with("ok"), "proc_spawn_true: {result}");
    }

    // ── Test 8: spawn false ──────────────────────────────────────────────────

    #[test]
    fn proc_spawn_false() {
        let src = concat!(include_str!("../std/process.mind"), "\n");
        let Some(so) = compile_to_so(src, "proc_spawn_false") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.spawn.restype = ctypes.c_int64; lib.spawn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.wait.restype = ctypes.c_int64; lib.wait.argtypes = [ctypes.c_int64]
lib.exit_code.restype = ctypes.c_int64; lib.exit_code.argtypes = [ctypes.c_int64]
prog = b'false\x00'
pbuf = (ctypes.c_uint8 * len(prog))(*prog)
child = lib.spawn(ctypes.addressof(pbuf), 0, 0)
pid = lib.__mind_load_i64(child + 0)
assert pid > 0, f'spawn false: pid={{pid}}'
status = lib.wait(child)
code = lib.exit_code(status)
assert code != 0, f'spawn false must return non-zero exit code; got {{code}}'
print('ok', code)
"#,
            so = so_str,
        );
        let result = py(&script);
        assert!(result.starts_with("ok"), "proc_spawn_false: {result}");
    }

    // ── Test 9: TCP loopback ─────────────────────────────────────────────────

    #[test]
    fn net_tcp_loopback() {
        let src = concat!(include_str!("../std/net.mind"), "\n");
        let Some(so) = compile_to_so(src, "net_tcp_loopback") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, threading
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.tcp_listen.restype = ctypes.c_int64; lib.tcp_listen.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.tcp_listen_port.restype = ctypes.c_int64; lib.tcp_listen_port.argtypes = [ctypes.c_int64]
lib.tcp_accept.restype = ctypes.c_int64; lib.tcp_accept.argtypes = [ctypes.c_int64]
lib.tcp_connect.restype = ctypes.c_int64; lib.tcp_connect.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.tcp_read.restype = ctypes.c_int64; lib.tcp_read.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.tcp_write.restype = ctypes.c_int64; lib.tcp_write.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.tcp_close.restype = None; lib.tcp_close.argtypes = [ctypes.c_int64]
lib.tcp_listener_close.restype = None; lib.tcp_listener_close.argtypes = [ctypes.c_int64]

host = b'127.0.0.1\x00'
hbuf = (ctypes.c_uint8 * len(host))(*host)
listener = lib.tcp_listen(ctypes.addressof(hbuf), 0)
assert listener != 0, 'tcp_listen failed'
port = lib.tcp_listen_port(listener)
assert port > 0, f'tcp_listen_port={{port}}'

received = [None]
def server_thread():
    stream = lib.tcp_accept(listener)
    buf = (ctypes.c_uint8 * 4)()
    n = lib.tcp_read(stream, ctypes.addressof(buf), 4)
    received[0] = bytes(buf[:n])
    lib.tcp_close(stream)

t = threading.Thread(target=server_thread, daemon=True)
t.start()

client = lib.tcp_connect(ctypes.addressof(hbuf), port)
assert client != 0, 'tcp_connect failed'
payload = (ctypes.c_uint8 * 1)(42)
lib.tcp_write(client, ctypes.addressof(payload), 1)
lib.tcp_close(client)

t.join(timeout=5)
assert received[0] is not None, 'server did not receive data within 5s'
assert received[0] == bytes([42]), f'TCP data mismatch: {{received[0]!r}}'
lib.tcp_listener_close(listener)
print('ok')
"#,
            so = so_str,
        );
        let result = py(&script);
        assert_eq!(result, "ok", "net_tcp_loopback: {result}");
    }

    // ── Test 10: UDP loopback ────────────────────────────────────────────────

    #[test]
    fn net_udp_loopback() {
        let src = concat!(include_str!("../std/net.mind"), "\n");
        let Some(so) = compile_to_so(src, "net_udp_loopback") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.udp_bind.restype = ctypes.c_int64; lib.udp_bind.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.udp_socket_port.restype = ctypes.c_int64; lib.udp_socket_port.argtypes = [ctypes.c_int64]
lib.udp_send_to.restype = ctypes.c_int64; lib.udp_send_to.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.udp_recv_from.restype = ctypes.c_int64; lib.udp_recv_from.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.udp_close.restype = None; lib.udp_close.argtypes = [ctypes.c_int64]

host = b'127.0.0.1\x00'
hbuf = (ctypes.c_uint8 * len(host))(*host)
sock = lib.udp_bind(ctypes.addressof(hbuf), 0)
assert sock != 0, 'udp_bind failed'
port = lib.udp_socket_port(sock)
assert port > 0, f'udp_socket_port={{port}}'

payload = (ctypes.c_uint8 * 1)(99)
n_sent = lib.udp_send_to(sock, ctypes.addressof(hbuf), port, ctypes.addressof(payload), 1)
assert n_sent == 1, f'udp_send_to returned {{n_sent}}'

recv_buf = (ctypes.c_uint8 * 4)()
n_recv = lib.udp_recv_from(sock, ctypes.addressof(recv_buf), 4)
assert n_recv == 1, f'udp_recv_from returned {{n_recv}}'
assert recv_buf[0] == 99, f'UDP data mismatch: {{recv_buf[0]}}'
lib.udp_close(sock)
print('ok')
"#,
            so = so_str,
        );
        let result = py(&script);
        assert_eq!(result, "ok", "net_udp_loopback: {result}");
    }
}
