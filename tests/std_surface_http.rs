// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/http.mind` surface tests — HTTP/1.1 client over `std.net` + `std.string`.
//!
//! Section A (gate: `std-surface`):
//!   Verifies `std/http.mind` parses cleanly and lowers to IR with all
//!   required `pub fn`s present.
//!
//! Section B (gate: `std-surface cross-module-imports`):
//!   Verifies `std.http` auto-exports its public symbols and that a program
//!   pulling in `std.http` (whose response bodies are handed straight to
//!   `std.json`) resolves both `std.http` and `std.json` symbols cleanly
//!   through a cross-module table alongside the `std.net` primitives it wraps.
//!
//! Section C (gate: `mlir-build cross-module-imports`, unix only):
//!   Real functional round-trips compiled via `mindc --emit-shared` and
//!   exercised through Python ctypes. A tiny MIND-based test server (`std.net`
//!   `tcp_listen`/`tcp_accept`/`tcp_write`) hands back a canned HTTP/1.1
//!   response with a known JSON body; the new `std.http` client
//!   (`http_get`/`http_post`) fetches it and the parsed status code + body are
//!   asserted byte-for-byte against the canned response:
//!     1. http_get_round_trip   — GET returns status 200 + exact JSON body
//!     2. http_post_round_trip  — POST sends Content-Length + body, parses 201
//!
//! Run with:
//!   cargo test --features "std-surface mlir-build cross-module-imports" \
//!     --test std_surface_http

#![cfg(feature = "std-surface")]

mod common;
// mindc_bin is only used by Section C (the mlir-build round-trip); keep the
// import scoped to the same cfg so a std-surface-only build stays warning-clean.
#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
use common::mindc_bin;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const HTTP_MIND_SRC: &str = include_str!("../std/http.mind");

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn has_fn(ir: &libmind::ir::IRModule, name: &str) -> bool {
    ir.instrs
        .iter()
        .any(|i| matches!(i, Instr::FnDef { name: n, .. } if n == name))
}

// ─── Section A: parse + lower ─────────────────────────────────────────────────

#[test]
fn http_mind_parses_and_lowers() {
    let module = parser::parse(HTTP_MIND_SRC).expect("std/http.mind must parse");
    let ir = lower_to_ir(&module);
    for want in [
        "http_get",
        "http_post",
        "http_status",
        "http_body_addr",
        "http_body_len",
        "http_header_get",
        "http_header_count",
        "http_header_name",
        "http_header_value",
    ] {
        assert!(has_fn(&ir, want), "std/http.mind: missing FnDef `{want}`");
    }
}

// ─── Section B: cross-module exports ─────────────────────────────────────────

#[cfg(feature = "cross-module-imports")]
mod cross_module {
    use super::*;
    use libmind::project::module_table::{build_module_table, collect_module_exports};

    const JSON_MIND_SRC: &str = include_str!("../std/json.mind");
    const NET_MIND_SRC: &str = include_str!("../std/net.mind");
    const STRING_MIND_SRC: &str = include_str!("../std/string.mind");

    #[test]
    fn http_mind_auto_exports_public_symbols() {
        let module = parser::parse(HTTP_MIND_SRC).expect("std/http.mind must parse");
        let ex = collect_module_exports("std.http", &module);
        for want in [
            "http_get",
            "http_post",
            "http_status",
            "http_body_addr",
            "http_body_len",
            "http_header_get",
            "http_header_count",
            "http_header_name",
            "http_header_value",
        ] {
            assert!(
                ex.exported.iter().any(|s| s == want),
                "std.http must auto-export `{want}`; got {:?}",
                ex.exported
            );
        }
    }

    #[test]
    fn program_using_std_http_and_std_json_resolves() {
        // http.mind wraps std.net + std.string; its response bodies are handed
        // straight to std.json's jv_parse. A downstream program therefore pulls
        // in all four — assert the cross-module table resolves the public
        // surface each `use` line depends on.
        let http = parser::parse(HTTP_MIND_SRC).expect("std/http.mind must parse");
        let json = parser::parse(JSON_MIND_SRC).expect("std/json.mind must parse");
        let net = parser::parse(NET_MIND_SRC).expect("std/net.mind must parse");
        let string = parser::parse(STRING_MIND_SRC).expect("std/string.mind must parse");

        let refs: Vec<(String, &libmind::ast::Module)> = vec![
            ("std.http".into(), &http),
            ("std.json".into(), &json),
            ("std.net".into(), &net),
            ("std.string".into(), &string),
        ];
        let table = build_module_table(&refs);

        assert!(
            table.resolves(&["std".into(), "http".into()], "http_get"),
            "table must expose std.http::http_get"
        );
        assert!(
            table.resolves(&["std".into(), "http".into()], "http_post"),
            "table must expose std.http::http_post"
        );
        assert!(
            table.resolves(&["std".into(), "http".into()], "http_status"),
            "table must expose std.http::http_status"
        );
        assert!(
            table.resolves(&["std".into(), "json".into()], "jv_parse"),
            "table must expose std.json::jv_parse (http bodies feed jv_parse)"
        );
        assert!(
            table.resolves(&["std".into(), "net".into()], "tcp_connect"),
            "table must expose std.net::tcp_connect (http wraps it)"
        );
    }
}

// ─── Section C: MLIR functional round-trip (unix only) ───────────────────────

#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
mod mlir_functional {
    use super::mindc_bin;
    use std::path::PathBuf;
    use std::process::Command;

    // mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

    fn out_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_http")
    }

    /// Compile source text to a shared library at `<tag>.so`.
    /// Returns `None` (soft-skip) if mindc is not available.
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

    /// Compile the `std.net` server-primitive library and the `std.http`
    /// client library. Returns `None` if mindc is unavailable (soft-skip).
    fn compile_pair(tag: &str) -> Option<(String, String)> {
        let net_src = concat!(include_str!("../std/net.mind"), "\n");
        let http_src = concat!(include_str!("../std/http.mind"), "\n");
        let net_so = compile_to_so(net_src, &format!("{tag}_net"))?;
        let http_so = compile_to_so(http_src, &format!("{tag}_http"))?;
        Some((
            net_so.to_string_lossy().into_owned(),
            http_so.to_string_lossy().into_owned(),
        ))
    }

    // ── Test 1: GET round-trip ───────────────────────────────────────────────

    #[test]
    fn http_get_round_trip() {
        let Some((net_so, http_so)) = compile_pair("http_get_round_trip") else {
            return;
        };
        // Canned response: known status + JSON body. http.mind's client must
        // read past the header terminator, parse the 3-digit code, honour
        // Content-Length, and expose the exact body bytes.
        let script = format!(
            r#"
import ctypes, threading
from ctypes import c_int64, addressof, c_uint8

# net.so provides the server primitives (tcp_listen/tcp_accept/tcp_write);
# it is loaded RTLD_GLOBAL so http.so's extern tcp_* references resolve to it.
net  = ctypes.CDLL('{net_so}',  mode=ctypes.RTLD_GLOBAL)
http = ctypes.CDLL('{http_so}', mode=ctypes.RTLD_GLOBAL)

for f, rt, at in [
    ("tcp_listen",         c_int64, [c_int64, c_int64]),
    ("tcp_listen_port",    c_int64, [c_int64]),
    ("tcp_accept",         c_int64, [c_int64]),
    ("tcp_read",           c_int64, [c_int64, c_int64, c_int64]),
    ("tcp_write",          c_int64, [c_int64, c_int64, c_int64]),
    ("tcp_close",          None,    [c_int64]),
    ("tcp_listener_close", None,    [c_int64]),
]:
    fn = getattr(net, f); fn.restype = rt; fn.argtypes = at

http.http_get.restype       = c_int64; http.http_get.argtypes       = [c_int64]*7
http.http_status.restype    = c_int64; http.http_status.argtypes    = [c_int64]
http.http_body_addr.restype = c_int64; http.http_body_addr.argtypes = [c_int64]
http.http_body_len.restype  = c_int64; http.http_body_len.argtypes  = [c_int64]

host = b"127.0.0.1\x00"
hbuf = (c_uint8 * len(host))(*host)
listener = net.tcp_listen(addressof(hbuf), 0)
assert listener != 0, "tcp_listen failed"
port = net.tcp_listen_port(listener)
assert port > 0, f"tcp_listen_port={{port}}"

body = b'{{"ok":true,"n":42}}'
response = (b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(body)) + body
rbuf = (c_uint8 * len(response))(*response)

def server():
    stream = net.tcp_accept(listener)
    req = (c_uint8 * 4096)()
    net.tcp_read(stream, addressof(req), 4096)   # drain the request line+headers
    total = len(response); w = 0
    while w < total:
        n = net.tcp_write(stream, addressof(rbuf) + w, total - w)
        if n <= 0:
            break
        w += n
    net.tcp_close(stream)

t = threading.Thread(target=server, daemon=True)
t.start()

path = b"/api/games"
pbuf = (c_uint8 * (len(path) + 1))(*path, 0)
h = http.http_get(addressof(hbuf), 9, port, addressof(pbuf), len(path), 0, 0)
assert h != 0, "http_get returned null handle"
st = http.http_status(h)
bl = http.http_body_len(h)
ba = http.http_body_addr(h)
got = bytes((c_uint8 * bl).from_address(ba))

t.join(timeout=5)
assert st == 200, f"status mismatch: {{st}}"
assert bl == len(body), f"body len mismatch: {{bl}} vs {{len(body)}}"
assert got == body, f"body mismatch: {{got!r}}"
net.tcp_listener_close(listener)
print("ok")
"#,
            net_so = net_so,
            http_so = http_so,
        );
        let result = py(&script);
        assert_eq!(result, "ok", "http_get_round_trip: {result}");
    }

    // ── Test 2: POST round-trip ──────────────────────────────────────────────

    #[test]
    fn http_post_round_trip() {
        let Some((net_so, http_so)) = compile_pair("http_post_round_trip") else {
            return;
        };
        // The client must emit an auto-computed Content-Length and append the
        // raw body; the server captures the request to assert both, then hands
        // back a canned 201 with its own JSON body.
        let script = format!(
            r#"
import ctypes, threading
from ctypes import c_int64, addressof, c_uint8

net  = ctypes.CDLL('{net_so}',  mode=ctypes.RTLD_GLOBAL)
http = ctypes.CDLL('{http_so}', mode=ctypes.RTLD_GLOBAL)

for f, rt, at in [
    ("tcp_listen",         c_int64, [c_int64, c_int64]),
    ("tcp_listen_port",    c_int64, [c_int64]),
    ("tcp_accept",         c_int64, [c_int64]),
    ("tcp_read",           c_int64, [c_int64, c_int64, c_int64]),
    ("tcp_write",          c_int64, [c_int64, c_int64, c_int64]),
    ("tcp_close",          None,    [c_int64]),
    ("tcp_listener_close", None,    [c_int64]),
]:
    fn = getattr(net, f); fn.restype = rt; fn.argtypes = at

http.http_post.restype      = c_int64; http.http_post.argtypes      = [c_int64]*11
http.http_status.restype    = c_int64; http.http_status.argtypes    = [c_int64]
http.http_body_addr.restype = c_int64; http.http_body_addr.argtypes = [c_int64]
http.http_body_len.restype  = c_int64; http.http_body_len.argtypes  = [c_int64]

host = b"127.0.0.1\x00"
hbuf = (c_uint8 * len(host))(*host)
listener = net.tcp_listen(addressof(hbuf), 0)
assert listener != 0, "tcp_listen failed"
port = net.tcp_listen_port(listener)
assert port > 0, f"tcp_listen_port={{port}}"

resp_body = b'{{"echo":"pong"}}'
response = (b"HTTP/1.1 201 Created\r\n"
            b"Content-Length: %d\r\n\r\n" % len(resp_body)) + resp_body
rbuf = (c_uint8 * len(response))(*response)

captured = [b""]
def server():
    stream = net.tcp_accept(listener)
    req = (c_uint8 * 8192)()
    n = net.tcp_read(stream, addressof(req), 8192)
    captured[0] = bytes(req[:n])
    total = len(response); w = 0
    while w < total:
        k = net.tcp_write(stream, addressof(rbuf) + w, total - w)
        if k <= 0:
            break
        w += k
    net.tcp_close(stream)

t = threading.Thread(target=server, daemon=True)
t.start()

path = b"/api/cmd/RESET"
pbuf = (c_uint8 * (len(path) + 1))(*path, 0)
ct   = b"application/json"
ctbuf = (c_uint8 * len(ct))(*ct)
req_body = b'{{"game":"x"}}'
bbuf = (c_uint8 * len(req_body))(*req_body)

h = http.http_post(addressof(hbuf), 9, port,
                   addressof(pbuf), len(path),
                   0, 0,
                   addressof(ctbuf), len(ct),
                   addressof(bbuf), len(req_body))
assert h != 0, "http_post returned null handle"
st = http.http_status(h)
bl = http.http_body_len(h)
ba = http.http_body_addr(h)
got = bytes((c_uint8 * bl).from_address(ba))

t.join(timeout=5)
assert st == 201, f"status mismatch: {{st}}"
assert got == resp_body, f"resp body mismatch: {{got!r}}"
# The client must have emitted a correct Content-Length and appended the body.
sent = captured[0]
assert b"Content-Length: %d" % len(req_body) in sent, f"missing/incorrect Content-Length: {{sent!r}}"
assert sent.endswith(req_body), f"request body not appended: {{sent!r}}"
assert b"Content-Type: application/json" in sent, f"missing Content-Type: {{sent!r}}"
net.tcp_listener_close(listener)
print("ok")
"#,
            net_so = net_so,
            http_so = http_so,
        );
        let result = py(&script);
        assert_eq!(result, "ok", "http_post_round_trip: {result}");
    }
}
