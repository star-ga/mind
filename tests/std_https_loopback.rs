// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std.https` against a hermetic loopback TLS 1.3 server.
//!
//! Two `mindc build` executables run side by side on 127.0.0.1: the server
//! (`tests/https_loopback/server.mind`, TEST-ONLY) and a client
//! (`tests/https_loopback/client.mind`) that calls `https_get_ca` /
//! `https_post_ca` once per server case. The PKI is test-only and committed
//! (`tests/https_loopback/`, see its README): an RSA-2048 root, an RSA-2048
//! intermediate and RSA-2048 leaves for `localhost`, all sha256WithRSAEncryption;
//! the client's bundle carries 40 filler anchors and comment lines before the
//! root. The server signs each CertificateVerify (rsa_pss_rsae_sha256) with the
//! leaf key in MIND.
//!
//! The client proves a GET (with a NewSessionTicket and a body over two
//! records), a POST echo, a chunked body, a 40000-byte body and an unfollowed
//! 301, and refuses with a named error: a wrong hostname, an expired leaf, an
//! unknown anchor, a corrupted CertificateVerify, a truncated record, an
//! oversized body, an ECDSA leaf and an RSA-3072 leaf; and, before any
//! connection, an http URL, userinfo, a malformed and a missing bundle and a
//! content type with a line break. A second run flips one expectation and must
//! report exactly one failed check, so the client cannot pass by checking
//! nothing.
//!
//! The ignored test is the manual release smoke: it fetches
//! https://three.arcprize.org/ over the network and prints the status line
//! (`cargo test --release --features "mlir-build std-surface
//! cross-module-imports" --test std_https_loopback -- --ignored --nocapture`).

#![cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/https_loopback");

/// The std modules both programs import.
const IMPORTS: &[&str] = &[
    "sha256",
    "rsa_pss",
    "x25519",
    "tls13_keyschedule",
    "tls13_finished",
    "tls13_client_core",
    "tls13_client_hello",
    "tls13_client_api",
    "https_url",
    "https_http1",
    "https",
    "dns",
    "net",
    "process",
];

/// A MIND function `name(dst) -> i64` that writes `bytes` at `dst` and returns
/// their count: whole little-endian words, then single bytes.
fn fixture_fn(name: &str, bytes: &[u8]) -> String {
    let words = bytes.len() / 8;
    let mut lines = Vec::with_capacity(words + 8);
    for w in 0..words {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[w * 8..w * 8 + 8]);
        let v = i64::from_le_bytes(word);
        let value = if v < 0 {
            format!("0 - {}", v.unsigned_abs())
        } else {
            v.to_string()
        };
        lines.push(format!("    __mind_store_i64(dst + {}, {value});", w * 8));
    }
    for (i, b) in bytes.iter().enumerate().skip(words * 8) {
        lines.push(format!("    __mind_store_i8(dst + {i}, {b});"));
    }
    format!(
        "fn {name}(dst: i64) -> i64 {{\n{}\n    return {};\n}}\n\n",
        lines.join("\n"),
        bytes.len()
    )
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex digit"))
        .collect()
}

/// `n=` / `d=` of the test-only server key.
fn key_field(field: &str) -> Vec<u8> {
    let text = std::fs::read_to_string(format!("{FIX}/server_key_test_only.txt"))
        .expect("read the test-only server key");
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{field}=")))
        .unwrap_or_else(|| panic!("server key has no {field}="));
    let bytes = hex_bytes(line.trim());
    assert_eq!(bytes.len(), 256, "{field} must be 256 bytes");
    bytes
}

fn der(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIX}/{name}.der")).unwrap_or_else(|e| panic!("read {name}.der: {e}"))
}

/// A NUL-terminated absolute path under the fixture directory.
fn path_bytes(file: &str) -> Vec<u8> {
    let mut b = format!("{FIX}/{file}").into_bytes();
    b.push(0);
    b
}

fn source(own: &str, fixtures: &[(&str, Vec<u8>)]) -> String {
    let mut src: String = IMPORTS
        .iter()
        .map(|m| format!("import std.{m};\n"))
        .collect();
    src.push('\n');
    for (name, bytes) in fixtures {
        src.push_str(&fixture_fn(name, bytes));
    }
    let body = std::fs::read_to_string(format!("{FIX}/{own}"))
        .unwrap_or_else(|e| panic!("read {own}: {e}"));
    src.push_str(&body);
    src
}

fn server_source() -> String {
    source(
        "server.mind",
        &[
            ("hl_leaf_good", der("leaf_good")),
            ("hl_leaf_expired", der("leaf_expired")),
            ("hl_leaf_wronghost", der("leaf_wronghost")),
            ("hl_leaf_ec", der("leaf_ec")),
            ("hl_leaf_rsa3072", der("leaf_rsa3072")),
            ("hl_inter", der("inter")),
            ("hl_key_n", key_field("n")),
            ("hl_key_d", key_field("d")),
        ],
    )
}

fn client_source() -> String {
    source(
        "client.mind",
        &[
            ("cl_ca_bundle", path_bytes("ca_bundle.pem")),
            ("cl_ca_other", path_bytes("ca_other.pem")),
            ("cl_ca_bad", path_bytes("ca_bad.pem")),
            ("cl_ca_missing", path_bytes("no_such_bundle.pem")),
        ],
    )
}

/// `mindc build` a one-file project named `case`; the executable, or `None`
/// when the build backend is a gap the shared gate allows to skip.
fn build_program(case: &str, main: &str) -> Option<PathBuf> {
    let dir = common::scratch_dir(&format!("std_https_{case}"));
    let manifest = format!(
        "[package]\nname = \"{case}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
         output = \"{case}\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
         sources = [\"main.mind\"]\n"
    );
    std::fs::write(dir.join("Mind.toml"), manifest).expect("write Mind.toml");
    std::fs::write(dir.join("main.mind"), main).expect("write main.mind");
    let out = Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(&dir)
        .output()
        .expect("spawn mindc build");
    if !common::gate::compiled(case, &out) {
        return None;
    }
    let exe = dir.join("target").join("debug").join(case);
    assert!(exe.is_file(), "{case}: no executable at {}", exe.display());
    Some(exe)
}

/// Kill a child that is still running when dropped.
struct Reaped(Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start the server, read its `PORT <n>` line.
fn start_server(exe: &Path) -> (Reaped, u16) {
    let mut child = Command::new(exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the loopback server");
    let stdout = child.stdout.take().expect("server stdout");
    let mut line = String::new();
    BufReader::new(stdout)
        .read_line(&mut line)
        .expect("read the server port line");
    let port = line
        .trim()
        .strip_prefix("PORT ")
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or_else(|| panic!("server printed {line:?}, not `PORT <n>`"));
    (Reaped(child), port)
}

/// Run the client against a fresh server; `(exit code, client stdout)`.
fn run_pair(server: &Path, client: &Path, mutate: bool, limit: Duration) -> (i32, String) {
    let (_server, port) = start_server(server);
    let mut cmd = Command::new(client);
    cmd.env("MIND_HTTPS_TEST_PORT", port.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if mutate {
        cmd.env("MIND_HTTPS_TEST_MUTATE", "1");
    }
    let mut child = Reaped(cmd.spawn().expect("spawn the client"));
    let started = Instant::now();
    let code = loop {
        if let Some(status) = child.0.try_wait().expect("wait on the client") {
            break status.code().expect("client exit code");
        }
        assert!(
            started.elapsed() < limit,
            "client still running after {limit:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut text = String::new();
    if let Some(mut out) = child.0.stdout.take() {
        out.read_to_string(&mut text).expect("read client stdout");
    }
    (code, text)
}

#[test]
fn https_get_and_post_against_a_loopback_tls13_server() {
    let Some(server) = build_program("hl_server", &server_source()) else {
        return;
    };
    let Some(client) = build_program("hl_client", &client_source()) else {
        return;
    };
    let limit = Duration::from_secs(900);
    let (code, out) = run_pair(&server, &client, false, limit);
    assert_eq!(
        code, 0,
        "every std.https case must hold; failed checks:\n{out}"
    );
    let (mcode, mout) = run_pair(&server, &client, true, limit);
    assert_eq!(
        mcode, 1,
        "the mutated expectation must be the one failed check:\n{mout}"
    );
    assert!(
        mout.starts_with("FAIL 1 error 0 status 200 "),
        "the mutated check must be case 0's status:\n{mout}"
    );
    println!("std_https_loopback: every case held; the mutated expectation was caught");
}

/// The manual release smoke (network): GET https://three.arcprize.org/ against
/// the system CA bundle and print the status line.
#[test]
#[ignore = "manual release smoke: reaches https://three.arcprize.org/ over the network"]
fn https_probe_three_arcprize_org() {
    let url = b"https://three.arcprize.org/";
    let probe = format!(
        "{}{}\nfn main() -> i64 {{\n    let u: i64 = __mind_alloc(64);\n    probe_url(u);\n\
         \x20   let r: i64 = https_get(u, {});\n    if https_error(r) != 0 {{\n\
         \x20       return 0 - https_error(r);\n    }}\n\
         \x20   let line: i64 = __mind_alloc(256);\n    let n: i64 = https_status_line_len(r);\n\
         \x20   let mut i: i64 = 0;\n    while i < n {{\n\
         \x20       __mind_store_i8(line + i, __mind_load_i8(https_status_line_addr(r) + i));\n\
         \x20       i = i + 1;\n    }}\n    __mind_store_i8(line + n, 10);\n\
         \x20   __mind_write(1, line, n + 1, 0 - 1);\n    return 0;\n}}\n",
        IMPORTS
            .iter()
            .map(|m| format!("import std.{m};\n"))
            .collect::<String>(),
        fixture_fn("probe_url", url),
        url.len()
    );
    let Some(exe) = build_program("hl_probe", &probe) else {
        return;
    };
    let out = Command::new(&exe).output().expect("run the probe");
    let text = String::from_utf8_lossy(&out.stdout);
    println!("three.arcprize.org: {text}");
    assert!(
        out.status.success() && text.starts_with("HTTP/1."),
        "probe failed (exit {:?}, the https_err code negated):\n{text}",
        out.status.code()
    );
}
