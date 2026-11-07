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

// The harness also serves std_https_network_safe.rs: its canned-case `End`
// variants are never built here.
#[path = "support/https_loopback_harness.rs"]
#[allow(dead_code)]
mod harness;

use harness::IMPORTS;
use harness::{Reaped, build_program, fixture_fn, path_bytes, server_source, source, start_server};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn client_source() -> String {
    source(
        "client.mind",
        &[],
        &[
            ("cl_ca_bundle", path_bytes("ca_bundle.pem")),
            ("cl_ca_other", path_bytes("ca_other.pem")),
            ("cl_ca_bad", path_bytes("ca_bad.pem")),
            ("cl_ca_missing", path_bytes("no_such_bundle.pem")),
        ],
        "",
    )
}

/// Run the client against a fresh server; `(exit code, client stdout)`.
fn run_pair(server: &Path, client: &Path, mutate: bool, limit: Duration) -> (i32, String) {
    let (_server, port) = start_server(server, None);
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
    let Some(server) = build_program("hl_server", &server_source(&[])) else {
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
