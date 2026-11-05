// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std.https` against `openssl s_server` on 127.0.0.1 (no outside network).
//!
//! An independent TLS 1.3 server: `openssl s_server` presents the committed
//! test-only chain (`tests/https_loopback/`: RSA-2048 root, intermediate and
//! leaves for `localhost`, keys `*_key_test_only.cnf`). s_server prints the
//! decrypted request on its stdout, where this harness asserts on it (method,
//! target, Host, X-API-Key, Cookie, Content-Type, Content-Length, body), and
//! sends whatever the harness writes to its stdin as the response, in records
//! of up to 16384 plaintext bytes, then close_notify. A TCP proxy between the
//! client and s_server records the length of every server record and can cut
//! the stream mid-record.
//!
//! The client is `tests/https_loopback/request.mind`, one `mindc build`
//! executable that performs one request per run (spec on stdin, result on
//! stdout). Cases: a 16 KiB record; a 5 MiB Content-Length body and a 5 MiB
//! chunked body, compared byte for byte; request headers asserted at the
//! server and Set-Cookie values captured; malformed chunk sizes, a control
//! byte in a response header and a malformed request header block refused;
//! a body truncated by close_notify, by a missing last chunk and by a TCP
//! cut refused; a wrong hostname, an untrusted chain and an RSA-3072 leaf
//! refused, each with its own `https_err_*` code.

#![cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

#[path = "support/https_openssl_harness.rs"]
mod harness;

use harness::*;

// std.https error codes (std/https_url.mind).
const E_HOSTNAME: i64 = -6;
const E_UNKNOWN_ANCHOR: i64 = -8;
const E_UNSUPPORTED_KEY: i64 = -11;
const E_TRUNCATED: i64 = -12;
const E_BODY_TOO_LARGE: i64 = -13;
const E_RESPONSE: i64 = -14;
const E_REQUEST_HEADERS: i64 = -19;

/// 5 MiB.
const BIG: usize = 5 * 1024 * 1024;

/// A full TLS 1.3 record of application data on the wire: 2^14 plaintext
/// bytes, the inner content type and a 16-byte AEAD tag.
const FULL_RECORD: usize = 16384 + 1 + 16;

#[test]
fn a_16_kib_record_reaches_the_caller() {
    let Some(env) = env() else { return };
    let body = pattern(16384);
    let seen = run_case(
        env,
        &Case::get("record16k", content_length_response(&body, "")),
    );
    assert_ok_body("record16k", &seen, &body);
    assert!(
        seen.records().contains(&FULL_RECORD),
        "s_server sent no full 16 KiB record: {:?}",
        seen.records
    );
}

#[test]
fn a_5_mib_content_length_body_streams_through() {
    let Some(env) = env() else { return };
    let body = pattern(BIG);
    let seen = run_case(env, &Case::get("cl5m", content_length_response(&body, "")));
    assert_ok_body("cl5m", &seen, &body);
    let full = seen.records().iter().filter(|&&n| n == FULL_RECORD).count();
    assert!(
        full >= BIG / 16384 - 2,
        "only {full} full-size records carried the 5 MiB body"
    );
}

#[test]
fn a_5_mib_chunked_body_streams_through() {
    let Some(env) = env() else { return };
    let body = pattern(BIG);
    let seen = run_case(env, &Case::get("chunk5m", chunked(&body)));
    assert_ok_body("chunk5m", &seen, &body);
}

#[test]
fn request_headers_reach_the_server_and_set_cookie_is_captured() {
    let Some(env) = env() else { return };
    let resp = content_length_response(
        b"{\"ok\":true}",
        "Set-Cookie: AWSALB=alb-1234; Expires=Thu, 01 Jan 2099 00:00:00 GMT; Path=/\r\n\
         Content-Type: application/json\r\nset-cookie: session=s-42 ;HttpOnly\r\n",
    );
    let mut case = Case::get("headers", resp);
    case.post = true;
    case.path = "/api/cmd/ACTION1?card=c-1";
    case.headers =
        b"X-API-Key: test-key-0123456789\r\nCookie: AWSALB=old; session=s-41\r\n".to_vec();
    case.content_type = b"application/json".to_vec();
    case.body = b"{\"game_id\":\"ls20\",\"x\":3}".to_vec();
    let seen = run_case(env, &case);
    assert_ok_body("headers", &seen, b"{\"ok\":true}");
    let content_length = format!("Content-Length: {}", case.body.len());
    let req = String::from_utf8(seen.request.clone()).expect("an ASCII request");
    let (head, body) = req.split_once("\r\n\r\n").expect("a request head");
    let lines: Vec<&str> = head.split("\r\n").collect();
    assert_eq!(lines[0], "POST /api/cmd/ACTION1?card=c-1 HTTP/1.1", "{req}");
    for want in [
        "X-API-Key: test-key-0123456789",
        "Cookie: AWSALB=old; session=s-41",
        "Content-Type: application/json",
        content_length.as_str(),
        "Connection: close",
    ] {
        assert!(
            lines.contains(&want),
            "the server did not get `{want}`:\n{req}"
        );
    }
    assert!(
        lines.iter().any(|l| l.starts_with("Host: localhost:")),
        "{req}"
    );
    assert_eq!(body, "{\"game_id\":\"ls20\",\"x\":3}");
    let got: Vec<(String, String)> = seen
        .reply
        .cookies
        .iter()
        .map(|(w, p)| {
            (
                String::from_utf8_lossy(w).into_owned(),
                String::from_utf8_lossy(p).into_owned(),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (
                "AWSALB=alb-1234; Expires=Thu, 01 Jan 2099 00:00:00 GMT; Path=/".to_string(),
                "AWSALB=alb-1234".to_string()
            ),
            (
                "session=s-42 ;HttpOnly".to_string(),
                "session=s-42".to_string()
            ),
        ]
    );
}

#[test]
fn malformed_chunk_sizes_are_refused() {
    let Some(env) = env() else { return };
    let head = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
    let cases: &[(&str, &str, i64)] = &[
        ("chunk_nohex", "zz\r\nhello\r\n0\r\n\r\n", E_RESPONSE),
        ("chunk_empty", "\r\nhello\r\n0\r\n\r\n", E_RESPONSE),
        ("chunk_neg", "-5\r\nhello\r\n0\r\n\r\n", E_RESPONSE),
        ("chunk_0x", "0x5\r\nhello\r\n0\r\n\r\n", E_RESPONSE),
        ("chunk_lead_sp", " 5\r\nhello\r\n0\r\n\r\n", E_RESPONSE),
        ("chunk_junk", "5 x\r\nhello\r\n0\r\n\r\n", E_RESPONSE),
        ("chunk_bare_lf", "5\nhello\r\n0\r\n\r\n", E_RESPONSE),
        (
            "chunk_ext_ctl",
            "5;a\x01b\r\nhello\r\n0\r\n\r\n",
            E_RESPONSE,
        ),
        ("chunk_no_crlf", "5\r\nhelloXY0\r\n\r\n", E_RESPONSE),
        (
            "chunk_overflow",
            "ffffffffffffffffffffffff\r\nhello\r\n0\r\n\r\n",
            E_BODY_TOO_LARGE,
        ),
        (
            "chunk_over_cap",
            "800001\r\nhello\r\n0\r\n\r\n",
            E_BODY_TOO_LARGE,
        ),
    ];
    for (tag, tail, code) in cases {
        let seen = run_case(env, &Case::get(tag, format!("{head}{tail}").into_bytes()));
        assert_refused(tag, &seen, *code);
    }
}

#[test]
fn a_control_byte_in_a_response_header_is_refused() {
    let Some(env) = env() else { return };
    for (tag, line) in [
        ("hdr_nul", "X-A: a\x00b\r\n"),
        ("hdr_bare_cr", "X-A: a\rb\r\n"),
        ("hdr_bare_lf", "X-A: a\nb\r\n"),
    ] {
        let resp = format!("HTTP/1.1 200 OK\r\n{line}Content-Length: 2\r\n\r\nok");
        let seen = run_case(env, &Case::get(tag, resp.into_bytes()));
        assert_refused(tag, &seen, E_RESPONSE);
    }
    // The positive control: the same response without the byte is accepted.
    let seen = run_case(
        env,
        &Case::get(
            "hdr_clean",
            b"HTTP/1.1 200 OK\r\nX-A: a b\r\nContent-Length: 2\r\n\r\nok".to_vec(),
        ),
    );
    assert_ok_body("hdr_clean", &seen, b"ok");
}

#[test]
fn a_truncated_body_is_refused() {
    let Some(env) = env() else { return };
    let body = pattern(1 << 20);
    // Content-Length 1 MiB, half sent, then close_notify.
    let mut short =
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
    short.extend_from_slice(&body[..body.len() / 2]);
    let seen = run_case(env, &Case::get("trunc_cl", short));
    assert_refused("trunc_cl", &seen, E_TRUNCATED);
    // A chunked body without its last chunk.
    let mut open = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
    open.extend_from_slice(format!("{:x}\r\n", 4096).as_bytes());
    open.extend_from_slice(&body[..4096]);
    open.extend_from_slice(b"\r\n");
    let seen = run_case(env, &Case::get("trunc_chunked", open));
    assert_refused("trunc_chunked", &seen, E_TRUNCATED);
    // The TCP stream cut in the middle of a record of the body.
    let mut case = Case::get("trunc_tcp", content_length_response(&body, ""));
    case.cut = Some(100_000);
    let seen = run_case(env, &case);
    assert_refused("trunc_tcp", &seen, E_TRUNCATED);
}

#[test]
fn a_malformed_request_header_block_is_refused_before_connecting() {
    let Some(env) = env() else { return };
    for (tag, block) in [
        ("rq_host", &b"Host: evil.example\r\n"[..]),
        ("rq_content_length", b"Content-Length: 0\r\n"),
        ("rq_te", b"Transfer-Encoding: chunked\r\n"),
        ("rq_no_crlf", b"X-API-Key: k"),
        ("rq_lf", b"X-API-Key: k\nCookie: a=b\r\n"),
        ("rq_cr", b"X-API-Key: k\rCookie: a=b\r\n"),
        ("rq_nul", b"X-API-Key: k\x00\r\n"),
        ("rq_name", b"X API Key: k\r\n"),
        ("rq_empty_name", b": k\r\n"),
    ] {
        let mut case = Case::get(tag, Vec::new());
        case.response = None;
        case.headers = block.to_vec();
        let seen = run_case(env, &case);
        assert_refused(tag, &seen, E_REQUEST_HEADERS);
        assert!(seen.records.is_none(), "{tag}: the client connected");
    }
}

#[test]
fn a_wrong_hostname_is_refused() {
    let Some(env) = env() else { return };
    let mut case = Case::get("wronghost", Vec::new());
    case.leaf = WRONG_HOST;
    case.response = None;
    let seen = run_case(env, &case);
    assert_refused("wronghost", &seen, E_HOSTNAME);
}

#[test]
fn an_untrusted_chain_is_refused() {
    let Some(env) = env() else { return };
    let mut case = Case::get("untrusted", Vec::new());
    case.ca = "ca_other.pem";
    case.response = None;
    let seen = run_case(env, &case);
    assert_refused("untrusted", &seen, E_UNKNOWN_ANCHOR);
}

#[test]
fn an_rsa_3072_leaf_is_refused_as_unsupported() {
    let Some(env) = env() else { return };
    let mut case = Case::get("rsa3072", Vec::new());
    case.leaf = RSA3072;
    case.response = None;
    let seen = run_case(env, &case);
    assert_refused("rsa3072", &seen, E_UNSUPPORTED_KEY);
}
