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
//! refused, each with its own `https_err_*` code. The framing guards: a
//! repeated Content-Length, a transfer coding other than chunked, and a
//! Content-Length over 18 digits (one that would wrap i64) are refused, with
//! accepted positive controls beside each; CR, LF, SP and other control bytes
//! in the URL request target are refused before any connection. An 8 MiB POST
//! body (the request maximum) reaches the server whole, and the same request
//! is refused after-send when a proxy resets the connection while the client
//! writes the body. Each refusal also has its error class: before-send for the
//! handshake and request refusals, after-send for a refused or cut response
//! and for a request whose write failed.

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
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

// std.https error codes (std/https_url.mind).
const E_URL: i64 = -1;
const E_HOSTNAME: i64 = -6;
const E_UNKNOWN_ANCHOR: i64 = -8;
const E_UNSUPPORTED_KEY: i64 = -11;
const E_TRUNCATED: i64 = -12;
const E_BODY_TOO_LARGE: i64 = -13;
const E_RESPONSE: i64 = -14;
const E_IO: i64 = -15;
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
        assert_refused_class(tag, &seen, *code, CLASS_AFTER_SEND);
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
        assert_refused_class(tag, &seen, E_RESPONSE, CLASS_AFTER_SEND);
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
    assert_refused_class("trunc_cl", &seen, E_TRUNCATED, CLASS_AFTER_SEND);
    // A chunked body without its last chunk.
    let mut open = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
    open.extend_from_slice(format!("{:x}\r\n", 4096).as_bytes());
    open.extend_from_slice(&body[..4096]);
    open.extend_from_slice(b"\r\n");
    let seen = run_case(env, &Case::get("trunc_chunked", open));
    assert_refused_class("trunc_chunked", &seen, E_TRUNCATED, CLASS_AFTER_SEND);
    // The TCP stream cut in the middle of a record of the body.
    let mut case = Case::get("trunc_tcp", content_length_response(&body, ""));
    case.cut = Some(100_000);
    let seen = run_case(env, &case);
    assert_refused_class("trunc_tcp", &seen, E_TRUNCATED, CLASS_AFTER_SEND);
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
        assert_refused_class(tag, &seen, E_REQUEST_HEADERS, CLASS_BEFORE_SEND);
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
    assert_refused_class("wronghost", &seen, E_HOSTNAME, CLASS_BEFORE_SEND);
}

#[test]
fn an_untrusted_chain_is_refused() {
    let Some(env) = env() else { return };
    let mut case = Case::get("untrusted", Vec::new());
    case.ca = "ca_other.pem";
    case.response = None;
    let seen = run_case(env, &case);
    assert_refused_class("untrusted", &seen, E_UNKNOWN_ANCHOR, CLASS_BEFORE_SEND);
}

#[test]
fn an_rsa_3072_leaf_is_refused_as_unsupported() {
    let Some(env) = env() else { return };
    let mut case = Case::get("rsa3072", Vec::new());
    case.leaf = RSA3072;
    case.response = None;
    let seen = run_case(env, &case);
    assert_refused_class("rsa3072", &seen, E_UNSUPPORTED_KEY, CLASS_BEFORE_SEND);
}

/// `head` + CRLF + `body`, as one response.
fn response(head: &str, body: &[u8]) -> Vec<u8> {
    let mut r = format!("HTTP/1.1 200 OK\r\n{head}\r\n").into_bytes();
    r.extend_from_slice(body);
    r
}

#[test]
fn a_repeated_content_length_is_refused() {
    let Some(env) = env() else { return };
    for (tag, head) in [
        (
            "cl_twice_diff",
            "Content-Length: 5\r\nContent-Length: 2\r\n",
        ),
        (
            "cl_twice_same",
            "Content-Length: 5\r\nContent-Length: 5\r\n",
        ),
        (
            "cl_twice_case",
            "content-length: 2\r\nCONTENT-LENGTH: 5\r\n",
        ),
        (
            "cl_apart",
            "Content-Length: 5\r\nX-A: b\r\nContent-Length: 4\r\n",
        ),
    ] {
        let seen = run_case(env, &Case::get(tag, response(head, b"hello")));
        assert_refused_class(tag, &seen, E_RESPONSE, CLASS_AFTER_SEND);
    }
    // The positive control: one Content-Length.
    let seen = run_case(
        env,
        &Case::get("cl_once", response("Content-Length: 5\r\n", b"hello")),
    );
    assert_ok_body("cl_once", &seen, b"hello");
}

#[test]
fn a_transfer_coding_other_than_chunked_is_refused() {
    let Some(env) = env() else { return };
    let framed = b"5\r\nhello\r\n0\r\n\r\n";
    for (tag, head, body) in [
        (
            "te_gzip_chunked",
            "Transfer-Encoding: gzip, chunked\r\n",
            &framed[..],
        ),
        (
            "te_chunked_twice",
            "Transfer-Encoding: chunked, chunked\r\n",
            framed,
        ),
        ("te_gzip", "Transfer-Encoding: gzip\r\n", b"hello"),
        ("te_identity", "Transfer-Encoding: identity\r\n", b"hello"),
        ("te_empty", "Transfer-Encoding: \r\n", b"hello"),
    ] {
        let seen = run_case(env, &Case::get(tag, response(head, body)));
        assert_refused_class(tag, &seen, E_RESPONSE, CLASS_AFTER_SEND);
    }
    // The positive controls: chunked, in any case.
    for (tag, head) in [
        ("te_chunked", "Transfer-Encoding: chunked\r\n"),
        ("te_chunked_upper", "Transfer-Encoding: CHUNKED\r\n"),
    ] {
        let seen = run_case(env, &Case::get(tag, response(head, framed)));
        assert_ok_body(tag, &seen, b"hello");
    }
}

#[test]
fn a_content_length_over_18_digits_is_refused() {
    let Some(env) = env() else { return };
    for (tag, value) in [
        ("cl_19_digits", "0000000000000000005"),
        ("cl_i64_wrap", "9223372036854775813"),
        ("cl_u64_wrap", "18446744073709551621"),
    ] {
        let head = format!("Content-Length: {value}\r\n");
        let seen = run_case(env, &Case::get(tag, response(&head, b"hello")));
        assert_refused_class(tag, &seen, E_RESPONSE, CLASS_AFTER_SEND);
    }
    // The positive control: 18 digits are read.
    let seen = run_case(
        env,
        &Case::get(
            "cl_18_digits",
            response("Content-Length: 000000000000000005\r\n", b"hello"),
        ),
    );
    assert_ok_body("cl_18_digits", &seen, b"hello");
}

#[test]
fn control_bytes_in_the_url_target_are_refused_before_connecting() {
    let Some(env) = env() else { return };
    for (tag, path) in [
        ("url_crlf", "/a\r\nX-Injected: 1"),
        ("url_lf", "/a\nb"),
        ("url_cr", "/a\rb"),
        ("url_sp", "/a b"),
        ("url_sp_query", "/?q=a b"),
        ("url_tab", "/a\tb"),
        ("url_nul", "/a\0b"),
        ("url_del", "/a\x7fb"),
    ] {
        let mut case = Case::get(tag, Vec::new());
        case.response = None;
        case.path = path;
        let seen = run_case(env, &case);
        assert_refused_class(tag, &seen, E_URL, CLASS_BEFORE_SEND);
        assert!(seen.records.is_none(), "{tag}: the client connected");
    }
    // The positive control: the same bytes percent-encoded reach the server.
    let mut case = Case::get("url_encoded", response("Content-Length: 2\r\n", b"ok"));
    case.path = "/a%20b?q=%0D%0A";
    let seen = run_case(env, &case);
    assert_ok_body("url_encoded", &seen, b"ok");
    let req = String::from_utf8_lossy(&seen.request).into_owned();
    assert!(
        req.starts_with("GET /a%20b?q=%0D%0A HTTP/1.1\r\n"),
        "request line: {req}"
    );
}

/// https_max_request_body() (std/https_url.mind): 8 MiB.
const MAX_REQUEST_BODY: usize = 8 * 1024 * 1024;

/// The client bytes the resetting proxy lets through: the handshake and the
/// first records of the request.
const RESET_AFTER: usize = 32 * 1024;

/// A one-connection proxy to s_server on `upstream` that forwards everything
/// the server sends and the first `after` bytes the client sends (the
/// handshake, then the start of the request), then stops reading and closes
/// the client's socket while request bytes are still arriving. A close with
/// unread bytes sends a TCP reset, so the client's next write fails. The
/// handle yields the client bytes forwarded and whether the client was still
/// writing, or `None` when no client connected before `stop`.
fn start_resetting_proxy(
    upstream: u16,
    after: usize,
    stop: Arc<AtomicBool>,
) -> (u16, thread::JoinHandle<Option<(usize, bool)>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind the proxy");
    let port = listener.local_addr().expect("proxy address").port();
    let handle = thread::spawn(move || {
        let mut client = accept_within(&listener, Duration::from_secs(300), &stop)?;
        let started = Instant::now();
        let mut server = loop {
            match TcpStream::connect(("127.0.0.1", upstream)) {
                Ok(s) => break s,
                Err(_) if started.elapsed() < Duration::from_secs(20) => {
                    thread::sleep(Duration::from_millis(20))
                }
                Err(e) => panic!("proxy: s_server never listened on {upstream}: {e}"),
            }
        };
        let mut s_in = server.try_clone().expect("clone server");
        let mut c_out = client.try_clone().expect("clone client");
        let down = thread::spawn(move || {
            let mut buf = vec![0u8; 65536];
            while let Ok(n) = s_in.read(&mut buf) {
                if n == 0 || c_out.write_all(&buf[..n]).is_err() {
                    break;
                }
            }
        });
        let mut forwarded = 0usize;
        let mut buf = vec![0u8; 65536];
        while forwarded < after {
            let want = (after - forwarded).min(buf.len());
            let n = match client.read(&mut buf[..want]) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            if server.write_all(&buf[..n]).is_err() {
                break;
            }
            forwarded += n;
        }
        client
            .set_read_timeout(Some(Duration::from_secs(60)))
            .expect("proxy read timeout");
        let writing = matches!(client.peek(&mut buf[..1]), Ok(n) if n > 0);
        // End s_server's side so the forwarder drops its copy of the client
        // socket: the close below is the last one, and it resets.
        let _ = server.shutdown(Shutdown::Both);
        let _ = down.join();
        drop(client);
        Some((forwarded, writing))
    });
    (port, handle)
}

/// POST `body` to s_server through [`start_resetting_proxy`]: the client's
/// result and what the proxy saw.
fn post_reset_mid_request(env: &Env, body: &[u8]) -> (Reply, Option<(usize, bool)>) {
    let s_port = server_port();
    // s_server's stdin stays open (held in `_server`) until the client is done.
    let (_server, _log) = start_server(env, GOOD, s_port, "reset_mid_request");
    let stop = Arc::new(AtomicBool::new(false));
    let (p_port, proxy) = start_resetting_proxy(s_port, RESET_AFTER, Arc::clone(&stop));
    let mut ca = fix("ca_bundle.pem").into_os_string().into_encoded_bytes();
    ca.push(0);
    let mut spec = Vec::new();
    put_field(&mut spec, b"P");
    put_field(
        &mut spec,
        format!("https://localhost:{p_port}/upload").as_bytes(),
    );
    put_field(&mut spec, b"");
    put_field(&mut spec, b"application/octet-stream");
    put_field(&mut spec, body);
    put_field(&mut spec, &ca);
    let reply = run_client(&env.client, spec);
    stop.store(true, Ordering::SeqCst);
    (reply, proxy.join().expect("proxy thread"))
}

#[test]
fn a_reset_while_the_request_is_written_is_after_send() {
    let Some(env) = env() else { return };
    let body = pattern(MAX_REQUEST_BODY);
    // The positive control: the same request, not cut, reaches s_server whole
    // and is answered.
    let mut case = Case::get("post_8m", content_length_response(b"stored", ""));
    case.post = true;
    case.path = "/upload";
    case.content_type = b"application/octet-stream".to_vec();
    case.body = body.clone();
    let seen = run_case(env, &case);
    assert_ok_body("post_8m", &seen, b"stored");
    assert!(
        seen.request.ends_with(&body),
        "post_8m: s_server got {} request bytes, not the whole 8 MiB body",
        seen.request.len()
    );
    // The connection is reset after the handshake and the first request
    // records, while the client still writes the body. The server may have
    // acted on what it got, so the failure is after-send: never safe to retry.
    let (reply, cut) = post_reset_mid_request(env, &body);
    let (forwarded, writing) = cut.expect("reset_mid_request: the client never connected");
    assert_eq!(
        forwarded, RESET_AFTER,
        "reset_mid_request: client bytes that reached s_server"
    );
    assert!(
        writing,
        "reset_mid_request: the client was no longer writing when the proxy reset the connection"
    );
    assert_eq!(reply.exit, 0, "reset_mid_request: client exit status");
    let seen = Seen {
        reply,
        request: Vec::new(),
        records: None,
    };
    assert_refused_class("reset_mid_request", &seen, E_IO, CLASS_AFTER_SEND);
}
