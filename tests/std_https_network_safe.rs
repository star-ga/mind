// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Network-safe use of `std.https`: the error class of every request says
//! whether the failure happened before any request byte was written (safe to
//! retry) or after (the server may have acted: pause, never count it), and
//! `std.https_probe` tells when the network is back.
//!
//! The reference server is the loopback TLS 1.3 server in MIND
//! (`tests/https_loopback/server.mind`, test-only chain for `localhost`). The
//! client is `tests/https_loopback/steps.mind`: one `mindc build` executable
//! that runs one call per step written on its stdin and answers it before
//! reading the next, so the test changes the network between two calls. The
//! client runs with a preloaded `getaddrinfo` (a MIND `--emit-shared` shim)
//! that answers 127.0.0.1 for every name, or EAI_AGAIN while the file named by
//! `MIND_TEST_DNS_OUTAGE` exists: a DNS outage the test switches on and off.
//!
//! Cases: a DNS outage between two requests (the probe reports it, the
//! request reports before-send, and after the outage the same request returns
//! the exact body); a refused port and a full accept queue (probe codes, and
//! before-send for the request); a TLS refusal before the request and a cut
//! after it; a body framed by the connection, accepted only on
//! close_notify: a fatal alert, a warning alert and a TCP close are refused
//! after-send; and a response that arrives before any request byte is written
//! is refused before-send, never a success. For that case the client also
//! preloads a `getrandom` that returns fixed bytes, so its second ClientHello
//! equals its first: a proxy records the server's flight, response and
//! close_notify on the first connection, and a replayer sends them all at once
//! after the second ClientHello, before the client has written anything else.

#![cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

#[path = "support/https_loopback_harness.rs"]
#[allow(dead_code)]
mod harness;

use harness::{End, Reaped};
use harness::{build_program, fixture_fn, path_bytes, server_source, source, start_server};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, channel};
use std::thread::JoinHandle;
use std::time::Duration;

// std.https codes (std/https_url.mind).
const E_CONNECT: i64 = -3;
const E_TLS: i64 = -5;
const E_HOSTNAME: i64 = -6;
const E_TRUNCATED: i64 = -12;
const E_RESPONSE: i64 = -14;
// https_error_class values.
const NONE: i64 = 0;
const BEFORE_SEND: i64 = 1;
const AFTER_SEND: i64 = 2;
// std.https_probe codes (std/https_probe.mind).
const PROBE_OK: i64 = 0;
const PROBE_DNS_AGAIN: i64 = -2;
const PROBE_REFUSED: i64 = -4;
const PROBE_TIMEOUT: i64 = -5;

/// The answer of the canned API case.
const API_BODY: &[u8] = br#"{"game_id":"ls20-1","state":"NOT_FINISHED","score":0}"#;
/// The body of the cases framed by the connection.
const CLOSE_BODY: &[u8] = b"a body that ends with the connection";

/// The canned server cases, ids `FIRST_CANNED` (13) up, in this order.
fn canned() -> Vec<(Vec<u8>, End)> {
    let mut api = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        API_BODY.len()
    )
    .into_bytes();
    api.extend_from_slice(API_BODY);
    let mut close = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n".to_vec();
    close.extend_from_slice(CLOSE_BODY);
    vec![
        (api, End::CloseNotify),            // 13
        (close.clone(), End::CloseNotify),  // 14
        (close.clone(), End::FatalAlert),   // 15
        (close.clone(), End::WarningAlert), // 16
        (close, End::TcpClose),             // 17
    ]
}

/// A preloaded `getaddrinfo` / `freeaddrinfo`: 127.0.0.1 for every name, or
/// EAI_AGAIN (-3) while the file named by MIND_TEST_DNS_OUTAGE exists.
const SHIM: &str = r#"extern "C" {
    unsafe fn getenv(name: *const u8) -> i64;
    unsafe fn access(path: *const u8, mode: i32) -> i32;
}

fn sh_zero(p: i64, n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        __mind_store_i8(p + i, 0);
        i = i + 1;
    }
    return 0;
}

pub fn getaddrinfo(node: i64, service: i64, hints: i64, res: i64) -> i64 {
    __mind_store_i64(res, 0);
    let name: i64 = __mind_alloc(32);
    sh_env_name(name);
    let flag: i64 = getenv(name as *const u8);
    if flag != 0 {
        if access(flag as *const u8, 0) as i64 == 0 {
            return 0 - 3;
        }
    }
    let nd: i64 = __mind_alloc(48);
    sh_zero(nd, 48);
    __mind_store_i32(nd + 4, 2);
    __mind_store_i32(nd + 8, 1);
    __mind_store_i32(nd + 16, 16);
    let sa: i64 = __mind_alloc(16);
    sh_zero(sa, 16);
    __mind_store_i16(sa, 2);
    __mind_store_i8(sa + 4, 127);
    __mind_store_i8(sa + 7, 1);
    __mind_store_i64(nd + 24, sa);
    __mind_store_i64(res, nd);
    return 0;
}

pub fn freeaddrinfo(res: i64) -> i64 {
    return 0;
}
"#;

/// A preloaded `getrandom` that writes the same bytes on every call: every
/// connection of a client process sends the same ClientHello and holds the
/// same key share, so a server flight recorded on one connection is valid on
/// the next. Only the replay test preloads it.
const FIXED_ENTROPY: &str = r#"pub fn getrandom(buf: i64, len: i64, flags: i64) -> i64 {
    let mut i: i64 = 0;
    while i < len {
        __mind_store_i8(buf + i, (i * 29 + 7) & 255);
        i = i + 1;
    }
    return len;
}
"#;

struct Env {
    server: PathBuf,
    steps: PathBuf,
    shim: PathBuf,
    fixed_entropy: PathBuf,
    dir: PathBuf,
}

/// `mindc --emit-shared` the MIND source `text` as `<dir>/<name>.so`.
/// `world`: the source reads the world, so the build allows it.
fn emit_shared(dir: &Path, name: &str, text: &str, world: bool) -> PathBuf {
    let src = dir.join(format!("{name}.mind"));
    std::fs::write(&src, text).expect("write the shim source");
    let so = dir.join(format!("{name}.so"));
    let mut cmd = Command::new(common::mindc_bin());
    cmd.arg(&src).arg("--emit-shared").arg(&so);
    if world {
        cmd.arg("--allow-nondeterministic");
    }
    let out = cmd.output().expect("spawn mindc --emit-shared");
    assert!(
        out.status.success(),
        "mindc --emit-shared failed for {name}\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    so
}

/// Build the server, the step client and the shim once per test binary;
/// `None` when the build backend is a gap the shared gate allows to skip.
fn env() -> Option<&'static Env> {
    static ENV: OnceLock<Option<Env>> = OnceLock::new();
    ENV.get_or_init(|| {
        let server = build_program("ns_server", &server_source(&canned()))?;
        let steps_src = source(
            "steps.mind",
            &["https_probe"],
            &[
                ("st_ca_bundle", path_bytes("ca_bundle.pem")),
                ("st_lit_prefix", b"https://localhost:".to_vec()),
                ("st_lit_ip", b"127.0.0.1\0".to_vec()),
            ],
            "",
        );
        let steps = build_program("ns_steps", &steps_src)?;
        let dir = common::scratch_dir("std_https_network_safe");
        let text = format!(
            "{}{SHIM}",
            fixture_fn("sh_env_name", b"MIND_TEST_DNS_OUTAGE\0")
        );
        // The resolver shim reads the environment and the file system.
        let shim = emit_shared(&dir, "gai_outage", &text, true);
        let fixed_entropy = emit_shared(&dir, "fixed_entropy", FIXED_ENTROPY, false);
        Some(Env {
            server,
            steps,
            shim,
            fixed_entropy,
            dir,
        })
    })
    .as_ref()
}

/// One answer of the step client.
#[derive(Debug, PartialEq)]
enum Answer {
    Get {
        error: i64,
        class: i64,
        status: i64,
        body: Vec<u8>,
    },
    Probe(i64),
}

/// The step client, run with the shim and its own outage flag file.
struct Steps {
    _child: Reaped,
    stdin: ChildStdin,
    answers: Receiver<Answer>,
}

fn numbers(line: &str, tag: char, n: usize) -> Vec<i64> {
    let mut parts = line.trim_end().split(' ');
    assert_eq!(
        parts.next(),
        Some(tag.to_string().as_str()),
        "answer {line:?}"
    );
    let v: Vec<i64> = parts.map(|p| p.parse().expect("a number")).collect();
    assert_eq!(v.len(), n, "answer {line:?}");
    v
}

impl Steps {
    fn start(env: &Env, flag: &Path) -> Self {
        Self::start_preloaded(env, flag, env.shim.clone().into_os_string())
    }

    /// The step client with the fixed-entropy `getrandom` preloaded too.
    fn start_fixed_entropy(env: &Env, flag: &Path) -> Self {
        let mut preload = env.shim.clone().into_os_string();
        preload.push(":");
        preload.push(&env.fixed_entropy);
        Self::start_preloaded(env, flag, preload)
    }

    fn start_preloaded(env: &Env, flag: &Path, preload: std::ffi::OsString) -> Self {
        let mut child = Command::new(&env.steps)
            .env("LD_PRELOAD", preload)
            .env("MIND_TEST_DNS_OUTAGE", flag)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn the step client");
        let stdin = child.stdin.take().expect("client stdin");
        let mut out = BufReader::new(child.stdout.take().expect("client stdout"));
        let (tx, answers) = channel();
        std::thread::spawn(move || {
            loop {
                let mut line = String::new();
                if out.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                let answer = if line.starts_with('G') {
                    let v = numbers(&line, 'G', 4);
                    let mut body = vec![0u8; v[3] as usize + 1];
                    out.read_exact(&mut body).expect("the body and its newline");
                    assert_eq!(body.pop(), Some(b'\n'), "body terminator");
                    Answer::Get {
                        error: v[0],
                        class: v[1],
                        status: v[2],
                        body,
                    }
                } else {
                    Answer::Probe(numbers(&line, 'P', 1)[0])
                };
                if tx.send(answer).is_err() {
                    return;
                }
            }
        });
        Steps {
            _child: Reaped(child),
            stdin,
            answers,
        }
    }

    fn step(&mut self, op: i64, arg: i64, url: &str) -> Answer {
        let mut msg = Vec::with_capacity(24 + url.len());
        for v in [op, arg, url.len() as i64] {
            msg.extend_from_slice(&v.to_le_bytes());
        }
        msg.extend_from_slice(url.as_bytes());
        self.stdin.write_all(&msg).expect("write a step");
        self.stdin.flush().expect("flush a step");
        self.answers
            .recv_timeout(Duration::from_secs(300))
            .expect("the step client answered within 300 s")
    }

    fn get(&mut self, url: &str) -> (i64, i64, i64, Vec<u8>) {
        match self.step(1, 0, url) {
            Answer::Get {
                error,
                class,
                status,
                body,
            } => (error, class, status, body),
            other => panic!("GET answered {other:?}"),
        }
    }

    fn probe(&mut self, url: &str, ms: i64) -> i64 {
        match self.step(2, ms, url) {
            Answer::Probe(code) => code,
            other => panic!("probe answered {other:?}"),
        }
    }

    fn probe_full_backlog(&mut self, ms: i64) -> i64 {
        match self.step(3, ms, "") {
            Answer::Probe(code) => code,
            other => panic!("probe answered {other:?}"),
        }
    }
}

/// A flag file path for one test; absent at the start.
fn flag(env: &Env, tag: &str) -> PathBuf {
    let p = env.dir.join(format!("{tag}.dns_down"));
    let _ = std::fs::remove_file(&p);
    p
}

/// A refusal: the code, its class, and no status or body.
fn refused(tag: &str, got: (i64, i64, i64, Vec<u8>), code: i64, class: i64) {
    assert_eq!(
        (got.0, got.1, got.2, got.3.len()),
        (code, class, 0, 0),
        "{tag}: error / class / status / body length"
    );
}

#[test]
fn a_dns_outage_between_two_requests_is_reported_before_send_and_recovers() {
    let Some(env) = env() else { return };
    let down = flag(env, "outage");
    let (_server, port) = start_server(&env.server, Some("13,13"));
    let mut st = Steps::start(env, &down);
    let url = format!("https://localhost:{port}/api/cmd/ACTION1?card=c-1");
    // Before the outage the request is answered.
    assert_eq!(
        st.get(&url),
        (0, NONE, 200, API_BODY.to_vec()),
        "the first request"
    );
    assert_eq!(
        st.probe(&url, 2000),
        PROBE_OK,
        "the probe before the outage"
    );
    // The resolver fails while the flag file exists.
    std::fs::write(&down, b"down").expect("start the outage");
    assert_eq!(
        st.probe(&url, 2000),
        PROBE_DNS_AGAIN,
        "the probe must report the outage"
    );
    refused("outage", st.get(&url), E_CONNECT, BEFORE_SEND);
    assert_eq!(st.probe(&url, 2000), PROBE_DNS_AGAIN, "still down");
    // After the outage the same request returns the exact body.
    std::fs::remove_file(&down).expect("end the outage");
    assert_eq!(st.probe(&url, 2000), PROBE_OK, "the probe after the outage");
    assert_eq!(
        st.get(&url),
        (0, NONE, 200, API_BODY.to_vec()),
        "the same request after the outage"
    );
}

#[test]
fn refused_and_unanswered_connections_are_named_and_before_send() {
    let Some(env) = env() else { return };
    let mut st = Steps::start(env, &flag(env, "refused"));
    // A port with no listener.
    let closed = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = closed.local_addr().expect("address").port();
    drop(closed);
    let url = format!("https://localhost:{port}/");
    assert_eq!(st.probe(&url, 2000), PROBE_REFUSED, "a closed port");
    refused("closed port", st.get(&url), E_CONNECT, BEFORE_SEND);
    // A listener whose accept queue is full never completes the connect.
    assert_eq!(
        st.probe_full_backlog(300),
        PROBE_TIMEOUT,
        "a full accept queue"
    );
}

#[test]
fn a_tls_refusal_is_before_send_and_a_cut_response_is_after_send() {
    let Some(env) = env() else { return };
    // 5: a leaf for another host; 9: half of a response record, then EOF.
    let (_server, port) = start_server(&env.server, Some("5,9"));
    let mut st = Steps::start(env, &flag(env, "classes"));
    let url = format!("https://localhost:{port}/");
    refused("wrong host", st.get(&url), E_HOSTNAME, BEFORE_SEND);
    refused("cut response", st.get(&url), E_TRUNCATED, AFTER_SEND);
}

#[test]
fn a_body_framed_by_the_connection_is_complete_only_on_close_notify() {
    let Some(env) = env() else { return };
    let (_server, port) = start_server(&env.server, Some("14,15,16,17"));
    let mut st = Steps::start(env, &flag(env, "alerts"));
    let url = format!("https://localhost:{port}/stream");
    assert_eq!(
        st.get(&url),
        (0, NONE, 200, CLOSE_BODY.to_vec()),
        "close_notify ends the body"
    );
    refused(
        "fatal internal_error alert",
        st.get(&url),
        E_TLS,
        AFTER_SEND,
    );
    refused(
        "user_canceled warning alert",
        st.get(&url),
        E_TLS,
        AFTER_SEND,
    );
    refused(
        "TCP close without an alert",
        st.get(&url),
        E_TRUNCATED,
        AFTER_SEND,
    );
}

/// Both directions of one connection, recorded by [`record_one`].
struct Recording {
    client: Vec<u8>,
    server: Vec<u8>,
}

/// Copy `from` to `to` until `from` ends, then end `to`'s write side; the
/// handle yields every byte read from `from`, kept even when `to` has
/// already closed (a client that stops after a complete response).
fn copy_and_record(mut from: TcpStream, mut to: TcpStream) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut seen = Vec::new();
        let mut buf = vec![0u8; 16640];
        let mut open = true;
        while let Ok(n) = from.read(&mut buf) {
            if n == 0 {
                break;
            }
            seen.extend_from_slice(&buf[..n]);
            open = open && to.write_all(&buf[..n]).is_ok();
        }
        let _ = to.shutdown(Shutdown::Write);
        seen
    })
}

/// A one-connection TCP proxy to 127.0.0.1:`upstream` that records both
/// directions; its port and a handle that yields the recording once both
/// sides have closed.
fn record_one(upstream: u16) -> (u16, JoinHandle<Recording>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind the recorder");
    let port = listener.local_addr().expect("recorder address").port();
    let handle = std::thread::spawn(move || {
        let (client, _) = listener.accept().expect("accept the client");
        let server = TcpStream::connect(("127.0.0.1", upstream)).expect("connect the server");
        let up = copy_and_record(
            client.try_clone().expect("clone the client"),
            server.try_clone().expect("clone the server"),
        );
        let down = copy_and_record(server, client);
        Recording {
            client: up.join().expect("the client copy"),
            server: down.join().expect("the server copy"),
        }
    });
    (port, handle)
}

/// The first TLS record of `bytes` (header and payload).
fn first_record(bytes: &[u8]) -> &[u8] {
    assert!(bytes.len() >= 5, "no TLS record header in {bytes:?}");
    let end = 5 + ((usize::from(bytes[3]) << 8) | usize::from(bytes[4]));
    assert!(bytes.len() >= end, "a cut TLS record in {bytes:?}");
    &bytes[..end]
}

/// The content type of every whole TLS record in `bytes`.
fn record_types(mut bytes: &[u8]) -> Vec<u8> {
    let mut types = Vec::new();
    while bytes.len() >= 5 {
        let rec = first_record(bytes);
        types.push(rec[0]);
        bytes = &bytes[rec.len()..];
    }
    types
}

/// A one-connection server that reads the client's first record (its
/// ClientHello), answers with all of `server_bytes` in one write, and then
/// reads what the client sends until it closes. The handle yields
/// (ClientHello, the client's bytes after it).
fn replay_one(server_bytes: Vec<u8>) -> (u16, JoinHandle<(Vec<u8>, Vec<u8>)>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind the replayer");
    let port = listener.local_addr().expect("replayer address").port();
    let handle = std::thread::spawn(move || {
        let (mut client, _) = listener.accept().expect("accept the client");
        client
            .set_read_timeout(Some(Duration::from_secs(300)))
            .expect("replayer read timeout");
        let mut hello = vec![0u8; 5];
        client.read_exact(&mut hello).expect("a record header");
        let len = (usize::from(hello[3]) << 8) | usize::from(hello[4]);
        hello.resize(5 + len, 0);
        client.read_exact(&mut hello[5..]).expect("the ClientHello");
        client
            .write_all(&server_bytes)
            .expect("send the recorded server bytes");
        let mut after = Vec::new();
        let _ = client.read_to_end(&mut after);
        (hello, after)
    });
    (port, handle)
}

#[test]
fn a_response_before_any_request_byte_is_refused_before_send() {
    let Some(env) = env() else { return };
    let (_server, port) = start_server(&env.server, Some("13"));
    let mut st = Steps::start_fixed_entropy(env, &flag(env, "early"));
    let url = |p: u16| format!("https://localhost:{p}/api/cmd/ACTION1?card=c-1");
    // Record one real exchange: the server's flight, the response and
    // close_notify. The positive control: the client accepts it.
    let (rport, recorder) = record_one(port);
    assert_eq!(
        st.get(&url(rport)),
        (0, NONE, 200, API_BODY.to_vec()),
        "the recorded request"
    );
    let rec = recorder.join().expect("the recorder");
    // Replay every server byte at once after the same ClientHello: the
    // response now arrives with the server's flight, before the client has
    // written its Finished or any request byte.
    let (eport, replayer) = replay_one(rec.server.clone());
    let got = st.get(&url(eport));
    let (hello, after) = replayer.join().expect("the replayer");
    assert_eq!(
        hello,
        first_record(&rec.client),
        "the replayed flight answers the recorded ClientHello only if the \
         client's entropy is fixed"
    );
    refused(
        "a response before the request",
        got,
        E_RESPONSE,
        BEFORE_SEND,
    );
    assert!(
        !record_types(&after).contains(&23),
        "the client wrote application data (its Finished or the request) \
         after the early response: {:?}",
        record_types(&after)
    );
}
