// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The `openssl s_server` harness of `tests/std_https_openssl.rs`: the client
//! build, the server, the recording TCP proxy, the client's stdin request and
//! stdout result, and the response builders.

use std::fs::File;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

pub const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/https_loopback");

/// The std modules the client imports.
pub const IMPORTS: &[&str] = &[
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
];

pub struct Env {
    pub client: PathBuf,
    pub openssl: PathBuf,
    pub inter_pem: PathBuf,
    pub dir: PathBuf,
}

/// The server certificate and its key.
#[derive(Clone, Copy)]
pub struct Leaf {
    pub cert: &'static str,
    pub key: &'static str,
}

pub const GOOD: Leaf = Leaf {
    cert: "leaf_good.der",
    key: "leaf_key_test_only.cnf",
};
pub const WRONG_HOST: Leaf = Leaf {
    cert: "leaf_wronghost.der",
    key: "leaf_key_test_only.cnf",
};
pub const RSA3072: Leaf = Leaf {
    cert: "leaf_rsa3072.der",
    key: "leaf_rsa3072_key_test_only.cnf",
};

pub fn fix(name: &str) -> PathBuf {
    Path::new(FIX).join(name)
}

/// `mindc build` the client once per test binary; `None` when the build
/// backend is a gap the shared gate allows to skip.
pub fn env() -> Option<&'static Env> {
    static ENV: OnceLock<Option<Env>> = OnceLock::new();
    ENV.get_or_init(|| {
        let openssl = which::which("openssl")
            .expect("std_https_openssl needs the openssl command line tool on PATH");
        let dir = crate::common::scratch_dir("std_https_openssl");
        let inter_pem = dir.join("inter.pem");
        let conv = Command::new(&openssl)
            .args(["x509", "-inform", "DER", "-in"])
            .arg(fix("inter.der"))
            .arg("-out")
            .arg(&inter_pem)
            .output()
            .expect("spawn openssl x509");
        assert!(conv.status.success(), "openssl x509: {conv:?}");
        // The test-only keys are committed as `asn1parse -genconf` text.
        for key in [GOOD.key, RSA3072.key] {
            let der = Command::new(&openssl)
                .args(["asn1parse", "-noout", "-genconf"])
                .arg(fix(key))
                .arg("-out")
                .arg(dir.join(format!("{key}.der")))
                .output()
                .expect("spawn openssl asn1parse");
            assert!(der.status.success(), "openssl asn1parse {key}: {der:?}");
        }
        let mut src: String = IMPORTS
            .iter()
            .map(|m| format!("import std.{m};\n"))
            .collect();
        src.push('\n');
        src.push_str(&std::fs::read_to_string(fix("request.mind")).expect("read request.mind"));
        let proj = dir.join("client");
        std::fs::create_dir_all(&proj).expect("create the client project");
        std::fs::write(
            proj.join("Mind.toml"),
            "[package]\nname = \"hs_request\"\nversion = \"0.1.0\"\n\n[build]\nentry = \
             \"main.mind\"\noutput = \"hs_request\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend \
             = \"cpu\"\nsources = [\"main.mind\"]\n",
        )
        .expect("write Mind.toml");
        std::fs::write(proj.join("main.mind"), src).expect("write main.mind");
        let out = Command::new(crate::common::mindc_bin())
            .args(["build", "--backend", "mlir"])
            .current_dir(&proj)
            .output()
            .expect("spawn mindc build");
        if !crate::common::gate::compiled("hs_request", &out) {
            return None;
        }
        let client = proj.join("target").join("debug").join("hs_request");
        assert!(client.is_file(), "no executable at {}", client.display());
        Some(Env {
            client,
            openssl,
            inter_pem,
            dir,
        })
    })
    .as_ref()
}

/// Kill a child that is still running when dropped.
pub struct Reaped(Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A port for s_server below the kernel's ephemeral range (32768 and up on
/// Linux), distinct for every call in this process. A port taken from an
/// ephemeral bind and released could be handed straight to another test's
/// proxy listener before s_server binds it, joining two cases' streams.
pub fn server_port() -> u16 {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let base = 20000 + (std::process::id() as usize % 400) * 16;
    for _ in 0..6000 {
        let k = NEXT.fetch_add(1, Ordering::SeqCst);
        let port = (base + k % 6000) as u16;
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    panic!("no free port in 20000..32767 for s_server");
}

/// `openssl s_server` for one connection on 127.0.0.1:`port`.
pub fn start_server(env: &Env, leaf: Leaf, port: u16, tag: &str) -> (Reaped, PathBuf) {
    let err = env.dir.join(format!("{tag}.s_server.err"));
    let child = Command::new(&env.openssl)
        .arg("s_server")
        .args(["-accept", &format!("127.0.0.1:{port}")])
        .arg("-cert")
        .arg(fix(leaf.cert))
        .args(["-certform", "DER", "-keyform", "DER", "-key"])
        .arg(env.dir.join(format!("{}.der", leaf.key)))
        .arg("-cert_chain")
        .arg(&env.inter_pem)
        .args(["-tls1_3", "-naccept", "1", "-quiet", "-no_ign_eof"])
        .args(["-num_tickets", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(
            File::create(&err).expect("create the s_server log"),
        ))
        .spawn()
        .expect("spawn openssl s_server");
    (Reaped(child), err)
}

/// Accept one connection before `limit` or `stop`, or `None`.
pub fn accept_within(
    listener: &TcpListener,
    limit: Duration,
    stop: &AtomicBool,
) -> Option<TcpStream> {
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");
    let started = Instant::now();
    while started.elapsed() < limit && !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((s, _)) => {
                s.set_nonblocking(false).expect("blocking stream");
                return Some(s);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10))
            }
            Err(e) => panic!("proxy accept: {e}"),
        }
    }
    None
}

/// TLS record lengths seen on a byte stream.
#[derive(Default)]
pub struct Records {
    pub lens: Vec<usize>,
    pub header: Vec<u8>,
    pub skip: usize,
}

impl Records {
    pub fn feed(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            if self.skip > 0 {
                let n = self.skip.min(data.len());
                self.skip -= n;
                data = &data[n..];
                continue;
            }
            let need = 5 - self.header.len();
            let n = need.min(data.len());
            self.header.extend_from_slice(&data[..n]);
            data = &data[n..];
            if self.header.len() == 5 {
                let len = (usize::from(self.header[3]) << 8) | usize::from(self.header[4]);
                self.lens.push(len);
                self.skip = len;
                self.header.clear();
            }
        }
    }
}

/// A one-connection TCP proxy to `upstream`: returns its port and a handle
/// that yields the server-to-client record lengths, or `None` when no client
/// connected before `stop` was set. With `cut`, the server-to-client stream
/// stops after that many bytes and both sockets close.
pub type ProxyHandle = thread::JoinHandle<Option<Vec<usize>>>;

pub fn start_proxy(upstream: u16, cut: Option<usize>, stop: Arc<AtomicBool>) -> (u16, ProxyHandle) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind the proxy");
    let port = listener.local_addr().expect("proxy address").port();
    let handle = thread::spawn(move || {
        let client = accept_within(&listener, Duration::from_secs(300), &stop)?;
        let started = Instant::now();
        let server = loop {
            match TcpStream::connect(("127.0.0.1", upstream)) {
                Ok(s) => break s,
                Err(_) if started.elapsed() < Duration::from_secs(20) => {
                    thread::sleep(Duration::from_millis(20))
                }
                Err(e) => panic!("proxy: s_server never listened on {upstream}: {e}"),
            }
        };
        let mut c_in = client.try_clone().expect("clone client");
        let mut s_out = server.try_clone().expect("clone server");
        let up = thread::spawn(move || {
            let mut buf = vec![0u8; 65536];
            loop {
                match c_in.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if s_out.write_all(&buf[..n]).is_err() {
                            break;
                        }
                    }
                }
            }
            let _ = s_out.shutdown(Shutdown::Write);
        });
        let mut s_in = server;
        let mut c_out = client;
        let mut records = Records::default();
        let mut sent = 0usize;
        let mut buf = vec![0u8; 65536];
        loop {
            let n = match s_in.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            let take = match cut {
                Some(c) => n.min(c - sent),
                None => n,
            };
            records.feed(&buf[..take]);
            if c_out.write_all(&buf[..take]).is_err() {
                break;
            }
            sent += take;
            if cut == Some(sent) {
                break;
            }
        }
        let _ = c_out.shutdown(Shutdown::Both);
        let _ = s_in.shutdown(Shutdown::Both);
        let _ = up.join();
        Some(records.lens)
    });
    (port, handle)
}

/// One field of the client's stdin request: length, bytes, zero padding.
pub fn put_field(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as i64).to_le_bytes());
    out.extend_from_slice(bytes);
    while out.len() % 8 != 0 {
        out.push(0);
    }
}

#[derive(Debug, Default)]
pub struct Reply {
    pub exit: i32,
    pub error: i64,
    /// `https_error_class`: 0 none, 1 before send, 2 after send.
    pub class: i64,
    pub status: i64,
    pub verdict: i64,
    pub cookies: Vec<(Vec<u8>, Vec<u8>)>,
    pub body: Vec<u8>,
}

pub fn take_line(s: &mut &[u8], tag: u8) -> i64 {
    let nl = s.iter().position(|&b| b == b'\n').expect("result line");
    let line = &s[..nl];
    assert!(
        line.len() > 2 && line[0] == tag && line[1] == b' ',
        "expected `{} <n>`, got {:?}",
        tag as char,
        String::from_utf8_lossy(line)
    );
    let v = std::str::from_utf8(&line[2..])
        .expect("ascii")
        .parse()
        .expect("a number");
    *s = &s[nl + 1..];
    v
}

pub fn take_blob(s: &mut &[u8], tag: u8) -> Vec<u8> {
    let n = take_line(s, tag) as usize;
    let v = s[..n].to_vec();
    assert_eq!(s[n], b'\n', "blob terminator");
    *s = &s[n + 1..];
    v
}

pub fn parse_reply(exit: i32, out: &[u8]) -> Reply {
    let mut s = out;
    let error = take_line(&mut s, b'E');
    let class = take_line(&mut s, b'L');
    let status = take_line(&mut s, b'S');
    let verdict = take_line(&mut s, b'V');
    let count = take_line(&mut s, b'C');
    let mut cookies = Vec::new();
    for _ in 0..count.min(8) {
        let whole = take_blob(&mut s, b'K');
        let pair = take_blob(&mut s, b'P');
        cookies.push((whole, pair));
    }
    let body = take_blob(&mut s, b'B');
    assert!(s.is_empty(), "trailing client output");
    Reply {
        exit,
        error,
        class,
        status,
        verdict,
        cookies,
        body,
    }
}

/// Run the client with `spec` on stdin; its parsed result.
pub fn run_client(client: &Path, spec: Vec<u8>) -> Reply {
    let mut child = Reaped(
        Command::new(client)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn the client"),
    );
    let mut stdin = child.0.stdin.take().expect("client stdin");
    let feeder = thread::spawn(move || {
        let _ = stdin.write_all(&spec);
    });
    let mut stdout = child.0.stdout.take().expect("client stdout");
    let reader = thread::spawn(move || {
        let mut v = Vec::new();
        let _ = stdout.read_to_end(&mut v);
        v
    });
    let started = Instant::now();
    let code = loop {
        if let Some(st) = child.0.try_wait().expect("wait on the client") {
            break st.code().unwrap_or(-1);
        }
        assert!(
            started.elapsed() < Duration::from_secs(600),
            "client still running after 600 s"
        );
        thread::sleep(Duration::from_millis(20));
    };
    let _ = feeder.join();
    let out = reader.join().expect("client stdout reader");
    parse_reply(code, &out)
}

/// What the server side observed.
pub struct Seen {
    pub reply: Reply,
    pub request: Vec<u8>,
    /// Server record lengths, `None` when the client never connected.
    pub records: Option<Vec<usize>>,
}

impl Seen {
    pub fn records(&self) -> &[usize] {
        self.records.as_deref().unwrap_or(&[])
    }
}

pub struct Case {
    pub tag: &'static str,
    pub leaf: Leaf,
    pub ca: &'static str,
    pub post: bool,
    pub path: &'static str,
    pub headers: Vec<u8>,
    pub content_type: Vec<u8>,
    pub body: Vec<u8>,
    /// The response s_server sends, or `None` when no request may arrive.
    pub response: Option<Vec<u8>>,
    pub cut: Option<usize>,
}

impl Case {
    pub fn get(tag: &'static str, response: Vec<u8>) -> Self {
        Case {
            tag,
            leaf: GOOD,
            ca: "ca_bundle.pem",
            post: false,
            path: "/",
            headers: Vec::new(),
            content_type: Vec::new(),
            body: Vec::new(),
            response: Some(response),
            cut: None,
        }
    }
}

/// End of the request head and the Content-Length it declares.
pub fn request_complete(buf: &[u8]) -> bool {
    let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    let head = String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase();
    let cl = head
        .split("\r\n")
        .find_map(|l| l.strip_prefix("content-length:"))
        .map(|v| v.trim().parse::<usize>().expect("request Content-Length"))
        .unwrap_or(0);
    buf.len() >= end + 4 + cl
}

pub fn run_case(env: &Env, case: &Case) -> Seen {
    let s_port = server_port();
    let (mut server, err_log) = start_server(env, case.leaf, s_port, case.tag);
    let stop = Arc::new(AtomicBool::new(false));
    let (p_port, proxy) = start_proxy(s_port, case.cut, Arc::clone(&stop));
    let seen = Arc::new(Mutex::new(Vec::<u8>::new()));
    let mut s_out = server.0.stdout.take().expect("s_server stdout");
    let sink = Arc::clone(&seen);
    let reader = thread::spawn(move || {
        let mut buf = vec![0u8; 65536];
        loop {
            match s_out.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => sink.lock().expect("lock").extend_from_slice(&buf[..n]),
            }
        }
    });
    let url = format!("https://localhost:{p_port}{}", case.path);
    let ca = if case.ca.is_empty() {
        Vec::new()
    } else {
        let mut p = fix(case.ca).into_os_string().into_encoded_bytes();
        p.push(0);
        p
    };
    let mut spec = Vec::new();
    put_field(&mut spec, if case.post { b"P" } else { b"G" });
    put_field(&mut spec, url.as_bytes());
    put_field(&mut spec, &case.headers);
    put_field(&mut spec, &case.content_type);
    put_field(&mut spec, &case.body);
    put_field(&mut spec, &ca);
    let mut stdin = server.0.stdin.take().expect("s_server stdin");
    let exe = env.client.clone();
    let client = thread::spawn(move || run_client(&exe, spec));
    // With no response expected, s_server's stdin stays open until the client
    // is done: its EOF would make s_server close the connection mid-handshake.
    let mut idle_stdin = None;
    let mut writer = None;
    if let Some(response) = case.response.clone() {
        let started = Instant::now();
        loop {
            if request_complete(&seen.lock().expect("lock")) {
                break;
            }
            if client.is_finished() {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(300),
                "{}: no request after 300 s; s_server log: {}",
                case.tag,
                std::fs::read_to_string(&err_log).unwrap_or_default()
            );
            thread::sleep(Duration::from_millis(10));
        }
        writer = Some(thread::spawn(move || {
            let _ = stdin.write_all(&response);
            drop(stdin);
        }));
    } else {
        idle_stdin = Some(stdin);
    }
    let reply = client.join().expect("client thread");
    drop(idle_stdin);
    stop.store(true, Ordering::SeqCst);
    if let Some(w) = writer {
        let _ = w.join();
    }
    drop(server);
    let _ = reader.join();
    let records = proxy.join().expect("proxy thread");
    let request = seen.lock().expect("lock").clone();
    assert_eq!(reply.exit, 0, "{}: client exit status", case.tag);
    Seen {
        reply,
        request,
        records,
    }
}

/// Byte `i` of the large test bodies.
pub fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| ((i * 7 + 3) % 251) as u8).collect()
}

pub fn content_length_response(body: &[u8], extra_headers: &str) -> Vec<u8> {
    let mut r = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n{extra_headers}\r\n",
        body.len()
    )
    .into_bytes();
    r.extend_from_slice(body);
    r
}

/// A chunked encoding of `body` with varied chunk sizes, upper- and
/// lower-case hex, a leading-zero size, an extension and a trailer.
pub fn chunked(body: &[u8]) -> Vec<u8> {
    let sizes = [1usize, 15, 4096, 16384, 65536, 100_003, 7, 300_000];
    let mut out = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
    let mut off = 0;
    let mut k = 0;
    while off < body.len() {
        let n = sizes[k % sizes.len()].min(body.len() - off);
        let line = match k % 4 {
            0 => format!("{n:x}\r\n"),
            1 => format!("{n:X}\r\n"),
            2 => format!("000{n:x};ext=\"v\"\r\n"),
            _ => format!("{n:x} \r\n"),
        };
        out.extend_from_slice(line.as_bytes());
        out.extend_from_slice(&body[off..off + n]);
        out.extend_from_slice(b"\r\n");
        off += n;
        k += 1;
    }
    out.extend_from_slice(b"0\r\nX-Trailer: done\r\n\r\n");
    out
}

pub fn assert_ok_body(tag: &str, seen: &Seen, want: &[u8]) {
    let r = &seen.reply;
    assert_eq!(
        (r.error, r.class, r.status, r.verdict),
        (0, CLASS_NONE, 200, 0),
        "{tag}: error / class / status / chain verdict"
    );
    assert_eq!(r.body.len(), want.len(), "{tag}: body length");
    assert!(r.body == want, "{tag}: body bytes differ");
}

/// `https_error_class` values (std/https_url.mind).
pub const CLASS_NONE: i64 = 0;
pub const CLASS_BEFORE_SEND: i64 = 1;
pub const CLASS_AFTER_SEND: i64 = 2;

/// A refusal with its code and the class that says whether the server can
/// have seen the request.
pub fn assert_refused_class(tag: &str, seen: &Seen, code: i64, class: i64) {
    assert_refused(tag, seen, code);
    assert_eq!(seen.reply.class, class, "{tag}: error class");
}

pub fn assert_refused(tag: &str, seen: &Seen, code: i64) {
    let r = &seen.reply;
    assert_eq!(
        (r.error, r.status, r.body.len()),
        (code, 0, 0),
        "{tag}: a refusal carries its code and no status or body"
    );
}
