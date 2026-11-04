// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std.entropy` and `std.dns` in executables built by `mindc build`.
//!
//! Each case builds a real project that imports the module, runs the
//! executable, and reads its exit code (42 = every check inside passed; any
//! other value names the check that failed, see the program text).
//!
//! The entropy error paths are driven through a test seam at the extern
//! boundary: a MIND shim, compiled with `mindc --emit-shared`, that defines
//! `getrandom` and is loaded with `LD_PRELOAD`, so the executable's call to the
//! C library's `getrandom` lands in the shim. No production code carries a
//! test hook.
//!
//! Network: localhost only. Names resolve through /etc/hosts or as numeric
//! literals; sockets bind 127.0.0.1.

#![cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Write a one-file project (`main.mind`) and build it.
fn build(dir: &Path, main: &str) -> Output {
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"ed\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"ed\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(dir.join("main.mind"), main).expect("write main.mind");
    Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(dir)
        .output()
        .expect("run mindc build")
}

/// Build `main` in a scratch project; `None` when routed through the
/// capability-skip gate.
fn build_exe(target: &str, main: &str) -> Option<std::path::PathBuf> {
    let dir = common::scratch_dir(target);
    let out = build(&dir, main);
    if !common::gate::compiled(target, &out) {
        return None;
    }
    Some(dir.join("target").join("debug").join("ed"))
}

/// Run `exe` (optionally with `LD_PRELOAD`), killing it after `limit`.
/// Returns the exit code and the elapsed time.
fn run(exe: &Path, preload: Option<&Path>, limit: Duration) -> (i32, Duration) {
    let mut cmd = Command::new(exe);
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    if let Some(p) = preload {
        cmd.env("LD_PRELOAD", p);
    }
    let start = Instant::now();
    let mut child = cmd.spawn().expect("spawn the built executable");
    loop {
        if let Some(st) = child.try_wait().expect("wait on the executable") {
            return (st.code().expect("exit code"), start.elapsed());
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            panic!("{} still running after {limit:?}; killed", exe.display());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

// ─── std.entropy ─────────────────────────────────────────────────────────────

/// 600 bytes = chunks of 256, 256 and 88. Exit 20/30 = the expected failure
/// with the buffer wiped; 21/31 = the failure left bytes behind.
const ENTROPY_MAIN: &str = r#"import std.entropy;

fn ent_zero(p: i64, n: i64) {
    let mut i: i64 = 0;
    while i < n {
        __mind_store_i8(p + i, 0);
        i = i + 1;
    }
}

fn ent_all_zero(p: i64, n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        if __mind_load_i8(p + i) != 0 {
            return 0;
        }
        i = i + 1;
    }
    return 1;
}

fn ent_same(a: i64, b: i64, n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        if __mind_load_i8(a + i) != __mind_load_i8(b + i) {
            return 0;
        }
        i = i + 1;
    }
    return 1;
}

fn main() -> i64 {
    let n: i64 = 600;
    let a: i64 = __mind_alloc(n);
    let b: i64 = __mind_alloc(n);
    ent_zero(a, n);
    ent_zero(b, n);
    let rc: i64 = entropy.entropy_fill(a, n);
    if rc == entropy.entropy_err_os() {
        if ent_all_zero(a, n) == 1 {
            return 20;
        }
        return 21;
    }
    if rc == entropy.entropy_err_short() {
        if ent_all_zero(a, n) == 1 {
            return 30;
        }
        return 31;
    }
    if rc != entropy.entropy_ok() {
        return 1;
    }
    if entropy.entropy_fill(b, n) != entropy.entropy_ok() {
        return 2;
    }
    if ent_all_zero(a, 256) == 1 {
        return 3;
    }
    if ent_all_zero(a + 256, 256) == 1 {
        return 4;
    }
    if ent_all_zero(a + 512, 88) == 1 {
        return 5;
    }
    if ent_same(a, b, n) == 1 {
        return 6;
    }
    if entropy.entropy_fill(0, 8) != entropy.entropy_err_arg() {
        return 7;
    }
    if entropy.entropy_fill(a, 0 - 1) != entropy.entropy_err_arg() {
        return 8;
    }
    if entropy.entropy_fill(a, entropy.entropy_max_len() + 1) != entropy.entropy_err_arg() {
        return 9;
    }
    if entropy.entropy_fill(a, 0) != entropy.entropy_ok() {
        return 10;
    }
    return 42;
}
"#;

/// A `getrandom` that fills every request with 0xAA, completes the 256-byte
/// chunks and fails the 88-byte tail with `TAIL_RC`, so the wipe must reach
/// the chunks that already succeeded.
fn getrandom_shim(tail_rc: &str) -> String {
    format!(
        "pub fn getrandom(buf: i64, buflen: i64, flags: i64) -> i64 {{\n    let mut i: i64 = 0;\n    while i < buflen {{\n        __mind_store_i8(buf + i, 170);\n        i = i + 1;\n    }}\n    if buflen == 256 {{\n        return buflen;\n    }}\n    return {tail_rc};\n}}\n"
    )
}

fn build_shim(dir: &Path, name: &str, tail_rc: &str) -> std::path::PathBuf {
    let src = common::scratch_src_in(dir, &format!("{name}.mind"), &getrandom_shim(tail_rc));
    common::emit_shared_in(dir, name, &src, &format!("{name}.so"), &[])
}

#[test]
fn entropy_fills_distinct_nonzero_buffers_and_fails_closed() {
    let target = "std_entropy_run";
    let Some(exe) = build_exe(target, ENTROPY_MAIN) else {
        return;
    };
    let limit = Duration::from_secs(60);

    // Two runs of the real getrandom: every check inside passes.
    assert_eq!(
        run(&exe, None, limit).0,
        42,
        "entropy_fill on the kernel CSPRNG"
    );
    assert_eq!(run(&exe, None, limit).0, 42, "a second run");

    let dir = common::scratch_dir(target);
    let fail = build_shim(&dir, "getrandom_fail", "0 - 1");
    let short = build_shim(&dir, "getrandom_short", "buflen - 1");
    assert_eq!(
        run(&exe, Some(&fail), limit).0,
        20,
        "getrandom error: entropy_err_os and every byte zeroed (21 = bytes left behind)"
    );
    assert_eq!(
        run(&exe, Some(&short), limit).0,
        30,
        "short getrandom: entropy_err_short and every byte zeroed (31 = bytes left behind)"
    );
}

// ─── std.dns: dns_resolve ────────────────────────────────────────────────────

/// `long_name(k)` builds a k-byte numeric literal, "000…0177.0.0.1", which the
/// C library reads as 127.0.0.1 (octal first part) without a lookup, so the
/// 253-byte bound is tested on both sides with no network.
const DNS_MAIN: &str = r#"import std.dns;
import std.string;

fn is_v4_loop(r: i64, i: i64) -> i64 {
    if dns.dns_family(r, i) != 4 {
        return 0;
    }
    if dns.dns_addr_byte(r, i, 0) != 127 {
        return 0;
    }
    if dns.dns_addr_byte(r, i, 1) != 0 {
        return 0;
    }
    if dns.dns_addr_byte(r, i, 2) != 0 {
        return 0;
    }
    if dns.dns_addr_byte(r, i, 3) != 1 {
        return 0;
    }
    return 1;
}

fn is_v6_loop(r: i64, i: i64) -> i64 {
    if dns.dns_family(r, i) != 6 {
        return 0;
    }
    let mut k: i64 = 0;
    while k < 15 {
        if dns.dns_addr_byte(r, i, k) != 0 {
            return 0;
        }
        k = k + 1;
    }
    if dns.dns_addr_byte(r, i, 15) != 1 {
        return 0;
    }
    return 1;
}

fn long_name(k: i64) -> i64 {
    let tail: String = "177.0.0.1";
    let p: i64 = __mind_alloc(k);
    let lead: i64 = k - 9;
    let mut i: i64 = 0;
    while i < lead {
        __mind_store_i8(p + i, 48);
        i = i + 1;
    }
    let mut j: i64 = 0;
    while j < 9 {
        __mind_store_i8(p + lead + j, string_get_byte(tail, j));
        j = j + 1;
    }
    return p;
}

fn main() -> i64 {
    let lh: String = "localhost";
    let r: i64 = dns.dns_resolve(string_addr(lh), string_len(lh));
    if dns.dns_status(r) != dns.dns_ok() {
        return 1;
    }
    let n: i64 = dns.dns_count(r);
    if n < 1 {
        return 2;
    }
    let mut i: i64 = 0;
    while i < n {
        if is_v4_loop(r, i) == 0 {
            if is_v6_loop(r, i) == 0 {
                return 3;
            }
        }
        i = i + 1;
    }
    if dns.dns_family(r, n) != 0 {
        return 4;
    }
    if dns.dns_addr_byte(r, 0, 16) != 0 - 1 {
        return 5;
    }

    let v4: String = "127.0.0.1";
    let r4: i64 = dns.dns_resolve(string_addr(v4), string_len(v4));
    if dns.dns_status(r4) != dns.dns_ok() {
        return 6;
    }
    if dns.dns_count(r4) != 1 {
        return 7;
    }
    if is_v4_loop(r4, 0) == 0 {
        return 8;
    }

    let v6: String = "::1";
    let r6: i64 = dns.dns_resolve(string_addr(v6), string_len(v6));
    if dns.dns_status(r6) != dns.dns_ok() {
        return 9;
    }
    if dns.dns_count(r6) != 1 {
        return 10;
    }
    if is_v6_loop(r6, 0) == 0 {
        return 11;
    }

    let re: i64 = dns.dns_resolve(string_addr(lh), 0);
    if dns.dns_status(re) != dns.dns_err_name() {
        return 12;
    }
    if dns.dns_count(re) != 0 {
        return 13;
    }
    if dns.dns_status(dns.dns_resolve(0, 9)) != dns.dns_err_name() {
        return 14;
    }

    let r253: i64 = dns.dns_resolve(long_name(253), 253);
    if dns.dns_status(r253) != dns.dns_ok() {
        return 15;
    }
    if is_v4_loop(r253, 0) == 0 {
        return 16;
    }
    let r254: i64 = dns.dns_resolve(long_name(254), 254);
    if dns.dns_status(r254) != dns.dns_err_name() {
        return 17;
    }
    if dns.dns_count(r254) != 0 {
        return 18;
    }

    let sp: String = "local host";
    if dns.dns_status(dns.dns_resolve(string_addr(sp), string_len(sp))) != dns.dns_err_name() {
        return 19;
    }
    let nul: String = "localhost\0";
    if dns.dns_status(dns.dns_resolve(string_addr(nul), string_len(nul))) != dns.dns_err_name() {
        return 20;
    }
    return 42;
}
"#;

#[test]
fn dns_resolves_localhost_and_refuses_bad_names() {
    let Some(exe) = build_exe("std_dns_resolve_run", DNS_MAIN) else {
        return;
    };
    let (code, _) = run(&exe, None, Duration::from_secs(60));
    assert_eq!(code, 42, "dns_resolve check {code} failed (see DNS_MAIN)");
}

// ─── std.dns: tcp_connect_host ───────────────────────────────────────────────

/// A listener on 127.0.0.1 only: "localhost" may list ::1 first, which refuses,
/// so the connect must move on to the next address. Out-of-range ports are
/// chosen so that, unchecked, their low 16 bits would reach the live listener.
const CONNECT_MAIN: &str = r#"import std.dns;
import std.net;
import std.string;

fn main() -> i64 {
    let any: String = "127.0.0.1\0";
    let l: i64 = net.tcp_listen(string_addr(any), 0);
    if l == 0 {
        return 1;
    }
    let port: i64 = net.tcp_listen_port(l);
    let lh: String = "localhost";
    let la: i64 = string_addr(lh);
    let ll: i64 = string_len(lh);
    let s: i64 = dns.tcp_connect_host(la, ll, port);
    if s == 0 {
        return 2;
    }
    let msg: String = "ping";
    if net.tcp_write(s, string_addr(msg), 4) != 4 {
        return 3;
    }
    let a: i64 = net.tcp_accept(l);
    if a == 0 {
        return 4;
    }
    let buf: i64 = __mind_alloc(8);
    if net.tcp_read(a, buf, 8) != 4 {
        return 5;
    }
    if __mind_load_i8(buf + 0) != 112 {
        return 6;
    }
    if __mind_load_i8(buf + 3) != 103 {
        return 7;
    }
    net.tcp_close(s);
    net.tcp_close(a);

    if dns.tcp_connect_host(la, ll, port + 65536) != 0 {
        return 8;
    }
    if dns.tcp_connect_host(la, ll, port - 65536) != 0 {
        return 9;
    }
    if dns.tcp_connect_host_timeout(la, ll, port, 0) != 0 {
        return 10;
    }
    if dns.tcp_connect_host_timeout(la, ll, port, dns.dns_max_timeout_ms() + 1) != 0 {
        return 11;
    }
    if dns.tcp_connect_host(la, 0, port) != 0 {
        return 12;
    }

    let l2: i64 = net.tcp_listen(string_addr(any), 0);
    if l2 == 0 {
        return 13;
    }
    let closed: i64 = net.tcp_listen_port(l2);
    net.tcp_listener_close(l2);
    if dns.tcp_connect_host(la, ll, closed) != 0 {
        return 14;
    }
    net.tcp_listener_close(l);
    return 42;
}
"#;

#[test]
fn tcp_connect_host_reaches_a_local_listener_and_refuses_bad_arguments() {
    let Some(exe) = build_exe("std_dns_connect_run", CONNECT_MAIN) else {
        return;
    };
    let (code, _) = run(&exe, None, Duration::from_secs(60));
    assert_eq!(
        code, 42,
        "tcp_connect_host check {code} failed (see CONNECT_MAIN)"
    );
}

/// The listener's accept queue holds one connection (backlog 0) and nobody
/// accepts, so the kernel drops the second client's SYN: without the timeout
/// that connect blocks for the SYN-retry period (minutes).
const TIMEOUT_MAIN: &str = r#"import std.dns;
import std.net;
import std.string;

extern "C" {
    unsafe fn listen(sockfd: i32, backlog: i32) -> i32;
}

fn main() -> i64 {
    let any: String = "127.0.0.1\0";
    let l: i64 = net.tcp_listen(string_addr(any), 0);
    if l == 0 {
        return 1;
    }
    let fd: i64 = net.tcp_listener_fd(l);
    let lr: i64 = listen(fd as i32, 0) as i64;
    if lr != 0 {
        return 2;
    }
    let port: i64 = net.tcp_listen_port(l);
    let ip: String = "127.0.0.1";
    let first: i64 = dns.tcp_connect_host_timeout(string_addr(ip), string_len(ip), port, 5000);
    if first == 0 {
        return 3;
    }
    let second: i64 = dns.tcp_connect_host_timeout(string_addr(ip), string_len(ip), port, 300);
    if second != 0 {
        return 4;
    }
    return 42;
}
"#;

#[test]
fn tcp_connect_host_gives_up_after_its_timeout() {
    let Some(exe) = build_exe("std_dns_timeout_run", TIMEOUT_MAIN) else {
        return;
    };
    let (code, took) = run(&exe, None, Duration::from_secs(60));
    assert_eq!(code, 42, "timeout check {code} failed (see TIMEOUT_MAIN)");
    assert!(
        took < Duration::from_secs(30),
        "a 300 ms connect timeout took {took:?}"
    );
}
