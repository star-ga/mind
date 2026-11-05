// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std.dns` reads the C library's `struct addrinfo` in the layout of the host
//! that `uname(2)` names, and refuses a host whose layout it does not implement.
//!
//! The host is faked at the extern boundary, as in `tests/std_entropy_dns_run.rs`:
//! a MIND shim compiled with `mindc --emit-shared` and loaded with `LD_PRELOAD`
//! defines `uname`, `getaddrinfo` and `freeaddrinfo`, so the executable's calls
//! land in the shim. No production code carries a test hook.
//!
//! * The Darwin shim builds macOS nodes: `ai_canonname` at +24 points at a decoy
//!   that reads as an IPv4 sockaddr for 0.0.0.0, `ai_addr` at +32, `sa_len` in
//!   byte 0 and `sa_family` in byte 1, `AF_INET6` = 30. A walk that read the
//!   Linux offsets, or took +24 for the address, fails a check below.
//! * The other-host shim names "FreeBSD" and answers with a valid Linux node, so
//!   only the refusal keeps the resolver and the connect from succeeding.

#![cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
mod std_entropy_dns_support;

use std::path::PathBuf;
use std::time::Duration;
use std_entropy_dns_support::{build_exe, run};

/// Shared shim helpers: zero a block, and the one node builder per layout.
const SHIM_COMMON: &str = r#"fn sh_zero(p: i64, n: i64) {
    let mut i: i64 = 0;
    while i < n {
        __mind_store_i8(p + i, 0);
        i = i + 1;
    }
}

// A macOS node. The address is b0.0.0.last (IPv4, alen 16) or b0::last with
// scope id 5 (IPv6, alen 28); salen / safam go to bytes 0 / 1 of the sockaddr.
fn mac_node(fam: i64, alen: i64, salen: i64, safam: i64, b0: i64, last: i64, next: i64) -> i64 {
    let nd: i64 = __mind_alloc(48);
    sh_zero(nd, 48);
    __mind_store_i32(nd + 4, fam);
    __mind_store_i32(nd + 8, 1);
    __mind_store_i32(nd + 16, alen);
    let decoy: i64 = __mind_alloc(28);
    sh_zero(decoy, 28);
    __mind_store_i8(decoy, 16);
    __mind_store_i8(decoy + 1, 2);
    __mind_store_i64(nd + 24, decoy);
    let sa: i64 = __mind_alloc(28);
    sh_zero(sa, 28);
    __mind_store_i8(sa, salen);
    __mind_store_i8(sa + 1, safam);
    if alen == 16 {
        __mind_store_i8(sa + 4, b0);
        __mind_store_i8(sa + 7, last);
    } else {
        __mind_store_i8(sa + 8, b0);
        __mind_store_i8(sa + 23, last);
        __mind_store_i32(sa + 24, 5);
    }
    __mind_store_i64(nd + 32, sa);
    __mind_store_i64(nd + 40, next);
    return nd;
}

// A Linux node for 127.0.0.1: ai_addr at +24, family u16 at +0.
fn linux_loop() -> i64 {
    let nd: i64 = __mind_alloc(48);
    sh_zero(nd, 48);
    __mind_store_i32(nd + 4, 2);
    __mind_store_i32(nd + 8, 1);
    __mind_store_i32(nd + 16, 16);
    let sa: i64 = __mind_alloc(28);
    sh_zero(sa, 28);
    __mind_store_i16(sa, 2);
    __mind_store_i8(sa + 4, 127);
    __mind_store_i8(sa + 7, 1);
    __mind_store_i64(nd + 24, sa);
    return nd;
}

pub fn freeaddrinfo(res: i64) -> i64 {
    return 0;
}

// Write `n` bytes of `word` (little-endian) and a NUL at `buf`.
fn sh_name(buf: i64, word: i64, n: i64) {
    let mut k: i64 = 0;
    while k < n {
        __mind_store_i8(buf + k, (word >> (8 * k)) & 255);
        k = k + 1;
    }
    __mind_store_i8(buf + n, 0);
}
"#;

/// macOS: `uname` says "Darwin"; the name's first byte picks the list.
/// `a` 10.0.0.7, `b` IPv6 32::1 scope 5, `c` both in that order, `d` an IPv6
/// sockaddr whose sa_len (16) disagrees with ai_addrlen (28), `e` the Linux
/// AF_INET6 value 10 as the family, `q` 127.0.0.1.
const DARWIN_SHIM: &str = r#"
pub fn uname(buf: i64) -> i64 {
    sh_name(buf, 121399254606148, 6);
    return 0;
}

pub fn getaddrinfo(node: i64, service: i64, hints: i64, res: i64) -> i64 {
    let b0: i64 = __mind_load_i8(node);
    let mut head: i64 = 0;
    if b0 == 97 {
        head = mac_node(2, 16, 16, 2, 10, 7, 0);
    }
    if b0 == 98 {
        head = mac_node(30, 28, 28, 30, 32, 1, 0);
    }
    if b0 == 99 {
        head = mac_node(2, 16, 16, 2, 10, 7, mac_node(30, 28, 28, 30, 32, 1, 0));
    }
    if b0 == 100 {
        head = mac_node(30, 28, 16, 30, 32, 1, 0);
    }
    if b0 == 101 {
        head = mac_node(10, 28, 28, 10, 32, 1, 0);
    }
    if b0 == 113 {
        head = mac_node(2, 16, 16, 2, 127, 1, 0);
    }
    __mind_store_i64(res, head);
    return 0;
}
"#;

/// Another host: `uname` says "FreeBSD"; every name resolves to a valid Linux
/// node for 127.0.0.1.
const OTHER_SHIM: &str = r#"
pub fn uname(buf: i64) -> i64 {
    sh_name(buf, 19231843050418758, 7);
    return 0;
}

pub fn getaddrinfo(node: i64, service: i64, hints: i64, res: i64) -> i64 {
    __mind_store_i64(res, linux_loop());
    return 0;
}
"#;

/// Compile `SHIM_COMMON` + `body` in `target`'s scratch directory as `name.so`.
fn shim(target: &str, name: &str, body: &str) -> PathBuf {
    let dir = common::scratch_dir(target);
    let text = format!("{SHIM_COMMON}{body}");
    let src = common::scratch_src_in(&dir, &format!("{name}.mind"), &text);
    common::emit_shared_in(&dir, name, &src, &format!("{name}.so"), &[])
}

/// Exit 42 when every check passes; any other value names the failed check.
const DARWIN_MAIN: &str = r#"import std.dns;
import std.net;
import std.string;

fn res(s: String) -> i64 {
    return dns.dns_resolve(string_addr(s), string_len(s));
}

fn main() -> i64 {
    let a: i64 = res("a");
    if dns.dns_status(a) != dns.dns_ok() {
        return 1;
    }
    if dns.dns_count(a) != 1 {
        return 2;
    }
    if dns.dns_family(a, 0) != 4 {
        return 3;
    }
    if dns.dns_addr_byte(a, 0, 0) != 10 {
        return 4;
    }
    if dns.dns_addr_byte(a, 0, 3) != 7 {
        return 5;
    }
    let b: i64 = res("b");
    if dns.dns_status(b) != dns.dns_ok() {
        return 6;
    }
    if dns.dns_family(b, 0) != 6 {
        return 7;
    }
    if dns.dns_addr_byte(b, 0, 0) != 32 {
        return 8;
    }
    if dns.dns_addr_byte(b, 0, 15) != 1 {
        return 9;
    }
    if dns.dns_scope_id(b, 0) != 5 {
        return 10;
    }
    let c: i64 = res("c");
    if dns.dns_count(c) != 2 {
        return 11;
    }
    if dns.dns_family(c, 0) != 4 {
        return 12;
    }
    if dns.dns_family(c, 1) != 6 {
        return 13;
    }
    if dns.dns_status(res("d")) != dns.dns_err_entry() {
        return 14;
    }
    if dns.dns_status(res("e")) != dns.dns_err_entry() {
        return 15;
    }
    if dns.dns_count(res("e")) != 0 {
        return 16;
    }
    // A live listener the resolved address reaches: only the Linux-only
    // refusal keeps tcp_connect_host from connecting to it.
    let any: String = "127.0.0.1\0";
    let l: i64 = net.tcp_listen(string_addr(any), 0);
    if l == 0 {
        return 17;
    }
    let port: i64 = net.tcp_listen_port(l);
    let q: String = "q";
    if dns.dns_addr_byte(res("q"), 0, 0) != 127 {
        return 18;
    }
    if dns.tcp_connect_host(string_addr(q), string_len(q), port) != 0 {
        return 19;
    }
    net.tcp_listener_close(l);
    return 42;
}
"#;

#[test]
fn dns_reads_the_darwin_addrinfo_layout_and_connects_only_on_linux() {
    let target = "std_dns_layout_darwin_run";
    let Some(exe) = build_exe(target, DARWIN_MAIN) else {
        return;
    };
    let so = shim(target, "darwin_shim", DARWIN_SHIM);
    let (code, _) = run(&exe, Some(&so), Duration::from_secs(60));
    assert_eq!(
        code, 42,
        "Darwin layout check {code} failed (see DARWIN_MAIN)"
    );
}

const OTHER_MAIN: &str = r#"import std.dns;
import std.net;
import std.string;

fn main() -> i64 {
    let q: String = "q";
    let r: i64 = dns.dns_resolve(string_addr(q), string_len(q));
    if dns.dns_status(r) != dns.dns_err_platform() {
        return 1;
    }
    if dns.dns_count(r) != 0 {
        return 2;
    }
    if dns.dns_family(r, 0) != 0 {
        return 3;
    }
    let any: String = "127.0.0.1\0";
    let l: i64 = net.tcp_listen(string_addr(any), 0);
    if l == 0 {
        return 4;
    }
    let port: i64 = net.tcp_listen_port(l);
    if dns.tcp_connect_host(string_addr(q), string_len(q), port) != 0 {
        return 5;
    }
    net.tcp_listener_close(l);
    // The new code is distinct from every other status.
    let mut c: i64 = 0;
    while c >= 0 - 17 {
        if c == dns.dns_err_platform() {
            return 6;
        }
        c = c - 1;
    }
    return 42;
}
"#;

#[test]
fn dns_refuses_a_host_whose_layout_it_does_not_implement() {
    let target = "std_dns_layout_other_run";
    let Some(exe) = build_exe(target, OTHER_MAIN) else {
        return;
    };
    let so = shim(target, "other_shim", OTHER_SHIM);
    let (code, _) = run(&exe, Some(&so), Duration::from_secs(60));
    assert_eq!(code, 42, "other-host check {code} failed (see OTHER_MAIN)");
}
