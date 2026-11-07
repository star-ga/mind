// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std.dns` releases every per-call allocation on every path.
//!
//! A built executable calls `tcp_connect_host` 300000 times against a closed
//! loopback port: each call resolves the name (uname, the getaddrinfo inputs,
//! the result record), builds a socket address and sets the send timeout
//! before the connect is refused, so the whole error path runs each time. The
//! process runs under `ulimit -v` 200 MB. A call that keeps even its
//! 1280-byte `uname` buffer would need 380 MB and the executable would die;
//! with every allocation released it exits 0.
//!
//! The control is the same loop with one deliberately leaked 4096-byte block
//! per iteration: it must die under the same limit, so a limit that never
//! binds cannot make this test pass.

#![cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
mod std_entropy_dns_support;

use std::path::Path;
use std::process::{Command, Stdio};
use std_entropy_dns_support::build_exe;

const ITERATIONS: i64 = 300_000;
const LIMIT_KB: i64 = 200_000;

fn program(leak_bytes: i64) -> String {
    format!(
        r#"import std.dns;

fn main() -> i64 {{
    let name: i64 = __mind_alloc(16);
    __mind_store_i8(name + 0, 49);
    __mind_store_i8(name + 1, 50);
    __mind_store_i8(name + 2, 55);
    __mind_store_i8(name + 3, 46);
    __mind_store_i8(name + 4, 48);
    __mind_store_i8(name + 5, 46);
    __mind_store_i8(name + 6, 48);
    __mind_store_i8(name + 7, 46);
    __mind_store_i8(name + 8, 49);
    let mut i: i64 = 0;
    let mut connected: i64 = 0;
    while i < {ITERATIONS} {{
        let h: i64 = tcp_connect_host(name, 9, 1);
        if h != 0 {{
            connected = connected + 1;
        }}
        if {leak_bytes} > 0 {{
            let leaked: i64 = __mind_alloc({leak_bytes});
            __mind_store_i8(leaked, 1);
        }}
        i = i + 1;
    }}
    if connected > 0 {{
        return 1;
    }}
    0
}}
"#
    )
}

/// Exit code of `exe` under the address-space limit; `None` when a signal
/// ended it (the allocator returned null and the program faulted).
fn run_limited(exe: &Path) -> Option<i32> {
    Command::new("sh")
        .arg("-c")
        .arg(format!("ulimit -v {LIMIT_KB}; exec \"$0\""))
        .arg(exe)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("spawn the limited executable")
        .code()
}

#[test]
fn tcp_connect_host_releases_its_allocations_on_the_error_path() {
    let Some(exe) = build_exe("std_dns_free_run", &program(0)) else {
        return;
    };
    assert_eq!(
        run_limited(&exe),
        Some(0),
        "300000 refused connects under {LIMIT_KB} KB must finish: a per-call allocation is leaked"
    );
    let Some(control) = build_exe("std_dns_free_run_control", &program(4096)) else {
        return;
    };
    assert_ne!(
        run_limited(&control),
        Some(0),
        "the control leaks 4096 bytes per call and must die under the limit"
    );
}
