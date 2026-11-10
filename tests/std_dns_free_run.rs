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
//!
//! `dns_read_path` (the CA bundle read of std.https_io) is bounded by its
//! buffer and closes its descriptor: 4000 reads finish under `ulimit -n 64`,
//! and the control (open without close) runs out of descriptors there.

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

/// Bytes of the file `dns_read_path` reads, and the descriptor limit it runs
/// under: 4000 reads under `ulimit -n 64` finish only if each read closes its
/// descriptor.
const FILE_LEN: usize = 1000;
const READS: i64 = 4000;
const FD_LIMIT: i64 = 64;

/// MIND statements that store the NUL-terminated `bytes` at `var`.
fn stores(var: &str, bytes: &[u8]) -> String {
    let mut s = format!("    let {var}: i64 = __mind_alloc({});\n", bytes.len() + 1);
    for (k, b) in bytes.iter().enumerate() {
        s.push_str(&format!("    __mind_store_i8({var} + {k}, {b});\n"));
    }
    s.push_str(&format!(
        "    __mind_store_i8({var} + {}, 0);\n",
        bytes.len()
    ));
    s
}

/// Reads `path` through `dns_read_path`, or (control) opens it `READS` times
/// with `__mind_open` and never closes it. Exit 0 only when every check holds.
fn read_path_program(path: &Path, leak_control: bool) -> String {
    let p = path.as_os_str().as_encoded_bytes();
    let mut missing = p.to_vec();
    missing.extend_from_slice(b".missing");
    let body = if leak_control {
        format!(
            "    let mut i: i64 = 0;\n    while i < {READS} {{\n        if __mind_open(p) < 0 {{\n            return 7;\n        }}\n        i = i + 1;\n    }}\n    0\n"
        )
    } else {
        format!(
            r#"    let buf: i64 = __mind_alloc(2048);
    if dns_read_path(q, buf, 2048) != 0 - 1 {{
        return 2;
    }}
    if dns_read_path(p, buf, 999) != 999 {{
        return 3;
    }}
    if dns_read_path(p, buf, {FILE_LEN}) != {FILE_LEN} {{
        return 4;
    }}
    if dns_read_path(p, buf, 2048) != {FILE_LEN} {{
        return 5;
    }}
    if __mind_load_i8(buf + 999) != {last} {{
        return 6;
    }}
    let mut i: i64 = 0;
    while i < {READS} {{
        if dns_read_path(p, buf, 2048) != {FILE_LEN} {{
            return 8;
        }}
        i = i + 1;
    }}
    0
"#,
            last = 999 % 90 + 32
        )
    };
    format!(
        "import std.dns;\n\nfn main() -> i64 {{\n{}{}{}}}\n",
        stores("p", p),
        stores("q", &missing),
        body
    )
}

/// Exit code of `exe` under the descriptor limit; `None` on a signal.
fn run_fd_limited(exe: &Path) -> Option<i32> {
    Command::new("sh")
        .arg("-c")
        .arg(format!("ulimit -n {FD_LIMIT}; exec \"$0\""))
        .arg(exe)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("spawn the fd-limited executable")
        .code()
}

#[test]
fn dns_read_path_bounds_the_read_and_closes_its_descriptor() {
    let dir = common::scratch_dir("std_dns_read_path_file");
    let file = dir.join("bundle.bin");
    let bytes: Vec<u8> = (0..FILE_LEN).map(|k| (k % 90 + 32) as u8).collect();
    std::fs::write(&file, &bytes).expect("write the read_path fixture");
    let Some(exe) = build_exe("std_dns_read_path", &read_path_program(&file, false)) else {
        return;
    };
    assert_eq!(
        run_fd_limited(&exe),
        Some(0),
        "dns_read_path: exit 2 = a missing file is not -1, 3/4/5 = the count is not bounded by cap, \
         6 = wrong bytes, 8 = a read failed under ulimit -n {FD_LIMIT} (a leaked descriptor)"
    );
    let Some(control) = build_exe("std_dns_read_path_control", &read_path_program(&file, true))
    else {
        return;
    };
    assert_eq!(
        run_fd_limited(&control),
        Some(7),
        "the control never closes and must run out of descriptors under ulimit -n {FD_LIMIT}"
    );
}
