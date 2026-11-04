// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Shared harness for `tests/std_entropy_dns_run.rs`: build a one-file project
//! with `mindc build`, run the executable with an optional `LD_PRELOAD`, and
//! the preloaded resolver shim. Split out so the test file stays under the
//! 800-line ceiling `tests/module_size_ratchet.rs` pins. This is a
//! `tests/<dir>/mod.rs` module, not a test target.

use std::path::{Path, PathBuf};
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
    Command::new(crate::common::mindc_bin())
        .arg("build")
        .current_dir(dir)
        .output()
        .expect("run mindc build")
}

/// Build `main` in a scratch project; `None` when routed through the
/// capability-skip gate.
pub fn build_exe(target: &str, main: &str) -> Option<PathBuf> {
    let dir = crate::common::scratch_dir(target);
    let out = build(&dir, main);
    if !crate::common::gate::compiled(target, &out) {
        return None;
    }
    Some(dir.join("target").join("debug").join("ed"))
}

/// Run `exe` (optionally with `LD_PRELOAD`), killing it after `limit`.
/// Returns the exit code and the elapsed time.
pub fn run(exe: &Path, preload: Option<&Path>, limit: Duration) -> (i32, Duration) {
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

/// A `getaddrinfo` / `freeaddrinfo` pair, preloaded, whose answer is chosen by
/// the first byte of the name: `eNN` returns -NN, `z` success with no list,
/// `k` 32 IPv4 nodes (10.0.0.0 .. 10.0.0.31), `o` 33 nodes, `f` family 99,
/// `l` an IPv4 node with a 28-byte address length, `w` an IPv6 node with a
/// 16-byte one, `n` no address, `s` a
/// sockaddr family that disagrees with ai_family, `x` one good node then a bad one,
/// `m` the loopback list 127.0.0.2, 127.0.0.4, 127.0.0.1 in that order, `p`
/// the list 127.0.0.4, 127.0.0.1.
pub const GAI_SHIM: &str = r#"fn sh_zero(p: i64, n: i64) {
    let mut i: i64 = 0;
    while i < n {
        __mind_store_i8(p + i, 0);
        i = i + 1;
    }
}

fn sh_node(fam: i64, safam: i64, alen: i64, with_addr: i64, octet: i64, next: i64) -> i64 {
    let nd: i64 = __mind_alloc(48);
    sh_zero(nd, 48);
    __mind_store_i32(nd + 4, fam);
    __mind_store_i32(nd + 8, 1);
    __mind_store_i32(nd + 16, alen);
    if with_addr == 1 {
        let sa: i64 = __mind_alloc(28);
        sh_zero(sa, 28);
        __mind_store_i16(sa, safam);
        __mind_store_i8(sa + 4, 10);
        __mind_store_i8(sa + 7, octet);
        __mind_store_i64(nd + 24, sa);
    }
    __mind_store_i64(nd + 40, next);
    return nd;
}

fn sh_loop(octet: i64, next: i64) -> i64 {
    let nd: i64 = sh_node(2, 2, 16, 1, octet, next);
    __mind_store_i8(__mind_load_i64(nd + 24) + 4, 127);
    return nd;
}

fn sh_chain(n: i64) -> i64 {
    let mut head: i64 = 0;
    let mut i: i64 = n - 1;
    while i >= 0 {
        head = sh_node(2, 2, 16, 1, i, head);
        i = i - 1;
    }
    return head;
}

pub fn getaddrinfo(node: i64, service: i64, hints: i64, res: i64) -> i64 {
    let b0: i64 = __mind_load_i8(node);
    __mind_store_i64(res, 0);
    if b0 == 101 {
        let d1: i64 = __mind_load_i8(node + 1) - 48;
        let d2: i64 = __mind_load_i8(node + 2) - 48;
        return 0 - (d1 * 10 + d2);
    }
    let mut head: i64 = 0;
    if b0 == 107 {
        head = sh_chain(32);
    }
    if b0 == 111 {
        head = sh_chain(33);
    }
    if b0 == 102 {
        head = sh_node(99, 99, 16, 1, 1, 0);
    }
    if b0 == 108 {
        head = sh_node(2, 2, 28, 1, 1, 0);
    }
    if b0 == 119 {
        head = sh_node(10, 10, 16, 1, 1, 0);
    }
    if b0 == 110 {
        head = sh_node(2, 2, 16, 0, 1, 0);
    }
    if b0 == 115 {
        head = sh_node(2, 10, 16, 1, 1, 0);
    }
    if b0 == 120 {
        head = sh_node(2, 2, 16, 1, 1, sh_node(99, 99, 16, 1, 2, 0));
    }
    if b0 == 109 {
        head = sh_loop(2, sh_loop(4, sh_loop(1, 0)));
    }
    if b0 == 112 {
        head = sh_loop(4, sh_loop(1, 0));
    }
    __mind_store_i64(res, head);
    return 0;
}

pub fn freeaddrinfo(res: i64) -> i64 {
    return 0;
}
"#;

/// Compile [`GAI_SHIM`] in `target`'s scratch directory; returns the `.so`.
pub fn gai_shim(target: &str) -> PathBuf {
    let dir = crate::common::scratch_dir(target);
    let src = crate::common::scratch_src_in(&dir, "gai_shim.mind", GAI_SHIM);
    crate::common::emit_shared_in(&dir, "gai_shim", &src, "gai_shim.so", &[])
}
