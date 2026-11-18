// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Callers of std functions that read the world (`std.entropy`, `std.dns`,
//! `std.https`, `std.https_probe`, `std.https_io`, and the CPU affinity calls of
//! `std.cpu_affinity` and `std.iouring`).
//!
//! Companion to `determinism_veto_control`. A std function's body is compiled
//! into a separate object, so the caller's IR holds only the call, and the
//! classifier must know the function by name. That case resolves a std import,
//! so it needs `cross-module-imports`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

mod common;

use std::path::Path;
use std::process::Command;

/// `(module, call, the function the refusal must name)`: a CALLER of a std
/// function whose body reads the world. The body is compiled into a separate
/// substrate object, so the caller's IR — what the veto, the evidence label and
/// `mindc verify` classify — holds only the call.
const STD_WORLD_CALLERS: &[(&str, &str, &str)] = &[
    ("entropy", "entropy.entropy_fill(p, 8)", "entropy_fill"),
    ("dns", "dns.dns_resolve(p, 9)", "dns_resolve"),
    ("dns", "dns.tcp_connect_host(p, 9, 80)", "tcp_connect_host"),
    (
        "dns",
        "dns.tcp_connect_host_timeout(p, 9, 80, 100)",
        "tcp_connect_host_timeout",
    ),
    ("dns", "dns.dns_tcp_socket(2)", "dns_tcp_socket"),
    ("dns", "dns.dns_connect_fd(p, p, 16)", "dns_connect_fd"),
    ("dns", "dns.dns_clock_gettime(1, p)", "dns_clock_gettime"),
    // The `::` and full-path spellings name the same std function.
    (
        "entropy",
        "std::entropy::entropy_fill(p, 8)",
        "entropy_fill",
    ),
    ("dns", "std.dns.dns_resolve(p, 9)", "dns_resolve"),
    // A bare call of an imported std function is the same call: lowering
    // keeps the bare name either way (the review's cases A and B).
    ("entropy", "entropy_fill(p, 8)", "entropy_fill"),
    ("dns", "dns_resolve(p, 9)", "dns_resolve"),
    ("dns", "tcp_connect_host(p, 9, 80)", "tcp_connect_host"),
    (
        "dns",
        "tcp_connect_host_timeout(p, 9, 80, 100)",
        "tcp_connect_host_timeout",
    ),
    ("https", "https.https_get(p, 9)", "https_get"),
    ("https", "https.https_get_ca(p, 9, p)", "https_get_ca"),
    ("https", "https.https_post(p, 9, p, 1, p, 1)", "https_post"),
    (
        "https",
        "https.https_post_ca(p, 9, p, 1, p, 1, p)",
        "https_post_ca",
    ),
    (
        "https",
        "https.https_get_with_headers(p, 9, p, 1, p)",
        "https_get_with_headers",
    ),
    (
        "https",
        "https.https_post_with_headers(p, 9, p, 1, p, 1, p, 1, p)",
        "https_post_with_headers",
    ),
    (
        "https_probe",
        "https_probe.https_probe(p, 9, 100)",
        "https_probe",
    ),
    ("https_io", "https_io.hio_now_unix()", "hio_now_unix"),
    ("https_io", "https_io.hio_now_mono_ms()", "hio_now_mono_ms"),
    (
        "https_io",
        "https_io.hio_remaining_ms(p)",
        "hio_remaining_ms",
    ),
    ("https_io", "https_io.hio_arm(p, 9)", "hio_arm"),
    ("https_io", "https_io.hio_io_failure(p)", "hio_io_failure"),
    // The CPU affinity calls make a real affinity system call: the std.cpu_affinity
    // functions, and the forwards of std.iouring. Qualified, then bare.
    (
        "cpu_affinity",
        "cpu_affinity.cpu_affinity_pin(p)",
        "cpu_affinity_pin",
    ),
    (
        "cpu_affinity",
        "cpu_affinity.cpu_affinity_mask()",
        "cpu_affinity_mask",
    ),
    (
        "cpu_affinity",
        "cpu_affinity.cpu_affinity_has(p)",
        "cpu_affinity_has",
    ),
    (
        "cpu_affinity",
        "cpu_affinity.cpu_affinity_pin_demo(p)",
        "cpu_affinity_pin_demo",
    ),
    (
        "iouring",
        "iouring.io_reactor_pin_cpu(p)",
        "io_reactor_pin_cpu",
    ),
    (
        "iouring",
        "iouring.io_reactor_affinity_mask()",
        "io_reactor_affinity_mask",
    ),
    (
        "iouring",
        "iouring.io_reactor_affinity_has(p)",
        "io_reactor_affinity_has",
    ),
    (
        "iouring",
        "iouring.io_reactor_pin_demo(p)",
        "io_reactor_pin_demo",
    ),
    ("cpu_affinity", "cpu_affinity_pin(p)", "cpu_affinity_pin"),
    ("cpu_affinity", "cpu_affinity_mask()", "cpu_affinity_mask"),
    ("cpu_affinity", "cpu_affinity_has(p)", "cpu_affinity_has"),
    (
        "cpu_affinity",
        "cpu_affinity_pin_demo(p)",
        "cpu_affinity_pin_demo",
    ),
    ("iouring", "io_reactor_pin_cpu(p)", "io_reactor_pin_cpu"),
    (
        "iouring",
        "io_reactor_affinity_mask()",
        "io_reactor_affinity_mask",
    ),
    (
        "iouring",
        "io_reactor_affinity_has(p)",
        "io_reactor_affinity_has",
    ),
    ("iouring", "io_reactor_pin_demo(p)", "io_reactor_pin_demo"),
    // The wrapper module std.cpu_affinity reaches the OS through, and the evidence
    // wrapper of std.os_evidence that runs cpu_affinity_pin: qualified, then bare.
    (
        "cpu_affinity_sys",
        "cpu_affinity_sys.cpu_affinity_sys_set(8, p)",
        "cpu_affinity_sys_set",
    ),
    (
        "cpu_affinity_sys",
        "cpu_affinity_sys.cpu_affinity_sys_get(8, p)",
        "cpu_affinity_sys_get",
    ),
    (
        "cpu_affinity_sys",
        "cpu_affinity_sys.cpu_affinity_sys_errno()",
        "cpu_affinity_sys_errno",
    ),
    (
        "os_evidence",
        "os_evidence.cpu_affinity_pin_ev(p, 0)",
        "cpu_affinity_pin_ev",
    ),
    (
        "cpu_affinity_sys",
        "cpu_affinity_sys_set(8, p)",
        "cpu_affinity_sys_set",
    ),
    (
        "cpu_affinity_sys",
        "cpu_affinity_sys_get(8, p)",
        "cpu_affinity_sys_get",
    ),
    (
        "cpu_affinity_sys",
        "cpu_affinity_sys_errno()",
        "cpu_affinity_sys_errno",
    ),
    (
        "os_evidence",
        "cpu_affinity_pin_ev(p, 0)",
        "cpu_affinity_pin_ev",
    ),
    // A local binding spelled like one row beside a real call of another row of
    // the same module: the binding hides nothing, the call is still the world read.
    (
        "dns",
        "let dns_connect_fd: i64 = p;\n    dns_connect_fd + dns_resolve(p, 9)",
        "dns_resolve",
    ),
    (
        "iouring",
        "let io_reactor_pin_cpu: i64 = p;\n    io_reactor_pin_cpu + io_reactor_affinity_has(p)",
        "io_reactor_affinity_has",
    ),
    (
        "cpu_affinity",
        "let cpu_affinity_has: i64 = p;\n    cpu_affinity_has + cpu_affinity_pin(p)",
        "cpu_affinity_pin",
    ),
    (
        "cpu_affinity_sys",
        "let cpu_affinity_sys_set: i64 = p;\n    cpu_affinity_sys_set + cpu_affinity_sys_errno()",
        "cpu_affinity_sys_errno",
    ),
    (
        "os_evidence",
        "let procgroup_spawn_ev: i64 = p;\n    procgroup_spawn_ev + cpu_affinity_pin_ev(p, 0)",
        "cpu_affinity_pin_ev",
    ),
];

/// Pure functions of the same modules: the positive control.
const STD_PURE_CALLERS: &[(&str, &str)] = &[
    ("entropy", "entropy.entropy_max_len()"),
    ("dns", "dns.dns_status(p)"),
    ("entropy", "entropy_max_len()"),
    ("dns", "dns_status(p)"),
    ("https", "https.https_deadline_s()"),
    ("https_http1", "https_http1.https_error_class(p)"),
    ("https_probe", "https_probe.https_probe_ok()"),
    ("https_io", "https_io.hio_deadline_s()"),
    ("https_record", "https_record.tls13rl_pend_ready(p)"),
    // A function of std.iouring that reads no World name stays deterministic.
    ("iouring", "iouring.iouring_cqe_f_more()"),
    ("iouring", "iouring_cqe_f_more()"),
    // A local binding spelled like a row of the imported module is a value, not a
    // call (a function name read as a value is a type error, E2037): no call, no
    // world read, with the std module imported.
    (
        "entropy",
        "let entropy_fill: i64 = p;\n    entropy_fill + 1",
    ),
    ("dns", "let dns_resolve: i64 = p;\n    dns_resolve + 1"),
    (
        "iouring",
        "let io_reactor_pin_cpu: i64 = p;\n    io_reactor_pin_cpu + 1",
    ),
    (
        "cpu_affinity",
        "let cpu_affinity_has: i64 = p;\n    cpu_affinity_has + 1",
    ),
    (
        "cpu_affinity_sys",
        "let cpu_affinity_sys_get: i64 = p;\n    cpu_affinity_sys_get + 1",
    ),
    (
        "os_evidence",
        "let cpu_affinity_pin_ev: i64 = p;\n    cpu_affinity_pin_ev + 1",
    ),
    // A function of std.os_evidence that reaches no World row stays deterministic.
    ("os_evidence", "os_evidence.procgroup_record_verdicts(p, 0)"),
];

/// Write `import std.<module>; pub fn f(p: i64) -> i64 { <call> }` to
/// `dir/<tag>.mind`; returns `(source, evidence path)`.
fn std_caller(dir: &Path, tag: &str, module: &str, call: &str) -> (String, String) {
    let text = format!("import std.{module};\n\npub fn f(p: i64) -> i64 {{\n    {call}\n}}\n");
    let src = common::scratch_src_in(dir, &format!("{tag}.mind"), &text);
    let ev = dir.join(format!("{tag}.ev"));
    (
        src.to_string_lossy().into_owned(),
        ev.to_string_lossy().into_owned(),
    )
}

/// `mindc <args>`: `(success, stdout + stderr)`.
fn mindc(args: &[&str]) -> (bool, String) {
    let out = Command::new(common::mindc_bin())
        .args(args)
        .output()
        .expect("spawn mindc");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

/// Copy the evidence at `src` to `dst` with only its stored determinism label
/// rewritten from `nondeterministic` to `deterministic` (its length byte too).
fn forge_deterministic(src: &Path, dst: &Path) {
    let bytes = std::fs::read(src).expect("read evidence");
    let marker = b"nondeterministic";
    let at = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .expect("stored label");
    assert!(at > 0 && bytes[at - 1] == marker.len() as u8);
    let parts: [&[u8]; 4] = [
        &bytes[..at - 1],
        &[b"deterministic".len() as u8],
        b"deterministic",
        &bytes[at + marker.len()..],
    ];
    std::fs::write(dst, parts.concat()).expect("write forged evidence");
}

/// A program that calls `entropy.entropy_fill` or a `std.dns` resolver must be
/// treated like one that calls `getrandom` / `getaddrinfo` itself: refused
/// without `--allow-nondeterministic`, and with it attested `nondeterministic`,
/// which `mindc verify --require-deterministic` rejects. Before these were
/// classified, the caller emitted, attested `deterministic` and passed that
/// verify. The pure accessors run the same three steps and must stay
/// `deterministic`, so a refusal for an unrelated reason cannot pass as a veto.
#[test]
fn callers_of_std_world_functions_are_refused_and_attested_nondeterministic() {
    let dir = common::scratch_dir("determinism_std_callers");
    let mut ran = 0usize;
    for (i, (module, call, offender)) in STD_WORLD_CALLERS.iter().enumerate() {
        let (src, ev) = std_caller(&dir, &format!("world{i}"), module, call);
        let (ok, text) = mindc(&[&src, "--emit-evidence", &ev]);
        assert!(
            !ok && text.contains("error[determinism]") && text.contains(&format!("{offender}()`")),
            "a caller of `{call}` was not refused naming `{offender}()`\n{text}"
        );
        let (ok, text) = mindc(&[&src, "--emit-evidence", &ev, "--allow-nondeterministic"]);
        assert!(
            ok,
            "`{call}` with --allow-nondeterministic did not emit\n{text}"
        );
        let (ok, text) = mindc(&["verify", "--require-deterministic", &ev]);
        assert!(
            !ok && text.contains("determinism:      nondeterministic"),
            "the evidence for a caller of `{call}` passed --require-deterministic\n{text}"
        );
        let forged = format!("{ev}.forged");
        forge_deterministic(Path::new(&ev), Path::new(&forged));
        let (ok, text) = mindc(&["verify", "--require-deterministic", &forged]);
        assert!(
            !ok && text.contains("TAMPERED"),
            "`{call}` relabelled deterministic was not refused as TAMPERED\n{text}"
        );
        ran += 1;
    }
    for (i, (module, call)) in STD_PURE_CALLERS.iter().enumerate() {
        let (src, ev) = std_caller(&dir, &format!("pure{i}"), module, call);
        let (ok, text) = mindc(&[&src, "--emit-evidence", &ev]);
        assert!(ok, "a caller of the pure `{call}` was refused\n{text}");
        let (ok, text) = mindc(&["verify", "--require-deterministic", &ev]);
        assert!(
            ok && text.contains("determinism:      deterministic"),
            "the evidence for a caller of the pure `{call}` is not deterministic\n{text}"
        );
        ran += 1;
    }
    println!("determinism_veto_control(std callers): ran={ran} fail=0");
    assert_eq!(
        ran,
        STD_WORLD_CALLERS.len() + STD_PURE_CALLERS.len(),
        "vacuous run: ran={ran}"
    );
}
