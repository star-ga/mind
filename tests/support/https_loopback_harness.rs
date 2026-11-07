// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The loopback TLS 1.3 server harness shared by `tests/std_https_loopback.rs`
//! and `tests/std_https_network_safe.rs`: MIND fixture functions, the server
//! and client sources, the `mindc build` of a one-file project, and the server
//! process.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

pub const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/https_loopback");

/// The std modules the server and the clients import.
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
    "process",
];

/// A MIND function `name(dst) -> i64` that writes `bytes` at `dst` and returns
/// their count: whole little-endian words, then single bytes.
pub fn fixture_fn(name: &str, bytes: &[u8]) -> String {
    let words = bytes.len() / 8;
    let mut lines = Vec::with_capacity(words + 8);
    for w in 0..words {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[w * 8..w * 8 + 8]);
        let v = i64::from_le_bytes(word);
        let value = if v < 0 {
            format!("0 - {}", v.unsigned_abs())
        } else {
            v.to_string()
        };
        lines.push(format!("    __mind_store_i64(dst + {}, {value});", w * 8));
    }
    for (i, b) in bytes.iter().enumerate().skip(words * 8) {
        lines.push(format!("    __mind_store_i8(dst + {i}, {b});"));
    }
    format!(
        "fn {name}(dst: i64) -> i64 {{\n{}\n    return {};\n}}\n\n",
        lines.join("\n"),
        bytes.len()
    )
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex digit"))
        .collect()
}

/// `n=` / `d=` of the test-only server key.
fn key_field(field: &str) -> Vec<u8> {
    let text = std::fs::read_to_string(format!("{FIX}/server_key_test_only.txt"))
        .expect("read the test-only server key");
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{field}=")))
        .unwrap_or_else(|| panic!("server key has no {field}="));
    let bytes = hex_bytes(line.trim());
    assert_eq!(bytes.len(), 256, "{field} must be 256 bytes");
    bytes
}

fn der(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIX}/{name}.der")).unwrap_or_else(|e| panic!("read {name}.der: {e}"))
}

/// A NUL-terminated absolute path under the fixture directory.
pub fn path_bytes(file: &str) -> Vec<u8> {
    let mut b = format!("{FIX}/{file}").into_bytes();
    b.push(0);
    b
}

/// The loopback server, as fixture files appended in this order: the listener
/// and case list, the RSA-2048 signer, the handshake helpers, request routing
/// and responses, and one connection.
const SERVER_PARTS: &[&str] = &[
    "server.mind",
    "server_rsa.mind",
    "server_tls.mind",
    "server_http.mind",
    "server_conn.mind",
];

/// [`IMPORTS`] and `more` imports, the fixture functions, any extra source,
/// then the fixture file `own`.
pub fn source(own: &str, more: &[&str], fixtures: &[(&str, Vec<u8>)], extra: &str) -> String {
    source_parts(&[own], more, fixtures, extra)
}

/// [`source`] with the fixture files `parts` appended in order.
fn source_parts(
    parts: &[&str],
    more: &[&str],
    fixtures: &[(&str, Vec<u8>)],
    extra: &str,
) -> String {
    let mut src: String = IMPORTS
        .iter()
        .chain(more)
        .map(|m| format!("import std.{m};\n"))
        .collect();
    src.push('\n');
    for (name, bytes) in fixtures {
        src.push_str(&fixture_fn(name, bytes));
    }
    src.push_str(extra);
    for own in parts {
        let body = std::fs::read_to_string(format!("{FIX}/{own}"))
            .unwrap_or_else(|e| panic!("read {own}: {e}"));
        src.push_str(&body);
    }
    src
}

/// How a canned server case ends after its response (`hl_canned_end`).
#[derive(Clone, Copy)]
pub enum End {
    CloseNotify = 0,
    FatalAlert = 1,
    WarningAlert = 2,
    TcpClose = 3,
}

/// The first canned case id (`hl_case_count()` in server.mind).
pub const FIRST_CANNED: usize = 13;

/// The server source; `canned[k]` is case `FIRST_CANNED + k`: its response
/// bytes and its end.
pub fn server_source(canned: &[(Vec<u8>, End)]) -> String {
    let mut fixtures = vec![
        ("hl_leaf_good", der("leaf_good")),
        ("hl_leaf_expired", der("leaf_expired")),
        ("hl_leaf_wronghost", der("leaf_wronghost")),
        ("hl_leaf_ec", der("leaf_ec")),
        ("hl_leaf_rsa3072", der("leaf_rsa3072")),
        ("hl_inter", der("inter")),
        ("hl_key_n", key_field("n")),
        ("hl_key_d", key_field("d")),
        ("hl_env_cases", b"MIND_HTTPS_SERVER_CASES".to_vec()),
    ];
    let names: Vec<String> = (0..canned.len())
        .map(|k| format!("hl_canned_{}", FIRST_CANNED + k))
        .collect();
    for (k, (bytes, _)) in canned.iter().enumerate() {
        fixtures.push((names[k].as_str(), bytes.clone()));
    }
    let mut dispatch = String::from("fn hl_canned(cid: i64, dst: i64) -> i64 {\n");
    let mut ends = String::from("fn hl_canned_end(cid: i64) -> i64 {\n");
    for (k, (_, end)) in canned.iter().enumerate() {
        let id = FIRST_CANNED + k;
        dispatch.push_str(&format!(
            "    if cid == {id} {{\n        return {}(dst);\n    }}\n",
            names[k]
        ));
        ends.push_str(&format!(
            "    if cid == {id} {{\n        return {};\n    }}\n",
            *end as i64
        ));
    }
    dispatch.push_str("    return 0;\n}\n\n");
    ends.push_str("    return 0;\n}\n\n");
    source_parts(SERVER_PARTS, &[], &fixtures, &format!("{dispatch}{ends}"))
}

/// `mindc build` a one-file project named `case`; the executable, or `None`
/// when the build backend is a gap the shared gate allows to skip.
pub fn build_program(case: &str, main: &str) -> Option<PathBuf> {
    let dir = crate::common::scratch_dir(&format!("std_https_{case}"));
    let manifest = format!(
        "[package]\nname = \"{case}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
         output = \"{case}\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
         sources = [\"main.mind\"]\n"
    );
    std::fs::write(dir.join("Mind.toml"), manifest).expect("write Mind.toml");
    std::fs::write(dir.join("main.mind"), main).expect("write main.mind");
    let out = Command::new(crate::common::mindc_bin())
        .arg("build")
        .current_dir(&dir)
        .output()
        .expect("spawn mindc build");
    if !crate::common::gate::compiled(case, &out) {
        return None;
    }
    let exe = dir.join("target").join("debug").join(case);
    assert!(exe.is_file(), "{case}: no executable at {}", exe.display());
    Some(exe)
}

/// Kill a child that is still running when dropped.
pub struct Reaped(pub Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start the server (serving `cases`, e.g. "13,14", or its default list), read
/// its `PORT <n>` line.
pub fn start_server(exe: &Path, cases: Option<&str>) -> (Reaped, u16) {
    let mut cmd = Command::new(exe);
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    if let Some(c) = cases {
        cmd.env("MIND_HTTPS_SERVER_CASES", c);
    }
    let mut child = cmd.spawn().expect("spawn the loopback server");
    let stdout = child.stdout.take().expect("server stdout");
    let mut line = String::new();
    BufReader::new(stdout)
        .read_line(&mut line)
        .expect("read the server port line");
    let port = line
        .trim()
        .strip_prefix("PORT ")
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or_else(|| panic!("server printed {line:?}, not `PORT <n>`"));
    (Reaped(child), port)
}
