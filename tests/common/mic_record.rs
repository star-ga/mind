// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Shared harness for the `std/mic_record.mind` + `std/mic_results.mind`
//! gates: an independent reference encoder built on the compiler's own mic@3
//! primitives, and a loader for the two modules compiled from the files ON
//! DISK.
//!
//! The behaviour gates compile the working-tree sources, not the copies that
//! `include_str!` bundles into `mindc`, so a mutation of either `.mind` file
//! reaches the gate without a rebuild of the compiler. The `import std.*`
//! wiring through the bundled copies has its own gate.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use libmind::ir::IRModule;
use libmind::ir::compact::v2::{uleb128_write, zigzag_encode};
use libmind::ir::compact::v3::emit_mic3;

/// One MAP value of the reference encoder.
#[derive(Clone, Copy, Debug)]
pub enum Val<'a> {
    Str(&'a str),
    Int(i64),
    Bytes(&'a [u8]),
}

/// The canonical record for `entries`: the compiler's own empty-module mic@3
/// body, the MAP sentinel, then the entries in key-byte order with the
/// compiler's ULEB128 and zigzag encoders. Independent of the MIND writer.
pub fn reference(entries: &[(&str, Val<'_>)]) -> Vec<u8> {
    let mut out = emit_mic3(&IRModule::new());
    out.push(0x4D);
    let mut sorted: Vec<&(&str, Val<'_>)> = entries.iter().collect();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    uleb128_write(&mut out, sorted.len() as u64).unwrap();
    for (key, val) in sorted {
        uleb128_write(&mut out, key.len() as u64).unwrap();
        out.extend_from_slice(key.as_bytes());
        match val {
            Val::Str(s) => {
                out.push(0x00);
                uleb128_write(&mut out, s.len() as u64).unwrap();
                out.extend_from_slice(s.as_bytes());
            }
            Val::Int(i) => {
                out.push(0x01);
                uleb128_write(&mut out, zigzag_encode(*i)).unwrap();
            }
            Val::Bytes(b) => {
                out.push(0x02);
                uleb128_write(&mut out, b.len() as u64).unwrap();
                out.extend_from_slice(b);
            }
        }
    }
    out
}

/// ULEB128 of `v`, for hand-built malformed records.
pub fn uleb(v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    uleb128_write(&mut out, v).unwrap();
    out
}

/// The record header: empty-module body, sentinel, entry count.
pub fn head(count: &[u8]) -> Vec<u8> {
    let mut out = emit_mic3(&IRModule::new());
    out.push(0x4D);
    out.extend_from_slice(count);
    out
}

/// The files of the `std.mic_record` codec, leaf module first. Each one's
/// `import std.mic_record_*;` lines name a module of this same list.
const CODEC_FILES: [&str; 3] = [
    "mic_record_wire.mind",
    "mic_record_table.mind",
    "mic_record.mind",
];

/// The codec and the results module as one translation unit, read from the
/// working tree. The codec's imports of its own parts, the results module's
/// `import std.mic_record;` and its `mic_record.` call qualifiers are dropped,
/// since the whole codec is defined in the same unit.
pub fn combined_source() -> String {
    let std_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("std");
    let read = |name: &str| {
        let p = std_dir.join(name);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
    };
    let codec: Vec<String> = CODEC_FILES
        .iter()
        .map(|f| {
            read(f)
                .lines()
                .filter(|l| !l.starts_with("import std.mic_record_"))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect();
    let codec = codec.join("\n");
    let results = read("mic_results.mind");
    let import = "import std.mic_record;\n";
    assert!(
        results.contains(import),
        "std/mic_results.mind must import std.mic_record"
    );
    let results = results.replace(import, "").replace("mic_record.", "");
    format!("{codec}\n{results}")
}

/// A dlopen'ed build of the combined source with typed `i64` calls.
pub struct Lib {
    lib: libloading::Library,
}

/// Compile `source` to a shared library in `target`'s scratch directory and
/// load it; `None` once a capability gap was routed through the skip gate.
pub fn load(target: &str, source: &str) -> Option<Lib> {
    let dir: PathBuf = super::scratch_dir(target);
    let src = super::scratch_src_in(&dir, "mic_record_gate.mind", source);
    let so = dir.join("libmic_record_gate.so");
    let out = Command::new(super::mindc_bin())
        .arg(&src)
        .arg("--emit-shared")
        .arg(&so)
        .output()
        .expect("run mindc --emit-shared");
    if !super::gate::compiled(target, &out) {
        return None;
    }
    let lib = unsafe { libloading::Library::new(&so) }
        .unwrap_or_else(|e| panic!("dlopen {}: {e}", so.display()));
    Some(Lib { lib })
}

impl Lib {
    fn sym<T>(&self, name: &str) -> libloading::Symbol<'_, T> {
        unsafe { self.lib.get::<T>(name.as_bytes()) }
            .unwrap_or_else(|e| panic!("symbol {name}: {e}"))
    }

    /// Call the exported `name` with `args` (every MIND `i64` ABI parameter).
    pub fn call(&self, name: &str, args: &[i64]) -> i64 {
        type F0 = unsafe extern "C" fn() -> i64;
        type F1 = unsafe extern "C" fn(i64) -> i64;
        type F2 = unsafe extern "C" fn(i64, i64) -> i64;
        type F3 = unsafe extern "C" fn(i64, i64, i64) -> i64;
        type F4 = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
        type F5 = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
        let a = args;
        unsafe {
            match a.len() {
                0 => self.sym::<F0>(name)(),
                1 => self.sym::<F1>(name)(a[0]),
                2 => self.sym::<F2>(name)(a[0], a[1]),
                3 => self.sym::<F3>(name)(a[0], a[1], a[2]),
                4 => self.sym::<F4>(name)(a[0], a[1], a[2], a[3]),
                5 => self.sym::<F5>(name)(a[0], a[1], a[2], a[3], a[4]),
                n => panic!("{name}: {n} arguments are not wired"),
            }
        }
    }

    /// The named error `mrec_e_<name>()`.
    pub fn err(&self, name: &str) -> i64 {
        self.call(&format!("mrec_e_{name}"), &[])
    }

    pub fn new_record(&self) -> i64 {
        let h = self.call("mrec_new", &[]);
        assert!(h > 0, "mrec_new must return a handle");
        h
    }

    pub fn put_int(&self, h: i64, key: &[u8], v: i64) -> i64 {
        self.call("mrec_put_int", &[h, ptr(key), key.len() as i64, v])
    }

    pub fn put_str(&self, h: i64, key: &[u8], v: &[u8]) -> i64 {
        let args = [h, ptr(key), key.len() as i64, ptr(v), v.len() as i64];
        self.call("mrec_put_str", &args)
    }

    pub fn put_bytes(&self, h: i64, key: &[u8], v: &[u8]) -> i64 {
        let args = [h, ptr(key), key.len() as i64, ptr(v), v.len() as i64];
        self.call("mrec_put_bytes", &args)
    }

    pub fn count(&self, h: i64) -> i64 {
        self.call("mrec_count", &[h])
    }

    pub fn find(&self, h: i64, key: &[u8]) -> i64 {
        self.call("mrec_find", &[h, ptr(key), key.len() as i64])
    }

    /// `mrec_get_int`: the status and the value it wrote.
    pub fn get_int(&self, h: i64, key: &[u8]) -> (i64, i64) {
        let mut out = [0i64; 1];
        let rc = self.call(
            "mrec_get_int",
            &[h, ptr(key), key.len() as i64, out.as_mut_ptr() as i64],
        );
        (rc, out[0])
    }

    /// The encoded record bytes.
    pub fn encode(&self, h: i64) -> Vec<u8> {
        let b = self.call("mrec_encode", &[h]);
        assert!(b > 0, "mrec_encode must return a buffer");
        let addr = self.call("mrec_buf_addr", &[b]);
        let len = self.call("mrec_buf_len", &[b]);
        bytes_at(addr, len)
    }

    pub fn decode(&self, bytes: &[u8]) -> i64 {
        self.call("mrec_decode", &[ptr(bytes), bytes.len() as i64])
    }
}

pub fn ptr(b: &[u8]) -> i64 {
    b.as_ptr() as i64
}

/// Copy `len` bytes the module owns at `addr`.
pub fn bytes_at(addr: i64, len: i64) -> Vec<u8> {
    assert!(len >= 0, "negative length {len}");
    if len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(addr as *const u8, len as usize) }.to_vec()
}

/// `VmRSS` and `VmHWM` (peak) of this process in KiB, from /proc/self/status.
#[cfg(target_os = "linux")]
pub fn rss_kib() -> (i64, i64) {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let field = |name: &str| -> i64 {
        let line = status
            .lines()
            .find(|l| l.starts_with(name))
            .unwrap_or_else(|| panic!("a {name} line"));
        line.split_whitespace()
            .nth(1)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("a {name} value in KiB"))
    };
    (field("VmRSS:"), field("VmHWM:"))
}

/// Run a resident-set bound alone. The resident set belongs to the process,
/// so a bound taken while sibling tests allocate on other threads measures
/// them too (measured: 26916 KiB beside the siblings, under the bound with
/// one test thread). The parent runs this test binary again with only
/// `test_name` selected and returns `false`; the child returns `true` and
/// takes the measurement.
#[cfg(target_os = "linux")]
pub fn alone(test_name: &str) -> bool {
    const ALONE_VAR: &str = "MIND_RSS_BOUND_ALONE";
    if std::env::var_os(ALONE_VAR).is_some_and(|v| v == test_name) {
        return true;
    }
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new(exe)
        .args([test_name, "--exact", "--test-threads", "1", "--nocapture"])
        .env(ALONE_VAR, test_name)
        .output()
        .expect("spawn this test binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // `--exact` on a name that matches no test still exits 0.
    assert!(
        out.status.success() && text.contains("test result: ok. 1 passed"),
        "{test_name} alone:\n{text}"
    );
    false
}

/// Growth in KiB of the resident set and of its peak, whichever is larger,
/// over `rounds` calls of `round`, after a warm-up of a tenth as many (at
/// least one) so the allocator's arenas are already mapped.
#[cfg(target_os = "linux")]
pub fn rss_growth_kib(rounds: usize, mut round: impl FnMut()) -> i64 {
    for _ in 0..(rounds / 10).max(1) {
        round();
    }
    let (rss0, peak0) = rss_kib();
    for _ in 0..rounds {
        round();
    }
    let (rss1, peak1) = rss_kib();
    (rss1 - rss0).max(peak1 - peak0)
}
