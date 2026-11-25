// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Growth paths of the runtime support library free the buffer they replace.
//!
//! A dynamic `array<T>` is an i64 handle to one `[addr | len | cap]` record.
//! `vec_push` grows that record IN PLACE: the handle value never changes, so
//! an alias taken before a push reads the grown data. When the backing store
//! grows, the old buffer is freed after the copy. Before this rule an array
//! of N elements kept every smaller buffer alive, about N more elements of
//! dead memory.
//!
//! How the tests measure a kept buffer: glibc `mallinfo2()` reports the bytes
//! the process holds through `malloc` (`uordblks` for the heap plus `hblkhd`
//! for mapped blocks). The test reads it in the same process just before and
//! just after one growth-heavy call. A final buffer of B bytes must show a
//! delta of at least B (positive control: the measure sees the buffer) and
//! below 1.5 B. A growth path that keeps its old buffers shows about 2 B, so
//! the test goes red. No runtime hook is needed.
//!
//! A growth inside a `region` block must keep the region's ownership: the
//! region frees what it tracks at exit, so the runtime moves the tracked entry
//! to the new buffer. Without that move the exit frees the old buffer a second
//! time and glibc aborts the process.
//!
//! Strings use another form: each string push returns a NEW header and the old
//! header stays a valid string. So a string growth takes a new buffer and
//! copies, and it never frees or reallocs the old buffer: an older header, or
//! a region frame, can still reach it. The string tests run each program in
//! five fresh processes: a read of freed memory gives a value that changes
//! from run to run, and a second free stops the process. The old string buffer
//! stays allocated until the container model of the next release (a known
//! limit), so no test here measures it.
//!
//! Gate: `cargo test --release --features "std-surface mlir-build cross-module-imports"
//!                   --test runtime_growth_buffer_run`

#![cfg(all(
    target_os = "linux",
    target_env = "gnu",
    feature = "mlir-build",
    feature = "std-surface"
))]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SRC: &str = r#"
pub fn fill(n: i64) -> i64 {
    let xs: array<i64> = [];
    let mut i = 0;
    while i < n {
        xs.push(i);
        i = i + 1;
    }
    return xs.length;
}

pub fn alias_sees_every_growth() -> i64 {
    let xs: array<i64> = [];
    let view = xs;
    let mut i = 0;
    while i < 100 {
        xs.push(i * 3);
        i = i + 1;
    }
    return view.length * 1000 + view[99] + view[0] * 7;
}

fn content_hash(x: string) -> i64 {
    let n = string_len(x);
    let mut h = n;
    let mut i = 0;
    while i < n {
        h = (h * 131 + string_get_byte(x, i)) % 1000000007;
        i = i + 1;
    }
    return h;
}

pub fn str_old_handle_after_growth() -> i64 {
    let s = "abcdefghijklmnopqrstuvwxyz0123456789";
    let pad = "0123456789012345678901234567890123456789012345678";
    let t = string_push_byte(s, 65);
    let moved: i64 = if string_addr(s) == string_addr(t) { 0 } else { 1 };
    return moved * 1000000 + string_get_byte(s, 0) * 1000 + string_get_byte(t, 0) + 0 * string_len(pad);
}

pub fn str_old_handle_content() -> i64 {
    let s = "abcdefghijklmnopqrstuvwxyz0123456789";
    let pad = "0123456789012345678901234567890123456789012345678";
    let t = string_push_byte(s, 65);
    return content_hash(s) * 1000000007 + content_hash(t) + 0 * string_len(pad);
}

pub fn str_two_pushes_grew() -> i64 {
    let s = "abcdefghijklmnopqrstuvwxyz0123456789";
    let pad = "0123456789012345678901234567890123456789012345678";
    let a = string_push_byte(s, 65);
    let b = string_push_byte(s, 66);
    let ga: i64 = if string_addr(a) == string_addr(s) { 0 } else { 1 };
    let gb: i64 = if string_addr(b) == string_addr(s) { 0 } else { 1 };
    let ab: i64 = if string_addr(a) == string_addr(b) { 0 } else { 1 };
    return ab * 100 + ga * 10 + gb + 0 * string_len(pad);
}

pub fn str_two_pushes_content() -> i64 {
    let s = "abcdefghijklmnopqrstuvwxyz0123456789";
    let pad = "0123456789012345678901234567890123456789012345678";
    let a = string_push_byte(s, 65);
    let b = string_push_byte(s, 66);
    return content_hash(a) * 1000000007 + content_hash(b) + 0 * string_len(pad);
}

pub fn region_str_growth() -> i64 {
    let r = region {
        let s = "abcdefghijklmnopqrstuvwxyz0123456789";
        let pad = "0123456789012345678901234567890123456789012345678";
        let t = string_push_byte(s, 65);
        string_len(t) + 0 * string_len(pad)
    };
    return r;
}

pub fn region_old_handle_after_growth() -> i64 {
    let r = region {
        let s = "abcdefghijklmnopqrstuvwxyz0123456789";
        let pad = "0123456789012345678901234567890123456789012345678";
        let t = string_push_byte(s, 65);
        content_hash(s) * 1000000007 + content_hash(t) + 0 * string_len(pad)
    };
    return r;
}

pub fn region_two_pushes_content() -> i64 {
    let r = region {
        let s = "abcdefghijklmnopqrstuvwxyz0123456789";
        let pad = "0123456789012345678901234567890123456789012345678";
        let a = string_push_byte(s, 65);
        let b = string_push_byte(s, 66);
        content_hash(a) * 1000000007 + content_hash(b) + 0 * string_len(pad)
    };
    return r;
}
"#;

/// Shared prologue of every driver: load the artifact and libc, declare the
/// i64 ABI of the runtime functions, and define the heap measure.
const PROLOGUE: &str = r#"import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
libc = ctypes.CDLL("libc.so.6")
class MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]
libc.mallinfo2.restype = MallInfo2
def held():
    m = libc.mallinfo2()
    return m.uordblks + m.hblkhd
I64 = ctypes.c_int64
for name in ("fill", "alias_sees_every_growth", "vec_new", "vec_push", "vec_len",
             "vec_get", "vec_addr", "map_new", "map_insert", "map_keys", "map_len",
             "string_to_utf8_bytes", "string_push_byte", "string_addr", "string_len",
             "string_cap", "string_from_utf8_bytes", "read_to_string",
             "__mind_alloc", "__mind_store_i64", "__mind_load_i64",
             "__mind_region_enter", "__mind_region_track", "__mind_region_exit"):
    f = getattr(lib, name)
    f.restype = I64
    f.argtypes = None
def i64(*xs):
    return [I64(x) for x in xs]
def check_delta(label, delta, final_bytes):
    assert delta >= final_bytes, f"{label}: delta {delta} < final buffer {final_bytes}: the measure saw nothing"
    assert delta < final_bytes * 3 // 2, f"{label}: delta {delta} >= 1.5 x final buffer {final_bytes}: old buffers kept"
    print(f"{label}: delta={delta} final={final_bytes}")
"#;

/// N = 100000 elements -> final capacity 131072 (doubling from 4) -> 1 MiB.
const N: i64 = 100_000;
const FINAL_BYTES: i64 = 131_072 * 8;

fn build(dir: &Path) -> PathBuf {
    let source = dir.join("growth.mind");
    let artifact = dir.join("growth.so");
    std::fs::write(&source, SRC).expect("write source");
    let out = Command::new(common::mindc_bin())
        .args([
            source.to_str().unwrap(),
            "--emit-shared",
            artifact.to_str().unwrap(),
        ])
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run mindc");
    assert!(out.status.success(), "compile failed:\n{}", combined(&out));
    artifact
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Run `body` after the shared prologue in a fresh `python3 -I` process.
fn run_driver(so: &Path, body: &str) -> Output {
    let script = format!("{PROLOGUE}{body}");
    Command::new("python3")
        .args(["-I", "-c", &script, so.to_str().unwrap()])
        .output()
        .expect("run python3 driver")
}

fn assert_driver_ok(so: &Path, body: &str) {
    let ran = run_driver(so, body);
    assert!(ran.status.success(), "driver failed:\n{}", combined(&ran));
    print!("{}", String::from_utf8_lossy(&ran.stdout));
}

#[test]
fn array_push_growth_frees_the_old_buffer() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_array");
    let so = build(&dir);
    assert_driver_ok(
        &so,
        &format!(
            r#"
assert lib.fill(I64(64)) == 64
before = held()
got = lib.fill(I64({N}))
after = held()
assert got == {N}, got
check_delta("array push", after - before, {FINAL_BYTES})
"#
        ),
    );
}

#[test]
fn handle_is_stable_and_alias_reads_grown_data() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_alias");
    let so = build(&dir);
    assert_driver_ok(
        &so,
        r#"
got = lib.alias_sees_every_growth()
assert got == 100297, f"alias_sees_every_growth={got}, expected 100297"
h = lib.vec_new()
first_addr = None
moved = 0
for i in range(1000):
    before_addr = lib.vec_addr(*i64(h))
    r = lib.vec_push(*i64(h, i * 5))
    assert r == h, f"push {i} returned handle {r}, expected {h}"
    if lib.vec_addr(*i64(h)) != before_addr:
        moved += 1
alias = h
assert lib.vec_len(*i64(alias)) == 1000
assert sum(lib.vec_get(*i64(alias, i)) for i in range(1000)) == 5 * 999 * 1000 // 2
assert moved >= 8, f"only {moved} growths moved the buffer"
print("handle stable over", moved, "growths")
"#,
    );
}

#[test]
fn vec_built_by_other_runtime_paths_frees_old_buffers() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_inherit");
    let so = build(&dir);
    assert_driver_ok(
        &so,
        &format!(
            r#"
n = {N}
buf = lib.__mind_alloc(*i64(n))
ctypes.memset(buf, 65, n)
s = lib.__mind_alloc(*i64(24))
lib.__mind_store_i64(*i64(s, buf))
lib.__mind_store_i64(*i64(s + 8, n))
lib.__mind_store_i64(*i64(s + 16, n))
before = held()
v = lib.string_to_utf8_bytes(*i64(s))
after = held()
assert lib.vec_len(*i64(v)) == n
check_delta("string_to_utf8_bytes", after - before, {FINAL_BYTES})
m = lib.map_new()
for i in range(n):
    m = lib.map_insert(*i64(m, i, i))
assert lib.map_len(*i64(m)) == n
before = held()
keys = lib.map_keys(*i64(m))
after = held()
assert lib.vec_len(*i64(keys)) == n
assert lib.vec_get(*i64(keys, n - 1)) == n - 1
check_delta("map_keys", after - before, {FINAL_BYTES})
"#
        ),
    );
}

#[test]
fn array_growth_inside_a_region_moves_the_tracked_entry() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_region_vec");
    let so = build(&dir);
    assert_driver_ok(
        &so,
        r#"
lib.__mind_region_enter()
buf = lib.__mind_region_track(*i64(lib.__mind_alloc(*i64(32))))
rec = lib.__mind_region_track(*i64(lib.__mind_alloc(*i64(24))))
lib.__mind_store_i64(*i64(rec, buf))
lib.__mind_store_i64(*i64(rec + 8, 0))
lib.__mind_store_i64(*i64(rec + 16, 4))
for i in range(5):
    assert lib.vec_push(*i64(rec, i)) == rec
assert lib.vec_addr(*i64(rec)) != buf, "precondition: the growth did not move the buffer"
assert [lib.vec_get(*i64(rec, i)) for i in range(5)] == [0, 1, 2, 3, 4]
lib.__mind_region_exit()
print("region exit ok")
"#,
    );
}

#[test]
fn string_growth_inside_a_region_keeps_the_tracked_buffer() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_region_str");
    let so = build(&dir);
    assert_driver_ok(
        &so,
        r#"
lib.__mind_region_enter()
# A buffer above the small-block cache, then a block of the same size that is
# carved right after it: the blocker stops an in-place extension, so a realloc
# would move the buffer and free the old one under the old header.
buf = lib.__mind_region_track(*i64(lib.__mind_alloc(*i64(3000))))
blocker = lib.__mind_region_track(*i64(lib.__mind_alloc(*i64(3000))))
rec = lib.__mind_region_track(*i64(lib.__mind_alloc(*i64(24))))
ctypes.memset(buf, 66, 3000)
lib.__mind_store_i64(*i64(rec, buf))
lib.__mind_store_i64(*i64(rec + 8, 3000))
lib.__mind_store_i64(*i64(rec + 16, 3000))
s2 = lib.string_push_byte(*i64(rec, 67))
assert lib.string_len(*i64(s2)) == 3001
assert lib.string_addr(*i64(s2)) != buf, "the growth did not take a new buffer"
assert ctypes.string_at(lib.string_addr(*i64(s2)), 3001) == b"B" * 3000 + b"C"
assert lib.string_addr(*i64(rec)) == buf
assert ctypes.string_at(buf, 3000) == b"B" * 3000, "the old header does not read its own bytes"
lib.__mind_region_exit()
print("region exit ok")
"#,
    );
}

/// The `content_hash` of the MIND source, for the expected values.
fn content_hash(bytes: &[u8]) -> i64 {
    let mut h = bytes.len() as i64;
    for &b in bytes {
        h = (h * 131 + i64::from(b)) % 1_000_000_007;
    }
    h
}

const LIT: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";

fn pushed(byte: u8) -> Vec<u8> {
    let mut v = LIT.to_vec();
    v.push(byte);
    v
}

fn pair(a: &[u8], b: &[u8]) -> i64 {
    content_hash(a) * 1_000_000_007 + content_hash(b)
}

/// Run one exported MIND function in five fresh processes. Every run must
/// exit with status 0 and give `want`: a read of freed memory changes from
/// run to run, and a second free stops the process.
fn assert_five_runs(so: &Path, name: &str, want: i64) {
    let body = format!(
        "f = getattr(lib, \"{name}\")\nf.restype = I64\nf.argtypes = None\nprint(\"{name}=\" + str(f()))\n"
    );
    let mut got = Vec::new();
    for run in 0..5 {
        let ran = run_driver(so, &body);
        assert!(
            ran.status.success(),
            "{name}: run {run} ended with {:?}:\n{}",
            ran.status,
            combined(&ran)
        );
        got.push(String::from_utf8_lossy(&ran.stdout).trim().to_string());
    }
    let expected = format!("{name}={want}");
    assert!(
        got.iter().all(|g| *g == expected),
        "{name}: five runs gave {got:?}, expected {expected} in each run"
    );
    println!("{expected} in 5 of 5 runs");
}

#[test]
fn string_old_handle_reads_its_own_bytes_after_growth() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_str_old");
    let so = build(&dir);
    // moved * 1000000 + s[0] * 1000 + t[0]: the growth took a new buffer, the
    // old handle still reads "a" (97), the new one too.
    assert_five_runs(&so, "str_old_handle_after_growth", 1_097_097);
    assert_five_runs(&so, "str_old_handle_content", pair(LIT, &pushed(b'A')));
}

#[test]
fn two_pushes_from_one_string_handle_both_grow() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_str_two");
    let so = build(&dir);
    // Each push had to grow (a new buffer for a, another for b).
    assert_five_runs(&so, "str_two_pushes_grew", 111);
    assert_five_runs(
        &so,
        "str_two_pushes_content",
        pair(&pushed(b'A'), &pushed(b'B')),
    );
}

#[test]
fn string_growth_inside_a_region_block() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_str_region");
    let so = build(&dir);
    assert_five_runs(&so, "region_str_growth", 37);
    assert_five_runs(
        &so,
        "region_old_handle_after_growth",
        pair(LIT, &pushed(b'A')),
    );
    assert_five_runs(
        &so,
        "region_two_pushes_content",
        pair(&pushed(b'A'), &pushed(b'B')),
    );
}

#[test]
fn string_from_utf8_bytes_keeps_no_step_buffer() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_from_utf8");
    let so = build(&dir);
    assert_driver_ok(
        &so,
        &format!(
            r#"
n = {N}
v = lib.vec_new()
for i in range(n):
    lib.vec_push(*i64(v, 65 + i % 26))
before = held()
s = lib.string_from_utf8_bytes(*i64(v))
after = held()
assert lib.string_len(*i64(s)) == n
assert lib.string_cap(*i64(s)) == 131072, lib.string_cap(*i64(s))
assert ctypes.string_at(lib.string_addr(*i64(s)), n) == bytes(65 + i % 26 for i in range(n))
check_delta("string_from_utf8_bytes", after - before, 131072)
"#
        ),
    );
}

#[test]
fn read_error_path_frees_its_buffer() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_read_err");
    let so = build(&dir);
    let plain = dir.join("plain.txt");
    assert_driver_ok(
        &so,
        &format!(
            r#"
import errno, os
bad = b"/sys/class/net/lo/speed"
try:
    open(bad, "rb").read()
    raise SystemExit("precondition: a read of /sys/class/net/lo/speed did not fail")
except OSError as e:
    assert e.errno == errno.EINVAL, e
fd = os.open(bad, os.O_RDONLY)
size = os.lseek(fd, 0, os.SEEK_END)
os.close(fd)
assert size > 0, f"precondition: size {{size}}"
plain = {plain:?}.encode()
open(plain, "wb").write(b"x" * size)
def reads(path, count):
    p = ctypes.create_string_buffer(path)
    lib.read_to_string(*i64(ctypes.addressof(p)))
    before = held()
    for _ in range(count):
        r = lib.read_to_string(*i64(ctypes.addressof(p)))
        out = (lib.__mind_load_i64(*i64(r)), lib.__mind_load_i64(*i64(r + 8)))
    return held() - before, out
# Positive control: a good read publishes a buffer of size + 1 bytes each time.
kept, (addr, length) = reads(plain, 200)
assert addr != 0 and length == size, (addr, length)
assert kept >= 200 * (size + 1), f"positive control: delta {{kept}} for 200 good reads"
delta, (addr, length) = reads(bad, 200)
assert addr == 0 and length == 0, (addr, length)
assert delta < 200 * 512, f"read error: delta {{delta}} for 200 failed reads, a buffer of {{size + 1}} bytes is kept"
print(f"read error: delta={{delta}} for 200 failed reads, good reads kept {{kept}}")
"#
        ),
    );
}
