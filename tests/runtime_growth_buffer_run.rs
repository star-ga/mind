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
fn string_growth_inside_a_region_moves_the_tracked_entry() {
    let dir = common::scratch_dir("runtime_growth_buffer_run_region_str");
    let so = build(&dir);
    assert_driver_ok(
        &so,
        r#"
lib.__mind_region_enter()
# A buffer above the small-block cache, then a block of the same size that is
# carved right after it: the blocker stops an in-place extension, so the
# growth must move the buffer (checked below).
buf = lib.__mind_region_track(*i64(lib.__mind_alloc(*i64(3000))))
blocker = lib.__mind_region_track(*i64(lib.__mind_alloc(*i64(3000))))
rec = lib.__mind_region_track(*i64(lib.__mind_alloc(*i64(24))))
ctypes.memset(buf, 66, 3000)
lib.__mind_store_i64(*i64(rec, buf))
lib.__mind_store_i64(*i64(rec + 8, 3000))
lib.__mind_store_i64(*i64(rec + 16, 3000))
s2 = lib.string_push_byte(*i64(rec, 67))
assert lib.string_len(*i64(s2)) == 3001
assert lib.string_addr(*i64(s2)) != buf, "precondition: realloc did not move the buffer"
assert ctypes.string_at(lib.string_addr(*i64(s2)), 3001) == b"B" * 3000 + b"C"
lib.__mind_region_exit()
print("region exit ok")
"#,
    );
}
