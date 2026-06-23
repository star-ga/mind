// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Struct-FIELD `array<T>` RUNTIME gate — construct, index-read, and
//! push-with-mutation-persistence on a collection held in a struct field.
//!
//! Three things that previously failed at the aggregate-call ABI boundary:
//!   * `Bag { items: [] }` — a struct literal whose `array<T>` field value is an
//!     array literal must lower onto the std.vec heap runtime (an i64 handle),
//!     not the generic const-array/tensor path (whose result the field store
//!     rejects as a non-i64 aggregate).
//!   * `b.items[i]` — index-read on a struct array field routes to `vec_get`.
//!   * `b.items.push(x)` — a mutating method on a struct array field is rebound
//!     to `b.items = b.items.push(x)` so the fresh-on-realloc std handle persists
//!     in the field (a bare statement would silently drop the growth).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test struct_array_field_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
struct Bag { items: array<i64> }

fn fill(b: Bag) -> Bag {
    b.items.push(10)
    b.items.push(20)
    b.items.push(30)
    return b
}

pub fn run() -> i64 {
    // Construct with an empty array-literal field, grow it through three
    // pushes (each must persist), then index-read the three elements back.
    let bag = Bag { items: [] }
    let bag = fill(bag)
    return bag.items[0] + bag.items[1] + bag.items[2]
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn struct_array_field_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("struct-array-field-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_struct_array_field_run.mind");
    let so = dir.join("mind_struct_array_field_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("struct-array-field-run: needs mlir-build; skipping");
            return;
        }
        panic!("struct-array-field-run: mindc --emit-shared failed:\n{stderr}");
    }

    // push 10 + 20 + 30, persisted, read back at [0]/[1]/[2] → 60.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 60, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "struct-array-field-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
