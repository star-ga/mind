// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Byte-reproducibility gate for programs with module-level constant tables.
//!
//! Twelve `const TAB<k>: [i64; N]` tables, each read by its own accessor, are
//! built to a shared library [`BUILDS`] times IN SEPARATE PROCESSES with
//! address-space randomisation left on, and every artifact must be the same
//! bytes. The tables have distinct lengths so their bufferized globals carry
//! distinct names, which is what made the defect visible as a layout change.
//!
//! The defect: on the tensor-aware preset, `one-shot-bufferize` with function
//! boundaries hoists each `arith.constant dense<...> : tensor<...>` into a
//! private `memref.global`, in the order it visits functions — an order read
//! out of a pointer-keyed map, so it moved with ASLR. Measured before the fix:
//! four builds of this program, four different `.so` hashes of one size, the
//! read-only data reordered. The MLIR text and the mic@3 were identical; only
//! `mlir-opt`'s output moved. The fix pre-declares the globals in source order
//! (`src/mlir/const_globals.rs`) so bufferization reuses them.
//!
//! Positive control: every artifact is loaded and every accessor is called at
//! its first and last index, so "identical" can never mean "identically empty"
//! or "identically wrong".
//!
//! Gate: `cargo test --release --features "mlir-build std-surface cross-module-imports"
//!                   --test const_table_build_determinism`

#![cfg(all(unix, feature = "mlir-build"))]

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Separate `mindc` processes per check. Each process gets fresh heap
/// addresses, so a pointer-ordered choice shows up as a second distinct hash
/// with overwhelming probability well before eight runs.
const BUILDS: usize = 8;
const TABLES: usize = 12;

/// Element `i` of table `k` (deterministic, distinct across tables).
fn elem(k: usize, i: usize) -> i64 {
    (((k * 41 + i) * 4729 + 7919 * k) % 100_000) as i64
}

fn table_len(k: usize) -> usize {
    40 + k
}

fn source() -> String {
    let mut s = String::new();
    for k in 0..TABLES {
        let vals: Vec<String> = (0..table_len(k)).map(|i| elem(k, i).to_string()).collect();
        s.push_str(&format!(
            "const TAB{k}: [i64; {}] = [{}];\npub fn t_{k}(i: i64) -> i64 {{\n    TAB{k}[i]\n}}\n\n",
            table_len(k),
            vals.join(", ")
        ));
    }
    s
}

/// Build `src` to `so`. `None` only for a tolerated capability gap (routed
/// through the fail-closed skip gate); any other failure panics.
fn build(mindc: &Path, src: &Path, so: &Path) -> Option<Vec<u8>> {
    let _ = std::fs::remove_file(so);
    let out = Command::new(mindc)
        .arg(src)
        .arg("--emit-shared")
        .arg(so)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("spawn mindc");
    if !common::gate::compiled("const_table_build_determinism", &out) {
        return None;
    }
    let bytes = std::fs::read(so).unwrap_or_else(|e| panic!("read {}: {e}", so.display()));
    // A launcher stub is ~1.2 KiB; twelve 40..51-element i64 tables alone are
    // over 4 KiB of read-only data.
    assert!(
        bytes.len() > 4096,
        "{} is {} bytes — not a real shared library",
        so.display(),
        bytes.len()
    );
    Some(bytes)
}

/// Call every accessor at its first and last index through ctypes.
fn check_values(so: &Path) {
    let mut py = format!(
        "import ctypes\nlib = ctypes.CDLL(r'{}')\nbad = []\n",
        so.display()
    );
    for k in 0..TABLES {
        let last = table_len(k) - 1;
        py.push_str(&format!(
            "f = lib.t_{k}; f.restype = ctypes.c_int64; f.argtypes = [ctypes.c_int64]\n\
             if f(0) != {} or f({last}) != {}: bad.append(({k}, f(0), f({last})))\n",
            elem(k, 0),
            elem(k, last)
        ));
    }
    py.push_str("assert not bad, bad\nprint('ok', len(bad))\n");
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("spawn python3");
    assert!(
        out.status.success(),
        "accessor values wrong in {}:\nstdout: {}\nstderr: {}",
        so.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn const_tables_build_to_one_shared_library() {
    let mindc = common::require_mindc();
    let dir = common::scratch_dir("const_table_build_determinism");
    let src = dir.join("tables.mind");
    std::fs::write(&src, source()).expect("write source");

    let sos: Vec<PathBuf> = (0..BUILDS)
        .map(|n| dir.join(format!("tables_{n}.so")))
        .collect();
    let mut builds: Vec<Vec<u8>> = Vec::with_capacity(BUILDS);
    for so in &sos {
        let Some(bytes) = build(&mindc, &src, so) else {
            return;
        };
        builds.push(bytes);
    }
    for so in &sos {
        check_values(so);
    }

    let distinct: Vec<usize> = (0..BUILDS)
        .filter(|&n| builds[..n].iter().all(|b| *b != builds[n]))
        .collect();
    let sizes: Vec<usize> = builds.iter().map(Vec::len).collect();
    assert_eq!(
        distinct.len(),
        1,
        "{BUILDS} builds of one const-table program produced {} distinct shared \
         libraries (first occurrences at builds {distinct:?}, sizes {sizes:?}) — \
         the artifact depends on something other than the source",
        distinct.len()
    );
    eprintln!(
        "const_table_build_determinism: {BUILDS} builds, 1 distinct artifact, {} bytes",
        sizes[0]
    );
}
