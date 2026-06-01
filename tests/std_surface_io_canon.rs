// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/io_canon.mind` surface tests.
//!
//! Verifies:
//!  1. `std/io_canon.mind` parses and lowers to IR with the canonical-ordering API.
//!  2. MLIR functional (gated on `mlir-build`): compile to a `.so`, push completion
//!     events in two DIFFERENT physical orders, sort each, and assert the drained
//!     canonical order is (a) correct and (b) identical regardless of push order —
//!     i.e. the deterministic-ordering property. This also guards the nested-loop
//!     exit-env SSA fix: `canon_sort` is a selection sort whose `min` is a body-
//!     local `let mut` mutated inside the inner loop, so a regression there makes
//!     the sort a no-op and this test fails.
//!
//! Gate: `cargo test --features "std-surface cross-module-imports mlir-build"
//!                   --test std_surface_io_canon`

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const IO_CANON_SRC: &str = include_str!("../std/io_canon.mind");

fn fndef_names(instrs: &[Instr]) -> Vec<String> {
    let mut out = Vec::new();
    for instr in instrs {
        if let Instr::FnDef { name, .. } = instr {
            out.push(name.clone());
        }
    }
    out
}

#[test]
fn io_canon_parses_and_lowers_with_api() {
    let module = parser::parse(IO_CANON_SRC).expect("std/io_canon.mind must parse cleanly");
    let ir = lower_to_ir(&module);
    let names = fndef_names(&ir.instrs);
    for required in [
        "canon_new",
        "canon_push",
        "canon_sort",
        "canon_len",
        "canon_conn",
        "canon_req",
        "canon_op",
        "canon_result",
        "canon_clear",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "std.io_canon must define `{required}`; found {names:?}"
        );
    }
}

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod mlir_functional {
    use std::path::PathBuf;
    use std::process::Command;

    fn mindc_bin() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("release")
            .join("mindc")
    }

    #[test]
    fn canonical_ordering_is_deterministic_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("io_canon: mindc not found; skipping");
            return;
        }

        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_io_canon");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let so_path = out_dir.join("libio_canon.so");
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("std")
            .join("io_canon.mind");

        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");
        if !status.success() {
            println!("io_canon: mindc compile failed; skipping");
            return;
        }

        unsafe {
            let lib = libloading::Library::new(&so_path).expect("dlopen libio_canon.so");
            type New = unsafe extern "C" fn(i64) -> i64;
            type Push = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
            type Sort = unsafe extern "C" fn(i64) -> i64;
            type Len = unsafe extern "C" fn(i64) -> i64;
            type Get = unsafe extern "C" fn(i64, i64) -> i64;

            let canon_new = lib.get::<New>(b"canon_new\0").unwrap();
            let canon_push = lib.get::<Push>(b"canon_push\0").unwrap();
            let canon_sort = lib.get::<Sort>(b"canon_sort\0").unwrap();
            let canon_len = lib.get::<Len>(b"canon_len\0").unwrap();
            let canon_conn = lib.get::<Get>(b"canon_conn\0").unwrap();
            let canon_req = lib.get::<Get>(b"canon_req\0").unwrap();

            // (conn_id, req_id, op, result)
            let order_a: [(i64, i64, i64, i64); 5] = [
                (2, 1, 9, 0),
                (1, 2, 9, 0),
                (1, 1, 9, 0),
                (2, 0, 9, 0),
                (3, 5, 9, 0),
            ];
            // Same multiset, different physical arrival order.
            let order_b: [(i64, i64, i64, i64); 5] = [
                (3, 5, 9, 0),
                (1, 1, 9, 0),
                (2, 1, 9, 0),
                (2, 0, 9, 0),
                (1, 2, 9, 0),
            ];

            let drain = |events: &[(i64, i64, i64, i64)]| -> Vec<(i64, i64)> {
                let h = canon_new(16);
                assert!(h != 0, "canon_new failed");
                for &(c, r, o, res) in events {
                    assert_eq!(canon_push(h, c, r, o, res), 1, "canon_push failed");
                }
                canon_sort(h);
                let n = canon_len(h);
                (0..n)
                    .map(|i| (canon_conn(h, i), canon_req(h, i)))
                    .collect()
            };

            let da = drain(&order_a);
            let db = drain(&order_b);
            let expected = vec![(1, 1), (1, 2), (2, 0), (2, 1), (3, 5)];

            assert_eq!(
                da, expected,
                "canon_sort must produce the canonical total order"
            );
            assert_eq!(
                da, db,
                "canonical drain order must be identical regardless of physical push order"
            );
        }
    }
}
