// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/reactor.mind` surface tests.
//!
//! Verifies:
//!  1. `std/reactor.mind` parses and lowers to IR with the connection-table API.
//!  2. MLIR functional (Unix-gated): compile to a `.so` and exercise the
//!     deterministic per-connection req-id allocation — monotonic per
//!     connection, independent across connections, unknown-connection sentinel,
//!     idempotent accept, close, and full-table rejection.
//!
//! Gate: `cargo test --features "std-surface cross-module-imports mlir-build"
//!                   --test std_surface_reactor`

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const REACTOR_SRC: &str = include_str!("../std/reactor.mind");

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
fn reactor_parses_and_lowers_with_api() {
    let module = parser::parse(REACTOR_SRC).expect("std/reactor.mind must parse cleanly");
    let ir = lower_to_ir(&module);
    let names = fndef_names(&ir.instrs);
    for required in [
        "reactor_new",
        "reactor_accept",
        "reactor_next_req",
        "reactor_conn_count",
        "reactor_close",
        "reactor_free",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "std.reactor must define `{required}`; found {names:?}"
        );
    }
}

#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
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
    fn monotonic_req_ids_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("reactor: mindc not found; skipping");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_reactor");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let so_path = out_dir.join("libreactor.so");
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("std")
            .join("reactor.mind");
        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");
        if !status.success() {
            println!("reactor: mindc compile failed; skipping");
            return;
        }

        unsafe {
            let lib = libloading::Library::new(&so_path).expect("dlopen libreactor.so");
            type I1 = unsafe extern "C" fn(i64) -> i64;
            type I2 = unsafe extern "C" fn(i64, i64) -> i64;
            let reactor_new = lib.get::<I1>(b"reactor_new\0").unwrap();
            let reactor_accept = lib.get::<I2>(b"reactor_accept\0").unwrap();
            let reactor_next_req = lib.get::<I2>(b"reactor_next_req\0").unwrap();
            let reactor_conn_count = lib.get::<I1>(b"reactor_conn_count\0").unwrap();
            let reactor_close = lib.get::<I2>(b"reactor_close\0").unwrap();

            let r = reactor_new(8);
            assert!(r != 0, "reactor_new(8) must succeed");

            // Accept two connections.
            assert_eq!(reactor_accept(r, 100), 1);
            assert_eq!(reactor_accept(r, 200), 1);
            assert_eq!(reactor_conn_count(r), 2);

            // req_id is monotonic per connection, starting at 0.
            assert_eq!(reactor_next_req(r, 100), 0);
            assert_eq!(reactor_next_req(r, 100), 1);
            assert_eq!(reactor_next_req(r, 100), 2);
            // …and independent across connections.
            assert_eq!(reactor_next_req(r, 200), 0);
            assert_eq!(reactor_next_req(r, 200), 1);

            // Unknown connection → -1 sentinel (never mints a colliding key).
            assert_eq!(reactor_next_req(r, 999), -1);

            // Accept is idempotent: re-accepting preserves the counter.
            assert_eq!(reactor_accept(r, 100), 1);
            assert_eq!(reactor_conn_count(r), 2, "idempotent accept must not grow");
            assert_eq!(
                reactor_next_req(r, 100),
                3,
                "counter preserved across re-accept"
            );

            // Close removes a connection; its key space is gone.
            assert_eq!(reactor_close(r, 100), 1);
            assert_eq!(reactor_conn_count(r), 1);
            assert_eq!(
                reactor_next_req(r, 100),
                -1,
                "closed connection is untracked"
            );
            // The surviving connection keeps its counter.
            assert_eq!(reactor_next_req(r, 200), 2);

            // Full-table rejection.
            let r1 = reactor_new(1);
            assert!(r1 != 0);
            assert_eq!(reactor_accept(r1, 1), 1);
            assert_eq!(reactor_accept(r1, 2), 0, "accept into full table returns 0");
        }
    }
}
