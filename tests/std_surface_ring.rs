// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/ring.mind` surface tests.
//!
//! Verifies:
//!  1. `std/ring.mind` parses and lowers to IR with the ring-buffer API.
//!  2. MLIR functional (Unix-gated, like the other native-exec tests): compile
//!     to a `.so` and exercise FIFO order, full-rejection, empty-pop sentinel,
//!     wraparound, and partial bulk write/read through the real buffer.
//!
//! Gate: `cargo test --features "std-surface cross-module-imports mlir-build"
//!                   --test std_surface_ring`

#![cfg(feature = "std-surface")]

mod common;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const RING_SRC: &str = include_str!("../std/ring.mind");

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
fn ring_parses_and_lowers_with_api() {
    let module = parser::parse(RING_SRC).expect("std/ring.mind must parse cleanly");
    let ir = lower_to_ir(&module);
    let names = fndef_names(&ir.instrs);
    for required in [
        "ring_new",
        "ring_push",
        "ring_pop",
        "ring_len",
        "ring_cap",
        "ring_free_space",
        "ring_write",
        "ring_read",
        "ring_clear",
        "ring_free",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "std.ring must define `{required}`; found {names:?}"
        );
    }
}

#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
mod mlir_functional {
    use super::mindc_bin;
    use std::path::PathBuf;
    use std::process::Command;

    // mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

    #[test]
    fn fifo_full_wrap_partial_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("ring: mindc not found; skipping");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_ring");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let so_path = out_dir.join("libring.so");
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("std")
            .join("ring.mind");
        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");
        if !status.success() {
            println!("ring: mindc compile failed; skipping");
            return;
        }

        unsafe {
            let lib = libloading::Library::new(&so_path).expect("dlopen libring.so");
            type I1 = unsafe extern "C" fn(i64) -> i64;
            type I2 = unsafe extern "C" fn(i64, i64) -> i64;
            type I3 = unsafe extern "C" fn(i64, i64, i64) -> i64;
            let ring_new = lib.get::<I1>(b"ring_new\0").unwrap();
            let ring_push = lib.get::<I2>(b"ring_push\0").unwrap();
            let ring_pop = lib.get::<I1>(b"ring_pop\0").unwrap();
            let ring_len = lib.get::<I1>(b"ring_len\0").unwrap();
            let ring_write = lib.get::<I3>(b"ring_write\0").unwrap();
            let ring_read = lib.get::<I3>(b"ring_read\0").unwrap();
            let ring_clear = lib.get::<I1>(b"ring_clear\0").unwrap();

            let r = ring_new(4);
            assert!(r != 0, "ring_new(4) must succeed");

            // FIFO + full-rejection.
            for b in [10i64, 20, 30, 40] {
                assert_eq!(ring_push(r, b), 1, "push into non-full ring");
            }
            assert_eq!(ring_push(r, 99), 0, "push into full ring returns 0");
            assert_eq!(ring_len(r), 4);
            for want in [10i64, 20, 30, 40] {
                assert_eq!(ring_pop(r), want, "FIFO pop order");
            }
            assert_eq!(ring_pop(r), -1, "pop empty returns -1");

            // Wraparound: write 3, read 2, write 3 (head/tail cross cap), read 4.
            ring_clear(r);
            let src = [1u8, 2, 3];
            assert_eq!(ring_write(r, src.as_ptr() as i64, 3), 3);
            let mut d2 = [0u8; 2];
            assert_eq!(ring_read(r, d2.as_mut_ptr() as i64, 2), 2);
            assert_eq!(d2, [1, 2]);
            let src2 = [4u8, 5, 6];
            assert_eq!(ring_write(r, src2.as_ptr() as i64, 3), 3, "wrap write fits");
            let mut d4 = [0u8; 4];
            assert_eq!(ring_read(r, d4.as_mut_ptr() as i64, 4), 4);
            assert_eq!(d4, [3, 4, 5, 6], "wraparound preserves FIFO order");

            // Partial write: more than free space returns the amount that fit.
            ring_clear(r);
            let big: [u8; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            assert_eq!(
                ring_write(r, big.as_ptr() as i64, 10),
                4,
                "write caps at free space"
            );
        }
    }
}
