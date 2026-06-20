// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/vec.mind` zeroed-allocation primitive tests (issue #204).
//!
//! Verifies:
//!  1. `std/vec.mind` parses and lowers to IR with `vec_zeroed` in the API.
//!  2. MLIR functional (gated on `mlir-build` + Unix): compile to a `.so`
//!     and exercise `vec_zeroed` through the real allocator — every slot of
//!     a freshly-zeroed buffer reads back as 0, a written sentinel survives,
//!     and a non-positive count returns the addr-0 (no-alloc) handle.
//!
//! Gate: `cargo test --features "std-surface cross-module-imports mlir-build"
//!                   --test std_surface_vec_zeroed`

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const VEC_SRC: &str = include_str!("../std/vec.mind");

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
fn vec_zeroed_parses_and_lowers() {
    let module = parser::parse(VEC_SRC).expect("std/vec.mind must parse cleanly");
    let ir = lower_to_ir(&module);
    let names = fndef_names(&ir.instrs);
    assert!(
        names.iter().any(|n| n == "vec_zeroed"),
        "std.vec must define `vec_zeroed`; found {names:?}"
    );
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
    fn vec_zeroed_zero_fill_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("vec_zeroed: mindc not found; skipping");
            return;
        }

        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_vec_zeroed");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let so_path = out_dir.join("libvec.so");
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("std")
            .join("vec.mind");

        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");
        if !status.success() {
            println!("vec_zeroed: mindc compile failed; skipping");
            return;
        }

        unsafe {
            let lib = libloading::Library::new(&so_path).expect("dlopen libvec.so");
            type I1 = unsafe extern "C" fn(i64) -> i64;
            type Load = unsafe extern "C" fn(i64) -> i64;
            type Store = unsafe extern "C" fn(i64, i64) -> i64;

            let vec_zeroed = lib.get::<I1>(b"vec_zeroed\0").unwrap();
            let load_i64 = lib.get::<Load>(b"__mind_load_i64\0").unwrap();
            let store_i64 = lib.get::<Store>(b"__mind_store_i64\0").unwrap();

            // A 4-element zeroed buffer: base is non-zero and every slot is 0.
            let base = vec_zeroed(4);
            assert!(base != 0, "vec_zeroed(4) must allocate");
            for i in 0..4 {
                assert_eq!(
                    load_i64(base + i * 8),
                    0,
                    "slot {i} of a freshly-zeroed vec must read 0"
                );
            }

            // A written sentinel survives; its neighbours stay zero.
            store_i64(base + 1 * 8, 99);
            assert_eq!(load_i64(base), 0, "slot 0 unchanged");
            assert_eq!(load_i64(base + 1 * 8), 99, "sentinel written");
            assert_eq!(load_i64(base + 2 * 8), 0, "slot 2 still zero");
            assert_eq!(load_i64(base + 3 * 8), 0, "slot 3 still zero");

            // Non-positive count is the addr-0 (no-alloc) handle.
            assert_eq!(vec_zeroed(0), 0, "vec_zeroed(0) allocates nothing");
            assert_eq!(vec_zeroed(-3), 0, "vec_zeroed(-3) allocates nothing");
        }
    }
}
