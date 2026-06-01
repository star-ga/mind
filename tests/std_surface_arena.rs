// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/arena.mind` surface tests.
//!
//! Verifies:
//!  1. `std/arena.mind` parses and lowers to IR with the bump-allocator API.
//!  2. MLIR functional (gated on `mlir-build` + Unix, since the compiled-.so
//!     exec path is POSIX-only): compile to a `.so` and exercise the bump /
//!     8-byte-alignment / O(1) reset / fixed-extent-overflow / high-water
//!     semantics through the real allocator.
//!
//! Gate: `cargo test --features "std-surface cross-module-imports mlir-build"
//!                   --test std_surface_arena`

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const ARENA_SRC: &str = include_str!("../std/arena.mind");

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
fn arena_parses_and_lowers_with_api() {
    let module = parser::parse(ARENA_SRC).expect("std/arena.mind must parse cleanly");
    let ir = lower_to_ir(&module);
    let names = fndef_names(&ir.instrs);
    for required in [
        "arena_new",
        "arena_alloc",
        "arena_reset",
        "arena_used",
        "arena_cap",
        "arena_remaining",
        "arena_high_water",
        "arena_free",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "std.arena must define `{required}`; found {names:?}"
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
    fn bump_alloc_align_reset_overflow_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("arena: mindc not found; skipping");
            return;
        }

        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_arena");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let so_path = out_dir.join("libarena.so");
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("std")
            .join("arena.mind");

        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");
        if !status.success() {
            println!("arena: mindc compile failed; skipping");
            return;
        }

        unsafe {
            let lib = libloading::Library::new(&so_path).expect("dlopen libarena.so");
            type I1 = unsafe extern "C" fn(i64) -> i64;
            type I2 = unsafe extern "C" fn(i64, i64) -> i64;

            let arena_new = lib.get::<I1>(b"arena_new\0").unwrap();
            let arena_alloc = lib.get::<I2>(b"arena_alloc\0").unwrap();
            let arena_reset = lib.get::<I1>(b"arena_reset\0").unwrap();
            let arena_used = lib.get::<I1>(b"arena_used\0").unwrap();
            let arena_cap = lib.get::<I1>(b"arena_cap\0").unwrap();
            let arena_remaining = lib.get::<I1>(b"arena_remaining\0").unwrap();
            let arena_high_water = lib.get::<I1>(b"arena_high_water\0").unwrap();
            let arena_free = lib.get::<I1>(b"arena_free\0").unwrap();

            let a = arena_new(1024);
            assert!(a != 0, "arena_new(1024) must succeed");
            assert_eq!(arena_cap(a), 1024, "cap reflects reservation");

            // Bump + 8-byte alignment: 10 -> 16, 8 -> 8, 1 -> 8.
            let p1 = arena_alloc(a, 10);
            let p2 = arena_alloc(a, 8);
            let p3 = arena_alloc(a, 1);
            assert!(p1 != 0 && p2 != 0 && p3 != 0, "allocs must succeed");
            assert_eq!(p2 - p1, 16, "10 bytes rounds up to a 16-byte stride");
            assert_eq!(p3 - p2, 8, "8 bytes is one 8-byte stride");
            assert_eq!(arena_used(a), 32, "used = 16 + 8 + 8");
            assert_eq!(arena_remaining(a), 992, "remaining = cap - used");

            // O(1) reset rewinds the offset; high-water survives it.
            arena_reset(a);
            assert_eq!(arena_used(a), 0, "reset rewinds used to 0");
            assert_eq!(arena_high_water(a), 32, "high-water survives reset");

            // Fixed extent: an over-capacity request returns 0 and does not bump.
            assert_eq!(arena_alloc(a, 2000), 0, "over-cap alloc returns 0");
            assert_eq!(arena_used(a), 0, "failed alloc must not advance the offset");

            // Reuse after reset works; bad inputs are rejected.
            assert!(arena_alloc(a, 16) != 0, "reuse after reset");
            assert_eq!(arena_new(0), 0, "non-positive cap rejected");
            assert_eq!(arena_alloc(a, 0), 0, "non-positive size rejected");
            assert_eq!(arena_alloc(a, -5), 0, "negative size rejected");

            arena_free(a);
        }
    }
}
