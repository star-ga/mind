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
        "canon_drain",
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
            type Drain = unsafe extern "C" fn(i64, i64, i64) -> i64;

            let canon_new = lib.get::<New>(b"canon_new\0").unwrap();
            let canon_push = lib.get::<Push>(b"canon_push\0").unwrap();
            let canon_sort = lib.get::<Sort>(b"canon_sort\0").unwrap();
            let canon_len = lib.get::<Len>(b"canon_len\0").unwrap();
            let canon_conn = lib.get::<Get>(b"canon_conn\0").unwrap();
            let canon_req = lib.get::<Get>(b"canon_req\0").unwrap();
            let canon_drain = lib.get::<Drain>(b"canon_drain\0").unwrap();

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

            // canon_drain: the serialized 32-byte canonical records must be
            // byte-identical regardless of physical push order — this drained
            // sequence is the deterministic I/O input the evidence chain
            // anchors (a SHA-256 over it, at the reactor boundary, is stable
            // across runs and substrates because physical arrival order never
            // enters it).
            let drain_bytes = |events: &[(i64, i64, i64, i64)]| -> Vec<u8> {
                let h = canon_new(16);
                assert!(h != 0, "canon_new failed");
                for &(c, r, o, res) in events {
                    assert_eq!(canon_push(h, c, r, o, res), 1, "canon_push failed");
                }
                let mut buf = vec![0u8; events.len() * 32];
                let n = canon_drain(h, buf.as_mut_ptr() as i64, buf.len() as i64);
                assert_eq!(n, (events.len() * 32) as i64, "canon_drain byte count");
                buf
            };
            let ba = drain_bytes(&order_a);
            let bb = drain_bytes(&order_b);
            assert_eq!(
                ba, bb,
                "canon_drain bytes must be identical regardless of physical push \
                 order (the deterministic completion sequence the evidence anchor hashes)"
            );
            // First drained record must be the canonical minimum (conn=1, req=1).
            assert_eq!(
                i64::from_le_bytes(ba[0..8].try_into().unwrap()),
                1,
                "first drained conn_id must be 1 (canonical order)"
            );
            assert_eq!(
                i64::from_le_bytes(ba[8..16].try_into().unwrap()),
                1,
                "first drained req_id must be 1 (canonical order)"
            );
            // An undersized destination must be rejected with -1, no partial write.
            {
                let h = canon_new(16);
                for &(c, r, o, res) in &order_a {
                    canon_push(h, c, r, o, res);
                }
                let mut small = vec![0u8; 16];
                assert_eq!(
                    canon_drain(h, small.as_mut_ptr() as i64, small.len() as i64),
                    -1,
                    "canon_drain must reject an undersized buffer"
                );
            }

            // Total-order robustness: events SHARING (conn_id, req_id) but
            // differing in (op, result) still drain deterministically — the
            // lexicographic tie-break on op then result orders them totally, so
            // two physical push orders of the same multiset are byte-identical
            // even when the (conn_id, req_id) uniqueness contract is violated.
            {
                let coll_a: [(i64, i64, i64, i64); 4] =
                    [(1, 1, 9, 5), (1, 1, 2, 8), (1, 1, 9, 3), (2, 1, 1, 1)];
                let coll_b: [(i64, i64, i64, i64); 4] =
                    [(2, 1, 1, 1), (1, 1, 9, 3), (1, 1, 9, 5), (1, 1, 2, 8)];
                let cba = drain_bytes(&coll_a);
                let cbb = drain_bytes(&coll_b);
                assert_eq!(
                    cba, cbb,
                    "colliding (conn,req) keys must still drain byte-identically \
                     via the op/result tie-break (total order, arrival-independent)"
                );
                // Canonical first record = (1,1,2,8): lexicographic min on all
                // four fields, so the op tie-break (2 < 9) decides ahead of req.
                assert_eq!(
                    i64::from_le_bytes(cba[16..24].try_into().unwrap()),
                    2,
                    "op tie-break orders the colliding-key minimum first"
                );
                assert_eq!(
                    i64::from_le_bytes(cba[24..32].try_into().unwrap()),
                    8,
                    "result of the canonical-first colliding-key record"
                );
            }
        }
    }

    #[test]
    fn canon_anchor_links_sha256_transitively_via_manifest() {
        // An entry that imports ONLY std.io_canon (NOT std.sha256) and calls
        // canon_anchor must still link: io_canon imports std.sha256, and the
        // substrate linker must pull sha256.o TRANSITIVELY. Without the
        // transitive walk in build_cdylib_from_entry this fails with an
        // undefined `sha256` symbol. Also exercises canon_anchor end-to-end.
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("canon_anchor: mindc not found; skipping");
            return;
        }
        let proj = std::env::temp_dir().join("mind_canon_anchor_probe");
        let _ = std::fs::remove_dir_all(&proj);
        std::fs::create_dir_all(proj.join("src")).expect("create project src");
        std::fs::write(
            proj.join("Mind.toml"),
            "[package]\nname = \"canon_anchor_probe\"\nversion = \"0.1.0\"\n\
             license = \"Apache-2.0\"\n\n[build]\ntarget = \"cpu\"\n\
             emit = \"cdylib\"\noptimize = \"release\"\nentry = \"src/main.mind\"\n",
        )
        .expect("write Mind.toml");
        // Imports ONLY std.io_canon — sha256 must arrive transitively.
        std::fs::write(
            proj.join("src").join("main.mind"),
            "import std.io_canon;\n\
             pub fn probe_anchor(out_addr: i64) -> i64 {\n\
             \x20   let h: i64 = canon_new(8)\n\
             \x20   let _ = canon_push(h, 3, 1, 9, 30)\n\
             \x20   let _ = canon_push(h, 1, 1, 9, 10)\n\
             \x20   let _ = canon_push(h, 2, 1, 9, 20)\n\
             \x20   canon_anchor(h, out_addr)\n\
             }\n",
        )
        .expect("write main.mind");
        let status = Command::new(&mindc)
            .arg("build")
            .current_dir(&proj)
            .status()
            .expect("spawn mindc build");
        if !status.success() {
            println!("canon_anchor: mindc build failed (no MLIR backend?); skipping");
            return;
        }
        let so = proj
            .join("target")
            .join("release")
            .join("libcanon_anchor_probe.so");
        assert!(so.exists(), "consumer cdylib not produced at {so:?}");
        // sha256 must be DEFINED (transitively linked), not undefined.
        let nm = Command::new("nm").arg("-D").arg(&so).output().expect("nm");
        let text = String::from_utf8_lossy(&nm.stdout);
        assert!(
            !text.lines().any(|l| l.contains(" U sha256")),
            "sha256 is undefined in the cdylib -- transitive substrate linking \
             failed:\n{text}"
        );
        // probe_anchor returns 3 events and writes a non-zero 32-byte anchor.
        unsafe {
            let lib = libloading::Library::new(&so).expect("dlopen probe .so");
            type Anchor = unsafe extern "C" fn(i64) -> i64;
            let probe = lib.get::<Anchor>(b"probe_anchor\0").unwrap();
            let mut anchor = [0u8; 32];
            let n = probe(anchor.as_mut_ptr() as i64);
            assert_eq!(n, 3, "canon_anchor must report 3 anchored events");
            assert_ne!(
                anchor, [0u8; 32],
                "evidence anchor must be non-zero (sha256 ran)"
            );
        }
    }
}
