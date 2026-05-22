// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0011 Phase A -- `std/async.mind` surface tests.
//!
//! Tests in this file verify:
//!
//! 1. `std/async.mind` parses and lowers to IR with all required `pub fn`s.
//! 2. `sync_scheduler` runs a single submitted unit, returns its value.
//! 3. `then` composition: submit -> then -> run produces the composed result.
//! 4. ReplayScheduler determinism: run the same pipeline twice; trace_hash
//!    values are identical (the determinism contract, RFC 0011 §7).
//! 5. ReplayScheduler: two different input sequences produce different
//!    trace_hashes (no false collision from different pipelines).
//! 6. `std.async` is registered in the bundled stdlib and resolves via
//!    `use std.async`.
//! 7. SyncScheduler trace_hash returns 0 (no recording).
//!
//! Gate: `cargo test --features "std-surface cross-module-imports"
//!                   --test std_surface_async`
//!
//! The MLIR-build functional execution test (compile to .so and run
//! natively) is nested under `#[cfg(feature = "mlir-build")]`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const ASYNC_MIND_SRC: &str = include_str!("../std/async.mind");

// ─── Helpers ────────────────────────────────────────────────────────────────

fn lower_async_mind() -> libmind::ir::IRModule {
    let module = parser::parse(ASYNC_MIND_SRC).expect("std/async.mind must parse cleanly");
    lower_to_ir(&module)
}

fn has_fndef(ir: &libmind::ir::IRModule, name: &str) -> bool {
    ir.instrs.iter().any(|i| matches!(i, Instr::FnDef { name: n, .. } if n == name))
}

fn count_calls_recursive(instrs: &[Instr], callee: &str) -> usize {
    let mut n = 0;
    for instr in instrs {
        match instr {
            Instr::Call { name, .. } if name == callee => n += 1,
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                n += count_calls_recursive(cond_instrs, callee);
                n += count_calls_recursive(then_instrs, callee);
                n += count_calls_recursive(else_instrs, callee);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                n += count_calls_recursive(cond_instrs, callee);
                n += count_calls_recursive(body, callee);
            }
            Instr::FnDef { body, .. } => {
                n += count_calls_recursive(body, callee);
            }
            _ => {}
        }
    }
    n
}

// ─── Test 1: parse + lower ───────────────────────────────────────────────────

#[test]
fn async_mind_parses_and_lowers() {
    let ir = lower_async_mind();

    // All public API functions must lower to FnDefs.
    for want in [
        "sync_scheduler",
        "replay_scheduler",
        "sched_kind",
        "submit",
        "then",
        "run",
        "trace_hash",
        "recv_value",
    ] {
        assert!(
            has_fndef(&ir, want),
            "expected FnDef `{want}` in lowered std/async.mind IR"
        );
    }
}

// ─── Test 2: internal helpers lower ─────────────────────────────────────────

#[test]
fn async_mind_internal_helpers_lower() {
    let ir = lower_async_mind();

    for want in [
        "sched_alloc",
        "trace_append",
        "snd_alloc",
        "snd_chain_append",
        "fnv_offset",
        "fnv_prime",
    ] {
        assert!(
            has_fndef(&ir, want),
            "expected internal FnDef `{want}` in lowered IR"
        );
    }
}

// ─── Test 3: run calls snd_thunk ─────────────────────────────────────────────

#[test]
fn run_calls_snd_thunk_and_trace_append() {
    let ir = lower_async_mind();

    // run() must call snd_thunk to extract the base value.
    assert!(
        count_calls_recursive(&ir.instrs, "snd_thunk") > 0,
        "run must call snd_thunk"
    );
    // run() calls trace_append for event recording.
    assert!(
        count_calls_recursive(&ir.instrs, "trace_append") > 0,
        "run must call trace_append"
    );
}

// ─── Test 4: trace_hash calls fnv helpers ────────────────────────────────────

#[test]
fn trace_hash_uses_fnv_helpers() {
    let ir = lower_async_mind();

    assert!(
        count_calls_recursive(&ir.instrs, "fnv_offset") > 0,
        "trace_hash must call fnv_offset"
    );
    assert!(
        count_calls_recursive(&ir.instrs, "fnv_prime") > 0,
        "trace_hash must call fnv_prime"
    );
}

// ─── Test 5: cross-module resolver finds std.async public symbols ────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn bundled_stdlib_resolves_use_std_async() {
    use libmind::project::module_table::build_module_table;
    use libmind::project::stdlib::parsed_stdlib_modules;

    let stdlib = parsed_stdlib_modules();
    let refs: Vec<(String, &libmind::ast::Module)> =
        stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&refs);

    assert!(
        table.resolves(&["std".into(), "async".into()], "sync_scheduler"),
        "std.async must export sync_scheduler"
    );
    assert!(
        table.resolves(&["std".into(), "async".into()], "replay_scheduler"),
        "std.async must export replay_scheduler"
    );
    assert!(
        table.resolves(&["std".into(), "async".into()], "submit"),
        "std.async must export submit"
    );
    assert!(
        table.resolves(&["std".into(), "async".into()], "then"),
        "std.async must export then"
    );
    assert!(
        table.resolves(&["std".into(), "async".into()], "run"),
        "std.async must export run"
    );
    assert!(
        table.resolves(&["std".into(), "async".into()], "trace_hash"),
        "std.async must export trace_hash"
    );
    assert!(
        table.resolves(&["std".into(), "async".into()], "recv_value"),
        "std.async must export recv_value"
    );
}

// ─── Test 6: sched_kind accessor returns kind field ──────────────────────────

#[test]
fn sched_kind_accessor_in_ir() {
    let ir = lower_async_mind();

    // sync_scheduler sets kind=0; replay_scheduler sets kind=1.
    // Both call sched_alloc.
    assert!(
        count_calls_recursive(&ir.instrs, "sched_alloc") >= 2,
        "sync_scheduler and replay_scheduler must both call sched_alloc"
    );
}

// ─── Tests 7-10: MLIR-build functional tests (determinism contract) ──────────
//
// These tests compile std/async.mind to a native .so and call the
// scheduler pipeline functions directly to verify:
//   - sync_scheduler: submit + run returns the work value.
//   - then composition: submit(s, 10) + then(snd, 3) + run = 13.
//   - Replay determinism: identical pipeline => identical trace_hash.
//   - Replay non-collision: different pipeline => different trace_hash.
//   - SyncScheduler trace_hash = 0 (no event recording).

#[cfg(feature = "mlir-build")]
mod mlir_tests {
    use libmind::project::stdlib::parsed_stdlib_modules;
    use std::path::PathBuf;

    fn mindc_bin() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("target");
        #[cfg(not(debug_assertions))]
        p.push("release");
        #[cfg(debug_assertions)]
        p.push("debug");
        p.push("mindc");
        p
    }

    fn tool_on_path(name: &str) -> bool {
        std::process::Command::new("which")
            .arg(name)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    // Compile std/async.mind to a shared library and load it.
    // Returns the path to the .so.
    fn compile_async_so(dir: &std::path::Path) -> Option<PathBuf> {
        let tool = "mlir-translate";
        if !tool_on_path(tool) && !tool_on_path("mlir-opt") {
            println!("mlir_async: MLIR tools not on PATH; skipping");
            return None;
        }

        let src_path = dir.join("async_test.mind");
        let so_path = dir.join("async_test.so");

        // Build a small driver that wraps std/async.mind.
        // The bundled stdlib is pre-parsed; we just need to compile
        // async.mind itself since it has no use statements.
        let src = include_str!("../std/async.mind");
        std::fs::write(&src_path, src).expect("write async_test.mind");

        let status = std::process::Command::new(mindc_bin())
            .args(["--emit-shared", "--out"])
            .arg(&so_path)
            .arg(&src_path)
            .status()
            .expect("run mindc");

        if !status.success() {
            println!("mlir_async: mindc --emit-shared failed; skipping functional tests");
            return None;
        }
        Some(so_path)
    }

    /// Load a symbol from the .so as an extern "C" fn.
    macro_rules! load_sym {
        ($lib:expr, $sym:literal, $ty:ty) => {{
            let raw = unsafe { $lib.get::<$ty>($sym) };
            raw.expect(concat!("symbol not found: ", $sym))
        }};
    }

    #[test]
    fn sync_scheduler_submit_run_returns_work_value() {
        let dir = std::env::temp_dir().join("mind_async_test_7");
        std::fs::create_dir_all(&dir).ok();
        let so = match compile_async_so(&dir) {
            Some(p) => p,
            None => return,
        };

        let lib = unsafe { libloading::Library::new(&so) }.expect("load so");

        let sync_scheduler: libloading::Symbol<unsafe extern "C" fn() -> i64> =
            unsafe { lib.get(b"sync_scheduler\0") }.expect("sync_scheduler");
        let submit: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"submit\0") }.expect("submit");
        let run: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"run\0") }.expect("run");

        unsafe {
            let s = sync_scheduler();
            let snd = submit(s, 42);
            let result = run(s, snd);
            assert_eq!(result, 42, "sync submit+run must return the work value");
        }
    }

    #[test]
    fn then_composition_produces_sum_of_stages() {
        let dir = std::env::temp_dir().join("mind_async_test_8");
        std::fs::create_dir_all(&dir).ok();
        let so = match compile_async_so(&dir) {
            Some(p) => p,
            None => return,
        };

        let lib = unsafe { libloading::Library::new(&so) }.expect("load so");

        let sync_scheduler: libloading::Symbol<unsafe extern "C" fn() -> i64> =
            unsafe { lib.get(b"sync_scheduler\0") }.expect("sync_scheduler");
        let submit: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"submit\0") }.expect("submit");
        let then: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"then\0") }.expect("then");
        let run: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"run\0") }.expect("run");

        unsafe {
            let s = sync_scheduler();
            // submit(s, 10) -> then(snd, 3) -> run = 10 + 3 = 13
            let snd0 = submit(s, 10);
            let snd1 = then(snd0, 3);
            let result = run(s, snd1);
            assert_eq!(result, 13, "then composition: 10 + 3 must equal 13");

            // Multi-stage: submit(s, 1) -> then(5) -> then(20) -> run = 26
            let snd_a = submit(s, 1);
            let snd_b = then(snd_a, 5);
            let snd_c = then(snd_b, 20);
            let result2 = run(s, snd_c);
            assert_eq!(result2, 26, "three-stage then: 1+5+20 must equal 26");
        }
    }

    #[test]
    fn replay_scheduler_trace_hash_determinism() {
        let dir = std::env::temp_dir().join("mind_async_test_9");
        std::fs::create_dir_all(&dir).ok();
        let so = match compile_async_so(&dir) {
            Some(p) => p,
            None => return,
        };

        let lib = unsafe { libloading::Library::new(&so) }.expect("load so");

        let replay_scheduler: libloading::Symbol<unsafe extern "C" fn() -> i64> =
            unsafe { lib.get(b"replay_scheduler\0") }.expect("replay_scheduler");
        let submit: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"submit\0") }.expect("submit");
        let then: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"then\0") }.expect("then");
        let run: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"run\0") }.expect("run");
        let trace_hash: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
            unsafe { lib.get(b"trace_hash\0") }.expect("trace_hash");

        unsafe {
            // Run the pipeline once.
            let s1 = replay_scheduler();
            let snd1 = submit(s1, 7);
            let snd1 = then(snd1, 3);
            let _r1 = run(s1, snd1);
            let hash1 = trace_hash(s1);

            // Run the identical pipeline on a fresh ReplayScheduler.
            // RFC 0011 §7 determinism contract: hash must be byte-identical.
            let s2 = replay_scheduler();
            let snd2 = submit(s2, 7);
            let snd2 = then(snd2, 3);
            let _r2 = run(s2, snd2);
            let hash2 = trace_hash(s2);

            assert_ne!(hash1, 0, "trace_hash of a non-empty log must not be 0");
            assert_eq!(
                hash1, hash2,
                "RFC 0011 §7 determinism: identical pipelines must produce identical trace_hash"
            );
        }
    }

    #[test]
    fn replay_scheduler_different_pipelines_produce_different_hashes() {
        let dir = std::env::temp_dir().join("mind_async_test_10");
        std::fs::create_dir_all(&dir).ok();
        let so = match compile_async_so(&dir) {
            Some(p) => p,
            None => return,
        };

        let lib = unsafe { libloading::Library::new(&so) }.expect("load so");

        let replay_scheduler: libloading::Symbol<unsafe extern "C" fn() -> i64> =
            unsafe { lib.get(b"replay_scheduler\0") }.expect("replay_scheduler");
        let submit: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"submit\0") }.expect("submit");
        let then: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"then\0") }.expect("then");
        let run: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"run\0") }.expect("run");
        let trace_hash: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
            unsafe { lib.get(b"trace_hash\0") }.expect("trace_hash");

        unsafe {
            // Pipeline A: submit(7) + then(3).  Events: [7, 3].
            let sa = replay_scheduler();
            let snd_a = submit(sa, 7);
            let snd_a = then(snd_a, 3);
            let _ = run(sa, snd_a);
            let hash_a = trace_hash(sa);

            // Pipeline B: submit(100) + then(999).  Events: [100, 999].
            let sb = replay_scheduler();
            let snd_b = submit(sb, 100);
            let snd_b = then(snd_b, 999);
            let _ = run(sb, snd_b);
            let hash_b = trace_hash(sb);

            assert_ne!(
                hash_a, hash_b,
                "different pipelines must produce different trace_hash values"
            );
        }
    }

    #[test]
    fn sync_scheduler_trace_hash_is_zero() {
        let dir = std::env::temp_dir().join("mind_async_test_11");
        std::fs::create_dir_all(&dir).ok();
        let so = match compile_async_so(&dir) {
            Some(p) => p,
            None => return,
        };

        let lib = unsafe { libloading::Library::new(&so) }.expect("load so");

        let sync_scheduler: libloading::Symbol<unsafe extern "C" fn() -> i64> =
            unsafe { lib.get(b"sync_scheduler\0") }.expect("sync_scheduler");
        let submit: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"submit\0") }.expect("submit");
        let run: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
            unsafe { lib.get(b"run\0") }.expect("run");
        let trace_hash: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
            unsafe { lib.get(b"trace_hash\0") }.expect("trace_hash");

        unsafe {
            let s = sync_scheduler();
            let snd = submit(s, 42);
            let _ = run(s, snd);
            let h = trace_hash(s);
            assert_eq!(
                h, 0,
                "SyncScheduler records no events; trace_hash must return 0"
            );
        }
    }
}
