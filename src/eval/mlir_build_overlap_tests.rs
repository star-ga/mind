// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The runtime-support object is compiled on a thread that overlaps the
//! mlir-opt / mlir-translate leg of `build_all_with_objects`. These tests pin
//! that the overlap is output-neutral: the shared library equals the one the
//! historical serial order (MLIR leg, then runtime object, then link) produces.

use super::*;

const ADD_MLIR: &str = "module {
  func.func @add(%a: i64, %b: i64) -> i64 {
    %c = arith.addi %a, %b : i64
    return %c : i64
  }
}
";

/// The MLIR toolchain, or `None` on a host without it. Under
/// `MIND_BENCH_REQUIRE=1` a missing toolchain is a failure, never a skip.
fn tools_or_skip() -> Option<BuildTools> {
    match resolve_tools() {
        Ok(tools) => Some(tools),
        Err(err) => {
            assert!(
                std::env::var("MIND_BENCH_REQUIRE").as_deref() != Ok("1"),
                "MIND_BENCH_REQUIRE=1 forbids a toolchain skip: {err}"
            );
            eprintln!("mlir_build overlap_tests: ran=0 (toolchain absent: {err})");
            None
        }
    }
}

#[test]
fn cache_switch_only_exact_zero_disables() {
    assert!(runtime_obj_cache_enabled(None));
    assert!(runtime_obj_cache_enabled(Some("1")));
    assert!(runtime_obj_cache_enabled(Some("")));
    assert!(runtime_obj_cache_enabled(Some("off")));
    assert!(!runtime_obj_cache_enabled(Some("0")));
    assert!(!runtime_obj_cache_enabled(Some(" 0\n")));
}

#[test]
fn overlapped_shared_build_matches_serial_reference() {
    let Some(tools) = tools_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let overlapped = dir.path().join("overlapped.so");
    let serial = dir.path().join("serial.so");
    let base = BuildOptions {
        preset: "core",
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: None,
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };

    let shared_opts = BuildOptions {
        emit_shared: Some(&overlapped),
        ..base
    };
    let products = build_all(ADD_MLIR, &tools, &shared_opts).expect("overlapped build");

    // Historical order, step by step: MLIR leg first, then the runtime
    // object, then the link with that object first in the extra list.
    let ir_only = build_all(ADD_MLIR, &tools, &base).expect("MLIR leg");
    assert_eq!(products.llvm_ir, ir_only.llvm_ir, "MLIR leg diverged");
    let runtime_obj = compile_runtime_support_obj(&tools, None, ADD_MLIR).expect("runtime object");
    let extra = vec![runtime_obj.path().to_path_buf()];
    run_clang_codegen(&ir_only.llvm_ir, &tools, None, &serial, true, &extra).expect("serial link");

    let a = std::fs::read(&overlapped).expect("read overlapped .so");
    let b = std::fs::read(&serial).expect("read serial .so");
    assert!(a.starts_with(b"\x7fELF"), "overlapped output is not an ELF");
    // Positive control: the runtime object is linked in full, not dropped.
    assert!(
        a.windows(b"__mind_load_i64".len())
            .any(|w| w == b"__mind_load_i64"),
        "runtime-support symbols missing from the overlapped .so"
    );
    assert!(a == b, "overlapped build differs from the serial reference");
}

fn object_has_symbol(path: &std::path::Path, name: &str) -> bool {
    let bytes = std::fs::read(path).expect("read object");
    let needle = name.as_bytes();
    bytes.windows(needle.len()).any(|window| window == needle)
}

#[test]
fn sys_twin_object_follows_mlir_text() {
    let Some(tools) = tools_or_skip() else {
        return;
    };
    let without = compile_runtime_support_obj(&tools, None, ADD_MLIR).expect("twin-free object");
    let with = compile_runtime_support_obj(
        &tools,
        None,
        "func.func private @__mind_sys_getuid() -> i64\n",
    )
    .expect("twin object");
    assert!(
        !object_has_symbol(without.path(), "__mind_sys_getuid"),
        "twin-free runtime object still defines __mind_sys_getuid"
    );
    assert!(
        object_has_symbol(with.path(), "__mind_sys_getuid"),
        "MLIR that names __mind_sys_ did not compile the twin"
    );
    assert!(
        object_has_symbol(without.path(), "__mind_load_i64"),
        "twin-free runtime object dropped an unrelated export"
    );
}

#[test]
fn sys_twin_flag_changes_runtime_obj_cache_key() {
    let Some(tools) = tools_or_skip() else {
        return;
    };
    let plain = vec!["-c".to_string(), "-O2".to_string()];
    let mut flagged = plain.clone();
    flagged.push("-DMIND_RAW_SYSCALLS=1".to_string());
    let bare = runtime_obj_cache_path(&tools, &plain);
    let marked = runtime_obj_cache_path(&tools, &flagged);
    if let (Some(bare), Some(marked)) = (bare, marked) {
        assert_ne!(
            bare, marked,
            "runtime-object cache key ignored the syscall define"
        );
    }
}

#[test]
fn linked_object_scan_matches_the_symbol_exactly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let named = dir.path().join("named.o");
    let plain = dir.path().join("plain.o");
    std::fs::write(&named, b"\0__mind_sys_getuid\0").expect("write named");
    std::fs::write(&plain, b"\0__mind_sys\0__mind_load_i64\0").expect("write plain");
    assert!(linked_object_names_sys_twin(&named), "named object missed");
    assert!(
        !linked_object_names_sys_twin(&plain),
        "plain object matched"
    );
    assert!(
        linked_object_names_sys_twin(dir.path().join("absent.o")),
        "unreadable object must count as named"
    );
}
