// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP` executable regression gate.
//!
//! A runtime out-of-bounds array index is a DETERMINISTIC BOUNDS TRAP — never a
//! clamp to `[0, len-1]`, never a silent wrong value. This gate compiles small
//! programs whose index is a genuine RUNTIME value (a function parameter, so the
//! compiler cannot fold it), runs the real artifact, and asserts:
//!   * OOB (negative, `== len`, and `> len`) exits with the pinned OOB code 77
//!     on BOTH the MLIR/clang backend AND the pure-MIND native-ELF backend
//!     (OOB_RUST_MLIR_NATIVE_PARITY);
//!   * an in-bounds index returns the correct element (the trap adds no
//!     behavioural change on the happy path);
//!   * OOB is NOT CLAMPED — a negative index does not return element 0 and a
//!     `>= len` index does not return element `len-1` (the exact regression a
//!     re-introduced `arith.maxsi/minsi` clamp would cause);
//!   * a COMPILE-TIME-PROVABLE OOB (a literal constant index) is REJECTED by
//!     `mindc check` (deterministic compile-time rejection, E2621), never lowered.
//!
//! The clamp lived in `src/mlir/lowering.rs` (`ArrayLoad`); it was replaced by a
//! call to the single trap authority `__mind_oob_check` (runtime-support), which
//! is emitted BEFORE the `tensor.extract` and whose result the extract consumes,
//! so the trap strictly dominates the memory access.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test array_oob_trap_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

const OOB_CODE: i32 = 77;

/// `at(k)` indexes a LOCAL `[i64;3]` by the runtime parameter `k`, so the index
/// is never a compile-time constant (never const-folded / never compile-rejected)
/// and always exercises the runtime bounds-trap path. `main` calls `at(<k>)`, so
/// the process exit is the in-range element for a valid `k`, or the trap for OOB.
fn runtime_src(k: i64) -> String {
    format!(
        "fn at(k: i64) -> i64 {{\n    let a: [i64; 3] = [10, 20, 30];\n    return a[k];\n}}\n\
         fn main() -> i64 {{ return at({k}); }}\n"
    )
}

/// Compile a cleanly-compiling program and run it; returns its real exit code.
/// `native` selects the pure-MIND native-ELF backend, else the MLIR/clang path.
/// Panics with the compiler diagnostics if the artifact is not produced (a
/// masked compile failure — build emits a JIT launcher stub — is caught here so
/// it can never be mistaken for a clean run).
fn build_and_run(mindc: &Path, src: &str, native: bool) -> i32 {
    let dir = common::scratch_dir("array_oob_trap_run").join(format!(
        "mind_oob_{}_{}",
        if native { "nat" } else { "mlir" },
        // vary by content so parallel cases never collide on one path
        {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            src.hash(&mut h);
            h.finish()
        }
    ));
    let _ = std::fs::create_dir_all(&dir);
    let srcf = dir.join("main.mind");
    let out = dir.join("p.bin");
    let _ = std::fs::remove_file(&out);
    std::fs::write(&srcf, src).expect("write src");

    let mut cmd = Command::new(mindc);
    cmd.arg("build");
    if native {
        cmd.arg("--backend=native");
    } else {
        cmd.args(["--backend=mlir", "--emit=binary"]);
    }
    // Run from the repo root so the native backend resolves `std/` modules.
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(format!("--out={}", out.display()))
        .arg(&srcf);
    let c = cmd.output().expect("run mindc build");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&c.stderr),
        String::from_utf8_lossy(&c.stdout)
    );
    // A build that could not natively compile emits a launcher stub, not a real
    // binary — treat that as a hard failure, never a clean run.
    assert!(
        !blob.contains("not natively compiled")
            && out.exists()
            && out.metadata().unwrap().len() > 0,
        "compile did not produce a real binary (native={native}):\n{blob}"
    );
    let mut perm = std::fs::metadata(&out).unwrap().permissions();
    perm.set_mode(0o755);
    std::fs::set_permissions(&out, perm).unwrap();

    let r = Command::new(&out).output().expect("run artifact");
    r.status.code().unwrap_or(-1)
}

/// True iff `mindc check` rejects the program with an out-of-bounds diagnostic.
fn check_rejects(mindc: &Path, src: &str) -> (bool, String) {
    let dir = common::scratch_dir("array_oob_trap_run").join(format!("mind_oob_chk_{}", src.len()));
    let _ = std::fs::create_dir_all(&dir);
    let srcf = dir.join("main.mind");
    std::fs::write(&srcf, src).expect("write src");
    let c = Command::new(mindc)
        .arg("check")
        .arg(&srcf)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run mindc check");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&c.stderr),
        String::from_utf8_lossy(&c.stdout)
    );
    (
        !c.status.success() && blob.to_lowercase().contains("out of bounds"),
        blob,
    )
}

#[test]
fn array_oob_is_a_deterministic_trap_not_a_clamp() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "array_oob_trap_run",
            "array-oob-trap: mindc not found; skipping",
        );
        return;
    }

    // Runtime OOB on BOTH backends -> pinned trap code, identical (parity).
    for k in [-1i64, 3, 5] {
        let src = runtime_src(k);
        let mlir = build_and_run(&mindc, &src, false);
        let native = build_and_run(&mindc, &src, true);
        assert_eq!(
            mlir, OOB_CODE,
            "MLIR OOB k={k} should trap {OOB_CODE}, got {mlir}"
        );
        assert_eq!(
            native, OOB_CODE,
            "native OOB k={k} should trap {OOB_CODE}, got {native}"
        );
        assert_eq!(mlir, native, "OOB_RUST_MLIR_NATIVE_PARITY broken at k={k}");
    }

    // In-bounds -> correct element, both backends (happy path unchanged).
    for (k, want) in [(0i64, 10i32), (1, 20), (2, 30)] {
        let src = runtime_src(k);
        assert_eq!(
            build_and_run(&mindc, &src, false),
            want,
            "MLIR in-bounds k={k}"
        );
        assert_eq!(
            build_and_run(&mindc, &src, true),
            want,
            "native in-bounds k={k}"
        );
    }

    // NOT-CLAMPED: a clamp would make a[-1] read a[0]==10 and a[5] read
    // a[len-1]==30. The trap makes both 77 instead — assert they are NOT the
    // clamped values on either backend.
    for native in [false, true] {
        let neg = build_and_run(&mindc, &runtime_src(-1), native);
        let high = build_and_run(&mindc, &runtime_src(5), native);
        assert_ne!(
            neg, 10,
            "negative index was CLAMPED to a[0] (native={native})"
        );
        assert_ne!(
            high, 30,
            "high index was CLAMPED to a[len-1] (native={native})"
        );
    }

    // Compile-time-provable OOB (literal index) -> deterministic compile-time
    // rejection, never a runnable artifact.
    for k in [3i64, 5, -1] {
        let src = format!(
            "fn main() -> i64 {{\n    let a: [i64; 3] = [10, 20, 30];\n    return a[{k}];\n}}\n"
        );
        let (rejected, blob) = check_rejects(&mindc, &src);
        assert!(
            rejected,
            "constant OOB a[{k}] must be rejected by `mindc check`:\n{blob}"
        );
    }
}

/// Array literal elements that are integer constant expressions (spec 3.2
/// fold): `[0 - 1, ...]` builds and runs with the same value on both routes as
/// the literal `[-1, ...]`. An index held in a `let mut` loop variable reads
/// each element (positive control), and stepping it past the end traps 77.
#[test]
fn folded_array_elements_run_on_both_routes() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "array_oob_trap_run",
            "array-oob-trap: mindc not found; skipping",
        );
        return;
    }
    let cases: [(&str, i32); 6] = [
        // Control: the literal spelling.
        (
            "fn main() -> i64 { let a = [-1, 2, 3]; return a[0] + 10; }\n",
            9,
        ),
        (
            "fn main() -> i64 { let a = [0 - 1, 2, 3]; return a[0] + 10; }\n",
            9,
        ),
        (
            "fn main() -> i64 {\n    let a: [i64; 3] = [5, 0 - 1, (2 * 3) - 7];\n    return a[0] + a[1] + a[2] + 100;\n}\n",
            103,
        ),
        // A `let mut` index inside a loop reads every element: 5 - 1 + 3 + 2 = 9.
        (
            "fn main() -> i64 {\n    let a = [5, 0 - 1, 3, 4 / 2];\n    let mut s = 0;\n    let mut i = 0;\n    while i < 4 {\n        s = s + a[i];\n        i = i + 1;\n    }\n    return s;\n}\n",
            9,
        ),
        // The same index stepped one past the end is the bounds trap.
        (
            "fn main() -> i64 {\n    let a = [5, 0 - 1, 3];\n    let mut s = 0;\n    let mut i = 0;\n    while i < 4 {\n        s = s + a[i];\n        i = i + 1;\n    }\n    return s;\n}\n",
            OOB_CODE,
        ),
        // A negative runtime index traps; it never reads a folded element.
        (
            "fn at(k: i64) -> i64 { let a = [0 - 7, 8]; return a[k]; }\nfn main() -> i64 { return at(0 - 1); }\n",
            OOB_CODE,
        ),
    ];
    for (src, want) in cases {
        let mlir = build_and_run(&mindc, src, false);
        let native = build_and_run(&mindc, src, true);
        assert_eq!(mlir, want, "MLIR route:\n{src}");
        assert_eq!(native, want, "native route:\n{src}");
    }
}

/// A typed dense tensor literal stores the folded element value: the old
/// encoder stored 0 for `0 - 1` (a silent wrong value). An element with no
/// constant value is refused by name on the MLIR route, never stored as 0.
/// Route A refuses the tensor annotation by name (`lane.type.tensor`).
#[test]
fn dense_tensor_literal_stores_folded_values_or_refuses() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "array_oob_trap_run",
            "array-oob-trap: mindc not found; skipping",
        );
        return;
    }
    let folded =
        "fn main() -> i64 {\n    let t: tensor<i64[2]> = [0 - 1, 2];\n    return t[0] + 10;\n}\n";
    // 10 + (-1) = 9; the old encoder gave 10 + 0 = 10.
    assert_eq!(
        build_and_run(&mindc, folded, false),
        9,
        "MLIR route:\n{folded}"
    );
    let runtime = "fn main() -> i64 {\n    let x: i64 = 3;\n    let t: tensor<i64[2]> = [x, 2];\n    return t[0] + 10;\n}\n";
    for (src, native, want) in [
        (runtime, false, "is not a constant"),
        (folded, true, "lane.type.tensor"),
    ] {
        let (built, blob) = build_only(&mindc, src, native);
        assert!(
            !built && blob.contains(want),
            "want refusal `{want}` (native={native}):\n{blob}"
        );
    }
}

/// Spec 3.3 on the CLI: every REFUSE row of the ruling's table exits non-zero
/// with `Q16ConstantOutOfRange` and writes no artifact; `as q16` and an inexact
/// integer fill refuse by name; a Q16.16 fill `1.5` emits Q(1.5) = 98304 and
/// never a `: q16` constant. Positive control: an exact integer fill runs.
#[test]
fn q16_and_integer_fill_constants_refuse_by_name() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "array_oob_trap_run",
            "array-oob-trap: mindc not found; skipping",
        );
        return;
    }
    let dense = |lit: &str| {
        format!("fn main() -> i64 {{\n    let t: Tensor<q16, [1]> = [{lit}];\n    return 0;\n}}\n")
    };
    let mut cases: Vec<(String, &str)> = [
        "32767.999992370605",
        "32768.0",
        "-32768.0000076294",
        "-32768.00001525879",
        "1e999",
        "-1e999",
        "1e+30",
        "-1e+30",
        "1.7976931348623157e+308",
    ]
    .iter()
    .map(|lit| (dense(lit), "Q16ConstantOutOfRange"))
    .collect();
    cases.push((
        "fn main() -> i64 {\n    return (1.5 as q16) as i64;\n}\n".into(),
        "Q16Unsupported",
    ));
    cases.push((
        "fn main() -> i64 {\n    let t: tensor<i32[2]> = 1.5;\n    return t[0];\n}\n".into(),
        "InexactIntegerFill",
    ));
    for (src, name) in &cases {
        let (built, blob) = build_only(&mindc, src, false);
        assert!(
            !built && blob.contains(name),
            "want refusal `{name}`:\n{src}\n{blob}"
        );
    }
    let exact = "fn main() -> i64 {\n    let t: tensor<i32[2]> = 3;\n    return t[0] + 10;\n}\n";
    assert_eq!(
        build_and_run(&mindc, exact, false),
        13,
        "integer fill control"
    );
    let dir = common::scratch_dir("array_oob_trap_run").join("mind_q16_fill_emit");
    let _ = std::fs::create_dir_all(&dir);
    let srcf = dir.join("main.mind");
    std::fs::write(
        &srcf,
        "fn main() -> i64 {\n    let t: Tensor<q16, [2]> = 1.5;\n    return 0;\n}\n",
    )
    .unwrap();
    let out = Command::new(&mindc)
        .arg(&srcf)
        .arg("--emit-mlir")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run mindc --emit-mlir");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && text.contains("98304") && !text.contains(": q16"),
        "Q16 fill 1.5 must emit 98304 and no `: q16`:\n{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// An i64 tensor fill passes through the IR's binary64 fill, so an integer
/// fill that binary64 rounds (2^53 + 1 became 2^53) must refuse by name with
/// no artifact; an integer binary64 holds builds and reads back exactly. Each
/// program returns `t[1] - fill + 10`, so the exact value exits 10.
#[test]
fn integer_fill_beyond_binary64_refuses_by_name() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "array_oob_trap_run",
            "array-oob-trap: mindc not found; skipping",
        );
        return;
    }
    let src = |fill: &str| {
        let back = match fill.strip_prefix('-') {
            Some(mag) => format!("+ {mag}"),
            None => format!("- {fill}"),
        };
        format!(
            "fn main() -> i64 {{\n    let t: tensor<i64[2]> = {fill};\n    return t[1] {back} + 10;\n}}\n"
        )
    };
    // 2^53, 2^53 - 1, -2^53, and i64::MIN + 2^53 (exact in binary64) build exactly.
    for fill in [
        "9007199254740992",
        "9007199254740991",
        "-9007199254740992",
        "-9214364837600034816",
    ] {
        assert_eq!(
            build_and_run(&mindc, &src(fill), false),
            10,
            "exact fill {fill}"
        );
    }
    // 2^53 + 1, i64::MAX, i64::MIN + 1, and the u64-range literal 2^63 + 1
    // (its i64 bits are i64::MIN + 1) have no binary64 value: refuse.
    for fill in [
        "9007199254740993",
        "9223372036854775807",
        "-9223372036854775807",
        "9223372036854775809",
    ] {
        let (built, blob) = build_only(&mindc, &src(fill), false);
        if built {
            let got = build_and_run(&mindc, &src(fill), false);
            panic!("fill {fill} built and exits {got} (exact is 10); want InexactIntegerFill");
        }
        assert!(blob.contains("InexactIntegerFill"), "fill {fill}:\n{blob}");
    }
}

/// `mindc --emit-ir` on a program: (exit 0, compiler output).
fn emit_ir(mindc: &Path, src: &str) -> (bool, String) {
    let dir =
        common::scratch_dir("array_oob_trap_run").join(format!("mind_nested_lit_{}", src.len()));
    let _ = std::fs::create_dir_all(&dir);
    let srcf = dir.join("main.mind");
    std::fs::write(&srcf, src).expect("write src");
    let c = Command::new(mindc)
        .arg(&srcf)
        .arg("--emit-ir")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run mindc --emit-ir");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&c.stderr),
        String::from_utf8_lossy(&c.stdout)
    );
    (c.status.success(), blob)
}

/// A nested constant literal of a dense binding compiles (the `linear_regression`
/// weights), and a ragged, miscounted or non-constant nest is refused by name on
/// the CLI with no IR text. The element values are exact in the IR test
/// `nested_dense_literals_flatten_row_major`.
#[test]
fn nested_dense_literal_compiles_or_refuses_by_name() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "array_oob_trap_run",
            "array-oob-trap: mindc not found; skipping",
        );
        return;
    }
    let prog = |ty: &str, lit: &str| {
        format!(
            "fn main() -> i64 {{\n    let x: i64 = 3;\n    let t: {ty} = {lit};\n    return 0;\n}}\n"
        )
    };
    let ok = [
        ("diff tensor<f32[4, 1]>", "[[0.1], [0.2], [-0.1], [0.3]]"),
        (
            "tensor<i64[2, 2, 2]>",
            "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
        ),
    ];
    for (ty, lit) in ok {
        let (built, blob) = emit_ir(&mindc, &prog(ty, lit));
        assert!(built, "{ty} {lit} must compile:\n{blob}");
    }
    let refused = [
        ("tensor<i64[2, 1]>", "[[x], [2]]", "NonConstantElement"),
        ("tensor<i64[2, 3]>", "[[1, 2], [3, 4]]", "DenseLeafCount"),
        ("tensor<i64[2, 2]>", "[[1, 2], [3]]", "RaggedNest"),
    ];
    for (ty, lit, name) in refused {
        let (built, blob) = emit_ir(&mindc, &prog(ty, lit));
        assert!(
            !built && blob.contains(name),
            "{ty} {lit}: want {name}:\n{blob}"
        );
    }
}

/// Build without running: (exit 0 or an artifact written, compiler output).
fn build_only(mindc: &Path, src: &str, native: bool) -> (bool, String) {
    let key = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        src.hash(&mut h);
        h.finish()
    };
    let dir =
        common::scratch_dir("array_oob_trap_run").join(format!("mind_oob_ref_{native}_{key}"));
    let _ = std::fs::create_dir_all(&dir);
    let (srcf, out) = (dir.join("main.mind"), dir.join("p.bin"));
    let _ = std::fs::remove_file(&out);
    std::fs::write(&srcf, src).expect("write src");
    let mut cmd = Command::new(mindc);
    cmd.arg("build");
    if native {
        cmd.arg("--backend=native");
    } else {
        cmd.args(["--backend=mlir", "--emit=binary"]);
    }
    let c = cmd
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(format!("--out={}", out.display()))
        .arg(&srcf)
        .output()
        .expect("run mindc build");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&c.stderr),
        String::from_utf8_lossy(&c.stdout)
    );
    // Any exit 0 or any file at --out counts as built: a refusal writes nothing.
    (c.status.success() || out.exists(), blob)
}

// The tree-evaluator's substrate representation of the SAME bounds-trap event —
// `EvalError::BoundsTrap` — is proven directly in a `src/eval/mod.rs` unit test
// (`eval_array_index_oob_is_bounds_trap`), which evaluates an out-of-bounds index
// and asserts the typed error. It is NOT exercised through `mindc test` here: the
// `mindc test` assert-condition evaluator has a pre-existing scope limitation
// (function-body locals are "unknown variable" inside an `assert` condition),
// orthogonal to the bounds-trap contract, that prevents reaching the index sites.
