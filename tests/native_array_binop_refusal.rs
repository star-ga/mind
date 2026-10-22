// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native backend refuses a binary operator whose operand is an array value.
//!
//! `[i64; N] + [i64; N]` (and the other binary operators) used to be admitted
//! by the frozen-profile fence. `mindc build --backend native` wrote an ELF
//! that ran and returned the wrong value. Element reads (`a[i] + b[i]`) are
//! scalars and stay admitted. This gate builds the probe programs and asserts
//! the named refusal `binop.array_operand`, no artifact, and that the element
//! program agrees with the MLIR backend and the tree evaluator.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{mindc_bin, scratch_dir};

const REFUSAL: &str = "out-of-profile construct: binop.array_operand";

const A1: &str = "\
fn main() -> i64 {
    let a: [i64; 1] = [7];
    let b: [i64; 1] = [8];
    let c = a + b;
    return c[0];
}
";

const A2: &str = "\
fn main() -> i64 {
    let a: [i64; 3] = [1, 2, 4];
    let b: [i64; 3] = [10, 20, 30];
    let c = a + b;
    if c[0] == 11 {
        if c[1] == 22 {
            if c[2] == 34 {
                return 1;
            }
        }
    }
    return 2;
}
";

const A3: &str = "\
fn main() -> i64 {
    let a: [i64; 4] = [3, 0, 0, 9];
    let b: [i64; 4] = [4, 0, 0, 8];
    let c = a + b;
    if c[0] == 7 {
        if c[3] == 17 {
            return 1;
        }
        return 3;
    }
    return 2;
}
";

const A4: &str = "\
fn main() -> i64 {
    let a = [5, 6];
    let b = [7, 8];
    let c = a + b;
    if c[0] == 12 {
        if c[1] == 14 {
            return 1;
        }
    }
    return 2;
}
";

const A5: &str = "\
fn main() -> i64 {
    let a: [i64; 3] = [1, 2, 4];
    let b: [i64; 3] = [10, 20, 30];
    if a[0] + b[0] == 11 {
        if a[2] + b[2] == 34 {
            return 1;
        }
    }
    return 2;
}
";

fn arith_src(op: &str) -> String {
    format!(
        "fn main() -> i64 {{\n    \
         let a: [i64; 2] = [8, 6];\n    \
         let b: [i64; 2] = [2, 3];\n    \
         let c = a {op} b;\n    \
         return c[0];\n}}\n"
    )
}

fn cmp_src(op: &str) -> String {
    format!(
        "fn main() -> i64 {{\n    \
         let a: [i64; 2] = [1, 2];\n    \
         let b: [i64; 2] = [1, 2];\n    \
         if a {op} b {{ return 1; }}\n    \
         return 0;\n}}\n"
    )
}

/// Scalar on the left, array on the right (`1 + a`).
fn rhs_arith_src(op: &str) -> String {
    format!(
        "fn main() -> i64 {{\n    \
         let a: [i64; 2] = [8, 6];\n    \
         let c = 1 {op} a;\n    \
         return c[0];\n}}\n"
    )
}

fn rhs_cmp_src(op: &str) -> String {
    format!(
        "fn main() -> i64 {{\n    \
         let a: [i64; 2] = [1, 2];\n    \
         if 1 {op} a {{ return 1; }}\n    \
         return 0;\n}}\n"
    )
}

struct Build {
    code: i32,
    blob: String,
    artifact: PathBuf,
}

fn build(native: bool, src: &str, label: &str) -> Build {
    let dir = scratch_dir("native_array_binop_refusal");
    let src_path = dir.join(format!("{label}.mind"));
    let artifact = dir.join(label);
    let _ = std::fs::remove_file(&artifact);
    std::fs::write(&src_path, src).expect("write source");

    let mut cmd = Command::new(mindc_bin());
    cmd.arg("build");
    if native {
        cmd.arg("--backend=native");
    } else {
        cmd.arg("--emit=binary");
    }
    let repo = env!("CARGO_MANIFEST_DIR");
    cmd.arg(format!("--out={}", artifact.display()))
        .arg(&src_path)
        .current_dir(repo)
        .env("MINDC_STD_DIR", format!("{repo}/std"))
        .env(
            "MINDC_NATIVE_ELF",
            format!("{repo}/examples/mindc_mind/testdata/selfhost_loop/stage1.elf"),
        );
    let out = cmd.output().expect("spawn mindc");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    Build {
        code: out.status.code().unwrap_or(-1),
        blob,
        artifact,
    }
}

fn assert_named_refusal(label: &str, src: &str) {
    let built = build(true, src, label);
    assert!(
        built.code != 0 && !built.artifact.exists() && built.blob.contains(REFUSAL),
        "{label}: native must refuse with `{REFUSAL}` and write no artifact \
         (code={} exists={})\n{}",
        built.code,
        built.artifact.exists(),
        built.blob
    );
}

fn run_elf(path: &Path) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = std::fs::metadata(path)
            .expect("artifact metadata")
            .permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(path, perm).expect("mark executable");
        let ran = Command::new(path).output().expect("run artifact");
        return ran.status.code().unwrap_or(-1);
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        panic!("executing a native artifact requires a unix host");
    }
}

#[test]
fn native_refuses_probe_array_adds() {
    assert!(mindc_bin().exists(), "mindc binary missing");
    assert_named_refusal("a1", A1);
    assert_named_refusal("a2", A2);
    assert_named_refusal("a3", A3);
    assert_named_refusal("a4", A4);
}

#[test]
fn native_refuses_every_array_binop() {
    assert!(mindc_bin().exists(), "mindc binary missing");
    for (i, op) in ["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"]
        .into_iter()
        .enumerate()
    {
        assert_named_refusal(&format!("arith{i}"), &arith_src(op));
    }
    for (i, op) in ["<", "<=", ">", ">=", "==", "!="].into_iter().enumerate() {
        assert_named_refusal(&format!("cmp{i}"), &cmp_src(op));
    }
    // The array is the right operand. A guard that inspects only the left
    // operand admits every program in these two loops.
    for (i, op) in ["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"]
        .into_iter()
        .enumerate()
    {
        assert_named_refusal(&format!("rhsarith{i}"), &rhs_arith_src(op));
    }
    for (i, op) in ["<", "<=", ">", ">=", "==", "!="].into_iter().enumerate() {
        assert_named_refusal(&format!("rhscmp{i}"), &rhs_cmp_src(op));
    }
}

#[test]
fn element_add_matches_mlir_and_evaluator() {
    assert!(mindc_bin().exists(), "mindc binary missing");
    // `eval_module` registers each `fn` and yields the last top-level value.
    // A function definition is not a call, so the same body is evaluated by
    // appending `main()`. Native and MLIR still compile `A5` unchanged.
    let eval_src = format!("{A5}main()\n");
    let module = libmind::parser::parse(&eval_src).expect("a5 parses");
    let evaluated = libmind::eval::eval_module(&module).expect("a5 evaluates");

    let native = build(true, A5, "a5_native");
    assert!(
        native.code == 0 && native.artifact.exists(),
        "element add must be admitted by the native backend\n{}",
        native.blob
    );
    assert!(
        !native.blob.contains("not natively compiled"),
        "native build wrote a launcher stub, not an executable\n{}",
        native.blob
    );
    let native_exit = run_elf(&native.artifact);

    let mlir = build(false, A5, "a5_mlir");
    assert!(
        mlir.code == 0 && mlir.artifact.exists(),
        "element add must build on the mlir backend\n{}",
        mlir.blob
    );
    let mlir_exit = run_elf(&mlir.artifact);

    assert_eq!(evaluated, 1, "evaluator");
    assert_eq!(native_exit, 1, "native exit");
    assert_eq!(mlir_exit, 1, "mlir exit");
    assert_eq!(native_exit as i64, evaluated);
    assert_eq!(mlir_exit as i64, evaluated);
}
