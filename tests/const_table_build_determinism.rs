// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Byte-reproducibility gate for programs whose tensor constants become
//! module globals.
//!
//! Every case below is built to a shared library [`BUILDS`] times IN SEPARATE
//! PROCESSES with address-space randomisation left on, and every artifact must
//! be the same bytes.
//!
//! The defect: on the tensor-aware preset, `one-shot-bufferize` with function
//! boundaries hoists each `arith.constant dense<...> : tensor<...>` into a
//! private `memref.global`, in the order it visits functions — an order read
//! out of a pointer-keyed map, so it moved with ASLR. The MLIR text and the
//! mic@3 were identical across runs; only `mlir-opt`'s output moved.
//!
//! The cases cover both kinds of constant the bufferizer sees:
//!
//! * constants written in the program — module `const` tables read by their
//!   own accessors ([`const_tables_build_to_one_shared_library`]);
//! * constants that exist only AFTER `canonicalize` folds tensor arithmetic on
//!   constants: elementwise `+` of two module tables, elementwise `+` of two
//!   function-body tensor literals, an axis `sum` (keepdims) that lowers to a
//!   `tensor.from_elements` of folded scalars, and a `sum` of a folded
//!   elementwise result. Pre-declaring only the constants in the emitted text
//!   left every one of these to the pointer-ordered path.
//!
//! Positive control: every artifact is loaded and every accessor is called on
//! every index, so "identical" can never mean "identically empty" or
//! "identically wrong".
//!
//! Gate: `cargo test --release --features "mlir-build std-surface cross-module-imports"
//!                   --test const_table_build_determinism`

#![cfg(all(unix, feature = "mlir-build"))]

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Separate `mindc` processes per case. Each process gets fresh heap
/// addresses, so a pointer-ordered choice shows up as a second distinct hash
/// with overwhelming probability well before eight runs.
const BUILDS: usize = 8;

/// `(accessor, argument, expected i64 result)`.
type Check = (String, i64, i64);

/// Deterministic, distinct-per-table integer values.
fn ival(k: usize, i: usize) -> i64 {
    (((k * 41 + i) * 4729 + 7919 * k) % 100_000) as i64
}

/// Exact IEEE bit pattern the accessors return via `__mind_f64_to_bits`.
fn bits(v: f64) -> i64 {
    v.to_bits() as i64
}

fn ints(v: &[i64]) -> String {
    v.iter().map(i64::to_string).collect::<Vec<_>>().join(", ")
}

/// `{:?}` keeps the `.0` on integral values, which the literal needs.
fn floats(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Pinned left-to-right fold, the order the reduction lowering emits.
fn fold_sum(v: &[f64]) -> f64 {
    v.iter().skip(1).fold(v[0], |acc, x| acc + x)
}

/// Build `src` to `so`. `None` only for a tolerated capability gap (routed
/// through the fail-closed skip gate); any other failure panics.
fn build(mindc: &Path, src: &Path, so: &Path) -> Option<Vec<u8>> {
    let _ = std::fs::remove_file(so);
    let out = Command::new(mindc)
        .arg(src)
        .arg("--emit-shared")
        .arg(so)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("spawn mindc");
    if !common::gate::compiled("const_table_build_determinism", &out) {
        return None;
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Toolchain: mlir-opt ") && stderr.contains("(LLVM "),
        "the build did not record the toolchain it used:\n{stderr}"
    );
    let bytes = std::fs::read(so).unwrap_or_else(|e| panic!("read {}: {e}", so.display()));
    // A launcher stub is ~1.2 KiB; every case links real code plus the
    // runtime-support object.
    assert!(
        bytes.len() > 4096,
        "{} is {} bytes — not a real shared library",
        so.display(),
        bytes.len()
    );
    Some(bytes)
}

/// Call every checked accessor through ctypes and compare exactly.
fn check_values(so: &Path, checks: &[Check]) {
    let rows: Vec<String> = checks
        .iter()
        .map(|(f, a, e)| format!("({f:?}, {a}, {e})"))
        .collect();
    let py = format!(
        "import ctypes\nlib = ctypes.CDLL(r'{}')\nbad = []\n\
         for fn, a, exp in [{}]:\n    \
         f = getattr(lib, fn); f.restype = ctypes.c_int64; f.argtypes = [ctypes.c_int64]\n    \
         got = f(a)\n    \
         if got != exp: bad.append((fn, a, exp, got))\n\
         assert not bad, bad[:10]\nprint('ok', {})\n",
        so.display(),
        rows.join(", "),
        checks.len()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("spawn python3");
    assert!(
        out.status.success(),
        "accessor values wrong in {}:\nstdout: {}\nstderr: {}",
        so.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Build `source` [`BUILDS`] times, check every artifact, assert ONE artifact.
fn assert_one_artifact(case: &str, source: &str, checks: &[Check]) {
    let mindc = common::require_mindc();
    let dir = common::scratch_dir("const_table_build_determinism").join(case);
    std::fs::create_dir_all(&dir).expect("create case dir");
    let src = dir.join("prog.mind");
    std::fs::write(&src, source).expect("write source");

    let sos: Vec<PathBuf> = (0..BUILDS).map(|n| dir.join(format!("b{n}.so"))).collect();
    let mut builds: Vec<Vec<u8>> = Vec::with_capacity(BUILDS);
    for so in &sos {
        let Some(bytes) = build(&mindc, &src, so) else {
            return;
        };
        builds.push(bytes);
    }
    for so in &sos {
        check_values(so, checks);
    }

    let distinct: Vec<usize> = (0..BUILDS)
        .filter(|&n| builds[..n].iter().all(|b| *b != builds[n]))
        .collect();
    let sizes: Vec<usize> = builds.iter().map(Vec::len).collect();
    assert_eq!(
        distinct.len(),
        1,
        "{case}: {BUILDS} builds of one program produced {} distinct shared \
         libraries (first occurrences at builds {distinct:?}, sizes {sizes:?}) — \
         the artifact depends on something other than the source",
        distinct.len()
    );
    eprintln!(
        "const_table_build_determinism[{case}]: {BUILDS} builds, 1 distinct artifact, \
         {} bytes, {} value checks",
        sizes[0],
        checks.len()
    );
}

/// Twelve module tables of distinct lengths, one accessor each.
#[test]
fn const_tables_build_to_one_shared_library() {
    let (mut src, mut checks) = (String::new(), Vec::new());
    for k in 0..12 {
        let v: Vec<i64> = (0..40 + k).map(|i| ival(k, i)).collect();
        src.push_str(&format!(
            "const TAB{k}: [i64; {}] = [{}];\npub fn t_{k}(i: i64) -> i64 {{\n    TAB{k}[i]\n}}\n\n",
            v.len(),
            ints(&v)
        ));
        checks.extend(
            v.iter()
                .enumerate()
                .map(|(i, &e)| (format!("t_{k}"), i as i64, e)),
        );
    }
    assert_one_artifact("module_tables", &src, &checks);
}

/// Six pairs of module tables summed elementwise: the sum is a NEW constant
/// that only exists after `canonicalize` folds `A + B`.
#[test]
fn folded_module_table_sums_build_to_one_shared_library() {
    let (mut src, mut checks) = (String::new(), Vec::new());
    for k in 0..6 {
        let n = 4 + k % 3;
        let a: Vec<i64> = (0..n).map(|i| ival(600 + k, i) - 777).collect();
        let b: Vec<i64> = (0..n).map(|i| ival(700 + k, i)).collect();
        src.push_str(&format!(
            "const A{k}: [i64; {n}] = [{}];\nconst B{k}: [i64; {n}] = [{}];\n\
             pub fn ab_{k}(i: i64) -> i64 {{\n    let s = A{k} + B{k};\n    s[i]\n}}\n\n",
            ints(&a),
            ints(&b)
        ));
        checks.extend((0..n).map(|i| (format!("ab_{k}"), i as i64, a[i] + b[i])));
    }
    assert_one_artifact("folded_module_sums", &src, &checks);
}

/// Function-body f64 tensor literals added elementwise in eight functions.
#[test]
fn folded_tensor_literal_sums_build_to_one_shared_library() {
    let (mut src, mut checks) = (String::new(), Vec::new());
    for k in 0..8 {
        let n = 3 + k % 3;
        let t: Vec<f64> = (0..n).map(|i| ival(400 + k, i) as f64 / 8.0).collect();
        let u: Vec<f64> = (0..n).map(|i| ival(500 + k, i) as f64 / 16.0).collect();
        src.push_str(&format!(
            "pub fn fold_{k}(i: i64) -> i64 {{\n    let t: tensor<f64[{n}]> = [{}];\n    \
             let u: tensor<f64[{n}]> = [{}];\n    let s = t + u;\n    \
             return __mind_f64_to_bits(s[i]);\n}}\n\n",
            floats(&t),
            floats(&u)
        ));
        checks.extend((0..n).map(|i| (format!("fold_{k}"), i as i64, bits(t[i] + u[i]))));
    }
    assert_one_artifact("folded_literal_sums", &src, &checks);
}

/// `sum(keepdims)` lowers to a `tensor.from_elements` of scalar adds. Over a
/// 1-D constant the result has ONE element, so it folds to a splat that
/// `tensor.extract` then folds to a scalar — no global is ever created, and
/// this case was already reproducible before the fix. It stays as a positive
/// control that splitting the pipeline did not break the reduction path; the
/// multi-element form that DOES create a global needs a 2-D constant, which no
/// MIND literal lowers to today, so it is driven from MLIR in
/// [`folded_from_elements_mlir_builds_to_one_shared_library`].
#[test]
fn folded_from_elements_reductions_build_to_one_shared_library() {
    let (mut src, mut checks) = (String::new(), Vec::new());
    for k in 0..8 {
        let n = 3 + k % 4;
        let t: Vec<f64> = (0..n).map(|i| ival(800 + k, i) as f64 / 4.0).collect();
        src.push_str(&format!(
            "pub fn red_{k}(i: i64) -> i64 {{\n    let t: tensor<f64[{n}]> = [{}];\n    \
             let s = tensor.sum(t, axes=[0], keepdims=true);\n    \
             return __mind_f64_to_bits(s[i]);\n}}\n\n",
            floats(&t)
        ));
        checks.push((format!("red_{k}"), 0, bits(fold_sum(&t))));
    }
    assert_one_artifact("folded_from_elements", &src, &checks);
}

/// An axis `sum` of a folded elementwise result: two folds feed one value.
/// Same single-element splat property (and the same positive-control role) as
/// [`folded_from_elements_reductions_build_to_one_shared_library`].
#[test]
fn folded_sum_of_sums_build_to_one_shared_library() {
    let (mut src, mut checks) = (String::new(), Vec::new());
    for k in 0..8 {
        let n = 2 + k % 4;
        let t: Vec<f64> = (0..n).map(|i| ival(900 + k, i) as f64 / 2.0).collect();
        let u: Vec<f64> = (0..n).map(|i| ival(950 + k, i) as f64 / 32.0).collect();
        let s: Vec<f64> = t.iter().zip(&u).map(|(a, b)| a + b).collect();
        src.push_str(&format!(
            "pub fn ss_{k}(i: i64) -> i64 {{\n    let t: tensor<f64[{n}]> = [{}];\n    \
             let u: tensor<f64[{n}]> = [{}];\n    \
             let s = tensor.sum(t + u, axes=[0], keepdims=true);\n    \
             return __mind_f64_to_bits(s[i]);\n}}\n\n",
            floats(&t),
            floats(&u)
        ));
        checks.push((format!("ss_{k}"), 0, bits(fold_sum(&s))));
    }
    assert_one_artifact("folded_sum_of_sums", &src, &checks);
}

/// One function of the MLIR-level case: a 2-D constant reduced over axis 0 the
/// way the pinned reduction lowering emits it (scalar extracts, a fixed
/// `arith.addf` chain, `tensor.from_elements`), then indexed at runtime.
/// `canonicalize` folds the whole chain into a NEW `tensor<Mxf64>` constant.
fn axis_sum_fn(k: usize, rows: &[Vec<f64>]) -> String {
    let m = rows[0].len();
    let dense: Vec<String> = rows.iter().map(|r| format!("[{}]", floats(r))).collect();
    let mut f = format!(
        "  func.func @fe_{k}(%i: i64) -> i64 {{\n    %t = arith.constant dense<[{}]> : tensor<{}x{m}xf64>\n",
        dense.join(", "),
        rows.len()
    );
    for r in 0..rows.len().max(m) {
        f.push_str(&format!("    %c{r} = arith.constant {r} : index\n"));
    }
    let mut outs = Vec::new();
    for j in 0..m {
        for r in 0..rows.len() {
            f.push_str(&format!(
                "    %x{r}_{j} = tensor.extract %t[%c{r}, %c{j}] : tensor<{}x{m}xf64>\n",
                rows.len()
            ));
        }
        f.push_str(&format!(
            "    %s0_{j} = arith.addf %x0_{j}, %x1_{j} : f64\n"
        ));
        for r in 2..rows.len() {
            f.push_str(&format!(
                "    %s{r}_{j} = arith.addf %s{}_{j}, %x{r}_{j} : f64\n",
                if r == 2 { 0 } else { r - 1 }
            ));
        }
        outs.push(format!(
            "%s{}_{j}",
            if rows.len() > 2 { rows.len() - 1 } else { 0 }
        ));
    }
    f.push_str(&format!(
        "    %r = tensor.from_elements {} : tensor<{m}xf64>\n    \
         %idx = arith.index_cast %i : i64 to index\n    \
         %e = tensor.extract %r[%idx] : tensor<{m}xf64>\n    \
         %b = arith.bitcast %e : f64 to i64\n    return %b : i64\n  }}\n",
        outs.join(", ")
    ));
    f
}

/// Folded `tensor.from_elements` axis reductions of 2-D constants in eight
/// functions, driven through the library build path on the tensor preset.
#[test]
fn folded_from_elements_mlir_builds_to_one_shared_library() {
    let tools = match libmind::eval::resolve_mlir_build_tools() {
        Ok(tools) => tools,
        Err(err) => {
            common::gate::skipped("const_table_build_determinism", &format!("{err}"));
            return;
        }
    };
    let (mut mlir, mut checks) = (String::from("module {\n"), Vec::new());
    for k in 0..8 {
        let m = 2 + k % 3;
        let rows: Vec<Vec<f64>> = (0..2 + k % 2)
            .map(|r| {
                (0..m)
                    .map(|j| ival(1000 + k * 7 + r, j) as f64 / 8.0)
                    .collect()
            })
            .collect();
        mlir.push_str(&axis_sum_fn(k, &rows));
        for j in 0..m {
            let col: Vec<f64> = rows.iter().map(|r| r[j]).collect();
            checks.push((format!("fe_{k}"), j as i64, bits(fold_sum(&col))));
        }
    }
    mlir.push_str("}\n");

    let dir = common::scratch_dir("const_table_build_determinism").join("mlir_from_elements");
    std::fs::create_dir_all(&dir).expect("create case dir");
    let mut outputs: Vec<(String, Vec<u8>)> = Vec::with_capacity(BUILDS);
    for n in 0..BUILDS {
        let so = dir.join(format!("b{n}.so"));
        let opts = libmind::eval::MlirBuildOptions {
            preset: "arith-linalg",
            emit_mlir_file: None,
            emit_llvm_file: None,
            emit_obj_file: None,
            emit_shared: Some(so.as_path()),
            opt_pipeline: None,
            target_triple: None,
        };
        let products = libmind::eval::build_mlir_artifacts(&mlir, &tools, &opts)
            .unwrap_or_else(|e| panic!("build {n} of the from_elements module failed: {e}"));
        assert!(
            products.optimized_mlir.contains("@__constant_"),
            "the folded reductions produced no global — the case no longer exercises \
             the ordering it guards:\n{}",
            products.optimized_mlir
        );
        check_values(&so, &checks);
        outputs.push((
            products.optimized_mlir,
            std::fs::read(&so).expect("read .so"),
        ));
    }
    let distinct = (0..BUILDS)
        .filter(|&n| outputs[..n].iter().all(|o| *o != outputs[n]))
        .count();
    assert_eq!(
        distinct, 1,
        "{BUILDS} builds of one MLIR module gave {distinct} distinct (optimized MLIR, .so) pairs"
    );
    eprintln!(
        "const_table_build_determinism[mlir_from_elements]: {BUILDS} builds, 1 distinct \
         artifact, {} value checks",
        checks.len()
    );
}

/// A two-module PROJECT whose second module spells strings as local byte
/// arrays (`let s: [i64; N] = [104, 101, ...]`) in eight functions — the shape
/// that made a multi-module native build differ in one object's read-only data
/// from build to build. Built with `mindc build --no-cache`, eight times.
#[test]
fn project_byte_string_arrays_build_to_one_binary() {
    let words = [
        "version",
        "tensors",
        "shape",
        "neuron_hash",
        "aggregate",
        "encoder.layer",
        "final_ln",
        "bias",
    ];
    let mut text = String::new();
    let mut expect = 0i64;
    for (k, w) in words.iter().enumerate() {
        let bytes: Vec<i64> = w.bytes().map(i64::from).collect();
        text.push_str(&format!(
            "pub fn s_{k}(i: i64) -> i64 {{\n    let s: [i64; {}] = [{}];\n    return s[i];\n}}\n\n",
            bytes.len(),
            ints(&bytes)
        ));
        expect += bytes[k % bytes.len()];
    }
    let calls: Vec<String> = (0..words.len())
        .map(|k| format!("s_{k}({})", k % words[k].len()))
        .collect();
    let main = format!(
        "import text;\n\nfn main() -> i64 {{\n    let t = {};\n    return t % 256;\n}}\n",
        calls.join(" + ")
    );

    let mindc = common::require_mindc();
    let dir = common::scratch_dir("const_table_build_determinism").join("project");
    std::fs::create_dir_all(&dir).expect("create project dir");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"strs\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
         output = \"strs\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
         sources = [\"main.mind\", \"text.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(dir.join("main.mind"), main).expect("write main.mind");
    std::fs::write(dir.join("text.mind"), text).expect("write text.mind");

    let mut bins: Vec<Vec<u8>> = Vec::with_capacity(BUILDS);
    for n in 0..BUILDS {
        let out = Command::new(&mindc)
            .args(["build", "--no-cache"])
            .current_dir(&dir)
            .output()
            .expect("spawn mindc build");
        if !common::gate::compiled("const_table_build_determinism", &out) {
            return;
        }
        let artifact = common::reported_artifact(&out);
        let bytes = std::fs::read(&artifact)
            .unwrap_or_else(|e| panic!("build {n}: read {}: {e}", artifact.display()));
        assert!(
            bytes.len() > 4096,
            "build {n}: {} bytes is not a real binary",
            bytes.len()
        );
        let run = Command::new(&artifact)
            .output()
            .expect("run the built binary");
        assert_eq!(
            run.status.code(),
            Some((expect % 256) as i32),
            "build {n}: the binary computed the wrong value"
        );
        bins.push(bytes);
    }
    let distinct = (0..BUILDS)
        .filter(|&n| bins[..n].iter().all(|b| *b != bins[n]))
        .count();
    assert_eq!(
        distinct, 1,
        "{BUILDS} `mindc build --no-cache` runs of one project gave {distinct} distinct binaries"
    );
    eprintln!(
        "const_table_build_determinism[project]: {BUILDS} builds, 1 distinct binary, {} bytes",
        bins[0].len()
    );
}

/// The tensor pipeline runs in two `mlir-opt` invocations; a failure in the
/// second (here: bufferization rejecting a `to_tensor` without `restrict`,
/// which `canonicalize` keeps) must still cite the line of the text the build
/// handed to `mlir-opt`, with that line's own source — never a line of the
/// intermediate text, and never the temporary file the first stage read.
#[test]
fn a_second_stage_error_cites_the_input_line() {
    let tools = match libmind::eval::resolve_mlir_build_tools() {
        Ok(tools) => tools,
        Err(err) => {
            common::gate::skipped("const_table_build_determinism", &format!("{err}"));
            return;
        }
    };
    let mlir = "module {\n  func.func @a(%m: memref<2xi64>, %i: index) -> i64 {\n    \
                %t = bufferization.to_tensor %m : memref<2xi64> to tensor<2xi64>\n    \
                %e = tensor.extract %t[%i] : tensor<2xi64>\n    return %e : i64\n  }\n}\n";
    let opts = libmind::eval::MlirBuildOptions {
        preset: "arith-linalg",
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: None,
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };
    let err = libmind::eval::build_mlir_artifacts(mlir, &tools, &opts)
        .expect_err("bufferization must reject a to_tensor without restrict")
        .to_string();
    assert!(
        err.contains("<stdin>:3:10: error: 'bufferization.to_tensor' op to_tensor ops without"),
        "the error must cite line 3 of the input:\n{err}"
    );
    assert!(
        err.contains("    %t = bufferization.to_tensor %m : memref<2xi64> to tensor<2xi64>"),
        "the error must show the input's own line, not the intermediate text:\n{err}"
    );
    assert!(
        !err.contains("mind-opt-"),
        "the temporary stage file leaked:\n{err}"
    );
}
