// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The self-host front-end accepts a `diff tensor<...>` parameter type exactly
//! when the reference compiler does.
//!
//! Every spelling below is compiled as the type of one parameter through both
//! paths: `mindc --emit-ir` (accepted = exit 0) and the pure-MIND
//! `mindc_compile` from a `libmindc_mind.so` built from this tree (accepted =
//! not an explicit refusal). The live reference verdict is checked against the
//! recorded one and the self-host verdict against the live one, so a shape that
//! either side starts to accept or refuse turns this red. The near-miss rows are
//! the shapes the first self-host fix over-accepted.
#![cfg(all(target_os = "linux", feature = "mlir-build"))]

mod common;

use libloading::Library;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const TARGET: &str = "g2_diff_tensor_param_grammar";

/// (case, parameter type, reference compiler accepts)
const CASES: &[(&str, &str, bool)] = &[
    ("diff_tensor_lit", "diff tensor<f32[10, 16]>", true),
    ("diff_tensor_sym", "diff tensor<f32[batch, features]>", true),
    ("diff_tensor_empty", "diff tensor<f32[]>", true),
    ("diff_tensor_noshape", "diff tensor<f32>", true),
    ("tensor_lit_nodiff", "tensor<f32[10, 16]>", true),
    ("tensor_empty_nodiff", "tensor<f32[]>", true),
    ("tensor_sym_nodiff", "tensor<f32[batch, features]>", true),
    ("tensor_noshape_nodiff", "tensor<f32>", true),
    ("diff_only", "diff", false),
    ("diff_tensor_no_lt", "diff tensor", false),
    ("diff_tensor_unclosed", "diff tensor<f32[10]", false),
    ("diff_wrong_word", "diff vector<f32[10]>", false),
    ("diff_mut_tensor", "diff mut tensor<f32[10]>", false),
    ("diff_const_tensor", "diff const tensor<f32[10]>", false),
    ("mut_tensor", "mut tensor<f32[10]>", false),
    ("const_tensor", "const tensor<f32[10]>", false),
    ("diff_tensor_no_dtype", "diff tensor<>", false),
    ("diff_tensor_two_dtype", "diff tensor<f32, i32>", false),
    ("diffx_tensor", "diffx tensor<f32[10]>", false),
    ("dif_tensor", "dif tensor<f32[10]>", false),
    ("diff_Tensor", "diff Tensor<f32[10]>", false),
    ("Diff_tensor", "Diff tensor<f32[10]>", false),
    ("diff_tensor_space_lt", "diff tensor <f32[10]>", true),
    ("diff_nl_tensor", "diff\ntensor<f32[10]>", true),
    ("diff_tensor_gt_extra", "diff tensor<f32[10]>>", false),
    ("amp_diff_tensor", "&diff tensor<f32[10]>", true),
    ("star_diff_tensor", "*diff tensor<f32[10]>", false),
    ("diff_tensor_i32", "diff tensor<i32[4]>", true),
    ("diff_tensor_f64", "diff tensor<f64[2,2]>", true),
    ("plain_i64", "i64", true),
    ("plain_ident", "Foo", true),
    ("generic_map", "map<i64, Value>", true),
    ("ref_u8", "&u8", true),
    ("ptr_const_u8", "*const u8", true),
    ("array_i64", "[i64]", true),
    ("diff_tensor_nested", "diff tensor<f32[[1,2],[3,4]]>", false),
    ("diff_tensor_dim3", "diff tensor<f32[1, 2, 3]>", true),
    ("diff_tensor_dbl_brack", "diff tensor<f32[10][]>", false),
    ("diff_tensor_trailing_comma", "diff tensor<f32[10,]>", true),
    ("diff_tensor_comma_only", "diff tensor<f32[,]>", false),
    ("diff_tensor_hex_dim", "diff tensor<f32[0x10]>", false),
    ("diff_tensor_nl_lt", "diff tensor\n<f32[10]>", false),
    ("diff_tensor_nl_inside", "diff tensor<f32\n[10]>", false),
    ("diff_comment_tensor", "diff // c\ntensor<f32>", true),
    (
        "diff_tensor_dotted",
        "diff tensor<fixed_point.Q16_16[4]>",
        false,
    ),
    ("diff_tensor_dot_space", "diff tensor<a. b>", false),
    ("diff_tensor_kw_dtype", "diff tensor<fn[4]>", true),
    ("diff_tensor_kw_dim", "diff tensor<f32[let]>", true),
    ("diff_tensor_true_dim", "diff tensor<f32[true]>", true),
    ("diff_tensor_neg_dim", "diff tensor<f32[-1]>", false),
    ("diff_tensor_float_dim", "diff tensor<f32[1.5]>", false),
    ("diff_tensor_suffix_dim", "diff tensor<f32[10u8]>", false),
    ("diff_tensor_tabs", "diff\ttensor\t<\tf32\t[\t1\t]\t>", true),
    (
        "diff_tensor_two_params",
        "diff tensor<f32[2]>, y: diff tensor<f32>",
        true,
    ),
    (
        "diff_tensor_sparse",
        "diff tensor<sparse[csr], f32[4]>",
        false,
    ),
    (
        "diff_tensor_blockcomment",
        "diff /* c */ tensor<f32>",
        false,
    ),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Build the self-host compiler from this tree's `main.mind` (via `Mind.toml`).
fn build_self_host_so(mindc: &Path, dir: &Path) -> Option<PathBuf> {
    let out = dir.join("libmindc_mind.so");
    let r = Command::new(mindc)
        .args([
            "build",
            "--backend=mlir",
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc build");
    if !common::gate::compiled(TARGET, &r) {
        return None;
    }
    let bytes = fs::read(&out).expect("read the built libmindc_mind.so");
    assert!(
        bytes.starts_with(b"\x7fELF"),
        "{} is not an ELF",
        out.display()
    );
    Some(out)
}

fn source_for(ty: &str) -> String {
    format!("fn fuzz_param(x: {ty}) -> i64 {{\n    1\n}}\n")
}

fn rust_accepts(mindc: &Path, file: &Path) -> bool {
    let r = Command::new(mindc)
        .arg(file)
        .arg("--emit-ir")
        .output()
        .expect("spawn mindc --emit-ir");
    r.status.success()
}

/// `mindc_compile(src, len)` returns an EmitState record: word 0 is the output
/// String handle (word 0 address, word 1 length), word 1 is `next_id`, -1 with
/// an empty buffer being the explicit refusal.
fn self_host_accepts(lib: &Library, src: &str) -> bool {
    type Compile = unsafe extern "C" fn(i64, i64) -> i64;
    let bytes = src.as_bytes().to_vec();
    unsafe {
        let compile: libloading::Symbol<Compile> =
            lib.get(b"mindc_compile\0").expect("mindc_compile export");
        let es = compile(bytes.as_ptr() as i64, bytes.len() as i64);
        assert_ne!(es, 0, "mindc_compile returned a null record");
        let word = |base: i64, i: usize| ((base as usize + 8 * i) as *const i64).read_unaligned();
        let buf = word(es, 0);
        let next_id = word(es, 1);
        let len = if buf == 0 { 0 } else { word(buf, 1) };
        assert!(next_id >= -1, "invalid refusal record (next_id {next_id})");
        !(next_id == -1 && len <= 0)
    }
}

#[test]
fn diff_tensor_parameter_types_match_the_reference_compiler() {
    let mindc = common::require_mindc();
    let dir = common::scratch_dir(TARGET);
    let Some(so) = build_self_host_so(&mindc, &dir) else {
        return;
    };
    let lib = unsafe { Library::new(&so) }.expect("dlopen libmindc_mind.so");
    let mut wrong = Vec::new();
    for (case, ty, want) in CASES {
        let src = source_for(ty);
        let file = dir.join(format!("{case}.mind"));
        fs::write(&file, &src).expect("write case source");
        let rust = rust_accepts(&mindc, &file);
        let mind = self_host_accepts(&lib, &src);
        if rust != *want || mind != rust {
            wrong.push(format!(
                "{case} `{ty}`: reference={rust} self-host={mind} recorded={want}"
            ));
        }
    }
    println!(
        "SDLC-GATE {TARGET} ran={} fail={}",
        CASES.len(),
        wrong.len()
    );
    assert!(
        wrong.is_empty(),
        "diff tensor parameter verdicts diverge:\n{}",
        wrong.join("\n")
    );
}
