// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native bridge refusals for programs it used to miscompile or crash on.
//!
//! * `fn main(argc: i64, argv: i64)` built natively and ran with both
//!   parameters 0: the probe below exited 40 where the MLIR build of the same
//!   source exits 20 + argc. The bridge now refuses an entry `main` with
//!   parameters by name (`native.entry_main_has_params`) and writes nothing.
//! * Undefined identifiers and symbolic tensor dimensions read as values
//!   reached lowering's fail-closed `panic!`, so `mindc build --backend native`
//!   died with exit 101. Every case of that negative corpus -- reduced from the
//!   MIND-Mem kernels (abstention, bm25, category, reranker) and the in-tree
//!   examples that panicked (cnn_classifier, zoo conv/logistic classifiers) --
//!   must now end in a named refusal (exit 3), never a panic.
#![cfg(all(feature = "cross-module-imports", feature = "std-surface"))]

mod native_bridge_support;
use native_bridge_support::{Project, assert_native_result, repo_path};

/// Build `src` as the project entry; assert a refusal of `kind` with no panic
/// and no artifact.
fn assert_refused(name: &str, src: &str, kind: &str) -> String {
    let p = Project::new(name);
    p.write("src/main.mind", src);
    let (code, stderr, bytes) = p.build_native("src/main.mind");
    assert_ne!(code, 101, "{name}: the native build panicked:\n{stderr}");
    assert!(
        !stderr.contains("panicked"),
        "{name}: the native build printed a panic:\n{stderr}"
    );
    assert_eq!(code, 3, "{name}: expected a refusal (exit 3):\n{stderr}");
    assert!(
        stderr.contains(kind),
        "{name}: expected refusal `{kind}`, got:\n{stderr}"
    );
    assert!(bytes.is_none(), "{name}: a refused build wrote an artifact");
    stderr
}

#[test]
fn an_entry_main_with_argc_argv_is_refused_not_run_with_zeros() {
    let stderr = assert_refused(
        "argc_probe",
        "fn main(argc: i64, argv: i64) -> i64 { if argv == 0 { return 40 + argc; } \
         return 20 + argc; }\n",
        "native.entry_main_has_params",
    );
    assert!(stderr.contains("--backend mlir"), "hint missing:\n{stderr}");
    assert!(
        stderr.contains("\"argc\""),
        "parameter list missing:\n{stderr}"
    );
}

#[test]
fn an_entry_main_with_one_parameter_is_refused() {
    assert_refused(
        "one_param",
        "fn main(n: i64) -> i64 { return n; }\n",
        "native.entry_main_has_params",
    );
}

#[test]
fn a_parameterless_main_still_builds_and_runs() {
    // Positive control: the refusal is keyed on parameters, not on `main`.
    let p = Project::new("no_params");
    p.write(
        "src/main.mind",
        "fn add(a: i64, b: i64) -> i64 { return a + b; }\n\
         fn main() -> i64 { return add(20, 3); }\n",
    );
    let (code, stderr, bytes) = p.build_native("src/main.mind");
    assert_eq!(code, 0, "{stderr}");
    let bytes = bytes.expect("artifact");
    assert_native_result(&p, &bytes, 23, "parameterless main");
}

#[test]
fn an_undefined_identifier_is_a_diagnostic_not_a_panic() {
    let stderr = assert_refused(
        "typo",
        "fn main() -> i64 {\n    let total = 1;\n    return totl;\n}\n",
        "native.unresolved_name",
    );
    assert!(stderr.contains("E2002"), "{stderr}");
    assert!(
        stderr.contains("main.mind:3:"),
        "no source location:\n{stderr}"
    );
}

#[test]
fn a_function_used_as_a_value_is_a_diagnostic_not_a_panic() {
    let stderr = assert_refused(
        "fn_value",
        "fn helper() -> i64 { return 1; }\n\
         fn main() -> i64 {\n    let f = helper;\n    return 0;\n}\n",
        "native.unresolved_name",
    );
    assert!(stderr.contains("E2037"), "{stderr}");
}

/// Reduced from the MIND-Mem kernels that panicked natively: the symbolic
/// dimension each one reads as a value, in the signature it is declared in.
const MIND_MEM_SYMBOLIC_DIM_CASES: &[(&str, &str, &str)] = &[
    (
        "mindmem_abstention",
        "fn entity_overlap(query_entities: tensor<f32[E]>, doc_entities: tensor<f32[N, E]>) \
         -> tensor<f32[N]> {\n    let q_broad = reshape(query_entities, [1, E]);\n    \
         return matmul(doc_entities, reshape(q_broad, [E, 1]));\n}\n",
        "`E`",
    ),
    (
        "mindmem_bm25",
        "fn bm25f_doc(tf: tensor<f32[F, T]>, idf: tensor<f32[T]>, norm: tensor<f32[F]>) \
         -> tensor<f32[T]> {\n    let norm_broad = reshape(norm, [F, 1]);\n    \
         let idf_broad = reshape(idf, [1, T]);\n    return tf;\n}\n",
        "`F`",
    ),
    (
        "mindmem_category",
        "fn query_category_relevance(query_kw: tensor<f32[K]>, cat_kw: tensor<f32[C, K]>) \
         -> tensor<f32[C]> {\n    let dots = matmul(cat_kw, reshape(query_kw, [K, 1]));\n    \
         return reshape(dots, [C]);\n}\n",
        "`K`",
    ),
    (
        "mindmem_reranker",
        "fn category_boost(query_cats: tensor<f32[C]>, doc_cats: tensor<f32[N, C]>) \
         -> tensor<f32[N]> {\n    let query_broad = reshape(query_cats, [1, C]);\n    \
         return matmul(doc_cats, reshape(query_broad, [C, 1]));\n}\n",
        "`C`",
    ),
];

#[test]
fn mind_mem_symbolic_dims_read_as_values_are_refused_not_panics() {
    for (name, src, dim) in MIND_MEM_SYMBOLIC_DIM_CASES {
        let stderr = assert_refused(name, src, "native.symbolic_dim_as_value");
        assert!(
            stderr.contains(dim),
            "{name}: dimension {dim} not named:\n{stderr}"
        );
    }
}

#[test]
fn in_tree_examples_that_panicked_are_refused_not_panics() {
    for rel in [
        "examples/cnn_classifier.mind",
        "examples/zoo/conv_classifier.mind",
        "examples/zoo/logistic_classifier.mind",
    ] {
        let src = std::fs::read_to_string(repo_path(rel)).expect("read example");
        let stderr = assert_refused(rel, &src, "native.symbolic_dim_as_value");
        assert!(
            stderr.contains("`batch`"),
            "{rel}: `batch` not named:\n{stderr}"
        );
    }
}
