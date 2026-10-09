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

/// A `let` (or loop variable) inside a loop body that re-declares a name bound
/// in an enclosing scope. The frozen native compiler resolves loop-carried
/// variables by NAME, so it merges the inner binding into the outer one: every
/// probe below built natively and exited with the inner value where the MLIR
/// build of the same source exits with the outer one (measured on the 0.10.3
/// candidate; native / MLIR exit codes in each comment).
const LOOP_BODY_SHADOW_CASES: &[(&str, &str, &str)] = &[
    (
        // native 3, MLIR 5.
        "outer_let",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let mut k: i64 = 0;\n    \
         while k < 2 {\n        k = k + 1;\n        let mut x: i64 = 7;\n        x = 3;\n    \
         }\n    return x;\n}\n",
        "`x`",
    ),
    (
        // native 3, MLIR 5.
        "outer_param",
        "fn f(x: i64) -> i64 {\n    let mut k: i64 = 0;\n    while k < 2 {\n        \
         k = k + 1;\n        let mut x: i64 = 7;\n        x = 3;\n    }\n    return x;\n}\n\
         fn main() -> i64 { return f(5); }\n",
        "`x`",
    ),
    (
        // native 3, MLIR 2.
        "blind_review_fzh",
        "fn fzh(p: i64, q0: i64, Pz: i64, a0: i64) -> i64 {\n    let mut x2: i64 = a0;\n    \
         x2 = q0;\n    let mut k0: i64 = 0;\n    while k0 < 2 {\n        k0 = k0 + 1;\n        \
         let mut q0: i64 = x2;\n        q0 = 3;\n    }\n    return q0;\n}\n\
         fn main() -> i64 { return fzh(1, 2, 3, 4); }\n",
        "`q0`",
    ),
    (
        // native 8, MLIR 2: the inner loop's `let a` merges into the outer
        // loop body's `a`.
        "nested_loop_body_let",
        "fn main() -> i64 {\n    let mut k: i64 = 0;\n    let mut r: i64 = 0;\n    \
         while k < 2 {\n        k = k + 1;\n        let mut a: i64 = 1;\n        \
         let mut j: i64 = 0;\n        while j < 1 {\n            j = j + 1;\n            \
         let mut a: i64 = 9;\n            a = 4;\n        }\n        r = r + a;\n    }\n    \
         return r;\n}\n",
        "`a`",
    ),
    (
        // native 3, MLIR 5 (`for` loop).
        "for_body_let",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    for i in 0..2 {\n        \
         let mut x: i64 = 7;\n        x = 3;\n    }\n    return x;\n}\n",
        "`x`",
    ),
    (
        // native 10, MLIR 30: the body's `let i` clobbers the induction variable.
        "let_shadows_for_variable",
        "fn main() -> i64 {\n    let mut s: i64 = 0;\n    for i in 0..3 {\n        \
         let i: i64 = 10;\n        s = s + i;\n    }\n    return s;\n}\n",
        "`i`",
    ),
    (
        // native 101, MLIR 47: a nested `for i` re-binds the outer `i`.
        "nested_for_same_variable",
        "fn main() -> i64 {\n    let mut s: i64 = 0;\n    for i in 0..3 {\n        \
         for i in 0..2 {\n            s = s + i;\n        }\n        s = s + 100;\n    }\n    \
         return s;\n}\n",
        "`i`",
    ),
    (
        // native 33, MLIR 43: the loop variable clobbers the parameter.
        "for_variable_shadows_param",
        "fn f(i: i64) -> i64 {\n    let mut s: i64 = 0;\n    for i in 0..3 {\n        \
         s = s + i;\n    }\n    return i * 10 + s;\n}\nfn main() -> i64 { return f(4); }\n",
        "`i`",
    ),
    (
        // native 5, MLIR 10: an inner `for k` re-binds the outer while's `k`.
        "for_variable_shadows_while_counter",
        "fn main() -> i64 {\n    let mut s: i64 = 0;\n    let mut k: i64 = 0;\n    \
         while k < 2 {\n        k = k + 1;\n        for k in 0..5 {\n            s = s + 1;\n        \
         }\n    }\n    return s;\n}\n",
        "`k`",
    ),
    (
        // native 10, MLIR 7.
        "shadow_then_outer_read_after_second_loop",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let mut k: i64 = 0;\n    \
         while k < 2 {\n        k = k + 1;\n        let mut x: i64 = 7;\n        x = x + 1;\n    \
         }\n    let mut j: i64 = 0;\n    while j < 2 { j = j + 1; x = x + 1; }\n    return x;\n}\n",
        "`x`",
    ),
    (
        // Inside an `if` inside the loop (the frozen compiler happens to reject
        // this shape today; the guard must not depend on that).
        "if_inside_loop",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let mut k: i64 = 0;\n    \
         while k < 2 {\n        k = k + 1;\n        let c: i64 = 1;\n        if c == 1 {\n            \
         let mut x: i64 = 7;\n            x = 3;\n        }\n    }\n    return x;\n}\n",
        "`x`",
    ),
    (
        // `loop { }` desugars to `while 1`.
        "loop_keyword",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let mut k: i64 = 0;\n    loop {\n        \
         k = k + 1;\n        let mut x: i64 = 7;\n        x = 3;\n        \
         if k >= 2 { break; }\n    }\n    return x;\n}\n",
        "`x`",
    ),
];

#[test]
fn a_loop_body_binding_shadowing_an_enclosing_name_is_refused() {
    for (name, src, shadowed) in LOOP_BODY_SHADOW_CASES {
        let stderr = assert_refused(name, src, "native.loop_body_shadow");
        assert!(
            stderr.contains(shadowed),
            "{name}: shadowed name {shadowed} not named:\n{stderr}"
        );
        assert!(
            stderr.contains("main.mind:"),
            "{name}: no source location:\n{stderr}"
        );
        assert!(
            stderr.contains("--backend mlir"),
            "{name}: hint missing:\n{stderr}"
        );
    }
}

#[test]
fn the_first_probe_names_the_inner_let_line() {
    let (_, src, _) = LOOP_BODY_SHADOW_CASES[0];
    let stderr = assert_refused("outer_let_line", src, "native.loop_body_shadow");
    assert!(
        stderr.contains("main.mind:6:"),
        "inner `let` line:\n{stderr}"
    );
}

/// Shapes the guard must NOT refuse. Each builds natively and exits with the
/// value the MLIR build of the same source exits with (measured).
const LOOP_BODY_SHADOW_CONTROLS: &[(&str, &str, i32)] = &[
    (
        // Shadowing in a plain `if` outside any loop.
        "if_block_shadow_outside_loop",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let c: i64 = 1;\n    \
         if c == 1 {\n        let mut x: i64 = 7;\n        x = 3;\n    }\n    return x;\n}\n",
        5,
    ),
    (
        // A parameter shadowed in a plain `if` outside any loop.
        "if_block_param_shadow_outside_loop",
        "fn f(x: i64, c: i64) -> i64 {\n    if c == 1 {\n        let mut x: i64 = 7;\n        \
         x = 3;\n    }\n    return x;\n}\nfn main() -> i64 { return f(5, 1); }\n",
        5,
    ),
    (
        // A loop-local `let` of a NEW name.
        "loop_local_new_name",
        "fn main() -> i64 {\n    let mut s: i64 = 0;\n    let mut k: i64 = 0;\n    \
         while k < 3 {\n        k = k + 1;\n        let t: i64 = k * 2;\n        s = s + t;\n    \
         }\n    return s;\n}\n",
        12,
    ),
    (
        // The same name in two sibling loops, with no outer binding.
        "sibling_loops_same_name",
        "fn main() -> i64 {\n    let mut s: i64 = 0;\n    let mut k: i64 = 0;\n    \
         while k < 3 {\n        k = k + 1;\n        let mut t: i64 = k;\n        t = t + 1;\n        \
         s = s + t;\n    }\n    let mut j: i64 = 0;\n    while j < 2 {\n        j = j + 1;\n        \
         let mut t: i64 = 10;\n        t = t + j;\n        s = s + t;\n    }\n    return s;\n}\n",
        32,
    ),
    (
        // A re-declaration in the SAME loop-body scope (no enclosing binding).
        "same_scope_redeclare_in_loop",
        "fn main() -> i64 {\n    let mut k: i64 = 0;\n    let mut s: i64 = 0;\n    \
         while k < 2 {\n        k = k + 1;\n        let mut a: i64 = k;\n        \
         let mut a: i64 = a + 10;\n        a = a + 1;\n        s = s + a;\n    }\n    return s;\n}\n",
        25,
    ),
    (
        // The loop-local name is bound in the outer scope only AFTER the loop.
        "outer_binding_after_loop",
        "fn main() -> i64 {\n    let mut k: i64 = 0;\n    let mut s: i64 = 0;\n    \
         while k < 2 {\n        k = k + 1;\n        let t: i64 = k;\n        s = s + t;\n    }\n    \
         let t: i64 = 100;\n    return s + t;\n}\n",
        103,
    ),
    (
        // Sibling `for` loops reusing an induction variable.
        "sibling_for_loops_same_variable",
        "fn main() -> i64 {\n    let mut s: i64 = 0;\n    let mut k: i64 = 0;\n    \
         while k < 3 {\n        k = k + 1;\n        for j in 0..2 {\n            s = s + j;\n        \
         }\n    }\n    for j in 0..4 {\n        s = s + j;\n    }\n    return s;\n}\n",
        9,
    ),
];

#[test]
fn shadowing_outside_loops_and_new_loop_locals_still_build_and_run() {
    for (name, src, expected) in LOOP_BODY_SHADOW_CONTROLS {
        let p = Project::new(name);
        p.write("src/main.mind", src);
        let (code, stderr, bytes) = p.build_native("src/main.mind");
        assert_eq!(code, 0, "{name}: control must build:\n{stderr}");
        let bytes = bytes.expect("artifact");
        assert_native_result(&p, &bytes, *expected, name);
    }
}
