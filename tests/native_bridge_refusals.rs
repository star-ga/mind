// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native bridge refusals for programs it used to miscompile or crash on.
//!
//! * `fn main(argc: i64, argv: i64)` built natively and ran with both
//!   parameters 0: the probe below exited 40 where the MLIR build of the same
//!   source exits 20 + argc. The bridge admits that signature only when the
//!   emitted entry binds the process arguments; unsupported signatures and
//!   older images without the binding still refuse without an artifact.
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
fn an_entry_main_with_argc_argv_is_admitted_and_bound_where_native_runs() {
    for (name, src) in [
        (
            "argc_ssa",
            "fn main(argc: i64, argv: i64) -> i64 { if argv == 0 { return 40 + argc; } \
             return 20 + argc; }\n",
        ),
        (
            "argc_home",
            "fn main(argc: i64, argv: i64) -> i64 { let mut i: i64 = 0; \
             while i < argc && argv != 0 { if i == 0 { return 20 + argc; } \
             i = i + 1; } return 40; }\n",
        ),
    ] {
        let p = Project::new(name);
        p.write("src/main.mind", src);
        let (code, stderr, bytes) = p.build_native("src/main.mind");
        assert_eq!(code, 0, "{name}: {stderr}");
        let bytes = bytes.expect("admitted argc/argv program must emit an artifact");
        // Linux x86-64 executes the real image with only argv[0], hence
        // argc == 1 and argv != 0. Other hosts check admission and exact
        // transport via the drain fixture; they cannot attest entry semantics.
        assert_native_result(&p, &bytes, 21, name);
    }
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

/// The cases above whose shape a compiler carrying the scoped-nested-bindings
/// mark admits (a plain `let` over an outer `let` or parameter across `if`
/// blocks and `while`/`loop` bodies only), with the exit code of the MLIR build
/// of the same source (the comments above).
const PROVEN_SHADOW_MLIR_EXIT: &[(&str, i32)] = &[
    ("outer_let", 5),
    ("outer_param", 5),
    ("blind_review_fzh", 2),
    ("nested_loop_body_let", 2),
    ("shadow_then_outer_read_after_second_loop", 7),
    ("if_inside_loop", 5),
    ("loop_keyword", 5),
    ("if_shadow_loop_in_nested_if", 5),
    ("if_shadow_loop_in_nested_if_then_writes", 6),
    ("if_shadow_write_in_nested_if", 5),
    ("param_if_shadow_write_in_nested_if", 5),
    ("helper_if_shadow_write_in_nested_if", 3),
    ("loop_param_if_shadow_write_in_nested_if", 28),
    ("if_block_shadow_outside_loop", 5),
    ("if_block_param_shadow_outside_loop", 5),
    ("imported_block_shadow", 3),
];

/// True when the compiler these builds run (`native_bridge_support`'s
/// `MINDC_NATIVE_ELF`) carries the scoped-nested-bindings mark, the bytes
/// `nb_cap_scoped_nested_bindings` in examples/mindc_mind/main.mind compiles to.
/// Off x86-64 Linux the compiler is a drain fixture without it.
fn compiler_scopes_nested_bindings() -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64"))
        && std::fs::read(repo_path(
            "examples/mindc_mind/testdata/selfhost_loop/stage1.elf",
        ))
        .is_ok_and(|b| b.windows(10).any(|w| w == b"\x48\xb8nbscope1"))
}

/// The MLIR exit of `name` when the compiler admits its (proven) shadow.
fn admitted_shadow(name: &str) -> Option<i32> {
    if !compiler_scopes_nested_bindings() {
        return None;
    }
    PROVEN_SHADOW_MLIR_EXIT
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, e)| *e)
}

/// An admitted shadow is never refused for shadowing; if the compiler builds
/// it, it exits with the MLIR value (a compiler refusal stays fail-closed).
fn assert_admitted_shadow(p: &Project, entry: &str, name: &str, expected: i32) {
    let (code, stderr, bytes) = p.build_native(entry);
    assert!(
        !stderr.contains("native.loop_body_shadow") && !stderr.contains("native.block_shadow"),
        "{name}: a proven shadow was refused by a scoping compiler:\n{stderr}"
    );
    if code == 0 {
        assert_native_result(p, &bytes.expect("artifact"), expected, name);
    } else {
        assert_eq!(code, 3, "{name}: neither built nor refused:\n{stderr}");
        assert!(bytes.is_none(), "{name}: a refused build wrote an artifact");
    }
}

#[test]
fn a_loop_body_binding_shadowing_an_enclosing_name_is_refused() {
    for (name, src, shadowed) in LOOP_BODY_SHADOW_CASES {
        if let Some(expected) = admitted_shadow(name) {
            let p = Project::new(name);
            p.write("src/main.mind", src);
            assert_admitted_shadow(&p, "src/main.mind", name, expected);
            continue;
        }
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
    // `for_body_let`: a `for` body is never a proven shape, so this stays a
    // refusal whatever compiler runs.
    let (name, src, _) = LOOP_BODY_SHADOW_CASES[4];
    assert_eq!(name, "for_body_let");
    let stderr = assert_refused("outer_let_line", src, "native.loop_body_shadow");
    assert!(
        stderr.contains("main.mind:4:"),
        "inner `let` line:\n{stderr}"
    );
}

/// Shapes the guard must NOT refuse. Each builds natively and exits with the
/// value the MLIR build of the same source exits with (measured).
const LOOP_BODY_SHADOW_CONTROLS: &[(&str, &str, i32)] = &[
    (
        // A parameter re-declared at the function's top level (one frame with
        // the parameters), then written from a nested `if`: native 40, MLIR 40.
        "param_redeclared_at_function_top",
        "fn f(x: i64, c: i64) -> i64 {\n    let mut x: i64 = x + 1;\n    if c == 1 {\n        \
         if c == 1 {\n            x = 40;\n        }\n    }\n    return x;\n}\n\
         fn main() -> i64 { return f(5, 1); }\n",
        40,
    ),
    (
        // A same-frame re-declaration, then a loop inside a nested `if` that
        // writes the name: native 9, MLIR 9.
        "same_frame_redeclare_loop_in_nested_if",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let mut x: i64 = x + 2;\n    \
         let c: i64 = 1;\n    let mut k: i64 = 0;\n    if c == 1 {\n        if c == 1 {\n            \
         while k < 2 { k = k + 1; x = x + 1; }\n        }\n    }\n    return x;\n}\n",
        9,
    ),
    (
        // Sibling `if` blocks binding the same name, no outer binding, written
        // from a nested `if` and a nested loop: native 29, MLIR 29.
        "sibling_if_blocks_same_name",
        "fn main() -> i64 {\n    let c: i64 = 1;\n    let mut s: i64 = 0;\n    if c == 1 {\n        \
         let mut t: i64 = 3;\n        if c == 1 {\n            t = t + 4;\n        }\n        \
         s = s + t;\n    }\n    if c == 1 {\n        let mut t: i64 = 20;\n        if c == 1 {\n            \
         let mut k: i64 = 0;\n            while k < 2 { k = k + 1; t = t + 1; }\n        }\n        \
         s = s + t;\n    }\n    return s;\n}\n",
        29,
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
fn same_frame_redeclarations_and_new_loop_locals_still_build_and_run() {
    for (name, src, expected) in LOOP_BODY_SHADOW_CONTROLS {
        let p = Project::new(name);
        p.write("src/main.mind", src);
        let (code, stderr, bytes) = p.build_native("src/main.mind");
        assert_eq!(code, 0, "{name}: control must build:\n{stderr}");
        let bytes = bytes.expect("artifact");
        assert_native_result(&p, &bytes, *expected, name);
    }
}

/// A binding in a nested block OUTSIDE any loop that re-declares an enclosing
/// name. The frozen compiler resolves a write from a further-nested block (an
/// inner `if`, or a loop inside one) to the OUTER binding, so these built
/// natively and exited with the inner value (measured on the 0.10.3 candidate,
/// native / MLIR in each comment). A single-level `if` shadow happens to run
/// correctly today, but that was measured at one nesting only, so every
/// cross-frame re-declaration is refused (`native.block_shadow`).
const BLOCK_SHADOW_CASES: &[(&str, &str, &str)] = &[
    (
        // native 11, MLIR 5: a loop inside a nested `if` carries `x` by name.
        "if_shadow_loop_in_nested_if",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let c: i64 = 1;\n    if c == 1 {\n        \
         let mut x: i64 = 9;\n        if c == 1 {\n            let mut k: i64 = 0;\n            \
         while k < 2 {\n                k = k + 1;\n                x = x + 1;\n            }\n        \
         }\n    }\n    return x;\n}\n",
        "`x`",
    ),
    (
        // native 112, MLIR 6.
        "if_shadow_loop_in_nested_if_then_writes",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let c: i64 = 1;\n    if c == 1 {\n        \
         let mut x: i64 = 9;\n        if c == 1 {\n            let mut k: i64 = 0;\n            \
         while k < 2 {\n                k = k + 1;\n                x = x + 1;\n            }\n        \
         }\n        x = x + 100;\n    }\n    x = x + 1;\n    return x;\n}\n",
        "`x`",
    ),
    (
        // native 40, MLIR 5: a write in a nested `if` lands on the outer `x`.
        "if_shadow_write_in_nested_if",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let c: i64 = 1;\n    if c == 1 {\n        \
         let mut x: i64 = 9;\n        if c == 1 {\n            x = 40;\n        }\n    }\n    \
         return x;\n}\n",
        "`x`",
    ),
    (
        // native 40, MLIR 5: the same with a parameter.
        "param_if_shadow_write_in_nested_if",
        "fn f(x: i64, c: i64) -> i64 {\n    if c == 1 {\n        let mut x: i64 = 9;\n        \
         if c == 1 {\n            x = 40;\n        }\n    }\n    return x;\n}\n\
         fn main() -> i64 {\n    return f(5, 1);\n}\n",
        "`x`",
    ),
    (
        // native 99, MLIR 3: in a helper called from `main`.
        "helper_if_shadow_write_in_nested_if",
        "fn h(a: i64) -> i64 {\n    let mut acc: i64 = a;\n    if a > 0 {\n        \
         let mut acc: i64 = 0;\n        if a > 1 {\n            acc = 99;\n        }\n    }\n    \
         return acc;\n}\nfn main() -> i64 {\n    return h(3);\n}\n",
        "`acc`",
    ),
    (
        // native 229, MLIR 28: a parameter carried by an earlier loop.
        "loop_param_if_shadow_write_in_nested_if",
        "fn f(n: i64, c: i64) -> i64 {\n    let mut s: i64 = 0;\n    let mut k: i64 = 0;\n    \
         while k < 3 {\n        k = k + 1;\n        s = s + n;\n    }\n    if c == 1 {\n        \
         let mut n: i64 = 1000;\n        if c == 1 {\n            n = 2000;\n        }\n        \
         s = s + 0;\n    }\n    return s + n;\n}\nfn main() -> i64 {\n    return f(7, 1);\n}\n",
        "`n`",
    ),
    (
        // Runs correctly today (native 5, MLIR 5) but only at this single
        // nesting level; refused because the family is not proven.
        "if_block_shadow_outside_loop",
        "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let c: i64 = 1;\n    \
         if c == 1 {\n        let mut x: i64 = 7;\n        x = 3;\n    }\n    return x;\n}\n",
        "`x`",
    ),
    (
        // As above, with a parameter (native 5, MLIR 5).
        "if_block_param_shadow_outside_loop",
        "fn f(x: i64, c: i64) -> i64 {\n    if c == 1 {\n        let mut x: i64 = 7;\n        \
         x = 3;\n    }\n    return x;\n}\nfn main() -> i64 { return f(5, 1); }\n",
        "`x`",
    ),
];

#[test]
fn a_block_binding_shadowing_an_enclosing_name_is_refused() {
    for (name, src, shadowed) in BLOCK_SHADOW_CASES {
        if let Some(expected) = admitted_shadow(name) {
            let p = Project::new(name);
            p.write("src/main.mind", src);
            assert_admitted_shadow(&p, "src/main.mind", name, expected);
            continue;
        }
        let stderr = assert_refused(name, src, "native.block_shadow");
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
fn a_block_shadow_in_an_imported_module_is_refused() {
    // native 99, MLIR 3: the shadow sits in `helper.mind`, not the entry.
    let p = Project::new("imported_block_shadow");
    p.write(
        "src/helper.mind",
        "pub fn hv(a: i64) -> i64 {\n    let mut acc: i64 = a;\n    if a > 0 {\n        \
         let mut acc: i64 = 0;\n        if a > 1 {\n            acc = 99;\n        }\n    }\n    \
         return acc;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return hv(3);\n}\n",
    );
    if let Some(expected) = admitted_shadow("imported_block_shadow") {
        assert_admitted_shadow(&p, "src/main.mind", "imported_block_shadow", expected);
        return;
    }
    let (code, stderr, bytes) = p.build_native("src/main.mind");
    assert_eq!(code, 3, "expected a refusal (exit 3):\n{stderr}");
    assert!(stderr.contains("native.block_shadow"), "{stderr}");
    assert!(stderr.contains("`acc`"), "{stderr}");
    assert!(bytes.is_none(), "a refused build wrote an artifact");
}
