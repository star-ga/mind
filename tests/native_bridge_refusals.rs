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

/// The struct names the std seed blob declares, read from the seed rows of the
/// std manifest the native bridge composes. Read here independently of the
/// bridge's own scan, so the two lists are checked against each other.
fn seed_struct_names() -> Vec<String> {
    let manifest = std::fs::read_to_string(repo_path(
        "examples/mindc_mind/testdata/stdlib_manifest.txt",
    ))
    .expect("std manifest");
    let mut names = Vec::new();
    for row in manifest.lines().filter(|l| !l.starts_with('#')) {
        let cols: Vec<&str> = row.split('\t').collect();
        if cols.get(1) != Some(&"seed") {
            continue;
        }
        let src = std::fs::read_to_string(repo_path(&format!("std/{}.mind", cols[0])))
            .expect("seed module");
        for line in src.lines() {
            let t = line.trim_start();
            let t = t.strip_prefix("pub ").unwrap_or(t);
            if let Some(rest) = t.strip_prefix("struct ") {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                names.push(name);
            }
        }
    }
    names
}

/// A program declaring a struct with the name of a std seed struct made the
/// frozen compiler die with SIGSEGV (exit 139) even when it never calls std:
/// `struct Value { h: i64 }` beside the seed's `std/json.mind` `Value`. It must be
/// a named refusal from the bridge, before the frozen compiler runs.
#[test]
fn a_struct_named_like_a_std_seed_struct_is_refused_not_a_crash() {
    let names = seed_struct_names();
    // Positive control on the scan itself: the measured collisions are all present.
    for want in [
        "Vec", "String", "Map", "File", "Value", "Box", "Text", "TermSize", "Args",
    ] {
        assert!(names.iter().any(|n| n == want), "{want} not in {names:?}");
    }
    for name in &names {
        let src = format!(
            "struct {name} {{ h: i64 }}\nfn main() -> i64 {{ let v = {name} {{ h: 33 }}; return v.h; }}\n"
        );
        let stderr = assert_refused(&format!("sc_{name}"), &src, "closure.std_struct_collision");
        assert!(
            stderr.contains(&format!("`{name}`")),
            "{name} not named:\n{stderr}"
        );
    }
    // Declared inside a module block and never used: still the same image, still refused.
    assert_refused(
        "sc_block",
        "module m { struct File { a: i64 } }\nfn main() -> i64 { return 7; }\n",
        "closure.std_struct_collision",
    );
}

/// Positive control: a struct whose name the seed does not declare (including a
/// case-variant of a seed name, and an enum or alias spelled like one) still builds
/// and runs.
#[test]
fn a_struct_not_named_like_a_seed_struct_still_builds() {
    for (name, src, want) in [
        (
            "sc_pt",
            "struct Pt { x: i64, y: i64 }\nfn main() -> i64 { let p = Pt { x: 4, y: 5 }; \
             return p.x * 10 + p.y; }\n",
            45,
        ),
        (
            "sc_lower",
            "struct value { h: i64 }\nfn main() -> i64 { let v = value { h: 9 }; return v.h; }\n",
            9,
        ),
        (
            "sc_enum",
            "enum Value { A, B }\nfn main() -> i64 { return 7; }\n",
            7,
        ),
    ] {
        let p = Project::new(name);
        p.write("src/main.mind", src);
        let (code, stderr, bytes) = p.build_native("src/main.mind");
        assert_eq!(code, 0, "{name}: {stderr}");
        assert_native_result(&p, &bytes.expect("artifact"), want, name);
    }
}

/// A struct the program itself declares twice made the frozen compiler die with
/// SIGSEGV (exit 139), the same crash as a collision with a seed struct, in every
/// shape measured: top level (unused, used, with differing fields), a module block
/// beside top level, two module blocks, and twice in one function body (which the
/// MLIR backend builds and runs, exit 3). Each must be a named refusal from the
/// bridge, before the frozen compiler runs.
#[test]
fn a_struct_declared_twice_in_the_program_is_refused_not_a_crash() {
    for (name, src) in [
        (
            "ds_same",
            "struct Q { h: i64 }\nstruct Q { h: i64 }\nfn main() -> i64 { return 1; }\n",
        ),
        (
            "ds_used",
            "struct Q { h: i64 }\nstruct Q { h: i64 }\n\
             fn main() -> i64 { let q = Q { h: 5 }; return q.h; }\n",
        ),
        (
            "ds_fields",
            "struct Q { h: i64 }\nstruct Q { a: i64, b: i64 }\n\
             fn main() -> i64 { let q = Q { a: 5, b: 6 }; return q.b; }\n",
        ),
        (
            "ds_mod_top",
            "module m { struct Q { h: i64 } }\nstruct Q { h: i64 }\n\
             fn main() -> i64 { return 1; }\n",
        ),
        (
            "ds_two_mods",
            "module a { struct Q { h: i64 } }\nmodule b { struct Q { h: i64 } }\n\
             fn main() -> i64 { return 1; }\n",
        ),
        (
            "ds_nested",
            "fn main() -> i64 { struct Q { h: i64 } struct Q { h: i64 } return 3; }\n",
        ),
    ] {
        let stderr = assert_refused(name, src, "closure.duplicate_struct");
        assert!(stderr.contains("`Q`"), "{name}: `Q` not named:\n{stderr}");
    }
}

/// Positive control for the duplicate-struct refusal: it is keyed on a STRUCT
/// name declared twice, not on any repeated type name. A struct and an enum
/// sharing a name builds natively and returns what the MLIR backend returns
/// (measured: 1 on both), and two distinct struct names still build.
#[test]
fn a_struct_sharing_a_name_with_an_enum_still_builds() {
    for (name, src, want) in [
        (
            "ds_struct_enum",
            "struct Q { h: i64 }\nenum Q { A, B }\nfn main() -> i64 { return 1; }\n",
            1,
        ),
        (
            "ds_two_structs",
            "struct P { h: i64 }\nstruct Q { h: i64 }\n\
             fn main() -> i64 { let p = P { h: 2 }; let q = Q { h: 5 }; return p.h * q.h; }\n",
            10,
        ),
    ] {
        let p = Project::new(name);
        p.write("src/main.mind", src);
        let (code, stderr, bytes) = p.build_native("src/main.mind");
        assert_eq!(code, 0, "{name}: {stderr}");
        assert_native_result(&p, &bytes.expect("artifact"), want, name);
    }
}

/// A value reaching a slot declared `f64` (a `let` annotation, a parameter, a
/// return type) that is not PROVEN an f64 built natively and ran on its raw bits:
/// `let x: f64 = __mind_load_i64(p); return x;` exits 0 natively and 7 under MLIR,
/// and an i64 struct field read into an `f64` let likewise. The rest were
/// refused by the MLIR type checker (E2015 / E2027 / E2010) yet built natively.
/// A `let` annotation never reaches the IR, so the operand proof could not see any
/// of them; the bridge now refuses each by name before the frozen compiler runs.
#[test]
fn a_value_not_proven_f64_reaching_an_f64_slot_is_refused() {
    for (name, src) in [
        (
            "f64_launder_ret",
            "fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, 7); \
             let x: f64 = __mind_load_i64(p); return x; }\n",
        ),
        (
            "f64_let_operand",
            "fn main() -> i64 { let a: i64 = 4621819117588971520; let p: i64 = __mind_alloc(8); \
             __mind_store_i64(p, a); let x: f64 = a; if x > 5.0 { return 1; } return 2; }\n",
        ),
        (
            "f64_param_wrap",
            "fn big(x: f64) -> i64 { if x > 5.0 { return 1; } return 2; }\n\
             fn w(a: i64) -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, a); \
             return big(a); }\nfn main() -> i64 { return w(4621819117588971520); }\n",
        ),
        (
            "f64_ret_operand",
            "fn g(a: i64) -> f64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, a); \
             return a; }\nfn main() -> i64 { if g(4621819117588971520) > 5.0 { return 1; } \
             return 2; }\n",
        ),
        (
            "f64_launder_i64_param",
            "fn g(y: i64) -> i64 { if y > 5 { return 1; } return 2; }\n\
             fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, 7); \
             let x: f64 = __mind_load_i64(p); return g(x); }\n",
        ),
        (
            "f64_from_i64_field",
            "struct S { v: i64 }\nfn main() -> i64 { let s = S { v: 7 }; let x: f64 = s.v; \
             return x; }\n",
        ),
        (
            "f64_param_int_literal",
            "fn big(x: f64) -> i64 { if x > 5.0 { return 1; } return 2; }\n\
             fn main() -> i64 { return big(7); }\n",
        ),
    ] {
        assert_refused(name, src, "type.f64_slot_unproven");
    }
}

/// The other direction: a value proven an f64 reaching a slot declared otherwise.
/// Each built natively (`fn f() -> i64 { let x: f64 = 2.5; return x; }` exits 0)
/// where the MLIR backend refuses the program (E2015 / E2027 / an `mlir-opt`
/// return-type error).
#[test]
fn an_f64_value_reaching_a_non_f64_slot_is_refused() {
    for (name, src) in [
        (
            "f64_into_i64_ret",
            "fn f() -> i64 { let x: f64 = 2.5; return x; }\nfn main() -> i64 { return f(); }\n",
        ),
        (
            "f64_into_i64_let",
            "fn main() -> i64 { let b: f64 = 2.5; let a: i64 = b; return a; }\n",
        ),
        (
            "f64_into_i64_param",
            "fn g(y: i64) -> i64 { if y > 5 { return 1; } return 2; }\n\
             fn main() -> i64 { return g(2.5); }\n",
        ),
        (
            // `x` is declared i64 and, shadowed in a block, f64: the write must
            // satisfy both (native 2, MLIR refuses).
            "f64_shadow_assign",
            "fn main() -> i64 { let mut x: i64 = 1; if x == 1 { let mut x: f64 = 1.0; } \
             x = 2.5; return x; }\n",
        ),
        // An f64 that is not a PROVEN one (an `if` / `match` value, a name bound to
        // one) reaching an i64 slot: native exits 1 / 1 / 1 / 0, MLIR refuses all.
        (
            "f64_if_into_i64_let",
            "fn main() -> i64 { let c: i64 = 1; \
             let a: i64 = if c == 1 { 1.5 } else { 2.5 }; return a; }\n",
        ),
        (
            "f64_match_into_i64_let",
            "fn main() -> i64 { let c: i64 = 1; \
             let a: i64 = match c { 1 => 1.5, _ => 2.5 }; return a; }\n",
        ),
        (
            "f64_if_name_into_i64_let",
            "fn main() -> i64 { let c: i64 = 1; let y = if c == 1 { 1.5 } else { 2.5 }; \
             let a: i64 = y; return a; }\n",
        ),
        (
            "f64_if_name_into_i64_ret",
            "fn g(c: i64) -> i64 { let y = if c == 1 { 1.5 } else { 2.5 }; return y; }\n\
             fn main() -> i64 { return g(1); }\n",
        ),
        (
            // No declared return type: the slot is i64 (native 0; `mlir-opt`
            // refuses `return` of an f64 from an i64 function).
            "f64_into_unannotated_ret",
            "fn h() { return 1.5; }\nfn main() -> i64 { let a: i64 = h(); return a; }\n",
        ),
    ] {
        assert_refused(name, src, "type.f64_value_in_non_f64_slot");
    }
}

/// Positive control: an `f64` slot fed a proven f64 (a float literal, float
/// arithmetic, an `f64` binding, including an unannotated `let` of a float
/// literal, a call returning `f64`, a trailing float expression) still builds
/// and returns what the MLIR backend returns (measured on both).
#[test]
fn a_proven_f64_slot_still_builds() {
    for (name, src, want) in [
        (
            "f64_ok_param",
            "fn k(x: f64, y: f64) -> i64 { if x < y { return 1; } return 2; }\n\
             fn main() -> i64 { let a: f64 = 1.5; let b: f64 = a * 2.0; \
             return k(a, b) + k(3.0, 0.0 - 1.0); }\n",
            3,
        ),
        (
            "f64_ok_ret",
            "fn h(x: f64) -> f64 { return x * 2.0; }\n\
             fn main() -> i64 { let a: f64 = h(1.5); if a > 2.5 { return 4; } return 5; }\n",
            4,
        ),
        (
            "f64_ok_untyped",
            "fn k(x: f64) -> i64 { if x > 1.0 { return 1; } return 2; }\n\
             fn main() -> i64 { let a = 1.5; return k(a); }\n",
            1,
        ),
        (
            "f64_ok_tail",
            "fn h(x: f64) -> f64 { x * 2.0 }\n\
             fn main() -> i64 { if h(1.5) > 2.5 { return 4; } return 5; }\n",
            4,
        ),
    ] {
        let p = Project::new(name);
        p.write("src/main.mind", src);
        let (code, stderr, bytes) = p.build_native("src/main.mind");
        assert_eq!(code, 0, "{name}: {stderr}");
        assert_native_result(&p, &bytes.expect("artifact"), want, name);
    }
}

/// `bool` is the i64 truthiness encoding on both backends (`true`/`false` parse as
/// 1 / 0), so a bool slot fed an i64 needs no slot proof: an i64 variable (one also
/// an intrinsic operand), the literal 256 and an i64 returned from a `-> bool`
/// function all read as true natively and under MLIR (measured: 1, 1, 1 on both).
#[test]
fn a_bool_slot_fed_an_i64_matches_the_mlir_result() {
    for (name, src, want) in [
        (
            "bool_param_intr_operand",
            "fn f(b: bool) -> i64 { if b { return 1; } return 2; }\n\
             fn main() -> i64 { let a: i64 = 2; let p: i64 = __mind_alloc(8); \
             __mind_store_i64(p, a); return f(a); }\n",
            1,
        ),
        (
            "bool_param_lit",
            "fn f(b: bool) -> i64 { if b { return 1; } return 2; }\n\
             fn main() -> i64 { return f(256); }\n",
            1,
        ),
        (
            "bool_ret_i64var",
            "fn g() -> bool { let a: i64 = 256; return a; }\n\
             fn main() -> i64 { let b = g(); if b { return 1; } return 2; }\n",
            1,
        ),
    ] {
        let p = Project::new(name);
        p.write("src/main.mind", src);
        let (code, stderr, bytes) = p.build_native("src/main.mind");
        assert_eq!(code, 0, "{name}: {stderr}");
        assert_native_result(&p, &bytes.expect("artifact"), want, name);
    }
}
