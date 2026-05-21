// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0010 Phase A — `extern "C"` parser + type-checker + LLVM call lowering.
//!
//! Seven tests that cover the full Phase A surface:
//!
//! 1. Parse a bare `extern "C"` block with one function declaration.
//! 2. Parse `safe`/`unsafe` per-symbol attribution.
//! 3. Parse the optional `callconv(.sysv)` annotation.
//! 4. Parse varargs `...` in the parameter list.
//! 5. Type-checker rejects aggregate types in extern signatures
//!    (`safety::extern_non_copy` diagnostic).
//! 6. `Instr::ExternFnDecl` causes the MLIR assembler to emit an
//!    `llvm.func` module-level declaration.
//! 7. A `func.call` that targets a symbol declared via `extern "C"`
//!    is emitted as `llvm.call` instead of `func.call`.
//!
//! Gated: `cargo test --features "std-surface mlir-lowering"
//!         --test extern_c_phase_a`.

#![cfg(all(feature = "std-surface", feature = "mlir-lowering"))]

use libmind::ast::{CallConv, ExternFn, Module, Node, Param, Span, TypeAnn};
use libmind::ir::{IRModule, Instr};
use libmind::mlir::lower_ir_to_mlir;
use libmind::parser;
use libmind::type_checker::{check_module_types_in_file, TypeEnv};

// ── helpers ──────────────────────────────────────────────────────────────────

fn sp() -> Span {
    Span::new(0, 0)
}

/// Build an `IRModule` that has one `ExternFnDecl` for `puts(i64) -> i64`.
fn module_with_extern_decl() -> IRModule {
    let mut m = IRModule::new();
    m.instrs.push(Instr::ExternFnDecl {
        name: "puts".to_string(),
        param_types: vec!["i64".to_string()],
        ret_type: Some("i64".to_string()),
        is_varargs: false,
        vararg_hints: Vec::new(),
        callconv: CallConv::SysV,
    });
    // Emit a zero constant so @main has at least one value.
    let id = m.fresh();
    m.instrs.push(Instr::ConstI64(id, 0));
    m.instrs.push(Instr::Output(id));
    m
}

/// Build an `IRModule` where `printf` is declared as a varargs extern and then
/// called with two concrete arguments so the `llvm.call` path is exercised.
fn module_with_extern_call() -> IRModule {
    let mut m = IRModule::new();
    // Declare printf(i64, ...) -> i64  (fmt ptr + vararg int).
    m.instrs.push(Instr::ExternFnDecl {
        name: "printf".to_string(),
        param_types: vec!["i64".to_string()],
        ret_type: Some("i64".to_string()),
        is_varargs: true,
        vararg_hints: Vec::new(),
        callconv: CallConv::SysV,
    });
    // Synthesise two integer arguments.
    let fmt_ptr = m.fresh();
    m.instrs.push(Instr::ConstI64(fmt_ptr, 0));
    let val = m.fresh();
    m.instrs.push(Instr::ConstI64(val, 42));
    // Call printf(fmt_ptr, val).
    let dst = m.fresh();
    m.instrs.push(Instr::Call {
        dst,
        name: "printf".to_string(),
        args: vec![fmt_ptr, val],
    });
    m.instrs.push(Instr::Output(dst));
    m
}

// ── test 1: parse a bare extern "C" block ────────────────────────────────────

/// A minimal `extern "C"` block with one opaque function declaration must
/// parse without error and produce exactly one `Node::ExternBlock` item.
#[test]
fn parse_bare_extern_c_block() {
    let src = r#"extern "C" { fn puts(s: i64) -> i64 }"#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert_eq!(module.items.len(), 1, "expected exactly one top-level item");
    match &module.items[0] {
        Node::ExternBlock { callconv, fns, .. } => {
            assert_eq!(*callconv, CallConv::C);
            assert_eq!(fns.len(), 1);
            assert_eq!(fns[0].name, "puts");
            assert_eq!(fns[0].params.len(), 1);
            assert_eq!(fns[0].params[0].name, "s");
        }
        other => panic!("expected ExternBlock, got {other:?}"),
    }
}

// ── test 2: safe / unsafe attribution ────────────────────────────────────────

/// `safe fn` and `unsafe fn` inside `extern "C"` must be correctly tagged in
/// the AST.  A bare `fn` (no keyword) defaults to `unsafe`.
#[test]
fn parse_safe_and_unsafe_attribution() {
    let src = r#"
        extern "C" {
            safe   fn query(x: i32) -> i32
            unsafe fn mutate(p: i64)
                   fn bare(n: i64) -> i64
        }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let Node::ExternBlock { fns, .. } = &module.items[0]
    else { panic!("expected ExternBlock") };

    assert!(!fns[0].is_unsafe, "`safe fn query` must have is_unsafe = false");
    assert!(fns[1].is_unsafe, "`unsafe fn mutate` must have is_unsafe = true");
    assert!(fns[2].is_unsafe, "bare `fn bare` must default to is_unsafe = true");
}

// ── test 3: callconv annotation ──────────────────────────────────────────────

/// `callconv(.sysv)`, `callconv(.win64)`, and `callconv(.aapcs)` must all
/// parse to the correct `CallConv` variant.  The block body is stored; the
/// annotation is informational in Phase A (lowering uses the platform default).
#[test]
fn parse_callconv_annotation() {
    for (tag, expected) in [
        ("sysv", CallConv::SysV),
        ("win64", CallConv::Win64),
        ("aapcs", CallConv::Aapcs),
        ("c", CallConv::C),
    ] {
        let src = format!(r#"extern "C" callconv(.{tag}) {{ fn f(x: i64) -> i64 }}"#);
        let module = parser::parse(&src)
            .unwrap_or_else(|e| panic!("parse failed for .{tag}: {e:?}"));
        let Node::ExternBlock { callconv, .. } = &module.items[0]
        else { panic!("expected ExternBlock for .{tag}") };
        assert_eq!(
            *callconv, expected,
            "callconv(.{tag}) must parse to {expected:?}"
        );
    }
}

// ── test 4: varargs `...` ─────────────────────────────────────────────────────

/// A declaration with a varargs tail `...` must set `is_varargs = true`.
/// The concrete parameters before `...` must be captured normally.
#[test]
fn parse_varargs_declaration() {
    let src = r#"extern "C" { fn printf(fmt: i64, ...) -> i64 }"#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let Node::ExternBlock { fns, .. } = &module.items[0]
    else { panic!("expected ExternBlock") };
    let printf = &fns[0];
    assert_eq!(printf.name, "printf");
    assert!(printf.is_varargs, "`printf` must be marked varargs");
    assert_eq!(printf.params.len(), 1, "one concrete param before `...`");
    assert_eq!(printf.params[0].name, "fmt");
}

// ── test 5: type-checker rejects aggregate types ──────────────────────────────

/// Any aggregate or non-Copy type in an `extern "C"` signature must produce a
/// `safety::extern_non_copy` diagnostic.  Tensors are the canonical example.
#[test]
fn typecheck_rejects_tensor_in_extern_signature() {
    // Build the AST directly — a Tensor param in an extern fn.
    let efn = ExternFn {
        is_unsafe: true,
        name: "bad_fn".to_string(),
        params: vec![Param {
            name: "x".to_string(),
            ty: TypeAnn::Tensor {
                dtype: "f32".to_string(),
                dims: vec![],
            },
            span: sp(),
        }],
        ret_type: Some(TypeAnn::ScalarI64),
        is_varargs: false,
        span: sp(),
    };
    let module = Module {
        items: vec![Node::ExternBlock {
            callconv: CallConv::C,
            fns: vec![efn],
            span: sp(),
        }],
    };

    let diags = check_module_types_in_file(&module, "", None, &TypeEnv::new());
    assert!(
        !diags.is_empty(),
        "expected at least one diagnostic for a Tensor param in extern fn"
    );
    let combined: String = diags.iter().map(|d| format!("{d:?}")).collect();
    assert!(
        combined.contains("extern_non_copy"),
        "diagnostic must reference `extern_non_copy`; got: {combined}"
    );
}

// ── test 6: ExternFnDecl → llvm.func declaration ─────────────────────────────

/// When an `Instr::ExternFnDecl` is present, the MLIR assembler must emit an
/// `llvm.func @name(type...) -> ret` module-level declaration.
#[test]
fn extern_fn_decl_emits_llvm_func_declaration() {
    let m = module_with_extern_decl();
    let text = lower_ir_to_mlir(&m)
        .expect("lowering must succeed")
        .text;

    assert!(
        text.contains("llvm.func @puts"),
        "expected `llvm.func @puts` declaration; got:\n{text}"
    );
    assert!(
        text.contains("-> i64"),
        "expected `-> i64` return in llvm.func; got:\n{text}"
    );
    // Must NOT fall through to the func.func private path for extern "C" fns.
    assert!(
        !text.contains("func.func private @puts"),
        "extern \"C\" symbols must not get a func.func private decl; got:\n{text}"
    );
}

// ── test 7: call to extern "C" fn → llvm.call ────────────────────────────────

/// A `func.call`-style `Instr::Call` whose callee was declared via
/// `Instr::ExternFnDecl` must lower to `llvm.call @name(...)` rather than
/// `func.call @name(...)`.  Varargs (`...`) must appear in the type signature.
#[test]
fn extern_c_call_lowers_to_llvm_call() {
    let m = module_with_extern_call();
    let text = lower_ir_to_mlir(&m)
        .expect("lowering must succeed")
        .text;

    assert!(
        text.contains("llvm.call @printf("),
        "call to extern \"C\" printf must lower to llvm.call; got:\n{text}"
    );
    assert!(
        !text.contains("func.call @printf"),
        "must NOT emit func.call for an extern \"C\" callee; got:\n{text}"
    );
    // Varargs must appear in the call type signature.
    assert!(
        text.contains(", ..."),
        "varargs `...` must appear in the llvm.call type signature; got:\n{text}"
    );
    // The llvm.func declaration must also be present.
    assert!(
        text.contains("llvm.func @printf"),
        "module must declare llvm.func @printf; got:\n{text}"
    );
}
