// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0010 Phase C — Win64 calling convention + f32 vararg promotion.
//!
//! Test coverage:
//!
//! 1. Parse `extern "C" callconv(.win64) { ... }` — AST contains `CallConv::Win64`.
//! 2. Win64 1-byte struct → i8 (not i64 as SysV would produce).
//! 3. Win64 16-byte struct → `!llvm.ptr` (MEMORY); SysV → `i64, i64`.
//! 4. Win64 8-byte struct → i64; SysV → i64 (same result, different paths).
//! 5. Variadic call with f32 hint promotes to f64 in emitted MLIR.
//! 6. `llvm.func` and `llvm.call` for Win64 carry `cconv = #llvm.cconv<win64cc>`.
//! 7. SysV `llvm.func` and `llvm.call` do NOT carry a cconv attribute.
//! 8. Win64 3-byte struct → `!llvm.ptr` (MEMORY, not a valid register size).
//! 9. Win64 4-byte struct → i32.
//! 10. End-to-end parse + lower — Win64 extern block produces Win64-classified IR.
//!
//! Gate: `cargo test --release
//!        --features "mlir-build std-surface cross-module-imports"
//!        extern_c_phase_c`.

#![cfg(all(
    feature = "std-surface",
    any(feature = "mlir-lowering", feature = "mlir-build")
))]

use libmind::ast::{CallConv, Node, TypeAnn};
use libmind::eval::lower::win64_classify_struct;
use libmind::ir::{IRModule, Instr};
use libmind::mlir::lower_ir_to_mlir;
use libmind::parser;
use libmind::type_checker::{check_module_types_in_file, TypeEnv};

// ── helpers ───────────────────────────────────────────────────────────────────

fn empty_repr_c() -> std::collections::BTreeMap<String, Vec<TypeAnn>> {
    std::collections::BTreeMap::new()
}

// ── test 1: parse callconv(.win64) ────────────────────────────────────────────

/// `extern "C" callconv(.win64) { ... }` must parse to an `ExternBlock`
/// with `callconv: CallConv::Win64`.
#[test]
fn parse_callconv_win64_produces_ast_variant() {
    let src = r#"
        extern "C" callconv(.win64) {
            safe fn get_tick_count() -> i64
        }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert_eq!(module.items.len(), 1, "expected one item");
    match &module.items[0] {
        Node::ExternBlock { callconv, fns, .. } => {
            assert_eq!(
                *callconv,
                CallConv::Win64,
                "callconv(.win64) must produce CallConv::Win64; got {callconv:?}"
            );
            assert_eq!(fns.len(), 1, "expected one fn declaration");
            assert_eq!(fns[0].name, "get_tick_count");
        }
        other => panic!("expected ExternBlock, got {other:?}"),
    }
}

// ── test 2: Win64 1-byte struct → i8 ─────────────────────────────────────────

/// Under Win64, a struct of exactly 1 byte passes by value as `i8`.
/// Under SysV, the same struct would pass as `i64` (integer eightbyte).
#[test]
fn win64_one_byte_struct_to_i8() {
    // struct S { x: i8 }  — 1 byte total
    let fields = vec![TypeAnn::Named("i8".to_string())];
    let result = win64_classify_struct(&fields, &empty_repr_c());
    assert_eq!(
        result,
        vec!["i8".to_string()],
        "1-byte struct must classify to i8 under Win64; got {result:?}"
    );
}

// ── test 3: Win64 vs SysV — 16-byte struct ───────────────────────────────────

/// Win64: a 16-byte struct passes by pointer (!llvm.ptr — MEMORY class).
/// SysV:  the same struct passes as two i64 eightbytes.
#[test]
fn win64_sixteen_byte_struct_to_llvm_ptr_vs_sysv_two_i64() {
    use libmind::eval::lower::sysv_classify_struct;

    // struct S { a: i64, b: i64 }  — 16 bytes total, all-integer
    let fields = vec![TypeAnn::ScalarI64, TypeAnn::ScalarI64];

    let win64_result = win64_classify_struct(&fields, &empty_repr_c());
    assert_eq!(
        win64_result,
        vec!["!llvm.ptr".to_string()],
        "16-byte struct must be MEMORY (!llvm.ptr) under Win64; got {win64_result:?}"
    );

    let sysv_result = sysv_classify_struct(&fields, &empty_repr_c());
    assert_eq!(
        sysv_result,
        vec!["i64".to_string(), "i64".to_string()],
        "16-byte all-int struct must be two i64s under SysV; got {sysv_result:?}"
    );

    // The two ABIs must produce different results for this case.
    assert_ne!(
        win64_result, sysv_result,
        "Win64 and SysV must classify 16-byte struct differently"
    );
}

// ── test 4: Win64 vs SysV — 8-byte struct produces same i64 ──────────────────

/// Both Win64 and SysV pass an exactly-8-byte struct as a single i64.
#[test]
fn win64_and_sysv_agree_on_eight_byte_struct() {
    use libmind::eval::lower::sysv_classify_struct;

    // struct S { a: i32, b: i32 }  — 8 bytes total
    let fields = vec![TypeAnn::ScalarI32, TypeAnn::ScalarI32];

    let win64_result = win64_classify_struct(&fields, &empty_repr_c());
    let sysv_result = sysv_classify_struct(&fields, &empty_repr_c());

    assert_eq!(
        win64_result,
        vec!["i64".to_string()],
        "8-byte struct must be i64 under Win64; got {win64_result:?}"
    );
    assert_eq!(
        sysv_result,
        vec!["i64".to_string()],
        "8-byte struct must be i64 under SysV; got {sysv_result:?}"
    );
    assert_eq!(win64_result, sysv_result, "Win64 and SysV must agree on 8-byte struct");
}

// ── test 5: f32 vararg hint promotes to f64 (R-03) ───────────────────────────

/// When a vararg hint specifies `f32`, the emitted `llvm.call` must use `f64`
/// per C11 §6.5.2.2p6 default argument promotions.
///
/// This addresses Phase B audit finding R-03 (F-10).
#[test]
fn vararg_f32_hint_promotes_to_f64_in_mlir() {
    let mut m = IRModule::new();
    // Declare a variadic function with an f32 vararg hint.
    m.instrs.push(Instr::ExternFnDecl {
        name: "log_float".to_string(),
        param_types: vec!["!llvm.ptr".to_string()], // fmt string
        ret_type: Some("i64".to_string()),
        is_varargs: true,
        // f32 hint — must be promoted to f64 in the emission.
        vararg_hints: vec!["f32".to_string()],
        callconv: CallConv::SysV,
    });

    let fmt = m.fresh();
    m.instrs.push(Instr::ConstI64(fmt, 0));
    let float_arg = m.fresh();
    m.instrs.push(Instr::ConstI64(float_arg, 1));

    let dst = m.fresh();
    m.instrs.push(Instr::Call {
        dst,
        name: "log_float".to_string(),
        args: vec![fmt, float_arg],
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m)
        .expect("lowering must succeed")
        .text;

    // The emitted call must use f64 for the vararg position, NOT f32.
    assert!(
        text.contains("f64"),
        "f32 vararg hint must be promoted to f64 in llvm.call; got:\n{text}"
    );
    assert!(
        !text.contains(", f32,") && !text.contains(", f32)"),
        "f32 must not appear as a vararg type in llvm.call; got:\n{text}"
    );
    assert!(
        text.contains("llvm.call @log_float"),
        "must emit llvm.call @log_float; got:\n{text}"
    );
}

// ── test 6: Win64 llvm.func and llvm.call carry cconv attribute ───────────────

/// An `extern "C" callconv(.win64)` declaration must produce:
/// - `llvm.func cconv = #llvm.cconv<win64cc> @name(...)` at module level.
/// - `llvm.call cconv = #llvm.cconv<win64cc> @name(...)` at call sites.
#[test]
fn win64_extern_decl_emits_cconv_attribute() {
    let mut m = IRModule::new();
    m.instrs.push(Instr::ExternFnDecl {
        name: "win_api_fn".to_string(),
        param_types: vec!["i64".to_string()],
        ret_type: Some("i64".to_string()),
        is_varargs: false,
        vararg_hints: Vec::new(),
        callconv: CallConv::Win64,
    });

    let arg = m.fresh();
    m.instrs.push(Instr::ConstI64(arg, 42));
    let dst = m.fresh();
    m.instrs.push(Instr::Call {
        dst,
        name: "win_api_fn".to_string(),
        args: vec![arg],
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m)
        .expect("lowering must succeed")
        .text;

    // llvm.func declaration must carry the cconv attribute.
    assert!(
        text.contains("llvm.func cconv = #llvm.cconv<win64cc> @win_api_fn"),
        "Win64 llvm.func must carry cconv = #llvm.cconv<win64cc>; got:\n{text}"
    );
    // llvm.call must also carry the cconv attribute.
    assert!(
        text.contains("llvm.call cconv = #llvm.cconv<win64cc> @win_api_fn"),
        "Win64 llvm.call must carry cconv = #llvm.cconv<win64cc>; got:\n{text}"
    );
}

// ── test 7: SysV llvm.func and llvm.call have no cconv attribute ──────────────

/// A SysV extern declaration must NOT carry `cconv = #llvm.cconv<win64cc>`.
/// The absence of a cconv attribute is MLIR's default (C/SysV on Linux).
#[test]
fn sysv_extern_decl_has_no_cconv_attribute() {
    let mut m = IRModule::new();
    m.instrs.push(Instr::ExternFnDecl {
        name: "posix_fn".to_string(),
        param_types: vec!["i64".to_string()],
        ret_type: Some("i64".to_string()),
        is_varargs: false,
        vararg_hints: Vec::new(),
        callconv: CallConv::SysV,
    });

    let arg = m.fresh();
    m.instrs.push(Instr::ConstI64(arg, 0));
    let dst = m.fresh();
    m.instrs.push(Instr::Call {
        dst,
        name: "posix_fn".to_string(),
        args: vec![arg],
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m)
        .expect("lowering must succeed")
        .text;

    assert!(
        !text.contains("cconv"),
        "SysV extern must not carry any cconv attribute; got:\n{text}"
    );
    assert!(
        text.contains("llvm.func @posix_fn"),
        "must emit llvm.func @posix_fn; got:\n{text}"
    );
    assert!(
        text.contains("llvm.call @posix_fn"),
        "must emit llvm.call @posix_fn; got:\n{text}"
    );
}

// ── test 8: Win64 3-byte struct → !llvm.ptr (MEMORY) ─────────────────────────

/// A 3-byte struct cannot fit in any standard register size; Win64 sends it
/// by pointer (MEMORY class). Size 3 is not in {1, 2, 4, 8}.
#[test]
fn win64_three_byte_struct_to_memory() {
    // struct S { a: i8, b: i16 }  — 1+2 = 3 bytes total
    let fields = vec![
        TypeAnn::Named("i8".to_string()),
        TypeAnn::Named("i16".to_string()),
    ];
    let result = win64_classify_struct(&fields, &empty_repr_c());
    assert_eq!(
        result,
        vec!["!llvm.ptr".to_string()],
        "3-byte struct must be MEMORY under Win64; got {result:?}"
    );
}

// ── test 9: Win64 4-byte struct → i32 ────────────────────────────────────────

/// A 4-byte struct passes by value as `i32` under Win64.
#[test]
fn win64_four_byte_struct_to_i32() {
    // struct S { a: i16, b: i16 }  — 2+2 = 4 bytes total
    let fields = vec![
        TypeAnn::Named("i16".to_string()),
        TypeAnn::Named("i16".to_string()),
    ];
    let result = win64_classify_struct(&fields, &empty_repr_c());
    assert_eq!(
        result,
        vec!["i32".to_string()],
        "4-byte struct must classify to i32 under Win64; got {result:?}"
    );
}

// ── test 10: end-to-end parse + lower for Win64 extern block ─────────────────

/// Parse a module with `extern "C" callconv(.win64)` referencing a
/// `#[repr(C)]` struct, lower to IR, and verify Win64-classified param types.
///
/// A 16-byte all-integer struct:
/// - Win64: `!llvm.ptr` (MEMORY)
/// - SysV:  `i64, i64` (two eightbytes)
#[test]
fn end_to_end_win64_struct_lower_to_ir() {
    let src = r#"
        [repr(C)]
        struct TwoI64 { a: i64, b: i64 }

        extern "C" callconv(.win64) {
            unsafe fn win_consume(s: TwoI64) -> i32
        }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    // Type-check: must succeed (struct is repr(C)).
    let diags = check_module_types_in_file(&module, src, None, &TypeEnv::new());
    assert!(
        diags.is_empty(),
        "type-check must succeed for Win64 repr(C) extern block; diags: {diags:?}"
    );

    // Lower to IR.
    use libmind::eval::lower_to_ir;
    let ir = lower_to_ir(&module);

    // Find the ExternFnDecl for win_consume.
    let extern_decl = ir.instrs.iter().find_map(|instr| {
        if let Instr::ExternFnDecl { name, param_types, callconv, .. } = instr {
            if name == "win_consume" {
                return Some((param_types.clone(), *callconv));
            }
        }
        None
    });

    let (param_types, callconv) = extern_decl
        .expect("must have ExternFnDecl for win_consume");

    // Win64 16-byte struct → MEMORY (!llvm.ptr).
    assert_eq!(
        param_types,
        vec!["!llvm.ptr".to_string()],
        "Win64: 16-byte struct must lower to !llvm.ptr; got {param_types:?}"
    );

    // Callconv must be recorded as Win64 in the IR.
    assert_eq!(
        callconv,
        CallConv::Win64,
        "ExternFnDecl must record CallConv::Win64; got {callconv:?}"
    );
}
