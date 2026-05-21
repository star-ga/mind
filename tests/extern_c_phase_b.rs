// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0010 Phase B — SysV x86_64 struct passing, variadics, callback pointers.
//!
//! Test coverage:
//!
//! 1. Parse `#[repr(C)]` struct attribute.
//! 2. `#[repr(C)]` struct accepted in extern "C" signature (type-checker).
//! 3. SysV classification — all-int 8B struct → single i64.
//! 4. SysV classification — all-int 16B struct → two i64s.
//! 5. SysV classification — all-float 8B struct → f64 or f32.
//! 6. SysV classification — all-float 16B struct → two float slots.
//! 7. SysV classification — >16B struct → MEMORY (!llvm.ptr).
//! 8. SysV classification — mixed int+float → MEMORY (!llvm.ptr).
//! 9. Parse `extern "C" fn(T) -> R` callback function pointer type.
//! 10. Callback FnPtr accepted in extern "C" signature (type-checker).
//! 11. Callback FnPtr lowers to !llvm.ptr in MLIR declaration.
//! 12. Variadic printf call — vararg_hints produce precise per-position types.
//! 13. MLIR emission for struct-valued extern parameter (SysV-classified).
//!
//! Gate: `cargo test --features "std-surface mlir-lowering"
//!        --test extern_c_phase_b`.

#![cfg(all(feature = "std-surface", feature = "mlir-lowering"))]

use libmind::ast::{CallConv, ExternFn, Module, Node, Param, Span, TypeAnn};
use libmind::eval::lower::sysv_classify_struct;
use libmind::eval::lower::SysVClass;
use libmind::eval::lower::classify_scalar_field;
use libmind::ir::{IRModule, Instr};
use libmind::mlir::lower_ir_to_mlir;
use libmind::parser;
use libmind::type_checker::{check_module_types_in_file, TypeEnv};

// ── helpers ───────────────────────────────────────────────────────────────────

fn sp() -> Span {
    Span::new(0, 0)
}

fn empty_repr_c() -> std::collections::BTreeMap<String, Vec<TypeAnn>> {
    std::collections::BTreeMap::new()
}

// ── test 1: parse #[repr(C)] attribute on a struct ───────────────────────────

/// A struct decorated with `[repr(C)]` must produce an `Attribute` whose
/// name is "repr" and whose single argument is "C".
#[test]
fn parse_repr_c_attribute_on_struct() {
    let src = r#"[repr(C)] struct Timeval { tv_sec: i64, tv_usec: i64 }"#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert_eq!(module.items.len(), 1);
    match &module.items[0] {
        Node::StructDef { name, attrs, fields, .. } => {
            assert_eq!(name, "Timeval");
            assert_eq!(fields.len(), 2);
            assert!(
                attrs.iter().any(|a| a.name == "repr" && a.args.iter().any(|s| s == "C")),
                "expected [repr(C)] attribute; attrs = {attrs:?}"
            );
        }
        other => panic!("expected StructDef, got {other:?}"),
    }
}

// ── test 2: repr(C) struct accepted in extern "C" signature ──────────────────

/// A `Named` type in an `extern "C"` signature must be accepted by the
/// Phase B type-checker when the struct carries `#[repr(C)]`.
/// The module must include the struct definition so the type-checker's
/// pre-pass can populate its repr(C) registry (audit fix F-06).
#[test]
fn typecheck_accepts_named_type_in_extern_signature() {
    // Use the parser so the repr_c pre-pass has a real StructDef to scan.
    let src = r#"
        [repr(C)]
        struct Timeval { tv_sec: i64, tv_usec: i64 }

        extern "C" {
            safe fn accept_timeval(tv: Timeval) -> i32
        }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let diags = check_module_types_in_file(&module, src, None, &TypeEnv::new());
    // A repr(C)-annotated struct must be accepted; no diagnostic expected.
    assert!(
        diags.is_empty(),
        "expected no diagnostics for repr(C) Named type in extern \"C\" signature; got: {diags:?}"
    );
}

// ── test 3: SysV — all-int 8B struct → single i64 ────────────────────────────

/// A `#[repr(C)]` struct with two i32 fields (total 8 bytes, all-integer)
/// classifies to a single `i64` eightbyte under SysV.
#[test]
fn sysv_all_int_8b_struct_to_single_i64() {
    let fields = vec![TypeAnn::ScalarI32, TypeAnn::ScalarI32]; // 4 + 4 = 8B
    let result = sysv_classify_struct(&fields, &empty_repr_c());
    assert_eq!(result, vec!["i64".to_string()],
        "8B all-int struct must classify to a single i64; got {result:?}");
}

// ── test 4: SysV — all-int 16B struct → two i64s ─────────────────────────────

/// A `#[repr(C)]` struct with two i64 fields (total 16 bytes, all-integer)
/// classifies to two `i64` eightbytes under SysV.
#[test]
fn sysv_all_int_16b_struct_to_two_i64() {
    let fields = vec![TypeAnn::ScalarI64, TypeAnn::ScalarI64]; // 8 + 8 = 16B
    let result = sysv_classify_struct(&fields, &empty_repr_c());
    assert_eq!(result, vec!["i64".to_string(), "i64".to_string()],
        "16B all-int struct must classify to two i64s; got {result:?}");
}

// ── test 5: SysV — all-float 8B struct → f64 ─────────────────────────────────

/// A single f64 field (8 bytes, float) classifies to a single `f64` eightbyte.
#[test]
fn sysv_all_float_8b_struct_to_f64() {
    let fields = vec![TypeAnn::ScalarF64]; // 8B float
    let result = sysv_classify_struct(&fields, &empty_repr_c());
    assert_eq!(result, vec!["f64".to_string()],
        "8B all-float struct (f64) must classify to f64; got {result:?}");
}

/// Two f32 fields (total 8 bytes, float) classify to a single `f32` eightbyte
/// (f32 does not promote to f64 when no f64 is present).
#[test]
fn sysv_all_float_8b_f32_pair_to_f32() {
    let fields = vec![TypeAnn::ScalarF32, TypeAnn::ScalarF32]; // 4 + 4 = 8B
    let result = sysv_classify_struct(&fields, &empty_repr_c());
    assert_eq!(result, vec!["f32".to_string()],
        "8B all-float struct (2xf32) must classify to f32; got {result:?}");
}

// ── test 6: SysV — all-float 16B struct → two float slots ────────────────────

/// Two f64 fields (total 16 bytes, all-float) classify to two `f64` slots.
#[test]
fn sysv_all_float_16b_struct_to_two_f64() {
    let fields = vec![TypeAnn::ScalarF64, TypeAnn::ScalarF64]; // 8 + 8 = 16B
    let result = sysv_classify_struct(&fields, &empty_repr_c());
    assert_eq!(result, vec!["f64".to_string(), "f64".to_string()],
        "16B all-float struct (2xf64) must classify to two f64s; got {result:?}");
}

// ── test 7: SysV — >16B struct → MEMORY (!llvm.ptr) ─────────────────────────

/// A struct larger than 16 bytes must classify to MEMORY (`!llvm.ptr`).
#[test]
fn sysv_large_struct_to_memory_class() {
    // Three i64 fields = 24 bytes > 16.
    let fields = vec![TypeAnn::ScalarI64, TypeAnn::ScalarI64, TypeAnn::ScalarI64];
    let result = sysv_classify_struct(&fields, &empty_repr_c());
    assert_eq!(result, vec!["!llvm.ptr".to_string()],
        ">16B struct must classify to MEMORY (!llvm.ptr); got {result:?}");
}

// ── test 8: SysV — mixed int+float → MEMORY ──────────────────────────────────

/// A struct with one integer field and one float field must classify to
/// MEMORY (`!llvm.ptr`) under Phase B's simplified mixed-class rule.
#[test]
fn sysv_mixed_int_float_struct_to_memory_class() {
    let fields = vec![TypeAnn::ScalarI32, TypeAnn::ScalarF32]; // 4B int + 4B float = 8B mixed
    let result = sysv_classify_struct(&fields, &empty_repr_c());
    assert_eq!(result, vec!["!llvm.ptr".to_string()],
        "mixed int+float struct must classify to MEMORY (!llvm.ptr); got {result:?}");
}

// ── test 9: parse extern "C" fn(T) -> R callback type ────────────────────────

/// A parameter type of the form `extern "C" fn(T, U) -> R` must parse to
/// `TypeAnn::FnPtr { params: [T, U], ret: Some(R) }`.
#[test]
fn parse_fn_ptr_callback_type() {
    let src = r#"
        extern "C" {
            unsafe fn qsort(
                base: *mut u8,
                nmemb: i64,
                size: i64,
                compar: extern "C" fn(*const u8, *const u8) -> i32
            )
        }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let Node::ExternBlock { fns, .. } = &module.items[0]
    else { panic!("expected ExternBlock") };

    let qsort = &fns[0];
    assert_eq!(qsort.name, "qsort");
    assert_eq!(qsort.params.len(), 4);

    let compar_ty = &qsort.params[3].ty;
    match compar_ty {
        TypeAnn::FnPtr { params, ret } => {
            assert_eq!(params.len(), 2, "compar must have 2 params");
            assert!(
                matches!(ret.as_deref(), Some(TypeAnn::ScalarI32)),
                "compar must return i32; got {ret:?}"
            );
        }
        other => panic!("expected FnPtr for compar, got {other:?}"),
    }
}

/// A no-parameter, no-return callback `extern "C" fn()` must parse correctly.
#[test]
fn parse_fn_ptr_no_params_no_ret() {
    let src = r#"
        extern "C" {
            unsafe fn atexit(func: extern "C" fn())
        }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let Node::ExternBlock { fns, .. } = &module.items[0]
    else { panic!("expected ExternBlock") };
    let atexit = &fns[0];
    assert_eq!(atexit.params.len(), 1);
    match &atexit.params[0].ty {
        TypeAnn::FnPtr { params, ret } => {
            assert!(params.is_empty(), "atexit callback must have no params");
            assert!(ret.is_none(), "atexit callback must return void (None)");
        }
        other => panic!("expected FnPtr, got {other:?}"),
    }
}

// ── test 10: FnPtr accepted in extern "C" signature (type-checker) ────────────

/// A `FnPtr` type in an `extern "C"` parameter must pass type-checking.
#[test]
fn typecheck_accepts_fn_ptr_in_extern_signature() {
    let efn = ExternFn {
        is_unsafe: true,
        name: "qsort".to_string(),
        params: vec![
            Param {
                name: "base".to_string(),
                ty: TypeAnn::RawPtr {
                    mutable: true,
                    pointee: Box::new(TypeAnn::Named("u8".to_string())),
                },
                span: sp(),
            },
            Param {
                name: "compar".to_string(),
                ty: TypeAnn::FnPtr {
                    params: vec![
                        TypeAnn::RawPtr {
                            mutable: false,
                            pointee: Box::new(TypeAnn::Named("u8".to_string())),
                        },
                        TypeAnn::RawPtr {
                            mutable: false,
                            pointee: Box::new(TypeAnn::Named("u8".to_string())),
                        },
                    ],
                    ret: Some(Box::new(TypeAnn::ScalarI32)),
                },
                span: sp(),
            },
        ],
        ret_type: None,
        is_varargs: false,
        span: sp(),
    };
    let module = Module {
        items: vec![Node::ExternBlock {
            callconv: CallConv::SysV,
            fns: vec![efn],
            span: sp(),
        }],
    };
    let diags = check_module_types_in_file(&module, "", None, &TypeEnv::new());
    assert!(
        diags.is_empty(),
        "FnPtr param must be accepted in extern \"C\" signature; got diags: {diags:?}"
    );
}

// ── test 11: FnPtr lowers to !llvm.ptr in MLIR declaration ───────────────────

/// When an `extern "C"` declaration has a `FnPtr` parameter type, the
/// generated `llvm.func` declaration must use `!llvm.ptr` for that parameter.
#[test]
fn fn_ptr_param_lowers_to_llvm_ptr_in_mlir() {
    let mut m = IRModule::new();
    // Declare signal(i64, !llvm.ptr) -> !llvm.ptr
    m.instrs.push(Instr::ExternFnDecl {
        name: "signal".to_string(),
        param_types: vec!["i64".to_string(), "!llvm.ptr".to_string()],
        ret_type: Some("!llvm.ptr".to_string()),
        is_varargs: false,
        vararg_hints: Vec::new(),
        callconv: CallConv::SysV,
    });
    let id = m.fresh();
    m.instrs.push(Instr::ConstI64(id, 0));
    m.instrs.push(Instr::Output(id));

    let text = lower_ir_to_mlir(&m)
        .expect("lowering must succeed")
        .text;

    assert!(
        text.contains("llvm.func @signal"),
        "must emit llvm.func @signal; got:\n{text}"
    );
    assert!(
        text.contains("!llvm.ptr"),
        "llvm.func @signal must use !llvm.ptr for fn-ptr param; got:\n{text}"
    );
}

// ── test 12: variadic printf with precise per-position types ─────────────────

/// `printf(fmt_ptr: !llvm.ptr, ...) -> i32` called with a format pointer and
/// an i64 integer value.  `vararg_hints` must produce `!llvm.ptr` for the
/// first variadic position and `i64` for the second.
#[test]
fn variadic_printf_call_uses_vararg_hints() {
    let mut m = IRModule::new();
    // Declare printf(!llvm.ptr, ...) -> i64 with vararg hints.
    m.instrs.push(Instr::ExternFnDecl {
        name: "printf".to_string(),
        param_types: vec!["!llvm.ptr".to_string()],
        ret_type: Some("i64".to_string()),
        is_varargs: true,
        vararg_hints: vec!["!llvm.ptr".to_string(), "i64".to_string()],
        callconv: CallConv::SysV,
    });

    // Three arguments: fmt pointer, string arg, int arg.
    let fmt = m.fresh();
    m.instrs.push(Instr::ConstI64(fmt, 0));
    let str_arg = m.fresh();
    m.instrs.push(Instr::ConstI64(str_arg, 1));
    let int_arg = m.fresh();
    m.instrs.push(Instr::ConstI64(int_arg, 42));

    let dst = m.fresh();
    m.instrs.push(Instr::Call {
        dst,
        name: "printf".to_string(),
        args: vec![fmt, str_arg, int_arg],
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m)
        .expect("lowering must succeed")
        .text;

    // The llvm.call type signature must list:
    //   (!llvm.ptr, !llvm.ptr, i64, ...) -> i64
    assert!(
        text.contains("llvm.call @printf"),
        "must emit llvm.call @printf; got:\n{text}"
    );
    assert!(
        text.contains(", ..."),
        "varargs must appear in llvm.call type signature; got:\n{text}"
    );
    // The first vararg position must use !llvm.ptr (from hints[0]).
    assert!(
        text.contains("!llvm.ptr, !llvm.ptr, i64, ..."),
        "llvm.call must use !llvm.ptr for ptr arg and i64 for int arg per hints;\ngot:\n{text}"
    );
}

// ── test 13: MLIR emission for struct-valued extern param ─────────────────────

/// A `#[repr(C)]` struct with two i64 fields (SysV: two i64 eightbytes) passed
/// to an extern "C" function: the `llvm.func` declaration must list two `i64`
/// parameters (not one struct type or !llvm.ptr).
#[test]
fn repr_c_struct_param_expands_to_two_i64_in_mlir() {
    let mut m = IRModule::new();
    // Pre-populate repr_c_structs: Timeval = { tv_sec: i64, tv_usec: i64 }
    m.repr_c_structs.insert(
        "Timeval".to_string(),
        vec![TypeAnn::ScalarI64, TypeAnn::ScalarI64],
    );

    // The ExternFnDecl carries the already-classified types (two i64s).
    // In real usage, lower_to_ir + extern_type_to_mlir_multi produces these.
    m.instrs.push(Instr::ExternFnDecl {
        name: "accept_timeval".to_string(),
        param_types: vec!["i64".to_string(), "i64".to_string()],
        ret_type: Some("i64".to_string()),
        is_varargs: false,
        vararg_hints: Vec::new(),
        callconv: CallConv::SysV,
    });
    let id = m.fresh();
    m.instrs.push(Instr::ConstI64(id, 0));
    m.instrs.push(Instr::Output(id));

    let text = lower_ir_to_mlir(&m)
        .expect("lowering must succeed")
        .text;

    assert!(
        text.contains("llvm.func @accept_timeval"),
        "must emit llvm.func @accept_timeval; got:\n{text}"
    );
    // The declaration must list (i64, i64) — two eightbytes for the struct.
    assert!(
        text.contains("llvm.func @accept_timeval(i64, i64)"),
        "16B all-int struct must expand to (i64, i64) in llvm.func; got:\n{text}"
    );
}

// ── test 14: end-to-end parse + type-check + IR lower with repr(C) struct ────

/// Parse a module that declares a `#[repr(C)]` struct and an `extern "C"` block
/// referencing it, then lower to IR. Verify:
/// - repr_c_structs contains the struct with correct field types.
/// - ExternFnDecl param_types for the struct parameter are SysV-classified.
#[test]
fn end_to_end_repr_c_struct_lower_to_ir() {
    let src = r#"
        [repr(C)]
        struct Point { x: i32, y: i32 }

        extern "C" {
            unsafe fn draw_point(p: Point) -> i32
        }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    // Type-check: must produce no errors.
    let diags = check_module_types_in_file(&module, src, None, &TypeEnv::new());
    assert!(
        diags.is_empty(),
        "type-check of repr(C) struct in extern signature must succeed; diags: {diags:?}"
    );

    // Lower to IR: repr_c_structs must be populated.
    use libmind::eval::lower_to_ir;
    let ir = lower_to_ir(&module);
    assert!(
        ir.repr_c_structs.contains_key("Point"),
        "repr_c_structs must contain 'Point'; got keys: {:?}",
        ir.repr_c_structs.keys().collect::<Vec<_>>()
    );
    let point_fields = &ir.repr_c_structs["Point"];
    assert_eq!(point_fields.len(), 2, "Point must have 2 fields");
    // Both i32 fields → SysV: total 8B all-int → single i64.
    let mlir_types = sysv_classify_struct(point_fields, &ir.repr_c_structs);
    assert_eq!(mlir_types, vec!["i64".to_string()],
        "Point (2xi32 = 8B all-int) must classify to single i64; got {mlir_types:?}");

    // Verify ExternFnDecl was emitted with i64 param type for the struct.
    let extern_decl = ir.instrs.iter().find_map(|instr| {
        #[cfg(feature = "std-surface")]
        if let Instr::ExternFnDecl { name, param_types, .. } = instr {
            if name == "draw_point" {
                return Some(param_types.clone());
            }
        }
        None
    });
    let param_types = extern_decl
        .expect("must have ExternFnDecl for draw_point");
    assert_eq!(param_types, vec!["i64".to_string()],
        "draw_point(Point) must lower to single i64 param; got {param_types:?}");
}

// ── test 15: SysV scalar field classification ─────────────────────────────────

/// `classify_scalar_field` must return correct (class, size) pairs for each
/// primitive type.
#[test]
fn sysv_scalar_field_classification() {
    let r = &empty_repr_c();
    assert_eq!(classify_scalar_field(&TypeAnn::ScalarI32, r), (Some(SysVClass::Integer), 4));
    assert_eq!(classify_scalar_field(&TypeAnn::ScalarI64, r), (Some(SysVClass::Integer), 8));
    assert_eq!(classify_scalar_field(&TypeAnn::ScalarF32, r), (Some(SysVClass::Float), 4));
    assert_eq!(classify_scalar_field(&TypeAnn::ScalarF64, r), (Some(SysVClass::Float), 8));
    assert_eq!(
        classify_scalar_field(&TypeAnn::RawPtr {
            mutable: true,
            pointee: Box::new(TypeAnn::ScalarI32),
        }, r),
        (Some(SysVClass::Integer), 8)
    );
    assert_eq!(
        classify_scalar_field(&TypeAnn::FnPtr { params: vec![], ret: None }, r),
        (Some(SysVClass::Integer), 8)
    );
}

// ── test 16: repr(C) struct declared AFTER extern "C" block (ordering hazard) ─

/// A `#[repr(C)]` struct declared *after* the `extern "C"` block that
/// references it must still classify correctly.  Before the two-pass fix,
/// the `repr_c_snapshot` was empty at ExternBlock processing time and the
/// struct fell through to the `i64` fallback — producing wrong classification
/// for any mixed int+float struct.
///
/// This test verifies the fix for the declaration-order hazard (audit F-05).
#[test]
fn repr_c_struct_after_extern_block_classifies_correctly() {
    // Mixed int+float struct should classify to MEMORY.
    // With the ordering bug: the struct is unknown at lowering time -> "i64" (WRONG).
    // With the two-pass fix: repr_c_structs is fully populated -> "!llvm.ptr" (CORRECT).
    let src = r#"
        extern "C" {
            unsafe fn use_mixed(m: Mixed) -> i32
        }

        [repr(C)]
        struct Mixed { x: i32, f: f32 }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    use libmind::eval::lower_to_ir;
    let ir = lower_to_ir(&module);

    // repr_c_structs must be populated regardless of declaration order.
    assert!(
        ir.repr_c_structs.contains_key("Mixed"),
        "repr_c_structs must contain 'Mixed' even when StructDef comes after ExternBlock; \
         keys: {:?}", ir.repr_c_structs.keys().collect::<Vec<_>>()
    );

    // The ExternFnDecl for use_mixed must have param_types = ["!llvm.ptr"]
    // because Mixed is a mixed int+float struct -> MEMORY class.
    let extern_decl = ir.instrs.iter().find_map(|instr| {
        #[cfg(feature = "std-surface")]
        if let Instr::ExternFnDecl { name, param_types, .. } = instr {
            if name == "use_mixed" {
                return Some(param_types.clone());
            }
        }
        None
    });
    let param_types = extern_decl
        .expect("must have ExternFnDecl for use_mixed");
    assert_eq!(
        param_types,
        vec!["!llvm.ptr".to_string()],
        "mixed int+float struct declared after ExternBlock must still classify \
         to MEMORY (!llvm.ptr); got {param_types:?}"
    );
}

// ── test 17: non-repr(C) struct in extern signature emits a diagnostic ─────

/// A `Named` type used in an `extern "C"` signature that is NOT annotated
/// with `#[repr(C)]` should produce at least one diagnostic.  Without the
/// fix, a non-repr(C) struct silently falls through to "i64" lowering, which
/// is unsound.
///
/// This test verifies audit finding F-06.
#[test]
fn non_repr_c_named_struct_in_extern_signature_emits_diagnostic() {
    // Declare a plain struct (no #[repr(C)]) and use it in an extern block.
    let src = r#"
        struct NotReprC { x: i32, y: i32 }

        extern "C" {
            unsafe fn bad(n: NotReprC) -> i32
        }
    "#;
    let module = parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let diags = check_module_types_in_file(&module, src, None, &TypeEnv::new());
    assert!(
        !diags.is_empty(),
        "using a non-repr(C) struct in an extern \"C\" signature must produce \
         at least one diagnostic (safety::extern_non_repr_c); got no diagnostics"
    );
}
