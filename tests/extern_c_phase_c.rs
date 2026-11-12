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
//! 6. `llvm.func` and `llvm.call` for Win64 carry the `win64cc` keyword, and a
//!    Win64 program builds through `mlir-opt`, `mlir-translate` and `clang`.
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
use libmind::type_checker::{TypeEnv, check_module_types_in_file};

#[cfg(all(unix, feature = "mlir-build"))]
mod common;

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
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
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
    assert_eq!(
        win64_result, sysv_result,
        "Win64 and SysV must agree on 8-byte struct"
    );
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
        binding: Default::default(),
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

        resolved_callee: None,
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m).expect("lowering must succeed").text;

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
/// - `llvm.func win64cc @name(...)` at module level.
/// - `llvm.call win64cc @name(...)` at call sites.
///
/// `mlir-opt` refuses `cconv = #llvm.cconv<win64cc>` before the symbol; the
/// run test `win64_program_builds_and_passes_args_in_win64_registers` builds it.
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
        binding: Default::default(),
    });

    let arg = m.fresh();
    m.instrs.push(Instr::ConstI64(arg, 42));
    let dst = m.fresh();
    m.instrs.push(Instr::Call {
        dst,
        name: "win_api_fn".to_string(),
        args: vec![arg],

        resolved_callee: None,
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m).expect("lowering must succeed").text;

    assert!(
        text.contains("llvm.func win64cc @win_api_fn"),
        "Win64 llvm.func must carry win64cc; got:\n{text}"
    );
    assert!(
        text.contains("llvm.call win64cc @win_api_fn"),
        "Win64 llvm.call must carry win64cc; got:\n{text}"
    );
}

// ── test 7: SysV llvm.func and llvm.call have no cconv attribute ──────────────

/// A SysV extern declaration must NOT carry a calling-convention keyword.
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
        binding: Default::default(),
    });

    let arg = m.fresh();
    m.instrs.push(Instr::ConstI64(arg, 0));
    let dst = m.fresh();
    m.instrs.push(Instr::Call {
        dst,
        name: "posix_fn".to_string(),
        args: vec![arg],

        resolved_callee: None,
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m).expect("lowering must succeed").text;

    assert!(
        !text.contains("cconv") && !text.contains("win64cc"),
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
        #[repr(C)]
        struct TwoI64 { a: i64, b: i64 }

        extern "C" callconv(.win64) {
            unsafe fn win_consume(s: TwoI64) -> i32
        }
    "#;
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    // The build path refuses the by-value struct (E2040); lowering below still
    // exercises the Win64 classifier.
    let diags = check_module_types_in_file(&module, src, None, &TypeEnv::default());
    let got: Vec<&str> = diags.iter().map(|d| d.code).collect();
    assert_eq!(
        got,
        ["E2040"],
        "by-value struct must be refused; diags: {diags:?}"
    );

    // Lower to IR.
    use libmind::eval::lower_to_ir;
    let ir = lower_to_ir(&module).expect("lowering");

    // Find the ExternFnDecl for win_consume.
    let extern_decl = ir.instrs.iter().find_map(|instr| {
        if let Instr::ExternFnDecl {
            name,
            param_types,
            callconv,
            ..
        } = instr
        {
            if name == "win_consume" {
                return Some((param_types.clone(), *callconv));
            }
        }
        None
    });

    let (param_types, callconv) = extern_decl.expect("must have ExternFnDecl for win_consume");

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

// ── test 11: `extern "C"` resolves from the compile TARGET, not the host ─────

fn lower_under(triple: &str, src: &str) -> IRModule {
    let target = libmind::target::Target::from_triple(triple).expect("triple");
    let _guard = libmind::target::CompileTargetGuard::install(target);
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    libmind::eval::lower_to_ir(&module).expect("lowering")
}

fn decl_of(ir: &IRModule, want: &str) -> (Vec<String>, Option<String>, CallConv) {
    ir.instrs
        .iter()
        .find_map(|instr| match instr {
            Instr::ExternFnDecl {
                name,
                param_types,
                ret_type,
                callconv,
                ..
            } if name == want => Some((param_types.clone(), ret_type.clone(), *callconv)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no ExternFnDecl for {want}"))
}

const C_SRC: &str = "extern \"C\" { fn f(x: u32, b: bool) -> u32; }";

/// One host, three targets: the callconv byte is a function of the target.
/// Resolving `C` from the mindc build host again turns the Windows row red on
/// a Linux host (and the Linux row red on a Windows host).
#[test]
fn c_callconv_follows_the_compile_target_not_the_host() {
    let names = vec!["u32".to_string(), "bool".to_string()];
    for (triple, expected) in [
        ("x86_64-unknown-linux-gnu", CallConv::SysV),
        ("x86_64-apple-darwin", CallConv::SysV),
        ("x86_64-w64-windows-gnu", CallConv::Win64),
    ] {
        let (params, ret, cc) = decl_of(&lower_under(triple, C_SRC), "f");
        assert_eq!(cc, expected, "{triple}");
        assert_eq!(
            params, names,
            "{triple}: declared names stay in the evidence"
        );
        assert_eq!(ret.as_deref(), Some("u32"), "{triple}");
    }
    let mic3 = |triple| libmind::ir::compact::v3::emit_mic3(&lower_under(triple, C_SRC));
    let linux = mic3("x86_64-unknown-linux-gnu");
    assert_eq!(
        linux,
        mic3("x86_64-unknown-linux-gnu"),
        "same target, same bytes"
    );
    assert_eq!(
        linux,
        mic3("x86_64-apple-darwin"),
        "same ABI row, same bytes"
    );
    let windows = mic3("x86_64-w64-windows-gnu");
    assert_ne!(
        linux, windows,
        "the Win64 row records its own callconv byte"
    );
    let diffs = linux.iter().zip(&windows).filter(|(a, b)| a != b).count();
    assert_eq!(
        (linux.len(), diffs),
        (windows.len(), 1),
        "only the callconv byte differs"
    );
}

fn codes_under(triple: &str, src: &str) -> Vec<String> {
    let target = libmind::target::Target::from_triple(triple).expect("triple");
    let _guard = libmind::target::CompileTargetGuard::install(target);
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    check_module_types_in_file(&module, src, None, &TypeEnv::default())
        .into_iter()
        .map(|d| d.code.to_string())
        .collect()
}

/// A declared convention with no row on the target is refused (E2039), never
/// lowered with another row. Each refusal has a same-shape accepted control.
#[test]
fn unrealized_callconv_is_refused_with_e2039() {
    let linux = "x86_64-unknown-linux-gnu";
    let windows = "x86_64-w64-windows-gnu";
    let arm = "aarch64-unknown-linux-gnu";
    let block = |cc: &str, sig: &str| format!("extern \"C\" {cc} {{ fn g({sig}) -> i64; }}");
    for (triple, cc, sig, refused) in [
        (linux, "callconv(.aapcs)", "x: i64", true),
        (linux, "callconv(.sysv)", "x: i64", false),
        (linux, "", "x: i64", false),
        (linux, "callconv(.win64)", "x: i64", false),
        (linux, "callconv(.win64)", "x: i64, ...", true),
        (linux, "", "x: i64, ...", false),
        (windows, "callconv(.sysv)", "x: i64", true),
        (windows, "", "x: i64", false),
        (arm, "callconv(.aapcs)", "x: i64", false),
        (arm, "", "x: i64", false),
        (arm, "callconv(.sysv)", "x: i64", true),
        (arm, "callconv(.win64)", "x: i64", true),
    ] {
        let codes = codes_under(triple, &block(cc, sig));
        assert_eq!(
            codes.iter().any(|c| c == "E2039"),
            refused,
            "{triple} {cc} ({sig}): {codes:?}"
        );
    }
}

/// Win64 keeps the declared name of each scalar in the evidence (before: u32
/// was recorded as i32). MLIR stays signless and carries the extension.
#[test]
fn win64_tokens_are_declared_names_and_mlir_stays_signless() {
    let src = "extern \"C\" callconv(.win64) {\n    fn w(a: i8, b: u8, c: i16, d: u16, e: i32, f: u32, g: bool, h: u64, k: f32) -> u32;\n    fn r() -> u32;\n}\nfn main() -> i64 { return r() as i64; }";
    let ir = lower_under("x86_64-unknown-linux-gnu", src);
    let (params, ret, cc) = decl_of(&ir, "w");
    assert_eq!(cc, CallConv::Win64);
    assert_eq!(
        params,
        ["i8", "u8", "i16", "u16", "i32", "u32", "bool", "i64", "f32"]
            .map(String::from)
            .to_vec()
    );
    assert_eq!(ret.as_deref(), Some("u32"));
    let text = lower_ir_to_mlir(&ir).expect("lowering must succeed").text;
    let decl = "llvm.func win64cc @w(i8 {llvm.signext}, i8 {llvm.zeroext}, \
                i16 {llvm.signext}, i16 {llvm.zeroext}, i32 {llvm.signext}, i32 {llvm.zeroext}, \
                i1 {llvm.zeroext}, i64, f32) -> i32";
    assert!(text.contains(decl), "win64 declaration; got:\n{text}");
    assert!(
        text.contains("arith.extui") && !text.contains("-> u32"),
        "the u32 return is read as i32 and zero-extended; got:\n{text}"
    );
}

// ── run tests: the emitted MLIR goes through mlir-opt, mlir-translate, clang ──

/// Build `src` to a `.so` with `mindc --emit-shared` (mlir-opt, mlir-translate
/// and clang all run). `None` only for a tolerated capability gap; any other
/// build failure panics with the compiler's stderr.
#[cfg(all(unix, feature = "mlir-build"))]
fn build_so(target: &str, src: &str) -> Option<std::path::PathBuf> {
    let dir = common::scratch_dir(target);
    let (mind, so) = (dir.join("t.mind"), dir.join("t.so"));
    std::fs::write(&mind, src).expect("write src");
    let out = std::process::Command::new(common::mindc_bin())
        .args([
            mind.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    common::gate::compiled(target, &out).then_some(so)
}

/// A `callconv(.win64)` call builds end to end (`mlir-opt` refused the old
/// `cconv = #llvm.cconv<win64cc>` form, so no Win64 program built), and the
/// convention reaches machine code: the first two integer arguments are set up
/// in `rcx`/`rdx`, not the System V `rdi`/`rsi`.
#[cfg(all(unix, target_arch = "x86_64", feature = "mlir-build"))]
#[test]
fn win64_program_builds_and_passes_args_in_win64_registers() {
    let src = "extern \"C\" callconv(.win64) {\n    fn w_add(a: i64, b: i64) -> i64;\n}\n\
               pub fn t_w() -> i64 {\n    return w_add(40, 2)\n}\n";
    let Some(so) = build_so("xc-win64-run", src) else {
        return;
    };
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn", so.to_str().unwrap()])
        .output()
        .expect("run objdump");
    let text = String::from_utf8_lossy(&out.stdout);
    let body: String = text
        .split("\n\n")
        .filter(|f| f.contains("<t_w>:"))
        .collect();
    assert!(
        body.contains("$0x28,%ecx") && body.contains("$0x2,%edx") && body.contains("w_add"),
        "t_w must pass 40 in rcx and 2 in rdx to w_add; got:\n{body}"
    );
}

/// An extern `f32`/`f64` return is used at its real type (before: every extern
/// call result was typed `i64`, so `return %r : i64` after `-> f32` and
/// `mlir-opt` refused the module). libm's `ldexpf`/`ldexp` are the callees.
#[cfg(all(unix, feature = "mlir-build"))]
#[test]
fn extern_float_returns_keep_their_float_type() {
    let src = "extern \"C\" {\n    fn ldexpf(x: f32, e: i32) -> f32;\n    fn ldexp(x: f64, e: i32) -> f64;\n}\n\
               pub fn t_f32() -> f32 {\n    return ldexpf(1.5 as f32, 2)\n}\n\
               pub fn t_f64() -> f64 {\n    return ldexp(2.25, 3) + 1.0\n}\n\
               pub fn t_f64_int() -> i64 {\n    return ldexp(2.25, 3) as i64\n}\n";
    let Some(so) = build_so("xc-float-ret", src) else {
        return;
    };
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.t_f32.restype = ctypes.c_float\n\
         lib.t_f64.restype = ctypes.c_double\n\
         lib.t_f64_int.restype = ctypes.c_int64\n\
         print(lib.t_f32(), lib.t_f64(), lib.t_f64_int())\n",
        so.to_string_lossy()
    );
    let out = std::process::Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "6.0 19.0 18",
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The declared width governs an extern argument: an `f64` literal or variable
/// passed to an extern `f32` parameter is converted (`arith.truncf`) at the call.
/// Before: `mlir-opt` refused the call (`f32` vs `f64`). libm `ldexpf` scales
/// by a power of two, so each result is exact: `0.1` reads back as the `f32`
/// nearest it (`0.10000000149011612`), not `0.1`.
#[cfg(all(unix, feature = "mlir-build"))]
#[test]
fn extern_f32_argument_takes_the_declared_width() {
    let src = "extern \"C\" {\n    fn ldexpf(x: f32, e: i32) -> f32;\n}\n\
               pub fn e_lit() -> f64 {\n    return ldexpf(0.1, 0) as f64\n}\n\
               pub fn e_param(x: f64) -> f64 {\n    return ldexpf(x * 3.0, 2) as f64\n}\n";
    let Some(so) = build_so("xc-f32-arg", src) else {
        return;
    };
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f, d = ctypes.c_float, ctypes.c_double\n\
         lib.e_lit.restype = d\n\
         lib.e_param.restype = d; lib.e_param.argtypes = [d]\n\
         print(lib.e_lit() == 0.10000000149011612, lib.e_param(0.1) == f(0.1 * 3.0).value * 4)\n",
        so.to_string_lossy()
    );
    let out = std::process::Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "True True",
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
