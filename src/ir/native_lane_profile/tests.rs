// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
use super::*;

fn lower(source: &str) -> (ast::Module, IRModule) {
    let ast = crate::parser::parse(source).expect("fixture parses");
    let ir = crate::eval::lower_to_ir(&ast).expect("fixture lowers");
    (ast, ir)
}
fn admits(source: &str) -> Check<()> {
    admits_with(source, true)
}
fn admits_with(source: &str, extern_fixed: bool) -> Check<()> {
    let (ast, ir) = lower(source);
    profile_lane_admits(&ast, &ir, extern_fixed)
}
fn body(ir: &mut IRModule) -> &mut Vec<Instr> {
    ir.instrs
        .iter_mut()
        .find_map(|i| match i {
            Instr::FnDef { body, .. } => Some(body),
            _ => None,
        })
        .unwrap()
}

#[test]
fn primitive_positive_controls() {
    for source in [
        "fn main() -> i64 { return 7; }",
        "fn f(x: i64, y: i64) -> i64 { return ((x >> 1) / y) % 7; }",
        "fn f(x: f64, y: f64) -> f64 { return x * y + 1.0; }",
        "fn f(x: f64, y: f64) -> i64 { if x < y { return 1; } return 0; }",
        "fn f(x: i64) -> i64 { return x + 1; } fn main() -> i64 { return f(2); }",
        "const A: i64 = 7; fn main() -> i64 { return A; }",
        "fn main() -> i64 { let x: f64 = __mind_bits_to_f64(0); return __mind_f64_to_bits(x); }",
        "fn main() -> i64 { let p = __mind_alloc(16); __mind_store_i64(p, 3); let x = __mind_load_i64(p); __mind_free(p); return x; }",
        "fn main() -> i64 { return __mind_read(0, 0, 0, -1); }",
        "fn main() -> i64 { return __mind_sys_getuid(); }",
        "fn main(argc: i64, argv: i64) -> i64 { return argc + (argv != 0); }",
        "fn main() -> i64 { let x = 0; while x < 3 { x = x + 1; } return x; }",
        "fn main() -> i64 { let x = 0; while x < 3 { x = x + 1; if x == 2 { continue; } if x == 3 { break; } } return x; }",
        "fn main() -> i64 { let x = 0; for i in 0..3 { x = x + i; } return x; }",
        "fn main() -> i64 { let x = 1; if x == 1 { x = 2; } else { x = 3; } return x; }",
        "fn main() -> i64 { let x = 1; if x != 0 && x < 3 { return 2; } return 0; }",
        "fn main() -> i64 { let x = 1; while x < 3 || x == 4 { if x == 2 { return x; } x = x + 1; } return x; }",
        "fn f(x: f64) -> f64 { return (0.0 - x) / 2.0; }",
        "fn f(x: i64) -> i64 { return !x; }",
        "fn f() -> f64 { let x = 1.0; let i = 0; while i < 3 { x = x * 2.0; i = i + 1; } return x; } fn main() -> i64 { return f() as i64; }",
        "fn main() -> i64 { let x = 1; if x == 1 { let x = 2.0; __mind_f64_to_bits(x); } return x; }",
        "fn f(c: i64) -> i64 { let x = if c { return 7; } else { 2 }; return x; }",
        "fn f(c: i64) -> f64 { let x = if c { return 7.0; } else { 2.0 }; return x; }",
        "fn f(c: i64) -> f64 { let x = 1.0; if c { x = 2.0; } else { x = 3.0; } return x; }",
        "const x: i64 = 1; fn main() -> i64 { if x == 1 { let x = 2.0; __mind_f64_to_bits(x); } return x; }",
    ] {
        assert_eq!(admits(source), Ok(()), "{source}");
    }
}

#[test]
fn source_types_cannot_hide_behind_width_erasure() {
    for source in [
        "fn f(x: u64) -> i64 { return 0; }",
        "fn f(x: i32) -> i64 { return x / 2; }",
        "fn f(x: u32) -> i64 { return x >> 1; }",
        "fn f(x: f32) -> i64 { return x < 2.0; }",
        "fn main() -> i64 { let x: i32 = 2; return x; }",
        "fn main() -> i64 { let x: f32 = 2.0; return x < 3.0; }",
        "fn main() -> i64 { let x = 7 as u64; return x / 2; }",
        "fn main() -> i64 { let x = 7 as i32; return x + 1; }",
        "fn f(x: *const u8) -> i64 { return 0; }",
        "fn f(x: [i64; 4]) -> i64 { return 0; }",
        "fn f(x: Tensor<f64, [4]>) -> i64 { return 0; }",
    ] {
        let ast = crate::parser::parse(source).expect(source);
        assert!(profile_lane_admits_source(&ast, true).is_err(), "{source}");
    }
}

#[test]
fn calls_have_exact_types_arity_and_closed_names() {
    for source in [
        "fn main() -> i64 { return __mind_alloc(1.0); }",
        "fn main() -> i64 { return __mind_alloc(1, 2); }",
        "fn main() -> i64 { return __mind_allocated(1); }",
        "fn main() -> i64 { return __mind_f64_to_bits(1); }",
        "fn main() -> i64 { return __mind_bits_to_f64(0); }",
        "fn main() -> i64 { return missing(1); }",
        "fn dead() -> i64 { return missing(1); } fn main() -> i64 { return 0; }",
        "fn __mind_alloc(x: i64) -> i64 { return x; }",
        "fn f(x: f64) -> i64 { return 0; } fn main() -> i64 { return f(1); }",
        "fn main() -> i64 { return __mind_sys_openat(0); }",
        // `close` is a closed-table row now; only the reserved-name override
        // still refuses at source.
        "extern \"C\" { fn __mind_alloc(x: i64) -> i64; } fn main() -> i64 { return 0; }",
    ] {
        let ast = crate::parser::parse(source).expect(source);
        assert!(profile_lane_admits_source(&ast, true).is_err(), "{source}");
    }
}

#[test]
fn float_bitwise_and_mixed_arithmetic_refuse() {
    for source in [
        "fn main() -> f64 { return 1.0 % 2.0; }",
        "fn main() -> f64 { return 1.0 + 2; }",
        "fn main() -> i64 { return 1.0 & 2.0; }",
        "fn main() -> i64 { if 1.0 { return 1; } return 0; }",
        "fn f(x: f64) -> f64 { return -x; }",
    ] {
        let ast = crate::parser::parse(source).expect(source);
        assert!(profile_lane_admits_source(&ast, true).is_err(), "{source}");
    }
}

#[test]
fn exact_intrinsic_contract_inventory() {
    for (name, n) in [
        ("__mind_alloc", 1),
        ("__mind_assert_fail", 1),
        ("__mind_read", 4),
        ("__mind_write", 4),
        ("__mind_free", 1),
        ("__mind_store_i32", 2),
        ("__mind_f64_to_bits", 1),
        ("__mind_open", 1),
        ("__mind_sys_fork", 0),
        ("__mind_sys_waitid", 5),
    ] {
        assert_eq!(intrinsic_arity(name), Some(n));
    }
    for name in [
        "alloc",
        "__mind_unknown",
        "__mind_sys_socket",
        "close",
        "__errno_location",
        "__mind_alloc_extra",
    ] {
        assert_eq!(intrinsic_arity(name), None);
    }
}

/// `__mind_open` is admitted on the native lane (census R9). The census shapes
/// (weights:243, naes_t1:732, selftest:65, main:230,367) all call
/// `__mind_open(path)` and use the returned fd. A mutant that removes the
/// `__mind_open` row from the admission table turns this test red.
#[test]
fn mind_open_is_admitted_on_the_native_lane() {
    // The census shapes: simple __mind_open calls with a path argument.
    for source in [
        // weights:243 — open a path, return fd + 100
        "fn main() -> i64 { let p = __mind_alloc(8); __mind_store_i8(p, 47); __mind_store_i8(p + 1, 0); let fd = __mind_open(p); return fd + 100; }",
        // naes_t1:732 — open a path, check fd >= 0
        "fn main() -> i64 { let p = __mind_alloc(8); __mind_store_i8(p, 47); __mind_store_i8(p + 1, 0); let fd = __mind_open(p); if fd < 0 { return 1; } return 0; }",
        // selftest:65 — open a path, read one byte
        "fn main() -> i64 { let p = __mind_alloc(8); __mind_store_i8(p, 47); __mind_store_i8(p + 1, 0); let fd = __mind_open(p); if fd < 0 { return 1; } let b = __mind_alloc(1); let n = __mind_read(fd, b, 1, 0 - 1); return n; }",
        // main:230 — open a path, use fd in a condition
        "fn main() -> i64 { let p = __mind_alloc(8); __mind_store_i8(p, 47); __mind_store_i8(p + 1, 0); let fd = __mind_open(p); if fd >= 0 { return 0; } return 1; }",
        // main:367 — open a path, return fd directly
        "fn main() -> i64 { let p = __mind_alloc(8); __mind_store_i8(p, 47); __mind_store_i8(p + 1, 0); return __mind_open(p); }",
    ] {
        assert!(admits(source).is_ok(), "should admit: {source}");
    }
    // Arity check: __mind_open takes exactly one argument.
    assert_eq!(intrinsic_arity("__mind_open"), Some(1));
    // Wrong arity refuses.
    let bad = "fn main() -> i64 { return __mind_open(); }";
    let ast = crate::parser::parse(bad).expect("parses");
    let refused = profile_lane_admits_source(&ast, true).map_err(|r| r.construct);
    assert_eq!(refused, Err("lane.call.arity"), "{bad}");
    let bad2 = "fn main() -> i64 { return __mind_open(1, 2); }";
    let ast2 = crate::parser::parse(bad2).expect("parses");
    let refused2 = profile_lane_admits_source(&ast2, true).map_err(|r| r.construct);
    assert_eq!(refused2, Err("lane.call.arity"), "{bad2}");
}

/// The kind 87 / 88 group-sum rows pass the native-lane front door with five
/// and six i64 arguments and an i64 value result. Without their
/// `intrinsic_signature` rows the lane refuses both calls.
#[test]
fn group_sum_rows_are_admitted_with_their_exact_arity() {
    let refusal = |source: &str| {
        let ast = crate::parser::parse(source).expect(source);
        profile_lane_admits_source(&ast, true).map_err(|r| r.construct)
    };
    let (bank, acc) = ("__mind_group_bank_i64", "__mind_group_acc32_i64");
    let ok = format!(
        "fn main() -> i64 {{ let p = __mind_alloc(64); let a = {bank}(p, 0, 1, 2, 3); \
         let b = {acc}(p, 8, 0, 1, 2, 3); return a + b; }}"
    );
    assert_eq!(refusal(&ok), Ok(()), "{ok}");
    assert!(admits(&ok).is_ok(), "{ok}");
    assert_eq!(intrinsic_arity(bank), Some(5));
    assert_eq!(intrinsic_arity(acc), Some(6));
    for bad in [
        format!("fn main() -> i64 {{ return {bank}(0, 1, 2, 3); }}"),
        format!("fn main() -> i64 {{ return {acc}(0, 1, 2, 3, 4); }}"),
    ] {
        assert_eq!(refusal(&bad), Err("lane.call.arity"), "{bad}");
    }
}

#[test]
fn malformed_ir_operands_and_calls_refuse() {
    let (ast, good) = lower("fn main() -> i64 { return 3 + 4; }");
    let mut unknown = good.clone();
    body(&mut unknown).insert(
        0,
        Instr::legacy_call(ValueId(99999), "__mind_forged", vec![]),
    );
    assert!(profile_lane_admits(&ast, &unknown, true).is_err());
    let mut mixed = good.clone();
    let b = body(&mut mixed);
    let item = b
        .iter_mut()
        .find(|i| matches!(i, Instr::ConstI64(..)))
        .unwrap();
    if let Instr::ConstI64(id, _) = item {
        *item = Instr::ConstF64(*id, 3.0);
    }
    assert!(profile_lane_admits(&ast, &mixed, true).is_err());
    let mut undefined = good.clone();
    let item = body(&mut undefined)
        .iter_mut()
        .find(|i| matches!(i, Instr::BinOp { .. }))
        .unwrap();
    if let Instr::BinOp { lhs, .. } = item {
        *lhs = ValueId(99999);
    }
    assert!(profile_lane_admits(&ast, &undefined, true).is_err());
}

#[test]
fn parameter_index_and_return_type_are_bound() {
    let (ast, good) = lower("fn f(x: f64) -> f64 { return x; }");
    let mut bad = good.clone();
    let p = body(&mut bad)
        .iter_mut()
        .find(|i| matches!(i, Instr::Param { .. }))
        .unwrap();
    if let Instr::Param { index, .. } = p {
        *index = 9;
    }
    assert!(profile_lane_admits(&ast, &bad, true).is_err());
    let mut bad = good.clone();
    body(&mut bad).insert(0, Instr::ConstI64(ValueId(99999), 1));
    for i in body(&mut bad) {
        if let Instr::Return { value } = i {
            *value = Some(ValueId(99999));
        }
    }
    assert!(profile_lane_admits(&ast, &bad, true).is_err());
}

#[test]
fn misplaced_loop_exit_is_refused() {
    let (ast, mut ir) = lower("fn main() -> i64 { return 1; }");
    body(&mut ir).insert(0, Instr::Break { live: vec![] });
    assert!(profile_lane_admits(&ast, &ir, true).is_err());
}

#[test]
fn typed_ir_call_arguments_are_rechecked() {
    let (ast, good) = lower("fn main() -> i64 { return __mind_alloc(8); }");
    let mut wrong_arity = good.clone();
    for i in body(&mut wrong_arity) {
        if let Instr::Call { args, .. } = i {
            args.clear();
        }
    }
    assert!(profile_lane_admits(&ast, &wrong_arity, true).is_err());
    let mut wrong_type = good.clone();
    for i in body(&mut wrong_type) {
        if let Instr::ConstI64(id, _) = i {
            *i = Instr::ConstF64(*id, 8.0);
        }
    }
    assert!(profile_lane_admits(&ast, &wrong_type, true).is_err());
}

#[test]
fn loop_carried_types_and_branch_scope_are_checked() {
    let (ast, mut ir) =
        lower("fn main() -> i64 { let x = 0; while x < 3 { x = x + 1; } return x; }");
    for i in body(&mut ir) {
        if let Instr::While { exit_ids, .. } = i {
            exit_ids.clear();
        }
    }
    assert!(profile_lane_admits(&ast, &ir, true).is_err());
    let (ast, mut ir) = lower("fn main() -> i64 { if 1 { let y = 7; } return 0; }");
    let mut hidden = None;
    for i in body(&mut ir) {
        if let Instr::If { then_instrs, .. } = i {
            hidden = then_instrs.iter().find_map(|i| match i {
                Instr::ConstI64(id, 7) => Some(*id),
                _ => None,
            });
        }
    }
    let hidden = hidden.expect("branch-local fixture value");
    for i in body(&mut ir) {
        if let Instr::Return { value } = i {
            *value = Some(hidden);
        }
    }
    assert!(profile_lane_admits(&ast, &ir, true).is_err());
}

fn source_admits(source: &str) -> Check<()> {
    source_admits_with(source, true)
}
fn source_admits_with(source: &str, extern_fixed: bool) -> Check<()> {
    profile_lane_admits_source(&crate::parser::parse(source).expect(source), extern_fixed)
}

/// The id of the first `Instr::Call` to `name`, anywhere in the first body.
fn call_dst(ir: &mut IRModule, name: &str) -> ValueId {
    body(ir)
        .iter()
        .find_map(|i| match i {
            Instr::Call { dst, name: n, .. } if n == name => Some(*dst),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no call to {name}"))
}

/// A store result is a statement, or the defined i64 value 0 where a value is
/// read: MLIR materialises `arith.constant 0` and the marked compiler writes 0
/// into the result slot (the downstream consumer `w_set` shape returns it). It never
/// becomes an f64, and a constant initialiser still refuses it.
#[test]
fn store_results_are_statements_or_the_defined_zero() {
    for name in ["__mind_store_i64", "__mind_store_i8", "__mind_store_i32"] {
        for source in [
            format!("fn main() -> i64 {{ return {name}(0, 1); }}"),
            format!("fn main() -> i64 {{ let x = {name}(0, 1); return x; }}"),
            format!("fn main() -> i64 {{ {name}(0, 1) }}"),
            format!(
                "fn w(p: i64, v: i64) -> i64 {{ {name}(p + 8, v) }} \
                 fn main() -> i64 {{ return w(0, 1) + 2; }}"
            ),
            format!("fn main() -> i64 {{ if {name}(0, 1) == 0 {{ return 1; }} return 2; }}"),
            format!(
                "fn f(k: i64) -> i64 {{ if k == 0 {{ {name}(0, 4); }} else {{ {name}(0, 6); }} \
                 return 0; }}"
            ),
            format!(
                "fn f(c: i64) -> i64 {{ let y = if c {{ {name}(0, 1) }} else {{ 5 }}; return y; }}"
            ),
            format!("fn main() -> i64 {{ {name}(0, 1); return 0; }}"),
        ] {
            assert_eq!(admits(&source), Ok(()), "{source}");
        }
        // A statement store ending a unit function stays admitted at source.
        assert_eq!(
            source_admits(&format!("fn f() {{ {name}(0, 1); }}")),
            Ok(())
        );
        for source in [
            format!("fn main() -> f64 {{ return {name}(0, 1); }}"),
            format!("fn main() -> i64 {{ let x: f64 = {name}(0, 1); return 0; }}"),
            format!(
                "fn g(x: f64) -> i64 {{ return 0; }} fn main() -> i64 {{ return g({name}(0, 1)); }}"
            ),
            format!("fn main() -> f64 {{ return {name}(0, 1) + 1.0; }}"),
            format!("const C: i64 = {name}(0, 1); fn main() -> i64 {{ return C; }}"),
            format!(
                "fn f(c: i64) -> f64 {{ let y = if c {{ {name}(0, 1) }} else {{ 5.0 }}; return y; }}"
            ),
        ] {
            assert!(source_admits(&source).is_err(), "{source}");
        }
        // IR: a store result cannot be forged into a float operation.
        let (ast, mut ir) = lower(&format!(
            "fn main() -> i64 {{ let x = {name}(0, 1); return x; }}"
        ));
        assert_eq!(profile_lane_admits(&ast, &ir, true), Ok(()));
        let dst = call_dst(&mut ir, name);
        let b = body(&mut ir);
        let at = b
            .iter()
            .position(|i| matches!(i, Instr::Return { .. }))
            .unwrap();
        b.insert(at, Instr::ConstF64(ValueId(99998), 1.0));
        b.insert(
            at + 1,
            Instr::BinOp {
                dst: ValueId(99999),
                op: BinOp::Add,
                lhs: dst,
                rhs: ValueId(99998),
            },
        );
        let err = profile_lane_admits(&ast, &ir, true).unwrap_err();
        assert_eq!(err.function.as_deref(), Some("main"));
    }
}

/// Only full-width i64 <-> f64 `as` conversions are admitted. Both backends
/// convert i64 -> f64 round-to-nearest-even and f64 -> i64 by saturating
/// truncation; narrow, unsigned, f32 and float-to-float casts still refuse, and
/// so does an explicit source call to a conversion intrinsic.
#[test]
fn full_width_i64_f64_casts_are_admitted_both_ways() {
    for source in [
        "fn f(x: i64) -> f64 { return x as f64; }",
        "fn f(x: f64) -> i64 { return x as i64; }",
        "fn f(x: i64) -> i64 { return (x as f64) as i64; }",
        "fn f(a: f64, b: f64) -> i64 { return (a * b) as i64; }",
        "fn f(n: i64, d: i64) -> f64 { return (n as f64) / (d as f64); }",
        "fn f(x: i64) -> i64 { return x as i64; }",
        "fn f(c: i64) -> f64 { return (c < 3) as f64; }",
        "fn f(p: i64) -> i64 { return __mind_load_i64(p) as f64 as i64; }",
    ] {
        assert_eq!(admits(source), Ok(()), "{source}");
    }
    for source in [
        "fn f(x: f64) -> f64 { return x as f64; }",
        "fn f(x: i64) -> f64 { return x as f32; }",
        "fn f(x: f64) -> i64 { return x as i32; }",
        "fn f(x: f64) -> i64 { return x as u64; }",
        "fn f(x: i64) -> i64 { return x as u8; }",
        "fn f(x: i64) -> f64 { return __mind_conv_f64(x); }",
        "fn f(x: f64) -> i64 { return __mind_conv_i64(x); }",
        "fn f(x: i64) -> i64 { return x as f64; }",
        "fn f(x: f64) -> f64 { return x as i64; }",
    ] {
        assert!(source_admits(source).is_err(), "{source}");
    }
    // IR: the conversion is typed by its SOURCE value, so a float fed to the
    // int->float conversion is not a known call.
    let (ast, mut ir) = lower("fn f(x: i64) -> f64 { return x as f64; }");
    assert_eq!(profile_lane_admits(&ast, &ir, true), Ok(()));
    let b = body(&mut ir);
    b.insert(0, Instr::ConstF64(ValueId(99999), 2.0));
    for i in b.iter_mut() {
        if let Instr::Call { name, args, .. } = i {
            if name == "__mind_conv_f64" {
                args[0] = ValueId(99999);
            }
        }
    }
    assert!(profile_lane_admits(&ast, &ir, true).is_err());
    assert_eq!(intrinsic_arity("__mind_conv_f64"), Some(1));
    assert_eq!(intrinsic_arity("__mind_conv_f32"), None);
}

/// `x as u32` lowers to `__mind_conv_u32(x)`. Both legs type it, so a `u32`
/// extern parameter has a working path; nothing else may hold the value.
#[test]
fn u32_casts_are_typed_in_both_legs() {
    let close_range =
        "extern \"C\" {\n    fn close_range(first: u32, last: u32, flags: i32) -> i32;\n}\n";
    for source in [
        "fn main() -> i64 { let x: i64 = (5 as u32) as i64; return x; }".to_string(),
        "fn main() -> i64 { let x: i64 = 0 - 1; return (x as u32) as i64; }".to_string(),
        format!(
            "{close_range}fn main() -> i64 {{ return close_range(1000 as u32, 2000 as u32, 0 as i32) as i64; }}"
        ),
    ] {
        assert_eq!(admits(&source), Ok(()), "{source}");
    }
    for (source, kind) in [
        (
            format!(
                "{close_range}fn main() -> i64 {{ return close_range(1000, 2000 as u32, 0 as i32) as i64; }}"
            ),
            "lane.type.mismatch",
        ),
        (
            "fn main() -> i64 { return 5 as u32; }".to_string(),
            "lane.type.mismatch",
        ),
        (
            "fn main() -> i64 { return ((5 as u32) + (6 as u32)) as i64; }".to_string(),
            "lane.binop.non_scalar",
        ),
        (
            "fn f(x: f64) -> i64 { return (x as u32) as i64; }".to_string(),
            "lane.source.cast",
        ),
        (
            "fn main() -> i64 { return __mind_conv_u32(5); }".to_string(),
            "lane.call.unknown",
        ),
    ] {
        assert_eq!(
            source_admits(&source).unwrap_err().construct,
            kind,
            "{source}"
        );
    }
    // IR: a float fed to the u32 conversion is not a known call.
    let (ast, mut ir) = lower("fn main() -> i64 { return (5 as u32) as i64; }");
    assert_eq!(profile_lane_admits(&ast, &ir, true), Ok(()));
    let b = body(&mut ir);
    b.insert(0, Instr::ConstF64(ValueId(99999), 2.0));
    let mut fed = 0;
    for i in b.iter_mut() {
        if let Instr::Call { name, args, .. } = i {
            if name == "__mind_conv_u32" {
                args[0] = ValueId(99999);
                fed += 1;
            }
        }
    }
    assert_eq!(fed, 1, "the cast lowers to one __mind_conv_u32 call");
    let err = profile_lane_admits(&ast, &ir, true).unwrap_err();
    assert_eq!(err.construct, "lane.call.unknown");
    assert_eq!(intrinsic_arity("__mind_conv_u32"), Some(1));
    assert_eq!(intrinsic_arity("__mind_conv_u16"), None);
}

/// A plain project import is a declaration the bridge already resolved; an
/// `as` rename is refused, and so is any other non-function top-level item.
#[test]
fn plain_project_imports_are_declarations() {
    assert_eq!(
        source_admits("import helper;\nfn main() -> i64 { return 0; }"),
        Ok(())
    );
    assert!(source_admits("import helper as h;\nfn main() -> i64 { return 0; }").is_err());
    assert!(source_admits("struct P { x: i64 }\nfn main() -> i64 { return 0; }").is_err());
}

/// A refusal names the function and the byte offset of the refused node, so a
/// large program's diagnostic can point at the line.
#[test]
fn rejections_name_the_function_and_offset() {
    let src = "fn ok() -> i64 { return 1; }\nfn bad(x: i64) -> i64 {\n    let y: f64 = x;\n    return 0;\n}\n";
    let err = source_admits(src).unwrap_err();
    assert_eq!(err.construct, "lane.type.mismatch");
    assert_eq!(err.function.as_deref(), Some("bad"));
    assert_eq!(err.at, Some(src.find("let y").unwrap()));
    let top = "fn main() -> i64 { return 0; }\nstruct P { x: i64 }\n";
    let err = source_admits(top).unwrap_err();
    assert_eq!(err.construct, "lane.source.top_level");
    assert_eq!(err.function, None);
    assert_eq!(err.at, Some(top.find("struct").unwrap()));
}

/// A branch that surely leaves does not reach the merge. Lowering forwards the
/// other branch's value in its place, so only the reaching side is typed; the
/// The downstream consumer UTF-8 decoder (`dcu_decode`) has exactly this shape.
#[test]
fn a_diverging_branch_is_not_typed_at_the_merge() {
    for source in [
        "fn f(first: i64) -> i64 { let mut w = 0; if first < 128 { w = 1; } else { \
         if first >= 194 { w = 2; } else { return 0 - 1; } } return w; }",
        "fn f(c: i64) -> f64 { let mut x = 1.0; if c == 1 { return 0.0; } else { x = 2.0; } \
         return x; }",
        "fn f(n: i64) -> i64 { let mut i = 0; let mut s = 0; while i < n { if i == 7 { \
         s = s + 1; } else { if i > 9 { break; } else { i = i + 1; continue; } } i = i + 1; } \
         return s; }",
    ] {
        assert_eq!(admits(source), Ok(()), "{source}");
    }
    // A reaching branch is still typed: an undefined forwarded value refuses.
    let (ast, mut ir) = lower(
        "fn f(c: i64) -> i64 { let mut x = 0; if c == 1 { x = 1; } else { x = 2; } return x; }",
    );
    assert_eq!(profile_lane_admits(&ast, &ir, true), Ok(()));
    for i in body(&mut ir) {
        if let Instr::If { merges, .. } = i {
            for (_, _, right) in merges.iter_mut() {
                *right = ValueId(99999);
            }
        }
    }
    assert!(profile_lane_admits(&ast, &ir, true).is_err());
}

#[test]
fn divergence_is_recognised_only_when_control_surely_leaves() {
    let ret = || Instr::Return { value: None };
    let brk = || Instr::Break { live: vec![] };
    let iff = |then_instrs: Vec<Instr>, else_instrs: Vec<Instr>| Instr::If {
        cond_id: ValueId(0),
        cond_instrs: vec![],
        then_instrs,
        then_result: ValueId(0),
        else_instrs,
        else_result: ValueId(0),
        dst: ValueId(1),
        branch_bindings: vec![],
        merges: vec![],
    };
    assert!(diverges(&[Instr::ConstI64(ValueId(0), 1), ret()]));
    assert!(diverges(&[brk()]));
    assert!(diverges(&[iff(vec![ret()], vec![brk()])]));
    assert!(!diverges(&[]));
    assert!(!diverges(&[iff(vec![ret()], vec![])]));
    assert!(!diverges(&[Instr::While {
        cond_id: ValueId(0),
        cond_instrs: vec![],
        body: vec![ret()],
        live_vars: vec![],
        init_ids: vec![],
        exit_ids: vec![],
    }]));
}

#[path = "tests_more.rs"]
mod more;
