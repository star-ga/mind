// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
use super::*;

fn lower(source: &str) -> (ast::Module, IRModule) {
    let ast = crate::parser::parse(source).expect("fixture parses");
    let ir = crate::eval::lower_to_ir(&ast).expect("fixture lowers");
    (ast, ir)
}
fn admits(source: &str) -> Check<()> {
    let (ast, ir) = lower(source);
    profile_lane_admits(&ast, &ir)
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
        "fn main() -> f64 { let x = 1.0; let i = 0; while i < 3 { x = x * 2.0; i = i + 1; } return x; }",
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
        assert!(profile_lane_admits_source(&ast).is_err(), "{source}");
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
        "extern \"C\" { fn close(fd: i32) -> i32; } fn main() -> i64 { return 0; }",
        "extern \"C\" { fn __mind_alloc(x: i64) -> i64; } fn main() -> i64 { return 0; }",
    ] {
        let ast = crate::parser::parse(source).expect(source);
        assert!(profile_lane_admits_source(&ast).is_err(), "{source}");
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
        assert!(profile_lane_admits_source(&ast).is_err(), "{source}");
    }
}

#[test]
fn exact_intrinsic_contract_inventory() {
    for (name, n) in [
        ("__mind_alloc", 1),
        ("__mind_read", 4),
        ("__mind_write", 4),
        ("__mind_free", 1),
        ("__mind_store_i32", 2),
        ("__mind_f64_to_bits", 1),
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

#[test]
fn malformed_ir_operands_and_calls_refuse() {
    let (ast, good) = lower("fn main() -> i64 { return 3 + 4; }");
    let mut unknown = good.clone();
    body(&mut unknown).insert(
        0,
        Instr::legacy_call(ValueId(99999), "__mind_forged", vec![]),
    );
    assert!(profile_lane_admits(&ast, &unknown).is_err());
    let mut mixed = good.clone();
    let b = body(&mut mixed);
    let item = b
        .iter_mut()
        .find(|i| matches!(i, Instr::ConstI64(..)))
        .unwrap();
    if let Instr::ConstI64(id, _) = item {
        *item = Instr::ConstF64(*id, 3.0);
    }
    assert!(profile_lane_admits(&ast, &mixed).is_err());
    let mut undefined = good.clone();
    let item = body(&mut undefined)
        .iter_mut()
        .find(|i| matches!(i, Instr::BinOp { .. }))
        .unwrap();
    if let Instr::BinOp { lhs, .. } = item {
        *lhs = ValueId(99999);
    }
    assert!(profile_lane_admits(&ast, &undefined).is_err());
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
    assert!(profile_lane_admits(&ast, &bad).is_err());
    let mut bad = good.clone();
    body(&mut bad).insert(0, Instr::ConstI64(ValueId(99999), 1));
    for i in body(&mut bad) {
        if let Instr::Return { value } = i {
            *value = Some(ValueId(99999));
        }
    }
    assert!(profile_lane_admits(&ast, &bad).is_err());
}

#[test]
fn misplaced_loop_exit_is_refused() {
    let (ast, mut ir) = lower("fn main() -> i64 { return 1; }");
    body(&mut ir).insert(0, Instr::Break { live: vec![] });
    assert!(profile_lane_admits(&ast, &ir).is_err());
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
    assert!(profile_lane_admits(&ast, &wrong_arity).is_err());
    let mut wrong_type = good.clone();
    for i in body(&mut wrong_type) {
        if let Instr::ConstI64(id, _) = i {
            *i = Instr::ConstF64(*id, 8.0);
        }
    }
    assert!(profile_lane_admits(&ast, &wrong_type).is_err());
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
    assert!(profile_lane_admits(&ast, &ir).is_err());
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
    assert!(profile_lane_admits(&ast, &ir).is_err());
}

#[test]
fn store_results_are_discard_only_in_source_and_ir() {
    for name in ["__mind_store_i64", "__mind_store_i8", "__mind_store_i32"] {
        let source = format!("fn main() -> i64 {{ return {name}(0, 1); }}");
        let ast = crate::parser::parse(&source).unwrap();
        assert!(profile_lane_admits_source(&ast).is_err());
        for expression in [
            format!("let x = {name}(0, 1); return 0;"),
            format!("{name}(0, 1)"),
        ] {
            let ast =
                crate::parser::parse(&format!("fn main() -> i64 {{ {expression} }}")).unwrap();
            assert!(profile_lane_admits_source(&ast).is_err());
        }
        let (ast, mut ir) = lower(&format!("fn main() -> i64 {{ {name}(0, 1); return 0; }}"));
        assert_eq!(profile_lane_admits(&ast, &ir), Ok(()));
        let b = body(&mut ir);
        let dst = b
            .iter()
            .find_map(|i| match i {
                Instr::Call { dst, .. } => Some(*dst),
                _ => None,
            })
            .unwrap();
        for i in b {
            if let Instr::Return { value } = i {
                *value = Some(dst);
            }
        }
        assert!(profile_lane_admits(&ast, &ir).is_err());
    }
}
