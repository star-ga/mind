// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Fixed i64 array reads and the named catch-all refusals.

use super::*;

fn lower(source: &str) -> (ast::Module, IRModule) {
    let ast = crate::parser::parse(source).expect("fixture parses");
    let ir = crate::eval::lower_to_ir(&ast).expect("fixture lowers");
    (ast, ir)
}
fn admits(source: &str) -> Check<()> {
    let (ast, ir) = lower(source);
    profile_lane_admits(&ast, &ir, true)
}
fn source_kind(source: &str) -> &'static str {
    profile_lane_admits_source(&crate::parser::parse(source).expect(source), true)
        .expect_err(source)
        .construct
}
fn first_body(ir: &mut IRModule) -> &mut Vec<Instr> {
    ir.instrs
        .iter_mut()
        .find_map(|i| match i {
            Instr::FnDef { body, .. } => Some(body),
            _ => None,
        })
        .expect("a function body")
}

/// The programs main's default native path builds and `array_oob_trap_run` /
/// `native_array_binop_refusal` / the RI-D1 gate execute.
#[test]
fn fixed_array_reads_are_admitted() {
    for source in [
        "fn main()->i64{let a=[1,2,3,4]; let mut s:i64=0; for i in 0..4 {s=s+a[i];} return s;}",
        "fn main() -> i64 { let a: [i64; 3] = [1, 2, 4]; let b: [i64; 3] = [10, 20, 30]; \
         if a[0] + b[0] == 11 { if a[2] + b[2] == 34 { return 1; } } return 2; }",
        "fn at(k: i64) -> i64 { let a: [i64; 3] = [10, 20, 30]; return a[k]; } \
         fn main() -> i64 { return at(-1); }",
        "fn main() -> i64 { let a = [-3, 5]; let i: i64 = 1; return a[i] + a[0]; }",
        "fn main() -> i64 { let a = [5, 6, 7]; let mut i: i64 = 0; let mut s: i64 = 0; \
         while i < 3 { s = s + a[i]; i = i + 1; } return s; }",
        "fn main() -> i64 { let c: i64 = 1; if c == 1 { let a = [4, 5]; return a[1]; } return 0; }",
        "fn main() -> i64 { let a = [5, 6]; if 1 == 1 { let a: i64 = 9; return a; } return a[0]; }",
        "fn main() -> i64 { let a: i64 = 2; let a = [7, 8]; return a[1]; }",
        // Integer constant expressions fold by the spec 3.2 rule.
        "fn main() -> i64 { let a = [0 - 1, 2, 3]; return a[0]; }",
        "fn main() -> i64 { let a: [i64; 3] = [(2 * 3) - 7, 9 / 2, -(4)]; return a[1]; }",
    ] {
        assert_eq!(admits(source), Ok(()), "{source}");
    }
}

/// A folded element lowers to the same `ConstArray` value as its literal, and an
/// element that does not fold (division by zero, overflow) still refuses.
#[test]
fn folded_elements_are_constants_and_bad_folds_refuse() {
    let values = |source: &str| {
        let (_, mut ir) = lower(source);
        first_body(&mut ir)
            .iter()
            .find_map(|i| match i {
                Instr::ConstArray { values, .. } => Some(values.clone()),
                _ => None,
            })
            .expect("the literal lowers to ConstArray")
    };
    assert_eq!(
        values("fn main() -> i64 { let a = [0 - 1, 2 * 3, 7 / 2]; return a[0]; }"),
        vec![-1, 6, 3]
    );
    assert_eq!(
        values("fn main() -> i64 { let a: [i64; 2] = [0 - 9, 4]; return a[0]; }"),
        vec![-9, 4]
    );
    for source in [
        "fn main() -> i64 { let a = [1 / 0, 2]; return a[1]; }",
        "fn main() -> i64 { let a = [9223372036854775807 + 1, 2]; return a[1]; }",
        "fn main() -> i64 { let a = [7 % 2, 2]; return a[1]; }",
    ] {
        assert_eq!(source_kind(source), "lane.source.array_element", "{source}");
    }
}

#[test]
fn every_other_array_use_refuses_by_name() {
    for (source, kind) in [
        (
            "fn main() -> i64 { let a = [1, 2]; return a; }",
            "lane.source.array_value",
        ),
        (
            "fn main() -> i64 { let a = [1, 2]; let b = a; return b[0]; }",
            "lane.source.array_value",
        ),
        (
            "fn main() -> i64 { let a = [1, 2]; let b = [3, 4]; let c = a + b; return 0; }",
            "lane.source.array_value",
        ),
        (
            "fn main() -> i64 { let a = [1, 2]; a = [3, 4]; return a[0]; }",
            "lane.source.array_value",
        ),
        (
            "fn f(x: i64) -> i64 { return x; } fn main() -> i64 { let a = [1, 2]; return f(a); }",
            "lane.source.array_value",
        ),
        (
            "fn main() -> i64 { let a = [1, 2]; a[0] = 5; return a[0]; }",
            "lane.source.index_store",
        ),
        (
            "fn main() -> i64 { let x: i64 = 3; return x[0]; }",
            "lane.source.index",
        ),
        (
            "fn main() -> i64 { return [1, 2][0]; }",
            "lane.source.index",
        ),
        (
            "fn main() -> i64 { let i = [1, 2]; let s: i64 = 0; for i in 0..2 { s = s + i[0]; } return s; }",
            "lane.source.index",
        ),
        (
            "fn main() -> i64 { let x: i64 = 1; let a = [x, 2]; return a[0]; }",
            "lane.source.array_element",
        ),
        (
            "fn main() -> i64 { let a = [1.0, 2.0]; return 0; }",
            "lane.source.array_element",
        ),
        (
            "fn main() -> i64 { let a: [i64; 3] = [1, 2]; return a[0]; }",
            "lane.type.array",
        ),
        (
            "fn main() -> i64 { let a: [f64; 2] = [1, 2]; return a[0]; }",
            "lane.type.array",
        ),
        (
            "fn main() -> i64 { let t: tensor<i64[2]> = [1, 2]; return t[0]; }",
            "lane.type.tensor",
        ),
        (
            "fn f(a: [i64; 2]) -> i64 { return 0; } fn main() -> i64 { return 0; }",
            "lane.type.array",
        ),
        (
            "fn f(x: i64) -> i64 { return x; } fn main() -> i64 { return f([1, 2]); }",
            "lane.source.array_literal",
        ),
        (
            "fn main() -> i64 { let a = [1, 2]; let i: f64 = 1.0; return a[i]; }",
            "lane.type.mismatch",
        ),
    ] {
        assert_eq!(source_kind(source), kind, "{source}");
    }
}

#[test]
fn catch_all_refusals_name_the_construct() {
    for (source, kind) in [
        (
            "fn sink(x: Tensor<f32, [4]>) -> i64 { return 0; } fn main() -> i64 { return 0; }",
            "lane.type.tensor",
        ),
        ("fn f(x: i32) -> i64 { return 0; }", "lane.type.int_width"),
        (
            "fn main() -> i64 { let x: u8 = 2; return 0; }",
            "lane.type.int_width",
        ),
        (
            "fn main() -> i64 { let t = (1, 2); return 0; }",
            "lane.source.tuple",
        ),
        (
            "fn main() -> i64 { let x: i64 = 1; return x.abs(); }",
            "lane.source.method_call",
        ),
    ] {
        assert_eq!(source_kind(source), kind, "{source}");
    }
}

/// The IR walk admits only an `ArrayLoad` of a `ConstArray` base, and keeps
/// the array value out of every other use.
#[test]
fn ir_arrays_are_read_only_by_array_load() {
    let source = "fn main() -> i64 { let a = [1, 2]; let i: i64 = 1; return a[i]; }";
    let (ast, mut ir) = lower(source);
    assert_eq!(profile_lane_admits(&ast, &ir, true), Ok(()));
    let base = first_body(&mut ir)
        .iter()
        .find_map(|i| match i {
            Instr::ConstArray { dst, .. } => Some(*dst),
            _ => None,
        })
        .expect("the literal lowers to ConstArray");
    let mut wrong_base = ir.clone();
    for instr in first_body(&mut wrong_base).iter_mut() {
        if let Instr::ArrayLoad { base: b, index, .. } = instr {
            *b = *index;
        }
    }
    assert_eq!(
        profile_lane_admits(&ast, &wrong_base, true).map_err(|r| r.construct),
        Err("lane.ir.array_base")
    );
    let mut used_as_value = ir.clone();
    for instr in first_body(&mut used_as_value).iter_mut() {
        if let Instr::Return { value } = instr {
            *value = Some(base);
        }
    }
    assert_eq!(
        profile_lane_admits(&ast, &used_as_value, true).map_err(|r| r.construct),
        Err("lane.ir.untyped_value")
    );
}
