// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Unit tests for the frozen-intrinsic admission rows.

use super::*;
use crate::ir::IRModule;
use crate::ir::frozen_profile::profile_frozen_admits;

/// The production entry over a bare instruction list.
fn admits(instrs: &[Instr]) -> Result<(), FrozenProfileRejection> {
    let module = IRModule {
        instrs: instrs.to_vec(),
        ..IRModule::default()
    };
    profile_frozen_admits(&module)
}

fn fndef(body: Vec<Instr>) -> Instr {
    Instr::FnDef {
        name: "f".into(),
        params: vec![],
        ret_id: None,
        body,
        reap_threshold: None,
        semantic_types: None,
        #[cfg(feature = "std-surface")]
        value_types: Default::default(),
    }
}

/// A call to `name` with `n` operands, each a defined positive i64 constant.
fn call_with(name: &str, n: usize) -> Vec<Instr> {
    let mut instrs: Vec<Instr> = (0..n).map(|i| Instr::ConstI64(ValueId(i), 8)).collect();
    instrs.push(Instr::legacy_call(
        ValueId(100),
        name,
        (0..n).map(ValueId).collect(),
    ));
    instrs
}

/// Admission is keyed on CALLEE and ARITY: every row is admitted at its own
/// arity, refused at any other, and a name `nb_intrinsic_id` does not know is
/// refused at every arity.
#[test]
fn intrinsic_admission_is_keyed_on_callee_and_arity() {
    for row in FROZEN_NATIVE_INTRINSICS {
        if row.offset_arg.is_some() {
            continue; // read/write: covered with a proven offset below
        }
        assert_eq!(
            admits(&call_with(row.name, row.arity)),
            Ok(()),
            "{}",
            row.name
        );
        assert_eq!(
            admits(&call_with(row.name, row.arity + 1))
                .unwrap_err()
                .construct,
            "call.intrinsic_arity",
            "{}",
            row.name
        );
    }
    for name in [
        "__mind_bits_to_f64",
        "__mind_f64_to_bits",
        "__mind_vec_zeroed",
        "__mind_now_ns",
        "__mind_calloc",
    ] {
        for n in 0..5 {
            assert_eq!(
                admits(&call_with(name, n)).unwrap_err().construct,
                "call.undefined_or_builtin",
                "{name}/{n}"
            );
        }
    }
}

/// The raw-IR form of the operand rule (no declared types needed): a float
/// constant, a comparison and an id the scope never defines are refused; an
/// integer constant and arithmetic over integer constants are admitted.
#[test]
fn intrinsic_operands_are_keyed_on_what_they_denote() {
    let wrap = |defs: Vec<Instr>| {
        let mut body = defs;
        body.push(Instr::legacy_call(
            ValueId(9),
            "__mind_wrap_i8",
            vec![ValueId(0)],
        ));
        admits(&[fndef(body)])
    };
    let refused = [
        vec![Instr::ConstF64(ValueId(0), 300.0)],
        vec![
            Instr::ConstI64(ValueId(1), 1),
            Instr::BinOp {
                dst: ValueId(0),
                op: BinOp::Lt,
                lhs: ValueId(1),
                rhs: ValueId(1),
            },
        ],
        vec![
            Instr::ConstF64(ValueId(1), 1.0),
            Instr::ConstI64(ValueId(2), 1),
            Instr::BinOp {
                dst: ValueId(0),
                op: BinOp::Add,
                lhs: ValueId(2),
                rhs: ValueId(1),
            },
        ],
        vec![],
    ];
    for defs in refused {
        assert_eq!(
            wrap(defs.clone()).unwrap_err().construct,
            "call.intrinsic_operand_not_i64",
            "{defs:?}"
        );
    }
    assert_eq!(wrap(vec![Instr::ConstI64(ValueId(0), 300)]), Ok(()));
    assert_eq!(
        wrap(vec![
            Instr::ConstI64(ValueId(1), 300),
            Instr::ConstI64(ValueId(2), 2),
            Instr::BinOp {
                dst: ValueId(0),
                op: BinOp::Mul,
                lhs: ValueId(1),
                rhs: ValueId(2),
            },
        ]),
        Ok(())
    );
}

/// `read`/`write` are admitted only with an offset proven `-1` in the same
/// straight-line slice: the frozen emitter ignores the offset, the runtime
/// honours it.
#[test]
fn io_offset_must_be_proven_minus_one() {
    let io = |offset: Vec<Instr>, name: &str| {
        let mut body = vec![
            Instr::ConstI64(ValueId(0), 1),
            Instr::ConstI64(ValueId(1), 4096),
            Instr::ConstI64(ValueId(2), 3),
        ];
        body.extend(offset);
        body.push(Instr::legacy_call(
            ValueId(9),
            name,
            vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
        ));
        admits(&[fndef(body)])
    };
    let minus_one = || vec![Instr::ConstI64(ValueId(3), -1)];
    let zero_minus_one = || {
        vec![
            Instr::ConstI64(ValueId(5), 0),
            Instr::ConstI64(ValueId(6), 1),
            Instr::BinOp {
                dst: ValueId(3),
                op: BinOp::Sub,
                lhs: ValueId(5),
                rhs: ValueId(6),
            },
        ]
    };
    for name in ["__mind_write", "__mind_read"] {
        assert_eq!(io(minus_one(), name), Ok(()), "{name}");
        assert_eq!(io(zero_minus_one(), name), Ok(()), "{name}");
        // A proven non-sentinel offset (pread/pwrite on MLIR) and an unproven one.
        assert_eq!(
            io(vec![Instr::ConstI64(ValueId(3), 0)], name)
                .unwrap_err()
                .construct,
            "call.io_offset_not_minus_one"
        );
        assert_eq!(
            io(
                vec![Instr::Param {
                    dst: ValueId(3),
                    name: "off".into(),
                    index: 0
                }],
                name
            )
            .unwrap_err()
            .construct,
            "call.io_offset_not_minus_one"
        );
    }
}

/// The two lists that must agree: this table and the frozen emitter's own
/// `nb_intrinsic_id`. Every id that function returns is a row, every row's id
/// is returned, and every `__mind_*` name its body mentions is a row.
#[test]
fn frozen_intrinsics_match_the_seed_emitter() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/mindc_mind/main.mind");
    let src = std::fs::read_to_string(path).expect("main.mind");
    let start = src
        .find("pub fn nb_intrinsic_id(")
        .expect("nb_intrinsic_id in main.mind");
    let len = src[start..].find("\n}\n").expect("end of nb_intrinsic_id");
    let body = &src[start..start + len];
    let mut ids = std::collections::BTreeSet::new();
    for piece in body.split("return ").skip(1) {
        let digits: String = piece.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && piece[digits.len()..].starts_with(';') {
            ids.insert(digits.parse::<u8>().expect("id"));
        }
    }
    let rows: std::collections::BTreeSet<u8> =
        FROZEN_NATIVE_INTRINSICS.iter().map(|r| r.seed_id).collect();
    assert_eq!(ids, rows, "nb_intrinsic_id ids vs FROZEN_NATIVE_INTRINSICS");
    assert_eq!(
        rows.len(),
        FROZEN_NATIVE_INTRINSICS.len(),
        "duplicate seed id"
    );
    let mut mentioned = std::collections::BTreeSet::new();
    for (at, _) in body.match_indices("__mind_") {
        let name: String = body[at..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.len() > "__mind_".len() {
            mentioned.insert(name);
        }
    }
    let names: std::collections::BTreeSet<String> = FROZEN_NATIVE_INTRINSICS
        .iter()
        .map(|r| r.name.to_string())
        .collect();
    assert_eq!(
        mentioned, names,
        "names nb_intrinsic_id documents vs the table"
    );
}

/// Every row the canonical intrinsic registry also describes agrees with it on
/// seed id, arity, result-use and offset policy.
#[test]
fn frozen_intrinsics_agree_with_the_contract_registry() {
    use crate::intrinsic_contract::{IntrinsicImplementation, IntrinsicResult};
    let mut checked = 0;
    for spec in crate::intrinsics::STD_SURFACE_INTRINSICS {
        let Some(contract) = spec.contract.as_ref() else {
            continue;
        };
        let IntrinsicImplementation::FrozenNative { seed_id } = contract.implementation;
        let row = frozen_native_intrinsic(spec.physical_name)
            .unwrap_or_else(|| panic!("{} has a FrozenNative contract", spec.physical_name));
        assert_eq!(row.seed_id, seed_id, "{}", row.name);
        assert_eq!(row.arity, spec.arity, "{}", row.name);
        assert_eq!(
            row.result == FrozenResult::DiscardOnly,
            contract.result == IntrinsicResult::DiscardOnly,
            "{}",
            row.name
        );
        assert_eq!(
            row.offset_arg.is_some(),
            contract.offset_policy
                == crate::intrinsic_contract::IntrinsicOffsetPolicy::MustBeMinusOneIgnored,
            "{}",
            row.name
        );
        checked += 1;
    }
    assert_eq!(checked, 10, "the registry's FrozenNative rows");
    // Rows the registry describes only as legacy keep the registry's arity.
    for spec in crate::intrinsics::STD_SURFACE_INTRINSICS {
        if let Some(row) = frozen_native_intrinsic(spec.physical_name) {
            assert_eq!(row.arity, spec.arity, "{}", row.name);
        }
    }
}

#[cfg(feature = "std-surface")]
fn lowered_admits(src: &str) -> Result<(), FrozenProfileRejection> {
    let module = crate::parser::parse(src).expect("test source parses");
    let ir = crate::eval::lower_to_ir(&module).expect("lowers");
    profile_frozen_admits(&ir)
}

/// The idioms consumers write: a statement store, a store as the last
/// statement of an `if`/`while` branch, `0 - 1` written inside a loop.
#[cfg(feature = "std-surface")]
#[test]
fn statement_stores_and_loop_io_are_admitted() {
    for src in [
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); __mind_store_i64(p, 5); \
         return __mind_load_i64(p); }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); let c: i64 = 1; \
         if c == 1 { __mind_store_i64(p, 5); } return __mind_load_i64(p); }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); let c: i64 = 1; \
         if c == 1 { __mind_store_i8(p, 5); } else { __mind_store_i8(p, 6); } \
         return __mind_load_i8(p); }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); let mut i: i64 = 0; \
         while i < 8 { __mind_store_i8(p + i, 65); i = i + 1; } \
         let mut sent: i64 = 0; \
         while sent < 8 { let n: i64 = __mind_write(1, p + sent, 8 - sent, 0 - 1); \
         sent = sent + n; } return 0; }",
        "fn set(c: i64, v: i64) -> i64 { __mind_store_i64(c, v); 0 } \
         fn main() -> i64 { let p: i64 = __mind_alloc(8); return set(p, 3); }",
    ] {
        assert_eq!(lowered_admits(src), Ok(()), "{src}");
    }
}

/// Reading a store's result, directly or through an `if` value, is refused.
#[cfg(feature = "std-surface")]
#[test]
fn reading_a_discard_only_result_is_refused() {
    for src in [
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); \
         let r: i64 = __mind_store_i64(p, 5); return r + 3; }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); \
         return __mind_store_i32(p, 5); }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); let c: i64 = 1; \
         let x: i64 = if c == 1 { __mind_store_i16(p, 5) } else { 0 }; return x; }",
        "fn f(p: i64) -> i64 { __mind_store_i8(p, 1) } \
         fn main() -> i64 { return f(__mind_alloc(8)); }",
    ] {
        assert_eq!(
            lowered_admits(src).unwrap_err().construct,
            "call.discard_only_result_used",
            "{src}"
        );
    }
}

/// A loop-carried variable inside the body is the header block argument, whose
/// value on every iteration after the first is the BACK-EDGE value. An i64
/// pre-loop value does not prove it: `x` starts as 1 and becomes 2.5.
#[cfg(feature = "std-surface")]
#[test]
fn a_loop_carried_value_needs_its_back_edge_proven() {
    let with_back_edge = |back: Instr| {
        let body = vec![
            Instr::ConstI64(ValueId(0), 1),
            Instr::ConstI64(ValueId(1), 0),
            Instr::While {
                cond_id: ValueId(3),
                cond_instrs: vec![
                    Instr::ConstI64(ValueId(2), 3),
                    Instr::BinOp {
                        dst: ValueId(3),
                        op: BinOp::Lt,
                        lhs: ValueId(1),
                        rhs: ValueId(2),
                    },
                ],
                body: vec![
                    Instr::legacy_call(ValueId(4), "__mind_wrap_i8", vec![ValueId(0)]),
                    back,
                    Instr::ConstI64(ValueId(6), 1),
                    Instr::BinOp {
                        dst: ValueId(7),
                        op: BinOp::Add,
                        lhs: ValueId(1),
                        rhs: ValueId(6),
                    },
                ],
                live_vars: vec![("x".into(), ValueId(5)), ("i".into(), ValueId(7))],
                init_ids: vec![ValueId(0), ValueId(1)],
                exit_ids: vec![ValueId(8), ValueId(9)],
            },
        ];
        admits(&[fndef(body)])
    };
    assert_eq!(
        with_back_edge(Instr::ConstF64(ValueId(5), 2.5))
            .unwrap_err()
            .construct,
        "call.intrinsic_operand_not_i64"
    );
    assert_eq!(with_back_edge(Instr::ConstI64(ValueId(5), 2)), Ok(()));
}

/// A declared signature is trusted for a parameter only when it lists exactly the
/// function's parameters: one recorded without an implicit receiver would shift
/// the f64 `x` (index 1) onto the i64 declared for `y`.
#[cfg(feature = "std-surface")]
#[test]
fn a_parameter_is_not_proven_through_a_shifted_signature() {
    use crate::ast::TypeAnn;
    let module_with = |declared: Vec<TypeAnn>| {
        let mut module = IRModule {
            instrs: vec![Instr::FnDef {
                name: "m".into(),
                params: vec![
                    ("self".into(), ValueId(0)),
                    ("x".into(), ValueId(1)),
                    ("y".into(), ValueId(2)),
                ],
                ret_id: None,
                body: vec![
                    Instr::Param {
                        dst: ValueId(1),
                        name: "x".into(),
                        index: 1,
                    },
                    Instr::legacy_call(ValueId(3), "__mind_wrap_i8", vec![ValueId(1)]),
                ],
                reap_threshold: None,
                semantic_types: None,
                value_types: Default::default(),
            }],
            ..IRModule::default()
        };
        module
            .fn_signatures
            .insert("m".into(), (declared, Some(TypeAnn::ScalarI64)));
        profile_frozen_admits(&module)
    };
    // Receiver omitted: [x: f64, y: i64] against three parameters.
    assert_eq!(
        module_with(vec![TypeAnn::ScalarF64, TypeAnn::ScalarI64])
            .unwrap_err()
            .construct,
        "call.intrinsic_operand_not_i64"
    );
    // Positive control: the same body with a full signature whose `x` is i64.
    assert_eq!(
        module_with(vec![
            TypeAnn::Named("S".into()),
            TypeAnn::ScalarI64,
            TypeAnn::ScalarF64
        ]),
        Ok(())
    );
}

/// The result side: an intrinsic's i64 used as a float (compared against a
/// float, mixed into float arithmetic, passed to an f64 parameter, returned
/// from an f64 function, merged with a float) is refused; used as an i64 it is
/// admitted.
#[cfg(feature = "std-surface")]
#[test]
fn intrinsic_results_must_stay_i64() {
    for src in [
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, 4609434218613702656); \
         let y: f64 = __mind_load_i64(p); if y == 1.5 { return 1; } return 2; }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); let y: f64 = __mind_load_i64(p); \
         let z: f64 = y + 1.0; if z > 2.0 { return 1; } return 2; }",
        "fn f(x: f64) -> i64 { if x > 1.0 { return 1; } return 2; } \
         fn main() -> i64 { let p: i64 = __mind_alloc(8); return f(__mind_load_i64(p)); }",
        "fn g() -> f64 { let p: i64 = __mind_alloc(8); return __mind_load_i64(p); } \
         fn main() -> i64 { if g() > 1.0 { return 1; } return 2; }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); let c: i64 = 1; \
         let x: f64 = if c == 1 { __mind_load_i64(p) } else { 1.5 }; \
         if x > 1.0 { return 1; } return 2; }",
    ] {
        assert_eq!(
            lowered_admits(src).unwrap_err().construct,
            "call.intrinsic_result_not_i64",
            "{src}"
        );
    }
    for src in [
        "fn k(x: i64) -> i64 { return x + 1; } fn main() -> i64 { let p: i64 = __mind_alloc(8); \
         __mind_store_i64(p, 5); let v: i64 = __mind_load_i64(p); \
         if v == 5 { return k(v); } return 0; }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); let mut i: i64 = 0; \
         let mut s: i64 = 0; while i < 4 { s = s + __mind_load_i8(p + i); i = i + 1; } return s; }",
    ] {
        assert_eq!(lowered_admits(src), Ok(()), "{src}");
    }
}

/// A variable offset — the shape three consumer readers use — is refused.
#[cfg(feature = "std-surface")]
#[test]
fn a_variable_io_offset_is_refused_from_source() {
    let src = "fn rd(fd: i64, p: i64, off: i64) -> i64 { return __mind_read(fd, p, 8, off); } \
               fn main() -> i64 { let p: i64 = __mind_alloc(8); return rd(0, p, 0 - 1); }";
    assert_eq!(
        lowered_admits(src).unwrap_err().construct,
        "call.io_offset_not_minus_one"
    );
}

/// `__mind_alloc(n <= 0)` is 0 by contract (C runtime, std/vec.mind) while the
/// frozen bump arena hands back a fresh pointer, and `realloc(p, n <= 0)` frees
/// and returns 0 on the C runtime. So the size operand must be PROVEN positive;
/// a size that is not a proven constant is refused, never assumed.
#[cfg(feature = "std-surface")]
#[test]
fn alloc_and_realloc_sizes_must_be_proven_positive() {
    for src in [
        "fn main() -> i64 { let p: i64 = __mind_alloc(0); if p == 0 { return 1; } return 2; }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); __mind_store_i64(p, 7); \
         let q: i64 = __mind_alloc(0 - 16); if q == 0 { return 1; } \
         __mind_store_i64(q, 99); return __mind_load_i64(p); }",
        "fn mk(n: i64) -> i64 { return __mind_alloc(n); } fn main() -> i64 { return mk(8); }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); let q: i64 = __mind_realloc(p, 0); \
         if q == 0 { return 1; } return 2; }",
        "fn grow(p: i64, n: i64) -> i64 { return __mind_realloc(p, n * 8); } \
         fn main() -> i64 { return grow(__mind_alloc(8), 2); }",
    ] {
        assert_eq!(
            lowered_admits(src).unwrap_err().construct,
            "call.alloc_size_not_proven_positive",
            "{src}"
        );
    }
    for src in [
        "fn main() -> i64 { let p: i64 = __mind_alloc(16); __mind_store_i64(p, 5); \
         return __mind_load_i64(p); }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8 * 2 + 1); \
         let q: i64 = __mind_realloc(p, 64); return __mind_load_i8(q); }",
        "struct P { a: i64, b: i64 } fn mk() -> P { return P { a: 3, b: 4 }; } \
         fn main() -> i64 { let p = mk(); return p.a + p.b; }",
    ] {
        assert_eq!(lowered_admits(src), Ok(()), "{src}");
    }
}

/// An intrinsic operand must DENOTE an i64 slot. A float operand reaches the
/// frozen emitter as its raw IEEE bits (`wrap_i8(300.0)` == 0, `alloc(16.0)`
/// asks for ~4.6e18 bytes) while the MLIR backend refuses the type mismatch; a
/// bool / i32 operand is a different MLIR type too. Admission is keyed on what
/// the operand is proven to be, not on the call's name.
#[cfg(feature = "std-surface")]
#[test]
fn intrinsic_operands_must_be_proven_i64() {
    for src in [
        "fn main() -> i64 { let x: f64 = 300.0; return __mind_wrap_i8(x); }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); let x: f64 = 1.5; \
         __mind_store_i64(p, x); return __mind_load_i64(p); }",
        "fn f(x: f64) -> i64 { return __mind_wrap_i16(x); } fn main() -> i64 { return f(2.0); }",
        "fn g() -> f64 { return 2.0; } fn main() -> i64 { return __mind_wrap_i32(g()); }",
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); let x: f64 = 1.5; \
         let y: f64 = x + 1.0; __mind_store_i8(p, y); return 0; }",
        "fn f(b: bool) -> i64 { return __mind_wrap_i8(b); } fn main() -> i64 { return f(true); }",
        "fn f(y: i32) -> i64 { return __mind_wrap_i8(y); } fn main() -> i64 { return f(2); }",
        "fn main() -> i64 { let a: i64 = 1; return __mind_wrap_i8(a < 2); }",
        // A float carried through a loop or an `if` stays a float.
        "fn main() -> i64 { let mut x: f64 = 1.0; let mut i: i64 = 0; \
         while i < 3 { x = x + 1.0; i = i + 1; } return __mind_wrap_i8(x); }",
        "fn main() -> i64 { let c: i64 = 1; let x: f64 = if c == 1 { 1.0 } else { 2.0 }; \
         return __mind_wrap_i8(x); }",
    ] {
        assert_eq!(
            lowered_admits(src).unwrap_err().construct,
            "call.intrinsic_operand_not_i64",
            "{src}"
        );
    }
    for src in [
        // i64 params, loop-carried and `if`-merged i64 values, intrinsic and
        // user-call results, a struct handle.
        "fn f(n: i64) -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; \
         while i < n { s = s + i; i = i + 1; } let t: i64 = if s > 2 { 1 } else { 2 }; \
         return __mind_wrap_i8(s + t); } fn main() -> i64 { return f(3); }",
        "fn k() -> i64 { return 300; } fn main() -> i64 { let p: i64 = __mind_alloc(8); \
         __mind_store_i64(p, k()); return __mind_wrap_i8(__mind_load_i64(p)); }",
        "struct Q { a: i64, b: i64 } fn get(q: Q) -> i64 { return q.a + q.b; } \
         fn main() -> i64 { let q = Q { a: 3, b: 4 }; return get(q); }",
    ] {
        assert_eq!(lowered_admits(src), Ok(()), "{src}");
    }
}
