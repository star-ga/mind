// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! The frozen stage1.elf's own intrinsics, as the RI-D1 frozen-profile fence admits
//! them (W1).
//!
//! `frozen_profile::admit_call` admits a call by CALLEE. A callee DEFINED in the
//! module is walked by the fence itself; this module decides the other admitted
//! family — the memory / I-O intrinsics the frozen compiler expands inline
//! (`nb_intrinsic_id` → `nb_emit_intrinsic` in `examples/mindc_mind/main.mind`) —
//! keyed on callee, arity, the I/O offset operand, and whether the program reads a
//! result slot the emitter never writes.

use crate::ir::frozen_profile::FrozenProfileRejection;
use crate::ir::{BinOp, Instr, ValueId};

fn reject(construct: &'static str) -> Result<(), FrozenProfileRejection> {
    Err(FrozenProfileRejection { construct })
}

/// Admit a call to the intrinsic `spec` with `args`. `consts` holds the integer
/// constants proven in the call's own straight-line slice (see [`fold_const`]).
pub(crate) fn admit_intrinsic_call(
    spec: &FrozenIntrinsic,
    args: &[ValueId],
    consts: &std::collections::BTreeMap<ValueId, i64>,
) -> Result<(), FrozenProfileRejection> {
    // The frozen emitter reads `ids[0..arity]` without checking how many arguments
    // the call has: a short call reads slots it never wrote, a long one drops
    // operands. Neither is a program the corpus proves.
    if args.len() != spec.arity {
        return reject("call.intrinsic_arity");
    }
    if let Some(at) = spec.offset_arg {
        // The frozen emitter IGNORES the offset and always issues plain read(2) /
        // write(2); the Rust/MLIR runtime issues pread/pwrite for an offset >= 0.
        // Only the `-1` sentinel means the same thing on both, so only a proven
        // `-1` is admitted. An unproven operand is refused, never assumed.
        if consts.get(&args[at]) != Some(&-1) {
            return reject("call.io_offset_not_minus_one");
        }
    }
    Ok(())
}

/// How the frozen emitter treats an admitted intrinsic's result slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrozenResult {
    /// The result slot is written (`nb_store_rax` into the call's dst).
    Value,
    /// The result slot is NEVER written: `nb_emit_intrinsic`'s store arms "leave
    /// dst untouched". Reading it returns whatever the stack slot last held, while
    /// the Rust/MLIR runtime returns 0. See `admit_discard_only_results`.
    DiscardOnly,
}

/// One intrinsic the frozen stage1.elf emits inline (`nb_intrinsic_id` →
/// `nb_emit_intrinsic` in `examples/mindc_mind/main.mind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrozenIntrinsic {
    /// The callee spelling `nb_intrinsic_id` matches byte-for-byte.
    pub(crate) name: &'static str,
    /// The `kind` id `nb_intrinsic_id` returns for it.
    pub(crate) seed_id: u8,
    /// How many argument slots `nb_emit_intrinsic` reads for it.
    pub(crate) arity: usize,
    pub(crate) result: FrozenResult,
    /// The argument the frozen emitter ignores and therefore admits only as `-1`.
    pub(crate) offset_arg: Option<usize>,
}

const fn fi(
    name: &'static str,
    seed_id: u8,
    arity: usize,
    result: FrozenResult,
    offset_arg: Option<usize>,
) -> FrozenIntrinsic {
    FrozenIntrinsic {
        name,
        seed_id,
        arity,
        result,
        offset_arg,
    }
}

/// The memory / I-O intrinsic family the FROZEN stage1.elf already emits, keyed by
/// CALLEE and ARITY — exactly the names `nb_intrinsic_id` recognises, and no other.
///
/// This is the corpus-derived intrinsic set the `admit_call` deferral used to name.
/// Each row is proven by an `IN_PROFILE` behaviour row in
/// `examples/mindc_mind/ri_d1_frozen_profile_gate.py` (native exit == MLIR exit
/// where MLIR builds the call, == an oracle computed independently of both for the
/// native-only `argc`/`argv`/`wrap_*`). Two lists that must agree are pinned by
/// tests: `frozen_intrinsics_match_the_seed_emitter` reads `nb_intrinsic_id`
/// itself, and `frozen_intrinsics_agree_with_the_contract_registry` checks every
/// row the canonical intrinsic registry also describes.
///
/// A name NOT in `nb_intrinsic_id` (`__mind_conv_*`, `__mind_bits_to_f64`, ...)
/// is out of scope: it needs a new seed intrinsic id and stays a clean
/// `call.undefined_or_builtin` refusal.
///
/// Known divergences admitted with these rows, all outside the value domain the
/// corpus proves (callers test `< 0` / `> 0`, never the exact error value):
/// `read`/`write`/`open` return `-errno` natively and `-1` from the C runtime on
/// failure; `read`/`write` with a null buffer or `count <= 0` return 0 from the C
/// runtime without a syscall, natively the syscall's own result; `alloc(n <= 0)`
/// returns 0 from the C runtime and a fresh arena pointer natively; `free` is a
/// no-op on the native bump arena.
///
/// deferred: these rows key on name + arity only. The memory rows do not see the
/// operand's DECLARED type (an f64 stored through `store_i64` is a bit pattern,
/// not a conversion), the same (op, operand_type) gap `admit_binop` defers — upgrade
/// path: thread `FnDef.value_types` into this predicate (task #313).
pub(crate) const FROZEN_NATIVE_INTRINSICS: [FrozenIntrinsic; 19] = [
    fi("__mind_alloc", 1, 1, FrozenResult::Value, None),
    fi("__mind_store_i64", 2, 2, FrozenResult::DiscardOnly, None),
    fi("__mind_load_i64", 3, 1, FrozenResult::Value, None),
    fi("__mind_store_i8", 4, 2, FrozenResult::DiscardOnly, None),
    fi("__mind_load_i8", 5, 1, FrozenResult::Value, None),
    fi("__mind_realloc", 6, 2, FrozenResult::Value, None),
    fi("__mind_free", 7, 1, FrozenResult::Value, None),
    fi("__mind_write", 8, 4, FrozenResult::Value, Some(3)),
    fi("__mind_read", 9, 4, FrozenResult::Value, Some(3)),
    fi("__mind_argc", 10, 0, FrozenResult::Value, None),
    fi("__mind_argv", 11, 1, FrozenResult::Value, None),
    fi("__mind_open", 12, 1, FrozenResult::Value, None),
    fi("__mind_store_i16", 13, 2, FrozenResult::DiscardOnly, None),
    fi("__mind_load_i16", 14, 1, FrozenResult::Value, None),
    fi("__mind_store_i32", 15, 2, FrozenResult::DiscardOnly, None),
    fi("__mind_load_i32", 16, 1, FrozenResult::Value, None),
    fi("__mind_wrap_i8", 17, 1, FrozenResult::Value, None),
    fi("__mind_wrap_i16", 18, 1, FrozenResult::Value, None),
    fi("__mind_wrap_i32", 19, 1, FrozenResult::Value, None),
];

/// The frozen-emitter row for `name`, if `nb_intrinsic_id` recognises it.
pub(crate) fn frozen_native_intrinsic(name: &str) -> Option<&'static FrozenIntrinsic> {
    FROZEN_NATIVE_INTRINSICS.iter().find(|row| row.name == name)
}

/// Refuse a program that READS the result of a [`FrozenResult::DiscardOnly`]
/// intrinsic. The frozen emitter never writes that slot, so `let r =
/// __mind_store_i64(p, v); return r;` returns a stale stack value natively and 0
/// from the Rust/MLIR runtime. A store used as a statement is admitted.
///
/// Judged per function. A store whose dst becomes an `if`'s branch result taints
/// that `if`'s dst (a statement-position `if` whose branch ends in a store is
/// fine; `let x = if c { store(..) } else { 0 };` is not). A tainted id that
/// appears in ANY operand position — the audited `for_each_operand` list plus the
/// control-flow ids it deliberately omits (`cond_id`, bindings, loop live/exit
/// ids) — is refused. Over-refusal is the safe direction.
pub(crate) fn admit_discard_only_results(instrs: &[Instr]) -> Result<(), FrozenProfileRejection> {
    for instr in instrs {
        if let Instr::FnDef { body, ret_id, .. } = instr {
            admit_discard_only_in_fn(body, *ret_id)?;
            admit_discard_only_results(body)?;
        }
    }
    Ok(())
}

fn admit_discard_only_in_fn(
    body: &[Instr],
    ret_id: Option<ValueId>,
) -> Result<(), FrozenProfileRejection> {
    use std::collections::BTreeSet;
    let mut tainted: BTreeSet<ValueId> = BTreeSet::new();
    #[cfg_attr(not(feature = "std-surface"), allow(unused_mut))]
    let mut ifs: Vec<(ValueId, ValueId, ValueId)> = Vec::new();
    visit_fn_instrs(body, &mut |instr| match instr {
        Instr::Call { dst, name, .. }
            if frozen_native_intrinsic(name)
                .is_some_and(|row| row.result == FrozenResult::DiscardOnly) =>
        {
            tainted.insert(*dst);
        }
        #[cfg(feature = "std-surface")]
        Instr::If {
            then_result,
            else_result,
            dst,
            ..
        } => ifs.push((*then_result, *else_result, *dst)),
        _ => {}
    });
    if tainted.is_empty() {
        return Ok(());
    }
    // An `if` result can feed another `if` result: propagate to a fixpoint.
    loop {
        let before = tainted.len();
        for (then_result, else_result, dst) in &ifs {
            if tainted.contains(then_result) || tainted.contains(else_result) {
                tainted.insert(*dst);
            }
        }
        if tainted.len() == before {
            break;
        }
    }
    // A body's tail expression is the function's implicit result (`ret_id`), read
    // by the caller: `fn f(p: i64) -> i64 { __mind_store_i8(p, 1) }`.
    let mut observed = ret_id.is_some_and(|id| tainted.contains(&id));
    visit_fn_instrs(body, &mut |instr| {
        crate::opt::ir_canonical::for_each_operand(instr, |v| observed |= tainted.contains(&v));
        for v in control_flow_ids(instr) {
            observed |= tainted.contains(&v);
        }
    });
    if observed {
        reject("call.discard_only_result_used")
    } else {
        Ok(())
    }
}

/// Visit every instruction of one function body, through nested control flow but
/// not into nested `FnDef` bodies (those are judged as their own functions).
fn visit_fn_instrs(instrs: &[Instr], f: &mut dyn FnMut(&Instr)) {
    for instr in instrs {
        f(instr);
        if matches!(instr, Instr::FnDef { .. }) {
            continue;
        }
        #[cfg(feature = "std-surface")]
        for nested in crate::ir::instr_bodies(instr) {
            visit_fn_instrs(nested, f);
        }
    }
}

/// The control-flow ids `for_each_operand` leaves out because they live in a
/// region's own namespace. For the discard-only check they are conservatively
/// treated as reads.
fn control_flow_ids(instr: &Instr) -> Vec<ValueId> {
    match instr {
        #[cfg(feature = "std-surface")]
        Instr::If {
            cond_id,
            branch_bindings,
            ..
        } => std::iter::once(*cond_id)
            .chain(branch_bindings.iter().map(|(_, v)| *v))
            .collect(),
        #[cfg(feature = "std-surface")]
        Instr::While {
            cond_id,
            live_vars,
            exit_ids,
            ..
        } => std::iter::once(*cond_id)
            .chain(live_vars.iter().map(|(_, v)| *v))
            .chain(exit_ids.iter().copied())
            .collect(),
        _ => Vec::new(),
    }
}

/// Fold the integer constants defined in ONE straight-line instruction slice.
///
/// Deliberately not inherited from enclosing slices: a loop body may name a
/// loop-carried block argument whose id the pre-loop value also used, so a
/// constant proven outside a region is not trusted inside it. A constant the
/// slice itself defines before the use is SSA-fixed. Under-proving only refuses.
pub(crate) fn fold_const(instr: &Instr, consts: &mut std::collections::BTreeMap<ValueId, i64>) {
    match instr {
        Instr::ConstI64(dst, v) => {
            consts.insert(*dst, *v);
        }
        Instr::BinOp { dst, op, lhs, rhs } => {
            let (Some(a), Some(b)) = (consts.get(lhs), consts.get(rhs)) else {
                return;
            };
            let folded = match op {
                BinOp::Add => a.checked_add(*b),
                BinOp::Sub => a.checked_sub(*b),
                BinOp::Mul => a.checked_mul(*b),
                _ => None,
            };
            if let Some(v) = folded {
                consts.insert(*dst, v);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
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

    fn call_with(name: &str, n: usize) -> Instr {
        Instr::legacy_call(ValueId(100), name, (0..n).map(ValueId).collect())
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
                admits(&[call_with(row.name, row.arity)]),
                Ok(()),
                "{}",
                row.name
            );
            assert_eq!(
                admits(&[call_with(row.name, row.arity + 1)])
                    .unwrap_err()
                    .construct,
                "call.intrinsic_arity",
                "{}",
                row.name
            );
        }
        for name in [
            "__mind_conv_i64",
            "__mind_conv_f64",
            "__mind_bits_to_f64",
            "__mind_f64_to_bits",
            "__mind_vec_zeroed",
            "__mind_now_ns",
            "__mind_calloc",
        ] {
            for n in 0..5 {
                assert_eq!(
                    admits(&[call_with(name, n)]).unwrap_err().construct,
                    "call.undefined_or_builtin",
                    "{name}/{n}"
                );
            }
        }
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
}
