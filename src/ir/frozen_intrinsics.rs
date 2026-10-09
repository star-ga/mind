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
    if let Some(at) = spec.size_arg {
        // `__mind_alloc(n <= 0)` is 0 by contract (`runtime-support/mind_intrinsics.c`,
        // relied on by std/vec.mind) and `realloc(p, n <= 0)` frees and returns 0 on
        // the C runtime, while the frozen bump arena hands back a fresh pointer for
        // both. Only a size PROVEN positive means the same thing on both backends.
        if !consts.get(&args[at]).is_some_and(|n| *n > 0) {
            return reject("call.alloc_size_not_proven_positive");
        }
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
    /// The byte-size argument, admitted only when proven `> 0` (`alloc`, `realloc`).
    pub(crate) size_arg: Option<usize>,
}

const VALUE: FrozenResult = FrozenResult::Value;
const DISCARD: FrozenResult = FrozenResult::DiscardOnly;

const fn fi(
    name: &'static str,
    seed_id: u8,
    arity: usize,
    result: FrozenResult,
    offset_arg: Option<usize>,
    size_arg: Option<usize>,
) -> FrozenIntrinsic {
    FrozenIntrinsic {
        name,
        seed_id,
        arity,
        result,
        offset_arg,
        size_arg,
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
/// Where the two runtimes could disagree, the rows close the gap or refuse:
/// `alloc`/`realloc` are admitted only with a size proven `> 0` (`size_arg`), since
/// a size `<= 0` is 0 from the C runtime and a fresh pointer natively;
/// `read`/`write` only with an offset proven `-1` (`offset_arg`); every operand
/// must be proven an i64 slot over the whole program ([`admit_frozen_program`]:
/// a declared `i64` parameter counts only when every call site passes a proven
/// value), since a float reaches the frozen emitter as its raw bits; and the C
/// runtime returns the raw syscall's `-errno` on an I/O failure, exactly as the
/// native syscalls do.
///
/// `free` is a no-op on the native bump arena, so native memory grows with every
/// byte EVER allocated while `malloc` memory grows with the peak. The two agree
/// only while the arena never runs out, so [`admit_frozen_program`] admits a
/// program only with a static proof that its total allocation fits a fixed budget
/// (no allocation in a loop or through recursion; `alloc_budget.rs`). Divergences
/// that remain are outside defined behaviour: a use-after-free or double free is
/// undefined on the C runtime, and fresh native memory reads as zero where
/// `malloc` memory is indeterminate.
pub(crate) const FROZEN_NATIVE_INTRINSICS: [FrozenIntrinsic; 19] = [
    fi("__mind_alloc", 1, 1, VALUE, None, Some(0)),
    fi("__mind_store_i64", 2, 2, DISCARD, None, None),
    fi("__mind_load_i64", 3, 1, VALUE, None, None),
    fi("__mind_store_i8", 4, 2, DISCARD, None, None),
    fi("__mind_load_i8", 5, 1, VALUE, None, None),
    fi("__mind_realloc", 6, 2, VALUE, None, Some(1)),
    fi("__mind_free", 7, 1, VALUE, None, None),
    fi("__mind_write", 8, 4, VALUE, Some(3), None),
    fi("__mind_read", 9, 4, VALUE, Some(3), None),
    fi("__mind_argc", 10, 0, VALUE, None, None),
    fi("__mind_argv", 11, 1, VALUE, None, None),
    fi("__mind_open", 12, 1, VALUE, None, None),
    fi("__mind_store_i16", 13, 2, DISCARD, None, None),
    fi("__mind_load_i16", 14, 1, VALUE, None, None),
    fi("__mind_store_i32", 15, 2, DISCARD, None, None),
    fi("__mind_load_i32", 16, 1, VALUE, None, None),
    fi("__mind_wrap_i8", 17, 1, VALUE, None, None),
    fi("__mind_wrap_i16", 18, 1, VALUE, None, None),
    fi("__mind_wrap_i32", 19, 1, VALUE, None, None),
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

mod alloc_budget;
mod operand_types;
mod program;

/// The whole-program half of the W1 fence, over the user module (`parts[0]`) and
/// every linked std seed function: the operand / result proof
/// (`operand_types::admit_program_operand_types`, which follows values across calls)
/// and the static allocation bound (`alloc_budget::admit_alloc_budget`).
pub(crate) fn admit_frozen_program(
    parts: &[&crate::ir::IRModule],
) -> Result<(), FrozenProfileRejection> {
    let prog = program::Program::new(parts);
    operand_types::admit_program_operand_types(&prog)?;
    alloc_budget::admit_alloc_budget(&prog)
}

#[cfg(test)]
mod program_tests;
#[cfg(test)]
mod tests;
