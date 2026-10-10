// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! A static bound on how many bytes an admitted program can ever allocate (W1).
//!
//! The frozen emitter's allocator is a bump arena with no reuse: `__mind_free` is
//! a no-op and `__mind_realloc` takes a fresh block. Native memory is therefore
//! O(total bytes EVER allocated), while the C runtime behind the MLIR backend is
//! O(peak live bytes). A loop that allocates and frees 1 MB two thousand times
//! peaks at 1 MB under `malloc` and exhausts the arena natively (exit 70,
//! "native arena exhausted"). Native and MLIR agree exactly when the arena never
//! runs out, so admission requires a PROOF that it cannot: every allocation is
//! counted with its multiplicity along the static call tree, and anything that
//! makes that count unbounded is refused.

use std::collections::BTreeMap;

use super::program::Program;
use super::{fold_const, frozen_native_intrinsic, reject};
use crate::ir::Instr;
use crate::ir::frozen_profile::FrozenProfileRejection;

/// The most bytes (block headers included) an admitted program may allocate in
/// total. The bounded profile's arena is 768 MiB (`NativeLayout::BOUNDED`), the
/// default one 1 GiB; struct literals also bump the arena through the same
/// `__mind_alloc`, and are counted here. 512 MiB leaves a quarter of the bounded
/// arena for anything this count cannot see, and keeps the C runtime's total far
/// inside the 2 GiB address-space limit the gates run under.
pub(crate) const NATIVE_ALLOC_BUDGET: u64 = 512 << 20;

/// The arena bytes one block of `n > 0` data bytes takes: the data rounded up to
/// 8 bytes plus the 8-byte size header `nb_emit_alloc_core` writes before it.
fn block_bytes(n: i64) -> Option<u64> {
    let n = u64::try_from(n).ok()?;
    n.checked_add(7).map(|r| (r & !7) + 8)
}

/// What one scope allocates directly and which functions it calls.
#[derive(Default)]
struct ScopeAllocs {
    direct: u64,
    alloc_in_loop: bool,
    /// (callee scope, call sits inside a loop)
    calls: Vec<(usize, bool)>,
    calls_ambiguous: bool,
}

/// Refuse a program whose total allocation is not statically bounded by
/// [`NATIVE_ALLOC_BUDGET`]:
///
/// * `alloc.in_loop` — an `__mind_alloc` / `__mind_realloc` (a struct literal
///   included) inside a `while`, or a call inside a `while` to a function that
///   can allocate: the trip count is not proven, so neither is the total;
/// * `alloc.recursive` — recursion through a function that can allocate;
/// * `alloc.unbounded_total` — an allocation that cannot be counted (a size not
///   folded to a constant, a call to an ambiguous name, an overflowing sum);
/// * `alloc.total_over_budget` — the counted total exceeds the budget.
///
/// A program with no allocation is not inspected.
pub(crate) fn admit_alloc_budget(
    prog: &Program<'_>,
    bounded: bool,
) -> Result<(), FrozenProfileRejection> {
    let mut infos: Vec<ScopeAllocs> = Vec::with_capacity(prog.scopes.len());
    let mut any_alloc = false;
    for scope in &prog.scopes {
        let mut info = ScopeAllocs::default();
        walk(prog, scope.body, false, &mut info, &mut any_alloc, bounded)?;
        infos.push(info);
    }
    if !any_alloc {
        return Ok(());
    }
    if infos.iter().any(|i| i.calls_ambiguous) {
        return reject("alloc.unbounded_total");
    }
    let bearing = allocating_scopes(&infos);
    for info in &infos {
        if !bounded
            && (info.alloc_in_loop
                || info
                    .calls
                    .iter()
                    .any(|(c, in_loop)| *in_loop && bearing[*c]))
        {
            return reject("alloc.in_loop");
        }
    }
    // Total per scope over the call tree; a scope on the current DFS path that is
    // reached again is recursion.
    let mut memo: BTreeMap<usize, u64> = BTreeMap::new();
    let mut total: u64 = 0;
    let called: Vec<bool> = {
        let mut c = vec![false; infos.len()];
        for info in &infos {
            for (callee, _) in &info.calls {
                c[*callee] = true;
            }
        }
        c
    };
    for root in (0..infos.len()).filter(|s| !called[*s] && bearing[*s]) {
        let bytes = scope_total(root, &infos, &bearing, &mut memo, &mut Vec::new())?;
        total = match total.checked_add(bytes) {
            Some(v) => v,
            None => return reject("alloc.unbounded_total"),
        };
    }
    // A bearing scope every one of whose callers is itself only reachable through
    // a cycle has no root: it is recursion with no entry from outside.
    if (0..infos.len()).any(|s| bearing[s] && !memo.contains_key(&s)) {
        return reject("alloc.recursive");
    }
    if !bounded && total > NATIVE_ALLOC_BUDGET {
        return reject("alloc.total_over_budget");
    }
    Ok(())
}

/// Walk one straight-line slice (constants are folded per slice, as admission
/// folds them) and the regions below it, not crossing into nested functions.
fn walk(
    prog: &Program<'_>,
    instrs: &[Instr],
    in_loop: bool,
    info: &mut ScopeAllocs,
    any_alloc: &mut bool,
    bounded: bool,
) -> Result<(), FrozenProfileRejection> {
    let mut consts = BTreeMap::new();
    for instr in instrs {
        fold_const(instr, &mut consts);
        match instr {
            Instr::FnDef { .. } => continue,
            Instr::Call { name, args, .. } => match frozen_native_intrinsic(name) {
                Some(row) => {
                    let Some(at) = row.size_arg else { continue };
                    *any_alloc = true;
                    let bytes = args
                        .get(at)
                        .and_then(|a| consts.get(a))
                        .and_then(|n| block_bytes(*n));
                    // In bounded mode count an allocation marker, not bytes:
                    // runtime owns the size/budget, while recursion/ambiguity
                    // remain static refusals. Unknown sizes still bear allocation.
                    let bytes = if bounded {
                        1
                    } else {
                        bytes.ok_or(FrozenProfileRejection {
                            construct: "alloc.unbounded_total",
                        })?
                    };
                    info.alloc_in_loop |= in_loop;
                    info.direct = match info.direct.checked_add(bytes) {
                        Some(v) => v,
                        None => return reject("alloc.unbounded_total"),
                    };
                }
                None => {
                    if let Some(c) = prog.callee(name) {
                        info.calls.push((c, in_loop));
                    } else if prog.is_ambiguous(name) {
                        info.calls_ambiguous = true;
                    }
                }
            },
            _ => {}
        }
        #[cfg(feature = "std-surface")]
        {
            let loops = matches!(instr, Instr::While { .. });
            for nested in crate::ir::instr_bodies(instr) {
                walk(prog, nested, in_loop || loops, info, any_alloc, bounded)?;
            }
        }
    }
    Ok(())
}

/// Scopes that can allocate: directly, or through any callee that can.
fn allocating_scopes(infos: &[ScopeAllocs]) -> Vec<bool> {
    let mut bearing: Vec<bool> = infos.iter().map(|i| i.direct > 0).collect();
    loop {
        let mut changed = false;
        for (s, info) in infos.iter().enumerate() {
            if !bearing[s] && info.calls.iter().any(|(c, _)| bearing[*c]) {
                bearing[s] = true;
                changed = true;
            }
        }
        if !changed {
            return bearing;
        }
    }
}

fn scope_total(
    s: usize,
    infos: &[ScopeAllocs],
    bearing: &[bool],
    memo: &mut BTreeMap<usize, u64>,
    path: &mut Vec<usize>,
) -> Result<u64, FrozenProfileRejection> {
    if let Some(v) = memo.get(&s) {
        return Ok(*v);
    }
    if path.contains(&s) {
        reject("alloc.recursive")?;
    }
    path.push(s);
    let mut total = infos[s].direct;
    for (c, _) in &infos[s].calls {
        if !bearing[*c] {
            continue;
        }
        let sub = scope_total(*c, infos, bearing, memo, path)?;
        total = match total.checked_add(sub) {
            Some(v) => v,
            None => return reject("alloc.unbounded_total").map(|()| 0),
        };
    }
    path.pop();
    memo.insert(s, total);
    Ok(total)
}
