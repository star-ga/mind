// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Regression: mic@3 emit must intern the `live` variable names carried by
//! `Instr::Break` / `Instr::Continue`.
//!
//! ## The bug (silent miscompile)
//!
//! `emit_instr` serializes `Break { live }` / `Continue { live }` via
//! `encode_named_vids`, which writes a **string-table index** for each `live`
//! variable name (`st.get(name)`). The pre-pass `collect_instr_strings` that
//! builds that table, however, had no arm for `Break` / `Continue` — those
//! variants fell into `_ => {}` and their names were never interned.
//!
//! Consequence for a name that appears ONLY inside a `Break`/`Continue` live
//! snapshot (e.g. a variable that is in scope at the break point but is not a
//! loop-carried `live_var` of the enclosing `while`):
//!   * debug build  → `StringTable::get`'s `debug_assert!` panics;
//!   * release build → `st.get` returns the fallback index **0**, so emit
//!     writes index 0 and `parse_mic3` decodes the name back as `strings[0]`
//!     — a DIFFERENT, WRONG string. The artifact round-trips to a corrupted
//!     program (wrong loop-control live binding) with no error.
//!
//! This test reconstructs that exact shape directly at the IR layer and proves
//! emit→parse no longer corrupts the name.

#![cfg(feature = "std-surface")]

use libmind::ir::compact::{emit_mic3, parse_mic3};
use libmind::ir::{IRModule, Instr};

/// Build a module whose only occurrence of the name `"brk_only"` is inside a
/// `Break`'s `live` snapshot. `strings[0]` is forced to be a different name
/// (`"sentinel0"`) so the index-0 fallback is observable as a wrong value, not
/// an accidental match.
fn module_with_break_only_name() -> IRModule {
    let mut m = IRModule::new();
    let p = m.fresh();
    let brk_val = m.fresh();

    // An exported function. Its name + a param establish other string-table
    // entries, and (critically) make `strings[0]` deterministically a name that
    // is NOT "brk_only", so the fallback-to-0 corruption is detectable.
    m.exports.insert("sentinel0".to_string());
    m.instrs.push(Instr::FnDef {
        name: "sentinel0".to_string(),
        params: vec![("p".to_string(), p)],
        ret_id: Some(brk_val),
        body: vec![
            Instr::Param {
                dst: p,
                name: "p".to_string(),
                index: 0,
            },
            Instr::ConstI64(brk_val, 7),
            // The `break` carries a live binding for a variable whose name
            // appears NOWHERE else in the module.
            Instr::Break {
                live: vec![("brk_only".to_string(), brk_val)],
            },
            Instr::Return {
                value: Some(brk_val),
            },
        ],
        reap_threshold: None,
    });
    m
}

/// Extract the `live` name of the first `Break` found anywhere in the module.
fn first_break_live_name(m: &IRModule) -> Option<String> {
    fn walk(instrs: &[Instr]) -> Option<String> {
        for i in instrs {
            match i {
                Instr::Break { live } | Instr::Continue { live } => {
                    return live.first().map(|(n, _)| n.clone());
                }
                Instr::FnDef { body, .. } => {
                    if let Some(n) = walk(body) {
                        return Some(n);
                    }
                }
                _ => {}
            }
        }
        None
    }
    walk(&m.instrs)
}

#[test]
fn break_live_name_survives_mic3_round_trip() {
    let m = module_with_break_only_name();
    assert_eq!(
        first_break_live_name(&m).as_deref(),
        Some("brk_only"),
        "fixture sanity: the original module's break live name is 'brk_only'"
    );

    // emit → parse. Pre-fix this either panics (debug) or silently writes the
    // wrong string index (release); post-fix the name is interned and survives.
    let bytes = emit_mic3(&m);
    let parsed = parse_mic3(&bytes).expect("mic@3 must re-parse");

    let decoded = first_break_live_name(&parsed);
    assert_eq!(
        decoded.as_deref(),
        Some("brk_only"),
        "Break live var name was corrupted by emit→parse (got {:?}); \
         collect_strings must intern Break/Continue live names",
        decoded
    );
}

#[test]
fn break_continue_module_is_mic3_fixed_point() {
    // The full fixed-point invariant `emit(parse(emit(m))) == emit(m)` must hold
    // for a module containing a break-only live name. Pre-fix the first emit
    // already encodes index 0, so even the fixed point is over a corrupted body.
    let m = module_with_break_only_name();
    let once = emit_mic3(&m);
    let twice = emit_mic3(&parse_mic3(&once).expect("re-parse"));
    assert_eq!(once, twice, "mic@3 must be a fixed point under emit→parse→emit");

    // And the round-tripped module must be byte-identical to a fresh build of
    // the same logical module — i.e. the name actually present, not strings[0].
    assert_ne!(
        emit_mic3(&{
            // A control module identical except the break carries strings[0]
            // ("sentinel0") instead of "brk_only". If the bug were present, the
            // buggy emission of `module_with_break_only_name` would equal THIS
            // control (both encode index 0). Post-fix they must differ.
            let mut c = IRModule::new();
            let p = c.fresh();
            let v = c.fresh();
            c.exports.insert("sentinel0".to_string());
            c.instrs.push(Instr::FnDef {
                name: "sentinel0".to_string(),
                params: vec![("p".to_string(), p)],
                ret_id: Some(v),
                body: vec![
                    Instr::Param {
                        dst: p,
                        name: "p".to_string(),
                        index: 0,
                    },
                    Instr::ConstI64(v, 7),
                    Instr::Break {
                        live: vec![("sentinel0".to_string(), v)],
                    },
                    Instr::Return { value: Some(v) },
                ],
                reap_threshold: None,
            });
            c
        }),
        once,
        "a break live name 'brk_only' must NOT serialize identically to one \
         carrying strings[0] ('sentinel0') — that is the index-0 corruption"
    );
}
