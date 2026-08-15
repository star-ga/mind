// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! #318 regression: F2-adjacent branch-local-`let` shadow leak in the IR merge
//! builder. When one branch writes an outer name and the OTHER branch only
//! *shadows* it with a branch-local `let`, the merge phi's edge for the
//! non-writing side must carry the PRE-IF value, not the block-scoped shadow.
//! Net-verified (multi-CLI audit 2026-08-15, Fable+qwen): the tree-evaluator is
//! correct, but pre-fix `lower.rs` lowered the then-edge to the shadow const —
//! a silent miscompile on the MLIR/IR path (and, separately, native codegen).

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;
use std::collections::HashMap;

fn collect_consts(instrs: &[Instr], out: &mut HashMap<usize, i64>) {
    for i in instrs {
        match i {
            Instr::ConstI64(vid, v) => {
                out.insert(vid.0, *v);
            }
            Instr::FnDef { body, .. } => collect_consts(body, out),
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                collect_consts(cond_instrs, out);
                collect_consts(then_instrs, out);
                collect_consts(else_instrs, out);
            }
            _ => {}
        }
    }
}

/// Find the `then_val` ValueId of the `"x"` merge in the first `Instr::If`
/// inside the first `FnDef`.
fn x_merge_then_val(instrs: &[Instr]) -> Option<usize> {
    for i in instrs {
        if let Instr::FnDef { body, .. } = i {
            for bi in body {
                if let Instr::If {
                    branch_bindings,
                    merges,
                    ..
                } = bi
                {
                    if let Some((_, mid)) = branch_bindings.iter().find(|(n, _)| n == "x") {
                        if let Some((_, then_val, _)) = merges.iter().find(|(m, _, _)| m == mid) {
                            return Some(then_val.0);
                        }
                    }
                }
            }
        }
    }
    None
}

#[test]
fn merge_then_edge_uses_preif_value_not_branch_shadow() {
    // else writes outer x=5; then only SHADOWS x with a branch-local `let x=99`.
    // For c==1 (then taken, outer x untouched) the correct value of x after the
    // if is the pre-if 1. The x-merge then-edge must therefore carry 1, not 99.
    let src = r#"
fn f(c: i64) -> i64 {
    let x: i64 = 1
    if c == 1 {
        let x: i64 = 99
        x
    } else {
        x = 5
    }
    return x
}
"#;
    let ir = lower_to_ir(&parser::parse(src).expect("parse"));
    let mut consts = HashMap::new();
    collect_consts(&ir.instrs, &mut consts);
    let then_val = x_merge_then_val(&ir.instrs).expect("x-merge then-edge exists");
    let v = consts.get(&then_val).copied();
    assert_eq!(
        v,
        Some(1),
        "#318: x-merge then-edge must carry the pre-if x=1, not the branch-local \
         shadow 99 — got {v:?} (then_val=%{then_val})"
    );
}
