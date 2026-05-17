// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use libmind::ast::Node;
use libmind::eval::lower::lower_to_ir;
use libmind::ir::{canonicalize_module, Instr};
use libmind::parser;

// ---------------------------------------------------------------------------
// Parser: [reap_threshold(t)] attribute on fn
// ---------------------------------------------------------------------------

#[test]
fn parse_reap_threshold_on_fn() {
    let src = r#"
        [reap_threshold(0.5)]
        fn expert_a(x: i32) -> i32 { return x; }
    "#;
    let m = parser::parse(src).expect("parse failed");
    assert_eq!(m.items.len(), 1);
    match &m.items[0] {
        Node::FnDef {
            name,
            reap_threshold,
            ..
        } => {
            assert_eq!(name, "expert_a");
            assert!(
                reap_threshold.is_some(),
                "reap_threshold should be Some after parsing [reap_threshold(0.5)]"
            );
            let t = reap_threshold.unwrap();
            assert!((t - 0.5).abs() < 1e-9, "expected 0.5, got {t}");
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_reap_threshold_zero() {
    let src = r#"
        [reap_threshold(0.0)]
        fn expert_b(x: i32) -> i32 { return x; }
    "#;
    let m = parser::parse(src).expect("parse failed");
    match &m.items[0] {
        Node::FnDef { reap_threshold, .. } => {
            let t = reap_threshold.expect("should be Some(0.0)");
            assert!((t - 0.0).abs() < 1e-9);
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_reap_threshold_high_value() {
    let src = r#"
        [reap_threshold(0.9)]
        fn expert_c(x: i32) -> i32 { return x; }
    "#;
    let m = parser::parse(src).expect("parse failed");
    match &m.items[0] {
        Node::FnDef { reap_threshold, .. } => {
            let t = reap_threshold.expect("should be Some(0.9)");
            assert!((t - 0.9).abs() < 1e-9, "expected ~0.9, got {t}");
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_fn_without_reap_threshold() {
    // Functions without the attribute must have reap_threshold = None.
    let src = "fn plain(x: i32) -> i32 { return x; }";
    let m = parser::parse(src).expect("parse failed");
    match &m.items[0] {
        Node::FnDef { reap_threshold, .. } => {
            assert!(
                reap_threshold.is_none(),
                "plain fn should have no reap_threshold"
            );
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_reap_threshold_out_of_range_rejected() {
    // threshold >= 1.0 must be silently ignored (treated as absent).
    let src = r#"
        [reap_threshold(1.0)]
        fn expert_d(x: i32) -> i32 { return x; }
    "#;
    let m = parser::parse(src).expect("parse succeeded (attribute is syntactically valid)");
    match &m.items[0] {
        Node::FnDef { reap_threshold, .. } => {
            assert!(
                reap_threshold.is_none(),
                "threshold >=1.0 should be None, got {reap_threshold:?}"
            );
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// IR lowering: reap_threshold propagates to Instr::FnDef
// ---------------------------------------------------------------------------

#[test]
fn lower_reap_threshold_propagated_to_ir() {
    let src = r#"
        [reap_threshold(0.5)]
        fn expert_a(x: i32) -> i32 { return x; }
    "#;
    let m = parser::parse(src).expect("parse failed");
    let ir = lower_to_ir(&m);
    let fn_def = ir.instrs.iter().find_map(|instr| {
        if let Instr::FnDef {
            name,
            reap_threshold,
            ..
        } = instr
        {
            if name == "expert_a" {
                return Some(*reap_threshold);
            }
        }
        None
    });
    let threshold = fn_def.expect("expert_a FnDef not found in IR");
    assert!(
        threshold.is_some(),
        "reap_threshold should be Some in IR Instr::FnDef"
    );
    assert!((threshold.unwrap() - 0.5).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// DCE pass: dead experts are pruned
// ---------------------------------------------------------------------------

#[test]
fn dce_prunes_unreachable_expert() {
    // expert_a is declared with reap_threshold but never called.
    // After canonicalization its body should be replaced with a tombstone.
    let src = r#"
        [reap_threshold(0.5)]
        fn expert_a(x: i32) -> i32 { return x; }
    "#;
    let m = parser::parse(src).expect("parse failed");
    let mut ir = lower_to_ir(&m);
    canonicalize_module(&mut ir);

    let fn_def = ir.instrs.iter().find_map(|instr| {
        if let Instr::FnDef { name, body, .. } = instr {
            if name == "expert_a" {
                return Some(body.clone());
            }
        }
        None
    });
    let body = fn_def.expect("expert_a should still exist in IR (as a tombstone)");
    // Body must be a single ConstI64 tombstone after DCE.
    assert_eq!(
        body.len(),
        1,
        "dead expert body should be a single tombstone"
    );
    assert!(
        matches!(body[0], Instr::ConstI64(..)),
        "tombstone should be ConstI64, got {:?}",
        body[0]
    );
}

#[test]
fn dce_preserves_called_expert() {
    // expert_a is declared with reap_threshold AND is called — must survive.
    let src = r#"
        [reap_threshold(0.5)]
        fn expert_a(x: i32) -> i32 { return x; }

        fn router() -> i32 {
            return expert_a(1);
        }
    "#;
    let m = parser::parse(src).expect("parse failed");
    let mut ir = lower_to_ir(&m);
    canonicalize_module(&mut ir);

    let fn_def = ir.instrs.iter().find_map(|instr| {
        if let Instr::FnDef { name, body, .. } = instr {
            if name == "expert_a" {
                return Some(body.clone());
            }
        }
        None
    });
    let body = fn_def.expect("expert_a should be in IR");
    // Body should have more than one instruction (not just a tombstone).
    assert!(
        body.len() > 1 || !matches!(body.first(), Some(Instr::ConstI64(..))),
        "called expert_a must not be tombstoned, got {body:?}"
    );
}

#[test]
fn dce_no_op_without_reap_threshold() {
    // Without any [reap_threshold] attribute, the REAP pass must not run.
    // We verify by checking that a plain fn body is untouched by canonicalize.
    let src = r#"
        fn plain_expert(x: i32) -> i32 { return x; }
    "#;
    let m = parser::parse(src).expect("parse failed");
    let mut ir = lower_to_ir(&m);
    let body_before = ir.instrs.iter().find_map(|instr| {
        if let Instr::FnDef { name, body, .. } = instr {
            if name == "plain_expert" {
                return Some(body.clone());
            }
        }
        None
    });
    canonicalize_module(&mut ir);
    let body_after = ir.instrs.iter().find_map(|instr| {
        if let Instr::FnDef { name, body, .. } = instr {
            if name == "plain_expert" {
                return Some(body.clone());
            }
        }
        None
    });
    // Body before and after canonicalization should have the same length.
    assert_eq!(
        body_before.map(|b| b.len()),
        body_after.map(|b| b.len()),
        "plain fn body should not be modified by canonicalize when no reap_threshold present"
    );
}
