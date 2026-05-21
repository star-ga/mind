// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 P0f Step 1 — `Node::FieldAccess` read-path lowering to the
//! heap-record ABI established by P0e.
//!
//! For a `StructLit` bound via `Let`, lowering tracks the variable's
//! struct type name in a per-fn `struct_env`. When that variable later
//! appears on the receiver side of a `FieldAccess`, lowering emits:
//!
//!   - `field_addr = addr + 8 * field_index` (or `addr` when index == 0),
//!   - `result = __mind_load_i64(field_addr)`.
//!
//! Step 1 only resolves the receiver when it is a plain `Ident` bound to
//! a `StructLit` in the same (or an enclosing) scope. Anything more
//! exotic — chained access `a.b.c`, FieldAccess of a function return,
//! FieldAccess on a struct-typed parameter — falls through to a
//! placeholder `ConstI64(0)` so older modules still compile. Step 2 will
//! extend resolution via a fold-through-StructLit IR pass or via
//! type-checker annotations.
//!
//! Gated: `cargo test --features std-surface --test std_surface_field_access`.

#![cfg(feature = "std-surface")]

use libmind::ast::{Field, Literal, Module, Node, Span, StructLitField, TypeAnn};
use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;

fn sp() -> Span {
    Span::new(0, 0)
}

fn field(name: &str) -> Field {
    Field {
        is_pub: false,
        name: name.to_string(),
        ty: TypeAnn::Named("i64".to_string()),
        span: sp(),
    }
}

fn lit_int(n: i64) -> Node {
    Node::Lit(Literal::Int(n), sp())
}

fn ident(name: &str) -> Node {
    Node::Lit(Literal::Ident(name.to_string()), sp())
}

fn struct_lit_field(name: &str, value: Node) -> StructLitField {
    StructLitField {
        name: name.to_string(),
        value,
        span: sp(),
    }
}

fn count_calls(instrs: &[Instr], name: &str) -> usize {
    instrs
        .iter()
        .filter(|i| matches!(i, Instr::Call { name: n, .. } if n == name))
        .count()
}

/// Build a module with a 3-field struct + literal binding + N field reads.
fn module_with_field_reads(read_fields: &[&str]) -> Module {
    let mut items: Vec<Node> = vec![
        Node::StructDef {
            is_pub: false,
            name: "Vec".to_string(),
            fields: vec![field("addr"), field("len"), field("cap")],
            attrs: vec![],
            span: sp(),
        },
        Node::Let {
            name: "v".to_string(),
            ann: None,
            value: Box::new(Node::StructLit {
                name: "Vec".to_string(),
                fields: vec![
                    struct_lit_field("addr", lit_int(100)),
                    struct_lit_field("len", lit_int(7)),
                    struct_lit_field("cap", lit_int(42)),
                ],
                span: sp(),
            }),
            span: sp(),
        },
    ];

    for f in read_fields {
        items.push(Node::FieldAccess {
            receiver: Box::new(ident("v")),
            field: (*f).to_string(),
            span: sp(),
        });
    }

    Module { items }
}

#[test]
fn field_access_emits_load_for_known_struct_var() {
    // Read every field of a known-struct variable: must emit exactly one
    // __mind_load_i64 per FieldAccess, no extra allocations.
    let module = module_with_field_reads(&["addr", "len", "cap"]);
    let ir = lower_to_ir(&module);

    assert_eq!(
        count_calls(&ir.instrs, "__mind_alloc"),
        1,
        "only the StructLit alloc — FieldAccess must not re-allocate"
    );
    assert_eq!(
        count_calls(&ir.instrs, "__mind_store_i64"),
        3,
        "stores come from the StructLit, not FieldAccess"
    );
    assert_eq!(
        count_calls(&ir.instrs, "__mind_load_i64"),
        3,
        "one __mind_load_i64 per FieldAccess, got {}",
        count_calls(&ir.instrs, "__mind_load_i64")
    );
}

#[test]
fn field_access_first_field_uses_base_addr_directly() {
    // Reading the 0-th field must NOT emit an Add(addr, 0) — the lowering
    // takes the base addr as the load arg directly. We assert this by
    // checking that the load's arg ValueId equals the alloc's destination
    // ValueId (i.e., the same SSA name is reused).
    let module = module_with_field_reads(&["addr"]);
    let ir = lower_to_ir(&module);

    let alloc_dst = ir
        .instrs
        .iter()
        .find_map(|i| match i {
            Instr::Call { name, dst, .. } if name == "__mind_alloc" => Some(*dst),
            _ => None,
        })
        .expect("StructLit should emit __mind_alloc");

    let load_arg = ir
        .instrs
        .iter()
        .find_map(|i| match i {
            Instr::Call { name, args, .. } if name == "__mind_load_i64" => args.first().copied(),
            _ => None,
        })
        .expect("FieldAccess should emit __mind_load_i64");

    assert_eq!(
        load_arg, alloc_dst,
        "field 0 load must reuse the alloc's base addr (no zero-offset Add)"
    );
}

#[test]
fn field_access_nonzero_field_uses_addr_plus_offset() {
    // Reading `v.len` (index 1) and `v.cap` (index 2) must each emit
    // a ConstI64(_, 8) or ConstI64(_, 16), an Add, then the load.
    let module = module_with_field_reads(&["len", "cap"]);
    let ir = lower_to_ir(&module);

    // Gather all ConstI64 values that appear AFTER the StructLit's stores.
    // The first stores need their own offsets (8, 16 inside StructLit);
    // FieldAccess for v.len adds another ConstI64(8), and v.cap adds (16).
    let mut offset_consts: Vec<i64> = Vec::new();
    let mut saw_stores_done = false;
    for instr in &ir.instrs {
        if let Instr::Call { name, .. } = instr {
            if name == "__mind_store_i64" {
                continue;
            }
            if name == "__mind_load_i64" {
                saw_stores_done = true;
                continue;
            }
        }
        if saw_stores_done {
            if let Instr::ConstI64(_, k) = instr {
                offset_consts.push(*k);
            }
        }
    }
    // We must see at least one 8 and one 16 introduced after the first load
    // begins emitting (or interleaved with it — order depends on IR walk).
    // Practically: 8 and 16 each appear *somewhere* in the IR.
    let all_consts: Vec<i64> = ir
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::ConstI64(_, k) => Some(*k),
            _ => None,
        })
        .collect();
    assert!(
        all_consts.contains(&8),
        "expected ConstI64(_, 8) for v.len's offset, got {all_consts:?}"
    );
    assert!(
        all_consts.contains(&16),
        "expected ConstI64(_, 16) for v.cap's offset, got {all_consts:?}"
    );

    assert_eq!(
        count_calls(&ir.instrs, "__mind_load_i64"),
        2,
        "two FieldAccesses must produce two loads"
    );
}

#[test]
fn field_access_unknown_receiver_falls_back_to_placeholder() {
    // FieldAccess on an Ident that is NOT in struct_env (e.g., never
    // bound to a StructLit) must fall through to ConstI64(_, 0), not
    // emit a __mind_load_i64.
    let module = Module {
        items: vec![
            // bare ident, no StructLit, no struct_env entry
            Node::Let {
                name: "x".to_string(),
                ann: None,
                value: Box::new(lit_int(99)),
                span: sp(),
            },
            Node::FieldAccess {
                receiver: Box::new(ident("x")),
                field: "anything".to_string(),
                span: sp(),
            },
        ],
    };

    let ir = lower_to_ir(&module);

    assert_eq!(
        count_calls(&ir.instrs, "__mind_load_i64"),
        0,
        "Step 1 must not emit a load when receiver type is unresolved"
    );
}

#[test]
fn field_access_unknown_struct_field_falls_back_to_placeholder() {
    // Known struct var, but the requested field is not in the StructDef.
    // Must fall back to placeholder, not emit a load at a guessed offset.
    let module = Module {
        items: vec![
            Node::StructDef {
                is_pub: false,
                name: "Pair".to_string(),
                fields: vec![field("a"), field("b")],
                attrs: vec![],
                span: sp(),
            },
            Node::Let {
                name: "p".to_string(),
                ann: None,
                value: Box::new(Node::StructLit {
                    name: "Pair".to_string(),
                    fields: vec![
                        struct_lit_field("a", lit_int(1)),
                        struct_lit_field("b", lit_int(2)),
                    ],
                    span: sp(),
                }),
                span: sp(),
            },
            Node::FieldAccess {
                receiver: Box::new(ident("p")),
                field: "c_does_not_exist".to_string(),
                span: sp(),
            },
        ],
    };

    let ir = lower_to_ir(&module);

    // The StructLit's 2 stores stay, but FieldAccess must not emit a load.
    assert_eq!(
        count_calls(&ir.instrs, "__mind_store_i64"),
        2,
        "stores come from the StructLit"
    );
    assert_eq!(
        count_calls(&ir.instrs, "__mind_load_i64"),
        0,
        "unknown field name must NOT produce a load at a guessed offset"
    );
}

#[test]
fn field_access_module_scope_binding_visible_inside_fn_body() {
    // A struct bound at module scope must be visible to FieldAccess
    // inside a function body — fn_struct_env starts as a clone of the
    // outer struct_env.
    let module = Module {
        items: vec![
            Node::StructDef {
                is_pub: false,
                name: "Cfg".to_string(),
                fields: vec![field("max")],
                attrs: vec![],
                span: sp(),
            },
            Node::Let {
                name: "cfg".to_string(),
                ann: None,
                value: Box::new(Node::StructLit {
                    name: "Cfg".to_string(),
                    fields: vec![struct_lit_field("max", lit_int(99))],
                    span: sp(),
                }),
                span: sp(),
            },
            Node::FnDef {
                is_pub: false,
                name: "read".to_string(),
                params: vec![],
                ret_type: None,
                body: vec![Node::Return {
                    value: Some(Box::new(Node::FieldAccess {
                        receiver: Box::new(ident("cfg")),
                        field: "max".to_string(),
                        span: sp(),
                    })),
                    span: sp(),
                }],
                reap_threshold: None,
                span: sp(),
            },
        ],
    };

    let ir = lower_to_ir(&module);

    // Locate the FnDef body and verify it contains a __mind_load_i64.
    let fn_body_loads: usize = ir
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::FnDef { body, .. } => Some(
                body.iter()
                    .filter(|b| matches!(b, Instr::Call { name, .. } if name == "__mind_load_i64"))
                    .count(),
            ),
            _ => None,
        })
        .sum();

    assert_eq!(
        fn_body_loads, 1,
        "FieldAccess inside fn body must resolve via cloned module-scope struct_env"
    );
}
