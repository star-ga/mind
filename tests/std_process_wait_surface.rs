// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/process.mind` checked-wait surface: the functions that keep exit,
//! signal, core-dump, stopped and `waitpid`-error statuses distinct, and that
//! publish a failed `exec` or a failed `pipe`/`fork` setup on the child record.
//!
//! Kept out of `tests/std_surface_net_fs_process.rs`, which is at its line
//! budget in `tests/module_size_ratchet.rs`. That file still asserts the
//! original process surface (`spawn`, `wait`, `exit_code`, ...); this one
//! asserts the additions. Native behaviour is exercised by the ctypes
//! acceptance driver in `tests/std_process_wait/`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const PROCESS_MIND_SRC: &str = include_str!("../std/process.mind");

/// The public functions the checked-wait work added to `std/process.mind`.
const WAIT_SURFACE: [&str; 6] = [
    "exit_error",
    "exit_core",
    "exit_stopped",
    "wait_flags",
    "child_exec_errno",
    "child_spawn_errno",
];

fn defined_fns(ir: &libmind::ir::IRModule) -> Vec<&str> {
    ir.instrs
        .iter()
        .filter_map(|i| match i {
            Instr::FnDef { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn process_mind_lowers_the_checked_wait_surface() {
    let module = parser::parse(PROCESS_MIND_SRC).expect("std/process.mind must parse");
    let ir = lower_to_ir(&module).expect("lowering");
    let fns = defined_fns(&ir);
    // Positive control: the module lowered to real functions, including the
    // pre-existing surface, so a missing name below is a real absence.
    assert!(
        fns.contains(&"wait") && fns.contains(&"spawn_capture"),
        "std/process.mind lowered without its base surface: {fns:?}"
    );
    let missing: Vec<_> = WAIT_SURFACE
        .iter()
        .filter(|want| !fns.contains(want))
        .collect();
    assert!(
        missing.is_empty(),
        "std/process.mind is missing checked-wait functions {missing:?}"
    );
}
