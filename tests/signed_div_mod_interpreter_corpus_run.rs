// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The tree-walking INTERPRETER runs the signed i64 div/mod edge corpus and must
//! produce the same exit status the frozen-profile native ELF and the MLIR backend
//! are pinned to.
//!
//! This is the third reader of
//! `examples/mindc_mind/testdata/signed_div_mod_edge_corpus.tsv`:
//!   * examples/mindc_mind/ri_d1_frozen_profile_gate.py        -> native ELF
//!   * tests/signed_div_mod_cross_backend_parity_run.rs        -> MLIR backend
//!   * THIS FILE                                               -> tree-walking interpreter
//!
//! Before it existed the interpreter was held to the contract only by unit tests with
//! their own hand-written operand lists — so editing a corpus row turned the two
//! compiled legs red and left the interpreter green. Reading the same file is what
//! makes "the tiers cannot drift" a property of the code rather than of a comment.
//!
//! `main()` returns an i64; the compiled legs observe it as a process exit status,
//! which Linux truncates to the low byte. The interpreter hands back the full i64, so
//! it is compared as `value & 0xFF` — the exact observation the other two legs make.
//!
//! This evaluator is also the conformance suite's VALUE oracle
//! (`conformance::VALUE_ORACLE_ENGINE == AstEvaluator`), so this test is what holds
//! conformance's expected values to the compiled contract for div/mod.
//!
//! NOT covered here: `eval::eval_ir` (the `mindc <file>` preview). It evaluates a flat
//! instruction list and does not call `main`, so it cannot run these programs; its
//! integer arms are held to the same values by
//! `eval::ir_interp::tests::ir_oracle_integer_arms_match_apply_int_op_and_the_artifact`.

#![cfg(feature = "std-surface")]

use std::collections::HashMap;
use std::path::PathBuf;

use libmind::eval::{self, value::Value};
use libmind::parser;

/// Exact row count of the shared corpus. Pinned HERE as well as in the other two
/// readers, so a deleted row is a failure in every tier instead of a silent shrink.
const CORPUS_ROWS: usize = 13;

/// Native stack for each interpreted program. See the comment at the spawn site.
const INTERP_STACK_BYTES: usize = 512 * 1024 * 1024;

fn load_corpus() -> Vec<(String, String, i64)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/mindc_mind/testdata");
    let path = dir.join("signed_div_mod_edge_corpus.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read the shared corpus at {}: {e}", path.display()));
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = raw.split('\t').collect();
        assert_eq!(
            f.len(),
            4,
            "{}:{}: want 4 TAB fields",
            path.display(),
            i + 1
        );
        let expect: i64 = f[1].trim().parse().expect("expected exit");
        let src = match f[2].trim() {
            "src" => f[3].trim().to_string(),
            "file" => std::fs::read_to_string(dir.join(f[3].trim())).expect("fixture file"),
            other => panic!("{}:{}: unknown kind '{other}'", path.display(), i + 1),
        };
        out.push((f[0].trim().to_string(), src, expect));
    }
    out
}

#[test]
fn interpreter_matches_the_pinned_exit_status_on_the_shared_div_mod_corpus() {
    let corpus = load_corpus();
    assert_eq!(
        corpus.len(),
        CORPUS_ROWS,
        "shared corpus has {} rows, pinned at {CORPUS_ROWS} — a deleted row must fail, not shrink",
        corpus.len()
    );

    let mut failures = Vec::new();
    for (name, src, expect) in &corpus {
        // Call `main` from a module-level binding: the evaluator returns the value of
        // the last item, which is how the existing interpreter tests observe a call.
        let program = format!("{src}\nlet __exit_status: i64 = main()\n");
        if let Err(e) = parser::parse(&program) {
            failures.push(format!("{name}: parse failed ({} error(s))", e.len()));
            continue;
        }
        // Run on a thread with an explicit stack. The Tier-0 drivers recurse ~120 deep
        // (`stop_time`), and the tree-walking evaluator spends far more native stack
        // per MIND frame than compiled code does: on the default 2 MiB test-thread
        // stack a debug build aborts the whole test binary with a stack overflow.
        // That is a property of this HARNESS'S thread, not of the program — the
        // compiled legs run the same recursion as ordinary processes.
        let result = std::thread::Builder::new()
            .stack_size(INTERP_STACK_BYTES)
            .spawn({
                let program = program.clone();
                move || {
                    let module = parser::parse(&program).expect("parsed above");
                    let mut env = HashMap::new();
                    eval::eval_module_value_with_env(&module, &mut env, Some(&program))
                }
            })
            .expect("spawn interpreter thread")
            .join()
            .unwrap_or_else(|_| panic!("{name}: interpreter thread panicked"));
        match result {
            Ok(Value::Int(v)) if v & 0xFF == *expect => {}
            Ok(Value::Int(v)) => failures.push(format!(
                "{name}: interpreter main() = {v} (low byte {}), pinned exit {expect}",
                v & 0xFF
            )),
            Ok(other) => failures.push(format!("{name}: expected an Int, got {other:?}")),
            Err(e) => failures.push(format!(
                "{name}: interpreter ERROR {e:?} where the compiled artifact returns exit {expect}"
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "interpreter disagrees with the pinned corpus on {}/{} rows:\n  - {}",
        failures.len(),
        corpus.len(),
        failures.join("\n  - ")
    );
}
