// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Artifact-path isolation contract for the integration-test harness.
//!
//! # The defect this gate defends against
//!
//! A test that compiles a workload to a FIXED path on the shared temp root
//! (`std::env::temp_dir().join("mind_x.so")`) is not private to its own run.
//! Two test processes on one box — two agents, two CI jobs on one runner, a
//! `cargo test` beside a preflight — pick the SAME path, and one `mindc
//! --emit-shared` truncates the artifact the other has just handed to
//! `dlopen`. The failure surfaces as a compile or load error inside the wedge
//! gate, i.e. it reads as a compiler regression, and it does not reproduce.
//! The path is also world-writable and predictable, so another user on the box
//! can pre-create it as a symlink and redirect the write.
//!
//! # The rule
//!
//! `std::env::temp_dir()` has exactly ONE sanctioned caller in `tests/`:
//! [`common::scratch_dir`], which appends the target name and the pid. Every
//! other call site is a shared-root path, and every one of them is recorded
//! below with its exact count.
//!
//! The backlog is a SET WITH COUNTS, not a ceiling, and it is checked in BOTH
//! directions:
//!
//! * a file with sites that is not on the list is RED — a new shared path
//!   cannot be added while an old one is routed away, which is the hole a
//!   total-count ratchet leaves open;
//! * a listed file whose measured count dropped is RED — routing a file
//!   REQUIRES deleting (or lowering) its row, so the list can never overstate
//!   the debt;
//! * a listed file that is gone or clean is RED for the same reason.
//!
//! This is what makes the count in `common::scratch_dir`'s `deferred:` marker
//! a measurement rather than a sentence: the marker names this gate, and this
//! gate reads the tree.
//!
//! This file is deliberately NOT feature-gated and NOT unix-gated: the scan is
//! pure source reading, so the rule is enforced on every row of the CI matrix.

use std::path::{Path, PathBuf};

/// The shared-temp-root call, assembled from two pieces so the scanner cannot
/// match its OWN source. An exemption list would be a hole that erodes with
/// every entry; a marker unable to name itself has no hole to erode. (Same
/// construction as `harness_portability.rs`'s shell-stub marker.)
const TEMP_ROOT_MARKER: &str = concat!("temp", "_dir()");

/// The ONE sanctioned caller: `common::scratch_dir`, which derives a
/// per-target, per-process directory. Its count is pinned so the owner cannot
/// quietly grow a second, unscoped call.
const SCRATCH_OWNER: (&str, usize) = ("common/mod.rs", 1);

/// Files that still build artifact paths directly on the shared temp root.
///
/// deferred: routing all of these in one change would touch 125 test files and
/// could not be reviewed against the wedge gates it must not disturb. Upgrade
/// path: route a file through `common::scratch_dir` as it is NEXT TOUCHED for
/// any other reason, and delete its row here — a stale row fails this gate, so
/// the list cannot drift away from the tree.
const SHARED_TEMP_BACKLOG: &[(&str, usize)] = &[
    ("aggshape_reject.rs", 1),
    ("alias_miscompile_run.rs", 1),
    ("array_ctor_push_get_run.rs", 1),
    ("array_load_bounds_and_dtype.rs", 2),
    ("array_oob_trap_run.rs", 2),
    ("array_store_run.rs", 1),
    ("array_surface_run.rs", 1),
    ("array_u64_element_shift_run.rs", 1),
    ("bare_variant_ambiguity_run.rs", 1),
    ("bare_variant_ctor_run.rs", 1),
    ("bitwise_no_panic_any_feature.rs", 1),
    ("bool_literal_value_run.rs", 1),
    ("bug6_tuple_destructure_u64_run.rs", 1),
    ("bug_f4_closure_shadow_run.rs", 1),
    ("bytes_buffer_run.rs", 1),
    ("bytes_fixed_into_vec_run.rs", 1),
    ("bytes_zero_run.rs", 1),
    ("char_literal_run.rs", 1),
    ("closure_capture_reject.rs", 1),
    ("closure_i64_capture.rs", 2),
    ("collection_ctor_run.rs", 1),
    ("collection_mutation_expr_position_run.rs", 2),
    ("compound_assign.rs", 1),
    ("cond_truthiness.rs", 1),
    ("const_array_run.rs", 1),
    ("const_f64_array_run.rs", 1),
    ("continue_in_match_arm_run.rs", 1),
    ("cross_module_cdylib_compose.rs", 1),
    ("cross_module_enum_run.rs", 1),
    ("determinism_veto_control.rs", 1),
    ("digit_separator_run.rs", 1),
    ("dot_enum_variant_run.rs", 1),
    ("enum_match_collision_run.rs", 1),
    ("enum_match_run.rs", 1),
    ("enum_struct_variant_run.rs", 1),
    ("extern_narrow_ret_run.rs", 1),
    ("f3_bare_enum_collision_run.rs", 1),
    ("f64_abi_negative_control.rs", 1),
    ("f64_call_arg_run.rs", 1),
    ("f64_literal_envelope.rs", 1),
    ("f64_loop_run.rs", 1),
    ("fail_closed_capability_skip_stub_exec.rs", 1),
    ("fail_closed_cli_run.rs", 1),
    ("fn_value_call_reject.rs", 1),
    ("for_continue_advances_run.rs", 1),
    ("for_each_run.rs", 1),
    ("for_hygiene_run.rs", 1),
    ("g2_differential_mlir.rs", 3),
    ("int_determinism.rs", 1),
    ("int_suffix_literal.rs", 1),
    ("invariant_block_run.rs", 1),
    ("invariant_check_run.rs", 1),
    ("loop_run.rs", 1),
    ("manifest_pin_enforcement.rs", 1),
    ("map_get_inference_run.rs", 1),
    ("map_runtime_run.rs", 1),
    ("map_surface_run.rs", 1),
    ("match_arm_stmt_run.rs", 1),
    ("match_scrutinee_once.rs", 1),
    ("merge_kind_order_symmetry.rs", 1),
    ("mic3_cli_emit.rs", 1),
    ("mindc_build_phase_a.rs", 1),
    ("mindc_inspect.rs", 2),
    ("mindfuzz_cross_substrate.rs", 1),
    ("module_const_run.rs", 1),
    ("module_decl_run.rs", 1),
    ("module_enum_match_run.rs", 1),
    ("module_non_fn_call_reject.rs", 1),
    ("multimodule_determinism_run.rs", 3),
    ("narrow_call_abi.rs", 1),
    ("narrow_local_mask_run.rs", 2),
    ("narrow_locals_leak_run.rs", 2),
    ("narrow_reassign_mask_run.rs", 2),
    ("narrow_reassign_run.rs", 1),
    ("narrow_sig_abi_run.rs", 1),
    ("narrow_signedness_batch2_run.rs", 1),
    ("narrow_signedness_batch_run.rs", 1),
    ("narrow_tuple_pr216_run.rs", 1),
    ("narrow_unsigned_div_zero_run.rs", 2),
    ("nested_block_surface_run.rs", 1),
    ("nested_collection_run.rs", 1),
    ("nested_mut_thread_run.rs", 2),
    ("non_final_catch_all_match_run.rs", 1),
    ("pattern_guard_run.rs", 1),
    ("result_option_prelude_run.rs", 1),
    ("return_cond_type_reject.rs", 2),
    ("return_flow_tree_eval_run.rs", 1),
    ("scalar_cast_call_run.rs", 1),
    ("scalar_cast_unsigned_narrow_run.rs", 1),
    ("set_surface_run.rs", 1),
    ("statement_mutation_run.rs", 1),
    ("std_import_standalone_run.rs", 1),
    ("std_surface_cdylib_link.rs", 2),
    ("std_surface_i32_intrinsics.rs", 1),
    ("std_surface_io_canon.rs", 1),
    ("std_surface_phase_d_env_override.rs", 1),
    ("std_surface_promotion_compose.rs", 1),
    ("std_surface_self_emit_shared.rs", 1),
    ("string_escape_decode_run.rs", 1),
    ("string_from_bytes_run.rs", 1),
    ("string_runtime_shim_run.rs", 1),
    ("struct_array_field_run.rs", 1),
    ("struct_field_collection_run.rs", 1),
    ("struct_field_in_loop_run.rs", 1),
    ("struct_narrow_field.rs", 1),
    ("substrate_nonentry_import_link.rs", 1),
    ("tensor_param_2d_run.rs", 1),
    ("tensor_param_fail_loud_run.rs", 1),
    ("trait_static_dispatch_run.rs", 2),
    ("try_operator_run.rs", 1),
    ("tuple_destructure_run.rs", 1),
    ("turboquant_kernel_run.rs", 1),
    ("type_struct_run.rs", 1),
    ("typed_literal_match_pattern_run.rs", 1),
    ("typo_reject.rs", 1),
    ("u64_cast_signed_compare_run.rs", 1),
    ("u64_tag_survival_run.rs", 1),
    ("value_if_comparison.rs", 1),
    ("value_if_f64_let.rs", 1),
    ("verify_cli.rs", 1),
    ("verify_ssa.rs", 1),
];

/// Count the shared-root call sites in one source text.
///
/// Whole-line comments are excluded so a doc comment describing the rule (this
/// file, and `common::scratch_dir`'s own docs) is not counted as a violation.
/// Code is counted wherever it appears, so a site cannot hide behind a trailing
/// comment on the same line.
fn site_count(text: &str) -> usize {
    text.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .map(|l| l.matches(TEMP_ROOT_MARKER).count())
        .sum()
}

/// Every `.rs` under `tests/`, as (path relative to `tests/`, source text),
/// sorted. Separators are normalised to `/` so the recorded rows read the same
/// on every host in the CI matrix.
fn test_sources() -> Vec<(String, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).expect("read tests dir") {
            let p = e.expect("dir entry").path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut paths = Vec::new();
    walk(&root, &mut paths);
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let rel = p
                .strip_prefix(&root)
                .expect("path under tests/")
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(&p).expect("read test source");
            (rel, text)
        })
        .collect()
}

#[test]
fn shared_temp_root_sites_match_the_recorded_backlog() {
    let sources = test_sources();
    assert!(
        sources.len() > 100,
        "the scan found only {} test sources — the walk is broken and this \
         gate would pass vacuously",
        sources.len()
    );

    let mut measured: Vec<(String, usize)> = sources
        .iter()
        .map(|(rel, text)| (rel.clone(), site_count(text)))
        .filter(|(_, n)| *n > 0)
        .collect();
    measured.sort();

    let mut expected: Vec<(String, usize)> = SHARED_TEMP_BACKLOG
        .iter()
        .map(|(f, n)| ((*f).to_string(), *n))
        .chain(std::iter::once((
            SCRATCH_OWNER.0.to_string(),
            SCRATCH_OWNER.1,
        )))
        .collect();
    expected.sort();

    let mut unrecorded = Vec::new();
    let mut drifted = Vec::new();
    for (rel, n) in &measured {
        match expected.iter().find(|(f, _)| f == rel) {
            None => unrecorded.push(format!("  {rel}: {n} site(s)")),
            Some((_, want)) if want != n => {
                drifted.push(format!("  {rel}: recorded {want}, measured {n}"))
            }
            Some(_) => {}
        }
    }
    let mut cleaned = Vec::new();
    for (rel, want) in &expected {
        if !measured.iter().any(|(f, _)| f == rel) {
            cleaned.push(format!("  {rel}: recorded {want}, measured 0"));
        }
    }

    assert!(
        unrecorded.is_empty(),
        "these test sources build artifact paths on the SHARED temp root and \
         are not on the recorded backlog:\n{}\n\nA fixed shared path is not \
         private to one run: a concurrent test process compiling the same name \
         truncates the artifact this one is about to load. Route it through \
         `crate::common::scratch_dir(\"<target>\")`, keeping the file name.",
        unrecorded.join("\n")
    );
    assert!(
        drifted.is_empty(),
        "the recorded shared-temp-root counts no longer match the tree:\n{}\n\n\
         Routing a site REQUIRES updating its row (or deleting it) — the \
         backlog states the debt exactly, never approximately.",
        drifted.join("\n")
    );
    assert!(
        cleaned.is_empty(),
        "these rows are stale — the files are routed or gone, so their rows \
         must be DELETED:\n{}\n\nA backlog that outlives the debt is prose, \
         not a measurement.",
        cleaned.join("\n")
    );
}

#[test]
fn a_routed_target_does_not_also_write_to_the_shared_root() {
    // One file, one policy. A target that adopted `scratch_dir` but kept a
    // shared path for "just one" artifact still collides on that artifact, and
    // the mixed shape is what makes a half-routing look finished.
    let offenders: Vec<String> = test_sources()
        .into_iter()
        .filter(|(rel, _)| rel != SCRATCH_OWNER.0)
        .filter(|(_, text)| text.contains("scratch_dir("))
        .map(|(rel, text)| (rel, site_count(&text)))
        .filter(|(_, n)| *n > 0)
        .map(|(rel, n)| format!("  {rel}: {n} shared-root site(s) beside scratch_dir"))
        .collect();
    assert!(
        offenders.is_empty(),
        "these targets mix both artifact-path policies:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn the_detector_can_fire() {
    // A detector that matches nothing is the ran=0 defect in its active form,
    // so the counter is exercised on synthetic text built from the marker
    // itself (never spelled literally here, or the scan would flag this file).
    let m = TEMP_ROOT_MARKER;
    assert_eq!(
        site_count(&format!("let p = std::env::{m}.join(\"a.so\");")),
        1
    );
    assert_eq!(
        site_count(&format!("let a = {m}; let b = {m};")),
        2,
        "two sites on one line must both count"
    );
    assert_eq!(
        site_count(&format!("    // let p = std::env::{m};")),
        0,
        "a whole-line comment is documentation, not a call site"
    );
    assert_eq!(
        site_count(&format!("let p = {m}; // shared root")),
        1,
        "a trailing comment must not hide the call site on the same line"
    );
    assert_eq!(site_count("let d = crate::common::scratch_dir(\"t\");"), 0);
}
