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

//! Formatter idempotence gate — Phase 2A acceptance test (Step 3 of PR #3).
//!
//! For every source file in scope, asserts:
//!   `format_source(format_source(src)) == format_source(src)`
//!
//! This is the hard contract: round-trip stability must hold at 100% with
//! zero skips.  A formatter that fails idempotence is structurally broken.
//!
//! Scope:
//!   - `std/*.mind` (5 files, canonical stdlib)
//!   - `examples/**/*.mind` (all discoverable .mind sources)
//!   - `tests/mindcraft/fmt/*.in.mind` (7 Phase-2A fixture inputs)
//!
//! All files that parse successfully must be idempotent.  Files that fail to
//! parse (e.g. because they use features the parser doesn't yet support) are
//! skipped with a note — parse failures are not idempotence failures.

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

fn default_cfg() -> MindcraftFormatConfig {
    MindcraftFormatConfig::default()
}

fn manifest_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Produce a compact line-by-line diff for failure messages.
fn diff_lines(a: &str, b: &str) -> String {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();
    let max = a_lines.len().max(b_lines.len());
    let mut out = String::new();
    for i in 0..max {
        let la = a_lines.get(i).copied().unwrap_or("<missing>");
        let lb = b_lines.get(i).copied().unwrap_or("<missing>");
        if la != lb {
            out.push_str(&format!("  line {}: pass1={la:?}\n", i + 1));
            out.push_str(&format!("  line {}:  pass2={lb:?}\n", i + 1));
        }
    }
    out
}

/// Assert idempotence for a single source string.
///
/// Returns `true` if the file was exercised (either passed or failed),
/// `false` if skipped (parse error on pass 1 — not a formatter bug).
fn check_idempotence(label: &str, src: &str, cfg: &MindcraftFormatConfig) -> bool {
    let pass1 = match format_source(src, cfg) {
        Ok(s) => s,
        Err(_) => {
            // Parse failures are expected for some examples using tensor/
            // autodiff syntax not in the current formatter scope.
            return false;
        }
    };
    let pass2 = format_source(&pass1, cfg)
        .unwrap_or_else(|e| panic!("idempotence: pass-2 parse failed for {label}: {e}"));
    assert_eq!(
        pass1,
        pass2,
        "idempotence violated for {label}:\n{}",
        diff_lines(&pass1, &pass2),
    );
    true
}

// ---------------------------------------------------------------------------
// std/*.mind — 6 files (toml requires std-surface for while-loop parsing)
// ---------------------------------------------------------------------------

#[test]
fn idempotence_stdlib() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let mut passed = 0usize;
    let mut skipped = 0usize;

    // The three files below use only if/let/expression syntax — no while loops —
    // and must parse under the default build. `string.mind`, `toml.mind` and
    // `map.mind` use `while` loops and are tested separately under std-surface
    // (idempotence_stdlib_string / idempotence_stdlib_toml / idempotence_stdlib_map).
    for name in &["vec", "io", "blas"] {
        let path = base.join(format!("{name}.mind"));
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if check_idempotence(name, &src, &cfg) {
            passed += 1;
        } else {
            skipped += 1;
        }
    }

    assert_eq!(
        skipped, 0,
        "unexpected parse failures in std/ ({skipped} file(s))"
    );
    assert_eq!(passed, 3, "expected 3 stdlib files, got {passed}");
}

/// `string.mind` uses `while` loops (the byte-compare bodies of
/// `string_eq` / `string_starts_with`), which require the `std-surface`
/// feature for the formatter's parser. Gated separately, like toml.
#[test]
#[cfg(feature = "std-surface")]
fn idempotence_stdlib_string() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let path = base.join("string.mind");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let exercised = check_idempotence("string", &src, &cfg);
    assert!(
        exercised,
        "string.mind failed to parse under std-surface — unexpected"
    );
}

/// `sha256.mind` uses `while` loops in compress_block and sha256, which require
/// the `std-surface` feature for the formatter's parser.
#[test]
#[cfg(feature = "std-surface")]
fn idempotence_stdlib_sha256() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let path = base.join("sha256.mind");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let exercised = check_idempotence("sha256", &src, &cfg);
    assert!(
        exercised,
        "sha256.mind failed to parse under std-surface — unexpected"
    );
}

/// `map.mind` uses `while` loops, which require the `std-surface` feature to be
/// recognised by the formatter's parser.  This test is separately gated.
#[test]
#[cfg(feature = "std-surface")]
fn idempotence_stdlib_map() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let path = base.join("map.mind");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let exercised = check_idempotence("map", &src, &cfg);
    assert!(
        exercised,
        "map.mind failed to parse under std-surface — unexpected"
    );
}

/// `toml.mind` uses `while` loops, which require the `std-surface` feature to
/// be recognised by the formatter's parser.  This test is separately gated.
#[test]
#[cfg(feature = "std-surface")]
fn idempotence_stdlib_toml() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let path = base.join("toml.mind");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let exercised = check_idempotence("toml", &src, &cfg);
    assert!(
        exercised,
        "toml.mind failed to parse under std-surface — unexpected"
    );
}

// ---------------------------------------------------------------------------
// examples/**/*.mind
// ---------------------------------------------------------------------------

/// Ratchet floor for `idempotence_examples`: the number of files under examples/
/// that round-trip through the formatter today. Raising it is free; lowering it is a
/// deliberate, explained change.
// Ratchet floor for `idempotence_examples`, measured in a CLEAN `git worktree`.
//
// THE FLOOR IS FEATURE-DEPENDENT and must be, because how many examples PARSE depends on
// which language surface is compiled in:
//
//   --features std-surface …    50 passed,  8 skipped, 58 total
//   --no-default-features       31 passed, 27 skipped, 58 total   (ci.yml runs this)
//
// The 27 extra skips without `std-surface` are not a regression: those files use
// std-surface constructs that a default build genuinely cannot parse.
//
// Two calibration mistakes are baked into this comment so they are not repeated:
//   1. The first value (52) was measured in a working tree carrying two UNTRACKED fuzzer
//      artifacts under examples/mindc_mind/mindfuzz_self_host_staged/, inflating the
//      corpus 58 -> 60. Unreachable from any fresh clone. Calibrate ONLY from a clean
//      worktree — `git worktree add --detach /tmp/x HEAD`.
//   2. The second value (50) was measured with the FULL feature set only, and would still
//      have redded the `--no-default-features` tier at 31. Calibrate EVERY configuration
//      CI builds, not the one that happens to be in your shell history.
#[cfg(feature = "std-surface")]
const EXAMPLES_IDEMPOTENCE_FLOOR: usize = 50;
#[cfg(not(feature = "std-surface"))]
const EXAMPLES_IDEMPOTENCE_FLOOR: usize = 25;

// Measured at 68d8e172. These are an allow-list, not a skip target: any entry
// may start passing, while a new skip or a changed cause fails at that fixture.
#[cfg(not(feature = "std-surface"))]
const BARE_E1042_REFUSALS: &[&str] = &[
    "examples/dottie_collapse.mind",
    "examples/emit_ir/main.mind",
    "examples/grammar_mask/main.mind",
    "examples/lexer/main.mind",
    "examples/mindc_mind/main.mind",
    // The codec mirror uses bitwise masking/shifts; the bare parser must
    // classify those operators as the feature-specific E1042 refusal.
    "examples/mind_mirror_v04/mic3_v04_prefix_mirror.mind",
    "examples/parser/main.mind",
    "examples/remizov_feynman.mind",
    "examples/typecheck/main.mind",
    "examples/mindc_mind/testdata/native_record_array/computed_i64_bitwise.mind",
    // Measured after formatting, with
    // `cargo test --no-default-features --locked --test fmt_idempotence`.
    // Each path is a bare-parser refusal (E1042 or legacy-parse). The
    // std-surface formatter accepts the file; the bare grammar does not.
    // A formatter-only skip is not listed here.
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_prog0093_minimal.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_prog0178_minimal.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rt_nested_carry_init_s4242_p0231.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/shadow_assign_before_prog0040.mind",
    "examples/mindc_mind/testdata/native_misc/a8_str_const.mind",
];

#[cfg(not(feature = "std-surface"))]
const BARE_LEGACY_PARSE_SKIPS: &[&str] = &[
    "examples/anthropobrot.mind",
    "examples/collatz.mind",
    "examples/columnar/structural_scan_json.mind",
    "examples/columnar/tiled_fold.mind",
    "examples/compliance/auditable_model.mind",
    "examples/cos_dottie.mind",
    "examples/detmath_kat/main.mind",
    "examples/fft_q16.mind",
    "examples/fft_signal.mind",
    "examples/galperin_pi.mind",
    "examples/halbach_q16/main.mind",
    "examples/halbach_q16.mind",
    "examples/lorenz_f64.mind",
    "examples/lorenz_q16.mind",
    "examples/mandelbrot.mind",
    "examples/mandelbrot_strict.mind",
    "examples/mindc_mind/testdata/rh_f64_aggregate_canary.mind",
    "examples/native/ci_kernel.mind",
    "examples/native/loop.mind",
    // The native bridge corpus includes a while loop.  `while` belongs to the
    // std-surface grammar, so the bare formatter must classify this fixture as
    // a measured legacy parse skip rather than silently treating it as an
    // unexpected corpus regression.
    "examples/mindc_mind/testdata/backend_native_bridge/while_sum.mind",
    "examples/mindc_mind/testdata/native_record_array/extent_overflow_refuse.mind",
    "examples/mindc_mind/testdata/native_record_array/record_param_loop.mind",
    "examples/mindc_mind/testdata/native_record_array/reference.mind",
    // These loop fixtures also require std-surface; the enabled-profile
    // formatter pass below must parse and format them successfully.
    "examples/mindc_mind/testdata/native_record_array/reference_u8.mind",
    "examples/mindc_mind/testdata/native_record_array/u8_direct_loop.mind",
    "examples/policy.mind",
    // The quant examples use `while` and the `__mind_*` intrinsics, which belong to
    // the std-surface grammar; the enabled-profile pass formats them.
    "examples/quant/american_binomial/main.mind",
    "examples/quant/barrier_analytic/main.mind",
    "examples/quant/black_scholes/main.mind",
    "examples/quant/bond_curve/main.mind",
    "examples/quant/crr_binomial/main.mind",
    "examples/quant/greeks_higher_order/main.mind",
    "examples/quant/implied_vol/main.mind",
    "examples/quant/implied_vol_surface/main.mind",
    "examples/quant/monte_carlo_gbm/main.mind",
    "examples/quant/portfolio_risk/main.mind",
    "examples/remizov_benchmark.mind",
    "examples/remizov_gpu.mind",
    "examples/remizov_inverse.mind",
    "examples/remizov_solver.mind",
    "examples/remizov_verify.mind",
    // Measured after formatting, with
    // `cargo test --no-default-features --locked --test fmt_idempotence`.
    // Each path is a bare-parser refusal (E1042 or legacy-parse). The
    // std-surface formatter accepts the file; the bare grammar does not.
    // A formatter-only skip is not listed here.
    "examples/mindc_mind/native_c_call_emit.mind",
    "examples/mindc_mind/native_c_call_emit_driver.mind",
    "examples/mindc_mind/native_c_call_plan.mind",
    "examples/mindc_mind/native_c_call_plan_driver.mind",
    "examples/mindc_mind/native_c_variadic_emit.mind",
    "examples/mindc_mind/native_c_variadic_emit_driver.mind",
    "examples/mindc_mind/native_c_variadic_plan.mind",
    "examples/mindc_mind/native_c_variadic_plan_driver.mind",
    "examples/mindc_mind/native_c_variadic_verify.mind",
    "examples/mindc_mind/native_cabi_driver.mind",
    "examples/mindc_mind/native_dynamic.mind",
    "examples/mindc_mind/native_dynamic_allocator.mind",
    "examples/mindc_mind/native_dynamic_allocator_driver.mind",
    "examples/mindc_mind/native_dynamic_fixture.mind",
    "examples/mindc_mind/native_dynamic_image.mind",
    "examples/mindc_mind/native_dynamic_image_fixture.mind",
    "examples/mindc_mind/native_dynamic_layout.mind",
    "examples/mindc_mind/native_dynamic_layout_driver.mind",
    "examples/mindc_mind/native_dynamic_mic_fixture.mind",
    "examples/mindc_mind/native_emission_context.mind",
    "examples/mindc_mind/native_libc_binding.mind",
    "examples/mindc_mind/native_libc_binding_fixture.mind",
    "examples/mindc_mind/native_libc_io.mind",
    "examples/mindc_mind/native_libc_io_fixture.mind",
    "examples/mindc_mind/native_libc_startup.mind",
    "examples/mindc_mind/native_libc_startup_fixture.mind",
    "examples/mindc_mind/native_scoped_source_driver.mind",
    "examples/mindc_mind/native_scoped_source_image.mind",
    "examples/mindc_mind/native_source_driver.mind",
    "examples/mindc_mind/native_source_image.mind",
    "examples/mindc_mind/source_note_driver.mind",
    "examples/mindc_mind/string_escape_driver.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/alias_break.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/branch_alias_value_if_then.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_copy_chain3.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_copy_vs_param_k1.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_copy_vs_param_noread.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_copy_vs_param_p11.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_4_3.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_5_3.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_5_4.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_6_3.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_6_4.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_7_3.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_7_4.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_8_3.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_nested_while_n_8_4.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/idcollide_triple_nested_while.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/merge_binding_if_then_while_read.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/postloop_read_after_two_loops.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_loop_in_then_assigns_outer_and_shadow.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_native_shadow.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_native_shadow_param.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_r20_min.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rand_230_52_m0.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rand_230_52_m0_main_only.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rt_break_flush_carried.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rt_carried_ident_assign.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rt_dtk_letcopy_gen1234_p144.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rt_dtk_v_32_169.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rt_let_alias_of_carried.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_rt_shadow_nested_carry_s1234_p0105.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_shadow_then_loop_outer_assigned_in_loop.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_v1_noshadow.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_v6_let_then_loop_only_outer.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_v7_fn_param.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_wit_call_after.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_wit_nested_if_in_loop_in_then.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_wit_nested_while_in_loop_in_then.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_wit_outer_counter.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_wit_then_seq_after_if.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/rv3_wit_two_outer_one_prewritten.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/shadow_assign_before_in_while.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress/shadow_assign_before_while_after.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_e7_outer_else_nonempty.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_e8_no_outer_else.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_h1_pad1.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_h2_pad2.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_pcand_cond_read.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_pcand_min.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_pcand_nested.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_r20_helper_only.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_rand_110_20_m0.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_rand_412_4_m2.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_rand_530_61_m0.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_rand_530_61_m0_main_only.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_reg_min.mind",
    "examples/mindc_mind/testdata/mindfuzz_regress_failclosed/rv3_s3_412_helper.mind",
    "examples/mindc_mind/testdata/native_alloc_bounded/afl_inc.mind",
    "examples/mindc_mind/testdata/native_alloc_bounded/big_increasing.mind",
    "examples/mindc_mind/testdata/native_alloc_bounded/grow8.mind",
    "examples/mindc_mind/testdata/native_alloc_bounded/interleaved_growth.mind",
    "examples/mindc_mind/testdata/native_alloc_bounded/oom_exit.mind",
    "examples/mindc_mind/testdata/native_alloc_bounded/r18_free_loop.mind",
    "examples/mindc_mind/testdata/native_alloc_bounded/vec_grow.mind",
    "examples/mindc_mind/testdata/native_alloc_bounded/vec_std.mind",
    "examples/mindc_mind/testdata/native_cf/a1_logic_in_cond.mind",
    "examples/mindc_mind/testdata/native_cf/a6_vif_in_loop.mind",
    "examples/mindc_mind/testdata/native_cf/a9_read_then_branch_assign.mind",
    "examples/mindc_mind/testdata/native_cf/a9_vif_branch_effects.mind",
    "examples/mindc_mind/testdata/native_cf/deep_nesting.mind",
    "examples/mindc_mind/testdata/native_cf/f64_loop_vif_return.mind",
    "examples/mindc_mind/testdata/native_cf/l1_return_in_loop.mind",
    "examples/mindc_mind/testdata/native_cf/l2_loop_if_branches.mind",
    "examples/mindc_mind/testdata/native_cf/l3_break_continue_params.mind",
    "examples/mindc_mind/testdata/native_cf/mlir_break_in_cond_vif.mind",
    "examples/mindc_mind/testdata/native_cf/mlir_f64_loop_carry_continue.mind",
    "examples/mindc_mind/testdata/native_cf/mlir_f64_loop_carry_vif.mind",
    "examples/mindc_mind/testdata/native_cf/mlir_lost_cond_assign.mind",
    "examples/mindc_mind/testdata/native_cf/mlir_lost_loop_assign.mind",
    "examples/mindc_mind/testdata/native_cf/mlir_narrow_loop_carry.mind",
    "examples/mindc_mind/testdata/native_cf/mlir_shadow_exit_carry.mind",
    "examples/mindc_mind/testdata/native_cf/narrow_wrap_loop.mind",
    "examples/mindc_mind/testdata/native_cf/shadow_scopes.mind",
    "examples/mindc_mind/testdata/native_cf/shadow_scopes_ssa.mind",
    "examples/mindc_mind/testdata/native_lane_values/casts_exprs.mind",
    "examples/mindc_mind/testdata/native_libc_io/reference.mind",
    "examples/mindc_mind/testdata/native_misc/a8_const_nosemi.mind",
    "examples/mindc_mind/testdata/native_misc/st_cond_use.mind",
    "examples/mindc_mind/testdata/native_misc/st_pos_while_cond.mind",
    "examples/mindc_mind/testdata/native_scoped_fixed/observers/geometry-owner-mutant.mind",
    "examples/mindc_mind/testdata/native_scoped_fixed/observers/geometry.mind",
    "examples/mindc_mind/testdata/native_scoped_fixed/observers/staged-frame-range-mutant.mind",
    "examples/mindc_mind/testdata/native_scoped_fixed/observers/staged-rebase-mutant.mind",
    "examples/mindc_mind/testdata/native_scoped_fixed/observers/staged-rollback-mutant.mind",
    "examples/mindc_mind/testdata/native_scoped_fixed/observers/staged-scope-end-mutant.mind",
    "examples/mindc_mind/testdata/native_scoped_fixed/observers/staged-state.mind",
    "examples/mindc_mind/testdata/native_scoped_fixed/sources/P09-typed-param-loop-live-values.mind",
    "examples/mindc_mind/testdata/native_source_driver/note_fixture_tail.mind",
];

#[cfg(not(feature = "std-surface"))]
fn expected_bare_skip_cause(label: &str) -> Option<&'static str> {
    let normalized = label.replace('\\', "/");
    let label = normalized.as_str();
    if BARE_E1042_REFUSALS.contains(&label) {
        Some("E1042")
    } else if BARE_LEGACY_PARSE_SKIPS.contains(&label) {
        Some("legacy-parse")
    } else {
        None
    }
}

#[test]
fn idempotence_examples() {
    let base = manifest_dir();
    let cfg = default_cfg();
    let mut passed = 0usize;
    let mut skipped = 0usize;
    #[cfg(not(feature = "std-surface"))]
    let mut bitwise_refused = 0usize;
    #[cfg(not(feature = "std-surface"))]
    let mut legacy_parse_skipped = 0usize;
    #[cfg(not(feature = "std-surface"))]
    let mut skip_cause_mismatches = Vec::new();

    // Collect all .mind files under examples/
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let examples_dir = base.join("examples");
    collect_mind_files(&examples_dir, &mut paths);
    paths.sort();

    assert!(!paths.is_empty(), "no .mind files found under examples/");

    for path in &paths {
        let label = path
            .strip_prefix(&base)
            .unwrap_or(path)
            .display()
            .to_string();
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if check_idempotence(&label, &src, &cfg) {
            passed += 1;
        } else {
            skipped += 1;
            #[cfg(not(feature = "std-surface"))]
            {
                let cause = match libmind::parser::parse(&src) {
                    Err(errors) if errors.iter().any(|e| e.cause_code == Some("E1042")) => {
                        bitwise_refused += 1;
                        "E1042"
                    }
                    Err(_) => {
                        legacy_parse_skipped += 1;
                        "legacy-parse"
                    }
                    Ok(_) => "formatter-only",
                };
                let expected = expected_bare_skip_cause(&label);
                if expected != Some(cause) {
                    skip_cause_mismatches.push(format!(
                        "{label}: expected {}, got {cause}",
                        expected.unwrap_or("no skip")
                    ));
                }
            }
        }
    }

    // Parse skips are permitted (tensor/autodiff examples), but the
    // idempotence assertion inside check_idempotence must hold for every
    // file that does parse.
    //
    // The counts used to be discarded here (`let _ = (passed, skipped)`), which made
    // this a test that could not fail: `check_idempotence` returns false on a pass-1
    // parse/format error, so if EVERY example stopped parsing, `passed` was 0,
    // `skipped` was everything, and the only surviving assertion -- that the file list
    // is non-empty -- still held. A formatter or parser regression severe enough to
    // make the entire examples corpus unparseable was reported as ok.
    //
    // The sibling tests already demand positive counts (`idempotence_stdlib` requires
    // skipped == 0 && passed == 3; `idempotence_fmt_fixtures` requires passed == 7).
    // This one cannot demand zero skips -- some examples legitimately do not parse --
    // so it ratchets on what the corpus proves TODAY instead.
    eprintln!(
        "idempotence_examples: {passed} passed, {skipped} skipped, {} total",
        paths.len()
    );
    #[cfg(not(feature = "std-surface"))]
    eprintln!("bare_skip_causes: E1042={bitwise_refused}, legacy-parse={legacy_parse_skipped}");
    #[cfg(not(feature = "std-surface"))]
    assert!(
        skip_cause_mismatches.is_empty(),
        "bare formatter skip classification changed:\n{}",
        skip_cause_mismatches.join("\n")
    );
    #[cfg(not(feature = "std-surface"))]
    assert_eq!(
        skipped,
        bitwise_refused + legacy_parse_skipped,
        "every bare skip must have a measured parse-cause classification"
    );
    #[cfg(not(feature = "std-surface"))]
    assert!(
        bitwise_refused >= 6,
        "expected at least 6 std-surface examples refused with E1042, got {bitwise_refused}"
    );
    assert!(
        passed >= EXAMPLES_IDEMPOTENCE_FLOOR,
        "idempotence floor breached: {passed} of {} examples round-tripped, floor is \
         {EXAMPLES_IDEMPOTENCE_FLOOR}. Either a formatter/parser regression stopped \
         files parsing, or examples were removed -- if the drop is intentional, lower \
         the floor IN THE SAME CHANGE and say why.",
        paths.len(),
    );
}

// ---------------------------------------------------------------------------
// tests/mindcraft/fmt/*.in.mind — Phase-2A fixture inputs
// ---------------------------------------------------------------------------

#[test]
fn idempotence_fmt_fixtures() {
    let fixture_dir = manifest_dir().join("tests/mindcraft/fmt");
    let cfg = default_cfg();
    let mut passed = 0usize;

    for entry in
        std::fs::read_dir(&fixture_dir).unwrap_or_else(|e| panic!("cannot read fixture dir: {e}"))
    {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("mind")
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".in.mind"))
                .unwrap_or(false)
        {
            let label = path.file_name().unwrap().to_string_lossy().to_string();
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            let exercised = check_idempotence(&label, &src, &cfg);
            assert!(exercised, "fixture {label} failed to parse — unexpected");
            passed += 1;
        }
    }

    assert_eq!(passed, 7, "expected 7 fixture .in.mind files, got {passed}");
}

// ---------------------------------------------------------------------------
// Bare `[T]` dynamic-slice canonicalisation (parser accepts `[T]` as sugar
// for `&[T]`; both share the `Slice` AST node). The formatter has one
// surface form for that node — `&[T]` — so `[T]` canonicalises to `&[T]`.
// This must be a DECIDED, stable rewrite (lossless: MIND draws no
// borrow/owned distinction on slices), not an accidental drift. Asserts the
// canonical target and that a second pass is a fixpoint.
// ---------------------------------------------------------------------------

#[test]
fn bare_slice_canonicalises_to_borrowed_and_is_idempotent() {
    let cfg = default_cfg();
    let src = "struct S { items: [u32] }\n";

    let once = format_source(src, &cfg).expect("bare `[T]` must format");
    assert!(
        once.contains("&[u32]"),
        "bare `[T]` must canonicalise to `&[T]`, got:\n{once}"
    );
    assert!(
        !once.contains(": [u32]"),
        "no bare `[T]` should survive formatting, got:\n{once}"
    );

    let twice = format_source(&once, &cfg).expect("second pass must format");
    assert_eq!(once, twice, "`[T]` canonicalisation must be a fixpoint");
}

#[test]
fn fixed_array_is_not_canonicalised_to_slice() {
    let cfg = default_cfg();
    let src = "struct S { lut: [u32; 4] }\n";
    let once = format_source(src, &cfg).expect("`[T; N]` must format");
    assert!(
        once.contains("[u32; 4]"),
        "`[T; N]` must stay a fixed-size array, got:\n{once}"
    );
    assert!(
        !once.contains("&["),
        "`[T; N]` must not become a slice, got:\n{once}"
    );
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn collect_mind_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_mind_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("mind") {
            out.push(path);
        }
    }
}
