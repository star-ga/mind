#!/usr/bin/env bash
# fast_keystone.sh — fast LOCAL front-end keystone gate for the pure-MIND self-host
# driver (examples/mindc_mind/main.mind).
#
# Proves, in ~1-2 min, that the front-end emit is byte-identical and the cdylib build
# is deterministic — WITHOUT the ~25-minute `cargo test --release phase_g_keystone_bootstrap`
# release-test recompile. This is the per-increment gate U1 runs before committing a
# self-host change; CI still runs the full cargo keystone + cross_substrate.
#
# Checks:
#   1. deterministic build  — Mind.toml-driven cdylib == direct-path cdylib (byte-identical)
#   2. self_host_body_smoke — real-body emitter
#   3. mic3_flip_smoke      — whole-module mic@3 self-host FLIP byte-identical (nfn==--emit-mic3)
#   4. mic3_primitives_smoke— mic@3 codec primitives
#   5. self_host_mlir_smoke — per-construct MLIR emit. NOTE: post-pivot to the native-ELF
#                             self-host backend, MLIR text is a spec-NON-NORMATIVE surface,
#                             so this is a DEBUG AID, not the normative gate. The normative
#                             artifact gate (native-ELF byte-identity + program output-hash
#                             + mic@3 trace_hash) is added separately.
#
# Usage:  examples/mindc_mind/fast_keystone.sh                  # uses target/release/mindc
#         MINDC_REBUILD=1 examples/mindc_mind/fast_keystone.sh  # rebuild mindc first
#         MINDC_BIN=/path/to/mindc examples/mindc_mind/fast_keystone.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 2
MINDC="${MINDC_BIN:-$ROOT/target/release/mindc}"
ENTRY="examples/mindc_mind/main.mind"
DIR_SO="/tmp/fk_direct_$$.so"; MT_SO="/tmp/fk_mt_$$.so"
t0=$(date +%s); pass=0; fail=0
chk() { local n="$1"; shift
  if "$@" >/tmp/fk_step.log 2>&1; then echo "  PASS  $n"; pass=$((pass+1))
  else echo "  FAIL  $n"; tail -4 /tmp/fk_step.log | sed 's/^/        /'; fail=$((fail+1)); fi; }

if [ "${MINDC_REBUILD:-0}" = "1" ]; then
  echo "[rebuilding mindc release binary]"
  cargo build --release --no-default-features \
    --features "mlir-build std-surface cross-module-imports" --bin mindc \
    || { echo "ERROR: mindc build failed"; exit 2; }
fi
[ -x "$MINDC" ] || { echo "ERROR: mindc not found at $MINDC (try MINDC_REBUILD=1)"; exit 2; }

# Pre-flight: the smokes below need an mlir-build mindc (--emit-mlir / cdylib emit).
# A prior `cargo bench` (e.g. scripts/quick_perf.sh) rebuilds target/release/mindc with
# DEFAULT features — NO mlir-build — silently clobbering this binary. That would surface
# as a misleading `mlir_smoke` FAIL ("--emit-mlir requires the mlir-lowering/mlir-build
# feature"), reading like a front-end regression when it is only a stripped binary.
# Detect it precisely and fail loud with the fix instead of reporting a false regression.
_fk_probe="/tmp/fk_probe_$$.mind"
printf 'fn __fk_probe(a: i64) -> i64 {\n    return a;\n}\n' > "$_fk_probe"
if ! "$MINDC" "$_fk_probe" --emit-mlir >/dev/null 2>&1; then
  rm -f "$_fk_probe"
  echo "ERROR: $MINDC lacks mlir-build (--emit-mlir failed). A prior 'cargo bench' /"
  echo "       scripts/quick_perf.sh likely rebuilt it with default features. Rebuild the"
  echo "       gate binary before re-running the keystone:"
  echo "         MINDC_REBUILD=1 $0"
  echo "       or: cargo build --release --no-default-features \\"
  echo "             --features \"mlir-build std-surface cross-module-imports\" --bin mindc"
  exit 2
fi
rm -f "$_fk_probe"

echo "== fast front-end keystone =="
# 0. FORMAT GATE (fail fast). `mindc check` treats fmt::drift as a FATAL error
# (it reddens the "Mindcraft check" CI job), and NONE of the byte-identity gates
# below catch formatting — a self-host edit that leaves a trailing `;` or stray
# layout passes every smoke here yet fails CI. Catch it before the push.
if ! "$MINDC" fmt --check "$ENTRY" >/tmp/fk_fmt.log 2>&1; then
  echo "  FAIL  fmt drift in $ENTRY — run: $MINDC fmt --fix $ENTRY"
  head -8 /tmp/fk_fmt.log | sed 's/^/        /'; exit 1
fi
echo "  PASS  fmt check ($ENTRY canonical)"
# 1. deterministic build: Mind.toml-driven vs direct must be byte-identical.
# --no-cache is LOAD-BEARING for this gate: the module cache key (src/build/cache.rs)
# hashes source_bytes + target + optimize + compiler_version (semver, NOT a build hash),
# so a compiler change with unchanged .mind source + unchanged semver is a CACHE HIT that
# serves the PRE-change cdylib. Both build paths would then get the same stale bytes and
# `cmp` passes trivially — defeating the self-consistency check and hiding a real drift.
# Fresh compiles make this a true gate (a deterministic compiler still yields identical
# bytes for the two paths; a non-deterministic one is caught).
if ! "$MINDC" build "$ENTRY" --release --emit=cdylib --no-cache --out="$DIR_SO" >/tmp/fk_step.log 2>&1; then
  echo "  FAIL  direct cdylib build"; tail -6 /tmp/fk_step.log | sed 's/^/        /'; exit 1; fi
if ! "$MINDC" build --release --emit=cdylib --no-cache --out="$MT_SO" >/tmp/fk_step.log 2>&1; then
  echo "  FAIL  Mind.toml cdylib build"; tail -6 /tmp/fk_step.log | sed 's/^/        /'; exit 1; fi
if cmp -s "$DIR_SO" "$MT_SO"; then
  echo "  PASS  byte-identical build ($(stat -c%s "$MT_SO") B, sha=$(sha256sum "$MT_SO" | cut -c1-16))"; pass=$((pass+1))
else
  echo "  FAIL  build NOT byte-identical (Mind.toml-driven != direct-path)"; fail=$((fail+1)); fi

# 2-5. front-end + construct smokes against the built .so
export MINDC_SO="$MT_SO" MINDC="$MINDC" MINDC_BIN="$MINDC"
chk "body_smoke (real-body emit)"          python3 examples/mindc_mind/self_host_body_smoke.py
chk "mic3_flip (whole-module FLIP)"        python3 examples/mindc_mind/mic3_flip_smoke.py
chk "mic3_primitives (codec)"              python3 examples/mindc_mind/mic3_primitives_smoke.py
chk "mlir_smoke (constructs; debug aid)"   python3 examples/mindc_mind/self_host_mlir_smoke.py
chk "mod_operator (% remainder, both paths)" python3 examples/mindc_mind/mod_operator_smoke.py
chk "narrowint (i8/i16/i32 store/load)"    python3 examples/mindc_mind/self_host_native_narrowint_smoke.py
chk "scalar_f32 (native SSE f32 chain)"    python3 examples/mindc_mind/self_host_native_scalar_f32_smoke.py
chk "tc_narrowing (E2004 i64->i32 rule)"   python3 examples/mindc_mind/self_host_tc_narrowing_smoke.py
chk "div_shift_cmp (C3 signed edges)"      python3 examples/mindc_mind/div_shift_cmp_edge_smoke.py
chk "narrowwrap (i8/i16/i32 wrap arith)"   python3 examples/mindc_mind/self_host_native_narrowwrap_smoke.py
chk "tc_class (E2015 int/float class)"     python3 examples/mindc_mind/self_host_tc_class_mismatch_smoke.py
chk "tc_class_rules (E2010/11/13/16)"      python3 examples/mindc_mind/self_host_tc_class_rules_smoke.py
chk "tc_shape_rules (E2005/2101/2/3)"     python3 examples/mindc_mind/self_host_tc_shape_rules_smoke.py
chk "autowrap (declared-width driver)"     python3 examples/mindc_mind/self_host_native_autowrap_smoke.py
chk "tensor_ewadd (C4 native tensor)"       python3 examples/mindc_mind/self_host_native_tensor_ewadd_smoke.py
chk "tensor_dot (C4-T2 native MAC)"        python3 examples/mindc_mind/self_host_native_tensor_dot_smoke.py
chk "tensor_matmul (C4-T3 nested 2-D)"    python3 examples/mindc_mind/self_host_native_tensor_matmul_smoke.py
chk "tensor_rowsum (C4-T4 reduction)"     python3 examples/mindc_mind/self_host_native_tensor_rowsum_smoke.py
chk "tensor_transpose (C4-T4 transpose)"  python3 examples/mindc_mind/self_host_native_tensor_transpose_smoke.py
chk "tensor_ewmul (C4-T5 elementwise mul)" python3 examples/mindc_mind/self_host_native_tensor_ewmul_smoke.py
chk "tensor_colsum (C4-T5 col reduction)"  python3 examples/mindc_mind/self_host_native_tensor_colsum_smoke.py
chk "tensor_bcastadd (C4-T5 broadcast add)" python3 examples/mindc_mind/self_host_native_tensor_bcastadd_smoke.py
chk "standalone_driver (C8 no-Python mindc)" python3 examples/mindc_mind/self_host_standalone_driver_smoke.py
chk "tensor_maxrowmax (C4-T6 max-reduce)"  python3 examples/mindc_mind/self_host_native_tensor_maxrowmax_smoke.py
chk "tensor_relu (C4-T6 relu)"             python3 examples/mindc_mind/self_host_native_tensor_relu_smoke.py
chk "narrow_add_i8 (C2 wrap arithmetic)"   python3 examples/mindc_mind/self_host_native_narrow_add_i8_smoke.py
chk "narrow_arith_batch (C2 {sub,mul}i8+{add,mul}i16)" python3 examples/mindc_mind/self_host_native_narrow_arith_batch_smoke.py
chk "tensor_batchsum (C4-T6 3-D N-D idx)"  python3 examples/mindc_mind/self_host_native_tensor_batchsum_smoke.py
chk "tensor_rowmin (C4-T6 min-reduce)"     python3 examples/mindc_mind/self_host_native_tensor_rowmin_smoke.py
chk "param_mutation (reassigned-param live slot)" python3 examples/mindc_mind/self_host_param_mutation_smoke.py
chk "float_lit_exact (dyadic-only C1 guard)" python3 examples/mindc_mind/self_host_float_lit_exact_smoke.py
chk "carry_cap (256-cap scratch-table guard)" python3 examples/mindc_mind/self_host_carry_cap_smoke.py
chk "narrow_param (i8/i16/i32 param fail-closed)" python3 examples/mindc_mind/self_host_narrow_param_smoke.py
chk "if_region_carry (i64 branch loop-carry)"  python3 examples/mindc_mind/self_host_if_region_carry_smoke.py
chk "tc_let_class (E2015 let/assign class)" python3 examples/mindc_mind/self_host_tc_let_class_mismatch_smoke.py
chk "tc_fixed_bytes (E2006 bytes->vec)" python3 examples/mindc_mind/self_host_tc_fixed_bytes_into_vec_smoke.py
chk "tc_shape_annot (dtype/rank/dim compat)" python3 examples/mindc_mind/self_host_tc_shape_annot_compat_smoke.py
chk "tc_classify (error-code router)" python3 examples/mindc_mind/self_host_tc_classify_error_code_smoke.py
chk "tc_self_host_only (E2024 intrinsic-table advisory)" python3 examples/mindc_mind/self_host_tc_self_host_only_call_smoke.py
chk "arena_growth (self-host cap headroom)" python3 examples/mindc_mind/self_host_arena_growth_smoke.py
chk "toplevel_assign (C: straight-line i64 reassign)" python3 examples/mindc_mind/self_host_native_toplevel_assign_smoke.py
chk "narrow_paramret (i8/i16/i32 param+return wrap)" python3 examples/mindc_mind/self_host_native_narrow_paramret_smoke.py
# Grown-subset capability value-correctness gates (main.mind does not USE these
# features, so the self-host loop never exercises them — these harnesses are their
# ONLY regression gate; wired here so CI protects each landed rung).
chk "ref_netverify (refs value + cycle fail-closed)" python3 examples/mindc_mind/ref_netverify.py
chk "field_store_netverify (p.x=v mutable structs)"  python3 examples/mindc_mind/field_store_netverify.py
chk "enum_netverify (C-like enums + match)"          python3 examples/mindc_mind/enum_netverify.py
chk "option_netverify (Option/Result payload enums)" python3 examples/mindc_mind/option_netverify.py
chk "failclosed (poison boundary + float refusal)"   python3 examples/mindc_mind/self_host_failclosed_smoke.py
chk "closure_netverify (unresolved-callee fail-closed)" python3 examples/mindc_mind/closure_netverify.py
chk "fp_call (float call-return/args + unresolved-callee 0B)" python3 examples/mindc_mind/self_host_native_fp_call_smoke.py
chk "fp_field (float struct-field read (SSE))" python3 examples/mindc_mind/self_host_native_fp_field_smoke.py

rm -f "$DIR_SO" "$MT_SO"
echo "== $pass passed, $fail failed in $(($(date +%s)-t0))s =="
[ "$fail" -eq 0 ]
