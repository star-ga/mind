#!/usr/bin/env bash
# Execute every quant example's KAT and fail closed on the first red one.
#
# WHY THIS EXISTS, stated plainly: CI already runs `mindc check std/ examples/`,
# and that is NOT a substitute for this. `mindc check` type-checks; it does not
# execute. A KAT whose assertions all fail still type-checks perfectly, so a
# check-only gate reports green while every priced number in the file is wrong.
# Measured on this tree: `mindc check` exits 0 for a program whose main returns
# 3. These examples exist to prove numbers are right, and a gate that never runs
# them proves nothing about that.
#
# CONTRACT. Each example under examples/quant/ is a package whose `main`
# returns the NUMBER OF FAILED ASSERTIONS, so the process exit code is the
# failure count and 0 means every assertion held. This script runs each one from
# a CLEAN build directory and requires exit 0.
#
# The clean rebuild is not defensive tidiness — it is load-bearing. A stale
# target/ in an example directory makes `mindc run` report the exit code of a
# PREVIOUSLY built artifact rather than the current source. That has already
# produced three different exit codes (0, 1, 7) from one unmodified file in a
# single session, which reads exactly like nondeterminism and is not. Anything
# trusting an exit code here must delete target/ first.
#
# Exit 0 = every example's KAT passed. Exit 1 = at least one failed, or the
# toolchain/layout was not usable.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUANT="$ROOT/examples/quant"
MINDC="${MINDC:-mindc}"

if ! command -v "$MINDC" >/dev/null 2>&1; then
    echo "run_quant_examples: '$MINDC' not on PATH (set MINDC=/path/to/mindc)" >&2
    exit 1
fi

if [ ! -d "$QUANT" ]; then
    echo "run_quant_examples: no such directory: $QUANT" >&2
    exit 1
fi

# Sorted so the run order is deterministic and a log diff between two CI runs is
# meaningful.
mapfile -t EXAMPLES < <(find "$QUANT" -mindepth 2 -maxdepth 2 -name main.mind -printf '%h\n' | sort)

if [ "${#EXAMPLES[@]}" -eq 0 ]; then
    # An empty corpus must NOT pass. A glob that silently matches nothing is the
    # classic vacuous gate: it reports success without having tested anything.
    echo "run_quant_examples: FAIL — no examples found under $QUANT" >&2
    exit 1
fi

failed=0
for dir in "${EXAMPLES[@]}"; do
    name="$(basename "$dir")"
    rm -rf "$dir/target" "$dir/.mind-build.lock"

    # No pipe here on purpose: `cmd | tail` would make $? the exit status of
    # tail, which essentially always succeeds, and the gate would read green
    # whatever the example did.
    out="$(cd "$dir" && "$MINDC" run 2>&1)"
    rc=$?

    if [ "$rc" -eq 0 ]; then
        printf '  PASS  %s\n' "$name"
    else
        failed=$((failed + 1))
        # Distinguish the two failures that both arrive as a non-zero exit, because
        # they send you to completely different places. A KAT failure means the
        # numbers are wrong (fix the code or the references). A BUILD failure means
        # the example never ran at all, so the exit code is the toolchain's, not a
        # count of anything — reporting "N failed assertions" for a compile error
        # sends the reader hunting for N bad constants that do not exist.
        if printf '%s' "$out" | grep -qE '^error: |MLIR build failed|mlir-opt failed|mindc: error'; then
            printf '  FAIL  %s (BUILD ERROR — the example did not run; exit %s is the toolchain'"'"'s)\n' "$name" "$rc"
        else
            printf '  FAIL  %s (exit %s = failed assertion count)\n' "$name" "$rc"
        fi
        printf '%s\n' "$out" | sed 's/^/        /'
    fi

    # Leave no build artifacts behind: they are gitignored, but a left-behind
    # target/ is exactly what makes the NEXT run report a stale exit code.
    rm -rf "$dir/target" "$dir/.mind-build.lock"
done

echo
if [ "$failed" -ne 0 ]; then
    echo "run_quant_examples: FAIL — $failed of ${#EXAMPLES[@]} example(s) red." >&2
    echo "The exit code of each failing example is its count of failed assertions." >&2
    echo "Fix the code or the reference values — never widen a band to hide a miss." >&2
    exit 1
fi

echo "run_quant_examples: OK — ${#EXAMPLES[@]} example(s) passed."
