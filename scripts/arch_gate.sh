#!/usr/bin/env bash
# arch_gate.sh - architectural-governance gate (arch-mind).
#
# Re-scans the tree with arch-mind's MIND reflection at pinned commit fea599ec
# (fea599ec4d3695cb39754d0fd5dcc1f1fb101dab) and enforces .arch-mind/rules.mind.
# Fails closed when:
#   1. the arch-mind binary is missing or does not match the pinned tool hashes,
#   2. .arch-mind/rules.mind is missing or empty,
#   3. sidecar-scan fails or writes nothing,
#   4. check-rules reports 0 rules, or
#   5. any rule regresses against origin/main in its rule direction (merge-
#      procedure interim rule: no rule may get worse than origin/main, while
#      not demanding absolute floors main has never met).
# Prints asserted=<rules enforced>.
set -uo pipefail

ARCH_MIND_PIN="fea599ec4d3695cb39754d0fd5dcc1f1fb101dab"
CLI_SHA256="dee77ff9b2263a395d01fecc44f177276cf64c01efd608af0ce5652d011cc8d1"
REFLECT_SHA256="3d128ee4b72e23db0a7ab0d10f0aeaecfbbf0e04f9756d9ebeede9b5ba97b026"
ENGINE_SHA256="60956668517452a5cf13d5ac85cd19023d0d508b8ab8350cc3d048a60f27a50b"

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT" || exit 1

AM="${ARCH_MIND:-}"
if [ -z "$AM" ]; then
  if command -v arch-mind >/dev/null 2>&1; then AM="$(command -v arch-mind)"
  elif [ -x "$HOME/arch-mind/bin/arch-mind" ]; then AM="$HOME/arch-mind/bin/arch-mind"
  elif [ -x "$ROOT/../arch-mind/bin/arch-mind" ]; then AM="$ROOT/../arch-mind/bin/arch-mind"
  fi
fi
if [ -z "$AM" ]; then
  echo "::error::arch-mind not found (set ARCH_MIND or put arch-mind on PATH)"
  exit 1
fi
AM_BIN="$(command -v "$AM" 2>/dev/null || printf '%s' "$AM")"
if [ ! -f "$AM_BIN" ] || [ ! -x "$AM_BIN" ]; then
  echo "::error::arch-mind binary is missing or not executable: $AM"
  exit 1
fi

AM_ROOT="$(cd "$(dirname "$AM_BIN")/.." 2>/dev/null && pwd)"
if [ ! -d "$AM_ROOT/tools" ]; then
  echo "::error::arch-mind tools directory missing under $AM_ROOT"
  exit 1
fi
if [ -e "$AM_ROOT/.git" ]; then
  am_head="$(git -C "$AM_ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [ "$am_head" != "$ARCH_MIND_PIN" ]; then
    echo "::error::arch-mind HEAD is '$am_head', expected pinned '$ARCH_MIND_PIN'"
    exit 1
  fi
fi
printf '%s  %s\n%s  %s\n%s  %s\n' \
  "$CLI_SHA256" "$AM_ROOT/tools/arch_mind_cli.py" \
  "$REFLECT_SHA256" "$AM_ROOT/tools/mind_reflect.py" \
  "$ENGINE_SHA256" "$AM_ROOT/tools/rules_engine.py" | sha256sum -c - >/dev/null 2>&1 || {
  echo "::error::arch-mind tool sha256 mismatch against pinned commit ${ARCH_MIND_PIN:0:8}"
  exit 1
}

[ -s .arch-mind/rules.mind ] || { echo "::error::.arch-mind/rules.mind missing or empty"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/arch-gate.XXXXXX")" || exit 1
trap 'rm -rf "$WORK"' EXIT

FIX=.arch-mind/last_summary.json
rm -f "$FIX"
LOG="$WORK/head-scan.log"
"$AM_BIN" sidecar-scan --repo . --lang mind --out "$FIX" > "$LOG" 2>&1 || {
  tail -n 20 "$LOG"
  echo "::error::arch-mind sidecar-scan failed"
  exit 1
}
[ -s "$FIX" ] || { echo "::error::scan produced no fixture"; exit 1; }

out=$("$AM_BIN" check-rules --repo . --fixture "$FIX" --rules .arch-mind/rules.mind --mode enforce 2>&1)
rc=$?
echo "$out"
n=$(printf '%s\n' "$out" | sed -n 's/^rules: \([0-9][0-9]*\).*/\1/p' | head -1)
[ -n "$n" ] && [ "$n" -ge 1 ] || {
  echo "::error::check-rules reported no rules; a pass would assert nothing"
  exit 1
}

BASE_REF="${ARCH_BASE_REF:-}"
if [ -z "$BASE_REF" ]; then
  if git rev-parse --verify "origin/main^{commit}" >/dev/null 2>&1; then
    BASE_REF="origin/main"
  elif git rev-parse --verify "main^{commit}" >/dev/null 2>&1; then
    BASE_REF="main"
  else
    echo "::error::cannot resolve origin/main or main for baseline rule comparison"
    exit 1
  fi
else
  git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null 2>&1 || {
    echo "::error::ARCH_BASE_REF '$BASE_REF' does not resolve to a commit"
    exit 1
  }
fi

BASE_DIR="$WORK/base"
mkdir -p "$BASE_DIR" || exit 1
git archive "$BASE_REF" | tar -x -C "$BASE_DIR" || {
  echo "::error::failed to archive baseline ref '$BASE_REF'"
  exit 1
}
[ -s "$BASE_DIR/.arch-mind/rules.mind" ] || {
  echo "::error::baseline '$BASE_REF' is missing .arch-mind/rules.mind"
  exit 1
}

BASE_FIX="$WORK/base_summary.json"
BASE_LOG="$WORK/base-scan.log"
"$AM_BIN" sidecar-scan --repo "$BASE_DIR" --lang mind --out "$BASE_FIX" > "$BASE_LOG" 2>&1 || {
  tail -n 20 "$BASE_LOG"
  echo "::error::baseline arch-mind sidecar-scan failed on '$BASE_REF'"
  exit 1
}
[ -s "$BASE_FIX" ] || { echo "::error::baseline scan produced no fixture"; exit 1; }

HEAD_SCAN="$WORK/head_scan.json"
BASE_SCAN="$WORK/base_scan.json"
"$AM_BIN" scan --fixture "$FIX" --repo . --out "$HEAD_SCAN" >/dev/null 2>&1 || {
  echo "::error::failed to compute Q16 metric snapshot for working tree"
  exit 1
}
"$AM_BIN" scan --fixture "$BASE_FIX" --repo "$BASE_DIR" --out "$BASE_SCAN" >/dev/null 2>&1 || {
  echo "::error::failed to compute Q16 metric snapshot for baseline '$BASE_REF'"
  exit 1
}
[ -s "$HEAD_SCAN" ] && [ -s "$BASE_SCAN" ] || {
  echo "::error::metric snapshot is empty"
  exit 1
}

python3 - "$AM_ROOT/tools" "$BASE_DIR/.arch-mind/rules.mind" ".arch-mind/rules.mind" "$BASE_SCAN" "$HEAD_SCAN" "$BASE_REF" <<'PY'
import json
import sys
from pathlib import Path

tools_dir, base_rules_path, head_rules_path, base_scan_path, head_scan_path, base_ref = sys.argv[1:]
sys.path.insert(0, tools_dir)
import rules_engine as eng  # noqa: E402

base_rules = eng.load_rules(Path(base_rules_path))
head_rules = eng.load_rules(Path(head_rules_path))
if not base_rules or not head_rules:
    print("::error::0 rules loaded from rules.mind", file=sys.stderr)
    sys.exit(1)

base_by_name = {r.name: r for r in base_rules}
head_by_name = {r.name: r for r in head_rules}
for name, br in base_by_name.items():
    if name not in head_by_name:
        print(f"::error::rule {name} from {base_ref} was deleted in working tree", file=sys.stderr)
        sys.exit(1)
    hr = head_by_name[name]
    if (hr.metric, hr.op) != (br.metric, br.op):
        print(
            f"::error::rule {name} changed metric/op from {(br.metric, br.op)} to {(hr.metric, hr.op)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if br.op in ("ge", "gt") and hr.threshold_raw < br.threshold_raw:
        print(
            f"::error::rule {name} lowered floor from {br.threshold_raw} to {hr.threshold_raw}",
            file=sys.stderr,
        )
        sys.exit(1)
    if br.op in ("le", "lt") and hr.threshold_raw > br.threshold_raw:
        print(
            f"::error::rule {name} raised ceiling from {br.threshold_raw} to {hr.threshold_raw}",
            file=sys.stderr,
        )
        sys.exit(1)
    if br.op in ("eq", "ne") and hr.threshold_raw != br.threshold_raw:
        print(
            f"::error::rule {name} changed target from {br.threshold_raw} to {hr.threshold_raw}",
            file=sys.stderr,
        )
        sys.exit(1)

base_scan = json.loads(Path(base_scan_path).read_text(encoding="utf-8"))
head_scan = json.loads(Path(head_scan_path).read_text(encoding="utf-8"))

print(f"arch-mind baseline comparison against {base_ref} (pin={eng.__file__}):")
regressions = []
for rule in head_rules:
    if rule.metric not in base_scan or rule.metric not in head_scan:
        print(f"::error::metric {rule.metric!r} missing from scan output", file=sys.stderr)
        sys.exit(1)
    bq = int(base_scan[rule.metric])
    hq = int(head_scan[rule.metric])
    b_raw = bq // eng.Q16_ONE
    h_raw = hq // eng.Q16_ONE
    thresh_q16 = rule.threshold_raw * eng.Q16_ONE
    base_abs_ok = eng._compare(bq, rule.op, thresh_q16)
    head_abs_ok = eng._compare(hq, rule.op, thresh_q16)

    if rule.op in ("ge", "gt"):
        dir_ok = hq >= bq
    elif rule.op in ("le", "lt"):
        dir_ok = hq <= bq
    elif rule.op == "eq":
        if bq == thresh_q16:
            dir_ok = hq == thresh_q16
        else:
            dir_ok = abs(hq - thresh_q16) <= abs(bq - thresh_q16)
    elif rule.op == "ne":
        dir_ok = head_abs_ok if base_abs_ok else True
    else:
        dir_ok = False

    # Any rule that passed its absolute floor on base must still pass its
    # absolute floor on head AND not get worse than base. Rules whose absolute
    # floor main has not yet met must not get worse than base.
    rule_ok = dir_ok and (head_abs_ok or not base_abs_ok)
    verdict = "PASS" if rule_ok else "FAIL"
    print(
        f"  [{verdict}] {rule.name:<24} {rule.metric:<28} {rule.op} {rule.threshold_raw:>5} "
        f"base={b_raw:>5} (q16={bq}) -> head={h_raw:>5} (q16={hq})"
    )
    if not rule_ok:
        regressions.append(rule.name)

if regressions:
    print(
        f"::error::arch-mind regression vs {base_ref} on {len(regressions)} rule(s): "
        + ", ".join(regressions),
        file=sys.stderr,
    )
    sys.exit(1)
print(f"OK: all {len(head_rules)} rules no worse than {base_ref}.")
PY
cmp_rc=$?
[ "$cmp_rc" -eq 0 ] || exit "$cmp_rc"
echo "asserted=$n"
