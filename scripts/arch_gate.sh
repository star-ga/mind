#!/usr/bin/env bash
# arch_gate.sh - architectural-governance gate (arch-mind). Re-scans the tree
# with arch-mind's MIND reflection and enforces .arch-mind/rules.mind. Fails
# closed when arch-mind is missing, when the scan writes nothing, or when any
# rule breaks. Prints asserted=<rules enforced>.
set -uo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
AM="${ARCH_MIND:-}"
if [ -z "$AM" ]; then
  if command -v arch-mind >/dev/null 2>&1; then AM=arch-mind
  elif [ -x "$HOME/arch-mind/bin/arch-mind" ]; then AM="$HOME/arch-mind/bin/arch-mind"
  elif [ -x /home/user/arch-mind/bin/arch-mind ]; then AM=/home/user/arch-mind/bin/arch-mind
  fi
fi
[ -n "$AM" ] || { echo "::error::arch-mind not found (set ARCH_MIND or put arch-mind on PATH)"; exit 1; }
[ -f .arch-mind/rules.mind ] || { echo "::error::.arch-mind/rules.mind missing"; exit 1; }
FIX=.arch-mind/last_summary.json
rm -f "$FIX"
LOG=$(mktemp "${TMPDIR:-/tmp}/arch-scan.XXXXXX")
"$AM" sidecar-scan --repo . --lang mind --out "$FIX" > "$LOG" 2>&1 || { tail -n 20 "$LOG"; rm -f "$LOG"; echo "::error::arch-mind sidecar-scan failed"; exit 1; }
rm -f "$LOG"
[ -s "$FIX" ] || { echo "::error::scan produced no fixture"; exit 1; }
out=$("$AM" check-rules --repo . --fixture "$FIX" --rules .arch-mind/rules.mind --mode enforce 2>&1); rc=$?
echo "$out"
n=$(printf '%s\n' "$out" | sed -n 's/^rules: \([0-9][0-9]*\).*/\1/p' | head -1)
[ -n "$n" ] && [ "$n" -ge 1 ] || { echo "::error::check-rules reported no rules; a pass would assert nothing"; exit 1; }
[ "$rc" -eq 0 ] || exit "$rc"
echo "asserted=$n"
