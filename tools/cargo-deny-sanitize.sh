#!/usr/bin/env bash
set -euo pipefail

# Run cargo-deny but sanitize advisory entries that older cargo-deny versions
# cannot parse (e.g., CVSS v4 metadata).
COMMAND=${1:-}
if [[ -z "$COMMAND" ]]; then
  echo "usage: $(basename "$0") <cargo-deny-subcommand> [args...]" >&2
  exit 1
fi

# Fetch advisory database first so we can patch it if needed.
# Use separate args so we don't pass the subcommand twice.
FETCH_ARGS=("fetch")
for arg in "${@:2}"; do
  FETCH_ARGS+=("$arg")
done
cargo deny "${FETCH_ARGS[@]}"

CARGO_HOME_DIR=${CARGO_HOME:-"$HOME/.cargo"}
DB_ROOT="$CARGO_HOME_DIR/advisory-dbs"

sanitize_cvss_v4() {
  local advisory_file="$1"
  if rg -q '^cvss = "CVSS:4\.0/' "$advisory_file" 2>/dev/null; then
    echo "Stripping CVSS v4 line from $advisory_file" >&2
    # Remove only the CVSS line to keep the advisory content intact.
    tmpfile=$(mktemp)
    sed '/^cvss = "CVSS:4\.0\//d' "$advisory_file" > "$tmpfile"
    mv "$tmpfile" "$advisory_file"
  fi
}

if [[ -d "$DB_ROOT" ]]; then
  while IFS= read -r -d '' advisory; do
    sanitize_cvss_v4 "$advisory"
  done < <(find "$DB_ROOT" -name 'RUSTSEC-2024-0445.md' -print0)
fi

# Now run the actual command.
cargo deny "$@"
