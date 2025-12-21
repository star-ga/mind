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
# Do not forward subcommand-specific arguments here, as they may be invalid for fetch.
cargo deny fetch

CARGO_HOME_DIR=${CARGO_HOME:-"$HOME/.cargo"}
DB_ROOT="$CARGO_HOME_DIR/advisory-dbs"

# Check if file contains CVSS v4 line using grep (more portable than ripgrep)
has_cvss_v4() {
  grep -q '^cvss = "CVSS:4\.0/' "$1" 2>/dev/null
}

sanitize_cvss_v4() {
  local advisory_file="$1"
  if has_cvss_v4 "$advisory_file"; then
    echo "Stripping CVSS v4 line from $advisory_file" >&2
    # Remove only the CVSS line to keep the advisory content intact.
    local tmpfile
    tmpfile=$(mktemp) || { echo "Failed to create temporary file for $advisory_file" >&2; return 1; }
    sed '/^cvss = "CVSS:4\.0\//d' "$advisory_file" > "$tmpfile"
    mv "$tmpfile" "$advisory_file"
  fi
}

if [[ -d "$DB_ROOT" ]]; then
  # Find all advisory files containing CVSS v4 and sanitize them
  while IFS= read -r -d '' advisory; do
    sanitize_cvss_v4 "$advisory"
  done < <(find "$DB_ROOT" -name 'RUSTSEC-*.md' -print0)
fi

# Now run the actual command.
cargo deny "$@"
