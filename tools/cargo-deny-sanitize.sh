#!/usr/bin/env bash
set -euo pipefail

# Run cargo-deny but sanitize advisory entries that older cargo-deny versions
# cannot parse (e.g., CVSS v4 metadata).
#
# The issue is that `cargo deny fetch` fails during parsing, so we must
# fetch the advisory database manually via git, sanitize it, then run cargo deny.

COMMAND=${1:-}
if [[ -z "$COMMAND" ]]; then
  echo "usage: $(basename "$0") <cargo-deny-subcommand> [args...]" >&2
  exit 1
fi

CARGO_HOME_DIR=${CARGO_HOME:-"$HOME/.cargo"}
DB_ROOT="$CARGO_HOME_DIR/advisory-dbs"
RUSTSEC_REPO="https://github.com/rustsec/advisory-db.git"
RUSTSEC_DIR="$DB_ROOT/github.com-a946fc29ac602819"

# Manually fetch the advisory database via git (cargo deny fetch fails on CVSS v4)
mkdir -p "$DB_ROOT"
if [[ -d "$RUSTSEC_DIR/.git" ]]; then
  echo "Updating RustSec advisory database..." >&2
  git -C "$RUSTSEC_DIR" fetch --quiet origin main
  git -C "$RUSTSEC_DIR" reset --quiet --hard origin/main
else
  echo "Cloning RustSec advisory database..." >&2
  git clone --quiet --depth 1 "$RUSTSEC_REPO" "$RUSTSEC_DIR"
fi

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
    # Ensure tmpfile is cleaned up on exit from this function
    trap 'rm -f "$tmpfile"' RETURN
    sed '/^cvss = "CVSS:4\.0\//d' "$advisory_file" > "$tmpfile"
    mv "$tmpfile" "$advisory_file"
  fi
}

# Find all advisory files containing CVSS v4 and sanitize them
if [[ -d "$DB_ROOT" ]]; then
  while IFS= read -r -d '' advisory; do
    sanitize_cvss_v4 "$advisory"
  done < <(find "$DB_ROOT" -name 'RUSTSEC-*.md' -print0)
fi

# Now run the actual command (skip fetch since we already have the DB)
cargo deny "$@"
