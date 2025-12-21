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

# Sanitize CVSS v4 lines from advisory files
# Use grep -v with temp file for reliable modification
echo "Sanitizing CVSS v4 entries from advisory database..." >&2

for advisory_file in $(find "$DB_ROOT" -name 'RUSTSEC-*.md' -type f); do
  if grep -q 'CVSS:4\.' "$advisory_file" 2>/dev/null; then
    echo "  Removing CVSS v4 from: $advisory_file" >&2

    # Use grep -v to filter out CVSS v4 lines, write to temp, then replace
    tmpfile="${advisory_file}.tmp.$$"
    grep -v 'CVSS:4\.' "$advisory_file" > "$tmpfile"
    mv -f "$tmpfile" "$advisory_file"

    # Verify the line was removed
    if grep -q 'CVSS:4\.' "$advisory_file" 2>/dev/null; then
      echo "  ERROR: CVSS v4 line still present after removal!" >&2
      cat "$advisory_file" >&2
      exit 1
    else
      echo "  Successfully removed CVSS v4 line" >&2
    fi
  fi
done

echo "Sanitization complete. Running cargo deny with fetch disabled..." >&2

# Run cargo deny with --disable-fetch to use our pre-fetched, sanitized database
# This only disables advisory DB fetching, not cargo metadata (unlike --offline)
cargo deny --disable-fetch "$@"
