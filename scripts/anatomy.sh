#!/usr/bin/env bash
# anatomy — Generate ANATOMY.md for any repo
# Universal file index for coding agents (Claude Code, Codex, Gemini CLI, etc.)
# Usage: anatomy [dir] [--update] [--depth N] [--exclude pattern]
#
# Author: STARGA Inc <noreply@star.ga>
# License: MIT

set -euo pipefail

VERSION="1.0.0"
MAX_DEPTH=4
EXCLUDE_PATTERNS=()
TARGET_DIR="."
UPDATE_MODE=false
OUTPUT_FILE="ANATOMY.md"

# Token estimate: ~4 chars per token (typical BPE tokenizer)
estimate_tokens() {
  local file="$1"
  local chars
  chars=$(wc -c < "$file" 2>/dev/null || echo 0)
  echo $(( (chars + 3) / 4 ))
}

# Extract first meaningful line (skip shebangs, empty lines, imports)
extract_summary() {
  local file="$1"
  local ext="${file##*.}"
  
  case "$ext" in
    py)
      # Look for docstring or first comment
      grep -m1 -E '^\s*("""|#\s+\w)' "$file" 2>/dev/null | sed 's/^[[:space:]]*//' | sed 's/"""//' | utf8_trunc 120 || true
      ;;
    rs|ts|js|tsx|jsx|go|c|cpp|h|java|swift|kt)
      # First // or /// comment
      grep -m1 -E '^\s*//[/!]?\s+\w' "$file" 2>/dev/null | sed 's|^[[:space:]]*//[/!]*[[:space:]]*||' | utf8_trunc 120 || true
      ;;
    mind)
      # First // comment or fn declaration
      grep -m1 -E '^\s*(//|fn |module )' "$file" 2>/dev/null | sed 's|^[[:space:]]*//[[:space:]]*||' | utf8_trunc 120 || true
      ;;
    md)
      # First heading
      grep -m1 -E '^#' "$file" 2>/dev/null | sed 's/^#\+[[:space:]]*//' | utf8_trunc 120 || true
      ;;
    json)
      # Just say what it is
      local keys
      keys=$(python3 -c "import json,sys; d=json.load(open('$file')); print(', '.join(list(d.keys())[:5]))" 2>/dev/null || echo "")
      [ -n "$keys" ] && echo "Keys: $keys" || true
      ;;
    toml|yaml|yml)
      grep -m1 -E '^\[|^[a-z]' "$file" 2>/dev/null | utf8_trunc 120 || true
      ;;
    sh|bash|zsh)
      grep -m1 -E '^#\s+\w' "$file" 2>/dev/null | sed 's/^#[[:space:]]*//' | utf8_trunc 120 || true
      ;;
    *)
      head -1 "$file" 2>/dev/null | utf8_trunc 80 || true
      ;;
  esac
}

# Get file category for grouping
file_category() {
  local file="$1"
  local ext="${file##*.}"
  case "$ext" in
    py|rs|ts|js|tsx|jsx|go|c|cpp|h|java|swift|kt|mind) echo "source" ;;
    md|txt|rst) echo "docs" ;;
    json|toml|yaml|yml|ini|cfg|env) echo "config" ;;
    sh|bash|zsh) echo "scripts" ;;
    test.*|spec.*|*_test.*) echo "tests" ;;
    *) echo "other" ;;
  esac
}

# Size bucket for quick scanning
size_bucket() {
  local tokens="$1"
  if [ "$tokens" -lt 50 ]; then echo "tiny"
  elif [ "$tokens" -lt 200 ]; then echo "small"
  elif [ "$tokens" -lt 500 ]; then echo "medium"
  elif [ "$tokens" -lt 1500 ]; then echo "large"
  else echo "huge"
  fi
}

# Non-text files are indexed by path and size only. Copying their bytes
# makes ANATOMY.md invalid UTF-8. A NUL or a byte sequence that is not
# UTF-8 is the test. The extension denylist in filter_files does not see
# every fixture.
is_non_text() {
  local file="$1"
  if [ ! -s "$file" ]; then
    return 1
  fi
  # A NUL pattern cannot go through grep: an empty pattern matches every
  # file. Exit 0 when the bytes must not be copied into the index.
  python3 -c 'import sys
data = open(sys.argv[1], "rb").read()
if b"\0" in data:
    sys.exit(0)
try:
    data.decode("utf-8")
except UnicodeDecodeError:
    sys.exit(0)
sys.exit(1)' "$file"
}

# First N Unicode scalar values. head -c cuts bytes and can split a
# multibyte character, which makes the index invalid UTF-8.
utf8_trunc() {
  local n="$1"
  python3 -c 'import sys
n = int(sys.argv[1])
text = sys.stdin.buffer.read().decode("utf-8", "replace")
text = text.replace("\r", "").replace("\n", "")
sys.stdout.write(text[:n])' "$n"
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --update) UPDATE_MODE=true; shift ;;
    --depth) MAX_DEPTH="$2"; shift 2 ;;
    --exclude) EXCLUDE_PATTERNS+=("$2"); shift 2 ;;
    --output|-o) OUTPUT_FILE="$2"; shift 2 ;;
    --version) echo "anatomy $VERSION"; exit 0 ;;
    --help|-h)
      echo "anatomy $VERSION — Generate ANATOMY.md for coding agents"
      echo ""
      echo "Usage: anatomy [dir] [options]"
      echo ""
      echo "Options:"
      echo "  --update        Only update changed files (preserve manual edits)"
      echo "  --depth N       Max directory depth (default: 4)"
      echo "  --exclude PAT   Exclude glob pattern (repeatable)"
      echo "  --output FILE   Output file (default: ANATOMY.md)"
      echo "  --version       Show version"
      echo "  --help          Show this help"
      exit 0
      ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) TARGET_DIR="$1"; shift ;;
  esac
done

cd "$TARGET_DIR"

# Collect files (respect .gitignore if in a git repo)
#
# TRACKED FILES ONLY, deliberately. This once enumerated `--cached --others
# --exclude-standard`, i.e. untracked-but-not-ignored files too. ANATOMY.md is
# COMMITTED and PUBLIC, and it transcribes each indexed file's name and first
# meaningful line -- so any private, uncommitted note that happened to be
# sitting in the checkout when the index was regenerated was copied verbatim
# into a public document. That is exactly what happened: two local handoff
# notes, one of them naming a model as this compiler's owner, entered
# ANATOMY.md from a checkout that had them lying around untracked.
#
# The fix is at the input set, not at the output filter: with `--cached` the
# generator structurally cannot see a file that is not already public, so no
# amount of local scratch in a working tree can leak through it. A consequence
# worth stating: a NEW file must be `git add`ed before it appears in the index.
# That is the intended trade -- an index that lags a brand-new file by one
# `git add` is strictly safer than one that publishes private notes.
# Guarded by scripts/test_no_ai_attribution.py.
collect_files() {
  if git rev-parse --is-inside-work-tree &>/dev/null; then
    git ls-files --cached 2>/dev/null
  else
    find . -maxdepth "$MAX_DEPTH" -type f \
      -not -path '*/.git/*' \
      -not -path '*/node_modules/*' \
      -not -path '*/.venv/*' \
      -not -path '*/__pycache__/*' \
      -not -path '*/target/*' \
      -not -path '*/.wolf/*' \
      -not -path '*/dist/*' \
      | sed 's|^\./||' | sort
  fi
}

# Filter out excluded patterns and binary files
filter_files() {
  while IFS= read -r file; do
    # Skip the output file itself
    [ "$file" = "$OUTPUT_FILE" ] && continue
    
    # Skip binary files
    case "$file" in
      *.png|*.jpg|*.jpeg|*.gif|*.ico|*.woff|*.woff2|*.ttf|*.eot|*.svg|*.mp3|*.mp4|*.zip|*.tar|*.gz|*.bz2|*.xz|*.pyc|*.pyo|*.so|*.dylib|*.dll|*.exe|*.o|*.a|*.wasm|*.db|*.sqlite|*.lock|*.mic3) continue ;;
    esac
    
    # Skip excluded patterns
    local skip=false
    for pat in "${EXCLUDE_PATTERNS[@]+"${EXCLUDE_PATTERNS[@]}"}"; do
      if [[ "$file" == $pat ]]; then
        skip=true
        break
      fi
    done
    $skip && continue
    
    # Skip files larger than 100KB (likely generated)
    local size
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    [ "$size" -gt 102400 ] && continue
    
    echo "$file"
  done
}

# Build the anatomy
# Sort: root files first (no /), then by directory path
files=$(collect_files | filter_files | awk '{
  n = split($0, parts, "/")
  if (n == 1) printf "0\t%s\n", $0
  else printf "1\t%s\n", $0
}' | sort -t$'\t' -k1,1 -k2,2 | cut -f2)
total_files=0
total_tokens=0

# Get project name.
#
# NOT basename(pwd): ANATOMY.md is committed, so the checkout DIRECTORY name
# would end up in a public artifact and would differ between the main clone and
# every `git worktree` (`mind` vs `fix-wt-<branch>`), making a regeneration from
# a worktree look like a real content change and leaking local scratch paths.
# The remote's repository name is the same from every checkout of the same repo,
# which is the property the field actually wants. basename(pwd) stays as the
# fallback so this remains usable in a repo with no remote, or outside git.
project_name=$(basename -s .git "$(git config --get remote.origin.url 2>/dev/null)" 2>/dev/null || true)
[ -z "$project_name" ] && project_name=$(basename "$(pwd)")

# Count stats first
declare -A dir_tokens
declare -A dir_files

while IFS= read -r file; do
  [ -z "$file" ] && continue
  [ ! -f "$file" ] && continue
  
  tokens=$(estimate_tokens "$file")
  dir=$(dirname "$file")
  
  dir_tokens["$dir"]=$(( ${dir_tokens["$dir"]:-0} + tokens ))
  dir_files["$dir"]=$(( ${dir_files["$dir"]:-0} + 1 ))
  
  total_files=$((total_files + 1))
  total_tokens=$((total_tokens + tokens))
done <<< "$files"

# Generate output
{
  cat << 'EOF'
# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

EOF

  echo "**Project:** \`$project_name\`"
  echo "**Files:** $total_files | **Est. tokens:** ~$(printf "%'d" $total_tokens)"
  echo "**Generated:** $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo ""
  
  # Token budget guidance
  echo "## Token Budget Guide"
  echo ""
  echo "| Size | Tokens | Read strategy |"
  echo "|------|--------|---------------|"
  echo "| tiny | <50 | Always safe to read |"
  echo "| small | 50-200 | Read freely |"
  echo "| medium | 200-500 | Read if relevant |"
  echo "| large | 500-1500 | Use summary first, read specific sections |"
  echo "| huge | >1500 | Avoid full read — use grep or read specific lines |"
  echo ""
  
  # Directory overview
  echo "## Directory Overview"
  echo ""
  echo "| Directory | Files | Est. tokens |"
  echo "|-----------|-------|-------------|"
  for dir in $(echo "${!dir_tokens[@]}" | tr ' ' '\n' | sort); do
    printf "| \`%s/\` | %d | ~%s |\n" "$dir" "${dir_files[$dir]}" "$(printf "%'d" ${dir_tokens[$dir]})"
  done
  echo ""
  
  # File listing grouped by directory
  echo "## Files"
  echo ""
  
  current_dir=""
  while IFS= read -r file; do
    [ -z "$file" ] && continue
    [ ! -f "$file" ] && continue
    
    dir=$(dirname "$file")
    base=$(basename "$file")

    # New directory header
    if [ "$dir" != "$current_dir" ]; then
      current_dir="$dir"
      echo "### \`$dir/\`"
      echo ""
    fi

    if is_non_text "$file"; then
      bin_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
      echo "- \`$base\` (binary, ${bin_size} bytes)"
      continue
    fi

    tokens=$(estimate_tokens "$file")
    bucket=$(size_bucket "$tokens")
    summary=$(extract_summary "$file")
    
    # File entry
    if [ -n "$summary" ]; then
      echo "- \`$base\` (~${tokens} tok, $bucket) — $summary"
    else
      echo "- \`$base\` (~${tokens} tok, $bucket)"
    fi
    
  done <<< "$files"
  
  echo ""
  echo "---"
  echo "*Generated by \`anatomy $VERSION\`. Edit descriptions manually — re-run preserves structure.*"
  
} > "$OUTPUT_FILE"

echo "✓ $OUTPUT_FILE: $total_files files, ~$(printf "%'d" $total_tokens) estimated tokens"
