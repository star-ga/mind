#!/usr/bin/env sh
# install.sh — mindc one-line installer for Linux and macOS
#
# Usage:
#   curl -sSL https://mindlang.dev/install.sh | sh
#   sh install.sh
#   sh install.sh --dry-run        # print what would happen, do nothing
#
# Environment overrides:
#   MINDC_VERSION   pin a specific release tag, e.g. MINDC_VERSION=v0.6.9
#   MINDC_INSTALL_DIR  override install destination
#
# The script:
#   1. Detects OS + architecture
#   2. Fetches the latest (or pinned) release tag from GitHub
#   3. Downloads the appropriate asset + SHA256SUMS
#   4. Verifies the SHA256 checksum
#   5. Installs to /usr/local/bin (or ~/.local/bin if not writable)

set -e

REPO="star-ga/mind"
BINARY="mindc"

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

say() { printf '%s\n' "$*"; }
err() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || err "required tool not found: $1"; }

# ── Detect OS and architecture ────────────────────────────────────────────────

detect_platform() {
  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "$OS" in
    Linux)
      case "$ARCH" in
        x86_64|amd64) ASSET_SUFFIX="x86_64-unknown-linux-musl" ;;
        *) err "Unsupported Linux architecture: $ARCH. Download manually from https://github.com/${REPO}/releases" ;;
      esac
      ;;
    Darwin)
      # Universal binary works on both x86_64 and arm64
      ASSET_SUFFIX="x86_64-apple-darwin-universal"
      ;;
    *)
      err "Unsupported OS: $OS. For Windows use install.ps1, or download manually from https://github.com/${REPO}/releases"
      ;;
  esac

  say "Detected: OS=${OS}  ARCH=${ARCH}  asset suffix=${ASSET_SUFFIX}"
}

# ── Resolve version ───────────────────────────────────────────────────────────

resolve_version() {
  if [ -n "${MINDC_VERSION:-}" ]; then
    TAG="${MINDC_VERSION}"
    say "Pinned version: ${TAG}"
    return
  fi

  say "Fetching latest release tag from GitHub..."
  need curl

  LATEST_URL="https://api.github.com/repos/${REPO}/releases/latest"
  RESPONSE="$(curl -sSL --fail "${LATEST_URL}" 2>/dev/null)" || \
    err "Failed to fetch latest release from ${LATEST_URL}"

  # Extract tag_name without jq
  TAG="$(printf '%s' "${RESPONSE}" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"
  [ -n "${TAG}" ] || err "Could not parse tag_name from GitHub API response"

  say "Latest release: ${TAG}"
}

# ── Resolve install directory ─────────────────────────────────────────────────

resolve_install_dir() {
  if [ -n "${MINDC_INSTALL_DIR:-}" ]; then
    INSTALL_DIR="${MINDC_INSTALL_DIR}"
    return
  fi

  if [ -w "/usr/local/bin" ]; then
    INSTALL_DIR="/usr/local/bin"
  else
    INSTALL_DIR="${HOME}/.local/bin"
  fi
}

# ── Download + verify ─────────────────────────────────────────────────────────

download_and_verify() {
  VERSION="${TAG#v}"
  ASSET="${BINARY}-${TAG}-${ASSET_SUFFIX}"
  BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
  ASSET_URL="${BASE_URL}/${ASSET}"
  SUMS_URL="${BASE_URL}/SHA256SUMS"

  say "Asset URL : ${ASSET_URL}"
  say "Install to: ${INSTALL_DIR}/${BINARY}"

  if [ "${DRY_RUN}" -eq 1 ]; then
    say ""
    say "[dry-run] Would download: ${ASSET_URL}"
    say "[dry-run] Would verify  : ${SUMS_URL}"
    say "[dry-run] Would install : ${INSTALL_DIR}/${BINARY}"
    say "[dry-run] No files written."
    return
  fi

  TMPDIR_OWN="$(mktemp -d)"
  trap 'rm -rf "${TMPDIR_OWN}"' EXIT INT TERM

  say "Downloading ${ASSET}..."
  curl -sSL --fail --progress-bar "${ASSET_URL}" -o "${TMPDIR_OWN}/${ASSET}" || \
    err "Download failed: ${ASSET_URL}"

  say "Downloading SHA256SUMS..."
  curl -sSL --fail "${SUMS_URL}" -o "${TMPDIR_OWN}/SHA256SUMS" || \
    err "Download failed: ${SUMS_URL}"

  say "Verifying checksum..."
  cd "${TMPDIR_OWN}"

  # Grab only the line for our asset from the manifest
  EXPECTED="$(grep " ${ASSET}$" SHA256SUMS || true)"
  [ -n "${EXPECTED}" ] || err "No checksum entry for ${ASSET} in SHA256SUMS"

  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s\n' "${EXPECTED}" | sha256sum --check --status || \
      err "SHA256 verification FAILED for ${ASSET}"
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s\n' "${EXPECTED}" | shasum -a 256 --check --status || \
      err "SHA256 verification FAILED for ${ASSET}"
  else
    err "Neither sha256sum nor shasum found — cannot verify checksum"
  fi

  say "Checksum OK."
  cd - >/dev/null

  # ── Install ────────────────────────────────────────────────────────────────

  mkdir -p "${INSTALL_DIR}"
  cp "${TMPDIR_OWN}/${ASSET}" "${INSTALL_DIR}/${BINARY}"
  chmod +x "${INSTALL_DIR}/${BINARY}"

  say "Installed: ${INSTALL_DIR}/${BINARY}"
  say ""

  # PATH advisory if ~/.local/bin was used
  if printf '%s' "${INSTALL_DIR}" | grep -q "\.local/bin"; then
    case ":${PATH}:" in
      *":${INSTALL_DIR}:"*) ;;
      *)
        say "NOTE: ${INSTALL_DIR} is not in your PATH."
        say "Add the following to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
        say "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
        ;;
    esac
  fi

  say "Run: ${BINARY} --version"
  "${INSTALL_DIR}/${BINARY}" --version || true
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
  say "=== mindc installer ==="
  detect_platform
  resolve_version
  resolve_install_dir
  download_and_verify
}

main "$@"
