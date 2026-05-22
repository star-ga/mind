#!/usr/bin/env sh
# MIND compiler (mindc) installer — downloads a pre-built binary from the
# GitHub Releases page, verifies its SHA-256, and installs it on PATH.
#
#   curl -sSL https://mindlang.dev/install.sh | sh
#
# No Rust toolchain required. Asset scheme matches .github/workflows/release.yml:
#   mindc-linux-x64.tar.gz / mindc-linux-arm64.tar.gz
#   mindc-macos-x64.tar.gz / mindc-macos-arm64.tar.gz
#   mindc-windows-x64.zip   (Windows users: use install.ps1 instead)
# Each asset has a sibling <asset>.sha256 sidecar.
#
# Environment overrides:
#   MIND_INSTALL_DIR   target dir (default: /usr/local/bin or ~/.local/bin)
#   MIND_VERSION       pin a specific tag (default: latest release)
#   MIND_DRY_RUN=1     print what would happen, write nothing

set -eu

REPO="star-ga/mind"
DRY_RUN="${MIND_DRY_RUN:-0}"

info()  { printf '\033[0;36m[INFO]\033[0m %s\n' "$1"; }
ok()    { printf '\033[0;32m[OK]\033[0m %s\n' "$1"; }
err()   { printf '\033[0;31m[ERROR]\033[0m %s\n' "$1" >&2; exit 1; }

# -- Detect OS + arch -> release.yml asset name ---------------------------
os_raw="$(uname -s)"
arch_raw="$(uname -m)"

case "$os_raw" in
    Linux)  os="linux" ;;
    Darwin) os="macos" ;;
    *) err "unsupported OS '$os_raw' -- Windows users run install.ps1" ;;
esac

case "$arch_raw" in
    x86_64|amd64)  arch="x64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) err "unsupported architecture '$arch_raw'" ;;
esac

asset="mindc-${os}-${arch}.tar.gz"

# -- Resolve the release tag ----------------------------------------------
if [ -n "${MIND_VERSION:-}" ]; then
    tag="$MIND_VERSION"
    info "Using pinned version: $tag"
else
    info "Fetching latest release tag from GitHub..."
    tag="$(
        curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name"' | head -1 | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/'
    )"
    [ -n "$tag" ] || err "could not resolve latest release tag"
    info "Latest release: $tag"
fi

base="https://github.com/${REPO}/releases/download/${tag}"
asset_url="${base}/${asset}"
sha_url="${asset_url}.sha256"

# -- Choose install dir ---------------------------------------------------
if [ -n "${MIND_INSTALL_DIR:-}" ]; then
    install_dir="$MIND_INSTALL_DIR"
elif [ -w /usr/local/bin ] 2>/dev/null; then
    install_dir="/usr/local/bin"
else
    install_dir="${HOME}/.local/bin"
fi

info "Detected: OS=${os} ARCH=${arch} asset=${asset}"
info "Asset URL : ${asset_url}"
info "Install to: ${install_dir}/mindc"

if [ "$DRY_RUN" = "1" ]; then
    printf '\n[dry-run] Would download: %s\n' "$asset_url"
    printf '[dry-run] Would verify  : %s\n' "$sha_url"
    printf '[dry-run] Would install : %s/mindc\n' "$install_dir"
    printf '[dry-run] No files written.\n'
    exit 0
fi

# -- Download + verify + extract ------------------------------------------
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

info "Downloading $asset ..."
curl -fsSL "$asset_url" -o "${tmp}/${asset}" || err "download failed: $asset_url"

info "Verifying SHA-256 ..."
if curl -fsSL "$sha_url" -o "${tmp}/${asset}.sha256" 2>/dev/null; then
    expected="$(awk '{print $1}' "${tmp}/${asset}.sha256")"
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "${tmp}/${asset}" | awk '{print $1}')"
    else
        actual="$(shasum -a 256 "${tmp}/${asset}" | awk '{print $1}')"
    fi
    [ "$expected" = "$actual" ] || err "SHA-256 mismatch (expected $expected, got $actual)"
    ok "SHA-256 verified"
else
    info "no .sha256 sidecar published for this asset -- skipping verification"
fi

info "Extracting ..."
tar -xzf "${tmp}/${asset}" -C "$tmp" || err "extraction failed"

# The tarball contains the bare 'mindc' binary.
[ -f "${tmp}/mindc" ] || err "expected 'mindc' inside ${asset} but it was not found"

mkdir -p "$install_dir"
install -m 0755 "${tmp}/mindc" "${install_dir}/mindc" 2>/dev/null \
    || { cp "${tmp}/mindc" "${install_dir}/mindc" && chmod 0755 "${install_dir}/mindc"; }

ok "mindc ${tag} installed to ${install_dir}/mindc"

case ":${PATH}:" in
    *":${install_dir}:"*) ;;
    *) info "Add ${install_dir} to your PATH:  export PATH=\"${install_dir}:\$PATH\"" ;;
esac

"${install_dir}/mindc" --version 2>/dev/null || true
