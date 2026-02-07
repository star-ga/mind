#!/bin/bash
# Upload pre-built MIND runtime libraries to GitHub releases.
# Run after building mind-runtime with: cargo build --release --features eval
#
# Usage: ./scripts/release-runtime.sh [version]
# Example: ./scripts/release-runtime.sh 0.1.9
#
# Requires: gh CLI (authenticated)

set -e

VERSION="${1:-$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')}"
TAG="v${VERSION}"
RUNTIME_DIR="../mind-runtime/target/release"

echo "Releasing MIND runtime v${VERSION}"
echo "Tag: ${TAG}"
echo ""

# Check gh is installed
command -v gh >/dev/null 2>&1 || { echo "Error: gh CLI is required. Install: https://cli.github.com"; exit 1; }

# Check runtime builds exist
LINUX_SO="${RUNTIME_DIR}/libmind_runtime.so"
if [ ! -f "$LINUX_SO" ]; then
    echo "Error: Runtime .so not found at ${LINUX_SO}"
    echo "Build with: cd ../mind-runtime && cargo build --release --features eval"
    exit 1
fi

# Copy with canonical naming
DIST_DIR="/tmp/mind-runtime-dist"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

cp "$LINUX_SO" "$DIST_DIR/libmind_cpu_linux-x64.so"
echo "Prepared: libmind_cpu_linux-x64.so ($(du -h "$DIST_DIR/libmind_cpu_linux-x64.so" | cut -f1))"

# Create or update GitHub release
if gh release view "$TAG" >/dev/null 2>&1; then
    echo "Release ${TAG} exists, uploading assets..."
    gh release upload "$TAG" "$DIST_DIR/libmind_cpu_linux-x64.so" --clobber
else
    echo "Creating release ${TAG}..."
    gh release create "$TAG" \
        --title "MIND ${TAG}" \
        --notes "MIND Language v${VERSION}. See CHANGELOG.md for details." \
        "$DIST_DIR/libmind_cpu_linux-x64.so"
fi

echo ""
echo "Done! Users can install with:"
echo "  curl -fsSL https://mindlang.dev/install.sh | sh"
echo ""
echo "Direct download:"
echo "  https://github.com/star-ga/mind/releases/download/${TAG}/libmind_cpu_linux-x64.so"
