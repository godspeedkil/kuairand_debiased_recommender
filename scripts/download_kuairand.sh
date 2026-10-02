#!/usr/bin/env bash
# Download and verify a KuaiRand release into data/raw/.
# Usage: scripts/download_kuairand.sh [Pure|1K|27K]   (default: Pure)
# Source: https://github.com/chongminggao/KuaiRand
set -euo pipefail

VERSION="${1:-Pure}"
case "$VERSION" in
  Pure) MD5="0820331067a3784d9691136f772b35a7" ;;
  1K)   MD5="6b0b9c8222d67fcd4c676218edca3f1f" ;;
  27K)  MD5="3e3c799a24e2d23a4d2c757fbf9adf59" ;;
  *) echo "unknown version: $VERSION (expected Pure, 1K or 27K)"; exit 1 ;;
esac

ARCHIVE="KuaiRand-${VERSION}.tar.gz"
URL="https://zenodo.org/records/10439422/files/${ARCHIVE}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"
cd "$DEST"

if [ ! -f "$ARCHIVE" ]; then
  echo "Downloading $URL"
  curl -L --fail --retry 3 -o "$ARCHIVE.part" "$URL"
  mv "$ARCHIVE.part" "$ARCHIVE"
fi

echo "Verifying checksum"
if command -v md5sum >/dev/null; then ACTUAL=$(md5sum "$ARCHIVE" | cut -d' ' -f1)
else ACTUAL=$(md5 -q "$ARCHIVE"); fi
if [ "$ACTUAL" != "$MD5" ]; then
  echo "Checksum mismatch: expected $MD5, got $ACTUAL"; exit 1
fi

tar -xzf "$ARCHIVE"
echo "Extracted to $DEST/KuaiRand-${VERSION}"
