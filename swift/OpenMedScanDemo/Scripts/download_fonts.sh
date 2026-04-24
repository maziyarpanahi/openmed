#!/usr/bin/env bash
# Download OFL-licensed custom fonts for OpenMedScanDemo.
#
# Google's fonts repo now ships variable fonts (weight/opsz axes baked in)
# rather than static cuts. iOS supports variable fonts via Core Text, and
# SwiftUI's `.weight()` modifier picks the right instance at render time,
# so six files cover every weight we need.
#
# Output: OpenMedScanDemo/Fonts/*.ttf + OpenMedScanDemo/Fonts/LICENSES/*.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FONTS_DIR="$PROJECT_ROOT/OpenMedScanDemo/Fonts"
LICENSES_DIR="$FONTS_DIR/LICENSES"

mkdir -p "$FONTS_DIR" "$LICENSES_DIR"

BASE="https://raw.githubusercontent.com/google/fonts/main/ofl"

# Percent-encoded because the upstream filenames contain square brackets.
# local name                       upstream path (url-encoded)
FONT_FILES=(
    "Newsreader.ttf                 newsreader/Newsreader%5Bopsz%2Cwght%5D.ttf"
    "Newsreader-Italic.ttf          newsreader/Newsreader-Italic%5Bopsz%2Cwght%5D.ttf"
    "InterTight.ttf                 intertight/InterTight%5Bwght%5D.ttf"
    "InterTight-Italic.ttf          intertight/InterTight-Italic%5Bwght%5D.ttf"
    "JetBrainsMono.ttf              jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf"
    "JetBrainsMono-Italic.ttf       jetbrainsmono/JetBrainsMono-Italic%5Bwght%5D.ttf"
)

LICENSE_FILES=(
    "Newsreader-OFL.txt         newsreader/OFL.txt"
    "InterTight-OFL.txt         intertight/OFL.txt"
    "JetBrainsMono-OFL.txt      jetbrainsmono/OFL.txt"
)

MIN_FONT_SIZE=10000
MIN_LICENSE_SIZE=2000

download_file() {
    local label="$1" upstream="$2" out="$3" min_size="$4"
    local url="$BASE/$upstream"

    echo "-> $label"
    curl --fail --location --silent --show-error "$url" --output "$out.tmp"

    local bytes
    bytes=$(wc -c < "$out.tmp" | tr -d ' ')
    if [ "$bytes" -lt "$min_size" ]; then
        echo "   FAIL: $out (only $bytes bytes, expected >$min_size)" >&2
        rm -f "$out.tmp"
        exit 1
    fi
    mv "$out.tmp" "$out"
    printf "   OK  %-32s  %10d bytes\n" "$label" "$bytes"
}

echo "==> Fonts"
for entry in "${FONT_FILES[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    name="$1"
    upstream="$2"
    download_file "$name" "$upstream" "$FONTS_DIR/$name" "$MIN_FONT_SIZE"
done

echo
echo "==> Licenses"
for entry in "${LICENSE_FILES[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    name="$1"
    upstream="$2"
    download_file "$name" "$upstream" "$LICENSES_DIR/$name" "$MIN_LICENSE_SIZE"
done

echo
echo "Done. Fonts written to $FONTS_DIR"
