#!/usr/bin/env python3
"""Build deterministic WOFF2 browser fonts from canonical TTF inputs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

try:
    import fontTools
    from fontTools import subset
    from fontTools.ttLib import TTFont
    from fontTools.ttLib.woff2 import compress
except ImportError as exc:  # pragma: no cover - exercised in minimal installs
    raise SystemExit(
        "fonttools[woff] is required; install the repository dev extra"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = REPO_ROOT / "docs/brand/assets/fonts"
MANIFEST = FONT_DIR / "manifest.json"


def _parse_unicode_ranges(ranges: list[str]) -> set[int]:
    codepoints: set[int] = set()
    for value in ranges:
        if not value.startswith("U+"):
            raise ValueError(f"invalid Unicode range {value!r}")
        bounds = value[2:].split("-", 1)
        start = int(bounds[0], 16)
        end = int(bounds[-1], 16)
        if end < start:
            raise ValueError(f"invalid Unicode range {value!r}")
        codepoints.update(range(start, end + 1))
    return codepoints


def _subset_newsreader(
    source: Path,
    output: Path,
    settings: dict[str, object],
) -> None:
    options = subset.Options()
    options.flavor = "woff2"
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.name_legacy = True
    options.name_languages = ["*"]
    options.glyph_names = True
    options.symbol_cmap = True
    options.legacy_cmap = True
    options.notdef_glyph = True
    options.notdef_outline = True
    options.recommended_glyphs = True
    options.hinting = True
    options.canonical_order = True
    options.recalc_timestamp = False

    font = TTFont(source, recalcTimestamp=False)
    codepoints = _parse_unicode_ranges(settings["unicode_ranges"])  # type: ignore[arg-type]
    missing = sorted(codepoints - set(font.getBestCmap()))
    if missing:
        rendered = ", ".join(f"U+{value:04X}" for value in missing)
        raise ValueError(f"{source.name} lacks requested subset glyphs: {rendered}")
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=codepoints)
    subsetter.subset(font)
    font.flavor = "woff2"
    font.save(output, reorderTables=True)


def _build(output_dir: Path) -> list[Path]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_version = manifest["conversion"]["version"]
    if fontTools.__version__ != expected_version:
        raise RuntimeError(
            f"fontTools {expected_version} is required, found {fontTools.__version__}"
        )
    outputs = []
    for item in manifest["files"]:
        source = FONT_DIR / item["file"]
        output = output_dir / item["web_file"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if item.get("web_subset") == "openmed_latin":
            _subset_newsreader(
                source,
                output,
                manifest["conversion"]["newsreader_subset"],
            )
        else:
            compress(str(source), str(output))
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.write:
        _build(FONT_DIR)
        print("built canonical WOFF2 fonts")
        return 0

    with tempfile.TemporaryDirectory(prefix="openmed-web-fonts-") as temp:
        outputs = _build(Path(temp))
        stale = [
            output.name
            for output in outputs
            if (FONT_DIR / output.name).read_bytes() != output.read_bytes()
        ]
    if stale:
        print(
            "stale WOFF2 fonts: "
            + ", ".join(stale)
            + "; run python scripts/brand/build_web_fonts.py --write",
            file=sys.stderr,
        )
        return 1
    print("WOFF2 fonts reproduce exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
