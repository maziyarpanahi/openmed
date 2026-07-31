#!/usr/bin/env python3
"""Validate the repository-owned OpenMed brand system and its consumers."""

from __future__ import annotations

import datetime as dt
import hashlib
import html as html_lib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRAND_ROOT = REPO_ROOT / "docs/brand"
ASSET_ROOT = BRAND_ROOT / "assets"
TOKEN_ROOT = BRAND_ROOT / "system/tokens"

REQUIRED_HANDOFF_HASHES = {
    "SKILL.md": "a0916f07025494456f0c83d51af70560c967233a081c91d30519d1b7099c7f58",
    "readme.md": "253a2e9dfcbc2dc435f3c7c821117811b2bf10bb3626e55956c8c1ede896c4d4",
    "tokens/colors.css": (
        "fc010d24c177cbeeeb95d6847c292814323cbc9aa2f2dba8a7322243e8bf4789"
    ),
    "tokens/typography.css": (
        "e06a501ae0a754683b29348dab6b5a46e5c139dd994d884fb16479497a8a0778"
    ),
    "tokens/surfaces.css": (
        "acce3d9551fa4cd045bf77e515c2bfcd714f4e3f8000ad96d4200544e15856a2"
    ),
    "assets/logo.svg": (
        "43d553efb499c37f97e0752b0a9bed61b585edde983e4a8b35119442f8cbcf80"
    ),
    (
        "openmed.life/design_handoff_landing_redesign/OpenMed.life Landing v2.dc.html"
    ): "a2a1a321a25ef9154a8a6a275ba6d1c80037459501baf5b4c1dbcd4534df8a4b",
    "openmed.life/design_handoff_landing_redesign/README.md": (
        "be6d328d5a8c02de453b7203ae9bf88a45f12d56cbc6f01fb2bc68ee8e4e3716"
    ),
    "openmed.life/docs/docs.css": (
        "15ccf320178ec4febdfc399a3e404cead97cbfcffc8114c8b17ae8459e194944"
    ),
    "openmed.life/docs/docs.js": (
        "048620241fa3f3b5032b0c929bfb51464584fa2efcd6504a96508f2168b748dc"
    ),
    "OpenMed Social Cards.dc.html": (
        "a4d597baa83e71d8601d02d4ae9b2441b50cb69871887045d554c146384fe3d9"
    ),
    "social/README.md": (
        "e28ac10689d4ae392509bad8a7102fba63d3895ab1c6d862ff2e13bb5d5178a0"
    ),
}

REQUIRED_FILES = (
    "docs/brand/README.md",
    "docs/brand/MASCOT_BRIEF.md",
    "docs/brand/BRAND_SYSTEM_ROLLOUT_PLAN.md",
    "docs/brand/system/asset-register.md",
    "docs/brand/system/CHANGELOG.md",
    "docs/brand/system/claims.yml",
    "docs/brand/system/deprecation.md",
    "docs/brand/system/evidence/github-repository.json",
    "docs/brand/system/evidence/manual-accessibility-review.json",
    "docs/brand/system/evidence/social-visual-review.json",
    "docs/brand/system/handoff-provenance.json",
    "docs/brand/system/iconography.md",
    "docs/brand/system/ownership.md",
    "docs/brand/system/site-exceptions.md",
    "docs/brand/system/tokens.json",
    "docs/brand/system/version.json",
    "docs/brand/system/voice.md",
    "docs/brand/assets/open-cross.svg",
    "docs/brand/assets/open-cross-inverse.svg",
    "docs/brand/assets/openmed-wordmark.svg",
    "docs/brand/assets/cat-crest.png",
    "docs/brand/social/_src/exports.json",
    "docs/brand/social/_src/profile-copy.json",
    "docs/brand/social/_src/README.md",
    "docs/brand/social/exports/README.md",
    "docs/brand/social/PLATFORM_CUTOVER_RUNBOOK.md",
    "docs/brand/social/manifest.json",
    "docs/stylesheets/openmed-system.css",
    "docs/website/assets/openmed-system.css",
)

GENERATOR_CHECKS = (
    ("scripts/brand/update_claims.py", "--check"),
    ("scripts/brand/update_readme_brand.py", "--check"),
    ("scripts/brand/build_web_fonts.py", "--check"),
    ("scripts/brand/sync_consumers.py", "--check"),
    ("scripts/brand/render_social_assets.py", "--check"),
    ("scripts/i18n/check_readme_drift.py", ""),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256(path: Path) -> str:
    """Hash text provenance independently of checkout line-ending policy."""

    data = path.read_bytes()
    if path.suffix.lower() in {
        ".css",
        ".html",
        ".js",
        ".json",
        ".md",
        ".svg",
        ".txt",
        ".yaml",
        ".yml",
    }:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _load_json(relative: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def _plain_html_text(fragment: str) -> str:
    without_comments = re.sub(r"<!--.*?-->", " ", fragment, flags=re.DOTALL)
    without_tags = re.sub(r"<[^>]+>", " ", without_comments)
    return " ".join(html_lib.unescape(without_tags).split())


def _validate_faq_parity(
    website: str,
    faq_page: dict[str, Any],
    errors: list[str],
) -> None:
    section_match = re.search(
        r'<section\b[^>]*\bid="faq"[^>]*>(.*?)</section>',
        website,
        re.DOTALL,
    )
    if section_match is None:
        errors.append("website visible FAQ section is missing")
        return

    visible: dict[str, str] = {}
    for article in re.findall(
        r'<article\b[^>]*class="faq-item"[^>]*>(.*?)</article>',
        section_match.group(1),
        re.DOTALL,
    ):
        question_match = re.search(
            r"<button\b[^>]*>.*?<span>(.*?)</span>.*?</button>",
            article,
            re.DOTALL,
        )
        answer_match = re.search(
            r'<div\b[^>]*id="faq-answer-\d+"[^>]*>(.*?)</div>',
            article,
            re.DOTALL,
        )
        if question_match is None or answer_match is None:
            errors.append("website visible FAQ item lacks a question or answer")
            continue
        question = _plain_html_text(question_match.group(1))
        if question in visible:
            errors.append(f"website visible FAQ repeats question {question!r}")
        visible[question] = _plain_html_text(answer_match.group(1))

    structured: dict[str, str] = {}
    entities = faq_page.get("mainEntity")
    if not isinstance(entities, list):
        errors.append("website FAQ JSON-LD mainEntity must be a list")
        return
    for entity in entities:
        if not isinstance(entity, dict):
            errors.append("website FAQ JSON-LD contains a non-object entity")
            continue
        question = entity.get("name")
        accepted = entity.get("acceptedAnswer")
        answer = accepted.get("text") if isinstance(accepted, dict) else None
        if not isinstance(question, str) or not isinstance(answer, str):
            errors.append("website FAQ JSON-LD entity lacks text question/answer")
            continue
        if question in structured:
            errors.append(f"website FAQ JSON-LD repeats question {question!r}")
        structured[question] = " ".join(answer.split())

    if set(visible) != set(structured):
        missing = sorted(set(structured) - set(visible))
        extra = sorted(set(visible) - set(structured))
        errors.append(
            "website visible FAQ and JSON-LD questions differ: "
            f"missing={missing}, extra={extra}"
        )
    for question in sorted(set(visible) & set(structured)):
        if visible[question] != structured[question]:
            errors.append(
                "website visible FAQ disagrees with JSON-LD for "
                f"{question!r}: visible={visible[question]!r}, "
                f"structured={structured[question]!r}"
            )


def _png_info(path: Path) -> tuple[int, int, int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG with an IHDR header")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    return width, height, bit_depth, color_type


def _png_chunks(path: Path) -> list[tuple[str, bytes]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    chunks: list[tuple[str, bytes]] = []
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8].decode("ascii")
        payload = data[offset + 8 : offset + 8 + length]
        chunks.append((kind, payload))
        offset += 12 + length
        if kind == "IEND":
            break
    if not chunks or chunks[-1][0] != "IEND" or offset != len(data):
        raise ValueError("PNG chunk stream is incomplete")
    return chunks


def _css_declarations(body: str) -> list[tuple[str, str]]:
    without_comments = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return [
        (match.group(1), match.group(2).strip())
        for match in re.finditer(
            r"(?m)^\s*(--[\w-]+|color-scheme):\s*([^;]+);",
            without_comments,
        )
    ]


def _css_block(text: str, selector: str, *, start: int = 0) -> str:
    selector_at = text.find(selector, start)
    if selector_at < 0:
        raise ValueError(f"missing CSS selector {selector!r}")
    opening = text.find("{", selector_at + len(selector))
    if opening < 0:
        raise ValueError(f"missing CSS body for {selector!r}")
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise ValueError(f"unterminated CSS body for {selector!r}")


def _parse_color(value: str) -> tuple[float, float, float, float]:
    value = value.strip()
    if match := re.fullmatch(r"#([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})?", value):
        raw = match.group(1)
        alpha = int(match.group(2), 16) / 255 if match.group(2) else 1.0
        return (
            int(raw[0:2], 16) / 255,
            int(raw[2:4], 16) / 255,
            int(raw[4:6], 16) / 255,
            alpha,
        )
    match = re.fullmatch(
        r"rgb\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        r"(?:\s*/\s*([\d.]+)%?)?\s*\)",
        value,
    )
    if not match:
        raise ValueError(f"unsupported color {value!r}")
    alpha_text = match.group(4)
    alpha = 1.0
    if alpha_text is not None:
        alpha = float(alpha_text)
        if "%" in value:
            alpha /= 100
    return (
        float(match.group(1)) / 255,
        float(match.group(2)) / 255,
        float(match.group(3)) / 255,
        alpha,
    )


def _composite(
    foreground: tuple[float, float, float, float],
    background: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    alpha = foreground[3] + background[3] * (1 - foreground[3])
    if alpha == 0:
        return (0, 0, 0, 0)
    return (
        *(
            (
                foreground[index] * foreground[3]
                + background[index] * background[3] * (1 - foreground[3])
            )
            / alpha
            for index in range(3)
        ),
        alpha,
    )


def _linear_channel(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _encoded_channel(channel: float) -> float:
    channel = max(0.0, min(1.0, channel))
    return (
        channel * 12.92
        if channel <= 0.0031308
        else 1.055 * channel ** (1 / 2.4) - 0.055
    )


def _relative_luminance(color: tuple[float, float, float, float]) -> float:
    red, green, blue = (_linear_channel(channel) for channel in color[:3])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first: str, second: str) -> float:
    first_color = _parse_color(first)
    second_color = _parse_color(second)
    if first_color[3] < 1:
        first_color = _composite(first_color, second_color)
    if second_color[3] < 1:
        raise ValueError("contrast background must be opaque")
    lighter, darker = sorted(
        (_relative_luminance(first_color), _relative_luminance(second_color)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _cube_root(value: float) -> float:
    """Return a real cube root on every supported Python version."""

    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def _srgb_to_oklch(value: str) -> tuple[float, float, float]:
    red, green, blue, _ = _parse_color(value)
    red, green, blue = map(_linear_channel, (red, green, blue))
    l_root = _cube_root(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue)
    m_root = _cube_root(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue)
    s_root = _cube_root(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue)
    lightness = 0.2104542553 * l_root + 0.793617785 * m_root - 0.0040720468 * s_root
    axis_a = 1.9779984951 * l_root - 2.428592205 * m_root + 0.4505937099 * s_root
    axis_b = 0.0259040371 * l_root + 0.7827717662 * m_root - 0.808675766 * s_root
    chroma = math.hypot(axis_a, axis_b)
    hue = math.atan2(axis_b, axis_a)
    return lightness, chroma, hue


def _oklch_to_srgb(
    lightness: float,
    chroma: float,
    hue: float,
    alpha: float,
) -> tuple[float, float, float, float]:
    axis_a = chroma * math.cos(hue)
    axis_b = chroma * math.sin(hue)
    l_root = lightness + 0.3963377774 * axis_a + 0.2158037573 * axis_b
    m_root = lightness - 0.1055613458 * axis_a - 0.0638541728 * axis_b
    s_root = lightness - 0.0894841775 * axis_a - 1.291485548 * axis_b
    l_value, m_value, s_value = l_root**3, m_root**3, s_root**3
    red = 4.0767416621 * l_value - 3.3077115913 * m_value + 0.2309699292 * s_value
    green = -1.2684380046 * l_value + 2.6097574011 * m_value - 0.3413193965 * s_value
    blue = -0.0041960863 * l_value - 0.7034186147 * m_value + 1.707614701 * s_value
    return (
        _encoded_channel(red),
        _encoded_channel(green),
        _encoded_channel(blue),
        alpha,
    )


def _contrast_rgba(
    foreground: tuple[float, float, float, float],
    background: str,
) -> float:
    background_color = _parse_color(background)
    if foreground[3] < 1:
        foreground = _composite(foreground, background_color)
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background_color)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _unicode_range_codepoints(ranges: list[str]) -> set[int]:
    codepoints: set[int] = set()
    for value in ranges:
        if not (match := re.fullmatch(r"U\+([0-9A-F]+)(?:-([0-9A-F]+))?", value)):
            raise ValueError(f"invalid Unicode range {value!r}")
        first = int(match.group(1), 16)
        last = int(match.group(2), 16) if match.group(2) else first
        codepoints.update(range(first, last + 1))
    return codepoints


def _normalized_image_comparison(
    current_path: Path,
    golden_path: Path,
) -> tuple[float, float]:
    from PIL import Image, ImageChops, ImageStat  # noqa: PLC0415

    with Image.open(current_path) as current_source:
        current_rgba = current_source.convert("RGBA")
        current = (
            Image.alpha_composite(
                Image.new("RGBA", current_rgba.size, "white"),
                current_rgba,
            )
            .convert("RGB")
            .resize((64, 64), Image.Resampling.LANCZOS)
        )
    with Image.open(golden_path) as golden_source:
        golden_rgba = golden_source.convert("RGBA")
        golden = (
            Image.alpha_composite(
                Image.new("RGBA", golden_rgba.size, "white"),
                golden_rgba,
            )
            .convert("RGB")
            .resize((64, 64), Image.Resampling.LANCZOS)
        )
    difference = ImageChops.difference(current, golden)
    root_mean_square = math.sqrt(
        sum(value**2 for value in ImageStat.Stat(difference).rms) / 3
    )
    normalized_rms = root_mean_square / 255

    current_gray = current.convert("L")
    golden_gray = golden.convert("L")
    current_pixels = list(current_gray.get_flattened_data())
    golden_pixels = list(golden_gray.get_flattened_data())
    current_mean = sum(current_pixels) / 4096
    golden_mean = sum(golden_pixels) / 4096
    current_bits = [value >= current_mean for value in current_pixels]
    golden_bits = [value >= golden_mean for value in golden_pixels]
    hash_distance = sum(a != b for a, b in zip(current_bits, golden_bits, strict=True))
    return normalized_rms, hash_distance / 4096


def _crop_pixel_hash(path: Path, box: list[int]) -> str:
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as source:
        cropped = source.convert("RGBA").crop(tuple(box))
        return hashlib.sha256(cropped.tobytes()).hexdigest()


def _validate_required_files(errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (REPO_ROOT / relative).is_file():
            errors.append(f"missing required brand file: {relative}")


def _validate_provenance(errors: list[str]) -> None:
    provenance = _load_json("docs/brand/system/handoff-provenance.json")
    reviewed = {item["path"]: item["sha256"] for item in provenance["reviewed_sources"]}
    if reviewed != REQUIRED_HANDOFF_HASHES:
        missing = sorted(set(REQUIRED_HANDOFF_HASHES) - set(reviewed))
        extra = sorted(set(reviewed) - set(REQUIRED_HANDOFF_HASHES))
        wrong = sorted(
            path
            for path in set(reviewed) & set(REQUIRED_HANDOFF_HASHES)
            if reviewed[path] != REQUIRED_HANDOFF_HASHES[path]
        )
        errors.append(
            "handoff provenance mismatch "
            f"(missing={missing}, extra={extra}, wrong_hash={wrong})"
        )

    for name, expected in provenance["approved_exports"].items():
        path = BRAND_ROOT / "social/exports" / name
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"approved export hash mismatch: {name}")
    export_readme = BRAND_ROOT / "social/exports/README.md"
    if (
        not export_readme.is_file()
        or _source_sha256(export_readme) != REQUIRED_HANDOFF_HASHES["social/README.md"]
    ):
        errors.append("approved social export README differs from the handoff")

    repository_inputs = provenance["repository_inputs"]
    for record in repository_inputs.values():
        path = REPO_ROOT / record["path"]
        if not path.is_file() or _source_sha256(path) != record["sha256"]:
            errors.append(f"repository input hash mismatch: {record['path']}")


def _validate_tokens(errors: list[str]) -> None:
    colors = (TOKEN_ROOT / "colors.css").read_text(encoding="utf-8")
    typography = (TOKEN_ROOT / "typography.css").read_text(encoding="utf-8")
    surfaces = (TOKEN_ROOT / "surfaces.css").read_text(encoding="utf-8")
    tokens = _load_json("docs/brand/system/tokens.json")

    if tokens["version"] != "2026.07.0":
        errors.append("tokens.json is not at the released 2026.07.0 contract")
    if tokens["color"]["accent_base"] != "#B0413E":
        errors.append("tokens.json accent base is not #B0413E")
    if colors.count("--om-accent-base: #B0413E;") != 1:
        errors.append("colors.css must define one #B0413E accent base")

    selectors = {
        "root": ":root",
        "light": '[data-theme="light"],\n[data-md-color-scheme="default"]',
        "dark": '[data-theme="dark"],\n[data-md-color-scheme="slate"]',
        "system_dark": (
            ":root:not([data-theme]):not([data-md-color-scheme]):not(\n"
            "      :has([data-theme], [data-md-color-scheme])\n"
            "    )"
        ),
    }
    try:
        bodies = {
            name: _css_block(colors, selector) for name, selector in selectors.items()
        }
    except ValueError as exc:
        errors.append(str(exc))
        bodies = {}

    if bodies:
        dark_declarations = _css_declarations(bodies["dark"])
        if dark_declarations != _css_declarations(bodies["system_dark"]):
            errors.append(
                "explicit dark and unforced operating-system dark declarations differ"
            )
        light_declarations = _css_declarations(bodies["light"])
        dark_properties = {name for name, _ in dark_declarations}
        light_properties = {name for name, _ in light_declarations}
        if dark_properties - light_properties:
            errors.append(
                "explicit Material/generic light theme does not reset "
                f"{sorted(dark_properties - light_properties)}"
            )
        if (
            "prefers-color-scheme: dark" not in colors
            or ":has([data-theme]" not in colors
        ):
            errors.append(
                "operating-system dark theme does not exclude explicit themes"
            )

        for theme, records in tokens["color"]["derived"].items():
            targets = ("root",) if theme == "light" else ("dark", "system_dark")
            for role, record in records.items():
                property_name = record["property"]
                if record["base_role"] != "accent_base":
                    errors.append(
                        f"derived role {theme}.{role} has an unknown base role"
                    )
                if record["color_space"] != "oklch" or record["hue"] != "source":
                    errors.append(
                        f"derived role {theme}.{role} is not source-hue OKLCH"
                    )
                if record["fallback"] != tokens["color"][theme].get(
                    f"{role}_fallback", record["fallback"]
                ):
                    errors.append(
                        f"derived role {theme}.{role} fallback is inconsistent"
                    )
                for target in targets:
                    values = [
                        value
                        for name, value in _css_declarations(bodies[target])
                        if name == property_name
                    ]
                    if values != [record["fallback"], record["css"]]:
                        errors.append(
                            f"{target} {property_name} must contain its exact "
                            "fallback followed by its relative-color enhancement"
                        )
                if theme == "light" and property_name in {
                    item["property"]
                    for item in tokens["color"]["derived"]["dark"].values()
                }:
                    values = [
                        value
                        for name, value in light_declarations
                        if name == property_name
                    ]
                    if values != [record["fallback"], record["css"]]:
                        errors.append(
                            f"explicit light {property_name} does not reset its dark value"
                        )

    expected_css_values = {
        "--om-bg": (
            tokens["color"]["light"]["background"],
            tokens["color"]["dark"]["background"],
        ),
        "--om-surface": (
            tokens["color"]["light"]["surface"],
            tokens["color"]["dark"]["surface"],
        ),
        "--om-ink": (
            tokens["color"]["light"]["ink"],
            tokens["color"]["dark"]["ink"],
        ),
        "--om-body": (
            tokens["color"]["light"]["body"],
            tokens["color"]["dark"]["body"],
        ),
        "--om-line": (
            tokens["color"]["light"]["line"],
            tokens["color"]["dark"]["line"],
        ),
        "--om-success": (
            tokens["color"]["light"]["success"],
            tokens["color"]["dark"]["success"],
        ),
    }
    for property_name, values in expected_css_values.items():
        for value in values:
            if f"{property_name}: {value};" not in colors:
                errors.append(f"tokens.json/CSS mismatch for {property_name} {value}")

    numeric_pairs = {
        "--om-radius-card": (tokens["radius_px"]["card"], "px"),
        "--om-radius-control": (tokens["radius_px"]["control"], "px"),
        "--om-radius-tag": (tokens["radius_px"]["tag"], "px"),
        "--om-radius-badge": (tokens["radius_px"]["badge"], "px"),
        "--om-page-max": (tokens["layout_px"]["page_max"], "px"),
        "--om-gutter": (tokens["layout_px"]["gutter"], "px"),
        "--om-bp-mobile": (tokens["layout_px"]["mobile_breakpoint"], "px"),
    }
    for property_name, (value, unit) in numeric_pairs.items():
        if f"{property_name}: {value}{unit};" not in surfaces:
            errors.append(f"tokens.json/CSS mismatch for {property_name}")

    for family in ("IBM Plex Sans", "IBM Plex Mono", "Newsreader"):
        if family not in typography:
            errors.append(f"typography.css is missing {family}")
    if "openmed-editorial-exception:start" not in typography:
        errors.append("typography.css lacks the editorial exception boundary")

    governed_css = (
        colors
        + typography
        + surfaces
        + (REPO_ROOT / "docs/stylesheets/openmed-system.css").read_text(
            encoding="utf-8"
        )
        + (REPO_ROOT / "docs/website/assets/openmed-system.css").read_text(
            encoding="utf-8"
        )
    )
    if re.search(r"https?://(?:fonts\.googleapis|fonts\.gstatic)", governed_css):
        errors.append("a governed CSS consumer references remote fonts")

    for role, record in tokens["font"]["fallback_metrics"].items():
        family = re.escape(record["family"])
        match = re.search(
            rf'@font-face\s*\{{[^}}]*font-family:\s*"{family}";(?P<body>.*?)\}}',
            typography,
            re.DOTALL,
        )
        if not match:
            errors.append(f"typography.css lacks the {role} metric fallback face")
            continue
        body = match.group("body")
        expected_metrics = {
            "src": f'local("{record["local_source"]}")',
            "size-adjust": f"{record['size_adjust_percent']:g}%",
            "ascent-override": f"{record['ascent_override_percent']:g}%",
            "descent-override": f"{record['descent_override_percent']:g}%",
            "line-gap-override": f"{record['line_gap_override_percent']:g}%",
        }
        for property_name, expected in expected_metrics.items():
            if not re.search(
                rf"(?m)^\s*{re.escape(property_name)}:\s*"
                rf"{re.escape(expected)}\s*;",
                body,
            ):
                errors.append(
                    f"typography.css {record['family']} {property_name} "
                    "does not match tokens.json"
                )

    contrast_pairs = {
        "light ink/background": (
            tokens["color"]["light"]["ink"],
            tokens["color"]["light"]["background"],
            4.5,
        ),
        "light body/background": (
            tokens["color"]["light"]["body"],
            tokens["color"]["light"]["background"],
            4.5,
        ),
        "light accent/background": (
            tokens["color"]["derived"]["light"]["accent"]["fallback"],
            tokens["color"]["light"]["background"],
            4.5,
        ),
        "light success/background": (
            tokens["color"]["light"]["success"],
            tokens["color"]["light"]["background"],
            4.5,
        ),
        "dark ink/background": (
            tokens["color"]["dark"]["ink"],
            tokens["color"]["dark"]["background"],
            4.5,
        ),
        "dark body/background": (
            tokens["color"]["dark"]["body"],
            tokens["color"]["dark"]["background"],
            4.5,
        ),
        "dark accent/background": (
            tokens["color"]["derived"]["dark"]["accent"]["fallback"],
            tokens["color"]["dark"]["background"],
            4.5,
        ),
        "dark success/background": (
            tokens["color"]["dark"]["success"],
            tokens["color"]["dark"]["background"],
            4.5,
        ),
        "light accent button": (
            "#FFFFFF",
            tokens["color"]["derived"]["light"]["button_accent_background"]["fallback"],
            4.5,
        ),
        "dark accent button": (
            tokens["color"]["derived"]["dark"]["button_accent_foreground"]["fallback"],
            tokens["color"]["derived"]["dark"]["button_accent_background"]["fallback"],
            4.5,
        ),
    }
    for label, (foreground, background, minimum) in contrast_pairs.items():
        ratio = _contrast(foreground, background)
        if ratio < minimum:
            errors.append(
                f"{label} fallback contrast is {ratio:.2f}:1, below {minimum}:1"
            )

    _, source_chroma, source_hue = _srgb_to_oklch(tokens["color"]["accent_base"])
    enhanced: dict[str, dict[str, tuple[float, float, float, float]]] = {
        theme: {
            role: _oklch_to_srgb(
                record["lightness"],
                min(source_chroma, record["chroma_cap"]),
                source_hue,
                record["alpha"],
            )
            for role, record in roles.items()
        }
        for theme, roles in tokens["color"]["derived"].items()
    }
    enhanced_contrast = {
        "light accent/background": (
            enhanced["light"]["accent"],
            tokens["color"]["light"]["background"],
            4.5,
        ),
        "light strong/surface": (
            enhanced["light"]["accent_strong"],
            tokens["color"]["light"]["surface"],
            4.5,
        ),
        "dark accent/background": (
            enhanced["dark"]["accent"],
            tokens["color"]["dark"]["background"],
            4.5,
        ),
        "dark strong/surface": (
            enhanced["dark"]["accent_strong"],
            tokens["color"]["dark"]["surface"],
            4.5,
        ),
    }
    for label, (foreground, background, minimum) in enhanced_contrast.items():
        ratio = _contrast_rgba(foreground, background)
        if ratio < minimum:
            errors.append(
                f"{label} enhanced contrast is {ratio:.2f}:1, below {minimum}:1"
            )

    dark_button_background = enhanced["dark"]["button_accent_background"]
    dark_button_foreground = enhanced["dark"]["button_accent_foreground"]
    lighter, darker = sorted(
        (
            _relative_luminance(dark_button_background),
            _relative_luminance(dark_button_foreground),
        ),
        reverse=True,
    )
    if (ratio := (lighter + 0.05) / (darker + 0.05)) < 4.5:
        errors.append(f"dark enhanced accent button contrast is {ratio:.2f}:1")

    terminal = tokens["color"]["terminal"]
    for role in ("foreground", "body", "muted_text", "faint_text", "accent"):
        if (ratio := _contrast(terminal[role], terminal["background"])) < 4.5:
            errors.append(f"terminal {role} contrast is {ratio:.2f}:1")
    for role, syntax in tokens["color"]["syntax"].items():
        syntax_background = _composite(
            _parse_color(syntax["background"]),
            _parse_color(terminal["background"]),
        )
        foreground = _parse_color(syntax["foreground"])
        lighter, darker = sorted(
            (
                _relative_luminance(foreground),
                _relative_luminance(syntax_background),
            ),
            reverse=True,
        )
        if (ratio := (lighter + 0.05) / (darker + 0.05)) < 4.5:
            errors.append(f"terminal syntax {role} contrast is {ratio:.2f}:1")


def _validate_fonts_and_consumers(errors: list[str]) -> None:
    manifest = _load_json("docs/brand/assets/fonts/manifest.json")
    font_root = ASSET_ROOT / "fonts"
    destinations = {
        "docs": REPO_ROOT / "docs/assets/fonts",
        "website": REPO_ROOT / "docs/website/assets/fonts",
    }

    if manifest["schema_version"] != 2:
        errors.append("font manifest must use schema version 2")
    conversion = manifest["conversion"]
    if conversion["tool"] != "fontTools 4.61.1" or conversion["version"] != "4.61.1":
        errors.append("font conversion tool/version is not exactly fontTools 4.61.1")
    if conversion["ibm_plex_subsetting"] != "none":
        errors.append("IBM Plex must remain unsubscribed for localized documentation")

    licensed_families: set[str] = set()
    for license_record in manifest["licenses"]:
        license_path = font_root / license_record["file"]
        if license_record["spdx"] != "OFL-1.1":
            errors.append(
                f"font license {license_record['file']} has the wrong SPDX id"
            )
        if not license_record["source"].startswith("https://"):
            errors.append(
                f"font license {license_record['file']} lacks an HTTPS source"
            )
        if (
            not license_path.is_file()
            or _source_sha256(license_path) != license_record["sha256"]
        ):
            errors.append(f"font license hash mismatch: {license_record['file']}")
        licensed_families.update(license_record["applies_to"])

    from fontTools.ttLib import TTFont  # noqa: PLC0415

    expected_by_consumer: dict[str, set[str]] = {
        consumer: set() for consumer in destinations
    }
    for item in manifest["files"]:
        source = font_root / item["file"]
        web = font_root / item["web_file"]
        if item["family"] not in licensed_families:
            errors.append(f"font family {item['family']} has no applicable license")
        provenance = item["source"]
        required_provenance = {
            "project",
            "postscript_name",
            "font_version",
            "relationship",
        }
        if item["family"].startswith("IBM Plex"):
            required_provenance |= {
                "release",
                "archive",
                "archive_sha256",
                "upstream_path",
            }
        else:
            required_provenance |= {"distribution", "url"}
        missing_provenance = required_provenance - set(provenance)
        if missing_provenance:
            errors.append(
                f"font {item['file']} lacks provenance {sorted(missing_provenance)}"
            )
        for field in ("project", "archive", "url"):
            if field in provenance and not provenance[field].startswith("https://"):
                errors.append(f"font {item['file']} {field} is not an HTTPS source")
        if not source.is_file() or _sha256(source) != item["sha256"]:
            errors.append(f"font source hash mismatch: {item['file']}")
        if not web.is_file() or _sha256(web) != item["web_sha256"]:
            errors.append(f"web font hash mismatch: {item['web_file']}")
        if source.is_file():
            with TTFont(source, lazy=True) as source_font:
                names = source_font["name"]
                postscript_names = {
                    record.toUnicode() for record in names.names if record.nameID == 6
                }
                version_names = {
                    record.toUnicode().removeprefix("Version ")
                    for record in names.names
                    if record.nameID == 5
                }
                if provenance["postscript_name"] not in postscript_names:
                    errors.append(
                        f"font {item['file']} PostScript name does not match provenance"
                    )
                if provenance["font_version"] not in version_names:
                    errors.append(
                        f"font {item['file']} embedded version does not match provenance"
                    )
                source_codepoints = set(source_font.getBestCmap())
        else:
            source_codepoints = set()
        if web.is_file():
            with TTFont(web, lazy=True) as web_font:
                web_codepoints = set(web_font.getBestCmap())
                if web_font.flavor != "woff2":
                    errors.append(f"web font {item['web_file']} is not WOFF2")
        else:
            web_codepoints = set()
        if (
            item["family"].startswith("IBM Plex")
            and web_codepoints != source_codepoints
        ):
            errors.append(
                f"IBM Plex web font was unexpectedly subset: {item['web_file']}"
            )
        for consumer in item["consumers"]:
            expected_by_consumer[consumer].add(item["web_file"])
            copy = destinations[consumer] / item["web_file"]
            if not copy.is_file() or copy.read_bytes() != web.read_bytes():
                errors.append(f"{consumer} font copy mismatch: {item['web_file']}")

    published_total = 0
    for consumer, destination in destinations.items():
        actual = {path.name for path in destination.iterdir() if path.is_file()}
        if actual != expected_by_consumer[consumer]:
            errors.append(
                f"{consumer} font set mismatch: "
                f"expected={sorted(expected_by_consumer[consumer])}, "
                f"actual={sorted(actual)}"
            )
        total = sum((destination / name).stat().st_size for name in actual)
        published_total += total
        budget = manifest["budgets"][f"{consumer}_woff2_bytes_max"]
        if total > budget:
            errors.append(f"{consumer} font payload {total} exceeds {budget}")

    combined_budget = manifest["budgets"]["combined_published_woff2_bytes_max"]
    if published_total > combined_budget:
        errors.append(
            f"combined font payload {published_total} exceeds {combined_budget}"
        )

    subset = conversion["newsreader_subset"]
    requested_codepoints = _unicode_range_codepoints(subset["unicode_ranges"])
    if subset["coverage_sources"] != ["docs/website/index.html"]:
        errors.append("Newsreader subset coverage sources are not the owned consumers")
    for item in manifest["files"]:
        if item["family"] != "Newsreader":
            continue
        with TTFont(font_root / item["web_file"], lazy=True) as subset_font:
            if set(subset_font.getBestCmap()) != requested_codepoints:
                errors.append(
                    f"Newsreader subset coverage differs from declared ranges: "
                    f"{item['web_file']}"
                )

    website = (REPO_ROOT / "docs/website/index.html").read_text(encoding="utf-8")
    editorial_copy = "\n".join(
        re.findall(
            r'<(?:h1|h2|p)[^>]*class="[^"]*(?:display|lede)[^"]*"[^>]*>'
            r"(.*?)</(?:h1|h2|p)>",
            website,
            re.DOTALL,
        )
    )
    editorial_copy = _plain_html_text(editorial_copy)
    missing_editorial = sorted(
        {
            ord(character)
            for character in editorial_copy
            if not character.isspace() and ord(character) not in requested_codepoints
        }
    )
    if missing_editorial:
        errors.append(
            "Newsreader subset misses website editorial codepoints "
            f"{[f'U+{value:04X}' for value in missing_editorial]}"
        )

    docs_css = REPO_ROOT / "docs/stylesheets/openmed-system.css"
    website_css = REPO_ROOT / "docs/website/assets/openmed-system.css"
    docs_text = docs_css.read_text(encoding="utf-8")
    website_text = website_css.read_text(encoding="utf-8")
    if re.search(r"src:\s*url\([^)]*Newsreader", docs_text):
        errors.append("documentation CSS publishes the Newsreader exception")
    if len(re.findall(r"src:\s*url\([^)]*Newsreader", website_text)) != 2:
        errors.append("website CSS must publish exactly two Newsreader faces")
    expected_unicode_css = ", ".join(subset["unicode_ranges"])
    normalized_website_css = re.sub(r"\s+", " ", website_text)
    if normalized_website_css.count(f"unicode-range: {expected_unicode_css};") != 2:
        errors.append("website Newsreader unicode-range differs from the manifest")

    for css_path, css_text in ((docs_css, docs_text), (website_css, website_text)):
        for raw_url in re.findall(r"url\([\"']?([^\"')]+)", css_text):
            if "://" in raw_url or raw_url.startswith("data:"):
                continue
            target = (css_path.parent / raw_url).resolve()
            if not target.is_file():
                errors.append(
                    f"unresolved local CSS URL in "
                    f"{css_path.relative_to(REPO_ROOT)}: {raw_url}"
                )

    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    website = (REPO_ROOT / "docs/website/index.html").read_text(encoding="utf-8")
    if "stylesheets/openmed-system.css" not in mkdocs:
        errors.append("MkDocs does not load the generated token consumer")
    if 'href="assets/openmed-system.css"' not in website:
        errors.append("website does not load the generated token consumer")


def _validate_claims(errors: list[str]) -> None:
    registry = _load_json("docs/brand/system/claims.yml")
    if registry["schema_version"] != 2:
        errors.append("claims registry must use schema version 2")
    if registry["generation"] != {
        "command": "python scripts/brand/update_claims.py --write",
        "network": "forbidden",
        "network_refresh_command": (
            "python scripts/brand/update_claims.py --refresh-github-stars"
        ),
        "network_refresh_ci_policy": "never invoke from CI",
        "rounding": "none unless a claim definition explicitly says otherwise",
    }:
        errors.append("claims generation policy is not the offline governed contract")

    claims = registry["claims"]
    required_claims = {
        "dataset_count",
        "pii_checkpoint_count",
        "runtime_behavior_by_surface",
        "license_by_product_surface",
    }
    missing_claims = required_claims - set(claims)
    if missing_claims:
        errors.append(
            f"claims registry lacks required contract fields {sorted(missing_claims)}"
        )
    expected_values = {
        "package_version": "2.0.0",
        "repository_model_snapshot": 1520,
        "hugging_face_openmed_owned_snapshot": 1520,
        "supported_pii_languages": 34,
        "model_backed_pii_languages": 33,
        "placeholder_pii_languages": ["ru"],
        "user_supplied_model_languages": ["gu", "kn", "ml", "ne", "pa", "ur"],
        "pii_family_manifest_entries": 655,
        "mlx_manifest_entries": 662,
        "pii_entity_types": 50,
        "model_license_population": {
            "apache-2.0": 1511,
            "other": 5,
            "unknown": 4,
        },
        "sdk_license": "Apache-2.0",
        "runtime_locality": "local-first",
        "runtime_behavior_by_surface": {
            "core_sdk": (
                "Local processing after required artifacts are available; "
                "local_only mode blocks outbound sockets after model loading."
            ),
            "artifact_acquisition": (
                "Model, evaluation-data, and optional vocabulary downloads "
                "may use their configured network sources."
            ),
            "optional_integrations": (
                "Remote-provider adapters, telemetry-enabled paths, and "
                "user-configured integrations may use a network."
            ),
            "browser_demo": (
                "Accepts same-origin runtime and model URLs only; the page "
                "does not upload entered text."
            ),
        },
        "license_by_product_surface": {
            "openmed_sdk_source": "Apache-2.0",
            "catalog_models": "mixed and partially unknown",
            "referenced_datasets": "source-specific access and license terms",
            "openmed_agent": "separate product terms",
            "welna": "separate product terms",
            "external_integrations": "upstream terms",
        },
    }
    for claim_name, value in expected_values.items():
        if claims[claim_name]["value"] != value:
            errors.append(
                f"claim {claim_name} is {claims[claim_name]['value']!r}, "
                f"expected {value!r}"
            )
    if len(claims["national_id_only_languages"]["value"]) != 18:
        errors.append("national-ID-only language claim must contain 18 codes")

    try:
        generated_at = dt.date.fromisoformat(registry["generated_at"])
    except ValueError:
        errors.append("claims generated_at is not an ISO date")
        generated_at = dt.date.min
    required_fields = {
        "status",
        "value",
        "display",
        "definition",
        "source",
        "as_of",
        "owner",
        "public_wording",
        "qualification",
        "review_by",
        "follow_up_by",
    }
    for claim_name, claim in claims.items():
        missing = required_fields - set(claim)
        if missing:
            errors.append(f"claim {claim_name} lacks fields {sorted(missing)}")
            continue
        if not claim["owner"]:
            errors.append(f"claim {claim_name} has no owner")
        if claim["status"] == "unverified":
            if (
                claim["value"] is not None
                or claim["display"] is not None
                or claim["public_wording"]
                or claim["as_of"] is not None
                or claim["review_by"] is not None
            ):
                errors.append(
                    f"unverified claim {claim_name} exposes value, wording, or dates"
                )
            try:
                follow_up_by = dt.date.fromisoformat(claim["follow_up_by"])
            except (TypeError, ValueError):
                errors.append(
                    f"unverified claim {claim_name} lacks a valid follow-up date"
                )
                continue
            if follow_up_by <= generated_at:
                errors.append(
                    f"unverified claim {claim_name} has a non-forward follow-up date"
                )
            if follow_up_by < dt.date.today():
                errors.append(
                    f"unverified claim {claim_name} follow-up date has expired"
                )
            continue
        if claim["status"] != "verified":
            errors.append(f"claim {claim_name} has unknown status {claim['status']!r}")
            continue
        try:
            as_of = dt.date.fromisoformat(claim["as_of"])
            review_by = dt.date.fromisoformat(claim["review_by"])
        except (TypeError, ValueError):
            errors.append(f"verified claim {claim_name} lacks valid ISO dates")
            continue
        if as_of > generated_at:
            errors.append(f"claim {claim_name} is dated after registry generation")
        if review_by <= as_of:
            errors.append(f"claim {claim_name} has a non-forward review date")
        if review_by < dt.date.today():
            errors.append(f"claim {claim_name} review date has expired")
        if not claim["qualification"]:
            errors.append(f"verified claim {claim_name} has no qualification")
        if claim["follow_up_by"] is not None:
            errors.append(f"verified claim {claim_name} must not declare follow_up_by")

    evidence = _load_json("docs/brand/system/evidence/github-repository.json")
    expected_refresh = {
        "command": "python scripts/brand/update_claims.py --refresh-github-stars",
        "network": "explicit opt-in only",
        "ci": "forbidden",
    }
    if evidence["schema_version"] != 2 or evidence["refresh"] != expected_refresh:
        errors.append("GitHub repository evidence lacks schema-v2 offline policy")
    raw_stars = evidence["stargazers_count"]
    display = evidence["display"]
    if (
        not isinstance(raw_stars, int)
        or raw_stars < 0
        or display["method"] != "floor"
        or display["quantum"] != 100
        or display["value"] != raw_stars // 100 * 100
        or display["label"] != f"{display['value']:,}+ GitHub stars"
    ):
        errors.append("GitHub star evidence is not conservatively rounded")
    star_claim = claims["github_stars_snapshot"]
    if (
        star_claim["value"] != raw_stars
        or star_claim["display"] != display["label"]
        or star_claim["public_wording"] != display["label"]
        or star_claim["as_of"] != evidence["captured_at"][:10]
        or star_claim["review_by"] != evidence["review_by"]
    ):
        errors.append("GitHub star claim differs from committed offline evidence")

    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / ".github/workflows").glob("*.y*ml"))
    )
    if "--refresh-github-stars" in workflow_text:
        errors.append("CI must never invoke the networked GitHub stars refresh")

    package = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    expected_description = (
        'description = "Local-first SDK for clinical extraction and '
        'de-identification workflows on hardware you control."'
    )
    if expected_description not in package:
        errors.append("package description is not the governed provider-neutral copy")

    profile = _load_json("docs/brand/social/_src/profile-copy.json")
    if profile["schema_version"] != 2:
        errors.append("social profile copy must use schema version 2")
    for claim_name in profile["required_claims"]:
        if claim_name not in claims or claims[claim_name]["status"] != "verified":
            errors.append(f"profile copy depends on unverified claim {claim_name}")
    for placeholder, reference in profile["claim_placeholders"].items():
        claim_name, field = reference.split(".", 1)
        if (
            claim_name not in claims
            or field not in claims[claim_name]
            or not claims[claim_name][field]
        ):
            errors.append(f"profile placeholder {placeholder} has no verified value")

    website = (REPO_ROOT / "docs/website/index.html").read_text(encoding="utf-8")
    json_ld: list[dict[str, Any]] = []
    for payload in re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        website,
        re.DOTALL,
    ):
        try:
            json_ld.append(json.loads(payload))
        except json.JSONDecodeError as exc:
            errors.append(f"website JSON-LD is invalid: {exc}")
    by_type = {item.get("@type"): item for item in json_ld}
    if set(by_type) != {"Organization", "SoftwareSourceCode", "FAQPage"}:
        errors.append(
            "website must publish exactly Organization, software, and FAQ JSON-LD"
        )
    organization = by_type.get("Organization", {})
    expected_identity_links = [
        "https://github.com/maziyarpanahi/openmed",
        "https://huggingface.co/OpenMed",
        "https://x.com/OpenMed_AI",
        "https://www.linkedin.com/company/openmed-ai/",
    ]
    if (
        organization.get("url") != "https://openmed.life/"
        or organization.get("sameAs") != expected_identity_links
        or organization.get("founder") != {"@type": "Person", "name": "Maziyar Panahi"}
    ):
        errors.append("website Organization JSON-LD identity graph is not canonical")
    software = by_type.get("SoftwareSourceCode", {})
    if (
        software.get("softwareVersion") != claims["package_version"]["value"]
        or software.get("license") != "https://www.apache.org/licenses/LICENSE-2.0"
        or software.get("codeRepository") != "https://github.com/maziyarpanahi/openmed"
    ):
        errors.append("website software JSON-LD disagrees with governed claims")
    _validate_faq_parity(website, by_type.get("FAQPage", {}), errors)

    prohibited_public_claims = (
        r"\bstate[- ]of[- ]the[- ]art\b",
        r"\bSOTA\b",
        r"\bruns? everywhere\b",
        r"\bevery platform\b",
        r"\bnever leaves your device\b",
        r"\bnothing is sent to the cloud\b",
        r"\bentirely on the user(?:'s)? own device\b",
        r"\bno network calls?\b",
    )
    claim_surfaces = [
        REPO_ROOT / "docs/website/index.html",
        *sorted(REPO_ROOT.glob("README*.md")),
        REPO_ROOT / "docs/brand/MASCOT_BRIEF.md",
    ]
    for path in claim_surfaces:
        text = re.sub(
            r"```.*?```",
            "",
            path.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        for pattern in prohibited_public_claims:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(
                    f"{path.relative_to(REPO_ROOT)} exposes prohibited absolute "
                    f"claim pattern {pattern!r}"
                )

    def without_visual_descriptions(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_visual_descriptions(item)
                for key, item in value.items()
                if key != "alt_text" and not key.endswith("_alt")
            }
        if isinstance(value, list):
            return [without_visual_descriptions(item) for item in value]
        return value

    profile_copy = without_visual_descriptions(
        _load_json("docs/brand/social/_src/profile-copy.json")
    )
    profile_claim_text = json.dumps(profile_copy, ensure_ascii=False)
    for pattern in prohibited_public_claims:
        if re.search(pattern, profile_claim_text, re.IGNORECASE):
            errors.append(
                "docs/brand/social/_src/profile-copy.json exposes prohibited "
                f"non-alt claim pattern {pattern!r}"
            )


def _validate_readmes(errors: list[str]) -> None:
    registry = _load_json("docs/brand/system/claims.yml")
    star_claim = registry["claims"]["github_stars_snapshot"]
    root_readmes = sorted(REPO_ROOT.glob("README*.md"))
    if len(root_readmes) != 15:
        errors.append(f"expected 15 root READMEs, found {len(root_readmes)}")
    for path in root_readmes:
        text = path.read_text(encoding="utf-8")
        if text.count("docs/brand/openmed-readme-banner.png") != 1:
            errors.append(f"{path.name}: canonical banner count is not one")
        if re.search(r"(?m)^\|[^\n]*\*\*OpenMed\*\*[^\n]*$", text):
            errors.append(f"{path.name}: provider comparison table remains")
        if re.search(r"(?m)^## [^\n]*PII[^\n]*12[^\n]*$", text):
            errors.append(f"{path.name}: stale 12-language heading remains")
        if text.count("[Apache-2.0 License](LICENSE)") != 1:
            errors.append(f"{path.name}: canonical legal license link is missing")
        if "Apache-2.0 SDK License" in text:
            errors.append(f"{path.name}: SDK was incorrectly added to license name")
        if "api.star-history.com" in text:
            errors.append(f"{path.name}: remote star-history image remains")
        if text.count(star_claim["display"]) != 1:
            errors.append(
                f"{path.name}: governed GitHub star snapshot count is not one"
            )
        for prohibited_host in ("trendshift.io", "img.shields.io"):
            if prohibited_host in text:
                errors.append(
                    f"{path.name}: remote dynamic badge {prohibited_host} remains"
                )
        if "34" not in text or "33" not in text:
            errors.append(f"{path.name}: governed language counts are missing")
        if "Safe Harbor" not in text or "18" not in text:
            errors.append(f"{path.name}: qualified Safe Harbor capability is missing")


def _validate_approved_social(errors: list[str]) -> None:
    """Validate that social outputs use only the approved handoff exports."""

    from PIL import Image  # noqa: PLC0415

    manifest = _load_json("docs/brand/social/manifest.json")
    source = _load_json("docs/brand/social/_src/exports.json")
    profile_source = _load_json("docs/brand/social/_src/profile-copy.json")
    provenance = _load_json("docs/brand/system/handoff-provenance.json")
    visual_review = _load_json("docs/brand/system/evidence/social-visual-review.json")

    if manifest.get("schema_version") != 3 or source.get("schema_version") != 3:
        errors.append("social export source and manifest must use schema version 3")
    if profile_source.get("schema_version") != 2:
        errors.append("social profile copy must use schema version 2")
    if (
        source.get("export_scale") != 2
        or source.get("native_scale") != 1
        or source.get("network") != "forbidden"
        or source.get("animation_state") != "disabled"
        or "no visual reconstruction" not in source.get("derivative_policy", "").lower()
    ):
        errors.append(
            "social export source does not lock exact masters and size-only derivatives"
        )

    synchronizer = manifest.get("synchronizer", {})
    expected_synchronizer = {
        "pillow_version": "12.3.0",
        "master_policy": "byte-identical approved export",
        "resampling": "LANCZOS",
        "derivative_png_compress_level": 9,
        "network": "blocked",
        "master_color_profile": "source-preserved",
        "derivative_color_profile": "sRGB",
        "animation_state": "disabled",
        "visual_reconstruction": "forbidden",
    }
    for field, expected in expected_synchronizer.items():
        if synchronizer.get(field) != expected:
            errors.append(f"social synchronizer contract drifted: {field}")

    expected_source_hashes = {
        relative: _source_sha256(REPO_ROOT / relative)
        for relative in manifest.get("source_hashes", {})
    }
    if manifest.get("source_hashes") != expected_source_hashes:
        errors.append("social manifest source hashes are stale")

    approved_hashes = provenance.get("approved_exports", {})
    if manifest.get("approved_exports") != approved_hashes:
        errors.append("social manifest does not match approved handoff export hashes")
    export_dir = BRAND_ROOT / "social/exports"
    actual_export_names = {path.name for path in export_dir.glob("*.png")}
    if actual_export_names != set(approved_hashes):
        errors.append("canonical social export directory is incomplete or has extras")
    for name, expected_hash in approved_hashes.items():
        path = export_dir / name
        if not path.is_file() or _sha256(path) != expected_hash:
            errors.append(f"approved social export hash mismatch: {name}")

    forbidden_legacy_paths = (
        BRAND_ROOT / "social/golden",
        BRAND_ROOT / "social/_src/artboards.json",
        BRAND_ROOT / "social/_src/announcement-template.json",
        BRAND_ROOT / "social/_src/carousel-template.json",
    )
    for path in forbidden_legacy_paths:
        if path.exists():
            errors.append(f"legacy reconstructed social source remains: {path}")

    source_by_id = {asset["id"]: asset for asset in source.get("assets", [])}
    manifest_by_id = {asset["id"]: asset for asset in manifest.get("assets", [])}
    expected_ids = {
        "website-og",
        "github-social",
        "x-header",
        "hugging-face-card",
        "readme-banner",
        "linkedin-banner",
        "hugging-face-avatar",
        "x-avatar",
        "linkedin-company-tile",
        "favicon",
    }
    if set(source_by_id) != expected_ids or set(manifest_by_id) != expected_ids:
        errors.append("social approved-export mapping is incomplete or has unknown ids")

    source_names = {
        Path(asset.get("source", "")).name for asset in source_by_id.values()
    }
    if source_names != set(approved_hashes):
        errors.append("not every approved export is mapped exactly once")
    if len(source_names) != len(source_by_id):
        errors.append("an approved export is reused for multiple social masters")

    expected_master_paths = {asset.get("master", "") for asset in source_by_id.values()}
    actual_master_paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (BRAND_ROOT / "social").glob("*-2x.png")
    }
    if actual_master_paths != expected_master_paths:
        errors.append("top-level social masters are incomplete or include unknown art")

    for asset_id, source_record in source_by_id.items():
        record = manifest_by_id.get(asset_id, {})
        for field, expected in source_record.items():
            if record.get(field) != expected:
                errors.append(
                    f"{asset_id}: social manifest differs from source field {field}"
                )

        approved = REPO_ROOT / source_record["source"]
        master = REPO_ROOT / source_record["master"]
        native = REPO_ROOT / source_record["native"]
        if not approved.is_file() or not master.is_file() or not native.is_file():
            errors.append(f"{asset_id}: approved, master, or native asset is missing")
            continue
        if master.read_bytes() != approved.read_bytes():
            errors.append(f"{asset_id}: master is not an exact approved-export copy")
        if (
            record.get("approved_export_sha256") != _sha256(approved)
            or record.get("master_sha256") != _sha256(master)
            or record.get("native_sha256") != _sha256(native)
            or record.get("master_exact_copy") is not True
        ):
            errors.append(f"{asset_id}: social asset hashes are stale")

        width, height = source_record["size"]
        try:
            if _png_info(approved) != (width * 2, height * 2, 8, 6):
                errors.append(f"{asset_id}: approved export dimensions/mode mismatch")
            if _png_info(master) != (width * 2, height * 2, 8, 6):
                errors.append(f"{asset_id}: master dimensions/mode mismatch")
            if _png_info(native) != (width, height, 8, 6):
                errors.append(f"{asset_id}: native dimensions/mode mismatch")
            native_srgb = [
                payload for kind, payload in _png_chunks(native) if kind == "sRGB"
            ]
            if native_srgb != [b"\x00"]:
                errors.append(f"{asset_id}: native derivative lacks explicit sRGB")
        except ValueError as exc:
            errors.append(f"{asset_id}: {exc}")
            continue

        with Image.open(approved) as approved_image:
            approved_rgba = approved_image.convert("RGBA")
            expected_native = approved_rgba.resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )
        with Image.open(native) as native_image:
            native_rgba = native_image.convert("RGBA")
        approved_pixel_hash = hashlib.sha256(approved_rgba.tobytes()).hexdigest()
        native_pixel_hash = hashlib.sha256(native_rgba.tobytes()).hexdigest()
        if record.get("approved_export_pixel_sha256") != approved_pixel_hash:
            errors.append(f"{asset_id}: approved-export pixel hash is stale")
        if native_rgba.tobytes() != expected_native.tobytes():
            errors.append(f"{asset_id}: native is not the exact LANCZOS half-size")
        if record.get("native_pixel_sha256") != native_pixel_hash:
            errors.append(f"{asset_id}: native pixel hash is stale")
        derivative = record.get("native_derivative", {})
        if (
            derivative.get("method") != "Pillow Image.Resampling.LANCZOS"
            or derivative.get("source") != source_record["source"]
            or derivative.get("source_size") != [width * 2, height * 2]
            or derivative.get("output_size") != [width, height]
            or derivative.get("visual_edits") != "none"
        ):
            errors.append(f"{asset_id}: native derivative provenance is incomplete")

        x0, y0, x1, y1 = source_record["safe_zone"]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            errors.append(f"{asset_id}: safe zone escapes the native canvas")
        elif _crop_pixel_hash(native, source_record["safe_zone"]) != record.get(
            "safe_zone_crop_pixel_sha256"
        ):
            errors.append(f"{asset_id}: safe-zone crop hash is stale")

        if "safe_zone_preview" in record:
            preview = REPO_ROOT / record["safe_zone_preview"]
            if not preview.is_file() or _sha256(preview) != record.get(
                "safe_zone_preview_sha256"
            ):
                errors.append(f"{asset_id}: safe-zone preview hash is stale")
            else:
                try:
                    preview_srgb = [
                        payload
                        for kind, payload in _png_chunks(preview)
                        if kind == "sRGB"
                    ]
                    if preview_srgb != [b"\x00"]:
                        errors.append(
                            f"{asset_id}: safe-zone preview lacks explicit sRGB"
                        )
                except ValueError as exc:
                    errors.append(f"{asset_id}: invalid safe-zone preview: {exc}")
        for destination in source_record["copy_to"]:
            consumer = REPO_ROOT / destination
            if not consumer.is_file() or consumer.read_bytes() != native.read_bytes():
                errors.append(f"{asset_id}: stale consumer {destination}")

    for record in manifest.get("derived_assets", []):
        output = REPO_ROOT / record["output"]
        if not output.is_file() or _sha256(output) != record.get("sha256"):
            errors.append(f"derived social asset hash mismatch: {record['output']}")
            continue
        if record.get("visual_edits") != "none":
            errors.append(f"{record['id']}: derived visual edits are not forbidden")
        if record["kind"] == "resize":
            source_path = REPO_ROOT / record["source"]
            with Image.open(source_path) as source_image:
                expected = source_image.convert("RGBA").resize(
                    tuple(record["size"]),
                    Image.Resampling.LANCZOS,
                )
            with Image.open(output) as output_image:
                actual = output_image.convert("RGBA")
            if actual.tobytes() != expected.tobytes():
                errors.append(f"{record['id']}: resize is not source-faithful")
            try:
                if _png_info(output) != (*record["size"], 8, 6):
                    errors.append(f"{record['id']}: resize dimensions/mode mismatch")
                srgb = [
                    payload for kind, payload in _png_chunks(output) if kind == "sRGB"
                ]
                if srgb != [b"\x00"]:
                    errors.append(f"{record['id']}: resize lacks explicit sRGB")
            except ValueError as exc:
                errors.append(f"{record['id']}: invalid derived PNG: {exc}")
        elif record["kind"] == "favicon_ico":
            ico_data = output.read_bytes()
            reserved, image_type, count = struct.unpack("<HHH", ico_data[:6])
            if (reserved, image_type, count) != (0, 1, len(record["sizes"])):
                errors.append("favicon ICO directory header mismatch")
            actual_sizes: list[list[int]] = []
            for index in range(count):
                entry = ico_data[6 + index * 16 : 22 + index * 16]
                if len(entry) != 16:
                    errors.append("favicon ICO directory is truncated")
                    break
                actual_sizes.append([entry[0] or 256, entry[1] or 256])
            if actual_sizes != record["sizes"]:
                errors.append("favicon ICO does not contain the declared sizes")
        else:
            errors.append(f"unknown derived social asset kind: {record['kind']}")
        for destination in record["copy_to"]:
            consumer = REPO_ROOT / destination
            if not consumer.is_file() or consumer.read_bytes() != output.read_bytes():
                errors.append(f"{record['id']}: stale consumer {destination}")

    resolved_profile_bytes = json.dumps(
        manifest.get("resolved_profile_copy", {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(resolved_profile_bytes).hexdigest() != manifest.get(
        "resolved_profile_copy_sha256"
    ):
        errors.append("resolved social profile-copy hash is stale")
    resolved_profile = manifest.get("resolved_profile_copy", {})
    if "artwork" in resolved_profile:
        errors.append("social profile copy must not define reconstructed artwork")
    platforms = resolved_profile.get("platforms", {})
    expected_platforms = {
        "website",
        "github_repository",
        "hugging_face_organization",
        "hugging_face_card_repository",
        "x",
        "linkedin",
    }
    if set(platforms) != expected_platforms:
        errors.append("social profile plan does not cover every governed platform")
    for platform_name, platform in platforms.items():
        if (
            not platform.get("target")
            or not platform.get("link", "").startswith("https://")
            or not platform.get("authorization")
        ):
            errors.append(
                f"social profile {platform_name} lacks target/link/authorization"
            )
    hugging_face_card = platforms.get("hugging_face_card_repository", {})
    if (
        "Never read or change Hugging Face Space visibility or settings."
        not in hugging_face_card.get("space_policy", "")
    ):
        errors.append("Hugging Face card plan lacks the immutable Space-settings rule")
    expected_alt_ids = {
        "og-website",
        "github-social",
        "x-header",
        "hf-org",
        "readme-banner",
        "linkedin-banner",
        "avatar-cat",
        "avatar-x-circle",
        "avatar-linkedin",
        "favicon",
        "apple-touch",
        "favicon-ico",
    }
    if set(resolved_profile.get("alt_text", {})) != expected_alt_ids:
        errors.append("social alt-text registry does not cover approved outputs")

    expected_review_methods = {
        "original-size inspection",
        "320px max-edge thumbnail inspection",
        "safe-zone overlay inspection",
    }
    if (
        visual_review.get("schema_version") != 1
        or visual_review.get("review_scope") != "repository_candidate"
        or set(visual_review.get("methods", [])) != expected_review_methods
        or visual_review.get("external_platform_approval") != "pending_phase_5"
    ):
        errors.append("social visual-review evidence lacks the governed scope")
    try:
        reviewed_at = dt.date.fromisoformat(visual_review["reviewed_at"])
    except (KeyError, TypeError, ValueError):
        errors.append("social visual-review evidence has an invalid review date")
    else:
        if reviewed_at > dt.date.today():
            errors.append("social visual-review evidence is future-dated")

    review_groups = {
        "assets": {
            "website-og",
            "github-social",
            "x-header",
            "hugging-face-card",
            "readme-banner",
            "linkedin-banner",
        },
        "safe_zone_previews": {
            "github-social",
            "x-header",
            "hugging-face-card",
            "linkedin-banner",
        },
        "avatars": {
            "hugging-face-avatar",
            "x-avatar",
            "linkedin-company-tile",
        },
    }
    result_fields = {
        "assets": ("original_size", "thumbnail_320_max_edge"),
        "safe_zone_previews": ("crop_safe",),
        "avatars": ("crop_safe", "distinctive"),
    }
    for group, expected_group_ids in review_groups.items():
        records = visual_review.get(group, [])
        by_id = {record.get("id"): record for record in records}
        if set(by_id) != expected_group_ids:
            errors.append(f"social visual-review {group} coverage is incomplete")
            continue
        for record in records:
            path = REPO_ROOT / record.get("path", "")
            if not path.is_file() or _sha256(path) != record.get("sha256"):
                errors.append(
                    f"social visual-review evidence is stale: {record.get('id')}"
                )
            if any(record.get(field) != "pass" for field in result_fields[group]):
                errors.append(f"social visual-review did not pass: {record.get('id')}")
    if visual_review.get("findings") != {
        "clipping": "pass",
        "primary_headline_legibility_at_320px": "pass",
        "avatar_crop_and_distinction": "pass",
    }:
        errors.append("social visual-review findings are incomplete")
    if "not platform-owner approval" not in visual_review.get("qualification", ""):
        errors.append("social visual-review evidence lacks its approval boundary")


def _validate_accessibility_evidence(errors: list[str]) -> None:
    evidence = _load_json("docs/brand/system/evidence/manual-accessibility-review.json")
    expected_surfaces = [
        {"id": "website", "path": "/"},
        {"id": "docs-landing", "path": "/docs/"},
        {"id": "docs-guide", "path": "/docs/getting-started/"},
        {"id": "docs-api", "path": "/docs/api-reference/"},
        {"id": "docs-chinese", "path": "/docs/zh/"},
        {"id": "docs-hindi", "path": "/docs/hi/"},
        {
            "id": "leaderboard",
            "path": "/docs/eval/benchmark-leaderboard/",
        },
        {"id": "browser-demo", "path": "/docs/demo/web/"},
        {"id": "rtl-fixture", "path": "/docs/demo/rtl/"},
    ]
    expected_viewports = [
        "320x800",
        "390x844",
        "667x320",
        "768x1024",
        "1440x900",
        "1536x864",
    ]
    expected_themes = ["light", "dark", "system-light", "system-dark"]
    expected_presentation_modes = ["forced-colors", "print-where-relevant"]
    expected_methods = [
        "keyboard-only operation and visible-focus review",
        "accessible names, roles, and states review",
        "400 percent zoom proxy review",
        "WCAG text-spacing worst-case override review",
        "reduced-motion review",
        "forced-colors review",
        "automated axe WCAG 2.2 AA scan",
        "WCAG 2.2 AA contrast review",
        "manual visual snapshot review",
        "oversized WebKit screenshot tiling review",
        "print review where relevant",
    ]
    expected_result_keys = {
        "keyboard_only_and_visible_focus",
        "accessible_names_roles_states",
        "zoom_400_percent_proxy",
        "text_spacing_worst_case",
        "reduced_motion",
        "forced_colors",
        "axe_wcag_2_2_aa",
        "contrast_wcag_2_2_aa",
        "manual_snapshots",
        "oversized_webkit_tiling",
        "print_where_relevant",
    }

    if (
        evidence.get("schema_version") != 1
        or evidence.get("review_scope") != "repository_candidate"
        or evidence.get("reviewer_role")
        != "repository-candidate manual accessibility review"
        or evidence.get("artifact") != "staged repository candidate"
    ):
        errors.append("manual accessibility evidence lacks the governed scope")
    try:
        reviewed_at = dt.date.fromisoformat(evidence["reviewed_at"])
    except (KeyError, TypeError, ValueError):
        errors.append("manual accessibility evidence has an invalid review date")
    else:
        if reviewed_at != dt.date(2026, 7, 29):
            errors.append("manual accessibility evidence has the wrong review date")
        if reviewed_at > dt.date.today():
            errors.append("manual accessibility evidence is future-dated")

    if evidence.get("surfaces") != expected_surfaces:
        errors.append("manual accessibility evidence surface coverage is incomplete")
    if evidence.get("viewports") != expected_viewports:
        errors.append("manual accessibility evidence viewport coverage is incomplete")
    if evidence.get("themes") != expected_themes:
        errors.append("manual accessibility evidence theme coverage is incomplete")
    if evidence.get("presentation_modes") != expected_presentation_modes:
        errors.append(
            "manual accessibility evidence forced-color/print coverage is incomplete"
        )
    if evidence.get("methods") != expected_methods:
        errors.append("manual accessibility evidence methods are incomplete")

    results = evidence.get("results", {})
    if set(results) != expected_result_keys or any(
        result != "pass" for result in results.values()
    ):
        errors.append("manual accessibility evidence contains an incomplete result")
    if evidence.get("axe") != {
        "standard": "WCAG 2.2 AA",
        "unwaived_violations": 0,
    }:
        errors.append("manual accessibility evidence axe result is incomplete")
    if evidence.get("waivers") != []:
        errors.append("manual accessibility evidence contains an ungoverned waiver")
    if evidence.get("external_live_phase") != "pending_phase_5":
        errors.append("manual accessibility evidence lacks the live-phase boundary")
    qualification = evidence.get("qualification", "")
    if (
        "not user testing" not in qualification
        or "not live-platform approval" not in qualification
    ):
        errors.append("manual accessibility evidence lacks its qualification")


def _validate_governance(errors: list[str]) -> None:
    register = (REPO_ROOT / "docs/brand/system/asset-register.md").read_text(
        encoding="utf-8"
    )
    for path in sorted(ASSET_ROOT.rglob("*")):
        if not path.is_file():
            continue
        registered = f"`assets/{path.relative_to(ASSET_ROOT).as_posix()}`"
        if registered not in register:
            errors.append(f"asset register omits {registered}")
    for disposition in ("Retained", "Superseded", "Removed"):
        if disposition not in register:
            errors.append(f"asset register lacks {disposition!r} disposition")
    if (
        "`docs/website/assets/openmed-tui-preview.png` | Removed" not in register
        or (REPO_ROOT / "docs/website/assets/openmed-tui-preview.png").exists()
    ):
        errors.append("retired website preview is not removed and registered")

    exception_text = (REPO_ROOT / "docs/brand/system/site-exceptions.md").read_text(
        encoding="utf-8"
    )
    table_lines = [
        line
        for line in exception_text.splitlines()
        if line.startswith("|") and not re.fullmatch(r"\|[-|]+\|", line)
    ]
    if not table_lines:
        errors.append("site exceptions table is missing")
    else:
        headers = [part.strip() for part in table_lines[0].strip("|").split("|")]
        expected_headers = [
            "Exception",
            "Exact role",
            "Guardrail",
            "Owner",
            "Reviewed",
            "Review by",
        ]
        if headers != expected_headers:
            errors.append("site exception table columns are not governed")
        records = table_lines[1:]
        if len(records) != 7:
            errors.append(f"expected 7 registered exceptions, found {len(records)}")
        for line in records:
            fields = [part.strip() for part in line.strip("|").split("|")]
            if len(fields) != 6 or not all(fields):
                errors.append(f"incomplete site exception row: {line}")
                continue
            try:
                reviewed = dt.date.fromisoformat(fields[4])
                review_by = dt.date.fromisoformat(fields[5])
            except ValueError:
                errors.append(f"invalid site exception date: {line}")
                continue
            if review_by <= reviewed:
                errors.append(f"site exception is not time-bounded: {line}")
            if review_by < dt.date.today():
                errors.append(f"site exception review date has expired: {line}")

    version = _load_json("docs/brand/system/version.json")
    tokens = _load_json("docs/brand/system/tokens.json")
    expected_version = version["brand_system_version"]
    if (
        version["schema_version"] != 1
        or version["status"] != "current"
        or version["tokens_version"] != expected_version
        or tokens["version"] != expected_version
    ):
        errors.append("brand-system and token versions do not agree")
    try:
        released_at = dt.date.fromisoformat(version["released_at"])
        minimum_review_by = dt.date.fromisoformat(version["minimum_review_by"])
    except ValueError:
        errors.append("brand version dates are invalid")
    else:
        if minimum_review_by <= released_at:
            errors.append("brand-system review date is not forward-looking")
        if minimum_review_by < dt.date.today():
            errors.append("brand-system review date has expired")
    changelog = (BRAND_ROOT / "system/CHANGELOG.md").read_text(encoding="utf-8")
    deprecation = (BRAND_ROOT / "system/deprecation.md").read_text(encoding="utf-8")
    if f"## {expected_version}" not in changelog:
        errors.append("brand changelog lacks the current system version")
    for required in (
        "replacement",
        "owner",
        "removal date",
        "Regenerate every consumer",
    ):
        if required.lower() not in deprecation.lower():
            errors.append(f"deprecation policy lacks {required!r}")

    brand_readme = (BRAND_ROOT / "README.md").read_text(encoding="utf-8")
    required_command = (
        "uv run --frozen --extra dev --extra docs python "
        "scripts/brand/validate_system.py"
    )
    if required_command not in brand_readme:
        errors.append("brand README lacks the locked validator command")
    if "`multimodal` extra" in brand_readme:
        errors.append("brand README still gives stale Pillow dependency guidance")
    if (
        "scripts/brand/update_claims.py --refresh-github-stars" not in brand_readme
        or "must never run in CI" not in brand_readme
    ):
        errors.append("brand README lacks the explicit offline star-refresh boundary")

    cutover = (BRAND_ROOT / "social/PLATFORM_CUTOVER_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    for platform in (
        "Website",
        "GitHub",
        "Hugging Face",
        "X",
        "LinkedIn",
    ):
        if platform not in cutover:
            errors.append(f"social cutover runbook omits {platform}")
    if (
        "explicit approval" not in cutover.lower()
        or "space visibility or settings" not in cutover.lower()
        or "never read, change" not in cutover.lower()
    ):
        errors.append("social cutover runbook lacks authorization/Space guardrails")

    website_css_path = REPO_ROOT / "docs/website/assets/style.css"
    docs_css_path = REPO_ROOT / "docs/stylesheets/openmed-brand.css"
    website_css = website_css_path.read_text(encoding="utf-8")
    docs_css = docs_css_path.read_text(encoding="utf-8")
    for path, text in ((website_css_path, website_css), (docs_css_path, docs_css)):
        print_block = ""
        if "@media print" in text:
            try:
                print_block = _css_block(text, "@media print")
            except ValueError as exc:
                errors.append(str(exc))
        non_print = text.replace(print_block, "")
        for match in re.finditer(r"#[0-9A-Fa-f]{3,8}|rgba?\([^)]*\)", non_print):
            errors.append(
                f"{path.relative_to(REPO_ROOT)} uses a private raw color "
                f"{match.group(0)} outside the registered print exception"
            )
        if print_block:
            allowed_print_colors = {"#FFFFFF", "#000000", "#BBBBBB"}
            actual_print_colors = set(
                re.findall(r"#[0-9A-Fa-f]{3,8}|rgba?\([^)]*\)", print_block)
            )
            if not actual_print_colors <= allowed_print_colors:
                errors.append(
                    f"{path.relative_to(REPO_ROOT)} print palette is not registered"
                )

    if website_css.count("background-image: linear-gradient(") != 1 or not re.search(
        r"(?s)\.rotating-word\s*\{[^}]*"
        r"background-image:\s*linear-gradient\(",
        website_css,
    ):
        errors.append("the sole decorative gradient is not scoped to rotating-word")
    if re.search(r"(?:backdrop-filter|filter)\s*:\s*blur\(", website_css + docs_css):
        errors.append("consumer CSS uses prohibited blur/glass styling")
    for selector, body in re.findall(
        r"([^{}]+)\{([^{}]*)\}",
        website_css + "\n" + docs_css,
        re.DOTALL,
    ):
        if ":hover" in selector and re.search(
            r"transform:\s*(?:translate|scale)", body
        ):
            errors.append(f"hover lift/scale is prohibited in {selector.strip()!r}")

    editorial_blocks = re.findall(
        r"([^{}]+)\{([^{}]*font-family:\s*var\(--font-editorial\)[^{}]*)\}",
        website_css,
        re.DOTALL,
    )
    expected_editorial_selector = (
        ".community-number,\n.numbers-wall strong,\n.research-stats strong"
    )
    if len(editorial_blocks) != 1 or editorial_blocks[0][0].strip() != (
        expected_editorial_selector
    ):
        errors.append("Newsreader escaped its registered website display roles")
    if "--om-font-editorial" in docs_css or "Newsreader" in docs_css:
        errors.append("Newsreader must not be used by documentation prose")

    breakpoint_values = [
        int(value)
        for value in re.findall(r"@media\s*\(max-width:\s*(\d+)px\)", website_css)
    ]
    if sorted(breakpoint_values) != [900, 1080]:
        errors.append(
            f"website uses private breakpoints {sorted(set(breakpoint_values))}"
        )
    try:
        comparison_breakpoint = _css_block(
            website_css,
            "@media (max-width: 1080px)",
        )
    except ValueError as exc:
        errors.append(str(exc))
        comparison_breakpoint = ""
    comparison_selectors = {
        selector.strip()
        for selector in re.findall(r"([^{}]+)\{", comparison_breakpoint)
    }
    if comparison_selectors != {".table-scroll", ".comparison-table"}:
        errors.append("1080 px exception must contain only comparison overflow rules")
    if "overflow-x: auto;" not in _css_block(
        comparison_breakpoint, ".table-scroll"
    ) or "min-width: 850px;" not in _css_block(
        comparison_breakpoint, ".comparison-table"
    ):
        errors.append("comparison overflow exception lacks its scoped scroll contract")
    website = (REPO_ROOT / "docs/website/index.html").read_text(encoding="utf-8")
    table_scroll_tags = re.findall(
        r'<div\s+class="table-scroll"(?P<attributes>.*?)>',
        website,
        re.DOTALL,
    )
    if len(table_scroll_tags) != 1 or not all(
        attribute in table_scroll_tags[0]
        for attribute in ('role="region"', 'tabindex="0"', "aria-label=")
    ):
        errors.append("comparison overflow lacks its keyboard/accessibility affordance")

    chrome_sources = (
        (REPO_ROOT / "docs/website/index.html").read_text(encoding="utf-8")
        + website_css
        + (REPO_ROOT / "docs/website/assets/script.js").read_text(encoding="utf-8")
    )
    if re.search(r"[✓✕×→←↗●■◆★☆⚕⚙☰☀☾🔒🏥🧬]", chrome_sources):
        errors.append("website UI chrome contains an emoji or font glyph icon")
    if re.search(r"content:\s*['\"][^'\"]+['\"]", website_css):
        errors.append("website CSS uses font glyphs as generated UI chrome")

    owned_browser_sources = [
        REPO_ROOT / "docs/website/index.html",
        REPO_ROOT / "docs/website/assets/script.js",
        REPO_ROOT / "docs/website/assets/style.css",
        REPO_ROOT / "docs/stylesheets/openmed-brand.css",
        REPO_ROOT / "docs/javascripts/openmed.js",
    ]
    browser_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in owned_browser_sources
        if path.is_file()
    )
    remote_runtime_patterns = (
        r'<script[^>]+src=["\']https?://',
        r'<link[^>]+rel=["\'](?:stylesheet|icon)["\'][^>]+https?://',
        r"https?://fonts\.(?:googleapis|gstatic)",
        r"\bfetch\s*\(\s*[\"']https?://",
        r"\bXMLHttpRequest\b",
    )
    for pattern in remote_runtime_patterns:
        if re.search(pattern, browser_text, re.IGNORECASE):
            errors.append(
                f"owned browser surface performs remote runtime fetch: {pattern}"
            )

    banned_space_mutations = (
        "private=False",
        "update_repo_visibility",
        "update_repo_settings",
        "repo update --visibility",
    )
    brand_scripts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "scripts/brand").glob("*.py"))
        if path.name != "validate_system.py"
    )
    for mutation in banned_space_mutations:
        if mutation in brand_scripts:
            errors.append(f"brand tooling contains forbidden Space mutation {mutation}")


def _run_generator_checks(errors: list[str]) -> None:
    for script, mode in GENERATOR_CHECKS:
        command = [sys.executable, script]
        if mode:
            command.append(mode)
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            errors.append(f"{script} {mode} failed: {detail}")


def validate(*, reproduce: bool = True) -> list[str]:
    """Return every brand-system validation error."""

    errors: list[str] = []
    _validate_required_files(errors)
    if errors:
        return errors
    _validate_provenance(errors)
    _validate_tokens(errors)
    _validate_fonts_and_consumers(errors)
    _validate_claims(errors)
    _validate_readmes(errors)
    _validate_approved_social(errors)
    _validate_accessibility_evidence(errors)
    _validate_governance(errors)
    if reproduce:
        _run_generator_checks(errors)
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print("brand system, claims, assets, and consumers are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
