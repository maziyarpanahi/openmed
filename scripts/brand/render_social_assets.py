#!/usr/bin/env python3
"""Synchronize approved OpenMed social exports and their size derivatives."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import re
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

try:
    import PIL
    from PIL import Image, ImageDraw, PngImagePlugin
except ImportError as exc:  # pragma: no cover - exercised in minimal installs
    raise SystemExit(
        "Pillow is required to derive brand assets. Install the repository dev extra."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "docs/brand/social/_src/exports.json"
PROFILE_COPY = REPO_ROOT / "docs/brand/social/_src/profile-copy.json"
CLAIMS = REPO_ROOT / "docs/brand/system/claims.yml"
MANIFEST = REPO_ROOT / "docs/brand/social/manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256(path: Path) -> str:
    """Hash text inputs independently of checkout line-ending policy."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _pixel_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()


def _claim_value(claims: dict[str, Any], pointer: str) -> Any:
    value: Any = claims
    for part in pointer.split("."):
        value = value[part]
    return value


def _expand_templates(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand_templates(item, replacements) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_templates(item, replacements) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in replacements:
            raise ValueError(f"unknown profile-copy claim placeholder: {key}")
        return replacements[key]

    expanded = re.sub(r"\{([a-z_]+)\}", replace, value)
    if re.search(r"\{[^{}]+\}", expanded):
        raise ValueError(f"unresolved profile-copy placeholder: {expanded}")
    return expanded


def _load_profile_copy() -> dict[str, Any]:
    claims_registry = json.loads(CLAIMS.read_text(encoding="utf-8"))
    profile_source = json.loads(PROFILE_COPY.read_text(encoding="utf-8"))
    claims = claims_registry["claims"]
    for claim_name in profile_source["required_claims"]:
        if claims[claim_name]["status"] != "verified":
            raise ValueError(f"profile copy depends on unverified claim {claim_name}")
    replacements = {
        name: str(_claim_value(claims, pointer))
        for name, pointer in profile_source["claim_placeholders"].items()
    }
    return _expand_templates(profile_source, replacements)


def _save_png(image: Image.Image, path: Path) -> None:
    """Save a deterministic derivative with an explicit sRGB intent chunk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add(b"sRGB", b"\x00")
    image.save(
        path,
        format="PNG",
        pnginfo=pnginfo,
        compress_level=9,
        optimize=False,
    )


def _save_ico(image: Image.Image, path: Path, sizes: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        path,
        format="ICO",
        sizes=[tuple(size) for size in sizes],
        bitmap_format="png",
    )


def _render_preview(native: Image.Image, safe_zone: list[int], path: Path) -> None:
    """Draw a review-only safe-zone overlay over an approved native derivative."""

    preview = native.copy()
    draw = ImageDraw.Draw(preview, "RGBA")
    x0, y0, x1, y1 = safe_zone
    accent = (176, 65, 62, 255)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=accent, width=2)
    arm = max(8, min(native.size) // 30)
    for x, direction in ((x0, 1), (x1 - 1, -1)):
        draw.line((x, y0, x + direction * arm, y0), fill=accent, width=4)
        draw.line((x, y1 - 1, x + direction * arm, y1 - 1), fill=accent, width=4)
    for y, direction in ((y0, 1), (y1 - 1, -1)):
        draw.line((x0, y, x0, y + direction * arm), fill=accent, width=4)
        draw.line((x1 - 1, y, x1 - 1, y + direction * arm), fill=accent, width=4)
    _save_png(preview, path)


def _copy_to_consumers(root: Path, source: Path, destinations: list[str]) -> None:
    for destination in destinations:
        destination_path = root / destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination_path)


def _validate_source_contract(source: dict[str, Any]) -> None:
    if source["schema_version"] != 3:
        raise ValueError("social export source must use schema version 3")
    if (
        source["export_scale"] != 2
        or source["native_scale"] != 1
        or source["network"] != "forbidden"
        or source["animation_state"] != "disabled"
    ):
        raise ValueError(
            "social export source must lock DPR2, DPR1, offline, static use"
        )
    if "no visual reconstruction" not in source["derivative_policy"].lower():
        raise ValueError("social export source must prohibit visual reconstruction")
    ids = [asset["id"] for asset in source["assets"]]
    if len(ids) != len(set(ids)):
        raise ValueError("social export source contains duplicate asset ids")


def render_all(root: Path) -> dict[str, Any]:
    """Copy approved masters and derive only the sizes declared by the handoff."""

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    _validate_source_contract(source)
    profile = _load_profile_copy()
    preview_ids = set(source["preview_ids"])
    assets: list[dict[str, Any]] = []
    approved_exports: dict[str, str] = {}

    for asset in source["assets"]:
        approved_path = REPO_ROOT / asset["source"]
        if not approved_path.is_file():
            raise ValueError(f"{asset['id']}: approved export is missing")
        approved_bytes = approved_path.read_bytes()
        approved_hash = hashlib.sha256(approved_bytes).hexdigest()
        approved_exports[approved_path.name] = approved_hash

        native_size = tuple(asset["size"])
        expected_master_size = tuple(
            value * source["export_scale"] for value in native_size
        )
        with Image.open(approved_path) as approved_source:
            approved = approved_source.convert("RGBA")
            if approved_source.mode != "RGBA" or approved.size != expected_master_size:
                raise ValueError(
                    f"{asset['id']}: approved export must be RGBA "
                    f"{expected_master_size[0]}x{expected_master_size[1]}"
                )
            native = approved.resize(native_size, Image.Resampling.LANCZOS)
            approved_pixel_hash = _pixel_sha256(approved)

        master_path = root / asset["master"]
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_bytes(approved_bytes)

        native_path = root / asset["native"]
        _save_png(native, native_path)
        _copy_to_consumers(root, native_path, asset["copy_to"])

        entry = {
            **asset,
            "dpr": {"master": source["export_scale"], "native": source["native_scale"]},
            "mode": "RGBA",
            "approved_export_sha256": approved_hash,
            "approved_export_pixel_sha256": approved_pixel_hash,
            "master_sha256": _sha256(master_path),
            "master_exact_copy": master_path.read_bytes() == approved_bytes,
            "native_sha256": _sha256(native_path),
            "native_pixel_sha256": _pixel_sha256(native),
            "native_derivative": {
                "method": "Pillow Image.Resampling.LANCZOS",
                "source": asset["source"],
                "source_size": list(expected_master_size),
                "output_size": list(native_size),
                "visual_edits": "none",
            },
            "safe_zone_crop_pixel_sha256": hashlib.sha256(
                native.crop(tuple(asset["safe_zone"])).tobytes()
            ).hexdigest(),
        }
        if asset["id"] in preview_ids:
            preview_path = (
                root / "docs/brand/social/previews" / f"{asset['id']}-safe-zone.png"
            )
            _render_preview(native, asset["safe_zone"], preview_path)
            entry["safe_zone_preview"] = preview_path.relative_to(root).as_posix()
            entry["safe_zone_preview_sha256"] = _sha256(preview_path)
        assets.append(entry)

    derived_assets: list[dict[str, Any]] = []
    for item in source["derived"]:
        output = root / item["output"]
        source_path = root / item["source"]
        if not source_path.is_file():
            source_path = REPO_ROOT / item["source"]
        with Image.open(source_path) as source_image:
            image = source_image.convert("RGBA")
            if item["kind"] == "resize":
                image = image.resize(tuple(item["size"]), Image.Resampling.LANCZOS)
                _save_png(image, output)
            elif item["kind"] == "favicon_ico":
                _save_ico(image, output, item["sizes"])
            else:
                raise ValueError(f"unknown derived kind: {item['kind']}")
        _copy_to_consumers(root, output, item["copy_to"])
        derived_assets.append(
            {
                **item,
                "mode": image.mode,
                "sha256": _sha256(output),
                "pixel_sha256": _pixel_sha256(image),
                "visual_edits": "none",
            }
        )

    profile_bytes = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    source_readme = REPO_ROOT / source["source_readme"]
    source_paths = (SOURCE, PROFILE_COPY, CLAIMS, source_readme)
    return {
        "schema_version": 3,
        "source": SOURCE.relative_to(REPO_ROOT).as_posix(),
        "profile_copy": PROFILE_COPY.relative_to(REPO_ROOT).as_posix(),
        "source_hashes": {
            path.relative_to(REPO_ROOT).as_posix(): _source_sha256(path)
            for path in source_paths
        },
        "approved_exports": dict(sorted(approved_exports.items())),
        "resolved_profile_copy": profile,
        "resolved_profile_copy_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "synchronizer": {
            "implementation": Path(__file__)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix(),
            "engine": "byte copy plus Pillow size derivatives",
            "pillow_version": PIL.__version__,
            "master_policy": "byte-identical approved export",
            "resampling": "LANCZOS",
            "derivative_png_compress_level": 9,
            "network": "blocked",
            "master_color_profile": "source-preserved",
            "derivative_color_profile": "sRGB",
            "animation_state": "disabled",
            "visual_reconstruction": "forbidden",
        },
        "assets": assets,
        "derived_assets": derived_assets,
        "external_distribution": [
            {
                "source": "docs/brand/social/github-social.png",
                "approved_export": ("docs/brand/social/exports/github-social-2x.png"),
                "target": "maziyarpanahi/openmed social preview",
                "authorization": "explicit external profile edit",
            },
            {
                "source": "docs/brand/social/hf-card.png",
                "approved_export": "docs/brand/social/exports/hf-org-2x.png",
                "target": "OpenMed/README openmed-social-card.png",
                "authorization": "explicit external repository update",
            },
        ],
    }


@contextlib.contextmanager
def _network_blocked() -> Iterator[None]:
    original_socket = socket.socket
    original_connection = socket.create_connection

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access is forbidden during brand synchronization")

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_connection  # type: ignore[assignment]


def _manifest_without_derivative_byte_hashes(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return the portable manifest contract used by cross-platform checks.

    Pillow produces identical RGBA pixels for these derivatives on supported
    platforms, but its PNG/ICO compression bytes can differ with the native
    codec build. The committed manifest still records canonical repository
    byte hashes; cross-platform reproduction compares those derivatives by
    decoded pixels and keeps byte equality for approved master exports.
    """

    portable = copy.deepcopy(manifest)
    for asset in portable["assets"]:
        asset.pop("native_sha256", None)
        asset.pop("safe_zone_preview_sha256", None)
    for asset in portable["derived_assets"]:
        asset.pop("sha256", None)
    return portable


def _image_pixel_state(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return rgba.size, _pixel_sha256(rgba)


def _ico_pixel_state(path: Path) -> dict[str, str]:
    with Image.open(path) as image:
        sizes = sorted(image.ico.sizes())  # type: ignore[attr-defined]
        return {
            f"{width}x{height}": _pixel_sha256(
                image.ico.getimage((width, height))  # type: ignore[attr-defined]
            )
            for width, height in sizes
        }


def _same_rendered_pixels(actual: Path, expected: Path) -> bool:
    if actual.suffix.lower() == ".ico":
        return _ico_pixel_state(actual) == _ico_pixel_state(expected)
    return _image_pixel_state(actual) == _image_pixel_state(expected)


def _validate_committed_output_hashes(
    manifest: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for asset in manifest["assets"]:
        for field, key in (
            ("master", "master_sha256"),
            ("native", "native_sha256"),
            ("safe_zone_preview", "safe_zone_preview_sha256"),
        ):
            relative = asset.get(field)
            expected_hash = asset.get(key)
            if relative is None or expected_hash is None:
                continue
            path = REPO_ROOT / relative
            if not path.is_file() or _sha256(path) != expected_hash:
                mismatches.append(relative)
    for asset in manifest["derived_assets"]:
        relative = asset["output"]
        path = REPO_ROOT / relative
        if not path.is_file() or _sha256(path) != asset["sha256"]:
            mismatches.append(relative)
    return mismatches


def _validate_rendered_outputs(
    rendered_manifest: dict[str, Any],
    temp_root: Path,
) -> list[str]:
    mismatches: list[str] = []
    for asset in rendered_manifest["assets"]:
        master = asset["master"]
        actual_master = REPO_ROOT / master
        expected_master = temp_root / master
        if (
            not actual_master.is_file()
            or actual_master.read_bytes() != expected_master.read_bytes()
        ):
            mismatches.append(master)

        native = asset["native"]
        actual_native = REPO_ROOT / native
        expected_native = temp_root / native
        if not actual_native.is_file() or not _same_rendered_pixels(
            actual_native, expected_native
        ):
            mismatches.append(native)
        for relative in asset["copy_to"]:
            consumer = REPO_ROOT / relative
            if (
                not consumer.is_file()
                or consumer.read_bytes() != actual_native.read_bytes()
            ):
                mismatches.append(relative)

        preview = asset.get("safe_zone_preview")
        if preview is not None:
            actual_preview = REPO_ROOT / preview
            expected_preview = temp_root / preview
            if not actual_preview.is_file() or not _same_rendered_pixels(
                actual_preview, expected_preview
            ):
                mismatches.append(preview)

    for asset in rendered_manifest["derived_assets"]:
        output = asset["output"]
        actual_output = REPO_ROOT / output
        expected_output = temp_root / output
        if not actual_output.is_file() or not _same_rendered_pixels(
            actual_output, expected_output
        ):
            mismatches.append(output)
        for relative in asset["copy_to"]:
            consumer = REPO_ROOT / relative
            if (
                not consumer.is_file()
                or consumer.read_bytes() != actual_output.read_bytes()
            ):
                mismatches.append(relative)
    return mismatches


def _check() -> int:
    if not MANIFEST.exists():
        print("social manifest is missing; synchronize with --write", file=sys.stderr)
        return 1
    committed_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="openmed-social-sync-") as temp:
        root = Path(temp)
        with _network_blocked():
            rendered_manifest = render_all(root)
        if _manifest_without_derivative_byte_hashes(
            committed_manifest
        ) != _manifest_without_derivative_byte_hashes(rendered_manifest):
            print("social manifest is stale; synchronize with --write", file=sys.stderr)
            return 1
        mismatches = _validate_committed_output_hashes(committed_manifest)
        mismatches.extend(_validate_rendered_outputs(rendered_manifest, root))
        if mismatches:
            print(
                "stale social outputs: " + ", ".join(sorted(set(mismatches))),
                file=sys.stderr,
            )
            return 1
    print(
        "approved social exports reproduce byte-exactly and derivatives "
        "reproduce pixel-exactly offline"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        return _check()

    with _network_blocked():
        manifest = render_all(REPO_ROOT)
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"synchronized {len(manifest['assets'])} approved social exports")
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
