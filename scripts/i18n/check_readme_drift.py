#!/usr/bin/env python3
"""Detect source README changes that have not been reviewed for translations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("docs/i18n/readme_section_hashes.json")
DEFAULT_TRANSLATIONS = (Path("README.zh-CN.md"), Path("README.hi.md"))
GLOSSARY = Path("docs/i18n/glossary.md")
PREAMBLE = "__preamble__"

_H2 = re.compile(r"^##\s+(.+?)\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")


class DriftError(RuntimeError):
    """Raised when a README translation or its manifest is stale."""


@dataclass(frozen=True)
class Section:
    """One README preamble or H2 section."""

    heading: str
    content: str


def split_h2_sections(text: str) -> list[Section]:
    """Split Markdown into the preamble followed by fenced-code-aware H2 sections."""
    sections: list[Section] = []
    heading = PREAMBLE
    lines: list[str] = []
    fence: str | None = None

    for line in text.splitlines(keepends=True):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None

        heading_match = _H2.match(line) if fence is None else None
        if heading_match:
            sections.append(Section(heading=heading, content="".join(lines)))
            heading = heading_match.group(1)
            lines = []
        else:
            lines.append(line)

    sections.append(Section(heading=heading, content="".join(lines)))
    return sections


def section_sha256(section: Section) -> str:
    """Return the stable SHA-256 digest for a section body."""
    normalized = section.content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_sections(path: Path) -> list[Section]:
    if not path.is_file():
        raise DriftError(f"README file does not exist: {path}")
    return split_h2_sections(path.read_text(encoding="utf-8"))


def _display_heading(heading: str) -> str:
    return "README preamble" if heading == PREAMBLE else f"section {heading!r}"


def _relative_targets(text: str) -> set[str]:
    targets = {match.group(1) for match in _MARKDOWN_LINK.finditer(text)}
    targets.update(match.group(1) for match in _HTML_LINK.finditer(text))
    return targets


def validate_relative_links(root: Path, readme: Path) -> None:
    """Raise when a local link or image in a README does not resolve."""
    missing: list[str] = []
    text = readme.read_text(encoding="utf-8")
    for target in sorted(_relative_targets(text)):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith(("#", "/", "//")):
            continue
        relative_path = unquote(parsed.path)
        if relative_path and not (root / relative_path).exists():
            missing.append(target)

    if missing:
        formatted = "\n".join(f"- {readme.name}: {target}" for target in missing)
        raise DriftError(f"README contains unresolved local links:\n{formatted}")


def _validate_translation_structure(
    source_path: Path,
    source_sections: list[Section],
    translation_path: Path,
    translation_sections: list[Section],
) -> None:
    if len(source_sections) != len(translation_sections):
        raise DriftError(
            f"{translation_path.name} has {len(translation_sections) - 1} H2 sections; "
            f"{source_path.name} has {len(source_sections) - 1}. Add a translated "
            "counterpart for every source H2 section before updating the manifest."
        )

    source_link = f'href="{translation_path.name}"'
    translation_link = f'href="{source_path.name}"'
    if source_link not in source_sections[0].content:
        raise DriftError(
            f"{source_path.name} language switcher must link to "
            f"{translation_path.name}."
        )
    if translation_link not in translation_sections[0].content:
        raise DriftError(
            f"{translation_path.name} language switcher must link to "
            f"{source_path.name}."
        )

    source_targets = _relative_targets(
        "".join(section.content for section in source_sections)
    )
    translation_targets = _relative_targets(
        "".join(section.content for section in translation_sections)
    )
    source_targets.discard(translation_path.name)
    translation_targets.discard(source_path.name)
    if source_targets != translation_targets:
        missing = sorted(source_targets - translation_targets)
        extra = sorted(translation_targets - source_targets)
        details = []
        if missing:
            details.append(f"missing targets: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected targets: {', '.join(extra)}")
        raise DriftError(
            f"{translation_path.name} must preserve every link, badge, and image "
            f"target from {source_path.name} ({'; '.join(details)})."
        )


def build_manifest(root: Path, manifest_path: Path) -> dict[str, object]:
    """Build a manifest from the current source and reviewed translations."""
    if not (root / GLOSSARY).is_file():
        raise DriftError(f"README translation glossary is missing: {GLOSSARY}")

    source_rel = Path("README.md")
    source_path = root / source_rel
    source_sections = _read_sections(source_path)
    available_defaults = tuple(
        path for path in DEFAULT_TRANSLATIONS if (root / path).is_file()
    )

    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_names = tuple(Path(name) for name in current.get("translations", {}))
        translation_names = tuple(dict.fromkeys((*current_names, *available_defaults)))
    else:
        translation_names = available_defaults

    translations: dict[str, object] = {}
    for translation_rel in translation_names:
        translation_path = root / translation_rel
        translation_sections = _read_sections(translation_path)
        _validate_translation_structure(
            source_path,
            source_sections,
            translation_path,
            translation_sections,
        )
        validate_relative_links(root, translation_path)

        entries = []
        for source, translation in zip(
            source_sections, translation_sections, strict=True
        ):
            entries.append(
                {
                    "source_heading": source.heading,
                    "translation_heading": translation.heading,
                    "source_sha256": section_sha256(source),
                    "translation_sha256": section_sha256(translation),
                }
            )
        translations[translation_rel.as_posix()] = {"sections": entries}

    return {
        "version": 1,
        "source": source_rel.as_posix(),
        "glossary": GLOSSARY.as_posix(),
        "translations": translations,
    }


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise DriftError(
            f"README translation manifest is missing: {path}. "
            "Run scripts/i18n/check_readme_drift.py --update after reviewing "
            f"translations against {GLOSSARY}."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DriftError(f"Cannot read README translation manifest: {exc}") from exc
    if manifest.get("version") != 1:
        raise DriftError("README translation manifest must use version 1.")
    return manifest


def check_repository(root: Path, manifest_path: Path) -> None:
    """Validate README section parity, hashes, switchers, and local links."""
    manifest = _load_manifest(manifest_path)
    glossary_rel = Path(str(manifest.get("glossary", "")))
    if glossary_rel != GLOSSARY or not (root / glossary_rel).is_file():
        raise DriftError(
            f"README translation manifest must reference the existing {GLOSSARY}."
        )

    source_rel = Path(str(manifest.get("source", "")))
    source_path = root / source_rel
    source_sections = _read_sections(source_path)
    translations = manifest.get("translations")
    if not isinstance(translations, dict) or not translations:
        raise DriftError("README translation manifest has no translations.")

    errors: list[str] = []
    for translation_name, translation_manifest in translations.items():
        translation_path = root / translation_name
        translation_sections = _read_sections(translation_path)
        try:
            _validate_translation_structure(
                source_path,
                source_sections,
                translation_path,
                translation_sections,
            )
            validate_relative_links(root, translation_path)
        except DriftError as exc:
            errors.append(str(exc))
            continue

        if not isinstance(translation_manifest, dict):
            errors.append(f"Manifest entry for {translation_name} must be an object.")
            continue
        entries = translation_manifest.get("sections")
        if not isinstance(entries, list) or len(entries) != len(source_sections):
            errors.append(
                f"Manifest for {translation_name} has the wrong section count."
            )
            continue

        for source, translation, entry in zip(
            source_sections, translation_sections, entries, strict=True
        ):
            if not isinstance(entry, dict):
                errors.append(f"Manifest section for {translation_name} is invalid.")
                continue
            if entry.get("source_heading") != source.heading:
                errors.append(
                    f"{translation_name} has no reviewed manifest entry for "
                    f"{_display_heading(source.heading)}."
                )
                continue
            if entry.get("translation_heading") != translation.heading:
                errors.append(
                    f"{translation_name} heading changed for "
                    f"{_display_heading(source.heading)}."
                )
            if entry.get("source_sha256") != section_sha256(source):
                errors.append(
                    f"{translation_name} is stale for "
                    f"{_display_heading(source.heading)} because {source_rel.name} "
                    "changed. Review the translation using "
                    f"{GLOSSARY}, then update the manifest."
                )
            if entry.get("translation_sha256") != section_sha256(translation):
                errors.append(
                    f"The manifest entry for {translation_name} "
                    f"{_display_heading(translation.heading)} is stale. Review the "
                    "translation, then update the manifest."
                )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise DriftError(
            "README translation drift detected:\n"
            f"{details}\n"
            "Run `python scripts/i18n/check_readme_drift.py --update` only after "
            f"the paired translation has been reviewed against {GLOSSARY}."
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (defaults to the root containing this script).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest path, relative to --root unless absolute.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Refresh hashes after reviewing every changed translation section.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the README translation drift check or refresh its manifest."""
    args = _parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    try:
        if args.update:
            manifest = build_manifest(root, manifest_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            display_path = (
                manifest_path.relative_to(root)
                if manifest_path.is_relative_to(root)
                else manifest_path
            )
            print(f"Updated {display_path}")

        check_repository(root, manifest_path)
    except DriftError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("README translation drift check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
