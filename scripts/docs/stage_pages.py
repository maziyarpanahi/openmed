#!/usr/bin/env python3
"""Build and verify the exact OpenMed GitHub Pages artifact.

This is the single staging entrypoint used by both local development and CI.
It renders committed generated docs, builds MkDocs below ``/docs/``, overlays
the marketing site at the artifact root, and writes a content-addressed
manifest for every staged route and asset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from fnmatch import fnmatchcase
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
MKDOCS_CONFIG = ROOT / "mkdocs.yml"
DEFAULT_OUTPUT_DIR = ROOT / "site"
DEFAULT_WEBSITE_DIR = ROOT / "docs" / "website"
DEFAULT_PUBLICATION = ROOT / "docs" / "brand" / "system" / "publication.yml"
DEFAULT_LEADERBOARD_DIR = ROOT / "docs" / "eval" / "benchmark-leaderboard"
MANIFEST_NAME = "pages-manifest.json"
MANIFEST_SCHEMA_VERSION = 1
PUBLIC_ORIGIN = "https://openmed.life"

_LOCALE_SOURCE_RE = re.compile(r"^(?P<stem>.+)\.(?P<locale>[a-z]{2,3})\.md$")
_RELEASE_TAG_RE = re.compile(r"v?[0-9]+(?:\.[0-9]+){2}(?:[-+][0-9A-Za-z.-]+)?")
_CSS_REFERENCE_RE = re.compile(
    r"""(?:url\(\s*|@import\s+)(?P<quote>["']?)(?P<url>[^"')\s;]+)(?P=quote)""",
    re.IGNORECASE,
)
_INTERNAL_HOSTS = frozenset({"openmed.life", "www.openmed.life"})


class PageStagingError(RuntimeError):
    """Raised when the Pages artifact cannot be staged safely."""


class _HTMLReferenceParser(HTMLParser):
    """Collect URL-bearing attributes and fragment targets from HTML."""

    _URL_ATTRIBUTES = frozenset({"action", "href", "poster", "src"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.alternates: list[tuple[str, str]] = []
        self.canonicals: list[str] = []
        self.direction = ""
        self.document_language = ""
        self.icons: list[str] = []
        self.ids: set[str] = set()
        self.meta: dict[str, list[str]] = {}
        self.references: list[str] = []
        self.scripts: list[str] = []
        self.stylesheets: list[str] = []
        self._in_title = False
        self._title_parts: list[str] = []

    @property
    def title(self) -> str:
        """Return the normalized document title."""

        return " ".join("".join(self._title_parts).split())

    def meta_values(self, key: str) -> list[str]:
        """Return metadata values for one case-insensitive name/property."""

        return self.meta.get(key.casefold(), [])

    def handle_starttag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        lowered_tag = tag.casefold()
        attribute_map = {
            name.casefold(): value for name, value in attributes if value is not None
        }
        if lowered_tag == "html":
            self.document_language = attribute_map.get("lang", "").strip()
            self.direction = attribute_map.get("dir", "").strip().casefold()
        elif lowered_tag == "title":
            self._in_title = True
        elif lowered_tag == "meta":
            key = (
                attribute_map.get("name")
                or attribute_map.get("property")
                or attribute_map.get("http-equiv")
            )
            content = attribute_map.get("content")
            if key and content is not None:
                self.meta.setdefault(key.casefold(), []).append(content.strip())
        elif lowered_tag == "link":
            rel = {
                value.casefold()
                for value in attribute_map.get("rel", "").split()
                if value
            }
            href = attribute_map.get("href", "").strip()
            if href and "canonical" in rel:
                self.canonicals.append(href)
            if href and "alternate" in rel and attribute_map.get("hreflang"):
                self.alternates.append(
                    (attribute_map["hreflang"].strip().casefold(), href)
                )
            if href and rel & {"apple-touch-icon", "icon", "shortcut"}:
                self.icons.append(href)
            if href and "stylesheet" in rel:
                self.stylesheets.append(href)
        elif lowered_tag == "script":
            source = attribute_map.get("src", "").strip()
            if source:
                self.scripts.append(source)

        for name, value in attributes:
            if value is None:
                continue
            lowered = name.casefold()
            if lowered in {"id", "name"} and value:
                self.ids.add(value)
            elif lowered in self._URL_ATTRIBUTES:
                self.references.append(value)
            elif lowered == "srcset":
                self.references.extend(
                    candidate.strip().split(maxsplit=1)[0]
                    for candidate in value.split(",")
                    if candidate.strip()
                )

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(root: Path) -> dict[str, str]:
    """Return a deterministic relative-path to SHA-256 mapping."""

    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def changed_snapshot_paths(
    before: Mapping[str, str], after: Mapping[str, str]
) -> list[str]:
    """Return sorted paths added, removed, or changed between snapshots."""

    return sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )


def resolve_release_tag(requested: str | None) -> str:
    """Resolve and validate the release tag used by generated public pages."""

    if requested is None:
        from openmed.__about__ import __version__

        requested = f"v{__version__}"
    tag = requested.strip()
    if not _RELEASE_TAG_RE.fullmatch(tag):
        raise PageStagingError(
            "Release tag must be a semantic version such as v2.0.0; "
            f"received {requested!r}"
        )
    return tag if tag.startswith("v") else f"v{tag}"


def load_publication(path: Path) -> dict[str, Any]:
    """Load and minimally validate the documentation publication contract."""

    if not path.is_file():
        raise PageStagingError(f"Publication contract does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PageStagingError(f"Could not read publication contract: {exc}") from exc
    if not isinstance(payload, dict):
        raise PageStagingError("Publication contract must be a YAML mapping")
    if payload.get("version") != 1:
        raise PageStagingError("Publication contract requires version: 1")
    if payload.get("fallback_policy") != "disabled":
        raise PageStagingError(
            "Publication contract must set fallback_policy: disabled"
        )
    for key in (
        "expected_routes",
        "expected_assets",
        "translation_groups",
        "classification",
        "metadata_policy",
        "fixtures",
    ):
        if key not in payload:
            raise PageStagingError(f"Publication contract is missing {key!r}")
    if not isinstance(payload["expected_routes"], list):
        raise PageStagingError("publication expected_routes must be a list")
    if not isinstance(payload["expected_assets"], list):
        raise PageStagingError("publication expected_assets must be a list")
    if not isinstance(payload["translation_groups"], dict):
        raise PageStagingError("publication translation_groups must be a mapping")
    if not isinstance(payload["classification"], dict):
        raise PageStagingError("publication classification must be a mapping")
    if not isinstance(payload["metadata_policy"], dict):
        raise PageStagingError("publication metadata_policy must be a mapping")
    if not isinstance(payload["fixtures"], list):
        raise PageStagingError("publication fixtures must be a list")
    validate_publication_contract(payload)
    return payload


def _safe_relative_source(source: object, *, label: str, suffix: str) -> str:
    """Return one normalized repository-relative publication source."""

    if not isinstance(source, str) or not source.strip():
        raise PageStagingError(f"{label} must be non-empty text")
    candidate = source.strip().replace("\\", "/")
    if candidate.startswith("/") or candidate.endswith("/"):
        raise PageStagingError(f"Unsafe {label}: {source!r}")
    path = Path(candidate)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != candidate:
        raise PageStagingError(f"Unsafe {label}: {source!r}")
    if path.suffix.casefold() != suffix:
        raise PageStagingError(f"{label} must end in {suffix}: {source!r}")
    return path.as_posix()


def _nav_sources(node: object) -> list[str]:
    """Flatten Markdown source paths from a MkDocs nav tree."""

    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [source for child in node for source in _nav_sources(child)]
    if isinstance(node, Mapping):
        return [source for child in node.values() for source in _nav_sources(child)]
    return []


def markdown_output_path(source: str) -> str:
    """Map a default-language Markdown source to its staged HTML path."""

    source_path = Path(
        _safe_relative_source(source, label="Markdown source", suffix=".md")
    )
    stem_path = source_path.with_suffix("")
    if stem_path.name == "index":
        relative = stem_path.parent / "index.html"
    else:
        relative = stem_path / "index.html"
    return (Path("docs") / relative).as_posix()


def _excluded_source(source: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(source, pattern) for pattern in patterns)


def _publication_fixture_map(
    publication: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    fixtures = publication.get("fixtures", [])
    if not isinstance(fixtures, list):
        raise PageStagingError("publication fixtures must be a list")
    mapped: dict[str, Mapping[str, Any]] = {}
    expected = set(normalized_expected_paths(publication))
    for position, raw_fixture in enumerate(fixtures):
        if not isinstance(raw_fixture, Mapping):
            raise PageStagingError(f"Publication fixture {position} must be a mapping")
        source = _safe_relative_source(
            raw_fixture.get("source"),
            label=f"fixture {position} source",
            suffix=".html",
        )
        route_values = normalized_route_values(
            [raw_fixture.get("route")],
            label=f"fixture {position} route",
        )
        route = route_values[0]
        if route != (Path("docs") / source).as_posix():
            raise PageStagingError(
                f"Fixture {source!r} must stage at {(Path('docs') / source).as_posix()!r}"
            )
        if route not in expected:
            raise PageStagingError(
                f"Fixture route {route!r} is not listed in expected_routes"
            )
        if raw_fixture.get("indexing") != "noindex,nofollow":
            raise PageStagingError(
                f"Fixture {source!r} must set indexing: noindex,nofollow"
            )
        if raw_fixture.get("data_policy") != "synthetic_only":
            raise PageStagingError(
                f"Fixture {source!r} must set data_policy: synthetic_only"
            )
        if raw_fixture.get("social_metadata") != "prohibited":
            raise PageStagingError(
                f"Fixture {source!r} must set social_metadata: prohibited"
            )
        if (
            not isinstance(raw_fixture.get("purpose"), str)
            or not str(raw_fixture["purpose"]).strip()
        ):
            raise PageStagingError(f"Fixture {source!r} requires a purpose")
        if not (DOCS_DIR / source).is_file():
            raise PageStagingError(f"Fixture source does not exist: {source}")
        if route in mapped:
            raise PageStagingError(f"Duplicate fixture route: {route}")
        mapped[route] = raw_fixture
    return mapped


def validate_publication_contract(publication: Mapping[str, Any]) -> None:
    """Validate classification, translation, fixture, and metadata policy."""

    expected_routes = set(normalized_expected_paths(publication))
    expected_assets = set(normalized_expected_assets(publication))
    if overlap := expected_routes & expected_assets:
        raise PageStagingError(
            "Publication paths cannot be both routes and assets: "
            + ", ".join(sorted(overlap))
        )
    website_assets = {
        path.relative_to(DEFAULT_WEBSITE_DIR).as_posix()
        for path in DEFAULT_WEBSITE_DIR.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "index.html"
    }
    expected_website_assets = {
        path for path in expected_assets if not path.startswith("docs/")
    }
    if website_assets != expected_website_assets:
        raise PageStagingError(
            "publication expected_assets differs from website source assets "
            f"(missing: {sorted(website_assets - expected_website_assets)}; "
            f"stale: {sorted(expected_website_assets - website_assets)})"
        )

    classification = publication.get("classification")
    if not isinstance(classification, Mapping):
        raise PageStagingError("publication classification must be a mapping")
    classified: dict[str, list[str]] = {}
    for key in ("navigated", "link_only", "excluded"):
        values = classification.get(key)
        if not isinstance(values, list):
            raise PageStagingError(f"publication classification.{key} must be a list")
        normalized: list[str] = []
        for value in values:
            if key == "excluded":
                if not isinstance(value, str) or not value.strip():
                    raise PageStagingError(
                        "publication excluded patterns must be non-empty text"
                    )
                candidate = value.strip().replace("\\", "/")
                if candidate.startswith("/") or ".." in Path(candidate).parts:
                    raise PageStagingError(
                        f"Unsafe publication excluded pattern: {value!r}"
                    )
            else:
                candidate = _safe_relative_source(
                    value,
                    label=f"classification.{key} source",
                    suffix=".md",
                )
                if not (DOCS_DIR / candidate).is_file():
                    raise PageStagingError(
                        f"Classified documentation source does not exist: {candidate}"
                    )
            normalized.append(candidate)
        if len(normalized) != len(set(normalized)):
            raise PageStagingError(f"publication classification.{key} has duplicates")
        classified[key] = normalized

    navigated = set(classified["navigated"])
    link_only = set(classified["link_only"])
    if overlap := navigated & link_only:
        raise PageStagingError(
            "Documentation sources cannot be both navigated and link-only: "
            + ", ".join(sorted(overlap))
        )

    translations = publication.get("translation_groups")
    if not isinstance(translations, Mapping):
        raise PageStagingError("publication translation_groups must be a mapping")
    translated_sources: set[str] = set()
    group_sources: set[str] = set()
    for group_name, raw_group in translations.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise PageStagingError("Translation group names must be non-empty text")
        if not isinstance(raw_group, Mapping):
            raise PageStagingError(
                f"Translation group {group_name!r} must be a mapping"
            )
        source = _safe_relative_source(
            raw_group.get("source"),
            label=f"translation group {group_name!r} source",
            suffix=".md",
        )
        if source not in navigated | link_only:
            raise PageStagingError(
                f"Translation group source is not classified: {source}"
            )
        if source in group_sources:
            raise PageStagingError(f"Duplicate translation group source: {source}")
        group_sources.add(source)
        raw_translations = raw_group.get("translations")
        if not isinstance(raw_translations, Mapping) or not raw_translations:
            raise PageStagingError(
                f"Translation group {group_name!r} requires translations"
            )
        for locale, translated in raw_translations.items():
            if (
                not isinstance(locale, str)
                or re.fullmatch(r"[a-z]{2,3}", locale) is None
                or not isinstance(translated, str)
            ):
                raise PageStagingError(
                    f"Translation group {group_name!r} has an invalid translation"
                )
            source_path = _safe_relative_source(
                translated,
                label=f"translation group {group_name!r} {locale} source",
                suffix=".md",
            )
            translated_output_path(source_path, locale)
            if not (DOCS_DIR / source_path).is_file():
                raise PageStagingError(
                    f"Translated documentation source does not exist: {source_path}"
                )
            if source_path in translated_sources:
                raise PageStagingError(f"Duplicate translated source: {source_path}")
            translated_sources.add(source_path)

    actual_translations = {
        path.relative_to(DOCS_DIR).as_posix()
        for path in DOCS_DIR.rglob("*.md")
        if _LOCALE_SOURCE_RE.fullmatch(path.name)
        and not _excluded_source(
            path.relative_to(DOCS_DIR).as_posix(), classified["excluded"]
        )
    }
    if actual_translations != translated_sources:
        missing = sorted(actual_translations - translated_sources)
        stale = sorted(translated_sources - actual_translations)
        raise PageStagingError(
            "Translation group coverage differs from localized sources "
            f"(unclassified: {missing}; missing: {stale})"
        )

    default_sources = {
        path.relative_to(DOCS_DIR).as_posix()
        for path in DOCS_DIR.rglob("*.md")
        if path.relative_to(DOCS_DIR).as_posix() not in translated_sources
        and not _excluded_source(
            path.relative_to(DOCS_DIR).as_posix(), classified["excluded"]
        )
    }
    if navigated | link_only != default_sources:
        unclassified = sorted(default_sources - navigated - link_only)
        stale = sorted((navigated | link_only) - default_sources)
        raise PageStagingError(
            "Documentation classification differs from publishable Markdown "
            f"(unclassified: {unclassified}; missing: {stale})"
        )

    try:
        mkdocs = yaml.load(
            MKDOCS_CONFIG.read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PageStagingError(f"Could not validate MkDocs navigation: {exc}") from exc
    if not isinstance(mkdocs, Mapping):
        raise PageStagingError("mkdocs.yml must be a mapping")
    nav_sources = _nav_sources(mkdocs.get("nav"))
    if nav_sources != classified["navigated"]:
        raise PageStagingError(
            "publication classification.navigated must exactly match mkdocs.yml nav"
        )
    copyright_text = str(mkdocs.get("copyright", ""))
    if "OpenMed SDK" not in copyright_text or "Apache-2.0" not in copyright_text:
        raise PageStagingError(
            "MkDocs copyright must scope Apache-2.0 to the OpenMed SDK"
        )
    identity_pairs = (
        (
            DOCS_DIR / "brand" / "assets" / "open-cross.svg",
            DOCS_DIR / "assets" / "openmed-mark.svg",
        ),
        (
            DOCS_DIR / "brand" / "assets" / "open-cross.svg",
            DOCS_DIR / "assets" / "openmed-favicon.svg",
        ),
        (
            DOCS_DIR / "brand" / "assets" / "open-cross-inverse.svg",
            DOCS_DIR / "assets" / "openmed-mark-inverse.svg",
        ),
    )
    for canonical, consumer in identity_pairs:
        try:
            canonical_bytes = canonical.read_bytes()
            consumer_bytes = consumer.read_bytes()
        except OSError as exc:
            raise PageStagingError(
                f"Could not validate documentation identity asset: {exc}"
            ) from exc
        if consumer_bytes != canonical_bytes:
            raise PageStagingError(
                f"Documentation identity asset drift: {consumer.relative_to(ROOT)} "
                f"must match {canonical.relative_to(ROOT)}"
            )

    metadata_policy = publication.get("metadata_policy")
    expected_metadata_policy = {
        "indexable_pages": {
            "title": "required",
            "description": "required",
            "canonical": "required",
            "favicon": "required_local",
            "open_graph": "required",
            "twitter_card": "required",
        },
        "localized_pages": {
            "alternates": "real_translations_only",
            "default_language_fallback": "prohibited",
        },
    }
    if metadata_policy != expected_metadata_policy:
        raise PageStagingError(
            "publication metadata_policy does not match the enforced policy"
        )

    _publication_fixture_map(publication)


def normalized_route_values(values: Iterable[object], *, label: str) -> list[str]:
    """Return safe, normalized staged paths from arbitrary route values."""

    normalized: list[str] = []
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise PageStagingError(f"Every {label} must be text")
        candidate = raw.strip().replace("\\", "/")
        path = Path(candidate)
        if (
            candidate.startswith("/")
            or path.is_absolute()
            or ".." in path.parts
            or candidate.endswith("/")
            or path.as_posix() != candidate
        ):
            raise PageStagingError(f"Unsafe {label}: {raw!r}")
        normalized.append(path.as_posix())
    if len(normalized) != len(set(normalized)):
        raise PageStagingError(f"{label} contains duplicates")
    return sorted(normalized)


def normalized_expected_paths(publication: Mapping[str, Any]) -> list[str]:
    """Return safe, normalized staged paths from ``expected_routes``."""

    return normalized_route_values(
        publication.get("expected_routes", []),
        label="publication expected_routes",
    )


def normalized_expected_assets(publication: Mapping[str, Any]) -> list[str]:
    """Return safe, normalized staged paths from ``expected_assets``."""

    return normalized_route_values(
        publication.get("expected_assets", []),
        label="publication expected_assets",
    )


def translated_output_path(source: str, locale: str) -> str:
    """Map a docs-relative localized Markdown source to its staged HTML path."""

    source_path = Path(source)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise PageStagingError(f"Unsafe translated source path: {source!r}")
    match = _LOCALE_SOURCE_RE.fullmatch(source_path.name)
    if match is None or match.group("locale") != locale:
        raise PageStagingError(
            f"Translation {source!r} does not use the .{locale}.md suffix"
        )
    stem_path = source_path.with_name(match.group("stem"))
    if stem_path.name == "index":
        relative = stem_path.parent / "index.html"
    else:
        relative = stem_path / "index.html"
    return (Path("docs") / locale / relative).as_posix()


def translated_output_paths(publication: Mapping[str, Any]) -> list[str]:
    """Return every real localized HTML output declared by publication.yml."""

    outputs: list[str] = []
    groups = publication.get("translation_groups", {})
    if not isinstance(groups, Mapping):
        raise PageStagingError("publication translation_groups must be a mapping")
    for group_name, group in groups.items():
        if not isinstance(group, Mapping):
            raise PageStagingError(
                f"Translation group {group_name!r} must be a mapping"
            )
        translations = group.get("translations", {})
        if not isinstance(translations, Mapping):
            raise PageStagingError(
                f"Translation group {group_name!r} translations must be a mapping"
            )
        for locale, source in translations.items():
            if not isinstance(locale, str) or not isinstance(source, str):
                raise PageStagingError(
                    f"Translation group {group_name!r} has an invalid translation"
                )
            outputs.append(translated_output_path(source, locale))
    if len(outputs) != len(set(outputs)):
        raise PageStagingError("Publication translation groups emit duplicate routes")
    return sorted(outputs)


def find_overlay_collisions(source: Path, destination: Path) -> list[str]:
    """Find website paths that would replace an existing staged docs path."""

    collisions: set[str] = set()
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            collisions.add(f"{relative.as_posix()} (symlink is not allowed)")
            continue
        if target.exists():
            collisions.add(relative.as_posix())
            continue
        for parent in relative.parents:
            if parent == Path("."):
                break
            if (destination / parent).is_file():
                collisions.add(relative.as_posix())
                break
    return sorted(collisions)


def public_route_for_path(path: str) -> str:
    """Return the public URL path for one staged artifact path."""

    normalized = path.strip("/")
    if normalized == "index.html":
        return "/"
    if normalized.endswith("/index.html"):
        return f"/{normalized[: -len('index.html')]}"
    return f"/{normalized}"


def artifact_path_for_url(output_dir: Path, url_path: str) -> Path | None:
    """Resolve a public URL path to an existing staged file."""

    decoded = unquote(url_path).lstrip("/")
    if "\x00" in decoded:
        return None
    relative = Path(decoded)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidates: list[Path]
    if not decoded or url_path.endswith("/"):
        candidates = [relative / "index.html"]
    else:
        candidates = [relative, relative / "index.html"]
        if not relative.suffix:
            candidates.append(Path(f"{relative}.html"))
    for candidate in candidates:
        path = output_dir / candidate
        if path.is_file() and not path.is_symlink():
            return path
    return None


def _html_document(path: Path) -> _HTMLReferenceParser:
    parser = _HTMLReferenceParser()
    try:
        parser.feed(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise PageStagingError(f"Could not parse staged HTML {path}: {exc}") from exc
    return parser


def _normalized_public_metadata_url(
    reference: str,
    *,
    source_route: str,
    label: str,
) -> str:
    """Resolve one metadata URL and require the canonical production origin."""

    absolute = urlsplit(urljoin(f"{PUBLIC_ORIGIN}{source_route}", reference.strip()))
    if (
        absolute.scheme != "https"
        or absolute.hostname != "openmed.life"
        or absolute.port is not None
        or absolute.username is not None
        or absolute.password is not None
        or absolute.query
        or absolute.fragment
    ):
        raise PageStagingError(
            f"{label} must be an HTTPS openmed.life URL without query or fragment: "
            f"{reference!r}"
        )
    return f"{PUBLIC_ORIGIN}{absolute.path or '/'}"


def _required_meta(
    document: _HTMLReferenceParser,
    key: str,
    *,
    source: str,
) -> str:
    values = document.meta_values(key)
    if len(values) != 1 or not values[0]:
        raise PageStagingError(
            f"{source}: metadata {key!r} must appear exactly once with content"
        )
    return values[0]


def _expected_hreflang(
    publication: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Return exact artifact-path hreflang maps for published Markdown."""

    classification = publication["classification"]
    expected = {
        markdown_output_path(source): {
            "en": f"{PUBLIC_ORIGIN}{public_route_for_path(markdown_output_path(source))}"
        }
        for source in (
            list(classification["navigated"]) + list(classification["link_only"])
        )
    }
    for group in publication["translation_groups"].values():
        source = str(group["source"])
        group_paths = {"en": markdown_output_path(source)}
        group_paths.update(
            {
                str(locale): translated_output_path(str(translated), str(locale))
                for locale, translated in group["translations"].items()
            }
        )
        alternates = {
            locale: f"{PUBLIC_ORIGIN}{public_route_for_path(path)}"
            for locale, path in group_paths.items()
        }
        for path in group_paths.values():
            expected[path] = dict(alternates)
    return expected


def _sitemap_contract(
    path: Path,
) -> tuple[set[str], dict[str, dict[str, str]]]:
    """Return loc URLs plus hreflang links grouped by sitemap URL."""

    try:
        tree = ElementTree.parse(path)
    except (OSError, ElementTree.ParseError) as exc:
        raise PageStagingError(f"Could not parse sitemap {path}: {exc}") from exc
    urls: set[str] = set()
    alternates: dict[str, dict[str, str]] = {}
    for element in tree.iter():
        if not element.tag.endswith("url"):
            continue
        loc = next(
            (
                child.text.strip()
                for child in element
                if child.tag.endswith("loc") and child.text and child.text.strip()
            ),
            "",
        )
        if not loc:
            raise PageStagingError(f"{path}: sitemap URL entry is missing loc")
        if loc in urls:
            raise PageStagingError(f"{path}: duplicate sitemap URL {loc}")
        urls.add(loc)
        language_map: dict[str, str] = {}
        for child in element:
            if not child.tag.endswith("link") or child.attrib.get("rel") != "alternate":
                continue
            language = child.attrib.get("hreflang", "").strip().casefold()
            href = child.attrib.get("href", "").strip()
            if not language or not href or language in language_map:
                raise PageStagingError(
                    f"{path}: invalid or duplicate sitemap hreflang for {loc}"
                )
            language_map[language] = href
        alternates[loc] = language_map
    return urls, alternates


def _assert_link_only_reachability(
    output_dir: Path,
    publication: Mapping[str, Any],
    documents: Mapping[Path, _HTMLReferenceParser],
) -> None:
    classification = publication["classification"]
    navigated = {
        output_dir / markdown_output_path(source)
        for source in classification["navigated"]
    }
    link_only = {
        output_dir / markdown_output_path(source)
        for source in classification["link_only"]
    }
    reachable = set(navigated)
    pending = list(navigated)
    while pending:
        source = pending.pop()
        document = documents.get(source)
        if document is None:
            continue
        source_route = public_route_for_path(source.relative_to(output_dir).as_posix())
        for reference in document.references:
            resolved = _internal_reference(reference, source_route=source_route)
            if resolved is None:
                continue
            target = artifact_path_for_url(output_dir, resolved[0])
            if target is not None and target in documents and target not in reachable:
                reachable.add(target)
                pending.append(target)
    if missing := sorted(
        path.relative_to(output_dir).as_posix() for path in link_only - reachable
    ):
        raise PageStagingError(
            "Publication link-only pages are not reachable from navigated pages: "
            + ", ".join(missing)
        )


def _validate_local_icons(
    output_dir: Path,
    document: _HTMLReferenceParser,
    *,
    source_route: str,
    source: str,
) -> None:
    """Require one or more unique, local favicon references that resolve."""

    if not document.icons:
        raise PageStagingError(f"{source}: at least one local favicon is required")
    normalized_icons = [
        _normalized_public_metadata_url(
            reference,
            source_route=source_route,
            label=f"{source} favicon",
        )
        for reference in document.icons
    ]
    if len(normalized_icons) != len(set(normalized_icons)):
        raise PageStagingError(f"{source}: duplicate favicon reference")
    for icon in normalized_icons:
        if artifact_path_for_url(output_dir, urlsplit(icon).path) is None:
            raise PageStagingError(
                f"{source}: favicon does not resolve in the artifact: {icon}"
            )


def _validated_social_image(
    output_dir: Path,
    document: _HTMLReferenceParser,
    key: str,
    *,
    source_route: str,
    source: str,
) -> str:
    """Return one normalized social image URL that resolves in the artifact."""

    image_url = _normalized_public_metadata_url(
        _required_meta(document, key, source=source),
        source_route=source_route,
        label=f"{source} {key}",
    )
    if artifact_path_for_url(output_dir, urlsplit(image_url).path) is None:
        raise PageStagingError(f"{source}: {key} does not resolve in the artifact")
    return image_url


def validate_publication_metadata(
    output_dir: Path,
    publication: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate indexability, canonical, social, locale, and fixture metadata."""

    documents = {
        path: _html_document(path)
        for path in sorted(output_dir.rglob("*.html"))
        if path.is_file() and not path.is_symlink()
    }
    fixtures = _publication_fixture_map(publication)
    expected_hreflang = _expected_hreflang(publication)
    metadata: dict[str, dict[str, Any]] = {}
    canonical_owners: dict[str, str] = {}
    description_owners: dict[str, str] = {}
    title_owners: dict[str, str] = {}

    for path, document in documents.items():
        relative = path.relative_to(output_dir).as_posix()
        route = public_route_for_path(relative)
        if relative == "docs/404.html":
            if document.document_language != "en":
                raise PageStagingError(
                    "docs/404.html: canonical fallback page must declare html lang=en"
                )
            continue

        if not document.document_language:
            raise PageStagingError(f"{relative}: html lang is required")
        if relative in fixtures:
            fixture = fixtures[relative]
            expected_indexing = {
                directive.strip().casefold()
                for directive in str(fixture["indexing"]).split(",")
            }
            actual_indexing = {
                directive.strip().casefold()
                for value in document.meta_values("robots")
                for directive in value.split(",")
                if directive.strip()
            }
            if actual_indexing != expected_indexing:
                raise PageStagingError(
                    f"{relative}: robots metadata must be {fixture['indexing']}"
                )
            if (
                fixture.get("purpose") == "rtl_layout_and_accessibility"
                and document.direction != "rtl"
            ):
                raise PageStagingError(
                    f"{relative}: RTL fixture must declare html dir=rtl"
                )
            if len(document.canonicals) != 1:
                raise PageStagingError(
                    f"{relative}: fixture canonical must appear exactly once"
                )
            canonical = _normalized_public_metadata_url(
                document.canonicals[0],
                source_route=route,
                label=f"{relative} canonical",
            )
            expected_canonical = f"{PUBLIC_ORIGIN}{route}"
            if canonical != expected_canonical:
                raise PageStagingError(
                    f"{relative}: canonical {canonical!r} does not self-match "
                    f"{expected_canonical!r}"
                )
            if fixture.get("social_metadata") != "prohibited":
                raise PageStagingError(
                    f"{relative}: fixture social metadata policy must be prohibited"
                )
            social_keys = {
                key
                for key in document.meta
                if key.startswith("og:") or key.startswith("twitter:")
            }
            if social_keys:
                raise PageStagingError(
                    f"{relative}: fixture social metadata is prohibited: "
                    + ", ".join(sorted(social_keys))
                )
            _validate_local_icons(
                output_dir,
                document,
                source_route=route,
                source=relative,
            )
            metadata[route] = {
                "canonical": canonical,
                "hreflang": {},
                "indexing": str(fixture["indexing"]),
                "lang": document.document_language,
                "social_metadata": "prohibited",
                "title": document.title,
            }
            continue

        robots = {
            directive.strip().casefold()
            for value in document.meta_values("robots")
            for directive in value.split(",")
            if directive.strip()
        }
        if "noindex" in robots:
            raise PageStagingError(f"{relative}: indexable page declares noindex")
        if not document.title:
            raise PageStagingError(f"{relative}: title is required")
        if prior := title_owners.get(document.title):
            raise PageStagingError(
                f"{relative}: duplicate title {document.title!r} also belongs to {prior}"
            )
        title_owners[document.title] = relative

        description = _required_meta(document, "description", source=relative)
        normalized_description = " ".join(description.casefold().split())
        if len(description) < 40:
            raise PageStagingError(
                f"{relative}: description is too short or boilerplate-like"
            )
        if len(set(re.findall(r"[\w'-]+", normalized_description))) < 4:
            raise PageStagingError(
                f"{relative}: description lacks useful page-specific detail"
            )
        if prior := description_owners.get(normalized_description):
            raise PageStagingError(
                f"{relative}: duplicate description also belongs to {prior}"
            )
        description_owners[normalized_description] = relative
        if len(document.canonicals) != 1:
            raise PageStagingError(f"{relative}: canonical must appear exactly once")
        canonical = _normalized_public_metadata_url(
            document.canonicals[0],
            source_route=route,
            label=f"{relative} canonical",
        )
        expected_canonical = f"{PUBLIC_ORIGIN}{route}"
        if canonical != expected_canonical:
            raise PageStagingError(
                f"{relative}: canonical {canonical!r} does not self-match "
                f"{expected_canonical!r}"
            )
        if prior := canonical_owners.get(canonical):
            raise PageStagingError(
                f"{relative}: canonical {canonical!r} also belongs to {prior}"
            )
        canonical_owners[canonical] = relative

        expected_alternates = expected_hreflang.get(relative, {})
        actual_alternates: dict[str, str] = {}
        for language, reference in document.alternates:
            if language in actual_alternates:
                raise PageStagingError(f"{relative}: duplicate hreflang {language!r}")
            actual_alternates[language] = _normalized_public_metadata_url(
                reference,
                source_route=route,
                label=f"{relative} hreflang {language}",
            )
        if actual_alternates != expected_alternates:
            raise PageStagingError(
                f"{relative}: hreflang differs from real translations "
                f"(expected {expected_alternates}; found {actual_alternates})"
            )

        _validate_local_icons(
            output_dir,
            document,
            source_route=route,
            source=relative,
        )
        og_url = _required_meta(document, "og:url", source=relative)
        normalized_og_url = _normalized_public_metadata_url(
            og_url,
            source_route=route,
            label=f"{relative} og:url",
        )
        if normalized_og_url != canonical:
            raise PageStagingError(f"{relative}: og:url must match canonical")
        og_type = _required_meta(document, "og:type", source=relative)
        if og_type not in {"article", "website"}:
            raise PageStagingError(f"{relative}: unsupported og:type {og_type!r}")
        og_title = _required_meta(document, "og:title", source=relative)
        twitter_title = _required_meta(document, "twitter:title", source=relative)
        if og_title != document.title or twitter_title != document.title:
            raise PageStagingError(
                f"{relative}: document, Open Graph, and X titles must agree exactly"
            )
        og_description = _required_meta(document, "og:description", source=relative)
        twitter_description = _required_meta(
            document, "twitter:description", source=relative
        )
        if og_description != description or twitter_description != description:
            raise PageStagingError(
                f"{relative}: description, Open Graph, and X copy must agree exactly"
            )
        og_image = _validated_social_image(
            output_dir,
            document,
            "og:image",
            source_route=route,
            source=relative,
        )
        twitter_image = _validated_social_image(
            output_dir,
            document,
            "twitter:image",
            source_route=route,
            source=relative,
        )
        if og_image != twitter_image:
            raise PageStagingError(
                f"{relative}: Open Graph and X images must agree exactly"
            )
        og_image_alt = _required_meta(document, "og:image:alt", source=relative)
        twitter_image_alt = _required_meta(
            document, "twitter:image:alt", source=relative
        )
        if og_image_alt != twitter_image_alt or len(og_image_alt.split()) < 3:
            raise PageStagingError(
                f"{relative}: Open Graph and X image alt text must agree and be useful"
            )
        twitter_card = _required_meta(document, "twitter:card", source=relative)
        if twitter_card != "summary_large_image":
            raise PageStagingError(
                f"{relative}: twitter:card must be summary_large_image"
            )
        metadata[route] = {
            "canonical": canonical,
            "description": description,
            "hreflang": actual_alternates,
            "indexing": "index",
            "lang": document.document_language,
            "open_graph": True,
            "title": document.title,
            "twitter_card": True,
        }

    expected_docs_urls = {
        f"{PUBLIC_ORIGIN}{public_route_for_path(path)}" for path in expected_hreflang
    }
    docs_sitemap = output_dir / "docs" / "sitemap.xml"
    sitemap_urls, sitemap_alternates = _sitemap_contract(docs_sitemap)
    if sitemap_urls != expected_docs_urls:
        missing = sorted(expected_docs_urls - sitemap_urls)
        unexpected = sorted(sitemap_urls - expected_docs_urls)
        raise PageStagingError(
            "Documentation sitemap differs from publication classification "
            f"(missing: {missing}; unexpected: {unexpected})"
        )
    expected_sitemap_alternates = {
        f"{PUBLIC_ORIGIN}{public_route_for_path(path)}": languages
        for path, languages in expected_hreflang.items()
    }
    if sitemap_alternates != expected_sitemap_alternates:
        raise PageStagingError(
            "Documentation sitemap hreflang differs from translation groups"
        )

    expected_locale_urls: dict[str, set[str]] = {}
    for group in publication["translation_groups"].values():
        for locale, translated in group["translations"].items():
            output_path = translated_output_path(str(translated), str(locale))
            expected_locale_urls.setdefault(str(locale), set()).add(
                f"{PUBLIC_ORIGIN}{public_route_for_path(output_path)}"
            )
    for locale, expected_urls in sorted(expected_locale_urls.items()):
        locale_sitemap = output_dir / "docs" / locale / "sitemap.xml"
        locale_urls, locale_alternates = _sitemap_contract(locale_sitemap)
        if locale_urls != expected_urls:
            missing = sorted(expected_urls - locale_urls)
            unexpected = sorted(locale_urls - expected_urls)
            raise PageStagingError(
                f"Documentation {locale} sitemap differs from real translations "
                f"(missing: {missing}; unexpected: {unexpected})"
            )
        expected_alternates = {
            url: expected_sitemap_alternates[url] for url in expected_urls
        }
        if locale_alternates != expected_alternates:
            raise PageStagingError(
                f"Documentation {locale} sitemap hreflang differs from "
                "real translation groups"
            )

    all_sitemap_urls: set[str] = set()
    for sitemap in sorted(output_dir.rglob("sitemap*.xml")):
        urls, _ = _sitemap_contract(sitemap)
        all_sitemap_urls.update(urls)
    for relative in fixtures:
        fixture_url = f"{PUBLIC_ORIGIN}{public_route_for_path(relative)}"
        if fixture_url in all_sitemap_urls:
            raise PageStagingError(
                f"Non-indexed fixture appears in a sitemap: {fixture_url}"
            )

    _assert_link_only_reachability(output_dir, publication, documents)
    return metadata


def _internal_reference(
    reference: str,
    *,
    source_route: str,
) -> tuple[str, str] | None:
    candidate = reference.strip()
    if not candidate or candidate.startswith(
        ("#!", "data:", "javascript:", "mailto:", "tel:")
    ):
        return None
    absolute = urlsplit(urljoin(f"https://openmed.life{source_route}", candidate))
    if absolute.scheme not in {"http", "https"}:
        return None
    if absolute.hostname not in _INTERNAL_HOSTS:
        return None
    return absolute.path or "/", unquote(absolute.fragment)


def validate_internal_references(output_dir: Path) -> None:
    """Crawl staged HTML/CSS plus advertised sitemap and robots references."""

    html_documents = {
        path: _html_document(path)
        for path in sorted(output_dir.rglob("*.html"))
        if path.is_file() and not path.is_symlink()
    }
    errors: list[str] = []
    for source, document in html_documents.items():
        source_relative = source.relative_to(output_dir).as_posix()
        source_route = public_route_for_path(source_relative)
        for kind, references in (
            ("script", document.scripts),
            ("stylesheet", document.stylesheets),
        ):
            seen: set[str] = set()
            for reference in references:
                if reference in seen:
                    errors.append(f"{source_relative}: duplicate {kind} {reference}")
                seen.add(reference)
        for reference in document.references:
            resolved = _internal_reference(reference, source_route=source_route)
            if resolved is None:
                continue
            target_url, fragment = resolved
            target = artifact_path_for_url(output_dir, target_url)
            if target is None:
                errors.append(f"{source_relative}: missing {reference}")
                continue
            if fragment and target.suffix.casefold() == ".html":
                target_document = html_documents.get(target)
                if target_document is None:
                    target_document = _html_document(target)
                if fragment not in target_document.ids:
                    errors.append(f"{source_relative}: missing fragment {reference}")

    for source in sorted(output_dir.rglob("*.css")):
        if not source.is_file() or source.is_symlink():
            continue
        source_relative = source.relative_to(output_dir).as_posix()
        source_route = f"/{source_relative}"
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise PageStagingError(
                f"Could not parse staged CSS {source}: {exc}"
            ) from exc
        for match in _CSS_REFERENCE_RE.finditer(text):
            reference = match.group("url")
            resolved = _internal_reference(reference, source_route=source_route)
            if resolved is None:
                continue
            target_url, _ = resolved
            if artifact_path_for_url(output_dir, target_url) is None:
                errors.append(f"{source_relative}: missing {reference}")

    for sitemap in sorted(output_dir.rglob("sitemap*.xml")):
        try:
            tree = ElementTree.parse(sitemap)
        except (OSError, ElementTree.ParseError) as exc:
            raise PageStagingError(f"Could not parse sitemap {sitemap}: {exc}") from exc
        for element in tree.iter():
            if not element.tag.endswith("loc") or not element.text:
                continue
            resolved = _internal_reference(element.text, source_route="/")
            if (
                resolved is not None
                and artifact_path_for_url(output_dir, resolved[0]) is None
            ):
                errors.append(
                    f"{sitemap.relative_to(output_dir).as_posix()}: "
                    f"missing {element.text.strip()}"
                )

    for robots in sorted(output_dir.rglob("robots.txt")):
        for line in robots.read_text(encoding="utf-8").splitlines():
            if not line.casefold().startswith("sitemap:"):
                continue
            reference = line.split(":", maxsplit=1)[1].strip()
            resolved = _internal_reference(reference, source_route="/")
            if (
                resolved is not None
                and artifact_path_for_url(output_dir, resolved[0]) is None
            ):
                errors.append(
                    f"{robots.relative_to(output_dir).as_posix()}: missing {reference}"
                )

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:100])
        remainder = len(errors) - min(len(errors), 100)
        suffix = f"\n  - ... and {remainder} more" if remainder else ""
        raise PageStagingError(
            f"Staged artifact contains broken internal references:\n{preview}{suffix}"
        )


def owner_for_path(path: str) -> str:
    """Return the build owner for one staged artifact path."""

    if path == MANIFEST_NAME:
        return "staging"
    if path.startswith("docs/eval/benchmark-leaderboard/"):
        return "leaderboard"
    if path.startswith("docs/demo/web/"):
        return "browser-demo"
    if path.startswith("docs/"):
        return "mkdocs"
    return "marketing"


def build_manifest(
    output_dir: Path,
    *,
    release_tag: str,
    publication: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the exact content-addressed route and asset manifest."""

    files: list[dict[str, Any]] = []
    route_index: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise PageStagingError(
                "Staged Pages artifacts cannot contain symlinks: "
                + path.relative_to(output_dir).as_posix()
            )
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(output_dir).as_posix()
        route = public_route_for_path(relative)
        if route in route_index:
            raise PageStagingError(
                f"Staged route collision: {relative} and {route_index[route]} "
                f"both publish {route}"
            )
        route_index[route] = relative
        files.append(
            {
                "bytes": path.stat().st_size,
                "owner": owner_for_path(relative),
                "path": relative,
                "route": route,
                "sha256": sha256_file(path),
            }
        )

    owners: dict[str, dict[str, int]] = {}
    for item in files:
        owner = str(item["owner"])
        summary = owners.setdefault(owner, {"bytes": 0, "files": 0})
        summary["bytes"] += int(item["bytes"])
        summary["files"] += 1

    translation_groups = publication.get("translation_groups", {})
    return {
        "expected_paths": sorted(
            set(normalized_expected_paths(publication))
            | set(normalized_expected_assets(publication))
        ),
        "files": files,
        "manifest": {
            "owner": "staging",
            "path": MANIFEST_NAME,
            "route": f"/{MANIFEST_NAME}",
        },
        "metadata": dict(metadata or {}),
        "owners": owners,
        "release_tag": release_tag,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "translation_groups": translation_groups,
    }


def write_manifest(
    output_dir: Path,
    *,
    release_tag: str,
    publication: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Write the deterministic Pages manifest and return its path."""

    manifest_path = output_dir / MANIFEST_NAME
    payload = build_manifest(
        output_dir,
        release_tag=release_tag,
        publication=publication,
        metadata=metadata,
    )
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def assert_paths_exist(output_dir: Path, paths: Iterable[str]) -> None:
    """Require every relative path in *paths* to exist as a non-empty file."""

    missing: list[str] = []
    empty: list[str] = []
    for relative in sorted(set(paths)):
        path = output_dir / relative
        if path.is_symlink():
            missing.append(f"{relative} (symlink is not allowed)")
        elif not path.is_file():
            missing.append(relative)
        elif path.stat().st_size == 0:
            empty.append(relative)
    messages: list[str] = []
    if missing:
        messages.append(f"missing: {', '.join(missing)}")
    if empty:
        messages.append(f"empty: {', '.join(empty)}")
    if messages:
        raise PageStagingError(
            "Staged Pages artifact is incomplete (" + "; ".join(messages) + ")"
        )


def validate_llm_feeds(
    output_dir: Path,
    *,
    config_path: Path = MKDOCS_CONFIG,
) -> None:
    """Require the configured LLM index and full feed to contain every page."""

    try:
        mkdocs = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PageStagingError(f"Could not load the LLM feed contract: {exc}") from exc
    if not isinstance(mkdocs, Mapping):
        raise PageStagingError("The LLM feed MkDocs config must be a mapping")

    plugins = mkdocs.get("plugins")
    if not isinstance(plugins, list):
        raise PageStagingError("The LLM feed MkDocs config requires a plugin list")
    llmstxt: Mapping[str, Any] | None = None
    for plugin in plugins:
        if isinstance(plugin, Mapping) and "llmstxt" in plugin:
            candidate = plugin["llmstxt"]
            if not isinstance(candidate, Mapping):
                raise PageStagingError("The llmstxt plugin config must be a mapping")
            llmstxt = candidate
            break
    if llmstxt is None:
        raise PageStagingError("The MkDocs config does not declare the llmstxt plugin")

    site_url = mkdocs.get("site_url")
    if not isinstance(site_url, str) or not site_url.strip():
        raise PageStagingError("The LLM feed contract requires a site_url")
    site_url = site_url.rstrip("/") + "/"

    full_output = llmstxt.get("full_output")
    if not isinstance(full_output, str) or not full_output.strip():
        raise PageStagingError("The llmstxt plugin requires a full_output path")
    full_output_path = Path(full_output.strip())
    if (
        full_output_path.is_absolute()
        or ".." in full_output_path.parts
        or full_output_path.suffix != ".txt"
    ):
        raise PageStagingError(f"Unsafe llmstxt full_output path: {full_output!r}")

    sections = llmstxt.get("sections")
    if not isinstance(sections, Mapping) or not sections:
        raise PageStagingError("The llmstxt plugin requires curated sections")

    expected_headings: list[str] = []
    expected_links: list[tuple[str, str]] = []
    generated_pages: list[tuple[str, Path]] = []
    for raw_heading, raw_entries in sections.items():
        if not isinstance(raw_heading, str) or not raw_heading.strip():
            raise PageStagingError("LLM feed section names must be non-empty text")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise PageStagingError(
                f"LLM feed section {raw_heading!r} must contain pages"
            )
        heading = raw_heading.strip()
        expected_headings.append(heading)
        for raw_entry in raw_entries:
            description = ""
            if isinstance(raw_entry, str):
                source = raw_entry
            elif isinstance(raw_entry, Mapping) and len(raw_entry) == 1:
                source, raw_description = next(iter(raw_entry.items()))
                description = str(raw_description).strip()
            else:
                raise PageStagingError(
                    f"Invalid llmstxt entry in section {heading!r}: {raw_entry!r}"
                )
            source = _safe_relative_source(
                source,
                label=f"llmstxt section {heading!r} source",
                suffix=".md",
            )
            if any(character in source for character in "*?["):
                raise PageStagingError(
                    f"LLM feed sources must be explicit, not globs: {source}"
                )
            generated_path = Path(markdown_output_path(source)).with_suffix(".md")
            feed_relative = generated_path.relative_to("docs").as_posix()
            expected_links.append((urljoin(site_url, feed_relative), description))
            generated_pages.append((source, output_dir / generated_path))

    index_path = output_dir / "docs" / "llms.txt"
    full_path = output_dir / "docs" / full_output_path
    try:
        index_text = index_path.read_text(encoding="utf-8")
        full_text = full_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PageStagingError(f"Could not read staged LLM feeds: {exc}") from exc

    actual_headings = re.findall(r"(?m)^## ([^\n]+?)\s*$", index_text)
    if actual_headings != expected_headings:
        raise PageStagingError(
            "Staged llms.txt section coverage differs from mkdocs.yml "
            f"(expected: {expected_headings}; actual: {actual_headings})"
        )
    actual_links = [
        (url, (description or "").strip())
        for url, description in re.findall(
            r"(?m)^- \[[^\]\n]+\]\(([^)\n]+)\)(?:: ([^\n]+))?$",
            index_text,
        )
    ]
    if actual_links != expected_links:
        raise PageStagingError(
            "Staged llms.txt page links or descriptions differ from mkdocs.yml"
        )

    for heading in expected_headings:
        if full_text.count(f"# {heading}\n") != 1:
            raise PageStagingError(
                f"Staged {full_output_path} is missing section {heading!r}"
            )
    for source, generated_path in generated_pages:
        try:
            content = generated_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise PageStagingError(
                f"Could not read generated LLM page for {source}: {exc}"
            ) from exc
        if len(content) < 80:
            raise PageStagingError(
                f"Generated LLM page for {source} has no substantive content"
            )
        if content not in full_text:
            raise PageStagingError(
                f"Staged {full_output_path} does not embed generated page {source}"
            )


def ensure_safe_output_dir(output_dir: Path) -> Path:
    """Resolve an output directory and reject broad or out-of-repo targets."""

    if output_dir.is_symlink():
        raise PageStagingError("Refusing to use a symlink as the Pages output")
    resolved = output_dir.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise PageStagingError(
            f"Output directory must be inside the repository: {resolved}"
        ) from exc
    if relative == Path(".") or not relative.parts:
        raise PageStagingError("Refusing to use the repository root as output")
    if resolved != DEFAULT_OUTPUT_DIR and not (
        resolved.parent == ROOT and resolved.name.startswith("site-")
    ):
        raise PageStagingError(
            "Pages output must be site/ or a root-level site-* preview directory"
        )
    return resolved


def run_checked(command: Sequence[str], *, label: str) -> None:
    """Run a build command at the repository root with a concise error."""

    print(f"[pages] {label}")
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PageStagingError(f"{label} failed: {exc}") from exc


def render_leaderboard(release_tag: str) -> None:
    """Regenerate the public leaderboard and require committed output parity."""

    before = snapshot_tree(DEFAULT_LEADERBOARD_DIR)
    run_checked(
        (
            sys.executable,
            "-m",
            "openmed.eval.leaderboard",
            "--reports-dir",
            "docs/benchmarks",
            "--output-dir",
            "docs/eval/benchmark-leaderboard",
            "--manifest",
            "models.jsonl",
            "--release-tag",
            release_tag,
        ),
        label=f"rendering leaderboard for {release_tag}",
    )
    after = snapshot_tree(DEFAULT_LEADERBOARD_DIR)
    changed = changed_snapshot_paths(before, after)
    if changed:
        raise PageStagingError(
            "Committed leaderboard output was stale and has been regenerated: "
            + ", ".join(changed)
            + ". Review the generated files, then stage again."
        )


def build_docs(output_dir: Path) -> None:
    """Build strict MkDocs output below the staged ``docs`` prefix."""

    run_checked(
        (
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--clean",
            "--site-dir",
            str(output_dir / "docs"),
        ),
        label="building MkDocs in strict mode",
    )


def copy_website(source: Path, output_dir: Path) -> None:
    """Copy the static marketing site without replacing generated docs."""

    if not source.is_dir():
        raise PageStagingError(f"Marketing site does not exist: {source}")
    collisions = find_overlay_collisions(source, output_dir)
    if collisions:
        raise PageStagingError(
            "Marketing overlay would replace staged output: " + ", ".join(collisions)
        )
    print("[pages] copying marketing site")
    shutil.copytree(source, output_dir, dirs_exist_ok=True, copy_function=shutil.copy2)


def copy_locale_sitemaps(output_dir: Path, publication: Mapping[str, Any]) -> None:
    """Write locale-scoped sitemaps for Material's locale-relative links."""

    source = output_dir / "docs" / "sitemap.xml"
    if not source.is_file():
        raise PageStagingError("MkDocs did not produce docs/sitemap.xml")
    try:
        source_tree = ElementTree.parse(source)
    except (OSError, ElementTree.ParseError) as exc:
        raise PageStagingError(
            f"Could not parse documentation sitemap {source}: {exc}"
        ) from exc
    source_root = source_tree.getroot()
    entries: dict[str, Any] = {}
    for element in source_root:
        if not element.tag.endswith("url"):
            continue
        loc = next(
            (
                child.text.strip()
                for child in element
                if child.tag.endswith("loc") and child.text and child.text.strip()
            ),
            "",
        )
        if loc:
            entries[loc] = element

    locale_urls: dict[str, set[str]] = {}
    for group in publication["translation_groups"].values():
        for locale, translated in group["translations"].items():
            output_path = translated_output_path(str(translated), str(locale))
            locale_urls.setdefault(str(locale), set()).add(
                f"{PUBLIC_ORIGIN}{public_route_for_path(output_path)}"
            )

    ElementTree.register_namespace("", "http://www.sitemaps.org/schemas/sitemap/0.9")
    ElementTree.register_namespace("xhtml", "http://www.w3.org/1999/xhtml")
    for locale, expected_urls in sorted(locale_urls.items()):
        missing = sorted(expected_urls - entries.keys())
        if missing:
            raise PageStagingError(
                f"Global documentation sitemap is missing {locale} translations: "
                + ", ".join(missing)
            )
        destination = output_dir / "docs" / locale / "sitemap.xml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        locale_root = ElementTree.Element(source_root.tag, source_root.attrib)
        for url in sorted(expected_urls):
            locale_root.append(copy.deepcopy(entries[url]))
        ElementTree.ElementTree(locale_root).write(
            destination,
            encoding="utf-8",
            xml_declaration=True,
        )


def prune_source_maps(output_dir: Path) -> list[str]:
    """Remove non-runtime source maps from the production Pages artifact."""

    removed: list[str] = []
    for path in sorted(output_dir.rglob("*.map")):
        if path.is_symlink():
            raise PageStagingError(
                "Staged source maps cannot be symlinks: "
                + path.relative_to(output_dir).as_posix()
            )
        if not path.is_file():
            continue
        removed.append(path.relative_to(output_dir).as_posix())
        path.unlink()
    if removed:
        print(f"[pages] removed {len(removed)} non-runtime source maps")
    return removed


def stage_pages(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    release_tag: str | None = None,
    publication_path: Path = DEFAULT_PUBLICATION,
) -> Path:
    """Build, validate, and manifest the complete Pages artifact."""

    output_dir = ensure_safe_output_dir(output_dir)
    publication = load_publication(publication_path)
    resolved_tag = resolve_release_tag(release_tag)

    render_leaderboard(resolved_tag)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    build_docs(output_dir)
    copy_website(DEFAULT_WEBSITE_DIR, output_dir)
    copy_locale_sitemaps(output_dir, publication)
    prune_source_maps(output_dir)

    required = {
        *normalized_expected_paths(publication),
        *normalized_expected_assets(publication),
        *translated_output_paths(publication),
    }
    assert_paths_exist(output_dir, required)
    print("[pages] validating LLM feeds")
    validate_llm_feeds(output_dir)
    print("[pages] crawling internal routes and assets")
    validate_internal_references(output_dir)
    print("[pages] validating publication metadata and locale policy")
    metadata = validate_publication_metadata(output_dir, publication)
    manifest_path = write_manifest(
        output_dir,
        release_tag=resolved_tag,
        publication=publication,
        metadata=metadata,
    )
    print(f"[pages] staged {output_dir}")
    print(f"[pages] manifest {manifest_path}")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory inside the repository (default: site)",
    )
    parser.add_argument(
        "--release-tag",
        help="Semantic release tag for generated pages (default: package version)",
    )
    parser.add_argument(
        "--publication",
        type=Path,
        default=DEFAULT_PUBLICATION,
        help="Documentation publication contract",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Stage Pages from command-line arguments."""

    args = build_arg_parser().parse_args(argv)
    try:
        stage_pages(
            output_dir=args.output_dir,
            release_tag=args.release_tag,
            publication_path=args.publication,
        )
    except PageStagingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
