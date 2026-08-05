"""Validated local crosswalk resources for multilingual concept grounding.

Crosswalks are ordinary JSON files loaded from local storage.  The bundled
starter resources are deliberately small, permissively licensed mapping
tables; callers can supply larger resources using the same schema.  This
module performs no downloads and rejects resources that do not explicitly
declare themselves redistributable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .vocab import normalize_alias, normalize_language

__all__ = [
    "CrosswalkEntry",
    "CrosswalkFormatError",
    "CrosswalkLicenseError",
    "CrosswalkResource",
    "DEFAULT_CROSSWALK_RESOURCES",
    "load_crosswalk",
    "load_default_crosswalks",
]

CROSSWALK_SCHEMA_VERSION = 1
DEFAULT_CROSSWALK_RESOURCES: tuple[str, ...] = (
    "icd10cn_icd10.json",
    "chpo_hpo.json",
    "indic_hpo_aliases.json",
)
_SUPPORTED_TARGET_SYSTEMS = frozenset({"HPO", "ICD10"})


class CrosswalkFormatError(ValueError):
    """Raised when a local crosswalk does not satisfy the public schema."""


class CrosswalkLicenseError(CrosswalkFormatError):
    """Raised when a crosswalk is not declared freely redistributable."""


@dataclass(frozen=True)
class CrosswalkEntry:
    """One source-locale concept mapped to an international concept."""

    source_system: str
    source_code: str
    locale: str
    aliases: tuple[str, ...]
    target_system: str
    target_code: str
    target_display: str

    def __post_init__(self) -> None:
        source_system = _required_text(self.source_system, "source_system")
        source_code = _required_text(self.source_code, "source_code")
        locale = _normalize_locale(self.locale)
        aliases = _unique_text(self.aliases)
        target_system = _required_text(self.target_system, "target_system").upper()
        target_code = _required_text(self.target_code, "target_code")
        target_display = _required_text(self.target_display, "target_display")
        if not aliases:
            raise CrosswalkFormatError("crosswalk aliases must not be empty")
        if target_system not in _SUPPORTED_TARGET_SYSTEMS:
            raise CrosswalkFormatError(
                "free crosswalk target_system must be ICD10 or HPO"
            )
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "source_code", source_code)
        object.__setattr__(self, "locale", locale)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "target_system", target_system)
        object.__setattr__(self, "target_code", target_code)
        object.__setattr__(self, "target_display", target_display)

    @property
    def language(self) -> str:
        """Return the normalized primary language used for alias routing."""

        return normalize_language(self.locale)

    @property
    def surfaces(self) -> tuple[str, ...]:
        """Return source code and aliases in deterministic matching order."""

        return _unique_text((self.source_code, *self.aliases))


@dataclass(frozen=True)
class CrosswalkResource:
    """A versioned, redistributable local mapping table."""

    name: str
    version: str
    license_id: str
    redistributable: bool
    entries: tuple[CrosswalkEntry, ...]
    content_hash: str

    def __post_init__(self) -> None:
        name = _required_text(self.name, "name")
        version = _required_text(self.version, "version")
        license_id = _required_text(self.license_id, "license")
        entries = tuple(self.entries)
        if not self.redistributable:
            raise CrosswalkLicenseError(
                f"crosswalk {name!r} is not declared redistributable"
            )
        if not entries:
            raise CrosswalkFormatError(f"crosswalk {name!r} has no entries")
        if not self.content_hash.startswith("sha256:"):
            raise CrosswalkFormatError("crosswalk content_hash must use SHA-256")
        identities: set[tuple[str, str, str, str, str]] = set()
        for entry in entries:
            identity = (
                entry.source_system.casefold(),
                entry.source_code.casefold(),
                entry.locale,
                entry.target_system,
                entry.target_code.casefold(),
            )
            if identity in identities:
                raise CrosswalkFormatError(
                    f"crosswalk {name!r} contains duplicate mapping {identity!r}"
                )
            identities.add(identity)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "license_id", license_id)
        object.__setattr__(self, "entries", entries)

    @property
    def resource_version(self) -> str:
        """Return a stable name/version/content identifier for provenance."""

        return f"{self.name}@{self.version}+{self.content_hash}"

    def entries_for_locale(self, locale: str) -> tuple[CrosswalkEntry, ...]:
        """Return entries routed to ``locale`` using its primary language."""

        language = normalize_language(locale)
        return tuple(entry for entry in self.entries if entry.language == language)

    def entries_for_source_code(
        self,
        source_code: str,
        *,
        source_system: str | None = None,
    ) -> tuple[CrosswalkEntry, ...]:
        """Return exact mappings for one source code."""

        normalized_code = normalize_alias(source_code)
        normalized_system = (
            normalize_alias(source_system) if source_system is not None else None
        )
        return tuple(
            entry
            for entry in self.entries
            if normalize_alias(entry.source_code) == normalized_code
            and (
                normalized_system is None
                or normalize_alias(entry.source_system) == normalized_system
            )
        )

    def entries_for_target_code(
        self,
        target_code: str,
        *,
        target_system: str | None = None,
    ) -> tuple[CrosswalkEntry, ...]:
        """Return exact reverse mappings for one international code."""

        normalized_code = normalize_alias(target_code)
        normalized_system = target_system.strip().upper() if target_system else None
        return tuple(
            entry
            for entry in self.entries
            if normalize_alias(entry.target_code) == normalized_code
            and (normalized_system is None or entry.target_system == normalized_system)
        )


def load_crosswalk(path: str | Path) -> CrosswalkResource:
    """Load a caller-supplied crosswalk JSON file without network access."""

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise CrosswalkFormatError(f"crosswalk file does not exist: {resolved}")
    if resolved.suffix.casefold() != ".json":
        raise CrosswalkFormatError("crosswalk resources must be JSON files")
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrosswalkFormatError(f"invalid crosswalk JSON: {resolved}") from exc
    return _resource_from_payload(payload, raw=raw, source=str(resolved))


def load_default_crosswalks() -> tuple[CrosswalkResource, ...]:
    """Load the bundled CC0 starter crosswalks from package resources."""

    data_root = resources.files(__package__).joinpath("data")
    loaded: list[CrosswalkResource] = []
    for filename in DEFAULT_CROSSWALK_RESOURCES:
        resource = data_root.joinpath(filename)
        raw = resource.read_bytes()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CrosswalkFormatError(
                f"invalid bundled crosswalk JSON: {filename}"
            ) from exc
        loaded.append(_resource_from_payload(payload, raw=raw, source=filename))
    return tuple(loaded)


def _resource_from_payload(
    payload: Any,
    *,
    raw: bytes,
    source: str,
) -> CrosswalkResource:
    if not isinstance(payload, Mapping):
        raise CrosswalkFormatError(f"crosswalk {source!r} must contain an object")
    if payload.get("schema_version") != CROSSWALK_SCHEMA_VERSION:
        raise CrosswalkFormatError(
            f"crosswalk {source!r} requires schema_version {CROSSWALK_SCHEMA_VERSION}"
        )
    rows = payload.get("entries")
    if not isinstance(rows, list):
        raise CrosswalkFormatError(f"crosswalk {source!r} entries must be a list")
    entries = tuple(_entry_from_payload(row, source=source) for row in rows)
    return CrosswalkResource(
        name=str(payload.get("name", "")),
        version=str(payload.get("version", "")),
        license_id=str(payload.get("license", "")),
        redistributable=payload.get("redistributable") is True,
        entries=entries,
        content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
    )


def _entry_from_payload(row: Any, *, source: str) -> CrosswalkEntry:
    if not isinstance(row, Mapping):
        raise CrosswalkFormatError(f"crosswalk {source!r} entry must be an object")
    raw_aliases = row.get("aliases")
    if isinstance(raw_aliases, str) or not isinstance(raw_aliases, Iterable):
        raise CrosswalkFormatError(f"crosswalk {source!r} entry aliases must be a list")
    return CrosswalkEntry(
        source_system=str(row.get("source_system", "")),
        source_code=str(row.get("source_code", "")),
        locale=str(row.get("locale", "")),
        aliases=tuple(str(alias) for alias in raw_aliases),
        target_system=str(row.get("target_system", "")),
        target_code=str(row.get("target_code", "")),
        target_display=str(row.get("target_display", "")),
    )


def _normalize_locale(locale: str) -> str:
    value = _required_text(locale, "locale").replace("_", "-")
    parts = [part for part in value.split("-") if part]
    language = normalize_language(value)
    if len(parts) == 1:
        return language
    return "-".join((language, *(part.upper() for part in parts[1:])))


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CrosswalkFormatError(f"crosswalk {field_name} must be non-empty text")
    return value.strip()


def _unique_text(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        normalized = normalize_alias(text)
        if text and normalized not in seen:
            result.append(text)
            seen.add(normalized)
    return tuple(result)
