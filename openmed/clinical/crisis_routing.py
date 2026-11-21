"""Offline display-only routing for explicit, caller-supplied crisis signals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from openmed.core.intended_use import IntendedUseError, load_intended_use_registry

GENERIC_NOTICE = (
    "If someone is in immediate danger, contact local emergency services now."
)
_LOCALE = re.compile(r"([a-z]{2,3})(?:-([a-z]{4}))?(?:-([a-z]{2}|[0-9]{3}))?", re.I)


class CrisisRoutingError(ValueError):
    """Invalid routing configuration or a refused suppression, by reason code."""


@dataclass(frozen=True, slots=True)
class CrisisResource:
    """Display metadata and source provenance for a locally reviewed resource."""

    id: str
    label: str
    url: str
    source_url: str
    verified_on: str


@dataclass(frozen=True, slots=True)
class CrisisNotice:
    """Required display data. The kit performs no contact or navigation."""

    resources: tuple[CrisisResource, ...]
    matched_locale: str | None
    text: str = GENERIC_NOTICE
    mandatory: bool = True


def _locale_keys(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not (
        match := _LOCALE.fullmatch(value.replace("_", "-"))
    ):
        return ()
    language, script, region = match.groups()
    language = language.lower()
    parts = [language]
    if script:
        parts.append(script.title())
    if region:
        parts.append(region.upper())
    keys = ["-".join(parts)]
    if script and region:
        keys.extend((language + "-" + region.upper(), language + "-" + script.title()))
    return tuple(dict.fromkeys([*keys, language]))


def _safe_link(value: object) -> bool:
    if (
        not isinstance(value, str)
        or len(value) > 2048
        or not value.isascii()
        or any(ord(c) <= 32 or ord(c) == 127 for c in value)
        or "\\" in value
    ):
        return False
    parts = urlsplit(value)
    return (
        parts.scheme == "https"
        and bool(parts.hostname)
        and (parts.port is None or parts.port > 0)
        and parts.username is None
        and parts.password is None
    )


def _resource(row: dict) -> CrisisResource:
    if not isinstance(row, dict) or set(row) != {
        "id",
        "label",
        "url",
        "source_url",
        "verified_on",
    }:
        raise CrisisRoutingError("invalid_crisis_resource")
    item = CrisisResource(**row)
    if not (
        isinstance(item.id, str)
        and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", item.id)
        and isinstance(item.label, str)
        and item.label.strip()
        and item.label.isprintable()
        and len(item.label) <= 160
        and _safe_link(item.url)
        and _safe_link(item.source_url)
        and date.fromisoformat(item.verified_on).isoformat() == item.verified_on
    ):
        raise CrisisRoutingError("invalid_crisis_resource")
    return item


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CrisisRoutingError("invalid_crisis_table")
        result[key] = value
    return result


def _table(source) -> dict[str, tuple[CrisisResource, ...]]:
    try:
        data = json.loads(source.read_bytes(), object_pairs_hook=_unique)
        if (
            not isinstance(data, dict)
            or set(data) != {"schema_version", "locales"}
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
        ):
            raise CrisisRoutingError("invalid_crisis_table")
        entries = {}
        for locale, rows in data["locales"].items():
            names = _locale_keys(locale)
            if (
                not names
                or names[0] in entries
                or not isinstance(rows, list)
                or not rows
            ):
                raise CrisisRoutingError("invalid_crisis_table")
            resources = tuple(_resource(row) for row in rows)
            if len({item.id for item in resources}) != len(resources):
                raise CrisisRoutingError("invalid_crisis_table")
            entries[names[0]] = resources
        return entries
    except CrisisRoutingError:
        raise
    except (AttributeError, KeyError, ValueError, TypeError, OSError, RecursionError):
        raise CrisisRoutingError("invalid_crisis_table") from None


class CrisisRouting:
    """Resolve local resources without guessing a country from language alone."""

    def __init__(self, *, override_file: Path | None = None) -> None:
        bundled = files("openmed.clinical").joinpath("data", "crisis_resources.json")
        self._entries = _table(bundled)
        if override_file is not None:
            self._entries.update(_table(Path(override_file)))

    def route(
        self,
        *,
        capability: str,
        locale: str,
        signal: bool,
        suppress_notice: bool = False,
    ) -> CrisisNotice | None:
        """Return a mandatory notice for a signal or refuse an unsafe suppression."""
        if type(signal) is not bool or type(suppress_notice) is not bool:
            raise CrisisRoutingError("invalid_crisis_signal")
        if not signal:
            return None
        if suppress_notice:
            try:
                declaration = load_intended_use_registry().resolve(capability)
            except (IntendedUseError, TypeError):
                raise CrisisRoutingError("crisis_notice_required") from None
            if "crisis_routing_required" in declaration.regulatory_tier.obligations:
                raise CrisisRoutingError("crisis_notice_required")
            return None
        for key in _locale_keys(locale):
            if key in self._entries:
                return CrisisNotice(self._entries[key], key)
        return CrisisNotice((), None)
