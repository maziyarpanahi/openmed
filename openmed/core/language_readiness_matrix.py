"""Deterministic clinical language-pack readiness matrices (#3100).

Maintainers and release reviewers need one content-free view of *how far* each
clinical language pack is from a shippable state: does routing detect the
language, are surrogates real or approximated, do national-ID validators
resolve, and is there fixture or release evidence to cite? This module turns
readiness records into a bounded JSON matrix and a deterministic Markdown
rendering.

The matrix is a pure projection: it never loads models, never downloads
fixtures, and never scores language quality. Every cell carries a closed
capability name, a closed state, and (when the state is ``complete``) the
``sha256:`` evidence digest that justifies it.

Two entry points are supported:

* :func:`build_language_readiness_matrix` accepts readiness entries directly,
  which is what release tooling and tests use;
* :func:`readiness_entries_from_packs` derives entries from the process-local
  :data:`~openmed.core.language_pack.LANGUAGE_PACK_REGISTRY` via the existing
  coherence report, so registered packs get a readiness row without a second
  registry.

Legacy spellings are folded before a record exists:
:meth:`LanguageReadinessEntry.from_tag` resolves caller-supplied aliases with
the shared :func:`~openmed.core.locale_tag.normalize_locale_tag` helper, so a
registry that still writes ``en_US`` or ``iw`` renders the same cells as its
canonical form. Underscores are never converted implicitly: without an explicit
alias such a tag fails as ``locale_tag_invalid``.

Rendering is byte-stable: capabilities and states are closed vocabularies,
locales are normalized BCP 47 tags sorted ordinally, and JSON is emitted with
sorted keys and compact separators (no trailing newline).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from .language_pack import LANGUAGE_PACK_REGISTRY, LanguagePackRegistry
from .language_pack_coherence import APPROXIMATED, FILLED, pack_coherence_report
from .locale_tag import LocaleTagError, normalize_locale_tag

MATRIX_SCHEMA_VERSION = "openmed.language_readiness_matrix.v1"

# Capability slots reported per locale, in stable display order.
READINESS_CAPABILITIES: tuple[str, ...] = (
    "detection",
    "surrogates",
    "validation",
    "fixtures",
    "release_evidence",
)

# Closed state vocabulary.
COMPLETE = "complete"
PARTIAL = "partial"
BLOCKED = "blocked"
MISSING_EVIDENCE = "missing_evidence"

READINESS_STATES: tuple[str, ...] = (
    COMPLETE,
    PARTIAL,
    BLOCKED,
    MISSING_EVIDENCE,
)

# Rendering placeholder for a locale/capability pair that has no record. It is
# intentionally outside READINESS_STATES: the closed state vocabulary applies to
# reported cells, and the JSON matrix carries only reported cells.
NOT_REPORTED = "not_reported"

_EVIDENCE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CAPABILITY_ORDER = {
    capability: index for index, capability in enumerate(READINESS_CAPABILITIES)
}
_EMPTY_NOTE = "_No readiness entries are registered._"


class LanguageReadinessError(ValueError):
    """Raised when a readiness record cannot be represented deterministically."""


def _resolve_aliases(
    aliases: Optional[Mapping[str, str]],
) -> Optional[dict[str, str]]:
    """Validate the shape of a caller alias mapping without mutating it.

    Alias key grammar, canonical targets, duplicate keys, and alias chains stay
    the responsibility of :func:`openmed.core.locale_tag.normalize_locale_tag`;
    this guard only makes wrong-type failures read as
    :class:`LanguageReadinessError`.
    """

    if aliases is None:
        return None
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Mapping):
        raise LanguageReadinessError("aliases must be a mapping of locale tags")
    for key, value in aliases.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise LanguageReadinessError("alias keys and values must be strings")
    return dict(aliases)


@dataclass(frozen=True, slots=True)
class LanguageReadinessEntry:
    """One locale's readiness state for one capability.

    Args:
        locale: Normalized BCP 47 locale tag. Use :meth:`from_tag` when the tag
            comes from a registry that still uses legacy spellings.
        capability: One of :data:`READINESS_CAPABILITIES`.
        state: One of :data:`READINESS_STATES`.
        evidence_digest: ``sha256:<64 lowercase hex>`` evidence pointer. It is
            required for ``complete`` capabilities and omitted for states that
            cannot cite evidence.
    """

    locale: str
    capability: str
    state: str
    evidence_digest: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the closed vocabulary and the evidence digest shape."""

        if not isinstance(self.locale, str) or not self.locale.strip():
            raise LanguageReadinessError("locale must be a non-empty string")
        try:
            object.__setattr__(self, "locale", normalize_locale_tag(self.locale))
        except LocaleTagError as exc:
            raise LanguageReadinessError(str(exc)) from exc

        if self.capability not in READINESS_CAPABILITIES:
            raise LanguageReadinessError(
                f"unknown readiness capability {self.capability!r}"
            )
        if self.state not in READINESS_STATES:
            raise LanguageReadinessError(f"unknown readiness state {self.state!r}")

        digest = self.evidence_digest
        if digest is not None:
            if not isinstance(digest, str) or not _EVIDENCE_DIGEST.fullmatch(digest):
                raise LanguageReadinessError(
                    "evidence_digest must be 'sha256:' followed by 64 lowercase "
                    "hexadecimal characters"
                )
        if self.state == COMPLETE and digest is None:
            raise LanguageReadinessError(
                "complete readiness requires an evidence digest"
            )

    @classmethod
    def from_tag(
        cls,
        locale: str,
        capability: str,
        state: str,
        evidence_digest: Optional[str] = None,
        *,
        aliases: Optional[Mapping[str, str]] = None,
    ) -> LanguageReadinessEntry:
        """Build an entry from a raw tag, resolving explicit aliases first.

        Alias resolution delegates to
        :func:`openmed.core.locale_tag.normalize_locale_tag`, which validates
        the caller mapping (key grammar, canonical targets, duplicate keys, and
        forbidden alias chains) and never mutates it. A tag that is neither a
        supported structural tag nor covered by an alias fails as
        ``locale_tag_invalid``; underscores are never converted implicitly.

        Raises:
            LanguageReadinessError: If the tag cannot be normalized or the
                aliases are malformed.
        """

        if not isinstance(locale, str) or not locale.strip():
            raise LanguageReadinessError("locale must be a non-empty string")
        try:
            canonical = normalize_locale_tag(locale, aliases=_resolve_aliases(aliases))
        except LocaleTagError as exc:
            raise LanguageReadinessError(str(exc)) from exc
        return cls(canonical, capability, state, evidence_digest)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly cell payload."""

        return {
            "capability": self.capability,
            "evidence_digest": self.evidence_digest,
            "locale": self.locale,
            "state": self.state,
        }


def normalize_readiness_entries(
    entries: Iterable[LanguageReadinessEntry],
) -> tuple[LanguageReadinessEntry, ...]:
    """Reject duplicates and return a stable ordering.

    Entries are ordered by locale and then by :data:`READINESS_CAPABILITIES`
    order, so reordered inputs render identically.

    Raises:
        LanguageReadinessError: If ``entries`` is not an iterable of
            :class:`LanguageReadinessEntry` values or if a locale and capability
            pair appears twice.
    """

    if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
        raise LanguageReadinessError(
            "entries must be an iterable of LanguageReadinessEntry values"
        )

    normalized: list[LanguageReadinessEntry] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, LanguageReadinessEntry):
            raise LanguageReadinessError(
                "entries must contain LanguageReadinessEntry values"
            )
        key = (entry.locale, entry.capability)
        if key in seen:
            raise LanguageReadinessError(
                f"duplicate readiness entry for locale {entry.locale!r} "
                f"capability {entry.capability!r}"
            )
        seen.add(key)
        normalized.append(entry)

    normalized.sort(key=lambda item: (item.locale, _CAPABILITY_ORDER[item.capability]))
    return tuple(normalized)


def build_language_readiness_matrix(
    entries: Iterable[LanguageReadinessEntry],
    *,
    version: str,
) -> dict[str, Any]:
    """Return the bounded, JSON-friendly readiness matrix.

    Args:
        entries: Readiness records to project.
        version: Non-empty readiness registry version recorded with the matrix.

    Returns:
        A fresh dictionary with closed capabilities and states, sorted locales,
        one cell per entry, and per-state totals.
    """

    if not isinstance(version, str) or not version.strip():
        raise LanguageReadinessError("version must be a non-empty string")

    normalized = normalize_readiness_entries(entries)
    totals = {
        state: sum(1 for entry in normalized if entry.state == state)
        for state in READINESS_STATES
    }
    return {
        "capabilities": list(READINESS_CAPABILITIES),
        "cells": [entry.to_dict() for entry in normalized],
        "locales": sorted({entry.locale for entry in normalized}),
        "schema_version": MATRIX_SCHEMA_VERSION,
        "states": sorted(READINESS_STATES),
        "totals": totals,
        "version": version,
    }


def render_language_readiness_json(
    entries: Iterable[LanguageReadinessEntry],
    *,
    version: str,
) -> str:
    """Render the readiness matrix as byte-stable, compact JSON."""

    matrix = build_language_readiness_matrix(entries, version=version)
    return json.dumps(
        matrix,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _matrix_cell(
    by_key: Mapping[tuple[str, str], LanguageReadinessEntry],
    locale: str,
    capability: str,
) -> str:
    """Return the Markdown cell text for one locale and capability."""

    entry = by_key.get((locale, capability))
    if entry is None:
        return f"`{NOT_REPORTED}`"
    return f"`{entry.state}`"


def render_language_readiness_markdown(
    entries: Iterable[LanguageReadinessEntry],
    *,
    version: str,
) -> str:
    """Render the readiness matrix as deterministic Markdown.

    The capability matrix, the per-state totals, and the evidence digest table
    are derived from the same normalized snapshot the JSON renderer uses, so
    both formats always agree.
    """

    normalized = normalize_readiness_entries(entries)
    matrix = build_language_readiness_matrix(normalized, version=version)
    locales: list[str] = list(matrix["locales"])
    by_key = {(entry.locale, entry.capability): entry for entry in normalized}

    header = "| Locale | " + " | ".join(READINESS_CAPABILITIES) + " |"
    separator = "| --- |" + " --- |" * len(READINESS_CAPABILITIES)
    lines = [
        "# Language readiness matrix",
        "",
        f"Schema: `{matrix['schema_version']}`",
        "",
        f"Version: `{matrix['version']}`",
        "",
        "## Capability matrix",
        "",
        header,
        separator,
    ]
    if not locales:
        lines.extend(["", _EMPTY_NOTE])
        return "\n".join(lines) + "\n"

    for locale in locales:
        cells = [
            _matrix_cell(by_key, locale, capability)
            for capability in READINESS_CAPABILITIES
        ]
        lines.append(f"| `{locale}` | " + " | ".join(cells) + " |")

    lines.extend(["", "## State totals", "", "| State | Count |", "| --- | --- |"])
    for state in READINESS_STATES:
        lines.append(f"| `{state}` | {matrix['totals'][state]} |")

    lines.extend(
        [
            "",
            "## Evidence digests",
            "",
            "| Locale | Capability | Digest |",
            "| --- | --- | --- |",
        ]
    )
    digested = [entry for entry in normalized if entry.evidence_digest is not None]
    if digested:
        for entry in digested:
            lines.append(
                f"| `{entry.locale}` | {entry.capability} | `{entry.evidence_digest}` |"
            )
    else:
        lines.extend(["", "_No evidence digests are recorded._"])

    return "\n".join(lines) + "\n"


def _evidence_digest(payload: object) -> str:
    """Return the canonical ``sha256:`` digest of a JSON-compatible payload."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _slot_state(status: str) -> str:
    """Map a coherence slot status onto the closed readiness vocabulary."""

    if status == FILLED:
        return COMPLETE
    if status == APPROXIMATED:
        return PARTIAL
    return MISSING_EVIDENCE


def _detection_state(statuses: Iterable[str]) -> str:
    """Combine routing slot statuses into one detection state."""

    states = list(statuses)
    if states and all(status == FILLED for status in states):
        return COMPLETE
    if any(status == FILLED for status in states):
        return PARTIAL
    return MISSING_EVIDENCE


def _validation_state(status: object) -> str:
    """Map the coherence report's national-ID status onto a readiness state."""

    if status == FILLED:
        return COMPLETE
    if status == "absent":
        return MISSING_EVIDENCE
    return BLOCKED


def readiness_entries_from_packs(
    *,
    registry: Optional[LanguagePackRegistry] = None,
) -> tuple[LanguageReadinessEntry, ...]:
    """Derive readiness entries from registered language packs.

    Routing, surrogate, and national-ID capability states come from the existing
    coherence report; ``fixtures`` and ``release_evidence`` stay at
    ``missing_evidence`` because the pack registry does not record them. The
    evidence digest of a ``complete`` cell is the SHA-256 of the canonical JSON
    of the pack evidence that produced it, so the digest changes exactly when
    the underlying declaration changes.
    """

    resolved = registry if registry is not None else LANGUAGE_PACK_REGISTRY
    entries: list[LanguageReadinessEntry] = []

    for row in pack_coherence_report(registry=resolved):
        locale = normalize_locale_tag(str(row["language"]))
        slots = row["coverage"]["slots"]  # type: ignore[index]
        detection = _detection_state(
            (slots["script"], slots["segmenter"], slots["recognizers"])
        )
        routing_evidence = {
            "recognizers": row["recognizers"],
            "scripts": row["scripts"],
            "segmenter": row["segmenter"],
        }
        surrogate_state = _slot_state(slots["surrogate_locale"])
        surrogate_evidence = {"surrogate_locale": row["surrogate_locale"]}
        validation_state = _validation_state(row["national_id"]["status"])  # type: ignore[index]
        validation_evidence = {"national_id": row["national_id"]}

        entries.append(
            LanguageReadinessEntry(
                locale=locale,
                capability="detection",
                state=detection,
                evidence_digest=_evidence_digest(routing_evidence)
                if detection == COMPLETE
                else None,
            )
        )
        entries.append(
            LanguageReadinessEntry(
                locale=locale,
                capability="surrogates",
                state=surrogate_state,
                evidence_digest=_evidence_digest(surrogate_evidence)
                if surrogate_state == COMPLETE
                else None,
            )
        )
        entries.append(
            LanguageReadinessEntry(
                locale=locale,
                capability="validation",
                state=validation_state,
                evidence_digest=_evidence_digest(validation_evidence)
                if validation_state == COMPLETE
                else None,
            )
        )
        entries.append(
            LanguageReadinessEntry(
                locale=locale,
                capability="fixtures",
                state=MISSING_EVIDENCE,
            )
        )
        entries.append(
            LanguageReadinessEntry(
                locale=locale,
                capability="release_evidence",
                state=MISSING_EVIDENCE,
            )
        )

    return normalize_readiness_entries(entries)


__all__ = [
    "BLOCKED",
    "COMPLETE",
    "LanguageReadinessEntry",
    "LanguageReadinessError",
    "MATRIX_SCHEMA_VERSION",
    "MISSING_EVIDENCE",
    "NOT_REPORTED",
    "PARTIAL",
    "READINESS_CAPABILITIES",
    "READINESS_STATES",
    "build_language_readiness_matrix",
    "normalize_readiness_entries",
    "readiness_entries_from_packs",
    "render_language_readiness_json",
    "render_language_readiness_markdown",
]
