"""Protect common clinical terms from ambiguous PII over-redaction."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar

from .labels import (
    FIRST_NAME,
    LAST_NAME,
    LOCATION,
    MIDDLE_NAME,
    ORGANIZATION,
    PERSON,
    normalize_label,
)

T = TypeVar("T")

DATA_FILE = Path(__file__).with_name("data") / "clinical_protect_terms.txt"
PROTECTION_SOURCE = "openmed/core/data/clinical_protect_terms.txt"
PROTECTION_VERSION = "clinical-protect-terms-v1"

_PROTECTED_CANONICAL_LABELS = frozenset(
    {
        PERSON,
        FIRST_NAME,
        LAST_NAME,
        MIDDLE_NAME,
        LOCATION,
        ORGANIZATION,
    }
)
_RUNTIME_TERMS: set[str] = set()
_RUNTIME_LOCK = RLock()
_WHITESPACE_RE = re.compile(r"\s+")
_DIRECT_IDENTIFIER_RE = (
    re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"),
    re.compile(
        r"^(?:mrn|medical record|patient id|id)[:#\s-]*[a-z0-9][a-z0-9-]{3,}$", re.I
    ),
    re.compile(r"^[a-z]{0,4}\d[a-z0-9-]{5,}$", re.I),
)


@dataclass(frozen=True)
class ClinicalProtectionResult:
    """Filtered span result plus safe aggregate suppression metadata."""

    spans: list[Any]
    suppressed_count: int
    checked_count: int
    protected_term_count: int

    @property
    def metadata(self) -> dict[str, Any]:
        """Return safe metadata without raw clinical or patient text."""

        return {
            "clinical_protection": {
                "source": PROTECTION_SOURCE,
                "version": PROTECTION_VERSION,
                "protected_term_count": self.protected_term_count,
                "checked_spans": self.checked_count,
                "suppressed_spans": self.suppressed_count,
            }
        }


def load_bundled_terms() -> frozenset[str]:
    """Load normalized bundled clinical protection terms.

    Returns:
        A frozenset of normalized protected terms from the bundled data file.
    """

    return frozenset(_load_bundled_terms_cached())


@lru_cache(maxsize=1)
def _load_bundled_terms_cached() -> tuple[str, ...]:
    terms: list[str] = []
    with DATA_FILE.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            normalized = normalize_term(line)
            if normalized:
                terms.append(normalized)
    return tuple(sorted(set(terms)))


def normalize_term(term: str) -> str:
    """Normalize a clinical term for exact whole-span matching.

    Args:
        term: Candidate term or span surface.

    Returns:
        Case-folded, NFC-normalized text with internal whitespace collapsed.
    """

    normalized = unicodedata.normalize("NFC", str(term or "").strip())
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.casefold()


def add_protected_terms(terms: Iterable[str]) -> None:
    """Add clinical protection terms for the current Python process.

    Args:
        terms: Terms to add. Empty strings and whitespace-only values are ignored.
    """

    additions = {normalize_term(term) for term in _coerce_terms(terms)}
    additions.discard("")
    if not additions:
        return
    with _RUNTIME_LOCK:
        _RUNTIME_TERMS.update(additions)


def protected_terms(
    *,
    extra_terms: Iterable[str] | None = None,
    include_builtin: bool = True,
) -> frozenset[str]:
    """Return active protected terms.

    Args:
        extra_terms: Per-call or config-supplied terms to include.
        include_builtin: Include the bundled OpenMed-maintained term list.

    Returns:
        Active normalized terms, including runtime additions.
    """

    active: set[str] = set(load_bundled_terms() if include_builtin else ())
    with _RUNTIME_LOCK:
        active.update(_RUNTIME_TERMS)
    if extra_terms is not None:
        active.update(normalize_term(term) for term in _coerce_terms(extra_terms))
    active.discard("")
    return frozenset(active)


def protection_options_from_config(config: Any) -> dict[str, Any]:
    """Resolve clinical protection options from an optional OpenMed config.

    Args:
        config: Optional config object with clinical protection attributes.

    Returns:
        Keyword arguments suitable for :func:`filter_protected_spans`.
    """

    if config is None:
        return {
            "enabled": True,
            "extra_terms": None,
            "include_builtin": True,
        }
    return {
        "enabled": bool(getattr(config, "clinical_protect_enabled", True)),
        "extra_terms": getattr(config, "clinical_protect_terms", None),
        "include_builtin": bool(getattr(config, "clinical_protect_use_builtin", True)),
    }


def protect_spans(
    spans: Sequence[T],
    text: str,
    *,
    extra_terms: Iterable[str] | None = None,
    include_builtin: bool = True,
    lang: str = "en",
    enabled: bool = True,
) -> list[T]:
    """Drop ambiguous PII spans whose surface is a protected clinical term.

    Protection only applies to PERSON/name, LOCATION, and ORGANIZATION-class
    labels. Direct identifiers such as SSNs, MRNs, dates, and IDs are never
    suppressed by this filter.

    Args:
        spans: Entity or span records with offsets and labels.
        text: Source text for exact surface extraction.
        extra_terms: Per-call terms to include with the runtime/bundled list.
        include_builtin: Include the bundled OpenMed-maintained term list.
        lang: Label normalization language hint.
        enabled: Return spans unchanged when false.

    Returns:
        The retained spans in their original order.
    """

    return filter_protected_spans(
        spans,
        text,
        extra_terms=extra_terms,
        include_builtin=include_builtin,
        lang=lang,
        enabled=enabled,
    ).spans


def filter_protected_spans(
    spans: Sequence[T],
    text: str,
    *,
    extra_terms: Iterable[str] | None = None,
    include_builtin: bool = True,
    lang: str = "en",
    enabled: bool = True,
) -> ClinicalProtectionResult:
    """Filter protected clinical terms and return aggregate metadata.

    Args:
        spans: Entity or span records with offsets and labels.
        text: Source text for exact surface extraction.
        extra_terms: Per-call terms to include with the runtime/bundled list.
        include_builtin: Include the bundled OpenMed-maintained term list.
        lang: Label normalization language hint.
        enabled: Return spans unchanged when false.

    Returns:
        A :class:`ClinicalProtectionResult` with retained spans and counts.
    """

    span_list = list(spans)
    terms = protected_terms(extra_terms=extra_terms, include_builtin=include_builtin)
    if not enabled or not terms:
        return ClinicalProtectionResult(
            spans=span_list,
            suppressed_count=0,
            checked_count=0,
            protected_term_count=len(terms),
        )

    retained: list[T] = []
    checked_count = 0
    suppressed_count = 0
    for span in span_list:
        if _is_ambiguous_label(span, lang):
            checked_count += 1
            surface = _span_surface(span, text)
            if (
                surface is not None
                and normalize_term(surface) in terms
                and not _looks_like_direct_identifier(surface)
            ):
                suppressed_count += 1
                continue
        retained.append(span)

    return ClinicalProtectionResult(
        spans=retained,
        suppressed_count=suppressed_count,
        checked_count=checked_count,
        protected_term_count=len(terms),
    )


def _is_ambiguous_label(span: Any, lang: str) -> bool:
    label = (
        getattr(span, "canonical_label", None)
        or getattr(span, "label", None)
        or getattr(span, "entity_type", None)
        or ""
    )
    return normalize_label(str(label), lang=lang) in _PROTECTED_CANONICAL_LABELS


def _coerce_terms(terms: Iterable[str]) -> tuple[str, ...]:
    if isinstance(terms, str):
        values = terms.split(",")
    else:
        values = terms
    return tuple(str(term).strip() for term in values if str(term).strip())


def _looks_like_direct_identifier(surface: str) -> bool:
    text = surface.strip()
    return any(pattern.fullmatch(text) for pattern in _DIRECT_IDENTIFIER_RE)


def _span_surface(span: Any, text: str) -> str | None:
    start = getattr(span, "start", None)
    end = getattr(span, "end", None)
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    ):
        return text[start:end]
    surface = getattr(span, "text", None)
    if surface is not None:
        return str(surface)
    return None


__all__ = [
    "ClinicalProtectionResult",
    "PROTECTION_SOURCE",
    "PROTECTION_VERSION",
    "add_protected_terms",
    "filter_protected_spans",
    "load_bundled_terms",
    "normalize_term",
    "protect_spans",
    "protected_terms",
    "protection_options_from_config",
]
