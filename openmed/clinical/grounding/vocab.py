"""Lightweight free-vocabulary index for concept-grounding linkers.

The loader accepts either a JSONL file path or in-memory rows, so no full
vocabulary is bundled: callers provide their own (open) vocabulary data at
runtime, while tests use a small committed fixture. Each row carries a code, a
display name, and synonyms/aliases; key names are matched leniently so the same
loader serves RxNorm, ICD-10-CM, LOINC and HPO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

_CODE_KEYS = ("code", "rxcui", "cui", "id")
_DISPLAY_KEYS = ("display", "name", "str", "label")
_SYNONYM_KEYS = ("synonyms", "aliases", "terms")


def normalize_term(term: Any) -> str:
    """Casefold, trim and collapse whitespace for alias matching."""
    return " ".join(str(term).strip().casefold().split())


@dataclass(frozen=True)
class VocabEntry:
    """One vocabulary concept: a code, display name and its aliases."""

    code: str
    display: str
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class VocabIndex:
    """An indexed vocabulary with exact-alias lookup and alias iteration."""

    entries: tuple[VocabEntry, ...]
    alias_map: Mapping[str, tuple[VocabEntry, ...]]

    def exact(self, term: str) -> list[VocabEntry]:
        """Return entries whose alias exactly matches ``term`` (normalized)."""
        return list(self.alias_map.get(normalize_term(term), ()))

    def iter_alias_entries(self) -> Iterator[tuple[str, VocabEntry]]:
        """Yield ``(normalized_alias, entry)`` for every alias in the index."""
        for entry in self.entries:
            for synonym in entry.synonyms:
                yield normalize_term(synonym), entry


def load_vocab(source: str | Path | Iterable[Mapping[str, Any]]) -> VocabIndex:
    """Load a vocabulary from a JSONL path or an iterable of row mappings."""
    entries = tuple(_entry_from_row(row) for row in _iter_rows(source))

    alias_map: dict[str, list[VocabEntry]] = {}
    for entry in entries:
        aliases = entry.synonyms or (entry.display,)
        for alias in aliases:
            key = normalize_term(alias)
            if not key:
                continue
            bucket = alias_map.setdefault(key, [])
            if entry not in bucket:
                bucket.append(entry)

    return VocabIndex(
        entries=entries,
        alias_map={key: tuple(value) for key, value in alias_map.items()},
    )


def _iter_rows(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return [dict(row) for row in source]


def _entry_from_row(row: Mapping[str, Any]) -> VocabEntry:
    code = _first(row, _CODE_KEYS)
    if code is None:
        raise ValueError(f"vocabulary row is missing a code field: {row!r}")
    display = _first(row, _DISPLAY_KEYS, default=code)
    synonyms = _first(row, _SYNONYM_KEYS, default=None)
    if isinstance(synonyms, Sequence) and not isinstance(synonyms, (str, bytes)):
        synonym_tuple = tuple(str(item) for item in synonyms)
    else:
        synonym_tuple = (str(display),)
    return VocabEntry(code=str(code), display=str(display), synonyms=synonym_tuple)


def _first(
    row: Mapping[str, Any], keys: tuple[str, ...], *, default: Any = None
) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default
