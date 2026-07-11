"""Flat-table exporter for coded clinical entities.

The exporter emits one row per entity using a fixed, documented schema:

``entity_label``
    Clinical entity label such as ``"condition"`` or ``"medication"``.
``normalized_text``
    Normalized entity text supplied by the caller.
``system``, ``code``, ``display``
    Bound vocabulary coding fields when present.
``negation``, ``temporality``, ``certainty``
    Context axes when present.
``start``, ``end``
    Character offsets for the entity span when present.
``section``
    Source section label when present.

Only these whitelisted fields are copied into the flat rows. Missing values are
left blank so CSV and DataFrame exports keep the same columns even for empty or
partially annotated inputs.
"""

from __future__ import annotations

import csv
import io
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TextIO

__all__ = [
    "FLAT_TABLE_COLUMNS",
    "flatten_entities",
    "flatten_clinical_entities",
    "to_csv",
    "to_dataframe",
]

FLAT_TABLE_COLUMNS: tuple[str, ...] = (
    "entity_label",
    "normalized_text",
    "system",
    "code",
    "display",
    "negation",
    "temporality",
    "certainty",
    "start",
    "end",
    "section",
)

_MISSING = object()

_ENTITY_LABEL_FIELDS = (
    "entity_label",
    "label",
    "entity_type",
    "entity_group",
    "entity",
)
_NORMALIZED_TEXT_FIELDS = ("normalized_text", "text", "entity_text", "word", "surface")
_SYSTEM_FIELDS = ("system", "code_system", "coding_system")
_CODE_FIELDS = ("code", "code_value", "coding_code")
_DISPLAY_FIELDS = ("display", "code_display", "coding_display")
_NEGATION_FIELDS = ("negation", "polarity")
_TEMPORALITY_FIELDS = ("temporality", "temporal_status")
_CERTAINTY_FIELDS = ("certainty", "assertion_certainty")
_SECTION_FIELDS = ("section", "section_label", "section_name")
_METADATA_FIELDS = ("metadata", "meta")
_CONTEXT_FIELDS = (
    "context",
    "clinical_context",
    "assertion",
    "clinical_assertion",
    "context_axes",
)


def flatten_entities(entities: Iterable[Any]) -> list[dict[str, Any]]:
    """Flatten clinical entities into stable row dictionaries.

    Args:
        entities: Iterable of entity mappings or simple objects. Fields may be
            present directly on the entity or inside ``metadata``. Coding may
            be supplied as direct ``system``/``code``/``display`` fields, a
            ``coding`` mapping, a ``codings`` list, or a FHIR-like
            ``code.coding`` / ``codeable_concept.coding`` structure.

    Returns:
        A list containing one dictionary per entity. Every dictionary has keys
        ordered exactly as :data:`FLAT_TABLE_COLUMNS`. Missing scalar values are
        emitted as ``""``; offsets are emitted as integers when possible and
        ``""`` otherwise.
    """

    return [_row_for_entity(entity) for entity in entities]


def flatten_clinical_entities(entities: Iterable[Any]) -> list[dict[str, Any]]:
    """Alias for :func:`flatten_entities` with a more explicit name."""

    return flatten_entities(entities)


def to_csv(
    entities: Iterable[Any],
    output: str | os.PathLike[str] | TextIO | None = None,
) -> str | None:
    """Write clinical entities to CSV using the fixed flat-table schema.

    Args:
        entities: Iterable of entity mappings or simple objects.
        output: Optional text stream or filesystem path. When omitted, the CSV
            document is returned as a string.

    Returns:
        The CSV text when *output* is omitted; otherwise ``None``.
    """

    rows = flatten_entities(entities)
    if output is None:
        stream = io.StringIO(newline="")
        _write_csv(rows, stream)
        return stream.getvalue()

    if isinstance(output, (str, os.PathLike)):
        with open(output, "w", encoding="utf-8", newline="") as stream:
            _write_csv(rows, stream)
        return None

    _write_csv(rows, output)
    return None


def to_dataframe(entities: Iterable[Any]) -> Any:
    """Return a pandas DataFrame using the fixed flat-table schema.

    pandas is imported lazily so importing this module has no pandas
    dependency. Install pandas only when DataFrame export is needed.

    Args:
        entities: Iterable of entity mappings or simple objects.

    Returns:
        ``pandas.DataFrame`` with columns ordered as
        :data:`FLAT_TABLE_COLUMNS`.

    Raises:
        ImportError: If pandas is not installed.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - covered via monkeypatch
        raise ImportError(
            "to_dataframe requires pandas; install pandas to use DataFrame export."
        ) from exc

    return pd.DataFrame(flatten_entities(entities), columns=list(FLAT_TABLE_COLUMNS))


def _write_csv(rows: Sequence[Mapping[str, Any]], stream: TextIO) -> None:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(FLAT_TABLE_COLUMNS),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)


def _row_for_entity(entity: Any) -> dict[str, Any]:
    coding = _primary_coding(entity)

    row = {
        "entity_label": _first_scalar(_sources(entity), _ENTITY_LABEL_FIELDS),
        "normalized_text": _first_scalar(_sources(entity), _NORMALIZED_TEXT_FIELDS),
        "system": _first_scalar(_coding_sources(entity, coding), _SYSTEM_FIELDS),
        "code": _code_value(entity, coding),
        "display": _first_scalar(_coding_sources(entity, coding), _DISPLAY_FIELDS),
        "negation": _first_scalar(_context_sources(entity), _NEGATION_FIELDS),
        "temporality": _first_scalar(_context_sources(entity), _TEMPORALITY_FIELDS),
        "certainty": _first_scalar(_context_sources(entity), _CERTAINTY_FIELDS),
        "start": _offset_value(_first_value(_sources(entity), ("start",))),
        "end": _offset_value(_first_value(_sources(entity), ("end",))),
        "section": _first_scalar(_context_sources(entity), _SECTION_FIELDS),
    }
    return {column: row[column] for column in FLAT_TABLE_COLUMNS}


def _sources(entity: Any) -> tuple[Any, ...]:
    sources: list[Any] = [entity]
    for name in _METADATA_FIELDS:
        metadata = _value(entity, name)
        if metadata is not _MISSING:
            sources.append(metadata)
    return tuple(sources)


def _context_sources(entity: Any) -> tuple[Any, ...]:
    sources = list(_sources(entity))
    for source in tuple(sources):
        for name in _CONTEXT_FIELDS:
            nested = _value(source, name)
            if nested is not _MISSING:
                sources.append(nested)
    return tuple(sources)


def _coding_sources(entity: Any, coding: Mapping[str, Any] | None) -> tuple[Any, ...]:
    sources = list(_sources(entity))
    if coding is not None:
        sources.insert(0, coding)
    return tuple(sources)


def _code_value(entity: Any, coding: Mapping[str, Any] | None) -> str:
    for source in _sources(entity):
        for name in _CODE_FIELDS:
            value = _value(source, name)
            if value is not _MISSING and not isinstance(value, Mapping):
                return _scalar_text(value)
    if coding is None:
        return ""
    return _first_scalar((coding,), _CODE_FIELDS)


def _primary_coding(entity: Any) -> Mapping[str, Any] | None:
    for source in _sources(entity):
        for name in ("coding", "codings", "code", "codeable_concept"):
            candidate = _value(source, name)
            coding = _coerce_coding(candidate)
            if coding is not None:
                return coding
        candidate = _value(source, "candidates")
        coding = _coerce_coding(candidate)
        if coding is not None:
            return coding
    return None


def _coerce_coding(value: Any) -> Mapping[str, Any] | None:
    if value is _MISSING or value is None or isinstance(value, (str, bytes)):
        return None

    if isinstance(value, Mapping):
        nested = _value(value, "coding")
        if nested is not _MISSING:
            coding = _coerce_coding(nested)
            if coding is not None:
                return coding

        nested = _value(value, "codings")
        if nested is not _MISSING:
            coding = _coerce_coding(nested)
            if coding is not None:
                return coding

        if any(
            _value(value, key) is not _MISSING for key in ("system", "code", "display")
        ):
            return value
        return None

    if isinstance(value, Sequence):
        for item in value:
            coding = _coerce_coding(item)
            if coding is not None:
                return coding
        return None

    if any(_value(value, key) is not _MISSING for key in ("system", "code", "display")):
        return {
            key: _value(value, key)
            for key in ("system", "code", "display")
            if _value(value, key) is not _MISSING
        }
    return None


def _first_scalar(sources: Iterable[Any], names: Iterable[str]) -> str:
    return _scalar_text(_first_value(sources, names))


def _first_value(sources: Iterable[Any], names: Iterable[str]) -> Any:
    for source in sources:
        for name in names:
            value = _value(source, name)
            if value is not _MISSING:
                return value
    return _MISSING


def _value(source: Any, name: str) -> Any:
    if source is None:
        return _MISSING
    if isinstance(source, Mapping):
        return source.get(name, _MISSING)
    return getattr(source, name, _MISSING)


def _scalar_text(value: Any) -> str:
    if value is _MISSING or value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return ""


def _offset_value(value: Any) -> int | str:
    if value is _MISSING or value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return ""
    return ""
