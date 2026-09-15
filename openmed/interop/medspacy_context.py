"""Optional medspaCy ConText annotations for OpenMed spans.

The adapter consumes a processed medspaCy/spaCy ``Doc`` and never constructs a
pipeline or downloads a model. A caller may therefore keep medspaCy entirely
outside the OpenMed core and pass the resulting document to
:func:`to_canonical`.

medspaCy and spaCy character offsets are Python string, half-open offsets:
``[start_char, end_char)``. The alignment contract is intentionally exact: an
OpenMed span is enriched only when its ``(start, end)`` pair is identical to a
medspaCy span from the same, unmodified ``Doc.text``. Token offsets, normalized
text, and overlap-based guesses are not interchangeable with this contract.
This avoids attaching assertion context to a neighboring or partially
overlapping entity.

Context is stored in the existing OpenMed ``metadata["clinical_context"]``
shape. medspaCy's booleans are also retained under
``metadata["medspacy_context"]`` as safe, structured provenance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import import_module as _import_module
from typing import Any, Final

from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.labels import normalize_label
from openmed.core.pii import PIIEntity
from openmed.core.schemas import OpenMedSpan

_MISSING: Final = object()
_CONTEXT_ATTRIBUTES: Final = (
    "is_negated",
    "is_uncertain",
    "is_historical",
    "is_family",
)
_FALSE_VALUES: Final = frozenset({"", "0", "false", "no", "off", "none"})
_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class MedspacyContextAdapterConfig:
    """Configuration for converting medspaCy ConText spans.

    ``default_confidence`` applies because ConText flags are rule annotations,
    not model scores. ``preserve_context_flags`` controls whether the raw
    medspaCy booleans are retained alongside the canonical clinical context;
    the canonical fields are always written.
    """

    source: str = "medspacy"
    default_confidence: float = 1.0
    preserve_context_flags: bool = True

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if not 0.0 <= float(self.default_confidence) <= 1.0:
            raise ValueError("default_confidence must be between 0.0 and 1.0")


def to_canonical(
    doc: Any,
    *,
    text: str | None = None,
    openmed_spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]] | None = None,
    existing_spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]] | None = None,
    spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]] | None = None,
    config: MedspacyContextAdapterConfig | None = None,
) -> list[Any]:
    """Convert a processed medspaCy ``Doc`` to canonical OpenMed spans.

    With no supplied OpenMed spans, each medspaCy entity becomes a
    :class:`~openmed.core.pii.PIIEntity`. When ``openmed_spans`` (or one of its
    aliases) is supplied, the input spans are copied and only exact offset
    matches receive the context metadata. ``PIIEntity`` and ``OpenMedSpan``
    inputs retain their respective types; mapping inputs are copied as
    mappings. The input objects are never mutated.

    The aliases exist to make the alignment use explicit at call sites while
    remaining compatible with both the PIIEntity and OpenMedSpan surfaces.
    Supplying more than one alias is an error.
    """

    cfg = config or MedspacyContextAdapterConfig()
    supplied = _coalesce_existing_spans(
        openmed_spans=openmed_spans,
        existing_spans=existing_spans,
        spans=spans,
    )
    document_text = _document_text(doc, explicit_text=text)
    context_records = _context_records(doc, text=document_text, config=cfg)

    if supplied is None:
        return [
            _record_to_entity(record, text=document_text, config=cfg)
            for record in context_records
        ]
    return _attach_records(
        _as_list(supplied),
        context_records,
        text=document_text,
        config=cfg,
    )


def attach_context(
    doc: Any,
    openmed_spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]],
    *,
    text: str | None = None,
    config: MedspacyContextAdapterConfig | None = None,
) -> list[Any]:
    """Attach exact-offset ConText flags to existing OpenMed spans.

    This is the explicit alignment-oriented form of :func:`to_canonical`.
    Unmatched OpenMed spans are copied unchanged, while matched spans receive
    ``metadata["clinical_context"]`` and optional raw medspaCy flags.
    """

    return to_canonical(
        doc,
        text=text,
        openmed_spans=openmed_spans,
        config=config,
    )


def process_to_canonical(
    text: str,
    *,
    nlp: Any,
    openmed_spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]] | None = None,
    config: MedspacyContextAdapterConfig | None = None,
) -> list[Any]:
    """Run a caller-configured medspaCy pipeline and convert its ``Doc``.

    The optional ``medspacy`` and ``spacy`` packages are imported only when
    this function is explicitly called. No default pipeline or model is
    created, and callers remain responsible for local model configuration.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not callable(nlp):
        raise TypeError("nlp must be a callable, configured medspaCy pipeline")
    _require_runtime_dependencies()
    return to_canonical(
        nlp(text),
        text=text,
        openmed_spans=openmed_spans,
        config=config,
    )


def _require_runtime_dependencies() -> None:
    """Import optional runtime packages only on explicit pipeline use."""

    try:
        _import_module("medspacy")
        _import_module("spacy")
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            package="medspacy",
            feature="medspaCy ConText support",
            extra="medspacy",
        ) from exc


def _context_records(
    doc: Any,
    *,
    text: str | None,
    config: MedspacyContextAdapterConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for span in _span_records(doc):
        start, end = _span_offsets(span, text=text)
        flags = _context_flags(span)
        records.append(
            {
                "start": start,
                "end": end,
                "label": _span_label(span),
                "confidence": _span_confidence(
                    span,
                    default=config.default_confidence,
                ),
                "text": _span_surface(span, text=text, start=start, end=end),
                "flags": flags,
            }
        )
    return records


def _record_to_entity(
    record: Mapping[str, Any],
    *,
    text: str | None,
    config: MedspacyContextAdapterConfig,
) -> PIIEntity:
    start = int(record["start"])
    end = int(record["end"])
    surface = str(record["text"])
    label = normalize_label(str(record["label"]))
    metadata = _context_metadata(
        flags=record["flags"],
        source=config.source,
        preserve_context_flags=config.preserve_context_flags,
    )
    return PIIEntity(
        text=surface,
        label=label,
        confidence=float(record["confidence"]),
        start=start,
        end=end,
        entity_type=label,
        original_text=surface,
        metadata=metadata,
    )


def _attach_records(
    spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    text: str | None,
    config: MedspacyContextAdapterConfig,
) -> list[Any]:
    by_offsets: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (int(record["start"]), int(record["end"]))
        by_offsets.setdefault(key, []).append(record)

    attached: list[Any] = []
    for span in spans:
        offsets = _existing_offsets(span)
        matched = by_offsets.get(offsets) if offsets is not None else None
        if not matched:
            attached.append(_copy_without_context(span))
            continue

        flags = _merge_flags(record["flags"] for record in matched)
        metadata = _context_metadata(
            flags=flags,
            source=config.source,
            preserve_context_flags=config.preserve_context_flags,
            existing_metadata=_existing_metadata(span),
        )
        attached.append(_copy_with_metadata(span, metadata, text=text))
    return attached


def _context_metadata(
    *,
    flags: Mapping[str, bool],
    source: str,
    preserve_context_flags: bool,
    existing_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(existing_metadata or {})
    metadata["adapter"] = source
    metadata["source"] = source
    metadata["clinical_context"] = _clinical_context(flags)
    metadata["clinical_context_sources"] = {
        key: source for key in metadata["clinical_context"]
    }
    metadata["medspacy_offset_alignment"] = "exact"
    if preserve_context_flags:
        metadata["medspacy_context"] = dict(flags)
    else:
        metadata.pop("medspacy_context", None)
    return metadata


def _clinical_context(flags: Mapping[str, bool]) -> dict[str, str]:
    return {
        "negation": "negated" if flags["is_negated"] else "affirmed",
        "uncertainty": "uncertain" if flags["is_uncertain"] else "certain",
        "temporality": "historical" if flags["is_historical"] else "recent",
        "experiencer": "family" if flags["is_family"] else "patient",
    }


def _context_flags(span: Any) -> dict[str, bool]:
    return {
        name: _as_bool(_extension_value(span, name)) for name in _CONTEXT_ATTRIBUTES
    }


def _merge_flags(flag_sets: Iterable[Mapping[str, bool]]) -> dict[str, bool]:
    merged = {name: False for name in _CONTEXT_ATTRIBUTES}
    for flags in flag_sets:
        for name in _CONTEXT_ATTRIBUTES:
            merged[name] = merged[name] or bool(flags.get(name, False))
    return merged


def _span_records(doc: Any) -> list[Any]:
    if doc is None:
        return []
    if _has_offsets(doc):
        return [doc]
    if isinstance(doc, Mapping):
        for key in ("ents", "entities", "spans"):
            if key in doc:
                if key == "spans" and isinstance(doc[key], Mapping):
                    return [
                        item for group in doc[key].values() for item in _as_list(group)
                    ]
                return _as_list(doc[key])
        return []

    entities = getattr(doc, "ents", _MISSING)
    if entities is not _MISSING:
        return _as_list(entities)
    for key in ("entities", "spans"):
        value = getattr(doc, key, _MISSING)
        if value is not _MISSING:
            if key == "spans" and isinstance(value, Mapping):
                return [item for group in value.values() for item in _as_list(group)]
            return _as_list(value)
    if isinstance(doc, Iterable) and not isinstance(doc, (str, bytes)):
        return list(doc)
    return []


def _document_text(doc: Any, *, explicit_text: str | None) -> str | None:
    if explicit_text is not None:
        if not isinstance(explicit_text, str):
            raise TypeError("text must be a string")
        return explicit_text
    if _has_offsets(doc):
        return None
    value = _value(doc, ("text", "document_text", "full_text"))
    return value if isinstance(value, str) else None


def _span_offsets(span: Any, *, text: str | None) -> tuple[int, int]:
    start = _value(span, ("start_char", "start"))
    end = _value(span, ("end_char", "end"))
    if start is _MISSING or end is _MISSING:
        raise ValueError("medspaCy span must provide start_char and end_char")
    try:
        start_int = int(start)
        end_int = int(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("medspaCy span offsets must be integers") from exc
    if start_int < 0 or end_int < start_int:
        raise ValueError("medspaCy span offsets must be non-negative and half-open")
    if text is not None and end_int > len(text):
        raise ValueError("medspaCy span offsets exceed the document text")
    return start_int, end_int


def _existing_offsets(span: Any) -> tuple[int, int] | None:
    start = _value(span, ("start", "start_char"))
    end = _value(span, ("end", "end_char"))
    if start is _MISSING or end is _MISSING:
        return None
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None


def _span_label(span: Any) -> str:
    label = _value(span, ("label_", "label", "entity_type", "canonical_label"))
    if label is _MISSING or label is None or not str(label).strip():
        return "CONDITION"
    return str(label)


def _span_confidence(span: Any, *, default: float) -> float:
    value = _extension_value(span, "score")
    if value is _MISSING:
        value = _value(span, ("score", "confidence"))
    if value is _MISSING or value is None:
        return float(default)
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return float(default)
    return confidence if 0.0 <= confidence <= 1.0 else float(default)


def _span_surface(
    span: Any,
    *,
    text: str | None,
    start: int,
    end: int,
) -> str:
    surface = _value(span, ("text", "surface", "word"))
    if surface is not _MISSING and surface is not None:
        return str(surface)
    if text is not None:
        return text[start:end]
    return ""


def _existing_metadata(span: Any) -> Mapping[str, Any] | None:
    metadata = _value(span, "metadata")
    return metadata if isinstance(metadata, Mapping) else None


def _copy_without_context(span: Any) -> Any:
    if isinstance(span, PIIEntity):
        return replace(span, metadata=dict(span.metadata) if span.metadata else None)
    if isinstance(span, OpenMedSpan):
        return replace(span, metadata=dict(span.metadata))
    if isinstance(span, Mapping):
        return dict(span)
    return span


def _copy_with_metadata(
    span: Any, metadata: Mapping[str, Any], *, text: str | None
) -> Any:
    if isinstance(span, PIIEntity):
        return replace(span, metadata=dict(metadata))
    if isinstance(span, OpenMedSpan):
        return replace(span, metadata=dict(metadata))
    if isinstance(span, Mapping):
        copied = dict(span)
        copied["metadata"] = dict(metadata)
        return copied

    offsets = _existing_offsets(span)
    if offsets is None:
        raise TypeError("OpenMed spans must expose start/end offsets")
    surface = _span_surface(span, text=text, start=offsets[0], end=offsets[1])
    label = normalize_label(_span_label(span))
    return PIIEntity(
        text=surface,
        label=label,
        confidence=_span_confidence(span, default=1.0),
        start=offsets[0],
        end=offsets[1],
        entity_type=label,
        original_text=surface,
        metadata=dict(metadata),
    )


def _coalesce_existing_spans(
    *,
    openmed_spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]] | None,
    existing_spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]] | None,
    spans: Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]] | None,
) -> Sequence[PIIEntity | OpenMedSpan | Mapping[str, Any]] | None:
    values = [
        value for value in (openmed_spans, existing_spans, spans) if value is not None
    ]
    if len(values) > 1:
        raise TypeError("provide only one OpenMed span argument")
    return values[0] if values else None


def _extension_value(span: Any, name: str) -> Any:
    value = _value(span, name)
    if value is not _MISSING:
        return value
    extensions = _value(span, "_")
    if extensions is _MISSING or extensions is None:
        return _MISSING
    return _value(extensions, name)


def _value(record: Any, names: str | Sequence[str], default: Any = _MISSING) -> Any:
    candidates = (names,) if isinstance(names, str) else names
    for name in candidates:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        value = getattr(record, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _has_offsets(record: Any) -> bool:
    return (
        _value(record, ("start_char", "start")) is not _MISSING
        and _value(
            record,
            ("end_char", "end"),
        )
        is not _MISSING
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _as_bool(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _FALSE_VALUES:
            return False
        if normalized in _TRUE_VALUES:
            return True
    return bool(value)


__all__ = [
    "MedspacyContextAdapterConfig",
    "attach_context",
    "process_to_canonical",
    "to_canonical",
]
