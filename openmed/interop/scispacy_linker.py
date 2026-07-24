"""Canonical adapter for user-configured scispaCy UMLS linker output.

OpenMed does not bundle, download, or initialize UMLS resources. Callers keep
responsibility for licensing and configuring the scispaCy pipeline; this module
only converts the resulting ``Span._.kb_ents`` candidates.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any, Iterable, Mapping

UMLS_SYSTEM = "UMLS"
_MISSING = object()


class ScispaCyLinkerResourceError(RuntimeError):
    """Raised when a configured, user-supplied UMLS linker is unavailable."""


def to_canonical(linked: Any) -> list[dict[str, Any]]:
    """Map scispaCy linked spans to canonical span dictionaries.

    ``linked`` may be a spaCy ``Doc``, one span, an iterable of spans, or
    equivalent mapping-based stubs. Each returned span has a ``codes`` list of
    ``{"system": "UMLS", "code": <CUI>, "score": <score>}`` mappings.

    This conversion path never imports scispaCy and never accesses the network
    or a vocabulary resource.
    """

    canonical: list[dict[str, Any]] = []
    for span in _span_records(linked):
        candidates = _candidate_records(span)
        if candidates is _MISSING:
            raise ScispaCyLinkerResourceError(_resource_message())
        if not isinstance(candidates, list):
            raise TypeError(f"candidates must be a list, got {type(candidates).__name__}")

        output: dict[str, Any] = {}
        text = _value(span, ("text", "ngram"))
        start = _value(span, ("start_char", "start"))
        end = _value(span, ("end_char", "end"))
        if text is not _MISSING:
            output["text"] = str(text)
        if start is not _MISSING:
            output["start"] = int(start)
        if end is not _MISSING:
            output["end"] = int(end)
        output["codes"] = [_canonical_code(candidate) for candidate in candidates]
        canonical.append(output)
    return canonical


def link_to_canonical(text: str, *, nlp: Any) -> list[dict[str, Any]]:
    """Run a caller-configured scispaCy pipeline and convert its linked spans.

    The pipeline must already contain ``scispacy_linker`` backed by UMLS data
    obtained and prepared by the user. OpenMed intentionally provides no
    automatic resource download or default linker construction.
    """

    if nlp is None:
        raise ScispaCyLinkerResourceError(_resource_message())
    try:
        _import_module("scispacy")
    except ImportError as exc:
        raise ImportError(
            "scispaCy support requires the optional dependency. "
            "Install with `pip install openmed[scispacy]`."
        ) from exc

    has_pipe = getattr(nlp, "has_pipe", None)
    if callable(has_pipe) and not has_pipe("scispacy_linker"):
        raise ScispaCyLinkerResourceError(_resource_message())
    if not callable(nlp):
        raise TypeError("nlp must be a callable, configured scispaCy pipeline")
    return to_canonical(nlp(text))


def _span_records(linked: Any) -> list[Any]:
    if linked is None:
        return []
    if _candidate_value(linked) is not _MISSING:
        return [linked]
    if isinstance(linked, Mapping):
        for key in ("ents", "entities", "spans"):
            records = linked.get(key, _MISSING)
            if records is not _MISSING:
                return _as_list(records)
        return [linked]

    entities = getattr(linked, "ents", _MISSING)
    if entities is not _MISSING:
        return _as_list(entities)
    return _as_list(linked)


def _candidate_records(span: Any) -> list[Any] | object:
    candidates = _candidate_value(span)
    if candidates is _MISSING:
        return _MISSING
    return _as_list(candidates)


def _candidate_value(span: Any) -> Any:
    if isinstance(span, Mapping):
        for key in ("kb_ents", "candidates"):
            if key in span:
                return span[key]

    direct = getattr(span, "kb_ents", _MISSING)
    if direct is not _MISSING:
        return direct
    extensions = getattr(span, "_", _MISSING)
    if extensions is not _MISSING:
        return getattr(extensions, "kb_ents", _MISSING)
    return _MISSING


def _canonical_code(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        cui = _value(candidate, ("cui", "concept_id", "code"))
        score = _value(candidate, ("score", "similarity", "confidence"))
    elif isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
        cui, score = candidate[0], candidate[1]
    else:
        cui = _value(candidate, ("cui", "concept_id", "code"))
        score = _value(candidate, ("score", "similarity", "confidence"))

    if cui is _MISSING or not str(cui).strip():
        raise ValueError("scispaCy candidate is missing a non-empty CUI")
    if score is _MISSING:
        raise ValueError(f"scispaCy candidate {cui!r} is missing a score")
    confidence = float(score)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"scispaCy candidate score must be between 0 and 1: {score!r}")
    return {"system": UMLS_SYSTEM, "code": str(cui), "score": confidence}


def _value(record: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(record, Mapping):
        for key in keys:
            if key in record:
                return record[key]
        return _MISSING
    for key in keys:
        value = getattr(record, key, _MISSING)
        if value is not _MISSING:
            return value
    return _MISSING


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _resource_message() -> str:
    return (
        "scispaCy UMLS linker candidates are unavailable. Configure a pipeline "
        "with 'scispacy_linker' backed by user-supplied licensed UMLS resources; "
        "OpenMed does not bundle or download UMLS data."
    )


__all__ = [
    "ScispaCyLinkerResourceError",
    "UMLS_SYSTEM",
    "link_to_canonical",
    "to_canonical",
]
