"""Canonical adapter for QuickUMLS matcher output.

QuickUMLS data must be created from a UMLS installation licensed and supplied
by the caller. OpenMed ships no UMLS vocabulary files and performs no resource
download from this adapter.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from pathlib import Path
from typing import Any, Iterable, Mapping

UMLS_SYSTEM = "UMLS"
_MISSING = object()
_REQUIRED_RESOURCE_NAMES = ("umls-simstring.db", "cui-semtypes.db")


class QuickUMLSResourceError(RuntimeError):
    """Raised when a user-supplied QuickUMLS resource is unavailable."""


def to_canonical(matches: Any) -> list[dict[str, Any]]:
    """Map QuickUMLS match dictionaries to canonical span dictionaries.

    Both the matcher's nested candidate-group output and flat mapping-based
    stubs are accepted. Candidates sharing offsets become one span with a
    ``codes`` list using the same UMLS fields as the scispaCy adapter.
    """

    canonical: list[dict[str, Any]] = []
    for group in _match_groups(matches):
        if not group:
            continue
        first = group[0]
        start = _required_value(first, ("start", "start_char"), "start")
        end = _required_value(first, ("end", "end_char"), "end")
        text = _value(first, ("ngram", "text", "term"))
        output: dict[str, Any] = {
            "start": int(start),
            "end": int(end),
            "codes": _canonical_codes(group),
        }
        if text is not _MISSING:
            output["text"] = str(text)
        canonical.append(output)
    return canonical


def match_to_canonical(
    text: str,
    *,
    matcher: Any | None = None,
    resource_path: str | Path | None = None,
    best_match: bool = True,
    ignore_syntax: bool = False,
    **matcher_options: Any,
) -> list[dict[str, Any]]:
    """Run QuickUMLS with a caller matcher or a user-supplied resource path.

    Passing ``matcher`` avoids adapter-side construction. Otherwise
    ``resource_path`` must point to QuickUMLS data prepared from a licensed
    UMLS installation; the adapter never discovers or downloads such data.
    """

    matcher_ = matcher
    if matcher_ is None:
        resource = _validated_resource_path(resource_path)
        try:
            module = _import_module("quickumls")
        except ImportError as exc:
            raise ImportError(
                "QuickUMLS support requires the optional dependency. "
                "Install with `pip install openmed[quickumls]`."
            ) from exc
        matcher_ = module.QuickUMLS(str(resource), **matcher_options)
    elif matcher_options:
        raise TypeError("matcher_options are only valid when constructing a matcher")

    match = getattr(matcher_, "match", None)
    if not callable(match):
        raise TypeError("matcher must provide a callable match(text, ...) method")
    return to_canonical(match(text, best_match=best_match, ignore_syntax=ignore_syntax))


def _validated_resource_path(resource_path: str | Path | None) -> Path:
    if resource_path is None:
        raise QuickUMLSResourceError(_resource_message())
    resource = Path(resource_path).expanduser()
    missing = [
        name for name in _REQUIRED_RESOURCE_NAMES if not (resource / name).exists()
    ]
    if not resource.is_dir() or missing:
        detail = (
            f" Missing expected resources: {', '.join(missing)}." if missing else ""
        )
        raise QuickUMLSResourceError(f"{_resource_message()}{detail}")
    return resource


def _match_groups(matches: Any) -> list[list[Any]]:
    if matches is None:
        return []
    if isinstance(matches, Mapping):
        return [[matches]]
    items = _as_list(matches)
    if not items:
        return []
    if all(_looks_like_match(item) for item in items):
        return _group_flat_matches(items)
    return [_as_list(item) for item in items]


def _group_flat_matches(matches: list[Any]) -> list[list[Any]]:
    grouped: dict[tuple[int, int, str], list[Any]] = {}
    for match in matches:
        start = int(_required_value(match, ("start", "start_char"), "start"))
        end = int(_required_value(match, ("end", "end_char"), "end"))
        text = _value(match, ("ngram", "text", "term"))
        key = (start, end, "" if text is _MISSING else str(text))
        grouped.setdefault(key, []).append(match)
    return list(grouped.values())


def _canonical_codes(group: list[Any]) -> list[dict[str, Any]]:
    codes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in group:
        cui = _required_value(match, ("cui", "concept_id", "code"), "CUI")
        code = str(cui).strip()
        if not code:
            raise ValueError("QuickUMLS match is missing a non-empty CUI")
        if code in seen:
            continue
        score = _required_value(
            match, ("similarity", "score", "confidence"), "similarity"
        )
        confidence = float(score)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"QuickUMLS similarity must be between 0 and 1: {score!r}")
        codes.append({"system": UMLS_SYSTEM, "code": code, "score": confidence})
        seen.add(code)
    return codes


def _looks_like_match(value: Any) -> bool:
    return _value(value, ("cui", "concept_id", "code")) is not _MISSING


def _required_value(record: Any, keys: tuple[str, ...], label: str) -> Any:
    value = _value(record, keys)
    if value is _MISSING:
        raise ValueError(f"QuickUMLS match is missing {label}")
    return value


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
        "QuickUMLS requires a user-supplied QuickUMLS data directory prepared "
        "from a licensed UMLS installation. Pass resource_path or a configured "
        "matcher; OpenMed does not bundle or download UMLS data."
    )


__all__ = [
    "QuickUMLSResourceError",
    "UMLS_SYSTEM",
    "match_to_canonical",
    "to_canonical",
]
