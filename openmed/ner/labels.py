"""Domain default label management for zero-shot NER."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


_RESOURCE_PACKAGE = "label_maps"
_DEFAULT_RESOURCE = "defaults.json"
_GENERIC_DOMAIN = "generic"


def load_default_label_map(overrides_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load the curated default label map.

    Args:
        overrides_path: Optional filesystem path that, when provided, overrides
            the packaged defaults. Primarily intended for testing.

    Returns:
        Mapping of normalised domain identifiers to label lists.
    """

    if overrides_path is not None:
        data = _load_from_path(overrides_path)
    else:
        data = _load_from_resource()
    return _normalise_label_map(data)


@lru_cache()
def _load_from_resource() -> Mapping[str, Iterable[str]]:
    with resources.files(_RESOURCE_PACKAGE).joinpath(_DEFAULT_RESOURCE).open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def _load_from_path(path: Path) -> Mapping[str, Iterable[str]]:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def _normalise_label_map(
    raw: Mapping[str, Iterable[str]]
) -> Dict[str, List[str]]:
    normalised: Dict[str, List[str]] = {}
    for domain, labels in raw.items():
        norm_domain = _normalise_domain(domain)
        if not norm_domain:
            continue
        norm_labels = _deduplicate_labels(labels)
        normalised[norm_domain] = norm_labels
    return normalised


def _normalise_domain(domain: str) -> str:
    return domain.strip().lower().replace(" ", "_")


def _deduplicate_labels(labels: Iterable[str]) -> List[str]:
    seen: Dict[str, None] = {}
    cleaned: List[str] = []
    for label in labels:
        label_str = str(label).strip()
        if not label_str:
            continue
        if label_str.lower() in seen:
            continue
        seen[label_str.lower()] = None
        cleaned.append(label_str)
    return cleaned


def available_domains(label_map: Optional[Mapping[str, Iterable[str]]] = None) -> List[str]:
    mapping = label_map or load_default_label_map()
    return sorted(mapping.keys())


def get_default_labels(
    domain: Optional[str],
    *,
    label_map: Optional[Mapping[str, Iterable[str]]] = None,
    inherit_generic: bool = True,
) -> List[str]:
    mapping = label_map or load_default_label_map()

    key = _normalise_domain(domain) if domain else None
    if key and key in mapping:
        labels = mapping[key]
    elif inherit_generic and _GENERIC_DOMAIN in mapping:
        labels = mapping[_GENERIC_DOMAIN]
    else:
        labels = []
    return list(labels)


def reload_default_label_map() -> Dict[str, List[str]]:
    """Clear caches and return freshly loaded defaults."""

    _load_from_resource.cache_clear()  # type: ignore[attr-defined]
    return load_default_label_map()


__all__ = [
    "load_default_label_map",
    "reload_default_label_map",
    "get_default_labels",
    "available_domains",
]
