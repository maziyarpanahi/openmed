"""Load data-driven African healthcare-context safety-sweep configuration."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DATA_FILE = Path(__file__).with_name("data") / "africa_context_terms.json"
DATA_SOURCE = "openmed/core/data/africa_context_terms.json"
SUPPORTED_SCHEMA_VERSION = 1

_TERM_PLACEHOLDER_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")


@lru_cache(maxsize=1)
def load_africa_context_data() -> Mapping[str, Any]:
    """Return the validated bundled African context configuration."""

    with DATA_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(f"{DATA_SOURCE} must contain a JSON object")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{DATA_SOURCE} schema_version must be {SUPPORTED_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("term_sets"), Mapping):
        raise ValueError(f"{DATA_SOURCE} term_sets must be an object")
    if not isinstance(payload.get("patterns"), list):
        raise ValueError(f"{DATA_SOURCE} patterns must be a list")
    if not isinstance(payload.get("african_profile_defaults"), Mapping):
        raise ValueError(f"{DATA_SOURCE} african_profile_defaults must be an object")
    return payload


def rendered_pattern_entries() -> tuple[Mapping[str, Any], ...]:
    """Render configured term-set placeholders into regex pattern entries."""

    payload = load_africa_context_data()
    term_sets = payload["term_sets"]
    if not isinstance(term_sets, Mapping):
        raise TypeError(f"term_sets must be a Mapping, got {type(term_sets).__name__}")

    entries: list[Mapping[str, Any]] = []
    for index, raw_entry in enumerate(payload["patterns"]):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{DATA_SOURCE} patterns[{index}] must be an object")

        template = _required_string(raw_entry, "template", index=index)
        entity_type = _required_string(raw_entry, "entity_type", index=index)
        rendered = template
        for term_set_name in _TERM_PLACEHOLDER_RE.findall(template):
            if term_set_name not in term_sets:
                raise ValueError(
                    f"{DATA_SOURCE} patterns[{index}] references unknown term set "
                    f"{term_set_name!r}"
                )
            terms = _flatten_terms(term_sets[term_set_name], term_set_name)
            alternation = "|".join(
                re.escape(term)
                for term in sorted(terms, key=lambda item: (-len(item), item))
            )
            rendered = rendered.replace(f"{{{term_set_name}}}", alternation)

        context_words: tuple[str, ...] = ()
        context_term_set = raw_entry.get("context_term_set")
        if context_term_set is not None:
            context_term_set = str(context_term_set)
            if context_term_set not in term_sets:
                raise ValueError(
                    f"{DATA_SOURCE} patterns[{index}] references unknown context term "
                    f"set {context_term_set!r}"
                )
            context_words = _flatten_terms(
                term_sets[context_term_set],
                context_term_set,
            )

        entries.append(
            {
                "pattern": rendered,
                "entity_type": entity_type,
                "priority": int(raw_entry.get("priority", 0)),
                "flags": int(raw_entry.get("flags", re.IGNORECASE)),
                "base_score": float(raw_entry.get("base_score", 0.5)),
                "context_boost": float(raw_entry.get("context_boost", 0.35)),
                "context_words": context_words,
                "requires_context": bool(raw_entry.get("requires_context", False)),
                "safety_sweep_requires_context": bool(
                    raw_entry.get("safety_sweep_requires_context", False)
                ),
            }
        )
    return tuple(entries)


def profile_defaults_for(profile_name: str) -> Mapping[str, Mapping[str, str]]:
    """Return configured context-action defaults for an African profile."""

    defaults = load_africa_context_data()["african_profile_defaults"]
    if not isinstance(defaults, Mapping):
        raise TypeError(f"african_profile_defaults must be a Mapping, got {type(defaults).__name__}")
    value = defaults.get(str(profile_name), {})
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{DATA_SOURCE} profile defaults for {profile_name!r} must be an object"
        )

    normalized: dict[str, Mapping[str, str]] = {}
    for key in ("policy_label_actions", "actions"):
        raw_actions = value.get(key, {})
        if not isinstance(raw_actions, Mapping):
            raise ValueError(
                f"{DATA_SOURCE} profile defaults {profile_name!r}.{key} "
                "must be an object"
            )
        normalized[key] = {
            str(label): str(action) for label, action in raw_actions.items()
        }
    return normalized


def _required_string(
    entry: Mapping[str, Any],
    key: str,
    *,
    index: int,
) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{DATA_SOURCE} patterns[{index}].{key} must be a non-empty string"
        )
    return value


def _flatten_terms(value: Any, term_set_name: str) -> tuple[str, ...]:
    terms: list[str] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            terms.extend(_flatten_terms(nested, term_set_name))
    elif isinstance(value, list):
        for term in value:
            if not isinstance(term, str) or not term.strip():
                raise ValueError(
                    f"{DATA_SOURCE} term set {term_set_name!r} must contain "
                    "non-empty strings"
                )
            terms.append(term.strip())
    else:
        raise ValueError(
            f"{DATA_SOURCE} term set {term_set_name!r} must be a list or object"
        )
    return tuple(sorted(set(terms)))


__all__ = [
    "DATA_FILE",
    "DATA_SOURCE",
    "load_africa_context_data",
    "profile_defaults_for",
    "rendered_pattern_entries",
]
