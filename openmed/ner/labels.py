"""Domain default label management for zero-shot NER."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from openmed.core.labels import (
    normalize_label,
    policy_label_for,
    risk_level_for,
    system_hints_for,
)

_RESOURCE_PACKAGE = "openmed.zero_shot.data.label_maps"
_DEFAULT_RESOURCE = "defaults.json"
_GENERIC_DOMAIN = "generic"


def load_default_label_map(
    overrides_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
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
    with (
        resources.files(_RESOURCE_PACKAGE)
        .joinpath(_DEFAULT_RESOURCE)
        .open("r", encoding="utf-8") as handle
    ):
        return json.load(handle)


def _load_from_path(path: Path) -> Mapping[str, Iterable[str]]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in label file {path}: {exc}") from exc


def _normalise_label_map(raw: Mapping[str, Iterable[str]]) -> Dict[str, List[str]]:
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


def available_domains(
    label_map: Optional[Mapping[str, Iterable[str]]] = None,
) -> List[str]:
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


def generate_clinical_domains_markdown() -> str:
    """Offline, generate the clinical domains markdown documentation string."""
    disclaimer = (
        "Deterministic status vocabulary helpers for SDOH cue normalization. "
        "The helpers in this module normalize explicit status cues into compact "
        "canonical values for downstream SDOH extractors. Outputs are advisory "
        "labels for review and downstream processing, not clinical decisions. "
        "They deliberately do not infer a status from absence of mention; text "
        "with no configured cue returns 'unknown' unless an explicit context "
        "axis is supplied by the caller."
    )

    final_markdown = f"**Disclaimer:** {disclaimer}\n\n"

    default_label_map = load_default_label_map()

    # Create a markdown string for each domain and its labels
    for domain, labels in default_label_map.items():
        table_header = f"### {domain.replace('_', ' ').title()}\n\n"
        table_header += (
            "| Label | Category | Risk Level | System Hints | Fixture Path |\n"
        )
        table_header += (
            "|-------|----------|------------|--------------|--------------|\n"
        )

        domain_markdown = table_header

        for label in labels:
            normalized_label = normalize_label(label)
            category = policy_label_for(normalized_label)
            risk_level = risk_level_for(normalized_label)
            system_hints = system_hints_for(normalized_label)
            system_hints_str = ", ".join(system_hints) if system_hints else "None"
            fixture_path = "tests/fixtures/clinical/context_traps.jsonl"

            domain_markdown += f"| {label} | {category} | {risk_level} | {system_hints_str} | {fixture_path} |\n"

        final_markdown += domain_markdown + "\n"

    return final_markdown


__all__ = [
    "load_default_label_map",
    "reload_default_label_map",
    "get_default_labels",
    "available_domains",
    "generate_clinical_domains_markdown",
]

if __name__ == "__main__":
    # Generate the clinical domains markdown and save to docs
    markdown_output = generate_clinical_domains_markdown()
    output_path = os.path.join("docs", "clinical-domains.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)
