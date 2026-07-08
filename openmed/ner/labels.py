"""Domain default label management for zero-shot NER."""

from __future__ import annotations

import json
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
_CLINICAL_DOMAINS_DOC = Path("docs/clinical-domains.md")
_NO_FIXTURE = "Not shipped"
_CLINICAL_DOMAINS_DISCLAIMER = (
    "This catalog is generated from the packaged zero-shot domain label map "
    "and canonical label metadata. Labels, policy classes, risk levels, coding "
    "system hints, and fixture pointers are for offline evaluation and review "
    "workflow planning only; they are not clinical guidance and must not be "
    "used to infer diagnosis, treatment, coding, or coverage decisions."
)
_DOMAIN_FIXTURE_PATHS: Mapping[str, str] = {
    "anesthesia": "tests/fixtures/clinical/anesthesia.jsonl",
    "endocrinology": "tests/fixtures/clinical/endocrinology.jsonl",
    "gastroenterology": "tests/fixtures/clinical/gastroenterology.jsonl",
    "genomic_variant": "tests/fixtures/clinical/genomic_variant.jsonl",
    "nephrology_renal": "tests/fixtures/clinical/nephrology_renal.jsonl",
    "nutrition_diet": "tests/fixtures/clinical/nutrition_diet.jsonl",
}


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
    """Generate the clinical domain catalog markdown fully offline."""

    sections = [
        "# Clinical Domain Label Catalog",
        "",
        f"**Disclaimer:** {_CLINICAL_DOMAINS_DISCLAIMER}",
        "",
    ]
    for domain, labels in load_default_label_map().items():
        sections.extend(
            [
                f"## {_format_domain_name(domain)}",
                "",
                (
                    "| Label | Canonical Label | Category | Risk Level | "
                    "System Hints | Fixture Path |"
                ),
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        fixture_path = _DOMAIN_FIXTURE_PATHS.get(domain, _NO_FIXTURE)
        for label in labels:
            canonical_label = normalize_label(label)
            system_hints = system_hints_for(canonical_label)
            sections.append(
                "| "
                f"{_markdown_cell(label)} | "
                f"{_markdown_cell(canonical_label)} | "
                f"{policy_label_for(canonical_label)} | "
                f"{risk_level_for(canonical_label)} | "
                f"{_markdown_cell(', '.join(system_hints) if system_hints else 'None')} | "
                f"{_markdown_cell(fixture_path)} |"
            )
        sections.append("")
    return "\n".join(sections)


def write_clinical_domains_markdown(
    output_path: Path | str = _CLINICAL_DOMAINS_DOC,
) -> Path:
    """Write the generated clinical domain catalog and return its path."""

    path = Path(output_path)
    path.write_text(generate_clinical_domains_markdown(), encoding="utf-8")
    return path


def _format_domain_name(domain: str) -> str:
    return domain.replace("_", " ").title()


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


__all__ = [
    "load_default_label_map",
    "reload_default_label_map",
    "get_default_labels",
    "available_domains",
    "generate_clinical_domains_markdown",
    "write_clinical_domains_markdown",
]

if __name__ == "__main__":
    write_clinical_domains_markdown()
