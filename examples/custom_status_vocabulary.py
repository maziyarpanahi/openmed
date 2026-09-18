#!/usr/bin/env python3
"""Extend the SDOH status vocabulary with a synthetic local domain.

``openmed.clinical.status_vocab`` ships bundled cue tables for three SDOH
domains (substance use, employment, living situation) plus a
``load_status_vocab(path)`` entry point for loading and validating a vocabulary
from an explicit local path. This example walks through adding a fourth,
synthetic domain ("mobility") without bundling it into the package, and shows
the fail-closed behavior a contributor should expect from an invalid table:
missing advisory provenance, and a cue accidentally listed under two statuses.

Everything here runs offline against a vocabulary defined in this file. No
network access, bundled terminology change, or clinical decision is involved;
see ``openmed.clinical.status_vocab.STATUS_NORMALIZATION_ADVISORY``.
"""

from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openmed.clinical import load_status_vocab

_MOBILITY_DOMAIN = "mobility"

EXAMPLE_VOCABULARY_YAML = """schema_version: 1
provenance:
  source: Synthetic example vocabulary for the local-extension walkthrough.
  license: Apache-2.0 repository asset.
  disclaimer: Advisory normalization only; not a clinical decision rule or a substitute for review.
defaults:
  unknown_status: unknown
vocabularies:
  mobility:
    priority: [never, former, assisted, ambulatory]
    current_statuses: [assisted, ambulatory]
    axis_overrides:
      negated: never
      historical_current: former
    statuses:
      ambulatory:
        cues: ["walks independently", "ambulates without assistance", "no mobility aid"]
      assisted:
        cues: ["uses a cane", "uses a walker", "uses a wheelchair", "requires assistance walking"]
      former:
        cues: ["formerly used a wheelchair", "no longer uses a cane", "previously required assistance walking"]
      never:
        cues: ["denies mobility limitation", "never used a mobility aid"]
"""


def write_example_vocabulary(directory: Path) -> Path:
    """Write the synthetic vocabulary to an explicit local path and return it."""
    path = directory / "custom_status_vocab.yaml"
    path.write_text(EXAMPLE_VOCABULARY_YAML, encoding="utf-8")
    return path


def find_duplicate_cues(vocabulary: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Return cues that collide, after normalization, across statuses.

    ``load_status_vocab`` checks structure and provenance but does not check
    whether the same cue was listed under two statuses by mistake; matching
    resolves ties by priority order, so a duplicate silently favors whichever
    status is checked first. Contributors extending a vocabulary locally
    should run this check before trusting normalization output.
    """
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for status, entry in vocabulary["statuses"].items():
        for cue in entry["cues"]:
            normalized = _normalize_phrase(cue)
            existing = seen.setdefault(normalized, status)
            if existing != status:
                collisions.append((cue, existing, status))
    return collisions


def validate_no_duplicate_cues(vocabulary: Mapping[str, Any], *, domain: str) -> None:
    """Raise ``ValueError`` if any cue in ``vocabulary`` collides across statuses."""
    collisions = find_duplicate_cues(vocabulary)
    if collisions:
        cue, first_status, second_status = collisions[0]
        raise ValueError(
            f"{domain} vocabulary lists {cue!r} under both "
            f"{first_status!r} and {second_status!r}"
        )


def normalize_mobility_status(
    phrase: object,
    vocabulary: Mapping[str, Any],
    *,
    negated: bool = False,
    temporality: str | None = None,
) -> str:
    """Normalize a mobility-status phrase against a loaded local vocabulary.

    ``status_vocab`` ships ``normalize_substance_status``,
    ``normalize_employment_status``, and ``normalize_living_status`` for its
    three bundled domains, and each always loads the packaged vocabulary. A
    locally extended domain such as this one has no packaged helper, so this
    function re-implements the same deterministic, priority-ordered
    substring match documented on ``openmed.clinical.status_vocab`` against
    the vocabulary payload returned by ``load_status_vocab(path)``.
    """
    overrides = vocabulary["axis_overrides"]
    if negated:
        return str(overrides["negated"])

    text = _normalize_phrase(phrase)
    status = "unknown"
    if text:
        for candidate in vocabulary["priority"]:
            cues = vocabulary["statuses"][candidate]["cues"]
            if any(_cue_pattern(_normalize_phrase(cue)).search(text) for cue in cues):
                status = candidate
                break

    if temporality == "historical" and status in vocabulary["current_statuses"]:
        return str(overrides["historical_current"])
    return status


def _normalize_phrase(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _cue_pattern(cue: str) -> re.Pattern[str]:
    escaped = re.escape(cue).replace(r"\ ", r"\s+")
    suffix = "" if cue.endswith("-") else r"(?!\w)"
    return re.compile(r"(?<!\w)" + escaped + suffix)


def broken_provenance_yaml() -> str:
    """Return the example vocabulary with its advisory disclaimer stripped."""
    return EXAMPLE_VOCABULARY_YAML.replace(
        "disclaimer: Advisory normalization only; not a clinical decision rule"
        " or a substitute for review.",
        "disclaimer: Local mobility-status cue table.",
    )


def broken_duplicate_cue_yaml() -> str:
    """Return the example vocabulary with one cue duplicated across statuses."""
    return EXAMPLE_VOCABULARY_YAML.replace(
        'cues: ["denies mobility limitation", "never used a mobility aid"]',
        'cues: ["denies mobility limitation", "never used a mobility aid", "uses a cane"]',
    )


def main() -> dict[str, Any]:
    """Run the local-extension walkthrough and return a JSON-serializable summary."""
    with tempfile.TemporaryDirectory() as workdir:
        directory = Path(workdir)
        mobility = load_status_vocab(write_example_vocabulary(directory))[
            "vocabularies"
        ][_MOBILITY_DOMAIN]
        validate_no_duplicate_cues(mobility, domain=_MOBILITY_DOMAIN)

        normalized = {
            "uses a cane": normalize_mobility_status("uses a cane", mobility),
            "walks independently": normalize_mobility_status(
                "walks independently", mobility
            ),
            "formerly used a wheelchair": normalize_mobility_status(
                "formerly used a wheelchair", mobility
            ),
            "walks independently (historical)": normalize_mobility_status(
                "walks independently", mobility, temporality="historical"
            ),
            "uses a cane (negated)": normalize_mobility_status(
                "uses a cane", mobility, negated=True
            ),
            "chart silent on mobility": normalize_mobility_status(
                "chart silent on mobility", mobility
            ),
        }

        invalid_provenance_path = directory / "invalid_provenance.yaml"
        invalid_provenance_path.write_text(broken_provenance_yaml(), encoding="utf-8")
        try:
            load_status_vocab(invalid_provenance_path)
        except ValueError as error:
            invalid_provenance_rejected = str(error)
        else:
            raise AssertionError(
                "expected load_status_vocab to reject a missing disclaimer"
            )

        duplicate_cue_path = directory / "duplicate_cues.yaml"
        duplicate_cue_path.write_text(broken_duplicate_cue_yaml(), encoding="utf-8")
        duplicate_mobility = load_status_vocab(duplicate_cue_path)["vocabularies"][
            _MOBILITY_DOMAIN
        ]
        try:
            validate_no_duplicate_cues(duplicate_mobility, domain=_MOBILITY_DOMAIN)
        except ValueError as error:
            duplicate_cue_rejected = str(error)
        else:
            raise AssertionError(
                "expected validate_no_duplicate_cues to reject a duplicate cue"
            )

    summary = {
        "normalized": normalized,
        "invalid_provenance_rejected": invalid_provenance_rejected,
        "duplicate_cue_rejected": duplicate_cue_rejected,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
