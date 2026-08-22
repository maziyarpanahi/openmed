"""Eval-only loader for the n2c2 2018 Track 2 relation corpus.

The n2c2 2018 Track 2 corpus is distributed as paired BRAT text and
standoff files under the n2c2/DBMI data-use agreement.  This module exposes a
dedicated medication/ADE API while keeping the relation representation shared
with the existing DrugProt relation view.  No corpus rows are bundled or
downloaded.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from openmed.core.labels import (
    CANONICAL_LABELS,
    CONDITION,
    DOSAGE,
    DURATION,
    FORM,
    FREQUENCY,
    INDICATION,
    MEDICATION,
    ROUTE,
    STRENGTH,
)
from openmed.eval.datasets.drugprot import DrugProtRelationFixture
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.datasets.i2b2 import (
    N2C2_2018_PATH_ENV as N2C2_2018_GENERIC_PATH_ENV,
)
from openmed.eval.datasets.i2b2 import (
    load_n2c2_2018_relation_fixtures,
)
from openmed.eval.datasets.licenses import license_for
from openmed.eval.harness import BenchmarkFixture

N2C2_2018_TRACK2 = "n2c2-2018-track2"
N2C2_TRACK2 = N2C2_2018_TRACK2
N2C2_2018_TRACK2_DUA_NAME = "n2c2/DBMI 2018 Track 2 DUA"
N2C2_2018_TRACK2_PATH_ENV = "OPENMED_N2C2_2018_TRACK2_PATH"
N2C2_TRACK2_PATH_ENV = N2C2_2018_TRACK2_PATH_ENV

N2C2_2018_TRACK2_ENTITY_TYPES: tuple[str, ...] = (
    "Drug",
    "Strength",
    "Route",
    "Form",
    "ADE",
    "Dosage",
    "Reason",
    "Frequency",
    "Duration",
)
N2C2_2018_TRACK2_ENTITY_TO_CANONICAL: Mapping[str, str] = {
    "Drug": MEDICATION,
    "Strength": STRENGTH,
    "Route": ROUTE,
    "Form": FORM,
    "ADE": CONDITION,
    "Dosage": DOSAGE,
    "Reason": INDICATION,
    "Frequency": FREQUENCY,
    "Duration": DURATION,
}

N2C2_2018_TRACK2_RELATION_TYPES: tuple[str, ...] = (
    "Frequency-Drug",
    "Strength-Drug",
    "Route-Drug",
    "Dosage-Drug",
    "ADE-Drug",
    "Reason-Drug",
    "Duration-Drug",
    "Form-Drug",
)
N2C2_2018_TRACK2_RELATION_TO_CANONICAL: Mapping[str, str] = {
    "Frequency-Drug": "DRUG_TO_FREQUENCY",
    "Strength-Drug": "DRUG_TO_STRENGTH",
    "Route-Drug": "DRUG_TO_ROUTE",
    "Dosage-Drug": "DRUG_TO_DOSE",
    "ADE-Drug": "DRUG_TO_ADE",
    "Reason-Drug": "DRUG_TO_INDICATION",
    "Duration-Drug": "DRUG_TO_DURATION",
    "Form-Drug": "DRUG_TO_FORM",
}

N2C2_2018_TRACK2_SUITE_METADATA: Mapping[str, Any] = {
    "access": (
        "credentialed local path only; pass path=... or set "
        f"{N2C2_2018_TRACK2_PATH_ENV} (or {N2C2_2018_GENERIC_PATH_ENV})"
    ),
    "annotation_format": "BRAT standoff (.txt + .ann)",
    "cache_corpus_rows": False,
    "dataset": N2C2_2018_TRACK2,
    "dua": N2C2_2018_TRACK2_DUA_NAME,
    "entity_label_mapping": dict(sorted(N2C2_2018_TRACK2_ENTITY_TO_CANONICAL.items())),
    "eval_only": True,
    "license": license_for("n2c2-2018").to_dict(),
    "network_fetch": False,
    "redistribution": "credentialed eval-only; never redistributed",
    "relation_type_mapping": dict(
        sorted(N2C2_2018_TRACK2_RELATION_TO_CANONICAL.items())
    ),
    "suite": N2C2_2018_TRACK2,
    "task": "relation",
    "track": 2,
    "year": 2018,
}

_REPO_ROOT = Path(__file__).resolve().parents[3]


class N2C2Track2CredentialRequired(DUACredentialRequired):
    """Raised when the n2c2 Track 2 DUA path is not configured."""


def map_n2c2_2018_track2_entity_label(label: str) -> str:
    """Map one n2c2 Track 2 entity type to a canonical label."""

    key = _mapping_key(label)
    aliases = {
        "drugname": "Drug",
        "medication": "Drug",
        "adversedrugevent": "ADE",
    }
    source_label = aliases.get(key, label)
    canonical = _lookup_mapping(
        source_label,
        N2C2_2018_TRACK2_ENTITY_TO_CANONICAL,
        kind="entity label",
    )
    _ensure_canonical(canonical, source_label)
    return canonical


def map_n2c2_2018_track2_relation_type(relation_type: str) -> str:
    """Map one n2c2 Track 2 relation type to the shared relation schema."""

    canonical = _lookup_mapping(
        relation_type,
        N2C2_2018_TRACK2_RELATION_TO_CANONICAL,
        kind="relation type",
    )
    if not canonical:
        raise RuntimeError(f"n2c2 Track 2 relation mapping is empty: {relation_type!r}")
    return canonical


def n2c2_2018_track2_suite_metadata() -> dict[str, Any]:
    """Return row-free metadata for the n2c2 Track 2 suite."""

    return dict(N2C2_2018_TRACK2_SUITE_METADATA)


def load_n2c2_2018_track2_relation_fixtures(
    path: str | Path | None = None,
) -> list[DrugProtRelationFixture]:
    """Load n2c2 Track 2 relation fixtures from an approved local path.

    The parser accepts the paired ``.txt``/``.ann`` BRAT files used by the
    challenge.  Source text and annotations remain in memory for evaluation;
    they are never copied into the repository or a cache.
    """

    source = _credentialed_path(path)
    fixtures = load_n2c2_2018_relation_fixtures(source)
    return [_retag_relation_fixture(fixture) for fixture in fixtures]


def load_n2c2_2018_track2_ner_fixtures(
    path: str | Path | None = None,
) -> list[BenchmarkFixture]:
    """Load the n2c2 Track 2 entity view as benchmark NER fixtures."""

    relation_fixtures = load_n2c2_2018_track2_relation_fixtures(path)
    return [_ner_fixture(fixture) for fixture in relation_fixtures]


def load_n2c2_2018_track2_fixtures(
    path: str | Path | None = None,
    *,
    task: str = "relation",
) -> list[BenchmarkFixture | DrugProtRelationFixture]:
    """Load the requested n2c2 Track 2 NER or relation view."""

    normalized_task = str(task).strip().casefold().replace("-", "_")
    if normalized_task in {"ner", "entity", "entities"}:
        return load_n2c2_2018_track2_ner_fixtures(path)
    if normalized_task in {"relation", "relations", "re"}:
        return load_n2c2_2018_track2_relation_fixtures(path)
    raise ValueError("n2c2 Track 2 task must be 'ner' or 'relation'")


def load_n2c2_2018_track2(
    path: str | Path | None = None,
    *,
    task: str = "relation",
) -> list[BenchmarkFixture | DrugProtRelationFixture]:
    """Load the requested n2c2 Track 2 view.

    This short name is kept as the natural entry point for callers focused on
    the medication/ADE relation number; use
    :func:`load_n2c2_2018_track2_ner_fixtures` for the NER view.
    """

    return load_n2c2_2018_track2_fixtures(path, task=task)


load_n2c2_track2 = load_n2c2_2018_track2
load_n2c2_track2_ner_fixtures = load_n2c2_2018_track2_ner_fixtures
load_n2c2_track2_relation_fixtures = load_n2c2_2018_track2_relation_fixtures


def _credentialed_path(path: str | Path | None) -> Path:
    raw_path = path
    if raw_path is None:
        raw_path = os.environ.get(N2C2_2018_TRACK2_PATH_ENV) or os.environ.get(
            N2C2_2018_GENERIC_PATH_ENV
        )
    if raw_path is None or not str(raw_path).strip():
        raise N2C2Track2CredentialRequired(
            f"{N2C2_2018_TRACK2_DUA_NAME} credentialed local path is required; "
            f"pass path=... or set {N2C2_2018_TRACK2_PATH_ENV} "
            f"(or {N2C2_2018_GENERIC_PATH_ENV}). "
            "No corpus rows were loaded."
        )
    candidate = Path(raw_path).expanduser().resolve(strict=False)
    if _is_relative_to(candidate, _REPO_ROOT):
        raise N2C2Track2CredentialRequired(
            f"{N2C2_2018_TRACK2_DUA_NAME} data must stay outside the repository "
            f"tree; refusing to read {candidate}. No corpus rows were loaded."
        )
    if not candidate.exists():
        raise N2C2Track2CredentialRequired(
            f"{N2C2_2018_TRACK2_DUA_NAME} credentialed path does not exist: "
            f"{candidate}. No corpus rows were loaded."
        )
    if not candidate.is_file() and not candidate.is_dir():
        raise N2C2Track2CredentialRequired(
            f"{N2C2_2018_TRACK2_DUA_NAME} path must be a file or directory: "
            f"{candidate}. No corpus rows were loaded."
        )
    return candidate


def _retag_relation_fixture(
    fixture: DrugProtRelationFixture,
) -> DrugProtRelationFixture:
    metadata = {
        **dict(fixture.metadata or {}),
        **N2C2_2018_TRACK2_SUITE_METADATA,
        "source_path_hash": dict(fixture.metadata or {}).get("source_path_hash"),
        "task": "relation",
    }
    if metadata["source_path_hash"] is None:
        metadata.pop("source_path_hash")
    return replace(fixture, metadata=metadata)


def _ner_fixture(fixture: DrugProtRelationFixture) -> BenchmarkFixture:
    metadata = {
        **dict(fixture.metadata or {}),
        "relation_count": len(fixture.relations),
        "task": "ner",
    }
    return BenchmarkFixture(
        fixture_id=fixture.fixture_id,
        text=fixture.text,
        gold_spans=tuple(
            entity.to_eval_span(fixture.text) for entity in fixture.entities
        ),
        language=fixture.language,
        metadata=metadata,
    )


def _lookup_mapping(
    value: str,
    mapping: Mapping[str, str],
    *,
    kind: str,
) -> str:
    key = _mapping_key(value)
    for source, canonical in mapping.items():
        if _mapping_key(source) == key:
            return canonical
    allowed = ", ".join(mapping)
    raise ValueError(
        f"unknown n2c2 Track 2 {kind} {value!r}; expected one of: {allowed}"
    )


def _ensure_canonical(canonical: str, source_label: str) -> None:
    if canonical not in CANONICAL_LABELS:
        raise RuntimeError(
            f"n2c2 Track 2 mapping for {source_label!r} is not canonical: {canonical!r}"
        )


def _mapping_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


_missing_entity_mappings = sorted(
    set(N2C2_2018_TRACK2_ENTITY_TYPES) - set(N2C2_2018_TRACK2_ENTITY_TO_CANONICAL)
)
_extra_entity_mappings = sorted(
    set(N2C2_2018_TRACK2_ENTITY_TO_CANONICAL) - set(N2C2_2018_TRACK2_ENTITY_TYPES)
)
_missing_relation_mappings = sorted(
    set(N2C2_2018_TRACK2_RELATION_TYPES) - set(N2C2_2018_TRACK2_RELATION_TO_CANONICAL)
)
_extra_relation_mappings = sorted(
    set(N2C2_2018_TRACK2_RELATION_TO_CANONICAL) - set(N2C2_2018_TRACK2_RELATION_TYPES)
)
if (
    _missing_entity_mappings
    or _extra_entity_mappings
    or _missing_relation_mappings
    or _extra_relation_mappings
):
    raise RuntimeError(
        "n2c2 Track 2 mappings must cover the source tables exactly; "
        f"missing_entities={_missing_entity_mappings}, "
        f"extra_entities={_extra_entity_mappings}, "
        f"missing_relations={_missing_relation_mappings}, "
        f"extra_relations={_extra_relation_mappings}"
    )
for _source_label, _canonical_label in N2C2_2018_TRACK2_ENTITY_TO_CANONICAL.items():
    _ensure_canonical(_canonical_label, _source_label)


__all__ = [
    "N2C2_2018_TRACK2",
    "N2C2_TRACK2",
    "N2C2_2018_TRACK2_DUA_NAME",
    "N2C2_2018_TRACK2_PATH_ENV",
    "N2C2_TRACK2_PATH_ENV",
    "N2C2_2018_TRACK2_ENTITY_TYPES",
    "N2C2_2018_TRACK2_ENTITY_TO_CANONICAL",
    "N2C2_2018_TRACK2_RELATION_TYPES",
    "N2C2_2018_TRACK2_RELATION_TO_CANONICAL",
    "N2C2_2018_TRACK2_SUITE_METADATA",
    "N2C2Track2CredentialRequired",
    "load_n2c2_2018_track2",
    "load_n2c2_2018_track2_fixtures",
    "load_n2c2_2018_track2_ner_fixtures",
    "load_n2c2_2018_track2_relation_fixtures",
    "load_n2c2_track2",
    "load_n2c2_track2_ner_fixtures",
    "load_n2c2_track2_relation_fixtures",
    "map_n2c2_2018_track2_entity_label",
    "map_n2c2_2018_track2_relation_type",
    "n2c2_2018_track2_suite_metadata",
]
