"""License-aware CMeEE loader for user-supplied CBLUE data.

CMeEE uses nine source categories: ``bod`` (body site), ``dep`` (department),
``dis`` (disease), ``dru`` (drug), ``equ`` (equipment), ``ite`` (test item),
``mic`` (microorganism), ``pro`` (procedure), and ``sym`` (symptom). The
benchmark's inclusive code-point offsets are converted to OpenMed's exclusive
end offsets without flattening nested entities.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from openmed.core.labels import (
    BODY_SITE,
    CONDITION,
    JOB_DEPARTMENT,
    LAB_TEST,
    MEDICATION,
    MICROORGANISM,
    OTHER,
    PROCEDURE,
)
from openmed.eval.datasets.licenses import license_for
from openmed.eval.datasets.multilingual_ner import (
    CMEEE,
    LabelMappingResult,
    MultilingualNerLoadResult,
    load_multilingual_ner_benchmark,
    map_multilingual_ner_label,
    source_for,
)
from openmed.eval.harness import BenchmarkFixture

CMEEE_SOURCE = source_for(CMEEE)
CMEEE_PATH_ENV = "OPENMED_CMEEE_PATH"
CMEEE_SCRIPT = "Han"
CMEEE_ENTITY_TYPES: Mapping[str, str] = {
    "bod": BODY_SITE,
    "dep": JOB_DEPARTMENT,
    "dis": CONDITION,
    "dru": MEDICATION,
    "equ": OTHER,
    "ite": LAB_TEST,
    "mic": MICROORGANISM,
    "pro": PROCEDURE,
    "sym": CONDITION,
}


def configured_cmeee_path(path: str | Path | None = None) -> Path | None:
    """Return the explicit or environment-configured CMeEE path."""

    raw_path = path if path is not None else os.environ.get(CMEEE_PATH_ENV)
    if raw_path is None or not str(raw_path).strip():
        return None
    return Path(raw_path).expanduser()


def load_cmeee(
    path: str | Path | None = None,
    *,
    split: str = "test",
    allow_repo_path: bool = False,
) -> MultilingualNerLoadResult:
    """Load CMeEE records from an explicit local path.

    OpenMed does not bundle CMeEE corpus text. Pass a local path to data you
    are licensed to use, or set ``OPENMED_CMEEE_PATH``. Synthetic smoke
    fixtures may opt into ``allow_repo_path=True`` in tests.
    """

    configured_path = configured_cmeee_path(path)
    source_path = _select_split_source(configured_path, split=split)
    result = load_multilingual_ner_benchmark(
        CMEEE,
        source_path,
        split=split,
        allow_repo_path=allow_repo_path,
    )
    dataset_license = license_for(CMEEE).to_dict()
    return replace(
        result,
        records=tuple(
            replace(
                record,
                metadata={
                    **dict(record.metadata),
                    "license": dataset_license,
                    "script": CMEEE_SCRIPT,
                },
            )
            for record in result.records
        ),
    )


def load_cmeee_fixtures(
    path: str | Path | None = None,
    *,
    split: str = "test",
    allow_repo_path: bool = False,
) -> list[BenchmarkFixture]:
    """Load CMeEE fixtures and reject configured-but-empty sources."""

    result = load_cmeee(path, split=split, allow_repo_path=allow_repo_path)
    fixtures = result.to_benchmark_fixtures()
    if not fixtures:
        raise ValueError(
            f"{CMEEE_PATH_ENV} is configured but the CMeEE {split!r} source "
            "contains no benchmark records"
        )
    if not any(fixture.gold_spans for fixture in fixtures):
        raise ValueError(
            f"{CMEEE_PATH_ENV} is configured but the CMeEE {split!r} source "
            "contains no annotated entity spans"
        )
    return [
        replace(
            fixture,
            metadata={**dict(fixture.metadata), "script": CMEEE_SCRIPT, "suite": CMEEE},
        )
        for fixture in fixtures
    ]


def cmeee_suite_metadata(path: str | Path | None = None) -> dict[str, Any]:
    """Return CMeEE suite metadata without reading benchmark content."""

    configured = configured_cmeee_path(path) is not None
    reason = "" if configured else f"{CMEEE_PATH_ENV} is not set"
    return {
        "availability": {
            "configured": configured,
            "path_env": CMEEE_PATH_ENV,
            "reason": reason,
            "status": "configured" if configured else "skipped",
        },
        "entity_types": dict(CMEEE_ENTITY_TYPES),
        "language": "zh",
        "license": license_for(CMEEE).to_dict(),
        "script": CMEEE_SCRIPT,
        "suite": CMEEE,
        "task": "clinical_ner",
    }


def map_cmeee_label(label: str) -> LabelMappingResult:
    """Map a CMeEE source label to an OpenMed canonical label."""

    return map_multilingual_ner_label(CMEEE, label)


def _select_split_source(path: Path | None, *, split: str) -> Path | None:
    if path is None or not path.is_dir():
        return path
    split_aliases = {split.lower()}
    if split.lower() in {"dev", "val", "validation"}:
        split_aliases.update({"dev", "val", "validation"})
    candidates = [
        candidate
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
        and candidate.suffix.lower() in {".json", ".jsonl", ".ndjson"}
        and "cmeee" in candidate.name.lower()
        and any(
            alias in candidate.stem.lower().replace("-", "_").split("_")
            for alias in split_aliases
        )
    ]
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates)
        raise ValueError(f"multiple CMeEE {split!r} sources found: {names}")
    return candidates[0] if candidates else path


__all__ = [
    "CMEEE",
    "CMEEE_ENTITY_TYPES",
    "CMEEE_PATH_ENV",
    "CMEEE_SCRIPT",
    "CMEEE_SOURCE",
    "cmeee_suite_metadata",
    "configured_cmeee_path",
    "load_cmeee",
    "load_cmeee_fixtures",
    "map_cmeee_label",
]
