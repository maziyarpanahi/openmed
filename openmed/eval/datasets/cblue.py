"""License-aware CBLUE task-shape loaders for user-supplied benchmark data.

CBLUE publishes several Chinese biomedical tasks under one access-controlled
release. OpenMed supports the task shapes that carry clinical entity
annotations:

* ``chip-cdn`` -- clinical diagnosis normalization. Each row pairs a raw
  diagnosis mention with one or more standard terms joined by ``##``.
* ``imcs-v2-ner`` -- medical-dialogue entity recognition. Each row pairs a
  character sequence with per-character BIO tags over ``Symptom``, ``Drug``,
  ``Drug_Category``, ``Medical_Examination``, and ``Operation``.

Relation decoding (``cmeie``) is out of scope for this loader and is
deliberately not registered; :func:`cblue_task_shape` rejects it with an
explicit message rather than falling through to a generic unknown-task error.

No CBLUE record is bundled, downloaded, cached, or redistributed. Callers pass
an explicit local path to data they are licensed to use, or set the documented
environment variables. The tiny fixtures committed with OpenMed are synthetic
smoke inputs composed from an invented vocabulary and carry no benchmark text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.eval.datasets.licenses import license_for
from openmed.eval.datasets.multilingual_ner import (
    CHIP_CDN,
    IMCS_V2_NER,
    LabelMappingResult,
    MultilingualNerLoadResult,
    load_multilingual_ner_benchmark,
    map_multilingual_ner_label,
    source_for,
)
from openmed.eval.harness import BenchmarkFixture

CBLUE = "cblue"
CMEIE = "cmeie"
CBLUE_PATH_ENV = "OPENMED_CBLUE_PATH"
CBLUE_LANGUAGE = "zh"
CBLUE_SCRIPT = "Han"
CBLUE_TASKS: tuple[str, ...] = (CHIP_CDN, IMCS_V2_NER)

#: CBLUE tasks that OpenMed deliberately does not decode. Relation extraction
#: stays out of scope, so ``cmeie`` never reaches the benchmark interface.
CBLUE_UNSUPPORTED_TASKS: tuple[str, ...] = (CMEIE,)

CHIP_CDN_NORMALIZED_SEPARATOR = "##"

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@dataclass(frozen=True)
class CblueTaskShape:
    """Descriptor for one supported CBLUE task shape."""

    task: str
    display_name: str
    shape: str
    path_env: str
    license_key: str
    entity_types: Mapping[str, str]
    synthetic_fixture: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready descriptor with no benchmark content."""

        return {
            "display_name": self.display_name,
            "entity_types": dict(sorted(self.entity_types.items())),
            "language": CBLUE_LANGUAGE,
            "license": license_for(self.license_key).to_dict(),
            "path_env": self.path_env,
            "script": CBLUE_SCRIPT,
            "shape": self.shape,
            "task": self.task,
        }


CBLUE_TASK_SHAPES: Mapping[str, CblueTaskShape] = {
    CHIP_CDN: CblueTaskShape(
        task=CHIP_CDN,
        display_name="CHIP-CDN",
        shape="entity_normalization",
        path_env="OPENMED_CHIP_CDN_PATH",
        license_key="chip_cdn",
        entity_types=dict(source_for(CHIP_CDN).label_mapping),
        synthetic_fixture="cblue_chip_cdn_synthetic.jsonl",
    ),
    IMCS_V2_NER: CblueTaskShape(
        task=IMCS_V2_NER,
        display_name="IMCS-V2-NER",
        shape="dialogue_ner",
        path_env="OPENMED_IMCS_V2_NER_PATH",
        license_key="imcs_v2_ner",
        entity_types=dict(source_for(IMCS_V2_NER).label_mapping),
        synthetic_fixture="cblue_imcs_v2_ner_synthetic.jsonl",
    ),
}


def cblue_task_shape(task: str) -> CblueTaskShape:
    """Return the descriptor for a supported CBLUE task shape.

    Raises:
        ValueError: If *task* is unknown, or is a CBLUE task that OpenMed
            deliberately leaves undecoded.
    """

    key = _task_key(task)
    if key in CBLUE_UNSUPPORTED_TASKS:
        raise ValueError(
            f"CBLUE task {task!r} is intentionally out of scope: OpenMed does "
            "not implement relation decoding"
        )
    try:
        return CBLUE_TASK_SHAPES[key]
    except KeyError as exc:
        allowed = ", ".join(CBLUE_TASKS)
        raise ValueError(
            f"unknown CBLUE task {task!r}; expected one of: {allowed}"
        ) from exc


def synthetic_cblue_fixture_path(task: str) -> Path:
    """Return the committed synthetic smoke fixture path for *task*."""

    return _FIXTURE_DIR / cblue_task_shape(task).synthetic_fixture


def configured_cblue_task_path(
    task: str,
    path: str | Path | None = None,
) -> Path | None:
    """Return the explicit or environment-configured path for *task*.

    Resolution order is the explicit argument, then the task-specific
    environment variable, then a ``<task>`` child of ``OPENMED_CBLUE_PATH``.
    """

    shape = cblue_task_shape(task)
    if path is not None and str(path).strip():
        return Path(path).expanduser()

    task_env = os.environ.get(shape.path_env)
    if task_env is not None and task_env.strip():
        return Path(task_env).expanduser()

    root_env = os.environ.get(CBLUE_PATH_ENV)
    if root_env is not None and root_env.strip():
        return Path(root_env).expanduser() / shape.task
    return None


def load_cblue_task(
    task: str,
    path: str | Path | None = None,
    *,
    split: str = "test",
    allow_repo_path: bool = False,
) -> MultilingualNerLoadResult:
    """Load one CBLUE task split from an explicit local path.

    OpenMed does not bundle CBLUE corpus text. Pass a local path to data you
    are licensed to use, or set the task's documented environment variable.
    Synthetic smoke fixtures may opt into ``allow_repo_path=True`` in tests.
    """

    shape = cblue_task_shape(task)
    configured_path = configured_cblue_task_path(shape.task, path)
    result = load_multilingual_ner_benchmark(
        shape.task,
        configured_path,
        split=split,
        allow_repo_path=allow_repo_path,
        row_adapter=_ROW_ADAPTERS[shape.task],
    )
    dataset_license = license_for(shape.license_key).to_dict()
    return replace(
        result,
        records=tuple(
            replace(
                record,
                metadata={
                    **dict(record.metadata),
                    "benchmark_family": CBLUE,
                    "license": dataset_license,
                    "script": CBLUE_SCRIPT,
                    "task_shape": shape.shape,
                },
            )
            for record in result.records
        ),
    )


def load_cblue_task_fixtures(
    task: str,
    path: str | Path | None = None,
    *,
    split: str = "test",
    allow_repo_path: bool = False,
) -> list[BenchmarkFixture]:
    """Load one CBLUE task as harness fixtures, rejecting empty sources."""

    shape = cblue_task_shape(task)
    result = load_cblue_task(
        shape.task,
        path,
        split=split,
        allow_repo_path=allow_repo_path,
    )
    fixtures = result.to_benchmark_fixtures()
    if not fixtures:
        raise ValueError(
            f"{shape.path_env} is configured but the {shape.display_name} "
            f"{split!r} source contains no benchmark records"
        )
    if not any(fixture.gold_spans for fixture in fixtures):
        raise ValueError(
            f"{shape.path_env} is configured but the {shape.display_name} "
            f"{split!r} source contains no annotated entity spans"
        )
    # Namespace ids by task: CBLUE releases number rows per file, so two task
    # files both starting at "1" would otherwise collide across a combined run.
    return [
        replace(
            fixture,
            fixture_id=f"{shape.task}/{fixture.fixture_id}",
            metadata={
                **dict(fixture.metadata),
                "cblue_task": shape.task,
                "script": CBLUE_SCRIPT,
                "source_record_id": fixture.fixture_id,
                "task_shape": shape.shape,
            },
        )
        for fixture in fixtures
    ]


def load_cblue_fixtures(
    paths: Mapping[str, str | Path] | None = None,
    *,
    tasks: Sequence[str] = CBLUE_TASKS,
    split: str = "test",
    allow_repo_path: bool = False,
) -> list[BenchmarkFixture]:
    """Load every requested CBLUE task shape as harness fixtures."""

    fixtures: list[BenchmarkFixture] = []
    for task in tasks:
        shape = cblue_task_shape(task)
        task_path = None if paths is None else paths.get(shape.task)
        fixtures.extend(
            load_cblue_task_fixtures(
                shape.task,
                task_path,
                split=split,
                allow_repo_path=allow_repo_path,
            )
        )
    return fixtures


def cblue_task_metadata(
    task: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return task metadata without reading any benchmark content."""

    shape = cblue_task_shape(task)
    configured = configured_cblue_task_path(shape.task, path) is not None
    reason = "" if configured else f"{shape.path_env} is not set"
    return {
        **shape.to_dict(),
        "availability": {
            "configured": configured,
            "path_env": shape.path_env,
            "reason": reason,
            "status": "configured" if configured else "skipped",
        },
        "benchmark_family": CBLUE,
    }


def cblue_suite_metadata(
    paths: Mapping[str, str | Path] | None = None,
    *,
    tasks: Sequence[str] = CBLUE_TASKS,
) -> dict[str, Any]:
    """Return PHI-free metadata covering every requested CBLUE task shape."""

    resolved = [cblue_task_shape(task) for task in tasks]
    return {
        "benchmark_family": CBLUE,
        "data_boundary": (
            "CBLUE task records are user-supplied local inputs; OpenMed "
            "bundles only synthetic smoke records generated from an invented "
            "vocabulary."
        ),
        "language": CBLUE_LANGUAGE,
        "license": license_for(CBLUE).to_dict(),
        "redistribution": "no licensed benchmark corpus text is bundled",
        "root_path_env": CBLUE_PATH_ENV,
        "script": CBLUE_SCRIPT,
        "tasks": {
            shape.task: cblue_task_metadata(
                shape.task,
                None if paths is None else paths.get(shape.task),
            )
            for shape in resolved
        },
        "unsupported_tasks": {
            CMEIE: "relation decoding is out of scope",
        },
    }


def map_cblue_label(task: str, label: str) -> LabelMappingResult:
    """Map a CBLUE source label to an OpenMed canonical label."""

    return map_multilingual_ner_label(cblue_task_shape(task).task, label)


def _chip_cdn_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize one CHIP-CDN row into the shared text-plus-spans shape.

    The raw mention becomes the record text and carries a single diagnosis
    span, so normalization rows score through the same offset validation and
    canonical label mapping as span-native task shapes. Standard terms are
    retained as metadata for normalization scoring.
    """

    mention = str(row.get("text") or row.get("mention") or "").strip()
    if not mention:
        raise ValueError("CHIP-CDN row requires a non-empty diagnosis mention")

    raw_normalized = row.get("normalized_result")
    if raw_normalized is None:
        raw_normalized = row.get("normalized_terms")
    if isinstance(raw_normalized, str):
        candidates = raw_normalized.split(CHIP_CDN_NORMALIZED_SEPARATOR)
    elif isinstance(raw_normalized, Sequence) and not isinstance(
        raw_normalized, (str, bytes)
    ):
        candidates = [str(item) for item in raw_normalized]
    else:
        candidates = []
    normalized_terms = [term.strip() for term in candidates if str(term).strip()]
    if not normalized_terms:
        raise ValueError("CHIP-CDN row requires at least one standard term")

    label = str(row.get("label") or row.get("type") or "dis")
    return {
        **{key: value for key, value in row.items() if key not in _CHIP_CDN_RAW_KEYS},
        "metadata": {
            **dict(row.get("metadata") or {}),
            "normalized_terms": normalized_terms,
        },
        "spans": [
            {
                "end": len(mention),
                "label": label,
                "start": 0,
                "text": mention,
            }
        ],
        "text": mention,
    }


def _imcs_v2_ner_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize one IMCS-V2-NER row into the shared text-plus-spans shape.

    ``sentence`` may arrive as a character list or a string, and ``BIO_label``
    as a tag list or a whitespace-joined string. Character offsets are decoded
    from the tag sequence rather than read from the source row.
    """

    characters = _character_sequence(row)
    tags = _tag_sequence(row)
    if len(characters) != len(tags):
        raise ValueError(
            f"IMCS-V2-NER row has {len(characters)} characters and {len(tags)} BIO tags"
        )

    text = "".join(characters)
    spans: list[dict[str, Any]] = []
    active_label = ""
    active_start = 0
    for index, tag in enumerate(tags):
        prefix, label = _bio_tag(tag)
        starts_entity = bool(label) and (prefix == "B" or label != active_label)
        if active_label and (starts_entity or not label):
            spans.append(_imcs_span(text, active_start, index, active_label))
            active_label = ""
        if starts_entity:
            active_label = label
            active_start = index
    if active_label:
        spans.append(_imcs_span(text, active_start, len(tags), active_label))

    return {
        **{key: value for key, value in row.items() if key not in _IMCS_RAW_KEYS},
        "spans": spans,
        "text": text,
    }


def _imcs_span(text: str, start: int, end: int, label: str) -> dict[str, Any]:
    return {"end": end, "label": label, "start": start, "text": text[start:end]}


def _character_sequence(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("sentence")
    if raw is None:
        raw = row.get("text")
    if isinstance(raw, str):
        return list(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, bytes):
        return [str(item) for item in raw]
    raise ValueError("IMCS-V2-NER row requires a sentence character sequence")


def _tag_sequence(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("BIO_label")
    if raw is None:
        raw = row.get("bio_label")
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, Sequence) and not isinstance(raw, bytes):
        return [str(item) for item in raw]
    raise ValueError("IMCS-V2-NER row requires a BIO_label tag sequence")


def _bio_tag(tag: str) -> tuple[str, str]:
    normalized = str(tag or "O").strip()
    if not normalized or normalized.upper() == "O":
        return "O", ""
    if len(normalized) > 2 and normalized[1] == "-":
        return normalized[0].upper(), normalized[2:]
    return "B", normalized


def _task_key(task: str) -> str:
    return str(task).strip().lower().replace("_", "-")


_CHIP_CDN_RAW_KEYS = frozenset(
    {"metadata", "normalized_result", "normalized_terms", "spans", "text"}
)
_IMCS_RAW_KEYS = frozenset({"BIO_label", "bio_label", "sentence", "spans", "text"})
_ROW_ADAPTERS = {
    CHIP_CDN: _chip_cdn_row,
    IMCS_V2_NER: _imcs_v2_ner_row,
}


__all__ = [
    "CBLUE",
    "CBLUE_LANGUAGE",
    "CBLUE_PATH_ENV",
    "CBLUE_SCRIPT",
    "CBLUE_TASKS",
    "CBLUE_TASK_SHAPES",
    "CBLUE_UNSUPPORTED_TASKS",
    "CHIP_CDN",
    "CHIP_CDN_NORMALIZED_SEPARATOR",
    "CMEIE",
    "IMCS_V2_NER",
    "CblueTaskShape",
    "cblue_suite_metadata",
    "cblue_task_metadata",
    "cblue_task_shape",
    "configured_cblue_task_path",
    "load_cblue_fixtures",
    "load_cblue_task",
    "load_cblue_task_fixtures",
    "map_cblue_label",
    "synthetic_cblue_fixture_path",
]
